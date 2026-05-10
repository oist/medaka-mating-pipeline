from __future__ import annotations

"""
Blob-first top-camera identity workflow.

This script is deliberately idtracker-inspired without depending on
idtracker.ai internals:

1. Mine fish-shaped blob candidates from the top camera.
2. Curate candidates manually.
3. Link accepted blobs into fragments.
4. Train a small fragment-level appearance model.
5. Apply IDs to fragments, leaving uncertain cases unknown.

CLI:
    python -m scripts.tracking.top_camera_identity mine-blobs EXP ...
    python -m scripts.tracking.top_camera_identity review blob_candidates.csv
    python -m scripts.tracking.top_camera_identity train-id reviewed.csv
    python -m scripts.tracking.top_camera_identity bootstrap-id reviewed.csv
    python -m scripts.tracking.top_camera_identity apply-id candidates.csv MODEL
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.experiment_metadata import VIDEO_BASENAMES
from scripts.common.local_paths import choose_path, load_local_paths


ID_LABELS = ("ID_1", "ID_2", "ID_3")
UNKNOWN_ID = "unknown"


@dataclass
class BlobCandidate:
    candidate_id: str
    frame: int
    x: float
    y: float
    area: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    mask_path: str
    source: str
    near_sleap_det: int
    quality_score: float
    status: str = "review"
    id: str = ""
    bad_reason: str = ""
    crop_path: str = ""
    sam_score: float | None = None


@dataclass
class FragmentSummary:
    fragment_id: int
    start_frame: int
    end_frame: int
    n_candidates: int
    identity: str
    distance: float | None
    margin: float | None
    unknown_reason: str


def _as_path(path: str | Path, *, base: Path | None = None) -> Path:
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_top_sleap_points(
    slp_path: Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[int, list[tuple[float, float, float]]]:
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError("sleap-io is required. Install with `uv sync --extra sleap`.") from exc

    labels = sio.load_slp(str(slp_path))
    out: dict[int, list[tuple[float, float, float]]] = {}
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if start_frame is not None and frame < start_frame:
            continue
        if end_frame is not None and frame > end_frame:
            continue
        pts_out: list[tuple[float, float, float]] = []
        for inst in lf.instances:
            pts = inst.numpy()
            valid = np.isfinite(pts).all(axis=1)
            if not valid.any():
                continue
            xy = pts[valid].mean(axis=0)
            score = getattr(inst, "score", None)
            pts_out.append((float(xy[0]), float(xy[1]), _safe_float(score)))
        if pts_out:
            out[frame] = pts_out
    return out


def estimate_median_background(
    video_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    max_samples: int = 80,
) -> np.ndarray:
    """Estimate a grayscale median background from sparse frames."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = np.linspace(start_frame, end_frame, max(1, max_samples)).round().astype(int)
    samples: list[np.ndarray] = []
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, img = cap.read()
            if not ok:
                continue
            samples.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if not samples:
        raise RuntimeError(f"No frames read while estimating background: {video_path}")
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def foreground_mask_from_frame(
    frame_bgr: np.ndarray,
    background_gray: np.ndarray,
    *,
    threshold: int = 24,
    open_px: int = 3,
    close_px: int = 7,
) -> np.ndarray:
    """Return a cleaned binary foreground mask for one top-camera frame."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def component_candidates(
    mask: np.ndarray,
    *,
    frame: int,
    source: str,
    candidate_prefix: str,
    min_area: int = 80,
    max_area: int = 8000,
    min_aspect: float = 0.15,
    max_aspect: float = 6.0,
    min_fill: float = 0.08,
    sam_score: float | None = None,
) -> list[BlobCandidate]:
    """Convert connected components into unfiltered blob candidates."""
    import cv2

    n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out: list[BlobCandidate] = []
    seq = 0
    for label in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area or w <= 0 or h <= 0:
            continue
        aspect = w / max(1, h)
        fill = area / max(1, w * h)
        if aspect < min_aspect or aspect > max_aspect or fill < min_fill:
            continue
        cx, cy = centroids[label]
        out.append(
            BlobCandidate(
                candidate_id=f"{candidate_prefix}_{frame}_{seq:03d}",
                frame=int(frame),
                x=float(cx),
                y=float(cy),
                area=area,
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                mask_path="",
                source=source,
                near_sleap_det=-1,
                quality_score=float(area),
                sam_score=sam_score,
            )
        )
        seq += 1
    return out


def filter_candidates_by_sleap(
    candidates: list[BlobCandidate],
    sleap_points: list[tuple[float, float, float]],
    *,
    max_distance_px: float = 80.0,
) -> list[BlobCandidate]:
    """Keep candidates near a SLEAP mouth point and score by distance/area."""
    if not sleap_points:
        return []
    filtered: list[BlobCandidate] = []
    pts = np.array([[p[0], p[1]] for p in sleap_points], dtype=float)
    for cand in candidates:
        d = np.linalg.norm(pts - np.array([[cand.x, cand.y]]), axis=1)
        det_idx = int(np.argmin(d))
        dist = float(d[det_idx])
        if dist > max_distance_px:
            continue
        cand.near_sleap_det = det_idx
        cand.quality_score = float(cand.area / (1.0 + dist))
        filtered.append(cand)
    return filtered


def suppress_overlapping_candidates(
    candidates: list[BlobCandidate],
    *,
    max_center_distance_px: float = 20.0,
    same_det_only: bool = True,
) -> list[BlobCandidate]:
    """
    Keep one candidate when multiple masks explain the same fish.

    This mainly deduplicates bgsub/SAM candidates, but also helps when a noisy
    threshold produces two nearby components for one fish.
    """
    ranked = sorted(candidates, key=lambda c: (-c.quality_score, c.source, c.candidate_id))
    kept: list[BlobCandidate] = []
    for cand in ranked:
        duplicate = False
        for prev in kept:
            if same_det_only and cand.near_sleap_det != prev.near_sleap_det:
                continue
            dist = float(np.hypot(cand.x - prev.x, cand.y - prev.y))
            if dist <= max_center_distance_px:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return sorted(kept, key=lambda c: c.candidate_id)


def keep_best_candidate_per_sleap_detection(
    candidates: list[BlobCandidate],
    *,
    max_per_detection: int = 1,
) -> list[BlobCandidate]:
    """Limit candidates so one SLEAP mouth cannot spawn many fish blobs."""
    if max_per_detection <= 0:
        return candidates
    grouped: dict[int, list[BlobCandidate]] = {}
    passthrough: list[BlobCandidate] = []
    for cand in candidates:
        if cand.near_sleap_det < 0:
            passthrough.append(cand)
        else:
            grouped.setdefault(cand.near_sleap_det, []).append(cand)
    kept = list(passthrough)
    for group in grouped.values():
        kept.extend(
            sorted(group, key=lambda c: (-c.quality_score, c.source, c.candidate_id))[
                :max_per_detection
            ]
        )
    return sorted(kept, key=lambda c: c.candidate_id)


def mask_iou(
    a_mask: np.ndarray,
    a_bbox: tuple[int, int, int, int],
    b_mask: np.ndarray,
    b_bbox: tuple[int, int, int, int],
) -> float:
    """IoU for cropped binary masks with full-frame bboxes."""
    ax, ay, aw, ah = a_bbox
    bx, by, bw, bh = b_bbox
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    a_crop = a_mask[iy0 - ay : iy1 - ay, ix0 - ax : ix1 - ax] > 0
    b_crop = b_mask[iy0 - by : iy1 - by, ix0 - bx : ix1 - bx] > 0
    inter = int(np.logical_and(a_crop, b_crop).sum())
    union = int((a_mask > 0).sum() + (b_mask > 0).sum() - inter)
    return inter / union if union > 0 else 0.0


def bbox_iou(a_bbox: tuple[int, int, int, int], b_bbox: tuple[int, int, int, int]) -> float:
    """IoU for full-frame bboxes in x,y,w,h format."""
    ax, ay, aw, ah = a_bbox
    bx, by, bw, bh = b_bbox
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _load_mask_for_candidate(row: pd.Series, *, csv_dir: Path) -> np.ndarray | None:
    mask_path = str(row.get("mask_path", ""))
    if not mask_path:
        return None
    p = _as_path(mask_path, base=csv_dir)
    if not p.exists():
        return None
    import cv2

    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return mask


def _candidate_bbox(row: pd.Series) -> tuple[int, int, int, int]:
    return (
        int(row["bbox_x"]),
        int(row["bbox_y"]),
        int(row["bbox_w"]),
        int(row["bbox_h"]),
    )


def _row_id_label(row: pd.Series) -> str:
    identity = str(row.get("id", ""))
    return identity if identity in ID_LABELS else ""


def candidate_mask_to_full_frame(
    row: pd.Series,
    *,
    csv_dir: Path,
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """Expand a cropped candidate mask to full-frame coordinates."""
    full_h, full_w = int(frame_shape[0]), int(frame_shape[1])
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    mask = _load_mask_for_candidate(row, csv_dir=csv_dir)
    if mask is None:
        return full
    x, y, w, h = _candidate_bbox(row)
    if w <= 0 or h <= 0:
        return full
    if mask.shape[:2] != (h, w):
        import cv2

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(full_w, x + w), min(full_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return full
    mx0, my0 = x0 - x, y0 - y
    mx1, my1 = mx0 + (x1 - x0), my0 + (y1 - y0)
    full[y0:y1, x0:x1] = (mask[my0:my1, mx0:mx1] > 0).astype(np.uint8) * 255
    return full


def save_edited_candidate_mask(
    df: pd.DataFrame,
    row_idx: int,
    full_mask: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    csv_dir: Path,
) -> None:
    """Persist an edited full-frame mask and update candidate geometry columns."""
    import cv2

    mask = (full_mask > 0).astype(np.uint8) * 255
    area = int((mask > 0).sum())
    if area <= 0:
        df.at[row_idx, "area"] = 0
        df.at[row_idx, "status"] = "reject"
        df.at[row_idx, "id"] = ""
        df.at[row_idx, "bad_reason"] = "empty_mask_after_edit"
        return

    x, y, w, h = cv2.boundingRect(mask)
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2)
        cy = float(y + h / 2)

    candidate_id = str(df.at[row_idx, "candidate_id"])
    mask_path = str(df.at[row_idx, "mask_path"]) if "mask_path" in df.columns else ""
    if not mask_path or _as_path(mask_path, base=csv_dir).is_absolute() and not Path(mask_path).is_relative_to(csv_dir):
        mask_path = str(Path("masks") / f"{candidate_id}.png")
        df.at[row_idx, "mask_path"] = mask_path
    mask_out = _as_path(mask_path, base=csv_dir)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_out), mask[y : y + h, x : x + w])

    crop_path = str(df.at[row_idx, "crop_path"]) if "crop_path" in df.columns else ""
    if not crop_path:
        crop_path = str(Path("crops") / f"{candidate_id}.png")
        df.at[row_idx, "crop_path"] = crop_path
    crop_out = _as_path(crop_path, base=csv_dir)
    crop_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crop_out), frame_bgr[y : y + h, x : x + w])

    df.at[row_idx, "x"] = cx
    df.at[row_idx, "y"] = cy
    df.at[row_idx, "area"] = area
    df.at[row_idx, "bbox_x"] = int(x)
    df.at[row_idx, "bbox_y"] = int(y)
    df.at[row_idx, "bbox_w"] = int(w)
    df.at[row_idx, "bbox_h"] = int(h)
    if str(df.at[row_idx, "bad_reason"]) == "empty_mask_after_edit":
        df.at[row_idx, "bad_reason"] = ""


def smart_fill_region_from_click(
    frame_bgr: np.ndarray,
    seed_xy: tuple[int, int],
    *,
    search_radius: int = 90,
    tolerance: float = 30.0,
    close_px: int = 5,
) -> np.ndarray:
    """
    Segment a connected, locally color-similar region from one click.

    This is a small Labelbox-style smart-fill helper: it thresholds Lab color
    distance inside a local window, then keeps only the connected component
    touched by the click. It is intentionally simple and inspectable.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if sx < 0 or sy < 0 or sx >= w or sy >= h:
        return np.zeros((h, w), dtype=np.uint8)

    r = max(8, int(search_radius))
    x0, y0 = max(0, sx - r), max(0, sy - r)
    x1, y1 = min(w, sx + r + 1), min(h, sy + r + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    local_sx, local_sy = sx - x0, sy - y0
    seed_y0, seed_y1 = max(0, local_sy - 1), min(lab.shape[0], local_sy + 2)
    seed_x0, seed_x1 = max(0, local_sx - 1), min(lab.shape[1], local_sx + 2)
    seed_color = np.median(lab[seed_y0:seed_y1, seed_x0:seed_x1].reshape(-1, 3), axis=0)

    yy, xx = np.ogrid[: lab.shape[0], : lab.shape[1]]
    radial_limit = ((xx - local_sx) ** 2 + (yy - local_sy) ** 2) <= r**2
    color_dist = np.linalg.norm(lab - seed_color.reshape(1, 1, 3), axis=2)
    local_mask = ((color_dist <= float(tolerance)) & radial_limit).astype(np.uint8) * 255
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)

    n, labels, _stats, _centroids = cv2.connectedComponentsWithStats((local_mask > 0).astype(np.uint8), 8)
    seed_label = int(labels[local_sy, local_sx])
    full = np.zeros((h, w), dtype=np.uint8)
    if n <= 1 or seed_label == 0:
        return full
    full[y0:y1, x0:x1] = (labels == seed_label).astype(np.uint8) * 255
    return full


def merge_same_frame_same_id_candidates(
    df: pd.DataFrame,
    *,
    csv_dir: Path,
    out_subdir: str = "merged_same_id",
) -> tuple[pd.DataFrame, dict]:
    """
    Merge manually accepted split blobs with the same ID in the same frame.

    Review can intentionally label partial boxes with the same ID. For training
    appearance, those parts should become one virtual fish-shaped crop/mask so
    the classifier learns the whole animal rather than competing fragments.
    """
    import cv2

    if df.empty or "id" not in df.columns or "status" not in df.columns:
        return df.copy(), {"n_merged_groups": 0, "n_merged_parts": 0}

    out = df.copy()
    keep_indices: set[int] = set(out.index)
    merged_rows: list[pd.Series] = []
    n_parts = 0
    masks_dir = csv_dir / out_subdir / "masks"
    crops_dir = csv_dir / out_subdir / "crops"

    accepted = out[(out["status"] == "accept") & (out["id"].isin(ID_LABELS))]
    for (frame, identity), group in accepted.groupby(["frame", "id"], sort=True):
        if len(group) <= 1:
            continue

        pieces: list[tuple[pd.Series, np.ndarray, np.ndarray]] = []
        x0 = y0 = 10**9
        x1 = y1 = -1
        for row_idx, row in group.iterrows():
            mask = _load_mask_for_candidate(row, csv_dir=csv_dir)
            crop_path = str(row.get("crop_path", ""))
            crop = None
            if crop_path:
                cp = _as_path(crop_path, base=csv_dir)
                if cp.exists():
                    crop = cv2.imread(str(cp), cv2.IMREAD_COLOR)
            if mask is None or crop is None:
                continue
            bx, by, bw, bh = _candidate_bbox(row)
            if bw <= 0 or bh <= 0:
                continue
            if mask.shape[:2] != (bh, bw):
                mask = cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
            if crop.shape[:2] != (bh, bw):
                crop = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_AREA)
            pieces.append((row, mask, crop))
            x0, y0 = min(x0, bx), min(y0, by)
            x1, y1 = max(x1, bx + bw), max(y1, by + bh)

        if len(pieces) <= 1 or x1 <= x0 or y1 <= y0:
            continue

        merged_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        merged_crop = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        for row, mask, crop in pieces:
            bx, by, bw, bh = _candidate_bbox(row)
            yy = by - y0
            xx = bx - x0
            part = mask > 0
            merged_mask[yy : yy + bh, xx : xx + bw][part] = 255
            merged_crop[yy : yy + bh, xx : xx + bw][part] = crop[part]

        area = int((merged_mask > 0).sum())
        if area <= 0:
            continue
        mx, my, mw, mh = cv2.boundingRect(merged_mask)
        merged_mask = merged_mask[my : my + mh, mx : mx + mw]
        merged_crop = merged_crop[my : my + mh, mx : mx + mw]
        gx, gy = x0 + mx, y0 + my
        moments = cv2.moments(merged_mask, binaryImage=True)
        if moments["m00"] > 0:
            cx = gx + float(moments["m10"] / moments["m00"])
            cy = gy + float(moments["m01"] / moments["m00"])
        else:
            cx = float(gx + mw / 2)
            cy = float(gy + mh / 2)

        candidate_id = f"merged_f{int(frame)}_{identity}_{len(merged_rows):04d}"
        masks_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{candidate_id}.png"
        crop_path = crops_dir / f"{candidate_id}.png"
        cv2.imwrite(str(mask_path), merged_mask)
        cv2.imwrite(str(crop_path), merged_crop)

        new_row = group.iloc[0].copy()
        new_row["candidate_id"] = candidate_id
        new_row["x"] = cx
        new_row["y"] = cy
        new_row["area"] = area
        new_row["bbox_x"] = int(gx)
        new_row["bbox_y"] = int(gy)
        new_row["bbox_w"] = int(mw)
        new_row["bbox_h"] = int(mh)
        new_row["mask_path"] = str(mask_path.relative_to(csv_dir))
        new_row["crop_path"] = str(crop_path.relative_to(csv_dir))
        new_row["source"] = "merged_same_id"
        new_row["quality_score"] = float(pd.to_numeric(group["quality_score"], errors="coerce").fillna(0).sum())
        new_row["near_sleap_det"] = -1
        new_row["merged_from"] = "|".join(group["candidate_id"].astype(str))
        new_row["n_merged_parts"] = int(len(group))
        merged_rows.append(new_row)
        n_parts += int(len(group))
        keep_indices.difference_update(set(group.index))

    merged = out.loc[sorted(keep_indices)].copy()
    if merged_rows:
        merged = pd.concat([merged, pd.DataFrame(merged_rows)], ignore_index=True, sort=False)
    return merged.reset_index(drop=True), {
        "n_merged_groups": int(len(merged_rows)),
        "n_merged_parts": int(n_parts),
    }


def _track_prediction(
    df: pd.DataFrame,
    last_idx: int,
    prev_idx: int | None,
    *,
    target_frame: int,
) -> np.ndarray:
    last_xy = np.array([df.at[last_idx, "x"], df.at[last_idx, "y"]], dtype=float)
    if prev_idx is None:
        return last_xy
    dt = max(1, int(df.at[last_idx, "frame"]) - int(df.at[prev_idx, "frame"]))
    velocity = (
        last_xy - np.array([df.at[prev_idx, "x"], df.at[prev_idx, "y"]], dtype=float)
    ) / dt
    gap = max(1, target_frame - int(df.at[last_idx, "frame"]))
    return last_xy + velocity * gap


def _fragment_endpoint_summary(group: pd.DataFrame) -> dict:
    ordered = group.sort_values("frame")
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    ids = ordered["id"].dropna().astype(str) if "id" in ordered else pd.Series([], dtype=str)
    ids = ids[ids.isin(ID_LABELS)]
    identity = str(ids.mode().iloc[0]) if not ids.empty else ""
    return {
        "start_frame": int(first["frame"]),
        "end_frame": int(last["frame"]),
        "start_xy": np.array([float(first["x"]), float(first["y"])]),
        "end_xy": np.array([float(last["x"]), float(last["y"])]),
        "start_bbox": _candidate_bbox(first),
        "end_bbox": _candidate_bbox(last),
        "identity": identity,
        "n": int(len(ordered)),
    }


def stitch_short_fragment_gaps(
    df: pd.DataFrame,
    *,
    max_gap_frames: int = 20,
    max_distance_px: float = 80.0,
) -> pd.DataFrame:
    """Offline tracklet stitching with manual-ID and motion constraints."""
    if df.empty or "fragment_id" not in df.columns:
        return df
    out = df.copy()
    while True:
        summaries = {
            int(fid): _fragment_endpoint_summary(group)
            for fid, group in out.groupby("fragment_id")
        }
        fids = sorted(summaries)
        best_pair: tuple[int, int] | None = None
        best_cost = float("inf")
        for a in fids:
            sa = summaries[a]
            for b in fids:
                if a == b:
                    continue
                sb = summaries[b]
                gap = int(sb["start_frame"] - sa["end_frame"])
                if gap <= 0 or gap > max_gap_frames:
                    continue
                if sa["identity"] and sb["identity"] and sa["identity"] != sb["identity"]:
                    continue
                dist = float(np.linalg.norm(sb["start_xy"] - sa["end_xy"]))
                if dist > max_distance_px * max(1, gap):
                    continue
                box = bbox_iou(sa["end_bbox"], sb["start_bbox"])
                cost = dist / max(1, gap) - box * max_distance_px
                if sa["identity"] and sa["identity"] == sb["identity"]:
                    cost -= max_distance_px
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (a, b)
        if best_pair is None:
            break
        keep, merge = best_pair
        out.loc[out["fragment_id"] == merge, "fragment_id"] = keep
    return out


def link_blob_fragments(
    candidates: pd.DataFrame,
    *,
    csv_dir: Path | None = None,
    max_distance_px: float = 60.0,
    min_iou: float = 0.02,
    min_bbox_iou: float = 0.01,
    max_gap_frames: int = 5,
    stitch_gap_frames: int = 20,
) -> pd.DataFrame:
    """Mask-OCSORT-style linking with bbox/mask costs and offline stitching."""
    if candidates.empty:
        out = candidates.copy()
        out["fragment_id"] = []
        return out
    csv_dir = csv_dir or Path(".")
    df = candidates.copy().sort_values(["frame", "candidate_id"]).reset_index(drop=True)
    if "status" in df.columns:
        df = df[df["status"].fillna("review").isin(["accept", "review", "unknown", ""])].copy()
    df["fragment_id"] = -1

    active: dict[int, dict[str, int | None]] = {}  # fragment_id -> last/prev row indices
    next_fragment = 0
    masks: dict[int, np.ndarray | None] = {}

    for frame, idxs in df.groupby("frame", sort=True).groups.items():
        frame = int(frame)
        stale = [
            fid for fid, state in active.items()
            if frame - int(df.at[int(state["last_idx"]), "frame"]) > max_gap_frames
        ]
        for fid in stale:
            del active[fid]

        curr = list(idxs)
        if not active:
            for ridx in curr:
                df.at[ridx, "fragment_id"] = next_fragment
                active[next_fragment] = {"last_idx": ridx, "prev_idx": None}
                next_fragment += 1
            continue

        fids = list(active.keys())
        prev_rows = [int(active[fid]["last_idx"]) for fid in fids]
        cost = np.full((len(prev_rows), len(curr)), 1e9, dtype=float)
        for i, prev_idx in enumerate(prev_rows):
            prev_xy = np.array([df.at[prev_idx, "x"], df.at[prev_idx, "y"]], dtype=float)
            pred_xy = _track_prediction(
                df,
                prev_idx,
                active[fids[i]].get("prev_idx"),
                target_frame=frame,
            )
            prev_bbox = _candidate_bbox(df.loc[prev_idx])
            prev_identity = _row_id_label(df.loc[prev_idx])
            prev_mask = masks.get(prev_idx)
            if prev_idx not in masks:
                prev_mask = _load_mask_for_candidate(df.loc[prev_idx], csv_dir=csv_dir)
                masks[prev_idx] = prev_mask
            for j, curr_idx in enumerate(curr):
                curr_xy = np.array([df.at[curr_idx, "x"], df.at[curr_idx, "y"]], dtype=float)
                curr_identity = _row_id_label(df.loc[curr_idx])
                if prev_identity and curr_identity and prev_identity != curr_identity:
                    continue
                dist = float(np.linalg.norm(curr_xy - pred_xy))
                raw_dist = float(np.linalg.norm(curr_xy - prev_xy))
                gap = max(1, frame - int(df.at[prev_idx, "frame"]))
                if dist > max_distance_px * gap and raw_dist > max_distance_px * gap:
                    continue
                curr_bbox = _candidate_bbox(df.loc[curr_idx])
                box_iou = bbox_iou(prev_bbox, curr_bbox)
                iou = 0.0
                curr_mask = masks.get(curr_idx)
                if curr_idx not in masks:
                    curr_mask = _load_mask_for_candidate(df.loc[curr_idx], csv_dir=csv_dir)
                    masks[curr_idx] = curr_mask
                if prev_mask is not None and curr_mask is not None:
                    iou = mask_iou(
                        prev_mask,
                        _candidate_bbox(df.loc[prev_idx]),
                        curr_mask,
                        _candidate_bbox(df.loc[curr_idx]),
                    )
                if iou < min_iou and box_iou < min_bbox_iou and min(dist, raw_dist) > max_distance_px * 0.5:
                    continue
                label_bonus = max_distance_px if prev_identity and prev_identity == curr_identity else 0.0
                area_prev = max(1.0, float(df.at[prev_idx, "area"]))
                area_curr = max(1.0, float(df.at[curr_idx, "area"]))
                area_penalty = abs(np.log(area_curr / area_prev)) * (max_distance_px * 0.25)
                cost[i, j] = (
                    dist
                    - (iou * max_distance_px)
                    - (box_iou * max_distance_px * 0.75)
                    + area_penalty
                    - label_bonus
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_curr: set[int] = set()
        assigned_fids: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= 1e9:
                continue
            fid = fids[int(ri)]
            curr_idx = curr[int(ci)]
            df.at[curr_idx, "fragment_id"] = fid
            active[fid] = {"last_idx": curr_idx, "prev_idx": active[fid]["last_idx"]}
            assigned_curr.add(curr_idx)
            assigned_fids.add(fid)
        # Fragments not assigned this frame stay active until max_gap.
        for curr_idx in curr:
            if curr_idx in assigned_curr:
                continue
            df.at[curr_idx, "fragment_id"] = next_fragment
            active[next_fragment] = {"last_idx": curr_idx, "prev_idx": None}
            next_fragment += 1

    if stitch_gap_frames > 0:
        df = stitch_short_fragment_gaps(
            df,
            max_gap_frames=stitch_gap_frames,
            max_distance_px=max_distance_px,
        )
    return df


def crop_masked_feature(
    crop_bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    feature_size: int = 32,
) -> np.ndarray:
    """Normalized full-blob appearance feature from crop and optional mask."""
    import cv2

    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        crop_bgr = crop_bgr.copy()
        crop_bgr[mask == 0] = 0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (feature_size, feature_size), interpolation=cv2.INTER_AREA)
    feat = small.astype(np.float32).ravel()
    feat -= float(feat.mean())
    std = float(feat.std())
    if std > 1e-6:
        feat /= std
    norm = float(np.linalg.norm(feat))
    if norm > 1e-6:
        feat /= norm
    return feat


def fit_identity_centroids(features_by_id: dict[str, list[np.ndarray]], *, min_samples: int = 5) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for identity, feats in features_by_id.items():
        if identity not in ID_LABELS or len(feats) < min_samples:
            continue
        centroid = np.mean(np.vstack(feats), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-6:
            centroid = centroid / norm
        centroids[identity] = centroid.astype(np.float32)
    return centroids


def classify_fragment_feature(
    feature: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    min_margin: float = 0.05,
    max_distance: float = 1.25,
) -> tuple[str, float | None, float | None, str]:
    if not centroids:
        return UNKNOWN_ID, None, None, "no_model"
    dists = sorted(
        ((identity, float(np.linalg.norm(feature - centroid))) for identity, centroid in centroids.items()),
        key=lambda x: x[1],
    )
    identity, best = dists[0]
    margin = (dists[1][1] - best) if len(dists) > 1 else float("inf")
    if best > max_distance:
        return UNKNOWN_ID, best, margin, "far_from_centroid"
    if margin < min_margin:
        return UNKNOWN_ID, best, margin, "low_margin"
    return identity, best, margin, ""


def aggregate_fragment_features(
    df: pd.DataFrame,
    *,
    csv_dir: Path,
    feature_size: int = 32,
) -> dict[int, np.ndarray]:
    """Average crop features per fragment."""
    feats: dict[int, list[np.ndarray]] = {}
    for _, row in df.iterrows():
        crop_path = str(row.get("crop_path", ""))
        if not crop_path:
            continue
        cp = _as_path(crop_path, base=csv_dir)
        if not cp.exists():
            continue
        import cv2

        crop = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        mask = _load_mask_for_candidate(row, csv_dir=csv_dir)
        feat = crop_masked_feature(crop, mask, feature_size=feature_size)
        feats.setdefault(int(row["fragment_id"]), []).append(feat)
    out: dict[int, np.ndarray] = {}
    for fid, values in feats.items():
        arr = np.mean(np.vstack(values), axis=0)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-6:
            arr = arr / norm
        out[fid] = arr.astype(np.float32)
    return out


def assign_fragment_identities(
    fragment_features: dict[int, np.ndarray],
    centroids: dict[str, np.ndarray],
    *,
    min_margin: float = 0.05,
    max_distance: float = 1.25,
) -> dict[int, tuple[str, float | None, float | None, str]]:
    return {
        fid: classify_fragment_feature(
            feat,
            centroids,
            min_margin=min_margin,
            max_distance=max_distance,
        )
        for fid, feat in fragment_features.items()
    }


def evaluate_curated_fragments(
    fragment_features: dict[int, np.ndarray],
    fragments: pd.DataFrame,
    *,
    min_margin: float,
    max_distance: float,
) -> dict:
    """
    Report a conservative leave-one-fragment-out diagnostic on curated IDs.

    This is intentionally small: the curated set may have only a handful of
    fragments, but it still catches the obvious "all identities collapse
    together" failure before applying the model to uncurated data.
    """
    labels_by_fragment: dict[int, str] = {}
    for fid, group in fragments.groupby("fragment_id"):
        ids = group["id"].dropna().astype(str)
        ids = ids[ids.isin(ID_LABELS)]
        if ids.empty:
            continue
        labels_by_fragment[int(fid)] = str(ids.mode().iloc[0])

    confusion: dict[str, dict[str, int]] = {
        label: {pred: 0 for pred in (*ID_LABELS, UNKNOWN_ID)}
        for label in ID_LABELS
    }
    margins: list[float] = []
    n_eval = 0
    n_correct = 0
    for fid, true_id in labels_by_fragment.items():
        train: dict[str, list[np.ndarray]] = {label: [] for label in ID_LABELS}
        for other_fid, other_id in labels_by_fragment.items():
            if other_fid == fid or other_fid not in fragment_features:
                continue
            train[other_id].append(fragment_features[other_fid])
        centroids = fit_identity_centroids(train, min_samples=1)
        if fid not in fragment_features or not centroids:
            continue
        pred, _dist, margin, _reason = classify_fragment_feature(
            fragment_features[fid],
            centroids,
            min_margin=min_margin,
            max_distance=max_distance,
        )
        confusion[true_id][pred] += 1
        if margin is not None and np.isfinite(margin):
            margins.append(float(margin))
        n_eval += 1
        if pred == true_id:
            n_correct += 1

    return {
        "n_eval_fragments": n_eval,
        "accuracy": (n_correct / n_eval) if n_eval else None,
        "confusion": confusion,
        "margin_min": min(margins) if margins else None,
        "margin_median": float(np.median(margins)) if margins else None,
        "margin_max": max(margins) if margins else None,
    }


def enforce_one_identity_per_frame(
    df: pd.DataFrame,
    assignments: dict[int, tuple[str, float | None, float | None, str]],
) -> dict[int, tuple[str, float | None, float | None, str]]:
    """Resolve same-frame ID collisions by keeping the lower-distance fragment."""
    out = dict(assignments)
    for _frame, g in df.groupby("frame"):
        by_id: dict[str, list[int]] = {}
        for fid in sorted(set(g["fragment_id"].astype(int))):
            identity = out.get(fid, (UNKNOWN_ID, None, None, "missing"))[0]
            if identity in ID_LABELS:
                by_id.setdefault(identity, []).append(fid)
        for identity, fids in by_id.items():
            if len(fids) <= 1:
                continue
            best_fid = min(
                fids,
                key=lambda fid: out[fid][1] if out[fid][1] is not None else float("inf"),
            )
            for fid in fids:
                if fid == best_fid:
                    continue
                _old_id, dist, margin, _reason = out[fid]
                out[fid] = (UNKNOWN_ID, dist, margin, "same_frame_identity_conflict")
    return out


def _write_candidate_artifacts(
    frame_bgr: np.ndarray,
    component_mask: np.ndarray,
    cand: BlobCandidate,
    *,
    out_dir: Path,
) -> BlobCandidate:
    import cv2

    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = cand.bbox_x, cand.bbox_y, cand.bbox_w, cand.bbox_h
    crop = frame_bgr[y : y + h, x : x + w]
    mask_crop = component_mask[y : y + h, x : x + w]
    mask_name = f"{cand.candidate_id}.png"
    crop_name = f"{cand.candidate_id}.png"
    cv2.imwrite(str(masks_dir / mask_name), mask_crop)
    cv2.imwrite(str(crops_dir / crop_name), crop)
    cand.mask_path = str(Path("masks") / mask_name)
    cand.crop_path = str(Path("crops") / crop_name)
    return cand


def _draw_preview(frame: np.ndarray, candidates: list[BlobCandidate]) -> np.ndarray:
    import cv2

    out = frame.copy()
    for cand in candidates:
        color = (0, 220, 0) if cand.source == "bgsub" else (255, 160, 0)
        cv2.rectangle(
            out,
            (cand.bbox_x, cand.bbox_y),
            (cand.bbox_x + cand.bbox_w, cand.bbox_y + cand.bbox_h),
            color,
            2,
        )
        cv2.putText(
            out,
            f"{cand.near_sleap_det}:{cand.quality_score:.0f}",
            (cand.bbox_x, max(15, cand.bbox_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _read_sam_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SAM manifest not found: {path}")
    return pd.read_csv(path)


def _sam_candidates_for_frame(
    sam_df: pd.DataFrame | None,
    *,
    frame: int,
    manifest_dir: Path,
    min_area: int,
    max_area: int,
) -> list[tuple[BlobCandidate, np.ndarray]]:
    if sam_df is None or sam_df.empty:
        return []
    import cv2

    out: list[tuple[BlobCandidate, np.ndarray]] = []
    rows = sam_df[sam_df["frame"].astype(int) == int(frame)]
    for idx, row in rows.iterrows():
        p = _as_path(str(row["mask_path"]), base=manifest_dir)
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        source = str(row.get("source", "sam"))
        score = _safe_float(row.get("score"), default=float("nan"))
        for cand in component_candidates(
            mask,
            frame=frame,
            source=source,
            candidate_prefix=f"sam{idx}",
            min_area=min_area,
            max_area=max_area,
            sam_score=score if np.isfinite(score) else None,
        ):
            out.append((cand, (mask > 0).astype(np.uint8) * 255))
    return out


def mine_blobs(
    *,
    experiment: str,
    video_path: Path,
    slp_path: Path | None,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    every: int = 1,
    sam_manifest: Path | None = None,
    threshold: int = 24,
    min_area: int = 80,
    max_area: int = 8000,
    max_sleap_distance_px: float = 80.0,
    max_candidates_per_sleap: int = 1,
    background_samples: int = 80,
    preview_video: bool = True,
    no_sleap_filter: bool = False,
    pure_bgsub: bool = False,
) -> dict:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    if pure_bgsub:
        no_sleap_filter = True
        sam_manifest = None
    if no_sleap_filter:
        sleap_by_frame: dict[int, list[tuple[float, float, float]]] = {}
    elif slp_path is not None:
        sleap_by_frame = _load_top_sleap_points(slp_path, start_frame=start_frame, end_frame=end_frame)
    else:
        raise RuntimeError("SLP path is required unless --no-sleap-filter or --pure-bgsub is used.")
    background = estimate_median_background(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        max_samples=background_samples,
    )
    sam_df = _read_sam_manifest(sam_manifest) if sam_manifest else None
    sam_dir = sam_manifest.parent if sam_manifest else out_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 166.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if preview_video:
        writer = cv2.VideoWriter(
            str(out_dir / "blob_candidates_preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, every),
            (width, height),
        )

    rows: list[BlobCandidate] = []
    try:
        for frame in range(start_frame, end_frame + 1, every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                continue
            fg_mask = foreground_mask_from_frame(img, background, threshold=threshold)
            bg_cands = component_candidates(
                fg_mask,
                frame=frame,
                source="bgsub",
                candidate_prefix="bg",
                min_area=min_area,
                max_area=max_area,
            )
            frame_candidates: list[tuple[BlobCandidate, np.ndarray]] = [
                (cand, fg_mask) for cand in bg_cands
            ]
            frame_candidates.extend(
                _sam_candidates_for_frame(
                    sam_df,
                    frame=frame,
                    manifest_dir=sam_dir,
                    min_area=min_area,
                    max_area=max_area,
                )
            )

            filtered_pairs: list[tuple[BlobCandidate, np.ndarray]] = []
            if no_sleap_filter:
                filtered = suppress_overlapping_candidates(
                    [cand for cand, _mask in frame_candidates],
                    same_det_only=False,
                )
            else:
                filtered = filter_candidates_by_sleap(
                    [cand for cand, _mask in frame_candidates],
                    sleap_by_frame.get(frame, []),
                    max_distance_px=max_sleap_distance_px,
                )
                filtered = suppress_overlapping_candidates(filtered)
                filtered = keep_best_candidate_per_sleap_detection(
                    filtered,
                    max_per_detection=max_candidates_per_sleap,
                )
            keep_ids = {id(c) for c in filtered}
            for cand, mask in frame_candidates:
                if id(cand) not in keep_ids:
                    continue
                cand.candidate_id = f"{experiment}_f{frame}_{len(rows):06d}"
                saved = _write_candidate_artifacts(img, mask, cand, out_dir=out_dir)
                rows.append(saved)
                filtered_pairs.append((saved, mask))
            if writer is not None:
                writer.write(_draw_preview(img, [cand for cand, _m in filtered_pairs]))
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    csv_path = out_dir / "blob_candidates.csv"
    pd.DataFrame([asdict(r) for r in rows]).to_csv(csv_path, index=False)
    return {
        "csv_path": str(csv_path),
        "n_candidates": len(rows),
        "frames": [start_frame, end_frame],
        "mode": "pure_bgsub" if pure_bgsub else ("bgsub_no_sleap_filter" if no_sleap_filter else "sleap_filtered"),
        "preview_video": str(out_dir / "blob_candidates_preview.mp4") if preview_video else None,
    }


def review_candidates(
    *,
    candidates_csv: Path,
    output_csv: Path | None = None,
    video_path: Path | None = None,
    full_frame: bool = True,
    play_fps: float = 2.0,
    display_scale: float = 1.0,
) -> dict:
    """Keyboard reviewer for candidate blobs."""
    if full_frame and video_path is not None:
        return review_candidates_full_frame(
            candidates_csv=candidates_csv,
            output_csv=output_csv,
            video_path=video_path,
            play_fps=play_fps,
            display_scale=display_scale,
        )
    return review_candidates_crops(candidates_csv=candidates_csv, output_csv=output_csv)


def review_candidates_crops(
    *,
    candidates_csv: Path,
    output_csv: Path | None = None,
) -> dict:
    """Keyboard reviewer for candidate crops when no video is available."""
    import cv2

    df = pd.read_csv(candidates_csv)
    csv_dir = candidates_csv.parent
    if output_csv is None:
        output_csv = candidates_csv.with_name("blob_candidates.reviewed.csv")
    idx = 0
    instructions = "1/2/3=ID, u=unknown, r=reject, n/p=next/prev, s=save, q=quit"
    while 0 <= idx < len(df):
        row = df.iloc[idx]
        crop_path = _as_path(str(row.get("crop_path", "")), base=csv_dir)
        img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((160, 240, 3), dtype=np.uint8)
        canvas = cv2.copyMakeBorder(img, 60, 30, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        title = (
            f"{idx + 1}/{len(df)} f={int(row['frame'])} "
            f"status={row.get('status', '')} id={row.get('id', '')}"
        )
        cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, instructions, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.imshow("top-camera blob reviewer", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("1"), ord("2"), ord("3")):
            df.at[df.index[idx], "status"] = "accept"
            df.at[df.index[idx], "id"] = f"ID_{chr(key)}"
            df.at[df.index[idx], "bad_reason"] = ""
            idx += 1
        elif key == ord("u"):
            df.at[df.index[idx], "status"] = "unknown"
            df.at[df.index[idx], "id"] = UNKNOWN_ID
            df.at[df.index[idx], "bad_reason"] = ""
            idx += 1
        elif key == ord("r"):
            df.at[df.index[idx], "status"] = "reject"
            df.at[df.index[idx], "id"] = ""
            df.at[df.index[idx], "bad_reason"] = "manual_reject"
            idx += 1
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("s"):
            df.to_csv(output_csv, index=False)
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    df.to_csv(output_csv, index=False)
    return {"reviewed_csv": str(output_csv), "n_rows": int(len(df))}


def _ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in (
        ("status", "review"),
        ("id", ""),
        ("bad_reason", ""),
        ("id_source", ""),
    ):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
    return df


def _candidate_color(row: pd.Series, *, selected: bool = False) -> tuple[int, int, int]:
    status = str(row.get("status", "review"))
    identity = str(row.get("id", ""))
    if selected:
        return (0, 255, 255)
    if status == "reject":
        return (60, 60, 220)
    if identity == "ID_1":
        return (180, 119, 31)
    if identity == "ID_2":
        return (14, 127, 255)
    if identity == "ID_3":
        return (44, 160, 44)
    if identity == UNKNOWN_ID or status == "unknown":
        return (180, 180, 180)
    return (0, 220, 0)


def _zoom_view_transform(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    *,
    view_zoom: float,
    view_center_xy: tuple[float, float] | None,
    display_scale: float,
) -> dict[str, float]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    zoom = max(1.0, float(view_zoom))
    crop_w = max(1, min(w, int(round(w / zoom))))
    crop_h = max(1, min(h, int(round(h / zoom))))
    cx = float(view_center_xy[0]) if view_center_xy is not None else w / 2.0
    cy = float(view_center_xy[1]) if view_center_xy is not None else h / 2.0
    crop_x = int(round(cx - crop_w / 2))
    crop_y = int(round(cy - crop_h / 2))
    crop_x = max(0, min(crop_x, max(0, w - crop_w)))
    crop_y = max(0, min(crop_y, max(0, h - crop_h)))
    return {
        "crop_x": float(crop_x),
        "crop_y": float(crop_y),
        "crop_w": float(crop_w),
        "crop_h": float(crop_h),
        "frame_w": float(w),
        "frame_h": float(h),
        "display_scale": float(display_scale),
        "view_zoom": zoom,
        "scale_x": float(w / crop_w),
        "scale_y": float(h / crop_h),
    }


def _frame_to_display_xy(
    x: float,
    y: float,
    transform: dict[str, float],
) -> tuple[int, int]:
    sx = transform["frame_w"] / max(1.0, transform["crop_w"])
    sy = transform["frame_h"] / max(1.0, transform["crop_h"])
    scale = transform["display_scale"]
    return (
        int(round((x - transform["crop_x"]) * sx * scale)),
        int(round((y - transform["crop_y"]) * sy * scale)),
    )


def _draw_full_frame_review(
    frame_bgr: np.ndarray,
    df: pd.DataFrame,
    frame_indices: list[int],
    selected_row_index: int | None,
    *,
    csv_dir: Path,
    display_scale: float,
    playing: bool,
    play_fps: float,
    edit_mode: str,
    brush_radius: int,
    smart_radius: int,
    smart_tolerance: float,
    view_zoom: float,
    view_center_xy: tuple[float, float] | None,
    preview_mask: np.ndarray | None = None,
    mouse_xy: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], dict[str, float]]:
    import cv2

    canvas = frame_bgr.copy()
    bbox_display: dict[int, tuple[int, int, int, int]] = {}
    overlay = canvas.copy()
    for label_idx, row_idx in enumerate(frame_indices, start=1):
        row = df.loc[row_idx]
        x, y, w, h = _candidate_bbox(row)
        selected = row_idx == selected_row_index
        color = _candidate_color(row, selected=selected)
        full_mask = candidate_mask_to_full_frame(row, csv_dir=csv_dir, frame_shape=canvas.shape)
        if full_mask.any():
            overlay[full_mask > 0] = color
        thickness = 4 if selected else 2
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        identity = str(row.get("id", "")) or str(row.get("status", "review"))
        label = f"{label_idx}:{identity}"
        cv2.putText(
            canvas,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0)
    erase_modes = {"erase", "smart_erase"}
    if preview_mask is not None and preview_mask.shape[:2] == canvas.shape[:2] and preview_mask.any():
        preview_color = (0, 0, 255) if edit_mode in erase_modes else (0, 180, 255)
        preview_overlay = canvas.copy()
        preview_overlay[preview_mask > 0] = preview_color
        canvas = cv2.addWeighted(preview_overlay, 0.42, canvas, 0.58, 0.0)
        contours, _hier = cv2.findContours((preview_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, preview_color, 2, cv2.LINE_AA)
    if mouse_xy is not None and edit_mode in {"add", "erase", "fill", "smart_erase"}:
        mx, my = mouse_xy
        if 0 <= mx < canvas.shape[1] and 0 <= my < canvas.shape[0]:
            radius = int(smart_radius if edit_mode in {"fill", "smart_erase"} else brush_radius)
            cursor_color = (0, 0, 255) if edit_mode in erase_modes else (0, 180, 255)
            cv2.circle(canvas, (mx, my), radius, cursor_color, 1, cv2.LINE_AA)

    transform = _zoom_view_transform(
        canvas.shape,
        view_zoom=view_zoom,
        view_center_xy=view_center_xy,
        display_scale=display_scale,
    )
    cx, cy = int(transform["crop_x"]), int(transform["crop_y"])
    cw, ch = int(transform["crop_w"]), int(transform["crop_h"])
    if cw != canvas.shape[1] or ch != canvas.shape[0]:
        canvas = canvas[cy : cy + ch, cx : cx + cw]
        canvas = cv2.resize(
            canvas,
            (int(transform["frame_w"]), int(transform["frame_h"])),
            interpolation=cv2.INTER_LINEAR,
        )
    if display_scale != 1.0:
        canvas = cv2.resize(
            canvas,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )

    frame_no = int(df.loc[frame_indices[0], "frame"]) if frame_indices else -1
    selected_text = "-"
    if selected_row_index is not None:
        row = df.loc[selected_row_index]
        selected_text = (
            f"{frame_indices.index(selected_row_index) + 1}/"
            f"{len(frame_indices)} {row['candidate_id']}"
        )
    header = (
        f"frame={frame_no} candidates={len(frame_indices)} selected={selected_text} "
        f"{'PLAY' if playing else 'PAUSE'} {play_fps:g}fps "
        f"mode={edit_mode} zoom={view_zoom:g}x brush={brush_radius}px smart={smart_radius}px/{smart_tolerance:g}"
    )
    help1 = "click/[/]/tab select | 1/2/3 ID | u unknown | d/r reject | n/p frame | space play | s save | q quit"
    help2 = "wheel zoom | f smart fill | x smart erase | a add | e erase | z undo | ,/. brush/fill size"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 82), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, help1, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, help2, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    for row_idx in frame_indices:
        row = df.loc[row_idx]
        x, y, w, h = _candidate_bbox(row)
        x0, y0 = _frame_to_display_xy(float(x), float(y), transform)
        x1, y1 = _frame_to_display_xy(float(x + w), float(y + h), transform)
        bbox_display[row_idx] = (
            x0,
            y0,
            max(0, x1 - x0),
            max(0, y1 - y0),
        )
    return canvas, bbox_display, transform


def review_candidates_full_frame(
    *,
    candidates_csv: Path,
    video_path: Path,
    output_csv: Path | None = None,
    play_fps: float = 2.0,
    display_scale: float = 1.0,
) -> dict:
    """Full-frame blob reviewer with mask/box overlays and slow playback."""
    import cv2

    using_default_output = output_csv is None
    if output_csv is None:
        output_csv = candidates_csv.with_name("blob_candidates.reviewed.csv")
    should_resume_reviewed = (
        using_default_output
        and candidates_csv.name == "blob_candidates.csv"
        and output_csv.exists()
        and output_csv != candidates_csv
    )
    read_csv = output_csv if should_resume_reviewed else candidates_csv
    df = _ensure_review_columns(pd.read_csv(read_csv))
    csv_dir = candidates_csv.parent

    frames = sorted(int(f) for f in df["frame"].dropna().unique())
    if not frames:
        df.to_csv(output_csv, index=False)
        return {"reviewed_csv": str(output_csv), "n_rows": 0, "mode": "full_frame"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for review: {video_path}")

    state = {
        "frame_pos": 0,
        "selected_pos": 0,
        "playing": False,
        "bboxes": {},
        "window_sized": False,
        "edit_mode": "select",
        "brush_radius": 14,
        "smart_radius": 90,
        "smart_tolerance": 30.0,
        "mouse_down": False,
        "current_frame_bgr": None,
        "mouse_xy": None,
        "preview_mask": None,
        "undo_stack": [],
        "view_zoom": 1.0,
        "view_center_xy": None,
        "view_transform": None,
        "mask_edits_made": False,
    }
    window_name = "top-camera blob reviewer"

    def _save_review_state() -> None:
        df.to_csv(output_csv, index=False)
        if using_default_output and bool(state["mask_edits_made"]) and output_csv != candidates_csv:
            df.to_csv(candidates_csv, index=False)

    def _frame_indices() -> list[int]:
        frame = frames[state["frame_pos"]]
        return list(df.index[df["frame"].astype(int) == frame])

    def _selected_row_index() -> int | None:
        idxs = _frame_indices()
        if not idxs:
            return None
        state["selected_pos"] = max(0, min(int(state["selected_pos"]), len(idxs) - 1))
        return idxs[state["selected_pos"]]

    def _move_frame(delta: int) -> None:
        state["frame_pos"] = max(0, min(int(state["frame_pos"]) + delta, len(frames) - 1))
        state["selected_pos"] = 0
        state["mouse_down"] = False
        state["preview_mask"] = None

    def _move_selection(delta: int) -> None:
        idxs = _frame_indices()
        if not idxs:
            state["selected_pos"] = 0
        else:
            state["selected_pos"] = (int(state["selected_pos"]) + delta) % len(idxs)
        state["preview_mask"] = None

    def _label_selected(status: str, identity: str, reason: str = "") -> None:
        row_idx = _selected_row_index()
        if row_idx is None:
            return
        df.at[row_idx, "status"] = status
        df.at[row_idx, "id"] = identity
        df.at[row_idx, "bad_reason"] = reason
        df.at[row_idx, "id_source"] = "manual" if identity in ID_LABELS else ""
        _move_selection(1)

    def _display_to_frame(display_x: int, display_y: int) -> tuple[int, int] | None:
        img = state.get("current_frame_bgr")
        if img is None:
            return None
        transform = state.get("view_transform")
        if transform is None:
            transform = _zoom_view_transform(
                img.shape,
                view_zoom=float(state["view_zoom"]),
                view_center_xy=state["view_center_xy"],
                display_scale=display_scale,
            )
        orig_x = int(round(
            transform["crop_x"]
            + (display_x / max(transform["display_scale"], 1e-6))
            * transform["crop_w"]
            / max(1.0, transform["frame_w"])
        ))
        orig_y = int(round(
            transform["crop_y"]
            + (display_y / max(transform["display_scale"], 1e-6))
            * transform["crop_h"]
            / max(1.0, transform["frame_h"])
        ))
        if orig_x < 0 or orig_y < 0 or orig_x >= img.shape[1] or orig_y >= img.shape[0]:
            return None
        return orig_x, orig_y

    def _push_undo(row_idx: int, img: np.ndarray) -> None:
        stack = state["undo_stack"]
        stack.append(
            {
                "row_idx": row_idx,
                "row": df.loc[row_idx].copy(),
                "mask": candidate_mask_to_full_frame(df.loc[row_idx], csv_dir=csv_dir, frame_shape=img.shape),
            }
        )
        del stack[:-30]

    def _undo_last_edit() -> None:
        stack = state["undo_stack"]
        if not stack:
            return
        snap = stack.pop()
        row_idx = int(snap["row_idx"])
        if row_idx not in df.index:
            return
        row_before = snap["row"]
        for col, value in row_before.items():
            df.at[row_idx, col] = value
        img = state.get("current_frame_bgr")
        if img is not None:
            save_edited_candidate_mask(df, row_idx, snap["mask"], img, csv_dir=csv_dir)
            for col, value in row_before.items():
                if col in df.columns:
                    df.at[row_idx, col] = value
            state["mask_edits_made"] = True
        state["preview_mask"] = None

    def _paint_selected(frame_x: int, frame_y: int) -> None:
        row_idx = _selected_row_index()
        img = state.get("current_frame_bgr")
        if row_idx is None or img is None:
            return
        if frame_x < 0 or frame_y < 0 or frame_x >= img.shape[1] or frame_y >= img.shape[0]:
            return
        full_mask = candidate_mask_to_full_frame(df.loc[row_idx], csv_dir=csv_dir, frame_shape=img.shape)
        value = 255 if state["edit_mode"] == "add" else 0
        cv2.circle(full_mask, (frame_x, frame_y), int(state["brush_radius"]), value, -1)
        save_edited_candidate_mask(df, row_idx, full_mask, img, csv_dir=csv_dir)
        state["mask_edits_made"] = True

    def _smart_edit_selected(frame_x: int, frame_y: int, *, erase: bool) -> None:
        row_idx = _selected_row_index()
        img = state.get("current_frame_bgr")
        if row_idx is None or img is None:
            return
        if frame_x < 0 or frame_y < 0 or frame_x >= img.shape[1] or frame_y >= img.shape[0]:
            return
        full_mask = candidate_mask_to_full_frame(df.loc[row_idx], csv_dir=csv_dir, frame_shape=img.shape)
        fill = smart_fill_region_from_click(
            img,
            (frame_x, frame_y),
            search_radius=int(state["smart_radius"]),
            tolerance=float(state["smart_tolerance"]),
        )
        full_mask[fill > 0] = 0 if erase else 255
        save_edited_candidate_mask(df, row_idx, full_mask, img, csv_dir=csv_dir)
        state["mask_edits_made"] = True

    left_arrow_keys = {81, 2424832, 65361}
    right_arrow_keys = {83, 2555904, 65363}

    def _mouse(event, x, y, _flags, _param):
        frame_xy = _display_to_frame(x, y)
        state["mouse_xy"] = frame_xy
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(_flags) if hasattr(cv2, "getMouseWheelDelta") else _flags
            current = float(state["view_zoom"])
            factor = 1.25 if delta > 0 else 0.8
            state["view_zoom"] = max(1.0, min(8.0, current * factor))
            if frame_xy is not None:
                state["view_center_xy"] = frame_xy
            if state["view_zoom"] <= 1.001:
                state["view_zoom"] = 1.0
                state["view_center_xy"] = None
            return
        if event == cv2.EVENT_LBUTTONUP:
            state["mouse_down"] = False
            return
        if state["edit_mode"] in {"fill", "smart_erase"}:
            img = state.get("current_frame_bgr")
            if frame_xy is not None and img is not None:
                state["preview_mask"] = smart_fill_region_from_click(
                    img,
                    frame_xy,
                    search_radius=int(state["smart_radius"]),
                    tolerance=float(state["smart_tolerance"]),
                )
            if event == cv2.EVENT_LBUTTONDOWN:
                row_idx = _selected_row_index()
                img = state.get("current_frame_bgr")
                if row_idx is not None and img is not None:
                    _push_undo(row_idx, img)
                if frame_xy is not None:
                    _smart_edit_selected(*frame_xy, erase=state["edit_mode"] == "smart_erase")
            return
        if state["edit_mode"] in {"add", "erase"}:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["mouse_down"] = True
                row_idx = _selected_row_index()
                img = state.get("current_frame_bgr")
                if row_idx is not None and img is not None:
                    _push_undo(row_idx, img)
                if frame_xy is not None:
                    _paint_selected(*frame_xy)
            elif event == cv2.EVENT_MOUSEMOVE and state["mouse_down"]:
                if frame_xy is not None:
                    _paint_selected(*frame_xy)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for pos, row_idx in enumerate(_frame_indices()):
            bx, by, bw, bh = state["bboxes"].get(row_idx, (0, 0, 0, 0))
            if bx <= x <= bx + bw and by <= y <= by + bh:
                state["selected_pos"] = pos
                break

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.setMouseCallback(window_name, _mouse)
    try:
        while True:
            frame = frames[state["frame_pos"]]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((720, 1280, 3), dtype=np.uint8)
            state["current_frame_bgr"] = img
            idxs = _frame_indices()
            selected = _selected_row_index()
            canvas, bboxes, transform = _draw_full_frame_review(
                img,
                df,
                idxs,
                selected,
                csv_dir=csv_dir,
                display_scale=display_scale,
                playing=bool(state["playing"]),
                play_fps=play_fps,
                edit_mode=str(state["edit_mode"]),
                brush_radius=int(state["brush_radius"]),
                smart_radius=int(state["smart_radius"]),
                smart_tolerance=float(state["smart_tolerance"]),
                view_zoom=float(state["view_zoom"]),
                view_center_xy=state["view_center_xy"],
                preview_mask=state["preview_mask"],
                mouse_xy=state["mouse_xy"],
            )
            state["bboxes"] = bboxes
            state["view_transform"] = transform
            if not state["window_sized"]:
                cv2.resizeWindow(window_name, canvas.shape[1], canvas.shape[0])
                state["window_sized"] = True
            cv2.imshow(window_name, canvas)
            delay = max(1, int(round(1000.0 / max(0.1, play_fps)))) if state["playing"] else 30
            key = cv2.waitKeyEx(delay)

            if state["playing"] and key < 0:
                _move_frame(1)
                if state["frame_pos"] == len(frames) - 1:
                    state["playing"] = False
                continue

            if key < 0:
                continue
            if key in (ord("1"), ord("2"), ord("3")):
                _label_selected("accept", f"ID_{chr(key)}")
            elif key == ord("u"):
                _label_selected("unknown", UNKNOWN_ID)
            elif key in (ord("r"), ord("d")):
                _label_selected("reject", "", "manual_reject")
            elif key == ord("f"):
                state["edit_mode"] = "fill"
                state["playing"] = False
            elif key == ord("x"):
                state["edit_mode"] = "smart_erase"
                state["playing"] = False
            elif key == ord("a"):
                state["edit_mode"] = "add"
                state["playing"] = False
                state["preview_mask"] = None
            elif key == ord("e"):
                state["edit_mode"] = "erase"
                state["playing"] = False
                state["preview_mask"] = None
            elif key == ord("z"):
                _undo_last_edit()
            elif key == ord("v"):
                state["edit_mode"] = "select"
                state["mouse_down"] = False
                state["preview_mask"] = None
            elif key in (ord("."), ord(">")):
                if state["edit_mode"] in {"fill", "smart_erase"}:
                    state["smart_radius"] = min(260, int(state["smart_radius"]) + 10)
                    state["preview_mask"] = None
                else:
                    state["brush_radius"] = min(80, int(state["brush_radius"]) + 3)
            elif key in (ord(","), ord("<")):
                if state["edit_mode"] in {"fill", "smart_erase"}:
                    state["smart_radius"] = max(12, int(state["smart_radius"]) - 10)
                    state["preview_mask"] = None
                else:
                    state["brush_radius"] = max(2, int(state["brush_radius"]) - 3)
            elif key in (ord("]"), ord("\t")):
                _move_selection(1)
            elif key == ord("["):
                _move_selection(-1)
            elif key == ord("n") or key in right_arrow_keys:
                _move_frame(1)
            elif key == ord("p") or key in left_arrow_keys:
                _move_frame(-1)
            elif key == ord(" "):
                state["playing"] = not state["playing"]
            elif key in (ord("+"), ord("=")):
                play_fps = min(30.0, play_fps * 1.5)
            elif key in (ord("-"), ord("_")):
                play_fps = max(0.2, play_fps / 1.5)
            elif key == ord("s"):
                _save_review_state()
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    _save_review_state()
    return {
        "reviewed_csv": str(output_csv),
        "n_rows": int(len(df)),
        "mode": "full_frame",
        "video": str(video_path),
    }


def train_identity_model(
    *,
    reviewed_csv: Path,
    output_model: Path,
    output_report: Path | None = None,
    min_samples: int = 5,
    feature_size: int = 32,
) -> dict:
    df = pd.read_csv(reviewed_csv)
    csv_dir = reviewed_csv.parent
    accepted = df[(df["status"] == "accept") & (df["id"].isin(ID_LABELS))].copy()
    training_rows, merge_report = merge_same_frame_same_id_candidates(accepted, csv_dir=csv_dir)
    fragments = link_blob_fragments(training_rows, csv_dir=csv_dir)
    features = aggregate_fragment_features(fragments, csv_dir=csv_dir, feature_size=feature_size)
    features_by_id: dict[str, list[np.ndarray]] = {label: [] for label in ID_LABELS}
    for fid, feat in features.items():
        ids = fragments[fragments["fragment_id"] == fid]["id"].dropna().astype(str)
        ids = ids[ids.isin(ID_LABELS)]
        if ids.empty:
            continue
        identity = ids.mode().iloc[0]
        features_by_id[identity].append(feat)
    centroids = fit_identity_centroids(features_by_id, min_samples=min_samples)
    diagnostics = evaluate_curated_fragments(features, fragments, min_margin=0.05, max_distance=1.25)
    model = {
        "version": 1,
        "centroids": centroids,
        "feature_size": feature_size,
        "id_labels": list(ID_LABELS),
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)

    report = {
        "model_path": str(output_model),
        "n_reviewed_rows": int(len(df)),
        "n_accepted_rows": int(len(accepted)),
        "n_training_rows_after_same_id_merge": int(len(training_rows)),
        "same_id_merge": merge_report,
        "n_fragments": int(fragments["fragment_id"].nunique()) if not fragments.empty else 0,
        "samples_per_id": {k: len(v) for k, v in features_by_id.items()},
        "trained_identities": sorted(centroids),
        "curated_fragment_eval": diagnostics,
    }
    if output_report is None:
        output_report = output_model.with_name("top_id_model_report.json")
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def bootstrap_identity_labels(
    *,
    candidates_csv: Path,
    output_csv: Path | None = None,
    model_path: Path | None = None,
    min_seed_samples: int = 1,
    min_margin: float = 0.08,
    max_distance: float = 1.10,
    feature_size: int = 32,
    auto_accept: bool = True,
    include_bootstrap_labels: bool = False,
) -> dict:
    """
    Pre-label review candidates from curated seeds or an existing model.

    Manual labels are rows with status=accept, id in ID_*, and id_source not
    equal to "bootstrap". Automatic predictions get id_source="bootstrap" so a
    later bootstrap round can avoid training on its own guesses unless requested.
    """
    df = _ensure_review_columns(pd.read_csv(candidates_csv))
    csv_dir = candidates_csv.parent
    if output_csv is None:
        output_csv = candidates_csv.with_name("blob_candidates.bootstrapped.csv")

    usable_mask = df["status"].fillna("review").isin(["accept", "review", "unknown", ""])
    usable = df[usable_mask].copy()
    usable["__source_row"] = usable.index
    fragments = link_blob_fragments(usable, csv_dir=csv_dir)
    if fragments.empty:
        df.to_csv(output_csv, index=False)
        return {
            "output_csv": str(output_csv),
            "n_rows": int(len(df)),
            "n_fragments": 0,
            "n_auto_labeled_rows": 0,
            "n_unknown_rows": 0,
            "trained_identities": [],
        }

    model = joblib.load(model_path) if model_path is not None else None
    fs = int(feature_size if model is None else model.get("feature_size", feature_size))
    fragment_features = aggregate_fragment_features(fragments, csv_dir=csv_dir, feature_size=fs)

    if model is not None:
        centroids = model.get("centroids", {})
    else:
        seed_rows = fragments[
            (fragments["status"] == "accept")
            & (fragments["id"].isin(ID_LABELS))
        ].copy()
        if not include_bootstrap_labels:
            seed_rows = seed_rows[seed_rows["id_source"].fillna("").astype(str) != "bootstrap"]
        labels_by_fragment: dict[int, str] = {}
        for fid, group in seed_rows.groupby("fragment_id"):
            ids = group["id"].dropna().astype(str)
            ids = ids[ids.isin(ID_LABELS)]
            if not ids.empty:
                labels_by_fragment[int(fid)] = str(ids.mode().iloc[0])
        features_by_id: dict[str, list[np.ndarray]] = {label: [] for label in ID_LABELS}
        for fid, identity in labels_by_fragment.items():
            if fid in fragment_features:
                features_by_id[identity].append(fragment_features[fid])
        centroids = fit_identity_centroids(features_by_id, min_samples=min_seed_samples)

    assignments = assign_fragment_identities(
        fragment_features,
        centroids,
        min_margin=min_margin,
        max_distance=max_distance,
    )
    assignments = enforce_one_identity_per_frame(fragments, assignments)

    for col, default in (
        ("fragment_id", -1),
        ("suggested_id", ""),
        ("suggested_distance", np.nan),
        ("suggested_margin", np.nan),
        ("suggested_unknown_reason", ""),
        ("review_priority", ""),
    ):
        if col not in df.columns:
            df[col] = default
    df["suggested_distance"] = pd.to_numeric(df["suggested_distance"], errors="coerce")
    df["suggested_margin"] = pd.to_numeric(df["suggested_margin"], errors="coerce")

    manual_rows = set(
        df.index[
            (df["status"] == "accept")
            & (df["id"].isin(ID_LABELS))
            & (df["id_source"].fillna("").astype(str) != "bootstrap")
        ]
    )
    n_auto = 0
    n_unknown = 0
    n_preserved = 0
    for fid, group in fragments.groupby("fragment_id"):
        fid = int(fid)
        identity, dist, margin, reason = assignments.get(fid, (UNKNOWN_ID, None, None, "no_feature"))
        source_rows = [int(i) for i in group["__source_row"].tolist()]
        has_manual = any(i in manual_rows for i in source_rows)
        for row_idx in source_rows:
            df.at[row_idx, "fragment_id"] = fid
            df.at[row_idx, "suggested_id"] = identity
            df.at[row_idx, "suggested_distance"] = np.nan if dist is None else float(dist)
            df.at[row_idx, "suggested_margin"] = np.nan if margin is None else float(margin)
            df.at[row_idx, "suggested_unknown_reason"] = reason
            if row_idx in manual_rows:
                df.at[row_idx, "review_priority"] = "manual_seed"
                n_preserved += 1
                continue
            if has_manual:
                df.at[row_idx, "review_priority"] = "same_fragment_as_manual_seed"
            if identity in ID_LABELS and reason == "":
                df.at[row_idx, "id"] = identity
                df.at[row_idx, "id_source"] = "bootstrap"
                df.at[row_idx, "bad_reason"] = ""
                df.at[row_idx, "status"] = "accept" if auto_accept else "review"
                df.at[row_idx, "review_priority"] = "check_auto_label"
                n_auto += 1
            else:
                if str(df.at[row_idx, "id_source"]) == "bootstrap":
                    df.at[row_idx, "id"] = UNKNOWN_ID
                    df.at[row_idx, "status"] = "unknown"
                    df.at[row_idx, "bad_reason"] = reason
                df.at[row_idx, "review_priority"] = reason or "unknown"
                n_unknown += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return {
        "output_csv": str(output_csv),
        "n_rows": int(len(df)),
        "n_fragments": int(fragments["fragment_id"].nunique()),
        "n_seed_rows": int(len(manual_rows)),
        "n_preserved_seed_rows": int(n_preserved),
        "n_auto_labeled_rows": int(n_auto),
        "n_unknown_rows": int(n_unknown),
        "trained_identities": sorted(centroids),
        "auto_accept": bool(auto_accept),
        "min_margin": float(min_margin),
        "max_distance": float(max_distance),
    }


def apply_identity_model(
    *,
    candidates_csv: Path,
    model_path: Path,
    output_tracks: Path,
    output_fragments: Path | None = None,
    min_margin: float = 0.05,
    max_distance: float = 1.25,
    feature_size: int | None = None,
) -> dict:
    df = pd.read_csv(candidates_csv)
    csv_dir = candidates_csv.parent
    usable = df[df["status"].fillna("review").isin(["accept", "review", "unknown", ""])].copy()
    fragments = link_blob_fragments(usable, csv_dir=csv_dir)
    model = joblib.load(model_path)
    centroids = model.get("centroids", {})
    fs = int(feature_size or model.get("feature_size", 32))
    fragment_features = aggregate_fragment_features(fragments, csv_dir=csv_dir, feature_size=fs)
    assignments = assign_fragment_identities(
        fragment_features,
        centroids,
        min_margin=min_margin,
        max_distance=max_distance,
    )
    assignments = enforce_one_identity_per_frame(fragments, assignments)

    id_to_numeric = {label: i for i, label in enumerate(ID_LABELS)}
    track_rows: list[dict] = []
    fragment_rows: list[FragmentSummary] = []
    for fid, group in fragments.groupby("fragment_id"):
        fid = int(fid)
        identity, dist, margin, reason = assignments.get(fid, (UNKNOWN_ID, None, None, "no_feature"))
        if identity not in ID_LABELS:
            track_id = -1
        else:
            track_id = id_to_numeric[identity]
        fragment_rows.append(
            FragmentSummary(
                fragment_id=fid,
                start_frame=int(group["frame"].min()),
                end_frame=int(group["frame"].max()),
                n_candidates=int(len(group)),
                identity=identity,
                distance=dist,
                margin=margin,
                unknown_reason=reason,
            )
        )
        for _, row in group.iterrows():
            track_rows.append(
                {
                    "frame": int(row["frame"]),
                    "track_id": track_id,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "score": row.get("quality_score", ""),
                    "node": "mouth",
                    "identity": identity,
                    "fragment_id": fid,
                    "appearance_distance": dist,
                    "appearance_margin": margin,
                    "unknown_reason": reason,
                }
            )

    output_tracks.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(track_rows).sort_values(["frame", "track_id"]).to_csv(output_tracks, index=False)
    if output_fragments is None:
        output_fragments = output_tracks.with_name("top_cam.blob_identity_fragments.csv")
    pd.DataFrame([asdict(r) for r in fragment_rows]).to_csv(output_fragments, index=False)
    return {
        "tracks_csv": str(output_tracks),
        "fragments_csv": str(output_fragments),
        "n_fragments": len(fragment_rows),
        "n_track_rows": len(track_rows),
    }


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path | None, Path, Path, str]:
    slp_path = Path(args.slp) if getattr(args, "slp", None) else None
    video_path = Path(args.video) if getattr(args, "video", None) else None
    experiment = getattr(args, "experiment", None)
    slp_optional = bool(getattr(args, "pure_bgsub", False) or getattr(args, "no_sleap_filter", False))
    if slp_path is None or video_path is None:
        if not experiment and not (slp_optional and video_path is not None):
            raise SystemExit("Pass an experiment name or both --slp and --video.")
        if experiment:
            local = load_local_paths(args.paths_config)
            sleap_output_dir = choose_path(args.sleap_output_dir, local.sleap_output_dir if local else None)
            medaka_data_dir = choose_path(args.medaka_data_dir, local.medaka_data_dir if local else None)
            if medaka_data_dir is None or (sleap_output_dir is None and not slp_optional):
                raise SystemExit("Need medaka_data_dir and, unless using pure bgsub/no SLEAP filter, sleap_output_dir.")
            if slp_path is None and sleap_output_dir is not None:
                slp_path = Path(sleap_output_dir) / experiment / "top_cam.slp"
            video_path = video_path or Path(medaka_data_dir) / experiment / "top_cam" / VIDEO_BASENAMES["top_cam"]
    if slp_path is not None and not slp_path.exists() and not slp_optional:
        raise SystemExit(f"SLP not found: {slp_path}")
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if slp_path is not None and not slp_path.exists():
        slp_path = None
    experiment_name = experiment or (slp_path.parent.name if slp_path is not None else "top_camera")
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else Path("runs") / "top_id" / experiment_name
    return slp_path, video_path, out_dir, experiment_name


def _resolve_review_video(args: argparse.Namespace) -> Path | None:
    if getattr(args, "video", None):
        video = Path(args.video)
        if not video.exists():
            raise SystemExit(f"Video not found: {video}")
        return video
    experiment = getattr(args, "experiment", None)
    if not experiment:
        return None
    local = load_local_paths(args.paths_config)
    medaka_data_dir = choose_path(args.medaka_data_dir, local.medaka_data_dir if local else None)
    if medaka_data_dir is None:
        raise SystemExit("Need medaka_data_dir to resolve review video from --experiment.")
    video = Path(medaka_data_dir) / experiment / "top_cam" / VIDEO_BASENAMES["top_cam"]
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    return video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blob-first top-camera ID workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mine-blobs", help="Mine blob candidates from top camera.")
    p.add_argument("experiment", nargs="?")
    p.add_argument("--slp")
    p.add_argument("--video")
    p.add_argument("--out-dir")
    p.add_argument("--paths-config")
    p.add_argument("--sleap-output-dir")
    p.add_argument("--medaka-data-dir")
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--sam-manifest", type=Path, default=None)
    p.add_argument("--threshold", type=int, default=24)
    p.add_argument("--min-area", type=int, default=80)
    p.add_argument("--max-area", type=int, default=8000)
    p.add_argument("--max-sleap-distance-px", type=float, default=80.0)
    p.add_argument("--max-candidates-per-sleap", type=int, default=1)
    p.add_argument("--background-samples", type=int, default=80)
    p.add_argument("--no-sleap-filter", action="store_true", help="Keep bg/SAM components without SLEAP proximity filtering.")
    p.add_argument("--pure-bgsub", action="store_true", help="Use only background-subtraction components; skip SLEAP filtering and SAM.")
    p.add_argument("--no-preview-video", action="store_true")

    p = sub.add_parser("review", help="Keyboard-review blob candidates.")
    p.add_argument("candidates_csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None, help="Top-camera video path for full-frame review.")
    p.add_argument("--experiment", default=None, help="Resolve top-camera video from paths.local.toml.")
    p.add_argument("--paths-config", default=None)
    p.add_argument("--medaka-data-dir", default=None)
    p.add_argument("--crop-only", action="store_true", help="Use the old crop-only reviewer.")
    p.add_argument("--play-fps", type=float, default=2.0, help="Slow-play review speed.")
    p.add_argument("--display-scale", type=float, default=1.0, help="Scale full-frame review window.")

    p = sub.add_parser("train-id", help="Train fragment-level ID centroids.")
    p.add_argument("reviewed_csv", type=Path)
    p.add_argument("--output-model", type=Path, default=None)
    p.add_argument("--output-report", type=Path, default=None)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--feature-size", type=int, default=32)

    p = sub.add_parser("bootstrap-id", help="Pre-label candidates from seed labels or a model.")
    p.add_argument("candidates_csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None, help="Optional existing top_id_model.joblib.")
    p.add_argument("--min-seed-samples", type=int, default=1)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--max-distance", type=float, default=1.10)
    p.add_argument("--feature-size", type=int, default=32)
    p.add_argument("--suggest-only", action="store_true", help="Fill suggested IDs without auto-accepting rows.")
    p.add_argument(
        "--include-bootstrap-labels",
        action="store_true",
        help="Allow previous bootstrap labels to become training seeds.",
    )

    p = sub.add_parser("apply-id", help="Apply fragment-level ID model.")
    p.add_argument("candidates_csv", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("--output-tracks", type=Path, default=None)
    p.add_argument("--output-fragments", type=Path, default=None)
    p.add_argument("--min-margin", type=float, default=0.05)
    p.add_argument("--max-distance", type=float, default=1.25)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "mine-blobs":
        slp_path, video_path, out_dir, experiment = _resolve_inputs(args)
        result = mine_blobs(
            experiment=experiment,
            video_path=video_path,
            slp_path=slp_path,
            out_dir=out_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            every=args.every,
            sam_manifest=args.sam_manifest,
            threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            max_sleap_distance_px=args.max_sleap_distance_px,
            max_candidates_per_sleap=args.max_candidates_per_sleap,
            background_samples=args.background_samples,
            preview_video=not args.no_preview_video,
            no_sleap_filter=args.no_sleap_filter,
            pure_bgsub=args.pure_bgsub,
        )
    elif args.command == "review":
        video_path = _resolve_review_video(args)
        result = review_candidates(
            candidates_csv=args.candidates_csv,
            output_csv=args.output,
            video_path=video_path,
            full_frame=not args.crop_only,
            play_fps=args.play_fps,
            display_scale=args.display_scale,
        )
    elif args.command == "train-id":
        output_model = args.output_model or args.reviewed_csv.with_name("top_id_model.joblib")
        result = train_identity_model(
            reviewed_csv=args.reviewed_csv,
            output_model=output_model,
            output_report=args.output_report,
            min_samples=args.min_samples,
            feature_size=args.feature_size,
        )
    elif args.command == "bootstrap-id":
        result = bootstrap_identity_labels(
            candidates_csv=args.candidates_csv,
            output_csv=args.output,
            model_path=args.model,
            min_seed_samples=args.min_seed_samples,
            min_margin=args.min_margin,
            max_distance=args.max_distance,
            feature_size=args.feature_size,
            auto_accept=not args.suggest_only,
            include_bootstrap_labels=args.include_bootstrap_labels,
        )
    elif args.command == "apply-id":
        output_tracks = args.output_tracks or args.candidates_csv.with_name("top_cam.blob_identity_tracks.csv")
        result = apply_identity_model(
            candidates_csv=args.candidates_csv,
            model_path=args.model,
            output_tracks=output_tracks,
            output_fragments=args.output_fragments,
            min_margin=args.min_margin,
            max_distance=args.max_distance,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
