from __future__ import annotations

"""Build a conservative, geometry-supervised fish appearance dataset.

The dataset contains only local same/different constraints.  In particular,
``geometry_source_idx_local`` is a frame/window-local reconstruction slot and is
never treated as a biological identity.  Positive labels come from either the
same continuous 2D fragment or the same synchronized reconstruction source;
negative labels come from different fish visible in the same camera frame.

Example::

    python -m sam2_id_experiment.scripts.48_build_geometry_contrastive_dataset \
      --experiment 20251225_K_1_2_FCMC \
      --identified-2d ..._identified_2d.csv \
      --dispositions ..._detection_dispositions.csv \
      --best-pre-link-json ..._best_pre_link.json \
      --view-input top_cam cam0_top.mp4 top_masks.h5 top.silhouette_fused_tracklets.csv \
      --view-input side1_cam cam1_side.mp4 side1_masks.h5 side1.silhouette_fused_tracklets.csv \
      --view-input side2_cam cam2_side.mp4 side2_masks.h5 side2.silhouette_fused_tracklets.csv \
      --window development:10000:69999:train \
      --window calibration:70500:74499:calibration \
      --window heldout:75000:79999:test \
      --output-dir runs/geometry_clean_embedding_v1

``SOURCE`` in ``--view-input`` can be a video or a directory containing images
whose filenames end in a frame number.  Inputs are read-only.  The output
directory must be absent or empty.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "geometry_contrastive_v1"
VALID_SPLITS = ("train", "calibration", "test")
PAIR_COLUMNS = [
    "pair_id", "sample_a_id", "sample_b_id", "pair_type", "pair_frame_delta", "window",
    "local_fragment_id", "geometry_source_idx_local",
]


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int
    split: str | None = None


@dataclass(frozen=True)
class ViewInput:
    view: str
    source: Path
    masks: Path
    associations: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--identified-2d", required=True)
    p.add_argument("--dispositions", required=True)
    p.add_argument("--best-pre-link-json", required=True)
    p.add_argument(
        "--view-input", nargs=4, action="append", required=True,
        metavar=("VIEW", "VIDEO_OR_IMAGE_DIR", "MASKS_H5", "ASSOCIATIONS"),
        help="Repeat once per camera. ASSOCIATIONS is a silhouette-fused CSV/parquet.",
    )
    p.add_argument(
        "--window", action="append", default=[], metavar="NAME:START:END[:SPLIT]",
        help="Non-overlapping inclusive frame range. SPLIT is train, calibration, or test.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--max-residual-px2", type=float, default=500.0)
    p.add_argument("--min-separation-mm", type=float, default=15.0)
    p.add_argument("--max-speed-mm-per-frame", type=float, default=8.0)
    p.add_argument("--min-mask-score", type=float, default=0.5)
    p.add_argument("--min-mutual-confidence", type=float, default=0.25)
    p.add_argument("--association-max-distance-px", type=float, default=12.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--crop-padding-fraction", type=float, default=0.15)
    p.add_argument(
        "--edge-margin-px",
        type=int,
        default=2,
        help="Reject masks whose bounding box approaches the image edge by this many pixels.",
    )
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--min-positive-frame-delta", type=int, default=10)
    p.add_argument("--max-positive-frame-delta", type=int, default=1000)
    p.add_argument("--max-positive-pairs-per-fragment", type=int, default=2000)
    p.add_argument("--max-fragment-gap", type=int, default=100)
    p.add_argument("--block-size-frames", type=int, default=500)
    p.add_argument("--embargo-frames", type=int, default=500)
    p.add_argument("--split-fractions", nargs=3, type=float, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "CALIBRATION", "TEST"))
    p.add_argument("--include-cross-view-negatives", action="store_true")
    p.add_argument("--random-seed", type=int, default=13)
    p.add_argument("--contact-sheet-samples", type=int, default=48)
    return p


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _require_columns(table: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def parse_windows(values: list[str]) -> list[Window]:
    windows: list[Window] = []
    for value in values:
        fields = value.split(":")
        if len(fields) not in {3, 4}:
            raise ValueError(f"Invalid --window {value!r}; expected NAME:START:END[:SPLIT]")
        name, start_text, end_text = fields[:3]
        split = fields[3].lower() if len(fields) == 4 else None
        # Backward-compatible spelling for early pilot commands.  The persisted
        # split is always "calibration" because thresholds are fitted there.
        if split in {"validation", "val"}:
            split = "calibration"
        if not name:
            raise ValueError("Window names must be non-empty.")
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"Window {name!r} ends before it starts.")
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r}; choose from {VALID_SPLITS}.")
        windows.append(Window(name, start, end, split))
    windows.sort(key=lambda w: (w.start, w.end))
    if len({w.name for w in windows}) != len(windows):
        raise ValueError("Window names must be unique.")
    for left, right in zip(windows, windows[1:]):
        if right.start <= left.end:
            raise ValueError(f"Windows {left.name!r} and {right.name!r} overlap.")
    explicit = [w.split is not None for w in windows]
    if any(explicit) and not all(explicit):
        raise ValueError("Either assign a split to every --window or to none of them.")
    return windows


def _split_counts(n: int, fractions: tuple[float, float, float], names: list[str]) -> dict[str, int]:
    weights = np.asarray([fractions[VALID_SPLITS.index(name)] for name in names], dtype=float)
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Split fractions must be finite, non-negative, and not all zero.")
    weights /= weights.sum()
    raw = weights * n
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[index] += 1
    if n >= len(names):
        for index in range(len(names)):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def assign_splits(
    frames: Iterable[int], windows: list[Window], *, block_size: int,
    embargo_frames: int, fractions: tuple[float, float, float],
) -> pd.DataFrame:
    """Assign complete temporal blocks and reserve complete embargo blocks."""
    if block_size <= 0 or embargo_frames < 0:
        raise ValueError("block_size must be positive and embargo_frames non-negative.")
    unique_frames = np.asarray(sorted(set(int(frame) for frame in frames)), dtype=int)
    if not len(unique_frames):
        return pd.DataFrame(columns=["frame", "window", "split", "split_block_id"])
    if not windows:
        windows = [Window("all", int(unique_frames.min()), int(unique_frames.max()), None)]

    if all(window.split is not None for window in windows):
        for left, right in zip(windows, windows[1:]):
            if left.split != right.split:
                separation = right.start - left.end - 1
                if separation < embargo_frames:
                    raise ValueError(
                        f"Explicit windows {left.name!r}/{right.name!r} have only {separation} "
                        f"embargo frames; require {embargo_frames}."
                    )
        rows = []
        for frame in unique_frames:
            matches = [window for window in windows if window.start <= frame <= window.end]
            if matches:
                window = matches[0]
                rows.append({"frame": frame, "window": window.name, "split": window.split,
                             "split_block_id": f"window:{window.name}"})
        return pd.DataFrame(rows)

    blocks: list[tuple[Window, int, int, int]] = []
    for window in windows:
        present = unique_frames[(unique_frames >= window.start) & (unique_frames <= window.end)]
        if not len(present):
            continue
        for block_index in sorted(set(((present - window.start) // block_size).tolist())):
            start = window.start + int(block_index) * block_size
            blocks.append((window, int(block_index), start, start + block_size - 1))
    nonzero_names = [name for name, fraction in zip(VALID_SPLITS, fractions) if fraction > 0]
    embargo_blocks = int(math.ceil(embargo_frames / block_size)) if embargo_frames else 0
    reserve = embargo_blocks * max(0, len(nonzero_names) - 1)
    usable_count = len(blocks) - reserve
    if usable_count < len(nonzero_names):
        raise ValueError(
            f"Only {len(blocks)} temporal blocks are available; need at least "
            f"{len(nonzero_names) + reserve} for requested splits and whole-block embargo."
        )
    counts = _split_counts(usable_count, fractions, nonzero_names)
    block_assignments: dict[tuple[str, int], str] = {}
    cursor = 0
    for split_index, split in enumerate(nonzero_names):
        for window, block_index, _start, _end in blocks[cursor : cursor + counts[split]]:
            block_assignments[(window.name, block_index)] = split
        cursor += counts[split]
        if split_index < len(nonzero_names) - 1:
            for window, block_index, _start, _end in blocks[cursor : cursor + embargo_blocks]:
                block_assignments[(window.name, block_index)] = "embargo"
            cursor += embargo_blocks

    rows = []
    for frame in unique_frames:
        for window in windows:
            if window.start <= frame <= window.end:
                block_index = (frame - window.start) // block_size
                split = block_assignments.get((window.name, int(block_index)), "excluded")
                rows.append({
                    "frame": int(frame), "window": window.name, "split": split,
                    "split_block_id": f"{window.name}:b{int(block_index):05d}",
                })
                break
    return pd.DataFrame(rows)


def load_geometry_qc(
    path: str | Path, *, num_fish: int, max_residual: float,
    min_separation: float, max_speed: float,
) -> tuple[pd.DataFrame, Counter]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_frames = payload.get("frames", payload)
    labels = [chr(ord("a") + index) for index in range(num_fish)]
    rows: list[dict] = []
    counts: Counter = Counter()
    previous: dict[int, tuple[int, np.ndarray]] = {}
    for frame_text, info in sorted(raw_frames.items(), key=lambda item: int(item[0])):
        frame = int(frame_text)
        groups = info.get("groups", {})
        errors = info.get("errors", {})
        points = [np.asarray(groups.get(label, []), dtype=float) for label in labels]
        if not all(point.shape == (3,) and np.isfinite(point).all() for point in points):
            counts["reject_nonfinite_or_missing_3d"] += 1
            continue
        residuals = np.asarray([errors.get(label, np.nan) for label in labels], dtype=float)
        if not np.isfinite(residuals).all() or np.any(residuals > max_residual):
            counts["reject_high_residual"] += 1
            continue
        if info.get("dropped_high_residual"):
            counts["reject_dropped_high_residual"] += 1
            continue
        recovered = list(info.get("recovered", [False] * num_fish))
        synthetic = list(info.get("synthetic_views", [-1] * num_fish))
        if any(bool(value) for value in recovered) or any(int(value) >= 0 for value in synthetic):
            counts["reject_partial_or_recovered_geometry"] += 1
            continue
        pairwise = [float(np.linalg.norm(points[a] - points[b]))
                    for a, b in combinations(range(num_fish), 2)]
        minimum_separation = min(pairwise) if pairwise else float("inf")
        if minimum_separation < min_separation:
            counts["reject_3d_too_close"] += 1
            continue
        speeds = []
        for source_idx, point in enumerate(points):
            prior = previous.get(source_idx)
            speed = np.nan
            if prior is not None and frame - prior[0] == 1:
                speed = float(np.linalg.norm(point - prior[1]))
            speeds.append(speed)
            previous[source_idx] = (frame, point)
        finite_speeds = [speed for speed in speeds if np.isfinite(speed)]
        if finite_speeds and max(finite_speeds) > max_speed:
            counts["reject_speed_spike"] += 1
            continue
        for source_idx, (point, residual, speed) in enumerate(zip(points, residuals, speeds)):
            rows.append({
                "frame": frame, "geometry_source_idx_local": source_idx,
                "geometry_x_mm": float(point[0]), "geometry_y_mm": float(point[1]),
                "geometry_z_mm": float(point[2]), "residual_px2": float(residual),
                "min_fish_separation_mm": float(minimum_separation),
                "speed_mm_per_frame": float(speed) if np.isfinite(speed) else np.nan,
            })
        counts["accept_geometry_frame"] += 1
    return pd.DataFrame(rows), counts


def _bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    return series.fillna(default).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _tracklet_key(value: object) -> str:
    text = str(value)
    return text.rsplit(":", 1)[-1]


def build_candidate_table(
    *, identified: pd.DataFrame, dispositions: pd.DataFrame, associations: pd.DataFrame,
    geometry: pd.DataFrame, split_map: pd.DataFrame, view: str, min_mask_score: float,
    min_mutual_confidence: float, max_association_distance: float, num_fish: int,
) -> tuple[pd.DataFrame, Counter]:
    """Join strict frame-local geometry, SLEAP, and mask evidence."""
    counts: Counter = Counter()
    _require_columns(
        identified,
        ["frame", "view", "node_id", "tracklet_id", "x", "y", "component_id"],
        "identified_2d",
    )
    _require_columns(
        dispositions,
        ["frame", "view", "tracklet_node", "selected_source_idx", "disposition", "synthetic_view"],
        "detection dispositions",
    )
    _require_columns(
        associations,
        ["frame", "tracklet_id", "x", "y", "matched_mask_instance_id", "matched_mask_track_id",
         "mask_score", "n_candidate_masks", "n_skeletons_for_mask", "split_endpoints_across_masks",
         "silhouette_status", "silhouette_evidence_state", "mutual_confidence"],
        f"{view} associations",
    )
    disp = dispositions[dispositions["view"].astype(str) == view].copy()
    counts["disposition_rows"] = len(disp)
    disp["geometry_source_idx_local"] = pd.to_numeric(disp["selected_source_idx"], errors="coerce")
    valid_disp = (
        disp["geometry_source_idx_local"].between(0, num_fish - 1)
        & disp["disposition"].astype(str).eq("assigned_2d_valid_3d")
        & pd.to_numeric(disp["synthetic_view"], errors="coerce").fillna(-1).eq(-1)
    )
    counts["reject_disposition_not_clean_three_view"] = int((~valid_disp).sum())
    disp = disp[valid_disp].copy()
    disp["geometry_source_idx_local"] = disp["geometry_source_idx_local"].astype(int)
    disp["node_id"] = disp["tracklet_node"].astype(str)
    disp = disp.merge(geometry, on=["frame", "geometry_source_idx_local"], how="inner")
    disp = disp.merge(split_map, on="frame", how="inner")
    counts["reject_embargo_or_outside_window"] = int(disp["split"].isin(["embargo", "excluded"]).sum())
    disp = disp[disp["split"].isin(VALID_SPLITS)].copy()

    id2 = identified[identified["view"].astype(str) == view].copy()
    id2 = id2.drop_duplicates(["frame", "node_id"], keep=False)
    work = disp.merge(
        id2[["frame", "node_id", "tracklet_id", "x", "y", "component_id"]
            if "component_id" in id2.columns else ["frame", "node_id", "tracklet_id", "x", "y"]],
        on=["frame", "node_id"], how="inner", suffixes=("_disposition", ""),
    )
    work["tracklet_key"] = work["tracklet_id"].map(_tracklet_key)

    assoc = associations.copy()
    assoc["tracklet_key"] = assoc["tracklet_id"].map(_tracklet_key)
    assoc["association_row"] = np.arange(len(assoc), dtype=int)
    work = work.merge(assoc, on=["frame", "tracklet_key"], how="inner", suffixes=("", "_association"))
    work["association_distance_px"] = np.hypot(
        pd.to_numeric(work["x_association"], errors="coerce") - pd.to_numeric(work["x"], errors="coerce"),
        pd.to_numeric(work["y_association"], errors="coerce") - pd.to_numeric(work["y"], errors="coerce"),
    )
    strict = (
        pd.to_numeric(work["matched_mask_instance_id"], errors="coerce").ge(0)
        & pd.to_numeric(work["mask_score"], errors="coerce").ge(min_mask_score)
        & pd.to_numeric(work["mutual_confidence"], errors="coerce").ge(min_mutual_confidence)
        & pd.to_numeric(work["n_candidate_masks"], errors="coerce").eq(1)
        & pd.to_numeric(work["n_skeletons_for_mask"], errors="coerce").eq(1)
        & ~_bool_series(work["split_endpoints_across_masks"])
        & work["silhouette_evidence_state"].astype(str).eq("support")
        & work["silhouette_status"].astype(str).isin({"strong_match", "supported_match"})
        & work["association_distance_px"].le(max_association_distance)
    )
    ambiguity_columns = [
        column
        for column in (
            "is_reflection",
            "reflection_candidate",
            "is_merge",
            "is_split",
            "duplicate_mask_assignment",
        )
        if column in work.columns
    ]
    for column in ambiguity_columns:
        ambiguous = _bool_series(work[column])
        counts[f"reject_explicit_ambiguity_{column}"] = int(ambiguous.sum())
        strict &= ~ambiguous
    counts["reject_mask_not_strict_one_to_one_support"] = int((~strict).sum())
    work = work[strict].copy()
    work["matched_mask_instance_id"] = work["matched_mask_instance_id"].astype(int)

    # A duplicated SLEAP node or mask is a merge/split/reflection ambiguity.  Use
    # keep=False so neither side of the ambiguous relationship enters training.
    node_dup = work.duplicated(["frame", "node_id"], keep=False)
    mask_dup = work.duplicated(["frame", "matched_mask_instance_id"], keep=False)
    counts["reject_duplicate_node_or_mask_mapping"] = int((node_dup | mask_dup).sum())
    work = work[~node_dup & ~mask_dup].copy()

    # Require the full simultaneous set in this view: one row per local source.
    cardinality = work.groupby("frame")["geometry_source_idx_local"].agg(["nunique", "size"])
    clean_frames = cardinality.index[(cardinality["nunique"] == num_fish) & (cardinality["size"] == num_fish)]
    counts["reject_rows_without_complete_view_frame"] = int((~work["frame"].isin(clean_frames)).sum())
    work = work[work["frame"].isin(clean_frames)].copy()
    counts["accept_candidate_rows"] = len(work)
    return work, counts


class MaskStore:
    """Read one HDF5 or a read-only chunk manifest without rewriting masks."""

    def __init__(self, path: Path, *, view: str | None = None):
        self.path = Path(path)
        self.handles: dict[Path, h5py.File] = {}
        self.chunks: list[tuple[int | None, int | None, Path]] = []
        if self.path.is_dir():
            pattern = f"{view}.masks_*.h5" if view else "*.masks_*.h5"
            paths = sorted(self.path.glob(pattern))
            if not paths:
                raise FileNotFoundError(f"no mask chunks matching {pattern!r} under {self.path}")
            self.chunks = [(*self._range_from_name(item), item) for item in paths]
        elif self.path.suffix.lower() == ".json":
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "sam2_mask_chunk_manifest_v1":
                raise ValueError("mask chunk manifest has unsupported schema_version")
            if view and payload.get("view") not in {None, view}:
                raise ValueError(
                    f"mask chunk manifest view {payload.get('view')!r} does not match {view!r}"
                )
            for entry in payload.get("chunks", []):
                item = Path(entry["path"])
                if not item.is_absolute():
                    item = self.path.parent / item
                self.chunks.append(
                    (int(entry["start_frame"]), int(entry["end_frame"]), item)
                )
        else:
            self.chunks = [(None, None, self.path)]
        if not self.chunks:
            raise ValueError("mask chunk source contains no chunks")
        for _start, _end, item in self.chunks:
            if not item.exists():
                raise FileNotFoundError(item)
        ranged = sorted(
            (int(start), int(end), item)
            for start, end, item in self.chunks
            if start is not None and end is not None
        )
        for (start, end, _item), (next_start, _next_end, _next_item) in zip(ranged, ranged[1:]):
            if end >= next_start:
                raise ValueError(
                    f"mask chunk frame ranges overlap: [{start}, {end}] and [{next_start}, ...]"
                )

    @staticmethod
    def _range_from_name(path: Path) -> tuple[int, int]:
        match = re.search(r"masks_(\d+)-(\d+)\.h5$", path.name)
        if not match:
            raise ValueError(f"cannot infer mask chunk frame range from {path.name!r}")
        return int(match.group(1)), int(match.group(2))

    def _chunk_for_frame(self, frame: int) -> Path | None:
        matches = [
            item
            for start, end, item in self.chunks
            if start is None or (int(start) <= int(frame) <= int(end))
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    def _open(self, path: Path) -> h5py.File:
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        return self.handles[path]

    def read(self, frame: int, instance_id: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        chunk = self._chunk_for_frame(int(frame))
        if chunk is None:
            return None
        h5 = self._open(chunk)
        frame_key = f"frame_{int(frame):06d}"
        if frame_key not in h5:
            return None
        matches = []
        frame_group = h5[frame_key]
        for instance_key in (key for key in frame_group.keys() if key.startswith("instance_")):
            group = frame_group[instance_key]
            resolved = int(group.attrs.get("instance_id", int(instance_key.split("_")[1])))
            if resolved == int(instance_id):
                matches.append(group)
        if len(matches) != 1:
            return None
        group = matches[0]
        mask = np.asarray(group["mask"], dtype=bool)
        bbox = tuple(int(value) for value in np.asarray(group["bbox"], dtype=int).tolist())
        return mask, bbox

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


class FrameSource:
    def __init__(self, path: Path):
        self.path = path
        self.cap: cv2.VideoCapture | None = None
        self.images: dict[int, Path] | None = None
        self.next_video_frame = 0
        if path.is_dir():
            images: dict[int, Path] = {}
            for item in sorted(path.iterdir()):
                if item.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                    continue
                matches = re.findall(r"(\d+)", item.stem)
                if matches:
                    images[int(matches[-1])] = item
            self.images = images
        else:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")

    def read(self, frame: int) -> np.ndarray | None:
        if self.images is not None:
            path = self.images.get(int(frame))
            return cv2.imread(str(path), cv2.IMREAD_COLOR) if path is not None else None
        assert self.cap is not None
        frame = int(frame)
        # Dataset rows are frame-sorted.  Decode short gaps sequentially instead
        # of doing tens of thousands of expensive key-frame seeks.
        if frame < self.next_video_frame or frame - self.next_video_frame > 128:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.next_video_frame = frame
        image = None
        while self.next_video_frame <= frame:
            ok, image = self.cap.read()
            if not ok:
                return None
            self.next_video_frame += 1
        return image

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def _square_masked_crop(
    image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int],
    *, size: int, padding_fraction: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    half = max(1, int(math.ceil(max(width, height) * (1.0 + padding_fraction) / 2.0)))
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    crop_x0, crop_y0 = int(math.floor(center_x - half)), int(math.floor(center_y - half))
    crop_x1, crop_y1 = crop_x0 + 2 * half, crop_y0 + 2 * half
    output = np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
    local_mask = np.zeros((2 * half, 2 * half), dtype=bool)
    image_h, image_w = image.shape[:2]
    source_x0, source_y0 = max(0, crop_x0), max(0, crop_y0)
    source_x1, source_y1 = min(image_w, crop_x1), min(image_h, crop_y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        target_x0, target_y0 = source_x0 - crop_x0, source_y0 - crop_y0
        target_x1 = target_x0 + source_x1 - source_x0
        target_y1 = target_y0 + source_y1 - source_y0
        patch = image[source_y0:source_y1, source_x0:source_x1]
        patch_mask = mask[source_y0:source_y1, source_x0:source_x1]
        output[target_y0:target_y1, target_x0:target_x1][patch_mask] = patch[patch_mask]
        local_mask[target_y0:target_y1, target_x0:target_x1] = patch_mask
    resized = cv2.resize(output, (size, size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(local_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    resized[resized_mask == 0] = 0
    return resized, (crop_x0, crop_y0, crop_x1, crop_y1)


def _assign_fragments(samples: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    samples = samples.sort_values(["experiment", "window", "split", "view", "node_id", "frame"]).copy()
    fragment_ids: dict[int, str] = {}
    fragment_index = 0
    # node_id is the must-link evidence.  Do not require a stable pre-link
    # source index across time: that slot is deliberately frame-local.
    keys = ["experiment", "window", "split", "view", "node_id"]
    for _key, group in samples.groupby(keys, sort=False, dropna=False):
        previous = None
        for index, frame in zip(group.index, group["frame"].astype(int)):
            if previous is None or frame - previous > max_gap:
                fragment_index += 1
            fragment_ids[index] = f"local_fragment_{fragment_index:08d}"
            previous = frame
    samples["local_fragment_id"] = pd.Series(fragment_ids)
    return samples.sort_values("sample_id").reset_index(drop=True)


def materialize_crops(
    candidates_by_view: dict[str, pd.DataFrame], view_inputs: dict[str, ViewInput],
    *, output_h5: Path, experiment: str, crop_size: int, padding_fraction: float,
    sample_every: int, max_fragment_gap: int, edge_margin_px: int,
) -> tuple[pd.DataFrame, Counter]:
    counts: Counter = Counter()
    metadata: list[dict] = []
    with h5py.File(output_h5, "x") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["identity_semantics"] = "local constraints only; no biological identity labels"
        output.attrs["color_order"] = "RGB"
        crops = output.create_dataset(
            "crops", shape=(0, crop_size, crop_size, 3), maxshape=(None, crop_size, crop_size, 3),
            dtype=np.uint8, chunks=(min(64, max(1, 4096 // crop_size)), crop_size, crop_size, 3),
            compression="gzip", shuffle=True,
        )
        buffer: list[np.ndarray] = []

        def flush() -> None:
            if not buffer:
                return
            start = len(crops)
            crops.resize(start + len(buffer), axis=0)
            crops[start:] = np.stack(buffer)
            buffer.clear()

        for view, candidate_table in candidates_by_view.items():
            selected = candidate_table[
                candidate_table["frame"].astype(int).mod(max(1, sample_every)).eq(0)
            ].copy()
            store = MaskStore(view_inputs[view].masks, view=view)
            source = FrameSource(view_inputs[view].source)
            try:
                for frame, frame_group in selected.groupby("frame", sort=True):
                    image = source.read(int(frame))
                    if image is None:
                        counts["reject_missing_source_frame"] += len(frame_group)
                        continue
                    for row in frame_group.to_dict("records"):
                        result = store.read(int(frame), int(row["matched_mask_instance_id"]))
                        if result is None:
                            counts["reject_missing_or_duplicate_mask_instance"] += 1
                            continue
                        mask, bbox = result
                        if mask.shape != image.shape[:2] or not mask.any():
                            counts["reject_invalid_mask_shape_or_empty"] += 1
                            continue
                        height, width = image.shape[:2]
                        if (
                            bbox[0] <= edge_margin_px
                            or bbox[1] <= edge_margin_px
                            or bbox[2] >= width - 1 - edge_margin_px
                            or bbox[3] >= height - 1 - edge_margin_px
                        ):
                            counts["reject_truncated_mask_at_image_edge"] += 1
                            continue
                        crop_bgr, crop_bounds = _square_masked_crop(
                            image, mask, bbox, size=crop_size, padding_fraction=padding_fraction,
                        )
                        crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        sample_id = len(metadata)
                        buffer.append(crop)
                        record = {
                            "sample_id": sample_id, "crop_index": sample_id,
                            "experiment": experiment, "view": view,
                            "frame": int(frame), "node_id": str(row["node_id"]),
                            "tracklet_id": str(row["tracklet_id"]),
                            "component_id_provenance": str(row.get("component_id", "")),
                            "matched_mask_instance_id": int(row["matched_mask_instance_id"]),
                            "matched_mask_track_id": int(row["matched_mask_track_id"]),
                            "geometry_source_idx_local": int(row["geometry_source_idx_local"]),
                            "split": str(row["split"]), "window": str(row["window"]),
                            "split_block_id": str(row["split_block_id"]),
                            "sleap_x": float(row["x"]), "sleap_y": float(row["y"]),
                            "geometry_x_mm": float(row["geometry_x_mm"]),
                            "geometry_y_mm": float(row["geometry_y_mm"]),
                            "geometry_z_mm": float(row["geometry_z_mm"]),
                            "residual_px2": float(row["residual_px2"]),
                            "min_fish_separation_mm": float(row["min_fish_separation_mm"]),
                            "speed_mm_per_frame": float(row["speed_mm_per_frame"]),
                            "matched_mask_score": float(row["mask_score"]),
                            "matched_mask_area_px": int(np.count_nonzero(mask)),
                            "silhouette_status": str(row["silhouette_status"]),
                            "mutual_confidence": float(row["mutual_confidence"]),
                            "n_candidate_masks": int(row["n_candidate_masks"]),
                            "n_skeletons_for_mask": int(row["n_skeletons_for_mask"]),
                            "association_distance_px": float(row["association_distance_px"]),
                            "mask_bbox_x0": int(bbox[0]), "mask_bbox_y0": int(bbox[1]),
                            "mask_bbox_x1": int(bbox[2]), "mask_bbox_y1": int(bbox[3]),
                            "crop_x0": crop_bounds[0], "crop_y0": crop_bounds[1],
                            "crop_x1": crop_bounds[2], "crop_y1": crop_bounds[3],
                            "qc_status": "clean_geometry_contrastive",
                        }
                        metadata.append(record)
                        if len(buffer) >= 128:
                            flush()
                flush()
            finally:
                source.close()
                store.close()
    samples = pd.DataFrame(metadata)
    if samples.empty:
        raise RuntimeError("No crops survived materialization.")
    samples = _assign_fragments(samples, max_fragment_gap)
    counts["accept_samples"] = len(samples)
    return samples, counts


def _sample_temporal_pairs(
    group: pd.DataFrame, *, minimum_delta: int, maximum_delta: int,
    limit: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    ids = group["sample_id"].astype(int).to_numpy()
    frames = group["frame"].astype(int).to_numpy()
    if len(ids) < 2 or limit <= 0:
        return []
    possible: list[tuple[int, int]] = []
    if len(ids) <= 300:
        possible = [
            (int(ids[left]), int(ids[right]))
            for left in range(len(ids)) for right in range(left + 1, len(ids))
            if minimum_delta <= abs(int(frames[right] - frames[left])) <= maximum_delta
        ]
        if len(possible) > limit:
            chosen = rng.choice(len(possible), size=limit, replace=False)
            possible = [possible[int(index)] for index in chosen]
        return possible
    pairs: set[tuple[int, int]] = set()
    for _ in range(limit * 50):
        left = int(rng.integers(0, len(ids)))
        valid = np.flatnonzero(
            (np.abs(frames - frames[left]) >= minimum_delta)
            & (np.abs(frames - frames[left]) <= maximum_delta)
        )
        if len(valid):
            right = int(rng.choice(valid))
            if right != left:
                pair = tuple(sorted((int(ids[left]), int(ids[right]))))
                pairs.add(pair)
        if len(pairs) >= limit:
            break
    return sorted(pairs)


def build_pairs(
    samples: pd.DataFrame, *, min_positive_delta: int, max_positive_delta: int,
    max_positive_per_fragment: int, include_cross_view_negatives: bool,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    sample_lookup = samples.set_index("sample_id", drop=False)
    positive_rows: list[dict] = []
    for fragment, group in samples.groupby("local_fragment_id", sort=True):
        pairs = _sample_temporal_pairs(
            group.sort_values("frame"), minimum_delta=min_positive_delta,
            maximum_delta=max_positive_delta, limit=max_positive_per_fragment, rng=rng,
        )
        for left, right in pairs:
            a, b = sample_lookup.loc[left], sample_lookup.loc[right]
            positive_rows.append(_pair_record(a, b, "same_local_fragment", fragment))

    synchronization_keys = ["experiment", "window", "split", "frame", "geometry_source_idx_local"]
    for _key, group in samples.groupby(synchronization_keys, sort=False):
        for left, right in combinations(group.to_dict("records"), 2):
            if left["view"] != right["view"]:
                positive_rows.append(_pair_record(left, right, "synchronized_cross_camera_same_source", ""))

    negative_rows: list[dict] = []
    negative_keys = ["experiment", "window", "split", "frame"]
    for _key, frame_group in samples.groupby(negative_keys, sort=False):
        view_groups: Iterator[pd.DataFrame]
        if include_cross_view_negatives:
            view_groups = iter([frame_group])
        else:
            view_groups = (group for _view, group in frame_group.groupby("view", sort=False))
        for group in view_groups:
            by_source = {int(source): source_group.to_dict("records")
                         for source, source_group in group.groupby("geometry_source_idx_local")}
            for source_a, source_b in combinations(sorted(by_source), 2):
                for left, right in product(by_source[source_a], by_source[source_b]):
                    kind = "simultaneous_cross_view_different_source" if left["view"] != right["view"] \
                        else "simultaneous_same_view_different_source"
                    negative_rows.append(_pair_record(left, right, kind, ""))

    positives = pd.DataFrame(positive_rows, columns=PAIR_COLUMNS)
    negatives = pd.DataFrame(negative_rows, columns=PAIR_COLUMNS)
    positives.insert(0, "pair_label", "same")
    negatives.insert(0, "pair_label", "different")
    positives["pair_id"] = np.arange(len(positives), dtype=int)
    negatives["pair_id"] = np.arange(len(negatives), dtype=int)
    return positives, negatives


def validate_dataset_gates(
    samples: pd.DataFrame,
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
) -> dict:
    """Fail closed on split, pair, provenance, and synchronization integrity."""
    split_counts = samples["split"].value_counts()
    missing_splits = [split for split in VALID_SPLITS if int(split_counts.get(split, 0)) == 0]
    if missing_splits:
        raise RuntimeError(f"Dataset gate failed: empty sample splits {missing_splits}")
    if samples["component_id_provenance"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Dataset gate failed: blank component_id_provenance")

    frame_scope = [column for column in ("experiment", "frame") if column in samples.columns]
    frame_overlap = samples.groupby(frame_scope, dropna=False)["split"].nunique()
    if (frame_overlap > 1).any():
        raise RuntimeError("Dataset gate failed: synchronized frames cross data splits")
    for column, label in (("split_block_id", "temporal block"), ("local_fragment_id", "fragment")):
        overlap = samples.groupby(column, dropna=False)["split"].nunique()
        if (overlap > 1).any():
            raise RuntimeError(f"Dataset gate failed: {label} leakage across splits")

    split_for_sample = samples.set_index("sample_id")["split"]
    frame_for_sample = samples.set_index("sample_id")["frame"]
    pair_counts: dict[str, dict[str, int]] = {split: {} for split in VALID_SPLITS}
    for label, table in (("positive", positives), ("negative", negatives)):
        left_split = table["sample_a_id"].map(split_for_sample)
        right_split = table["sample_b_id"].map(split_for_sample)
        if left_split.isna().any() or right_split.isna().any():
            raise RuntimeError("Dataset gate failed: pair references an unknown sample")
        if (left_split != right_split).any():
            raise RuntimeError("Dataset gate failed: pair crosses data splits")
        for split in VALID_SPLITS:
            count = int(left_split.eq(split).sum())
            pair_counts[split][label] = count
            if count == 0:
                raise RuntimeError(f"Dataset gate failed: split {split!r} has no {label} pairs")

    negative_left_frame = negatives["sample_a_id"].map(frame_for_sample)
    negative_right_frame = negatives["sample_b_id"].map(frame_for_sample)
    if not negative_left_frame.eq(negative_right_frame).all():
        raise RuntimeError("Dataset gate failed: negative pairs are not simultaneous")
    return {
        "passed": True,
        "samples_by_split": {split: int(split_counts.get(split, 0)) for split in VALID_SPLITS},
        "pairs_by_split_and_label": pair_counts,
        "synchronized_frames_kept_together": True,
        "temporal_blocks_disjoint": True,
        "fragments_disjoint": True,
        "component_provenance_nonblank": True,
        "negative_pairs_simultaneous": True,
        "color_order": "RGB",
    }


def _pair_record(left: pd.Series | dict, right: pd.Series | dict, kind: str, fragment: str) -> dict:
    left, right = dict(left), dict(right)
    source = (
        left["geometry_source_idx_local"]
        if kind == "synchronized_cross_camera_same_source"
        and left["geometry_source_idx_local"] == right["geometry_source_idx_local"]
        else -1
    )
    return {
        # Keep frame/view/split metadata in samples only.  Repeating those names
        # in pair tables makes validated trainer joins ambiguous (``frame_a_x``
        # versus ``frame_a_y``).  pair_frame_delta is safe diagnostic metadata.
        "pair_id": -1, "sample_a_id": int(left["sample_id"]),
        "sample_b_id": int(right["sample_id"]), "pair_type": kind,
        "pair_frame_delta": abs(int(left["frame"]) - int(right["frame"])),
        "window": str(left["window"]),
        "local_fragment_id": fragment, "geometry_source_idx_local": int(source),
    }


def _write_table_both(table: pd.DataFrame, stem: Path) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=False)
    table.to_parquet(stem.with_suffix(".parquet"), index=False)


def write_contact_sheet(crops_h5: Path, samples: pd.DataFrame, path: Path, count: int, seed: int) -> None:
    if count <= 0 or samples.empty:
        return
    rng = np.random.default_rng(seed)
    chosen = samples.iloc[np.sort(rng.choice(len(samples), size=min(count, len(samples)), replace=False))]
    with h5py.File(crops_h5, "r") as h5:
        images = np.asarray(h5["crops"][chosen["sample_id"].astype(int).to_numpy()])
    columns = min(8, len(images))
    rows = int(math.ceil(len(images) / columns))
    tile_h, tile_w = images.shape[1] + 36, images.shape[2]
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 238, dtype=np.uint8)
    for index, (image, row) in enumerate(zip(images, chosen.to_dict("records"))):
        y, x = divmod(index, columns)
        y0, x0 = y * tile_h, x * tile_w
        sheet[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image
        label = f"{row['view']} f{row['frame']} s{row['sample_id']}"
        cv2.putText(sheet, label, (x0 + 2, y0 + image.shape[0] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, str(row["local_fragment_id"]), (x0 + 2, y0 + image.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _validate_inputs(args: argparse.Namespace, views: list[ViewInput]) -> None:
    paths = [Path(args.identified_2d), Path(args.dispositions), Path(args.best_pre_link_json)]
    for view in views:
        paths.extend([view.source, view.masks, view.associations])
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input paths: {[str(path) for path in missing]}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if len({view.view for view in views}) != len(views):
        raise ValueError("Each --view-input VIEW must be unique.")
    if args.num_fish < 2 or args.crop_size <= 0 or args.sample_every <= 0 or args.edge_margin_px < 0:
        raise ValueError(
            "num_fish >= 2, crop_size > 0, sample_every > 0, and edge_margin_px >= 0 are required."
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    view_inputs = [ViewInput(str(view), Path(source), Path(masks), Path(associations))
                   for view, source, masks, associations in args.view_input]
    _validate_inputs(args, view_inputs)
    windows = parse_windows(args.window)

    geometry, geometry_counts = load_geometry_qc(
        args.best_pre_link_json, num_fish=args.num_fish,
        max_residual=args.max_residual_px2, min_separation=args.min_separation_mm,
        max_speed=args.max_speed_mm_per_frame,
    )
    if geometry.empty:
        raise RuntimeError("No geometry frames passed QC.")
    split_map = assign_splits(
        geometry["frame"], windows, block_size=args.block_size_frames,
        embargo_frames=args.embargo_frames, fractions=tuple(args.split_fractions),
    )
    identified = _read_table(args.identified_2d)
    dispositions = _read_table(args.dispositions)
    candidates_by_view: dict[str, pd.DataFrame] = {}
    join_counts: dict[str, dict[str, int]] = {}
    for view_input in view_inputs:
        candidates, counts = build_candidate_table(
            identified=identified, dispositions=dispositions,
            associations=_read_table(view_input.associations), geometry=geometry,
            split_map=split_map, view=view_input.view, min_mask_score=args.min_mask_score,
            min_mutual_confidence=args.min_mutual_confidence,
            max_association_distance=args.association_max_distance_px, num_fish=args.num_fish,
        )
        candidates_by_view[view_input.view] = candidates
        join_counts[view_input.view] = {str(key): int(value) for key, value in counts.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_path = output_dir / "crops.h5"
    samples, materialization_counts = materialize_crops(
        candidates_by_view, {view.view: view for view in view_inputs}, output_h5=crops_path,
        experiment=args.experiment, crop_size=args.crop_size,
        padding_fraction=args.crop_padding_fraction, sample_every=args.sample_every,
        max_fragment_gap=args.max_fragment_gap, edge_margin_px=args.edge_margin_px,
    )
    positives, negatives = build_pairs(
        samples, min_positive_delta=args.min_positive_frame_delta,
        max_positive_delta=args.max_positive_frame_delta,
        max_positive_per_fragment=args.max_positive_pairs_per_fragment,
        include_cross_view_negatives=args.include_cross_view_negatives,
        random_seed=args.random_seed,
    )
    if positives.empty or negatives.empty:
        raise RuntimeError(
            f"Dataset is not trainable: generated {len(positives)} positives and {len(negatives)} negatives."
        )
    dataset_gates = validate_dataset_gates(samples, positives, negatives)

    _write_table_both(samples, output_dir / "samples")
    _write_table_both(positives, output_dir / "positive_pairs")
    _write_table_both(negatives, output_dir / "negative_pairs")
    write_contact_sheet(crops_path, samples, output_dir / "contact_sheet.png",
                        args.contact_sheet_samples, args.random_seed)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "identity_semantics": (
            "No global biological identity labels. geometry_source_idx_local is used only within "
            "a synchronized frame/window; local_fragment_id is a must-link fragment, not a fish name."
        ),
        "experiment": args.experiment,
        "inputs": {
            "identified_2d": str(Path(args.identified_2d)),
            "dispositions": str(Path(args.dispositions)),
            "best_pre_link_json": str(Path(args.best_pre_link_json)),
            "views": [{"view": view.view, "source": str(view.source), "masks": str(view.masks),
                       "associations": str(view.associations)} for view in view_inputs],
        },
        "parameters": vars(args),
        "geometry_qc_counts": dict(geometry_counts),
        "join_qc_counts_by_view": join_counts,
        "materialization_counts": dict(materialization_counts),
        "dataset_gates": dataset_gates,
        "n_samples": int(len(samples)),
        "n_local_fragments": int(samples["local_fragment_id"].nunique()),
        "samples_by_split": {str(k): int(v) for k, v in samples["split"].value_counts().items()},
        "samples_by_view": {str(k): int(v) for k, v in samples["view"].value_counts().items()},
        "n_positive_pairs": int(len(positives)),
        "positive_pairs_by_kind": {str(k): int(v) for k, v in positives["pair_type"].value_counts().items()},
        "n_negative_pairs": int(len(negatives)),
        "negative_pairs_by_kind": {str(k): int(v) for k, v in negatives["pair_type"].value_counts().items()},
        "outputs": ["crops.h5", "samples.csv", "samples.parquet", "positive_pairs.csv",
                    "positive_pairs.parquet", "negative_pairs.csv", "negative_pairs.parquet",
                    "summary.json", "contact_sheet.png"],
    }
    # argparse Paths are strings in this CLI, but tuples/lists should stay JSON friendly.
    summary["parameters"]["view_input"] = args.view_input
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "n_samples", "n_local_fragments", "samples_by_split", "n_positive_pairs", "n_negative_pairs"
    )}, indent=2))


if __name__ == "__main__":
    main()
