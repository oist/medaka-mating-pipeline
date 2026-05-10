from __future__ import annotations

"""
Interactive 3D + multiview visualization of medaka tracking via Rerun.

Loads the same outputs as visualize_multiview.py and logs them to the
Rerun viewer for interactive scrubbing, panning, and inspection:

  - 3 synchronized camera video streams (top, side1, side2)
  - 2D SLEAP detections per view, colored by 3D fish ID
  - 3D fish positions with per-ID color and trail
  - Optional pre-link 3D candidates and registration-error timelines
  - Per-ID 3D jump timelines and optional relink decision markers
  - Tank wireframe (200 mm cube)

Typical usage:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml \\
        --start-frame 171800 --end-frame 176799

    # Save for offline viewing:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml \\
        --save output.rrd

    # Full-run diagnostic pass without enormous video payloads:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml --no-video --save full_qc.rrd

    # View relinked outputs from scripts.reconstruction_3d.relink:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml --tag 20251225_K_1_2_FCMC_relinked

Requires: rerun-sdk (install via `uv sync --extra rerun`).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.load_2d_slp import VIEW_TO_CAM

VIEW_ORDER = ["top_cam", "side1_cam", "side2_cam"]
VIEW_TO_VIDEO = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}

ID_COLORS_RGBA: Dict[str, Tuple[int, int, int, int]] = {
    "ID_1": (31, 119, 180, 255),
    "ID_2": (255, 127, 14, 255),
    "ID_3": (44, 160, 44, 255),
    "ID_4": (214, 39, 40, 255),
    "ID_5": (148, 103, 189, 255),
    "ID_6": (140, 86, 75, 255),
    "ID_7": (227, 119, 194, 255),
    "ID_8": (127, 127, 127, 255),
}
UNASSIGNED_RGBA = (110, 110, 110, 180)
PRE_LINK_RGBA = (210, 210, 210, 150)
BAD_RESIDUAL_RGBA = (220, 50, 47, 230)


def _id_color(fish_id: str | None) -> Tuple[int, int, int, int]:
    if not fish_id:
        return UNASSIGNED_RGBA
    return ID_COLORS_RGBA.get(fish_id, UNASSIGNED_RGBA)


def _source_labels(num_individuals: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(num_individuals)]


def _finite_point(pt) -> bool:
    arr = np.asarray(pt, dtype=float)
    return arr.shape == (3,) and np.isfinite(arr).all()


def _set_time(rr, frame: int, fps: float) -> None:
    """Set timelines across Rerun SDK versions used by this project."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=int(frame))
        rr.set_time("time", duration=float(frame) / fps)
    else:
        rr.set_time_sequence("frame", int(frame))
        rr.set_time_seconds("time", float(frame) / fps)


def _scalar_archetype(rr, value: float):
    if hasattr(rr, "Scalar"):
        return rr.Scalar(float(value))
    return rr.Scalars(float(value))


def _log_scalar(rr, entity_path: str, value: float | None) -> None:
    if value is None:
        value = float("nan")
    rr.log(entity_path, _scalar_archetype(rr, float(value)))


def _json_path_or_none(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _first_existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _path_candidates(
    reconstruction_output_dir: Path,
    tag: str,
    *,
    kind: str,
) -> List[Path]:
    if kind == "csv":
        return [reconstruction_output_dir / f"{tag}_3d.csv"]
    if kind == "best":
        paths = [reconstruction_output_dir / f"{tag}_best.json"]
        if tag.endswith("_relinked"):
            base = tag[: -len("_relinked")]
            paths.insert(0, reconstruction_output_dir / f"{base}_best_relinked.json")
        return paths
    if kind == "pre_link":
        paths = [reconstruction_output_dir / f"{tag}_best_pre_link.json"]
        if tag.endswith("_relinked"):
            base = tag[: -len("_relinked")]
            paths.insert(0, reconstruction_output_dir / f"{base}_best_pre_link.json")
        return paths
    if kind == "decisions":
        paths = [reconstruction_output_dir / f"{tag}_relink_decisions.csv"]
        if tag.endswith("_relinked"):
            base = tag[: -len("_relinked")]
            paths.insert(0, reconstruction_output_dir / f"{base}_relink_decisions.csv")
        return paths
    raise ValueError(f"Unknown path kind: {kind}")


def _tank_wireframe_strips(
    lo: float = 0.0, hi: float = 200.0,
) -> list[list[list[float]]]:
    c = [
        [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
        [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [[c[a], c[b]] for a, b in edges]


def _estimate_json_to_csv_offset(
    df_3d: pd.DataFrame,
    per_frame_mapping: Dict[int, dict],
    id_labels: List[str],
    *,
    max_samples: int = 5000,
) -> np.ndarray:
    """
    Estimate coordinate offset between raw JSON groups and exported CSV.

    `reconstruct_from_slp` often writes the CSV after shifting the minimum
    coordinate to zero, while the JSON preserves the raw model coordinates.
    This lets optional pre-link candidate points sit in the same scene as the
    CSV track without changing reconstruction outputs.
    """
    offsets: list[np.ndarray] = []
    sample_rows = df_3d.dropna(subset=["X", "Y", "Z"])
    if sample_rows.empty:
        return np.zeros(3, dtype=float)

    for row in sample_rows.itertuples(index=False):
        frame = int(row.frame)
        fish_id = str(row.fish_id)
        if fish_id not in id_labels:
            continue
        groups = (per_frame_mapping.get(frame) or {}).get("groups", {}) or {}
        raw_pt = groups.get(fish_id)
        if not _finite_point(raw_pt):
            continue
        csv_pt = np.asarray([row.X, row.Y, row.Z], dtype=float)
        offsets.append(csv_pt - np.asarray(raw_pt, dtype=float))
        if len(offsets) >= max_samples:
            break

    if not offsets:
        return np.zeros(3, dtype=float)
    return np.nanmedian(np.vstack(offsets), axis=0)


def _load_pre_link_payload(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def log_pre_link_diagnostics(
    rr,
    pre_link_payload: dict,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    coordinate_offset: np.ndarray,
    residual_warning: float,
) -> None:
    """
    Log pre-link triangulated candidates and registration-error timeseries.

    The pre-link artifact is the best per-frame triangulation before temporal
    identity linking. It is the most direct way to separate "triangulation got
    weak" from "linking swapped an otherwise good point".
    """
    frames_payload = pre_link_payload.get("frames", {}) or {}
    num_individuals = int(pre_link_payload.get("num_individuals", 3))
    src_labels = list(pre_link_payload.get("src_labels") or _source_labels(num_individuals))

    for frame in range(start_frame, end_frame + 1):
        info = frames_payload.get(str(frame)) or frames_payload.get(frame) or {}
        groups = info.get("groups", {}) or {}
        errors = info.get("errors", {}) or {}
        _set_time(rr, frame, fps)

        positions: list[list[float]] = []
        colors: list[Tuple[int, int, int, int]] = []
        labels: list[str] = []
        max_err = float("nan")
        finite_errors: list[float] = []

        for src_label in src_labels:
            err = errors.get(src_label)
            err_val = float(err) if err is not None and np.isfinite(err) else float("nan")
            _log_scalar(rr, f"diagnostics/registration_error/{src_label}", err_val)
            if np.isfinite(err_val):
                finite_errors.append(err_val)

            pt = groups.get(src_label)
            if not _finite_point(pt):
                continue
            shifted = np.asarray(pt, dtype=float) + coordinate_offset
            color = BAD_RESIDUAL_RGBA if np.isfinite(err_val) and err_val > residual_warning else PRE_LINK_RGBA
            positions.append([float(v) for v in shifted])
            colors.append(color)
            if np.isfinite(err_val):
                labels.append(f"{src_label} err={err_val:.0f}")
            else:
                labels.append(src_label)

        if finite_errors:
            max_err = float(max(finite_errors))
        _log_scalar(rr, "diagnostics/registration_error/max", max_err)

        if positions:
            rr.log(
                "world/pre_link_candidates",
                rr.Points3D(
                    positions=positions,
                    colors=colors,
                    radii=[1.0] * len(positions),
                    labels=labels,
                ),
            )
        else:
            rr.log("world/pre_link_candidates", rr.Clear(recursive=False))


def log_relink_decisions(
    rr,
    decisions_csv: Path | None,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    if decisions_csv is None or not decisions_csv.exists():
        return
    decisions = pd.read_csv(decisions_csv)
    if "frame" not in decisions.columns:
        return
    decisions = decisions[
        (decisions["frame"] >= start_frame) & (decisions["frame"] <= end_frame)
    ]
    for row in decisions.itertuples(index=False):
        frame = int(row.frame)
        _set_time(rr, frame, fps)
        _log_scalar(rr, "diagnostics/relink_event", 1.0)
        if hasattr(row, "fwd_jump_mm"):
            _log_scalar(rr, "diagnostics/relink/fwd_jump_mm", float(row.fwd_jump_mm))
        if hasattr(row, "bwd_jump_mm"):
            _log_scalar(rr, "diagnostics/relink/bwd_jump_mm", float(row.bwd_jump_mm))


def _draw_detections_on_frame(
    frame: np.ndarray,
    instances: list,
    cam_to_id: list,
    num_individuals: int,
) -> np.ndarray:
    """Draw ID-colored keypoint circles for assigned detections and dim grey
    dots for unassigned ones — so it's visible when a real fish gets culled
    by the 3D reconstruction's tank-bounds / mutually-exclusive triplet pick.
    """
    import cv2

    ranked = sorted(
        instances,
        key=lambda e: (
            -(e[1] if e[1] is not None and np.isfinite(e[1]) else -np.inf),
            e[0][0][0] if len(e[0]) else np.inf,
        ),
    )
    # Cull-overlay style for detections the reconstructor didn't pick.
    cull_fill_bgr = (180, 180, 180)
    cull_outline_bgr = (60, 60, 60)

    for slot_idx, (pts, score, _orig) in enumerate(ranked):
        fish_id = cam_to_id[slot_idx] if slot_idx < len(cam_to_id) else None
        if fish_id is not None:
            rgba = _id_color(fish_id)
            bgr = (rgba[2], rgba[1], rgba[0])
            for x, y in pts:
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                cx, cy = int(round(x)), int(round(y))
                cv2.circle(frame, (cx, cy), 10, bgr, 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, bgr, -1, cv2.LINE_AA)
            if pts is not None and len(pts) > 0 and np.isfinite(pts[0]).all():
                anchor = pts[0]
                cv2.putText(
                    frame, fish_id,
                    (int(anchor[0]) + 12, int(anchor[1]) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA,
                )
        else:
            # Unassigned raw prediction (extra n=6-fish candidate or one the
            # reconstructor rejected). Draw small dim dot.
            for x, y in pts:
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                cx, cy = int(round(x)), int(round(y))
                cv2.circle(frame, (cx, cy), 4, cull_outline_bgr, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, cull_fill_bgr, -1, cv2.LINE_AA)
    return frame


def log_camera_frames(
    rr,
    medaka_data_dir: Path,
    experiment: str,
    fps: float,
    start_frame: int,
    end_frame: int,
    preds_by_view: Dict[str, dict],
    per_frame_mapping: Dict[int, dict],
    num_individuals: int = 3,
) -> None:
    import cv2

    caps = {}
    for view in VIEW_ORDER:
        video_path = medaka_data_dir / experiment / view / VIEW_TO_VIDEO[view]
        if not video_path.exists():
            print(f"Warning: video not found, skipping: {video_path}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: cannot open video, skipping: {video_path}")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        caps[view] = cap

    for fi in range(start_frame, end_frame + 1):
        _set_time(rr, fi, fps)
        mapping_info = per_frame_mapping.get(fi, {})

        for view, cap in caps.items():
            ok, frame = cap.read()
            if not ok:
                continue

            cam = VIEW_TO_CAM[view]
            instances = preds_by_view.get(view, {}).get(fi, [])
            cam_to_id = mapping_info.get(f"{cam}_to_id", [])
            if instances:
                frame = _draw_detections_on_frame(
                    frame, instances, cam_to_id, num_individuals,
                )

            _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rr.log(
                f"cameras/{view}",
                rr.EncodedImage(contents=jpeg_buf.tobytes(), media_type="image/jpeg"),
            )

    for cap in caps.values():
        cap.release()


def log_3d_trajectories(
    rr,
    df_3d: pd.DataFrame,
    id_labels: List[str],
    start_frame: int,
    end_frame: int,
    fps: float,
    jump_warning_mm: float,
    trail_seconds: float = 1.0,
) -> None:
    frames = np.arange(start_frame, end_frame + 1)
    trail_window = max(1, int(round(fps * trail_seconds)))
    for fid in id_labels:
        sub = df_3d[df_3d["fish_id"] == fid].set_index("frame")
        sub = sub.reindex(frames)
        xyz = sub[["X", "Y", "Z"]].to_numpy()

        color = _id_color(fid)
        prev_valid: np.ndarray | None = None
        for i, fi in enumerate(frames):
            _set_time(rr, int(fi), fps)
            pt = xyz[i]
            if np.isfinite(pt).all():
                jump = (
                    float(np.linalg.norm(pt - prev_valid))
                    if prev_valid is not None and np.isfinite(prev_valid).all()
                    else float("nan")
                )
                _log_scalar(rr, f"diagnostics/jump_mm/{fid}", jump)
                rr.log(
                    f"world/{fid}",
                    rr.Points3D(
                        positions=[pt.tolist()],
                        colors=[
                            BAD_RESIDUAL_RGBA
                            if np.isfinite(jump) and jump > jump_warning_mm
                            else color
                        ],
                        radii=[3.5 if np.isfinite(jump) and jump > jump_warning_mm else 2.0],
                        labels=[fid],
                    ),
                )

                # Trailing polyline of the last `trail_window` positions, with
                # NaN-gaps split into separate strips so missing frames don't
                # connect across discontinuities.
                lo = max(0, i - trail_window + 1)
                window = xyz[lo : i + 1]
                strips: list[list[list[float]]] = []
                run: list[list[float]] = []
                for p in window:
                    if np.isfinite(p).all():
                        run.append(p.tolist())
                    elif run:
                        if len(run) >= 2:
                            strips.append(run)
                        run = []
                if len(run) >= 2:
                    strips.append(run)
                if strips:
                    rr.log(
                        f"world/{fid}/trail",
                        rr.LineStrips3D(
                            strips,
                            colors=[(*color[:3], 200)] * len(strips),
                            radii=0.6,
                        ),
                    )
                else:
                    rr.log(f"world/{fid}/trail", rr.Clear(recursive=False))

                prev_valid = pt.copy()
            else:
                rr.log(f"world/{fid}", rr.Clear(recursive=False))
                rr.log(f"world/{fid}/trail", rr.Clear(recursive=False))
                _log_scalar(rr, f"diagnostics/jump_mm/{fid}", float("nan"))


def run_viewer(
    *,
    experiment: str,
    tag: str | None,
    sleap_output_dir: Path,
    reconstruction_output_dir: Path,
    medaka_data_dir: Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    fps: float = 166.0,
    save_path: str | None = None,
    no_video: bool = False,
    no_2d: bool = False,
    csv_path_override: Path | None = None,
    best_json_override: Path | None = None,
    pre_link_json_override: Path | None = None,
    decisions_csv_override: Path | None = None,
    slp_dir_override: Path | None = None,
    no_pre_link: bool = False,
    residual_warning: float = 5000.0,
    jump_warning_mm: float = 15.0,
    trail_seconds: float = 1.0,
) -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "rerun-sdk is required. Install with:\n"
            "    uv sync --extra rerun\n"
            "or  pip install rerun-sdk"
        ) from exc

    effective_tag = tag or experiment

    csv_path = (
        csv_path_override
        if csv_path_override is not None
        else _first_existing(_path_candidates(reconstruction_output_dir, effective_tag, kind="csv"))
    )
    json_path = (
        best_json_override
        if best_json_override is not None
        else _first_existing(_path_candidates(reconstruction_output_dir, effective_tag, kind="best"))
    )
    pre_link_path = (
        pre_link_json_override
        if pre_link_json_override is not None
        else _first_existing(_path_candidates(reconstruction_output_dir, effective_tag, kind="pre_link"))
    )
    decisions_csv = (
        decisions_csv_override
        if decisions_csv_override is not None
        else _first_existing(_path_candidates(reconstruction_output_dir, effective_tag, kind="decisions"))
    )
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"3D CSV not found: {csv_path}")

    df_3d = pd.read_csv(csv_path)
    all_frames = sorted(df_3d["frame"].unique())
    f_min, f_max = int(all_frames[0]), int(all_frames[-1])
    if start_frame is None:
        start_frame = f_min
    if end_frame is None:
        end_frame = f_max

    start_frame = max(start_frame, f_min)
    end_frame = min(end_frame, f_max)
    n_frames = end_frame - start_frame + 1
    print(f"Frames {start_frame}..{end_frame} ({n_frames} frames, {n_frames/fps:.1f}s)")

    df_3d = df_3d[(df_3d["frame"] >= start_frame) & (df_3d["frame"] <= end_frame)].copy()
    id_labels = sorted(df_3d["fish_id"].dropna().unique())

    has_best_json = json_path is not None and json_path.exists()
    per_frame_mapping: Dict[int, dict] = {}
    if has_best_json:
        with json_path.open("r", encoding="utf-8") as fh:
            best_json = json.load(fh)
        id_labels = best_json.get("id_labels", id_labels)
        per_frame_mapping = {int(k): v for k, v in best_json["frames"].items()}
    else:
        print("Warning: best JSON not found; 2D overlays will be unassigned.")

    preds_by_view: Dict[str, dict] = {}
    if not no_video and not no_2d and has_best_json:
        from scripts.sleap.visualize_predictions import index_predictions_by_frame

        slp_dir = slp_dir_override if slp_dir_override is not None else sleap_output_dir / experiment
        for view in VIEW_ORDER:
            slp_path = slp_dir / f"{view}.slp"
            if slp_path.exists():
                print(f"Loading SLEAP predictions: {view}...")
                by_frame, _ = index_predictions_by_frame(slp_path)
                preds_by_view[view] = by_frame
            else:
                print(f"Warning: {slp_path} not found, skipping 2D detections for {view}")

    app_id = f"medaka_{experiment}"
    rr.init(app_id, spawn=save_path is None)
    if save_path:
        rr.save(save_path)
        print(f"Saving to {save_path}...")

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log(
        "world/tank",
        rr.LineStrips3D(
            strips=_tank_wireframe_strips(0.0, 200.0),
            colors=[(100, 100, 100, 120)],
        ),
        static=True,
    )

    rr.log(
        "world",
        rr.AnnotationContext([
            rr.AnnotationInfo(id=i + 1, label=fid, color=_id_color(fid)[:3])
            for i, fid in enumerate(id_labels)
        ]),
        static=True,
    )

    print("Logging 3D trajectories...")
    log_3d_trajectories(
        rr,
        df_3d,
        id_labels,
        start_frame,
        end_frame,
        fps,
        jump_warning_mm,
        trail_seconds=trail_seconds,
    )

    if not no_pre_link:
        pre_link_payload = _load_pre_link_payload(pre_link_path)
        if pre_link_payload:
            print(f"Logging pre-link diagnostics: {pre_link_path}")
            offset = _estimate_json_to_csv_offset(df_3d, per_frame_mapping, id_labels)
            log_pre_link_diagnostics(
                rr,
                pre_link_payload,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                coordinate_offset=offset,
                residual_warning=residual_warning,
            )
        else:
            print("Pre-link JSON not found; skipping residual/candidate diagnostics.")

    if decisions_csv is not None and decisions_csv.exists():
        print(f"Logging relink decisions: {decisions_csv}")
        log_relink_decisions(
            rr,
            decisions_csv,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
        )

    if not no_video:
        print("Logging camera frames...")
        log_camera_frames(
            rr, medaka_data_dir, experiment, fps,
            start_frame, end_frame,
            preds_by_view, per_frame_mapping,
            num_individuals=len(id_labels),
        )

    print(f"Done. Logged {n_frames} frames for {len(id_labels)} fish.")
    print(f"3D CSV: {csv_path}")
    if json_path:
        print(f"Best JSON: {json_path}")
    if save_path:
        print(f"Saved: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive 3D + multiview visualization via Rerun.",
    )
    parser.add_argument(
        "experiment",
        help="Experiment folder name, e.g. 20251225_K_1_2_FCMC.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Reconstruction tag (default: same as experiment name).",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fps", type=float, default=166.0)
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save to .rrd file instead of spawning the viewer.",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video streams (faster full-run 3D diagnostics).",
    )
    parser.add_argument(
        "--no-2d", action="store_true",
        help="Skip 2D detection overlays (no SLP loading).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Explicit 3D CSV path. Overrides --tag path resolution.",
    )
    parser.add_argument(
        "--best-json",
        default=None,
        help="Explicit post-link best JSON path. Supports *_best_relinked.json.",
    )
    parser.add_argument(
        "--pre-link-json",
        default=None,
        help="Explicit pre-link JSON path for residual/candidate diagnostics.",
    )
    parser.add_argument(
        "--decisions-csv",
        default=None,
        help="Explicit relink decisions CSV path for event markers.",
    )
    parser.add_argument(
        "--slp-dir",
        default=None,
        help=(
            "Explicit directory containing top_cam.slp / side1_cam.slp / "
            "side2_cam.slp. Overrides the default <sleap_output_dir>/<experiment> "
            "lookup; useful when comparing alternate inference rounds."
        ),
    )
    parser.add_argument(
        "--no-pre-link",
        action="store_true",
        help="Skip pre-link candidate points and registration-error timelines.",
    )
    parser.add_argument(
        "--residual-warning",
        type=float,
        default=5000.0,
        help="Color pre-link candidates red above this registration error.",
    )
    parser.add_argument(
        "--jump-warning-mm",
        type=float,
        default=15.0,
        help="Color 3D track points red above this inter-sample jump.",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=1.0,
        help=(
            "How many seconds of past 3D positions to draw as a trailing "
            "polyline behind each fish (default 1.0). Use 0 to disable trails."
        ),
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--sleap-output-dir", default=None)
    parser.add_argument("--reconstruction-output-dir", default=None)
    parser.add_argument("--medaka-data-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local.sleap_output_dir if local else None,
    )
    reconstruction_output_dir = choose_path(
        args.reconstruction_output_dir,
        local.reconstruction_output_dir if local else None,
    )
    medaka_data_dir = choose_path(
        args.medaka_data_dir,
        local.medaka_data_dir if local else None,
    )
    if sleap_output_dir is None and not (args.no_video or args.no_2d):
        parser.error("sleap_output_dir required (paths.local.toml or --sleap-output-dir).")
    if reconstruction_output_dir is None:
        parser.error(
            "reconstruction_output_dir required "
            "(paths.local.toml or --reconstruction-output-dir)."
        )
    if medaka_data_dir is None and not args.no_video:
        parser.error("medaka_data_dir required (paths.local.toml or --medaka-data-dir).")

    run_viewer(
        experiment=args.experiment,
        tag=args.tag,
        sleap_output_dir=Path(sleap_output_dir) if sleap_output_dir else Path("."),
        reconstruction_output_dir=Path(reconstruction_output_dir),
        medaka_data_dir=Path(medaka_data_dir) if medaka_data_dir else Path("."),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        save_path=args.save,
        no_video=args.no_video,
        no_2d=args.no_2d,
        csv_path_override=_json_path_or_none(args.csv),
        best_json_override=_json_path_or_none(args.best_json),
        pre_link_json_override=_json_path_or_none(args.pre_link_json),
        decisions_csv_override=_json_path_or_none(args.decisions_csv),
        slp_dir_override=_json_path_or_none(args.slp_dir),
        no_pre_link=args.no_pre_link,
        residual_warning=args.residual_warning,
        jump_warning_mm=args.jump_warning_mm,
        trail_seconds=args.trail_seconds,
    )


if __name__ == "__main__":
    main()
