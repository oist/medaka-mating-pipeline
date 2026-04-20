from __future__ import annotations

"""
Interactive 3D + multiview visualization of medaka tracking via Rerun.

Loads the same outputs as visualize_multiview.py and logs them to the
Rerun viewer for interactive scrubbing, panning, and inspection:

  - 3 synchronized camera video streams (top, side1, side2)
  - 2D SLEAP detections per view, colored by 3D fish ID
  - 3D fish positions with per-ID color and trail
  - Tank wireframe (200 mm cube)

Typical usage:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml \\
        --start-frame 171800 --end-frame 176799

    # Save for offline viewing:
    python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC \\
        --paths-config configs/paths.local.toml \\
        --save output.rrd

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


def _id_color(fish_id: str | None) -> Tuple[int, int, int, int]:
    if not fish_id:
        return UNASSIGNED_RGBA
    return ID_COLORS_RGBA.get(fish_id, UNASSIGNED_RGBA)


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


def _draw_detections_on_frame(
    frame: np.ndarray,
    instances: list,
    cam_to_id: list,
    num_individuals: int,
) -> np.ndarray:
    """Draw ID-colored keypoint circles and labels onto a BGR frame."""
    import cv2

    ranked = sorted(
        instances,
        key=lambda e: (
            -(e[1] if e[1] is not None and np.isfinite(e[1]) else -np.inf),
            e[0][0][0] if len(e[0]) else np.inf,
        ),
    )
    for slot_idx, (pts, score, _orig) in enumerate(ranked[:num_individuals]):
        fish_id = cam_to_id[slot_idx] if slot_idx < len(cam_to_id) else None
        rgba = _id_color(fish_id)
        bgr = (rgba[2], rgba[1], rgba[0])
        for x, y in pts:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(frame, (cx, cy), 10, bgr, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, bgr, -1, cv2.LINE_AA)
        if pts is not None and len(pts) > 0 and np.isfinite(pts[0]).all():
            label = fish_id if fish_id else "?"
            anchor = pts[0]
            cv2.putText(
                frame, label,
                (int(anchor[0]) + 12, int(anchor[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA,
            )
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
        rr.set_time("frame", sequence=fi)
        rr.set_time("time", duration=fi / fps)
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
) -> None:
    frames = np.arange(start_frame, end_frame + 1)
    for fid in id_labels:
        sub = df_3d[df_3d["fish_id"] == fid].set_index("frame")
        sub = sub.reindex(frames)
        xyz = sub[["X", "Y", "Z"]].to_numpy()

        color = _id_color(fid)
        for i, fi in enumerate(frames):
            rr.set_time("frame", sequence=int(fi))
            rr.set_time("time", duration=fi / fps)
            pt = xyz[i]
            if np.isfinite(pt).all():
                rr.log(
                    f"world/{fid}",
                    rr.Points3D(
                        positions=[pt.tolist()],
                        colors=[color],
                        radii=[2.0],
                        labels=[fid],
                    ),
                )
            else:
                rr.log(f"world/{fid}", rr.Clear(recursive=False))


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

    csv_path = reconstruction_output_dir / f"{effective_tag}_3d.csv"
    json_path = reconstruction_output_dir / f"{effective_tag}_best.json"
    if not csv_path.exists():
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

    has_best_json = json_path.exists()
    per_frame_mapping: Dict[int, dict] = {}
    if has_best_json:
        with json_path.open("r", encoding="utf-8") as fh:
            best_json = json.load(fh)
        id_labels = best_json.get("id_labels", id_labels)
        per_frame_mapping = {int(k): v for k, v in best_json["frames"].items()}

    preds_by_view: Dict[str, dict] = {}
    if not no_2d and has_best_json:
        from scripts.sleap.visualize_predictions import index_predictions_by_frame

        slp_dir = sleap_output_dir / experiment
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
    log_3d_trajectories(rr, df_3d, id_labels, start_frame, end_frame, fps)

    if not no_video:
        print("Logging camera frames...")
        log_camera_frames(
            rr, medaka_data_dir, experiment, fps,
            start_frame, end_frame,
            preds_by_view, per_frame_mapping,
            num_individuals=len(id_labels),
        )

    print(f"Done. Logged {n_frames} frames for {len(id_labels)} fish.")
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
        help="Skip video streams (faster logging, 3D + 2D overlays only).",
    )
    parser.add_argument(
        "--no-2d", action="store_true",
        help="Skip 2D detection overlays (no SLP loading).",
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
    if sleap_output_dir is None:
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
        sleap_output_dir=Path(sleap_output_dir),
        reconstruction_output_dir=Path(reconstruction_output_dir),
        medaka_data_dir=Path(medaka_data_dir) if medaka_data_dir else Path("."),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        save_path=args.save,
        no_video=args.no_video,
        no_2d=args.no_2d,
    )


if __name__ == "__main__":
    main()
