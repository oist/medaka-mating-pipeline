from __future__ import annotations

"""
Render SLEAP predictions on top of a video for QC.

Loads a `.slp` file (or chunks via the experiment_metadata.json), pulls the
matching frames from the source video, draws each predicted instance's
keypoints, and writes an annotated mp4.

Typical usage:
    python -m scripts.sleap.visualize_predictions \\
        --slp /path/to/top_cam.slp \\
        --video /path/to/cam0_top.mp4 \\
        --frames 0-1660 \\
        --output /path/to/top_cam.viz.mp4
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths


# Distinct BGR colors for up to 8 instances per frame (cv2 uses BGR).
INSTANCE_COLORS = [
    (0, 255, 0),     # green
    (0, 165, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 0, 0),     # blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
    (200, 200, 200), # gray
]


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    track_id: int
    x: float
    y: float
    identity: str = ""
    score: str = ""


def parse_frame_range(spec: str | None, total_frames: int) -> tuple[int, int]:
    if spec is None:
        return 0, total_frames - 1
    if "-" not in spec:
        raise ValueError(f"--frames must look like START-END, got {spec!r}")
    start_s, end_s = spec.split("-", 1)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else total_frames - 1
    if start < 0 or end >= total_frames or start > end:
        raise ValueError(
            f"frame range {start}-{end} out of bounds for video with "
            f"{total_frames} frames"
        )
    return start, end


def load_track_csv(csv_path: Path) -> dict[tuple[int, int, int], int]:
    """
    Load a tracking CSV (frame, track_id, x, y, ...) and return a mapping
    {(frame_idx, x_int, y_int): track_id}. We index by rounded coordinates so
    we can look up the track for a given prediction without re-running the
    matching.
    """
    import csv as _csv
    out: dict[tuple[int, int, int], int] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                fi = int(row["frame"])
                tid = int(row["track_id"])
                xi = int(round(float(row["x"])))
                yi = int(round(float(row["y"])))
            except (KeyError, ValueError):
                continue
            out[(fi, xi, yi)] = tid
    return out


def load_track_points_by_frame(csv_path: Path) -> dict[int, list[TrackPoint]]:
    """Load track rows grouped by frame for direct drawing and nearest matching."""
    import csv as _csv

    out: dict[int, list[TrackPoint]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                fi = int(row["frame"])
                tid = int(row["track_id"])
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, ValueError):
                continue
            out.setdefault(fi, []).append(
                TrackPoint(
                    frame=fi,
                    track_id=tid,
                    x=x,
                    y=y,
                    identity=str(row.get("identity", "")),
                    score=str(row.get("score", "")),
                )
            )
    return out


def index_predictions_by_frame(slp_path: Path) -> tuple[dict[int, list], list[str]]:
    """
    Load `.slp` predictions and return:
        - mapping {frame_idx: [(numpy_keypoints, score, color_idx), ...]}
        - skeleton node names (in order)

    `numpy_keypoints` is an (n_nodes, 2) float array; NaN where the node is
    missing.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required. Install with `pip install sleap-io` "
            "or `uv sync --extra sleap`."
        ) from exc

    labels = sio.load_slp(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError(f"No skeleton found in {slp_path}")
    node_names = [n.name for n in labels.skeletons[0].nodes]

    by_frame: dict[int, list] = {}
    for f in labels.labeled_frames:
        instances = []
        for i, inst in enumerate(f.instances):
            pts = inst.numpy()  # (n_nodes, 2)
            score = getattr(inst, "score", None)
            instances.append((pts, score, i))
        by_frame[int(f.frame_idx)] = instances
    return by_frame, node_names


def draw_predictions(
    frame: np.ndarray,
    instances: list,
    node_names: list[str],
    radius: int = 6,
    show_node_names: bool = False,
    frame_idx: int | None = None,
    track_lookup: dict[tuple[int, int, int], int] | None = None,
    track_points_by_frame: dict[int, list[TrackPoint]] | None = None,
    track_match_distance: float = 60.0,
) -> np.ndarray:
    out = frame
    frame_track_points = track_points_by_frame.get(frame_idx, []) if (
        track_points_by_frame is not None and frame_idx is not None
    ) else []
    for pts, score, idx in instances:
        # Default colour: per-instance index. If a tracking CSV is available
        # for this frame, override using the persistent track_id. Exact point
        # lookup supports SLEAP-point tracks; nearest lookup supports blob
        # centroid tracks from scripts.tracking.top_camera_identity.
        track_id = None
        if track_lookup is not None and frame_idx is not None:
            for (x, y) in pts:
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                key = (frame_idx, int(round(x)), int(round(y)))
                if key in track_lookup:
                    track_id = track_lookup[key]
                    break
        if track_id is None and frame_track_points:
            best_tid = None
            best_dist = float("inf")
            for (x, y) in pts:
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                for tp in frame_track_points:
                    if tp.track_id < 0:
                        continue
                    dist = float(np.hypot(float(x) - tp.x, float(y) - tp.y))
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tp.track_id
            if best_tid is not None and best_dist <= track_match_distance:
                track_id = best_tid
        color_idx = track_id if track_id is not None else idx
        color = INSTANCE_COLORS[color_idx % len(INSTANCE_COLORS)]

        for ni, (x, y) in enumerate(pts):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(out, (cx, cy), radius, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 2, color, thickness=-1, lineType=cv2.LINE_AA)
            if show_node_names and ni < len(node_names):
                cv2.putText(
                    out, node_names[ni], (cx + radius + 2, cy - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )
        if len(pts) > 0:
            anchor = pts[0]
            if np.isfinite(anchor[0]) and np.isfinite(anchor[1]):
                label_parts = []
                if track_id is not None:
                    label_parts.append(f"id={track_id}")
                if score is not None:
                    label_parts.append(f"{score:.2f}")
                if label_parts:
                    cv2.putText(
                        out, " ".join(label_parts),
                        (int(anchor[0]) + radius + 2, int(anchor[1]) + radius + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                    )
    return out


def draw_track_points(
    frame: np.ndarray,
    track_points: list[TrackPoint],
    *,
    radius: int = 8,
) -> np.ndarray:
    """Draw CSV track points directly, independent of SLEAP instance order."""
    out = frame
    for tp in track_points:
        color = INSTANCE_COLORS[tp.track_id % len(INSTANCE_COLORS)] if tp.track_id >= 0 else INSTANCE_COLORS[-1]
        cx, cy = int(round(tp.x)), int(round(tp.y))
        cv2.circle(out, (cx, cy), radius, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, color, thickness=-1, lineType=cv2.LINE_AA)
        label = tp.identity or (f"id={tp.track_id}" if tp.track_id >= 0 else "unknown")
        cv2.putText(
            out,
            label,
            (cx + radius + 2, cy + radius + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def render(
    video_path: Path,
    slp_path: Path,
    output_path: Path,
    frames_spec: str | None = None,
    every: int = 1,
    radius: int = 6,
    show_node_names: bool = False,
    fps: float | None = None,
    track_csv: Path | None = None,
    draw_track_csv_points: bool = True,
    draw_sleap_points: bool = True,
    track_match_distance: float = 60.0,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start, end = parse_frame_range(frames_spec, total_frames)
    out_fps = fps if fps else src_fps / max(1, every)

    print(f"Source video : {video_path} ({total_frames} frames, {src_fps:.1f} fps)")
    print(f"Predictions  : {slp_path}")
    print(f"Frame range  : {start}-{end}  (every {every}, output fps={out_fps:.1f})")

    by_frame, node_names = index_predictions_by_frame(slp_path)
    n_pred_total = sum(len(v) for v in by_frame.values())
    print(f"Loaded {n_pred_total} instances across {len(by_frame)} frames; "
          f"nodes: {node_names}")

    track_lookup = None
    track_points_by_frame = None
    if track_csv is not None:
        track_lookup = load_track_csv(track_csv)
        track_points_by_frame = load_track_points_by_frame(track_csv)
        n_track_points = sum(len(v) for v in track_points_by_frame.values())
        print(f"Loaded {n_track_points} track rows from {track_csv}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    annotated = 0
    for fi in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - start) % every != 0:
            continue
        instances = by_frame.get(fi, [])
        if instances and draw_sleap_points:
            annotated += 1
            frame = draw_predictions(
                frame, instances, node_names,
                radius=radius, show_node_names=show_node_names,
                frame_idx=fi,
                track_lookup=track_lookup,
                track_points_by_frame=track_points_by_frame,
                track_match_distance=track_match_distance,
            )
        elif instances:
            annotated += 1
        if draw_track_csv_points and track_points_by_frame is not None:
            frame = draw_track_points(frame, track_points_by_frame.get(fi, []), radius=radius + 2)
        cv2.putText(
            frame, f"frame {fi}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    print(f"Wrote {written} frames ({annotated} with predictions) -> {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Overlay SLEAP predictions on a video and save an mp4."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help=(
            "Experiment folder name (e.g. 20251220_D_1_1_FCMC). When provided, "
            "--slp/--video are auto-resolved from paths.local.toml. Otherwise "
            "pass --slp and --video explicitly."
        ),
    )
    parser.add_argument("--view", default="top_cam",
                        help="View to render (default top_cam).")
    parser.add_argument("--slp", default=None, help="Path to a merged .slp file.")
    parser.add_argument("--video", default=None, help="Path to the source video.")
    parser.add_argument("--output", default=None,
                        help="Output mp4 path. Defaults to <slp>.viz.mp4.")
    parser.add_argument("--frames", default=None,
                        help="Frame range, e.g. '0-1660' (inclusive).")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Render the first N seconds (overrides --frames).")
    parser.add_argument("--every", type=int, default=1,
                        help="Subsample factor (1 = every frame).")
    parser.add_argument("--radius", type=int, default=6,
                        help="Keypoint marker radius in px.")
    parser.add_argument("--show-node-names", action="store_true",
                        help="Draw node names next to each keypoint.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override output fps (otherwise src_fps/every).")
    parser.add_argument("--track-csv", default=None,
                        help="Tracking CSV (frame,track_id,x,y,...) to colour by track_id.")
    parser.add_argument("--no-draw-track-points", action="store_true",
                        help="Use track CSV only for recolouring SLEAP points; do not draw CSV points.")
    parser.add_argument("--track-only", action="store_true",
                        help="Draw only track CSV points, hiding raw SLEAP keypoints.")
    parser.add_argument("--track-match-distance", type=float, default=60.0,
                        help="Nearest-pixel distance for matching track CSV points to SLEAP points.")
    parser.add_argument("--paths-config", default=None,
                        help="paths.local.toml path (for experiment lookup).")
    parser.add_argument("--sleap-output-dir", default=None)
    parser.add_argument("--medaka-data-dir", default=None)
    args = parser.parse_args(argv)

    slp_path: Path | None = Path(args.slp) if args.slp else None
    video_path: Path | None = Path(args.video) if args.video else None

    if args.experiment is not None and (slp_path is None or video_path is None):
        local = load_local_paths(args.paths_config)
        sleap_output_dir = choose_path(
            args.sleap_output_dir,
            local.sleap_output_dir if local else None,
        )
        medaka_data_dir = choose_path(
            args.medaka_data_dir,
            local.medaka_data_dir if local else None,
        )
        if sleap_output_dir is None or medaka_data_dir is None:
            parser.error(
                "Need both sleap_output_dir and medaka_data_dir (paths.local.toml "
                "or --sleap-output-dir/--medaka-data-dir) to resolve from an experiment name."
            )
        if slp_path is None:
            slp_path = Path(sleap_output_dir) / args.experiment / f"{args.view}.slp"
        if video_path is None:
            from scripts.common.experiment_metadata import VIDEO_BASENAMES
            video_path = (
                Path(medaka_data_dir) / args.experiment / args.view / VIDEO_BASENAMES[args.view]
            )

    if slp_path is None or video_path is None:
        parser.error("--slp and --video are required (or pass an experiment name).")

    if not slp_path.exists():
        parser.error(f"SLP file not found: {slp_path}")
    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    output_path = Path(args.output) if args.output else slp_path.with_suffix(".viz.mp4")

    frames_spec = args.frames
    if args.seconds is not None:
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        n_frames = int(round(args.seconds * src_fps))
        frames_spec = f"0-{n_frames - 1}"

    track_csv = Path(args.track_csv) if args.track_csv else None
    if track_csv is not None and not track_csv.exists():
        parser.error(f"Track CSV not found: {track_csv}")
    if args.track_only and track_csv is None:
        parser.error("--track-only requires --track-csv")

    render(
        video_path=video_path,
        slp_path=slp_path,
        output_path=output_path,
        frames_spec=frames_spec,
        every=args.every,
        radius=args.radius,
        show_node_names=args.show_node_names,
        fps=args.fps,
        track_csv=track_csv,
        draw_track_csv_points=not args.no_draw_track_points,
        draw_sleap_points=not args.track_only,
        track_match_distance=args.track_match_distance,
    )


if __name__ == "__main__":
    main()
