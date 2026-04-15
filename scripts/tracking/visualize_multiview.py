from __future__ import annotations

"""
Render a multiview QC clip: top, side1, side2 with predictions coloured by
their 3D fish ID, plus a diagonal 3D panel showing the reconstructed
trajectories (trails + current position + tank wireframe) up to the current
frame.

Expects the outputs of `scripts.reconstruction_3d.reconstruct_from_slp`:
    - <tag>_3d.csv              (positions per frame per fish_id)
    - <tag>_best.json           (per-frame detection -> ID mapping)
    - <view>.slp                (merged SLEAP predictions per view)
    - <view>.mp4                (raw recording per view)

Panel layout (2 x 2):
    +----------------+----------------+
    |    top_cam     |    side1_cam   |
    +----------------+----------------+
    |    side2_cam   |    3D diagonal |
    +----------------+----------------+

The 3D panel is drawn with cv2 (no matplotlib): a rotation matrix projects
world XYZ (mm) onto the 2D panel (default `azim=-60°, elev=25°`, black
background, 0-200 mm tank wireframe for reference). World X is flipped
before projection (`invert_x=True`) so left-right matches cam0; switch off
via `--no-invert-x` on the CLI.

Typical usage:
    python -m scripts.tracking.visualize_multiview 20251220_D_1_1_FCMC \\
        --paths-config configs/paths.local.toml \\
        --tag 20251220_D_1_1_FCMC_mid5k_gated \\
        --start-frame 171800 --end-frame 172629 \\
        --output C:/tmp/medaka_3d/multiview.mp4

Optional 3D-panel CLI flags:
    --azim <deg>        rotation around world Z (default -60)
    --elev <deg>        tilt above horizontal (default 25)
    --no-invert-x       keep world X unflipped
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.load_2d_slp import VIEW_TO_CAM
from scripts.sleap.visualize_predictions import index_predictions_by_frame


VIEW_ORDER = ["top_cam", "side1_cam", "side2_cam"]
VIEW_TO_VIDEO = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
# Colour cycle in BGR (matches the matplotlib tab10 order we used for the 3D plots).
ID_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "ID_1": (180, 119, 31),   # matplotlib tab:blue
    "ID_2": (14, 127, 255),   # tab:orange
    "ID_3": (44, 160, 44),    # tab:green
    "ID_4": (40, 39, 214),    # tab:red
    "ID_5": (189, 103, 148),  # tab:purple
    "ID_6": (75, 86, 140),    # tab:brown
    "ID_7": (194, 119, 227),  # tab:pink
    "ID_8": (127, 127, 127),  # tab:gray
}
UNASSIGNED_BGR = (110, 110, 110)


def _id_color(fish_id: str | None) -> Tuple[int, int, int]:
    if not fish_id:
        return UNASSIGNED_BGR
    return ID_COLORS_BGR.get(fish_id, UNASSIGNED_BGR)


def _sorted_instance_points(labels_for_view) -> List[Tuple[int, list, float]]:
    """Return [(original_index, pts, score)] sorted by score-desc as load_2d_slp does."""
    # `labels_for_view` is a list of (numpy_pts, score, idx_in_file).
    ranked = sorted(
        labels_for_view,
        key=lambda entry: (
            -(entry[1] if entry[1] is not None and np.isfinite(entry[1]) else -np.inf),
            entry[0][0][0] if len(entry[0]) else np.inf,
        ),
    )
    return ranked


def _draw_frame_overlay(
    frame: np.ndarray,
    instances,
    cam_to_id: List[str | None],
    *,
    frame_idx: int,
    view: str,
    radius: int = 10,
) -> np.ndarray:
    """Draw keypoints coloured by ID on one camera frame."""
    out = frame
    ranked = _sorted_instance_points(instances)
    for slot_idx, (pts, score, _orig) in enumerate(ranked):
        fish_id = None
        if slot_idx < len(cam_to_id):
            fish_id = cam_to_id[slot_idx]
        color = _id_color(fish_id)
        for x, y in pts:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(out, (cx, cy), radius, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 2, color, thickness=-1, lineType=cv2.LINE_AA)
        if len(pts) > 0 and np.isfinite(pts[0]).all():
            anchor = pts[0]
            label = fish_id if fish_id else "?"
            if score is not None and np.isfinite(score):
                label = f"{label} {score:.2f}"
            cv2.putText(
                out,
                label,
                (int(anchor[0]) + radius + 2, int(anchor[1]) + radius + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        out, f"{view}  f={frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


def _make_projector(
    size: int,
    *,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    invert_x: bool = True,
    invert_y: bool = False,
    padding: float = 0.80,
):
    """
    Return a function `project(xyz[N,3]) -> uv[N,2]` for a diagonal 3D panel.

    Rotation convention mirrors matplotlib's 3D axis: azim rotates around
    world-Z, elev tilts the camera up/down. `invert_x` flips world X before
    projection — set to True so the 3D panel's left-right matches cam0
    (world X is oriented opposite to cam0 pixel-X in our calibration).
    """
    cx = 0.5 * (x_range[0] + x_range[1])
    cy = 0.5 * (y_range[0] + y_range[1])
    cz = 0.5 * (z_range[0] + z_range[1])
    diag = float(
        np.sqrt(
            (x_range[1] - x_range[0]) ** 2
            + (y_range[1] - y_range[0]) ** 2
            + (z_range[1] - z_range[0]) ** 2
        )
    )
    scale = padding * size / max(1e-6, diag)

    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]]
    )
    rot = rx @ rz

    def project(xyz: np.ndarray) -> np.ndarray:
        xyz = np.atleast_2d(xyz).astype(float)
        centered = xyz - np.array([cx, cy, cz])
        if invert_x:
            centered[:, 0] = -centered[:, 0]
        if invert_y:
            centered[:, 1] = -centered[:, 1]
        rotated = centered @ rot.T
        u = rotated[:, 0] * scale + size / 2
        # Image y grows downward; negate rotated z so higher world-Z draws higher.
        v = -rotated[:, 2] * scale + size / 2
        return np.stack([u, v], axis=1)

    return project


def _render_3d_panel(
    size: int,
    xyz_by_id: Dict[str, np.ndarray],
    *,
    trail_len: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    current_idx: int,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    invert_x: bool = True,
) -> np.ndarray:
    """
    Draw a diagonal-view 3D panel with trails, current markers, and a
    wireframe of the tank cube for reference. Black background.

    `xyz_by_id[id]` is an (N, 3) array of positions up to `current_idx`
    inclusive. NaN rows are tolerated.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    project = _make_projector(
        size,
        azim_deg=azim_deg,
        elev_deg=elev_deg,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        invert_x=invert_x,
    )

    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    corners = np.array(
        [
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    cpts = project(corners).round().astype(int)
    for a, b in edges:
        cv2.line(
            canvas, tuple(cpts[a]), tuple(cpts[b]), (60, 60, 60), 1, cv2.LINE_AA
        )

    # Axis labels + short reference arrows from the origin corner.
    origin_xyz = np.array([[xmin, ymin, zmin]])
    for tip_xyz, label, color in [
        (np.array([[xmin + 40, ymin, zmin]]), "X", (80, 80, 220)),
        (np.array([[xmin, ymin + 40, zmin]]), "Y", (80, 220, 80)),
        (np.array([[xmin, ymin, zmin + 40]]), "Z", (220, 80, 80)),
    ]:
        p0 = project(origin_xyz)[0].round().astype(int)
        p1 = project(tip_xyz)[0].round().astype(int)
        cv2.arrowedLine(canvas, tuple(p0), tuple(p1), color, 1, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(
            canvas, label, (int(p1[0]) + 4, int(p1[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    for fid, pts in xyz_by_id.items():
        color = _id_color(fid)
        start = max(0, current_idx - trail_len + 1)
        segment = pts[start : current_idx + 1]
        if len(segment) < 2:
            continue
        finite = np.isfinite(segment).all(axis=1)
        i = 0
        while i < len(segment):
            if not finite[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(segment) and finite[j + 1]:
                j += 1
            run = segment[i : j + 1]
            px_run = project(run).round().astype(np.int32)
            cv2.polylines(canvas, [px_run], False, color, 1, cv2.LINE_AA)
            i = j + 1

    for fid, pts in xyz_by_id.items():
        color = _id_color(fid)
        if current_idx >= len(pts):
            continue
        p = pts[current_idx]
        if not np.isfinite(p).all():
            continue
        proj = project(p.reshape(1, 3))[0].round().astype(int)
        px, py = int(proj[0]), int(proj[1])
        cv2.circle(canvas, (px, py), 7, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 3, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas, fid, (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    cv2.putText(
        canvas, "3D (diagonal view, mm)", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA,
    )
    return canvas


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def render(
    *,
    experiment: str,
    tag: str,
    sleap_output_dir: Path,
    medaka_data_dir: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    panel_size: int = 600,
    trail_seconds: float = 2.0,
    fps: float | None = None,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    invert_x: bool = True,
) -> None:
    slp_dir = sleap_output_dir / experiment
    json_path = slp_dir / f"{tag}_best.json"
    csv_path = slp_dir / f"{tag}_3d.csv"
    if not json_path.exists():
        raise FileNotFoundError(f"Best-tracked JSON not found: {json_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"3D CSV not found: {csv_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        best_json = json.load(fh)
    id_labels: List[str] = best_json["id_labels"]
    per_frame = {int(k): v for k, v in best_json["frames"].items()}

    df_3d = pd.read_csv(csv_path)
    df_3d = df_3d[(df_3d["frame"] >= start_frame) & (df_3d["frame"] <= end_frame)].copy()

    # Full per-fish XYZ array aligned to frame_idx in [start_frame, end_frame]
    frames = np.arange(start_frame, end_frame + 1)
    xyz_by_id: Dict[str, np.ndarray] = {}
    for fid in id_labels:
        sub = df_3d[df_3d["fish_id"] == fid].set_index("frame")
        sub = sub.reindex(frames)
        xyz_by_id[fid] = sub[["X", "Y", "Z"]].to_numpy()

    # Load per-view predictions and videos.
    slps = {view: slp_dir / f"{view}.slp" for view in VIEW_ORDER}
    videos = {
        view: medaka_data_dir / experiment / view / VIEW_TO_VIDEO[view]
        for view in VIEW_ORDER
    }
    for view in VIEW_ORDER:
        if not slps[view].exists():
            raise FileNotFoundError(f"Missing merged .slp for {view}: {slps[view]}")
        if not videos[view].exists():
            raise FileNotFoundError(f"Missing video for {view}: {videos[view]}")

    preds_by_view: Dict[str, dict] = {}
    for view in VIEW_ORDER:
        by_frame, _ = index_predictions_by_frame(slps[view])
        preds_by_view[view] = by_frame

    caps = {view: cv2.VideoCapture(str(videos[view])) for view in VIEW_ORDER}
    try:
        src_fps = caps[VIEW_ORDER[0]].get(cv2.CAP_PROP_FPS) or 166.0
        out_fps = fps if fps else src_fps
        trail_len = max(2, int(round(trail_seconds * src_fps)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, out_fps, (panel_size * 2, panel_size * 2)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer for {output_path}")

        for view in VIEW_ORDER:
            caps[view].set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        written = 0
        for fi_offset, fi in enumerate(frames):
            panels: Dict[str, np.ndarray] = {}
            for view in VIEW_ORDER:
                ok, frame = caps[view].read()
                if not ok:
                    frame = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                instances = preds_by_view[view].get(int(fi), [])
                mapping_info = per_frame.get(int(fi), {})
                cam_to_id = mapping_info.get(f"{VIEW_TO_CAM[view]}_to_id", [])
                annotated = _draw_frame_overlay(
                    frame, instances, cam_to_id, frame_idx=int(fi), view=view,
                    radius=max(6, panel_size // 120),
                )
                panels[view] = _resize_square(annotated, panel_size)

            three_d_panel = _render_3d_panel(
                panel_size,
                xyz_by_id,
                trail_len=trail_len,
                x_range=(-10.0, 210.0),
                y_range=(-10.0, 210.0),
                z_range=(-10.0, 210.0),
                current_idx=fi_offset,
                azim_deg=azim_deg,
                elev_deg=elev_deg,
                invert_x=invert_x,
            )

            top_row = np.hstack([panels["top_cam"], panels["side1_cam"]])
            bot_row = np.hstack([panels["side2_cam"], three_d_panel])
            composite = np.vstack([top_row, bot_row])
            writer.write(composite)
            written += 1

        writer.release()
        print(f"Wrote {written} frames -> {output_path}")
    finally:
        for cap in caps.values():
            cap.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a 2x2 multiview QC clip with 2D overlays and a diagonal 3D panel."
    )
    parser.add_argument(
        "experiment",
        help="Experiment folder name, e.g. 20251220_D_1_1_FCMC.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Reconstruction tag (produces <tag>_best.json and <tag>_3d.csv).",
    )
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument(
        "--output", required=True, help="Output mp4 path.",
    )
    parser.add_argument(
        "--panel-size", type=int, default=600, help="Size of each panel in px (default 600).",
    )
    parser.add_argument(
        "--trail-seconds", type=float, default=2.0,
        help="Length of the trail history in the 3D panel (seconds).",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output FPS (defaults to source FPS).",
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0,
        help="3D panel azimuth around world-Z in degrees (default -60).",
    )
    parser.add_argument(
        "--elev", type=float, default=25.0,
        help="3D panel elevation above horizontal in degrees (default 25).",
    )
    parser.add_argument(
        "--no-invert-x", action="store_true",
        help="Do not flip world X before projecting. "
             "Default flips X so the 3D panel's left-right matches cam0.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--sleap-output-dir", default=None)
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
    medaka_data_dir = choose_path(
        args.medaka_data_dir,
        local.medaka_data_dir if local else None,
    )
    if sleap_output_dir is None:
        parser.error("sleap_output_dir required (paths.local.toml or --sleap-output-dir).")
    if medaka_data_dir is None:
        parser.error("medaka_data_dir required (paths.local.toml or --medaka-data-dir).")

    render(
        experiment=args.experiment,
        tag=args.tag,
        sleap_output_dir=Path(sleap_output_dir),
        medaka_data_dir=Path(medaka_data_dir),
        output_path=Path(args.output),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        panel_size=args.panel_size,
        trail_seconds=args.trail_seconds,
        azim_deg=args.azim,
        elev_deg=args.elev,
        invert_x=not args.no_invert_x,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
