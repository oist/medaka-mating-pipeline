"""
Project 3D trajectories back into 2D video frames and render an overlay video.

This helper does not display IDs. It projects 3D points into the three camera
views using the calibration model, draws simple points on the videos, and then
reuses the three-view overlay writer to export an MP4.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_tracking_video import render_three_view_video


def find_videos_by_set_key(
    video_dir: Path,
    set_key: str,
    video_top: Optional[Path] = None,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
) -> Dict[str, Path]:
    """Find the three view videos for one set key using the time range and PLAYER labels."""
    out: Dict[str, Path] = {}
    if video_top and video_top.exists():
        out["top"] = Path(video_top)
    if video_side1 and video_side1.exists():
        out["side1"] = Path(video_side1)
    if video_side2 and video_side2.exists():
        out["side2"] = Path(video_side2)
    if len(out) == 3:
        return out

    m = re.search(r"(\d+\.\d+-\d+\.\d+)", set_key)
    time_part = m.group(1) if m else ""
    if not video_dir.exists() or not time_part:
        return out
    all_mp4 = list(video_dir.rglob("*.mp4"))

    def by_view(name_upper: str) -> Optional[str]:
        if "PLAYER1" in name_upper:
            return "top"
        if "PLAYER2" in name_upper:
            return "side1"
        if "PLAYER3" in name_upper:
            return "side2"
        return None

    for p in all_mp4:
        if time_part not in p.stem:
            continue
        v = by_view(p.stem.upper())
        if v and v not in out:
            out[v] = p.resolve()
    return out


def project_3d_to_2d_views(
    df_3d: pd.DataFrame,
    cal: Calibration,
) -> Dict[str, pd.DataFrame]:
    """
    Project a 3D trajectory CSV (`frame, fish_id, X, Y, Z`) into the three
    camera views.

    The output intentionally drops identity text and returns only
    `frame, individual, x, y` per view for point overlays.
    """
    frames = sorted(df_3d["frame"].unique())
    view_order = ["top", "side1", "side2"]
    view_rows: Dict[str, List[Dict]] = {v: [] for v in view_order}

    for f in frames:
        sub = df_3d[df_3d["frame"] == f]
        if sub.empty:
            continue
        if "fish_id" in sub.columns:
            sub = sub.sort_values("fish_id").head(3)
        else:
            sub = sub.head(3)
        xyz = sub[["X", "Y", "Z"]].values.astype(float)
        valid = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[valid]
        if xyz.size == 0:
            continue
        uv_all = cal.compute_imCoords_from_XYZ(xyz)
        n_pt = uv_all.shape[0]
        for i in range(n_pt):
            u, v = float(uv_all[i, 0, 0]), float(uv_all[i, 0, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["top"].append({"frame": f, "individual": i, "x": u, "y": v})
            u, v = float(uv_all[i, 1, 0]), float(uv_all[i, 1, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["side1"].append({"frame": f, "individual": i, "x": u, "y": v})
            u, v = float(uv_all[i, 2, 0]), float(uv_all[i, 2, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["side2"].append({"frame": f, "individual": i, "x": u, "y": v})

    view_data: Dict[str, pd.DataFrame] = {}
    for v in view_order:
        if view_rows[v]:
            view_data[v] = pd.DataFrame(view_rows[v])
        else:
            view_data[v] = pd.DataFrame(columns=["frame", "individual", "x", "y"])
    return view_data


def run_render_3d_projection_video(
    csv_3d_path: str | Path,
    model_folder: str | Path,
    video_dir: str | Path,
    set_key: str,
    output_path: str | Path,
    fps: float,
    *,
    video_top: Optional[Path] = None,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
    trail_duration_s: float = 0.5,
    show_trail_row: bool = False,
) -> Path:
    """Project 3D trajectories into all views and export a point-overlay video."""
    csv_3d_path = Path(csv_3d_path)
    model_folder = Path(model_folder)
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    if not csv_3d_path.exists():
        raise FileNotFoundError(f"3D CSV does not exist: {csv_3d_path}")
    if not model_folder.is_dir():
        raise FileNotFoundError(f"Calibration directory does not exist: {model_folder}")

    df_3d = pd.read_csv(csv_3d_path)
    for c in ["frame", "X", "Y", "Z"]:
        if c not in df_3d.columns:
            raise ValueError(f"3D CSV must contain column {c}; current columns: {list(df_3d.columns)}")
    cal = Calibration(model_folder)
    view_data = project_3d_to_2d_views(df_3d, cal)

    # The CSV may cover only a slice of a long recording (e.g. frames
    # 171800-176799); bound the writer to that window so it doesn't render every
    # untouched frame of the full-length source video.
    f_min = int(df_3d["frame"].min())
    f_max = int(df_3d["frame"].max())

    video_paths = find_videos_by_set_key(
        video_dir, set_key, video_top, video_side1, video_side2
    )
    missing = [v for v in ["top", "side1", "side2"] if v not in video_paths]
    if missing:
        raise FileNotFoundError(
            f"Missing view video(s): {missing}. Place PLAYER1/2/3 mp4 files under {video_dir} "
            f"or pass video_top/video_side1/video_side2 explicitly."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_three_view_video(
        view_data,
        video_paths,
        output_path,
        fps,
        trail_duration_s=trail_duration_s,
        show_trail_row=show_trail_row,
        simple_dots=True,
        start_frame=f_min,
        end_frame=f_max,
    )
    return output_path


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Project 3D trajectories into 2D videos and generate an overlay video without ID labels."
    )
    p.add_argument("--csv-3d", type=Path, required=True, help="3D trajectory CSV (frame, fish_id, X, Y, Z)")
    p.add_argument("--model-folder", type=Path, required=True, help="Calibration model directory containing XYZ_to_imCoords.joblib")
    p.add_argument("--video-dir", type=Path, required=True, help="Video root directory")
    p.add_argument("--set-key", type=str, required=True, help="Clip set key used to match the source videos")
    p.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    p.add_argument("--fps", type=float, default=166.0, help="Video frame rate")
    p.add_argument("--video-top", type=Path, default=None)
    p.add_argument("--video-side1", type=Path, default=None)
    p.add_argument("--video-side2", type=Path, default=None)
    p.add_argument("--trail-duration-s", type=float, default=0.5)
    p.add_argument("--no-trail-row", action="store_true", help="Hide the lower trail row")
    args = p.parse_args()

    out = args.output
    if out is None:
        out = args.csv_3d.parent / (args.csv_3d.stem.replace("_3d", "") + "_3d_projection_overlay.mp4")
    run_render_3d_projection_video(
        args.csv_3d,
        args.model_folder,
        args.video_dir,
        args.set_key,
        out,
        args.fps,
        video_top=args.video_top,
        video_side1=args.video_side1,
        video_side2=args.video_side2,
        trail_duration_s=args.trail_duration_s,
        show_trail_row=not args.no_trail_row,
    )
    print("Wrote:", out)


if __name__ == "__main__":
    main()
