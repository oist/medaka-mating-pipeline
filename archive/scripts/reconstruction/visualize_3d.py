"""
Visualize already-generated 3D trajectories after identity tracking.

This script creates trajectory PNG or MP4 outputs from existing 3D CSV files.
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.calibration.trajectory_3d import plot_3d_trajectories, plot_3d_trajectories_video


PREFERRED_GROUP = ("ID_1", "ID_2", "ID_3")
FALLBACK_GROUP = ("a", "b", "c")


def load_trajectories_from_csv(output_dir: str, set_key: str) -> Dict[str, pd.DataFrame]:
    """Load 3D trajectories from either the long CSV or legacy per-group CSV files."""
    out = Path(output_dir)
    single = out / f"{set_key}_3d.csv"
    if single.exists():
        df = pd.read_csv(single)
        if "fish_id" not in df.columns:
            raise ValueError(f"{single} is missing the fish_id column")
        return {fid: gr[["frame", "X", "Y", "Z"]].copy() for fid, gr in df.groupby("fish_id")}

    for name in PREFERRED_GROUP:
        p = out / f"{set_key}_{name}_3d.csv"
        if p.exists():
            return {
                n: pd.read_csv(out / f"{set_key}_{n}_3d.csv")
                for n in PREFERRED_GROUP
                if (out / f"{set_key}_{n}_3d.csv").exists()
            }
    result = {}
    for name in FALLBACK_GROUP:
        p = out / f"{set_key}_{name}_3d.csv"
        if p.exists():
            result[name] = pd.read_csv(p)
    if not result:
        raise FileNotFoundError(f"3D CSV not found: {out}/{set_key}_3d.csv or {set_key}_*_3d.csv")
    return result


def run_video(
    output_dir: str,
    set_key: str,
    fps: float = 30,
    subsample: int = 5,
    trail_seconds: float = 5.0,
    data_fps: float | None = None,
    xyz_lim: tuple[float, float] | None = (0, 200),
) -> str:
    """Generate an MP4 trajectory video from existing 3D CSV files."""
    trajectories = load_trajectories_from_csv(output_dir, set_key)
    out_path = Path(output_dir) / f"{set_key}_3d_trajectories.mp4"
    plot_3d_trajectories_video(
        trajectories, str(out_path),
        fps=fps, subsample=subsample, trail_seconds=trail_seconds,
        data_fps=data_fps, xyz_lim=xyz_lim,
    )
    return str(out_path)


def run_plot(
    output_dir: str,
    set_key: str,
    xyz_lim: tuple[float, float] | None = (0, 200),
) -> str:
    """Generate a PNG trajectory plot from existing 3D CSV files."""
    trajectories = load_trajectories_from_csv(output_dir, set_key)
    out_path = Path(output_dir) / f"{set_key}_3d_trajectories.png"
    plot_3d_trajectories(trajectories, str(out_path), xyz_lim=xyz_lim)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="Render 3D trajectory plots or MP4 videos from CSV")
    ap.add_argument("--output_dir", required=True, help="Directory containing the 3D CSV files")
    ap.add_argument("--set_key", required=True, help="Set key")
    ap.add_argument("--fps", type=float, default=30, help="Output video frame rate")
    ap.add_argument("--subsample", type=int, default=5, help="Frame subsampling interval")
    ap.add_argument("--trail_seconds", type=float, default=5.0, help="How many recent seconds of trajectory to keep visible")
    ap.add_argument("--format", choices=["mp4", "png"], default="mp4", help="Output format")
    args = ap.parse_args()

    if args.format == "mp4":
        out = run_video(
            args.output_dir, args.set_key, args.fps, args.subsample,
            trail_seconds=args.trail_seconds,
        )
    else:
        out = run_plot(args.output_dir, args.set_key)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
