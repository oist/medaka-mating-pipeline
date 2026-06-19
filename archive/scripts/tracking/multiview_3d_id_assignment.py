"""
Geometry-based multi-view 3D ID assignment wrapper.

This is a lightweight wrapper around the legacy reconstruction steps:

1. load 2D points from processed CSV files
2. compute 3D candidates and registration costs
3. choose the best a/b/c correspondence
4. optionally stabilize labels across frames in 3D

Typical use case:
- you already produced `*_processed.csv` files with `scripts.tracking.process_tracking`
- you want a geometry-driven a/b/c assignment that is cleaner than a
  single-view 2D identity stream
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from scripts.reconstruction.step1_get_points import (
    load_processed_dfs,
    get_points_per_frame,
    fill_missing_2d_from_two_views,
)
from scripts.reconstruction.step2_compute_3d_candidates import compute_candidates
from scripts.reconstruction.step3_select_best_correspondence import select_best_all_frames
from scripts.reconstruction.step4_cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction.export_3d_groups import best_by_frame_to_dfs, dfs_to_single_csv
from scripts.reconstruction.fill_missing_3d import ensure_all_frames_and_fill_spline
from scripts.reconstruction.interpolate_3d_trajectories import interpolate_and_save
from scripts.reconstruction.visualize_3d import run_plot

VIEW_TO_CAM = {"top": "cam0", "side1": "cam1", "side2": "cam2"}
LABELS = ["a", "b", "c"]


def _build_best_2d_per_frame(
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    best: Dict[int, Dict],
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    Convert `points_by_frame` plus the selected `best` solution into per-frame
    per-view 2D coordinates for the final a/b/c identities.
    """
    out: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for frame, info in best.items():
        groups = info.get("groups")
        perm_cam1 = info.get("perm_cam1")
        perm_cam2 = info.get("perm_cam2")
        if not groups or perm_cam1 is None or perm_cam2 is None:
            continue
        pts = points_by_frame.get(frame)
        if not pts:
            continue
        frame_2d: Dict[str, Dict[str, List[float]]] = {}
        for view, cam in VIEW_TO_CAM.items():
            idx_list = list(range(len(groups))) if cam == "cam0" else (perm_cam1 if cam == "cam1" else perm_cam2)
            arr = pts[cam]
            frame_2d[view] = {}
            for i, label in enumerate(LABELS):
                if i >= len(idx_list):
                    continue
                idx = idx_list[i]
                if idx >= len(arr) or label not in groups:
                    continue
                x, y = arr[idx]
                frame_2d[view][label] = [float(x), float(y)]
        out[frame] = frame_2d
    return out


def run_multiview_id_pipeline(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    output_dir: str | None = None,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
    use_fish_id: bool = False,
    save_best_json: bool = True,
    shift_to_positive: bool = False,
    interpolate_3d: bool = True,
    fill_missing_2d: bool = True,
    cross_frame_tracking: bool = True,
    flow_weight: float = 10.0,
    fill_missing_3d: bool = True,
) -> Dict:
    """
    Run Steps 1-4 in one call and optionally export the 3D results.
    """
    processed_dir_path = Path(processed_dir)
    if not processed_dir_path.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir_path}")

    from scripts.reconstruction.calibration import Calibration

    cal = Calibration(model_folder)

    dfs = load_processed_dfs(str(processed_dir_path), set_key)
    if fill_missing_2d and "fish_id" in next(iter(dfs.values())).columns:
        dfs = fill_missing_2d_from_two_views(cal, dfs)
    points_by_frame = get_points_per_frame(dfs, use_fish_id=use_fish_id)

    candidates_by_frame = compute_candidates(cal, points_by_frame, xyz_bounds=xyz_bounds)
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)

    if cross_frame_tracking:
        best_for_export = apply_cross_frame_tracking(best, flow_weight=flow_weight)
    else:
        best_for_export = best

    best_2d_per_frame = _build_best_2d_per_frame(points_by_frame, best_for_export)

    csv_paths: Dict[str, str] = {}
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_best_json:
            import json
            best_path = out_dir / f"{set_key}_best.json"
            with best_path.open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_for_export.items()}, f, indent=2, ensure_ascii=False)

        dfs_3d = best_by_frame_to_dfs(best_for_export, shift_to_positive=shift_to_positive)
        all_frames = sorted(points_by_frame.keys())
        if fill_missing_3d and dfs_3d and all_frames:
            dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
        if dfs_3d:
            single_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out_dir))
            csv_paths["3d"] = single_path

        if interpolate_3d and dfs_3d:
            interpolate_and_save(
                dfs_3d,
                set_key=set_key,
                output_dir=str(out_dir),
                suffix="_3d_interp",
            )

        if best_2d_per_frame:
            import json
            best_2d_path = out_dir / f"{set_key}_best_2d.json"
            with best_2d_path.open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_2d_per_frame.items()}, f, indent=2, ensure_ascii=False)

    return {
        "points_by_frame": points_by_frame,
        "candidates_by_frame": candidates_by_frame,
        "best": best,
        "best_tracked": best_for_export,
        "best_2d_per_frame": best_2d_per_frame,
        "csv_paths": csv_paths,
    }


def quick_plot_3d_trajectories(
    output_dir: str,
    set_key: str,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
) -> str:
    """Generate a simple 3D trajectory PNG from the exported 3D CSV."""
    return run_plot(output_dir, set_key, xyz_lim=xyz_bounds)


def main():
    """CLI entry point for one-shot three-view 3D ID assignment."""
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Geometry-based multi-view 3D ID assignment from processed CSV files. "
            "Runs the legacy reconstruction steps and exports a/b/c 3D trajectories."
        )
    )
    ap.add_argument("--processed_dir", required=True, help="Directory containing the *_processed.csv files")
    ap.add_argument("--set_key", required=True, help="Clip set key, consistent with the tracking stage")
    ap.add_argument("--model_folder", required=True, help="Calibration model directory containing imCoords_to_XYZ.joblib")
    ap.add_argument("--output_dir", required=True, help="3D result output directory")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, 200.0),
        help="Valid 3D range [LOW, HIGH]; candidates outside this range are rejected",
    )
    ap.add_argument(
        "--use_fish_id",
        action="store_true",
        help="Use 2D fish_id ordering a/b/c during Step 1 instead of score/x sorting",
    )
    ap.add_argument(
        "--no_save_best_json",
        action="store_true",
        help="Do not save {set_key}_best.json",
    )
    ap.add_argument(
        "--shift_to_positive",
        action="store_true",
        help="Shift all 3D coordinates into the non-negative range",
    )
    ap.add_argument(
        "--no_cross_frame_tracking",
        action="store_true",
        help="Disable Step 4 cross-frame tracking; labels may change frame to frame",
    )
    ap.add_argument(
        "--flow_weight",
        type=float,
        default=10.0,
        help="Motion-direction penalty weight used during cross-frame tracking",
    )
    ap.add_argument(
        "--no_fill_missing_3d",
        action="store_true",
        help="Do not fill missing 3D frames before export",
    )
    args = ap.parse_args()

    bounds: Tuple[float, float] | None = None
    if args.xyz_bounds is not None:
        bounds = (float(args.xyz_bounds[0]), float(args.xyz_bounds[1]))

    result = run_multiview_id_pipeline(
        processed_dir=args.processed_dir,
        set_key=args.set_key,
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        xyz_bounds=bounds,
        use_fish_id=bool(args.use_fish_id),
        save_best_json=not args.no_save_best_json,
        shift_to_positive=bool(args.shift_to_positive),
        cross_frame_tracking=not args.no_cross_frame_tracking,
        flow_weight=float(args.flow_weight),
        fill_missing_3d=not args.no_fill_missing_3d,
    )

    n_frames = len(result["best"])
    print(f"Completed three-view 3D ID assignment. {n_frames} frame(s) contain valid a/b/c results.")


if __name__ == "__main__":
    main()
