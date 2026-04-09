"""
3D reconstruction without a calibration model.

Assumptions:

- top view -> (X, Y)
- side1 -> (X, Z)
- side2 -> (Y, Z)

Missing dimensions remain `NaN`.
"""

from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scripts.reconstruction_3d.load_2d import (
    NUM_INDIVIDUALS,
    get_points_per_frame,
    load_processed_dfs,
    run_load_2d,
)
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.cross_frame_tracking import apply_first_frame_id_only
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.visualize_3d import run_plot

LABELS = ["a", "b", "c"]


def _image_to_xyz_one(
    pt_top: Tuple[float, float],
    pt_side1: Tuple[float, float],
    pt_side2: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    """
    Convert one top/side1/side2 triplet into an approximate 3D point.

    The cost is the sum of squared disagreements between shared coordinates.
    """
    x0, y0 = float(pt_top[0]), float(pt_top[1])
    x1, y1 = float(pt_side1[0]), float(pt_side1[1])
    x2, y2 = float(pt_side2[0]), float(pt_side2[1])

    X = np.nanmean(np.array([x0, x1])[np.isfinite([x0, x1])]) if np.any(np.isfinite([x0, x1])) else np.nan
    Y = np.nanmean(np.array([y0, x2])[np.isfinite([y0, x2])]) if np.any(np.isfinite([y0, x2])) else np.nan
    Z = np.nanmean(np.array([y1, y2])[np.isfinite([y1, y2])]) if np.any(np.isfinite([y1, y2])) else np.nan

    cost = 0.0
    if np.isfinite(x0) and np.isfinite(x1):
        cost += (x0 - x1) ** 2
    if np.isfinite(y0) and np.isfinite(x2):
        cost += (y0 - x2) ** 2
    if np.isfinite(y1) and np.isfinite(y2):
        cost += (y1 - y2) ** 2
    if not np.any(np.isfinite([X, Y, Z])):
        cost = np.inf

    return np.array([X, Y, Z], dtype=float), float(cost)


def compute_candidates_no_calib(
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    *,
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict[int, List[Dict]]:
    """
    Enumerate permutations per frame and compute 3D candidates without a
    calibration model.
    """
    candidates_by_frame: Dict[int, List[Dict]] = {}
    low = high = None
    if xyz_bounds is not None:
        low, high = xyz_bounds[0], xyz_bounds[1]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    for frame in tqdm(sorted(points_by_frame.keys()), desc="3D (no calibration)", unit="frame"):
        cams = points_by_frame[frame]
        pts0 = list(cams.get("cam0", []))
        pts1 = list(cams.get("cam1", []))
        pts2 = list(cams.get("cam2", []))
        while len(pts0) < NUM_INDIVIDUALS:
            pts0.append((np.nan, np.nan))
        while len(pts1) < NUM_INDIVIDUALS:
            pts1.append((np.nan, np.nan))
        while len(pts2) < NUM_INDIVIDUALS:
            pts2.append((np.nan, np.nan))
        pts0, pts1, pts2 = pts0[:NUM_INDIVIDUALS], pts1[:NUM_INDIVIDUALS], pts2[:NUM_INDIVIDUALS]

        perms1 = list(permutations(range(NUM_INDIVIDUALS)))
        perms2 = list(permutations(range(NUM_INDIVIDUALS)))
        frame_candidates = []

        for perm1 in perms1:
            for perm2 in perms2:
                groups = {}
                total_cost = 0.0
                for i, label in enumerate(LABELS):
                    j1, j2 = perm1[i], perm2[i]
                    xyz, c = _image_to_xyz_one(pts0[i], pts1[j1], pts2[j2])
                    if np.any(np.isfinite(xyz)):
                        if low is not None and high is not None:
                            out = (xyz < low) | (xyz > high)
                            if np.any(np.isfinite(xyz) & out):
                                xyz = np.array([np.nan, np.nan, np.nan])
                                c = np.inf
                        groups[label] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                        total_cost += c
                    else:
                        groups[label] = [np.nan, np.nan, np.nan]
                        total_cost += np.inf

                frame_candidates.append({
                    "groups": groups,
                    "perm_cam1": list(perm1),
                    "perm_cam2": list(perm2),
                    "cost": float(total_cost),
                })
        candidates_by_frame[frame] = frame_candidates

    return candidates_by_frame


def run_pipeline_no_calibration(
    processed_dir: str,
    set_key: str,
    *,
    output_dir: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
    fill_missing_3d: bool = True,
    plot: bool = True,
    first_frame_id_only: bool = True,
) -> Dict:
    """
    Full no-calibration 3D pipeline: load 2D points, map them into pseudo-3D,
    choose the best permutation, assign IDs from the first valid frame, fill
    missing frames, and export the CSV/plot.
    """
    points_by_frame = run_load_2d(processed_dir, set_key, use_fish_id=False)
    candidates_by_frame = compute_candidates_no_calib(
        points_by_frame, xyz_bounds=xyz_bounds
    )
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)
    best_tracked = apply_first_frame_id_only(best)

    out = Path(output_dir or processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = None
    png_path = None

    dfs_3d = best_by_frame_to_dfs(
        best_tracked,
        shift_to_positive=True,
        ensure_non_negative=False,
    )
    if dfs_3d:
        if fill_missing_3d and points_by_frame:
            all_frames = sorted(points_by_frame.keys())
            dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
        csv_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out))
        if plot:
            # In no-calibration mode, coordinates may remain at pixel scale.
            png_path = run_plot(str(out), set_key, xyz_lim=xyz_bounds)

    return {
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "csv_path": csv_path,
        "png_path": png_path,
    }
