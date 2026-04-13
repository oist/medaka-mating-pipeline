"""
Multi-view 3D multi-target tracking with identities bound to 3D tracks.

The goal is stable IDs across frames and, by extension, across clips if the
caller applies additional cross-video logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.reconstruction_3d.load_2d import run_load_2d
from scripts.reconstruction_3d.reconstruct_no_calibration import compute_candidates_no_calib
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.visualize_3d import run_plot

TRACK_IDS = ["ID_1", "ID_2", "ID_3"]
_SRC = ["a", "b", "c"]


def _groups_to_points(groups: Dict) -> np.ndarray:
    """Convert groups (`a/b/c` or `ID_1/2/3`) into a `(3, 3)` array."""
    out = np.full((3, 3), np.nan)
    labels = TRACK_IDS if "ID_1" in groups else _SRC
    for i, lb in enumerate(labels):
        if lb in groups and groups[lb] is not None:
            pt = groups[lb]
            if len(pt) == 3 and np.any(np.isfinite(pt)):
                out[i, :] = pt
    return out


def _cost_matrix(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    prev_vel: Optional[np.ndarray] = None,
    motion_weight: float = 0.0,
) -> np.ndarray:
    """
    Build the geometry + motion-consistency cost matrix:
        C[i, j] = d_ij + lambda * m_ij
    """
    C = np.full((3, 3), np.inf)
    for i in range(3):
        if np.any(np.isnan(prev_pts[i])) or not np.any(np.isfinite(prev_pts[i])):
            continue
        for j in range(3):
            if np.any(np.isnan(curr_pts[j])) or not np.any(np.isfinite(curr_pts[j])):
                continue
            d = float(np.linalg.norm(prev_pts[i] - curr_pts[j]))
            m = 0.0
            if motion_weight > 0 and prev_vel is not None and np.any(np.isfinite(prev_vel[i])):
                delta = curr_pts[j] - prev_pts[i]
                v = prev_vel[i]
                nv = np.linalg.norm(v)
                nd = np.linalg.norm(delta)
                if nv > 1e-6 and nd > 1e-6:
                    cos_a = np.dot(v, delta) / (nv * nd)
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    m = 1.0 - cos_a
            C[i, j] = d + motion_weight * m
    return C


def assign_3d_tracks(
    best_by_frame: Dict[int, Dict],
    *,
    motion_weight: float = 10.0,
) -> Dict[int, Dict]:
    """
    Bind stable IDs to 3D tracks.

    - first frame: assign `ID_1/ID_2/ID_3` by x-order
    - later frames: match previous tracks to current points in 3D using
      Hungarian assignment
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}
    prev_pts = np.full((3, 3), np.nan)
    prev_pts_2 = np.full((3, 3), np.nan)

    for idx, f in enumerate(frames):
        info = best_by_frame[f]
        groups = info.get("groups", {})
        curr_pts = _groups_to_points(groups)

        if idx == 0:
            valid = [i for i in range(3) if np.any(np.isfinite(curr_pts[i]))]
            valid_sorted = sorted(valid, key=lambda i: curr_pts[i, 0])
            new_groups = {}
            for i, tid in enumerate(TRACK_IDS):
                if i < len(valid_sorted):
                    new_groups[tid] = list(curr_pts[valid_sorted[i]])
                else:
                    new_groups[tid] = [np.nan, np.nan, np.nan]
            result[f] = {
                "groups": new_groups,
                "perm_cam1": list(info.get("perm_cam1", [])),
                "perm_cam2": list(info.get("perm_cam2", [])),
            }
            prev_pts = _groups_to_points(new_groups)
            prev_pts_2 = prev_pts.copy()
            continue

        dt = float(f - frames[idx - 1])
        prev_vel = (prev_pts - prev_pts_2) / dt if dt > 0 and idx >= 2 else None

        C = _cost_matrix(prev_pts, curr_pts, prev_vel=prev_vel, motion_weight=motion_weight)
        n_track, n_pt = C.shape
        n_pt_orig = n_pt
        if n_pt < n_track:
            C = np.hstack([C, np.full((n_track, n_track - n_pt), np.inf)])
        feasible = np.isfinite(C) & (C < 1e9)
        if not np.any(feasible):
            new_groups = {tid: [np.nan, np.nan, np.nan] for tid in TRACK_IDS}
        else:
            C_safe = np.where(np.isfinite(C), C, 1e10)
            row_ind, col_ind = linear_sum_assignment(C_safe)
            new_groups = {}
            for i, tid in enumerate(TRACK_IDS):
                j = int(col_ind[np.where(row_ind == i)[0][0]])
                if j < n_pt_orig and np.isfinite(C[i, j]) and C[i, j] < 1e9:
                    new_groups[tid] = list(curr_pts[j])
                else:
                    new_groups[tid] = [np.nan, np.nan, np.nan]
        result[f] = {
            "groups": new_groups,
            "perm_cam1": list(info.get("perm_cam1", [])),
            "perm_cam2": list(info.get("perm_cam2", [])),
        }
        prev_pts_2 = prev_pts.copy()
        prev_pts = _groups_to_points(new_groups)

    return result


def run_full_pipeline(
    processed_dir: str,
    set_key: str,
    *,
    model_folder: Optional[str] = None,
    output_dir: Optional[str] = None,
    xyz_bounds: Optional[Tuple[float, float]] = None,
    use_fish_id: bool | None = None,
    allow_permutations: bool | None = None,
    fill_missing_3d: bool = True,
    motion_weight: float = 10.0,
    plot: bool = True,
) -> Dict:
    """
    Full pipeline:
    2D -> 3D reconstruction -> 3D tracking -> optional interpolation -> export.
    """
    points_by_frame = run_load_2d(processed_dir, set_key, use_fish_id=use_fish_id)
    if model_folder:
        cal = Calibration(model_folder)
        if allow_permutations is None:
            allow_permutations = bool(cal.has_geometric_scoring)
        candidates_by_frame = compute_candidates(
            cal,
            points_by_frame,
            xyz_bounds=xyz_bounds,
            allow_permutations=allow_permutations,
        )
    else:
        if allow_permutations is None:
            allow_permutations = False
        candidates_by_frame = compute_candidates_no_calib(
            points_by_frame,
            xyz_bounds=xyz_bounds,
            allow_permutations=allow_permutations,
        )
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)
    best_tracked = assign_3d_tracks(best, motion_weight=motion_weight)

    out = Path(output_dir or processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = None
    png_path = None

    dfs_3d = best_by_frame_to_dfs(best_tracked, shift_to_positive=True, ensure_non_negative=False)
    if dfs_3d:
        if fill_missing_3d and points_by_frame:
            all_frames = sorted(points_by_frame.keys())
            dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
        csv_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out))
        if plot:
            png_path = run_plot(str(out), set_key, xyz_lim=xyz_bounds)

    return {
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "csv_path": csv_path,
        "png_path": png_path,
    }


FAILURE_ANALYSIS = r"""
1) Reconstruction failure (a frame has no valid 3D point)
   - Cause: too many missing 2D views, all candidate permutations fall outside
     `xyz_bounds`, or the geometry is inconsistent.
   - Symptom: `groups` becomes all-NaN or partly NaN for that frame.
   - Handling: keep NaNs, continue tracking, then interpolate later if possible.

2) Association ambiguity (two targets are very close)
   - Cause: two current 3D points produce nearly identical costs relative to
     the previous tracks.
   - Symptom: short ID flips or crossing trajectories.
   - Handling: increase `motion_weight` or add smoothing / post-processing.

3) Occlusion / short disappearance
   - Cause: one fish is missing in all views or reconstruction fails in that frame.
   - Symptom: that track becomes NaN in the frame.
   - Handling: preserve the ID, leave NaN for the frame, interpolate later.

4) Number of targets is not 3
   - This implementation assumes exactly 3 tracks. If a frame has fewer points,
     the remaining tracks stay NaN.

5) Cross-video identity consistency
   - One run only guarantees stable IDs within a single clip. Cross-video
     consistency still requires an external rule.
"""


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Multi-view 3D multi-target tracking with stable IDs bound to 3D tracks."
    )
    p.add_argument("--processed_dir", required=True, help="Directory containing the *_processed.csv files")
    p.add_argument("--set_key", required=True, help="Clip set key")
    p.add_argument("--model_folder", default=None, help="Calibration model directory; if omitted use no-calibration mode")
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional valid 3D range [LOW, HIGH]. Omit to keep the model output scale unchanged.",
    )
    fish_id_group = p.add_mutually_exclusive_group()
    fish_id_group.add_argument(
        "--use_fish_id",
        dest="use_fish_id",
        action="store_true",
        help="Force a/b/c ordering from processed CSV fish_id values when available",
    )
    fish_id_group.add_argument(
        "--ignore_fish_id",
        dest="use_fish_id",
        action="store_false",
        help="Ignore fish_id and order 2D points by score/x instead",
    )
    p.set_defaults(use_fish_id=None)
    p.add_argument("--motion_weight", type=float, default=10.0, help="Motion-consistency cost weight")
    p.add_argument("--no_fill", action="store_true", help="Do not interpolate missing 3D frames")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()
    xyz_bounds = (float(args.xyz_bounds[0]), float(args.xyz_bounds[1])) if args.xyz_bounds else None
    res = run_full_pipeline(
        processed_dir=args.processed_dir,
        set_key=args.set_key,
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        xyz_bounds=xyz_bounds,
        use_fish_id=args.use_fish_id,
        fill_missing_3d=not args.no_fill,
        motion_weight=args.motion_weight,
        plot=not args.no_plot,
    )
    if res.get("csv_path"):
        print("3D CSV:", res["csv_path"])
    if res.get("png_path"):
        print("Trajectory plot:", res["png_path"])


if __name__ == "__main__":
    main()
