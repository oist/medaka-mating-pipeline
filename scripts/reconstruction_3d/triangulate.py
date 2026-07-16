"""
Build 3D candidates from the three-view 2D points.

For each frame, the pipeline enumerates permutations across the three views and
produces multiple candidate groups of three 3D points.

- Exactly one missing camera is delegated to the backend's two-view policy;
  fewer than two finite views produce `NaN` 3D points.
- Invalid or out-of-bounds 3D predictions are rejected as `NaN`.
- Candidate selection happens later in `select_best.py`.
"""

from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d import NUM_INDIVIDUALS

CAM_ORDER = ["cam0", "cam1", "cam2"]
LABELS = ["a", "b", "c"]


def _default_labels(num_individuals: int) -> List[str]:
    """Generate per-candidate labels ('a', 'b', ...) for any N."""
    return [chr(ord("a") + i) for i in range(num_individuals)]


def compute_candidates(
    cal: Calibration,
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    *,
    scoring_cal: Calibration | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
    allow_permutations: bool = True,
    num_individuals: int | None = None,
) -> Dict[int, List[Dict]]:
    """
    Enumerate permutations of cam1/cam2 assignments and compute candidate 3D
    groups with registration cost for each frame.

    When `allow_permutations=False`, the function keeps the incoming camera
    ordering and emits a single candidate per frame.
    """
    scoring_cal = scoring_cal or cal
    low, high = (xyz_bounds or (None, None))[0], (xyz_bounds or (None, None))[1]
    candidates_by_frame: Dict[int, List[Dict]] = {}

    n = num_individuals if num_individuals is not None else NUM_INDIVIDUALS
    labels = _default_labels(n)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    for frame in tqdm(sorted(points_by_frame.keys()), desc="3D candidates", unit="frame"):
        cams = points_by_frame[frame]
        pts0 = list(cams.get("cam0", []))
        pts1 = list(cams.get("cam1", []))
        pts2 = list(cams.get("cam2", []))
        while len(pts0) < n:
            pts0.append((np.nan, np.nan))
        while len(pts1) < n:
            pts1.append((np.nan, np.nan))
        while len(pts2) < n:
            pts2.append((np.nan, np.nan))
        pts0, pts1, pts2 = pts0[:n], pts1[:n], pts2[:n]

        if allow_permutations:
            perms1 = list(permutations(range(n)))
            perms2 = list(permutations(range(n)))
        else:
            identity_perm = tuple(range(n))
            perms1 = [identity_perm]
            perms2 = [identity_perm]
        permutation_pairs = [
            (perm1, perm2) for perm1 in perms1 for perm2 in perms2
        ]
        observations = np.empty((len(permutation_pairs), n, 3, 2), dtype=float)
        pts0_array = np.asarray(pts0, dtype=float)
        pts1_array = np.asarray(pts1, dtype=float)
        pts2_array = np.asarray(pts2, dtype=float)
        observations[:, :, 0, :] = pts0_array
        for candidate_index, (perm1, perm2) in enumerate(permutation_pairs):
            observations[candidate_index, :, 1, :] = pts1_array[list(perm1)]
            observations[candidate_index, :, 2, :] = pts2_array[list(perm2)]

        flat_observations = observations.reshape(-1, 3, 2)
        if scoring_cal is cal and hasattr(cal, "evaluate_hypotheses"):
            flat_xyz, flat_errors = cal.evaluate_hypotheses(
                flat_observations, profile="legacy_combined"
            )
        else:
            flat_xyz = cal.triangulate(flat_observations)
            flat_errors = scoring_cal.score_hypotheses(
                flat_observations, profile="legacy_combined"
            )
        xyz_all = np.asarray(flat_xyz, dtype=float).reshape(len(permutation_pairs), n, 3)
        error_all = np.asarray(flat_errors, dtype=float).reshape(len(permutation_pairs), n)
        enough_views = (
            np.isfinite(observations).all(axis=-1).sum(axis=-1) >= 2
        )
        valid_xyz = enough_views & np.isfinite(xyz_all).all(axis=-1)
        if low is not None and high is not None:
            valid_xyz &= np.all((xyz_all >= low) & (xyz_all <= high), axis=-1)
        error_all = np.where(valid_xyz & np.isfinite(error_all), error_all, np.inf)

        frame_candidates = []
        for candidate_index, (perm1, perm2) in enumerate(permutation_pairs):
            groups = {}
            errors: Dict[str, float] = {}
            missing_views: list[int] = []
            for fish_index, label in enumerate(labels):
                missing = np.flatnonzero(
                    ~np.isfinite(observations[candidate_index, fish_index]).all(axis=-1)
                )
                missing_views.append(int(missing[0]) if len(missing) == 1 else -1)
                if not valid_xyz[candidate_index, fish_index]:
                    groups[label] = [np.nan, np.nan, np.nan]
                    errors[label] = float("inf")
                    continue
                point = xyz_all[candidate_index, fish_index]
                groups[label] = [float(point[0]), float(point[1]), float(point[2])]
                errors[label] = float(error_all[candidate_index, fish_index])
            total_cost = float(sum(errors.values()))
            frame_candidates.append({
                    "groups": groups,
                    "perm_cam1": list(perm1),
                    "perm_cam2": list(perm2),
                    "recovered": [value >= 0 for value in missing_views],
                    "missing_views": missing_views,
                    # Historical key retained for older downstream readers.
                    "synthetic_views": missing_views,
                    "cost": total_cost,
                    "errors": errors,
                    "score_unit": scoring_cal.score_unit,
                    "triangulation_backend": cal.backend_name,
                    "scoring_backend": scoring_cal.backend_name,
                    "scoring_profile": "legacy_combined",
            })
        candidates_by_frame[frame] = frame_candidates

    return candidates_by_frame
