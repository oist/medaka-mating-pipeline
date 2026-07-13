"""
Build 3D candidates from the three-view 2D points.

For each frame, the pipeline enumerates permutations across the three views and
produces multiple candidate groups of three 3D points.

- Missing 2D values produce `NaN` 3D points.
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
        frame_candidates = []

        for perm1 in perms1:
            for perm2 in perms2:
                groups = {}
                errors: Dict[str, float] = {}
                total_cost = 0.0
                for i, label in enumerate(labels):
                    j1, j2 = perm1[i], perm2[i]
                    im0 = np.array(pts0[i], dtype=float)
                    im1 = np.array(pts1[j1], dtype=float)
                    im2 = np.array(pts2[j2], dtype=float)
                    triplet = scoring_cal.complete_triplet(np.vstack([im0, im1, im2]))
                    if np.any(np.isnan(triplet)):
                        total_cost += np.inf
                        groups[label] = [np.nan, np.nan, np.nan]
                        errors[label] = float("inf")
                        continue
                    xyz = cal.compute_XYZ_from_imageCoord_triplet(triplet)
                    # Invalid or out-of-bounds predictions are left as NaN.
                    if np.any(np.isnan(xyz)) or np.any(np.isinf(xyz)):
                        total_cost += np.inf
                        groups[label] = [np.nan, np.nan, np.nan]
                        errors[label] = float("inf")
                        continue
                    if low is not None and high is not None:
                        if np.any(xyz < low) or np.any(xyz > high):
                            total_cost += np.inf
                            groups[label] = [np.nan, np.nan, np.nan]
                            errors[label] = float("inf")
                            continue
                    err = scoring_cal.compute_triplet_registration_error(triplet)
                    if np.isfinite(err):
                        total_cost += err
                        errors[label] = float(err)
                    else:
                        total_cost += np.inf
                        errors[label] = float("inf")
                    groups[label] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                frame_candidates.append({
                    "groups": groups,
                    "perm_cam1": list(perm1),
                    "perm_cam2": list(perm2),
                    "cost": float(total_cost),
                    "errors": errors,
                })
        candidates_by_frame[frame] = frame_candidates

    return candidates_by_frame
