"""Choose the best 3D candidate per frame."""

from typing import Dict, List, Tuple

import numpy as np

LABELS = ["a", "b", "c"]


def _point_in_bounds(
    pt: List[float],
    low: float | None,
    high: float | None,
) -> bool:
    if low is None or high is None:
        return True
    if any(np.isnan(pt)):
        return False
    return low <= pt[0] <= high and low <= pt[1] <= high and low <= pt[2] <= high


def select_best_all_frames(
    candidates_by_frame: Dict[int, List[Dict]],
    *,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
) -> Dict[int, Dict]:
    """
    For each frame, pick the minimum-cost candidate whose points stay inside
    `xyz_bounds`. If none satisfy the bounds, choose the minimum-cost candidate
    regardless of bounds.
    """
    low, high = (xyz_bounds or (None, None))[0], (xyz_bounds or (None, None))[1]
    best_by_frame: Dict[int, Dict] = {}

    for frame, cands in candidates_by_frame.items():
        if not cands:
            best_by_frame[frame] = {"groups": {}, "perm_cam1": [], "perm_cam2": []}
            continue
        valid = [
            c for c in cands
            if all(
                _point_in_bounds(c["groups"].get(lb, [np.nan] * 3), low, high)
                for lb in LABELS
            )
        ]
        if valid:
            chosen = min(valid, key=lambda c: c["cost"])
        else:
            chosen = min(cands, key=lambda c: c["cost"])
        best_by_frame[frame] = {
            "groups": dict(chosen["groups"]),
            "perm_cam1": list(chosen["perm_cam1"]),
            "perm_cam2": list(chosen["perm_cam2"]),
        }
    return best_by_frame
