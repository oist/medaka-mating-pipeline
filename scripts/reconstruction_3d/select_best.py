"""Choose the best 3D candidate per frame."""

from typing import Dict, List, Tuple

import numpy as np

LABELS = ["a", "b", "c"]


def _labels_for(num_individuals: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(num_individuals)]


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
    xyz_bounds: Tuple[float, float] | None = None,
    num_individuals: int | None = None,
    max_residual_per_fish: float | None = None,
) -> Dict[int, Dict]:
    """
    For each frame, pick the minimum-cost candidate whose points stay inside
    `xyz_bounds`. If none satisfy the bounds, choose the minimum-cost candidate
    regardless of bounds.

    When `max_residual_per_fish` is set, any fish whose per-candidate
    registration error exceeds the threshold is zeroed out to NaN even when
    the candidate itself wins the frame — this drops geometrically
    implausible individual matches while keeping the rest of the frame.
    """
    low, high = (xyz_bounds or (None, None))[0], (xyz_bounds or (None, None))[1]
    labels = _labels_for(num_individuals) if num_individuals is not None else LABELS
    best_by_frame: Dict[int, Dict] = {}

    for frame, cands in candidates_by_frame.items():
        if not cands:
            best_by_frame[frame] = {"groups": {}, "perm_cam1": [], "perm_cam2": []}
            continue
        valid = [
            c for c in cands
            if all(
                _point_in_bounds(c["groups"].get(lb, [np.nan] * 3), low, high)
                for lb in labels
            )
        ]
        if valid:
            chosen = min(valid, key=lambda c: c["cost"])
        else:
            chosen = min(cands, key=lambda c: c["cost"])

        groups_out = dict(chosen["groups"])
        errors = chosen.get("errors") or {}
        dropped_labels: List[str] = []
        if max_residual_per_fish is not None:
            for lb in labels:
                err = errors.get(lb)
                if err is None or not np.isfinite(err) or err > max_residual_per_fish:
                    if lb in groups_out:
                        groups_out[lb] = [np.nan, np.nan, np.nan]
                    dropped_labels.append(lb)

        best_by_frame[frame] = {
            "groups": groups_out,
            "perm_cam1": list(chosen["perm_cam1"]),
            "perm_cam2": list(chosen["perm_cam2"]),
            "errors": dict(errors),
            "dropped_high_residual": dropped_labels,
        }
    return best_by_frame
