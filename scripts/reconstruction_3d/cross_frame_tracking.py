"""
Unify identities across frames in 3D space.

Missing (`NaN`) points do not participate in the distance calculation.

- First frame: assign `ID_1`, `ID_2`, `ID_3` by increasing X coordinate.
- Later frames: match the current 3D points to the previous IDs using
  Hungarian assignment on Euclidean distance.
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

# Reconstructed IDs; no longer use `a/b/c` once tracking is in 3D.
ID_LABELS = ["ID_1", "ID_2", "ID_3"]

# Upstream `select_best` still uses `a/b/c` labels for per-frame geometry.
_SRC_LABELS = ["a", "b", "c"]


def _get_three_points_from_groups(groups: Dict) -> np.ndarray:
    """Extract three points from `groups` into a `(3, 3)` array."""
    out = np.full((3, 3), np.nan)
    labels = ID_LABELS if "ID_1" in groups else _SRC_LABELS
    for i, lb in enumerate(labels):
        if lb in groups and groups[lb] is not None:
            pt = groups[lb]
            if len(pt) == 3 and not (np.isnan(pt).any() or np.isinf(pt).any()):
                out[i, :] = pt
    return out


def apply_cross_frame_tracking(
    best_by_frame: Dict[int, Dict],
    *,
    flow_weight: float = 10.0,
) -> Dict[int, Dict]:
    """
    Track identities across frames and return groups keyed by `ID_1/2/3`.
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}

    # First frame: assign IDs by x-order.
    groups0 = best_by_frame[frames[0]].get("groups", {})
    pts0 = _get_three_points_from_groups(groups0)
    valid = [i for i in range(3) if not np.any(np.isnan(pts0[i]))]
    valid_sorted = sorted(valid, key=lambda i: pts0[i, 0])
    new_groups_0 = {}
    for i, id_name in enumerate(ID_LABELS):
        if i < len(valid_sorted):
            new_groups_0[id_name] = list(pts0[valid_sorted[i]])
        else:
            new_groups_0[id_name] = [np.nan, np.nan, np.nan]
    result[frames[0]] = {
        "groups": new_groups_0,
        "perm_cam1": list(best_by_frame[frames[0]].get("perm_cam1", [])),
        "perm_cam2": list(best_by_frame[frames[0]].get("perm_cam2", [])),
    }

    if len(frames) == 1:
        return result

    # Later frames: match previous IDs to the current a/b/c points by distance.
    for idx in range(1, len(frames)):
        t_prev, t_curr = frames[idx - 1], frames[idx]
        prev_groups = result[t_prev].get("groups", {})
        curr_info = best_by_frame[t_curr]
        curr_groups = curr_info.get("groups", {})
        pts_prev = _get_three_points_from_groups(prev_groups)
        pts_curr = _get_three_points_from_groups(curr_groups)

        cost = np.full((3, 3), 1e10)
        for i in range(3):
            if np.any(np.isnan(pts_prev[i])):
                continue
            for j in range(3):
                if np.any(np.isnan(pts_curr[j])):
                    continue
                cost[i, j] = float(np.linalg.norm(pts_prev[i] - pts_curr[j]))

        row_ind, col_ind = linear_sum_assignment(cost)
        # Map previous IDs to current-frame points.
        new_groups = {}
        for i in range(3):
            id_name = ID_LABELS[i]
            j = col_ind[np.where(row_ind == i)[0][0]]
            new_groups[id_name] = list(pts_curr[j])
        result[t_curr] = {
            "groups": new_groups,
            "perm_cam1": list(curr_info.get("perm_cam1", [])),
            "perm_cam2": list(curr_info.get("perm_cam2", [])),
        }

    return result


def apply_first_frame_id_only(best_by_frame: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Determine the label-to-ID mapping from the first valid frame only and keep
    that mapping fixed for the whole clip.
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    # Find the first frame where all three points are valid.
    label_to_id: Dict[str, str] = {}
    for f in frames:
        groups = best_by_frame[f].get("groups", {})
        valid = [
            (lb, groups[lb])
            for lb in _SRC_LABELS
            if lb in groups
            and groups[lb] is not None
            and len(groups[lb]) == 3
            and not (np.isnan(np.asarray(groups[lb])).any() or np.isinf(np.asarray(groups[lb])).any())
        ]
        if len(valid) == 3:
            valid_sorted = sorted(valid, key=lambda x: x[1][0])
            label_to_id = {
                valid_sorted[0][0]: ID_LABELS[0],
                valid_sorted[1][0]: ID_LABELS[1],
                valid_sorted[2][0]: ID_LABELS[2],
            }
            break
    if not label_to_id:
        label_to_id = {"a": ID_LABELS[0], "b": ID_LABELS[1], "c": ID_LABELS[2]}

    result: Dict[int, Dict] = {}
    for f in frames:
        info = best_by_frame[f]
        groups = info.get("groups", {})
        new_groups = {}
        for lb, id_name in label_to_id.items():
            if lb in groups and groups[lb] is not None:
                pt = groups[lb]
                if len(pt) == 3:
                    new_groups[id_name] = list(pt)
                else:
                    new_groups[id_name] = [np.nan, np.nan, np.nan]
            else:
                new_groups[id_name] = [np.nan, np.nan, np.nan]
        result[f] = {
            "groups": new_groups,
            "perm_cam1": list(info.get("perm_cam1", [])),
            "perm_cam2": list(info.get("perm_cam2", [])),
        }
    return result
