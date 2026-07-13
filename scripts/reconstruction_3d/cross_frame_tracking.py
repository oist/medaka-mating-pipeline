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


def _id_labels_for(n: int) -> List[str]:
    return [f"ID_{i + 1}" for i in range(n)]


def _src_labels_for(n: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(n)]


def _get_points_from_groups(
    groups: Dict, num_individuals: int
) -> np.ndarray:
    """Extract N 3D points from `groups` into an (N, 3) array."""
    out = np.full((num_individuals, 3), np.nan)
    if groups and any(k.startswith("ID_") for k in groups.keys()):
        labels = _id_labels_for(num_individuals)
    else:
        labels = _src_labels_for(num_individuals)
    for i, lb in enumerate(labels):
        if lb in groups and groups[lb] is not None:
            pt = groups[lb]
            if len(pt) == 3 and not (np.isnan(pt).any() or np.isinf(pt).any()):
                out[i, :] = pt
    return out


# Back-compat alias used by legacy callers/tests.
def _get_three_points_from_groups(groups: Dict) -> np.ndarray:
    return _get_points_from_groups(groups, 3)


def apply_cross_frame_tracking(
    best_by_frame: Dict[int, Dict],
    *,
    flow_weight: float = 10.0,
    num_individuals: int | None = None,
    max_jump_mm: float | None = None,
    max_jump_hard_cap_mm: float | None = None,
    max_gap_frames: int | None = None,
    source_id_hints_by_frame: Dict[int, Dict[int, str]] | None = None,
    source_hints_override_motion_gate: bool = True,
) -> Dict[int, Dict]:
    """
    Track identities across frames and return groups keyed by `ID_1/2/3/...`.

    Matching uses the last *valid* (non-NaN) 3D position of each ID instead of
    the immediately previous frame's assignment. This keeps an ID alive
    through short NaN gaps.

    Optional gating:
      - `max_jump_mm`: reject Hungarian matches where the per-frame distance
        exceeds `max_jump_mm * gap_frames`. An ID whose match is rejected
        keeps its last valid position and the current-frame slot is NaN.
      - `max_jump_hard_cap_mm`: absolute ceiling on the per-match distance,
        applied in addition to the gap-scaled gate. Prevents long NaN gaps
        from making the gate permissive enough to pick up a different fish
        (at gap=N the effective threshold is
        `min(max_jump_mm * N, max_jump_hard_cap_mm)`).
      - `max_gap_frames`: retire an ID whose last valid position is older
        than this many frames. Retired slots can be re-seeded by unmatched
        current-frame points.
      - `source_id_hints_by_frame`: optional high-confidence mapping
        ``frame -> {source_group_index: ID_name}``, normally derived from a
        well-separated tracked camera view. Hinted pairs are assigned before
        3D Hungarian matching; all remaining IDs retain the normal fallback.
      - `source_hints_override_motion_gate`: legacy anchor hints bypass the 3D
        gate when True. Tracklet-first reconstruction sets this False because
        conflicting candidates have already been reoptimized and a forced
        jump would reintroduce an ID swap.
    """
    n = num_individuals if num_individuals is not None else 3
    id_labels = _id_labels_for(n)

    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}

    # Seed on the first frame with any valid point, sorted left-to-right by X.
    groups0 = best_by_frame[frames[0]].get("groups", {})
    pts0 = _get_points_from_groups(groups0, n)
    valid = [i for i in range(n) if not np.any(np.isnan(pts0[i]))]
    valid_sorted = sorted(valid, key=lambda i: pts0[i, 0])

    # last_valid[id_name] = (frame, np.array([x, y, z])) or None.
    last_valid: Dict[str, tuple[int, np.ndarray] | None] = {
        id_name: None for id_name in id_labels
    }
    new_groups_0: Dict[str, List[float]] = {}
    id_mapping_0: List[int] = [-1] * n  # id_mapping[i] = src label idx for ID_(i+1)
    hints0 = (source_id_hints_by_frame or {}).get(frames[0], {})
    hinted_src0: set[int] = set()
    hinted_ids0: set[int] = set()
    for src_idx, id_name in hints0.items():
        if id_name not in id_labels or src_idx not in valid:
            continue
        id_idx = id_labels.index(id_name)
        if id_idx in hinted_ids0 or src_idx in hinted_src0:
            continue
        pt = pts0[src_idx]
        new_groups_0[id_name] = [float(pt[0]), float(pt[1]), float(pt[2])]
        last_valid[id_name] = (frames[0], pt.copy())
        id_mapping_0[id_idx] = int(src_idx)
        hinted_ids0.add(id_idx)
        hinted_src0.add(src_idx)

    remaining_src0 = [src_idx for src_idx in valid_sorted if src_idx not in hinted_src0]
    for i, id_name in enumerate(id_labels):
        if i in hinted_ids0:
            continue
        if remaining_src0:
            src_idx = remaining_src0.pop(0)
            pt = pts0[src_idx]
            new_groups_0[id_name] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[id_name] = (frames[0], pt.copy())
            id_mapping_0[i] = int(src_idx)
        else:
            new_groups_0[id_name] = [np.nan, np.nan, np.nan]
    first_info = best_by_frame[frames[0]]
    first_recovered_src = list(first_info.get("recovered", []))
    # id_mapping_0[id_idx] = src label idx for ID_(id_idx+1); reorder
    # the recovered flag to follow IDs instead of source labels.
    first_recovered_by_id = [
        bool(first_recovered_src[src_idx])
        if (0 <= src_idx < len(first_recovered_src))
        else False
        for src_idx in id_mapping_0
    ]
    first_synthetic_src = list(first_info.get("synthetic_views", []))
    first_synthetic_by_id = [
        int(first_synthetic_src[src_idx])
        if (0 <= src_idx < len(first_synthetic_src))
        else -1
        for src_idx in id_mapping_0
    ]
    result[frames[0]] = {
        "groups": new_groups_0,
        "perm_cam0": list(first_info.get("perm_cam0", [])),
        "perm_cam1": list(first_info.get("perm_cam1", [])),
        "perm_cam2": list(first_info.get("perm_cam2", [])),
        "recovered": first_recovered_by_id,
        "synthetic_views": first_synthetic_by_id,
        "id_mapping": id_mapping_0,
        "id_anchor": [i in hinted_ids0 for i in range(n)],
    }

    if len(frames) == 1:
        return result

    for idx in range(1, len(frames)):
        t_curr = frames[idx]
        curr_info = best_by_frame[t_curr]
        curr_groups = curr_info.get("groups", {})
        pts_curr = _get_points_from_groups(curr_groups, n)
        valid_curr = [j for j in range(n) if not np.any(np.isnan(pts_curr[j]))]

        # Retire stale IDs if they exceed the gap budget.
        if max_gap_frames is not None:
            for id_name in id_labels:
                lv = last_valid[id_name]
                if lv is not None and (t_curr - lv[0]) > max_gap_frames:
                    last_valid[id_name] = None

        cost = np.full((n, n), 1e10)
        for i, id_name in enumerate(id_labels):
            lv = last_valid[id_name]
            if lv is None:
                continue
            fr_prev, pt_prev = lv
            gap = max(1, t_curr - fr_prev)
            # Effective per-match threshold: the minimum of the gap-scaled
            # gate and the absolute hard cap (if either is set).
            threshold = None
            if max_jump_mm is not None:
                threshold = max_jump_mm * gap
            if max_jump_hard_cap_mm is not None:
                threshold = (
                    max_jump_hard_cap_mm
                    if threshold is None
                    else min(threshold, max_jump_hard_cap_mm)
                )
            for j in valid_curr:
                d = float(np.linalg.norm(pt_prev - pts_curr[j]))
                if threshold is not None and d > threshold:
                    continue
                cost[i, j] = d

        hints = (source_id_hints_by_frame or {}).get(t_curr, {})
        forced: Dict[int, int] = {}
        forced_src: set[int] = set()
        for src_idx, id_name in hints.items():
            if id_name not in id_labels or src_idx not in valid_curr:
                continue
            id_idx = id_labels.index(id_name)
            if (
                not source_hints_override_motion_gate
                and last_valid[id_name] is not None
                and cost[id_idx, src_idx] >= 1e10
            ):
                continue
            if id_idx in forced or src_idx in forced_src:
                continue
            forced[id_idx] = int(src_idx)
            forced_src.add(int(src_idx))

        new_groups: Dict[str, List[float]] = {
            id_name: [np.nan, np.nan, np.nan] for id_name in id_labels
        }
        id_mapping: List[int] = [-1] * n
        matched_curr: set[int] = set()

        # A reliable tracked-view identity overrides the ambiguous 3D gate.
        for i, j in forced.items():
            id_name = id_labels[i]
            pt = pts_curr[j]
            new_groups[id_name] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[id_name] = (t_curr, pt.copy())
            id_mapping[i] = j
            matched_curr.add(j)

        if not forced:
            # Preserve the established geometry-only linker exactly when no
            # high-confidence view hint is available.
            row_ind, col_ind = linear_sum_assignment(cost)
            candidate_pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
        else:
            remaining_ids = [i for i in range(n) if i not in forced]
            remaining_curr = [j for j in valid_curr if j not in forced_src]
            candidate_pairs = []
            if remaining_ids and remaining_curr:
                sub_cost = cost[np.ix_(remaining_ids, remaining_curr)]
                sub_rows, sub_cols = linear_sum_assignment(sub_cost)
                candidate_pairs = [
                    (remaining_ids[int(sub_i)], remaining_curr[int(sub_j)])
                    for sub_i, sub_j in zip(sub_rows, sub_cols)
                ]

        for i, j in candidate_pairs:
            if cost[i, j] >= 1e10:
                continue
            id_name = id_labels[i]
            pt = pts_curr[j]
            new_groups[id_name] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[id_name] = (t_curr, pt.copy())
            id_mapping[i] = j
            matched_curr.add(j)

        # Re-seed IDs whose last_valid is None using any unmatched current point.
        unmatched = [j for j in valid_curr if j not in matched_curr]
        for i, id_name in enumerate(id_labels):
            if last_valid[id_name] is not None:
                continue
            if not unmatched:
                break
            j = unmatched.pop(0)
            pt = pts_curr[j]
            new_groups[id_name] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[id_name] = (t_curr, pt.copy())
            id_mapping[i] = j

        recovered_src = list(curr_info.get("recovered", []))
        recovered_by_id = [
            bool(recovered_src[src_idx])
            if (0 <= src_idx < len(recovered_src))
            else False
            for src_idx in id_mapping
        ]
        synthetic_src = list(curr_info.get("synthetic_views", []))
        synthetic_by_id = [
            int(synthetic_src[src_idx])
            if (0 <= src_idx < len(synthetic_src))
            else -1
            for src_idx in id_mapping
        ]
        result[t_curr] = {
            "groups": new_groups,
            "perm_cam0": list(curr_info.get("perm_cam0", [])),
            "perm_cam1": list(curr_info.get("perm_cam1", [])),
            "perm_cam2": list(curr_info.get("perm_cam2", [])),
            "recovered": recovered_by_id,
            "synthetic_views": synthetic_by_id,
            "id_mapping": id_mapping,
            "id_anchor": [i in forced for i in range(n)],
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
            "perm_cam0": list(info.get("perm_cam0", [])),
            "perm_cam1": list(info.get("perm_cam1", [])),
            "perm_cam2": list(info.get("perm_cam2", [])),
        }
    return result
