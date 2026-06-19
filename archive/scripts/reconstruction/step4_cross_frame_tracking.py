"""
Step 4: track identities across frames in 3D space.

This follows the same principle as the reference implementation in
Dynamics_of_dominance: use the previous frame's identities and current-frame
3D observations, then match them with Hungarian assignment on 3D distance.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUP_LABELS = ["a", "b", "c"]
UNMATCHABLE_COST = 1e10


def _distance_3d(p: List[float], q: List[float]) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)))


def apply_cross_frame_tracking(
    best_by_frame: Dict[int, Dict],
    flow_weight: float = 0.0,
) -> Dict[int, Dict]:
    """
    Track labels across frames.

    - First frame: use the Step 3 labels directly as the initial identity state.
    - Later frames: match current observations to the previous identities in 3D.
    """
    frames = sorted(best_by_frame.keys())
    if len(frames) == 0:
        return best_by_frame

    result: Dict[int, Dict] = {}
    result[frames[0]] = dict(best_by_frame[frames[0]])
    groups0 = result[frames[0]].get("groups", {})
    last_known: List[List[float] | None] = [
        list(groups0[la]) if la in groups0 else None
        for la in GROUP_LABELS
    ]

    for idx in range(1, len(frames)):
        t1 = frames[idx]
        raw_t1 = best_by_frame[t1]
        groups_t1_raw = raw_t1.get("groups", {})
        curr_obs = [groups_t1_raw[la] for la in GROUP_LABELS if la in groups_t1_raw]
        n_curr = len(curr_obs)

        if n_curr == 0:
            result[t1] = dict(raw_t1)
            continue

        cost = np.full((len(GROUP_LABELS), n_curr), UNMATCHABLE_COST, dtype=float)
        for i in range(len(GROUP_LABELS)):
            if last_known[i] is None:
                continue
            for j in range(n_curr):
                d = _distance_3d(last_known[i], curr_obs[j])
                cost[i, j] = d if np.isfinite(d) else UNMATCHABLE_COST

        row_ind, col_ind = linear_sum_assignment(cost)

        new_groups: Dict[str, List[float]] = {}
        for k in range(len(row_ind)):
            i = int(row_ind[k])
            j = int(col_ind[k])
            if i >= len(GROUP_LABELS) or j >= n_curr or cost[i, j] >= UNMATCHABLE_COST:
                continue
            new_groups[GROUP_LABELS[i]] = list(curr_obs[j])

        for i, la in enumerate(GROUP_LABELS):
            if la not in new_groups and last_known[i] is not None:
                new_groups[la] = list(last_known[i])

        perm1_raw = raw_t1.get("perm_cam1")
        perm2_raw = raw_t1.get("perm_cam2")
        new_perm_cam1 = []
        new_perm_cam2 = []
        if perm1_raw is not None and perm2_raw is not None and n_curr <= len(perm1_raw):
            for i in range(len(GROUP_LABELS)):
                idx_arr = np.where(row_ind == i)[0]
                if len(idx_arr) > 0:
                    j = int(col_ind[idx_arr[0]])
                    if j < len(perm1_raw):
                        new_perm_cam1.append(perm1_raw[j])
                        new_perm_cam2.append(perm2_raw[j])
            if len(new_perm_cam1) != len(new_groups) and len(perm1_raw) == 3:
                new_perm_cam1 = list(perm1_raw)
                new_perm_cam2 = list(perm2_raw)

        result[t1] = {"groups": new_groups}
        if new_perm_cam1 and new_perm_cam2:
            result[t1]["perm_cam1"] = new_perm_cam1
            result[t1]["perm_cam2"] = new_perm_cam2
        elif raw_t1.get("perm_cam1") is not None and raw_t1.get("perm_cam2") is not None:
            result[t1]["perm_cam1"] = raw_t1["perm_cam1"]
            result[t1]["perm_cam2"] = raw_t1["perm_cam2"]

        for i, la in enumerate(GROUP_LABELS):
            if la in new_groups:
                last_known[i] = new_groups[la]

    return result


def run(best_json_path: str, output_path: str | None = None, flow_weight: float = 0.0) -> Dict:
    """Run Step 4."""
    with open(best_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    best_by_frame = {int(k): v for k, v in raw.items()}

    tracked = apply_cross_frame_tracking(best_by_frame, flow_weight=flow_weight)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in tracked.items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
    return tracked


def main():
    ap = argparse.ArgumentParser(description="Step 4: track identities across frames in 3D")
    ap.add_argument("--best_json", required=True, help="best.json from Step 3")
    ap.add_argument("--output", default=None, help="Output JSON path (defaults to overwrite input)")
    ap.add_argument("--flow_weight", type=float, default=0.0, help="Unused compatibility parameter")
    args = ap.parse_args()
    output = args.output or args.best_json
    run(args.best_json, output, args.flow_weight)


if __name__ == "__main__":
    main()
