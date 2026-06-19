"""
Step 3: choose the best correspondence from the 3D candidates.

This implementation does not use KMeans. It chooses the minimum-total-cost
assignment under the per-camera indexing constraints.
"""

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GROUP_LABELS = ["a", "b", "c"]


def _infer_dimensions(candidates: List[Dict]) -> Tuple[int, int, int]:
    """Infer the cam0/cam1/cam2 index ranges from the candidate list."""
    n0 = n1 = n2 = 0
    for c in candidates:
        n0 = max(n0, c.get("cam0_idx", 0) + 1)
        n1 = max(n1, c.get("cam1_idx", 0) + 1)
        n2 = max(n2, c.get("cam2_idx", 0) + 1)
    return n0, n1, n2


def _build_cost_matrix(
    candidates: List[Dict],
    n0: int,
    n1: int,
    n2: int,
    xyz_bounds: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Build a cost tensor of shape `(n0, n1, n2)`."""
    cost_mat = np.full((n0, n1, n2), np.inf, dtype=float)
    for c in candidates:
        i0 = c.get("cam0_idx", 0)
        i1 = c.get("cam1_idx", 0)
        i2 = c.get("cam2_idx", 0)
        if i0 >= n0 or i1 >= n1 or i2 >= n2:
            continue
        cost = c.get("cost")
        if cost is not None and not (isinstance(cost, float) and np.isnan(cost)):
            cost_mat[i0, i1, i2] = float(cost)
    return cost_mat


def _select_best_assignment(
    candidates: List[Dict],
    xyz_bounds: Tuple[float, float] | None = None,
) -> tuple[Dict[str, List[float]] | None, List[int] | None, List[int] | None]:
    """
    Select `k` triplets (`k=min(3,n0,n1,n2)`) with minimum total cost.
    Supports frames containing only one or two fish as well.
    """
    if not candidates:
        return None, None, None

    n0, n1, n2 = _infer_dimensions(candidates)
    k = min(3, n0, n1, n2)
    if k == 0:
        return None, None, None

    cost_mat = _build_cost_matrix(candidates, n0, n1, n2, xyz_bounds)
    cand_by_idx: Dict[Tuple[int, int, int], Dict] = {}
    for c in candidates:
        key = (c["cam0_idx"], c["cam1_idx"], c["cam2_idx"])
        cand_by_idx[key] = c

    indices = list(range(k))
    perms = list(permutations(indices))
    best_total = np.inf
    best_assign: List[Tuple[int, int, int]] = []

    for p in perms:
        for q in perms:
            total = 0.0
            assign = []
            for i in range(k):
                i0, i1, i2 = i, p[i], q[i]
                if i0 >= n0 or i1 >= n1 or i2 >= n2:
                    total = np.inf
                    break
                c = cost_mat[i0, i1, i2]
                if np.isinf(c):
                    total = np.inf
                    break
                total += c
                assign.append((i0, i1, i2))
            if total < best_total:
                best_total = total
                best_assign = assign

    if not best_assign or np.isinf(best_total):
        return None, None, None

    groups: Dict[str, List[float]] = {}
    perm_cam1: List[int] = []
    perm_cam2: List[int] = []
    for idx in range(k):
        label = GROUP_LABELS[idx]
        i0, i1, i2 = best_assign[idx]
        key = (i0, i1, i2)
        if key in cand_by_idx:
            groups[label] = cand_by_idx[key]["xyz"]
            perm_cam1.append(i1)
            perm_cam2.append(i2)
    return groups, perm_cam1, perm_cam2


def select_best_all_frames(
    candidates_by_frame: Dict[int, List[Dict]],
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict[int, Dict]:
    """Choose the best `a/b/c` assignment per frame from the candidate set."""
    frames = sorted(candidates_by_frame.keys())
    result: Dict[int, Dict] = {}
    for frame in frames:
        cands = candidates_by_frame[frame]
        groups, perm_cam1, perm_cam2 = _select_best_assignment(cands, xyz_bounds)
        if groups:
            result[frame] = {"groups": groups}
            if perm_cam1 is not None and perm_cam2 is not None:
                result[frame]["perm_cam1"] = perm_cam1
                result[frame]["perm_cam2"] = perm_cam2
    return result


def run(
    candidates_path: str,
    output_path: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict:
    """Run Step 3."""
    with open(candidates_path, encoding="utf-8") as f:
        raw = json.load(f)
    candidates_by_frame = {int(k): v for k, v in raw.items()}

    best = select_best_all_frames(candidates_by_frame, xyz_bounds)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in best.items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
    return best


def main():
    ap = argparse.ArgumentParser(description="Step 3: choose the best a/b/c correspondence by registration cost")
    ap.add_argument("--candidates", required=True, help="JSON output from Step 2")
    ap.add_argument("--output", default=None, help="Output JSON path")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Valid 3D range [LOW, HIGH], for example --xyz_bounds 0 200",
    )
    args = ap.parse_args()
    bounds = tuple(args.xyz_bounds) if args.xyz_bounds is not None else None
    run(args.candidates, args.output, xyz_bounds=bounds)


if __name__ == "__main__":
    main()
