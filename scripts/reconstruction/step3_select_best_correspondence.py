"""
Step 3: 最適な対応の選択（参考 Dynamics_of_dominance 逻辑）
========================================================

不使用 KMeans 聚类，而是基于配准误差（registration cost）选择最优对应。

对每一帧的 27 个 3D 候选点，选择 3 个 triplets (cam0_idx, cam1_idx, cam2_idx)
使得：
  - 每个相机索引 0,1,2 各被使用一次
  - 总配准误差最小

即：在 6×6=36 种有效分配中，选择总 cost 最小的分配，得到 a/b/c 三条鱼的 3D 坐标。

输出 best.json 结构:
    {
      "0": {
        "groups": {
          "a": [Xa, Ya, Za],
          "b": [Xb, Yb, Zb],
          "c": [Xc, Yc, Zc]
        }
      },
      ...
    }

用法:
    python -m scripts.reconstruction.step3_select_best_correspondence --candidates candidates.json --output best.json
"""

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GROUP_LABELS = ["a", "b", "c"]


def _build_cost_matrix(
    candidates: List[Dict],
    xyz_bounds: Tuple[float, float] | None = None,
) -> np.ndarray | None:
    """
    构建 cost[i0, i1, i2] 矩阵。
    若某组合不存在、cost 为 None、或 xyz 超出 xyz_bounds，则视为 inf。
    """
    cost_mat = np.full((3, 3, 3), np.inf, dtype=float)
    for c in candidates:
        i0 = c.get("cam0_idx", 0)
        i1 = c.get("cam1_idx", 0)
        i2 = c.get("cam2_idx", 0)
        cost = c.get("cost")
        xyz = c.get("xyz", [])
        # 负坐标候选视为无效
        if xyz and any(v < 0 for v in xyz):
            continue
        # 若指定了物理范围，超出范围的候选也视为无效
        if xyz and xyz_bounds is not None:
            low, high = xyz_bounds
            if any(v < low or v > high for v in xyz):
                continue
        if cost is not None and not (isinstance(cost, float) and np.isnan(cost)):
            cost_mat[i0, i1, i2] = float(cost)
    return cost_mat


def _select_best_assignment(
    candidates: List[Dict],
    xyz_bounds: Tuple[float, float] | None = None,
) -> tuple[Dict[str, List[float]] | None, List[int] | None, List[int] | None]:
    """
    从 27 个候选中选择 3 个 triplets，使总 cost 最小。
    xyz_bounds 超出范围的候选在 _build_cost_matrix 中已排除。
    """
    if len(candidates) < 3:
        return None, None, None

    cost_mat = _build_cost_matrix(candidates, xyz_bounds)
    cand_by_idx: Dict[Tuple[int, int, int], Dict] = {}
    for c in candidates:
        key = (c["cam0_idx"], c["cam1_idx"], c["cam2_idx"])
        cand_by_idx[key] = c

    perms = list(permutations([0, 1, 2]))
    best_total = np.inf
    best_assign: List[Tuple[int, int, int]] = []

    for p in perms:
        for q in perms:
            total = 0.0
            assign = []
            for i in range(3):
                i0, i1, i2 = i, p[i], q[i]
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
    for idx, (i0, i1, i2) in enumerate(best_assign):
        key = (i0, i1, i2)
        if key in cand_by_idx:
            label = GROUP_LABELS[idx]
            groups[label] = cand_by_idx[key]["xyz"]
            perm_cam1.append(i1)
            perm_cam2.append(i2)
    if len(groups) != 3:
        return None, None, None
    return groups, perm_cam1, perm_cam2


def select_best_all_frames(
    candidates_by_frame: Dict[int, List[Dict]],
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict[int, Dict]:
    """
    对每帧基于配准误差选择最优 a/b/c 对应。
    xyz_bounds: 与 step2 一致的有效范围，超出范围的候选不参与选择。
    """
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
    """
    执行 Step 3。
    xyz_bounds: 例如 (0, 200)，与 step2 一致；超出范围的候选不参与最优选择。
    """
    with open(candidates_path, encoding="utf-8") as f:
        raw = json.load(f)
    candidates_by_frame = {int(k): v for k, v in raw.items()}

    best = select_best_all_frames(candidates_by_frame, xyz_bounds)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in best.items()}, f, indent=2, ensure_ascii=False)
        print(f"已保存: {output_path}")
    return best


def main():
    ap = argparse.ArgumentParser(description="Step 3: 基于配准误差选择最优 a/b/c 对应（参考 Dynamics_of_dominance）")
    ap.add_argument("--candidates", required=True, help="Step 2 输出的 JSON")
    ap.add_argument("--output", default=None, help="输出 JSON 路径")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="3D 有效范围 [LOW, HIGH]，与 step2 一致，例如 --xyz_bounds 0 200",
    )
    args = ap.parse_args()
    bounds = tuple(args.xyz_bounds) if args.xyz_bounds is not None else None
    run(args.candidates, args.output, xyz_bounds=bounds)


if __name__ == "__main__":
    main()
