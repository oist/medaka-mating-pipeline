"""
Step 4: 跨帧个体追踪（完全对照 Dynamics_of_dominance 的 track_segment_in_3D_if_possible）
===============================================================================

Reference: Dynamics_of_dominance-main/tracking_code/lib/tracking.py
  - 用上一帧每条鱼的 3D 位置 last_known_positions 与当前帧 3 个 3D 观测做匈牙利匹配；
  - 代价 = 3D 欧氏距离（reference 里是 compute_distance_between_3D_fish，我们单点即欧氏距离）；
  - 无 flow 惩罚，无首帧按 X 排序（首帧直接用 Step3 的 a/b/c 顺序，作为初始 last_known）。

与 reference 一致：仅用「上一帧身份→当前观测」的 3D 距离最小化确定身份。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUP_LABELS = ["a", "b", "c"]

# 用大有限数代替 np.inf，避免 linear_sum_assignment 报 "cost matrix is infeasible"
UNMATCHABLE_COST = 1e10


def _distance_3d(p: List[float], q: List[float]) -> float:
    """单点 3D 欧氏距离（reference 的 compute_distance_between_3D_fish 在单 bodypoint 时即此）。"""
    return float(np.linalg.norm(np.array(p, dtype=float) - np.array(q, dtype=float)))


def apply_cross_frame_tracking(
    best_by_frame: Dict[int, Dict],
    flow_weight: float = 0.0,
) -> Dict[int, Dict]:
    """
    跨帧追踪：与 Dynamics_of_dominance 的 track_segment 一致。
    - 第一帧：用 Step3 结果作为初始 last_known（不重排）。
    - 后续帧：cost[i,j] = distance(last_known[i], curr_obs[j])，匈牙利后更新 last_known。
    flow_weight 保留参数兼容，此处不使用（reference 无 flow）。
    """
    frames = sorted(best_by_frame.keys())
    if len(frames) == 0:
        return best_by_frame

    result: Dict[int, Dict] = {}
    # 第一帧：直接采用 Step3 结果，作为 last_known 的初值
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

        # 代价矩阵：cost[i,j] = distance(last_known[i], curr_obs[j])
        # i = 身份(0,1,2)，j = 当前观测索引（与 reference 一致）
        # 用大有限数代替 inf，否则 linear_sum_assignment 会报 "cost matrix is infeasible"
        cost = np.full((len(GROUP_LABELS), n_curr), UNMATCHABLE_COST, dtype=float)
        for i in range(len(GROUP_LABELS)):
            if last_known[i] is None:
                continue
            for j in range(n_curr):
                d = _distance_3d(last_known[i], curr_obs[j])
                cost[i, j] = d if np.isfinite(d) else UNMATCHABLE_COST

        row_ind, col_ind = linear_sum_assignment(cost)

        # (row_ind[k], col_ind[k]) 表示 身份 row_ind[k] 匹配 观测 col_ind[k]
        new_groups: Dict[str, List[float]] = {}
        for k in range(len(row_ind)):
            i = int(row_ind[k])
            j = int(col_ind[k])
            if i >= len(GROUP_LABELS) or j >= n_curr or cost[i, j] >= UNMATCHABLE_COST:
                continue
            new_groups[GROUP_LABELS[i]] = list(curr_obs[j])

        # 未匹配的身份保留上一帧位置（与 reference 一致）
        for i, la in enumerate(GROUP_LABELS):
            if la not in new_groups and last_known[i] is not None:
                new_groups[la] = list(last_known[i])

        # perm_cam1/cam2：按 a,b,c 顺序取观测下标
        perm1_raw = raw_t1.get("perm_cam1")
        perm2_raw = raw_t1.get("perm_cam2")
        new_perm_cam1 = []
        new_perm_cam2 = []
        if perm1_raw is not None and perm2_raw is not None and n_curr <= len(perm1_raw):
            for i in range(len(GROUP_LABELS)):
                idx = np.where(row_ind == i)[0]
                if len(idx) > 0:
                    j = int(col_ind[idx[0]])
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

        # 更新 last_known（与 reference 一致）
        for i, la in enumerate(GROUP_LABELS):
            if la in new_groups:
                last_known[i] = new_groups[la]
            # 否则 last_known[i] 保持不变

    return result


def run(best_json_path: str, output_path: str | None = None, flow_weight: float = 0.0) -> Dict:
    """执行 Step 4。"""
    with open(best_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    best_by_frame = {int(k): v for k, v in raw.items()}

    tracked = apply_cross_frame_tracking(best_by_frame, flow_weight=flow_weight)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in tracked.items()}, f, indent=2, ensure_ascii=False)
        print(f"已保存: {output_path}")
    return tracked


def main():
    ap = argparse.ArgumentParser(description="Step 4: 跨帧追踪（对照 Dynamics_of_dominance，3D 距离匈牙利）")
    ap.add_argument("--best_json", required=True, help="Step 3 的 best.json")
    ap.add_argument("--output", default=None, help="输出 JSON 路径（默认覆盖输入）")
    ap.add_argument("--flow_weight", type=float, default=0.0, help="未使用，保留兼容")
    args = ap.parse_args()
    output = args.output or args.best_json
    run(args.best_json, output, args.flow_weight)


if __name__ == "__main__":
    main()
