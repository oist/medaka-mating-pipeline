"""
Step 4: 跨帧个体追踪（匈牙利算法 + 运动方向 flow）
==================================================

对 Step 3 输出的每帧 a/b/c 3D 坐标进行跨帧关联，保证同一物理个体在时间上保持同一标签。

使用匈牙利算法最小化匹配代价，代价 = 3D 距离 + flow 惩罚（运动方向不一致时增加代价）。

输入: best.json（Step 3 输出）
输出: best_tracked.json（跨帧追踪后的 best.json，结构相同）

用法:
    python -m scripts.reconstruction.step4_cross_frame_tracking --best_json best.json --output best_tracked.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUP_LABELS = ["a", "b", "c"]


def _compute_cost_matrix(
    groups_t: Dict[str, List[float]],
    groups_t1_raw: Dict[str, List[float]],
    groups_t_prev: Dict[str, List[float]] | None,
    flow_weight: float,
) -> np.ndarray:
    """
    构建 3x3 代价矩阵：cost[i,j] = 将 t 帧的 canonical 标签 i 匹配到 t+1 帧的 raw 标签 j 的代价。

    代价 = 3D 欧氏距离 + flow 惩罚（运动方向不一致时增加）
    """
    cost_mat = np.zeros((3, 3))
    for i, la in enumerate(GROUP_LABELS):
        pos_i = np.array(groups_t[la], dtype=float)
        for j, lb in enumerate(GROUP_LABELS):
            pos_j = np.array(groups_t1_raw[lb], dtype=float)
            dist = float(np.linalg.norm(pos_i - pos_j))

            flow_penalty = 0.0
            if groups_t_prev is not None and flow_weight > 0:
                pos_prev = np.array(groups_t_prev[la], dtype=float)
                v_prev = pos_i - pos_prev
                v_new = pos_j - pos_i
                n_prev = np.linalg.norm(v_prev)
                n_new = np.linalg.norm(v_new)
                if n_prev > 1e-6 and n_new > 1e-6:
                    cos_angle = np.dot(v_prev, v_new) / (n_prev * n_new)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    flow_penalty = (1.0 - cos_angle) * flow_weight

            cost_mat[i, j] = dist + flow_penalty
    return cost_mat


def apply_cross_frame_tracking(
    best_by_frame: Dict[int, Dict],
    flow_weight: float = 10.0,
) -> Dict[int, Dict]:
    """
    对 best.json 进行跨帧追踪，使同一物理个体在时间上保持同一标签。

    Args:
        best_by_frame: {frame: {"groups": {...}, "perm_cam1": [...], "perm_cam2": [...]}}
        flow_weight: flow 方向惩罚权重（mm），运动方向相反时约增加 2*flow_weight 代价

    Returns:
        结构同 best_by_frame，但 a/b/c 已按跨帧一致性重排
    """
    frames = sorted(best_by_frame.keys())
    if len(frames) < 2:
        return best_by_frame

    result: Dict[int, Dict] = {}
    result[frames[0]] = dict(best_by_frame[frames[0]])

    for idx in range(len(frames) - 1):
        t, t1 = frames[idx], frames[idx + 1]
        groups_t = result[t]["groups"]
        raw_t1 = best_by_frame[t1]
        groups_t1_raw = raw_t1.get("groups", {})
        groups_t_prev = result[frames[idx - 1]]["groups"] if idx > 0 else None

        if len(groups_t) != 3 or len(groups_t1_raw) != 3:
            result[t1] = dict(raw_t1)
            continue

        cost_mat = _compute_cost_matrix(
            groups_t, groups_t1_raw, groups_t_prev, flow_weight
        )

        row_ind, col_ind = linear_sum_assignment(cost_mat)

        new_groups: Dict[str, List[float]] = {}
        new_perm_cam1: List[int] = []
        new_perm_cam2: List[int] = []
        perm1_raw = raw_t1.get("perm_cam1")
        perm2_raw = raw_t1.get("perm_cam2")

        for i, la in enumerate(GROUP_LABELS):
            raw_label = GROUP_LABELS[col_ind[i]]
            new_groups[la] = list(groups_t1_raw[raw_label])
            if perm1_raw is not None and perm2_raw is not None:
                raw_idx = col_ind[i]
                new_perm_cam1.append(int(perm1_raw[raw_idx]))
                new_perm_cam2.append(int(perm2_raw[raw_idx]))

        result[t1] = {"groups": new_groups}
        if new_perm_cam1 and new_perm_cam2:
            result[t1]["perm_cam1"] = new_perm_cam1
            result[t1]["perm_cam2"] = new_perm_cam2

    return result


def run(best_json_path: str, output_path: str | None = None, flow_weight: float = 10.0) -> Dict:
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
    ap = argparse.ArgumentParser(description="Step 4: cross-frame tracking (Hungarian + flow)")
    ap.add_argument("--best_json", required=True, help="best.json from Step 3")
    ap.add_argument("--output", default=None, help="output JSON path (default: overwrite input)")
    ap.add_argument("--flow_weight", type=float, default=10.0, help="flow direction penalty weight (mm)")
    args = ap.parse_args()
    output = args.output or args.best_json
    run(args.best_json, output, args.flow_weight)


if __name__ == "__main__":
    main()
