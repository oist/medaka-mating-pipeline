"""
跨帧 ID 统一：在 3D 坐标上做 ID 匹配，空缺（NaN）不参与距离计算、匹配时忽略。

- 第一帧：按 X 坐标从小到大分配 ID_1, ID_2, ID_3（空缺 slot 仍占位，填 NaN）。
- 后续帧：用匈牙利算法按 3D 距离匹配到上一帧的 ID_1/ID_2/ID_3；代价矩阵中若任一点为 NaN 则代价为 inf，
  即忽视空缺，只根据有效点做匹配；匹配后该 slot 仍可得到 ID，3D 为 NaN 的留待最后统一插值。
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

# 重建后的个体 ID，不再使用 a/b/c
ID_LABELS = ["ID_1", "ID_2", "ID_3"]

# 上游 select_best 仍用 a,b,c 表示三种几何对应，此处仅用于读取每帧的 3 个点
_SRC_LABELS = ["a", "b", "c"]


def _get_three_points_from_groups(groups: Dict) -> np.ndarray:
    """从 groups（key 为 a,b,c 或 ID_1,ID_2,ID_3）取出 3 个点，返回 (3, 3) 数组，无效行为 NaN。"""
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
    对 best_by_frame 做跨帧追踪，输出 ID_1 / ID_2 / ID_3，与 a/b/c 无关。

    - 第一帧：取本帧 3 个 3D 点，按 X 坐标**从小到大**排序，依次赋 ID_1, ID_2, ID_3。
    - 后续帧：用上一帧的 ID_1/ID_2/ID_3 的 3 个点与当前帧的 3 个点建代价矩阵（欧氏距离），
      匈牙利算法得到最优匹配，当前帧的 3 个点按匹配结果赋 ID_1, ID_2, ID_3。

    返回的 best_by_frame 每帧的 groups 的 key 为 "ID_1", "ID_2", "ID_3"。
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}

    # 第一帧：按 X 大小分配 ID_1, ID_2, ID_3（X 最小=ID_1，次之=ID_2，最大=ID_3）
    groups0 = best_by_frame[frames[0]].get("groups", {})
    pts0 = _get_three_points_from_groups(groups0)
    valid = [i for i in range(3) if not np.any(np.isnan(pts0[i]))]
    valid_sorted = sorted(valid, key=lambda i: pts0[i, 0])  # 按 X 升序
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

    # 后续帧：上一帧为 ID_1/ID_2/ID_3 顺序，当前帧为 a,b,c 顺序，用距离匹配
    for idx in range(1, len(frames)):
        t_prev, t_curr = frames[idx - 1], frames[idx]
        prev_groups = result[t_prev].get("groups", {})
        curr_info = best_by_frame[t_curr]
        curr_groups = curr_info.get("groups", {})
        pts_prev = _get_three_points_from_groups(prev_groups)   # 行 0,1,2 = ID_1, ID_2, ID_3
        pts_curr = _get_three_points_from_groups(curr_groups)   # 行 0,1,2 = 当前帧的 a,b,c 对应点

        cost = np.full((3, 3), 1e10)
        for i in range(3):
            if np.any(np.isnan(pts_prev[i])):
                continue
            for j in range(3):
                if np.any(np.isnan(pts_curr[j])):
                    continue
                cost[i, j] = float(np.linalg.norm(pts_prev[i] - pts_curr[j]))

        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind[k], col_ind[k]：上一帧的 row_ind[k] 号 ID（即 ID_{row_ind[k]+1}）匹配当前帧的 col_ind[k] 号点
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
    仅用第一帧（或首个三点点全有效的帧）决定 ID 映射，之后全片固定：a→ID_1, b→ID_2, c→ID_3。

    - 在首个有效帧内，将 a/b/c 按 X 坐标从小到大排序，依次赋 ID_1, ID_2, ID_3。
    - 得到映射 label_to_id（哪个字母对应哪个 ID）。
    - 后续所有帧直接沿用该映射，不再做匈牙利匹配。
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    # 找首个三点都有效的帧
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
            valid_sorted = sorted(valid, key=lambda x: x[1][0])  # 按 X 升序
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
