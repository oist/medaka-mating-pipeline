"""
ID 分配算法：严格保证每一帧 ID 一一对应（a, b, c），用 Hungarian 做最优分配。

核心逻辑：
1. 第一帧：按 x 坐标排序，把 3 个检测依次标为 a, b, c。
2. 后续帧：构造代价矩阵 cost[i,j] = 距离(i,j) + 运动方向惩罚，上一帧无位置的鱼代价设为无穷；
   用 Hungarian 算法求总代价最小的分配。不额外指定「最大距离」——谁近、谁运动方向一致，谁总代价小，算法自然选最优配对。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

# 上一帧无位置的鱼：与所有检测的代价设为此值，由后续「补充分配」按 x 顺序处理
FORBIDDEN_COST = 1e9


def _compute_cost_matrix(prev_positions, curr_positions,
                        prev_prev_positions=None, flow_weight=0.0):
    """
    代价矩阵：cost[i,j] = 距离 + 运动方向惩罚。
    仅当上一帧鱼 i 无位置时 cost[i,:] = FORBIDDEN_COST；其余一律用实际距离+flow，不设距离上限。
    """
    num_fish = len(prev_positions)
    num_det = len(curr_positions)
    cost = np.full((num_fish, num_det), FORBIDDEN_COST, dtype=float)

    for i in range(num_fish):
        p_prev = prev_positions[i]
        if p_prev is None:
            continue
        p_prev = np.array(p_prev, dtype=float)
        p_prev_prev = prev_prev_positions[i] if prev_prev_positions is not None else None
        for j in range(num_det):
            p_curr = curr_positions[j]
            if p_curr is None:
                continue
            p_curr = np.array(p_curr, dtype=float)
            dist = float(np.linalg.norm(p_prev - p_curr))
            flow_penalty = 0.0
            if flow_weight > 0 and p_prev_prev is not None:
                p_prev_prev = np.array(p_prev_prev, dtype=float)
                v_prev = p_prev - p_prev_prev
                v_new = p_curr - p_prev
                n_prev = np.linalg.norm(v_prev)
                n_new = np.linalg.norm(v_new)
                if n_prev > 1e-6 and n_new > 1e-6:
                    cos_angle = np.dot(v_prev, v_new) / (n_prev * n_new)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    flow_penalty = (1.0 - cos_angle) * flow_weight
            cost[i, j] = dist + flow_penalty
    return cost


def assign_ids_frame(prev_positions_dict, curr_positions_dict, fish_ids,
                     prev_prev_positions_dict=None, flow_weight=0.0, max_distance=100):
    """
    对单帧做 ID 分配。
    - 第一帧：按 x 排序分配 a/b/c。
    - 后续帧：代价 = 距离 + 运动方向，Hungarian 求最小总代价。max_distance 保留为兼容参数，不参与代价计算。
    返回: {track_id: fish_id}
    """
    track_ids = sorted(curr_positions_dict.keys())
    if len(track_ids) == 0:
        return {}

    curr_positions = [curr_positions_dict[t][:2] for t in track_ids]
    prev_positions = [prev_positions_dict.get(fid, None) for fid in fish_ids]
    prev_prev_positions = None
    if prev_prev_positions_dict is not None and flow_weight > 0:
        prev_prev_positions = [prev_prev_positions_dict.get(fid, None) for fid in fish_ids]

    # 第一帧：无历史位置，按 x 坐标从左到右依次标为 a, b, c
    if all(p is None for p in prev_positions):
        xs = [p[0] for p in curr_positions]
        order = np.argsort(xs)
        assignment = {}
        for i, idx in enumerate(order):
            if i < len(fish_ids):
                assignment[track_ids[idx]] = fish_ids[i]
        return assignment

    cost = _compute_cost_matrix(
        prev_positions, curr_positions,
        prev_prev_positions=prev_prev_positions,
        flow_weight=flow_weight,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if c >= len(track_ids):
            continue
        if cost[r, c] >= FORBIDDEN_COST:
            continue
        assignment[track_ids[c]] = fish_ids[r]

    # 补充分配（仅针对「从未出现过」的鱼，与缺失值填充无关）：
    # 仅当 prev_positions_dict.get(f) is None 时才参与——即该鱼在之前所有帧都没有被分配过位置（例如首帧只有 2 个检测，只给了 a、b，c 从未出现）。
    # 中途某一帧缺失 2 个个体时，那两条鱼会保留「上一次被匹配时的位置」（见 assign_ids_temporal 里未匹配则 previous_positions[fid]=原位置），
    # 因此 prev_positions_dict.get(f) 不是 None，不会进这里，不会按 x 硬配；未分配的检测和未分配的鱼本帧就保持未分配，由后续缺失值填充补坐标。
    unassigned_tracks = [t for t in track_ids if t not in assignment]
    unassigned_fish = [f for f in fish_ids if f not in assignment.values()]
    unassigned_fish_no_pos = [f for f in unassigned_fish if prev_positions_dict.get(f) is None]
    if unassigned_tracks and unassigned_fish_no_pos:
        order_by_x = np.argsort([curr_positions_dict[t][0] for t in unassigned_tracks])
        sorted_tracks = [unassigned_tracks[i] for i in order_by_x]
        sorted_fish = sorted(unassigned_fish_no_pos, key=lambda f: fish_ids.index(f))
        for i in range(min(len(sorted_tracks), len(sorted_fish))):
            assignment[sorted_tracks[i]] = sorted_fish[i]

    return assignment


def assign_ids_temporal(data_dict, num_fish=3, max_distance=100, flow_weight=0.0):
    """
    对三个视角的数据进行时间序列 ID 分配。
    flow_weight>0 时，代价矩阵加入运动方向一致性惩罚，减少跨帧 ID 交换。
    """
    common_frames = None
    for player, data in data_dict.items():
        if common_frames is None:
            common_frames = data['frames']
        else:
            common_frames = np.intersect1d(common_frames, data['frames'])

    assignments = {player: {} for player in data_dict.keys()}
    fish_ids = (["a", "b", "c"] + [chr(ord("d") + k) for k in range(max(0, num_fish - 3))])[:num_fish]
    previous_positions = {player: {fid: None for fid in fish_ids} for player in data_dict.keys()}
    prev_prev_positions = {player: {fid: None for fid in fish_ids} for player in data_dict.keys()}

    for frame in common_frames:
        for player in data_dict.keys():
            current_positions = data_dict[player]['positions'].get(frame, {})

            assignment = assign_ids_frame(
                previous_positions[player],
                current_positions,
                fish_ids,
                prev_prev_positions_dict=prev_prev_positions[player] if flow_weight != 0 else None,
                flow_weight=flow_weight,
                max_distance=max_distance,
            )

            assignments[player][frame] = assignment

            # 更新下一帧的「上一帧」位置：被匹配到的 fish 用本帧检测位置；未匹配的保留原位置，供下一帧继续匹配
            prev_prev_positions[player] = dict(previous_positions[player])
            matched_fish = {fid for _, fid in assignment.items()}
            for fid in fish_ids:
                if fid in matched_fish:
                    tid = next(t for t, f in assignment.items() if f == fid)
                    if tid in current_positions:
                        previous_positions[player][fid] = current_positions[tid][:2]
                    else:
                        previous_positions[player][fid] = prev_prev_positions[player].get(fid)
                else:
                    previous_positions[player][fid] = prev_prev_positions[player].get(fid)

    return assignments


def create_tracked_dataframe(data_dict, assignments):
    """
    创建带ID的DataFrame
    
    Args:
        data_dict: 原始数据字典
        assignments: ID分配结果
        
    Returns:
        dict: {player: DataFrame with 'fish_id' column}
    """
    tracked_dfs = {}
    
    for player in data_dict.keys():
        df = data_dict[player]['df'] if 'df' in data_dict[player] else None
        if df is None:
            # 需要从positions重建DataFrame
            rows = []
            for frame, positions in data_dict[player]['positions'].items():
                for track_id, (x, y, score) in positions.items():
                    fish_id = assignments[player].get(frame, {}).get(track_id, 'unassigned')
                    rows.append({
                        'frame': frame,
                        'track': track_id,
                        'fish_id': fish_id,
                        'x': x,
                        'y': y,
                        'score': score
                    })
            df = pd.DataFrame(rows)
        
        # 添加fish_id列
        if 'fish_id' not in df.columns:
            df['fish_id'] = df.apply(
                lambda row: assignments[player].get(row['frame'], {}).get(row['track'], 'unassigned'),
                axis=1
            )
        
        tracked_dfs[player] = df
    
    return tracked_dfs
