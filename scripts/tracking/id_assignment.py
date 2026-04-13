"""
Identity assignment for 2D detections.

The algorithm enforces a one-to-one mapping between fish IDs (`a`, `b`, `c`)
and detections in each frame using Hungarian assignment.

Core logic:
1. On the first frame, sort detections by x and assign `a`, `b`, `c`.
2. On later frames, minimize:
      cost[i, j] = distance + motion-direction penalty
   Fish with no previous position receive a forbidden cost until they are
   introduced by the fallback assignment step.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

# Fish with no previous position receive this forbidden cost and may be filled
# in later by the x-ordered fallback assignment.
FORBIDDEN_COST = 1e9


def _compute_cost_matrix(prev_positions, curr_positions,
                        prev_prev_positions=None, flow_weight=0.0,
                        max_distance=None):
    """
    Build the assignment cost matrix.

    `cost[i, j] = distance + motion-direction penalty`

    Rows with no previous position stay at `FORBIDDEN_COST`. Candidates whose
    Euclidean distance exceeds `max_distance` are also rejected.
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
            if max_distance is not None and np.isfinite(max_distance) and dist > float(max_distance):
                continue
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
    Assign fish IDs for a single frame.

    - First frame: assign `a/b/c` by x-order.
    - Later frames: use Hungarian assignment on distance plus motion direction.
    - `max_distance` is a hard rejection threshold for implausible matches.

    Returns:
        Mapping `{track_id: fish_id}`.
    """
    track_ids = sorted(curr_positions_dict.keys())
    if len(track_ids) == 0:
        return {}

    curr_positions = [curr_positions_dict[t][:2] for t in track_ids]
    prev_positions = [prev_positions_dict.get(fid, None) for fid in fish_ids]
    prev_prev_positions = None
    if prev_prev_positions_dict is not None and flow_weight > 0:
        prev_prev_positions = [prev_prev_positions_dict.get(fid, None) for fid in fish_ids]

    # First frame: assign IDs from left to right.
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
        max_distance=max_distance,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if c >= len(track_ids):
            continue
        if cost[r, c] >= FORBIDDEN_COST:
            continue
        assignment[track_ids[c]] = fish_ids[r]

    # Fallback assignment is only for fish that have never been seen before.
    # Fish that are temporarily missing keep their previous position and do not
    # enter this path, so they are not force-assigned by x-order mid-sequence.
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
    Assign IDs over time for all three views.

    When `flow_weight > 0`, motion-direction consistency is added to reduce
    cross-frame ID swaps. Each view is tracked on its own frame timeline, so a
    dropped frame in one camera does not suppress assignment in the others.
    """
    assignments = {player: {} for player in data_dict.keys()}
    fish_ids = (["a", "b", "c"] + [chr(ord("d") + k) for k in range(max(0, num_fish - 3))])[:num_fish]
    previous_positions = {player: {fid: None for fid in fish_ids} for player in data_dict.keys()}
    prev_prev_positions = {player: {fid: None for fid in fish_ids} for player in data_dict.keys()}
    previous_frames = {player: None for player in data_dict.keys()}

    for player in data_dict.keys():
        player_frames = np.sort(np.asarray(data_dict[player].get('frames', []), dtype=int))
        for frame in player_frames:
            current_positions = data_dict[player]['positions'].get(frame, {})
            prev_frame = previous_frames[player]
            frame_gap = 1 if prev_frame is None else max(1, int(frame) - int(prev_frame))
            effective_max_distance = None
            if max_distance is not None and np.isfinite(max_distance):
                effective_max_distance = float(max_distance) * float(frame_gap)

            assignment = assign_ids_frame(
                previous_positions[player],
                current_positions,
                fish_ids,
                prev_prev_positions_dict=prev_prev_positions[player] if flow_weight != 0 else None,
                flow_weight=flow_weight,
                max_distance=effective_max_distance,
            )

            assignments[player][frame] = assignment

            # Update the position history used by the next frame.
            prev_prev_positions[player] = dict(previous_positions[player])
            track_by_fish = {fid: tid for tid, fid in assignment.items()}
            for fid in fish_ids:
                tid = track_by_fish.get(fid)
                if tid is not None and tid in current_positions:
                    previous_positions[player][fid] = current_positions[tid][:2]
                else:
                    previous_positions[player][fid] = prev_prev_positions[player].get(fid)
            previous_frames[player] = int(frame)

    return assignments


def create_tracked_dataframe(data_dict, assignments):
    """
    Build DataFrames with a `fish_id` column from the assignment results.
    """
    tracked_dfs = {}
    
    for player in data_dict.keys():
        df = data_dict[player]['df'] if 'df' in data_dict[player] else None
        if df is None:
            # Rebuild the DataFrame from the positions structure if needed.
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
        
        # Add the `fish_id` column if it does not exist yet.
        if 'fish_id' not in df.columns:
            df['fish_id'] = df.apply(
                lambda row: assignments[player].get(row['frame'], {}).get(row['track'], 'unassigned'),
                axis=1
            )
        
        tracked_dfs[player] = df
    
    return tracked_dfs
