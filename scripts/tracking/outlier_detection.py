"""
异常值检测和处理算法
逻辑：
1. 检测连续超过最大距离的帧（连续2帧、3帧等）
2. 删除这些连续异常帧（设为NaN）
3. 用插值补齐
4. 迭代检查，直到每一帧都不超过设定距离为止
"""
import numpy as np
import pandas as pd


def detect_frame_jumps(positions, max_jump_distance=200):
    """
    检测帧间位置跳跃（异常大的位移）
    
    逻辑：计算相邻帧之间的距离，如果距离超过阈值，标记为异常值
    
    Args:
        positions: (n_frames, 2) 位置数组，按frame排序
        max_jump_distance: 最大允许跳跃距离（像素）
        
    Returns:
        array: 布尔数组，True表示异常值（需要删除的点）
    """
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=bool)
    
    # 计算相邻帧之间的距离
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # 标记异常值：距离超过阈值的点
    outlier_mask = np.zeros(len(positions), dtype=bool)
    outlier_mask[1:] = distances > max_jump_distance  # 第二帧开始，如果与前帧距离过大，标记为异常
    
    return outlier_mask


def detect_consecutive_outliers(positions, max_jump_distance=200):
    """
    检测连续超过阈值的帧
    
    逻辑：
    - 如果连续2帧都超过最大距离，就都删除
    - 如果连续3帧都超过，就3帧都删除
    - 以此类推
    
    Args:
        positions: (n_frames, 2) 位置数组，按frame排序
        max_jump_distance: 最大允许跳跃距离
        
    Returns:
        array: 布尔数组，True表示需要删除的帧
    """
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=bool)
    
    # 计算相邻帧之间的距离
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # 标记超过阈值的帧
    exceeds_threshold = distances > max_jump_distance
    
    # 找出连续超过阈值的区域
    outlier_mask = np.zeros(len(positions), dtype=bool)
    
    i = 0
    while i < len(exceeds_threshold):
        if exceeds_threshold[i]:
            # 找到连续超过阈值的起始位置
            start_idx = i
            # 找到连续超过阈值的结束位置
            while i < len(exceeds_threshold) and exceeds_threshold[i]:
                i += 1
            end_idx = i
            
            # 标记连续超过阈值的所有帧（包括起始帧和结束帧）
            # 如果连续2帧超过，删除2帧；连续3帧超过，删除3帧
            outlier_mask[start_idx:end_idx+1] = True  # end_idx+1因为要包括结束帧本身
        else:
            i += 1
    
    return outlier_mask


def detect_outliers_frame_gap_scaled(positions, frame_indices, max_jump_per_frame=100):
    """
    按帧间隔缩放的异常值检测：1 帧内跳动 > max_jump_per_frame，2 帧 > 2*max_jump_per_frame，以此类推。
    
    对每一对“上一有效点 → 当前点”，若间隔为 k 帧且距离 > max_jump_per_frame * k，则标记当前点为异常。
    
    Args:
        positions: (n_points, 2) 位置数组，按 frame 排序
        frame_indices: (n_points,) 对应的帧号
        max_jump_per_frame: 每帧允许的最大位移（像素），默认 100
        
    Returns:
        array: 布尔数组，True 表示需要删除的点
    """
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=bool)
    
    outlier_mask = np.zeros(len(positions), dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=int)
    
    for i in range(1, len(positions)):
        gap = frame_indices[i] - frame_indices[i - 1]
        if gap <= 0:
            continue
        dist = np.linalg.norm(positions[i] - positions[i - 1])
        threshold = max_jump_per_frame * gap
        if dist > threshold:
            outlier_mask[i] = True
    
    return outlier_mask


def remove_outliers_from_trajectory(
    df, fish_id, max_jump_distance=200,
    use_frame_gap_scaling=False, max_jump_per_frame=100,
):
    """
    从轨迹中移除异常值。
    
    逻辑（二选一）：
    - use_frame_gap_scaling=False：连续超过 max_jump_distance 的帧整段删除（原逻辑）。
    - use_frame_gap_scaling=True：按帧间隔缩放，1 帧内 > max_jump_per_frame、2 帧 > 2*max_jump_per_frame 等。
    
    Args:
        df: DataFrame with columns ['frame', 'fish_id', 'x', 'y']
        fish_id: 要处理的鱼ID
        max_jump_distance: 最大允许跳跃距离（仅当 use_frame_gap_scaling=False 时使用）
        use_frame_gap_scaling: 是否按帧间隔缩放阈值（1帧>100, 2帧>200...）
        max_jump_per_frame: 每帧允许最大位移（仅当 use_frame_gap_scaling=True 时使用）
        
    Returns:
        DataFrame: 移除异常值后的DataFrame（异常值设为NaN）
    """
    fish_data = df[df['fish_id'] == fish_id].copy().sort_values('frame')
    
    if len(fish_data) < 2:
        return df
    
    positions = fish_data[['x', 'y']].values
    frames = fish_data['frame'].values
    
    if use_frame_gap_scaling:
        # 只对有效点做检测；先取有效行再算 mask，再映射回原索引
        valid = ~(np.isnan(positions[:, 0]) | np.isnan(positions[:, 1]))
        if np.sum(valid) < 2:
            return df
        pos_valid = positions[valid]
        frame_valid = frames[valid]
        mask_valid = detect_outliers_frame_gap_scaled(
            pos_valid, frame_valid, max_jump_per_frame=max_jump_per_frame
        )
        outlier_mask = np.zeros(len(fish_data), dtype=bool)
        outlier_mask[valid] = mask_valid
    else:
        outlier_mask = detect_consecutive_outliers(
            positions, max_jump_distance=max_jump_distance
        )
    
    result_df = df.copy()
    outlier_indices = fish_data.index[outlier_mask]
    result_df.loc[outlier_indices, ['x', 'y']] = np.nan
    
    return result_df


def check_remaining_outliers(
    df, fish_id, max_jump_distance=200,
    use_frame_gap_scaling=False, max_jump_per_frame=100,
):
    """
    检查是否还有异常值（规则需与 remove_outliers_from_trajectory 一致）。
    """
    fish_data = df[df['fish_id'] == fish_id].copy().sort_values('frame')
    
    if len(fish_data) < 2:
        return False
    
    valid_data = fish_data.dropna(subset=['x', 'y'])
    if len(valid_data) < 2:
        return False
    
    positions = valid_data[['x', 'y']].values
    frames = valid_data['frame'].values
    
    if use_frame_gap_scaling:
        for i in range(1, len(positions)):
            gap = int(frames[i] - frames[i - 1])
            if gap <= 0:
                continue
            dist = np.linalg.norm(positions[i] - positions[i - 1])
            if dist > max_jump_per_frame * gap:
                return True
        return False
    
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.any(distances > max_jump_distance)


def fill_missing_values_trajectory_simple(df, fish_id, method='linear'):
    """
    简单的缺失值填充（用于迭代处理中）
    
    Args:
        df: DataFrame with columns ['frame', 'fish_id', 'x', 'y']
        fish_id: 要处理的鱼ID
        method: 插值方法
        
    Returns:
        DataFrame: 填充后的DataFrame
    """
    fish_data = df[df['fish_id'] == fish_id].copy().sort_values('frame')
    
    if len(fish_data) == 0:
        return df
    
    result_df = df.copy()
    
    # 对x坐标插值
    x_values = fish_data['x'].values
    if np.any(~np.isnan(x_values)):
        df_x = pd.DataFrame({'value': x_values})
        if method == 'linear':
            interp_x = df_x.interpolate(method='linear', limit_direction='both')
        else:
            interp_x = df_x.interpolate(method='cubic', limit_direction='both')
        result_df.loc[fish_data.index, 'x'] = interp_x['value'].values
    
    # 对y坐标插值
    y_values = fish_data['y'].values
    if np.any(~np.isnan(y_values)):
        df_y = pd.DataFrame({'value': y_values})
        if method == 'linear':
            interp_y = df_y.interpolate(method='linear', limit_direction='both')
        else:
            interp_y = df_y.interpolate(method='cubic', limit_direction='both')
        result_df.loc[fish_data.index, 'y'] = interp_y['value'].values
    
    return result_df


def process_all_fish_outliers_iterative(
    df_dict, max_jump_distance=200, max_iterations=10,
    use_frame_gap_scaling=False, max_jump_per_frame=100,
):
    """
    迭代处理所有鱼的异常值，直到没有异常值为止。
    支持按帧间隔缩放：use_frame_gap_scaling=True 时，1帧>max_jump_per_frame，2帧>2*max_jump_per_frame...
    """
    processed_dict = {}
    for player, df in df_dict.items():
        processed_df = df.copy()
        fish_ids = [fid for fid in df['fish_id'].unique() if fid != 'unassigned']
        for fish_id in fish_ids:
            iteration = 0
            has_outliers = True
            while has_outliers and iteration < max_iterations:
                iteration += 1
                processed_df = remove_outliers_from_trajectory(
                    processed_df, fish_id,
                    max_jump_distance=max_jump_distance,
                    use_frame_gap_scaling=use_frame_gap_scaling,
                    max_jump_per_frame=max_jump_per_frame,
                )
                processed_df = fill_missing_values_trajectory_simple(
                    processed_df, fish_id, method='linear'
                )
                has_outliers = check_remaining_outliers(
                    processed_df, fish_id,
                    max_jump_distance=max_jump_distance,
                    use_frame_gap_scaling=use_frame_gap_scaling,
                    max_jump_per_frame=max_jump_per_frame,
                )
                if has_outliers:
                    print(f"  {player} {fish_id}: 第{iteration}次迭代后仍有异常值，继续处理...")
                elif iteration > 1:
                    print(f"  {player} {fish_id}: 经过{iteration}次迭代，所有异常值已处理")
            if iteration >= max_iterations and has_outliers:
                print(f"  警告: {player} {fish_id} 达到最大迭代次数({max_iterations})，仍有异常值")
        processed_dict[player] = processed_df
    return processed_dict


def process_all_fish_outliers(
    df_dict, max_jump_distance=200, iterative=True, max_iterations=10,
    use_frame_gap_scaling=False, max_jump_per_frame=100,
):
    """
    处理所有鱼的异常值。
    use_frame_gap_scaling=True 时采用「1帧>100px, 2帧>200px...」的缩放规则。
    """
    if iterative:
        return process_all_fish_outliers_iterative(
            df_dict, max_jump_distance, max_iterations,
            use_frame_gap_scaling=use_frame_gap_scaling,
            max_jump_per_frame=max_jump_per_frame,
        )
    processed_dict = {}
    for player, df in df_dict.items():
        processed_df = df.copy()
        fish_ids = [fid for fid in df['fish_id'].unique() if fid != 'unassigned']
        for fish_id in fish_ids:
            processed_df = remove_outliers_from_trajectory(
                processed_df, fish_id,
                max_jump_distance=max_jump_distance,
                use_frame_gap_scaling=use_frame_gap_scaling,
                max_jump_per_frame=max_jump_per_frame,
            )
        processed_dict[player] = processed_df
    return processed_dict
