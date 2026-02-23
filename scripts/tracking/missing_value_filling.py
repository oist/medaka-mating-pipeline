"""
缺失值填充算法
逻辑：
1. 根据文件名中的时间信息（秒）和fps，生成完整的帧序列
2. 每一帧都要有3个个体的坐标数据
3. 使用插值方法科学地补齐缺失数据（用上下两帧的数据）
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import re


def extract_time_from_filename(filename):
    """
    从文件名中提取时间范围（秒）
    
    例如: top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv
    提取: (1905.506, 1913.422)
    
    Returns:
        tuple: (start_time, end_time) in seconds
    """
    basename = filename.split('\\')[-1] if '\\' in filename else filename.split('/')[-1]
    # 查找时间模式：数字.数字-数字.数字
    pattern = r'(\d+\.\d+)-(\d+\.\d+)'
    match = re.search(pattern, basename)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    return None, None


def generate_complete_frame_sequence(start_time, end_time, fps=166):
    """
    生成完整的帧序列
    
    Args:
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        fps: 帧率（默认166）
        
    Returns:
        array: 完整的帧号数组
    """
    if start_time is None or end_time is None:
        return None
    
    # 计算总帧数
    duration = end_time - start_time
    total_frames = int(duration * fps)
    
    # 生成帧序列（从0开始）
    frames = np.arange(0, total_frames)
    return frames


def interpolate_1d_timeseries(data, method='linear', limit=None):
    """
    对1D时间序列进行插值
    
    逻辑：用上下两帧的数据进行插值补齐
    
    Args:
        data: 1D数组，可能包含NaN
        method: 插值方法 ('linear', 'cubic')
        limit: 最大连续NaN数量，超过此值不插值（None表示不限制）
        
    Returns:
        array: 插值后的数组
    """
    if len(data) == 0:
        return data
    
    df = pd.DataFrame({'value': data})
    
    if method == 'linear':
        interp_df = df.interpolate(method='linear', limit=limit, limit_direction='both')
    elif method == 'cubic':
        interp_df = df.interpolate(method='cubic', limit=limit, limit_direction='both')
    else:
        raise ValueError(f"未知的插值方法: {method}")
    
    return interp_df['value'].values


def fill_missing_values_trajectory(df, fish_id, method='linear', interp_limit=None, smooth=True, smooth_window=9, smooth_order=2):
    """
    填充轨迹中的缺失值
    
    逻辑：用上下两帧的数据进行插值补齐
    
    Args:
        df: DataFrame with columns ['frame', 'fish_id', 'x', 'y']
        fish_id: 要处理的鱼ID
        method: 插值方法
        interp_limit: 最大连续缺失值数量（None表示不限制）
        smooth: 是否使用Savitzky-Golay滤波平滑
        smooth_window: 平滑窗口大小
        smooth_order: 平滑多项式阶数
        
    Returns:
        DataFrame: 填充后的DataFrame
    """
    fish_data = df[df['fish_id'] == fish_id].copy().sort_values('frame')
    
    if len(fish_data) == 0:
        return df
    
    result_df = df.copy()
    
    # 对x坐标插值
    x_values = fish_data['x'].values
    if np.any(~np.isnan(x_values)):  # 如果有有效数据
        x_interp = interpolate_1d_timeseries(x_values, method=method, limit=interp_limit)
        
        if smooth and len(x_interp[~np.isnan(x_interp)]) >= smooth_window:
            try:
                # 只对非NaN值进行平滑
                valid_mask = ~np.isnan(x_interp)
                if valid_mask.sum() >= smooth_window:
                    x_smooth = x_interp.copy()
                    x_smooth[valid_mask] = savgol_filter(x_interp[valid_mask], smooth_window, smooth_order)
                    x_interp = x_smooth
            except:
                pass
        
        result_df.loc[fish_data.index, 'x'] = x_interp
    
    # 对y坐标插值
    y_values = fish_data['y'].values
    if np.any(~np.isnan(y_values)):  # 如果有有效数据
        y_interp = interpolate_1d_timeseries(y_values, method=method, limit=interp_limit)
        
        if smooth and len(y_interp[~np.isnan(y_interp)]) >= smooth_window:
            try:
                valid_mask = ~np.isnan(y_interp)
                if valid_mask.sum() >= smooth_window:
                    y_smooth = y_interp.copy()
                    y_smooth[valid_mask] = savgol_filter(y_interp[valid_mask], smooth_window, smooth_order)
                    y_interp = y_smooth
            except:
                pass
        
        result_df.loc[fish_data.index, 'y'] = y_interp
    
    return result_df


def ensure_complete_frames(df_dict, file_set_info, fps=166):
    """
    确保每一帧都存在，且每一帧都有3个个体的坐标数据
    
    逻辑：
    1. 从文件名提取时间范围
    2. 根据fps生成完整的帧序列
    3. 为缺失的帧创建数据（使用插值）
    
    Args:
        df_dict: {player: DataFrame}
        file_set_info: dict包含player1_file等信息
        fps: 帧率（默认166）
        
    Returns:
        dict: {player: 完整的DataFrame，每一帧都有3个个体的数据}
    """
    # 从第一个文件提取时间信息
    first_file = file_set_info.get('player1_file', '')
    start_time, end_time = extract_time_from_filename(first_file)
    
    if start_time is None or end_time is None:
        print("警告: 无法从文件名提取时间信息，跳过帧补齐")
        return df_dict
    
    # 生成完整的帧序列
    complete_frames = generate_complete_frame_sequence(start_time, end_time, fps)
    if complete_frames is None:
        return df_dict
    
    complete_dfs = {}
    
    for player, df in df_dict.items():
        # 获取该视角的所有 fish_id，并保证至少包含 a, b, c（缺的用 NaN 行补全，供 3D 用）
        present = [fid for fid in df['fish_id'].unique() if fid != 'unassigned']
        fish_ids = [f for f in ['a', 'b', 'c'] if f in present]
        fish_ids += sorted([f for f in present if f not in fish_ids])
        for i in range(3):
            if ['a', 'b', 'c'][i] not in fish_ids:
                fish_ids.append(['a', 'b', 'c'][i])
        
        # 为每个fish_id创建完整的帧序列
        complete_rows = []
        
        for frame in complete_frames:
            for fish_id in fish_ids:
                # 查找该帧该fish_id的数据
                frame_data = df[(df['frame'] == frame) & (df['fish_id'] == fish_id)]
                
                if len(frame_data) > 0:
                    # 如果存在数据，使用现有数据
                    row = frame_data.iloc[0].to_dict()
                else:
                    # 如果不存在，创建空行（后续用插值填充）
                    row = {
                        'frame': frame,
                        'fish_id': fish_id,
                        'x': np.nan,
                        'y': np.nan,
                        'score': np.nan
                    }
                    # 保留其他列
                    if 'track' in df.columns:
                        row['track'] = np.nan
                
                complete_rows.append(row)
        
        complete_df = pd.DataFrame(complete_rows)
        
        # 对每个fish_id进行插值，补齐缺失的帧
        for fish_id in fish_ids:
            complete_df = fill_missing_values_trajectory(
                complete_df, fish_id, method='linear', interp_limit=None, smooth=True
            )
        
        complete_dfs[player] = complete_df
    
    return complete_dfs


def process_all_fish_missing_values(df_dict, file_set_info=None, fps=166, method='linear', interp_limit=None, smooth=True, **kwargs):
    """
    处理所有鱼的缺失值
    
    Args:
        df_dict: {player: DataFrame}
        file_set_info: 文件信息（用于提取时间范围）
        fps: 帧率
        method: 插值方法
        interp_limit: 最大连续缺失值数量
        smooth: 是否平滑
        **kwargs: 其他参数
        
    Returns:
        dict: {player: 处理后的DataFrame}
    """
    # 首先确保每一帧都存在
    if file_set_info:
        df_dict = ensure_complete_frames(df_dict, file_set_info, fps=fps)
    
    # 然后对每个fish_id进行插值
    processed_dict = {}
    
    for player, df in df_dict.items():
        processed_df = df.copy()
        
        # 获取所有fish_id
        fish_ids = df['fish_id'].unique()
        fish_ids = [fid for fid in fish_ids if fid != 'unassigned']
        
        for fish_id in fish_ids:
            processed_df = fill_missing_values_trajectory(
                processed_df, fish_id, method=method, 
                interp_limit=interp_limit, smooth=smooth, **kwargs
            )
        
        processed_dict[player] = processed_df
    
    return processed_dict
