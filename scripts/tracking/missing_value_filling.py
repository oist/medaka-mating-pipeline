"""
缺失值填充与轨迹平滑
====================

设计原则
--------
- 本模块只负责「在 x,y 为 NaN 的地方做插值 + 可选平滑」，以及「补齐所有帧」。
- 异常点的检测与置 NaN 由 `outlier_detection.py` 完成，这样 2D 清洗的职责更清晰。

主要功能
--------
- 从文件名推断时间范围，生成完整帧序列；
- 确保每一帧每条鱼都有一行记录（缺失处先用 NaN 占位）；
- 对每条轨迹按时间插值补洞，可选 Savitzky-Golay 平滑。
"""

from __future__ import annotations

import re
from typing import Dict, Hashable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_time_from_filename(filename: str) -> tuple[float | None, float | None]:
    """
    从文件名中提取时间范围（秒）。

    例如:
        top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv
    提取:
        (1905.506, 1913.422)
    """
    basename = filename.split("\\")[-1] if "\\" in filename else filename.split("/")[-1]
    pattern = r"(\\d+\\.\\d+)-(\\d+\\.\\d+)"
    match = re.search(pattern, basename)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    return None, None


def generate_complete_frame_sequence(
    start_time: float | None,
    end_time: float | None,
    fps: int = 166,
) -> np.ndarray | None:
    """
    根据起止时间和帧率生成完整帧序列（从 0 开始）。
    """
    if start_time is None or end_time is None:
        return None
    duration = max(0.0, float(end_time) - float(start_time))
    total_frames = int(duration * fps)
    return np.arange(0, total_frames, dtype=int)


def _interpolate_1d_timeseries(
    data: np.ndarray,
    *,
    method: str = "linear",
    limit: int | None = None,
) -> np.ndarray:
    """
    对 1D 时间序列做插值，使用 pandas.interpolate 封装。

    Args:
        data: 1D 数组，允许包含 NaN。
        method: 'linear' 或 'cubic'（由 pandas 提供）。
        limit: 最大连续 NaN 长度；超过则不填（None 表示不限制）。
    """
    if data.size == 0:
        return data

    s = pd.Series(data.copy())
    interp = s.interpolate(method=method, limit=limit, limit_direction="both")
    return interp.to_numpy()


def _smooth_series_inplace(
    values: np.ndarray,
    *,
    smooth: bool,
    smooth_window: int,
    smooth_order: int,
) -> np.ndarray:
    """
    仅在需要且数据长度足够时，对非 NaN 段应用 Savitzky-Golay 平滑。
    """
    if not smooth:
        return values

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < smooth_window:
        return values

    try:
        smoothed = values.copy()
        smoothed[valid_mask] = savgol_filter(
            smoothed[valid_mask],
            smooth_window,
            smooth_order,
        )
        return smoothed
    except Exception:
        return values


def fill_missing_values_trajectory(
    df: pd.DataFrame,
    entity_col: str,
    entity_id: Hashable,
    *,
    method: str = "linear",
    interp_limit: int | None = None,
    smooth: bool = True,
    smooth_window: int = 9,
    smooth_order: int = 2,
) -> pd.DataFrame:
    """
    对单个实体（如某一条鱼）的轨迹做插值补洞 + 可选平滑。

    仅对 `x` 和 `y` 列做处理，其它列保持不变。
    """
    mask_entity = df[entity_col] == entity_id
    entity_df = df.loc[mask_entity].copy().sort_values("frame")
    if entity_df.empty:
        return df

    result_df = df.copy()

    # x 方向
    xs = entity_df["x"].to_numpy(dtype=float)
    if np.any(~np.isnan(xs)):
        xs_interp = _interpolate_1d_timeseries(xs, method=method, limit=interp_limit)
        xs_interp = _smooth_series_inplace(
            xs_interp,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
        )
        result_df.loc[entity_df.index, "x"] = xs_interp

    # y 方向
    ys = entity_df["y"].to_numpy(dtype=float)
    if np.any(~np.isnan(ys)):
        ys_interp = _interpolate_1d_timeseries(ys, method=method, limit=interp_limit)
        ys_interp = _smooth_series_inplace(
            ys_interp,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
        )
        result_df.loc[entity_df.index, "y"] = ys_interp

    return result_df


def _ensure_complete_frames_for_player(
    df: pd.DataFrame,
    *,
    file_path: str | None,
    fps: int,
    entity_col: str = "fish_id",
) -> pd.DataFrame:
    """
    为单个视角补齐所有帧和实体行：
    - 根据文件名时间范围生成完整帧号；
    - 对每个实体在每一帧都保留一行（缺失处 x,y,score=NaN）。
    """
    if file_path is None:
        return df

    start_time, end_time = extract_time_from_filename(file_path)
    frames = generate_complete_frame_sequence(start_time, end_time, fps=fps)
    if frames is None:
        return df

    # 已有的实体集合（排除 'unassigned'）
    present = [e for e in df[entity_col].unique() if e != "unassigned"]
    # 至少保证包含 a,b,c 三个 label，顺序优先 a,b,c
    base_labels = ["a", "b", "c"]
    entities: list[Hashable] = [e for e in base_labels if e in present]
    entities += [e for e in present if e not in entities]
    for lbl in base_labels:
        if lbl not in entities:
            entities.append(lbl)

    rows = []
    for frame in frames:
        for ent in entities:
            sub = df[(df["frame"] == frame) & (df[entity_col] == ent)]
            if not sub.empty:
                rows.append(sub.iloc[0].to_dict())
            else:
                row: dict = {
                    "frame": int(frame),
                    entity_col: ent,
                    "x": np.nan,
                    "y": np.nan,
                    "score": np.nan,
                }
                if "track" in df.columns:
                    row["track"] = np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def ensure_complete_frames(
    df_dict: Dict[str, pd.DataFrame],
    file_set_info: Dict[str, str],
    fps: int = 166,
) -> Dict[str, pd.DataFrame]:
    """
    为每个视角补齐完整帧序列和实体行。

    Args:
        df_dict: {player: DataFrame}
        file_set_info: 包含 player1_file / player2_file / player3_file 等路径。
        fps: 帧率。
    """
    out: Dict[str, pd.DataFrame] = {}
    for player, df in df_dict.items():
        file_key = f"{player}_file"
        file_path = file_set_info.get(file_key)
        out[player] = _ensure_complete_frames_for_player(
            df,
            file_path=file_path,
            fps=fps,
            entity_col="fish_id",
        )
    return out


def process_all_fish_missing_values(
    df_dict: Dict[str, pd.DataFrame],
    file_set_info: Dict[str, str] | None = None,
    *,
    fps: int = 166,
    method: str = "linear",
    interp_limit: int | None = None,
    smooth: bool = True,
    smooth_window: int = 9,
    smooth_order: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    对所有视角、所有鱼的轨迹做缺失值填充 + 可选平滑。

    一般调用顺序应为：
        1. outlier_detection.process_all_fish_outliers → 标记异常点为 NaN；
        2. process_all_fish_missing_values → 插值补洞 + 平滑。
    """
    # 先补齐帧序列（可选）
    if file_set_info is not None:
        df_dict = ensure_complete_frames(df_dict, file_set_info=file_set_info, fps=fps)

    processed: Dict[str, pd.DataFrame] = {}
    for player, df in df_dict.items():
        out_df = df.copy()
        if "fish_id" not in out_df.columns:
            processed[player] = out_df
            continue

        fish_ids = [fid for fid in out_df["fish_id"].unique() if fid != "unassigned"]
        for fid in fish_ids:
            out_df = fill_missing_values_trajectory(
                out_df,
                entity_col="fish_id",
                entity_id=fid,
                method=method,
                interp_limit=interp_limit,
                smooth=smooth,
                smooth_window=smooth_window,
                smooth_order=smooth_order,
            )
        processed[player] = out_df

    return processed


__all__ = [
    "extract_time_from_filename",
    "generate_complete_frame_sequence",
    "fill_missing_values_trajectory",
    "ensure_complete_frames",
    "process_all_fish_missing_values",
]

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
