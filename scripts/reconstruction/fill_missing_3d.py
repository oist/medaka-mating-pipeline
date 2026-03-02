"""
3D 轨迹缺失帧补全（保证每帧都有 3 个个体坐标）
===============================================

在 Step 4 跨帧追踪之后、导出 CSV 之前或之后使用：
- 将每条轨迹扩展到「全部帧」范围，缺失帧先填 NaN；
- 用 **三次样条** 沿时间轴插值补全 NaN（不用线性插值，曲线更自然）；
- 有效点过少（<4）时退化为「最近有效帧」复制，保证稳定。

保证最终 CSV 每帧都有 3 个个体，且跨帧 ID 已由 Step 4 确定。
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

GROUP_LABELS = ["a", "b", "c"]


def ensure_all_frames(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
) -> Dict[str, pd.DataFrame]:
    """
    保证每个 df 都包含 all_frames 中每一帧；缺失行用 NaN 占位。
    """
    out: Dict[str, pd.DataFrame] = {}
    all_frames = sorted(all_frames)
    for label in GROUP_LABELS:
        df = dfs_3d.get(label)
        if df is None or df.empty:
            out[label] = pd.DataFrame({"frame": all_frames, "X": np.nan, "Y": np.nan, "Z": np.nan})
            continue
        df = df[["frame", "X", "Y", "Z"]].copy().drop_duplicates(subset=["frame"])
        existing = df.set_index("frame")
        full_index = pd.Index(all_frames, name="frame")
        reindexed = existing.reindex(full_index).reset_index()
        out[label] = reindexed
    return out


def _fill_xyz_cubic_spline_one_series(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    对一维序列（frames 为 x，values 为 y）用三次样条补全 NaN。
    frames 与 values 长度相同；NaN 位置由样条在对应 frame 处求值填充。
    若有效点 < 4，则用最近有效帧的值填充每个 NaN。
    """
    valid = ~np.isnan(values) & ~np.isnan(frames)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return values
    if n_valid == 1:
        return np.full_like(values, values[valid][0])
    frames_valid = np.asarray(frames[valid], dtype=float)
    values_valid = np.asarray(values[valid], dtype=float)
    order = np.argsort(frames_valid)
    u = frames_valid[order]
    v = values_valid[order]
    # 去重 frame（保留首次），保证严格单调供 CubicSpline
    keep = np.concatenate([[True], u[1:] != u[:-1]])
    u, v = u[keep], v[keep]
    if len(u) < 2:
        return values
    nan_mask = np.isnan(values)
    if not np.any(nan_mask):
        return values
    frames_nan = frames[nan_mask]
    if len(u) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(u, v)
            v_nan = np.where(
                frames_nan < u.min(),
                v[0],
                np.where(frames_nan > u.max(), v[-1], cs(frames_nan)),
            )
            out = values.copy()
            out[nan_mask] = v_nan
            return out
        except Exception:
            pass
    # 有效点 < 4 或样条失败：用最近有效帧的值
    valid_idx = np.where(valid)[0]
    for i in np.where(nan_mask)[0]:
        dist = np.abs(frames[valid_idx] - frames[i])
        j = valid_idx[np.argmin(dist)]
        values[i] = values[j]
    return values


def fill_xyz_cubic_spline(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 的 X, Y, Z 列按「帧」用三次样条补全 NaN。
    df 需含 frame, X, Y, Z；frame 需单调（已 sort）。
    """
    df = df.sort_values("frame").reset_index(drop=True)
    frames = df["frame"].values.astype(float)
    for col in ["X", "Y", "Z"]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float).copy()
        df[col] = _fill_xyz_cubic_spline_one_series(frames, vals)
    return df


def ensure_all_frames_and_fill_spline(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
) -> Dict[str, pd.DataFrame]:
    """
    先扩展为全部帧（缺失帧 NaN），再对每条轨迹用三次样条补全 X,Y,Z。
    保证返回的每个 df 行数 = len(all_frames)，且无 NaN。
    """
    expanded = ensure_all_frames(dfs_3d, all_frames)
    out: Dict[str, pd.DataFrame] = {}
    for label, df in expanded.items():
        df = fill_xyz_cubic_spline(df.copy())
        # 若这一条轨迹在所有帧上始终没有任何有效点（X/Y/Z 全是 NaN），
        # 说明整个片段里根本没有这个个体的 3D 结果；不要强行造一条全 0 轨迹，直接跳过。
        if df[["X", "Y", "Z"]].isna().all(axis=None):
            continue
        # 若仅部分帧存在 NaN，用前向/后向填充；极端情况下仍有 NaN 再用 0 填满
        for col in ["X", "Y", "Z"]:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0.0)
        out[label] = df
    return out
