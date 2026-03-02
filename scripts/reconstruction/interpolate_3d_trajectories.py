"""
3D 轨迹缺失帧插值
=================

对 a/b/c 三条 3D 轨迹（每条约 frame, X, Y, Z），在缺失帧上做 3D 空间线性插值，
输出插值后的 DataFrame 或写入 *_3d_interp.csv。
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

GROUP_LABELS = ["a", "b", "c"]


def interpolate_3d_trajectories(
    dfs: Dict[str, pd.DataFrame],
    frame_range: Tuple[int, int] | None = None,
    method: str = "linear",
) -> Dict[str, pd.DataFrame]:
    """
    对每条 3D 轨迹（列 frame, X, Y, Z）在缺失帧上插值。

    Args:
        dfs: {"a": df_a, "b": df_b, "c": df_c}，每个 df 含 frame, X, Y, Z
        frame_range: 若给定 (min_f, max_f)，则在该范围内插值；否则用所有 df 的 frame 并集
        method: 插值方法，如 "linear"（默认）、"cubic" 等（见 pandas interpolate）

    Returns:
        插值后的 {label: df}，每个 df 在完整帧序列上有 X,Y,Z（缺失首尾用外推或留 NaN 取决于 method）
    """
    if not dfs:
        return {}

    all_frames: set = set()
    for df in dfs.values():
        if df is not None and not df.empty and "frame" in df.columns:
            all_frames.update(df["frame"].astype(int).tolist())
    if frame_range is not None:
        low, high = frame_range
        full_frames = pd.Index(range(low, high + 1), name="frame")
    else:
        if not all_frames:
            return {k: v for k, v in dfs.items() if v is not None and not v.empty}
        full_frames = pd.Index(sorted(all_frames), name="frame")

    result: Dict[str, pd.DataFrame] = {}
    for label in GROUP_LABELS:
        df = dfs.get(label)
        if df is None or df.empty:
            continue
        if "frame" not in df.columns or "X" not in df.columns:
            continue
        df = df[["frame", "X", "Y", "Z"]].copy()
        df = df.drop_duplicates(subset=["frame"]).sort_values("frame")
        df = df.set_index("frame")
        df = df.reindex(full_frames)
        df = df.interpolate(method=method, limit_direction="both")
        df = df.reset_index()
        result[label] = df
    return result


def interpolate_and_save(
    dfs: Dict[str, pd.DataFrame],
    set_key: str,
    output_dir: str,
    *,
    frame_range: Tuple[int, int] | None = None,
    method: str = "linear",
    suffix: str = "_3d_interp",
) -> Dict[str, pd.DataFrame]:
    """
    插值并写入 {set_key}_a_3d_interp.csv 等；列 frame, X, Y, Z。
    """
    interp = interpolate_3d_trajectories(dfs, frame_range=frame_range, method=method)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, df in interp.items():
        path = out_dir / f"{set_key}_{label}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"已保存: {path}")
    return interp
