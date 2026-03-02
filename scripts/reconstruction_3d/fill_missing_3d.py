"""
将每条轨迹扩展到全部帧，缺失帧用线性插值补全。
个体 ID 为 ID_1, ID_2, ID_3。
约定：空缺一律用 NaN 表示，不填 0；插值前不把负值 clip 成 0，避免把无效/缺失变成 (0,0,0)。
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# 与 cross_frame_tracking 一致，导出与插值均用 ID_1, ID_2, ID_3
ID_LABELS = ["ID_1", "ID_2", "ID_3"]


def best_by_frame_to_dfs(
    best_by_frame: Dict[int, Dict],
    *,
    shift_to_positive: bool = True,
    ensure_non_negative: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    从 best_by_frame 转为 {label: DataFrame 含 frame, X, Y, Z}。
    label 为 ID_1, ID_2, ID_3（由 cross_frame_tracking 输出）。
    只写入有效点（非 nan/inf）；空缺不写行，留待后续统一插值。
    shift_to_positive: 整体平移使最小坐标为 0（仅用有效点算 min）。
    ensure_non_negative: 若 True 则将 X,Y,Z 与 0 取 max（易把无效变成 0，默认关闭）。
    """
    dfs: Dict[str, pd.DataFrame] = {lb: [] for lb in ID_LABELS}
    for frame, info in sorted(best_by_frame.items()):
        groups = info.get("groups", {})
        for lb in ID_LABELS:
            pt = groups.get(lb)
            if pt is not None and len(pt) == 3 and not (np.isnan(pt).any() or np.isinf(pt).any()):
                dfs[lb].append({"frame": frame, "X": pt[0], "Y": pt[1], "Z": pt[2]})

    out = {}
    for lb in ID_LABELS:
        if dfs[lb]:
            out[lb] = pd.DataFrame(dfs[lb])
        else:
            out[lb] = pd.DataFrame(columns=["frame", "X", "Y", "Z"])

    if shift_to_positive and out:
        arrs = [df[["X", "Y", "Z"]].values for df in out.values() if not df.empty]
        if arrs:
            all_xyz = np.vstack(arrs)
        else:
            all_xyz = np.array([]).reshape(0, 3)
        if all_xyz.size > 0:
            min_vals = np.nanmin(all_xyz, axis=0)
            for lb in out:
                if not out[lb].empty:
                    out[lb] = out[lb].copy()
                    out[lb][["X", "Y", "Z"]] -= min_vals

    if ensure_non_negative and out:
        for lb in out:
            if not out[lb].empty:
                out[lb] = out[lb].copy()
                out[lb][["X", "Y", "Z"]] = out[lb][["X", "Y", "Z"]].clip(lower=0)
    return out  # 默认不 clip，避免把负值/无效变成 0


def ensure_all_frames_and_fill_spline(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
) -> Dict[str, pd.DataFrame]:
    """
    对每条轨迹扩展到 all_frames；缺失帧用普通线性插值，超出观测范围的帧填 NaN。
    """
    if not all_frames:
        return dfs_3d
    all_frames = sorted(all_frames)
    all_frames_arr = np.array(all_frames, dtype=float)
    result = {}
    for lb, df in dfs_3d.items():
        if df.empty:
            result[lb] = pd.DataFrame({
                "frame": all_frames,
                "X": np.nan, "Y": np.nan, "Z": np.nan,
            })
            result[lb] = result[lb].sort_values("frame").reset_index(drop=True)
            continue
        if len(df) < 2:
            result[lb] = df.copy()
            continue
        frames = df["frame"].values.astype(float)
        xyz = df[["X", "Y", "Z"]].values
        _, idx = np.unique(frames, return_index=True)
        frames = frames[idx]
        xyz = xyz[idx]
        sort_idx = np.argsort(frames)
        frames = frames[sort_idx]
        xyz = xyz[sort_idx]
        if len(frames) < 2:
            result[lb] = df.copy()
            continue
        f_min, f_max = float(frames.min()), float(frames.max())
        # 线性插值：仅对 [f_min, f_max] 内插值，之外保持 NaN
        interp_x = np.interp(all_frames_arr, frames, xyz[:, 0])
        interp_y = np.interp(all_frames_arr, frames, xyz[:, 1])
        interp_z = np.interp(all_frames_arr, frames, xyz[:, 2])
        out_of_range = (all_frames_arr < f_min) | (all_frames_arr > f_max)
        interp_x = np.where(out_of_range, np.nan, interp_x)
        interp_y = np.where(out_of_range, np.nan, interp_y)
        interp_z = np.where(out_of_range, np.nan, interp_z)
        out_df = pd.DataFrame({
            "frame": all_frames,
            "X": interp_x, "Y": interp_y, "Z": interp_z,
        })
        # 仅对有限值 clip 到 0，保留 NaN 不变
        for col in ["X", "Y", "Z"]:
            out_df[col] = out_df[col].where(~np.isfinite(out_df[col]), out_df[col].clip(lower=0))
        result[lb] = out_df.sort_values("frame").reset_index(drop=True)
    return result


def dfs_to_single_csv(
    dfs_3d: Dict[str, pd.DataFrame],
    set_key: str,
    output_dir: str,
    *,
    ensure_non_negative: bool = False,
) -> str:
    """将 {ID_1: df, ID_2: df, ID_3: df} 合并为单 CSV：frame, fish_id, X, Y, Z。空缺为 NaN，不填 0。"""
    from pathlib import Path
    rows = []
    for fish_id, df in dfs_3d.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            x, y, z = float(r["X"]), float(r["Y"]), float(r["Z"])
            if ensure_non_negative and np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                x, y, z = max(0, x), max(0, y), max(0, z)
            rows.append({
                "frame": int(r["frame"]),
                "fish_id": fish_id,
                "X": x,
                "Y": y,
                "Z": z,
            })
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_path = Path(output_dir) / f"{set_key}_3d.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        return str(out_path)
    out_df = out_df.sort_values(["frame", "fish_id"]).reset_index(drop=True)
    out_path = Path(output_dir) / f"{set_key}_3d.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return str(out_path)
