"""
Step 6: 3D 轨迹异常值去除与插值补全
==================================

对每条 3D 轨迹（单位 mm）：按帧间隔缩放阈值（一帧 > max_mm_per_frame，两帧 > 2*max_mm_per_frame…）
**循环**检测异常 → 剔除 → 插值 → 再检测，直到无异常或达到最大迭代次数，保证每帧都有 3 个个体。

输入: output_dir 下 {set_key}_a_3d.csv / _ID_1_3d.csv 等（frame, X, Y, Z）
输出: 同路径覆盖或输出到 _3d_cleaned（可配置）
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ID_LABELS = ["ID_1", "ID_2", "ID_3"]
GROUP_LABELS = ["a", "b", "c"]


def _detect_outliers_frame_gap_scaled(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    max_mm_per_frame: float,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
) -> np.ndarray:
    """
    positions (n, 3), frame_indices (n,). 返回 bool 数组，True 表示异常。
    轨迹末尾（最后 tail_fraction 比例帧）使用更宽松阈值：threshold * tail_relax_factor，
    避免整段尾巴被标为异常后插值成「最后一格不动」，导致 ID 在视频末尾完全静止。
    """
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=bool)
    mask = np.zeros(len(positions), dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=int)
    f_min, f_max = int(frame_indices.min()), int(frame_indices.max())
    tail_threshold_frame = f_max - max(1, int((f_max - f_min) * tail_fraction))
    for i in range(1, len(positions)):
        gap = frame_indices[i] - frame_indices[i - 1]
        if gap <= 0:
            continue
        dist = np.linalg.norm(positions[i] - positions[i - 1])
        threshold = max_mm_per_frame * gap
        if frame_indices[i] >= tail_threshold_frame:
            threshold *= tail_relax_factor
        if dist > threshold:
            mask[i] = True
    return mask


def _check_remaining_outliers_3d(
    df: pd.DataFrame,
    max_mm_per_frame: float,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
) -> bool:
    """检查该条 3D 轨迹是否仍存在异常（与 _detect_outliers_frame_gap_scaled 规则一致，含末尾放宽）。"""
    df = df.sort_values("frame")
    pos = df[["X", "Y", "Z"]].values
    frames = df["frame"].values
    valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
    if np.sum(valid) < 2:
        return False
    pos_valid = pos[valid]
    frame_valid = frames[valid]
    f_min, f_max = int(frame_valid.min()), int(frame_valid.max())
    tail_threshold_frame = f_max - max(1, int((f_max - f_min) * tail_fraction))
    for i in range(1, len(pos_valid)):
        gap = int(frame_valid[i] - frame_valid[i - 1])
        if gap <= 0:
            continue
        dist = np.linalg.norm(pos_valid[i] - pos_valid[i - 1])
        threshold = max_mm_per_frame * gap
        if frame_valid[i] >= tail_threshold_frame:
            threshold *= tail_relax_factor
        if dist > threshold:
            return True
    return False


def _interpolate_xyz(df: pd.DataFrame, method: str = "cubic") -> pd.DataFrame:
    """对 frame, X, Y, Z 做插值补全 NaN。method: 'cubic'（三次样条，默认）或 'linear'。"""
    df = df.copy().sort_values("frame")
    for col in ["X", "Y", "Z"]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if np.any(~np.isnan(vals)):
            if method == "cubic" and np.sum(~np.isnan(vals)) >= 4:
                try:
                    from scipy.interpolate import CubicSpline
                    valid = ~np.isnan(vals)
                    x = np.arange(len(vals))[valid]
                    y = vals[valid]
                    cs = CubicSpline(x, y)
                    nan_idx = np.where(np.isnan(vals))[0]
                    vals[nan_idx] = cs(nan_idx)
                    df[col] = vals
                except Exception:
                    df[col] = pd.Series(vals).interpolate(method="linear", limit_direction="both").values
            else:
                df[col] = pd.Series(vals).interpolate(method="linear", limit_direction="both").values
    return df


def _ensure_complete_frames(df: pd.DataFrame, all_frames: np.ndarray) -> pd.DataFrame:
    """保证 df 包含 all_frames 中每一帧；缺失行用 NaN 占位。"""
    if all_frames is None or len(all_frames) == 0:
        return df
    df = df.sort_values("frame")
    existing = set(df["frame"].astype(int))
    missing = [f for f in all_frames if int(f) not in existing]
    if not missing:
        return df
    rows = []
    for f in missing:
        rows.append({"frame": int(f), "X": np.nan, "Y": np.nan, "Z": np.nan})
    extra = pd.DataFrame(rows)
    return pd.concat([df, extra], ignore_index=True).sort_values("frame").reset_index(drop=True)


def process_set(
    output_dir: Path,
    set_key: str,
    max_mm_per_frame: float = 10.0,
    use_id_labels: bool = True,
    in_place: bool = True,
    max_iterations: int = 10,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
    interp_method: str = "cubic",
) -> List[Path]:
    """
    处理单个 set_key 的 3D CSV：**循环**异常值去除 + 插值，直到无异常或达到 max_iterations。
    规则：一帧间隔 > max_mm_per_frame (mm)，两帧 > 2*max_mm_per_frame，以此类推。
    """
    labels = ID_LABELS if use_id_labels else GROUP_LABELS

    dfs = {}
    all_frames = None
    for g in labels:
        p = output_dir / f"{set_key}_{g}_3d.csv"
        if not p.exists():
            raise FileNotFoundError(f"未找到: {p}")
        dfs[g] = pd.read_csv(p)
        frames = dfs[g]["frame"].astype(int).values
        if all_frames is None:
            all_frames = np.unique(frames)
        else:
            all_frames = np.union1d(all_frames, frames)

    if all_frames is not None:
        all_frames = np.sort(all_frames)

    out_suffix = "" if in_place else "_cleaned"
    written = []

    for g in labels:
        df = dfs[g].copy()
        df = _ensure_complete_frames(df, all_frames)
        df = df.sort_values("frame").reset_index(drop=True)

        iteration = 0
        has_outliers = True
        while has_outliers and iteration < max_iterations:
            iteration += 1
            pos = df[["X", "Y", "Z"]].values
            frames = df["frame"].values
            valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
            if np.sum(valid) >= 2:
                pos_valid = pos[valid]
                frame_valid = frames[valid]
                mask_valid = _detect_outliers_frame_gap_scaled(
                    pos_valid, frame_valid, max_mm_per_frame,
                    tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
                )
                outlier_in_full = np.zeros(len(df), dtype=bool)
                outlier_in_full[valid] = mask_valid
                df.loc[outlier_in_full, ["X", "Y", "Z"]] = np.nan
            df = _interpolate_xyz(df, method=interp_method)
            has_outliers = _check_remaining_outliers_3d(
                df, max_mm_per_frame,
                tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
            )
            if has_outliers:
                pass  # 下一轮继续
            elif iteration > 1:
                pass  # 多轮后干净了，静默结束即可

        if iteration >= max_iterations and has_outliers:
            print(f"    警告: {set_key} {g} 达到最大迭代次数 ({max_iterations})，仍有 3D 异常点")

        out_path = output_dir / f"{set_key}_{g}_3d{out_suffix}.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def process_set_single_csv(
    output_dir: Path,
    set_key: str,
    max_mm_per_frame: float = 10.0,
    in_place: bool = True,
    max_iterations: int = 10,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
    interp_method: str = "cubic",
) -> List[Path]:
    """
    读单 CSV {set_key}_3d.csv（列 frame, fish_id, X, Y, Z），按 fish_id 拆开做异常值+插值，再合并写回单 CSV。
    """
    single_path = output_dir / f"{set_key}_3d.csv"
    if not single_path.exists():
        raise FileNotFoundError(f"未找到: {single_path}")
    full_df = pd.read_csv(single_path)
    if "fish_id" not in full_df.columns:
        raise ValueError(f"{single_path} 缺少列 fish_id")
    labels = full_df["fish_id"].unique().tolist()
    dfs = {g: full_df[full_df["fish_id"] == g][["frame", "X", "Y", "Z"]].copy() for g in labels}
    all_frames = np.unique(np.concatenate([df["frame"].astype(int).values for df in dfs.values()]))
    all_frames = np.sort(all_frames)

    out_suffix = "" if in_place else "_cleaned"
    for g in labels:
        df = dfs[g].copy()
        df = _ensure_complete_frames(df, all_frames)
        df = df.sort_values("frame").reset_index(drop=True)
        iteration = 0
        has_outliers = True
        while has_outliers and iteration < max_iterations:
            iteration += 1
            pos = df[["X", "Y", "Z"]].values
            frames = df["frame"].values
            valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
            if np.sum(valid) >= 2:
                pos_valid = pos[valid]
                frame_valid = frames[valid]
                mask_valid = _detect_outliers_frame_gap_scaled(
                    pos_valid, frame_valid, max_mm_per_frame,
                    tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
                )
                outlier_in_full = np.zeros(len(df), dtype=bool)
                outlier_in_full[valid] = mask_valid
                df.loc[outlier_in_full, ["X", "Y", "Z"]] = np.nan
            df = _interpolate_xyz(df, method=interp_method)
            has_outliers = _check_remaining_outliers_3d(
                df, max_mm_per_frame,
                tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
            )
        if iteration >= max_iterations and has_outliers:
            print(f"    警告: {set_key} {g} 达到最大迭代次数 ({max_iterations})，仍有 3D 异常点")
        dfs[g] = df

    out_path = output_dir / f"{set_key}_3d{out_suffix}.csv"
    rows = []
    for fid, df in dfs.items():
        for _, r in df.iterrows():
            rows.append({"frame": int(r["frame"]), "fish_id": fid, "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"])})
    pd.DataFrame(rows).sort_values(["frame", "fish_id"]).to_csv(out_path, index=False)
    return [out_path]


def run(
    output_dir: str,
    set_keys: List[str] | None = None,
    max_mm_per_frame: float = 10.0,
    use_id_labels: bool = True,
    in_place: bool = True,
    max_iterations: int = 10,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
    interp_method: str = "cubic",
) -> List[Path]:
    """
    对 output_dir 下多组 3D CSV 做**循环**异常值去除与插值补全。
    set_keys 为 None 时：优先扫描单 CSV {set_key}_3d.csv，否则扫描 *_a_3d.csv / *_ID_1_3d.csv。
    """
    out_dir = Path(output_dir)
    if set_keys is None:
        single_files = list(out_dir.glob("*_3d.csv"))
        single_files = [
            f for f in single_files
            if f.stem.endswith("_3d") and "_3d_cleaned" not in f.stem and "_3d_interp" not in f.stem
            and not any(x in f.stem for x in ["_a_3d", "_b_3d", "_c_3d", "_ID_1_3d", "_ID_2_3d", "_ID_3_3d"])
        ]
        a_files = list(out_dir.glob("*_a_3d.csv"))
        id_files = list(out_dir.glob("*_ID_1_3d.csv"))
        if single_files:
            set_keys = sorted({f.stem.replace("_3d", "") for f in single_files})
            use_single_csv = True
        elif id_files:
            set_keys = sorted({f.stem.replace("_ID_1_3d", "") for f in id_files})
            use_id_labels = True
            use_single_csv = False
        else:
            set_keys = sorted({f.stem.replace("_a_3d", "") for f in a_files})
            use_id_labels = False
            use_single_csv = False
    else:
        use_single_csv = None

    if not set_keys:
        print("未找到 3D CSV，跳过 step6。")
        return []

    all_written = []
    for set_key in set_keys:
        try:
            if use_single_csv is True or (use_single_csv is None and (out_dir / f"{set_key}_3d.csv").exists()):
                written = process_set_single_csv(
                    out_dir, set_key,
                    max_mm_per_frame=max_mm_per_frame,
                    in_place=in_place,
                    max_iterations=max_iterations,
                    tail_fraction=tail_fraction,
                    tail_relax_factor=tail_relax_factor,
                    interp_method=interp_method,
                )
            else:
                written = process_set(
                    out_dir, set_key,
                    max_mm_per_frame=max_mm_per_frame,
                    use_id_labels=use_id_labels,
                    in_place=in_place,
                    max_iterations=max_iterations,
                    tail_fraction=tail_fraction,
                    tail_relax_factor=tail_relax_factor,
                    interp_method=interp_method,
                )
            all_written.extend(written)
            print(f"  已处理 {set_key}")
        except FileNotFoundError as e:
            print(f"  跳过 {set_key}: {e}")
    return all_written


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 6: 3D 异常值去除与插值补全（循环至无异常）")
    ap.add_argument("--output-dir", required=True, help="3D CSV 目录")
    ap.add_argument("--set-keys", nargs="*", default=None)
    ap.add_argument("--max-mm-per-frame", type=float, default=10.0, help="每帧允许最大位移 (mm)，k 帧间隔阈值为 10*k")
    ap.add_argument("--max-iterations", type=int, default=10, help="最大迭代次数")
    ap.add_argument("--no-in-place", action="store_true", help="输出 _3d_cleaned.csv 不覆盖")
    ap.add_argument("--tail-fraction", type=float, default=0.15, help="轨迹末尾比例，该段内放宽阈值，默认 0.15")
    ap.add_argument("--tail-relax-factor", type=float, default=2.0, help="末尾段阈值放宽倍数，默认 2.0")
    ap.add_argument("--interp-method", choices=("cubic", "linear"), default="cubic", help="补全 NaN 的插值方法，默认 cubic")
    args = ap.parse_args()
    run(
        args.output_dir, args.set_keys, args.max_mm_per_frame,
        in_place=not args.no_in_place,
        max_iterations=args.max_iterations,
        tail_fraction=args.tail_fraction,
        tail_relax_factor=args.tail_relax_factor,
        interp_method=args.interp_method,
    )


if __name__ == "__main__":
    main()
