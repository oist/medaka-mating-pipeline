"""
从 best 导出 3D 轨迹为 CSV（单文件 long 格式，与 2D processed 一致）
==========================================

支持两种输出：
- 单 CSV：{set_key}_3d.csv，列 frame, fish_id, X, Y, Z（每帧 3 行，与 2D 一致）
- 原三文件：{set_key}_a_3d.csv 等（保留兼容）
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

GROUP_LABELS = ["a", "b", "c"]


def best_by_frame_to_dfs(
    best_by_frame: Dict[int, Dict],
    *,
    shift_to_positive: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    从 best_by_frame 构建 per-label 的 DataFrame（不写文件）。
    与 export_groups_to_csv 内部逻辑一致。
    """
    data_per_group: Dict[str, List[Dict]] = {g: [] for g in GROUP_LABELS}
    for frame in sorted(best_by_frame.keys()):
        info = best_by_frame[frame]
        groups = info.get("groups", {})
        for g in GROUP_LABELS:
            if g in groups:
                Xv, Yv, Zv = groups[g]
                data_per_group[g].append(
                    {"frame": frame, "X": float(Xv), "Y": float(Yv), "Z": float(Zv)}
                )
    if shift_to_positive:
        all_pts = []
        for rows in data_per_group.values():
            for r in rows:
                all_pts.append([r["X"], r["Y"], r["Z"]])
        if all_pts:
            arr = np.array(all_pts)
            mins = arr.min(axis=0)
            offset = np.maximum(0, -mins)
            if np.any(offset > 0):
                for rows in data_per_group.values():
                    for r in rows:
                        r["X"] += float(offset[0])
                        r["Y"] += float(offset[1])
                        r["Z"] += float(offset[2])
    result: Dict[str, pd.DataFrame] = {}
    for g, rows in data_per_group.items():
        if rows:
            result[g] = pd.DataFrame(rows)
    return result


def dfs_to_single_csv(
    dfs_3d: Dict[str, pd.DataFrame],
    set_key: str,
    output_dir: str,
) -> str:
    """
    将 a/b/c 三个 DataFrame 合并为一张 long 表并写入单 CSV。
    列: frame, fish_id, X, Y, Z（与 2D processed 的 frame + fish_id 一致）。
    返回写入路径。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for fish_id, df in dfs_3d.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "frame": int(r["frame"]),
                "fish_id": fish_id,
                "X": float(r["X"]),
                "Y": float(r["Y"]),
                "Z": float(r["Z"]),
            })
    out_df = pd.DataFrame(rows).sort_values(["frame", "fish_id"]).reset_index(drop=True)
    path = out_dir / f"{set_key}_3d.csv"
    out_df.to_csv(path, index=False)
    print(f"已保存: {path}")
    return str(path)


def export_groups_to_csv(
    best_json_path: str,
    set_key: str,
    output_dir: str,
    *,
    shift_to_positive: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Args:
        shift_to_positive: 若为 True，将所有坐标平移使最小值为 0（保持相对几何不变）
    """
    best_path = Path(best_json_path)
    if not best_path.exists():
        raise FileNotFoundError(f"未找到 best.json: {best_path}")

    with open(best_path, encoding="utf-8") as f:
        raw = json.load(f)
    best_by_frame = {int(k): v for k, v in raw.items()}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 为每个分组 label 累积所有帧
    data_per_group: Dict[str, List[Dict]] = {g: [] for g in GROUP_LABELS}

    for frame in sorted(best_by_frame.keys()):
        info = best_by_frame[frame]
        groups = info.get("groups", {})
        for g in GROUP_LABELS:
            if g in groups:
                Xv, Yv, Zv = groups[g]
                data_per_group[g].append(
                    {"frame": frame, "X": float(Xv), "Y": float(Yv), "Z": float(Zv)}
                )

    if shift_to_positive:
        all_pts = []
        for rows in data_per_group.values():
            for r in rows:
                all_pts.append([r["X"], r["Y"], r["Z"]])
        if all_pts:
            arr = np.array(all_pts)
            mins = arr.min(axis=0)
            offset = np.maximum(0, -mins)
            if np.any(offset > 0):
                for rows in data_per_group.values():
                    for r in rows:
                        r["X"] += float(offset[0])
                        r["Y"] += float(offset[1])
                        r["Z"] += float(offset[2])

    result: Dict[str, pd.DataFrame] = {}
    for g, rows in data_per_group.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"{set_key}_{g}_3d.csv"
        df.to_csv(csv_path, index=False)
        print(f"已保存: {csv_path}")
        result[g] = df

    return result


def main():
    import argparse

    ap = argparse.ArgumentParser(description="从 best.json 导出 a/b/c 三条 3D 轨迹为 CSV")
    ap.add_argument("--best_json", required=True, help="best.json 路径")
    ap.add_argument("--set_key", required=True, help="set key")
    ap.add_argument("--output_dir", required=True, help="输出 CSV 目录")
    ap.add_argument("--shift_to_positive", action="store_true", help="将坐标平移使最小值为 0")
    args = ap.parse_args()

    export_groups_to_csv(
        args.best_json, args.set_key, args.output_dir,
        shift_to_positive=args.shift_to_positive,
    )


if __name__ == "__main__":
    main()

