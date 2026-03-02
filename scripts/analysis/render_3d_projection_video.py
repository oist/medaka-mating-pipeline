"""
3D 轨迹投影到 2D 视频并生成叠加视频
====================================

放弃 ID 显示：仅将 3D 点用校准模型投影到三视角 2D，在视频上画点（不标 ID 文字）。
流程：加载 3D CSV → 用校准 XYZ_to_imCoords 逐帧投影到 top/side1/side2 → 复用三视角叠加逻辑写出 MP4。

依赖：opencv-python, numpy, pandas；校准目录需含 XYZ_to_imCoords.joblib。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_tracking_video import (
    render_three_view_video,
    find_view_csvs,
    resolve_video_paths,
)


def find_videos_by_set_key(
    video_dir: Path,
    set_key: str,
    video_top: Optional[Path] = None,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
) -> Dict[str, Path]:
    """按 set_key 中的时间段与 PLAYER1/2/3 在 video_dir 下找三路 mp4。"""
    out: Dict[str, Path] = {}
    if video_top and video_top.exists():
        out["top"] = Path(video_top)
    if video_side1 and video_side1.exists():
        out["side1"] = Path(video_side1)
    if video_side2 and video_side2.exists():
        out["side2"] = Path(video_side2)
    if len(out) == 3:
        return out
    m = re.search(r"(\d+\.\d+-\d+\.\d+)", set_key)
    time_part = m.group(1) if m else ""
    if not video_dir.exists() or not time_part:
        return out
    all_mp4 = list(video_dir.rglob("*.mp4"))

    def by_view(name_upper: str) -> Optional[str]:
        if "PLAYER1" in name_upper:
            return "top"
        if "PLAYER2" in name_upper:
            return "side1"
        if "PLAYER3" in name_upper:
            return "side2"
        return None

    for p in all_mp4:
        if time_part not in p.stem:
            continue
        v = by_view(p.stem.upper())
        if v and v not in out:
            out[v] = p.resolve()
    return out


def project_3d_to_2d_views(
    df_3d: pd.DataFrame,
    cal: Calibration,
) -> Dict[str, pd.DataFrame]:
    """
    将 3D CSV（frame, fish_id, X, Y, Z）按帧投影到三视角 2D。
    放弃 ID：输出仅含 frame, individual(0/1/2), x, y，用于画点不标 ID。
    返回 { "top": df, "side1": df, "side2": df }，每 df 列 frame, individual, x, y。
    """
    # 按帧分组，每帧最多 3 个点（按 fish_id 或行序）
    frames = sorted(df_3d["frame"].unique())
    view_order = ["top", "side1", "side2"]  # 对应 cam0, cam1, cam2
    view_rows: Dict[str, List[Dict]] = {v: [] for v in view_order}

    for f in frames:
        sub = df_3d[df_3d["frame"] == f]
        if sub.empty:
            continue
        # 每帧取最多 3 行，按 fish_id 或索引顺序
        if "fish_id" in sub.columns:
            sub = sub.sort_values("fish_id").head(3)
        else:
            sub = sub.head(3)
        xyz = sub[["X", "Y", "Z"]].values.astype(float)
        valid = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[valid]
        if xyz.size == 0:
            continue
        # 投影: (n, 3, 2) -> [i, cam, 0/1] = (u,v)
        uv_all = cal.compute_imCoords_from_XYZ(xyz)
        n_pt = uv_all.shape[0]
        for i in range(n_pt):
            u, v = float(uv_all[i, 0, 0]), float(uv_all[i, 0, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["top"].append({"frame": f, "individual": i, "x": u, "y": v})
            u, v = float(uv_all[i, 1, 0]), float(uv_all[i, 1, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["side1"].append({"frame": f, "individual": i, "x": u, "y": v})
            u, v = float(uv_all[i, 2, 0]), float(uv_all[i, 2, 1])
            if np.isfinite(u) and np.isfinite(v):
                view_rows["side2"].append({"frame": f, "individual": i, "x": u, "y": v})

    view_data: Dict[str, pd.DataFrame] = {}
    for v in view_order:
        if view_rows[v]:
            view_data[v] = pd.DataFrame(view_rows[v])
        else:
            view_data[v] = pd.DataFrame(columns=["frame", "individual", "x", "y"])
    return view_data


def run_render_3d_projection_video(
    csv_3d_path: str | Path,
    model_folder: str | Path,
    video_dir: str | Path,
    set_key: str,
    output_path: str | Path,
    fps: float,
    *,
    video_top: Optional[Path] = None,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
    trail_duration_s: float = 0.5,
    show_trail_row: bool = False,
) -> Path:
    """
    3D CSV → 校准投影到 2D → 三视角视频叠加，写出 MP4（不标 ID，仅画点）。

    csv_3d_path: 3D 轨迹 CSV，列 frame, fish_id, X, Y, Z
    model_folder: 校准目录，需含 XYZ_to_imCoords.joblib
    video_dir: 视频根目录，下含与 set_key 时间段+PLAYER1/2/3 匹配的 mp4
    set_key: 片段 key，用于匹配视频文件名
    output_path: 输出 mp4 路径
    fps: 视频帧率
    """
    csv_3d_path = Path(csv_3d_path)
    model_folder = Path(model_folder)
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    if not csv_3d_path.exists():
        raise FileNotFoundError(f"3D CSV 不存在: {csv_3d_path}")
    if not model_folder.is_dir():
        raise FileNotFoundError(f"校准目录不存在: {model_folder}")

    df_3d = pd.read_csv(csv_3d_path)
    for c in ["frame", "X", "Y", "Z"]:
        if c not in df_3d.columns:
            raise ValueError(f"3D CSV 需含列 {c}，当前: {list(df_3d.columns)}")
    cal = Calibration(model_folder)
    view_data = project_3d_to_2d_views(df_3d, cal)

    video_paths = find_videos_by_set_key(
        video_dir, set_key, video_top, video_side1, video_side2
    )
    missing = [v for v in ["top", "side1", "side2"] if v not in video_paths]
    if missing:
        raise FileNotFoundError(
            f"未找到视角视频: {missing}。请在 {video_dir} 下放置含 set_key 时间段且文件名含 PLAYER1/2/3 的 mp4，或传入 video_top/side1/side2。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_three_view_video(
        view_data,
        video_paths,
        output_path,
        fps,
        trail_duration_s=trail_duration_s,
        show_trail_row=show_trail_row,
        simple_dots=True,  # 放弃 ID：仅画点，不标文字
    )
    return output_path


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="将 3D 轨迹投影到 2D 视频并生成叠加视频（不标 ID）。"
    )
    p.add_argument("--csv-3d", type=Path, required=True, help="3D 轨迹 CSV（frame, fish_id, X, Y, Z）")
    p.add_argument("--model-folder", type=Path, required=True, help="校准模型目录（含 XYZ_to_imCoords.joblib）")
    p.add_argument("--video-dir", type=Path, required=True, help="视频根目录")
    p.add_argument("--set-key", type=str, required=True, help="片段 set key，用于匹配视频")
    p.add_argument("--output", type=Path, default=None, help="输出 mp4 路径")
    p.add_argument("--fps", type=float, default=166.0, help="视频帧率")
    p.add_argument("--video-top", type=Path, default=None)
    p.add_argument("--video-side1", type=Path, default=None)
    p.add_argument("--video-side2", type=Path, default=None)
    p.add_argument("--trail-duration-s", type=float, default=0.5)
    p.add_argument("--no-trail-row", action="store_true", help="不显示下方轨迹行")
    args = p.parse_args()

    out = args.output
    if out is None:
        out = args.csv_3d.parent / (args.csv_3d.stem.replace("_3d", "") + "_3d_projection_overlay.mp4")
    run_render_3d_projection_video(
        args.csv_3d,
        args.model_folder,
        args.video_dir,
        args.set_key,
        out,
        args.fps,
        video_top=args.video_top,
        video_side1=args.video_side1,
        video_side2=args.video_side2,
        trail_duration_s=args.trail_duration_s,
        show_trail_row=not args.no_trail_row,
    )
    print("已写出:", out)


if __name__ == "__main__":
    main()
