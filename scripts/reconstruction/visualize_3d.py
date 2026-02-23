"""
3D 轨迹可视化（追踪 ID 之后的可视化步骤）
=========================================

从已有 3D CSV 生成 3D 轨迹视频 MP4（ffmpeg 编码，轨迹仅保留最近 N 秒），不重新跑三角化。

用法:
    python -m scripts.reconstruction.visualize_3d ^
      --output_dir "H:\3d_trajectories" ^
      --set_key "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from scripts.calibration.trajectory_3d import plot_3d_trajectories, plot_3d_trajectories_video


# 只选一组 3 条轨迹：若存在 ID_1/2/3 则只用它们，否则用 a/b/c（避免 a/b/c 与 ID_1/2/3 同时存在时画出 6 条线）
PREFERRED_GROUP = ("ID_1", "ID_2", "ID_3")
FALLBACK_GROUP = ("a", "b", "c")


def load_trajectories_from_csv(output_dir: str, set_key: str) -> Dict[str, pd.DataFrame]:
    """从 output_dir 读取 {set_key}_*_3d.csv，构建 trajectories 字典。只加载一组（ID_1/2/3 或 a/b/c），避免重复 6 条线。"""
    out = Path(output_dir)
    # 优先使用跨视频 ID（Step 5 输出）
    for name in PREFERRED_GROUP:
        p = out / f"{set_key}_{name}_3d.csv"
        if p.exists():
            return {
                name: pd.read_csv(out / f"{set_key}_{name}_3d.csv")
                for name in PREFERRED_GROUP
                if (out / f"{set_key}_{name}_3d.csv").exists()
            }
    # 否则使用 a/b/c（Step 4 导出）
    result = {}
    for name in FALLBACK_GROUP:
        p = out / f"{set_key}_{name}_3d.csv"
        if p.exists():
            result[name] = pd.read_csv(p)
    if not result:
        raise FileNotFoundError(f"未找到 3D CSV: {out}/{set_key}_*_3d.csv（需要 a/b/c 或 ID_1/2/3）")
    return result


def run_video(
    output_dir: str,
    set_key: str,
    fps: float = 30,
    subsample: int = 5,
    trail_seconds: float = 5.0,
    data_fps: float | None = None,
    xyz_lim: tuple[float, float] | None = (0, 200),
) -> str:
    """从 3D CSV 生成 MP4 视频。fps=输出视频帧率（越小播放越慢）；data_fps=原始数据帧率（用于 trail 计算）。"""
    trajectories = load_trajectories_from_csv(output_dir, set_key)
    out_path = Path(output_dir) / f"{set_key}_3d_trajectories.mp4"
    plot_3d_trajectories_video(
        trajectories, str(out_path),
        fps=fps, subsample=subsample, trail_seconds=trail_seconds,
        data_fps=data_fps, xyz_lim=xyz_lim,
    )
    return str(out_path)


def run_plot(
    output_dir: str,
    set_key: str,
    xyz_lim: tuple[float, float] | None = (0, 200),
) -> str:
    """从 3D CSV 生成 PNG 轨迹图。xyz_lim 固定坐标轴范围，默认 (0, 200)。"""
    trajectories = load_trajectories_from_csv(output_dir, set_key)
    out_path = Path(output_dir) / f"{set_key}_3d_trajectories.png"
    plot_3d_trajectories(trajectories, str(out_path), xyz_lim=xyz_lim)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(description="从 3D CSV 生成 3D 轨迹视频 MP4")
    ap.add_argument("--output_dir", required=True, help="3D CSV 所在目录")
    ap.add_argument("--set_key", required=True, help="set key")
    ap.add_argument("--fps", type=float, default=30, help="帧率")
    ap.add_argument("--subsample", type=int, default=5, help="帧间隔")
    ap.add_argument("--trail_seconds", type=float, default=5.0, help="轨迹只保留最近多少秒（默认 5）")
    ap.add_argument("--format", choices=["mp4", "png"], default="mp4", help="输出格式")
    args = ap.parse_args()

    if args.format == "mp4":
        out = run_video(
            args.output_dir, args.set_key, args.fps, args.subsample,
            trail_seconds=args.trail_seconds,
        )
    else:
        out = run_plot(args.output_dir, args.set_key)
    print(f"完成: {out}")


if __name__ == "__main__":
    main()
