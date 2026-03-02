"""
3D 轨迹绘图与视频
=================

从 trajectories 字典（label -> DataFrame 含 frame, X, Y, Z）生成 3D 轨迹图 PNG 或 MP4。
供 scripts.reconstruction.visualize_3d 调用。
视频使用 ffmpeg 编码（优先使用 imageio-ffmpeg 提供的 ffmpeg，无需系统 PATH）。
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_EXE = "ffmpeg"

# 3D CSV 列名（兼容大小写）
FRAME_COL = "frame"
XYZ_COLS = ["X", "Y", "Z"]


def _get_xyz(df: pd.DataFrame):
    """从 DataFrame 取 frame 与 X,Y,Z，列名兼容 X/x 等。"""
    if "X" in df.columns and "Y" in df.columns and "Z" in df.columns:
        xyz = df[["X", "Y", "Z"]]
    elif "x" in df.columns and "y" in df.columns and "z" in df.columns:
        xyz = df[["x", "y", "z"]].rename(columns={"x": "X", "y": "Y", "z": "Z"})
    else:
        raise ValueError(f"DataFrame 需含 X,Y,Z 或 x,y,z 列，当前列: {list(df.columns)}")
    frame = df[FRAME_COL] if FRAME_COL in df.columns else np.arange(len(df))
    return frame.values, xyz["X"].values, xyz["Y"].values, xyz["Z"].values


def _sorted_xyz_per_label(trajectories: Dict[str, pd.DataFrame]):
    """返回 {label: (frames, xs, ys, zs)} 已按 frame 排序。"""
    out = {}
    for label, df in trajectories.items():
        if df.empty:
            continue
        frame, x, y, z = _get_xyz(df)
        order = np.argsort(frame)
        out[label] = (frame[order], x[order], y[order], z[order])
    return out


# 默认每条轨迹颜色（与 qc 等保持一致风格）
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# 固定 3D 坐标轴范围 (mm)，便于观看时坐标系不动
DEFAULT_XYZ_LIM = (0, 200)


def _apply_calibration_view(ax, xyz_lim: tuple[float, float] | None = DEFAULT_XYZ_LIM) -> None:
    """
    与相机校准 02_calibration / qc_3d_template 一致的 3D 坐标系：
    - Y 轴反向（invert_yaxis）
    - 轴标签 X/Y/Z (mm)，原点与校准一致
    - 等比例显示（set_box_aspect [1,1,1]）保证坐标轴方向不畸变
    - 可选固定范围 xyz_lim
    """
    ax.invert_yaxis()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    if xyz_lim is not None:
        low, high = xyz_lim
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_zlim(low, high)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def plot_3d_trajectories(
    trajectories: Dict[str, pd.DataFrame],
    out_path: str,
    xyz_lim: tuple[float, float] | None = DEFAULT_XYZ_LIM,
) -> None:
    """
    绘制所有个体的 3D 轨迹线图并保存为 PNG。
    坐标即普通 X,Y,Z；可选固定坐标轴范围（默认 0~200）。

    trajectories: {个体标签: DataFrame 含 frame, X, Y, Z}
    out_path: 输出 PNG 路径
    xyz_lim: 固定 X/Y/Z 轴范围 (low, high)，默认 (0, 200)；None 表示自动
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = _sorted_xyz_per_label(trajectories)
    if not data:
        raise ValueError("trajectories 为空或所有 DataFrame 为空")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = DEFAULT_COLORS
    for i, (label, (frames, xs, ys, zs)) in enumerate(data.items()):
        c = colors[i % len(colors)]
        ax.plot(xs, ys, zs, color=c, label=label, linewidth=1.5, alpha=0.9)
        ax.scatter(xs[-1], ys[-1], zs[-1], color=c, s=40, marker="o")

    _apply_calibration_view(ax, xyz_lim=xyz_lim)
    ax.legend()
    ax.set_title("3D trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_trajectories_video(
    trajectories: Dict[str, pd.DataFrame],
    out_path: str,
    fps: float = 30,
    subsample: int = 1,
    trail_seconds: float = 5.0,
    data_fps: float | None = None,
    xyz_lim: tuple[float, float] | None = DEFAULT_XYZ_LIM,
) -> None:
    """
    按时间推进绘制 3D 轨迹，轨迹只保留最近 trail_seconds 秒，用 ffmpeg 编码为 MP4。
    data_fps 用于计算 trail 帧数；fps 为输出视频帧率（越小播放越慢）。

    trajectories: {个体标签: DataFrame 含 frame, X, Y, Z}
    out_path: 输出 MP4 路径
    fps: 输出视频帧率（播放速度，建议 24~30 便于观看）
    subsample: 帧间隔，1=每帧，10=每 10 帧取 1 帧
    trail_seconds: 轨迹只显示最近多少秒内的点（默认 5 秒）
    data_fps: 原始数据帧率（用于 trail 计算）；若 None 则用 fps
    xyz_lim: 固定 X/Y/Z 轴范围，默认 (0, 200)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = _sorted_xyz_per_label(trajectories)
    if not data:
        raise ValueError("trajectories 为空或所有 DataFrame 为空")

    all_frames = set()
    for frames, *_ in data.values():
        all_frames.update(frames.tolist())
    frames_sorted = sorted(all_frames)
    if subsample > 1:
        frames_sorted = frames_sorted[::subsample]
    if not frames_sorted:
        raise ValueError("没有有效帧")

    src_fps = data_fps if data_fps is not None else fps
    trail_frames = max(1, int(src_fps * trail_seconds))
    colors = DEFAULT_COLORS

    with tempfile.TemporaryDirectory(prefix="traj3d_") as tmpdir:
        tmpdir = Path(tmpdir)
        for idx, t in enumerate(frames_sorted):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            t_start = max(frames_sorted[0], t - trail_frames)

            for i, (label, (frames, xs, ys, zs)) in enumerate(data.items()):
                c = colors[i % len(colors)]
                mask = (frames >= t_start) & (frames <= t)
                if np.any(mask):
                    ax.plot(xs[mask], ys[mask], zs[mask], color=c, label=label, linewidth=1.2, alpha=0.9)
                    last = np.where(frames <= t)[0][-1]
                    ax.scatter(xs[last], ys[last], zs[last], color=c, s=80, marker="o")
            _apply_calibration_view(ax, xyz_lim=xyz_lim)
            ax.legend()
            ax.set_title(f"3D trajectories (frame {int(t)}, trail {trail_seconds}s)")
            fig.tight_layout()
            fig.savefig(tmpdir / f"frame_{idx:05d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        pattern = str(tmpdir / "frame_%05d.png")
        cmd = [
            _FFMPEG_EXE, "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"ffmpeg 失败: {ret.stderr or ret.stdout}")


# 单色 3D 轨迹视频（不区分 ID，三条轨迹同一颜色，轨迹尾迹 0.5 秒）
SINGLE_COLOR = "#00b4d8"  # 同一颜色


def render_3d_trajectory_video_single_color(
    csv_3d_path: str | Path,
    out_path: str | Path,
    *,
    fps: float = 30,
    trail_seconds: float = 0.5,
    data_fps: float = 166,
    subsample: int = 1,
    xyz_lim: tuple[float, float] | None = DEFAULT_XYZ_LIM,
) -> Path:
    """
    从 3D CSV 生成 3D 轨迹视频：不区分 ID，三个点用同一颜色，轨迹显示最近 trail_seconds 秒。
    """
    csv_3d_path = Path(csv_3d_path)
    out_path = Path(out_path)
    if not csv_3d_path.exists():
        raise FileNotFoundError(f"3D CSV 不存在: {csv_3d_path}")

    df = pd.read_csv(csv_3d_path)
    if "frame" not in df.columns:
        raise ValueError(f"3D CSV 需含 frame 列，当前: {list(df.columns)}")
    if "X" not in df.columns and "x" in df.columns:
        df = df.rename(columns={"x": "X", "y": "Y", "z": "Z"})
    if "X" not in df.columns or "Y" not in df.columns or "Z" not in df.columns:
        raise ValueError(f"3D CSV 需含 X,Y,Z 列，当前: {list(df.columns)}")

    trajectories = {}
    id_col = "fish_id" if "fish_id" in df.columns else "individual"
    if id_col not in df.columns:
        trajectories["all"] = df[["frame", "X", "Y", "Z"]].copy()
    else:
        for fid in df[id_col].unique():
            trajectories[str(fid)] = df[df[id_col] == fid][["frame", "X", "Y", "Z"]].copy()

    data = _sorted_xyz_per_label(trajectories)
    if not data:
        raise ValueError("3D CSV 无有效轨迹数据")

    all_frames = set()
    for frames, *_ in data.values():
        all_frames.update(frames.tolist())
    frames_sorted = sorted(all_frames)
    if subsample > 1:
        frames_sorted = frames_sorted[::subsample]
    if not frames_sorted:
        raise ValueError("没有有效帧")

    trail_frames = max(1, int(data_fps * trail_seconds))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="traj3d_") as tmpdir:
        tmpdir = Path(tmpdir)
        for idx, t in enumerate(frames_sorted):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            t_start = max(frames_sorted[0], t - trail_frames)

            for _label, (frames, xs, ys, zs) in data.items():
                mask = (frames >= t_start) & (frames <= t)
                valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                mask = mask & valid
                if np.any(mask):
                    ax.plot(xs[mask], ys[mask], zs[mask], color=SINGLE_COLOR, linewidth=1.2, alpha=0.9)
                idx_last = np.where((frames <= t) & valid)[0]
                if len(idx_last) > 0:
                    last = idx_last[-1]
                    ax.scatter(xs[last], ys[last], zs[last], color=SINGLE_COLOR, s=80, marker="o")

            _apply_calibration_view(ax, xyz_lim=xyz_lim)
            ax.set_title(f"3D trajectory (frame {int(t)}, trail {trail_seconds}s)")
            fig.tight_layout()
            fig.savefig(tmpdir / f"frame_{idx:05d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        pattern = str(tmpdir / "frame_%05d.png")
        cmd = [
            _FFMPEG_EXE, "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"ffmpeg 失败: {ret.stderr or ret.stdout}")

    return out_path
