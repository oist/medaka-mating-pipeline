"""
3D 轨迹可视化：从 3D CSV 或 trajectories 字典绘制 PNG。
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.calibration.trajectory_3d import _apply_calibration_view

DEFAULT_XYZ_LIM = (0, 200)
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def _get_xyz(df: pd.DataFrame):
    if "X" in df.columns and "Y" in df.columns and "Z" in df.columns:
        xyz = df[["X", "Y", "Z"]]
    elif "x" in df.columns and "y" in df.columns and "z" in df.columns:
        xyz = df[["x", "y", "z"]].rename(columns={"x": "X", "y": "Y", "z": "Z"})
    else:
        raise ValueError(f"DataFrame 需含 X,Y,Z 或 x,y,z 列")
    frame = df["frame"].values if "frame" in df.columns else np.arange(len(df))
    return frame, xyz["X"].values, xyz["Y"].values, xyz["Z"].values


def _sorted_xyz_per_label(trajectories: Dict[str, pd.DataFrame]) -> Dict[str, Tuple]:
    out = {}
    for label, df in trajectories.items():
        if df.empty:
            continue
        frame, x, y, z = _get_xyz(df)
        order = np.argsort(frame)
        out[label] = (frame[order], x[order], y[order], z[order])
    return out


def plot_3d_trajectories(
    trajectories: Dict[str, pd.DataFrame],
    out_path: str,
    xyz_lim: Tuple[float, float] | None = DEFAULT_XYZ_LIM,
) -> None:
    """
    绘制所有个体的 3D 轨迹线图并保存为 PNG。
    trajectories: {个体标签: DataFrame 含 frame, X, Y, Z}
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


def run_plot(
    output_dir: str,
    set_key: str,
    *,
    xyz_lim: Tuple[float, float] | None = DEFAULT_XYZ_LIM,
    csv_name: str | None = None,
) -> str:
    """
    从 output_dir 下 {set_key}_3d.csv 读取，绘制 3D 轨迹 PNG，保存为 {set_key}_3d_trajectories.png。
    返回生成的 PNG 路径。
    """
    out_dir = Path(output_dir)
    csv_path = out_dir / (csv_name or f"{set_key}_3d.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 3D CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "fish_id" not in df.columns:
        raise ValueError("3D CSV 需含 fish_id 列")
    trajectories = {}
    for fid in df["fish_id"].unique():
        sub = df[df["fish_id"] == fid].copy()
        if "X" not in sub.columns and "x" in sub.columns:
            sub = sub.rename(columns={"x": "X", "y": "Y", "z": "Z"})
        trajectories[str(fid)] = sub

    png_path = out_dir / f"{set_key}_3d_trajectories.png"
    plot_3d_trajectories(trajectories, str(png_path), xyz_lim=xyz_lim)
    return str(png_path)
