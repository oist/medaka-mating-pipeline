"""
2×2 布局合成视频：3D 轨迹 + 三视角 2D 原视频叠加「由 3D 对应的 2D 轨迹」

布局：
  - 第 1 行：左 = 3D 轨迹视频；右 = top 视角原视频 + 3D 对应 2D 轨迹叠加
  - 第 2 行：左 = side1 视角原视频 + 2D 叠加；右 = side2 视角原视频 + 2D 叠加

依赖：opencv-python, matplotlib, numpy, pandas
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.calibration.trajectory_3d import _apply_calibration_view

# 与 trajectory_3d 一致
def _get_xyz(df: pd.DataFrame):
    if "X" in df.columns and "Y" in df.columns and "Z" in df.columns:
        xyz = df[["X", "Y", "Z"]]
    elif "x" in df.columns and "y" in df.columns and "z" in df.columns:
        xyz = df[["x", "y", "z"]].rename(columns={"x": "X", "y": "Y", "z": "Z"})
    else:
        raise ValueError(f"DataFrame 需含 X,Y,Z 或 x,y,z 列")
    frame = df["frame"].values if "frame" in df.columns else np.arange(len(df))
    return frame, xyz["X"].values, xyz["Y"].values, xyz["Z"].values


def _sorted_xyz_per_label(trajectories: Dict[str, pd.DataFrame]):
    out = {}
    for label, df in trajectories.items():
        if df.empty:
            continue
        frame, x, y, z = _get_xyz(df)
        order = np.argsort(frame)
        out[label] = (frame[order], x[order], y[order], z[order])
    return out


DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
DEFAULT_XYZ_LIM = (0, 200)


def _render_one_3d_frame(
    trajectories: Dict[str, pd.DataFrame],
    frame_idx: int,
    trail_frames: int,
    xyz_lim: Tuple[float, float],
    out_height: int,
    out_width: int,
    dpi: int = 100,
) -> np.ndarray:
    """将当前帧的 3D 轨迹（含 trail）渲染为 RGB 图像 (H,W,3) uint8。"""
    data = _sorted_xyz_per_label(trajectories)
    if not data:
        return np.zeros((out_height, out_width, 3), dtype=np.uint8)

    all_frames = set()
    for frames, *_ in data.values():
        all_frames.update(frames.tolist())
    frames_sorted = sorted(all_frames)
    t_start = max(frames_sorted[0], frame_idx - trail_frames)

    fig = plt.figure(figsize=(out_width / dpi, out_height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _apply_calibration_view(ax, xyz_lim=xyz_lim)

    for i, (label, (frames, xs, ys, zs)) in enumerate(data.items()):
        c = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        mask = (frames >= t_start) & (frames <= frame_idx)
        if np.any(mask):
            ax.plot(xs[mask], ys[mask], zs[mask], color=c, label=label, linewidth=1.2, alpha=0.9)
            last = np.where(frames <= frame_idx)[0]
            if len(last) > 0:
                j = last[-1]
                ax.scatter(xs[j], ys[j], zs[j], color=c, s=80, marker="o")
    ax.legend()
    ax.set_title(f"3D (frame {int(frame_idx)})")
    fig.tight_layout()
    buf_io = io.BytesIO()
    fig.savefig(buf_io, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf_io.seek(0)
    if cv2 is not None:
        buf_arr = np.frombuffer(buf_io.getvalue(), dtype=np.uint8)
        buf = cv2.imdecode(buf_arr, cv2.IMREAD_COLOR)
        if buf is not None:
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            if buf.shape[0] != out_height or buf.shape[1] != out_width:
                buf = cv2.resize(buf, (out_width, out_height), interpolation=cv2.INTER_AREA)
            return buf
    return np.zeros((out_height, out_width, 3), dtype=np.uint8)


def _overlay_2d_on_frame(
    frame: np.ndarray,
    points_xy: Dict[str, List[float]],
    colors_bgr: List[Tuple[int, int, int]],
    labels: List[str],
) -> np.ndarray:
    """在 frame 上画 points_xy（label -> [x,y]），frame 已是目标分辨率。"""
    if cv2 is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    for i, label in enumerate(labels):
        if label not in points_xy:
            continue
        xy = points_xy[label]
        if len(xy) < 2:
            continue
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = colors_bgr[i % len(colors_bgr)]
        cv2.circle(out, (x, y), 8, color, 2)
        cv2.putText(out, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def load_best_2d(best_2d_path: str | Path) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """加载 best_2d.json：{frame: {"top": {"a": [x,y], ...}, "side1": {...}, "side2": {...}}}"""
    with open(best_2d_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_trajectories_from_csv(output_dir: str | Path, set_key: str) -> Dict[str, pd.DataFrame]:
    """从 output_dir 读取 3D 轨迹。优先单 CSV {set_key}_3d.csv，否则 {set_key}_a_3d.csv 等。"""
    out = Path(output_dir)
    single = out / f"{set_key}_3d.csv"
    if single.exists():
        df = pd.read_csv(single)
        if "fish_id" not in df.columns:
            raise ValueError(f"{single} 缺少列 fish_id")
        return {fid: gr[["frame", "X", "Y", "Z"]].copy() for fid, gr in df.groupby("fish_id")}
    result = {}
    for label in ["a", "b", "c"]:
        p = out / f"{set_key}_{label}_3d.csv"
        if p.exists():
            result[label] = pd.read_csv(p)
    return result


def _build_best_2d_from_3d(
    trajectories: Dict[str, pd.DataFrame],
    calibration,
    n_frames: int,
    view_order: List[str],
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    用 3D 轨迹 + 校准的 3D→2D 模型生成每帧每视角的 2D 点（重投影），用于叠加。
    view_order: ["top", "side1", "side2"] 对应 cam0, cam1, cam2。
    """
    # 轨迹 key 可能为 0,1,2 或 "a","b","c"，统一映射到叠加用的标签 a/b/c
    sorted_keys = sorted(trajectories.keys(), key=lambda x: (str(x), x))
    key_to_label = {k: ["a", "b", "c"][i] for i, k in enumerate(sorted_keys) if i < 3}
    best_2d = {}
    for frame_idx in range(n_frames):
        best_2d[frame_idx] = {v: {} for v in view_order}
        points_3d = []
        labels_at_frame = []
        for traj_key, df in trajectories.items():
            row = df[df["frame"] == frame_idx]
            if not row.empty:
                points_3d.append(row[["X", "Y", "Z"]].values[0])
                labels_at_frame.append(key_to_label.get(traj_key, str(traj_key)))
        if not points_3d:
            continue
        xyz = np.array(points_3d, dtype=float)
        imcoords = calibration.compute_imCoords_from_XYZ(xyz)  # (n, 3, 2)
        for i, label in enumerate(labels_at_frame):
            for j, v in enumerate(view_order):
                best_2d[frame_idx][v][label] = [float(imcoords[i, j, 0]), float(imcoords[i, j, 1])]
    return best_2d


def _build_best_2d_from_3d_orthogonal(
    trajectories: Dict[str, pd.DataFrame],
    n_frames: int,
    view_order: List[str],
    ref_wh: Dict[str, Tuple[int, int]],
    xyz_lim: Tuple[float, float],
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    用 3D 轨迹 + 正交投影生成每帧每视角的 2D 点，用于叠加（无校准模型时）。
    top=XY, side1=XZ, side2=YZ；世界坐标按 xyz_lim 线性映射到各视角图像尺寸。
    返回的 2D 坐标为各视角原始分辨率下的像素坐标，与 ref_wh 一致。
    """
    x0, x1 = float(xyz_lim[0]), float(xyz_lim[1])
    if x1 <= x0:
        x1 = x0 + 1.0
    sorted_keys = sorted(trajectories.keys(), key=lambda x: (str(x), x))
    key_to_label = {k: ["a", "b", "c"][i] for i, k in enumerate(sorted_keys) if i < 3}
    best_2d = {}
    for frame_idx in range(n_frames):
        best_2d[frame_idx] = {v: {} for v in view_order}
        for traj_key, df in trajectories.items():
            row = df[df["frame"] == frame_idx]
            if row.empty:
                continue
            label = key_to_label.get(traj_key, str(traj_key))
            X, Y, Z = float(row["X"].iloc[0]), float(row["Y"].iloc[0]), float(row["Z"].iloc[0])
            for v in view_order:
                w, h = ref_wh[v][0], ref_wh[v][1]
                if w <= 0 or h <= 0:
                    continue
                if v == "top":
                    u = (X - x0) / (x1 - x0) * (w - 1)
                    v_px = (h - 1) - (Y - x0) / (x1 - x0) * (h - 1)
                elif v == "side1":
                    u = (X - x0) / (x1 - x0) * (w - 1)
                    v_px = (h - 1) - (Z - x0) / (x1 - x0) * (h - 1)
                else:
                    u = (Y - x0) / (x1 - x0) * (w - 1)
                    v_px = (h - 1) - (Z - x0) / (x1 - x0) * (h - 1)
                best_2d[frame_idx][v][label] = [u, v_px]
    return best_2d


def render_2x2_video(
    output_dir: str | Path,
    set_key: str,
    video_paths: Dict[str, Path],
    best_2d_per_frame: Dict[int, Dict[str, Dict[str, List[float]]]] | str | Path | None = None,
    *,
    calibration_folder: str | Path | None = None,
    fps: float = 30,
    trail_seconds: float = 3.0,
    data_fps: float = 166,
    xyz_lim: Tuple[float, float] = (0, 200),
    cell_height: int = 360,
    out_filename: str | None = None,
) -> str:
    """
    生成 2×2 布局 MP4：左上 3D 轨迹，右上 top，左下 side1，右下 side2（后三格为原视频 + 2D 叠加）。

    2D 叠加完全由 3D 结果得到：若提供 calibration_folder 且含 XYZ_to_imCoords.joblib，则用
    校准 3D→2D 重投影；否则用正交投影（top=XY, side1=XZ, side2=YZ，世界坐标按 xyz_lim 线性映射）。
    不再使用 best_2d，保证 2D 与 3D 画面一致。

    Args:
        output_dir: 3D CSV 所在目录，输出 MP4 也写在此目录。
        set_key: 片段 key。
        video_paths: {"top": Path, "side1": Path, "side2": Path} 原视频路径。
        best_2d_per_frame: 已废弃，忽略；2D 叠加始终从 3D 投影得到。
        calibration_folder: 校准目录（含 XYZ_to_imCoords.joblib 时用 3D→2D，否则用正交投影）。
        fps: 输出视频帧率。
        trail_seconds: 3D 轨迹尾迹时长（秒）。
        data_fps: 数据帧率，用于计算 trail 帧数。
        xyz_lim: 3D 轴范围。
        cell_height: 每个格子的高度（像素），宽度按原视频比例缩放。
        out_filename: 输出文件名，默认 {set_key}_3d_plus_views_2x2.mp4

    Returns:
        输出 MP4 的绝对路径。
    """
    if cv2 is None:
        raise RuntimeError("需要 opencv-python：pip install opencv-python")

    output_dir = Path(output_dir)
    trajectories = load_trajectories_from_csv(output_dir, set_key)
    if not trajectories:
        raise FileNotFoundError(f"未找到 3D CSV：{output_dir / set_key}_a_3d.csv 等")

    order = ["top", "side1", "side2"]
    caps = {}
    ref_wh: Dict[str, Tuple[int, int]] = {}
    for v in order:
        p = video_paths.get(v)
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"缺少视角视频: {v} -> {p}")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {p}")
        caps[v] = cap
        ref_wh[v] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    n_frames = int(min(caps[v].get(cv2.CAP_PROP_FRAME_COUNT) for v in order))
    if n_frames <= 0:
        raise ValueError("视频无有效帧数")

    # 2D 叠加：完全从 3D 结果得到。优先校准 3D→2D 重投影，否则用正交投影（top=XY, side1=XZ, side2=YZ），不再使用 best_2d。
    best_2d_per_frame = None
    if calibration_folder is not None:
        try:
            from scripts.reconstruction.calibration import Calibration
            cal = Calibration(str(calibration_folder))
            if cal.has_imcoords_from_XYZ():
                best_2d_per_frame = _build_best_2d_from_3d(trajectories, cal, n_frames, order)
        except Exception:
            pass
    if best_2d_per_frame is None:
        # 无校准 3D→2D 时：用正交投影，保证 2D 与 3D 完全一致
        best_2d_per_frame = _build_best_2d_from_3d_orthogonal(
            trajectories, n_frames, order, ref_wh, xyz_lim
        )

    # 每个格子的宽高（三视角格子同高，宽度按比例）
    cell_w = cell_height
    for v in order:
        w, h = ref_wh[v]
        if h > 0:
            cell_w = max(cell_w, int(w * cell_height / h))
    # 统一为正方形格子便于 2x2 对齐
    cell_size = max(cell_height, cell_w)
    out_w = cell_size * 2
    out_h = cell_size * 2

    trail_frames = max(1, int(round(trail_seconds * data_fps)))
    colors_bgr = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]  # a, b, c
    labels = ["a", "b", "c"]

    out_path = output_dir / (out_filename or f"{set_key}_3d_plus_views_2x2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {out_path}")

    # 简单一行进度条：按帧数粗略更新百分比
    progress_step = max(1, n_frames // 100)
    print(f"开始写入 2×2 视频，共 {n_frames} 帧...", flush=True)

    for frame_idx in range(n_frames):
        # 左上：3D 轨迹
        img_3d = _render_one_3d_frame(
            trajectories,
            frame_idx,
            trail_frames,
            xyz_lim,
            cell_size,
            cell_size,
        )
        if len(img_3d.shape) == 2:
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_GRAY2BGR)
        elif img_3d.shape[2] == 3:
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
        if img_3d.shape[0] != cell_size or img_3d.shape[1] != cell_size:
            img_3d = cv2.resize(img_3d, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

        # 右上、左下、右下：三视角视频 + 2D 叠加
        row_videos = []
        for v in order:
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            # 叠加该帧、该视角的 2D 点（best_2d 为原始像素，需缩放）
            b2 = best_2d_per_frame.get(frame_idx, {}).get(v, {})
            if b2 and ref_wh[v][0] > 0 and ref_wh[v][1] > 0:
                scale_x = cell_size / ref_wh[v][0]
                scale_y = cell_size / ref_wh[v][1]
                scaled = {}
                for lb, xy in b2.items():
                    if len(xy) >= 2:
                        scaled[lb] = [xy[0] * scale_x, xy[1] * scale_y]
                frame = _overlay_2d_on_frame(frame, scaled, colors_bgr, labels)
            cv2.putText(frame, v, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            row_videos.append(frame)

        top_row = np.hstack([img_3d, row_videos[0]])
        bottom_row = np.hstack([row_videos[1], row_videos[2]])
        combined = np.vstack([top_row, bottom_row])
        writer.write(combined)

        if (frame_idx + 1) % progress_step == 0 or (frame_idx + 1) == n_frames:
            pct = (frame_idx + 1) * 100.0 / n_frames
            print(f"\r写入 2×2 视频: {frame_idx + 1}/{n_frames} ({pct:5.1f}%)", end="", flush=True)

    print()

    for cap in caps.values():
        cap.release()
    writer.release()
    return str(out_path.resolve())


__all__ = [
    "load_best_2d",
    "load_trajectories_from_csv",
    "render_2x2_video",
]
