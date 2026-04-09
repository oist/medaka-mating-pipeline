"""
Create a 2x2 composite video:

- top-left: 3D trajectory video
- top-right: top-view raw video with 2D points corresponding to the 3D track
- bottom-left: side1 raw video with 2D overlay
- bottom-right: side2 raw video with 2D overlay
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.calibration.trajectory_3d import _apply_calibration_view


def _get_xyz(df: pd.DataFrame):
    if "X" in df.columns and "Y" in df.columns and "Z" in df.columns:
        xyz = df[["X", "Y", "Z"]]
    elif "x" in df.columns and "y" in df.columns and "z" in df.columns:
        xyz = df[["x", "y", "z"]].rename(columns={"x": "X", "y": "Y", "z": "Z"})
    else:
        raise ValueError("DataFrame must contain X,Y,Z or x,y,z columns")
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
    """Render one 3D trajectory frame to an RGB image."""
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
