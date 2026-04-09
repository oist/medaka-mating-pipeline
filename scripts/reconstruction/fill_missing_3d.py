"""
Fill missing frames in 3D trajectories so each frame has all three fish.

This legacy helper uses cubic spline interpolation along the time axis and
falls back to nearest valid-frame copying when there are too few valid points.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

GROUP_LABELS = ["a", "b", "c"]


def ensure_all_frames(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
) -> Dict[str, pd.DataFrame]:
    """Ensure that every DataFrame contains every frame in `all_frames`."""
    out: Dict[str, pd.DataFrame] = {}
    all_frames = sorted(all_frames)
    for label in GROUP_LABELS:
        df = dfs_3d.get(label)
        if df is None or df.empty:
            out[label] = pd.DataFrame({"frame": all_frames, "X": np.nan, "Y": np.nan, "Z": np.nan})
            continue
        df = df[["frame", "X", "Y", "Z"]].copy().drop_duplicates(subset=["frame"])
        existing = df.set_index("frame")
        full_index = pd.Index(all_frames, name="frame")
        reindexed = existing.reindex(full_index).reset_index()
        out[label] = reindexed
    return out


def _fill_xyz_cubic_spline_one_series(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Fill NaNs in a 1D series with cubic spline interpolation.

    If there are fewer than four valid points, copy from the nearest valid
    frame instead.
    """
    valid = ~np.isnan(values) & ~np.isnan(frames)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return values
    if n_valid == 1:
        return np.full_like(values, values[valid][0])
    frames_valid = np.asarray(frames[valid], dtype=float)
    values_valid = np.asarray(values[valid], dtype=float)
    order = np.argsort(frames_valid)
    u = frames_valid[order]
    v = values_valid[order]
    keep = np.concatenate([[True], u[1:] != u[:-1]])
    u, v = u[keep], v[keep]
    if len(u) < 2:
        return values
    nan_mask = np.isnan(values)
    if not np.any(nan_mask):
        return values
    frames_nan = frames[nan_mask]
    if len(u) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(u, v)
            v_nan = np.where(
                frames_nan < u.min(),
                v[0],
                np.where(frames_nan > u.max(), v[-1], cs(frames_nan)),
            )
            out = values.copy()
            out[nan_mask] = v_nan
            return out
        except Exception:
            pass
    valid_idx = np.where(valid)[0]
    for i in np.where(nan_mask)[0]:
        dist = np.abs(frames[valid_idx] - frames[i])
        j = valid_idx[np.argmin(dist)]
        values[i] = values[j]
    return values


def fill_xyz_cubic_spline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cubic-spline filling to the X/Y/Z columns of one trajectory."""
    df = df.sort_values("frame").reset_index(drop=True)
    frames = df["frame"].values.astype(float)
    for col in ["X", "Y", "Z"]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float).copy()
        df[col] = _fill_xyz_cubic_spline_one_series(frames, vals)
    return df


def ensure_all_frames_and_fill_spline(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
) -> Dict[str, pd.DataFrame]:
    """Expand to all frames first, then fill X/Y/Z with cubic spline interpolation."""
    expanded = ensure_all_frames(dfs_3d, all_frames)
    out: Dict[str, pd.DataFrame] = {}
    for label, df in expanded.items():
        df = fill_xyz_cubic_spline(df.copy())
        if df[["X", "Y", "Z"]].isna().all(axis=None):
            continue
        for col in ["X", "Y", "Z"]:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(0.0)
        out[label] = df
    return out
