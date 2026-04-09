"""
Missing-value filling and optional trajectory smoothing for 2D tracks.

Responsibilities:

- interpolate `x` and `y` where values are `NaN`
- optionally smooth valid interpolated segments
- optionally expand each fish trajectory to the full frame sequence for a clip

Outlier detection belongs in `outlier_detection.py`.
"""

from __future__ import annotations

import re
from typing import Dict, Hashable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_time_from_filename(filename: str) -> tuple[float | None, float | None]:
    """
    Extract the start and end time in seconds from a clip filename.

    Example:
        top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv
        -> (1905.506, 1913.422)
    """
    basename = filename.split("\\")[-1] if "\\" in filename else filename.split("/")[-1]
    pattern = r"(\d+\.\d+)-(\d+\.\d+)"
    match = re.search(pattern, basename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def generate_complete_frame_sequence(
    start_time: float | None,
    end_time: float | None,
    fps: int = 166,
) -> np.ndarray | None:
    """Generate the full frame sequence for the clip starting at frame 0."""
    if start_time is None or end_time is None:
        return None
    duration = max(0.0, float(end_time) - float(start_time))
    total_frames = int(duration * fps)
    return np.arange(0, total_frames, dtype=int)


def _interpolate_1d_timeseries(
    data: np.ndarray,
    *,
    method: str = "linear",
    limit: int | None = None,
) -> np.ndarray:
    """Interpolate a 1D time series using pandas."""
    if data.size == 0:
        return data

    series = pd.Series(data.copy())
    return series.interpolate(
        method=method,
        limit=limit,
        limit_direction="both",
    ).to_numpy()


def _smooth_series_inplace(
    values: np.ndarray,
    *,
    smooth: bool,
    smooth_window: int,
    smooth_order: int,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing when there are enough valid points."""
    if not smooth:
        return values

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < smooth_window:
        return values

    try:
        smoothed = values.copy()
        smoothed[valid_mask] = savgol_filter(
            smoothed[valid_mask],
            smooth_window,
            smooth_order,
        )
        return smoothed
    except Exception:
        return values


def fill_missing_values_trajectory(
    df: pd.DataFrame,
    fish_id: Hashable,
    method: str = "linear",
    interp_limit: int | None = None,
    smooth: bool = True,
    smooth_window: int = 9,
    smooth_order: int = 2,
    entity_col: str = "fish_id",
) -> pd.DataFrame:
    """
    Interpolate one entity trajectory and optionally smooth it.

    Only the `x` and `y` columns are modified.
    """
    mask_entity = df[entity_col] == fish_id
    entity_df = df.loc[mask_entity].copy().sort_values("frame")
    if entity_df.empty:
        return df

    result_df = df.copy()

    xs = entity_df["x"].to_numpy(dtype=float)
    if np.any(~np.isnan(xs)):
        xs_interp = _interpolate_1d_timeseries(xs, method=method, limit=interp_limit)
        xs_interp = _smooth_series_inplace(
            xs_interp,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
        )
        result_df.loc[entity_df.index, "x"] = xs_interp

    ys = entity_df["y"].to_numpy(dtype=float)
    if np.any(~np.isnan(ys)):
        ys_interp = _interpolate_1d_timeseries(ys, method=method, limit=interp_limit)
        ys_interp = _smooth_series_inplace(
            ys_interp,
            smooth=smooth,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
        )
        result_df.loc[entity_df.index, "y"] = ys_interp

    return result_df


def _ensure_complete_frames_for_player(
    df: pd.DataFrame,
    *,
    file_path: str | None,
    fps: int,
    entity_col: str = "fish_id",
) -> pd.DataFrame:
    """
    Ensure one row per frame and per entity for a single view.

    Missing rows are created with `NaN` coordinates and score.
    """
    if file_path is None:
        return df

    start_time, end_time = extract_time_from_filename(file_path)
    frames = generate_complete_frame_sequence(start_time, end_time, fps=fps)
    if frames is None:
        return df

    present = [e for e in df[entity_col].unique() if e != "unassigned"]
    base_labels = ["a", "b", "c"]
    entities: list[Hashable] = [e for e in base_labels if e in present]
    entities += [e for e in present if e not in entities]
    for label in base_labels:
        if label not in entities:
            entities.append(label)

    rows = []
    for frame in frames:
        for ent in entities:
            sub = df[(df["frame"] == frame) & (df[entity_col] == ent)]
            if not sub.empty:
                rows.append(sub.iloc[0].to_dict())
            else:
                row: dict = {
                    "frame": int(frame),
                    entity_col: ent,
                    "x": np.nan,
                    "y": np.nan,
                    "score": np.nan,
                }
                if "track" in df.columns:
                    row["track"] = np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def ensure_complete_frames(
    df_dict: Dict[str, pd.DataFrame],
    file_set_info: Dict[str, str],
    fps: int = 166,
) -> Dict[str, pd.DataFrame]:
    """Expand all views to the full frame range implied by the clip filename."""
    out: Dict[str, pd.DataFrame] = {}
    for player, df in df_dict.items():
        file_key = f"{player}_file"
        file_path = file_set_info.get(file_key)
        out[player] = _ensure_complete_frames_for_player(
            df,
            file_path=file_path,
            fps=fps,
            entity_col="fish_id",
        )
    return out


def process_all_fish_missing_values(
    df_dict: Dict[str, pd.DataFrame],
    file_set_info: Dict[str, str] | None = None,
    fps: int = 166,
    method: str = "linear",
    interp_limit: int | None = None,
    smooth: bool = True,
    smooth_window: int = 9,
    smooth_order: int = 2,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Interpolate and optionally smooth trajectories for all fish in all views.

    Typical call order:
        1. `outlier_detection.process_all_fish_outliers()` marks bad points as NaN
        2. `process_all_fish_missing_values()` fills short gaps and smooths them
    """
    smooth_window = int(kwargs.get("smooth_window", smooth_window))
    smooth_order = int(kwargs.get("smooth_order", smooth_order))

    if file_set_info is not None:
        df_dict = ensure_complete_frames(df_dict, file_set_info=file_set_info, fps=fps)

    processed: Dict[str, pd.DataFrame] = {}
    for player, df in df_dict.items():
        out_df = df.copy()
        if "fish_id" not in out_df.columns:
            processed[player] = out_df
            continue

        fish_ids = [fid for fid in out_df["fish_id"].unique() if fid != "unassigned"]
        for fid in fish_ids:
            out_df = fill_missing_values_trajectory(
                out_df,
                fid,
                method=method,
                interp_limit=interp_limit,
                smooth=smooth,
                smooth_window=smooth_window,
                smooth_order=smooth_order,
            )
        processed[player] = out_df

    return processed


__all__ = [
    "extract_time_from_filename",
    "generate_complete_frame_sequence",
    "fill_missing_values_trajectory",
    "ensure_complete_frames",
    "process_all_fish_missing_values",
]
