"""
Expand each 3D trajectory to all frames and fill missing points with linear
interpolation when requested.

IDs are `ID_1`, `ID_2`, and `ID_3`.
Missing values stay as `NaN`; they are never silently converted to zero.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# Keep ID labels aligned with `cross_frame_tracking.py`.
ID_LABELS = ["ID_1", "ID_2", "ID_3"]


def _id_labels_for(n: int) -> List[str]:
    return [f"ID_{i + 1}" for i in range(n)]


def best_by_frame_to_dfs(
    best_by_frame: Dict[int, Dict],
    *,
    shift_to_positive: bool = True,
    ensure_non_negative: bool = False,
    num_individuals: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convert `best_by_frame` into `{label: DataFrame(frame, X, Y, Z)}`.

    Only valid points are written immediately. Missing points are left for the
    later fill stage.
    """
    id_labels = _id_labels_for(num_individuals) if num_individuals is not None else ID_LABELS
    dfs: Dict[str, pd.DataFrame] = {lb: [] for lb in id_labels}
    for frame, info in sorted(best_by_frame.items()):
        groups = info.get("groups", {})
        for lb in id_labels:
            pt = groups.get(lb)
            if pt is not None and len(pt) == 3 and not (np.isnan(pt).any() or np.isinf(pt).any()):
                dfs[lb].append({"frame": frame, "X": pt[0], "Y": pt[1], "Z": pt[2]})

    out = {}
    for lb in id_labels:
        if dfs[lb]:
            out[lb] = pd.DataFrame(dfs[lb])
        else:
            out[lb] = pd.DataFrame(columns=["frame", "X", "Y", "Z"])

    if shift_to_positive and out:
        arrs = [df[["X", "Y", "Z"]].values for df in out.values() if not df.empty]
        if arrs:
            all_xyz = np.vstack(arrs)
        else:
            all_xyz = np.array([]).reshape(0, 3)
        if all_xyz.size > 0:
            min_vals = np.nanmin(all_xyz, axis=0)
            for lb in out:
                if not out[lb].empty:
                    out[lb] = out[lb].copy()
                    out[lb][["X", "Y", "Z"]] -= min_vals

    if ensure_non_negative and out:
        for lb in out:
            if not out[lb].empty:
                out[lb] = out[lb].copy()
                out[lb][["X", "Y", "Z"]] = out[lb][["X", "Y", "Z"]].clip(lower=0)
    return out  # Do not clip by default; keep invalid values away from 0.


def ensure_all_frames_and_fill_spline(
    dfs_3d: Dict[str, pd.DataFrame],
    all_frames: List[int],
    max_gap_frames: int | None = None,
    *,
    ensure_non_negative: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Expand each trajectory to `all_frames` and linearly interpolate internal
    gaps. Frames outside the observed range remain `NaN`.

    `max_gap_frames`: do NOT interpolate across a gap longer than this many frames
    between consecutive *observed* points — those frames stay `NaN` (a transient
    detection then a long silence must not be bridged to the next, unrelated
    detection — the cause of the entry-phantom artifact). `None` = no cap (legacy).

    `ensure_non_negative`: clip finite coordinates to zero or above after
    interpolation. Keep this enabled for legacy shifted-coordinate workflows;
    disable it when the caller requests preservation of calibrated negative
    coordinates.

    Every output frame carries an `is_interpolated` flag (True only where a value was
    filled, i.e. finite-after-fill and not an originally-observed frame), so downstream
    ID estimation can tell real geometry from interpolated and weight it accordingly.
    """
    if not all_frames:
        return dfs_3d
    all_frames = sorted(all_frames)
    all_frames_arr = np.array(all_frames, dtype=float)
    result = {}
    for lb, df in dfs_3d.items():
        if df.empty:
            result[lb] = pd.DataFrame({
                "frame": all_frames,
                "X": np.nan, "Y": np.nan, "Z": np.nan, "is_interpolated": False,
            })
            result[lb] = result[lb].sort_values("frame").reset_index(drop=True)
            continue
        if len(df) < 2:
            out = df.copy()
            if "is_interpolated" not in out.columns:
                out["is_interpolated"] = False
            result[lb] = out
            continue
        frames = df["frame"].values.astype(float)
        xyz = df[["X", "Y", "Z"]].values
        _, idx = np.unique(frames, return_index=True)
        frames = frames[idx]
        xyz = xyz[idx]
        sort_idx = np.argsort(frames)
        frames = frames[sort_idx]
        xyz = xyz[sort_idx]
        if len(frames) < 2:
            out = df.copy()
            if "is_interpolated" not in out.columns:
                out["is_interpolated"] = False
            result[lb] = out
            continue
        f_min, f_max = float(frames.min()), float(frames.max())
        # Interpolate only inside the observed frame range.
        interp_x = np.interp(all_frames_arr, frames, xyz[:, 0])
        interp_y = np.interp(all_frames_arr, frames, xyz[:, 1])
        interp_z = np.interp(all_frames_arr, frames, xyz[:, 2])
        out_of_range = (all_frames_arr < f_min) | (all_frames_arr > f_max)
        observed = np.isin(all_frames_arr, frames)
        # Frames sitting inside a gap longer than the cap are left NaN, not bridged.
        in_big_gap = np.zeros(len(all_frames_arr), dtype=bool)
        if max_gap_frames is not None:
            li = np.clip(np.searchsorted(frames, all_frames_arr, side="right") - 1, 0, len(frames) - 1)
            left = frames[li]
            right = frames[np.clip(li + 1, 0, len(frames) - 1)]
            gap = right - left
            in_big_gap = (~observed) & (~out_of_range) & (gap > max_gap_frames)
        nan_mask = out_of_range | in_big_gap
        interp_x = np.where(nan_mask, np.nan, interp_x)
        interp_y = np.where(nan_mask, np.nan, interp_y)
        interp_z = np.where(nan_mask, np.nan, interp_z)
        is_interp = (~observed) & (~nan_mask)
        out_df = pd.DataFrame({
            "frame": all_frames,
            "X": interp_x, "Y": interp_y, "Z": interp_z,
            "is_interpolated": is_interp,
        })
        if ensure_non_negative:
            # Clip finite values to zero or above while preserving NaNs.
            for col in ["X", "Y", "Z"]:
                out_df[col] = out_df[col].where(
                    ~np.isfinite(out_df[col]),
                    out_df[col].clip(lower=0),
                )
        result[lb] = out_df.sort_values("frame").reset_index(drop=True)
    return result


def dfs_to_single_csv(
    dfs_3d: Dict[str, pd.DataFrame],
    set_key: str,
    output_dir: str,
    *,
    ensure_non_negative: bool = False,
) -> str:
    """Merge the per-ID DataFrames into one CSV with frame, fish_id, X, Y, Z."""
    from pathlib import Path
    rows = []
    for fish_id, df in dfs_3d.items():
        if df.empty:
            continue
        has_interp = "is_interpolated" in df.columns
        for _, r in df.iterrows():
            x, y, z = float(r["X"]), float(r["Y"]), float(r["Z"])
            if ensure_non_negative and np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                x, y, z = max(0, x), max(0, y), max(0, z)
            row = {
                "frame": int(r["frame"]),
                "fish_id": fish_id,
                "X": x,
                "Y": y,
                "Z": z,
            }
            if has_interp:
                row["is_interpolated"] = bool(r["is_interpolated"])
            rows.append(row)
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_path = Path(output_dir) / f"{set_key}_3d.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        return str(out_path)
    out_df = out_df.sort_values(["frame", "fish_id"]).reset_index(drop=True)
    out_path = Path(output_dir) / f"{set_key}_3d.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return str(out_path)
