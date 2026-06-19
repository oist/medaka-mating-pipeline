"""
Step 6: remove 3D outliers and refill trajectories.

For each 3D trajectory (in millimeters), repeatedly:

- detect outliers with a frame-gap-scaled motion threshold
- remove them
- interpolate the missing points

The loop stops when no outliers remain or when the maximum number of
iterations is reached.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ID_LABELS = ["ID_1", "ID_2", "ID_3"]
GROUP_LABELS = ["a", "b", "c"]


def _detect_outliers_frame_gap_scaled(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    max_mm_per_frame: float,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
) -> np.ndarray:
    """Return a boolean outlier mask using a frame-gap-scaled threshold."""
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=bool)
    mask = np.zeros(len(positions), dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=int)
    f_min, f_max = int(frame_indices.min()), int(frame_indices.max())
    tail_threshold_frame = f_max - max(1, int((f_max - f_min) * tail_fraction))
    for i in range(1, len(positions)):
        gap = frame_indices[i] - frame_indices[i - 1]
        if gap <= 0:
            continue
        dist = np.linalg.norm(positions[i] - positions[i - 1])
        threshold = max_mm_per_frame * gap
        if frame_indices[i] >= tail_threshold_frame:
            threshold *= tail_relax_factor
        if dist > threshold:
            mask[i] = True
    return mask


def _check_remaining_outliers_3d(
    df: pd.DataFrame,
    max_mm_per_frame: float,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
) -> bool:
    """Check whether a trajectory still contains outliers."""
    df = df.sort_values("frame")
    pos = df[["X", "Y", "Z"]].values
    frames = df["frame"].values
    valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
    if np.sum(valid) < 2:
        return False
    pos_valid = pos[valid]
    frame_valid = frames[valid]
    f_min, f_max = int(frame_valid.min()), int(frame_valid.max())
    tail_threshold_frame = f_max - max(1, int((f_max - f_min) * tail_fraction))
    for i in range(1, len(pos_valid)):
        gap = int(frame_valid[i] - frame_valid[i - 1])
        if gap <= 0:
            continue
        dist = np.linalg.norm(pos_valid[i] - pos_valid[i - 1])
        threshold = max_mm_per_frame * gap
        if frame_valid[i] >= tail_threshold_frame:
            threshold *= tail_relax_factor
        if dist > threshold:
            return True
    return False


def _interpolate_xyz(df: pd.DataFrame, method: str = "cubic") -> pd.DataFrame:
    """Interpolate X/Y/Z across frame order."""
    df = df.copy().sort_values("frame")
    for col in ["X", "Y", "Z"]:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        if np.any(~np.isnan(vals)):
            if method == "cubic" and np.sum(~np.isnan(vals)) >= 4:
                try:
                    from scipy.interpolate import CubicSpline
                    valid = ~np.isnan(vals)
                    x = np.arange(len(vals))[valid]
                    y = vals[valid]
                    cs = CubicSpline(x, y)
                    nan_idx = np.where(np.isnan(vals))[0]
                    vals[nan_idx] = cs(nan_idx)
                    df[col] = vals
                except Exception:
                    df[col] = pd.Series(vals).interpolate(method="linear", limit_direction="both").values
            else:
                df[col] = pd.Series(vals).interpolate(method="linear", limit_direction="both").values
    return df


def _ensure_complete_frames(df: pd.DataFrame, all_frames: np.ndarray) -> pd.DataFrame:
    """Ensure that a DataFrame contains every frame in `all_frames`."""
    if all_frames is None or len(all_frames) == 0:
        return df
    df = df.sort_values("frame")
    existing = set(df["frame"].astype(int))
    missing = [f for f in all_frames if int(f) not in existing]
    if not missing:
        return df
    rows = [{"frame": int(f), "X": np.nan, "Y": np.nan, "Z": np.nan} for f in missing]
    extra = pd.DataFrame(rows)
    return pd.concat([df, extra], ignore_index=True).sort_values("frame").reset_index(drop=True)


def process_set(
    output_dir: Path,
    set_key: str,
    max_mm_per_frame: float = 10.0,
    use_id_labels: bool = True,
    in_place: bool = True,
    max_iterations: int = 10,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
    interp_method: str = "cubic",
) -> List[Path]:
    """Process one set of 3D CSV files with iterative outlier removal."""
    labels = ID_LABELS if use_id_labels else GROUP_LABELS

    dfs = {}
    all_frames = None
    for g in labels:
        p = output_dir / f"{set_key}_{g}_3d.csv"
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        dfs[g] = pd.read_csv(p)
        frames = dfs[g]["frame"].astype(int).values
        if all_frames is None:
            all_frames = np.unique(frames)
        else:
            all_frames = np.union1d(all_frames, frames)

    if all_frames is not None:
        all_frames = np.sort(all_frames)

    out_suffix = "" if in_place else "_cleaned"
    written = []

    for g in labels:
        df = dfs[g].copy()
        df = _ensure_complete_frames(df, all_frames)
        df = df.sort_values("frame").reset_index(drop=True)

        iteration = 0
        has_outliers = True
        while has_outliers and iteration < max_iterations:
            iteration += 1
            pos = df[["X", "Y", "Z"]].values
            frames = df["frame"].values
            valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
            if np.sum(valid) >= 2:
                pos_valid = pos[valid]
                frame_valid = frames[valid]
                mask_valid = _detect_outliers_frame_gap_scaled(
                    pos_valid, frame_valid, max_mm_per_frame,
                    tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
                )
                outlier_in_full = np.zeros(len(df), dtype=bool)
                outlier_in_full[valid] = mask_valid
                df.loc[outlier_in_full, ["X", "Y", "Z"]] = np.nan
            df = _interpolate_xyz(df, method=interp_method)
            has_outliers = _check_remaining_outliers_3d(
                df, max_mm_per_frame,
                tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
            )

        if iteration >= max_iterations and has_outliers:
            print(f"    Warning: {set_key} {g} reached max_iterations ({max_iterations}) and still contains 3D outliers")

        out_path = output_dir / f"{set_key}_{g}_3d{out_suffix}.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def process_set_single_csv(
    output_dir: Path,
    set_key: str,
    max_mm_per_frame: float = 10.0,
    in_place: bool = True,
    max_iterations: int = 10,
    tail_fraction: float = 0.15,
    tail_relax_factor: float = 2.0,
    interp_method: str = "cubic",
) -> List[Path]:
    """Process one long-format `{set_key}_3d.csv` file fish-by-fish."""
    single_path = output_dir / f"{set_key}_3d.csv"
    if not single_path.exists():
        raise FileNotFoundError(f"File not found: {single_path}")
    full_df = pd.read_csv(single_path)
    if "fish_id" not in full_df.columns:
        raise ValueError(f"{single_path} is missing the fish_id column")
    labels = full_df["fish_id"].unique().tolist()
    dfs = {g: full_df[full_df["fish_id"] == g][["frame", "X", "Y", "Z"]].copy() for g in labels}
    all_frames = np.unique(np.concatenate([df["frame"].astype(int).values for df in dfs.values()]))
    all_frames = np.sort(all_frames)

    out_suffix = "" if in_place else "_cleaned"
    for g in labels:
        df = dfs[g].copy()
        df = _ensure_complete_frames(df, all_frames)
        df = df.sort_values("frame").reset_index(drop=True)
        iteration = 0
        has_outliers = True
        while has_outliers and iteration < max_iterations:
            iteration += 1
            pos = df[["X", "Y", "Z"]].values
            frames = df["frame"].values
            valid = ~(np.isnan(pos[:, 0]) | np.isnan(pos[:, 1]) | np.isnan(pos[:, 2]))
            if np.sum(valid) >= 2:
                pos_valid = pos[valid]
                frame_valid = frames[valid]
                mask_valid = _detect_outliers_frame_gap_scaled(
                    pos_valid, frame_valid, max_mm_per_frame,
                    tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
                )
                outlier_in_full = np.zeros(len(df), dtype=bool)
                outlier_in_full[valid] = mask_valid
                df.loc[outlier_in_full, ["X", "Y", "Z"]] = np.nan
            df = _interpolate_xyz(df, method=interp_method)
            has_outliers = _check_remaining_outliers_3d(
                df, max_mm_per_frame,
                tail_fraction=tail_fraction, tail_relax_factor=tail_relax_factor,
            )
        if iteration >= max_iterations and has_outliers:
            print(f"    Warning: {set_key} {g} reached max_iterations ({max_iterations}) and still contains 3D outliers")
        dfs[g] = df

    out_path = output_dir / f"{set_key}_3d{out_suffix}.csv"
    rows = []
    for fid, df in dfs.items():
        for _, r in df.iterrows():
            rows.append({
                "frame": int(r["frame"]),
                "fish_id": fid,
                "X": float(r["X"]),
                "Y": float(r["Y"]),
                "Z": float(r["Z"]),
            })
    pd.DataFrame(rows).sort_values(["frame", "fish_id"]).to_csv(out_path, index=False)
    return [out_path]
