"""
2D trajectory outlier detection and mirror filtering.

This module focuses only on:

- marking implausible 2D points as outliers
- setting those points to `NaN`
- optionally logging removed points for review

Interpolation and smoothing belong in `missing_value_filling.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Hashable, List

import numpy as np
import pandas as pd


def detect_outliers_frame_gap_scaled(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    max_jump_per_frame: float = 100.0,
) -> np.ndarray:
    """
    Detect single-point outliers using a frame-gap-scaled distance threshold.

    A point is marked as an outlier when:

        distance(previous_valid_point, current_point) > max_jump_per_frame * gap

    where `gap` is the frame interval between the two points.
    """
    n = len(positions)
    if n < 2:
        return np.zeros(n, dtype=bool)

    outlier_mask = np.zeros(n, dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=int)

    valid = ~(np.isnan(positions[:, 0]) | np.isnan(positions[:, 1]))
    if valid.sum() < 2:
        return outlier_mask

    valid_idx = np.where(valid)[0]
    prev_i = valid_idx[0]
    prev_pos = positions[prev_i]
    prev_frame = frame_indices[prev_i]

    for curr_i in valid_idx[1:]:
        curr_pos = positions[curr_i]
        curr_frame = frame_indices[curr_i]
        gap = curr_frame - prev_frame
        if gap <= 0:
            continue

        dist = float(np.linalg.norm(curr_pos - prev_pos))
        threshold = max_jump_per_frame * gap
        if dist > threshold:
            outlier_mask[curr_i] = True
        else:
            prev_i = curr_i
            prev_pos = curr_pos
            prev_frame = curr_frame

    return outlier_mask


def detect_loop_segment_outliers(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    *,
    max_loop_frames: int = 10,
    return_dist_thresh: float = 30.0,
    excursion_dist_thresh: float = 200.0,
) -> np.ndarray:
    """
    Detect short erroneous loop segments that leave a location and quickly
    return to nearly the same place.
    """
    n = len(positions)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out

    frames = np.asarray(frame_indices, dtype=int)
    valid = ~(np.isnan(positions[:, 0]) | np.isnan(positions[:, 1]))
    idx = np.where(valid)[0]
    if idx.size < 3:
        return out

    for ii, i0 in enumerate(idx):
        p0 = positions[i0]
        f0 = frames[i0]
        for jj in range(ii + 2, idx.size):
            i1 = idx[jj]
            f1 = frames[i1]
            if f1 - f0 > max_loop_frames:
                break
            p1 = positions[i1]
            if np.linalg.norm(p1 - p0) > return_dist_thresh:
                continue
            mid_idx = idx[ii + 1 : jj]
            if mid_idx.size == 0:
                continue
            mid_pos = positions[mid_idx]
            dist_to_ends = np.minimum(
                np.linalg.norm(mid_pos - p0, axis=1),
                np.linalg.norm(mid_pos - p1, axis=1),
            )
            if np.all(dist_to_ends > excursion_dist_thresh):
                out[mid_idx] = True

    return out


def _is_mirror_pair(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
) -> bool:
    """Return True when two points look like a reflection pair."""
    if abs(y1 - y2) > y_tol:
        return False
    axis = (x1 + x2) / 2.0
    d1 = x1 - axis
    d2 = x2 - axis
    if d1 * d2 >= 0:
        return False
    if abs(abs(d1) - abs(d2)) > sym_tol:
        return False
    return True


def filter_mirror_in_view(
    df: pd.DataFrame,
    entity_col: str = "fish_id",
    *,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
    player: str | None = None,
    summary: List[dict] | None = None,
) -> pd.DataFrame:
    """
    Detect likely fish/reflection pairs within one view and frame, then set the
    less plausible point to `NaN`.
    """
    if entity_col not in df.columns or "frame" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        return df

    out = df.copy().sort_values("frame")
    last_pos: Dict[Hashable, np.ndarray] = {}
    to_nan = set()

    for frame in out["frame"].unique():
        sub = out[(out["frame"] == frame)].dropna(subset=["x", "y"])
        if len(sub) < 2:
            for _, row in sub.iterrows():
                last_pos[row[entity_col]] = np.array([float(row["x"]), float(row["y"])])
            continue

        rows = sub.index.tolist()
        ids = sub[entity_col].values
        pos = sub[["x", "y"]].values.astype(float)

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if not _is_mirror_pair(
                    pos[i, 0],
                    pos[i, 1],
                    pos[j, 0],
                    pos[j, 1],
                    y_tol=y_tol,
                    sym_tol=sym_tol,
                ):
                    continue

                idi, idj = ids[i], ids[j]
                pi, pj = pos[i], pos[j]

                dist_i = np.inf
                dist_j = np.inf
                if idi in last_pos:
                    dist_i = float(np.linalg.norm(pi - last_pos[idi]))
                if idj in last_pos:
                    dist_j = float(np.linalg.norm(pj - last_pos[idj]))

                if dist_i < dist_j:
                    to_nan.add(rows[j])
                    if summary is not None:
                        row_j = sub.loc[rows[j]]
                        summary.append(
                            {
                                "player": player or "",
                                "fish_id": idj,
                                "frame": int(frame),
                                "x": float(row_j["x"]),
                                "y": float(row_j["y"]),
                                "reason": "mirror",
                            }
                        )
                elif dist_j < dist_i:
                    to_nan.add(rows[i])
                    if summary is not None:
                        row_i = sub.loc[rows[i]]
                        summary.append(
                            {
                                "player": player or "",
                                "fish_id": idi,
                                "frame": int(frame),
                                "x": float(row_i["x"]),
                                "y": float(row_i["y"]),
                                "reason": "mirror",
                            }
                        )

        for idx in sub.index:
            if idx in to_nan:
                continue
            row = out.loc[idx]
            last_pos[row[entity_col]] = np.array([float(row["x"]), float(row["y"])])

    if to_nan:
        out.loc[list(to_nan), ["x", "y"]] = np.nan

    return out


def process_mirror_filter(
    df_dict: Dict[str, pd.DataFrame],
    *,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
    log_dir: str | None = None,
    set_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Apply in-view mirror filtering to all views."""
    summary: List[dict] = []
    processed: Dict[str, pd.DataFrame] = {}

    for player, df in df_dict.items():
        if "fish_id" not in df.columns:
            processed[player] = df.copy()
            continue
        processed[player] = filter_mirror_in_view(
            df,
            entity_col="fish_id",
            y_tol=y_tol,
            sym_tol=sym_tol,
            player=player,
            summary=summary,
        )

    if log_dir and summary:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        name = f"{set_key}_mirror_outliers.csv" if set_key else "mirror_outliers.csv"
        out_csv = log_path / name
        pd.DataFrame(summary).sort_values(["player", "fish_id", "frame"]).to_csv(out_csv, index=False)
        print(f"  Wrote mirror-filter summary: {out_csv}")

    return processed


def _remove_outliers_for_entity(
    df: pd.DataFrame,
    entity_col: str,
    entity_id: Hashable,
    *,
    max_jump_per_frame: float,
    player: str | None = None,
    summary: List[dict] | None = None,
) -> pd.DataFrame:
    """Set outlier points to `NaN` for one entity trajectory."""
    mask_entity = df[entity_col] == entity_id
    entity_df = df.loc[mask_entity].copy().sort_values("frame")
    if len(entity_df) < 2:
        return df

    positions = entity_df[["x", "y"]].to_numpy(dtype=float)
    frames = entity_df["frame"].to_numpy()

    jump_mask = detect_outliers_frame_gap_scaled(
        positions,
        frames,
        max_jump_per_frame=max_jump_per_frame,
    )
    loop_mask = detect_loop_segment_outliers(
        positions,
        frames,
        max_loop_frames=10,
        return_dist_thresh=30.0,
        excursion_dist_thresh=max_jump_per_frame * 2.0,
    )
    combined_mask = jump_mask | loop_mask

    result_df = df.copy()
    outlier_indices = entity_df.index[combined_mask]
    if len(outlier_indices) > 0:
        if summary is not None:
            for idx in outlier_indices:
                row = entity_df.loc[idx]
                summary.append(
                    {
                        "player": player or "",
                        "fish_id": entity_id,
                        "frame": int(row["frame"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                    }
                )
        result_df.loc[outlier_indices, ["x", "y"]] = np.nan

    return result_df


def process_all_fish_outliers(
    df_dict: Dict[str, pd.DataFrame],
    max_jump_distance: float = 200.0,
    iterative: bool = True,
    max_iterations: int = 10,
    use_frame_gap_scaling: bool = True,
    max_jump_per_frame: float = 100.0,
    *,
    log_dir: str | None = None,
    set_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a single-pass outlier scan for every fish in every view.

    Older parameters are still accepted for compatibility, but the active
    logic is intentionally simple.
    """
    jump_thr = float(max_jump_per_frame) if use_frame_gap_scaling else float(max_jump_distance)

    processed: Dict[str, pd.DataFrame] = {}
    summary: List[dict] = []

    for player, df in df_dict.items():
        if "fish_id" not in df.columns:
            processed[player] = df.copy()
            continue

        out_df = df.copy()
        fish_ids = [fid for fid in df["fish_id"].unique() if fid != "unassigned"]
        for fid in fish_ids:
            out_df = _remove_outliers_for_entity(
                out_df,
                entity_col="fish_id",
                entity_id=fid,
                max_jump_per_frame=jump_thr,
                player=player,
                summary=summary,
            )
        processed[player] = out_df

    if log_dir is not None and summary:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        out_csv = log_path / (f"{set_key}_outliers.csv" if set_key else "outliers.csv")
        df_summary = pd.DataFrame(summary)
        df_summary.sort_values(["player", "fish_id", "frame"]).to_csv(out_csv, index=False)
        print(f"  Wrote outlier summary: {out_csv}")

        counts = (
            df_summary.groupby(["player", "fish_id"], as_index=False)["frame"]
            .count()
            .rename(columns={"frame": "n_outliers"})
        )
        summary_csv = log_path / (
            f"{set_key}_outliers_summary.csv" if set_key else "outliers_summary.csv"
        )
        counts.to_csv(summary_csv, index=False)
        print(f"  Wrote outlier count summary: {summary_csv}")

    return processed


__all__ = [
    "detect_outliers_frame_gap_scaled",
    "process_all_fish_outliers",
    "filter_mirror_in_view",
    "process_mirror_filter",
]
