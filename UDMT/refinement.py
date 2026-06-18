from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from UDMT.io import identity_keys, normalize_tracks, require_columns


@dataclass(frozen=True)
class RefinementConfig:
    """Settings for mask-centroid localization refinement."""

    max_distance_px: float = 30.0
    min_mask_area: float = 20.0
    max_mask_area: float | None = None
    min_quality: float = 0.0
    blend_alpha: float = 1.0
    prefer_identity_match: bool = True
    strict_identity_match: bool = False
    identity_mismatch_penalty_px: float = 10000.0


@dataclass(frozen=True)
class RefinementSummary:
    input_rows: int
    mask_rows: int
    refined_rows: int
    unmatched_track_rows: int
    rejected_mask_rows: int
    max_distance_px: float

    def to_dict(self) -> dict:
        return asdict(self)


def refine_tracks_with_masks(
    tracks: pd.DataFrame,
    masks: pd.DataFrame,
    config: RefinementConfig | None = None,
) -> tuple[pd.DataFrame, RefinementSummary]:
    """Refine top-camera track centers by nearest accepted mask centroids.

    Accepted input schemas:
    - tracks: `frame, track_id, x, y` plus optional identity/score columns.
    - masks: `frame` plus either `centroid_x, centroid_y` or `x, y`; optional
      `area`, `quality_score`, `confidence`, `score`, `candidate_id`, and
      identity/status columns.
    """

    cfg = config or RefinementConfig()
    track_df = normalize_tracks(tracks)
    mask_df, rejected_masks = _prepare_masks(masks, cfg)

    refined = track_df.copy()
    refined["pre_refine_x"] = refined["x"].astype(float)
    refined["pre_refine_y"] = refined["y"].astype(float)
    refined["udmt_refined"] = False
    refined["refine_distance_px"] = np.nan
    refined["refine_mask_area"] = np.nan
    refined["refine_candidate_id"] = ""
    refined["refine_source"] = ""

    if track_df.empty or mask_df.empty:
        summary = RefinementSummary(
            input_rows=int(len(track_df)),
            mask_rows=int(len(mask_df)),
            refined_rows=0,
            unmatched_track_rows=int(len(track_df)),
            rejected_mask_rows=int(rejected_masks),
            max_distance_px=float(cfg.max_distance_px),
        )
        return refined, summary

    track_keys = identity_keys(track_df)
    mask_keys = identity_keys(mask_df)
    refined_count = 0

    for frame, frame_tracks in track_df.groupby("frame", sort=False):
        frame_masks = mask_df[mask_df["frame"] == frame]
        if frame_masks.empty:
            continue

        track_xy = frame_tracks[["x", "y"]].to_numpy(dtype=float)
        mask_xy = frame_masks[["_mask_x", "_mask_y"]].to_numpy(dtype=float)
        cost = np.linalg.norm(track_xy[:, None, :] - mask_xy[None, :, :], axis=2)
        raw_distance = cost.copy()

        if cfg.prefer_identity_match:
            tk = track_keys.loc[frame_tracks.index].to_numpy(dtype=object)
            mk = mask_keys.loc[frame_masks.index].to_numpy(dtype=object)
            mismatch = (tk[:, None] != "") & (mk[None, :] != "") & (tk[:, None] != mk[None, :])
            if cfg.strict_identity_match:
                cost[mismatch] = np.inf
            else:
                cost[mismatch] += float(cfg.identity_mismatch_penalty_px)

        finite = np.isfinite(cost)
        if not finite.any():
            continue
        lap_cost = cost.copy()
        lap_cost[~finite] = 1e12
        rows, cols = linear_sum_assignment(lap_cost)
        for ri, ci in zip(rows, cols):
            if not np.isfinite(cost[ri, ci]):
                continue
            if raw_distance[ri, ci] > cfg.max_distance_px:
                continue
            track_index = frame_tracks.index[ri]
            mask_row = frame_masks.iloc[ci]
            alpha = float(np.clip(cfg.blend_alpha, 0.0, 1.0))
            refined.at[track_index, "x"] = (
                (1.0 - alpha) * float(refined.at[track_index, "x"])
                + alpha * float(mask_row["_mask_x"])
            )
            refined.at[track_index, "y"] = (
                (1.0 - alpha) * float(refined.at[track_index, "y"])
                + alpha * float(mask_row["_mask_y"])
            )
            refined.at[track_index, "udmt_refined"] = True
            refined.at[track_index, "refine_distance_px"] = float(raw_distance[ri, ci])
            refined.at[track_index, "refine_mask_area"] = float(mask_row.get("area", np.nan))
            refined.at[track_index, "refine_candidate_id"] = str(mask_row["_candidate_id"])
            refined.at[track_index, "refine_source"] = str(mask_row.get("_source", "mask_centroid"))
            refined_count += 1

    summary = RefinementSummary(
        input_rows=int(len(track_df)),
        mask_rows=int(len(mask_df)),
        refined_rows=int(refined_count),
        unmatched_track_rows=int(len(track_df) - refined_count),
        rejected_mask_rows=int(rejected_masks),
        max_distance_px=float(cfg.max_distance_px),
    )
    return refined.sort_values(["frame", "track_id"]).reset_index(drop=True), summary


def _prepare_masks(
    masks: pd.DataFrame,
    cfg: RefinementConfig,
) -> tuple[pd.DataFrame, int]:
    if "centroid_x" in masks.columns and "centroid_y" in masks.columns:
        x_col, y_col = "centroid_x", "centroid_y"
    else:
        x_col, y_col = "x", "y"
    require_columns(masks, ("frame", x_col, y_col), label="masks")

    out = masks.copy()
    before = len(out)
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out["_mask_x"] = pd.to_numeric(out[x_col], errors="coerce")
    out["_mask_y"] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=["frame", "_mask_x", "_mask_y"])
    out["frame"] = out["frame"].astype(int)

    if "area" in out.columns:
        out["area"] = pd.to_numeric(out["area"], errors="coerce")
        out = out[out["area"].fillna(0.0) >= float(cfg.min_mask_area)]
        if cfg.max_mask_area is not None:
            out = out[out["area"].fillna(np.inf) <= float(cfg.max_mask_area)]
    else:
        out["area"] = np.nan

    quality = _first_numeric_column(out, ("quality_score", "confidence", "score"), default=1.0)
    out["_quality"] = quality
    out = out[out["_quality"] >= float(cfg.min_quality)]

    for status_col in ("status", "qc_status"):
        if status_col in out.columns:
            status = out[status_col].fillna("").astype(str).str.lower()
            out = out[~status.isin(["reject", "rejected", "bad"])]

    out["_candidate_id"] = _first_text_column(
        out,
        ("candidate_id", "tracklet_id", "source_tracklet_id", "identity_trajectory_id"),
        default_prefix="mask",
    )
    out["_source"] = _first_text_column(out, ("source", "node"), default_prefix="mask_centroid")
    rejected = before - len(out)
    return out.reset_index(drop=True), int(rejected)


def _first_numeric_column(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    default: float,
) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _first_text_column(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    default_prefix: str,
) -> pd.Series:
    value = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in columns:
        if col not in df.columns:
            continue
        candidate = df[col].fillna("").astype(str).str.strip()
        value = value.mask(value == "", candidate)
    fallback = pd.Series([f"{default_prefix}_{i}" for i in range(len(df))], index=df.index)
    return value.mask(value == "", fallback)
