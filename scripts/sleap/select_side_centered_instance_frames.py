"""Select leakage-safe side-view frames for centered-instance HITL training.

This is a two-stage active-learning workflow:

1. ``mine`` uses complete Round-7 side predictions to build a broad,
   experiment-balanced pool without running full-video inference again.
2. Run the current model only on that sparse pool with
   :mod:`scripts.sleap.prepare_prediction_review_candidates`.
3. ``finalize`` combines current-model evidence with the Round-7 features,
   writes the final train/validation manifest, and builds a blank embedded
   package for human review.

Predictions are never converted into authoritative labels. The blank package
and the separate current-model candidate overlay are intentionally distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths
from scripts.sleap.mine_round8_hard_frames import (
    FeatureConfig,
    load_or_build_view_features,
)
from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
)


EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
SIDE_VIEWS = ("side1_cam", "side2_cam")
RANDOM_SEED = 20260727
SELECTION_VERSION = "side-centered-instance-hitl-v1"

DEFAULT_TRAIN_QUOTAS: dict[str, dict[str, int]] = {
    "20251225_K_1_1_FCMC": {
        "round7_round8_tail_disagreement": 17,
        "temporal_or_shared_tail": 14,
        "close_contact_or_crossing": 17,
        "long_body_crop_margin_proxy": 11,
        "count_partial_or_low_confidence": 7,
        "clean_diverse_control": 4,
    },
    "20251225_K_1_2_FCMC": {
        "round7_round8_tail_disagreement": 10,
        "temporal_or_shared_tail": 8,
        "close_contact_or_crossing": 10,
        "long_body_crop_margin_proxy": 5,
        "count_partial_or_low_confidence": 5,
        "clean_diverse_control": 2,
    },
    "20251225_K_1_3_FCMC": {
        "round7_round8_tail_disagreement": 18,
        "temporal_or_shared_tail": 14,
        "close_contact_or_crossing": 18,
        "long_body_crop_margin_proxy": 11,
        "count_partial_or_low_confidence": 6,
        "clean_diverse_control": 3,
    },
}
DEFAULT_VALIDATION_QUOTAS = {
    "clean_diverse_control": 4,
    "close_contact_or_crossing": 2,
    "round7_round8_tail_disagreement": 2,
    "temporal_or_shared_tail": 1,
    "count_partial_or_low_confidence": 1,
}
DEFAULT_POOL_QUOTAS = {
    "tail_confidence": 45,
    "temporal_or_shared_tail": 40,
    "close_contact_or_crossing": 40,
    "long_body_crop_margin_proxy": 30,
    "count_partial_or_low_confidence": 20,
    "clean_diverse_control": 35,
}


@dataclass(frozen=True)
class SelectionConfig:
    seed: int = RANDOM_SEED
    skip_initial_frames: int = 49_800
    time_bins: int = 30
    selection_spacing_frames: int = 166
    training_overlap_guard_frames: int = 20
    locked_guard_frames: int = 166
    planned_crop_size: int = 448
    crop_ambiguity_px: int = 16
    plausible_score: float = 0.20


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _stable_tie(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .isin({"1", "true", "yes", "y"})
    )


def _rank_high(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    return numeric.rank(method="average", pct=True).fillna(0.0)


def _rank_low(series: pd.Series) -> pd.Series:
    return _rank_high(-pd.to_numeric(series, errors="coerce"))


def _merge_intervals(
    intervals: Iterable[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    ordered = sorted(
        (int(min(start, end)), int(max(start, end)), str(reason))
        for start, end, reason in intervals
    )
    merged: list[tuple[int, int, str]] = []
    for start, end, reason in ordered:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, reason))
            continue
        old_start, old_end, old_reason = merged[-1]
        reasons = sorted(set(old_reason.split(";") + reason.split(";")))
        merged[-1] = (old_start, max(old_end, end), ";".join(reasons))
    return merged


def _interval_mask(
    frames: np.ndarray, intervals: Sequence[tuple[int, int, str]]
) -> np.ndarray:
    result = np.zeros(len(frames), dtype=bool)
    for start, end, _reason in intervals:
        result |= (frames >= int(start)) & (frames <= int(end))
    return result


def _add_manifest_intervals(
    destination: dict[str, list[tuple[int, int, str]]],
    path: Path,
    *,
    reason: str,
    guard_frames: int,
    eligibility_column: str | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"exclusion manifest not found: {path}")
    table = pd.read_csv(path)
    if eligibility_column is not None:
        if eligibility_column not in table:
            raise ValueError(
                f"{path} is missing eligibility column {eligibility_column!r}"
            )
        table = table.loc[_as_bool(table[eligibility_column])]
    required = {"experiment", "frame_idx"}
    missing = required - set(table)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for row in table.drop_duplicates(
        ["experiment", "frame_idx"], keep="first"
    ).to_dict("records"):
        experiment = str(row["experiment"])
        start = row.get("event_start_frame", row["frame_idx"])
        end = row.get("event_end_frame", row["frame_idx"])
        if pd.isna(start):
            start = row["frame_idx"]
        if pd.isna(end):
            end = row["frame_idx"]
        destination[experiment].append(
            (
                int(start) - int(guard_frames),
                int(end) + int(guard_frames),
                reason,
            )
        )


def _add_existing_label_intervals(
    destination: dict[str, list[tuple[int, int, str]]],
    path: Path,
    *,
    guard_frames: int,
    reason: str,
) -> int:
    import sleap_io as sio

    if not path.is_file():
        raise FileNotFoundError(f"existing label package not found: {path}")
    labels = sio.load_slp(str(path), open_videos=False, lazy=True)
    count = 0
    try:
        for labeled_frame in labels.labeled_frames:
            source = _video_source_path(labeled_frame.video)
            experiment = experiment_from_video(source)
            view = classify_view(source)
            if experiment not in EXPERIMENTS or view not in SIDE_VIEWS:
                continue
            frame = int(labeled_frame.frame_idx)
            destination[experiment].append(
                (
                    frame - int(guard_frames),
                    frame + int(guard_frames),
                    reason,
                )
            )
            count += 1
    finally:
        labels.close()
    return count


def build_exclusion_intervals(
    *,
    locked_manifest: Path,
    future_locked_manifest: Path,
    previous_selection_manifest: Path,
    existing_train_labels: Path,
    existing_val_labels: Path,
    config: SelectionConfig,
) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, Any]]:
    """Build experiment-specific intervals protected from active learning."""
    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    _add_manifest_intervals(
        intervals,
        locked_manifest,
        reason="locked_hitl_v2",
        guard_frames=config.locked_guard_frames,
    )
    _add_manifest_intervals(
        intervals,
        future_locked_manifest,
        reason="future_locked_test",
        guard_frames=config.locked_guard_frames,
        eligibility_column="future_lock_eligible",
    )

    previous = pd.read_csv(previous_selection_manifest)
    required = {"experiment", "frame", "split"}
    missing = required - set(previous)
    if missing:
        raise ValueError(
            f"{previous_selection_manifest} is missing columns: {sorted(missing)}"
        )
    for row in previous.to_dict("records"):
        experiment = str(row["experiment"])
        frame = int(row["frame"])
        if str(row["split"]) == "holdout":
            start = int(row.get("event_start_frame", frame))
            end = int(row.get("event_end_frame", frame))
            guard = config.locked_guard_frames
            reason = "round8_locked_holdout"
        else:
            start = end = frame
            guard = config.training_overlap_guard_frames
            reason = "existing_round8_training_selection"
        intervals[experiment].append(
            (start - guard, end + guard, reason)
        )

    train_lf_count = _add_existing_label_intervals(
        intervals,
        existing_train_labels,
        guard_frames=config.training_overlap_guard_frames,
        reason="existing_side_training_label",
    )
    val_lf_count = _add_existing_label_intervals(
        intervals,
        existing_val_labels,
        guard_frames=config.locked_guard_frames,
        reason="existing_side_validation_label",
    )
    intervals["20251225_K_1_2_FCMC"].append(
        (
            96_612 - config.locked_guard_frames,
            97_802 + config.locked_guard_frames,
            "round8_sentinel_event",
        )
    )
    merged = {
        experiment: _merge_intervals(intervals.get(experiment, []))
        for experiment in EXPERIMENTS
    }
    report = {
        "sources": {
            "locked_manifest": _fingerprint(locked_manifest),
            "future_locked_manifest": _fingerprint(future_locked_manifest),
            "previous_selection_manifest": _fingerprint(
                previous_selection_manifest
            ),
            "existing_train_labels": _fingerprint(existing_train_labels),
            "existing_val_labels": _fingerprint(existing_val_labels),
        },
        "existing_side_train_labeled_frames_read": train_lf_count,
        "existing_side_val_labeled_frames_read": val_lf_count,
        "merged_interval_counts": {
            experiment: len(merged[experiment]) for experiment in EXPERIMENTS
        },
    }
    return merged, report


def _minimum_pair_distance(
    table: pd.DataFrame, *, x_column: str, y_column: str
) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=float)
    maximum_instances = int(table["instance_index"].max()) + 1
    n_frames = int(table["frame"].max()) + 1
    xs = np.full((n_frames, maximum_instances), np.nan, dtype=float)
    ys = np.full_like(xs, np.nan)
    frame = table["frame"].to_numpy(dtype=np.int64)
    slot = table["instance_index"].to_numpy(dtype=np.int64)
    xs[frame, slot] = table[x_column].to_numpy(dtype=float)
    ys[frame, slot] = table[y_column].to_numpy(dtype=float)
    minimum = np.full(n_frames, np.inf, dtype=float)
    for first in range(maximum_instances):
        for second in range(first + 1, maximum_instances):
            distance = np.hypot(
                xs[:, first] - xs[:, second],
                ys[:, first] - ys[:, second],
            )
            # Ignore pairs whose slot is absent in this frame. A plain
            # ``minimum`` would propagate NaN from any globally allocated but
            # locally unused slot.
            minimum = np.fmin(minimum, distance)
    minimum[~np.isfinite(minimum)] = np.nan
    return pd.Series(minimum, index=np.arange(n_frames))


def aggregate_view_features(
    frames: pd.DataFrame,
    instances: pd.DataFrame,
    *,
    view: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add centered-instance-specific tail, body, and shared-tail signals."""
    table = frames.copy().set_index("frame", drop=False)
    plausible = instances.loc[instances["plausible"].astype(bool)].copy()
    plausible["body_length_px"] = np.hypot(
        plausible["tail_x"] - plausible["mouth_x"],
        plausible["tail_y"] - plausible["mouth_y"],
    )
    length = plausible["body_length_px"].to_numpy(dtype=float)
    finite_length = length[np.isfinite(length)]
    median = float(np.median(finite_length)) if len(finite_length) else np.nan
    mad = (
        float(np.median(np.abs(finite_length - median)))
        if len(finite_length)
        else np.nan
    )
    scale = max(1.0, 1.4826 * mad) if np.isfinite(mad) else 1.0
    plausible["body_length_robust_z"] = (
        (plausible["body_length_px"] - median).abs() / scale
    )
    plausible["mouth_tail_score_gap"] = (
        plausible["mouth_score"] - plausible["tail_score"]
    )

    tracked = plausible.loc[
        plausible["track_id"].astype(str).ne("")
        & np.isfinite(plausible["tail_x"])
        & np.isfinite(plausible["tail_y"])
    ].sort_values(["track_id", "frame"], kind="stable")
    if tracked.empty:
        tracked["tail_midpoint_residual_px"] = np.nan
    else:
        grouped = tracked.groupby("track_id", sort=False)
        previous_frame = grouped["frame"].shift(1)
        next_frame = grouped["frame"].shift(-1)
        previous_x = grouped["tail_x"].shift(1)
        previous_y = grouped["tail_y"].shift(1)
        next_x = grouped["tail_x"].shift(-1)
        next_y = grouped["tail_y"].shift(-1)
        exact = (
            tracked["frame"].sub(previous_frame).eq(1)
            & next_frame.sub(tracked["frame"]).eq(1)
        )
        residual = np.hypot(
            tracked["tail_x"] - 0.5 * (previous_x + next_x),
            tracked["tail_y"] - 0.5 * (previous_y + next_y),
        )
        tracked["tail_midpoint_residual_px"] = residual.where(exact)

    aggregates = plausible.groupby("frame", sort=False).agg(
        minimum_tail_score=("tail_score", "min"),
        minimum_mouth_score=("mouth_score", "min"),
        maximum_mouth_tail_score_gap=("mouth_tail_score_gap", "max"),
        maximum_body_length_px=("body_length_px", "max"),
        maximum_body_length_robust_z=("body_length_robust_z", "max"),
    )
    tail_residual = tracked.groupby("frame", sort=False)[
        "tail_midpoint_residual_px"
    ].max()
    tail_distance = _minimum_pair_distance(
        plausible.loc[
            np.isfinite(plausible["tail_x"])
            & np.isfinite(plausible["tail_y"])
        ],
        x_column="tail_x",
        y_column="tail_y",
    ).rename("minimum_tail_distance_px")
    table = table.join(aggregates, how="left")
    table = table.join(tail_residual, how="left")
    table = table.join(tail_distance, how="left")
    keep = [
        "frame",
        "instance_count",
        "plausible_count",
        "partial_instance_count",
        "minimum_mouth_distance_px",
        "minimum_score",
        "maximum_score",
        "score_std",
        "score_range",
        "maximum_xy_residual_px",
        "outlier_count",
        "minimum_tail_score",
        "minimum_mouth_score",
        "maximum_mouth_tail_score_gap",
        "maximum_body_length_px",
        "maximum_body_length_robust_z",
        "tail_midpoint_residual_px",
        "minimum_tail_distance_px",
    ]
    renamed = table[keep].rename(
        columns={
            column: f"{view}_{column}"
            for column in keep
            if column != "frame"
        }
    )
    pool_instances = plausible[
        [
            "frame",
            "instance_index",
            "mouth_x",
            "mouth_y",
            "tail_x",
            "tail_y",
            "mouth_score",
            "tail_score",
            "instance_score",
            "visible_node_count",
            "body_length_px",
        ]
    ].copy()
    pool_instances["view"] = view
    return renamed.reset_index(drop=True), pool_instances


def combine_side_features(
    side1: pd.DataFrame,
    side2: pd.DataFrame,
    *,
    experiment: str,
    intervals: Sequence[tuple[int, int, str]],
    config: SelectionConfig,
) -> pd.DataFrame:
    table = side1.merge(side2, on="frame", how="inner", validate="one_to_one")
    table["experiment"] = experiment
    frames = table["frame"].to_numpy(dtype=np.int64)
    table["excluded"] = _interval_mask(frames, intervals)
    table["eligible"] = (
        ~table["excluded"]
        & table["frame"].ge(config.skip_initial_frames)
    )
    table["r7_count_or_partial"] = False
    for view in SIDE_VIEWS:
        table["r7_count_or_partial"] |= (
            table[f"{view}_plausible_count"].ne(3)
            | table[f"{view}_partial_instance_count"].gt(0)
        )
    table["r7_good_three_both_views"] = ~table["r7_count_or_partial"]

    def maximum(column: str) -> pd.Series:
        return table[
            [f"{view}_{column}" for view in SIDE_VIEWS]
        ].max(axis=1, skipna=True)

    def minimum(column: str) -> pd.Series:
        return table[
            [f"{view}_{column}" for view in SIDE_VIEWS]
        ].min(axis=1, skipna=True)

    table["r7_minimum_tail_score"] = minimum("minimum_tail_score")
    table["r7_minimum_mouth_score"] = minimum("minimum_mouth_score")
    table["r7_maximum_mouth_tail_score_gap"] = maximum(
        "maximum_mouth_tail_score_gap"
    )
    table["r7_maximum_tail_midpoint_residual_px"] = maximum(
        "tail_midpoint_residual_px"
    )
    table["r7_maximum_xy_residual_px"] = maximum(
        "maximum_xy_residual_px"
    )
    table["r7_minimum_mouth_distance_px"] = minimum(
        "minimum_mouth_distance_px"
    )
    table["r7_minimum_tail_distance_px"] = minimum(
        "minimum_tail_distance_px"
    )
    table["r7_maximum_body_length_px"] = maximum(
        "maximum_body_length_px"
    )
    table["r7_maximum_body_length_robust_z"] = maximum(
        "maximum_body_length_robust_z"
    )
    eligible = table["eligible"]
    table["tail_confidence_score"] = 0.0
    table.loc[eligible, "tail_confidence_score"] = (
        _rank_low(table.loc[eligible, "r7_minimum_tail_score"])
        + _rank_high(
            table.loc[eligible, "r7_maximum_mouth_tail_score_gap"]
        )
    ) / 2.0
    table["temporal_or_shared_tail_score"] = 0.0
    table.loc[eligible, "temporal_or_shared_tail_score"] = (
        _rank_high(
            table.loc[eligible, "r7_maximum_tail_midpoint_residual_px"]
        )
        + _rank_low(
            table.loc[eligible, "r7_minimum_tail_distance_px"]
        )
    ) / 2.0
    table["close_contact_score"] = 0.0
    table.loc[eligible, "close_contact_score"] = _rank_low(
        table.loc[eligible, "r7_minimum_mouth_distance_px"]
    )
    table["crop_margin_proxy_score"] = 0.0
    table.loc[eligible, "crop_margin_proxy_score"] = (
        _rank_high(
            table.loc[eligible, "r7_maximum_body_length_px"]
        )
        + _rank_high(
            table.loc[eligible, "r7_maximum_body_length_robust_z"]
        )
    ) / 2.0
    table["count_partial_low_confidence_score"] = (
        table["r7_count_or_partial"].astype(float)
        + _rank_low(table["r7_minimum_tail_score"])
    ) / 2.0
    table["clean_control_score"] = 0.0
    table.loc[eligible, "clean_control_score"] = (
        _rank_high(table.loc[eligible, "r7_minimum_tail_score"])
        + _rank_high(
            table.loc[eligible, "r7_minimum_mouth_distance_px"]
        )
        + _rank_low(
            table.loc[eligible, "r7_maximum_tail_midpoint_residual_px"]
        )
    ) / 3.0
    crop_limit = (
        config.planned_crop_size / 2.0 - config.crop_ambiguity_px
    )
    table["planned_crop_margin_risk"] = (
        table["r7_maximum_body_length_px"].ge(crop_limit)
    )
    table["shared_tail_prediction"] = (
        table["r7_minimum_tail_distance_px"].le(5.0)
    )
    table["time_bin"] = np.minimum(
        config.time_bins - 1,
        (
            table["frame"]
            / max(1, int(table["frame"].max()) + 1)
            * config.time_bins
        ).astype(int),
    )
    return table


def _diverse_pick(
    candidates: pd.DataFrame,
    *,
    wanted: int,
    score_column: str,
    selected_frames: Sequence[int],
    spacing_frames: int,
    seed: int,
) -> pd.DataFrame:
    if wanted <= 0:
        return candidates.iloc[:0].copy()
    work = candidates.copy()
    if work.empty:
        return work
    work["_tie"] = [
        _stable_tie(seed, row.experiment, int(row.frame), score_column)
        for row in work.itertuples()
    ]
    work["_bin_rank"] = work.groupby("time_bin", sort=True)[
        score_column
    ].rank(method="first", ascending=False)
    work = work.sort_values(
        ["_bin_rank", "time_bin", score_column, "_tie"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    chosen: list[int] = []
    occupied = [int(frame) for frame in selected_frames]
    for spacing in (spacing_frames, max(20, spacing_frames // 2), 20, 0):
        for index, row in work.iterrows():
            if index in chosen:
                continue
            frame = int(row["frame"])
            if spacing and any(abs(frame - other) <= spacing for other in occupied):
                continue
            chosen.append(index)
            occupied.append(frame)
            if len(chosen) == wanted:
                return work.loc[chosen].drop(
                    columns=["_tie", "_bin_rank"]
                )
    return work.loc[chosen].drop(columns=["_tie", "_bin_rank"])


def select_coarse_pool(
    frame_table: pd.DataFrame,
    *,
    quotas: Mapping[str, int] = DEFAULT_POOL_QUOTAS,
    config: SelectionConfig,
) -> pd.DataFrame:
    """Select one experiment's broad Round-7 candidate pool."""
    eligible = frame_table.loc[frame_table["eligible"]].copy()
    good = eligible["r7_good_three_both_views"]
    masks = {
        "tail_confidence": good
        & eligible["tail_confidence_score"].ge(
            eligible.loc[good, "tail_confidence_score"].quantile(0.70)
        ),
        "temporal_or_shared_tail": good
        & (
            eligible["temporal_or_shared_tail_score"].ge(
                eligible.loc[
                    good, "temporal_or_shared_tail_score"
                ].quantile(0.70)
            )
            | eligible["shared_tail_prediction"]
        ),
        "close_contact_or_crossing": good
        & eligible["close_contact_score"].ge(
            eligible.loc[good, "close_contact_score"].quantile(0.70)
        ),
        "long_body_crop_margin_proxy": good
        & (
            eligible["crop_margin_proxy_score"].ge(
                eligible.loc[
                    good, "crop_margin_proxy_score"
                ].quantile(0.70)
            )
            | eligible["planned_crop_margin_risk"]
        ),
        "count_partial_or_low_confidence": (
            eligible["r7_count_or_partial"]
            | eligible["count_partial_low_confidence_score"].ge(0.75)
        ),
        "clean_diverse_control": good
        & eligible["clean_control_score"].ge(
            eligible.loc[good, "clean_control_score"].quantile(0.55)
        ),
    }
    score_columns = {
        "tail_confidence": "tail_confidence_score",
        "temporal_or_shared_tail": "temporal_or_shared_tail_score",
        "close_contact_or_crossing": "close_contact_score",
        "long_body_crop_margin_proxy": "crop_margin_proxy_score",
        "count_partial_or_low_confidence": (
            "count_partial_low_confidence_score"
        ),
        "clean_diverse_control": "clean_control_score",
    }
    selected: list[pd.DataFrame] = []
    selected_frames: list[int] = []
    for offset, (category, wanted) in enumerate(quotas.items()):
        available = eligible.loc[
            masks[category] & ~eligible["frame"].isin(selected_frames)
        ]
        picked = _diverse_pick(
            available,
            wanted=int(wanted),
            score_column=score_columns[category],
            selected_frames=selected_frames,
            spacing_frames=config.selection_spacing_frames,
            seed=config.seed + offset,
        )
        if len(picked) < wanted:
            fallback = eligible.loc[
                ~eligible["frame"].isin(
                    selected_frames + picked["frame"].astype(int).tolist()
                )
            ].copy()
            fallback["_fallback_score"] = fallback[
                list(score_columns.values())
            ].max(axis=1)
            extra = _diverse_pick(
                fallback,
                wanted=int(wanted) - len(picked),
                score_column="_fallback_score",
                selected_frames=(
                    selected_frames
                    + picked["frame"].astype(int).tolist()
                ),
                spacing_frames=config.selection_spacing_frames,
                seed=config.seed + 100 + offset,
            ).drop(columns=["_fallback_score"])
            picked = pd.concat([picked, extra], ignore_index=False)
        if len(picked) != wanted:
            raise ValueError(
                f"could not select {wanted} {category} frames; got {len(picked)}"
            )
        picked = picked.copy()
        picked["pool_category"] = category
        picked["pool_category_score"] = picked[score_columns[category]]
        selected.append(picked)
        selected_frames.extend(picked["frame"].astype(int).tolist())
    result = pd.concat(selected, ignore_index=True)
    if len(result) != sum(int(value) for value in quotas.values()):
        raise ValueError("coarse pool total does not match requested quotas")
    return result.sort_values("frame", kind="stable").reset_index(drop=True)


def _prediction_rows(labels_or_path: Any) -> pd.DataFrame:
    import sleap_io as sio

    labels = (
        sio.load_slp(str(labels_or_path))
        if isinstance(labels_or_path, (str, Path))
        else labels_or_path
    )
    rows: list[dict[str, Any]] = []
    for labeled_frame in labels.labeled_frames:
        source = _video_source_path(labeled_frame.video)
        experiment = experiment_from_video(source)
        view = classify_view(source)
        if experiment not in EXPERIMENTS or view not in SIDE_VIEWS:
            continue
        for index, instance in enumerate(labeled_frame.instances):
            points = np.asarray(instance.numpy(), dtype=float)
            if len(points) < 2:
                continue
            point_records = getattr(instance, "points", None)

            def score(node_index: int) -> float:
                if point_records is None or node_index >= len(point_records):
                    return np.nan
                try:
                    return float(point_records[node_index]["score"])
                except (KeyError, TypeError, ValueError):
                    return np.nan

            rows.append(
                {
                    "experiment": experiment,
                    "view": view,
                    "frame": int(labeled_frame.frame_idx),
                    "instance_index": index,
                    "mouth_x": float(points[0, 0]),
                    "mouth_y": float(points[0, 1]),
                    "tail_x": float(points[1, 0]),
                    "tail_y": float(points[1, 1]),
                    "mouth_score": score(0),
                    "tail_score": score(1),
                    "instance_score": float(
                        getattr(instance, "score", np.nan) or np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _match_by_mouth(
    old: pd.DataFrame, current: pd.DataFrame, *, gate_px: float = 80.0
) -> list[tuple[int, int, float]]:
    if old.empty or current.empty:
        return []
    old_xy = old[["mouth_x", "mouth_y"]].to_numpy(dtype=float)
    current_xy = current[["mouth_x", "mouth_y"]].to_numpy(dtype=float)
    cost = np.linalg.norm(
        old_xy[:, None, :] - current_xy[None, :, :], axis=2
    )
    cost[~np.isfinite(cost)] = 1e9
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(cost)
        pairs = zip(rows.tolist(), columns.tolist())
    except ImportError:  # pragma: no cover - scipy is a project dependency
        import itertools

        width = min(len(old), len(current))
        best: tuple[float, tuple[int, ...]] | None = None
        for columns in itertools.permutations(range(len(current)), width):
            value = sum(cost[row, column] for row, column in enumerate(columns))
            if best is None or value < best[0]:
                best = (float(value), columns)
        pairs = enumerate(best[1] if best is not None else ())
    return [
        (int(old_index), int(current_index), float(cost[old_index, current_index]))
        for old_index, current_index in pairs
        if cost[old_index, current_index] <= gate_px
    ]


def compare_round7_round8(
    round7_instances: pd.DataFrame,
    round8_instances: pd.DataFrame,
) -> pd.DataFrame:
    """Return source-frame metrics from sparse current-model predictions."""
    key_columns = ["experiment", "view", "frame"]
    round7_groups = {
        key: group.reset_index(drop=True)
        for key, group in round7_instances.groupby(key_columns, sort=False)
    }
    round8_groups = {
        key: group.reset_index(drop=True)
        for key, group in round8_instances.groupby(key_columns, sort=False)
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(set(round7_groups) | set(round8_groups)):
        old = round7_groups.get(key, pd.DataFrame())
        current = round8_groups.get(key, pd.DataFrame())
        matches = _match_by_mouth(old, current)
        tail_disagreement: list[float] = []
        mouth_disagreement: list[float] = []
        for old_index, current_index, mouth_distance in matches:
            old_tail = old.loc[old_index, ["tail_x", "tail_y"]].to_numpy(float)
            current_tail = current.loc[
                current_index, ["tail_x", "tail_y"]
            ].to_numpy(float)
            if np.isfinite(old_tail).all() and np.isfinite(current_tail).all():
                tail_disagreement.append(
                    float(np.linalg.norm(old_tail - current_tail))
                )
            mouth_disagreement.append(float(mouth_distance))
        current_tails = current[["tail_x", "tail_y"]].to_numpy(float)
        minimum_tail_distance = np.nan
        if len(current_tails) >= 2:
            distances = [
                float(np.linalg.norm(current_tails[first] - current_tails[second]))
                for first in range(len(current_tails))
                for second in range(first + 1, len(current_tails))
                if np.isfinite(current_tails[first]).all()
                and np.isfinite(current_tails[second]).all()
            ]
            if distances:
                minimum_tail_distance = min(distances)
        body_length = np.hypot(
            current.get("tail_x", pd.Series(dtype=float))
            - current.get("mouth_x", pd.Series(dtype=float)),
            current.get("tail_y", pd.Series(dtype=float))
            - current.get("mouth_y", pd.Series(dtype=float)),
        )
        rows.append(
            {
                "experiment": key[0],
                "view": key[1],
                "frame": int(key[2]),
                "round7_instance_count": len(old),
                "round8_instance_count": len(current),
                "matched_instance_count": len(matches),
                "maximum_mouth_disagreement_px": (
                    max(mouth_disagreement) if mouth_disagreement else np.nan
                ),
                "maximum_tail_disagreement_px": (
                    max(tail_disagreement) if tail_disagreement else np.nan
                ),
                "minimum_round8_tail_score": (
                    pd.to_numeric(
                        current.get("tail_score", pd.Series(dtype=float)),
                        errors="coerce",
                    ).min()
                    if len(current)
                    else np.nan
                ),
                "maximum_round8_body_length_px": (
                    float(np.nanmax(body_length)) if len(body_length) else np.nan
                ),
                "minimum_round8_tail_distance_px": minimum_tail_distance,
            }
        )
    per_view = pd.DataFrame(rows)
    if per_view.empty:
        raise ValueError("current-model candidate overlay contained no side instances")
    source_rows: list[dict[str, Any]] = []
    for (experiment, frame), group in per_view.groupby(
        ["experiment", "frame"], sort=True
    ):
        source_rows.append(
            {
                "experiment": experiment,
                "frame": int(frame),
                "round8_view_count": int(group["view"].nunique()),
                "round8_any_count_not_three": bool(
                    group["round8_instance_count"].ne(3).any()
                ),
                "round8_minimum_tail_score": float(
                    group["minimum_round8_tail_score"].min()
                ),
                "round8_maximum_mouth_disagreement_px": float(
                    group["maximum_mouth_disagreement_px"].max()
                ),
                "round8_maximum_tail_disagreement_px": float(
                    group["maximum_tail_disagreement_px"].max()
                ),
                "round8_maximum_body_length_px": float(
                    group["maximum_round8_body_length_px"].max()
                ),
                "round8_minimum_tail_distance_px": float(
                    group["minimum_round8_tail_distance_px"].min()
                ),
            }
        )
    return pd.DataFrame(source_rows)


def add_current_model_scores(
    pool: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    table = pool.merge(
        comparison,
        on=["experiment", "frame"],
        how="left",
        validate="one_to_one",
    )
    if table["round8_view_count"].fillna(0).lt(2).any():
        bad = table.loc[
            table["round8_view_count"].fillna(0).lt(2),
            ["experiment", "frame", "round8_view_count"],
        ]
        raise ValueError(
            "current-model overlay is missing selected side views: "
            f"{bad.head(10).to_dict('records')}"
        )
    table["round7_round8_tail_disagreement_score"] = (
        _rank_high(table["round8_maximum_tail_disagreement_px"])
        + _rank_low(table["round8_minimum_tail_score"])
    ) / 2.0
    # Restrict the CI-oriented disagreement signal to frames whose matched
    # mouth anchors stayed close. Large mouth shifts are centroid evidence.
    table["ci_oriented_tail_disagreement"] = (
        table["round8_maximum_tail_disagreement_px"].ge(5.0)
        & table["round8_maximum_mouth_disagreement_px"].le(10.0)
    )
    table["current_count_partial_low_confidence_score"] = (
        table["round8_any_count_not_three"].astype(float)
        + _rank_low(table["round8_minimum_tail_score"])
    ) / 2.0
    return table


def _category_mask_and_score(
    table: pd.DataFrame, category: str
) -> tuple[pd.Series, str]:
    good_current_count = ~table["round8_any_count_not_three"]
    if category == "round7_round8_tail_disagreement":
        return (
            good_current_count
            & (
                table["ci_oriented_tail_disagreement"]
                | table["round7_round8_tail_disagreement_score"].ge(0.70)
            ),
            "round7_round8_tail_disagreement_score",
        )
    if category == "temporal_or_shared_tail":
        return (
            good_current_count
            & (
                table["temporal_or_shared_tail_score"].ge(0.70)
                | table["shared_tail_prediction"]
                | table["round8_minimum_tail_distance_px"].le(5.0)
            ),
            "temporal_or_shared_tail_score",
        )
    if category == "close_contact_or_crossing":
        return (
            good_current_count & table["close_contact_score"].ge(0.70),
            "close_contact_score",
        )
    if category == "long_body_crop_margin_proxy":
        return (
            good_current_count
            & (
                table["crop_margin_proxy_score"].ge(0.70)
                | table["planned_crop_margin_risk"]
            ),
            "crop_margin_proxy_score",
        )
    if category == "count_partial_or_low_confidence":
        return (
            table["round8_any_count_not_three"]
            | table["r7_count_or_partial"]
            | table["current_count_partial_low_confidence_score"].ge(0.75),
            "current_count_partial_low_confidence_score",
        )
    if category == "clean_diverse_control":
        return (
            good_current_count
            & table["r7_good_three_both_views"]
            & table["round8_minimum_tail_score"].ge(
                table["round8_minimum_tail_score"].median()
            ),
            "clean_control_score",
        )
    raise KeyError(category)


def _select_by_quotas(
    table: pd.DataFrame,
    *,
    quotas: Mapping[str, int],
    selected_frames: Sequence[int],
    guard_frames: Sequence[int],
    config: SelectionConfig,
    seed_offset: int,
) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    occupied = [int(frame) for frame in selected_frames]
    guarded = [int(frame) for frame in guard_frames]
    for offset, (category, wanted) in enumerate(quotas.items()):
        mask, score_column = _category_mask_and_score(table, category)
        available = table.loc[
            mask
            & ~table["frame"].isin(occupied)
            & ~table["frame"].map(
                lambda frame: any(
                    abs(int(frame) - other) <= config.locked_guard_frames
                    for other in guarded
                )
            )
        ]
        picked = _diverse_pick(
            available,
            wanted=int(wanted),
            score_column=score_column,
            selected_frames=occupied,
            spacing_frames=config.selection_spacing_frames,
            seed=config.seed + seed_offset + offset,
        )
        if len(picked) < wanted:
            fallback = table.loc[
                ~table["frame"].isin(
                    occupied + picked["frame"].astype(int).tolist()
                )
                & ~table["frame"].map(
                    lambda frame: any(
                        abs(int(frame) - other)
                        <= config.locked_guard_frames
                        for other in guarded
                    )
                )
            ].copy()
            fallback["_fallback_score"] = fallback[
                [
                    "round7_round8_tail_disagreement_score",
                    "temporal_or_shared_tail_score",
                    "close_contact_score",
                    "crop_margin_proxy_score",
                    "current_count_partial_low_confidence_score",
                    "clean_control_score",
                ]
            ].max(axis=1)
            extra = _diverse_pick(
                fallback,
                wanted=int(wanted) - len(picked),
                score_column="_fallback_score",
                selected_frames=occupied
                + picked["frame"].astype(int).tolist(),
                spacing_frames=config.selection_spacing_frames,
                seed=config.seed + seed_offset + 100 + offset,
            ).drop(columns=["_fallback_score"])
            picked = pd.concat([picked, extra], ignore_index=False)
        if len(picked) != wanted:
            raise ValueError(
                f"could not select {wanted} final {category} frames; "
                f"got {len(picked)}"
            )
        picked = picked.copy()
        picked["selection_category"] = category
        picked["selection_category_score"] = picked[score_column]
        chosen.append(picked)
        occupied.extend(picked["frame"].astype(int).tolist())
    return pd.concat(chosen, ignore_index=True)


def select_final_set(
    scored_pool: pd.DataFrame,
    *,
    config: SelectionConfig,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for experiment in EXPERIMENTS:
        subset = scored_pool.loc[
            scored_pool["experiment"].eq(experiment)
        ].copy()
        validation = _select_by_quotas(
            subset,
            quotas=DEFAULT_VALIDATION_QUOTAS,
            selected_frames=[],
            guard_frames=[],
            config=config,
            seed_offset=1_000,
        )
        validation["split_role"] = "validation"
        train = _select_by_quotas(
            subset,
            quotas=DEFAULT_TRAIN_QUOTAS[experiment],
            selected_frames=[],
            guard_frames=validation["frame"].astype(int).tolist(),
            config=config,
            seed_offset=2_000,
        )
        train["split_role"] = "train"
        selections.extend([validation, train])
    result = pd.concat(selections, ignore_index=True)
    expected = 30 + sum(
        sum(quotas.values()) for quotas in DEFAULT_TRAIN_QUOTAS.values()
    )
    if len(result) != expected:
        raise ValueError(f"final selection count {len(result)} != {expected}")
    for experiment in EXPERIMENTS:
        validation_frames = result.loc[
            result["experiment"].eq(experiment)
            & result["split_role"].eq("validation"),
            "frame",
        ].to_numpy(dtype=int)
        train_frames = result.loc[
            result["experiment"].eq(experiment)
            & result["split_role"].eq("train"),
            "frame",
        ].to_numpy(dtype=int)
        violations = [
            (int(train), int(validation))
            for train in train_frames
            for validation in validation_frames
            if abs(int(train) - int(validation)) <= config.locked_guard_frames
        ]
        if violations:
            raise ValueError(
                f"train/validation guard violation in {experiment}: "
                f"{violations[:10]}"
            )
    result["_batch_tie"] = [
        _stable_tie(config.seed, row.experiment, int(row.frame), "batch")
        for row in result.itertuples()
    ]
    result = result.sort_values(
        ["split_role", "experiment", "_batch_tie"], kind="stable"
    ).reset_index(drop=True)
    result["review_batch"] = (np.arange(len(result)) % 3) + 1
    return result.drop(columns=["_batch_tie"]).sort_values(
        ["review_batch", "experiment", "frame"], kind="stable"
    ).reset_index(drop=True)


def _resolve_video_paths(
    *,
    experiments: Iterable[str],
    raw_data_root: Path,
    round7_output_root: Path,
    round7_tag: str,
) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path]]:
    import sleap_io as sio

    videos: dict[tuple[str, str], Path] = {}
    slps: dict[tuple[str, str], Path] = {}
    for experiment in experiments:
        for view in SIDE_VIEWS:
            slp_path = (
                round7_output_root / experiment / round7_tag / f"{view}.slp"
            )
            if not slp_path.is_file():
                raise FileNotFoundError(f"Round-7 prediction not found: {slp_path}")
            labels = sio.load_slp(str(slp_path), open_videos=False, lazy=True)
            try:
                if not labels.videos:
                    raise ValueError(f"{slp_path} has no source video metadata")
                stored = Path(_video_source_path(labels.videos[0]))
            finally:
                labels.close()
            raw_video = raw_data_root / experiment / view / stored.name
            if not raw_video.is_file():
                raise FileNotFoundError(f"raw video not found: {raw_video}")
            videos[(experiment, view)] = raw_video
            slps[(experiment, view)] = slp_path
    return videos, slps


def build_blank_side_package(
    manifest: pd.DataFrame,
    *,
    output_path: Path,
    video_paths: Mapping[tuple[str, str], Path],
    selection_manifest_path: Path,
    embed: bool,
) -> dict[str, Any]:
    import sleap_io as sio

    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"], edges=[("mouth", "tail")]
    )
    videos = {
        key: sio.Video.from_filename(str(path))
        for key, path in video_paths.items()
    }
    labeled_frames: list[Any] = []
    suggestions: list[Any] = []
    key_rows = manifest[
        ["experiment", "frame", "split_role", "review_batch"]
    ].drop_duplicates(["experiment", "frame"])
    for row in key_rows.to_dict("records"):
        for view in SIDE_VIEWS:
            video = videos[(str(row["experiment"]), view)]
            frame = int(row["frame"])
            labeled_frames.append(
                sio.LabeledFrame(video=video, frame_idx=frame, instances=[])
            )
            suggestions.append(
                sio.SuggestionFrame(video=video, frame_idx=frame)
            )
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=list(videos.values()),
        labeled_frames=labeled_frames,
        suggestions=suggestions,
        provenance={
            "schema_version": SELECTION_VERSION,
            "purpose": "blank_authoritative_side_centered_instance_hitl",
            "contains_model_predictions": False,
            "selection_manifest": str(selection_manifest_path.resolve()),
            "selection_manifest_sha256": hashlib.sha256(
                selection_manifest_path.read_bytes()
            ).hexdigest(),
            "instructions": [
                "review all three real fish in both side views",
                "correct mouth and tail for every visible fish",
                "delete reflections and other false positives",
                "add missing fish when the model returns fewer than three",
                "do not copy locked HITL-v2 benchmark labels into training",
            ],
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(output_path), embed="all" if embed else False)
    return {
        "path": str(output_path.resolve()),
        "size_bytes": int(output_path.stat().st_size),
        "source_frame_count": int(len(key_rows)),
        "labeled_frame_count": int(len(labeled_frames)),
        "suggestion_count": int(len(suggestions)),
        "embedded": bool(embed),
    }


def validate_blank_side_package(
    path: Path, manifest: pd.DataFrame
) -> dict[str, Any]:
    import sleap_io as sio

    labels = sio.load_slp(str(path))
    if len(labels.skeletons) != 1:
        raise ValueError("blank side package must contain one skeleton")
    nodes = [node.name for node in labels.skeletons[0].nodes]
    if nodes != ["mouth", "tail"]:
        raise ValueError(f"unexpected skeleton node order: {nodes}")
    expected = {
        (str(row.experiment), view, int(row.frame))
        for row in manifest[
            ["experiment", "frame"]
        ].drop_duplicates().itertuples()
        for view in SIDE_VIEWS
    }
    observed: set[tuple[str, str, int]] = set()
    nonempty = 0
    for labeled_frame in labels.labeled_frames:
        source = _video_source_path(labeled_frame.video)
        key = (
            str(experiment_from_video(source)),
            str(classify_view(source)),
            int(labeled_frame.frame_idx),
        )
        if key in observed:
            raise ValueError(f"duplicate labeled frame: {key}")
        observed.add(key)
        nonempty += int(bool(labeled_frame.instances))
    if observed != expected:
        raise ValueError(
            f"blank package coverage mismatch: missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    if nonempty:
        raise ValueError(f"blank package contains {nonempty} non-empty frames")
    return {
        "node_names": nodes,
        "expected_labeled_frames": len(expected),
        "observed_labeled_frames": len(observed),
        "suggestions": len(labels.suggestions),
        "nonempty_labeled_frames": nonempty,
        "passed": True,
    }


def _manifest_columns(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "frame",
        "split_role",
        "review_batch",
        "selection_category",
        "selection_category_score",
        "pool_category",
        "pool_category_score",
        "r7_good_three_both_views",
        "r7_count_or_partial",
        "r7_minimum_tail_score",
        "r7_maximum_mouth_tail_score_gap",
        "r7_maximum_tail_midpoint_residual_px",
        "r7_minimum_mouth_distance_px",
        "r7_minimum_tail_distance_px",
        "r7_maximum_body_length_px",
        "planned_crop_margin_risk",
        "shared_tail_prediction",
        "round8_any_count_not_three",
        "round8_minimum_tail_score",
        "round8_maximum_mouth_disagreement_px",
        "round8_maximum_tail_disagreement_px",
        "ci_oriented_tail_disagreement",
        "round8_maximum_body_length_px",
        "round8_minimum_tail_distance_px",
    ]
    return table[[column for column in columns if column in table]].copy()


def mine(args: argparse.Namespace) -> int:
    config = SelectionConfig(
        seed=args.seed,
        skip_initial_frames=args.skip_initial_frames,
        selection_spacing_frames=args.spacing_frames,
    )
    intervals, exclusion_report = build_exclusion_intervals(
        locked_manifest=args.locked_manifest,
        future_locked_manifest=args.future_locked_manifest,
        previous_selection_manifest=args.previous_selection_manifest,
        existing_train_labels=args.existing_train_labels,
        existing_val_labels=args.existing_val_labels,
        config=config,
    )
    video_paths, slp_paths = _resolve_video_paths(
        experiments=EXPERIMENTS,
        raw_data_root=args.raw_data_root,
        round7_output_root=args.round7_output_root,
        round7_tag=args.round7_tag,
    )
    all_pools: list[pd.DataFrame] = []
    selected_instances: list[pd.DataFrame] = []
    feature_reports: dict[str, Any] = {}
    feature_config = FeatureConfig(plausible_score=config.plausible_score)
    for experiment in EXPERIMENTS:
        view_tables: dict[str, pd.DataFrame] = {}
        view_instances: dict[str, pd.DataFrame] = {}
        feature_reports[experiment] = {}
        for view in SIDE_VIEWS:
            cache_dir = args.cache_dir / experiment
            frames, instances, metadata = load_or_build_view_features(
                slp_paths[(experiment, view)],
                cache_dir=cache_dir,
                view=view,
                config=feature_config,
                force=args.force_features,
            )
            view_tables[view], view_instances[view] = aggregate_view_features(
                frames, instances, view=view
            )
            feature_reports[experiment][view] = metadata
        combined = combine_side_features(
            view_tables["side1_cam"],
            view_tables["side2_cam"],
            experiment=experiment,
            intervals=intervals[experiment],
            config=config,
        )
        pool = select_coarse_pool(combined, config=config)
        all_pools.append(pool)
        frames = set(pool["frame"].astype(int))
        for view in SIDE_VIEWS:
            chosen = view_instances[view].loc[
                view_instances[view]["frame"].isin(frames)
            ].copy()
            chosen["experiment"] = experiment
            selected_instances.append(chosen)
    pool = pd.concat(all_pools, ignore_index=True).sort_values(
        ["experiment", "frame"], kind="stable"
    )
    pool["split_role"] = "candidate_pool"
    pool["review_batch"] = 0
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.report_dir / "candidate_pool_630.csv"
    instance_path = args.report_dir / "candidate_pool_630_round7_instances.csv"
    pool.to_csv(pool_path, index=False)
    pd.concat(selected_instances, ignore_index=True).to_csv(
        instance_path, index=False
    )
    pool_seed_path = args.artifact_dir / "candidate_pool_630_blank.slp"
    package = build_blank_side_package(
        pool,
        output_path=pool_seed_path,
        video_paths=video_paths,
        selection_manifest_path=pool_path,
        embed=False,
    )
    validation = validate_blank_side_package(pool_seed_path, pool)
    report = {
        "schema_version": SELECTION_VERSION,
        "stage": "mine",
        "config": asdict(config),
        "pool_counts": {
            "source_frames": int(len(pool)),
            "side_view_images": int(len(pool) * len(SIDE_VIEWS)),
            "by_experiment": {
                key: int(value)
                for key, value in pool["experiment"].value_counts().to_dict().items()
            },
            "by_category": {
                key: int(value)
                for key, value in pool["pool_category"].value_counts().to_dict().items()
            },
        },
        "exclusions": exclusion_report,
        "round7_features": feature_reports,
        "artifacts": {
            "pool_manifest": _fingerprint(pool_path),
            "pool_round7_instances": _fingerprint(instance_path),
            "pool_blank_seed": package,
        },
        "validation": validation,
    }
    _write_json(args.report_dir / "candidate_pool_630_summary.json", report)
    print(f"Candidate pool manifest: {pool_path}")
    print(f"Round-7 pool instances: {instance_path}")
    print(f"Sparse inference seed:  {pool_seed_path}")
    return 0


def _write_runbook(
    path: Path,
    *,
    seed_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = f"""# Side centered-instance HITL review

This package contains 210 synchronized source frames from both side views:
180 training frames and 30 event-guarded validation frames. Review batches are
recorded in `selection_210.csv`.

The blank package is authoritative. Round-8 predictions are only a candidate
overlay and become user labels only after explicit review.

```powershell
Set-Location "{repo_root}"
.venv\\Scripts\\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in "{seed_path}" `
  --candidate-slp "{candidate_path}" `
  --out "{output_path}" `
  --paths-config "configs\\paths.local.toml" `
  --instance-review --independent --min-views 1 `
  --calibration-backend auto
```

Review every proposed fish with Left/Right and accept or correct it. Afterwards,
open the saved package in normal review mode and add any missing fish so each
side image contains exactly three real fish. Mouth must be finite for every
centered-instance training crop. Delete reflections and other false positives.
The GUI autosaves every two minutes. Press Ctrl+S for a quick save and
Ctrl+Shift+S for a fully embedded save.
"""
    path.write_text(text, encoding="utf-8")


def _write_launcher(
    path: Path,
    *,
    seed_path: Path,
    candidate_path: Path,
    output_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    text = f"""@echo off
setlocal
cd /d "{repo_root}"
".venv\\Scripts\\python.exe" -m scripts.sleap.pyqt_multiview_annotator ^
  --in "{seed_path}" ^
  --candidate-slp "{candidate_path}" ^
  --out "{output_path}" ^
  --paths-config "configs\\paths.local.toml" ^
  --calibration-backend auto ^
  --min-views 1 ^
  --instance-review ^
  --independent
endlocal
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finalize(args: argparse.Namespace) -> int:
    config = SelectionConfig(
        seed=args.seed,
        skip_initial_frames=args.skip_initial_frames,
        selection_spacing_frames=args.spacing_frames,
    )
    pool = pd.read_csv(args.pool_manifest)
    round7_instances = pd.read_csv(args.round7_instances)
    round8_instances = _prediction_rows(args.round8_pool_candidates)
    comparison = compare_round7_round8(round7_instances, round8_instances)
    scored = add_current_model_scores(pool, comparison)
    final = select_final_set(scored, config=config)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.report_dir / "selection_210.csv"
    _manifest_columns(final).to_csv(manifest_path, index=False)

    video_paths, _slp_paths = _resolve_video_paths(
        experiments=EXPERIMENTS,
        raw_data_root=args.raw_data_root,
        round7_output_root=args.round7_output_root,
        round7_tag=args.round7_tag,
    )
    seed_path = args.artifact_dir / "side_ci_review_210_blank.pkg.slp"
    package = build_blank_side_package(
        final,
        output_path=seed_path,
        video_paths=video_paths,
        selection_manifest_path=manifest_path,
        embed=True,
    )
    validation = validate_blank_side_package(seed_path, final)
    output_path = args.artifact_dir / "output" / "side_ci_review_210_labeled.pkg.slp"
    candidate_path = args.artifact_dir / "side_ci_review_210_round8_candidates.slp"
    _write_runbook(
        args.artifact_dir / "README.md",
        seed_path=seed_path,
        candidate_path=candidate_path,
        output_path=output_path,
    )
    summary = {
        "schema_version": SELECTION_VERSION,
        "stage": "finalize",
        "config": asdict(config),
        "inputs": {
            "pool_manifest": _fingerprint(args.pool_manifest),
            "round7_instances": _fingerprint(args.round7_instances),
            "round8_pool_candidates": _fingerprint(
                args.round8_pool_candidates
            ),
        },
        "selection": {
            "source_frames": int(len(final)),
            "side_view_images": int(len(final) * len(SIDE_VIEWS)),
            "expected_fish_instances": int(len(final) * len(SIDE_VIEWS) * 3),
            "by_split": {
                key: int(value)
                for key, value in final["split_role"].value_counts().to_dict().items()
            },
            "by_experiment_and_split": (
                final.groupby(["experiment", "split_role"]).size()
                .unstack(fill_value=0)
                .astype(int)
                .to_dict(orient="index")
            ),
            "by_category": {
                key: int(value)
                for key, value in final[
                    "selection_category"
                ].value_counts().to_dict().items()
            },
            "by_review_batch": {
                str(key): int(value)
                for key, value in final["review_batch"].value_counts().to_dict().items()
            },
        },
        "artifacts": {
            "manifest": _fingerprint(manifest_path),
            "blank_seed": package,
            "expected_candidate_overlay": str(candidate_path.resolve()),
            "expected_labeled_output": str(output_path.resolve()),
        },
        "validation": validation,
        "scientific_notes": [
            "Round 7 was used only for broad all-frame mining.",
            "Round 8 sparse predictions were used for current-model reranking.",
            "Large mouth disagreement is treated as centroid evidence rather than centered-instance disagreement.",
            "Long body length is a crop-margin proxy, not a measured crop-offset sensitivity test.",
            "The locked HITL-v2 benchmark and guard intervals remain excluded from training.",
        ],
    }
    _write_json(args.report_dir / "selection_210_summary.json", summary)
    print(f"Final selection manifest: {manifest_path}")
    print(f"Blank review package:    {seed_path}")
    print(f"Next candidate output:   {candidate_path}")
    return 0


def audit(args: argparse.Namespace) -> int:
    """Validate the final blank package and separate model overlay."""
    import sleap_io as sio

    manifest = pd.read_csv(args.manifest)
    expected = [
        (str(row.experiment), view, int(row.frame))
        for row in manifest[
            ["experiment", "frame"]
        ].drop_duplicates().itertuples()
        for view in SIDE_VIEWS
    ]
    seed = sio.load_slp(str(args.seed_slp))
    candidates = sio.load_slp(str(args.candidate_slp))

    def key(labeled_frame: Any) -> tuple[str, str, int]:
        source = _video_source_path(labeled_frame.video)
        return (
            str(experiment_from_video(source)),
            str(classify_view(source)),
            int(labeled_frame.frame_idx),
        )

    seed_keys = [key(labeled_frame) for labeled_frame in seed.labeled_frames]
    candidate_keys = [
        key(labeled_frame) for labeled_frame in candidates.labeled_frames
    ]
    if seed_keys != expected:
        raise ValueError("blank seed order/coverage does not match final manifest")
    if candidate_keys != expected:
        raise ValueError(
            "candidate overlay order/coverage does not match final manifest"
        )
    seed_instances = sum(
        len(labeled_frame.instances) for labeled_frame in seed.labeled_frames
    )
    if seed_instances:
        raise ValueError(
            f"authoritative blank seed contains {seed_instances} instances"
        )
    candidate_types = sorted(
        {
            type(instance).__name__
            for labeled_frame in candidates.labeled_frames
            for instance in labeled_frame.instances
        }
    )
    if candidate_types != ["PredictedInstance"]:
        raise ValueError(
            f"candidate overlay has unexpected instance types: {candidate_types}"
        )
    maximum_instances = max(
        len(labeled_frame.instances)
        for labeled_frame in candidates.labeled_frames
    )
    if maximum_instances > args.expected_instances:
        raise ValueError(
            f"candidate overlay exceeds expected instance cap: {maximum_instances}"
        )
    deficient_rows: list[dict[str, Any]] = []
    total_candidates = 0
    for labeled_frame in candidates.labeled_frames:
        count = len(labeled_frame.instances)
        total_candidates += count
        if count >= args.expected_instances:
            continue
        experiment, view, frame = key(labeled_frame)
        deficient_rows.append(
            {
                "experiment": experiment,
                "frame": frame,
                "view": view,
                "candidate_instances": count,
                "expected_instances": args.expected_instances,
                "missing_instances": args.expected_instances - count,
                "missing_instance_target": f"{experiment}:{frame}:{view}",
            }
        )
    readable = 0
    shapes: set[tuple[int, ...]] = set()
    if args.decode_all:
        for labeled_frame in seed.labeled_frames:
            image = labeled_frame.video[int(labeled_frame.frame_idx)]
            shapes.add(tuple(int(value) for value in image.shape))
            readable += 1

    deficient = pd.DataFrame(
        deficient_rows,
        columns=[
            "experiment",
            "frame",
            "view",
            "candidate_instances",
            "expected_instances",
            "missing_instances",
            "missing_instance_target",
        ],
    )
    args.deficient_csv.parent.mkdir(parents=True, exist_ok=True)
    deficient.to_csv(args.deficient_csv, index=False)
    report = {
        "schema_version": SELECTION_VERSION,
        "stage": "audit",
        "passed": True,
        "artifacts": {
            "manifest": {
                **_fingerprint(args.manifest),
                "sha256": _sha256_file(args.manifest),
            },
            "blank_seed": {
                **_fingerprint(args.seed_slp),
                "sha256": _sha256_file(args.seed_slp),
            },
            "candidate_overlay": {
                **_fingerprint(args.candidate_slp),
                "sha256": _sha256_file(args.candidate_slp),
            },
            "deficient_views": _fingerprint(args.deficient_csv),
        },
        "coverage": {
            "source_frames": int(
                len(manifest.drop_duplicates(["experiment", "frame"]))
            ),
            "expected_side_view_images": len(expected),
            "blank_seed_labeled_frames": len(seed_keys),
            "candidate_labeled_frames": len(candidate_keys),
            "same_order_as_manifest": True,
        },
        "authoritative_seed": {
            "instance_count": seed_instances,
            "node_names": [node.name for node in seed.skeletons[0].nodes],
            "contains_model_predictions": False,
        },
        "candidate_overlay": {
            "instance_count": total_candidates,
            "maximum_instances_per_view": maximum_instances,
            "instance_types": candidate_types,
            "deficient_view_count": len(deficient),
            "missing_instance_count": int(
                deficient["missing_instances"].sum()
            )
            if len(deficient)
            else 0,
            "authoritative_annotation": False,
        },
        "embedded_frame_decode": {
            "requested": bool(args.decode_all),
            "readable_frames": readable,
            "unique_shapes": [list(shape) for shape in sorted(shapes)],
        },
    }
    _write_json(args.output_json, report)
    if args.readme is not None or args.launcher_bat is not None:
        if args.labeled_output is None:
            raise ValueError(
                "--labeled-output is required with --readme or --launcher-bat"
            )
        if args.readme is not None:
            _write_runbook(
                args.readme,
                seed_path=args.seed_slp,
                candidate_path=args.candidate_slp,
                output_path=args.labeled_output,
            )
        if args.launcher_bat is not None:
            _write_launcher(
                args.launcher_bat,
                seed_path=args.seed_slp,
                candidate_path=args.candidate_slp,
                output_path=args.labeled_output,
            )
    print(f"Bundle audit:      {args.output_json}")
    print(f"Deficient views:   {args.deficient_csv}")
    print(f"Candidate fish:    {total_candidates}")
    print(f"Missing fish slots:{report['candidate_overlay']['missing_instance_count']}")
    return 0


def _common_parser(parser: argparse.ArgumentParser) -> None:
    paths = load_local_paths(DEFAULT_PATHS_CONFIG)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-initial-frames", type=int, default=49_800)
    parser.add_argument("--spacing-frames", type=int, default=166)
    parser.add_argument("--round7-tag", default="round7_260704")
    parser.add_argument(
        "--round7-output-root",
        type=Path,
        default=paths.sleap_output_dir if paths is not None else None,
        required=paths is None or paths.sleap_output_dir is None,
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=paths.medaka_data_dir if paths is not None else None,
        required=paths is None or paths.medaka_data_dir is None,
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    mine_parser = subparsers.add_parser("mine")
    _common_parser(mine_parser)
    paths = load_local_paths(DEFAULT_PATHS_CONFIG)
    default_model_side = (
        paths.sleap_models_dir / "retrain_20260722_round8" / "side"
        if paths is not None and paths.sleap_models_dir is not None
        else None
    )
    mine_parser.add_argument("--cache-dir", type=Path, required=True)
    mine_parser.add_argument("--force-features", action="store_true")
    mine_parser.add_argument(
        "--locked-manifest",
        type=Path,
        default=Path(
            "report/20260726/sleap_hitl_benchmark_v2/"
            "shared_benchmark_40_hitl_v2_manifest.csv"
        ),
    )
    mine_parser.add_argument(
        "--future-locked-manifest",
        type=Path,
        default=Path(
            "report/20260724/sleap_topdown_optimization/"
            "future_locked_test_annotation_queue_v1.csv"
        ),
    )
    mine_parser.add_argument(
        "--previous-selection-manifest",
        type=Path,
        default=Path("report/20260721/round8_active_learning/selection_360.csv"),
    )
    mine_parser.add_argument(
        "--existing-train-labels",
        type=Path,
        default=(
            default_model_side / "train_labels.pkg.slp"
            if default_model_side is not None
            else None
        ),
        required=default_model_side is None,
    )
    mine_parser.add_argument(
        "--existing-val-labels",
        type=Path,
        default=(
            default_model_side / "val_labels.pkg.slp"
            if default_model_side is not None
            else None
        ),
        required=default_model_side is None,
    )
    mine_parser.set_defaults(func=mine)

    finalize_parser = subparsers.add_parser("finalize")
    _common_parser(finalize_parser)
    finalize_parser.add_argument("--pool-manifest", type=Path, required=True)
    finalize_parser.add_argument("--round7-instances", type=Path, required=True)
    finalize_parser.add_argument(
        "--round8-pool-candidates", type=Path, required=True
    )
    finalize_parser.set_defaults(func=finalize)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument("--seed-slp", type=Path, required=True)
    audit_parser.add_argument("--candidate-slp", type=Path, required=True)
    audit_parser.add_argument("--output-json", type=Path, required=True)
    audit_parser.add_argument("--deficient-csv", type=Path, required=True)
    audit_parser.add_argument("--expected-instances", type=int, default=3)
    audit_parser.add_argument("--decode-all", action="store_true")
    audit_parser.add_argument("--readme", type=Path)
    audit_parser.add_argument("--launcher-bat", type=Path)
    audit_parser.add_argument("--labeled-output", type=Path)
    audit_parser.set_defaults(func=audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in {"mine", "finalize"}:
        for name in (
            "report_dir",
            "artifact_dir",
            "raw_data_root",
            "round7_output_root",
        ):
            value = getattr(args, name)
            if value is None:
                raise ValueError(f"--{name.replace('_', '-')} is required")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
