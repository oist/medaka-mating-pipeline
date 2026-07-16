"""Leakage-safe comparison of position-only and motion-aware 3D linkers.

The input is a clean reviewed mouth-3D CSV containing ``human_identity`` and
``X/Y/Z``. Human labels are retained only as held-out truth. Before temporal
association, every frame is converted to anonymous candidates ordered solely
by the numeric tuple ``(X, Y, Z)``. This deterministic ordering must never use
``human_identity``, row order, a reviewed tracklet ID, or any other label proxy.

Both methods call :func:`apply_cross_frame_tracking` on the same anonymous
candidate stream. The arbitrary ``ID_1/...`` outputs are mapped one-to-one to
reviewed H labels using only a requested calibration/context frame range, with
all held-out episode frames explicitly removed from mapping fit. When an
episode CSV is supplied, every reported metric is limited to those contiguous
held-out intervals. Otherwise, evaluation uses frames outside the context
range.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
)


XYZ = ("X", "Y", "Z")


@dataclass(frozen=True)
class ComparisonConfig:
    calibration_start_frame: int
    calibration_end_frame: int
    num_individuals: int | None = None
    max_jump_mm: float | None = 8.0
    max_jump_hard_cap_mm: float | None = 80.0
    max_gap_frames: int | None = 100
    velocity_prediction_weight: float = 1.0
    flow_weight: float = 2.0
    acceleration_weight: float = 0.0
    motion_uncertainty_growth_per_frame: float = 0.25
    max_motion_gap_frames: int | None = 5
    jump_threshold_mm: float = 8.0
    severe_jump_threshold_mm: float = 20.0

    def __post_init__(self) -> None:
        if self.calibration_end_frame < self.calibration_start_frame:
            raise ValueError("calibration_end_frame precedes calibration_start_frame")
        if self.num_individuals is not None and self.num_individuals < 1:
            raise ValueError("num_individuals must be positive or None")
        for name in (
            "velocity_prediction_weight",
            "flow_weight",
            "acceleration_weight",
            "motion_uncertainty_growth_per_frame",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_motion_gap_frames is not None and self.max_motion_gap_frames < 1:
            raise ValueError("max_motion_gap_frames must be positive or None")
        if self.max_gap_frames is not None and self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative or None")
        if self.jump_threshold_mm <= 0:
            raise ValueError("jump_threshold_mm must be positive")
        if self.severe_jump_threshold_mm <= self.jump_threshold_mm:
            raise ValueError(
                "severe_jump_threshold_mm must exceed jump_threshold_mm"
            )


def _integer_frames(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{name} must contain finite integer values")
    return numeric.astype(np.int64)


def validate_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """Validate reviewed truth without changing its identity labels."""
    required = {"frame", "human_identity", *XYZ}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(
            "reviewed mouth-3D table is missing columns: "
            + ", ".join(sorted(missing))
        )
    result = reference.drop(columns=["reference_row_index"], errors="ignore").copy()
    result["frame"] = _integer_frames(result["frame"], "frame")
    if result["human_identity"].isna().any():
        raise ValueError("human_identity cannot be missing in reviewed truth")
    result["human_identity"] = result["human_identity"].astype(str).str.strip()
    if result["human_identity"].eq("").any():
        raise ValueError("human_identity cannot be blank in reviewed truth")
    if result.duplicated(["frame", "human_identity"]).any():
        raise ValueError("duplicate frame/human_identity rows in reviewed truth")
    for column in XYZ:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.sort_values(["frame", "human_identity"], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    # This key is audit plumbing used only to reattach withheld truth after
    # association. It is assigned after validation/sorting for repeatability.
    result["reference_row_index"] = np.arange(len(result), dtype=np.int64)
    return result


def _source_labels_for(n: int) -> list[str]:
    if n > 26:
        raise ValueError("anonymous source labels support at most 26 candidates")
    return [chr(ord("a") + index) for index in range(n)]


def build_anonymous_observations(
    reference: pd.DataFrame,
    *,
    num_individuals: int | None = None,
) -> tuple[dict[int, dict[str, object]], pd.DataFrame, int]:
    """Build label-free linker input and a post-hoc truth lookup.

    Candidate order is lexicographic ``(X, Y, Z)`` within each frame. Exact
    duplicate coordinates are rejected because no label-independent ordering
    can distinguish them. ``reference_row_index`` is kept only in the returned
    lookup and is never passed to the linker.
    """
    truth = validate_reference(reference)
    finite = np.isfinite(truth[list(XYZ)].to_numpy(dtype=float)).all(axis=1)
    observed = truth.loc[finite].copy()
    if observed.empty:
        raise ValueError("reviewed mouth-3D table has no finite XYZ observations")
    inferred_n = int(observed.groupby("frame", sort=False).size().max())
    if num_individuals is not None and num_individuals < 1:
        raise ValueError("num_individuals must be positive or None")
    n = int(num_individuals if num_individuals is not None else inferred_n)
    if inferred_n > n:
        raise ValueError(
            f"a frame contains {inferred_n} candidates, more than "
            f"num_individuals={n}"
        )
    labels = _source_labels_for(n)
    best_by_frame: dict[int, dict[str, object]] = {}
    index_rows: list[dict[str, object]] = []
    for frame, group in observed.groupby("frame", sort=True):
        if group.duplicated(list(XYZ)).any():
            raise ValueError(
                f"frame {int(frame)} contains duplicate XYZ candidates; "
                "anonymous deterministic ordering is undefined"
            )
        # Do not add a label-derived tie breaker here. The strict duplicate
        # check above makes XYZ a complete deterministic ordering key.
        ordered = group.sort_values(list(XYZ), kind="stable").reset_index(drop=True)
        groups: dict[str, list[float]] = {}
        for source_index, row in ordered.iterrows():
            source_slot = labels[int(source_index)]
            xyz = [float(row[column]) for column in XYZ]
            groups[source_slot] = xyz
            index_rows.append(
                {
                    "frame": int(frame),
                    "source_index": int(source_index),
                    "source_slot": source_slot,
                    "reference_row_index": int(row["reference_row_index"]),
                    "X": xyz[0],
                    "Y": xyz[1],
                    "Z": xyz[2],
                }
            )
        best_by_frame[int(frame)] = {"groups": groups}
    candidate_index = pd.DataFrame(index_rows)
    return best_by_frame, candidate_index, n


def normalize_episodes(
    episodes: pd.DataFrame,
    *,
    frame_min: int,
    frame_max: int,
) -> pd.DataFrame:
    """Validate non-overlapping, inclusive held-out frame intervals."""
    table = episodes.copy()
    if table.empty:
        raise ValueError("episode CSV must contain at least one held-out interval")
    if {"frame_start", "frame_end"}.issubset(table.columns):
        table = table.rename(
            columns={"frame_start": "start_frame", "frame_end": "end_frame"}
        )
    required = {"start_frame", "end_frame"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            "episode CSV is missing columns: " + ", ".join(sorted(missing))
        )
    table["start_frame"] = _integer_frames(table["start_frame"], "start_frame")
    table["end_frame"] = _integer_frames(table["end_frame"], "end_frame")
    if (table["end_frame"] < table["start_frame"]).any():
        raise ValueError("episode end_frame precedes start_frame")
    if (
        (table["start_frame"] < frame_min).any()
        or (table["end_frame"] > frame_max).any()
    ):
        raise ValueError("episode intervals must lie inside the reference range")
    if "episode_id" not in table:
        table["episode_id"] = [
            f"episode_{index + 1:03d}" for index in range(len(table))
        ]
    table["episode_id"] = table["episode_id"].astype(str)
    if table["episode_id"].duplicated().any():
        raise ValueError("episode_id values must be unique")
    table.sort_values(["start_frame", "end_frame"], kind="stable", inplace=True)
    if len(table) > 1:
        starts = table["start_frame"].to_numpy(dtype=np.int64)[1:]
        previous_ends = table["end_frame"].to_numpy(dtype=np.int64)[:-1]
        if np.any(starts <= previous_ends):
            raise ValueError("held-out episode intervals cannot overlap")
    table["n_frames"] = table["end_frame"] - table["start_frame"] + 1
    return table.reset_index(drop=True)


def _outside_context_episodes(
    *,
    frame_min: int,
    frame_max: int,
    context_start: int,
    context_end: int,
) -> pd.DataFrame:
    rows = []
    if frame_min < context_start:
        rows.append(
            {
                "episode_id": "outside_context_before",
                "start_frame": frame_min,
                "end_frame": min(frame_max, context_start - 1),
            }
        )
    if context_end < frame_max:
        rows.append(
            {
                "episode_id": "outside_context_after",
                "start_frame": max(frame_min, context_end + 1),
                "end_frame": frame_max,
            }
        )
    if not rows:
        raise ValueError(
            "no evaluation frames remain outside the calibration/context range"
        )
    return normalize_episodes(
        pd.DataFrame(rows), frame_min=frame_min, frame_max=frame_max
    )


def _frames_in_episodes(frames: np.ndarray, episodes: pd.DataFrame) -> np.ndarray:
    selected = np.zeros(len(frames), dtype=bool)
    values = np.asarray(frames, dtype=np.int64)
    for row in episodes.itertuples(index=False):
        selected |= (values >= int(row.start_frame)) & (
            values <= int(row.end_frame)
        )
    return selected


def _prediction_rows(
    tracked: dict[int, dict],
    candidate_index: pd.DataFrame,
    *,
    method: str,
    num_individuals: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_frame = {
        int(frame): group
        for frame, group in candidate_index.groupby("frame", sort=False)
    }
    for frame in sorted(by_frame):
        source_to_track: dict[int, str] = {}
        mapping = tracked.get(frame, {}).get("id_mapping", [-1] * num_individuals)
        for track_index, source_index in enumerate(mapping):
            source_index = int(source_index)
            if source_index < 0:
                continue
            if source_index in source_to_track:
                raise AssertionError("linker produced a non-one-to-one source mapping")
            source_to_track[source_index] = f"ID_{track_index + 1}"
        for row in by_frame[frame].itertuples(index=False):
            rows.append(
                {
                    "method": method,
                    "frame": int(frame),
                    "source_index": int(row.source_index),
                    "source_slot": str(row.source_slot),
                    "reference_row_index": int(row.reference_row_index),
                    "track_id": source_to_track.get(int(row.source_index), pd.NA),
                    "X": float(row.X),
                    "Y": float(row.Y),
                    "Z": float(row.Z),
                }
            )
    return pd.DataFrame(rows)


def _fit_global_mapping(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    context_mask: np.ndarray,
    num_individuals: int,
) -> tuple[dict[str, str], dict[str, object]]:
    context_truth = reference.loc[context_mask]
    human_labels = sorted(context_truth["human_identity"].astype(str).unique())
    if len(human_labels) != num_individuals:
        raise ValueError(
            "calibration/context rows must contain exactly "
            f"{num_individuals} reviewed identities; found {human_labels}"
        )
    context_indices = set(
        context_truth.loc[
            np.isfinite(context_truth[list(XYZ)].to_numpy(dtype=float)).all(axis=1),
            "reference_row_index",
        ].astype(int)
    )
    fit = predictions[
        predictions["reference_row_index"].isin(context_indices)
        & predictions["track_id"].notna()
    ].merge(
        reference[["reference_row_index", "human_identity"]],
        on="reference_row_index",
        how="left",
        validate="many_to_one",
    )
    if fit.empty:
        raise ValueError("no associated rows are available for mapping calibration")
    track_ids = [f"ID_{index + 1}" for index in range(num_individuals)]
    counts = pd.crosstab(fit["track_id"], fit["human_identity"]).reindex(
        index=track_ids, columns=human_labels, fill_value=0
    )
    rows, columns = linear_sum_assignment(-counts.to_numpy(dtype=float))
    mapping = {
        track_ids[int(row)]: human_labels[int(column)]
        for row, column in zip(rows, columns)
    }
    support = {
        "context_associated_rows": int(len(fit)),
        "count_matrix": counts.astype(int).to_dict(orient="index"),
        "assigned_support": {
            track_id: int(counts.loc[track_id, human])
            for track_id, human in mapping.items()
        },
        "zero_support_track_ids": [
            track_id for track_id in track_ids if int(counts.loc[track_id].sum()) == 0
        ],
    }
    return mapping, support


def _contiguous_spans(frames: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted({int(frame) for frame in frames})
    if not values:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = values[0]
    for frame in values[1:]:
        if frame != previous + 1:
            spans.append((start, previous))
            start = frame
        previous = frame
    spans.append((start, previous))
    return spans


def _score_gaps(
    scored: pd.DataFrame,
    *,
    human_labels: list[str],
    episodes: pd.DataFrame,
) -> dict[str, object]:
    spans: list[dict[str, object]] = []
    per_identity: dict[str, dict[str, int]] = {}
    for human in human_labels:
        human_spans: list[dict[str, object]] = []
        for episode in episodes.itertuples(index=False):
            covered = set(
                scored.loc[
                    scored["human_identity"].eq(human)
                    & scored["track_id"].notna()
                    & scored["frame"].between(
                        int(episode.start_frame), int(episode.end_frame)
                    ),
                    "frame",
                ].astype(int)
            )
            missing = (
                frame
                for frame in range(
                    int(episode.start_frame), int(episode.end_frame) + 1
                )
                if frame not in covered
            )
            for start, end in _contiguous_spans(missing):
                item = {
                    "episode_id": str(episode.episode_id),
                    "human_identity": human,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": end - start + 1,
                }
                spans.append(item)
                human_spans.append(item)
        per_identity[human] = {
            "gap_spans": len(human_spans),
            "missing_frames": int(sum(item["n_frames"] for item in human_spans)),
            "longest_gap_frames": max(
                (int(item["n_frames"]) for item in human_spans), default=0
            ),
        }
    return {
        "per_identity": per_identity,
        "longest_gap_frames": max(
            (int(item["n_frames"]) for item in spans), default=0
        ),
        "gap_spans": spans,
    }


def _score_switches_and_errors(
    scored: pd.DataFrame,
    *,
    human_labels: list[str],
    episodes: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    switch_events: list[dict[str, object]] = []
    error_spans: list[dict[str, object]] = []
    for human in human_labels:
        for episode in episodes.itertuples(index=False):
            group = scored[
                scored["human_identity"].eq(human)
                & scored["frame"].between(
                    int(episode.start_frame), int(episode.end_frame)
                )
                & scored["mapped_human_identity"].notna()
            ].sort_values("frame", kind="stable")
            frames = group["frame"].to_numpy(dtype=np.int64)
            labels = group["mapped_human_identity"].astype(str).to_numpy()
            for index in range(1, len(group)):
                if labels[index] == labels[index - 1]:
                    continue
                switch_events.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "human_identity": human,
                        "from_identity": labels[index - 1],
                        "to_identity": labels[index],
                        "previous_frame": int(frames[index - 1]),
                        "frame": int(frames[index]),
                        "frame_gap": int(frames[index] - frames[index - 1]),
                    }
                )

            wrong = group[group["mapped_human_identity"].ne(human)]
            for start, end in _contiguous_spans(wrong["frame"].astype(int)):
                span_rows = wrong[wrong["frame"].between(start, end)]
                error_spans.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "human_identity": human,
                        "start_frame": start,
                        "end_frame": end,
                        "n_frames": end - start + 1,
                        "observed_identities": sorted(
                            span_rows["mapped_human_identity"].astype(str).unique()
                        ),
                    }
                )
    return (
        {
            "identity_switches": len(switch_events),
            "adjacent_identity_switches": sum(
                int(event["frame_gap"]) == 1 for event in switch_events
            ),
            "switch_events": switch_events,
        },
        error_spans,
    )


def _jump_metrics(
    scored: pd.DataFrame,
    *,
    episodes: pd.DataFrame,
    threshold: float,
    severe_threshold: float,
) -> dict[str, object]:
    steps: list[float] = []
    jump_events: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        selected = scored[
            scored["frame"].between(
                int(episode.start_frame), int(episode.end_frame)
            )
            & scored["track_id"].notna()
        ]
        for track_id, group in selected.groupby("track_id", sort=True):
            ordered = group.sort_values("frame", kind="stable")
            frames = ordered["frame"].to_numpy(dtype=np.int64)
            xyz = ordered[list(XYZ)].to_numpy(dtype=float)
            if len(ordered) < 2:
                continue
            displacement = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            adjacent = np.diff(frames) == 1
            for index in np.flatnonzero(adjacent & np.isfinite(displacement)):
                value = float(displacement[index])
                steps.append(value)
                if value > threshold:
                    jump_events.append(
                        {
                            "episode_id": str(episode.episode_id),
                            "track_id": str(track_id),
                            "frame": int(frames[index + 1]),
                            "displacement_mm": value,
                        }
                    )
    values = np.asarray(steps, dtype=float)
    return {
        "adjacent_steps": int(len(values)),
        "jumps_gt8": int(np.sum(values > threshold)),
        "jumps_gt20": int(np.sum(values > severe_threshold)),
        "jump_rate_gt8": float(np.mean(values > threshold)) if len(values) else None,
        "jump_rate_gt20": (
            float(np.mean(values > severe_threshold)) if len(values) else None
        ),
        "max_step_mm": float(np.max(values)) if len(values) else None,
        "jump_events_gt8": jump_events,
    }


def _score_method(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    mapping: dict[str, str],
    mapping_support: dict[str, object],
    episodes: pd.DataFrame,
    human_labels: list[str],
    config: ComparisonConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    scored = predictions.merge(
        reference[["reference_row_index", "human_identity"]],
        on="reference_row_index",
        how="left",
        validate="many_to_one",
    )
    scored["mapped_human_identity"] = scored["track_id"].map(mapping)
    scored["identity_correct"] = (
        scored["mapped_human_identity"].astype("string")
        == scored["human_identity"].astype("string")
    ).fillna(False)
    test_mask = _frames_in_episodes(scored["frame"].to_numpy(), episodes)
    scored["is_test"] = test_mask
    test = scored.loc[test_mask].copy()

    total_test_frames = int(episodes["n_frames"].sum())
    expected_grid_rows = total_test_frames * len(human_labels)
    reference_test_mask = _frames_in_episodes(
        reference["frame"].to_numpy(), episodes
    )
    reference_test = reference.loc[reference_test_mask]
    finite_reference_rows = int(
        np.isfinite(reference_test[list(XYZ)].to_numpy(dtype=float)).all(axis=1).sum()
    )
    associated = test["track_id"].notna()
    associated_rows = int(associated.sum())
    correct = int(test.loc[associated, "identity_correct"].sum())
    by_frame = test.loc[associated].groupby("frame")["track_id"].nunique()

    switches, error_spans = _score_switches_and_errors(
        test, human_labels=human_labels, episodes=episodes
    )
    gaps = _score_gaps(test, human_labels=human_labels, episodes=episodes)
    summary = {
        "global_mapping": mapping,
        "mapping_fit": mapping_support,
        "identity": {
            "associated_rows": associated_rows,
            "correct_identity_rows": correct,
            "error_rows": associated_rows - correct,
            "identity_accuracy_on_associated": (
                float(correct / associated_rows) if associated_rows else None
            ),
            "identity_recall_of_expected_timeline": (
                float(correct / expected_grid_rows) if expected_grid_rows else None
            ),
        },
        "coverage": {
            "test_frames": total_test_frames,
            "expected_identity_rows": expected_grid_rows,
            "available_finite_reference_rows": finite_reference_rows,
            "associated_rows": associated_rows,
            "available_reference_coverage": (
                float(associated_rows / finite_reference_rows)
                if finite_reference_rows
                else None
            ),
            "timeline_coverage": (
                float(associated_rows / expected_grid_rows)
                if expected_grid_rows
                else None
            ),
            "complete_association_frames": int((by_frame == len(human_labels)).sum()),
            "complete_association_frame_fraction": (
                float((by_frame == len(human_labels)).sum() / total_test_frames)
                if total_test_frames
                else None
            ),
        },
        "gaps": gaps,
        "switches": switches,
        "error_spans": error_spans,
        "jumps": _jump_metrics(
            test,
            episodes=episodes,
            threshold=config.jump_threshold_mm,
            severe_threshold=config.severe_jump_threshold_mm,
        ),
    }
    return summary, scored


def compare_motion_linkers(
    reference: pd.DataFrame,
    *,
    config: ComparisonConfig,
    episodes: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run both linkers and return summaries plus an inspectable row table."""
    truth = validate_reference(reference)
    frame_min = int(truth["frame"].min())
    frame_max = int(truth["frame"].max())
    if config.calibration_start_frame < frame_min or config.calibration_end_frame > frame_max:
        raise ValueError("calibration/context range must lie inside reference frames")
    heldout = (
        normalize_episodes(episodes, frame_min=frame_min, frame_max=frame_max)
        if episodes is not None
        else _outside_context_episodes(
            frame_min=frame_min,
            frame_max=frame_max,
            context_start=config.calibration_start_frame,
            context_end=config.calibration_end_frame,
        )
    )
    reference_test = _frames_in_episodes(truth["frame"].to_numpy(), heldout)
    requested_context = truth["frame"].between(
        config.calibration_start_frame, config.calibration_end_frame
    ).to_numpy()
    context_mask = requested_context & ~reference_test
    if not context_mask.any():
        raise ValueError(
            "no calibration/context rows remain after excluding held-out episodes"
        )

    best_by_frame, candidate_index, n = build_anonymous_observations(
        truth, num_individuals=config.num_individuals
    )
    common = {
        "num_individuals": n,
        "max_jump_mm": config.max_jump_mm,
        "max_jump_hard_cap_mm": config.max_jump_hard_cap_mm,
        "max_gap_frames": config.max_gap_frames,
    }
    position = apply_cross_frame_tracking(
        best_by_frame,
        flow_weight=0.0,
        velocity_prediction_weight=0.0,
        acceleration_weight=0.0,
        **common,
    )
    motion = apply_cross_frame_tracking(
        best_by_frame,
        flow_weight=config.flow_weight,
        velocity_prediction_weight=config.velocity_prediction_weight,
        acceleration_weight=config.acceleration_weight,
        motion_uncertainty_growth_per_frame=(
            config.motion_uncertainty_growth_per_frame
        ),
        max_motion_gap_frames=config.max_motion_gap_frames,
        **common,
    )

    method_tracks = {
        "position_only": position,
        "motion_aware": motion,
    }
    summaries: dict[str, object] = {}
    scored_parts: list[pd.DataFrame] = []
    human_labels = sorted(
        truth.loc[context_mask, "human_identity"].astype(str).unique()
    )
    for method, tracked in method_tracks.items():
        predictions = _prediction_rows(
            tracked,
            candidate_index,
            method=method,
            num_individuals=n,
        )
        mapping, mapping_support = _fit_global_mapping(
            predictions,
            truth,
            context_mask=context_mask,
            num_individuals=n,
        )
        summary, scored = _score_method(
            predictions,
            truth,
            mapping=mapping,
            mapping_support=mapping_support,
            episodes=heldout,
            human_labels=human_labels,
            config=config,
        )
        scored["is_context"] = scored["reference_row_index"].isin(
            set(truth.loc[context_mask, "reference_row_index"].astype(int))
        )
        summaries[method] = summary
        scored_parts.append(scored)

    predictions = pd.concat(scored_parts, ignore_index=True)
    test_frame_values = set(
        frame
        for row in heldout.itertuples(index=False)
        for frame in range(int(row.start_frame), int(row.end_frame) + 1)
    )
    requested_context_frames = set(
        range(config.calibration_start_frame, config.calibration_end_frame + 1)
    )
    effective_context_frames = set(
        truth.loc[context_mask, "frame"].astype(int).unique()
    )
    result = {
        "schema_version": "motion_linker_comparison_v1",
        "config": asdict(config),
        "input": {
            "reference_rows": int(len(truth)),
            "frame_min": frame_min,
            "frame_max": frame_max,
            "num_individuals": n,
        },
        "anonymous_candidate_policy": {
            "ordering": "per-frame lexicographic numeric (X, Y, Z)",
            "tie_policy": "reject exact duplicate XYZ candidates",
            "warning": (
                "Source ordering is deterministic plumbing only and must not "
                "carry human_identity or any label-derived proxy."
            ),
            "human_identity_passed_to_linker": False,
        },
        "episodes": heldout.to_dict(orient="records"),
        "leakage_audit": {
            "mapping_fit_policy": (
                "requested calibration/context range minus every held-out "
                "episode frame"
            ),
            "requested_context_frames": len(requested_context_frames),
            "effective_context_frames_with_reference": len(effective_context_frames),
            "heldout_frames": len(test_frame_values),
            "requested_context_heldout_overlap_frames_excluded": len(
                requested_context_frames & test_frame_values
            ),
            "effective_context_heldout_intersection_frames": len(
                effective_context_frames & test_frame_values
            ),
            "human_identity_passed_to_association": False,
            "source_ordering_uses_human_identity": False,
        },
        "methods": summaries,
        "predictions": predictions,
    }
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(
    result: dict[str, object],
    *,
    output_json: Path,
    predictions_csv: Path | None = None,
) -> dict[str, Path]:
    """Write machine-readable summary JSON and optional per-row predictions."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    predictions = result["predictions"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("result predictions must be a DataFrame")
    report = {key: value for key, value in result.items() if key != "predictions"}
    outputs = {"report_json": str(output_json)}
    paths = {"report_json": output_json}
    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_csv, index=False)
        outputs["predictions_csv"] = str(predictions_csv)
        paths["predictions_csv"] = predictions_csv
    report["outputs"] = outputs
    output_json.write_text(
        json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8"
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--episodes-csv", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--calibration-start-frame", type=int, required=True)
    parser.add_argument("--calibration-end-frame", type=int, required=True)
    parser.add_argument("--num-individuals", type=int)
    parser.add_argument("--max-jump-mm", type=float, default=8.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--velocity-prediction-weight", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=2.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.0)
    parser.add_argument(
        "--motion-uncertainty-growth-per-frame", type=float, default=0.25
    )
    parser.add_argument("--max-motion-gap-frames", type=int, default=5)
    parser.add_argument("--jump-threshold-mm", type=float, default=8.0)
    parser.add_argument("--severe-jump-threshold-mm", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = pd.read_csv(args.input_csv, low_memory=False)
    episodes = (
        pd.read_csv(args.episodes_csv, low_memory=False)
        if args.episodes_csv is not None
        else None
    )
    config = ComparisonConfig(
        calibration_start_frame=args.calibration_start_frame,
        calibration_end_frame=args.calibration_end_frame,
        num_individuals=args.num_individuals,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        velocity_prediction_weight=args.velocity_prediction_weight,
        flow_weight=args.flow_weight,
        acceleration_weight=args.acceleration_weight,
        motion_uncertainty_growth_per_frame=(
            args.motion_uncertainty_growth_per_frame
        ),
        max_motion_gap_frames=args.max_motion_gap_frames,
        jump_threshold_mm=args.jump_threshold_mm,
        severe_jump_threshold_mm=args.severe_jump_threshold_mm,
    )
    result = compare_motion_linkers(reference, config=config, episodes=episodes)
    paths = write_outputs(
        result,
        output_json=args.output_json,
        predictions_csv=args.predictions_csv,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
