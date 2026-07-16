"""Event-centered held-out benchmark for three-fish 3D data association.

The benchmark deliberately does not make a random row split.  It builds
contiguous episodes around reviewed (or automatically detected) association
events, merges intervals whose guard bands would overlap, and learns the
arbitrary prediction-label -> reviewed-identity permutation only from frames
outside every held-out episode and its guard band.

The strict reviewed mouth reconstruction is used as the reference.  Prediction
rows are associated to it independently in each frame by a small spatial
Hungarian assignment, then the baseline and optional optimized label streams
are scored on exactly the same held-out episodes.

Example (the currently reviewed K_1_2 frames 0--15000)::

    python -m scripts.analysis.heldout_association_benchmark \
      --reference-csv runs/round7_full_qc/20251225_K_1_2_FCMC/geometry_swap_review/review_0_15000_consensus/clean_reviewed_multiview_mouth_3d_f0_15000.csv \
      --events-csv runs/round7_full_qc/20251225_K_1_2_FCMC/geometry_swap_review/review_0_15000_consensus/suggested_3d_mapping_events.csv \
      --baseline-track-csv runs/round7_full_qc/20251225_K_1_2_FCMC/motion_identity/round7_260704_3d_individual_id_motion_layered_local.csv \
      --baseline-label-col id_final_pre_motion --optimized-label-col id_final \
      --output-dir runs/round7_full_qc/20251225_K_1_2_FCMC/heldout_association

This module is evaluation-only: it neither trains nor modifies an association
model and it never writes back into an input CSV.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


XYZ = ("X", "Y", "Z")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for event selection, matching, and physical jump QC."""

    episode_radius_frames: int = 166
    guard_frames: int = 166
    max_episodes: int | None = None
    event_distance_mm: float = 20.0
    match_distance_mm: float = 35.0
    jump_threshold_mm: float = 8.0
    severe_jump_threshold_mm: float = 20.0

    def __post_init__(self) -> None:
        if self.episode_radius_frames < 0:
            raise ValueError("episode_radius_frames must be non-negative")
        if self.guard_frames < 0:
            raise ValueError("guard_frames must be non-negative")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("max_episodes must be positive or None")
        for name in (
            "event_distance_mm",
            "match_distance_mm",
            "jump_threshold_mm",
            "severe_jump_threshold_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.severe_jump_threshold_mm <= self.jump_threshold_mm:
            raise ValueError("severe_jump_threshold_mm must exceed jump_threshold_mm")


def _as_int_frames(values: pd.Series, name: str = "frame") -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{name} must contain finite integer values")
    return numeric.astype(np.int64)


def _validate_reference(reference: pd.DataFrame) -> pd.DataFrame:
    required = {"frame", "human_identity", *XYZ}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError("reference is missing columns: " + ", ".join(sorted(missing)))
    out = reference.copy()
    out["frame"] = _as_int_frames(out["frame"])
    if out.duplicated(["frame", "human_identity"]).any():
        raise ValueError("reference has duplicate frame/human_identity rows")
    for column in XYZ:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["human_identity"] = out["human_identity"].astype(str)
    return out.sort_values(["frame", "human_identity"], kind="stable").reset_index(drop=True)


def _validate_track(track: pd.DataFrame, label_col: str) -> pd.DataFrame:
    required = {"frame", label_col, *XYZ}
    missing = required - set(track.columns)
    if missing:
        raise ValueError("track is missing columns: " + ", ".join(sorted(missing)))
    out = track.copy()
    out["frame"] = _as_int_frames(out["frame"])
    for column in XYZ:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[label_col] = out[label_col].astype("string")
    return out


def load_event_candidates(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize an event table to ``event_frame/event_type/source_event_id``.

    Tables with a point ``frame`` or an interval ``frame_start/frame_end`` are
    accepted.  Interval events are centered deterministically.
    """

    if "frame" in events:
        frames = _as_int_frames(events["frame"])
    elif {"frame_start", "frame_end"}.issubset(events.columns):
        starts = _as_int_frames(events["frame_start"], "frame_start")
        ends = _as_int_frames(events["frame_end"], "frame_end")
        if (ends < starts).any():
            raise ValueError("event frame_end precedes frame_start")
        frames = ((starts + ends) // 2).astype(np.int64)
    else:
        raise ValueError("events must contain frame or frame_start/frame_end")
    event_type = (
        events["event_type"].fillna("reviewed_event").astype(str)
        if "event_type" in events
        else pd.Series("reviewed_event", index=events.index)
    )
    if "event_id" in events:
        source_id = events["event_id"].fillna("").astype(str)
    elif "candidate_id" in events:
        source_id = events["candidate_id"].fillna("").astype(str)
    else:
        source_id = pd.Series([f"event_{i:06d}" for i in range(len(events))])
    out = pd.DataFrame(
        {"event_frame": frames.to_numpy(), "event_type": event_type.to_numpy(),
         "source_event_id": source_id.to_numpy()}
    )
    return out.sort_values("event_frame", kind="stable").reset_index(drop=True)


def _contiguous_true_runs(frames: np.ndarray, selected: np.ndarray) -> Iterable[np.ndarray]:
    chosen = np.asarray(frames, dtype=np.int64)[np.asarray(selected, dtype=bool)]
    if chosen.size == 0:
        return []
    splits = np.flatnonzero(np.diff(chosen) > 1) + 1
    return np.split(chosen, splits)


def derive_event_candidates(reference: pd.DataFrame, distance_mm: float = 20.0) -> pd.DataFrame:
    """Derive close-approach and reference-gap event centers without randomness."""

    ref = _validate_reference(reference)
    rows: list[dict[str, Any]] = []
    identities = sorted(ref["human_identity"].unique())
    by_identity = {
        identity: group.set_index("frame")[list(XYZ)]
        for identity, group in ref.groupby("human_identity", sort=True)
    }
    for identity_a, identity_b in permutations(identities, 2):
        if identity_a >= identity_b:
            continue
        joined = by_identity[identity_a].join(
            by_identity[identity_b], how="inner", lsuffix="_a", rsuffix="_b"
        )
        if joined.empty:
            continue
        a = joined[[f"{c}_a" for c in XYZ]].to_numpy(float)
        b = joined[[f"{c}_b" for c in XYZ]].to_numpy(float)
        distance = np.linalg.norm(a - b, axis=1)
        finite_close = np.isfinite(distance) & (distance <= distance_mm)
        frames = joined.index.to_numpy(np.int64)
        for run in _contiguous_true_runs(frames, finite_close):
            positions = np.searchsorted(frames, run)
            center = int(run[int(np.nanargmin(distance[positions]))])
            rows.append(
                {"event_frame": center, "event_type": "close_approach",
                 "source_event_id": f"close_{identity_a}_{identity_b}_{center}"}
            )

    for identity, group in ref.groupby("human_identity", sort=True):
        frames = np.sort(group["frame"].unique().astype(np.int64))
        for left, right in zip(frames[:-1], frames[1:]):
            if right - left > 1:
                center = int((left + right) // 2)
                rows.append(
                    {"event_frame": center, "event_type": "reference_gap",
                     "source_event_id": f"gap_{identity}_{left}_{right}"}
                )
    if not rows:
        center = int((ref["frame"].min() + ref["frame"].max()) // 2)
        rows.append(
            {"event_frame": center, "event_type": "timeline_midpoint",
             "source_event_id": "fallback_midpoint"}
        )
    return pd.DataFrame(rows).sort_values("event_frame", kind="stable").reset_index(drop=True)


def build_heldout_episodes(
    candidates: pd.DataFrame,
    *,
    frame_min: int,
    frame_max: int,
    radius_frames: int,
    guard_frames: int,
    max_episodes: int | None = None,
) -> pd.DataFrame:
    """Create non-overlapping contiguous episodes with disjoint guard bands.

    Candidate windows are merged whenever their *guarded* intervals overlap.
    This guarantees that no reference row can be training context for one event
    while also belonging to another held-out episode.
    """

    if frame_max < frame_min:
        raise ValueError("frame_max precedes frame_min")
    required = {"event_frame", "event_type", "source_event_id"}
    if required - set(candidates.columns):
        raise ValueError("candidate table is not normalized")
    raw: list[dict[str, Any]] = []
    for row in candidates.sort_values("event_frame", kind="stable").itertuples(index=False):
        center = int(row.event_frame)
        if center < frame_min or center > frame_max:
            continue
        raw.append(
            {"start": max(frame_min, center - radius_frames),
             "end": min(frame_max, center + radius_frames),
             "centers": [center], "types": [str(row.event_type)],
             "source_ids": [str(row.source_event_id)]}
        )
    if not raw:
        raise ValueError("no event candidates fall inside the reference frame range")

    merged: list[dict[str, Any]] = []
    for item in raw:
        if merged and item["start"] - guard_frames <= merged[-1]["end"] + guard_frames:
            previous = merged[-1]
            previous["end"] = max(previous["end"], item["end"])
            previous["centers"].extend(item["centers"])
            previous["types"].extend(item["types"])
            previous["source_ids"].extend(item["source_ids"])
        else:
            merged.append(item)

    if max_episodes is not None and len(merged) > max_episodes:
        # Deterministic temporal coverage rather than a random subset.
        indices = np.unique(np.linspace(0, len(merged) - 1, max_episodes).round().astype(int))
        merged = [merged[index] for index in indices]

    rows = []
    for index, item in enumerate(merged, start=1):
        start, end = int(item["start"]), int(item["end"])
        rows.append(
            {"episode_id": f"episode_{index:03d}", "start_frame": start,
             "end_frame": end, "n_frames": end - start + 1,
             "guard_start_frame": max(frame_min, start - guard_frames),
             "guard_end_frame": min(frame_max, end + guard_frames),
             "event_frames": "|".join(map(str, item["centers"])),
             "event_types": "|".join(sorted(set(item["types"]))),
             "source_event_ids": "|".join(item["source_ids"]),
             "n_source_events": len(item["centers"])}
        )
    episodes = pd.DataFrame(rows)
    if len(episodes) > 1:
        if (episodes["start_frame"].to_numpy()[1:] <= episodes["end_frame"].to_numpy()[:-1]).any():
            raise AssertionError("held-out episodes overlap")
        if (
            episodes["guard_start_frame"].to_numpy()[1:]
            <= episodes["guard_end_frame"].to_numpy()[:-1]
        ).any():
            raise AssertionError("held-out guard bands overlap")
    return episodes


def frame_split_masks(frames: np.ndarray, episodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(heldout, train)`` masks; train excludes all episode guard bands."""

    values = np.asarray(frames, dtype=np.int64)
    heldout = np.zeros(values.shape, dtype=bool)
    excluded = np.zeros(values.shape, dtype=bool)
    for row in episodes.itertuples(index=False):
        heldout |= (values >= row.start_frame) & (values <= row.end_frame)
        excluded |= (values >= row.guard_start_frame) & (values <= row.guard_end_frame)
    train = ~excluded
    if np.any(heldout & train):
        raise AssertionError("held-out frames leaked into the training mask")
    return heldout, train


def spatial_matches(
    reference: pd.DataFrame,
    track: pd.DataFrame,
    *,
    label_col: str,
    match_distance_mm: float,
) -> pd.DataFrame:
    """One-to-one spatial association of reviewed and predicted points per frame."""

    ref = _validate_reference(reference)
    pred = _validate_track(track, label_col)
    # Work on arrays and positional group indices.  Constructing 30,000 tiny
    # DataFrames and calling SciPy once per frame was needlessly dominant on a
    # 15k-frame review.  These groups contain at most three fish, so exhaustive
    # rectangular assignment has at most six alternatives and is substantially
    # faster while remaining exactly optimal.
    ref_groups = ref.groupby("frame", sort=False).indices
    pred_groups = pred.groupby("frame", sort=False).indices
    ref_xyz = ref[list(XYZ)].to_numpy(float)
    pred_xyz = pred[list(XYZ)].to_numpy(float)
    ref_human = ref["human_identity"].to_numpy(str)
    pred_label = pred[label_col].to_numpy(dtype=object)
    pred_slot = pred["geometry_slot"].to_numpy(dtype=object) if "geometry_slot" in pred else None
    rows: list[dict[str, Any]] = []
    optional_ref = [
        column for column in ("mean_reprojection_error_px", "max_reprojection_error_px",
                              "reviewed_tracklet_id") if column in ref
    ]
    optional_arrays = {column: ref[column].to_numpy(dtype=object) for column in optional_ref}

    def small_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_ref, n_pred = cost.shape
        if n_ref <= n_pred:
            best_columns = min(
                permutations(range(n_pred), n_ref),
                key=lambda columns: sum(cost[i, column] for i, column in enumerate(columns)),
            )
            return np.arange(n_ref, dtype=int), np.asarray(best_columns, dtype=int)
        pred_indices, ref_indices = small_assignment(cost.T)
        return ref_indices, pred_indices

    for frame in sorted(set(ref_groups) & set(pred_groups)):
        r_index = np.asarray(ref_groups[frame], dtype=int)
        p_index = np.asarray(pred_groups[frame], dtype=int)
        r_index = r_index[np.isfinite(ref_xyz[r_index]).all(axis=1)]
        p_valid = np.isfinite(pred_xyz[p_index]).all(axis=1) & pd.notna(pred_label[p_index])
        p_index = p_index[p_valid]
        if r_index.size == 0 or p_index.size == 0:
            continue
        distance = np.linalg.norm(
            ref_xyz[r_index, None, :] - pred_xyz[p_index][None, :, :],
            axis=2,
        )
        ref_indices, pred_indices = small_assignment(distance)
        for ri, pi in zip(ref_indices, pred_indices):
            value = float(distance[ri, pi])
            if not np.isfinite(value) or value > match_distance_mm:
                continue
            rr_index, pp_index = int(r_index[int(ri)]), int(p_index[int(pi)])
            row = {
                "frame": int(frame), "human_identity": str(ref_human[rr_index]),
                "predicted_label": str(pred_label[pp_index]), "distance_mm": value,
                "reference_row_index": rr_index, "track_row_index": pp_index,
            }
            if pred_slot is not None:
                row["geometry_slot"] = str(pred_slot[pp_index])
            for column in optional_ref:
                row[column] = optional_arrays[column][rr_index]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_label_mapping(matches: pd.DataFrame, train_mask: np.ndarray) -> tuple[dict[str, str], dict[str, Any]]:
    """Fit the arbitrary label permutation using training-context matches only."""

    if len(matches) != len(train_mask):
        raise ValueError("train mask is not aligned to matches")
    fit = matches.loc[np.asarray(train_mask, dtype=bool)]
    if fit.empty:
        raise ValueError("no training-context spatial matches are available")
    labels = sorted(fit["predicted_label"].astype(str).unique())
    humans = sorted(fit["human_identity"].astype(str).unique())
    counts = pd.crosstab(fit["predicted_label"], fit["human_identity"]).reindex(
        index=labels, columns=humans, fill_value=0
    )
    row_index, column_index = linear_sum_assignment(-counts.to_numpy(dtype=float))
    mapping = {labels[int(i)]: humans[int(j)] for i, j in zip(row_index, column_index)}
    support = {
        "training_matches": int(len(fit)),
        "labels": labels,
        "human_identities": humans,
        "count_matrix": counts.astype(int).to_dict(orient="index"),
        "assigned_support": {
            label: int(counts.loc[label, human]) for label, human in mapping.items()
        },
    }
    return mapping, support


def _longest_false_run(present_frames: set[int], start: int, end: int) -> int:
    longest = current = 0
    for frame in range(start, end + 1):
        if frame in present_frames:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def jump_metrics(
    track: pd.DataFrame,
    *,
    label_col: str,
    frame_ranges: list[tuple[int, int]],
    jump_threshold_mm: float,
    severe_jump_threshold_mm: float,
) -> dict[str, Any]:
    """Per-label adjacent-frame displacement QC inside held-out ranges."""

    data = _validate_track(track, label_col)
    selected = np.zeros(len(data), dtype=bool)
    frames = data["frame"].to_numpy(np.int64)
    for start, end in frame_ranges:
        selected |= (frames >= start) & (frames <= end)
    data = data.loc[selected].copy()
    steps: list[float] = []
    for _label, group in data.dropna(subset=[label_col]).groupby(label_col, sort=False):
        ordered = group.sort_values("frame", kind="stable")
        xyz = ordered[list(XYZ)].to_numpy(float)
        frame = ordered["frame"].to_numpy(np.int64)
        delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        valid = (np.diff(frame) == 1) & np.isfinite(delta)
        steps.extend(delta[valid].tolist())
    values = np.asarray(steps, dtype=float)
    return {
        "adjacent_steps": int(values.size),
        "jumps_gt8": int(np.sum(values > jump_threshold_mm)),
        "jumps_gt20": int(np.sum(values > severe_jump_threshold_mm)),
        "jump_rate_gt8": float(np.mean(values > jump_threshold_mm)) if values.size else None,
        "jump_rate_gt20": float(np.mean(values > severe_jump_threshold_mm)) if values.size else None,
        "p95_step_mm": _safe_percentile(values, 95),
        "max_step_mm": float(np.max(values)) if values.size else None,
    }


def _permutation_validity(track: pd.DataFrame, label_col: str, heldout_frames: set[int]) -> dict[str, Any]:
    data = track[track["frame"].isin(heldout_frames)].dropna(subset=[label_col])
    by_frame = data.groupby("frame")[label_col].agg(["count", "nunique"])
    duplicates = by_frame[by_frame["count"] != by_frame["nunique"]]
    complete = by_frame[(by_frame["count"] == 3) & (by_frame["nunique"] == 3)]
    return {
        "observed_frames": int(len(by_frame)),
        "one_each_frames": int(len(complete)),
        "one_each_frame_fraction": float(len(complete) / len(by_frame)) if len(by_frame) else None,
        "duplicate_label_frames": int(len(duplicates)),
    }


def score_method(
    *,
    method: str,
    reference: pd.DataFrame,
    track: pd.DataFrame,
    label_col: str,
    episodes: pd.DataFrame,
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Score one label stream and return summary, episode metrics, and matches."""

    matches = spatial_matches(
        reference, track, label_col=label_col, match_distance_mm=config.match_distance_mm
    )
    heldout_match, train_match = frame_split_masks(matches["frame"].to_numpy(), episodes)
    mapping, mapping_support = fit_label_mapping(matches, train_match)
    matches = matches.copy()
    matches["method"] = method
    matches["split"] = np.where(heldout_match, "heldout", np.where(train_match, "train", "guard"))
    matches["mapped_human_identity"] = matches["predicted_label"].map(mapping)
    matches["identity_correct"] = (
        matches["mapped_human_identity"].astype("string")
        == matches["human_identity"].astype("string")
    )

    episode_rows: list[dict[str, Any]] = []
    reference_identities = sorted(reference["human_identity"].unique())
    for episode in episodes.itertuples(index=False):
        ref_ep = reference[reference["frame"].between(episode.start_frame, episode.end_frame)]
        match_ep = matches[
            matches["frame"].between(episode.start_frame, episode.end_frame)
        ].sort_values(["human_identity", "frame"], kind="stable")
        n_frames = int(episode.n_frames)
        expected_rows = n_frames * len(reference_identities)
        correct = int(match_ep["identity_correct"].sum())
        switches = adjacent_switches = fragments = 0
        longest_gap = 0
        for identity in reference_identities:
            group = match_ep[match_ep["human_identity"] == identity]
            sequence = group["mapped_human_identity"].astype(str).to_numpy()
            identity_frames = group["frame"].to_numpy(np.int64)
            if len(group):
                changed = sequence[1:] != sequence[:-1]
                switches += int(changed.sum())
                adjacent_switches += int(np.sum(changed & (np.diff(identity_frames) == 1)))
                fragments += 1 + int(np.sum(changed | (np.diff(identity_frames) > 1)))
            longest_gap = max(
                longest_gap,
                _longest_false_run(set(map(int, identity_frames)), episode.start_frame, episode.end_frame),
            )
        reprojection = (
            pd.to_numeric(ref_ep["mean_reprojection_error_px"], errors="coerce").to_numpy(float)
            if "mean_reprojection_error_px" in ref_ep else np.array([], dtype=float)
        )
        episode_rows.append(
            {
                "method": method, "episode_id": episode.episode_id,
                "start_frame": int(episode.start_frame), "end_frame": int(episode.end_frame),
                "n_frames": n_frames, "reference_rows": int(len(ref_ep)),
                "reference_row_coverage": float(len(ref_ep) / expected_rows) if expected_rows else None,
                "matched_rows": int(len(match_ep)),
                "matched_reference_fraction": float(len(match_ep) / len(ref_ep)) if len(ref_ep) else None,
                "correct_identity_rows": correct,
                "identity_accuracy_on_matches": float(correct / len(match_ep)) if len(match_ep) else None,
                "identity_recall_of_reference": float(correct / len(ref_ep)) if len(ref_ep) else None,
                "identity_switches": switches, "adjacent_identity_switches": adjacent_switches,
                "identity_fragments": fragments, "longest_association_gap_frames": longest_gap,
                "mean_match_distance_mm": float(match_ep["distance_mm"].mean()) if len(match_ep) else None,
                "p95_match_distance_mm": _safe_percentile(match_ep["distance_mm"].to_numpy(), 95),
                "mean_reprojection_error_px": float(np.nanmean(reprojection)) if np.isfinite(reprojection).any() else None,
                "p95_reprojection_error_px": _safe_percentile(reprojection, 95),
            }
        )
    episode_metrics = pd.DataFrame(episode_rows)

    heldout = matches[matches["split"] == "heldout"]
    heldout_ref = reference[
        frame_split_masks(reference["frame"].to_numpy(), episodes)[0]
    ]
    correct = int(heldout["identity_correct"].sum())
    heldout_reprojection = (
        pd.to_numeric(heldout_ref["mean_reprojection_error_px"], errors="coerce").to_numpy(float)
        if "mean_reprojection_error_px" in heldout_ref else np.array([], dtype=float)
    )
    total_episode_frames = int(episodes["n_frames"].sum())
    expected_reference_rows = total_episode_frames * len(reference_identities)
    reference_counts = heldout_ref.groupby("frame")["human_identity"].nunique()
    ranges = [(int(r.start_frame), int(r.end_frame)) for r in episodes.itertuples(index=False)]
    heldout_frames = set(
        frame for start, end in ranges for frame in range(start, end + 1)
    )
    summary = {
        "method": method,
        "label_column": label_col,
        "label_to_reviewed_identity": mapping,
        "mapping_fit": mapping_support,
        "heldout_reference_rows": int(len(heldout_ref)),
        "heldout_reference_row_coverage": (
            float(len(heldout_ref) / expected_reference_rows) if expected_reference_rows else None
        ),
        "heldout_reference_frames_any": int(len(reference_counts)),
        "heldout_reference_frames_all_identities": int(
            np.sum(reference_counts.to_numpy() == len(reference_identities))
        ),
        "heldout_spatial_matches": int(len(heldout)),
        "heldout_match_coverage": float(len(heldout) / len(heldout_ref)) if len(heldout_ref) else None,
        "heldout_correct_identity_rows": correct,
        "heldout_identity_accuracy_on_matches": float(correct / len(heldout)) if len(heldout) else None,
        "heldout_identity_recall_of_reference": float(correct / len(heldout_ref)) if len(heldout_ref) else None,
        "heldout_idf1_like": (
            float(2 * correct / (len(heldout_ref) + len(heldout)))
            if len(heldout_ref) + len(heldout) else None
        ),
        "mean_match_distance_mm": float(heldout["distance_mm"].mean()) if len(heldout) else None,
        "p95_match_distance_mm": _safe_percentile(heldout["distance_mm"].to_numpy(), 95),
        "mean_reprojection_error_px": (
            float(np.nanmean(heldout_reprojection))
            if np.isfinite(heldout_reprojection).any() else None
        ),
        "p95_reprojection_error_px": _safe_percentile(heldout_reprojection, 95),
        "max_reprojection_error_px": (
            float(np.nanmax(heldout_reprojection))
            if np.isfinite(heldout_reprojection).any() else None
        ),
        "episode_identity_switches": int(episode_metrics["identity_switches"].sum()),
        "episode_adjacent_identity_switches": int(episode_metrics["adjacent_identity_switches"].sum()),
        "max_association_gap_frames": int(episode_metrics["longest_association_gap_frames"].max()),
        "jumps": jump_metrics(
            track, label_col=label_col, frame_ranges=ranges,
            jump_threshold_mm=config.jump_threshold_mm,
            severe_jump_threshold_mm=config.severe_jump_threshold_mm,
        ),
        "permutation_validity": _permutation_validity(track, label_col, heldout_frames),
    }
    return summary, episode_metrics, matches


def run_benchmark(
    *,
    reference: pd.DataFrame,
    baseline_track: pd.DataFrame,
    baseline_label_col: str,
    events: pd.DataFrame | None = None,
    optimized_track: pd.DataFrame | None = None,
    optimized_label_col: str | None = None,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> dict[str, Any]:
    """Run the complete leakage-safe held-out comparison in memory."""

    ref = _validate_reference(reference)
    candidates = (
        load_event_candidates(events)
        if events is not None
        else derive_event_candidates(ref, config.event_distance_mm)
    )
    episodes = build_heldout_episodes(
        candidates, frame_min=int(ref["frame"].min()), frame_max=int(ref["frame"].max()),
        radius_frames=config.episode_radius_frames, guard_frames=config.guard_frames,
        max_episodes=config.max_episodes,
    )
    ref_heldout, ref_train = frame_split_masks(ref["frame"].to_numpy(), episodes)
    if np.any(ref_heldout & ref_train):
        raise AssertionError("reference split leakage")

    methods: list[tuple[str, pd.DataFrame, str]] = [
        ("baseline", baseline_track, baseline_label_col)
    ]
    if optimized_label_col is not None:
        methods.append(
            ("optimized", optimized_track if optimized_track is not None else baseline_track,
             optimized_label_col)
        )
    summaries, episode_tables, match_tables = [], [], []
    for method, track, label_col in methods:
        summary, episode_metrics, matches = score_method(
            method=method, reference=ref, track=track, label_col=label_col,
            episodes=episodes, config=config,
        )
        summaries.append(summary)
        episode_tables.append(episode_metrics)
        match_tables.append(matches)

    leakage_audit = {
        "heldout_reference_rows": int(ref_heldout.sum()),
        "training_reference_rows": int(ref_train.sum()),
        "guard_only_reference_rows": int((~ref_heldout & ~ref_train).sum()),
        "heldout_training_intersection_rows": int((ref_heldout & ref_train).sum()),
        "episodes_non_overlapping": True,
        "guard_bands_non_overlapping": True,
        "identity_mapping_fit_uses": "training frames outside all held-out episodes and guard bands",
    }
    return {
        "config": config,
        "episodes": episodes,
        "method_summaries": summaries,
        "episode_metrics": pd.concat(episode_tables, ignore_index=True),
        "matches": pd.concat(match_tables, ignore_index=True),
        "leakage_audit": leakage_audit,
        "event_candidates": candidates,
    }


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


def write_benchmark_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write inspectable CSV/JSON benchmark artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "episodes": output_dir / "heldout_episodes.csv",
        "event_candidates": output_dir / "event_candidates.csv",
        "episode_metrics": output_dir / "episode_metrics.csv",
        "matches": output_dir / "association_matches.csv",
        "method_summary": output_dir / "method_summary.csv",
        "report": output_dir / "heldout_association_report.json",
    }
    result["episodes"].to_csv(paths["episodes"], index=False)
    result["event_candidates"].to_csv(paths["event_candidates"], index=False)
    result["episode_metrics"].to_csv(paths["episode_metrics"], index=False)
    result["matches"].to_csv(paths["matches"], index=False)
    summary_rows = []
    for item in result["method_summaries"]:
        row = {key: value for key, value in item.items() if not isinstance(value, dict)}
        row.update({f"jumps_{key}": value for key, value in item["jumps"].items()})
        row.update(
            {f"permutation_{key}": value for key, value in item["permutation_validity"].items()}
        )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(paths["method_summary"], index=False)
    report = {
        "schema_version": "heldout_association_benchmark_v1",
        "config": asdict(result["config"]),
        "leakage_audit": result["leakage_audit"],
        "episodes": result["episodes"].to_dict(orient="records"),
        "method_summaries": result["method_summaries"],
        "outputs": {key: str(path) for key, path in paths.items() if key != "report"},
    }
    paths["report"].write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--baseline-track-csv", type=Path, required=True)
    parser.add_argument("--baseline-label-col", default="id_final")
    parser.add_argument("--optimized-track-csv", type=Path)
    parser.add_argument("--optimized-label-col")
    parser.add_argument("--events-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-radius-frames", type=int, default=166)
    parser.add_argument("--guard-frames", type=int, default=166)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--event-distance-mm", type=float, default=20.0)
    parser.add_argument("--match-distance-mm", type=float, default=35.0)
    parser.add_argument("--jump-threshold-mm", type=float, default=8.0)
    parser.add_argument("--severe-jump-threshold-mm", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = pd.read_csv(args.reference_csv, low_memory=False)
    baseline = pd.read_csv(args.baseline_track_csv, low_memory=False)
    optimized = (
        pd.read_csv(args.optimized_track_csv, low_memory=False)
        if args.optimized_track_csv is not None else None
    )
    events = pd.read_csv(args.events_csv, low_memory=False) if args.events_csv else None
    config = BenchmarkConfig(
        episode_radius_frames=args.episode_radius_frames,
        guard_frames=args.guard_frames,
        max_episodes=args.max_episodes,
        event_distance_mm=args.event_distance_mm,
        match_distance_mm=args.match_distance_mm,
        jump_threshold_mm=args.jump_threshold_mm,
        severe_jump_threshold_mm=args.severe_jump_threshold_mm,
    )
    result = run_benchmark(
        reference=reference, baseline_track=baseline,
        baseline_label_col=args.baseline_label_col, events=events,
        optimized_track=optimized, optimized_label_col=args.optimized_label_col,
        config=config,
    )
    paths = write_benchmark_outputs(result, args.output_dir)
    print(json.dumps({"methods": result["method_summaries"],
                      "leakage_audit": result["leakage_audit"],
                      "report": str(paths["report"])}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
