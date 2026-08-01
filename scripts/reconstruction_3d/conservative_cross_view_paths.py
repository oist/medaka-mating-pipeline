from __future__ import annotations

"""Purity-first cross-view grouping and optional-unassigned three-path solver.

Inputs are hypotheses built from independently tracked strict-mouth fragments.
The objective uses robust per-group evidence and a bounded saturating support
bonus. It never sums residual by duration and never requires every fragment to
be assigned. Two-view groups that exhibit the reviewed count-mismatch failure
mode are explicitly rejected.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import CandidateWindow


VIEW_ORDER = ("top_cam", "side1_cam", "side2_cam")


@dataclass(frozen=True)
class CrossViewGroupConfig:
    minimum_support_frames: int = 3
    minimum_median_mouth_confidence: float = 0.3
    maximum_median_ray_residual_mm2: float = 5.0
    support_tau_frames: float = 20.0
    support_bonus_weight: float = 2.0
    residual_weight: float = 0.5
    residual_mad_weight: float = 0.25
    confidence_weight: float = 0.5
    missing_view_penalty: float = 0.75
    reject_retained_count_mismatch: bool = True

    def validate(self) -> None:
        positive = {
            "minimum_support_frames": self.minimum_support_frames,
            "support_tau_frames": self.support_tau_frames,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        finite_non_negative = {
            "minimum_median_mouth_confidence": (
                self.minimum_median_mouth_confidence
            ),
            "maximum_median_ray_residual_mm2": (
                self.maximum_median_ray_residual_mm2
            ),
            "support_bonus_weight": self.support_bonus_weight,
            "residual_weight": self.residual_weight,
            "residual_mad_weight": self.residual_mad_weight,
            "confidence_weight": self.confidence_weight,
            "missing_view_penalty": self.missing_view_penalty,
        }
        for name, value in finite_non_negative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ThreePathConfig:
    number_of_paths: int = 3
    beam_width: int = 256
    maximum_link_gap_frames: int = 100
    max_jump_mm: float = 8.0
    max_jump_hard_cap_mm: float = 80.0
    velocity_link_weight: float = 1.0
    maximum_velocity_angle_degrees: float = 120.0
    all_views_break_penalty: float = 0.5
    segment_restart_penalty: float = 1.0
    eligible_group_drop_cost: float = 0.5

    def validate(self) -> None:
        if self.number_of_paths != 3:
            raise ValueError("the controlled A4 comparison requires three paths")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.maximum_link_gap_frames < 0:
            raise ValueError("maximum_link_gap_frames must be non-negative")
        for name in (
            "max_jump_mm",
            "max_jump_hard_cap_mm",
            "velocity_link_weight",
            "maximum_velocity_angle_degrees",
            "all_views_break_penalty",
            "segment_restart_penalty",
            "eligible_group_drop_cost",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_velocity_angle_degrees > 180:
            raise ValueError("maximum_velocity_angle_degrees cannot exceed 180")


@dataclass(frozen=True)
class ConservativePathResult:
    selected_observations: pd.DataFrame
    groups: pd.DataFrame
    paths: pd.DataFrame
    links: pd.DataFrame
    dispositions: pd.DataFrame
    summary: Mapping[str, object]


def _stable_id(value: object, prefix: str) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _robust_velocity(frames: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Bounded robust velocity using deterministic logarithmic lags."""
    slopes: list[np.ndarray] = []
    for lag in (1, 2, 4, 8, 16, 32, 64):
        if lag >= len(frames):
            break
        delta = frames[lag:] - frames[:-lag]
        valid = delta > 0
        if np.any(valid):
            slopes.append(
                (xyz[lag:][valid] - xyz[:-lag][valid])
                / delta[valid, None]
            )
    return (
        np.median(np.concatenate(slopes, axis=0), axis=0)
        if slopes
        else np.zeros(3, dtype=float)
    )


def _row_evidence(
    hypotheses: pd.DataFrame,
    windows: Mapping[str, CandidateWindow],
    retained_tracklet_ids: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    required = {"frame", "view_mask", "ray_residual_mm2", "view_count"}
    required.update(f"slot_cam{index}" for index in range(3))
    required.update(f"tracklet_{view}" for view in VIEW_ORDER)
    missing = sorted(required - set(hypotheses.columns))
    if missing:
        raise ValueError(f"hypothesis table lacks required columns: {missing}")
    if set(windows) != set(VIEW_ORDER):
        raise ValueError(f"windows must contain {VIEW_ORDER}")
    if set(retained_tracklet_ids) != set(VIEW_ORDER):
        raise ValueError(f"retained_tracklet_ids must contain {VIEW_ORDER}")
    starts = {window.frame_start for window in windows.values()}
    ends = {window.frame_end for window in windows.values()}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("candidate windows are not aligned")
    frame_start = starts.pop()
    frame_end = ends.pop()
    table = hypotheses.copy()
    confidence: list[float] = []
    count_mismatch: list[bool] = []
    retained_counts = {}
    for view, window in windows.items():
        ids = np.asarray(retained_tracklet_ids[view])
        if ids.shape != window.mouth_xy.shape[:2]:
            raise ValueError(
                f"retained tracklet ID shape for {view} is {ids.shape}, "
                f"expected {window.mouth_xy.shape[:2]}"
            )
        retained_counts[view] = (ids >= 0).sum(axis=1)
    for row in table.itertuples(index=False):
        frame = int(row.frame)
        if not frame_start <= frame <= frame_end:
            raise ValueError(f"hypothesis frame {frame} lies outside windows")
        local = frame - frame_start
        values = []
        frame_retained_counts = []
        for view_index, view in enumerate(VIEW_ORDER):
            slot = int(getattr(row, f"slot_cam{view_index}"))
            count = int(retained_counts[view][local])
            frame_retained_counts.append(count)
            if slot >= 0:
                score = float(windows[view].mouth_score[local, slot])
                if np.isfinite(score):
                    values.append(score)
        confidence.append(float(np.median(values)) if values else np.nan)
        count_mismatch.append(
            bool(max(frame_retained_counts) > min(frame_retained_counts))
        )
    table["mouth_confidence"] = confidence
    table["retained_tracklet_count_mismatch"] = count_mismatch
    return table


def build_cross_view_groups(
    hypotheses: pd.DataFrame,
    windows: Mapping[str, CandidateWindow],
    retained_tracklet_ids: Mapping[str, np.ndarray],
    *,
    config: CrossViewGroupConfig = CrossViewGroupConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build conservative cross-view fragment groups and scored observations."""
    config.validate()
    table = _row_evidence(
        hypotheses, windows, retained_tracklet_ids
    )
    present_tracklets = np.ones(len(table), dtype=bool)
    for view_index, view in enumerate(VIEW_ORDER):
        declared = table.view_mask.astype(int).map(
            lambda mask, index=view_index: bool(mask & (1 << index))
        )
        present_tracklets &= ~declared | table[f"tracklet_{view}"].ge(0)
    table["all_declared_tracklets_retained"] = present_tracklets
    table = table[
        table.all_declared_tracklets_retained
        & np.isfinite(table.ray_residual_mm2)
        & np.isfinite(table.mouth_confidence)
    ].copy()
    if table.empty:
        return pd.DataFrame(), table

    key_columns = ["view_mask", *[f"tracklet_{view}" for view in VIEW_ORDER]]
    group_records: list[dict[str, object]] = []
    observation_parts: list[pd.DataFrame] = []
    for key, rows in table.groupby(key_columns, sort=True, dropna=False):
        rows = rows.sort_values(
            ["frame", "ray_residual_mm2", "hypothesis_id"]
        ).drop_duplicates("frame", keep="first")
        view_mask = int(key[0])
        tracklets = tuple(int(value) for value in key[1:])
        frames = rows.frame.to_numpy(dtype=int)
        xyz = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
        residual = rows.ray_residual_mm2.to_numpy(dtype=float)
        confidence = rows.mouth_confidence.to_numpy(dtype=float)
        support = int(len(np.unique(frames)))
        residual_median = float(np.median(residual))
        residual_mad = float(
            np.median(np.abs(residual - residual_median))
        )
        confidence_median = float(np.median(confidence))
        mismatch_fraction = float(
            rows.retained_tracklet_count_mismatch.mean()
        )
        support_bonus = float(
            config.support_bonus_weight
            * (1.0 - np.exp(-support / config.support_tau_frames))
        )
        assignment_cost = float(
            config.residual_weight * np.log1p(max(0.0, residual_median))
            + config.residual_mad_weight * residual_mad
            + config.confidence_weight * (1.0 - confidence_median)
            + config.missing_view_penalty * (3 - int(view_mask.bit_count()))
            - support_bonus
        )
        reasons: list[str] = []
        if support < config.minimum_support_frames:
            reasons.append("support_below_minimum")
        if confidence_median < config.minimum_median_mouth_confidence:
            reasons.append("mouth_confidence_below_minimum")
        if residual_median > config.maximum_median_ray_residual_mm2:
            reasons.append("median_residual_above_maximum")
        if (
            config.reject_retained_count_mismatch
            and mismatch_fraction > 0
        ):
            reasons.append("retained_tracklet_count_mismatch")
        group_id = _stable_id(
            {
                "view_mask": view_mask,
                "tracklets": tracklets,
                "frames": frames.tolist(),
            },
            "xvg",
        )
        velocity = _robust_velocity(frames, xyz)
        record = {
            "group_id": group_id,
            "view_mask": view_mask,
            "view_count": int(view_mask.bit_count()),
            **{
                f"tracklet_{view}": tracklets[index]
                for index, view in enumerate(VIEW_ORDER)
            },
            "frame_start": int(frames.min()),
            "frame_end": int(frames.max()),
            "support_frames": support,
            "span_frames": int(frames.max() - frames.min() + 1),
            "median_ray_residual_mm2": residual_median,
            "mad_ray_residual_mm2": residual_mad,
            "median_mouth_confidence": confidence_median,
            "retained_tracklet_count_mismatch_fraction": mismatch_fraction,
            "support_bonus": support_bonus,
            "assignment_cost": assignment_cost,
            "start_X": float(xyz[0, 0]),
            "start_Y": float(xyz[0, 1]),
            "start_Z": float(xyz[0, 2]),
            "end_X": float(xyz[-1, 0]),
            "end_Y": float(xyz[-1, 1]),
            "end_Z": float(xyz[-1, 2]),
            "velocity_X": float(velocity[0]),
            "velocity_Y": float(velocity[1]),
            "velocity_Z": float(velocity[2]),
            "eligible": not reasons,
            "rejection_reason": "|".join(reasons) if reasons else "eligible",
        }
        group_records.append(record)
        enriched = rows.copy()
        enriched["group_id"] = group_id
        enriched["group_assignment_cost"] = assignment_cost
        enriched["group_support_frames"] = support
        enriched["group_median_ray_residual_mm2"] = residual_median
        enriched["group_mad_ray_residual_mm2"] = residual_mad
        enriched["group_median_mouth_confidence"] = confidence_median
        enriched["group_retained_count_mismatch_fraction"] = mismatch_fraction
        enriched["group_eligible"] = not reasons
        enriched["group_rejection_reason"] = record["rejection_reason"]
        observation_parts.append(enriched)
    groups = pd.DataFrame.from_records(group_records).sort_values(
        ["frame_start", "frame_end", "group_id"]
    ).reset_index(drop=True)
    observations = pd.concat(observation_parts, ignore_index=True).sort_values(
        ["frame", "group_id", "ray_residual_mm2"]
    ).reset_index(drop=True)
    return groups, observations


@dataclass(frozen=True)
class _BeamState:
    cost: float
    last_groups: tuple[str | None, ...]
    resource_owner: tuple[tuple[str, int, int], ...]
    history_node: int


def _resources(group: Mapping[str, object]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (view, int(group[f"tracklet_{view}"]))
        for index, view in enumerate(VIEW_ORDER)
        if int(group["view_mask"]) & (1 << index)
    )


def _edge_cost(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    *,
    config: ThreePathConfig,
) -> tuple[float, dict[str, object]] | None:
    elapsed = int(current["frame_start"]) - int(previous["frame_end"])
    missing_gap = elapsed - 1
    if elapsed <= 0:
        return None
    previous_resources = set(_resources(previous))
    current_resources = set(_resources(current))
    all_views_break = not bool(previous_resources & current_resources)
    if missing_gap > config.maximum_link_gap_frames:
        return config.segment_restart_penalty, {
            "gap_frames": missing_gap,
            "elapsed_frames": elapsed,
            "direct_distance_mm": np.nan,
            "forward_prediction_error_mm": np.nan,
            "backward_prediction_error_mm": np.nan,
            "symmetric_prediction_error_mm": np.nan,
            "velocity_angle_degrees": np.nan,
            "hard_gate_mm": np.nan,
            "all_views_break": all_views_break,
            "segment_restart": True,
            "continuity_claimed": False,
            "edge_cost": config.segment_restart_penalty,
        }
    previous_end = np.asarray(
        [previous["end_X"], previous["end_Y"], previous["end_Z"]], dtype=float
    )
    velocity = np.asarray(
        [
            previous["velocity_X"],
            previous["velocity_Y"],
            previous["velocity_Z"],
        ],
        dtype=float,
    )
    current_velocity = np.asarray(
        [
            current["velocity_X"],
            current["velocity_Y"],
            current["velocity_Z"],
        ],
        dtype=float,
    )
    current_start = np.asarray(
        [current["start_X"], current["start_Y"], current["start_Z"]],
        dtype=float,
    )
    predicted = previous_end + velocity * elapsed
    forward_prediction_error = float(
        np.linalg.norm(current_start - predicted)
    )
    backward_predicted = current_start - current_velocity * elapsed
    backward_prediction_error = float(
        np.linalg.norm(previous_end - backward_predicted)
    )
    symmetric_prediction_error = (
        forward_prediction_error + backward_prediction_error
    ) / 2.0
    direct_distance = float(np.linalg.norm(current_start - previous_end))
    hard_gate = min(
        config.max_jump_mm * elapsed,
        config.max_jump_hard_cap_mm,
    )
    if direct_distance > hard_gate:
        return None
    previous_speed = float(np.linalg.norm(velocity))
    current_speed = float(np.linalg.norm(current_velocity))
    angle = np.nan
    if previous_speed > 1e-9 and current_speed > 1e-9:
        cosine = float(
            np.clip(
                np.dot(velocity, current_velocity)
                / (previous_speed * current_speed),
                -1.0,
                1.0,
            )
        )
        angle = float(np.degrees(np.arccos(cosine)))
        if angle > config.maximum_velocity_angle_degrees:
            return None
    cost = (
        config.velocity_link_weight
        * symmetric_prediction_error
        / max(hard_gate, np.finfo(float).eps)
        + (
            config.all_views_break_penalty
            if all_views_break
            else 0.0
        )
    )
    return cost, {
        "gap_frames": missing_gap,
        "elapsed_frames": elapsed,
        "direct_distance_mm": direct_distance,
        "forward_prediction_error_mm": forward_prediction_error,
        "backward_prediction_error_mm": backward_prediction_error,
        "symmetric_prediction_error_mm": symmetric_prediction_error,
        "velocity_angle_degrees": angle,
        "hard_gate_mm": hard_gate,
        "all_views_break": all_views_break,
        "segment_restart": False,
        "continuity_claimed": True,
        "edge_cost": cost,
    }


def solve_three_identity_paths(
    groups: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    config: ThreePathConfig = ThreePathConfig(),
    group_config: CrossViewGroupConfig = CrossViewGroupConfig(),
) -> ConservativePathResult:
    """Assign eligible groups to three paths while allowing explicit drops."""
    config.validate()
    group_config.validate()
    if groups.empty:
        return ConservativePathResult(
            pd.DataFrame(),
            groups.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "schema_version": 1,
                "status": "insufficient_eligible_groups",
                "group_config": asdict(group_config),
                "path_config": asdict(config),
            },
        )
    eligible = groups[groups.eligible].sort_values(
        ["frame_start", "frame_end", "group_id"]
    )
    records = {
        str(row.group_id): row._asdict()
        for row in eligible.itertuples(index=False)
    }
    record_items = list(records.items())
    final_resource_use: dict[tuple[str, int], int] = {}
    for group_index, (_group_id, group) in enumerate(record_items):
        for resource in _resources(group):
            final_resource_use[resource] = group_index
    # Each history node is (parent_node, group_id, path_index). Dropped groups
    # need no node because dispositions are reconstructed by set difference.
    history_nodes: list[tuple[int, str, int] | None] = [None]
    empty_last = tuple(None for _ in range(config.number_of_paths))
    states = [_BeamState(0.0, empty_last, (), 0)]
    for group_index, (group_id, group) in enumerate(record_items):
        expanded: list[_BeamState] = []
        for state in states:
            # Explicitly leave ambiguous/reflection-like groups unassigned.
            pruned_owners = tuple(
                entry
                for entry in state.resource_owner
                if final_resource_use[(entry[0], entry[1])] > group_index
            )
            expanded.append(
                _BeamState(
                    cost=state.cost + config.eligible_group_drop_cost,
                    last_groups=state.last_groups,
                    resource_owner=pruned_owners,
                    history_node=state.history_node,
                )
            )
            owners = {
                (view, tracklet): owner
                for view, tracklet, owner in state.resource_owner
            }
            nonempty = [
                index
                for index, last in enumerate(state.last_groups)
                if last is not None
            ]
            empty = [
                index
                for index, last in enumerate(state.last_groups)
                if last is None
            ]
            candidate_paths = [*nonempty, *(empty[:1])]
            for path_index in candidate_paths:
                resources = _resources(group)
                if any(
                    resource in owners and owners[resource] != path_index
                    for resource in resources
                ):
                    continue
                edge_value = 0.0
                previous_id = state.last_groups[path_index]
                if previous_id is not None:
                    edge = _edge_cost(
                        records[previous_id], group, config=config
                    )
                    if edge is None:
                        continue
                    edge_value = float(edge[0])
                last_groups = list(state.last_groups)
                last_groups[path_index] = group_id
                updated_owners = dict(owners)
                for resource in resources:
                    updated_owners[resource] = path_index
                kept_owners = tuple(
                    sorted(
                        (view, tracklet, owner)
                        for (view, tracklet), owner
                        in updated_owners.items()
                        if final_resource_use[(view, tracklet)] > group_index
                    )
                )
                history_nodes.append(
                    (state.history_node, group_id, path_index)
                )
                expanded.append(
                    _BeamState(
                        cost=(
                            state.cost
                            + float(group["assignment_cost"])
                            + edge_value
                        ),
                        last_groups=tuple(last_groups),
                        resource_owner=kept_owners,
                        history_node=len(history_nodes) - 1,
                    )
                )
        # Retain the best instance of each deterministic structural state.
        best_by_structure: dict[
            tuple[tuple[str | None, ...], tuple[tuple[str, int, int], ...]],
            _BeamState,
        ] = {}
        for state in expanded:
            key = (state.last_groups, state.resource_owner)
            old = best_by_structure.get(key)
            if old is None or state.cost < old.cost:
                best_by_structure[key] = state
        states = sorted(
            best_by_structure.values(),
            key=lambda state: (
                state.cost,
                tuple(value or "" for value in state.last_groups),
                state.resource_owner,
            ),
        )[: config.beam_width]
    complete = [
        state
        for state in states
        if all(value is not None for value in state.last_groups)
    ]
    if not complete:
        dispositions = groups[["group_id", "eligible", "rejection_reason"]].copy()
        dispositions["selected"] = False
        dispositions["path_id"] = ""
        return ConservativePathResult(
            pd.DataFrame(),
            groups.copy(),
            pd.DataFrame(),
            pd.DataFrame(),
            dispositions,
            {
                "schema_version": 1,
                "status": "insufficient_three_path_solution",
                "eligible_groups": int(len(eligible)),
                "group_config": asdict(group_config),
                "path_config": asdict(config),
            },
        )
    winner = min(
        complete,
        key=lambda state: (
            state.cost,
            tuple(value or "" for value in state.last_groups),
            state.resource_owner,
        ),
    )
    assignments: list[tuple[str, int]] = []
    node_index = winner.history_node
    while node_index:
        node = history_nodes[node_index]
        assert node is not None
        node_index, group_id, path_index = node
        assignments.append((group_id, path_index))
    assignments.reverse()
    winning_paths: list[list[str]] = [
        [] for _ in range(config.number_of_paths)
    ]
    for group_id, path_index in assignments:
        winning_paths[path_index].append(group_id)
    path_records: list[dict[str, object]] = []
    link_records: list[dict[str, object]] = []
    selected_group_to_path: dict[str, str] = {}
    selected_group_to_segment: dict[str, int] = {}
    for path_index, group_ids in enumerate(winning_paths):
        path_id = f"PATH_{path_index + 1}"
        segment_id = 0
        for order, group_id in enumerate(group_ids):
            edge = None
            if order:
                previous_id = group_ids[order - 1]
                edge = _edge_cost(
                    records[previous_id], records[group_id], config=config
                )
                assert edge is not None
                if bool(edge[1]["segment_restart"]):
                    segment_id += 1
            selected_group_to_path[group_id] = path_id
            selected_group_to_segment[group_id] = segment_id
            path_records.append(
                {
                    "path_id": path_id,
                    "path_segment": segment_id,
                    "path_order": order,
                    "group_id": group_id,
                }
            )
            if order:
                assert edge is not None
                link_records.append(
                    {
                        "path_id": path_id,
                        "source_group_id": previous_id,
                        "target_group_id": group_id,
                        **edge[1],
                        "accepted": True,
                    }
                )
    selected_ids = set(selected_group_to_path)
    selected = observations[
        observations.group_id.astype(str).isin(selected_ids)
    ].copy()
    selected["path_id"] = selected.group_id.astype(str).map(
        selected_group_to_path
    )
    selected["path_segment"] = selected.group_id.astype(str).map(
        selected_group_to_segment
    ).astype(int)
    selected = selected.sort_values(
        ["frame", "path_id", "ray_residual_mm2", "hypothesis_id"]
    ).drop_duplicates(["frame", "path_id"], keep="first")
    dispositions = groups[
        ["group_id", "eligible", "rejection_reason"]
    ].copy()
    dispositions["selected"] = dispositions.group_id.astype(str).isin(
        selected_ids
    )
    dispositions["path_id"] = dispositions.group_id.astype(str).map(
        selected_group_to_path
    ).fillna("")
    links = pd.DataFrame.from_records(link_records)
    paths = pd.DataFrame.from_records(path_records)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "objective_cost": winner.cost,
        "group_config": asdict(group_config),
        "path_config": asdict(config),
        "support_bonus_definition": (
            "weight * (1 - exp(-support_frames / tau_frames))"
        ),
        "residual_aggregation": "median plus median absolute deviation",
        "residual_duration_sum_used": False,
        "optional_unassigned": True,
        "eligible_groups": int(len(eligible)),
        "selected_groups": int(len(selected_ids)),
        "unassigned_eligible_groups": int(len(eligible) - len(selected_ids)),
        "eligible_group_coverage": (
            float(len(selected_ids) / len(eligible)) if len(eligible) else 0.0
        ),
        "drop_cost_definition": (
            "each unassigned eligible group adds eligible_group_drop_cost"
        ),
        "rejected_count_mismatch_groups": int(
            groups.rejection_reason.str.contains(
                "retained_tracklet_count_mismatch", regex=False
            ).sum()
        ),
        "selected_observations": int(len(selected)),
        "paths": config.number_of_paths,
        "all_views_break_links": int(
            links.all_views_break.sum() if len(links) else 0
        ),
        "segment_restart_links": int(
            links.segment_restart.sum() if len(links) else 0
        ),
    }
    return ConservativePathResult(
        selected_observations=selected.reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        paths=paths,
        links=links,
        dispositions=dispositions,
        summary=summary,
    )
