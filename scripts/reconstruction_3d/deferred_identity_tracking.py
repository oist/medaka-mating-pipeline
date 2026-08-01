from __future__ import annotations

"""Framewise 3D identity linking with bounded ambiguity deferral.

This module consumes already-selected per-frame geometry. It never changes
cross-view candidate pairing. Near-tied temporal assignments retain a short
beam until later separation provides a stable lead. Intervals that never
resolve, or where a high-confidence tracklet hint conflicts with the hard 3D
motion gate, keep geometry in audit fields but expose no forced identity.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.cross_frame_tracking import (
    _FORBIDDEN_COST,
    _TrackState,
    _get_points_from_groups,
    _motion_cost,
)


@dataclass(frozen=True)
class DeferredTrackingConfig:
    num_individuals: int = 3
    max_jump_mm: float = 8.0
    max_jump_hard_cap_mm: float = 80.0
    max_gap_frames: int = 100
    velocity_prediction_weight: float = 1.0
    tangent_weight: float = 0.0
    acceleration_weight: float = 0.05
    motion_uncertainty_growth_per_frame: float = 0.25
    max_motion_gap_frames: int | None = 20
    motion_history_frames: int = 5
    beam_width: int = 16
    entry_margin: float = 0.5
    retention_margin: float = 2.0
    resolution_margin: float = 1.0
    resolution_stability_frames: int = 3
    minimum_lookahead_frames: int = 3
    maximum_defer_horizon: int = 100

    def validate(self) -> None:
        finite_non_negative = {
            "max_jump_mm": self.max_jump_mm,
            "max_jump_hard_cap_mm": self.max_jump_hard_cap_mm,
            "velocity_prediction_weight": self.velocity_prediction_weight,
            "tangent_weight": self.tangent_weight,
            "acceleration_weight": self.acceleration_weight,
            "motion_uncertainty_growth_per_frame": (
                self.motion_uncertainty_growth_per_frame
            ),
            "entry_margin": self.entry_margin,
            "retention_margin": self.retention_margin,
            "resolution_margin": self.resolution_margin,
        }
        for name, value in finite_non_negative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        positive = {
            "num_individuals": self.num_individuals,
            "max_gap_frames": self.max_gap_frames,
            "motion_history_frames": self.motion_history_frames,
            "beam_width": self.beam_width,
            "resolution_stability_frames": self.resolution_stability_frames,
            "minimum_lookahead_frames": self.minimum_lookahead_frames,
            "maximum_defer_horizon": self.maximum_defer_horizon,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.motion_history_frames < 2:
            raise ValueError("motion_history_frames must be at least 2")


@dataclass(frozen=True)
class DeferredTrackingResult:
    tracked_by_frame: Dict[int, Dict]
    assignment_audit: pd.DataFrame
    unresolved_intervals: tuple[Mapping[str, object], ...]
    configuration: Mapping[str, object]


@dataclass
class _Branch:
    mapping: dict[int, int]
    local_cost: float
    diagnostics: dict[tuple[int, int], dict[str, object]]
    full_assignment: bool
    constraint_conflict: bool


@dataclass
class _Hypothesis:
    states: dict[str, _TrackState | None]
    cumulative_cost: float
    pending: list[tuple[int, Dict]]
    mapping_signature: tuple[tuple[int, ...], ...]
    constraint_conflict: bool = False


def _id_labels(count: int) -> list[str]:
    return [f"ID_{index + 1}" for index in range(count)]


def _retire_states(
    states: dict[str, _TrackState | None],
    frame: int,
    *,
    max_gap_frames: int,
) -> list[str]:
    retired: list[str] = []
    for identity, state in states.items():
        if state is not None and frame - state.frame > max_gap_frames:
            states[identity] = None
            retired.append(identity)
    return retired


def _enumerate_branches(
    states: dict[str, _TrackState | None],
    points: np.ndarray,
    frame: int,
    *,
    config: DeferredTrackingConfig,
    source_hints: Mapping[int, str] | None,
) -> list[_Branch]:
    labels = _id_labels(config.num_individuals)
    active = [
        index for index, identity in enumerate(labels) if states[identity] is not None
    ]
    valid = [
        index
        for index in range(config.num_individuals)
        if np.all(np.isfinite(points[index]))
    ]
    pair_cost = np.full(
        (config.num_individuals, config.num_individuals),
        _FORBIDDEN_COST,
        dtype=float,
    )
    pair_diagnostics: dict[tuple[int, int], dict[str, object]] = {}
    for identity_index in active:
        identity = labels[identity_index]
        state = states[identity]
        assert state is not None
        gap = max(1, int(frame - state.frame))
        threshold = min(
            config.max_jump_mm * gap,
            config.max_jump_hard_cap_mm,
        )
        for source_index in valid:
            motion_cost, diagnostics = _motion_cost(
                state,
                points[source_index],
                frame,
                velocity_prediction_weight=config.velocity_prediction_weight,
                flow_weight=config.tangent_weight,
                acceleration_weight=config.acceleration_weight,
                motion_uncertainty_growth_per_frame=(
                    config.motion_uncertainty_growth_per_frame
                ),
                max_motion_gap_frames=config.max_motion_gap_frames,
            )
            diagnostics = dict(diagnostics)
            within_gate = float(diagnostics["distance_mm"]) <= threshold
            diagnostics["within_motion_gate"] = within_gate
            diagnostics["motion_gate_mm"] = threshold
            pair_diagnostics[(identity_index, source_index)] = diagnostics
            if within_gate:
                pair_cost[identity_index, source_index] = (
                    float(diagnostics["distance_mm"]) + motion_cost
                )

    active_branches: list[dict[int, int]] = []
    maximum_cardinality = min(len(active), len(valid))
    for cardinality in range(maximum_cardinality, -1, -1):
        for identity_subset in combinations(active, cardinality):
            for source_order in permutations(valid, cardinality):
                mapping = dict(zip(identity_subset, source_order, strict=True))
                if all(
                    pair_cost[identity_index, source_index] < _FORBIDDEN_COST
                    for identity_index, source_index in mapping.items()
                ):
                    active_branches.append(mapping)
        if active_branches:
            break

    if not active:
        active_branches = [{}]
    branches: list[_Branch] = []
    for active_mapping in active_branches:
        mapping = dict(active_mapping)
        unmatched_sources = [
            source_index
            for source_index in valid
            if source_index not in mapping.values()
        ]
        unseen = [
            identity_index
            for identity_index, identity in enumerate(labels)
            if states[identity] is None and identity_index not in mapping
        ]
        unmatched_sources.sort(key=lambda index: (points[index, 0], index))
        for identity_index, source_index in zip(
            unseen, unmatched_sources, strict=False
        ):
            mapping[identity_index] = source_index
        cost = float(
            sum(
                pair_cost[identity_index, source_index]
                for identity_index, source_index in active_mapping.items()
            )
        )
        full_assignment = bool(
            len(valid) == config.num_individuals
            and len(mapping) == config.num_individuals
            and len(active_mapping) == len(active)
        )
        hints = source_hints or {}
        conflict = any(
            identity not in labels
            or mapping.get(labels.index(identity)) != int(source_index)
            for source_index, identity in hints.items()
            if int(source_index) in valid
        )
        branches.append(
            _Branch(
                mapping=mapping,
                local_cost=cost,
                diagnostics=pair_diagnostics,
                full_assignment=full_assignment,
                constraint_conflict=conflict,
            )
        )

    if source_hints:
        satisfying = [branch for branch in branches if not branch.constraint_conflict]
        if satisfying:
            branches = satisfying
    branches.sort(
        key=lambda branch: (
            branch.local_cost,
            tuple(branch.mapping.get(index, -1) for index in range(config.num_individuals)),
        )
    )
    return branches


def _apply_branch(
    states: dict[str, _TrackState | None],
    branch: _Branch,
    frame: int,
    frame_info: Mapping[str, object],
    points: np.ndarray,
    *,
    config: DeferredTrackingConfig,
) -> tuple[dict[str, _TrackState | None], Dict]:
    labels = _id_labels(config.num_individuals)
    updated = deepcopy(states)
    groups = {identity: [np.nan, np.nan, np.nan] for identity in labels}
    mapping = [-1] * config.num_individuals
    diagnostics_by_id: dict[str, dict[str, object]] = {}
    recovered_source = list(frame_info.get("recovered", []))
    synthetic_source = list(frame_info.get("synthetic_views", []))
    recovered: list[bool] = []
    synthetic: list[int] = []
    for identity_index, identity in enumerate(labels):
        source_index = branch.mapping.get(identity_index, -1)
        mapping[identity_index] = int(source_index)
        if source_index < 0:
            recovered.append(False)
            synthetic.append(-1)
            diagnostics_by_id[identity] = {
                "source_index": -1,
                "matched": False,
            }
            continue
        point = points[source_index]
        groups[identity] = [float(value) for value in point]
        state = updated[identity]
        seeded = state is None
        if state is None:
            updated[identity] = _TrackState(
                frame,
                point.copy(),
                history_size=config.motion_history_frames,
            )
        else:
            state.update(frame, point)
        pair = dict(branch.diagnostics.get((identity_index, source_index), {}))
        pair.update(
            {
                "source_index": int(source_index),
                "matched": True,
                "seeded": seeded,
            }
        )
        diagnostics_by_id[identity] = pair
        recovered.append(
            bool(recovered_source[source_index])
            if 0 <= source_index < len(recovered_source)
            else False
        )
        synthetic.append(
            int(synthetic_source[source_index])
            if 0 <= source_index < len(synthetic_source)
            else -1
        )
    result = {
        "groups": groups,
        "perm_cam0": list(frame_info.get("perm_cam0", [])),
        "perm_cam1": list(frame_info.get("perm_cam1", [])),
        "perm_cam2": list(frame_info.get("perm_cam2", [])),
        "recovered": recovered,
        "synthetic_views": synthetic,
        "id_mapping": mapping,
        "id_anchor": [False] * config.num_individuals,
        "tracking_diagnostics": diagnostics_by_id,
        "assignment_local_cost": branch.local_cost,
        "assignment_full": branch.full_assignment,
        "tracklet_constraint_conflict": branch.constraint_conflict,
    }
    return updated, result


def _unresolved_frame(frame_result: Mapping[str, object]) -> Dict:
    output = deepcopy(dict(frame_result))
    output["unresolved_geometry"] = deepcopy(output["groups"])
    output["groups"] = {
        identity: [np.nan, np.nan, np.nan]
        for identity in output["groups"]
    }
    output["identity_status"] = "unresolved"
    output["id_mapping"] = [-1] * len(output["groups"])
    return output


def _insufficient_geometry_frame(
    frame_info: Mapping[str, object],
    points: np.ndarray,
    *,
    config: DeferredTrackingConfig,
) -> Dict:
    labels = _id_labels(config.num_individuals)
    source_groups = {
        identity: [float(value) for value in points[index]]
        for index, identity in enumerate(labels)
    }
    result = {
        "groups": source_groups,
        "perm_cam0": list(frame_info.get("perm_cam0", [])),
        "perm_cam1": list(frame_info.get("perm_cam1", [])),
        "perm_cam2": list(frame_info.get("perm_cam2", [])),
        "recovered": list(frame_info.get("recovered", [])),
        "synthetic_views": list(frame_info.get("synthetic_views", [])),
        "id_mapping": list(range(config.num_individuals)),
        "id_anchor": [False] * config.num_individuals,
        "tracking_diagnostics": {},
        "assignment_local_cost": np.nan,
        "assignment_full": False,
        "tracklet_constraint_conflict": False,
    }
    unresolved = _unresolved_frame(result)
    unresolved["identity_status"] = "unresolved_insufficient_geometry"
    return unresolved


def apply_deferred_identity_tracking(
    best_by_frame: Dict[int, Dict],
    *,
    config: DeferredTrackingConfig = DeferredTrackingConfig(),
    source_id_hints_by_frame: Mapping[int, Mapping[int, str]] | None = None,
    forced_reset_frames: Iterable[int] = (),
) -> DeferredTrackingResult:
    """Link framewise geometry with a bounded, inspectable hypothesis beam."""
    config.validate()
    frames = sorted(best_by_frame)
    reset_frame_set = {int(frame) for frame in forced_reset_frames}
    result_configuration = {
        **asdict(config),
        "forced_reset_frames": sorted(reset_frame_set),
    }
    if not frames:
        return DeferredTrackingResult({}, pd.DataFrame(), (), result_configuration)
    labels = _id_labels(config.num_individuals)
    states: dict[str, _TrackState | None] = {
        identity: None for identity in labels
    }
    tracked: Dict[int, Dict] = {}
    audits: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    active_hypotheses: list[_Hypothesis] | None = None
    episode_start: int | None = None
    stable_resolution_frames = 0
    identity_segment = 0
    reset_pending = False
    insufficient_start: int | None = None
    insufficient_end: int | None = None

    for frame in frames:
        if frame in reset_frame_set and frame != frames[0]:
            if active_hypotheses is not None:
                assert episode_start is not None
                winner = active_hypotheses[0]
                for pending_frame, pending_result in winner.pending:
                    unresolved_result = _unresolved_frame(pending_result)
                    unresolved_result["identity_segment"] = identity_segment
                    tracked[pending_frame] = unresolved_result
                unresolved.append(
                    {
                        "frame_start": int(episode_start),
                        "frame_end": int(frame - 1),
                        "frames": int(frame - episode_start),
                        "reason": "forced_tracklet_segment_reset",
                        "hypotheses_retained": len(active_hypotheses),
                        "best_second_margin": np.nan,
                    }
                )
                active_hypotheses = None
                episode_start = None
                stable_resolution_frames = 0
            states = {identity: None for identity in labels}
            identity_segment += 1
            reset_pending = True
        frame_info = best_by_frame[frame]
        points = _get_points_from_groups(
            frame_info.get("groups", {}), config.num_individuals
        )
        hints = (source_id_hints_by_frame or {}).get(frame, {})
        valid_count = int(
            np.all(np.isfinite(points), axis=1).sum()
        )
        if valid_count < config.num_individuals:
            if active_hypotheses is not None:
                assert episode_start is not None
                winner = active_hypotheses[0]
                for pending_frame, pending_result in winner.pending:
                    unresolved_result = _unresolved_frame(pending_result)
                    unresolved_result["identity_segment"] = identity_segment
                    tracked[pending_frame] = unresolved_result
                unresolved.append(
                    {
                        "frame_start": int(episode_start),
                        "frame_end": int(frame - 1),
                        "frames": int(frame - episode_start),
                        "reason": "insufficient_geometry_during_defer",
                        "hypotheses_retained": len(active_hypotheses),
                        "best_second_margin": np.nan,
                    }
                )
                # No hidden hypothesis may bridge an episode whose evidence
                # disappeared before resolution.
                states = {identity: None for identity in labels}
                identity_segment += 1
                reset_pending = True
                active_hypotheses = None
                episode_start = None
                stable_resolution_frames = 0
            retired = _retire_states(
                states, frame, max_gap_frames=config.max_gap_frames
            )
            if retired:
                identity_segment += 1
                reset_pending = True
            unresolved_result = _insufficient_geometry_frame(
                frame_info, points, config=config
            )
            unresolved_result["identity_segment"] = identity_segment
            tracked[frame] = unresolved_result
            if insufficient_start is None:
                insufficient_start = int(frame)
            insufficient_end = int(frame)
            audits.append(
                {
                    "frame": int(frame),
                    "episode_start": None,
                    "best_cost": np.nan,
                    "second_cost": np.nan,
                    "assignment_margin": np.nan,
                    "hypotheses_retained": 0,
                    "ambiguous": True,
                    "constraint_conflict": False,
                    "resolved": False,
                }
            )
            continue
        if insufficient_start is not None:
            assert insufficient_end is not None
            unresolved.append(
                {
                    "frame_start": insufficient_start,
                    "frame_end": insufficient_end,
                    "frames": int(insufficient_end - insufficient_start + 1),
                    "reason": "insufficient_geometry",
                    "hypotheses_retained": 0,
                    "best_second_margin": np.nan,
                }
            )
            insufficient_start = None
            insufficient_end = None
        if active_hypotheses is None:
            working_states = deepcopy(states)
            _retire_states(
                working_states, frame, max_gap_frames=config.max_gap_frames
            )
            branches = _enumerate_branches(
                working_states,
                points,
                frame,
                config=config,
                source_hints=hints,
            )
            if not branches:
                continue
            best_cost = branches[0].local_cost
            second_cost = (
                branches[1].local_cost if len(branches) > 1 else np.inf
            )
            margin = (
                float(second_cost - best_cost)
                if np.isfinite(second_cost)
                else np.inf
            )
            ambiguous = bool(
                margin < config.entry_margin
                or not branches[0].full_assignment
                or branches[0].constraint_conflict
            )
            audits.append(
                {
                    "frame": int(frame),
                    "episode_start": int(frame) if ambiguous else None,
                    "best_cost": best_cost,
                    "second_cost": second_cost,
                    "assignment_margin": margin,
                    "hypotheses_retained": min(len(branches), config.beam_width),
                    "ambiguous": ambiguous,
                    "constraint_conflict": branches[0].constraint_conflict,
                    "resolved": not ambiguous,
                }
            )
            if not ambiguous:
                states, frame_result = _apply_branch(
                    working_states,
                    branches[0],
                    frame,
                    frame_info,
                    points,
                    config=config,
                )
                frame_result["identity_status"] = (
                    "new_segment_after_unresolved"
                    if reset_pending
                    else "resolved_immediate"
                )
                frame_result["identity_segment"] = identity_segment
                tracked[frame] = frame_result
                reset_pending = False
                continue
            episode_start = int(frame)
            active_hypotheses = []
            for branch in branches[: config.beam_width]:
                branch_states, frame_result = _apply_branch(
                    working_states,
                    branch,
                    frame,
                    frame_info,
                    points,
                    config=config,
                )
                active_hypotheses.append(
                    _Hypothesis(
                        states=branch_states,
                        cumulative_cost=branch.local_cost,
                        pending=[(frame, frame_result)],
                        mapping_signature=(
                            tuple(
                                branch.mapping.get(index, -1)
                                for index in range(config.num_individuals)
                            ),
                        ),
                        constraint_conflict=branch.constraint_conflict,
                    )
                )
            stable_resolution_frames = 0
            reset_pending = False
            continue

        expanded: list[_Hypothesis] = []
        for hypothesis in active_hypotheses:
            working_states = deepcopy(hypothesis.states)
            _retire_states(
                working_states, frame, max_gap_frames=config.max_gap_frames
            )
            branches = _enumerate_branches(
                working_states,
                points,
                frame,
                config=config,
                source_hints=hints,
            )
            for branch in branches:
                branch_states, frame_result = _apply_branch(
                    working_states,
                    branch,
                    frame,
                    frame_info,
                    points,
                    config=config,
                )
                signature = tuple(
                    branch.mapping.get(index, -1)
                    for index in range(config.num_individuals)
                )
                expanded.append(
                    _Hypothesis(
                        states=branch_states,
                        cumulative_cost=(
                            hypothesis.cumulative_cost + branch.local_cost
                        ),
                        pending=[*hypothesis.pending, (frame, frame_result)],
                        mapping_signature=(
                            *hypothesis.mapping_signature,
                            signature,
                        ),
                        # Track the current-frame conflict only. A temporary
                        # A4 hint/motion disagreement defers identity, but must
                        # not permanently poison a recoverable episode.
                        constraint_conflict=branch.constraint_conflict,
                    )
                )
        expanded.sort(
            key=lambda hypothesis: (
                hypothesis.cumulative_cost,
                hypothesis.mapping_signature,
            )
        )
        if not expanded:
            raise RuntimeError(f"no deferred assignment hypothesis at frame {frame}")
        best_total = expanded[0].cumulative_cost
        active_hypotheses = [
            hypothesis
            for hypothesis in expanded
            if hypothesis.cumulative_cost
            <= best_total + config.retention_margin
        ][: config.beam_width]
        second_total = (
            active_hypotheses[1].cumulative_cost
            if len(active_hypotheses) > 1
            else np.inf
        )
        lead = (
            float(second_total - best_total)
            if np.isfinite(second_total)
            else np.inf
        )
        conflict = active_hypotheses[0].constraint_conflict
        if lead >= config.resolution_margin and not conflict:
            stable_resolution_frames += 1
        else:
            stable_resolution_frames = 0
        assert episode_start is not None
        episode_length = int(frame - episode_start + 1)
        can_resolve = bool(
            episode_length >= config.minimum_lookahead_frames
            and stable_resolution_frames
            >= config.resolution_stability_frames
        )
        timed_out = episode_length >= config.maximum_defer_horizon
        audits.append(
            {
                "frame": int(frame),
                "episode_start": int(episode_start),
                "best_cost": best_total,
                "second_cost": second_total,
                "assignment_margin": lead,
                "hypotheses_retained": len(active_hypotheses),
                "ambiguous": True,
                "constraint_conflict": conflict,
                "resolved": can_resolve,
            }
        )
        if can_resolve:
            winner = active_hypotheses[0]
            for pending_frame, pending_result in winner.pending:
                pending_result["identity_status"] = "resolved_after_lookahead"
                pending_result["identity_segment"] = identity_segment
                tracked[pending_frame] = pending_result
            states = winner.states
            active_hypotheses = None
            episode_start = None
            stable_resolution_frames = 0
        elif timed_out:
            winner = active_hypotheses[0]
            for pending_frame, pending_result in winner.pending:
                unresolved_result = _unresolved_frame(pending_result)
                unresolved_result["identity_segment"] = identity_segment
                tracked[pending_frame] = unresolved_result
            unresolved.append(
                {
                    "frame_start": int(episode_start),
                    "frame_end": int(frame),
                    "frames": episode_length,
                    "reason": (
                        "tracklet_motion_gate_conflict"
                        if conflict
                        else "ambiguity_horizon_exhausted"
                    ),
                    "hypotheses_retained": len(active_hypotheses),
                    "best_second_margin": lead,
                }
            )
            # The best hidden hypothesis is not evidence. Do not let it silently
            # determine identities after a timed-out unresolved interval.
            states = {identity: None for identity in labels}
            identity_segment += 1
            reset_pending = True
            active_hypotheses = None
            episode_start = None
            stable_resolution_frames = 0

    if active_hypotheses is not None:
        winner = active_hypotheses[0]
        assert episode_start is not None
        final_frame = frames[-1]
        for pending_frame, pending_result in winner.pending:
            unresolved_result = _unresolved_frame(pending_result)
            unresolved_result["identity_segment"] = identity_segment
            tracked[pending_frame] = unresolved_result
        unresolved.append(
            {
                "frame_start": int(episode_start),
                "frame_end": int(final_frame),
                "frames": int(final_frame - episode_start + 1),
                "reason": (
                    "tracklet_motion_gate_conflict"
                    if winner.constraint_conflict
                    else "recording_ended_before_resolution"
                ),
                "hypotheses_retained": len(active_hypotheses),
                "best_second_margin": (
                    float(
                        active_hypotheses[1].cumulative_cost
                        - active_hypotheses[0].cumulative_cost
                    )
                    if len(active_hypotheses) > 1
                    else np.inf
                ),
            }
        )
    if insufficient_start is not None:
        assert insufficient_end is not None
        unresolved.append(
            {
                "frame_start": insufficient_start,
                "frame_end": insufficient_end,
                "frames": int(insufficient_end - insufficient_start + 1),
                "reason": "insufficient_geometry",
                "hypotheses_retained": 0,
                "best_second_margin": np.nan,
            }
        )
    audit_columns = [
        "frame",
        "episode_start",
        "best_cost",
        "second_cost",
        "assignment_margin",
        "hypotheses_retained",
        "ambiguous",
        "constraint_conflict",
        "resolved",
    ]
    return DeferredTrackingResult(
        tracked_by_frame=dict(sorted(tracked.items())),
        assignment_audit=pd.DataFrame.from_records(audits, columns=audit_columns),
        unresolved_intervals=tuple(unresolved),
        configuration=result_configuration,
    )
