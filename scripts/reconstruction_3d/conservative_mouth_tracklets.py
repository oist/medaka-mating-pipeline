from __future__ import annotations

"""High-purity per-view tracklets from cached strict-mouth candidates.

The tracker is deliberately conservative. Candidate slots are detections, not
biological identities. Directional constant-velocity matches are proposed in
both time directions and an edge is accepted only when the two passes agree.
Close detections, globally ambiguous assignments, implausible predictions, and
short fragments remain explicitly unassigned.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.reconstruction_3d.candidate_cache import CandidateWindow


_FORBIDDEN = 1e9


@dataclass(frozen=True)
class ConservativeTrackletConfig:
    separation_body_lengths: float = 0.5
    assignment_ambiguity_margin_body_lengths: float = 0.1
    maximum_missing_gap_frames: int = 2
    uncertainty_growth_body_lengths_per_missing_frame: float = 0.1
    base_prediction_gate_body_lengths: float = 1.0
    maximum_displacement_body_lengths_per_frame: float = 1.0
    motion_history_frames: int = 5
    minimum_mouth_score: float = 0.3
    minimum_fragment_observations: int = 2

    def validate(self) -> None:
        non_negative = {
            "separation_body_lengths": self.separation_body_lengths,
            "assignment_ambiguity_margin_body_lengths": (
                self.assignment_ambiguity_margin_body_lengths
            ),
            "uncertainty_growth_body_lengths_per_missing_frame": (
                self.uncertainty_growth_body_lengths_per_missing_frame
            ),
            "base_prediction_gate_body_lengths": (
                self.base_prediction_gate_body_lengths
            ),
            "maximum_displacement_body_lengths_per_frame": (
                self.maximum_displacement_body_lengths_per_frame
            ),
            "minimum_mouth_score": self.minimum_mouth_score,
        }
        for name, value in non_negative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_missing_gap_frames < 0:
            raise ValueError("maximum_missing_gap_frames must be non-negative")
        if self.motion_history_frames < 2:
            raise ValueError("motion_history_frames must be at least 2")
        if self.minimum_fragment_observations < 1:
            raise ValueError("minimum_fragment_observations must be positive")


@dataclass(frozen=True)
class TrackletBuildResult:
    tracklet_ids: np.ndarray
    tracklets: pd.DataFrame
    dispositions: pd.DataFrame
    bridges: pd.DataFrame
    summary: Mapping[str, object]


@dataclass
class _DirectionalState:
    frames: list[int]
    points: list[np.ndarray]
    observation_keys: list[tuple[int, int]]

    @property
    def frame(self) -> int:
        return int(self.frames[-1])

    @property
    def point(self) -> np.ndarray:
        return self.points[-1]

    @property
    def key(self) -> tuple[int, int]:
        return self.observation_keys[-1]

    def robust_velocity(self, history_size: int) -> np.ndarray:
        frames = self.frames[-history_size:]
        points = self.points[-history_size:]
        slopes: list[np.ndarray] = []
        for left, right in combinations(range(len(frames)), 2):
            delta = frames[right] - frames[left]
            if delta:
                slopes.append((points[right] - points[left]) / delta)
        return (
            np.median(np.asarray(slopes, dtype=float), axis=0)
            if slopes
            else np.zeros(2, dtype=float)
        )

    def append(
        self,
        frame: int,
        point: np.ndarray,
        key: tuple[int, int],
        *,
        history_size: int,
    ) -> None:
        self.frames.append(int(frame))
        self.points.append(point.copy())
        self.observation_keys.append(key)
        keep = max(2, int(history_size))
        self.frames = self.frames[-keep:]
        self.points = self.points[-keep:]
        self.observation_keys = self.observation_keys[-keep:]


def _robust_positive_median(values: np.ndarray) -> float:
    lengths = np.asarray(values, dtype=float)
    finite = lengths[np.isfinite(lengths) & (lengths > 0)]
    if not finite.size:
        raise ValueError("no finite positive values are available")
    median = float(np.median(finite))
    deviations = np.abs(finite - median)
    mad = float(np.median(deviations))
    if mad > 0:
        kept = finite[deviations <= 4.0 * 1.4826 * mad]
        if kept.size:
            median = float(np.median(kept))
    return median


def robust_body_length_px(window: CandidateWindow) -> float:
    """Estimate the per-view mouth-tail scale for a training-control window."""
    lengths = np.linalg.norm(window.tail_xy - window.mouth_xy, axis=2)
    return _robust_positive_median(lengths)


def frozen_body_length_px_from_training_controls(
    windows: Iterable[CandidateWindow],
) -> float:
    """Estimate one reusable per-view scale from training controls only.

    Callers must persist and hash this value before validation, held-out, or
    critical evaluation. Tail is consumed only here for normalization; it is
    never used for candidate admission or identity matching.
    """
    length_parts = [
        np.linalg.norm(window.tail_xy - window.mouth_xy, axis=2).reshape(-1)
        for window in windows
    ]
    if not length_parts:
        raise ValueError("at least one training-control window is required")
    return _robust_positive_median(np.concatenate(length_parts))


def _valid_observations(
    window: CandidateWindow,
    config: ConservativeTrackletConfig,
    body_length_px: float,
) -> tuple[dict[int, list[tuple[int, np.ndarray, float]]], dict[tuple[int, int], str]]:
    observations: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    status: dict[tuple[int, int], str] = {}
    separation_gate = config.separation_body_lengths * body_length_px
    for local, frame in enumerate(range(window.frame_start, window.frame_end + 1)):
        finite_slots = np.flatnonzero(
            np.all(np.isfinite(window.mouth_xy[local]), axis=1)
        )
        scored_slots: list[int] = []
        for slot in finite_slots:
            value = float(window.mouth_score[local, slot])
            key = (int(frame), int(slot))
            if not np.isfinite(value):
                status[key] = "mouth_score_missing"
            elif value < config.minimum_mouth_score:
                status[key] = "below_mouth_score"
            else:
                scored_slots.append(int(slot))
        points = (
            window.mouth_xy[local, scored_slots]
            if scored_slots
            else np.empty((0, 2), dtype=float)
        )
        unambiguous = np.ones(len(scored_slots), dtype=bool)
        if len(points) > 1:
            distances = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=2
            )
            np.fill_diagonal(distances, np.inf)
            unambiguous = np.min(distances, axis=1) >= separation_gate
        rows: list[tuple[int, np.ndarray, float]] = []
        for local_index, slot in enumerate(scored_slots):
            key = (int(frame), int(slot))
            if not bool(unambiguous[local_index]):
                status[key] = "mouth_proximity_ambiguous"
                continue
            score = float(window.mouth_score[local, slot])
            rows.append((slot, points[local_index].copy(), score))
            status[key] = "eligible"
        observations[int(frame)] = rows
    return observations, status


def _assignment(
    cost: np.ndarray,
) -> tuple[list[tuple[int, int]], float, float]:
    if cost.size == 0:
        return [], 0.0, np.inf
    safe = np.where(np.isfinite(cost), cost, _FORBIDDEN)
    rows, columns = linear_sum_assignment(safe)
    pairs = [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if safe[row, column] < _FORBIDDEN
    ]
    if not pairs:
        return [], 0.0, np.inf
    best = float(sum(safe[row, column] for row, column in pairs))
    alternatives: list[float] = []
    for blocked_row, blocked_column in pairs:
        alternative = safe.copy()
        alternative[blocked_row, blocked_column] = _FORBIDDEN
        alt_rows, alt_columns = linear_sum_assignment(alternative)
        alt_values = [
            alternative[row, column]
            for row, column in zip(alt_rows, alt_columns, strict=True)
        ]
        if alt_values and all(value < _FORBIDDEN for value in alt_values):
            alternatives.append(float(sum(alt_values)))
    second = min(alternatives) if alternatives else np.inf
    return pairs, best, second


def _directional_proposals(
    observations: Mapping[int, list[tuple[int, np.ndarray, float]]],
    *,
    body_length_px: float,
    config: ConservativeTrackletConfig,
    reverse: bool,
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], list[dict[str, object]]]:
    frames = sorted(observations, reverse=reverse)
    active: dict[int, _DirectionalState] = {}
    next_state_id = 0
    proposed: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    audits: list[dict[str, object]] = []
    direction = "backward" if reverse else "forward"
    for frame in frames:
        rows = observations[frame]
        points = (
            np.asarray([row[1] for row in rows], dtype=float)
            if rows
            else np.empty((0, 2), dtype=float)
        )
        active_ids = [
            state_id
            for state_id, state in sorted(active.items())
            if abs(frame - state.frame) - 1 <= config.maximum_missing_gap_frames
        ]
        cost = np.full((len(active_ids), len(rows)), np.inf, dtype=float)
        prediction_errors = np.full_like(cost, np.nan)
        displacement = np.full_like(cost, np.nan)
        for active_index, state_id in enumerate(active_ids):
            state = active[state_id]
            delta_frames = int(frame - state.frame)
            missing_gap = abs(delta_frames) - 1
            velocity = state.robust_velocity(config.motion_history_frames)
            predicted = state.point + velocity * delta_frames
            gate = body_length_px * (
                config.base_prediction_gate_body_lengths
                + config.uncertainty_growth_body_lengths_per_missing_frame
                * max(0, missing_gap)
            )
            displacement_gate = (
                body_length_px
                * config.maximum_displacement_body_lengths_per_frame
                * max(1, abs(delta_frames))
            )
            for current_index, point in enumerate(points):
                error = float(np.linalg.norm(point - predicted))
                jump = float(np.linalg.norm(point - state.point))
                prediction_errors[active_index, current_index] = error
                displacement[active_index, current_index] = jump
                if error <= gate and jump <= displacement_gate:
                    cost[active_index, current_index] = error / body_length_px
        pairs, best, second = _assignment(cost)
        margin = float(second - best) if np.isfinite(second) else np.inf
        frame_ambiguous = bool(
            pairs
            and margin
            < config.assignment_ambiguity_margin_body_lengths
        )
        accepted_pairs: list[tuple[int, int]] = []
        if not frame_ambiguous:
            accepted_pairs = pairs
        matched_states: set[int] = set()
        matched_current: set[int] = set()
        for active_index, current_index in accepted_pairs:
            state_id = active_ids[active_index]
            state = active[state_id]
            slot, point, _score = rows[current_index]
            current_key = (int(frame), int(slot))
            older, newer = (
                (current_key, state.key) if reverse else (state.key, current_key)
            )
            proposed.add((older, newer))
            audits.append(
                {
                    "direction": direction,
                    "source_frame": int(older[0]),
                    "source_slot": int(older[1]),
                    "target_frame": int(newer[0]),
                    "target_slot": int(newer[1]),
                    "prediction_error_px": float(
                        prediction_errors[active_index, current_index]
                    ),
                    "prediction_error_body_lengths": float(
                        prediction_errors[active_index, current_index]
                        / body_length_px
                    ),
                    "displacement_px": float(
                        displacement[active_index, current_index]
                    ),
                    "assignment_best_cost": best,
                    "assignment_second_cost": second,
                    "assignment_margin_body_lengths": margin,
                    "accepted_directionally": True,
                }
            )
            state.append(
                frame,
                point,
                current_key,
                history_size=config.motion_history_frames,
            )
            matched_states.add(state_id)
            matched_current.add(current_index)
        active = {
            state_id: state
            for state_id, state in active.items()
            if state_id in matched_states
            or abs(frame - state.frame) - 1 < config.maximum_missing_gap_frames
        }
        for current_index, (slot, point, _score) in enumerate(rows):
            if current_index in matched_current:
                continue
            active[next_state_id] = _DirectionalState(
                frames=[int(frame)],
                points=[point.copy()],
                observation_keys=[(int(frame), int(slot))],
            )
            next_state_id += 1
        if frame_ambiguous:
            audits.append(
                {
                    "direction": direction,
                    "source_frame": None,
                    "source_slot": None,
                    "target_frame": int(frame),
                    "target_slot": None,
                    "prediction_error_px": None,
                    "prediction_error_body_lengths": None,
                    "displacement_px": None,
                    "assignment_best_cost": best,
                    "assignment_second_cost": second,
                    "assignment_margin_body_lengths": margin,
                    "accepted_directionally": False,
                }
            )
    return proposed, audits


def _components(
    nodes: Iterable[tuple[int, int]],
    edges: Iterable[tuple[tuple[int, int], tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    parent = {node: node for node in nodes}

    def find(node: tuple[int, int]) -> tuple[int, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        root, child = sorted((left_root, right_root))
        parent[child] = root

    for left, right in edges:
        union(left, right)
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)
    return [
        sorted(component)
        for component in sorted(grouped.values(), key=lambda values: values[0])
    ]


def build_reciprocal_tracklets(
    window: CandidateWindow,
    *,
    config: ConservativeTrackletConfig = ConservativeTrackletConfig(),
    frozen_body_length_px: float | None = None,
) -> TrackletBuildResult:
    config.validate()
    body_length = (
        robust_body_length_px(window)
        if frozen_body_length_px is None
        else float(frozen_body_length_px)
    )
    if not np.isfinite(body_length) or body_length <= 0:
        raise ValueError("frozen_body_length_px must be finite and positive")
    observations, status = _valid_observations(window, config, body_length)
    forward, forward_audit = _directional_proposals(
        observations,
        body_length_px=body_length,
        config=config,
        reverse=False,
    )
    backward, backward_audit = _directional_proposals(
        observations,
        body_length_px=body_length,
        config=config,
        reverse=True,
    )
    accepted_edges = sorted(forward & backward)
    eligible_nodes = sorted(
        (int(frame), int(slot))
        for frame, rows in observations.items()
        for slot, _point, _score in rows
    )
    components = _components(eligible_nodes, accepted_edges)
    retained = [
        component
        for component in components
        if len(component) >= config.minimum_fragment_observations
    ]
    tracklet_by_node = {
        node: tracklet_id
        for tracklet_id, component in enumerate(retained)
        for node in component
    }
    ids = np.full(window.mouth_xy.shape[:2], -1, dtype=np.int32)
    tracklet_records: list[dict[str, object]] = []
    disposition_records: list[dict[str, object]] = []
    for local, frame in enumerate(range(window.frame_start, window.frame_end + 1)):
        for slot in range(window.mouth_xy.shape[1]):
            point = window.mouth_xy[local, slot]
            if not np.all(np.isfinite(point)):
                continue
            key = (int(frame), int(slot))
            tracklet_id = tracklet_by_node.get(key)
            disposition = status.get(key, "not_eligible")
            if tracklet_id is not None:
                ids[local, slot] = int(tracklet_id)
                disposition = "retained_tracklet"
                tracklet_records.append(
                    {
                        "tracklet_id": int(tracklet_id),
                        "frame": int(frame),
                        "candidate_slot": int(slot),
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "mouth_score": float(window.mouth_score[local, slot]),
                    }
                )
            elif disposition == "eligible":
                disposition = "short_or_unlinked"
            disposition_records.append(
                {
                    "frame": int(frame),
                    "candidate_slot": int(slot),
                    "disposition": disposition,
                    "tracklet_id": (
                        int(tracklet_id) if tracklet_id is not None else -1
                    ),
                }
            )
    bridge_records = []
    forward_rows = {
        (
            int(row["source_frame"]),
            int(row["source_slot"]),
            int(row["target_frame"]),
            int(row["target_slot"]),
        ): row
        for row in forward_audit
        if row["accepted_directionally"]
    }
    backward_rows = {
        (
            int(row["source_frame"]),
            int(row["source_slot"]),
            int(row["target_frame"]),
            int(row["target_slot"]),
        ): row
        for row in backward_audit
        if row["accepted_directionally"]
    }
    all_edges = sorted(forward | backward)
    for left, right in all_edges:
        key = (left[0], left[1], right[0], right[1])
        forward_row = forward_rows.get(key)
        backward_row = backward_rows.get(key)
        bridge_records.append(
            {
                "source_frame": int(left[0]),
                "source_slot": int(left[1]),
                "target_frame": int(right[0]),
                "target_slot": int(right[1]),
                "gap_frames": int(right[0] - left[0] - 1),
                "forward_proposed": key in forward_rows,
                "backward_proposed": key in backward_rows,
                "reciprocal_accepted": bool(
                    key in forward_rows and key in backward_rows
                ),
                "forward_prediction_error_body_lengths": (
                    forward_row["prediction_error_body_lengths"]
                    if forward_row is not None
                    else np.nan
                ),
                "backward_prediction_error_body_lengths": (
                    backward_row["prediction_error_body_lengths"]
                    if backward_row is not None
                    else np.nan
                ),
                "forward_assignment_margin_body_lengths": (
                    forward_row["assignment_margin_body_lengths"]
                    if forward_row is not None
                    else np.nan
                ),
                "backward_assignment_margin_body_lengths": (
                    backward_row["assignment_margin_body_lengths"]
                    if backward_row is not None
                    else np.nan
                ),
            }
        )
    tracklets = pd.DataFrame.from_records(
        tracklet_records,
        columns=[
            "tracklet_id",
            "frame",
            "candidate_slot",
            "x",
            "y",
            "mouth_score",
        ],
    )
    dispositions = pd.DataFrame.from_records(
        disposition_records,
        columns=[
            "frame",
            "candidate_slot",
            "disposition",
            "tracklet_id",
        ],
    )
    bridges = pd.DataFrame.from_records(
        bridge_records,
        columns=[
            "source_frame",
            "source_slot",
            "target_frame",
            "target_slot",
            "gap_frames",
            "forward_proposed",
            "backward_proposed",
            "reciprocal_accepted",
            "forward_prediction_error_body_lengths",
            "backward_prediction_error_body_lengths",
            "forward_assignment_margin_body_lengths",
            "backward_assignment_margin_body_lengths",
        ],
    )
    lengths = (
        tracklets.groupby("tracklet_id").size().to_numpy(dtype=int)
        if len(tracklets)
        else np.empty(0, dtype=int)
    )
    finite_candidates = int(len(dispositions))
    retained_candidates = int(
        (dispositions.disposition == "retained_tracklet").sum()
        if len(dispositions)
        else 0
    )
    summary = {
        "schema_version": 1,
        "anchor": "strict_finite_mouth",
        "candidate_slots_are_biological_identities": False,
        "body_length_px": body_length,
        "body_length_source": (
            "training_control_frozen"
            if frozen_body_length_px is not None
            else "current_window_unfrozen"
        ),
        "config": asdict(config),
        "finite_candidate_rows": finite_candidates,
        "retained_candidate_rows": retained_candidates,
        "retained_coverage": (
            retained_candidates / finite_candidates if finite_candidates else 0.0
        ),
        "fragment_count": int(len(lengths)),
        "fragment_length_median": (
            float(np.median(lengths)) if len(lengths) else 0.0
        ),
        "fragment_length_p95": (
            float(np.quantile(lengths, 0.95)) if len(lengths) else 0.0
        ),
        "reciprocal_accepted_bridges": int(len(accepted_edges)),
        "nonreciprocal_bridge_proposals": int(len(forward ^ backward)),
        "false_merge_proxy": int(
            (
                bridges.reciprocal_accepted
                & (
                    (
                        bridges.forward_prediction_error_body_lengths
                        > config.base_prediction_gate_body_lengths
                    )
                    | (
                        bridges.backward_prediction_error_body_lengths
                        > config.base_prediction_gate_body_lengths
                    )
                )
            ).sum()
            if len(bridges)
            else 0
        ),
        "false_break_proxy": int(len(forward ^ backward)),
        "unresolved_fraction": (
            float(
                (
                    dispositions.disposition.isin(
                        ["mouth_proximity_ambiguous", "short_or_unlinked"]
                    )
                ).mean()
            )
            if len(dispositions)
            else 0.0
        ),
    }
    return TrackletBuildResult(
        tracklet_ids=ids,
        tracklets=tracklets,
        dispositions=dispositions,
        bridges=bridges,
        summary=summary,
    )


def clean_motion_distributions(
    window: CandidateWindow,
    *,
    minimum_separation_body_lengths: float = 1.5,
    minimum_mouth_score: float = 0.3,
    frozen_body_length_px: float | None = None,
) -> dict[str, object]:
    """Estimate clean-frame displacement and prediction-error distributions."""
    body_length = (
        robust_body_length_px(window)
        if frozen_body_length_px is None
        else float(frozen_body_length_px)
    )
    if not np.isfinite(body_length) or body_length <= 0:
        raise ValueError("frozen_body_length_px must be finite and positive")
    config = ConservativeTrackletConfig(
        separation_body_lengths=minimum_separation_body_lengths,
        assignment_ambiguity_margin_body_lengths=0.0,
        maximum_missing_gap_frames=0,
        minimum_mouth_score=minimum_mouth_score,
        minimum_fragment_observations=1,
    )
    observations, _status = _valid_observations(window, config, body_length)
    displacements: list[float] = []
    prediction_errors: list[float] = []
    history: dict[int, _DirectionalState] = {}
    next_id = 0
    for frame in sorted(observations):
        rows = observations[frame]
        if len(rows) != 3:
            history = {}
            continue
        points = np.asarray([row[1] for row in rows], dtype=float)
        if not history:
            for index, (slot, point, _score) in enumerate(rows):
                history[next_id + index] = _DirectionalState(
                    [frame], [point.copy()], [(frame, slot)]
                )
            next_id += len(rows)
            continue
        state_ids = sorted(history)
        if any(frame - history[state_id].frame != 1 for state_id in state_ids):
            history = {}
            continue
        predicted = np.asarray(
            [
                history[state_id].point
                + history[state_id].robust_velocity(5)
                for state_id in state_ids
            ]
        )
        cost = np.linalg.norm(predicted[:, None] - points[None, :], axis=2)
        state_rows, point_columns = linear_sum_assignment(cost)
        for state_row, point_column in zip(
            state_rows, point_columns, strict=True
        ):
            state_id = state_ids[int(state_row)]
            state = history[state_id]
            point = points[int(point_column)]
            displacements.append(
                float(np.linalg.norm(point - state.point) / body_length)
            )
            if len(state.frames) >= 2:
                prediction_errors.append(
                    float(cost[state_row, point_column] / body_length)
                )
            slot = rows[int(point_column)][0]
            state.append(
                frame,
                point,
                (frame, slot),
                history_size=5,
            )

    def distribution(values: list[float]) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        return {
            "n": int(len(array)),
            "median_body_lengths": (
                float(np.median(array)) if len(array) else None
            ),
            "p95_body_lengths": (
                float(np.quantile(array, 0.95)) if len(array) else None
            ),
            "p99_body_lengths": (
                float(np.quantile(array, 0.99)) if len(array) else None
            ),
            "max_body_lengths": float(np.max(array)) if len(array) else None,
        }

    return {
        "schema_version": 1,
        "body_length_px": body_length,
        "body_length_source": (
            "training_control_frozen"
            if frozen_body_length_px is not None
            else "current_window_unfrozen"
        ),
        "clean_separation_body_lengths": minimum_separation_body_lengths,
        "displacement": distribution(displacements),
        "constant_velocity_prediction_error": distribution(prediction_errors),
    }
