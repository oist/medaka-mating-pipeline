"""Global motion-constrained optimization of F/M1/M2 over three 3D slots.

The geometry reconstruction supplies three continuous anonymous trajectories.
At each frame there are only six possible one-to-one mappings from those slots
to the biological identities ``F``, ``M1``, and ``M2``.  This module uses a
compact Viterbi dynamic program over those six states instead of committing to
appearance classifications independently at each sampled frame.

Transition costs use the complete previous-frame state before any identity is
updated.  They include displacement, a constant-velocity prediction, tangent
and acceleration changes, a permutation-switch penalty, and an optional hard
per-frame jump gate.  Appearance-derived labels remain soft evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any, Mapping

import numpy as np
import pandas as pd


FISH_IDS = ("F", "M1", "M2")
IDENTITY_INDEX = {name: index for index, name in enumerate(FISH_IDS)}
STATE_LABELS_BY_SLOT = np.asarray(list(permutations(FISH_IDS)), dtype=object)

# For each state and biological identity, which geometry slot carries it.
STATE_SLOT_BY_IDENTITY = np.empty((len(STATE_LABELS_BY_SLOT), 3), dtype=np.int8)
for state_index, labels_by_slot in enumerate(STATE_LABELS_BY_SLOT):
    for slot, label in enumerate(labels_by_slot):
        STATE_SLOT_BY_IDENTITY[state_index, IDENTITY_INDEX[str(label)]] = slot


DEFAULT_CONFIDENCE_WEIGHTS: Mapping[str, float] = {
    "high": 1.0,
    "interp": 0.25,
    "ambiguous": 0.02,
    "low": 0.10,
    "missing": 0.0,
}
DEFAULT_SOURCE_WEIGHTS: Mapping[str, float] = {
    "high": 1.0,
    "bridged": 0.70,
    "motion_localized": 0.25,
    "motion_debounced": 0.25,
    "edge": 0.10,
    "ambiguous": 0.0,
    "missing": 1.0,
}


@dataclass(frozen=True)
class MotionIdentityConfig:
    """Weights and gates for :func:`optimize_identity_track`.

    All costs are additive and non-negative.  ``hard_jump_mm=None`` disables
    the hard physical gate.  Across a frame gap, the gate scales by the gap up
    to ``max_gap_frames``; longer or position-missing transitions are not hard
    gated and rely on soft evidence.
    """

    position_weight: float = 1.0
    velocity_weight: float = 0.75
    tangent_weight: float = 1.5
    acceleration_weight: float = 0.25
    appearance_weight: float = 2.0
    switch_penalty: float = 1.0
    jump_penalty: float = 2.0
    soft_jump_mm: float = 2.5
    hard_jump_mm: float | None = 8.0
    max_gap_frames: int = 10
    confidence_weights: Mapping[str, float] | None = None
    source_weights: Mapping[str, float] | None = None
    low_margin_threshold: float = 1.0
    anchor_first_complete: bool = True
    anchor_min_evidence: float = 0.5
    stable_anchor_min_frames: int = 500
    stable_anchor_interval_frames: int = 5_000

    def __post_init__(self) -> None:
        non_negative = (
            "position_weight",
            "velocity_weight",
            "tangent_weight",
            "acceleration_weight",
            "appearance_weight",
            "switch_penalty",
            "jump_penalty",
            "soft_jump_mm",
            "low_margin_threshold",
            "anchor_min_evidence",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.hard_jump_mm is not None:
            if not np.isfinite(self.hard_jump_mm) or self.hard_jump_mm <= 0:
                raise ValueError("hard_jump_mm must be positive or None")
        if self.max_gap_frames < 1:
            raise ValueError("max_gap_frames must be at least 1")
        if self.stable_anchor_min_frames < 1:
            raise ValueError("stable_anchor_min_frames must be at least 1")
        if self.stable_anchor_interval_frames < 1:
            raise ValueError("stable_anchor_interval_frames must be at least 1")


@dataclass
class MotionIdentityResult:
    """Optimized row-aligned labels, output track, and compact diagnostics."""

    labels: np.ndarray
    optimized_track: pd.DataFrame
    state_index_by_frame: np.ndarray
    frame_numbers: np.ndarray
    path_margin_by_frame: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TransitionRelocationConfig:
    """Conservative repair of mislocalized appearance-state boundaries."""

    jump_threshold_mm: float = 8.0
    search_radius_frames: int = 350
    require_below_threshold: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.jump_threshold_mm) or self.jump_threshold_mm <= 0:
            raise ValueError("jump_threshold_mm must be positive")
        if self.search_radius_frames < 1:
            raise ValueError("search_radius_frames must be at least 1")


@dataclass(frozen=True)
class LayeredIdentityConfig:
    """Conservative two-layer repair used for production QC.

    The first layer removes a short permutation run only when it is bracketed
    by the same state and its entry causes an impossible 3D jump.  The second
    layer relocates the remaining implausible boundaries without changing
    their order.  This deliberately cannot invent a new long-lived identity
    regime; use :class:`MotionIdentityConfig` only for experimental global
    re-optimization.
    """

    jump_threshold_mm: float = 8.0
    max_flicker_frames: int = 166
    search_radius_frames: int = 350
    local_window_radius_frames: int = 166
    require_below_threshold: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.jump_threshold_mm) or self.jump_threshold_mm <= 0:
            raise ValueError("jump_threshold_mm must be positive")
        if self.max_flicker_frames < 1:
            raise ValueError("max_flicker_frames must be at least 1")
        if self.search_radius_frames < 1:
            raise ValueError("search_radius_frames must be at least 1")
        if self.local_window_radius_frames < 1:
            raise ValueError("local_window_radius_frames must be at least 1")


@dataclass(frozen=True)
class LocalWindowIdentityConfig:
    """Bounded minimum-edit repair around remaining discontinuities."""

    jump_threshold_mm: float = 8.0
    radius_frames: int = 166

    def __post_init__(self) -> None:
        if not np.isfinite(self.jump_threshold_mm) or self.jump_threshold_mm <= 0:
            raise ValueError("jump_threshold_mm must be positive")
        if self.radius_frames < 1:
            raise ValueError("radius_frames must be at least 1")


def _validate_track(track: pd.DataFrame) -> None:
    required = {"frame", "geometry_slot", "X", "Y", "Z", "id_final"}
    missing = required - set(track.columns)
    if missing:
        raise ValueError("track is missing columns: " + ", ".join(sorted(missing)))
    if track.empty:
        raise ValueError("track is empty")
    if track.duplicated(["frame", "geometry_slot"]).any():
        raise ValueError("track has duplicate frame/geometry_slot rows")
    slots = pd.to_numeric(track["geometry_slot"], errors="coerce")
    if slots.isna().any() or not slots.isin((0, 1, 2)).all():
        raise ValueError("geometry_slot must contain only 0, 1, and 2")
    frame = pd.to_numeric(track["frame"], errors="coerce")
    if frame.isna().any() or not np.allclose(frame, np.round(frame)):
        raise ValueError("frame must contain finite integers")


def _weight_for(value: object, weights: Mapping[str, float], missing: float) -> float:
    if pd.isna(value):
        return missing
    return float(weights.get(str(value), missing))


def _prepare_arrays(
    track: pd.DataFrame,
    config: MotionIdentityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frames, xyz, observed labels, evidence weights, and row lookup."""
    frames = np.sort(track["frame"].astype(np.int64).unique())
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    n_frames = len(frames)
    xyz = np.full((n_frames, 3, 3), np.nan, dtype=np.float64)
    observed = np.full((n_frames, 3), -1, dtype=np.int8)
    evidence = np.zeros((n_frames, 3), dtype=np.float64)
    row_lookup = np.empty(len(track), dtype=np.int64)

    confidence_weights = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if config.confidence_weights is not None:
        confidence_weights.update(config.confidence_weights)
    source_weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if config.source_weights is not None:
        source_weights.update(config.source_weights)

    has_conf = "id_conf" in track
    has_source = "id_source" in track
    for row_position, row in enumerate(track.itertuples(index=False)):
        frame = int(getattr(row, "frame"))
        slot = int(getattr(row, "geometry_slot"))
        frame_index = frame_to_index[frame]
        row_lookup[row_position] = frame_index * 3 + slot
        xyz[frame_index, slot] = (
            float(getattr(row, "X")),
            float(getattr(row, "Y")),
            float(getattr(row, "Z")),
        )
        label = str(getattr(row, "id_final"))
        observed[frame_index, slot] = IDENTITY_INDEX.get(label, -1)
        conf_weight = (
            _weight_for(getattr(row, "id_conf"), confidence_weights, 0.10)
            if has_conf
            else 1.0
        )
        source_weight = (
            _weight_for(getattr(row, "id_source"), source_weights, 1.0)
            if has_source
            else 1.0
        )
        # A biological label attached to a missing/interpolated-away geometry
        # position cannot safely cause a permutation transition. Keep it in the
        # output table, but give it no appearance-emission authority here.
        position_is_finite = np.isfinite(xyz[frame_index, slot]).all()
        evidence[frame_index, slot] = (
            min(conf_weight, source_weight) if position_is_finite else 0.0
        )

    return frames, xyz, observed, evidence, row_lookup


def _emission_cost(
    observed: np.ndarray,
    evidence: np.ndarray,
    config: MotionIdentityConfig,
) -> np.ndarray:
    state_codes = np.vectorize(IDENTITY_INDEX.__getitem__)(STATE_LABELS_BY_SLOT).astype(np.int8)
    valid = observed[:, :, None] >= 0
    mismatch = observed[:, :, None] != state_codes.T[None, :, :]
    return config.appearance_weight * np.sum(
        np.where(valid, mismatch * evidence[:, :, None], 0.0), axis=1
    )


def _transition_cost(
    previous_xyz: np.ndarray,
    current_xyz: np.ndarray,
    previous_slot_velocity: np.ndarray,
    *,
    frame_gap: int,
    velocity_available: np.ndarray,
    config: MotionIdentityConfig,
) -> tuple[np.ndarray, int]:
    """Return the 6x6 previous-state/current-state transition matrix."""
    slots = STATE_SLOT_BY_IDENTITY
    prev_by_state = previous_xyz[slots]  # previous state, identity, xyz
    curr_by_state = current_xyz[slots]   # current state, identity, xyz
    delta = curr_by_state[None, :, :, :] - prev_by_state[:, None, :, :]
    finite = np.isfinite(delta).all(axis=-1)
    distance = np.linalg.norm(np.where(finite[..., None], delta, 0.0), axis=-1)

    cost = config.position_weight * np.sum(np.where(finite, distance, 0.0), axis=-1)
    if config.jump_penalty > 0:
        excess = np.maximum(0.0, distance - config.soft_jump_mm)
        cost += config.jump_penalty * np.sum(np.where(finite, excess, 0.0), axis=-1)

    state_changed = np.arange(6)[:, None] != np.arange(6)[None, :]
    cost += config.switch_penalty * state_changed
    identity_changed_slot = slots[:, None, :] != slots[None, :, :]
    unevaluable_switch = np.any(identity_changed_slot & ~finite, axis=-1)
    # Never let a missing coordinate provide a free route between biological
    # permutations. Hold the previous state until every moved identity has a
    # physically evaluable transition.
    cost[state_changed & unevaluable_switch] = np.inf

    hard_rejections = 0
    if config.hard_jump_mm is not None and frame_gap <= config.max_gap_frames:
        threshold = config.hard_jump_mm * max(1, frame_gap)
        rejected = np.any(finite & (distance > threshold), axis=-1)
        hard_rejections = int(rejected.sum())
        cost[rejected] = np.inf

    motion_enabled = (
        config.velocity_weight > 0
        or config.tangent_weight > 0
        or config.acceleration_weight > 0
    )
    if motion_enabled:
        velocity_by_prev_state = previous_slot_velocity[slots]
        velocity_valid_by_prev_state = velocity_available[slots]
        predicted = (
            prev_by_state
            + velocity_by_prev_state * float(frame_gap)
        )
        prediction_delta = curr_by_state[None, :, :, :] - predicted[:, None, :, :]
        prediction_valid = (
            finite
            & velocity_valid_by_prev_state[:, None, :]
            & np.isfinite(prediction_delta).all(axis=-1)
        )
        prediction_error = np.linalg.norm(
            np.where(prediction_valid[..., None], prediction_delta, 0.0), axis=-1
        )
        cost += config.velocity_weight * np.sum(
            np.where(prediction_valid, prediction_error, 0.0), axis=-1
        )

        candidate_velocity = delta / float(max(1, frame_gap))
        previous_velocity = velocity_by_prev_state[:, None, :, :]
        previous_speed = np.linalg.norm(previous_velocity, axis=-1)
        candidate_speed = np.linalg.norm(candidate_velocity, axis=-1)
        tangent_valid = (
            prediction_valid
            & (previous_speed > 1e-9)
            & (candidate_speed > 1e-9)
        )
        dot = np.sum(previous_velocity * candidate_velocity, axis=-1)
        denominator = np.maximum(previous_speed * candidate_speed, 1e-9)
        tangent = 1.0 - np.clip(dot / denominator, -1.0, 1.0)
        cost += config.tangent_weight * np.sum(
            np.where(tangent_valid, tangent, 0.0), axis=-1
        )

        acceleration = candidate_velocity - previous_velocity
        acceleration_norm = np.linalg.norm(acceleration, axis=-1)
        cost += config.acceleration_weight * np.sum(
            np.where(prediction_valid, acceleration_norm, 0.0), axis=-1
        )

    return cost, hard_rejections


def _path_diagnostics(
    frames: np.ndarray,
    chosen_states: np.ndarray,
    margins: np.ndarray,
    observed: np.ndarray,
    config: MotionIdentityConfig,
    *,
    hard_rejections: int,
    normalized_path_cost: float,
) -> dict[str, Any]:
    optimized_codes = np.vectorize(IDENTITY_INDEX.__getitem__)(
        STATE_LABELS_BY_SLOT[chosen_states]
    )
    valid_observed = observed >= 0
    changed = valid_observed & (optimized_codes != observed)
    change_frames = frames[np.any(changed, axis=1)]
    permutation_change = np.r_[False, np.diff(chosen_states) != 0]
    finite_margins = margins[np.isfinite(margins)]
    low_margin = frames[np.isfinite(margins) & (margins < config.low_margin_threshold)]
    return {
        "algorithm": "six_state_motion_viterbi_v1",
        "config": asdict(config),
        "n_frames": int(len(frames)),
        "changed_rows": int(changed.sum()),
        "changed_frames": int(len(change_frames)),
        "changed_frame_examples": [int(frame) for frame in change_frames[:50]],
        "optimized_permutation_changes": int(permutation_change.sum()),
        "optimized_permutation_change_frames": [
            int(frame) for frame in frames[permutation_change][:100]
        ],
        "hard_transition_rejections_considered": int(hard_rejections),
        "normalized_path_cost": float(normalized_path_cost),
        "path_margin": {
            "median": float(np.median(finite_margins)) if finite_margins.size else None,
            "p05": float(np.quantile(finite_margins, 0.05)) if finite_margins.size else None,
            "minimum": float(np.min(finite_margins)) if finite_margins.size else None,
            "low_margin_frames": int(len(low_margin)),
            "low_margin_frame_examples": [int(frame) for frame in low_margin[:50]],
        },
    }


def optimize_identity_track(
    track: pd.DataFrame,
    *,
    config: MotionIdentityConfig | None = None,
) -> MotionIdentityResult:
    """Optimize biological labels over the complete frame sequence.

    The returned labels follow the input row order.  The output DataFrame is a
    copy; the caller's DataFrame is never modified.
    """
    config = config or MotionIdentityConfig()
    _validate_track(track)
    frames, xyz, observed, evidence, row_lookup = _prepare_arrays(track, config)
    emission = _emission_cost(observed, evidence, config)
    n_frames = len(frames)

    anchor_frame_indices: list[int] = []
    anchor_state_indices: list[int] = []
    if config.anchor_first_complete:
        state_codes = np.vectorize(IDENTITY_INDEX.__getitem__)(
            STATE_LABELS_BY_SLOT
        ).astype(np.int8)
        complete = (
            np.isfinite(xyz).all(axis=(1, 2))
            & (observed >= 0).all(axis=1)
            & (evidence >= config.anchor_min_evidence).all(axis=1)
        )
        for candidate_index in np.where(complete)[0]:
            matches = np.where(
                np.all(state_codes == observed[candidate_index], axis=1)
            )[0]
            if len(matches) == 1:
                anchor_frame_indices.append(int(candidate_index))
                anchor_state_indices.append(int(matches[0]))
                break

        # Long, internally stable appearance runs are trusted identity anchors.
        # Short runs remain deliberately unanchored: those are the flickers we
        # want motion continuity to remove or relocate.
        observed_state = np.full(n_frames, -1, dtype=np.int8)
        for state_index, state_code in enumerate(state_codes):
            matches = np.all(observed == state_code, axis=1)
            observed_state[matches] = state_index
        boundaries = np.r_[
            0,
            np.where(np.diff(observed_state) != 0)[0] + 1,
            n_frames,
        ]
        trusted = complete & (observed_state >= 0)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start < config.stable_anchor_min_frames:
                continue
            run_candidates = np.where(trusted[start:end])[0] + start
            if not len(run_candidates):
                continue
            targets = np.arange(
                start + (end - start) // 2,
                end,
                config.stable_anchor_interval_frames,
                dtype=int,
            )
            if not len(targets):
                targets = np.asarray([start + (end - start) // 2])
            for target in targets:
                candidate_index = int(
                    run_candidates[np.argmin(np.abs(run_candidates - target))]
                )
                state_index = int(observed_state[candidate_index])
                if candidate_index not in anchor_frame_indices:
                    anchor_frame_indices.append(candidate_index)
                    anchor_state_indices.append(state_index)

        for anchor_frame_index, anchor_state_index in zip(
            anchor_frame_indices, anchor_state_indices
        ):
            original_cost = float(emission[anchor_frame_index, anchor_state_index])
            emission[anchor_frame_index] = np.inf
            emission[anchor_frame_index, anchor_state_index] = original_cost

    backpointer = np.zeros((n_frames, 6), dtype=np.int8)
    incoming_margin = np.full((n_frames, 6), np.nan, dtype=np.float32)
    dp_previous = emission[0].astype(np.float64, copy=True)
    normalized_path_cost = float(np.min(dp_previous))
    dp_previous -= np.min(dp_previous)
    hard_rejections = 0

    for frame_index in range(1, n_frames):
        frame_gap = max(1, int(frames[frame_index] - frames[frame_index - 1]))
        previous_slot_velocity = np.full((3, 3), np.nan, dtype=np.float64)
        velocity_available = np.zeros(3, dtype=bool)
        if frame_index >= 2:
            previous_gap = int(frames[frame_index - 1] - frames[frame_index - 2])
            if 1 <= previous_gap <= config.max_gap_frames:
                previous_delta = xyz[frame_index - 1] - xyz[frame_index - 2]
                valid_velocity = np.isfinite(previous_delta).all(axis=1)
                previous_slot_velocity[valid_velocity] = (
                    previous_delta[valid_velocity] / float(previous_gap)
                )
                velocity_available = valid_velocity

        transition, rejected = _transition_cost(
            xyz[frame_index - 1],
            xyz[frame_index],
            previous_slot_velocity,
            frame_gap=frame_gap,
            velocity_available=velocity_available,
            config=config,
        )
        hard_rejections += rejected
        candidate_cost = dp_previous[:, None] + transition
        best_previous = np.argmin(candidate_cost, axis=0)
        best_cost = candidate_cost[best_previous, np.arange(6)]
        sorted_cost = np.sort(candidate_cost, axis=0)
        margin = np.full(6, np.inf, dtype=np.float64)
        both_finite = np.isfinite(sorted_cost[0]) & np.isfinite(sorted_cost[1])
        margin[both_finite] = sorted_cost[1, both_finite] - sorted_cost[0, both_finite]
        incoming_margin[frame_index] = margin.astype(np.float32)
        dp_current = best_cost + emission[frame_index]

        if not np.isfinite(dp_current).any():
            # This should only occur if every geometry slot itself violates the
            # hard gate. Preserve the sequence as auditable soft evidence
            # instead of crashing mid-experiment.
            soft_config = MotionIdentityConfig(
                **{**asdict(config), "hard_jump_mm": None}
            )
            transition, _ = _transition_cost(
                xyz[frame_index - 1],
                xyz[frame_index],
                previous_slot_velocity,
                frame_gap=frame_gap,
                velocity_available=velocity_available,
                config=soft_config,
            )
            candidate_cost = dp_previous[:, None] + transition
            best_previous = np.argmin(candidate_cost, axis=0)
            best_cost = candidate_cost[best_previous, np.arange(6)]
            dp_current = best_cost + emission[frame_index]

        backpointer[frame_index] = best_previous.astype(np.int8)
        frame_minimum = float(np.min(dp_current))
        normalized_path_cost += frame_minimum
        dp_previous = dp_current - frame_minimum

    chosen_states = np.empty(n_frames, dtype=np.int8)
    chosen_states[-1] = int(np.argmin(dp_previous))
    for frame_index in range(n_frames - 1, 0, -1):
        chosen_states[frame_index - 1] = backpointer[
            frame_index, chosen_states[frame_index]
        ]
    chosen_margins = incoming_margin[np.arange(n_frames), chosen_states]

    labels_grid = STATE_LABELS_BY_SLOT[chosen_states]
    flat_labels = labels_grid.reshape(-1)
    labels = flat_labels[row_lookup]
    optimized = track.copy()
    optimized["id_final"] = labels
    diagnostics = _path_diagnostics(
        frames,
        chosen_states,
        chosen_margins,
        observed,
        config,
        hard_rejections=hard_rejections,
        normalized_path_cost=normalized_path_cost,
    )
    ordered_anchors = sorted(
        zip(anchor_frame_indices, anchor_state_indices), key=lambda item: item[0]
    )
    diagnostics["anchors"] = {
        "count": int(len(ordered_anchors)),
        "first": (
            {
                "frame": int(frames[ordered_anchors[0][0]]),
                "state_index": int(ordered_anchors[0][1]),
                "labels_by_slot": [
                    str(value) for value in STATE_LABELS_BY_SLOT[ordered_anchors[0][1]]
                ],
            }
            if ordered_anchors
            else None
        ),
        "frame_examples": [
            int(frames[frame_index]) for frame_index, _ in ordered_anchors[:100]
        ],
    }
    return MotionIdentityResult(
        labels=np.asarray(labels, dtype=object),
        optimized_track=optimized,
        state_index_by_frame=chosen_states,
        frame_numbers=frames,
        path_margin_by_frame=chosen_margins,
        diagnostics=diagnostics,
    )


def _states_from_observed(observed: np.ndarray) -> np.ndarray:
    state_codes = np.vectorize(IDENTITY_INDEX.__getitem__)(
        STATE_LABELS_BY_SLOT
    ).astype(np.int8)
    states = np.full(len(observed), -1, dtype=np.int8)
    for state_index, state_code in enumerate(state_codes):
        states[np.all(observed == state_code, axis=1)] = state_index
    if np.any(states < 0):
        raise ValueError(
            "every frame must contain exactly one F, one M1, and one M2 "
            "for conservative transition relocation"
        )
    return states


def _boundary_motion(
    xyz: np.ndarray,
    frame_index: int,
    before_state: int,
    after_state: int,
) -> tuple[float, float]:
    if frame_index <= 0 or frame_index >= len(xyz):
        return np.inf, np.inf
    before = xyz[frame_index - 1, STATE_SLOT_BY_IDENTITY[before_state]]
    after = xyz[frame_index, STATE_SLOT_BY_IDENTITY[after_state]]
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return np.inf, np.inf
    step = np.linalg.norm(after - before, axis=1)
    return float(np.max(step)), float(np.sum(step))


def _jump_summary(
    xyz: np.ndarray,
    states: np.ndarray,
    threshold_mm: float,
) -> dict[str, int]:
    """Count finite biological-ID jumps for an identity-state sequence."""
    if len(states) < 2:
        return {"jump_rows": 0, "unique_jump_frames": 0}
    row = np.arange(len(states) - 1)[:, None]
    previous_slots = STATE_SLOT_BY_IDENTITY[states[:-1]]
    current_slots = STATE_SLOT_BY_IDENTITY[states[1:]]
    previous = xyz[:-1][row, previous_slots]
    current = xyz[1:][row, current_slots]
    delta = current - previous
    finite = np.isfinite(delta).all(axis=2)
    step = np.linalg.norm(np.where(finite[..., None], delta, 0.0), axis=2)
    jumping = finite & (step > threshold_mm)
    return {
        "jump_rows": int(jumping.sum()),
        "unique_jump_frames": int(np.any(jumping, axis=1).sum()),
    }


def _jumping_transition_indices(
    xyz: np.ndarray,
    states: np.ndarray,
    threshold_mm: float,
) -> np.ndarray:
    """Return frame-array indices whose incoming biological-ID step jumps."""
    if len(states) < 2:
        return np.empty(0, dtype=np.int64)
    row = np.arange(len(states) - 1)[:, None]
    previous = xyz[:-1][row, STATE_SLOT_BY_IDENTITY[states[:-1]]]
    current = xyz[1:][row, STATE_SLOT_BY_IDENTITY[states[1:]]]
    delta = current - previous
    finite = np.isfinite(delta).all(axis=2)
    step = np.linalg.norm(np.where(finite[..., None], delta, 0.0), axis=2)
    return np.where(np.any(finite & (step > threshold_mm), axis=1))[0] + 1


def relocate_identity_transitions(
    track: pd.DataFrame,
    *,
    config: TransitionRelocationConfig | None = None,
) -> MotionIdentityResult:
    """Move impossible permutation boundaries to nearby continuous crossings.

    Unlike the global Viterbi optimizer, this conservative layer preserves the
    observed order of appearance regimes.  Only their boundary frames may
    move, and only inside non-overlapping neighborhoods bounded by the
    midpoints to adjacent transitions.  A boundary is moved when its original
    transition exceeds ``jump_threshold_mm`` and a lower-cost candidate exists
    (and, by default, is at or below the threshold).
    """
    config = config or TransitionRelocationConfig()
    _validate_track(track)
    preparation_config = MotionIdentityConfig(anchor_first_complete=False)
    frames, xyz, observed, _evidence, row_lookup = _prepare_arrays(
        track, preparation_config
    )
    states = _states_from_observed(observed)
    transition_frames = np.where(np.diff(states) != 0)[0] + 1
    run_boundaries = np.r_[0, transition_frames, len(states)]
    run_states = states[run_boundaries[:-1]]
    relocated = transition_frames.astype(np.int64, copy=True)
    decisions: list[dict[str, Any]] = []

    for transition_index, original in enumerate(transition_frames):
        before_state = int(run_states[transition_index])
        after_state = int(run_states[transition_index + 1])
        original_max, original_sum = _boundary_motion(
            xyz, int(original), before_state, after_state
        )
        if not np.isfinite(original_max) or original_max <= config.jump_threshold_mm:
            continue

        previous_boundary = int(run_boundaries[transition_index])
        next_boundary = int(run_boundaries[transition_index + 2])
        midpoint_low = (previous_boundary + int(original)) // 2 + 1
        midpoint_high = (int(original) + next_boundary) // 2
        lower = max(
            1,
            int(original) - config.search_radius_frames,
            midpoint_low,
        )
        upper = min(
            len(states) - 1,
            int(original) + config.search_radius_frames,
            midpoint_high,
        )
        if lower > upper:
            continue

        best_frame = int(original)
        best_max, best_sum = original_max, original_sum
        for candidate in range(lower, upper + 1):
            candidate_max, candidate_sum = _boundary_motion(
                xyz, candidate, before_state, after_state
            )
            if (candidate_max, candidate_sum, abs(candidate - int(original))) < (
                best_max,
                best_sum,
                abs(best_frame - int(original)),
            ):
                best_frame, best_max, best_sum = candidate, candidate_max, candidate_sum

        acceptable = best_frame != int(original) and best_max < original_max
        if config.require_below_threshold:
            acceptable &= best_max <= config.jump_threshold_mm
        if acceptable:
            relocated[transition_index] = best_frame
            decisions.append(
                {
                    "original_frame": int(frames[original]),
                    "relocated_frame": int(frames[best_frame]),
                    "frame_offset": int(frames[best_frame] - frames[original]),
                    "before_state": before_state,
                    "after_state": after_state,
                    "original_max_step_mm": original_max,
                    "relocated_max_step_mm": best_max,
                }
            )

    # Midpoint-constrained search should already preserve order. The monotonic
    # clamp is a final guard against adjacent one-frame regimes.
    for index in range(1, len(relocated)):
        relocated[index] = max(relocated[index], relocated[index - 1] + 1)
    new_states = np.empty_like(states)
    starts = np.r_[0, relocated]
    ends = np.r_[relocated, len(states)]
    for state, start, end in zip(run_states, starts, ends):
        new_states[int(start):int(end)] = int(state)

    labels_grid = STATE_LABELS_BY_SLOT[new_states]
    labels = labels_grid.reshape(-1)[row_lookup]
    optimized = track.copy()
    optimized["id_final"] = labels
    changed = labels_grid != STATE_LABELS_BY_SLOT[states]
    before_jump = _jump_summary(xyz, states, config.jump_threshold_mm)
    after_jump = _jump_summary(xyz, new_states, config.jump_threshold_mm)
    diagnostics = {
        "algorithm": "conservative_transition_relocation_v1",
        "config": asdict(config),
        "n_frames": int(len(frames)),
        "input_permutation_changes": int(len(transition_frames)),
        "output_permutation_changes": int(np.sum(np.diff(new_states) != 0)),
        "relocated_boundaries": int(len(decisions)),
        "decisions": decisions,
        "changed_frames": int(np.any(changed, axis=1).sum()),
        "changed_rows": int(changed.sum()),
        "jump_qc_before": before_jump,
        "jump_qc_after": after_jump,
        # Back-compatible names denote unique source frames, not missing
        # transitions (which were previously represented as infinity).
        "transition_jumps_over_threshold_before": before_jump["unique_jump_frames"],
        "transition_jumps_over_threshold_after": after_jump["unique_jump_frames"],
    }
    return MotionIdentityResult(
        labels=np.asarray(labels, dtype=object),
        optimized_track=optimized,
        state_index_by_frame=new_states,
        frame_numbers=frames,
        path_margin_by_frame=np.full(len(frames), np.nan, dtype=np.float32),
        diagnostics=diagnostics,
    )


def repair_identity_local_windows(
    track: pd.DataFrame,
    *,
    config: LocalWindowIdentityConfig | None = None,
) -> MotionIdentityResult:
    """Minimize label edits inside bounded windows around remaining jumps.

    Each window fixes the biological permutation at both ends and searches all
    six permutations in between. A transition is allowed only when every
    finite affected identity moves no farther than ``jump_threshold_mm``. A
    window is applied only if the complete repaired path contains no remaining
    above-threshold transition; infeasible windows are left unchanged and
    reported for manual review.
    """
    config = config or LocalWindowIdentityConfig()
    _validate_track(track)
    preparation_config = MotionIdentityConfig(anchor_first_complete=False)
    frames, xyz, observed, _evidence, row_lookup = _prepare_arrays(
        track, preparation_config
    )
    input_states = _states_from_observed(observed)
    repaired_states = input_states.copy()
    jump_indices = _jumping_transition_indices(
        xyz, input_states, config.jump_threshold_mm
    )

    clusters: list[list[int]] = []
    for frame_index in jump_indices:
        value = int(frame_index)
        if (
            not clusters
            or value - clusters[-1][-1] > 2 * config.radius_frames
        ):
            clusters.append([value])
        else:
            clusters[-1].append(value)

    state_codes = np.vectorize(IDENTITY_INDEX.__getitem__)(
        STATE_LABELS_BY_SLOT
    ).astype(np.int8)
    observed_codes = state_codes[input_states]
    emission = np.sum(
        state_codes[None, :, :] != observed_codes[:, None, :], axis=2
    ).astype(float)
    transition_config = MotionIdentityConfig(
        position_weight=0.0,
        velocity_weight=0.0,
        tangent_weight=0.0,
        acceleration_weight=0.0,
        appearance_weight=1.0,
        switch_penalty=0.0,
        jump_penalty=0.0,
        hard_jump_mm=config.jump_threshold_mm,
        max_gap_frames=10,
        anchor_first_complete=False,
        stable_anchor_min_frames=max(1, len(frames) + 1),
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for event_indices in clusters:
        start = max(0, event_indices[0] - config.radius_frames)
        end = min(len(frames) - 1, event_indices[-1] + config.radius_frames)
        n_window = end - start + 1
        backpointer = np.zeros((n_window, 6), dtype=np.int8)
        cost = np.full(6, np.inf, dtype=float)
        cost[int(repaired_states[start])] = 0.0

        for local_index in range(1, n_window):
            frame_index = start + local_index
            frame_gap = max(
                1, int(frames[frame_index] - frames[frame_index - 1])
            )
            transition, _rejected_count = _transition_cost(
                xyz[frame_index - 1],
                xyz[frame_index],
                np.full((3, 3), np.nan),
                frame_gap=frame_gap,
                velocity_available=np.zeros(3, dtype=bool),
                config=transition_config,
            )
            candidate = cost[:, None] + transition
            previous = np.argmin(candidate, axis=0)
            backpointer[local_index] = previous.astype(np.int8)
            cost = candidate[previous, np.arange(6)] + emission[frame_index]
            if local_index == n_window - 1:
                target = int(repaired_states[end])
                target_cost = float(cost[target])
                cost[:] = np.inf
                cost[target] = target_cost

        target = int(repaired_states[end])
        if not np.isfinite(cost[target]):
            rejected.append(
                {
                    "start_frame": int(frames[start]),
                    "end_frame": int(frames[end]),
                    "jump_frames": [int(frames[index]) for index in event_indices],
                    "reason": "no_hard_continuity_path",
                }
            )
            continue

        path = np.empty(n_window, dtype=np.int8)
        path[-1] = target
        for local_index in range(n_window - 1, 0, -1):
            path[local_index - 1] = backpointer[
                local_index, path[local_index]
            ]
        after_qc = _jump_summary(
            xyz[start : end + 1], path, config.jump_threshold_mm
        )
        if after_qc["unique_jump_frames"]:
            rejected.append(
                {
                    "start_frame": int(frames[start]),
                    "end_frame": int(frames[end]),
                    "jump_frames": [int(frames[index]) for index in event_indices],
                    "reason": "candidate_retained_jump",
                }
            )
            continue

        changed = int(np.sum(path != repaired_states[start : end + 1]))
        repaired_states[start : end + 1] = path
        accepted.append(
            {
                "start_frame": int(frames[start]),
                "end_frame": int(frames[end]),
                "input_jump_frames": [
                    int(frames[index]) for index in event_indices
                ],
                "changed_frames": changed,
            }
        )

    labels_grid = STATE_LABELS_BY_SLOT[repaired_states]
    labels = labels_grid.reshape(-1)[row_lookup]
    optimized = track.copy()
    optimized["id_final"] = labels
    changed_grid = labels_grid != STATE_LABELS_BY_SLOT[input_states]
    diagnostics = {
        "algorithm": "bounded_local_minimum_edit_v1",
        "config": asdict(config),
        "n_frames": int(len(frames)),
        "input_jump_qc": _jump_summary(
            xyz, input_states, config.jump_threshold_mm
        ),
        "output_jump_qc": _jump_summary(
            xyz, repaired_states, config.jump_threshold_mm
        ),
        "input_jump_clusters": int(len(clusters)),
        "accepted_clusters": int(len(accepted)),
        "rejected_clusters": int(len(rejected)),
        "accepted": accepted,
        "rejected": rejected,
        "changed_frames": int(np.any(changed_grid, axis=1).sum()),
        "changed_rows": int(changed_grid.sum()),
    }
    return MotionIdentityResult(
        labels=np.asarray(labels, dtype=object),
        optimized_track=optimized,
        state_index_by_frame=repaired_states,
        frame_numbers=frames,
        path_margin_by_frame=np.full(len(frames), np.nan, dtype=np.float32),
        diagnostics=diagnostics,
    )


def repair_identity_track_layered(
    track: pd.DataFrame,
    *,
    config: LayeredIdentityConfig | None = None,
) -> MotionIdentityResult:
    """Apply bracketed-flicker removal followed by boundary relocation.

    This is the conservative default for existing appearance-labelled tracks.
    It preserves every sustained permutation regime and the order of the
    remaining regimes, limiting identity edits to locally auditable spans.
    """
    config = config or LayeredIdentityConfig()
    _validate_track(track)
    preparation_config = MotionIdentityConfig(anchor_first_complete=False)
    frames, xyz, observed, _evidence, row_lookup = _prepare_arrays(
        track, preparation_config
    )
    original_states = _states_from_observed(observed)
    debounced_states = original_states.copy()
    boundaries = np.r_[
        0,
        np.where(np.diff(original_states) != 0)[0] + 1,
        len(original_states),
    ]
    flicker_spans: list[dict[str, Any]] = []
    for run_index in range(1, len(boundaries) - 2):
        start = int(boundaries[run_index])
        end = int(boundaries[run_index + 1])
        before_state = int(original_states[boundaries[run_index - 1]])
        run_state = int(original_states[start])
        after_state = int(original_states[end])
        if before_state != after_state:
            continue
        duration = int(frames[end - 1] - frames[start] + 1)
        if duration > config.max_flicker_frames:
            continue
        entry_max, _entry_sum = _boundary_motion(
            xyz, start, before_state, run_state
        )
        if not np.isfinite(entry_max) or entry_max <= config.jump_threshold_mm:
            continue
        debounced_states[start:end] = before_state
        flicker_spans.append(
            {
                "start_frame": int(frames[start]),
                "end_frame": int(frames[end - 1]),
                "duration_frames": duration,
                "before_state": before_state,
                "removed_state": run_state,
                "entry_max_step_mm": entry_max,
            }
        )

    intermediate = track.copy()
    intermediate_labels = STATE_LABELS_BY_SLOT[debounced_states].reshape(-1)[
        row_lookup
    ]
    intermediate["id_final"] = intermediate_labels
    relocation_config = TransitionRelocationConfig(
        jump_threshold_mm=config.jump_threshold_mm,
        search_radius_frames=config.search_radius_frames,
        require_below_threshold=config.require_below_threshold,
    )
    relocated = relocate_identity_transitions(
        intermediate, config=relocation_config
    )
    localized = repair_identity_local_windows(
        relocated.optimized_track,
        config=LocalWindowIdentityConfig(
            jump_threshold_mm=config.jump_threshold_mm,
            radius_frames=config.local_window_radius_frames,
        ),
    )
    final_states = localized.state_index_by_frame
    final_labels = localized.labels
    optimized = track.copy()
    optimized["id_final"] = final_labels
    original_grid = STATE_LABELS_BY_SLOT[original_states]
    final_grid = STATE_LABELS_BY_SLOT[final_states]
    changed = original_grid != final_grid
    diagnostics = {
        "algorithm": "layered_identity_motion_repair_v1",
        "config": asdict(config),
        "n_frames": int(len(frames)),
        "input_permutation_changes": int(
            np.sum(np.diff(original_states) != 0)
        ),
        "after_flicker_permutation_changes": int(
            np.sum(np.diff(debounced_states) != 0)
        ),
        "output_permutation_changes": int(np.sum(np.diff(final_states) != 0)),
        "removed_flicker_spans": int(len(flicker_spans)),
        "flicker_spans": flicker_spans,
        "relocated_boundaries": int(
            relocated.diagnostics["relocated_boundaries"]
        ),
        "relocation_decisions": relocated.diagnostics["decisions"],
        "local_window_repair": localized.diagnostics,
        "changed_frames": int(np.any(changed, axis=1).sum()),
        "changed_rows": int(changed.sum()),
        "jump_qc": {
            "before": _jump_summary(
                xyz, original_states, config.jump_threshold_mm
            ),
            "after_flicker": _jump_summary(
                xyz, debounced_states, config.jump_threshold_mm
            ),
            "after_relocation": _jump_summary(
                xyz,
                relocated.state_index_by_frame,
                config.jump_threshold_mm,
            ),
            "after_local_windows": _jump_summary(
                xyz, final_states, config.jump_threshold_mm
            ),
        },
    }
    return MotionIdentityResult(
        labels=np.asarray(final_labels, dtype=object),
        optimized_track=optimized,
        state_index_by_frame=final_states,
        frame_numbers=frames,
        path_margin_by_frame=np.full(len(frames), np.nan, dtype=np.float32),
        diagnostics=diagnostics,
    )


__all__ = [
    "FISH_IDS",
    "MotionIdentityConfig",
    "MotionIdentityResult",
    "LayeredIdentityConfig",
    "LocalWindowIdentityConfig",
    "TransitionRelocationConfig",
    "optimize_identity_track",
    "repair_identity_track_layered",
    "repair_identity_local_windows",
    "relocate_identity_transitions",
]
