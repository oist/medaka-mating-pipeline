from __future__ import annotations

"""Geometry-preserving global stitching of identity fragments.

This module deliberately does not modify reconstructed coordinates.  It treats
3D geometry, reprojection, silhouettes, masks, and appearance as evidence on a
directed fragment graph.  Biological identities enter only through explicit
trusted anchors.  The solver is a constrained minimum-cost path cover: selected
edges have at most one predecessor and successor per fragment, anchor labels
propagate along selected paths, and simultaneous fragments cannot receive the
same biological identity.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


EVIDENCE_STATES = frozenset({"support", "unknown", "contradiction"})


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    frame_start: int
    frame_end: int
    trusted_identity: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if int(self.frame_end) < int(self.frame_start):
            raise ValueError(f"fragment {self.fragment_id!r} ends before it starts")


@dataclass(frozen=True)
class EdgeFeatures:
    source_id: str
    target_id: str
    gap_frames: int
    endpoint_displacement_mm: float | None = None
    motion_prediction_error_mm: float | None = None
    velocity_angle_deg: float | None = None
    reprojection_error_px: float | None = None
    silhouette_iou: float | None = None
    mask_overlap: float | None = None
    appearance_cosine_distance: float | None = None
    appearance_state: str = "unknown"
    silhouette_state: str = "unknown"
    strong_one_to_one_contradiction: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.silhouette_state not in EVIDENCE_STATES:
            raise ValueError(
                f"silhouette_state must be one of {sorted(EVIDENCE_STATES)}, "
                f"got {self.silhouette_state!r}"
            )
        if self.appearance_state not in EVIDENCE_STATES:
            raise ValueError(
                f"appearance_state must be one of {sorted(EVIDENCE_STATES)}, "
                f"got {self.appearance_state!r}"
            )


@dataclass(frozen=True)
class CostWeights:
    endpoint_displacement: float = 1.0
    motion_prediction: float = 2.0
    velocity_angle: float = 0.75
    reprojection: float = 1.0
    silhouette_continuity: float = 0.75
    mask_overlap: float = 0.5
    appearance: float = 1.5
    temporal_gap: float = 0.25
    silhouette_support_bonus: float = 0.35
    appearance_support_bonus: float = 0.5
    displacement_scale_mm: float = 8.0
    motion_scale_mm: float = 8.0
    angle_scale_deg: float = 90.0
    reprojection_scale_px: float = 20.0
    appearance_scale: float = 0.5
    gap_scale_frames: float = 30.0
    clip_normalized: float = 4.0


@dataclass(frozen=True)
class CostGates:
    max_gap_frames: int = 100
    max_jump_mm_per_frame: float = 8.0
    max_jump_hard_cap_mm: float = 80.0
    max_motion_prediction_error_mm: float = 80.0
    max_reprojection_error_px: float = 120.0


@dataclass(frozen=True)
class EdgeCost:
    source_id: str
    target_id: str
    feasible: bool
    total_cost: float
    reason: str
    terms: Mapping[str, float]
    features: EdgeFeatures = field(compare=False)


@dataclass(frozen=True)
class PathCoverResult:
    selected_edges: tuple[EdgeCost, ...]
    rejected_edges: tuple[EdgeCost, ...]
    fragment_identities: Mapping[str, str]
    component_by_fragment: Mapping[str, str]
    objective: float
    solver_status: str


def _finite(value: float | None) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _scaled(value: float | None, scale: float, clip: float) -> float:
    """Return a neutral zero penalty for missing/unknown evidence."""
    if not _finite(value):
        return 0.0
    return float(np.clip(max(0.0, float(value)) / max(float(scale), 1e-12), 0.0, clip))


def _overlap_penalty(value: float | None) -> float:
    if not _finite(value):
        return 0.0
    return float(1.0 - np.clip(float(value), 0.0, 1.0))


def calculate_edge_cost(
    features: EdgeFeatures,
    *,
    weights: CostWeights = CostWeights(),
    gates: CostGates = CostGates(),
) -> EdgeCost:
    """Calculate an inspectable edge cost without converting absence to veto.

    Missing masks, split/merge ambiguity, and unavailable appearance are neutral
    terms.  Only an explicitly validated one-fish-to-one-mask contradiction is
    a segmentation hard veto.
    """

    def reject(reason: str) -> EdgeCost:
        return EdgeCost(
            source_id=features.source_id,
            target_id=features.target_id,
            feasible=False,
            total_cost=float("inf"),
            reason=reason,
            terms={},
            features=features,
        )

    if features.source_id == features.target_id:
        return reject("self_link")
    if int(features.gap_frames) < 0:
        return reject("temporal_overlap")
    if int(features.gap_frames) > int(gates.max_gap_frames):
        return reject("gap_exceeds_gate")
    if features.strong_one_to_one_contradiction:
        return reject("strong_one_to_one_mask_contradiction")

    bridge_dt = max(1, int(features.gap_frames) + 1)
    jump_gate = min(
        float(gates.max_jump_mm_per_frame) * bridge_dt,
        float(gates.max_jump_hard_cap_mm),
    )
    if (
        _finite(features.endpoint_displacement_mm)
        and float(features.endpoint_displacement_mm) > jump_gate
    ):
        return reject("endpoint_displacement_exceeds_gate")
    if (
        _finite(features.motion_prediction_error_mm)
        and float(features.motion_prediction_error_mm)
        > float(gates.max_motion_prediction_error_mm)
    ):
        return reject("motion_prediction_exceeds_gate")
    if (
        _finite(features.reprojection_error_px)
        and float(features.reprojection_error_px) > float(gates.max_reprojection_error_px)
    ):
        return reject("reprojection_exceeds_gate")

    terms = {
        "endpoint_displacement": weights.endpoint_displacement
        * _scaled(
            features.endpoint_displacement_mm,
            weights.displacement_scale_mm * bridge_dt,
            weights.clip_normalized,
        ),
        "motion_prediction": weights.motion_prediction
        * _scaled(
            features.motion_prediction_error_mm,
            weights.motion_scale_mm,
            weights.clip_normalized,
        ),
        "velocity_angle": weights.velocity_angle
        * _scaled(
            features.velocity_angle_deg,
            weights.angle_scale_deg,
            weights.clip_normalized,
        ),
        "reprojection": weights.reprojection
        * _scaled(
            features.reprojection_error_px,
            weights.reprojection_scale_px,
            weights.clip_normalized,
        ),
        "silhouette_continuity": (
            weights.silhouette_continuity * _overlap_penalty(features.silhouette_iou)
            if features.silhouette_state == "support"
            else 0.0
        ),
        "mask_overlap": (
            weights.mask_overlap * _overlap_penalty(features.mask_overlap)
            if features.silhouette_state == "support"
            else 0.0
        ),
        "appearance": (
            weights.appearance
            * _scaled(
                features.appearance_cosine_distance,
                weights.appearance_scale,
                weights.clip_normalized,
            )
            if features.appearance_state in {"support", "contradiction"}
            else 0.0
        ),
        "appearance_support_bonus": (
            -float(weights.appearance_support_bonus)
            if features.appearance_state == "support"
            else 0.0
        ),
        "temporal_gap": weights.temporal_gap
        * _scaled(
            features.gap_frames,
            weights.gap_scale_frames,
            weights.clip_normalized,
        ),
        "silhouette_support_bonus": (
            -float(weights.silhouette_support_bonus)
            if features.silhouette_state == "support"
            else 0.0
        ),
    }
    return EdgeCost(
        source_id=features.source_id,
        target_id=features.target_id,
        feasible=True,
        total_cost=max(0.0, float(sum(terms.values()))),
        reason="feasible",
        terms=terms,
        features=features,
    )


def _overlap(left: Fragment, right: Fragment) -> bool:
    return max(int(left.frame_start), int(right.frame_start)) <= min(
        int(left.frame_end), int(right.frame_end)
    )


def solve_constrained_path_cover(
    fragments: Sequence[Fragment],
    edges: Sequence[EdgeCost],
    *,
    identity_labels: Sequence[str] = ("ID_1", "ID_2", "ID_3"),
    break_cost: float = 3.0,
    unanchored_label_penalty: float = 1e-5,
) -> PathCoverResult:
    """Solve a globally constrained path cover with trusted anchor propagation.

    Each selected link replaces a path death plus a path birth, so its MILP
    coefficient is ``edge_cost - 2 * break_cost``.  Unanchored fragments remain
    biologically unknown unless connected to a trusted anchor.  Geometry-local
    IDs are intentionally absent from this interface.
    """

    if break_cost < 0:
        raise ValueError("break_cost must be non-negative")
    labels = tuple(map(str, identity_labels))
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("identity_labels must be unique and non-empty")
    fragment_by_id = {str(fragment.fragment_id): fragment for fragment in fragments}
    if len(fragment_by_id) != len(fragments):
        raise ValueError("fragment_id values must be unique")
    ids = tuple(sorted(fragment_by_id))
    id_index = {fragment_id: i for i, fragment_id in enumerate(ids)}
    label_index = {label: i for i, label in enumerate(labels)}
    for fragment in fragments:
        if fragment.trusted_identity is not None and str(fragment.trusted_identity) not in label_index:
            raise ValueError(
                f"unknown trusted identity {fragment.trusted_identity!r} for {fragment.fragment_id!r}"
            )

    # Two simultaneous trusted anchors with the same identity are a ground-truth
    # conflict, not an optimization choice.
    anchored = [f for f in fragments if f.trusted_identity is not None]
    for i, left in enumerate(anchored):
        for right in anchored[i + 1 :]:
            if left.trusted_identity == right.trusted_identity and _overlap(left, right):
                raise ValueError(
                    "simultaneous trusted anchors share identity "
                    f"{left.trusted_identity!r}: {left.fragment_id!r}, {right.fragment_id!r}"
                )

    rejected: list[EdgeCost] = []
    feasible: list[EdgeCost] = []
    seen_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (str(edge.source_id), str(edge.target_id))
        if pair in seen_pairs:
            raise ValueError(f"duplicate directed edge {pair!r}")
        seen_pairs.add(pair)
        source = fragment_by_id.get(pair[0])
        target = fragment_by_id.get(pair[1])
        if source is None or target is None:
            raise ValueError(f"edge references unknown fragment: {pair!r}")
        if not edge.feasible:
            rejected.append(edge)
            continue
        if int(source.frame_end) >= int(target.frame_start):
            rejected.append(
                EdgeCost(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    feasible=False,
                    total_cost=float("inf"),
                    reason="non_forward_or_overlapping_fragments",
                    terms=edge.terms,
                    features=edge.features,
                )
            )
            continue
        feasible.append(edge)

    n_edges = len(feasible)
    n_fragments = len(ids)
    n_labels = len(labels)
    n_variables = n_edges + n_fragments * n_labels

    if n_fragments == 0:
        return PathCoverResult((), tuple(rejected), {}, {}, 0.0, "empty")

    def y_var(fragment_id: str, label: str) -> int:
        return n_edges + id_index[fragment_id] * n_labels + label_index[label]

    objective = np.zeros(n_variables, dtype=float)
    for edge_idx, edge in enumerate(feasible):
        objective[edge_idx] = float(edge.total_cost) - 2.0 * float(break_cost)
    for fragment_id in ids:
        fragment = fragment_by_id[fragment_id]
        for label in labels:
            if fragment.trusted_identity is None:
                objective[y_var(fragment_id, label)] = float(unanchored_label_penalty)

    lower = np.zeros(n_variables, dtype=float)
    upper = np.ones(n_variables, dtype=float)
    for fragment_id in ids:
        anchor = fragment_by_id[fragment_id].trusted_identity
        if anchor is None:
            continue
        for label in labels:
            index = y_var(fragment_id, label)
            fixed = 1.0 if str(anchor) == label else 0.0
            lower[index] = fixed
            upper[index] = fixed

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], lo: float, hi: float) -> None:
        row = len(constraint_lower)
        for column, value in coefficients.items():
            row_indices.append(row)
            column_indices.append(int(column))
            values.append(float(value))
        constraint_lower.append(float(lo))
        constraint_upper.append(float(hi))

    incoming: dict[str, list[int]] = {fragment_id: [] for fragment_id in ids}
    outgoing: dict[str, list[int]] = {fragment_id: [] for fragment_id in ids}
    for edge_idx, edge in enumerate(feasible):
        outgoing[str(edge.source_id)].append(edge_idx)
        incoming[str(edge.target_id)].append(edge_idx)
    for fragment_id in ids:
        if incoming[fragment_id]:
            add_constraint({index: 1.0 for index in incoming[fragment_id]}, 0.0, 1.0)
        if outgoing[fragment_id]:
            add_constraint({index: 1.0 for index in outgoing[fragment_id]}, 0.0, 1.0)
        add_constraint(
            {y_var(fragment_id, label): 1.0 for label in labels}, 0.0, 1.0
        )

    # A selected edge forces equality of all biological-label indicators.  A
    # path without an anchor therefore remains unlabeled at optimum.
    for edge_idx, edge in enumerate(feasible):
        source_id, target_id = str(edge.source_id), str(edge.target_id)
        for label in labels:
            source_y = y_var(source_id, label)
            target_y = y_var(target_id, label)
            add_constraint(
                {edge_idx: 1.0, source_y: 1.0, target_y: -1.0},
                -np.inf,
                1.0,
            )
            add_constraint(
                {edge_idx: 1.0, source_y: -1.0, target_y: 1.0},
                -np.inf,
                1.0,
            )

    # Simultaneous global fragments cannot be the same fish.  This assumes the
    # input nodes are multiview/3D identity fragments, not duplicate per-view
    # observations of one fish.
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            if not _overlap(fragment_by_id[left_id], fragment_by_id[right_id]):
                continue
            for label in labels:
                add_constraint(
                    {y_var(left_id, label): 1.0, y_var(right_id, label): 1.0},
                    -np.inf,
                    1.0,
                )

    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_lower), n_variables),
    ).tocsc()
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=float),
            np.asarray(constraint_upper, dtype=float),
        ),
        options={"presolve": True},
    )
    if not bool(result.success) or result.x is None:
        raise RuntimeError(
            f"global fragment path-cover MILP failed: status={result.status}, "
            f"message={result.message}"
        )

    chosen = tuple(
        edge for edge_idx, edge in enumerate(feasible) if float(result.x[edge_idx]) > 0.5
    )
    identities: dict[str, str] = {}
    for fragment_id in ids:
        selected = [
            label
            for label in labels
            if float(result.x[y_var(fragment_id, label)]) > 0.5
        ]
        if selected:
            identities[fragment_id] = selected[0]

    parent = {fragment_id: fragment_id for fragment_id in ids}

    def find(fragment_id: str) -> str:
        while parent[fragment_id] != fragment_id:
            parent[fragment_id] = parent[parent[fragment_id]]
            fragment_id = parent[fragment_id]
        return fragment_id

    for edge in chosen:
        left, right = find(str(edge.source_id)), find(str(edge.target_id))
        if left != right:
            parent[right] = left
    component = {fragment_id: find(fragment_id) for fragment_id in ids}
    return PathCoverResult(
        selected_edges=chosen,
        rejected_edges=tuple(rejected),
        fragment_identities=identities,
        component_by_fragment=component,
        objective=float(result.fun),
        solver_status=str(result.message),
    )
