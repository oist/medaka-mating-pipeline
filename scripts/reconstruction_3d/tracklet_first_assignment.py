from __future__ import annotations

"""Constraint primitives for tracklet-first multi-view reconstruction.

The functions in this module are intentionally independent of SLEAP and the
calibration implementation.  They provide the two pieces that must be shared
by reconstruction, diagnostics, and tests:

* all-different propagation over concrete-tracklet identity domains; and
* global selection of a mutually-exclusive set of candidate reconstructions.

Keeping these operations pure also makes the production six-candidate path
testable without loading videos, ``.slp`` files, or joblib models.
"""

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


DISPOSITION_REASONS = frozenset(
    {
        "reflection_or_extra_candidate",
        "no_compatible_cross_view_detection",
        "out_of_xyz_bounds",
        "residual_above_threshold",
        "globally_not_selected",
        "tracklet_identity_conflict",
        "motion_gate_rejection",
        "unresolved_identity",
        "partial_2d_identity_only",
    }
)


@dataclass(frozen=True)
class ReconstructionCandidate:
    """One real or completed multi-view reconstruction proposal."""

    slots: tuple[int, int, int]
    xyz: tuple[float, float, float]
    residual: float
    identity: str | None = None
    synthetic_view: int | None = None
    motion_cost: float = 0.0
    identity_conflict: bool = False
    motion_rejected: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    @property
    def cost(self) -> float:
        return float(self.residual + self.motion_cost)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[ReconstructionCandidate, ...]
    alternatives_considered: int
    rejected_identity_conflicts: int
    rejected_motion_gate: int


def _resource_masks(candidate: ReconstructionCandidate) -> tuple[int, int, int]:
    return tuple(0 if slot < 0 else 1 << int(slot) for slot in candidate.slots)


def select_global_candidate_set(
    candidates: Sequence[ReconstructionCandidate],
    *,
    max_selected: int = 3,
    max_states: int | None = 4096,
) -> SelectionResult:
    """Choose the best globally compatible candidate set.

    The dynamic program retains the cheapest partition for every used-slot and
    used-identity state.  Consequently a low-residual first proposal cannot
    consume a detection needed by a lower-cost complete set.  ``max_states``
    bounds production work; ``None`` gives an exact search for focused tests.
    The objective is lexicographic: maximize reconstructed fish first, then
    minimize total residual+motion cost.
    """

    viable = [
        c
        for c in candidates
        if np.isfinite(c.cost) and not c.identity_conflict and not c.motion_rejected
    ]
    viable.sort(key=lambda c: (c.cost, c.slots, c.identity or ""))
    if max_states is not None and len(viable) > 48:
        # The full 6^3 pool contains many reflection-driven variants of the
        # same detections.  Retain a resource-diverse production frontier:
        # the best proposal touching every slot, the best proposals for every
        # locked identity, then globally cheapest proposals up to 48.  Focused
        # audit/tests pass max_states=None and remain exhaustive.
        resource_mandatory: set[int] = set()
        for view_idx in range(3):
            slots = sorted({c.slots[view_idx] for c in viable if c.slots[view_idx] >= 0})
            for slot in slots:
                resource_mandatory.add(next(
                    i for i, c in enumerate(viable) if c.slots[view_idx] == slot
                ))
        identity_mandatory: set[int] = set()
        for identity in sorted({c.identity for c in viable if c.identity is not None}):
            identity_mandatory.update([
                i for i, c in enumerate(viable) if c.identity == identity
            ][:6])
        mandatory = resource_mandatory | identity_mandatory
        chosen_indices = list(sorted(mandatory, key=lambda i: viable[i].cost))
        for i in range(len(viable)):
            if i not in mandatory and len(chosen_indices) < 48:
                chosen_indices.append(i)
        viable = [viable[i] for i in sorted(chosen_indices, key=lambda i: viable[i].cost)]
    identities = sorted({c.identity for c in viable if c.identity is not None})
    identity_bit = {identity: 1 << i for i, identity in enumerate(identities)}

    # (mask0, mask1, mask2, identity_mask) -> (cost, chosen tuple)
    states: dict[tuple[int, int, int, int], tuple[float, tuple[ReconstructionCandidate, ...]]] = {
        (0, 0, 0, 0): (0.0, ())
    }
    best_by_count: dict[int, tuple[float, tuple[ReconstructionCandidate, ...]]] = {0: (0.0, ())}
    considered = 0
    for count in range(1, max_selected + 1):
        next_states: dict[
            tuple[int, int, int, int],
            tuple[float, tuple[ReconstructionCandidate, ...]],
        ] = {}
        for state_key, (state_cost, chosen) in states.items():
            m0, m1, m2, used_ids = state_key
            for candidate in viable:
                considered += 1
                cm0, cm1, cm2 = _resource_masks(candidate)
                if (m0 & cm0) or (m1 & cm1) or (m2 & cm2):
                    continue
                ibit = identity_bit.get(candidate.identity, 0)
                if ibit and (used_ids & ibit):
                    continue
                key = (m0 | cm0, m1 | cm1, m2 | cm2, used_ids | ibit)
                value = (state_cost + candidate.cost, chosen + (candidate,))
                old = next_states.get(key)
                if old is None or value[0] < old[0]:
                    next_states[key] = value
        if not next_states:
            break
        if max_states is not None and len(next_states) > max_states:
            ordered = sorted(
                next_states.items(),
                key=lambda item: (item[1][0], item[0]),
            )[:max_states]
            next_states = dict(ordered)
        best_by_count[count] = min(next_states.values(), key=lambda value: value[0])
        states = next_states

    best_count = max(best_by_count)
    selected = best_by_count[best_count][1]
    return SelectionResult(
        selected=tuple(sorted(selected, key=lambda c: (c.identity or "~", c.cost))),
        alternatives_considered=int(considered),
        rejected_identity_conflicts=sum(bool(c.identity_conflict) for c in candidates),
        rejected_motion_gate=sum(bool(c.motion_rejected) for c in candidates),
    )


def propagate_identity_domains(
    node_ids: Iterable[str],
    *,
    locked_identities: Mapping[str, str],
    simultaneous_groups: Iterable[Sequence[str]],
    identity_labels: Sequence[str] = ("ID_1", "ID_2", "ID_3"),
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Propagate one-to-one identity constraints to a fixed point.

    Each simultaneous group is an all-different constraint (normally the
    concrete tracklets visible in one camera at one frame).  Singleton domains
    are locked permanently.  In particular, when two of three simultaneous
    fish are known, the third immediately receives the remaining identity.
    Contradictory constraints never overwrite a locked identity; their domains
    become empty so callers can report/escalate the ambiguity.
    """

    labels = set(map(str, identity_labels))
    domains = {str(node): set(labels) for node in node_ids}
    sources: dict[str, str] = {}
    for node, identity in locked_identities.items():
        node, identity = str(node), str(identity)
        if node in domains and identity in labels:
            domains[node] = {identity}
            sources[node] = "locked_seed_or_geometry"

    groups = [tuple(dict.fromkeys(map(str, group))) for group in simultaneous_groups]
    groups = [group for group in groups if len(group) >= 2]
    changed = True
    while changed:
        changed = False
        for group in groups:
            present = [node for node in group if node in domains]
            singleton_owner: dict[str, str] = {}
            duplicate_singletons: set[str] = set()
            for node in present:
                if len(domains[node]) == 1:
                    identity = next(iter(domains[node]))
                    if identity in singleton_owner and singleton_owner[identity] != node:
                        duplicate_singletons.add(identity)
                    else:
                        singleton_owner[identity] = node
            fixed = set(singleton_owner) - duplicate_singletons
            for node in present:
                if len(domains[node]) == 1:
                    continue
                reduced = domains[node] - fixed
                if reduced != domains[node]:
                    domains[node] = reduced
                    changed = True
                    if len(reduced) == 1:
                        sources[node] = "one_to_one_exclusion"

            # Hall-set form specialized to the three-fish case.  If all but
            # one identities are owned inside a complete group, assign the
            # remaining domain even when the group arrived with duplicates.
            if len(present) == len(identity_labels):
                known = {
                    next(iter(domains[node]))
                    for node in present
                    if len(domains[node]) == 1
                }
                unresolved = [node for node in present if len(domains[node]) > 1]
                remaining = labels - known
                if len(unresolved) == 1 and len(remaining) == 1:
                    node = unresolved[0]
                    if domains[node] != remaining:
                        domains[node] = set(remaining)
                        sources[node] = "one_to_one_exclusion"
                        changed = True

    resolved = {
        node: next(iter(domain))
        for node, domain in domains.items()
        if len(domain) == 1
    }
    return resolved, sources, domains


def candidate_identity(
    slots: tuple[int, int, int],
    *,
    slot_nodes: Sequence[Mapping[int, str]],
    node_identities: Mapping[str, str],
) -> tuple[str | None, bool, tuple[str, ...]]:
    """Return consensus identity for the concrete tracklets in a proposal."""

    evidence = []
    for view_idx, slot in enumerate(slots):
        if slot < 0:
            continue
        node = slot_nodes[view_idx].get(int(slot))
        identity = node_identities.get(str(node)) if node is not None else None
        if identity is not None:
            evidence.append(str(identity))
    unique = tuple(sorted(set(evidence)))
    if len(unique) > 1:
        return None, True, unique
    return (unique[0] if unique else None), False, unique


def build_detection_dispositions(
    points_by_frame: Mapping[int, Mapping[str, Sequence[Sequence[float]]]],
    best_by_frame: Mapping[int, Mapping[str, object]],
    *,
    slot_nodes_by_frame: Mapping[int, Mapping[str, Mapping[int, str]]],
    node_identities: Mapping[str, str],
) -> list[dict[str, object]]:
    """Give every finite candidate-slot detection exactly one disposition."""

    rows: list[dict[str, object]] = []
    for frame in sorted(points_by_frame):
        info = best_by_frame.get(frame, {})
        selected_by_cam: dict[str, dict[int, tuple[int, bool, int]]] = {}
        recovered = list(info.get("recovered", []) or [])
        synthetic_views = list(info.get("synthetic_views", []) or [])
        source_hints = dict(info.get("source_id_hints", {}) or {})
        for view_idx, cam in enumerate(("cam0", "cam1", "cam2")):
            selected_by_cam[cam] = {}
            for source_idx, slot in enumerate(list(info.get(f"perm_{cam}", []) or [])):
                slot = int(slot)
                if slot >= 0:
                    selected_by_cam[cam][slot] = (
                        source_idx,
                        bool(source_idx < len(recovered) and recovered[source_idx]),
                        int(synthetic_views[source_idx]) if source_idx < len(synthetic_views) else -1,
                    )
        for view_idx, cam in enumerate(("cam0", "cam1", "cam2")):
            nodes = slot_nodes_by_frame.get(frame, {}).get(cam, {})
            for slot, xy in enumerate(points_by_frame[frame].get(cam, ())):
                xy_arr = np.asarray(xy, dtype=float)
                if xy_arr.shape != (2,) or not np.isfinite(xy_arr).all():
                    continue
                node = nodes.get(slot)
                identity = node_identities.get(str(node), "") if node is not None else ""
                selected = selected_by_cam[cam].get(slot)
                source_idx = selected[0] if selected is not None else -1
                selected_identity = source_hints.get(source_idx, identity)
                if selected is not None:
                    is_partial = bool(selected[1])
                    disposition = (
                        "assigned_3d_synthetic_missing_view" if is_partial else "assigned_2d_valid_3d"
                    )
                    reason = "assigned_two_view_completed" if is_partial else "assigned_three_view"
                elif identity:
                    disposition = "assigned_2d_only"
                    reason = "partial_2d_identity_only"
                elif node is not None:
                    disposition = "unresolved_detection"
                    reason = "unresolved_identity"
                else:
                    disposition = "extra_candidate"
                    reason = "reflection_or_extra_candidate"
                rows.append(
                    {
                        "frame": int(frame),
                        "view": ("top_cam", "side1_cam", "side2_cam")[view_idx],
                        "camera": cam,
                        "candidate_slot": int(slot),
                        "x": float(xy_arr[0]),
                        "y": float(xy_arr[1]),
                        "tracklet_node": str(node) if node is not None else "",
                        "final_fish_id": str(selected_identity or identity),
                        "disposition": disposition,
                        "primary_reason": reason,
                        "selected_source_idx": int(source_idx),
                        "synthetic_view": int(selected[2]) if selected is not None else -1,
                    }
                )
    return rows
