from __future__ import annotations

"""Pure array/table primitives for windowed mouth-tracklet association.

This module deliberately separates geometry observations from identity.  The
per-frame six-candidate selector owns raw-detection exclusivity.  Compact 3D
observations are then grouped into conservative fragments, a unit-aware
directed fragment graph is built, and interchangeable solvers label the same
graph.  Interpolation is intentionally outside this module so it can never
become association evidence.
"""

from dataclasses import asdict, dataclass
from itertools import permutations
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from scripts.reconstruction_3d.candidate_cache import CandidateWindow


VIEW_ORDER = ("top_cam", "side1_cam", "side2_cam")
CAM_ORDER = ("cam0", "cam1", "cam2")
ID_LABELS = ("ID_1", "ID_2", "ID_3")
SOURCE_LABELS = ("a", "b", "c")
RESIDUAL_UNIT = "mm^2"


@dataclass(frozen=True)
class WindowSpec:
    index: int
    core_start: int
    core_end: int
    context_start: int
    context_end: int


@dataclass(frozen=True)
class LinkGates:
    max_gap_frames: int = 100
    max_jump_mm_per_frame: float = 8.0
    max_jump_hard_cap_mm: float = 80.0


@dataclass(frozen=True)
class LinkWeights:
    endpoint: float = 1.0
    prediction: float = 2.0
    velocity_angle: float = 0.0
    temporal_gap: float = 0.25
    overlap_consistency: float = 1.0
    residual: float = 0.75
    mouth_confidence: float = 0.5
    endpoint_scale_mm: float = 8.0
    prediction_scale_mm: float = 8.0
    angle_scale_deg: float = 90.0
    gap_scale_frames: float = 100.0
    residual_scale_mm2: float = 2.5
    clip_normalized: float = 4.0
    tail_weight: float = 1.0 / 12.0
    tail_angle_scale_deg: float = 90.0
    mouth_ambiguity_margin: float = 0.25
    close_contact_mm: float = 12.0


@dataclass(frozen=True)
class SolverConfig:
    method: str = "sequential_hungarian"
    beam_width: int = 8
    max_global_states: int = 4096
    birth_cost: float = 3.0
    death_cost: float = 3.0
    ambiguity_margin: float = 0.5
    retention_margin: float = 2.0
    resolution_margin: float = 1.0
    resolution_stability_frames: int = 10
    deferred_horizon_frames: int = 250
    identity_policy: str = "solver_only"
    path_cover_mode: str = "exact_three"


@dataclass(frozen=True)
class SolverResult:
    fragment_identities: Mapping[str, str]
    objective: float
    hypotheses_peak: int
    pruned_states: int
    unresolved_intervals: tuple[Mapping[str, object], ...]
    deterministic_tie_count: int
    method: str
    selected_edges: tuple[str, ...] = ()
    paths: tuple[tuple[str, ...], ...] = ()
    feasible: bool = True
    status: str = "optimal"
    explored_states: int = 0


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_overlapping_windows(
    frame_start: int,
    frame_end: int,
    *,
    core_size: int,
    overlap: int,
    required_context: int = 100,
) -> list[WindowSpec]:
    """Return deterministic core ownership plus symmetric context halos."""
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("invalid inclusive frame range")
    if core_size < 1:
        raise ValueError("core_size must be positive")
    if overlap < required_context:
        raise ValueError(
            f"overlap={overlap} is smaller than required context {required_context}"
        )
    windows: list[WindowSpec] = []
    core_start = frame_start
    index = 0
    while core_start <= frame_end:
        core_end = min(frame_end, core_start + core_size - 1)
        windows.append(
            WindowSpec(
                index=index,
                core_start=core_start,
                core_end=core_end,
                context_start=max(frame_start, core_start - overlap),
                context_end=min(frame_end, core_end + overlap),
            )
        )
        core_start = core_end + 1
        index += 1
    return windows


def _resource_tuple(row: object) -> tuple[int, int, int, int]:
    return (
        int(getattr(row, "frame")),
        int(getattr(row, "slot_cam0")),
        int(getattr(row, "slot_cam1")),
        int(getattr(row, "slot_cam2")),
    )


def reconcile_window_fragments(
    window_tables: Sequence[pd.DataFrame],
    specs: Sequence[WindowSpec],
    *,
    min_alias_support_frames: int = 3,
    min_alias_consistency: float = 0.95,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonicalize exact-resource fragment aliases across adjacent windows.

    The routine never creates a biological bridge.  It only aliases local
    fragments that selected the same complete raw-resource tuple in their
    shared read interval.  Returns core-owned rows, the fragment-pair evidence
    table, and one strict resource-equivalence summary per seam.
    """
    if len(window_tables) != len(specs) or not window_tables:
        raise ValueError("window tables/specs must be non-empty and aligned")
    tables: list[pd.DataFrame] = []
    all_fragment_rows: dict[str, pd.DataFrame] = {}
    for table, spec in zip(window_tables, specs, strict=True):
        required = {
            "frame", "fragment_id", "X", "Y", "Z", "ray_residual_mm2",
            "slot_cam0", "slot_cam1", "slot_cam2",
        }
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"window {spec.index} lacks columns {sorted(missing)}")
        local = table.copy()
        local["local_fragment_id"] = local.fragment_id.astype(str)
        local["fragment_id"] = [
            value if value.startswith(f"w{spec.index:04d}:") else f"w{spec.index:04d}:{value}"
            for value in local.fragment_id.astype(str)
        ]
        if not local.frame.between(spec.context_start, spec.context_end).all():
            raise AssertionError(f"window {spec.index} emitted rows outside its context")
        local["window_index"] = int(spec.index)
        local["window_region"] = np.select(
            [local.frame < spec.core_start, local.frame > spec.core_end],
            ["left_context", "right_context"],
            default="core",
        )
        tables.append(local)
        for fragment_id, group in local.groupby("fragment_id", sort=True):
            all_fragment_rows[str(fragment_id)] = group

    parent = {fragment_id: fragment_id for fragment_id in all_fragment_rows}
    members_by_root = {
        fragment_id: {fragment_id} for fragment_id in all_fragment_rows
    }
    resources_by_root: dict[str, dict[tuple[int, int], int]] = {}
    for fragment_id, group in all_fragment_rows.items():
        resources: dict[tuple[int, int], int] = {}
        for row in group.itertuples(index=False):
            for view in range(3):
                slot = int(getattr(row, f"slot_cam{view}"))
                if slot < 0:
                    continue
                key = (int(row.frame), view)
                old = resources.get(key)
                if old is not None and old != slot:
                    raise AssertionError(
                        f"fragment {fragment_id} violates all-different at {key}"
                    )
                resources[key] = slot
        resources_by_root[fragment_id] = resources

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def compatible(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        left_resources = resources_by_root[left_root]
        right_resources = resources_by_root[right_root]
        smaller, larger = (
            (left_resources, right_resources)
            if len(left_resources) <= len(right_resources)
            else (right_resources, left_resources)
        )
        return all(larger.get(key, slot) == slot for key, slot in smaller.items())

    alias_records: list[dict[str, object]] = []
    seam_records: list[dict[str, object]] = []
    candidate_edges: list[tuple[tuple[object, ...], int]] = []
    resource_columns = ["frame", "slot_cam0", "slot_cam1", "slot_cam2"]
    for seam, (left, right) in enumerate(zip(tables[:-1], tables[1:], strict=True)):
        overlap_start = max(int(left.frame.min()), int(right.frame.min()))
        overlap_end = min(int(left.frame.max()), int(right.frame.max()))
        left_overlap = left[left.frame.between(overlap_start, overlap_end)]
        right_overlap = right[right.frame.between(overlap_start, overlap_end)]
        resource_join = left_overlap.merge(
            right_overlap,
            on=resource_columns,
            how="outer",
            suffixes=("_left", "_right"),
            indicator=True,
        )
        both = resource_join[resource_join._merge == "both"]
        xyz_delta = (
            np.max(
                np.abs(
                    both[["X_left", "Y_left", "Z_left"]].to_numpy(dtype=float)
                    - both[["X_right", "Y_right", "Z_right"]].to_numpy(dtype=float)
                ),
                axis=1,
            )
            if len(both)
            else np.empty(0)
        )
        residual_delta = (
            np.abs(
                both.ray_residual_mm2_left.to_numpy(dtype=float)
                - both.ray_residual_mm2_right.to_numpy(dtype=float)
            )
            if len(both)
            else np.empty(0)
        )
        strict_equal = bool(
            (resource_join._merge == "left_only").sum() == 0
            and (resource_join._merge == "right_only").sum() == 0
            and np.allclose(xyz_delta, 0.0, atol=atol, rtol=rtol)
            and np.allclose(residual_delta, 0.0, atol=atol, rtol=rtol)
        )
        seam_records.append(
            {
                "seam_index": seam,
                "left_window": int(specs[seam].index),
                "right_window": int(specs[seam + 1].index),
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "left_rows": int(len(left_overlap)),
                "right_rows": int(len(right_overlap)),
                "identical_resource_rows": int(len(both)),
                "left_only_rows": int((resource_join._merge == "left_only").sum()),
                "right_only_rows": int((resource_join._merge == "right_only").sum()),
                "max_xyz_delta_mm": float(np.max(xyz_delta)) if len(xyz_delta) else np.nan,
                "max_residual_delta_mm2": float(np.max(residual_delta)) if len(residual_delta) else np.nan,
                "strict_resource_equivalent": strict_equal,
                "unresolved": not strict_equal,
            }
        )
        frames_by_left = {
            str(fragment_id): set(group.frame.astype(int))
            for fragment_id, group in left_overlap.groupby("fragment_id")
        }
        frames_by_right = {
            str(fragment_id): set(group.frame.astype(int))
            for fragment_id, group in right_overlap.groupby("fragment_id")
        }
        # Only exact-resource pairs can reach the minimum alias support.  The
        # seam table above accounts for all nonmatching resources without a
        # quadratic fragment-pair scan.
        for (left_id, right_id), pair_rows in both.groupby(
            ["fragment_id_left", "fragment_id_right"], sort=True
        ):
                left_id, right_id = str(left_id), str(right_id)
                compared = len(frames_by_left[left_id] & frames_by_right[right_id])
                n_same = int(len(pair_rows))
                consistency = n_same / max(compared, 1)
                left_xyz = pair_rows[["X_left", "Y_left", "Z_left"]].to_numpy(dtype=float)
                right_xyz = pair_rows[["X_right", "Y_right", "Z_right"]].to_numpy(dtype=float)
                pair_xyz_delta = np.max(np.abs(left_xyz - right_xyz), axis=1)
                pair_residual_delta = np.abs(
                    pair_rows.ray_residual_mm2_left.to_numpy(dtype=float)
                    - pair_rows.ray_residual_mm2_right.to_numpy(dtype=float)
                )
                numerically_equal = bool(
                    np.allclose(left_xyz, right_xyz, atol=atol, rtol=rtol)
                    and np.allclose(
                        pair_rows.ray_residual_mm2_left.to_numpy(dtype=float),
                        pair_rows.ray_residual_mm2_right.to_numpy(dtype=float),
                        atol=atol,
                        rtol=rtol,
                    )
                )
                eligible = bool(
                    n_same >= min_alias_support_frames
                    and consistency >= min_alias_consistency
                    and numerically_equal
                )
                reason = (
                    "eligible" if eligible else
                    "support_below_minimum" if n_same < min_alias_support_frames else
                    "consistency_below_minimum" if consistency < min_alias_consistency else
                    "numerical_mismatch"
                )
                record = {
                    "seam_index": seam,
                    "left_fragment_id": left_id,
                    "right_fragment_id": right_id,
                    "n_same": n_same,
                    "n_compared": compared,
                    "consistency": consistency,
                    "max_xyz_delta_mm": float(np.max(pair_xyz_delta)),
                    "max_residual_delta_mm2": float(np.max(pair_residual_delta)),
                    "eligible": eligible,
                    "accepted": False,
                    "rejection_reason": reason,
                }
                alias_records.append(record)
                if eligible:
                    sort_key = (
                        -n_same,
                        -consistency,
                        float(np.max(pair_xyz_delta)),
                        int(specs[seam].index),
                        left_id,
                        right_id,
                    )
                    candidate_edges.append((sort_key, len(alias_records) - 1))

    for _sort_key, record_index in sorted(candidate_edges):
        record = alias_records[record_index]
        left_id = str(record["left_fragment_id"])
        right_id = str(record["right_fragment_id"])
        left_root, right_root = find(left_id), find(right_id)
        if left_root == right_root:
            record["accepted"] = True
            record["rejection_reason"] = "already_canonical"
            continue
        if not compatible(left_root, right_root):
            record["rejection_reason"] = "all_different_conflict"
            continue
        root, child = sorted((left_root, right_root))
        parent[child] = root
        members_by_root[root].update(members_by_root.pop(child))
        root_resources = resources_by_root[root]
        root_resources.update(resources_by_root.pop(child))
        record["accepted"] = True
        record["rejection_reason"] = "accepted"

    canonical_by_local: dict[str, str] = {}
    for root in sorted(members_by_root):
        members = sorted(members_by_root[root])
        resources = sorted(
            {
                _resource_tuple(row)
                for member in members
                for row in all_fragment_rows[member].itertuples(index=False)
            }
        )
        first_frame = resources[0][0] if resources else -1
        canonical = f"cfrag_{first_frame:06d}_{stable_hash(resources)[:16]}"
        for member in members:
            canonical_by_local[member] = canonical

    accepted_support: dict[str, tuple[int, int]] = {}
    for record in alias_records:
        if not record["accepted"]:
            continue
        canonical = canonical_by_local[str(record["left_fragment_id"])]
        agree, compared = accepted_support.get(canonical, (0, 0))
        accepted_support[canonical] = (
            agree + int(record["n_same"]),
            compared + int(record["n_compared"]),
        )
    cores = []
    for table, spec in zip(tables, specs, strict=True):
        core = table[table.frame.between(spec.core_start, spec.core_end)].copy()
        core["fragment_id"] = core.fragment_id.astype(str).map(canonical_by_local)
        core["overlap_agree_frames"] = core.fragment_id.map(
            lambda value: accepted_support.get(str(value), (0, 0))[0]
        )
        core["overlap_disagree_frames"] = core.fragment_id.map(
            lambda value: max(
                0,
                accepted_support.get(str(value), (0, 0))[1]
                - accepted_support.get(str(value), (0, 0))[0],
            )
        )
        cores.append(core)
    merged = pd.concat(cores, ignore_index=True).sort_values(
        ["frame", "fragment_id", "source_index"]
    ).reset_index(drop=True)
    if merged.duplicated(resource_columns).any():
        raise AssertionError("core ownership produced a duplicate resource tuple")
    return (
        merged,
        pd.DataFrame.from_records(alias_records),
        pd.DataFrame.from_records(seam_records),
    )


def candidate_windows_to_points(
    windows: Mapping[str, CandidateWindow],
) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """Legacy adapter used only around the verified candidate selector."""
    if set(windows) != set(VIEW_ORDER):
        raise ValueError(f"candidate windows must contain {VIEW_ORDER}")
    starts = {window.frame_start for window in windows.values()}
    ends = {window.frame_end for window in windows.values()}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("candidate windows are not frame-aligned")
    start, end = starts.pop(), ends.pop()
    output: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for local_index, frame in enumerate(range(start, end + 1)):
        cameras: dict[str, list[tuple[float, float]]] = {}
        for view, camera in zip(VIEW_ORDER, CAM_ORDER, strict=True):
            values = windows[view].mouth_xy[local_index]
            cameras[camera] = [
                (float(point[0]), float(point[1]))
                if np.isfinite(point).all()
                else (np.nan, np.nan)
                for point in values
            ]
        output[frame] = cameras
    return output


def best_to_observations(
    best_by_frame: Mapping[int, Mapping[str, object]],
    windows: Mapping[str, CandidateWindow],
    calibration: object,
    *,
    xyz_bounds: tuple[float, float] = (0.0, 200.0),
    tail_residual_max_mm2: float = 10.0,
    tail_length_bounds_mm: tuple[float, float] = (10.0, 60.0),
) -> pd.DataFrame:
    """Flatten selected geometry and add tail vectors using identical slots."""
    records: list[dict[str, object]] = []
    start = next(iter(windows.values())).frame_start
    tail_pixels: list[np.ndarray] = []
    tail_record_indices: list[int] = []
    for frame in sorted(best_by_frame):
        info = best_by_frame[frame]
        groups = info.get("groups", {})
        errors = info.get("errors", {})
        permutations_by_view = [
            list(info.get(f"perm_cam{view_index}", []))
            for view_index in range(3)
        ]
        recovered = list(info.get("recovered", []))
        missing_views = list(
            info.get("missing_views", info.get("synthetic_views", []))
        )
        for source_index, source_label in enumerate(SOURCE_LABELS):
            xyz = np.asarray(groups.get(source_label, [np.nan] * 3), dtype=float)
            if not np.isfinite(xyz).all():
                continue
            slots = tuple(
                int(values[source_index]) if source_index < len(values) else -1
                for values in permutations_by_view
            )
            local_frame = int(frame - start)
            confidence_values: list[float] = []
            pixels = np.full((3, 2), np.nan, dtype=float)
            tail_vectors_2d = np.full((3, 2), np.nan, dtype=float)
            for view_index, (view, slot) in enumerate(zip(VIEW_ORDER, slots, strict=True)):
                if slot < 0:
                    continue
                window = windows[view]
                mouth_score = float(window.mouth_score[local_frame, slot])
                instance_score = float(window.instance_score[local_frame, slot])
                confidence = (
                    mouth_score if np.isfinite(mouth_score) else instance_score
                )
                if np.isfinite(confidence):
                    confidence_values.append(confidence)
                tail = window.tail_xy[local_frame, slot]
                if np.isfinite(tail).all():
                    pixels[view_index] = tail
                    mouth = window.mouth_xy[local_frame, slot]
                    if np.isfinite(mouth).all():
                        tail_vectors_2d[view_index] = tail - mouth
            record = {
                "observation_id": f"f{int(frame):06d}_s{source_index}",
                "frame": int(frame),
                "source_index": int(source_index),
                "X": float(xyz[0]),
                "Y": float(xyz[1]),
                "Z": float(xyz[2]),
                "ray_residual_mm2": float(errors.get(source_label, np.inf)),
                "residual_unit": RESIDUAL_UNIT,
                "slot_cam0": slots[0],
                "slot_cam1": slots[1],
                "slot_cam2": slots[2],
                "mouth_confidence": (
                    float(np.mean(confidence_values)) if confidence_values else np.nan
                ),
                "view_count": int(sum(slot >= 0 for slot in slots)),
                "recovered": bool(
                    recovered[source_index]
                    if source_index < len(recovered)
                    else any(slot < 0 for slot in slots)
                ),
                "missing_view": int(
                    missing_views[source_index]
                    if source_index < len(missing_views)
                    else next((i for i, slot in enumerate(slots) if slot < 0), -1)
                ),
                "tail_X": np.nan,
                "tail_Y": np.nan,
                "tail_Z": np.nan,
                "tail_ray_residual_mm2": np.nan,
                "tail_vector_X": np.nan,
                "tail_vector_Y": np.nan,
                "tail_vector_Z": np.nan,
                "tail_state": "tail_triangulation_pending",
            }
            for view_index in range(3):
                record[f"tail2d_cam{view_index}_x"] = float(tail_vectors_2d[view_index, 0])
                record[f"tail2d_cam{view_index}_y"] = float(tail_vectors_2d[view_index, 1])
            records.append(record)
            if (
                not record["recovered"]
                and np.isfinite(pixels).all(axis=1).sum() >= 2
            ):
                tail_pixels.append(pixels)
                tail_record_indices.append(len(records) - 1)
            elif record["recovered"]:
                record["tail_state"] = "recovered_mouth_anchor"
            else:
                record["tail_state"] = "fewer_than_two_tail_views"
    if tail_pixels:
        observations = np.asarray(tail_pixels, dtype=float)
        tail_xyz, tail_scores = calibration.evaluate_hypotheses(
            observations, profile="production_fast"
        )
        for record_index, xyz, score in zip(
            tail_record_indices,
            np.asarray(tail_xyz, dtype=float),
            np.asarray(tail_scores, dtype=float),
            strict=True,
        ):
            record = records[record_index]
            if not np.isfinite(xyz).all() or not np.isfinite(score):
                record["tail_state"] = "tail_triangulation_invalid"
                continue
            low, high = xyz_bounds
            if not np.all((xyz >= low) & (xyz <= high)):
                record["tail_state"] = "tail_outside_xyz_bounds"
                continue
            if float(score) > float(tail_residual_max_mm2):
                record["tail_state"] = "tail_residual_over_10_mm2"
                continue
            mouth = np.asarray([record["X"], record["Y"], record["Z"]], dtype=float)
            vector = xyz - mouth
            length = float(np.linalg.norm(vector))
            if not tail_length_bounds_mm[0] <= length <= tail_length_bounds_mm[1]:
                record["tail_state"] = "tail_length_out_of_range"
                continue
            record.update(
                {
                    "tail_X": float(xyz[0]),
                    "tail_Y": float(xyz[1]),
                    "tail_Z": float(xyz[2]),
                    "tail_ray_residual_mm2": float(score),
                    "tail_vector_X": float(vector[0]),
                    "tail_vector_Y": float(vector[1]),
                    "tail_vector_Z": float(vector[2]),
                    "tail_state": "valid_3d",
                }
            )
    return pd.DataFrame.from_records(records)


def hypotheses_to_observations(
    selected: pd.DataFrame,
    windows: Mapping[str, CandidateWindow],
    calibration: object,
    *,
    xyz_bounds: tuple[float, float] = (0.0, 200.0),
    tail_residual_max_mm2: float = 10.0,
    tail_length_bounds_mm: tuple[float, float] = (10.0, 60.0),
) -> pd.DataFrame:
    """Attach strict mouth confidence and validated same-slot tail evidence."""
    table = selected.sort_values(
        ["frame", "ray_residual_mm2", "hypothesis_id"]
    ).reset_index(drop=True).copy()
    table["source_index"] = table.groupby("frame", sort=True).cumcount().astype(np.int8)
    table["observation_id"] = [
        f"f{int(frame):06d}_s{int(source)}"
        for frame, source in zip(table.frame, table.source_index, strict=True)
    ]
    start = next(iter(windows.values())).frame_start
    mouth_confidence = np.full(len(table), np.nan, dtype=float)
    tail_pixels = np.full((len(table), 3, 2), np.nan, dtype=float)
    tail_2d = np.full((len(table), 3, 2), np.nan, dtype=float)
    tail_view_count = np.zeros(len(table), dtype=np.int8)
    for row_index, row in enumerate(table.itertuples(index=False)):
        local_frame = int(row.frame) - start
        confidence_values: list[float] = []
        for view_index, view in enumerate(VIEW_ORDER):
            slot = int(getattr(row, f"slot_cam{view_index}"))
            if slot < 0:
                continue
            window = windows[view]
            mouth = window.mouth_xy[local_frame, slot]
            tail = window.tail_xy[local_frame, slot]
            point_score = float(window.mouth_score[local_frame, slot])
            instance_score = float(window.instance_score[local_frame, slot])
            confidence = point_score if np.isfinite(point_score) else instance_score
            if np.isfinite(confidence):
                confidence_values.append(confidence)
            if np.isfinite(mouth).all() and np.isfinite(tail).all():
                tail_pixels[row_index, view_index] = tail
                tail_2d[row_index, view_index] = tail - mouth
                tail_view_count[row_index] += 1
        if confidence_values:
            mouth_confidence[row_index] = float(np.mean(confidence_values))
    table["mouth_confidence"] = mouth_confidence
    table["recovered"] = table.view_count.to_numpy(dtype=int) < 3
    table["missing_view"] = [
        next((view for view in range(3) if not (int(mask) & (1 << view))), -1)
        for mask in table.view_mask
    ]
    table["tail_view_count"] = tail_view_count
    for view_index in range(3):
        table[f"tail2d_cam{view_index}_x"] = tail_2d[:, view_index, 0]
        table[f"tail2d_cam{view_index}_y"] = tail_2d[:, view_index, 1]
    table["tail_X"] = np.nan
    table["tail_Y"] = np.nan
    table["tail_Z"] = np.nan
    table["tail_ray_residual_mm2"] = np.nan
    table["tail_vector_X"] = np.nan
    table["tail_vector_Y"] = np.nan
    table["tail_vector_Z"] = np.nan
    table["tail_state"] = np.where(
        tail_view_count >= 2, "tail_triangulation_pending", "fewer_than_two_tail_views"
    )
    evaluable = tail_view_count >= 2
    if np.any(evaluable):
        tail_xyz, residual = calibration.evaluate_hypotheses(
            tail_pixels[evaluable], profile="production_fast"
        )
        evaluated_indices = np.flatnonzero(evaluable)
        low, high = xyz_bounds
        length_low, length_high = tail_length_bounds_mm
        for row_index, xyz, score in zip(
            evaluated_indices,
            np.asarray(tail_xyz, dtype=float),
            np.asarray(residual, dtype=float),
            strict=True,
        ):
            if bool(table.at[row_index, "recovered"]):
                table.at[row_index, "tail_state"] = "recovered_mouth_anchor"
                continue
            if not np.isfinite(xyz).all() or not np.isfinite(score):
                table.at[row_index, "tail_state"] = "tail_triangulation_invalid"
                continue
            if not np.all((xyz >= low) & (xyz <= high)):
                table.at[row_index, "tail_state"] = "tail_outside_xyz_bounds"
                continue
            if float(score) > float(tail_residual_max_mm2):
                table.at[row_index, "tail_state"] = "tail_residual_over_10_mm2"
                continue
            mouth = table.loc[row_index, ["X", "Y", "Z"]].to_numpy(dtype=float)
            vector = xyz - mouth
            length = float(np.linalg.norm(vector))
            if length < length_low or length > length_high:
                table.at[row_index, "tail_state"] = "tail_length_out_of_range"
                continue
            table.loc[row_index, ["tail_X", "tail_Y", "tail_Z"]] = xyz
            table.at[row_index, "tail_ray_residual_mm2"] = float(score)
            table.loc[
                row_index, ["tail_vector_X", "tail_vector_Y", "tail_vector_Z"]
            ] = vector
            table.at[row_index, "tail_state"] = "valid_3d"
    return table


def build_legacy_tracklet_ids(
    window: CandidateWindow,
    *,
    num_fish: int = 3,
    sep_body_lengths: float = 0.5,
    jump_body_lengths: float = 1.0,
    min_score: float = 0.3,
    max_gap: int = 0,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Dense-array equivalent of the conservative top-K tracklet builder."""
    from scripts.tracking.concrete_tracklets import build_concrete_tracklets

    body_lengths = np.linalg.norm(window.tail_xy - window.mouth_xy, axis=2)
    finite_lengths = body_lengths[np.isfinite(body_lengths) & (body_lengths > 0)]
    body_length = float(np.median(finite_lengths)) if finite_lengths.size else 1.0
    frame_data: dict[int, list[tuple[float, float, float]]] = {}
    for local, frame in enumerate(range(window.frame_start, window.frame_end + 1)):
        detections: list[tuple[float, float, float]] = []
        for slot in range(min(num_fish, window.mouth_xy.shape[1])):
            point = window.mouth_xy[local, slot]
            if not np.isfinite(point).all():
                continue
            score = float(window.instance_score[local, slot])
            if not np.isfinite(score):
                score = 0.0
            detections.append((float(point[0]), float(point[1]), score))
        frame_data[frame] = detections
    tracklets, stats = build_concrete_tracklets(
        frame_data,
        num_fish=num_fish,
        body_length=body_length,
        sep_bl=sep_body_lengths,
        jump_bl=jump_body_lengths,
        min_score=min_score,
        max_gap=max_gap,
    )
    output = np.full(window.mouth_xy.shape[:2], -1, dtype=np.int32)
    records: list[dict[str, object]] = []
    for tracklet in tracklets:
        tracklet_id = int(tracklet["id"])
        for frame, x, y, score in tracklet["points"]:
            local = int(frame - window.frame_start)
            coordinates = window.mouth_xy[local]
            distances = np.linalg.norm(coordinates - np.array([x, y]), axis=1)
            distances[~np.isfinite(distances)] = np.inf
            slot = int(np.argmin(distances))
            if not np.isfinite(distances[slot]) or distances[slot] > 1e-8:
                raise RuntimeError(
                    f"tracklet observation failed exact slot join at frame {frame}"
                )
            if output[local, slot] >= 0:
                raise RuntimeError("one detection was assigned to two tracklets")
            output[local, slot] = tracklet_id
            records.append(
                {
                    "tracklet_id": tracklet_id,
                    "frame": int(frame),
                    "candidate_slot": slot,
                    "x": float(x),
                    "y": float(y),
                    "score": float(score),
                }
            )
    stats = {**stats, "anchor": "strict_mouth", "body_length_px": body_length}
    return output, pd.DataFrame.from_records(records), stats


def build_candidate_tracklet_ids(
    window: CandidateWindow,
    *,
    sep_body_lengths: float = 0.5,
    jump_body_lengths: float = 1.0,
    min_score: float = 0.3,
    max_gap: int = 0,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Build conservative provenance tracklets for all finite candidate slots.

    Unlike the historical top-three comparator, ambiguity is handled per
    detection so an unrelated fourth--sixth slot can retain provenance.  A
    candidate closer than the separation gate to any rival is neither extended
    nor seeded; it becomes a unique one-frame resource in Item 2 instead of
    being conflated with a synthetic missing view.
    """
    body_lengths = np.linalg.norm(window.tail_xy - window.mouth_xy, axis=2)
    finite_lengths = body_lengths[np.isfinite(body_lengths) & (body_lengths > 0)]
    body_length = float(np.median(finite_lengths)) if finite_lengths.size else 1.0
    separation_gate = float(sep_body_lengths) * body_length
    jump_gate = float(jump_body_lengths) * body_length
    ids = np.full(window.mouth_xy.shape[:2], -1, dtype=np.int32)
    active: dict[int, dict[str, object]] = {}
    next_id = 0
    observations: list[dict[str, object]] = []
    for local, frame in enumerate(range(window.frame_start, window.frame_end + 1)):
        points_all = window.mouth_xy[local]
        scores_all = window.instance_score[local]
        valid_slots = np.flatnonzero(
            np.all(np.isfinite(points_all), axis=1)
            & np.isfinite(scores_all)
            & (scores_all >= float(min_score))
        )
        points = points_all[valid_slots]
        unambiguous = np.ones(len(valid_slots), dtype=bool)
        if len(points) >= 2:
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            unambiguous = np.min(distances, axis=1) >= separation_gate

        active_ids = sorted(active)
        matched_active: set[int] = set()
        matched_local: set[int] = set()
        if active_ids and len(valid_slots):
            predictions = []
            for tracklet_id in active_ids:
                state = active[tracklet_id]
                gap = int(frame - int(state["frame"]))
                velocity = np.asarray(state["velocity"], dtype=float)
                predictions.append(np.asarray(state["point"], dtype=float) + velocity * gap)
            cost = np.linalg.norm(np.asarray(predictions)[:, None] - points[None, :], axis=2)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns, strict=True):
                tracklet_id = active_ids[int(row)]
                state = active[tracklet_id]
                gap = int(frame - int(state["frame"]))
                if (
                    gap > max_gap + 1
                    or not unambiguous[int(column)]
                    or cost[int(row), int(column)] > jump_gate * max(1, gap)
                ):
                    continue
                slot = int(valid_slots[int(column)])
                previous = np.asarray(state["point"], dtype=float)
                point = points[int(column)]
                state["velocity"] = (point - previous) / max(1, gap)
                state["point"] = point.copy()
                state["frame"] = int(frame)
                ids[local, slot] = tracklet_id
                matched_active.add(tracklet_id)
                matched_local.add(int(column))
        active = {
            tracklet_id: state
            for tracklet_id, state in active.items()
            if tracklet_id in matched_active
            or int(frame - int(state["frame"])) <= max_gap
        }
        for local_candidate, slot in enumerate(valid_slots):
            if local_candidate in matched_local or not unambiguous[local_candidate]:
                continue
            tracklet_id = next_id
            next_id += 1
            point = points[local_candidate]
            active[tracklet_id] = {
                "point": point.copy(),
                "frame": int(frame),
                "velocity": np.zeros(2, dtype=float),
            }
            ids[local, int(slot)] = tracklet_id
        for slot in valid_slots:
            tracklet_id = int(ids[local, int(slot)])
            if tracklet_id < 0:
                continue
            point = points_all[int(slot)]
            observations.append(
                {
                    "tracklet_id": tracklet_id,
                    "frame": int(frame),
                    "candidate_slot": int(slot),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "score": float(scores_all[int(slot)]),
                }
            )
    table = pd.DataFrame.from_records(observations)
    lengths = (
        table.groupby("tracklet_id").size().to_numpy(dtype=int)
        if len(table)
        else np.empty(0, dtype=int)
    )
    summary = {
        "mode": "all_candidate_slots_conservative",
        "anchor": "strict_mouth",
        "body_length_px": body_length,
        "separation_gate_px": separation_gate,
        "jump_gate_px": jump_gate,
        "min_score": float(min_score),
        "max_gap": int(max_gap),
        "num_candidate_slots": int(window.mouth_xy.shape[1]),
        "n_fragments": int(len(lengths)),
        "covered_candidate_rows": int(len(table)),
        "fragment_length_median": float(np.median(lengths)) if len(lengths) else 0.0,
    }
    return ids, table, summary


def enumerate_refractive_hypotheses(
    windows: Mapping[str, CandidateWindow],
    tracklet_ids: Mapping[str, np.ndarray],
    calibration: object,
    *,
    xyz_bounds: tuple[float, float] = (0.0, 200.0),
    batch_frames: int = 256,
    include_two_view: bool = True,
) -> pd.DataFrame:
    """Freeze all finite in-bounds hypotheses before either Item 2 mode."""
    starts = {window.frame_start for window in windows.values()}
    ends = {window.frame_end for window in windows.values()}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("candidate windows are not aligned")
    frame_start, frame_end = starts.pop(), ends.pop()
    candidate_count = next(iter(windows.values())).mouth_xy.shape[1]
    low, high = map(float, xyz_bounds)
    records: list[pd.DataFrame] = []
    full_indices = np.stack(
        np.meshgrid(
            np.arange(candidate_count),
            np.arange(candidate_count),
            np.arange(candidate_count),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    pair_indices = np.stack(
        np.meshgrid(np.arange(candidate_count), np.arange(candidate_count), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)

    def evaluate_block(
        block_start: int,
        block_end: int,
        slots: np.ndarray,
        view_mask: int,
    ) -> None:
        count = block_end - block_start
        hypotheses = len(slots)
        pixels = np.full((count, hypotheses, 3, 2), np.nan, dtype=float)
        for view_index, view in enumerate(VIEW_ORDER):
            selected = slots[:, view_index]
            present = selected >= 0
            if np.any(present):
                pixels[:, present, view_index] = windows[view].mouth_xy[
                    block_start:block_end, selected[present]
                ]
        finite_views = np.all(np.isfinite(pixels), axis=3)
        finite = np.sum(finite_views, axis=2) >= 2
        flat_finite = finite.ravel()
        if not np.any(flat_finite):
            return
        flat_pixels = pixels.reshape(-1, 3, 2)
        xyz = np.full((count * hypotheses, 3), np.nan, dtype=float)
        residual = np.full(count * hypotheses, np.inf, dtype=float)
        evaluated_xyz, evaluated_residual = calibration.evaluate_hypotheses(
            flat_pixels[flat_finite], profile="production_fast"
        )
        xyz[flat_finite] = np.asarray(evaluated_xyz, dtype=float)
        residual[flat_finite] = np.asarray(evaluated_residual, dtype=float)
        valid = (
            flat_finite
            & np.all(np.isfinite(xyz), axis=1)
            & np.isfinite(residual)
            & np.all((xyz >= low) & (xyz <= high), axis=1)
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            return
        local_frames = indices // hypotheses
        hypothesis_indices = indices % hypotheses
        selected_slots = slots[hypothesis_indices]
        absolute_frames = frame_start + block_start + local_frames
        data: dict[str, object] = {
            "frame": absolute_frames.astype(np.int32),
            "view_mask": np.full(len(indices), view_mask, dtype=np.uint8),
            "X": xyz[indices, 0],
            "Y": xyz[indices, 1],
            "Z": xyz[indices, 2],
            "ray_residual_mm2": residual[indices],
            "residual_unit": np.repeat(RESIDUAL_UNIT, len(indices)),
        }
        for view_index, view in enumerate(VIEW_ORDER):
            slot_values = selected_slots[:, view_index].astype(np.int16)
            data[f"slot_cam{view_index}"] = slot_values
            ids = np.full(len(indices), -1, dtype=np.int64)
            observed = slot_values >= 0
            ids[observed] = tracklet_ids[view][
                local_frames[observed], slot_values[observed]
            ]
            # An observed but ambiguous/untracked candidate is a unique
            # one-frame tracklet, never the missing-view sentinel.
            ids[observed & (ids < 0)] = -(
                absolute_frames[observed & (ids < 0)].astype(np.int64)
                * candidate_count
                + slot_values[observed & (ids < 0)].astype(np.int64)
                + 2
            )
            data[f"tracklet_{view}"] = ids
        frame_table = pd.DataFrame(data)
        frame_table["hypothesis_id"] = [
            f"f{frame:06d}_m{view_mask}_s{s0}_{s1}_{s2}"
            for frame, s0, s1, s2 in zip(
                frame_table.frame,
                frame_table.slot_cam0,
                frame_table.slot_cam1,
                frame_table.slot_cam2,
                strict=True,
            )
        ]
        frame_table["view_count"] = int(view_mask.bit_count())
        records.append(frame_table)

    total_frames = frame_end - frame_start + 1
    for block_start in range(0, total_frames, batch_frames):
        block_end = min(total_frames, block_start + batch_frames)
        evaluate_block(block_start, block_end, full_indices, 0b111)
        if include_two_view:
            for missing in range(3):
                slots = np.full((len(pair_indices), 3), -1, dtype=int)
                present = [view for view in range(3) if view != missing]
                slots[:, present[0]] = pair_indices[:, 0]
                slots[:, present[1]] = pair_indices[:, 1]
                evaluate_block(block_start, block_end, slots, 0b111 ^ (1 << missing))
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True).sort_values(
        ["frame", "view_count", "ray_residual_mm2", "hypothesis_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _select_framewise_hypotheses(hypotheses: pd.DataFrame) -> pd.DataFrame:
    """Exact cardinality/view-count/residual selection within each frame."""
    from scripts.reconstruction_3d.tracklet_first_assignment import (
        ReconstructionCandidate,
        select_global_candidate_set,
    )

    selected_ids: list[str] = []
    for _frame, group in hypotheses.groupby("frame", sort=True):
        finite_residuals = group.ray_residual_mm2.to_numpy(dtype=float)
        residual_span = float(np.sum(np.abs(finite_residuals)) + 1.0)
        candidates = []
        by_id = {}
        for row in group.itertuples(index=False):
            # Cardinality is handled lexicographically by the selector.  This
            # penalty makes total observed views the second exact priority.
            cost = float(row.ray_residual_mm2) + (3 - int(row.view_count)) * residual_span
            candidate = ReconstructionCandidate(
                slots=(int(row.slot_cam0), int(row.slot_cam1), int(row.slot_cam2)),
                xyz=(float(row.X), float(row.Y), float(row.Z)),
                residual=cost,
                metadata={"hypothesis_id": str(row.hypothesis_id)},
            )
            candidates.append(candidate)
            by_id[id(candidate)] = str(row.hypothesis_id)
        selection = select_global_candidate_set(candidates, max_selected=3, max_states=None)
        selected_ids.extend(
            str(candidate.metadata["hypothesis_id"])
            for candidate in selection.selected
        )
    return hypotheses[hypotheses.hypothesis_id.isin(selected_ids)].copy()


def group_cross_view_tracklets(
    hypotheses: pd.DataFrame,
    *,
    mode: str,
    min_support_frames: int = 3,
    min_valid_fraction: float = 0.95,
    max_ray_residual_mm2: float = np.inf,
    tracklet_intervals: Mapping[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select framewise or robust multiframe groups from one frozen table.

    Returns ``(selected_observations, candidate_groups, conflicts,
    discarded_duplicates)``.  Multiframe selection is a three-stage binary
    MILP implementing the frozen lexicographic objective.
    """
    if mode == "framewise":
        selected = _select_framewise_hypotheses(hypotheses)
        selected["cross_view_group_id"] = "fw_" + selected.hypothesis_id.astype(str)
        selected["cross_view_group_support"] = 1
        selected["cross_view_group_trusted"] = False
        candidates = selected[["cross_view_group_id", "frame", "view_count", "ray_residual_mm2"]].copy()
        candidates["n_overlap"] = 1
        candidates["n_support"] = 1
        candidates["valid_fraction"] = 1.0
        candidates["residual_median_mm2"] = candidates.ray_residual_mm2
        candidates["residual_mad_mm2"] = 0.0
        candidates["residual_robust_upper_mm2"] = candidates.ray_residual_mm2
        candidates["eligible"] = True
        candidates["selected"] = True
        return selected, candidates, pd.DataFrame(), pd.DataFrame()
    if mode != "multiframe":
        raise ValueError("mode must be framewise or multiframe")
    key_columns = ["view_mask", *[f"tracklet_{view}" for view in VIEW_ORDER]]
    ordered = hypotheses.sort_values(
        [*key_columns, "frame", "ray_residual_mm2", "hypothesis_id"]
    )
    duplicate_mask = ordered.duplicated([*key_columns, "frame"], keep="first")
    duplicates = ordered[duplicate_mask].copy()
    unique = ordered[~duplicate_mask].copy()
    interval_by_tracklet: dict[tuple[int, int], tuple[int, int]] = dict(
        tracklet_intervals or {}
    )
    for view_index, view in enumerate(VIEW_ORDER):
        for tracklet_id, group in unique[unique[f"slot_cam{view_index}"] >= 0].groupby(
            f"tracklet_{view}", sort=False
        ):
            key = (view_index, int(tracklet_id))
            if key not in interval_by_tracklet:
                # Negative IDs are deliberately unique one-frame provenance
                # for observed but ambiguous detections.
                interval_by_tracklet[key] = (
                    int(group.frame.min()), int(group.frame.max())
                )
    group_records: list[dict[str, object]] = []
    support_rows: dict[str, np.ndarray] = {}
    for key, group in unique.groupby(key_columns, sort=True):
        view_mask = int(key[0])
        tracklet_key = tuple(map(int, key[1:]))
        intervals = [
            interval_by_tracklet[(view_index, tracklet_key[view_index])]
            for view_index in range(3)
            if view_mask & (1 << view_index)
        ]
        overlap_start = max(interval[0] for interval in intervals)
        overlap_end = min(interval[1] for interval in intervals)
        n_overlap = max(0, overlap_end - overlap_start + 1)
        frames = group.frame.to_numpy(dtype=int)
        n_support = int(np.unique(frames).size)
        valid_fraction = n_support / max(n_overlap, 1)
        residuals = group.ray_residual_mm2.to_numpy(dtype=float)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_upper = float(median + 3.0 * 1.4826 * mad)
        eligible = bool(
            n_support >= min_support_frames
            and valid_fraction >= min_valid_fraction
            and robust_upper <= max_ray_residual_mm2
        )
        resource_rows = [
            (
                int(row.frame),
                int(row.slot_cam0),
                int(row.slot_cam1),
                int(row.slot_cam2),
            )
            for row in group.itertuples(index=False)
        ]
        group_id = "mv_" + stable_hash(
            (view_mask, tracklet_key, sorted(map(int, frames)), resource_rows)
        )[:20]
        support_rows[group_id] = group.index.to_numpy(dtype=int)
        group_records.append(
            {
                "cross_view_group_id": group_id,
                "view_mask": view_mask,
                "tracklet_top_cam": tracklet_key[0],
                "tracklet_side1_cam": tracklet_key[1],
                "tracklet_side2_cam": tracklet_key[2],
                "frame_start": int(frames.min()),
                "frame_end": int(frames.max()),
                "n_overlap": n_overlap,
                "n_support": n_support,
                "valid_fraction": valid_fraction,
                "residual_median_mm2": median,
                "residual_mad_mm2": mad,
                "residual_robust_upper_mm2": robust_upper,
                "residual_gate_mm2": float(max_ray_residual_mm2),
                "residual_unit": RESIDUAL_UNIT,
                "view_count": int(view_mask.bit_count()),
                "eligible": eligible,
                "rejection_reason": (
                    "eligible"
                    if eligible
                    else "support_below_minimum"
                    if n_support < min_support_frames
                    else "valid_fraction_below_minimum"
                    if valid_fraction < min_valid_fraction
                    else "robust_residual_above_gate"
                ),
            }
        )
    candidates = pd.DataFrame.from_records(group_records)
    eligible = candidates[candidates.eligible].reset_index(drop=True)
    if eligible.empty:
        candidates["selected"] = False
        return hypotheses.iloc[0:0].copy(), candidates, pd.DataFrame(), duplicates

    resource_to_groups: dict[tuple[int, int, int], list[int]] = {}
    frame_to_groups: dict[int, list[int]] = {}
    conflicts: list[dict[str, object]] = []
    for group_index, row in eligible.iterrows():
        group_id = str(row.cross_view_group_id)
        rows = hypotheses.loc[support_rows[group_id]]
        for support in rows.itertuples(index=False):
            frame_to_groups.setdefault(int(support.frame), []).append(group_index)
            for view_index in range(3):
                slot = int(getattr(support, f"slot_cam{view_index}"))
                if slot >= 0:
                    resource_to_groups.setdefault(
                        (int(support.frame), view_index, slot), []
                    ).append(group_index)
    constraint_rows: list[tuple[list[int], float]] = []
    for resource, indices in sorted(resource_to_groups.items()):
        unique_indices = sorted(set(indices))
        if len(unique_indices) > 1:
            for left_index in unique_indices:
                for right_index in unique_indices:
                    if left_index < right_index:
                        conflicts.append(
                            {
                                "resource": str(resource),
                                "left_group_id": eligible.iloc[left_index].cross_view_group_id,
                                "right_group_id": eligible.iloc[right_index].cross_view_group_id,
                            }
                        )
        constraint_rows.append((unique_indices, 1.0))
    for _frame, indices in sorted(frame_to_groups.items()):
        constraint_rows.append((sorted(set(indices)), 3.0))

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, (indices, _upper) in enumerate(constraint_rows):
        for index in indices:
            row_indices.append(row_index)
            column_indices.append(index)
            values.append(1.0)
    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_rows), len(eligible)),
    ).tocsc()
    lower = np.full(len(constraint_rows), -np.inf)
    upper = np.asarray([upper for _indices, upper in constraint_rows], dtype=float)
    base_constraint = LinearConstraint(matrix, lower, upper)
    variable_bounds = Bounds(np.zeros(len(eligible)), np.ones(len(eligible)))
    integrality = np.ones(len(eligible), dtype=np.int8)
    support = eligible.n_support.to_numpy(dtype=float)
    views = support * eligible.view_count.to_numpy(dtype=float)
    residual_cost = support * eligible.residual_robust_upper_mm2.to_numpy(dtype=float)

    def solve(objective: np.ndarray, extra: Sequence[LinearConstraint] = ()):
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=variable_bounds,
            constraints=(base_constraint, *extra),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"multiframe set packing failed: {result.message}")
        return result

    first = solve(-support)
    best_support = float(np.rint(np.dot(support, first.x > 0.5)))
    support_constraint = LinearConstraint(support[None, :], best_support, best_support)
    second = solve(-views, (support_constraint,))
    best_views = float(np.rint(np.dot(views, second.x > 0.5)))
    view_constraint = LinearConstraint(views[None, :], best_views, best_views)
    rank_epsilon = np.arange(len(eligible), dtype=float) * 1e-12
    third = solve(residual_cost + rank_epsilon, (support_constraint, view_constraint))
    selected_group_ids = set(
        eligible.loc[third.x > 0.5, "cross_view_group_id"].astype(str)
    )
    candidates["selected"] = candidates.cross_view_group_id.astype(str).isin(selected_group_ids)
    selected_parts = []
    for group_id in sorted(selected_group_ids):
        rows = hypotheses.loc[support_rows[group_id]].copy()
        rows["cross_view_group_id"] = group_id
        rows["cross_view_group_support"] = int(
            candidates.loc[
                candidates.cross_view_group_id == group_id, "n_support"
            ].iloc[0]
        )
        rows["cross_view_group_trusted"] = True
        selected_parts.append(rows)
    selected = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["frame", "cross_view_group_id", "hypothesis_id"]
    )
    return selected, candidates, pd.DataFrame.from_records(conflicts), duplicates


def annotate_cross_view_groups(
    observations: pd.DataFrame,
    tracklet_ids: Mapping[str, np.ndarray],
    *,
    cache_frame_start: int,
    mode: str,
    min_support_frames: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach framewise or robust multiframe cross-view group evidence."""
    if mode not in {"framewise", "multiframe"}:
        raise ValueError("mode must be 'framewise' or 'multiframe'")
    table = observations.copy()
    frame_indices = table["frame"].to_numpy(dtype=np.int64) - int(cache_frame_start)
    for view_index, view in enumerate(VIEW_ORDER):
        slots = table[f"slot_cam{view_index}"].to_numpy(dtype=np.int64)
        values = np.full(len(table), -1, dtype=np.int64)
        valid = (slots >= 0) & (frame_indices >= 0) & (
            frame_indices < tracklet_ids[view].shape[0]
        )
        values[valid] = tracklet_ids[view][frame_indices[valid], slots[valid]]
        table[f"tracklet_{view}"] = values
    if mode == "framewise":
        table["cross_view_group_id"] = "fw_" + table["observation_id"].astype(str)
        table["cross_view_group_support"] = 1
        table["cross_view_group_trusted"] = False
        groups = table[
            ["cross_view_group_id", "frame", "ray_residual_mm2"]
        ].rename(columns={"frame": "frame_start"})
        groups["frame_end"] = groups["frame_start"]
        groups["support_frames"] = 1
        groups["residual_median_mm2"] = groups["ray_residual_mm2"]
        groups["residual_mad_mm2"] = 0.0
        groups["trusted"] = False
        return table, groups.drop(columns=["ray_residual_mm2"])

    key_columns = [f"tracklet_{view}" for view in VIEW_ORDER]
    key_values = table[key_columns].to_numpy(dtype=np.int64)
    present_views = (key_values >= 0).sum(axis=1)
    table["_key"] = [tuple(map(int, row)) for row in key_values]
    eligible = table[present_views >= 2]
    records: list[dict[str, object]] = []
    key_to_group: dict[tuple[int, int, int], str] = {}
    for key, group in eligible.groupby("_key", sort=True):
        frames = group["frame"].to_numpy(dtype=np.int64)
        residuals = group["ray_residual_mm2"].to_numpy(dtype=float)
        finite = residuals[np.isfinite(residuals)]
        median = float(np.median(finite)) if finite.size else np.inf
        mad = (
            float(np.median(np.abs(finite - median))) if finite.size else np.inf
        )
        support = int(np.unique(frames).size)
        group_id = "mv_" + stable_hash(key)[:16]
        trusted = support >= int(min_support_frames) and np.isfinite(median)
        records.append(
            {
                "cross_view_group_id": group_id,
                "tracklet_top_cam": int(key[0]),
                "tracklet_side1_cam": int(key[1]),
                "tracklet_side2_cam": int(key[2]),
                "frame_start": int(frames.min()),
                "frame_end": int(frames.max()),
                "support_frames": support,
                "residual_median_mm2": median,
                "residual_mad_mm2": mad,
                "residual_unit": RESIDUAL_UNIT,
                "trusted": bool(trusted),
            }
        )
        if trusted:
            key_to_group[key] = group_id
    group_ids = []
    supports = []
    trusted_flags = []
    support_by_key = {
        tuple(
            int(record[column])
            for column in (
                "tracklet_top_cam",
                "tracklet_side1_cam",
                "tracklet_side2_cam",
            )
        ): int(record["support_frames"])
        for record in records
    }
    for observation_id, key in zip(table["observation_id"], table["_key"], strict=True):
        trusted_id = key_to_group.get(key)
        group_ids.append(trusted_id or f"untrusted_{observation_id}")
        supports.append(support_by_key.get(key, 1))
        trusted_flags.append(trusted_id is not None)
    table["cross_view_group_id"] = group_ids
    table["cross_view_group_support"] = supports
    table["cross_view_group_trusted"] = trusted_flags
    table = table.drop(columns=["_key"])
    return table, pd.DataFrame.from_records(records)


def _pairwise_minimum(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.inf
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def build_fragments(
    observations: pd.DataFrame,
    *,
    max_jump_mm: float = 8.0,
    close_contact_mm: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build conservative 3D fragments, breaking at mouth ambiguity."""
    table = observations.sort_values(["frame", "source_index"]).reset_index(drop=True).copy()
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    frames = table["frame"].to_numpy(dtype=np.int64)
    group_ids = table["cross_view_group_id"].astype(str).to_numpy()
    trusted = table["cross_view_group_trusted"].to_numpy(dtype=bool)
    assigned = np.full(len(table), -1, dtype=np.int64)
    active: dict[int, int] = {}
    next_fragment = 0
    unique_frames, starts = np.unique(frames, return_index=True)
    ends = np.r_[starts[1:], len(table)]
    previous_close = False
    for frame, row_start, row_end in zip(unique_frames, starts, ends, strict=True):
        indices = np.arange(row_start, row_end, dtype=np.int64)
        current_close = _pairwise_minimum(xyz[indices]) < float(close_contact_mm)
        # A close-contact boundary is deliberately left for the fragment graph.
        if current_close or previous_close:
            active = {}
        usable_active = [
            (fragment_id, last_row)
            for fragment_id, last_row in sorted(active.items())
            if int(frame - frames[last_row]) == 1
        ]
        matched_rows: set[int] = set()
        if usable_active and len(indices):
            cost = np.full((len(usable_active), len(indices)), 1e9, dtype=float)
            physical_cost = np.full_like(cost, 1e9)
            for active_index, (_fragment_id, last_row) in enumerate(usable_active):
                distances = np.linalg.norm(xyz[indices] - xyz[last_row], axis=1)
                physical_cost[active_index] = distances
                same_group = (
                    trusted[last_row]
                    & trusted[indices]
                    & (group_ids[indices] == group_ids[last_row])
                )
                distances[same_group] *= 0.05
                distances[physical_cost[active_index] > float(max_jump_mm)] = 1e9
                cost[active_index] = distances
            rows, cols = linear_sum_assignment(cost)
            for active_index, current_index in zip(rows, cols, strict=True):
                if physical_cost[active_index, current_index] > float(max_jump_mm):
                    continue
                fragment_id = usable_active[int(active_index)][0]
                row = int(indices[int(current_index)])
                assigned[row] = fragment_id
                active[fragment_id] = row
                matched_rows.add(row)
        matched_fragments = set(assigned[list(matched_rows)]) if matched_rows else set()
        active = {
            fragment_id: last_row
            for fragment_id, last_row in active.items()
            if fragment_id in matched_fragments
        }
        for row in indices:
            if int(row) in matched_rows:
                continue
            fragment_id = next_fragment
            next_fragment += 1
            assigned[int(row)] = fragment_id
            active[fragment_id] = int(row)
        previous_close = current_close
    table["fragment_id"] = [f"frag_{value:07d}" for value in assigned]
    fragments = summarize_fragments(table)
    return table, fragments


def _endpoint_indices(frames: np.ndarray, *, from_end: bool) -> np.ndarray:
    order = np.argsort(frames)
    endpoint = int(frames[order[-1] if from_end else order[0]])
    if from_end:
        order = order[frames[order] >= endpoint - 29][-15:]
    else:
        order = order[frames[order] <= endpoint + 29][:15]
    return order


def _robust_velocity(frames: np.ndarray, xyz: np.ndarray, *, from_end: bool) -> np.ndarray:
    order = _endpoint_indices(frames, from_end=from_end)
    selected_frames = frames[order]
    selected_xyz = xyz[order]
    if len(selected_frames) < 3 or int(selected_frames.max() - selected_frames.min()) < 2:
        return np.full(3, np.nan)
    slopes: list[np.ndarray] = []
    for left in range(len(selected_frames) - 1):
        for right in range(left + 1, len(selected_frames)):
            dt = int(selected_frames[right] - selected_frames[left])
            if dt <= 0:
                continue
            difference = selected_xyz[right] - selected_xyz[left]
            if np.all(np.isfinite(difference)):
                slopes.append(difference / dt)
    if not slopes:
        return np.full(3, np.nan)
    return np.median(np.asarray(slopes), axis=0)


def _direction_summary(vectors: np.ndarray) -> tuple[np.ndarray, int, float]:
    valid = np.all(np.isfinite(vectors), axis=1)
    vectors = vectors[valid]
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-12
    if np.sum(keep) < 3:
        return np.full(vectors.shape[1] if vectors.ndim == 2 else 3, np.nan), 0, np.nan
    unit = vectors[keep] / lengths[keep, None]
    median = np.median(unit, axis=0)
    norm = float(np.linalg.norm(median))
    if norm < 0.5:
        return np.full(unit.shape[1], np.nan), int(len(unit)), norm
    return median / norm, int(len(unit)), norm


def summarize_fragments(observations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fragment_id, group in observations.groupby("fragment_id", sort=True):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy(dtype=np.int64)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        tail = group[["tail_vector_X", "tail_vector_Y", "tail_vector_Z"]].to_numpy(dtype=float)
        start_velocity = _robust_velocity(frames, xyz, from_end=False)
        end_velocity = _robust_velocity(frames, xyz, from_end=True)
        start_indices = _endpoint_indices(frames, from_end=False)
        end_indices = _endpoint_indices(frames, from_end=True)
        start_tail, start_tail_count, start_tail_norm = _direction_summary(tail[start_indices])
        end_tail, end_tail_count, end_tail_norm = _direction_summary(tail[end_indices])
        trusted_groups = sorted(
            set(
                group.loc[
                    group["cross_view_group_trusted"], "cross_view_group_id"
                ].astype(str)
            )
        )
        record = {
            "fragment_id": str(fragment_id),
            "frame_start": int(frames.min()),
            "frame_end": int(frames.max()),
            "observed_frames": int(np.unique(frames).size),
            "observation_count": int(len(group)),
            "start_X": float(xyz[0, 0]),
            "start_Y": float(xyz[0, 1]),
            "start_Z": float(xyz[0, 2]),
            "end_X": float(xyz[-1, 0]),
            "end_Y": float(xyz[-1, 1]),
            "end_Z": float(xyz[-1, 2]),
            "start_vX": float(start_velocity[0]),
            "start_vY": float(start_velocity[1]),
            "start_vZ": float(start_velocity[2]),
            "end_vX": float(end_velocity[0]),
            "end_vY": float(end_velocity[1]),
            "end_vZ": float(end_velocity[2]),
            "start_residual_median_mm2": float(np.nanmedian(group.iloc[start_indices]["ray_residual_mm2"])),
            "end_residual_median_mm2": float(np.nanmedian(group.iloc[end_indices]["ray_residual_mm2"])),
            "start_mouth_confidence_median": float(np.nanmedian(group.iloc[start_indices]["mouth_confidence"])),
            "end_mouth_confidence_median": float(np.nanmedian(group.iloc[end_indices]["mouth_confidence"])),
            "residual_median_mm2": float(np.nanmedian(group["ray_residual_mm2"].to_numpy(dtype=float))),
            "mouth_confidence_median": float(np.nanmedian(group["mouth_confidence"].to_numpy(dtype=float))),
            "trusted_group_ids": "|".join(trusted_groups),
            "start_tail3d_X": float(start_tail[0]),
            "start_tail3d_Y": float(start_tail[1]),
            "start_tail3d_Z": float(start_tail[2]),
            "start_tail3d_samples": start_tail_count,
            "start_tail3d_pre_norm": start_tail_norm,
            "end_tail3d_X": float(end_tail[0]),
            "end_tail3d_Y": float(end_tail[1]),
            "end_tail3d_Z": float(end_tail[2]),
            "end_tail3d_samples": end_tail_count,
            "end_tail3d_pre_norm": end_tail_norm,
            "tail_available_frames": int(np.all(np.isfinite(tail), axis=1).sum()),
            "tail_vector_X": float(end_tail[0]),
            "tail_vector_Y": float(end_tail[1]),
            "tail_vector_Z": float(end_tail[2]),
            "overlap_agree_frames": int(
                group["overlap_agree_frames"].max()
                if "overlap_agree_frames" in group
                else group.get("overlap_agree", pd.Series(False, index=group.index)).sum()
            ),
            "overlap_disagree_frames": int(
                group["overlap_disagree_frames"].max()
                if "overlap_disagree_frames" in group
                else group.get("overlap_disagree", pd.Series(False, index=group.index)).sum()
            ),
        }
        for view_index in range(3):
            columns = [f"tail2d_cam{view_index}_x", f"tail2d_cam{view_index}_y"]
            if all(column in group for column in columns):
                vectors_2d = group[columns].to_numpy(dtype=float)
                start_2d, start_count, start_norm = _direction_summary(vectors_2d[start_indices])
                end_2d, end_count, end_norm = _direction_summary(vectors_2d[end_indices])
            else:
                start_2d = end_2d = np.full(2, np.nan)
                start_count = end_count = 0
                start_norm = end_norm = np.nan
            record.update(
                {
                    f"start_tail2d_cam{view_index}_x": float(start_2d[0]),
                    f"start_tail2d_cam{view_index}_y": float(start_2d[1]),
                    f"start_tail2d_cam{view_index}_samples": start_count,
                    f"start_tail2d_cam{view_index}_pre_norm": start_norm,
                    f"end_tail2d_cam{view_index}_x": float(end_2d[0]),
                    f"end_tail2d_cam{view_index}_y": float(end_2d[1]),
                    f"end_tail2d_cam{view_index}_samples": end_count,
                    f"end_tail2d_cam{view_index}_pre_norm": end_norm,
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_fragments_compact(observations: pd.DataFrame) -> pd.DataFrame:
    """Vectorized fragment inventory for runs that do not build link edges.

    Link-specific robust endpoint velocities and tail direction summaries are
    intentionally omitted.  The inventory preserves the identifiers, temporal
    extent, counts, endpoint coordinates, and median QC values needed for
    inspection and immutable-graph provenance.
    """
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "fragment_id", "frame_start", "frame_end", "observed_frames",
                "observation_count", "start_X", "start_Y", "start_Z",
                "end_X", "end_Y", "end_Z", "residual_median_mm2",
                "mouth_confidence_median", "overlap_agree_frames",
                "overlap_disagree_frames",
            ]
        )
    ordered = observations.sort_values(["fragment_id", "frame", "source_index"])
    grouped = ordered.groupby("fragment_id", sort=True, observed=True)
    first = grouped[["X", "Y", "Z"]].first().rename(
        columns={"X": "start_X", "Y": "start_Y", "Z": "start_Z"}
    )
    last = grouped[["X", "Y", "Z"]].last().rename(
        columns={"X": "end_X", "Y": "end_Y", "Z": "end_Z"}
    )
    summary = grouped.agg(
        frame_start=("frame", "min"),
        frame_end=("frame", "max"),
        observed_frames=("frame", "nunique"),
        observation_count=("frame", "size"),
        residual_median_mm2=("ray_residual_mm2", "median"),
        mouth_confidence_median=("mouth_confidence", "median"),
        overlap_agree_frames=("overlap_agree_frames", "max"),
        overlap_disagree_frames=("overlap_disagree_frames", "max"),
    ).join(first).join(last).reset_index()
    integer_columns = [
        "frame_start", "frame_end", "observed_frames", "observation_count",
        "overlap_agree_frames", "overlap_disagree_frames",
    ]
    summary[integer_columns] = summary[integer_columns].astype(np.int64)
    return summary[
        [
            "fragment_id", "frame_start", "frame_end", "observed_frames",
            "observation_count", "start_X", "start_Y", "start_Z",
            "end_X", "end_Y", "end_Z", "residual_median_mm2",
            "mouth_confidence_median", "overlap_agree_frames",
            "overlap_disagree_frames",
        ]
    ]


def _scaled(value: float, scale: float, clip: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(max(0.0, value) / max(scale, 1e-12), 0.0, clip))


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return np.nan
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return np.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def build_fragment_link_edges(
    fragments: pd.DataFrame,
    *,
    gates: LinkGates = LinkGates(),
    weights: LinkWeights = LinkWeights(),
    tail_mode: str = "off",
) -> pd.DataFrame:
    """Build unit-aware directed edges and conditionally apply tail evidence."""
    if tail_mode not in {"off", "conditional"}:
        raise ValueError("tail_mode must be 'off' or 'conditional'")
    fragment_rows = list(fragments.itertuples(index=False))
    by_end: dict[int, list[object]] = {}
    for fragment in fragment_rows:
        by_end.setdefault(int(fragment.frame_end), []).append(fragment)
    records: list[dict[str, object]] = []
    for target in sorted(
        fragment_rows, key=lambda row: (int(row.frame_start), str(row.fragment_id))
    ):
        target_start = int(target.frame_start)
        for source_end in range(
            target_start - gates.max_gap_frames - 1, target_start
        ):
            for source in by_end.get(source_end, []):
                if str(source.fragment_id) == str(target.fragment_id):
                    continue
                gap = target_start - int(source.frame_end) - 1
                bridge_dt = gap + 1
                source_end_xyz = np.asarray(
                    [source.end_X, source.end_Y, source.end_Z], dtype=float
                )
                target_start_xyz = np.asarray(
                    [target.start_X, target.start_Y, target.start_Z], dtype=float
                )
                displacement = float(np.linalg.norm(target_start_xyz - source_end_xyz))
                jump_gate = min(
                    gates.max_jump_mm_per_frame * bridge_dt,
                    gates.max_jump_hard_cap_mm,
                )
                if displacement > jump_gate:
                    continue
                source_velocity = np.asarray(
                    [source.end_vX, source.end_vY, source.end_vZ], dtype=float
                )
                target_velocity = np.asarray(
                    [target.start_vX, target.start_vY, target.start_vZ], dtype=float
                )
                predicted = (
                    source_end_xyz + source_velocity * bridge_dt
                    if np.all(np.isfinite(source_velocity))
                    else None
                )
                forward_error = (
                    float(np.linalg.norm(target_start_xyz - predicted))
                    if predicted is not None
                    else np.nan
                )
                backward_predicted = (
                    target_start_xyz - target_velocity * bridge_dt
                    if np.all(np.isfinite(target_velocity))
                    else None
                )
                backward_error = (
                    float(np.linalg.norm(source_end_xyz - backward_predicted))
                    if backward_predicted is not None
                    else np.nan
                )
                available_prediction = np.asarray(
                    [forward_error, backward_error], dtype=float
                )
                available_prediction = available_prediction[
                    np.isfinite(available_prediction)
                ]
                prediction_error = (
                    float(np.median(available_prediction))
                    if len(available_prediction)
                    else np.nan
                )
                velocity_angle = _angle_deg(source_velocity, target_velocity)
                agree_frames = int(
                    min(
                        int(getattr(source, "overlap_agree_frames", 0)),
                        int(getattr(target, "overlap_agree_frames", 0)),
                    )
                )
                disagree_frames = int(
                    max(
                        int(getattr(source, "overlap_disagree_frames", 0)),
                        int(getattr(target, "overlap_disagree_frames", 0)),
                    )
                )
                overlap_denominator = agree_frames + disagree_frames
                overlap_consistency = (
                    agree_frames / overlap_denominator
                    if overlap_denominator >= 3
                    else np.nan
                )
                residual = float(
                    np.nanmedian(
                        [
                            source.end_residual_median_mm2,
                            target.start_residual_median_mm2,
                        ]
                    )
                )
                confidence_values = np.asarray(
                    [
                        source.end_mouth_confidence_median,
                        target.start_mouth_confidence_median,
                    ],
                    dtype=float,
                )
                confidence_values = confidence_values[np.isfinite(confidence_values)]
                confidence = (
                    float(np.min(confidence_values))
                    if len(confidence_values) == 2
                    else np.nan
                )
                terms = {
                    "endpoint_cost": weights.endpoint
                    * float(np.clip(displacement / max(jump_gate, 1e-12), 0.0, 1.0)),
                    "prediction_cost": weights.prediction
                    * (
                        float(np.clip(prediction_error / max(jump_gate, 1e-12), 0.0, 1.0))
                        if np.isfinite(prediction_error)
                        else 0.0
                    ),
                    "velocity_angle_cost": 0.0,
                    "gap_cost": weights.temporal_gap
                    * float(np.clip(gap / 100.0, 0.0, 1.0)),
                    "overlap_cost": (
                        weights.overlap_consistency * (1.0 - overlap_consistency)
                        if np.isfinite(overlap_consistency)
                        else 0.0
                    ),
                    "residual_cost": weights.residual
                    * _scaled(
                        residual,
                        weights.residual_scale_mm2,
                        weights.clip_normalized,
                    ),
                    "confidence_cost": (
                        weights.mouth_confidence
                        * float(np.clip(1.0 - confidence, 0.0, 1.0))
                        if np.isfinite(confidence)
                        else 0.0
                    ),
                }
                mouth_cost = float(sum(terms.values()))
                records.append(
                    {
                        "edge_id": "edge_"
                        + stable_hash((source.fragment_id, target.fragment_id))[:16],
                        "source_id": str(source.fragment_id),
                        "target_id": str(target.fragment_id),
                        "source_end": int(source.frame_end),
                        "target_start": target_start,
                        "gap_frames": int(gap),
                        "endpoint_displacement_mm": displacement,
                        "motion_prediction_error_mm": prediction_error,
                        "forward_prediction_error_mm": forward_error,
                        "backward_prediction_error_mm": backward_error,
                        "velocity_sources_available": int(len(available_prediction)),
                        "velocity_angle_deg": velocity_angle,
                        "overlap_consistency": overlap_consistency,
                        "overlap_agree_frames": agree_frames,
                        "overlap_disagree_frames": disagree_frames,
                        "ray_residual_median_mm2": residual,
                        "residual_unit": RESIDUAL_UNIT,
                        "mouth_confidence": confidence,
                        **terms,
                        "mouth_cost": mouth_cost,
                        "tail_eligible": False,
                        "tail_state": "disabled" if tail_mode == "off" else "not_ambiguous",
                        "tail_angle_deg": np.nan,
                        "tail_cost": 0.0,
                        "total_cost": mouth_cost,
                        "source_tail": np.asarray(
                            [source.end_tail3d_X, source.end_tail3d_Y, source.end_tail3d_Z],
                            dtype=float,
                        ),
                        "target_tail": np.asarray(
                            [target.start_tail3d_X, target.start_tail3d_Y, target.start_tail3d_Z],
                            dtype=float,
                        ),
                        "source_tail_2d": [
                            np.asarray(
                                [
                                    getattr(source, f"end_tail2d_cam{view}_x"),
                                    getattr(source, f"end_tail2d_cam{view}_y"),
                                ],
                                dtype=float,
                            )
                            for view in range(3)
                        ],
                        "target_tail_2d": [
                            np.asarray(
                                [
                                    getattr(target, f"start_tail2d_cam{view}_x"),
                                    getattr(target, f"start_tail2d_cam{view}_y"),
                                ],
                                dtype=float,
                            )
                            for view in range(3)
                        ],
                        "source_end_xyz": source_end_xyz,
                        "target_start_xyz": target_start_xyz,
                    }
                )
    if not records:
        return pd.DataFrame(
            columns=[
                "edge_id", "source_id", "target_id", "source_end",
                "target_start", "gap_frames", "endpoint_displacement_mm",
                "motion_prediction_error_mm", "mouth_cost", "tail_eligible",
                "tail_state", "tail_angle_deg", "tail_cost", "total_cost",
            ]
        )
    if tail_mode == "conditional":
        by_boundary: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            by_boundary.setdefault(int(record["target_start"]), []).append(index)
        for boundary_frame, indices in sorted(by_boundary.items()):
            targets = sorted({str(records[index]["target_id"]) for index in indices})
            sources = sorted({str(records[index]["source_id"]) for index in indices})
            edge_by_pair = {
                (str(records[index]["source_id"]), str(records[index]["target_id"])): index
                for index in indices
            }
            # Solve the rectangular source/dummy assignment exactly.  The
            # second-best complete assignment must exclude at least one cell
            # selected by the best; resolving those at-most-three exclusions
            # avoids the former O(n_sources ** n_targets) enumeration.
            large = 1e12
            matrix = np.full(
                (len(targets), len(sources) + len(targets)), large, dtype=float
            )
            edge_index_by_cell: dict[tuple[int, int], int] = {}
            source_index = {source_id: index for index, source_id in enumerate(sources)}
            for target_row, target_id in enumerate(targets):
                for source_id in sources:
                    record_index = edge_by_pair.get((source_id, target_id))
                    if record_index is None:
                        continue
                    column = source_index[source_id]
                    matrix[target_row, column] = float(records[record_index]["mouth_cost"])
                    edge_index_by_cell[(target_row, column)] = record_index
                matrix[target_row, len(sources) + target_row] = 6.0

            def solve_assignment(costs: np.ndarray) -> tuple[float, tuple[int, ...], tuple[tuple[int, int], ...]] | None:
                rows, columns = linear_sum_assignment(costs)
                values = costs[rows, columns]
                if len(rows) != len(targets) or np.any(values >= large / 2):
                    return None
                cells = tuple(sorted(zip(map(int, rows), map(int, columns))))
                edge_indices = tuple(
                    sorted(
                        edge_index_by_cell[cell]
                        for cell in cells
                        if cell in edge_index_by_cell
                    )
                )
                return float(np.sum(values)), edge_indices, cells

            best = solve_assignment(matrix)
            if best is None:
                continue
            best_cost, best_edges, best_cells = best
            runner_candidates = []
            for row, column in best_cells:
                altered = matrix.copy()
                altered[row, column] = large
                candidate = solve_assignment(altered)
                if candidate is not None and candidate[1] != best_edges:
                    runner_candidates.append(candidate)
            if not runner_candidates:
                continue
            runner = min(
                runner_candidates,
                key=lambda value: (
                    value[0],
                    tuple(records[index]["edge_id"] for index in value[1]),
                ),
            )
            if runner[0] - best_cost > weights.mouth_ambiguity_margin + 1e-12:
                continue
            within_margin = [(runner[0], runner[1])]
            competing_edge_sets = [set(best_edges), set(runner[1])]
            common = set.intersection(*competing_edge_sets) if competing_edge_sets else set()
            disputed = sorted(set.union(*competing_edge_sets) - common)
            source_points = {
                str(records[index]["source_id"]): records[index]["source_end_xyz"]
                for index in disputed
            }
            target_points = {
                str(records[index]["target_id"]): records[index]["target_start_xyz"]
                for index in disputed
            }
            competing_mouth_distance = min(
                _pairwise_minimum(np.asarray(list(source_points.values()))),
                _pairwise_minimum(np.asarray(list(target_points.values()))),
            )
            if competing_mouth_distance > weights.close_contact_mm:
                for index in disputed:
                    records[index]["tail_state"] = "mouths_not_close"
                continue
            all_3d = all(
                np.all(np.isfinite(records[index]["source_tail"]))
                and np.all(np.isfinite(records[index]["target_tail"]))
                for index in disputed
            )
            common_2d_views = [
                view
                for view in range(3)
                if all(
                    np.all(np.isfinite(records[index]["source_tail_2d"][view]))
                    and np.all(np.isfinite(records[index]["target_tail_2d"][view]))
                    for index in disputed
                )
            ]
            representation = "3d" if all_3d else "2d" if len(common_2d_views) >= 2 else None
            if representation is None:
                for index in disputed:
                    records[index]["tail_eligible"] = False
                    records[index]["tail_state"] = "tail_incomplete"
                    records[index]["tail_cost"] = 0.0
                    records[index]["total_cost"] = float(records[index]["mouth_cost"])
                continue
            for index in disputed:
                record = records[index]
                record["tail_eligible"] = True
                if representation == "3d":
                    angle = _angle_deg(
                        np.asarray(record["source_tail"], dtype=float),
                        np.asarray(record["target_tail"], dtype=float),
                    )
                else:
                    angles = [
                        _angle_deg(
                            np.asarray(record["source_tail_2d"][view], dtype=float),
                            np.asarray(record["target_tail_2d"][view], dtype=float),
                        )
                        for view in common_2d_views
                    ]
                    angle = float(np.median(angles))
                if not np.isfinite(angle):
                    raise RuntimeError("complete tail set produced a non-finite angle")
                raw_tail_cost = weights.tail_weight * float(
                    np.clip(angle / weights.tail_angle_scale_deg, 0.0, 1.0)
                )
                record["tail_state"] = representation
                record["tail_angle_deg"] = angle
                record["tail_cost"] = float(raw_tail_cost)
                record["total_cost"] = float(record["mouth_cost"] + record["tail_cost"])
            # The fixed per-edge 1/12 bound implies <=0.25 for any three-link
            # complete assignment; assert rather than silently rescale.
            for _cost, edge_indices in [(best_cost, best_edges), *within_margin]:
                tail_sum = sum(float(records[index]["tail_cost"]) for index in edge_indices)
                if tail_sum > 0.25 + 1e-12:
                    raise AssertionError(
                        f"tail assignment cap exceeded at frame {boundary_frame}: {tail_sum}"
                    )
    for record in records:
        record.pop("source_tail", None)
        record.pop("target_tail", None)
        record.pop("source_tail_2d", None)
        record.pop("target_tail_2d", None)
        record.pop("source_end_xyz", None)
        record.pop("target_start_xyz", None)
    return pd.DataFrame.from_records(records).sort_values(
        ["source_end", "target_start", "source_id", "target_id"]
    ).reset_index(drop=True)


def _edge_tables(
    edges: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    costs: dict[tuple[str, str], float] = {}
    identifiers: dict[tuple[str, str], str] = {}
    for index, row in enumerate(edges.itertuples(index=False)):
        pair = (str(row.source_id), str(row.target_id))
        if pair in costs:
            raise ValueError(f"duplicate directed fragment edge {pair}")
        costs[pair] = float(row.total_cost)
        identifiers[pair] = str(
            getattr(row, "edge_id", f"edge_{stable_hash(pair)[:16]}")
        )
    return costs, identifiers


def _paths_from_edges(
    fragment_ids: Sequence[str], selected_pairs: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    successor = {source: target for source, target in selected_pairs}
    predecessor = {target: source for source, target in selected_pairs}
    births = sorted(fragment_id for fragment_id in fragment_ids if fragment_id not in predecessor)
    paths = []
    for birth in births:
        path = [birth]
        while path[-1] in successor:
            path.append(successor[path[-1]])
        paths.append(tuple(path))
    return tuple(paths)


def _solve_global_path_cover(
    fragments: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> tuple[tuple[tuple[str, str], ...], float, int, str]:
    fragment_ids = sorted(fragments.fragment_id.astype(str))
    fragment_index = {fragment_id: index for index, fragment_id in enumerate(fragment_ids)}
    frame_start = dict(
        zip(fragments.fragment_id.astype(str), fragments.frame_start.astype(int), strict=True)
    )
    frame_end = dict(
        zip(fragments.fragment_id.astype(str), fragments.frame_end.astype(int), strict=True)
    )
    edge_rows = sorted(
        list(edges.itertuples(index=False)),
        key=lambda row: (str(row.source_id), str(row.target_id), str(getattr(row, "edge_id", ""))),
    )
    edge_count = len(edge_rows)
    fragment_count = len(fragment_ids)
    variable_count = edge_count + 2 * fragment_count

    def birth(index: int) -> int:
        return edge_count + index

    def death(index: int) -> int:
        return edge_count + fragment_count + index

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Mapping[int, float], bound: float) -> None:
        row = len(lower)
        for column, value in coefficients.items():
            rows.append(row)
            columns.append(int(column))
            values.append(float(value))
        lower.append(float(bound))
        upper.append(float(bound))

    incoming: dict[str, list[int]] = {fragment_id: [] for fragment_id in fragment_ids}
    outgoing: dict[str, list[int]] = {fragment_id: [] for fragment_id in fragment_ids}
    for edge_index, edge in enumerate(edge_rows):
        source, target = str(edge.source_id), str(edge.target_id)
        if source not in fragment_index or target not in fragment_index:
            raise ValueError(f"edge references unknown fragment {(source, target)}")
        if frame_end[source] >= frame_start[target]:
            raise ValueError(f"edge is not strictly time ordered: {(source, target)}")
        outgoing[source].append(edge_index)
        incoming[target].append(edge_index)
    for fragment_id in fragment_ids:
        index = fragment_index[fragment_id]
        add(
            {birth(index): 1.0, **{edge: 1.0 for edge in incoming[fragment_id]}},
            1.0,
        )
        add(
            {death(index): 1.0, **{edge: 1.0 for edge in outgoing[fragment_id]}},
            1.0,
        )
    add({birth(index): 1.0 for index in range(fragment_count)}, 3.0)
    add({death(index): 1.0 for index in range(fragment_count)}, 3.0)
    matrix = coo_array(
        (values, (rows, columns)), shape=(len(lower), variable_count)
    ).tocsc()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    objective = np.zeros(variable_count, dtype=float)
    objective[:edge_count] = [float(edge.total_cost) for edge in edge_rows]
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact three-path cover infeasible: {result.message}")
    primary = float(np.dot(objective, result.x))
    # Deterministic secondary solve.  The primary objective may increase by at
    # most the frozen numerical tolerance.
    primary_constraint = LinearConstraint(
        objective[None, :], -np.inf, primary + float(tolerance)
    )
    lexicographic = np.zeros(variable_count, dtype=float)
    lexicographic[:edge_count] = np.arange(1, edge_count + 1, dtype=float)
    secondary = milp(
        c=lexicographic,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=(constraints, primary_constraint),
        options={"presolve": True},
    )
    final = secondary if secondary.success and secondary.x is not None else result
    selected = tuple(
        (str(edge_rows[index].source_id), str(edge_rows[index].target_id))
        for index in range(edge_count)
        if float(final.x[index]) > 0.5
    )
    recomputed = float(
        sum(
            float(edge_rows[index].total_cost)
            for index in range(edge_count)
            if float(final.x[index]) > 0.5
        )
    )
    if recomputed > primary + tolerance + 1e-12:
        raise AssertionError("secondary path-cover tie break changed primary objective")
    return selected, recomputed, int(result.mip_node_count or 0), str(final.message)


def _solve_boundary_states(
    fragments: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    config: SolverConfig,
) -> tuple[
    tuple[tuple[str, str], ...],
    float,
    int,
    int,
    int,
    tuple[Mapping[str, object], ...],
    Mapping[str, tuple[int, int]],
    int,
]:
    if config.identity_policy not in {"solver_only", "immediate", "deferred"}:
        raise ValueError("identity_policy must be solver_only, immediate, or deferred")
    fragment_rows = {
        str(row.fragment_id): row for row in fragments.itertuples(index=False)
    }
    starts: dict[int, list[str]] = {}
    for fragment_id, row in fragment_rows.items():
        starts.setdefault(int(row.frame_start), []).append(fragment_id)
    edge_cost, edge_id = _edge_tables(edges)
    mouth_cost = {
        (str(row.source_id), str(row.target_id)): float(
            getattr(row, "mouth_cost", row.total_cost)
        )
        for row in edges.itertuples(index=False)
    }
    # History nodes avoid copying an ever-growing mapping into every state.
    histories: list[tuple[int | None, tuple[tuple[int, str, str | None], ...]]] = [
        (None, ())
    ]
    # cost, endpoints, active_end, history_index, episode_start, stable_frames
    states: list[
        tuple[float, tuple[str | None, ...], tuple[int, ...], int, int | None, int]
    ] = [(0.0, (None, None, None), (-1, -1, -1), 0, None, 0)]
    peak = 1
    pruned = 0
    explored = 0
    ties = 0
    ambiguity_records: list[dict[str, object]] = []
    for frame in sorted(starts):
        batch = sorted(
            starts[frame],
            key=lambda fragment_id: (
                float(fragment_rows[fragment_id].start_X), fragment_id
            ),
        )
        next_by_key: dict[
            tuple[tuple[str | None, ...], tuple[int, ...]],
            tuple[
                float, tuple[str | None, ...], tuple[int, ...], int,
                int | None, int, float, tuple[tuple[int, str, str | None], ...],
            ],
        ] = {}
        for state_index, (
            state_cost, endpoints, active_end, history_index, ambiguity_start,
            stable_frames,
        ) in enumerate(states):
            available = [index for index, end in enumerate(active_end) if end < frame]
            if len(batch) > len(available):
                continue
            if all(endpoint is None for endpoint in endpoints):
                # Remove arbitrary global label symmetry at the first observed
                # layer.  Biological ambiguity begins only after these three
                # geometry paths have canonical left-to-right labels.
                path_options = [tuple(available[: len(batch)])]
            else:
                path_options = permutations(available, len(batch))
            for assigned_paths in path_options:
                explored += 1
                new_endpoints = list(endpoints)
                new_active = list(active_end)
                assignments: list[tuple[int, str, str | None]] = []
                transition = 0.0
                transition_mouth = 0.0
                feasible = True
                for fragment_id, path_index in zip(batch, assigned_paths, strict=True):
                    previous = endpoints[path_index]
                    if previous is None:
                        selected_edge = None
                    else:
                        pair = (str(previous), fragment_id)
                        if pair not in edge_cost:
                            if config.path_cover_mode == "restarts":
                                transition += config.death_cost + config.birth_cost
                                transition_mouth += config.death_cost + config.birth_cost
                                selected_edge = None
                            else:
                                feasible = False
                                break
                        else:
                            transition += edge_cost[pair]
                            transition_mouth += mouth_cost[pair]
                            selected_edge = edge_id[pair]
                    new_endpoints[path_index] = fragment_id
                    new_active[path_index] = int(fragment_rows[fragment_id].frame_end)
                    assignments.append((path_index, fragment_id, selected_edge))
                if not feasible:
                    continue
                histories.append((history_index, tuple(assignments)))
                new_history = len(histories) - 1
                value = (
                    state_cost + transition,
                    tuple(new_endpoints),
                    tuple(new_active),
                    new_history,
                    ambiguity_start,
                    stable_frames,
                    transition_mouth,
                    tuple(assignments),
                )
                key = (value[1], value[2])
                old = next_by_key.get(key)
                if old is None or (value[0], value[7], value[3]) < (
                    old[0], old[7], old[3]
                ):
                    next_by_key[key] = value
        ordered = sorted(
            next_by_key.values(),
            key=lambda value: (
                value[0],
                tuple(edge or "" for _path, _fragment, edge in value[7]),
                value[1],
            ),
        )
        if not ordered:
            raise RuntimeError(f"no feasible three-path boundary assignment at frame {frame}")
        if len(ordered) > 1 and abs(ordered[1][0] - ordered[0][0]) <= 1e-12:
            ties += 1
        existing_episode = any(value[4] is not None for value in ordered)
        best_mouth = ordered[0][6]
        ambiguous_candidates = [
            value
            for value in ordered
            if value[7] != ordered[0][7]
            and value[6] - best_mouth <= config.ambiguity_margin + 1e-12
        ]
        ambiguous = bool(ambiguous_candidates)
        if config.method == "sequential_hungarian":
            limit = 1
        elif config.identity_policy == "solver_only":
            limit = int(config.beam_width)
        elif config.identity_policy == "immediate":
            if ambiguous:
                ambiguity_records.append(
                    {
                        "frame_start": int(frame),
                        "frame_end": int(frame),
                        "resolved": True,
                        "resolution_status": "forced_immediate",
                        "hypotheses_entered": 1 + len(ambiguous_candidates),
                        "peak_hypotheses": 1 + len(ambiguous_candidates),
                        "winning_margin": float(ordered[1][0] - ordered[0][0]),
                    }
                )
            limit = 1
        else:
            limit = int(config.beam_width)
            best_cost = ordered[0][0]
            ordered = [
                value for value in ordered
                if value[0] - best_cost <= config.retention_margin + 1e-12
            ]
            if ambiguous and not existing_episode:
                opened = int(frame)
                ordered = [
                    (*value[:4], opened, 0, *value[6:]) for value in ordered
                ]
                existing_episode = True
            if existing_episode:
                opened = min(int(value[4]) for value in ordered if value[4] is not None)
                margin = (
                    float(ordered[1][0] - ordered[0][0])
                    if len(ordered) > 1
                    else np.inf
                )
                direct_support = min(
                    int(fragment_rows[fragment_id].observed_frames)
                    for fragment_id in batch
                )
                stable = (
                    int(ordered[0][5]) + direct_support
                    if margin >= config.resolution_margin
                    else 0
                )
                ordered = [
                    (*value[:5], stable, *value[6:]) for value in ordered
                ]
                if len(ordered) == 1 or stable >= config.resolution_stability_frames:
                    ambiguity_records.append(
                        {
                            "frame_start": opened,
                            "frame_end": int(frame),
                            "resolved": True,
                            "resolution_status": "resolved_by_margin",
                            "hypotheses_at_resolution": len(ordered),
                            "winning_margin": margin,
                            "stable_direct_frames": stable,
                        }
                    )
                    ordered = [(*ordered[0][:4], None, 0, *ordered[0][6:])]
                elif int(frame) - opened >= config.deferred_horizon_frames:
                    ambiguity_records.append(
                        {
                            "frame_start": opened,
                            "frame_end": opened + int(config.deferred_horizon_frames),
                            "resolved": False,
                            "resolution_status": "unresolved_horizon",
                            "hypotheses_retained": len(ordered),
                            "reason": "deferred_horizon_exceeded",
                        }
                    )
                    ordered = [(*ordered[0][:4], None, 0, *ordered[0][6:])]
        if len(ordered) > limit:
            pruned += len(ordered) - limit
            ordered = ordered[:limit]
        states = [value[:6] for value in ordered]
        peak = max(peak, len(states))
    best = min(states, key=lambda value: (value[0], value[1]))
    if any(endpoint is None for endpoint in best[1]):
        raise RuntimeError("fewer than three paths were born")
    if best[4] is not None:
        ambiguity_records.append(
            {
                "frame_start": int(best[4]),
                "frame_end": int(fragments.frame_end.max()),
                "resolved": False,
                "hypotheses_retained": len(states),
                "reason": "recording_ended_before_resolution",
                "resolution_status": "unresolved_recording_end",
            }
        )
    assignment_batches: list[tuple[tuple[int, str, str | None], ...]] = []
    history_index: int | None = best[3]
    while history_index is not None:
        parent, batch_assignments = histories[history_index]
        assignment_batches.append(batch_assignments)
        history_index = parent
    assignments = [
        assignment
        for batch_assignments in reversed(assignment_batches)
        for assignment in batch_assignments
    ]
    selected_edge_ids = {edge for _path, _fragment, edge in assignments if edge is not None}
    fragment_paths: dict[str, tuple[int, int]] = {}
    restarts = 0
    previous_by_path: dict[int, str] = {}
    segment_by_path: dict[int, int] = {}
    for path, fragment, edge in assignments:
        if path not in previous_by_path:
            segment_by_path[path] = 0
        elif edge is None:
            restarts += 1
            segment_by_path[path] += 1
        fragment_paths[str(fragment)] = (int(path), int(segment_by_path[path]))
        previous_by_path[path] = fragment
    pair_by_edge_id = {identifier: pair for pair, identifier in edge_id.items()}
    selected_pairs = tuple(sorted(pair_by_edge_id[identifier] for identifier in selected_edge_ids))
    return (
        selected_pairs,
        float(
            best[0]
            + (
                3 * (config.birth_cost + config.death_cost)
                if config.path_cover_mode == "restarts"
                else 0.0
            )
        ),
        peak,
        pruned,
        ties,
        tuple(ambiguity_records),
        fragment_paths,
        restarts,
    )


def solve_fragment_identities(
    fragments: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    config: SolverConfig = SolverConfig(),
) -> SolverResult:
    """Partition every accepted fragment into exactly three time-ordered paths."""
    valid_methods = {"sequential_hungarian", "beam", "global_dag"}
    if config.method not in valid_methods:
        raise ValueError(f"unknown solver {config.method!r}; expected {valid_methods}")
    if config.path_cover_mode not in {"exact_three", "restarts"}:
        raise ValueError("path_cover_mode must be exact_three or restarts")
    if fragments.empty:
        return SolverResult({}, 0.0, 0, 0, (), 0, config.method, feasible=False, status="empty")
    if fragments.fragment_id.astype(str).duplicated().any():
        raise ValueError("fragment IDs must be unique")
    # At most three accepted nodes may overlap at a frame.
    events: list[tuple[int, int]] = []
    for row in fragments.itertuples(index=False):
        events.append((int(row.frame_start), 1))
        events.append((int(row.frame_end) + 1, -1))
    active = 0
    for _frame, delta in sorted(events, key=lambda value: (value[0], value[1])):
        active += delta
        if active > 3:
            raise ValueError("more than three accepted fragments overlap")
    edge_cost, edge_ids = _edge_tables(edges)
    if config.method == "global_dag":
        if config.path_cover_mode != "exact_three":
            raise ValueError("global_dag requires path_cover_mode=exact_three")
        selected_pairs, objective, explored, status = _solve_global_path_cover(
            fragments, edges
        )
        peak = pruned = ties = 0
        unresolved: tuple[Mapping[str, object], ...] = ()
        fragment_path_assignments: Mapping[str, tuple[int, int]] = {}
        restart_count = 0
    else:
        (
            selected_pairs,
            objective,
            peak,
            pruned,
            ties,
            unresolved,
            fragment_path_assignments,
            restart_count,
        ) = _solve_boundary_states(fragments, edges, config=config)
        explored = 0
        status = "optimal_boundary_state"
    fragment_ids = sorted(fragments.fragment_id.astype(str))
    if config.path_cover_mode == "restarts":
        segment_keys = sorted(set(fragment_path_assignments.values()))
        paths = tuple(
            tuple(
                sorted(
                    (
                        fragment_id for fragment_id, segment_key
                        in fragment_path_assignments.items()
                        if segment_key == key
                    ),
                    key=lambda fragment_id: (
                        int(
                            fragments.loc[
                                fragments.fragment_id.astype(str) == fragment_id,
                                "frame_start",
                            ].iloc[0]
                        ),
                        fragment_id,
                    ),
                )
            )
            for key in segment_keys
        )
    else:
        paths = _paths_from_edges(fragment_ids, selected_pairs)
        if len(paths) != 3 or sum(map(len, paths)) != len(fragment_ids):
            raise AssertionError(
                f"solver returned {len(paths)} paths covering {sum(map(len, paths))}/{len(fragment_ids)} fragments"
            )
    fragment_by_id = {
        str(row.fragment_id): row for row in fragments.itertuples(index=False)
    }
    ordered_paths = (
        paths
        if config.path_cover_mode == "restarts"
        else tuple(
            sorted(
                paths,
                key=lambda path: (
                    int(fragment_by_id[path[0]].frame_start),
                    float(fragment_by_id[path[0]].start_X),
                    path,
                ),
            )
        )
    )
    identities = {
        fragment_id: (
            f"SEG_{path_index + 1:06d}"
            if config.path_cover_mode == "restarts"
            else ID_LABELS[path_index]
        )
        for path_index, path in enumerate(ordered_paths)
        for fragment_id in path
    }
    selected_edge_ids = tuple(sorted(edge_ids[pair] for pair in selected_pairs))
    recomputed = float(
        sum(edge_cost[pair] for pair in selected_pairs)
        + restart_count * (config.birth_cost + config.death_cost)
        + (
            3 * (config.birth_cost + config.death_cost)
            if config.path_cover_mode == "restarts"
            else 0.0
        )
    )
    if not np.isclose(recomputed, objective, atol=1e-9, rtol=0.0):
        raise AssertionError("solver objective does not match selected graph edges")
    return SolverResult(
        fragment_identities=identities,
        objective=recomputed,
        hypotheses_peak=int(peak),
        pruned_states=int(pruned),
        unresolved_intervals=unresolved,
        deterministic_tie_count=int(ties),
        method=config.method,
        selected_edges=selected_edge_ids,
        paths=ordered_paths,
        feasible=True,
        status=status,
        explored_states=int(explored),
    )


def apply_fragment_identities(
    observations: pd.DataFrame,
    result: SolverResult,
    *,
    identity_policy: str = "immediate",
) -> pd.DataFrame:
    table = observations.copy()
    table["fish_id"] = table["fragment_id"].map(result.fragment_identities)
    table["identity_status"] = np.where(table["fish_id"].notna(), "resolved", "unmapped")
    if identity_policy == "deferred":
        for interval in result.unresolved_intervals:
            if bool(interval.get("resolved", False)):
                continue
            start = int(interval["frame_start"])
            end = int(interval["frame_end"])
            affected = table.frame.between(start, end)
            table.loc[affected, "fish_id"] = pd.NA
            table.loc[affected, "identity_status"] = "unresolved"
    elif identity_policy not in {"immediate", "solver_only"}:
        raise ValueError("identity_policy must be solver_only, immediate, or deferred")
    table["identity_resolved"] = table["identity_status"] == "resolved"
    return table


def track_observations_hungarian(
    observations: pd.DataFrame,
    *,
    max_jump_mm: float = 8.0,
    max_jump_hard_cap_mm: float = 80.0,
    max_gap_frames: int = 100,
) -> pd.DataFrame:
    """Compact position-only equivalent of the strict-mouth baseline linker."""
    table = observations.sort_values(["frame", "source_index"]).reset_index(drop=True).copy()
    frames = table["frame"].to_numpy(dtype=np.int64)
    xyz_all = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    fish_ids = np.empty(len(table), dtype=object)
    # Object-column NaN matches the Parquet round-trip representation of the
    # earlier pandas scalar-update implementation exactly.
    fish_ids.fill(np.nan)
    identity_status = np.full(len(table), "unmapped", dtype=object)
    states: list[tuple[int, np.ndarray] | None] = [None, None, None]
    unique_frames, starts = np.unique(frames, return_index=True)
    ends = np.r_[starts[1:], len(table)]
    for frame, row_start, row_end in zip(unique_frames, starts, ends, strict=True):
        xyz = xyz_all[row_start:row_end]
        # Retire stale states before constructing the common assignment matrix.
        for identity, state in enumerate(states):
            if state is not None and int(frame) - state[0] > int(max_gap_frames):
                states[identity] = None
        active = [identity for identity, state in enumerate(states) if state is not None]
        unmatched_rows = set(range(len(xyz)))
        if active and len(xyz):
            cost = np.full((len(active), len(xyz)), 1e12, dtype=float)
            for active_row, identity in enumerate(active):
                previous_frame, previous_xyz = states[identity]  # type: ignore[misc]
                gap = max(1, int(frame) - int(previous_frame))
                gate = min(float(max_jump_mm) * gap, float(max_jump_hard_cap_mm))
                distances = np.linalg.norm(xyz - previous_xyz, axis=1)
                cost[active_row, distances <= gate] = distances[distances <= gate]
            rows, columns = linear_sum_assignment(cost)
            for active_row, local_row in zip(rows, columns, strict=True):
                if cost[int(active_row), int(local_row)] >= 1e11:
                    continue
                identity = active[int(active_row)]
                row_index = int(row_start + int(local_row))
                fish_ids[row_index] = ID_LABELS[identity]
                identity_status[row_index] = "resolved"
                states[identity] = (int(frame), xyz[int(local_row)].copy())
                unmatched_rows.discard(int(local_row))
        inactive = [identity for identity, state in enumerate(states) if state is None]
        for local_row, identity in zip(
            sorted(unmatched_rows, key=lambda row: (float(xyz[row, 0]), row)),
            inactive,
            strict=False,
        ):
            row_index = int(row_start + int(local_row))
            fish_ids[row_index] = ID_LABELS[identity]
            identity_status[row_index] = "resolved"
            states[identity] = (int(frame), xyz[int(local_row)].copy())
        # Remaining unmatched geometry is deliberately left unmapped rather
        # than duplicating an active biological identity.
    table["fish_id"] = pd.Series(fish_ids, dtype="object")
    table["identity_status"] = identity_status
    table["identity_resolved"] = identity_status == "resolved"
    return table


def assert_tracking_invariants(observations: pd.DataFrame) -> dict[str, int]:
    """Raise on raw-resource reuse or more than three active identities."""
    resource_conflicts = 0
    for view_index in range(3):
        valid = observations[observations[f"slot_cam{view_index}"] >= 0]
        duplicated = valid.duplicated(["frame", f"slot_cam{view_index}"], keep=False)
        resource_conflicts += int(duplicated.sum())
    if resource_conflicts:
        raise AssertionError(f"raw detection resource conflicts: {resource_conflicts}")
    if "fish_id" in observations:
        mapped = observations.dropna(subset=["fish_id"])
        duplicated_identity = mapped.duplicated(["frame", "fish_id"], keep=False)
        if duplicated_identity.any():
            raise AssertionError(
                f"one biological identity occupies multiple rows: {int(duplicated_identity.sum())}"
            )
        active = mapped.groupby("frame")["fish_id"].nunique()
        too_many = int((active > 3).sum())
        if too_many:
            raise AssertionError(f"more than three active trajectories on {too_many} frames")
    return {"resource_conflicts": 0, "frames_over_three_active": 0}


def configuration_dict(*values: object) -> dict[str, object]:
    output: dict[str, object] = {}
    for value in values:
        if hasattr(value, "__dataclass_fields__"):
            output[value.__class__.__name__] = asdict(value)
    return output
