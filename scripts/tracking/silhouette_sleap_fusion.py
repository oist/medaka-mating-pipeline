from __future__ import annotations

"""Fuse SLEAP mouth/tail tracklets with independent instance silhouettes."""

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sam2_id_experiment.src.sam2_adapter.mask_io import MaskInstance


@dataclass(frozen=True)
class FusionConfig:
    min_mask_score: float = 0.35
    candidate_coverage: float = 0.20
    strong_skeleton_coverage: float = 0.72
    supported_skeleton_coverage: float = 0.45
    conflict_skeleton_coverage: float = 0.18
    strong_contradiction_skeleton_coverage: float = 0.08
    strong_contradiction_min_assignment_margin: float = 0.35
    strong_contradiction_min_mask_score: float = 0.70
    strong_contradiction_min_centroid_distance_radius: float = 0.75
    min_assignment_margin: float = 0.12
    sample_count: int = 41
    body_half_width_fraction: float = 0.06
    min_temporal_iou: float = 0.02
    max_temporal_centroid_step_bl: float = 0.35
    min_temporal_area_ratio: float = 0.45
    max_temporal_area_ratio: float = 2.20
    mask_track_max_gap_frames: int = 3
    mask_track_max_centroid_step_radius: float = 2.5
    mask_track_max_cost: float = 2.8
    mask_track_min_margin: float = 0.10


@dataclass
class MaskTrackState:
    track_id: int
    last_frame: int
    centroid: np.ndarray
    velocity: np.ndarray
    area: float
    axis: np.ndarray
    radius: float
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


def _mask_geometry(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, mask.shape[1], mask.shape[0]
    else:
        x0, y0, x1, y1 = (int(value) for value in bbox)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    xs = xs + x0
    ys = ys + y0
    if len(xs) == 0:
        return {
            "area": 0,
            "centroid": np.asarray([np.nan, np.nan]),
            "axis": np.asarray([np.nan, np.nan]),
            "radius": 1.0,
        }
    points = np.column_stack([xs.astype(float), ys.astype(float)])
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(points) >= 2:
        covariance = centered.T @ centered / max(1, len(points) - 1)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
    else:
        axis = np.asarray([1.0, 0.0])
    return {
        "area": int(len(points)),
        "centroid": centroid,
        "axis": np.asarray(axis, dtype=float),
        "radius": float(np.sqrt(len(points) / np.pi)),
    }


def _point_inside(mask: np.ndarray, point: np.ndarray) -> bool:
    if not np.isfinite(point).all():
        return False
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _sample_skeleton_coverage(
    mask: np.ndarray,
    mouth: np.ndarray,
    tail: np.ndarray,
    *,
    sample_count: int,
    half_width_fraction: float,
) -> tuple[float, float]:
    direction = mouth - tail
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return 0.0, 0.0
    unit = direction / length
    normal = np.asarray([-unit[1], unit[0]])
    alpha = np.linspace(0.0, 1.0, max(3, int(sample_count)))[:, None]
    centerline = tail[None, :] + alpha * direction[None, :]
    half_width = max(1.0, float(half_width_fraction) * length)
    offsets = np.asarray([-half_width, 0.0, half_width])
    inside_by_offset: list[float] = []
    for offset in offsets:
        points = centerline + normal[None, :] * offset
        inside = [_point_inside(mask, point) for point in points]
        inside_by_offset.append(float(np.mean(inside)))
    return inside_by_offset[1], float(np.mean(inside_by_offset))


def _axial_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-9:
        return 90.0
    cosine = abs(float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosine)))


def _pair_metrics(
    row: pd.Series,
    instance: MaskInstance,
    config: FusionConfig,
    *,
    geometry: dict[str, object] | None = None,
) -> dict[str, float | bool]:
    mask = np.asarray(instance.mask, dtype=bool)
    geometry = geometry or _mask_geometry(mask, instance.bbox)
    mouth = np.asarray([row["mouth_x"], row["mouth_y"]], dtype=float)
    tail = np.asarray([row["tail_x"], row["tail_y"]], dtype=float)
    centroid = 0.5 * (mouth + tail)
    body_vector = mouth - tail
    body_length = max(1.0, float(np.linalg.norm(body_vector)))
    skeleton_coverage, capsule_coverage = _sample_skeleton_coverage(
        mask,
        mouth,
        tail,
        sample_count=config.sample_count,
        half_width_fraction=config.body_half_width_fraction,
    )
    centroid_distance = float(np.linalg.norm(centroid - geometry["centroid"]))
    centroid_distance_bl = centroid_distance / body_length
    centroid_distance_radius = centroid_distance / max(1.0, float(geometry["radius"]))
    axis_angle = _axial_angle_deg(body_vector, np.asarray(geometry["axis"], dtype=float))
    mouth_inside = _point_inside(mask, mouth)
    tail_inside = _point_inside(mask, tail)
    endpoint_fraction = 0.5 * (float(mouth_inside) + float(tail_inside))
    score_penalty = max(0.0, 0.75 - min(1.0, float(instance.score)))
    cost = (
        1.25 * (1.0 - skeleton_coverage)
        + 0.65 * (1.0 - capsule_coverage)
        + 0.30 * min(2.0, centroid_distance_radius)
        + 0.25 * (axis_angle / 90.0)
        + 0.35 * (1.0 - endpoint_fraction)
        + 0.30 * score_penalty
    )
    return {
        "cost": float(cost),
        "mouth_inside": mouth_inside,
        "tail_inside": tail_inside,
        "skeleton_coverage": skeleton_coverage,
        "capsule_coverage": capsule_coverage,
        "centroid_distance_px": centroid_distance,
        "centroid_distance_bl": centroid_distance_bl,
        "centroid_distance_radius": centroid_distance_radius,
        "mask_axis_angle_deg": axis_angle,
        "mask_area_px": int(geometry["area"]),
        "mask_centroid_x": float(np.asarray(geometry["centroid"])[0]),
        "mask_centroid_y": float(np.asarray(geometry["centroid"])[1]),
    }


def associate_frame(
    observations: pd.DataFrame,
    masks: list[MaskInstance],
    *,
    config: FusionConfig,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    observations = observations.reset_index(drop=True)
    masks = [mask for mask in masks if float(mask.score) >= config.min_mask_score]
    if not masks:
        out = observations.copy()
        out["matched_mask_instance_id"] = -1
        out["matched_mask_track_id"] = -1
        out["silhouette_status"] = "no_mask"
        out["silhouette_evidence_state"] = "unknown"
        out["mutual_confidence"] = 0.0
        return out

    metrics: list[list[dict[str, float | bool]]] = []
    cost = np.empty((len(observations), len(masks)), dtype=float)
    geometries = [
        _mask_geometry(np.asarray(mask.mask, dtype=bool), mask.bbox) for mask in masks
    ]
    for row_index, row in observations.iterrows():
        row_metrics = []
        for mask_index, mask in enumerate(masks):
            pair = _pair_metrics(row, mask, config, geometry=geometries[mask_index])
            row_metrics.append(pair)
            cost[row_index, mask_index] = float(pair["cost"])
        metrics.append(row_metrics)

    row_index, column_index = linear_sum_assignment(cost)
    assignment = {int(left): int(right) for left, right in zip(row_index, column_index)}
    candidate_by_observation = {
        left: [
            right
            for right in range(len(masks))
            if float(metrics[left][right]["skeleton_coverage"]) >= config.candidate_coverage
            or bool(metrics[left][right]["mouth_inside"])
            or bool(metrics[left][right]["tail_inside"])
        ]
        for left in range(len(observations))
    }
    candidate_by_mask = {
        right: [left for left, candidates in candidate_by_observation.items() if right in candidates]
        for right in range(len(masks))
    }

    output_rows: list[dict[str, object]] = []
    for left, row in observations.iterrows():
        base = row.to_dict()
        if left not in assignment:
            output_rows.append(
                {
                    **base,
                    "matched_mask_instance_id": -1,
                    "matched_mask_track_id": -1,
                    "silhouette_status": "no_mask",
                    "silhouette_evidence_state": "unknown",
                    "mutual_confidence": 0.0,
                }
            )
            continue
        right = assignment[left]
        mask = masks[right]
        pair = metrics[left][right]
        alternatives = np.delete(cost[left], right)
        margin = float(np.min(alternatives) - cost[left, right]) if alternatives.size else np.inf
        candidates = candidate_by_observation[left]
        masks_for_mouth = [index for index in candidates if bool(metrics[left][index]["mouth_inside"])]
        masks_for_tail = [index for index in candidates if bool(metrics[left][index]["tail_inside"])]
        split_endpoints = bool(
            masks_for_mouth
            and masks_for_tail
            and not set(masks_for_mouth).intersection(masks_for_tail)
        )
        merge_count = len(candidate_by_mask[right])
        coverage = float(pair["skeleton_coverage"])
        endpoint_count = int(bool(pair["mouth_inside"])) + int(bool(pair["tail_inside"]))
        complete_one_to_one_frame = (
            len(observations) >= 2
            and len(observations) == len(masks) == len(assignment)
        )
        other_assignments_supported = all(
            (
                float(metrics[other_left][assignment[other_left]]["skeleton_coverage"])
                >= config.supported_skeleton_coverage
                or bool(metrics[other_left][assignment[other_left]]["mouth_inside"])
                or bool(metrics[other_left][assignment[other_left]]["tail_inside"])
            )
            and len(candidate_by_observation[other_left]) == 1
            and len(candidate_by_mask[assignment[other_left]]) == 1
            for other_left in assignment
            if other_left != left
        )
        strong_one_to_one_contradiction = (
            complete_one_to_one_frame
            and other_assignments_supported
            and len(candidates) == 0
            and len(candidate_by_mask[right]) == 0
            and coverage < config.strong_contradiction_skeleton_coverage
            and endpoint_count == 0
            and margin >= config.strong_contradiction_min_assignment_margin
            and float(mask.score) >= config.strong_contradiction_min_mask_score
            and float(pair["centroid_distance_radius"])
            >= config.strong_contradiction_min_centroid_distance_radius
        )
        if split_endpoints:
            status = "unknown_split_endpoints"
        elif merge_count > 1:
            status = "unknown_mask_merges_skeletons"
        elif strong_one_to_one_contradiction:
            status = "strong_contradiction_low_overlap"
        elif coverage < config.conflict_skeleton_coverage and endpoint_count == 0:
            status = "unknown_low_overlap"
        elif len(candidates) > 1 or margin < config.min_assignment_margin:
            status = "ambiguous_mask_assignment"
        elif coverage >= config.strong_skeleton_coverage and endpoint_count == 2:
            status = "strong_match"
        elif coverage >= config.supported_skeleton_coverage or endpoint_count >= 1:
            status = "supported_match"
        else:
            status = "ambiguous_low_support"
        evidence_state = (
            "support"
            if status in {"strong_match", "supported_match"}
            else "contradiction"
            if status == "strong_contradiction_low_overlap"
            else "unknown"
        )
        confidence = float(np.exp(-float(pair["cost"])))
        if np.isfinite(margin):
            confidence *= float(1.0 - np.exp(-max(0.0, margin)))
        if evidence_state == "contradiction":
            confidence = 0.0
        output_rows.append(
            {
                **base,
                **pair,
                "matched_mask_instance_id": int(mask.instance_id),
                "matched_mask_track_id": int(mask.sam2_track_id),
                "mask_score": float(mask.score),
                "association_margin": margin,
                "n_candidate_masks": len(candidates),
                "n_skeletons_for_mask": merge_count,
                "split_endpoints_across_masks": split_endpoints,
                "silhouette_status": status,
                "silhouette_evidence_state": evidence_state,
                "mutual_confidence": confidence,
                "_matched_mask_array": np.asarray(mask.mask, dtype=bool),
                "_matched_mask_bbox": tuple(mask.bbox),
            }
        )
    return pd.DataFrame(output_rows)


def _cropped_iou(
    left: np.ndarray,
    left_bbox: tuple[int, int, int, int],
    right: np.ndarray,
    right_bbox: tuple[int, int, int, int],
) -> float:
    x0 = max(0, min(left_bbox[0], right_bbox[0]))
    y0 = max(0, min(left_bbox[1], right_bbox[1]))
    x1 = min(left.shape[1], max(left_bbox[2], right_bbox[2]))
    y1 = min(left.shape[0], max(left_bbox[3], right_bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = left[y0:y1, x0:x1]
    b = right[y0:y1, x0:x1]
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _track_masks_frame(
    masks: list[MaskInstance],
    active: dict[int, MaskTrackState],
    *,
    frame: int,
    next_track_id: int,
    config: FusionConfig,
) -> tuple[list[MaskInstance], dict[int, MaskTrackState], int, dict[int, dict[str, object]]]:
    active = {
        track_id: state
        for track_id, state in active.items()
        if frame - state.last_frame <= config.mask_track_max_gap_frames
    }
    geometries = [
        _mask_geometry(np.asarray(mask.mask, dtype=bool), mask.bbox) for mask in masks
    ]
    track_ids = list(active)
    cost = np.full((len(track_ids), len(masks)), 1e6, dtype=float)
    pair_diagnostics: dict[tuple[int, int], dict[str, float]] = {}
    for row_index, track_id in enumerate(track_ids):
        state = active[track_id]
        gap = max(1, frame - state.last_frame)
        predicted = state.centroid + state.velocity * gap
        for column_index, (mask, geometry) in enumerate(zip(masks, geometries)):
            centroid = np.asarray(geometry["centroid"], dtype=float)
            radius = max(1.0, 0.5 * (state.radius + float(geometry["radius"])))
            distance = float(np.linalg.norm(centroid - predicted))
            distance_radius = distance / radius
            iou = _cropped_iou(
                state.mask,
                state.bbox,
                np.asarray(mask.mask, dtype=bool),
                tuple(mask.bbox),
            )
            area_ratio = float(geometry["area"]) / max(1.0, state.area)
            area_change = abs(float(np.log(max(area_ratio, 1e-6))))
            axis_angle = _axial_angle_deg(state.axis, np.asarray(geometry["axis"], dtype=float))
            pair_cost = (
                0.80 * min(3.0, distance_radius)
                + 1.20 * (1.0 - iou)
                + 0.30 * area_change
                + 0.20 * (axis_angle / 90.0)
                + 0.20 * max(0.0, 0.75 - min(1.0, float(mask.score)))
            )
            if distance_radius <= config.mask_track_max_centroid_step_radius * gap:
                cost[row_index, column_index] = pair_cost
            pair_diagnostics[(row_index, column_index)] = {
                "cost": pair_cost,
                "iou": iou,
                "centroid_distance_px": distance,
                "centroid_distance_radius": distance_radius,
                "area_ratio": area_ratio,
                "axis_angle_deg": axis_angle,
            }
    assignment: dict[int, int] = {}
    if track_ids and masks:
        rows, columns = linear_sum_assignment(cost)
        for row_index, column_index in zip(rows, columns):
            if cost[int(row_index), int(column_index)] <= config.mask_track_max_cost:
                assignment[int(column_index)] = int(row_index)

    linked: list[MaskInstance] = []
    diagnostics: dict[int, dict[str, object]] = {}
    for column_index, (mask, geometry) in enumerate(zip(masks, geometries)):
        if column_index in assignment:
            row_index = assignment[column_index]
            track_id = track_ids[row_index]
            state = active[track_id]
            gap = max(1, frame - state.last_frame)
            centroid = np.asarray(geometry["centroid"], dtype=float)
            observed_velocity = (centroid - state.centroid) / gap
            velocity = 0.4 * state.velocity + 0.6 * observed_velocity
            alternatives = np.delete(cost[row_index], column_index)
            finite_alternatives = alternatives[alternatives < 1e6]
            margin = (
                float(np.min(finite_alternatives) - cost[row_index, column_index])
                if finite_alternatives.size
                else np.inf
            )
            pair = pair_diagnostics[(row_index, column_index)]
            track_status = "ambiguous" if margin < config.mask_track_min_margin else "linked"
        else:
            track_id = next_track_id
            next_track_id += 1
            centroid = np.asarray(geometry["centroid"], dtype=float)
            velocity = np.zeros(2, dtype=float)
            margin = np.inf
            pair = {
                "cost": np.nan,
                "iou": np.nan,
                "centroid_distance_px": np.nan,
                "centroid_distance_radius": np.nan,
                "area_ratio": np.nan,
                "axis_angle_deg": np.nan,
            }
            track_status = "seeded"
        linked_mask = MaskInstance(
            frame=int(mask.frame),
            instance_id=int(mask.instance_id),
            sam2_track_id=int(track_id),
            mask=np.asarray(mask.mask, dtype=bool),
            bbox=tuple(mask.bbox),
            score=float(mask.score),
        )
        linked.append(linked_mask)
        active[track_id] = MaskTrackState(
            track_id=track_id,
            last_frame=frame,
            centroid=centroid,
            velocity=velocity,
            area=float(geometry["area"]),
            axis=np.asarray(geometry["axis"], dtype=float),
            radius=float(geometry["radius"]),
            mask=np.asarray(mask.mask, dtype=bool),
            bbox=tuple(mask.bbox),
        )
        diagnostics[int(mask.instance_id)] = {
            "mask_track_id": int(track_id),
            "mask_track_status": track_status,
            "mask_track_assignment_margin": margin,
            "mask_track_cost": float(pair["cost"]),
            "mask_track_iou": float(pair["iou"]),
            "mask_track_centroid_distance_px": float(pair["centroid_distance_px"]),
            "mask_track_area_ratio": float(pair["area_ratio"]),
        }
    return linked, active, next_track_id, diagnostics


def annotate_tracklets_with_masks(
    tracklets: pd.DataFrame,
    mask_path: str | Path,
    *,
    config: FusionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = config or FusionConfig()
    required = {"frame", "tracklet_id", "mouth_x", "mouth_y", "tail_x", "tail_y"}
    missing = required - set(tracklets.columns)
    if missing:
        raise ValueError(f"tracklet table missing columns: {sorted(missing)}")
    observations_by_frame = {
        int(frame): group.copy() for frame, group in tracklets.groupby("frame", sort=True)
    }
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for silhouette fusion") from exc
    output: list[pd.DataFrame] = []
    previous: dict[int, dict[str, object]] = {}
    active_mask_tracks: dict[int, MaskTrackState] = {}
    next_mask_track_id = 0
    with h5py.File(Path(mask_path), "r") as h5:
        frame_min = min(observations_by_frame)
        frame_max = max(observations_by_frame)
        for frame in range(frame_min, frame_max + 1):
            frame_key = f"frame_{int(frame):06d}"
            frame_instances: list[MaskInstance] = []
            if frame_key in h5:
                frame_group = h5[frame_key]
                for instance_key in sorted(
                    key for key in frame_group.keys() if key.startswith("instance_")
                ):
                    group = frame_group[instance_key]
                    mask = np.asarray(group["mask"][...], dtype=bool)
                    bbox = tuple(int(value) for value in np.asarray(group["bbox"][...]).tolist())
                    frame_instances.append(
                        MaskInstance(
                            frame=int(group.attrs.get("frame", frame)),
                            instance_id=int(group.attrs.get("instance_id", int(instance_key.split("_")[1]))),
                            sam2_track_id=int(group.attrs.get("sam2_track_id", -1)),
                            mask=mask,
                            bbox=bbox,
                            score=float(group.attrs.get("score", np.nan)),
                        )
                    )
            linked_masks, active_mask_tracks, next_mask_track_id, mask_track_diagnostics = _track_masks_frame(
                frame_instances,
                active_mask_tracks,
                frame=frame,
                next_track_id=next_mask_track_id,
                config=config,
            )
            if frame not in observations_by_frame:
                continue
            associated = associate_frame(observations_by_frame[frame], linked_masks, config=config)
            if "matched_mask_instance_id" in associated:
                for column in (
                    "mask_track_status",
                    "mask_track_assignment_margin",
                    "mask_track_cost",
                    "mask_track_iou",
                    "mask_track_centroid_distance_px",
                    "mask_track_area_ratio",
                ):
                    associated[column] = associated["matched_mask_instance_id"].map(
                        lambda value: mask_track_diagnostics.get(int(value), {}).get(column, np.nan)
                        if pd.notna(value) and int(value) >= 0
                        else np.nan
                    )
                downgrade = (
                    associated["mask_track_status"].eq("ambiguous")
                    & associated["silhouette_status"].isin(["strong_match", "supported_match"])
                )
                associated.loc[downgrade, "silhouette_status"] = "ambiguous_mask_track"
                associated.loc[downgrade, "silhouette_evidence_state"] = "unknown"
            temporal_iou: list[float] = []
            temporal_step: list[float] = []
            temporal_area_ratio: list[float] = []
            evidence_states: list[str] = []
            soft_reasons: list[str] = []
            break_reasons: list[str] = []
            for row in associated.to_dict("records"):
                tracklet_id = int(row["tracklet_id"])
                raw_current_mask = row.get("_matched_mask_array")
                current_mask = (
                    np.asarray(raw_current_mask, dtype=bool)
                    if isinstance(raw_current_mask, np.ndarray) and raw_current_mask.ndim == 2
                    else None
                )
                raw_current_bbox = row.get("_matched_mask_bbox")
                current_bbox = (
                    tuple(int(value) for value in raw_current_bbox)
                    if isinstance(raw_current_bbox, (tuple, list, np.ndarray)) and len(raw_current_bbox) == 4
                    else None
                )
                prior = previous.get(tracklet_id)
                iou = np.nan
                step = np.nan
                area_ratio = np.nan
                temporal_reason = ""
                if current_mask is not None and current_bbox is not None and prior is not None and int(row["frame"]) - int(prior["frame"]) == 1:
                    iou = _cropped_iou(
                        np.asarray(prior["mask"]),
                        tuple(prior["bbox"]),
                        current_mask,
                        current_bbox,
                    )
                    step = float(
                        np.hypot(
                            float(row.get("mask_centroid_x", np.nan)) - float(prior["centroid_x"]),
                            float(row.get("mask_centroid_y", np.nan)) - float(prior["centroid_y"]),
                        )
                    )
                    prior_area = max(1.0, float(prior["area"]))
                    area_ratio = float(row.get("mask_area_px", np.nan)) / prior_area
                    body_length = max(1.0, float(row.get("body_length_px", np.nan)))
                    if (
                        iou < config.min_temporal_iou
                        and step > config.max_temporal_centroid_step_bl * body_length
                    ):
                        temporal_reason = "mask_temporal_discontinuity"
                    elif not (config.min_temporal_area_ratio <= area_ratio <= config.max_temporal_area_ratio):
                        temporal_reason = "mask_area_discontinuity"
                status = str(row.get("silhouette_status", ""))
                evidence_state = str(row.get("silhouette_evidence_state", "unknown"))
                soft_reason = ""
                break_reason = ""
                if temporal_reason:
                    # A discontinuous mask track is unreliable segmentation
                    # evidence, not evidence that the SLEAP observation is
                    # wrong. Keep the row and stop only the trusted mask run.
                    evidence_state = "unknown"
                    soft_reason = temporal_reason
                elif evidence_state == "contradiction":
                    break_reason = status
                elif evidence_state == "unknown":
                    soft_reason = status
                temporal_iou.append(iou)
                temporal_step.append(step)
                temporal_area_ratio.append(area_ratio)
                evidence_states.append(evidence_state)
                soft_reasons.append(soft_reason)
                break_reasons.append(break_reason)
                if current_mask is not None and current_bbox is not None:
                    previous[tracklet_id] = {
                        "frame": int(row["frame"]),
                        "mask": current_mask,
                        "bbox": current_bbox,
                        "centroid_x": float(row.get("mask_centroid_x", np.nan)),
                        "centroid_y": float(row.get("mask_centroid_y", np.nan)),
                        "area": float(row.get("mask_area_px", np.nan)),
                    }
            associated["temporal_mask_iou"] = temporal_iou
            associated["temporal_mask_centroid_step_px"] = temporal_step
            associated["temporal_mask_area_ratio"] = temporal_area_ratio
            associated["silhouette_evidence_state"] = evidence_states
            associated["silhouette_soft_reason"] = soft_reasons
            associated["silhouette_break_reason"] = break_reasons
            output.append(associated)
    annotated = pd.concat(output, ignore_index=True).sort_values(["frame", "tracklet_id"])
    annotated = annotated.drop(columns=["_matched_mask_array", "_matched_mask_bbox"], errors="ignore")
    evidence_weight = {
        "strong_match": 1.0,
        "supported_match": 0.70,
        "ambiguous_mask_assignment": 0.20,
        "ambiguous_low_support": 0.10,
        "ambiguous_mask_track": 0.15,
        "no_mask": 0.0,
        "unknown_mask_merges_skeletons": -0.20,
        "unknown_split_endpoints": -0.20,
        "unknown_low_overlap": -0.25,
        "strong_contradiction_low_overlap": -1.0,
    }
    annotated["silhouette_evidence_weight"] = (
        annotated["silhouette_status"].map(evidence_weight).fillna(0.0).astype(float)
    )
    annotated.loc[
        annotated["silhouette_evidence_state"].eq("unknown")
        & annotated["silhouette_soft_reason"].isin(
            ["mask_temporal_discontinuity", "mask_area_discontinuity"]
        ),
        "silhouette_evidence_weight",
    ] = 0.0
    annotated.loc[
        annotated["silhouette_evidence_state"].eq("contradiction"),
        "silhouette_evidence_weight",
    ] = -1.0
    annotated = assign_trusted_mask_segments(
        annotated, max_gap_frames=config.mask_track_max_gap_frames
    )
    status_counts = annotated["silhouette_status"].value_counts().to_dict()
    break_counts = (
        annotated.loc[annotated["silhouette_break_reason"].fillna("").ne(""), "silhouette_break_reason"]
        .value_counts()
        .to_dict()
    )
    soft_counts = (
        annotated.loc[
            annotated["silhouette_soft_reason"].fillna("").ne(""),
            "silhouette_soft_reason",
        ]
        .value_counts()
        .to_dict()
    )
    evidence_counts = annotated["silhouette_evidence_state"].value_counts().to_dict()
    summary = {
        "rows": int(len(annotated)),
        "frames": int(annotated["frame"].nunique()),
        "tracklets": int(annotated["tracklet_id"].nunique()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "evidence_state_counts": {
            str(key): int(value) for key, value in evidence_counts.items()
        },
        "break_reason_counts": {str(key): int(value) for key, value in break_counts.items()},
        "soft_reason_counts": {str(key): int(value) for key, value in soft_counts.items()},
        "mean_mutual_confidence": float(annotated["mutual_confidence"].mean()),
        "median_temporal_mask_iou": float(annotated["temporal_mask_iou"].median()),
        "config": asdict(config),
    }
    return annotated, summary


def assign_trusted_mask_segments(
    annotated: pd.DataFrame,
    *,
    max_gap_frames: int = 3,
) -> pd.DataFrame:
    output = annotated.copy()
    segment_ids = pd.Series(-1, index=output.index, dtype=int)
    next_segment_id = 0
    matched = output[pd.to_numeric(output["matched_mask_track_id"], errors="coerce").ge(0)]
    for _, group in matched.groupby("matched_mask_track_id", sort=True):
        current_segment: int | None = None
        previous_frame: int | None = None
        for index, row in group.sort_values("frame").iterrows():
            status = str(row.get("silhouette_status", ""))
            evidence_state = str(row.get("silhouette_evidence_state", "unknown"))
            mask_track_status = str(row.get("mask_track_status", ""))
            raw_break_reason = row.get("silhouette_break_reason", "")
            has_break_reason = pd.notna(raw_break_reason) and bool(str(raw_break_reason))
            raw_soft_reason = row.get("silhouette_soft_reason", "")
            has_soft_reason = pd.notna(raw_soft_reason) and bool(str(raw_soft_reason))
            trusted = (
                evidence_state == "support"
                and status in {"strong_match", "supported_match"}
                and mask_track_status in {"linked", "seeded"}
                and not has_break_reason
                and not has_soft_reason
            )
            if not trusted:
                current_segment = None
                previous_frame = int(row["frame"])
                continue
            gap = previous_frame is not None and int(row["frame"]) - previous_frame > max_gap_frames
            if current_segment is None or gap:
                current_segment = next_segment_id
                next_segment_id += 1
            segment_ids.loc[index] = current_segment
            previous_frame = int(row["frame"])
    output["trusted_mask_segment_id"] = segment_ids
    return output


def split_on_silhouette_conflicts(annotated: pd.DataFrame) -> pd.DataFrame:
    output = annotated.copy().sort_values(["tracklet_id", "frame"])
    fused_ids = pd.Series(-1, index=output.index, dtype=int)
    next_id = 0
    for _, group in output.groupby("tracklet_id", sort=True):
        current_id: int | None = None
        previous_frame: int | None = None
        for index, row in group.iterrows():
            raw_break_reason = row.get("silhouette_break_reason", "")
            if "silhouette_evidence_state" in output.columns:
                conflict = str(row.get("silhouette_evidence_state", "unknown")) == "contradiction"
            else:
                # Backward-compatible behavior for old annotated CSVs.
                conflict = pd.notna(raw_break_reason) and bool(str(raw_break_reason))
            gap = previous_frame is not None and int(row["frame"]) != previous_frame + 1
            if conflict:
                current_id = None
                previous_frame = int(row["frame"])
                continue
            if current_id is None or gap:
                current_id = next_id
                next_id += 1
            fused_ids.loc[index] = current_id
            previous_frame = int(row["frame"])
    output["fused_tracklet_id"] = fused_ids
    output["silhouette_accepted"] = output["fused_tracklet_id"] >= 0
    return output.sort_values(["frame", "tracklet_id"])


def summarize_fused_tracklets(fused: pd.DataFrame) -> pd.DataFrame:
    accepted = fused[fused["fused_tracklet_id"].astype(int) >= 0].copy()
    rows: list[dict[str, object]] = []
    for fused_tracklet_id, group in accepted.groupby("fused_tracklet_id", sort=True):
        statuses = group["silhouette_status"].astype(str)
        states = group.get(
            "silhouette_evidence_state",
            pd.Series("unknown", index=group.index),
        ).astype(str)
        temporal = pd.to_numeric(group["temporal_mask_iou"], errors="coerce")
        rows.append(
            {
                "fused_tracklet_id": int(fused_tracklet_id),
                "source_tracklet_ids": "|".join(
                    str(value) for value in sorted(group["tracklet_id"].astype(int).unique())
                ),
                "frame_start": int(group["frame"].min()),
                "frame_end": int(group["frame"].max()),
                "n_rows": int(len(group)),
                "strong_fraction": float(statuses.eq("strong_match").mean()),
                "supported_fraction": float(
                    statuses.isin(["strong_match", "supported_match"]).mean()
                ),
                "ambiguous_fraction": float(statuses.str.startswith("ambiguous").mean()),
                "no_mask_fraction": float(statuses.eq("no_mask").mean()),
                "support_fraction": float(states.eq("support").mean()),
                "unknown_fraction": float(states.eq("unknown").mean()),
                "contradiction_fraction": float(states.eq("contradiction").mean()),
                "median_mutual_confidence": float(group["mutual_confidence"].median()),
                "median_temporal_mask_iou": float(temporal.median()),
                "q10_temporal_mask_iou": float(temporal.quantile(0.10)),
                "mean_silhouette_evidence_weight": float(
                    group["silhouette_evidence_weight"].mean()
                ),
                "trusted_mask_segment_ids": "|".join(
                    str(value)
                    for value in sorted(
                        group.loc[group["trusted_mask_segment_id"].astype(int) >= 0, "trusted_mask_segment_id"]
                        .astype(int)
                        .unique()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def build_silhouette_link_candidates(annotated: pd.DataFrame) -> pd.DataFrame:
    """Propose adjacent concrete-fragment links within one trusted mask run.

    When the silhouette conflict splitter has already run, candidates are keyed
    by ``fused_tracklet_id`` so they refer to the exact IDs consumed by the 3D
    tracklet graph. The pre-split ``tracklet_id`` remains a supported fallback
    for callers that only need diagnostics.
    """
    def safe_median(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.median(finite)) if finite.size else np.nan

    id_column = (
        "fused_tracklet_id"
        if "fused_tracklet_id" in annotated.columns
        else "tracklet_id"
    )
    trusted = annotated[
        (annotated["trusted_mask_segment_id"].astype(int) >= 0)
        & (pd.to_numeric(annotated[id_column], errors="coerce").fillna(-1).ge(0))
    ].copy()
    rows: list[dict[str, object]] = []
    for segment_id, segment in trusted.groupby("trusted_mask_segment_id", sort=True):
        fragments = []
        for tracklet_id, group in segment.groupby(id_column, sort=True):
            fragments.append(
                {
                    "tracklet_id": int(tracklet_id),
                    "frame_start": int(group["frame"].min()),
                    "frame_end": int(group["frame"].max()),
                    "n_rows": int(len(group)),
                    "mean_weight": float(group["silhouette_evidence_weight"].mean()),
                    "median_confidence": float(group["mutual_confidence"].median()),
                    "median_iou": float(group["temporal_mask_iou"].median()),
                }
            )
        fragments.sort(key=lambda value: (value["frame_start"], value["tracklet_id"]))
        # Only adjacent fragments are proposed. Transitive closure, if valid,
        # belongs to the global graph after each local edge passes 3D checks.
        for left, right in zip(fragments, fragments[1:]):
            overlap = min(left["frame_end"], right["frame_end"]) - max(
                left["frame_start"], right["frame_start"]
            ) + 1
            if left["frame_end"] < right["frame_start"]:
                gap = right["frame_start"] - left["frame_end"] - 1
            elif right["frame_end"] < left["frame_start"]:
                gap = left["frame_start"] - right["frame_end"] - 1
            else:
                gap = -max(1, overlap)
            rows.append(
                {
                    "trusted_mask_segment_id": int(segment_id),
                    "left_tracklet_id": left["tracklet_id"],
                    "right_tracklet_id": right["tracklet_id"],
                    "left_frame_end": left["frame_end"],
                    "right_frame_start": right["frame_start"],
                    "gap_frames": int(gap),
                    "temporal_overlap_frames": int(max(0, overlap)),
                    "silhouette_link_status": "conflict_overlap" if overlap > 0 else "candidate",
                    "support_rows": int(left["n_rows"] + right["n_rows"]),
                    "mean_silhouette_weight": float(
                        0.5 * (left["mean_weight"] + right["mean_weight"])
                    ),
                    "median_mutual_confidence": float(
                        safe_median([left["median_confidence"], right["median_confidence"]])
                    ),
                    "median_temporal_mask_iou": float(
                        safe_median([left["median_iou"], right["median_iou"]])
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="view=tracklets.csv,masks.h5; repeat for each available view",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "silhouette_fusion_summary.json"
    if aggregate_path.exists():
        try:
            summaries = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summaries = {}
    else:
        summaries = {}
    for specification in args.input:
        view, paths = specification.split("=", 1)
        tracklet_path, mask_path = paths.split(",", 1)
        print(f"[fuse] {view}")
        tracklets = pd.read_csv(tracklet_path)
        if args.frame_start is not None:
            tracklets = tracklets[tracklets["frame"] >= int(args.frame_start)]
        if args.frame_end is not None:
            tracklets = tracklets[tracklets["frame"] <= int(args.frame_end)]
        annotated, summary = annotate_tracklets_with_masks(tracklets, mask_path)
        fused = split_on_silhouette_conflicts(annotated)
        fused_summary = summarize_fused_tracklets(fused)
        link_candidates = build_silhouette_link_candidates(fused)
        concrete = fused[fused["silhouette_accepted"].astype(bool)].copy()
        concrete = concrete.rename(columns={"tracklet_id": "source_tracklet_id"})
        concrete["tracklet_id"] = concrete["fused_tracklet_id"].astype(int)
        annotated.to_csv(args.output_dir / f"{view}.silhouette_annotated.csv", index=False)
        fused.to_csv(args.output_dir / f"{view}.silhouette_fused_tracklets.csv", index=False)
        concrete.to_csv(
            args.output_dir / f"{view}.silhouette_concrete_tracklets.csv", index=False
        )
        fused_summary.to_csv(
            args.output_dir / f"{view}.silhouette_tracklet_summary.csv", index=False
        )
        link_candidates.to_csv(
            args.output_dir / f"{view}.silhouette_link_candidates.csv", index=False
        )
        (args.output_dir / f"{view}.silhouette_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries[view] = summary
        # Checkpoint each completed view so a long multi-view mask pass can be
        # resumed without losing already-finished summaries.
        aggregate_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"  statuses={summary['status_counts']} breaks={summary['break_reason_counts']}")


if __name__ == "__main__":
    main()
