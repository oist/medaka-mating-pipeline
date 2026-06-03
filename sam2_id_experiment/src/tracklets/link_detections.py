from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    iter_mask_instances_h5,
    write_mask_instances_h5,
)
from sam2_id_experiment.src.tracklets.mask_to_tracklet import _centroid


@dataclass
class ActiveTrack:
    tracklet_id: int
    frame: int
    mask: np.ndarray
    centroid: tuple[float, float]


def link_mask_detections_h5(
    *,
    input_masks: str | Path,
    output_masks: str | Path,
    max_frame_gap: int = 20,
    max_centroid_distance_px: float = 150.0,
    min_iou: float = 0.01,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    summary_path: str | Path | None = None,
) -> dict:
    instances = list(iter_mask_instances_h5(input_masks))
    linked = link_mask_instances(
        instances,
        max_frame_gap=max_frame_gap,
        max_centroid_distance_px=max_centroid_distance_px,
        min_iou=min_iou,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
    )
    image_shape = tuple(np.asarray(linked[0].mask).shape[:2]) if linked else None
    write_mask_instances_h5(output_masks, linked, image_shape=image_shape)
    tracklet_lengths: dict[int, int] = {}
    for inst in linked:
        tracklet_lengths[int(inst.sam2_track_id)] = tracklet_lengths.get(int(inst.sam2_track_id), 0) + 1
    summary = {
        "input_masks": str(input_masks),
        "output_masks": str(output_masks),
        "n_instances": len(linked),
        "n_tracklets": len(tracklet_lengths),
        "tracklet_lengths": {str(k): int(v) for k, v in sorted(tracklet_lengths.items())},
        "max_frame_gap": int(max_frame_gap),
        "max_centroid_distance_px": float(max_centroid_distance_px),
        "min_iou": float(min_iou),
    }
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def link_mask_instances(
    instances: list[MaskInstance],
    *,
    max_frame_gap: int = 20,
    max_centroid_distance_px: float = 150.0,
    min_iou: float = 0.01,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
) -> list[MaskInstance]:
    by_frame: dict[int, list[MaskInstance]] = {}
    for inst in instances:
        by_frame.setdefault(int(inst.frame), []).append(inst)

    active: dict[int, ActiveTrack] = {}
    next_tracklet_id = 0
    linked: list[MaskInstance] = []
    for frame in sorted(by_frame):
        detections = sorted(by_frame[frame], key=lambda inst: float(inst.score), reverse=True)
        active = {
            tid: track for tid, track in active.items()
            if int(frame) - int(track.frame) <= int(max_frame_gap)
        }
        matches = _greedy_matches(
            detections,
            active,
            max_centroid_distance_px=max_centroid_distance_px,
            min_iou=min_iou,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
        )
        used_detection_indices: set[int] = set()
        frame_linked: list[MaskInstance] = []
        for det_idx, tracklet_id in matches:
            inst = detections[det_idx]
            used_detection_indices.add(det_idx)
            frame_linked.append(_with_tracklet_id(inst, tracklet_id))
            active[tracklet_id] = ActiveTrack(
                tracklet_id=tracklet_id,
                frame=int(frame),
                mask=np.asarray(inst.mask, dtype=bool),
                centroid=_centroid(inst.mask),
            )
        for det_idx, inst in enumerate(detections):
            if det_idx in used_detection_indices:
                continue
            tracklet_id = next_tracklet_id
            next_tracklet_id += 1
            frame_linked.append(_with_tracklet_id(inst, tracklet_id))
            active[tracklet_id] = ActiveTrack(
                tracklet_id=tracklet_id,
                frame=int(frame),
                mask=np.asarray(inst.mask, dtype=bool),
                centroid=_centroid(inst.mask),
            )
        linked.extend(sorted(frame_linked, key=lambda inst: int(inst.instance_id)))
    return linked


def _greedy_matches(
    detections: list[MaskInstance],
    active: dict[int, ActiveTrack],
    *,
    max_centroid_distance_px: float,
    min_iou: float,
    iou_weight: float,
    distance_weight: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for det_idx, det in enumerate(detections):
        det_centroid = _centroid(det.mask)
        for tracklet_id, track in active.items():
            dist = _distance(det_centroid, track.centroid)
            if not np.isfinite(dist) or dist > max_centroid_distance_px:
                continue
            iou = mask_iou(det.mask, track.mask)
            if iou < min_iou:
                continue
            distance_score = 1.0 - min(1.0, dist / max_centroid_distance_px)
            score = float(iou_weight) * iou + float(distance_weight) * distance_score
            candidates.append((score, det_idx, int(tracklet_id)))
    candidates.sort(reverse=True)

    used_detections: set[int] = set()
    used_tracklets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, det_idx, tracklet_id in candidates:
        if det_idx in used_detections or tracklet_id in used_tracklets:
            continue
        used_detections.add(det_idx)
        used_tracklets.add(tracklet_id)
        matches.append((det_idx, tracklet_id))
    return matches


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(np.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _with_tracklet_id(inst: MaskInstance, tracklet_id: int) -> MaskInstance:
    return MaskInstance(
        frame=int(inst.frame),
        instance_id=int(inst.instance_id),
        sam2_track_id=int(tracklet_id),
        mask=np.asarray(inst.mask, dtype=bool),
        bbox=inst.bbox,
        score=float(inst.score),
    )
