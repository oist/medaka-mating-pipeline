"""Build clean three-view source-frame candidates for female-sex GT review.

This module ranks scene quality only.  It never predicts sex, chooses a female,
propagates biological identity, or uses persistent raw SLEAP instance IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from scripts.analysis.female_sex_gt import (normalize_path, stable_json,
                                            visible_node_centroids)
from scripts.analysis.geometry_swap_review import (VIDEO_NAMES,
                                                   IndexedSlpPredictions)
from scripts.common.local_paths import (DEFAULT_PATHS_CONFIG, REPO_ROOT,
                                        load_local_paths)
from scripts.reconstruction_3d.calibration import Calibration

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_INDEX = {view: index for index, view in enumerate(VIEWS)}
SOURCE = "clean_tracklet_three_view"
SCHEMA_VERSION = "clean_three_view_candidates_v2"
REVIEWED_SOURCE_MODE = "reviewed_clean_tracklets"
REFRACTIVE_SOURCE_MODE = "refractive_geometry_segments"
KNOWN_LABEL_STATUSES = {
    "clean",
    "interpolated",
    "missing",
    "unresolved",
    "ambiguous",
    "false_positive",
}
REQUIRED_CLEAN_COLUMNS = {
    "frame",
    "clean_tracklet_id",
    "geometry_source_id",
    "geometry_slot",
    "X",
    "Y",
    "Z",
    "label_status",
    "mapping_resolved",
    "is_interpolated",
}
OPTIONAL_BAD_STATUS_COLUMNS = (
    "is_ambiguous",
    "ambiguous",
    "is_unresolved",
    "unresolved",
    "is_false_positive",
    "false_positive",
)
OUTPUT_COLUMNS = [
    "experiment",
    "frame",
    "candidate_id",
    "source",
    "priority",
    "score",
    "reason",
    "view_hint",
    "clean_tracklet_ids",
    "geometry_ids",
    "geometry_tracklet_map_json",
    "geometry_source_mode",
    "top_visible_pose_count",
    "side1_visible_pose_count",
    "side2_visible_pose_count",
    "top_min_separation_px",
    "side1_min_separation_px",
    "side2_min_separation_px",
    "max_reprojection_distance_px",
    "min_border_distance_px",
    "is_interpolated",
    "temporal_bin",
    "quality_components_json",
    "clean_interval_id",
    "interval_section",
]
REJECTION_RULES = (
    "not_three_simultaneous_clean_tracklets",
    "interpolated_ambiguous_unresolved_or_false_positive",
    "nonfinite_3d_geometry",
    "outside_synchronized_video_or_slp_extent",
    "invalid_calibration_projection",
    "missing_raw_detection_in_a_view",
    "missing_mouth_or_tail_in_a_view",
    "malformed_slp_pose_data",
    "failed_or_over_gate_geometry_association",
    "insufficient_fish_separation_in_a_view",
    "border_adjacent_pose_in_a_view",
)


@dataclass(frozen=True)
class CandidateConfig:
    target_count: int = 300
    min_frame_spacing: int = 166
    match_gate_px: float = 80.0
    min_view_separation_px: float = 40.0
    border_margin_px: float = 20.0
    max_local_jump_mm: float = 8.0
    seed: int = 0
    side_hint_tie_margin: float = 0.05

    def validate(self) -> None:
        if self.target_count < 0:
            raise ValueError("target_count must be non-negative")
        if self.min_frame_spacing < 0:
            raise ValueError("min_frame_spacing must be non-negative")
        if self.match_gate_px <= 0:
            raise ValueError("match_gate_px must be positive")
        if self.min_view_separation_px < 0:
            raise ValueError("min_view_separation_px must be non-negative")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")
        if self.max_local_jump_mm <= 0:
            raise ValueError("max_local_jump_mm must be positive")
        if self.side_hint_tie_margin < 0:
            raise ValueError("side_hint_tie_margin must be non-negative")


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    frame_count: int
    width: int
    height: int
    reported_fps: float


@dataclass
class CandidateGenerationResult:
    candidates: pd.DataFrame
    diagnostics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return normalize_path(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _strict_integer_frames(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{label} must contain finite integer source frames")
    result = numeric.astype(np.int64)
    if (result < 0).any():
        raise ValueError(f"{label} contains negative source frames")
    return result


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    result: list[bool] = []
    for row_number, value in enumerate(values, start=1):
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            result.append(bool(value))
            continue
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
            result.append(bool(value))
            continue
        key = str(value).strip().lower() if pd.notna(value) else ""
        if key not in mapping:
            raise ValueError(
                f"{label} row {row_number} has unknown boolean value {value!r}"
            )
        result.append(mapping[key])
    return pd.Series(result, index=values.index, dtype=bool)


def validate_clean_tracklets(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the authoritative reviewed clean-tracklet row semantics."""
    missing = REQUIRED_CLEAN_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(
            "clean-tracklet CSV is not the authoritative reviewed schema; missing "
            f"required status/provenance columns: {sorted(missing)}. Use "
            "scripts.tracking.export_geometry_swap_tracklets to create "
            "clean_3d_tracklets.csv."
        )
    result = table.copy()
    result["frame"] = _strict_integer_frames(result["frame"], "clean-tracklet frame")
    for column in ("clean_tracklet_id", "geometry_source_id", "geometry_slot"):
        missing_value = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if missing_value.any():
            raise ValueError(f"clean-tracklet column {column!r} contains missing IDs")
        result[column] = result[column].astype(str)

    result["label_status"] = result["label_status"].astype(str).str.strip().str.lower()
    unknown = sorted(set(result["label_status"]) - KNOWN_LABEL_STATUSES)
    if unknown:
        raise ValueError(
            "clean-tracklet label_status contains unknown meanings; refusing to "
            f"infer cleanliness: {unknown}"
        )
    result["mapping_resolved"] = _strict_bool(
        result["mapping_resolved"], "mapping_resolved"
    )
    result["is_interpolated"] = _strict_bool(
        result["is_interpolated"], "is_interpolated"
    )
    for column in OPTIONAL_BAD_STATUS_COLUMNS:
        if column in result:
            result[column] = _strict_bool(result[column], column)

    for column in ("X", "Y", "Z"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    duplicate = result.duplicated(["frame", "geometry_source_id"], keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, ["frame", "geometry_source_id"]].head(5)
        raise ValueError(
            "clean-tracklet CSV has duplicate frame/geometry rows: "
            + examples.to_dict("records").__repr__()
        )
    return result.sort_values(
        ["frame", "geometry_source_id"], kind="stable", ignore_index=True
    )


def _standalone_geometry_id_column(
    geometry: pd.DataFrame, explicit: str | None = None
) -> str:
    if explicit is not None:
        if explicit not in geometry.columns:
            raise ValueError(f"geometry CSV has no requested ID column {explicit!r}")
        return explicit
    for column in (
        "fish_id",
        "geometry_source_id",
        "geometry_slot",
        "geometry_id",
        "final_fish_id",
    ):
        if column in geometry.columns:
            return column
    raise ValueError(
        "geometry CSV needs an ID column: fish_id, geometry_source_id, "
        "geometry_slot, geometry_id, or final_fish_id"
    )


def build_refractive_geometry_tracklets(
    geometry: pd.DataFrame,
    *,
    max_local_jump_mm: float,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build conservative provisional segments from a refractive 3D result.

    These IDs describe contiguous geometry fragments only. They are not
    biological identities and never propagate a female label. Missing,
    interpolated, and both sides of an over-threshold jump break a segment.
    """
    if max_local_jump_mm <= 0:
        raise ValueError("max_local_jump_mm must be positive")
    required = {"frame", "X", "Y", "Z", "is_interpolated"}
    missing = required - set(geometry.columns)
    if missing:
        raise ValueError(
            "direct reconstruction mode requires geometry columns: "
            f"{sorted(required)}; missing {sorted(missing)}"
        )
    result = geometry.copy()
    result["frame"] = _strict_integer_frames(result["frame"], "geometry frame")
    resolved_id_column = _standalone_geometry_id_column(result, id_column)
    missing_ids = result[resolved_id_column].isna() | result[
        resolved_id_column
    ].astype(str).str.strip().eq("")
    if missing_ids.any():
        raise ValueError(f"geometry ID column {resolved_id_column!r} contains missing IDs")
    result["geometry_source_id"] = result[resolved_id_column].astype(str)
    if result.duplicated(["frame", "geometry_source_id"]).any():
        raise ValueError(
            f"geometry CSV has duplicate (frame, {resolved_id_column}) rows"
        )
    geometry_ids = sorted(result["geometry_source_id"].unique())
    if len(geometry_ids) != 3:
        raise ValueError(
            "direct reconstruction mode requires exactly three geometry IDs, "
            f"found {len(geometry_ids)}: {geometry_ids}"
        )
    result["is_interpolated"] = _strict_bool(
        result["is_interpolated"], "geometry is_interpolated"
    )
    for column in ("X", "Y", "Z"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.sort_values(
        ["geometry_source_id", "frame"], kind="stable", inplace=True
    )
    result.reset_index(drop=True, inplace=True)

    xyz = result[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    result["label_status"] = "clean"
    result.loc[~finite, "label_status"] = "missing"
    result.loc[result["is_interpolated"], "label_status"] = "interpolated"

    frames = result["frame"].to_numpy(dtype=np.int64)
    geometry_id_values = result["geometry_source_id"].to_numpy(dtype=str)
    base_valid = finite & ~result["is_interpolated"].to_numpy(dtype=bool)
    same_geometry = np.zeros(len(result), dtype=bool)
    consecutive = np.zeros(len(result), dtype=bool)
    distances = np.full(len(result), np.nan, dtype=float)
    if len(result) > 1:
        same_geometry[1:] = geometry_id_values[1:] == geometry_id_values[:-1]
        consecutive[1:] = same_geometry[1:] & (frames[1:] == frames[:-1] + 1)
        comparable = consecutive[1:] & base_valid[1:] & base_valid[:-1]
        deltas = xyz[1:] - xyz[:-1]
        comparable_positions = np.flatnonzero(comparable) + 1
        distances[comparable_positions] = np.linalg.norm(
            deltas[comparable], axis=1
        )
    jump_current = np.isfinite(distances) & (distances > max_local_jump_mm)
    jump_boundary = jump_current.copy()
    if len(result) > 1:
        jump_boundary[:-1] |= jump_current[1:]
    jump_distances = distances[jump_current]
    result.loc[jump_boundary, "label_status"] = "unresolved"
    result["mapping_resolved"] = result["label_status"].eq("clean")
    result["geometry_slot"] = result["geometry_source_id"]

    clean_rows = result["label_status"].eq("clean")
    previous_clean = clean_rows.shift(fill_value=False)
    previous_geometry = result["geometry_source_id"].shift(fill_value="")
    previous_frame = result["frame"].shift(fill_value=-2)
    segment_start = clean_rows & (
        ~previous_clean
        | result["geometry_source_id"].ne(previous_geometry)
        | result["frame"].ne(previous_frame + 1)
    )
    segment_number = segment_start.groupby(result["geometry_source_id"]).cumsum()
    valid_tracklet_ids = (
        "R3D:"
        + result["geometry_source_id"]
        + ":S"
        + segment_number.astype(np.int64).astype(str).str.zfill(6)
    )
    invalid_tracklet_ids = (
        "R3D:"
        + result["geometry_source_id"]
        + ":INVALID:f"
        + result["frame"].astype(str).str.zfill(9)
    )
    result["clean_tracklet_id"] = valid_tracklet_ids.where(
        clean_rows, invalid_tracklet_ids
    )
    segment_counts = {
        str(key): int(value)
        for key, value in segment_number.groupby(result["geometry_source_id"]).max().items()
    }
    result["direct_geometry_segment"] = True

    status_counts = result["label_status"].value_counts().sort_index()
    diagnostics = {
        "source_mode": REFRACTIVE_SOURCE_MODE,
        "geometry_id_column": resolved_id_column,
        "geometry_ids": geometry_ids,
        "max_local_jump_mm": float(max_local_jump_mm),
        "over_threshold_jump_count": int(len(jump_distances)),
        "over_threshold_jump_distance_distribution_mm": _distribution(
            pd.Series(jump_distances, dtype=float)
        ),
        "segment_counts_by_geometry_id": segment_counts,
        "label_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
    }
    return result, diagnostics


def default_reconstruction_provenance_path(geometry_csv: Path) -> Path:
    suffix = "_3d"
    stem = geometry_csv.stem
    base = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return geometry_csv.with_name(base + "_reconstruction_provenance.json")


def validate_refractive_reconstruction_provenance(
    payload: Mapping[str, Any], calibration: Any
) -> dict[str, Any]:
    """Verify that geometry and reprojection use the same refractive backend."""
    triangulation = payload.get("triangulation", {})
    projection = payload.get("projection", {})
    if not isinstance(triangulation, Mapping) or not isinstance(projection, Mapping):
        raise ValueError("reconstruction provenance lacks triangulation/projection data")
    triangulation_backend = str(triangulation.get("backend", "")).lower()
    projection_backend = str(projection.get("backend", "")).lower()
    if triangulation_backend != "refractive" or projection_backend != "refractive":
        raise ValueError(
            "direct reconstruction mode requires refractive triangulation and "
            f"projection provenance, got {triangulation_backend!r} and "
            f"{projection_backend!r}"
        )
    score_unit = str(triangulation.get("score_unit", "")).lower()
    if score_unit != "mm^2":
        raise ValueError(
            "refractive reconstruction provenance must declare score_unit "
            f"'mm^2', got {score_unit!r}"
        )
    camera_order = payload.get("camera_order")
    if camera_order is None:
        camera_order = triangulation.get("provenance", {}).get("camera_order")
    if list(camera_order or []) != ["cam0", "cam1", "cam2"]:
        raise ValueError(
            "reconstruction provenance camera_order must be "
            "['cam0', 'cam1', 'cam2']"
        )
    if str(getattr(calibration, "backend_name", "")).lower() != "refractive":
        raise ValueError(
            "direct reconstruction mode requires a refractive calibration for "
            "geometry-to-SLEAP reprojection"
        )

    reconstruction_files = triangulation.get("provenance", {}).get(
        "model_files", {}
    )
    active_files = getattr(calibration, "provenance", {}).get("model_files", {})
    if not isinstance(reconstruction_files, Mapping) or not reconstruction_files:
        raise ValueError("reconstruction provenance has no calibration model files")
    if not isinstance(active_files, Mapping) or not active_files:
        raise ValueError("active calibration has no model-file provenance")
    reconstruction_names = {
        str(camera): Path(str(path)).name
        for camera, path in reconstruction_files.items()
    }
    active_names = {
        str(camera): Path(str(path)).name for camera, path in active_files.items()
    }
    if reconstruction_names != active_names:
        raise ValueError(
            "reconstruction and active calibration model files disagree: "
            f"{reconstruction_names} != {active_names}"
        )
    return {
        "triangulation_backend": triangulation_backend,
        "projection_backend": projection_backend,
        "score_unit": score_unit,
        "camera_order": list(camera_order),
        "model_files": reconstruction_files,
        "model_file_names": reconstruction_names,
    }


def _row_is_review_clean(table: pd.DataFrame) -> pd.Series:
    clean = (
        table["label_status"].eq("clean")
        & table["mapping_resolved"]
        & ~table["is_interpolated"]
    )
    for column in OPTIONAL_BAD_STATUS_COLUMNS:
        if column in table:
            clean &= ~table[column]
    return clean


def _geometry_id_column(clean: pd.DataFrame, geometry: pd.DataFrame) -> str:
    candidates = (
        "geometry_source_id",
        "fish_id",
        "geometry_slot",
        "geometry_id",
        "final_fish_id",
    )
    clean_keys = set(
        zip(clean["frame"].astype(int), clean["geometry_source_id"].astype(str))
    )
    for column in candidates:
        if column not in geometry:
            continue
        values = geometry[column].astype(str)
        keys = set(zip(geometry["frame"].astype(int), values))
        if clean_keys <= keys:
            return column
    raise ValueError(
        "geometry CSV cannot be reconciled with clean-tracklet geometry_source_id; "
        f"tried ID columns {[column for column in candidates if column in geometry]}"
    )


def reconcile_geometry(
    clean_tracklets: pd.DataFrame, geometry: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Attach exact immutable geometry rows and reject cross-run mismatches."""
    required = {"frame", "X", "Y", "Z"}
    missing = required - set(geometry.columns)
    if missing:
        raise ValueError(f"geometry CSV is missing columns: {sorted(missing)}")
    source = geometry.copy()
    source["frame"] = _strict_integer_frames(source["frame"], "geometry frame")
    id_column = _geometry_id_column(clean_tracklets, source)
    source["_geometry_join_id"] = source[id_column].astype(str)
    if source.duplicated(["frame", "_geometry_join_id"]).any():
        raise ValueError(f"geometry CSV has duplicate (frame, {id_column}) rows")
    for column in ("X", "Y", "Z"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    merged = clean_tracklets.merge(
        source[["frame", "_geometry_join_id", "X", "Y", "Z"]].rename(
            columns={
                "_geometry_join_id": "geometry_source_id",
                "X": "geometry_X",
                "Y": "geometry_Y",
                "Z": "geometry_Z",
            }
        ),
        on=["frame", "geometry_source_id"],
        how="left",
        validate="one_to_one",
        indicator="_geometry_merge_status",
    )
    missing_geometry_rows = merged["_geometry_merge_status"].eq("left_only")
    if missing_geometry_rows.any():
        bad = merged.loc[
            missing_geometry_rows,
            ["frame", "geometry_source_id"],
        ].head(5)
        raise ValueError(
            "geometry CSV is missing clean-tracklet source rows: "
            + repr(bad.to_dict("records"))
        )
    merged.drop(columns="_geometry_merge_status", inplace=True)

    clean_xyz = merged[["X", "Y", "Z"]].to_numpy(dtype=float)
    geometry_xyz = merged[["geometry_X", "geometry_Y", "geometry_Z"]].to_numpy(
        dtype=float
    )
    both_finite = np.isfinite(clean_xyz).all(axis=1) & np.isfinite(geometry_xyz).all(axis=1)
    disagree = both_finite & ~np.isclose(
        clean_xyz, geometry_xyz, rtol=1e-7, atol=1e-6
    ).all(axis=1)
    if disagree.any():
        bad = merged.loc[disagree, ["frame", "geometry_source_id"]].head(5)
        raise ValueError(
            "clean-tracklet coordinates disagree with the immutable geometry CSV; "
            "the files may come from different runs: "
            + repr(bad.to_dict("records"))
        )
    merged[["X", "Y", "Z"]] = geometry_xyz
    return merged.reset_index(drop=True), id_column


def _open_video_metadata(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open source video: {path}")
    try:
        metadata = VideoMetadata(
            path=Path(path),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            reported_fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    if metadata.frame_count <= 0 or metadata.width <= 0 or metadata.height <= 0:
        raise ValueError(f"video has invalid dimensions/frame count: {path}")
    return metadata


def _slp_extent(source: Any) -> tuple[int, int, int]:
    indices = np.asarray(getattr(source, "_frame_indices", []), dtype=np.int64)
    if not len(indices):
        return (0, -1, 0)
    return int(indices.min()), int(indices.max()), int(len(indices))


def _complete_pose_array(instances: np.ndarray | None) -> tuple[np.ndarray, int, int]:
    if instances is None:
        return np.empty((0, 2, 2), dtype=float), 0, 0
    array = np.asarray(instances, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 2 or array.shape[1] < 2:
        raise ValueError(
            "indexed SLP detections must have shape (instances, nodes>=2, 2)"
        )
    complete = np.isfinite(array[:, :2, :]).all(axis=(1, 2))
    return array[complete, :2, :], int(len(array)), int(complete.sum())


def _match_projected_geometry(
    projected: np.ndarray,
    instances: np.ndarray | None,
    gate_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    """Return three matched mouth/tail poses using a gated global assignment."""
    poses, raw_count, visible_count = _complete_pose_array(instances)
    if len(poses) < 3:
        return None, None, raw_count, visible_count
    centers = visible_node_centroids(poses)
    distances = np.linalg.norm(projected[:, None, :] - centers[None, :, :], axis=2)
    penalty = max(float(np.nanmax(distances)) if distances.size else 0.0, gate_px) + 1e6
    gated_cost = np.where(np.isfinite(distances) & (distances <= gate_px), distances, penalty)
    geometry_rows, pose_columns = linear_sum_assignment(gated_cost)
    if len(geometry_rows) != 3:
        return None, None, raw_count, visible_count
    order = np.argsort(geometry_rows, kind="stable")
    pose_columns = pose_columns[order]
    matched_distances = distances[np.arange(3), pose_columns]
    if not np.isfinite(matched_distances).all() or (matched_distances > gate_px).any():
        return None, None, raw_count, visible_count
    return poses[pose_columns], matched_distances, raw_count, visible_count


def _minimum_pair_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return math.nan
    distances = [
        float(np.linalg.norm(points[left] - points[right]))
        for left in range(len(points))
        for right in range(left + 1, len(points))
    ]
    return min(distances)


def _border_distance(poses: np.ndarray, width: int, height: int) -> float:
    points = poses.reshape(-1, 2)
    distances = np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            (width - 1) - points[:, 0],
            (height - 1) - points[:, 1],
        )
    )
    return float(np.min(distances))


def _clipped_quality(value: float, low: float, span: float) -> float:
    if span <= 0:
        return 1.0 if value >= low else 0.0
    return float(np.clip((value - low) / span, 0.0, 1.0))


def _quality_components(
    *,
    max_reprojection: float,
    min_separation: float,
    min_border: float,
    boundary_distance: int,
    config: CandidateConfig,
) -> tuple[float, dict[str, float]]:
    components = {
        "cross_view_pose_completeness": 1.0,
        "reprojection": float(
            np.clip(1.0 - max_reprojection / config.match_gate_px, 0.0, 1.0)
        ),
        "separation": _clipped_quality(
            min_separation,
            config.min_view_separation_px,
            max(2.0 * config.min_view_separation_px, 1.0),
        ),
        "border_clearance": _clipped_quality(
            min_border,
            config.border_margin_px,
            max(3.0 * config.border_margin_px, 1.0),
        ),
        "clean_boundary_clearance": float(
            np.clip(
                boundary_distance / max(config.min_frame_spacing, 1), 0.0, 1.0
            )
        ),
    }
    weights = {
        "cross_view_pose_completeness": 0.15,
        "reprojection": 0.35,
        "separation": 0.25,
        "border_clearance": 0.15,
        "clean_boundary_clearance": 0.10,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    components["score"] = float(score)
    return float(score), components


def _side_strength(
    *,
    max_reprojection: float,
    separation: float,
    border: float,
    config: CandidateConfig,
) -> float:
    reprojection = float(
        np.clip(1.0 - max_reprojection / config.match_gate_px, 0.0, 1.0)
    )
    separation_quality = _clipped_quality(
        separation,
        config.min_view_separation_px,
        max(2.0 * config.min_view_separation_px, 1.0),
    )
    border_quality = _clipped_quality(
        border,
        config.border_margin_px,
        max(3.0 * config.border_margin_px, 1.0),
    )
    return 0.5 * reprojection + 0.3 * separation_quality + 0.2 * border_quality


def _view_hint(side1_strength: float, side2_strength: float, tie_margin: float) -> str:
    if abs(side1_strength - side2_strength) <= tie_margin:
        return "both_sides"
    return "side1_cam" if side1_strength > side2_strength else "side2_cam"


def _three_fish_clean(group: pd.DataFrame) -> bool:
    return (
        len(group) == 3
        and group["clean_tracklet_id"].nunique() == 3
        and group["geometry_source_id"].nunique() == 3
        and group["geometry_slot"].nunique() == 3
    )


def _build_clean_intervals(clean: pd.DataFrame) -> tuple[dict[int, str], dict[str, dict[str, Any]]]:
    qualifying: list[tuple[int, tuple[str, ...]]] = []
    for frame, group in clean.groupby("frame", sort=True):
        if not _three_fish_clean(group):
            continue
        if not _row_is_review_clean(group).all():
            continue
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            continue
        qualifying.append(
            (int(frame), tuple(sorted(group["clean_tracklet_id"].astype(str))))
        )

    frame_to_interval: dict[int, str] = {}
    intervals: dict[str, dict[str, Any]] = {}
    previous_frame: int | None = None
    previous_key: tuple[str, ...] | None = None
    interval_id = ""
    for frame, key in qualifying:
        if previous_frame is None or frame != previous_frame + 1 or key != previous_key:
            interval_id = f"I{len(intervals) + 1:06d}"
            intervals[interval_id] = {
                "interval_id": interval_id,
                "start_frame": frame,
                "end_frame": frame,
                "clean_tracklet_ids": list(key),
            }
        intervals[interval_id]["end_frame"] = frame
        frame_to_interval[frame] = interval_id
        previous_frame, previous_key = frame, key
    for interval in intervals.values():
        interval["n_source_frames"] = (
            int(interval["end_frame"]) - int(interval["start_frame"]) + 1
        )
    return frame_to_interval, intervals


def _position_section(frame: int, start: int, end: int) -> str:
    if end <= start:
        return "middle"
    fraction = (frame - start) / (end - start)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction > 2.0 / 3.0:
        return "late"
    return "middle"


def _recording_bin(frame: int, start: int, end: int) -> str:
    return _position_section(frame, start, end)


def _seed_key(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _interval_spread_order(table: pd.DataFrame, seed: int) -> list[str]:
    interval_centers = (
        table.groupby("clean_interval_id", sort=True)["frame"].agg(["min", "max"])
    )
    if interval_centers.empty:
        return []
    centers = {
        str(index): (float(row["min"]) + float(row["max"])) / 2.0
        for index, row in interval_centers.iterrows()
    }
    recording_center = (float(table["frame"].min()) + float(table["frame"].max())) / 2.0
    remaining = set(centers)
    first = min(
        remaining,
        key=lambda item: (
            abs(centers[item] - recording_center),
            _seed_key(seed, "interval", item),
            item,
        ),
    )
    order = [first]
    remaining.remove(first)
    while remaining:
        next_id = min(
            remaining,
            key=lambda item: (
                -min(abs(centers[item] - centers[chosen]) for chosen in order),
                _seed_key(seed, "interval", item),
                item,
            ),
        )
        order.append(next_id)
        remaining.remove(next_id)
    return order


def _compatible(frame: int, selected_frames: list[int], spacing: int) -> bool:
    return all(abs(frame - selected) >= spacing for selected in selected_frames)


def select_temporally_diverse(
    eligible: pd.DataFrame, config: CandidateConfig
) -> pd.DataFrame:
    """Select coverage anchors, then fill fairly across clean intervals."""
    if eligible.empty or config.target_count == 0:
        return eligible.iloc[0:0].copy()
    target = min(config.target_count, len(eligible))
    interval_order = _interval_spread_order(eligible, config.seed)
    by_interval = {
        interval_id: eligible[eligible["clean_interval_id"].eq(interval_id)].copy()
        for interval_id in interval_order
    }
    selected_indices: list[int] = []
    selected_frames: list[int] = []
    priorities: dict[int, int] = {}

    def ranked(table: pd.DataFrame, phase: str) -> list[int]:
        records = table.assign(
            _seed_order=[
                _seed_key(config.seed, phase, int(frame)) for frame in table["frame"]
            ]
        ).sort_values(
            ["score", "_seed_order", "frame"],
            ascending=[False, True, True],
            kind="stable",
        )
        return [int(index) for index in records.index]

    # One middle anchor per interval first spreads even a small budget over time.
    # Early and late anchors follow for substantial intervals where NMS permits.
    for section, priority in (("middle", 3), ("early", 2), ("late", 2)):
        for interval_id in interval_order:
            if len(selected_indices) >= target:
                break
            candidates = by_interval[interval_id]
            candidates = candidates[candidates["interval_section"].eq(section)]
            for index in ranked(candidates, f"anchor:{section}:{interval_id}"):
                frame = int(eligible.loc[index, "frame"])
                if _compatible(frame, selected_frames, config.min_frame_spacing):
                    selected_indices.append(index)
                    selected_frames.append(frame)
                    priorities[index] = priority
                    break

    # Round-robin fill prevents the longest interval from consuming the budget.
    remaining = {
        interval_id: ranked(table, f"fill:{interval_id}")
        for interval_id, table in by_interval.items()
    }
    while len(selected_indices) < target:
        progress = False
        for interval_id in interval_order:
            queue = remaining[interval_id]
            while queue and queue[0] in priorities:
                queue.pop(0)
            chosen: int | None = None
            while queue:
                index = queue.pop(0)
                frame = int(eligible.loc[index, "frame"])
                if _compatible(frame, selected_frames, config.min_frame_spacing):
                    chosen = index
                    break
            if chosen is not None:
                selected_indices.append(chosen)
                selected_frames.append(int(eligible.loc[chosen, "frame"]))
                priorities[chosen] = 1
                progress = True
            if len(selected_indices) >= target:
                break
        if not progress:
            break

    selected = eligible.loc[selected_indices].copy()
    selected["priority"] = [priorities[int(index)] for index in selected.index]
    return selected.sort_values(
        ["priority", "score", "frame"],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[::-1].reset_index(drop=True)


def generate_candidates(
    *,
    experiment: str,
    clean_tracklets: pd.DataFrame,
    geometry: pd.DataFrame,
    calibration: Any,
    slp_predictions: Mapping[str, Any],
    videos: Mapping[str, VideoMetadata],
    config: CandidateConfig,
    source_mode: str = REVIEWED_SOURCE_MODE,
    source_diagnostics: Mapping[str, Any] | None = None,
    show_progress: bool = False,
) -> CandidateGenerationResult:
    """Generate and rank clean synchronized three-view candidate frames."""
    config.validate()
    missing_views = set(VIEWS) - set(slp_predictions)
    missing_videos = set(VIEWS) - set(videos)
    if missing_views or missing_videos:
        raise ValueError(
            f"all three views are required; missing SLP={sorted(missing_views)}, "
            f"video={sorted(missing_videos)}"
        )
    clean = validate_clean_tracklets(clean_tracklets)
    clean, geometry_id_column = reconcile_geometry(clean, geometry)
    frame_to_interval, interval_details = _build_clean_intervals(clean)

    # Project all finite geometry rows in one public-API batch. Full recordings
    # contain roughly one million rows, so avoiding one calibration call per
    # source frame materially reduces production runtime.
    all_xyz = clean[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite_geometry_rows = np.isfinite(all_xyz).all(axis=1)
    all_projected = np.full((len(clean), 3, 2), np.nan, dtype=float)
    if finite_geometry_rows.any():
        projected_finite = np.asarray(
            calibration.project_world(all_xyz[finite_geometry_rows]), dtype=float
        )
        expected_shape = (int(finite_geometry_rows.sum()), 3, 2)
        if projected_finite.shape != expected_shape:
            raise ValueError(
                "calibration.project_world returned unexpected shape "
                f"{projected_finite.shape}; expected {expected_shape}"
            )
        all_projected[finite_geometry_rows] = projected_finite

    slp_extents = {view: _slp_extent(slp_predictions[view]) for view in VIEWS}
    common_start = max([0, *(extent[0] for extent in slp_extents.values())])
    common_end = min(
        [
            *(videos[view].frame_count - 1 for view in VIEWS),
            *(extent[1] for extent in slp_extents.values()),
        ]
    )
    rejection_counts: Counter[str] = Counter({rule: 0 for rule in REJECTION_RULES})
    eligible_rows: list[dict[str, Any]] = []

    frame_groups = clean.groupby("frame", sort=True)
    frame_iterator = tqdm(
        frame_groups,
        total=int(clean["frame"].nunique()),
        desc="3-view candidate QC",
        unit="frame",
        disable=not show_progress,
    )
    for frame, group in frame_iterator:
        frame = int(frame)
        if not _three_fish_clean(group):
            rejection_counts["not_three_simultaneous_clean_tracklets"] += 1
            continue
        if not _row_is_review_clean(group).all():
            rejection_counts["interpolated_ambiguous_unresolved_or_false_positive"] += 1
            continue
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            rejection_counts["nonfinite_3d_geometry"] += 1
            continue
        if frame < common_start or frame > common_end:
            rejection_counts["outside_synchronized_video_or_slp_extent"] += 1
            continue

        ordered = group.sort_values("geometry_source_id", kind="stable")
        projected = all_projected[ordered.index.to_numpy(dtype=int)]
        if projected.shape != (3, 3, 2) or not np.isfinite(projected).all():
            rejection_counts["invalid_calibration_projection"] += 1
            continue

        view_matches: dict[str, np.ndarray] = {}
        view_distances: dict[str, np.ndarray] = {}
        raw_counts: dict[str, int] = {}
        visible_counts: dict[str, int] = {}
        rejection: str | None = None
        for view in VIEWS:
            instances = slp_predictions[view].get(frame, default=None)
            try:
                poses, distances, raw_count, visible_count = _match_projected_geometry(
                    projected[:, VIEW_INDEX[view], :], instances, config.match_gate_px
                )
            except ValueError:
                rejection = "malformed_slp_pose_data"
                break
            raw_counts[view] = raw_count
            visible_counts[view] = visible_count
            if raw_count == 0:
                rejection = "missing_raw_detection_in_a_view"
                break
            if visible_count < 3:
                rejection = "missing_mouth_or_tail_in_a_view"
                break
            if poses is None or distances is None:
                rejection = "failed_or_over_gate_geometry_association"
                break
            view_matches[view] = poses
            view_distances[view] = distances
        if rejection is not None:
            rejection_counts[rejection] += 1
            continue

        separations = {
            view: _minimum_pair_distance(view_matches[view].mean(axis=1))
            for view in VIEWS
        }
        if any(
            separation < config.min_view_separation_px
            for separation in separations.values()
        ):
            rejection_counts["insufficient_fish_separation_in_a_view"] += 1
            continue
        borders = {
            view: _border_distance(
                view_matches[view], videos[view].width, videos[view].height
            )
            for view in VIEWS
        }
        if any(border < config.border_margin_px for border in borders.values()):
            rejection_counts["border_adjacent_pose_in_a_view"] += 1
            continue

        interval_id = frame_to_interval.get(frame)
        if interval_id is None:
            raise RuntimeError(f"clean frame {frame} has no clean interval")
        interval = interval_details[interval_id]
        boundary_distance = min(
            frame - int(interval["start_frame"]), int(interval["end_frame"]) - frame
        )
        max_reprojection = max(float(values.max()) for values in view_distances.values())
        min_separation = min(separations.values())
        min_border = min(borders.values())
        score, components = _quality_components(
            max_reprojection=max_reprojection,
            min_separation=min_separation,
            min_border=min_border,
            boundary_distance=boundary_distance,
            config=config,
        )
        side_strengths = {
            view: _side_strength(
                max_reprojection=float(view_distances[view].max()),
                separation=separations[view],
                border=borders[view],
                config=config,
            )
            for view in ("side1_cam", "side2_cam")
        }
        components.update(
            {
                "side1_strength": side_strengths["side1_cam"],
                "side2_strength": side_strengths["side2_cam"],
                "boundary_distance_frames": float(boundary_distance),
            }
        )
        reason = (
            "complete three-view mouth/tail poses; globally matched clean geometry; "
            f"max reprojection {max_reprojection:.2f}px; minimum separation "
            f"{min_separation:.2f}px; minimum border clearance {min_border:.2f}px"
        )
        geometry_tracklet_map = {
            str(row["geometry_source_id"]): str(row["clean_tracklet_id"])
            for _, row in ordered.iterrows()
        }
        eligible_rows.append(
            {
                "experiment": experiment,
                "frame": frame,
                "source": SOURCE,
                "score": score,
                "reason": reason,
                "view_hint": _view_hint(
                    side_strengths["side1_cam"],
                    side_strengths["side2_cam"],
                    config.side_hint_tie_margin,
                ),
                "clean_tracklet_ids": ";".join(
                    sorted(ordered["clean_tracklet_id"].astype(str))
                ),
                "geometry_ids": ";".join(
                    sorted(ordered["geometry_source_id"].astype(str))
                ),
                "geometry_tracklet_map_json": stable_json(geometry_tracklet_map),
                "geometry_source_mode": source_mode,
                "top_visible_pose_count": visible_counts["top_cam"],
                "side1_visible_pose_count": visible_counts["side1_cam"],
                "side2_visible_pose_count": visible_counts["side2_cam"],
                "top_min_separation_px": separations["top_cam"],
                "side1_min_separation_px": separations["side1_cam"],
                "side2_min_separation_px": separations["side2_cam"],
                "max_reprojection_distance_px": max_reprojection,
                "min_border_distance_px": min_border,
                "is_interpolated": False,
                "temporal_bin": _recording_bin(frame, common_start, common_end),
                "quality_components_json": stable_json(components),
                "clean_interval_id": interval_id,
                "interval_section": _position_section(
                    frame, int(interval["start_frame"]), int(interval["end_frame"])
                ),
            }
        )

    eligible = pd.DataFrame(eligible_rows)
    if eligible.empty:
        eligible = pd.DataFrame(
            columns=[
                column
                for column in OUTPUT_COLUMNS
                if column not in {"candidate_id", "priority"}
            ]
        )
    selected = select_temporally_diverse(eligible, config)
    if not selected.empty:
        selected["candidate_id"] = [
            f"{experiment}:clean3v:f{int(frame):09d}" for frame in selected["frame"]
        ]
        selected = selected.sort_values(
            ["priority", "score", "frame"],
            ascending=[False, False, True],
            kind="stable",
            ignore_index=True,
        )
    for column in OUTPUT_COLUMNS:
        if column not in selected:
            selected[column] = pd.Series(dtype=object)
    selected = selected[OUTPUT_COLUMNS]

    for interval_id, details in interval_details.items():
        interval_eligible = eligible[eligible["clean_interval_id"].eq(interval_id)]
        interval_selected = selected[selected["clean_interval_id"].eq(interval_id)]
        details["eligible_frames"] = int(len(interval_eligible))
        details["selected_frames"] = int(len(interval_selected))
        details["selected_sections"] = sorted(
            interval_selected["interval_section"].astype(str).unique().tolist()
        )

    diagnostics = {
        "geometry_source_mode": source_mode,
        "geometry_source_diagnostics": dict(source_diagnostics or {}),
        "geometry_id_column": geometry_id_column,
        "video_extents": {
            view: {
                "frame_start": 0,
                "frame_end": videos[view].frame_count - 1,
                "frame_count": videos[view].frame_count,
                "width": videos[view].width,
                "height": videos[view].height,
                "reported_fps": videos[view].reported_fps,
            }
            for view in VIEWS
        },
        "slp_extents": {
            view: {
                "indexed_frame_start": slp_extents[view][0],
                "indexed_frame_end": slp_extents[view][1],
                "indexed_frame_count": slp_extents[view][2],
            }
            for view in VIEWS
        },
        "common_source_frame_extent": {
            "frame_start": common_start,
            "frame_end": common_end,
        },
        "total_examined_frames": int(clean["frame"].nunique()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "eligible_frame_count": int(len(eligible)),
        "selected_candidate_count": int(len(selected)),
        "eligible_score_distribution": _distribution(eligible["score"]),
        "clean_intervals": list(interval_details.values()),
    }
    return CandidateGenerationResult(selected, diagnostics)


def _distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    quantiles = numeric.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(numeric.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(path: Path, *, hash_limit_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": normalize_path(path),
        "size_bytes": int(stat.st_size),
        "modified_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
    }
    if stat.st_size <= hash_limit_bytes:
        result["sha256"] = _sha256(path)
    else:
        result["sha256"] = None
        result["hash_note"] = f"omitted because file exceeds {hash_limit_bytes} bytes"
    return result


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolved_thresholds(config: CandidateConfig) -> dict[str, Any]:
    return {
        "target_count": config.target_count,
        "min_frame_spacing": config.min_frame_spacing,
        "match_gate_px": config.match_gate_px,
        "min_view_separation_px": config.min_view_separation_px,
        "border_margin_px": config.border_margin_px,
        "max_local_jump_mm": config.max_local_jump_mm,
        "seed": config.seed,
        "side_hint_tie_margin": config.side_hint_tie_margin,
        "quality_weights": {
            "cross_view_pose_completeness": 0.15,
            "reprojection": 0.35,
            "separation": 0.25,
            "border_clearance": 0.15,
            "clean_boundary_clearance": 0.10,
        },
        "sharpness_component": "not computed; avoids decoding every remote source frame",
    }


def run_from_paths(
    *,
    experiment: str,
    clean_tracklets_csv: Path | None,
    geometry_csv: Path,
    reconstruction_provenance_json: Path | None,
    geometry_id_column: str | None,
    slp_dir: Path,
    video_dir: Path,
    calibration_model_dir: Path,
    calibration_backend: str,
    output_csv: Path,
    summary_json: Path,
    config: CandidateConfig,
    show_progress: bool = True,
) -> dict[str, Any]:
    source_paths = [geometry_csv]
    if clean_tracklets_csv is not None:
        source_paths.append(clean_tracklets_csv)
    else:
        reconstruction_provenance_json = (
            reconstruction_provenance_json
            or default_reconstruction_provenance_path(geometry_csv)
        )
        source_paths.append(reconstruction_provenance_json)
    video_paths = {view: video_dir / view / VIDEO_NAMES[view] for view in VIEWS}
    slp_paths = {view: slp_dir / f"{view}.slp" for view in VIEWS}
    source_paths.extend(video_paths.values())
    source_paths.extend(slp_paths.values())
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required read-only inputs:\n  " + "\n  ".join(missing))
    if not calibration_model_dir.is_dir():
        raise FileNotFoundError(
            f"calibration model directory does not exist: {calibration_model_dir}"
        )
    normalized_sources = {normalize_path(path) for path in source_paths}
    if normalize_path(output_csv) == normalize_path(summary_json):
        raise ValueError("--output-csv and --summary-json must be different paths")
    for output in (output_csv, summary_json):
        if normalize_path(output) in normalized_sources:
            raise ValueError(f"refusing to overwrite an input file: {output}")

    geometry = pd.read_csv(geometry_csv, low_memory=False)
    videos = {view: _open_video_metadata(path) for view, path in video_paths.items()}
    calibration = Calibration(calibration_model_dir, backend=calibration_backend)
    reconstruction_provenance: Mapping[str, Any] | None = None
    provenance_validation: dict[str, Any] | None = None
    if clean_tracklets_csv is None:
        if reconstruction_provenance_json is None:
            raise RuntimeError("direct mode did not resolve reconstruction provenance")
        reconstruction_provenance = json.loads(
            reconstruction_provenance_json.read_text(encoding="utf-8")
        )
        if not isinstance(reconstruction_provenance, Mapping):
            raise ValueError("reconstruction provenance JSON must contain an object")
        provenance_validation = validate_refractive_reconstruction_provenance(
            reconstruction_provenance, calibration
        )
        clean, source_diagnostics = build_refractive_geometry_tracklets(
            geometry,
            max_local_jump_mm=config.max_local_jump_mm,
            id_column=geometry_id_column,
        )
        source_mode = REFRACTIVE_SOURCE_MODE
        source_diagnostics["reconstruction_provenance_validation"] = (
            provenance_validation
        )
    else:
        clean = pd.read_csv(clean_tracklets_csv, low_memory=False)
        source_mode = REVIEWED_SOURCE_MODE
        source_diagnostics = {"source_mode": source_mode}
    slps: dict[str, IndexedSlpPredictions] = {}
    try:
        slps = {
            view: IndexedSlpPredictions(path, node_names=("mouth", "tail"))
            for view, path in slp_paths.items()
        }
        result = generate_candidates(
            experiment=experiment,
            clean_tracklets=clean,
            geometry=geometry,
            calibration=calibration,
            slp_predictions=slps,
            videos=videos,
            config=config,
            source_mode=source_mode,
            source_diagnostics=source_diagnostics,
            show_progress=show_progress,
        )
    finally:
        for source in slps.values():
            source.close()

    _atomic_write_csv(result.candidates, output_csv)
    inputs = {
        "clean_tracklets_csv": (
            _file_provenance(clean_tracklets_csv)
            if clean_tracklets_csv is not None
            else None
        ),
        "geometry_csv": _file_provenance(geometry_csv),
        "reconstruction_provenance_json": (
            _file_provenance(reconstruction_provenance_json)
            if reconstruction_provenance_json is not None
            else None
        ),
        "slp": {view: _file_provenance(path) for view, path in slp_paths.items()},
        "videos": {view: _file_provenance(path) for view, path in video_paths.items()},
        "calibration_model_dir": normalize_path(calibration_model_dir),
    }
    temporal_coverage = {
        str(key): int(value)
        for key, value in result.candidates["temporal_bin"].value_counts().sort_index().items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "software": {
            "module": "scripts.analysis.build_female_sex_gt_candidates",
            "version": SCHEMA_VERSION,
            "git_commit": _git_commit(),
        },
        "created_at_utc": _utc_now(),
        "experiment": experiment,
        "inputs": inputs,
        "output_csv": normalize_path(output_csv),
        "output_csv_sha256": _sha256(output_csv),
        "calibration": {
            "requested_backend": calibration_backend,
            "resolved_backend": calibration.backend_name,
            "score_unit": calibration.score_unit,
            "provenance": calibration.provenance,
        },
        "reconstruction_provenance_validation": provenance_validation,
        "reconstruction_provenance": reconstruction_provenance,
        **result.diagnostics,
        "resolved_thresholds": _resolved_thresholds(config),
        "score_distributions": {
            "eligible_quality_score": result.diagnostics[
                "eligible_score_distribution"
            ],
            "selected_quality_score": _distribution(result.candidates["score"]),
            "selected_max_reprojection_distance_px": _distribution(
                result.candidates["max_reprojection_distance_px"]
            ),
            "selected_min_border_distance_px": _distribution(
                result.candidates["min_border_distance_px"]
            ),
        },
        "temporal_coverage": temporal_coverage,
        "clean_tracklet_interval_coverage": result.diagnostics["clean_intervals"],
    }
    _atomic_write_json(summary, summary_json)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--clean-tracklets-csv",
        type=Path,
        help=(
            "optional reviewed clean_3d_tracklets.csv; when omitted, build "
            "provisional segments directly from a verified refractive geometry CSV"
        ),
    )
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-provenance-json",
        type=Path,
        help=(
            "refractive reconstruction provenance; defaults beside --geometry-csv"
        ),
    )
    parser.add_argument("--geometry-id-column")
    parser.add_argument("--slp-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument(
        "--calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default="auto",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--min-frame-spacing", type=int, default=166)
    parser.add_argument("--match-gate-px", type=float, default=80.0)
    parser.add_argument("--min-view-separation-px", type=float, default=40.0)
    parser.add_argument("--border-margin-px", type=float, default=20.0)
    parser.add_argument("--max-local-jump-mm", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _resolve_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    local = load_local_paths(args.paths_config)
    if args.video_dir is None and local is not None and local.medaka_data_dir is not None:
        args.video_dir = local.medaka_data_dir / args.experiment
    if args.slp_dir is None and local is not None and local.sleap_output_dir is not None:
        args.slp_dir = local.sleap_output_dir / args.experiment
    if (
        args.calibration_model_dir is None
        and local is not None
        and local.calibration_model_dir is not None
    ):
        args.calibration_model_dir = local.calibration_model_dir
    if args.output_csv is None:
        args.output_csv = (
            REPO_ROOT
            / "annotations"
            / "fish_sex_gt"
            / args.experiment
            / "clean_three_view_candidates.csv"
        )
    if args.summary_json is None:
        args.summary_json = args.output_csv.with_name(
            args.output_csv.stem + "_summary.json"
        )
    for option in ("video_dir", "slp_dir", "calibration_model_dir"):
        if getattr(args, option) is None:
            parser.error(
                f"--{option.replace('_', '-')} is required when it cannot be resolved "
                "from --paths-config"
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _resolve_args(parser.parse_args(argv), parser)
    config = CandidateConfig(
        target_count=args.target_count,
        min_frame_spacing=args.min_frame_spacing,
        match_gate_px=args.match_gate_px,
        min_view_separation_px=args.min_view_separation_px,
        border_margin_px=args.border_margin_px,
        max_local_jump_mm=args.max_local_jump_mm,
        seed=args.seed,
    )
    summary = run_from_paths(
        experiment=args.experiment,
        clean_tracklets_csv=args.clean_tracklets_csv,
        geometry_csv=args.geometry_csv,
        reconstruction_provenance_json=args.reconstruction_provenance_json,
        geometry_id_column=args.geometry_id_column,
        slp_dir=args.slp_dir,
        video_dir=args.video_dir,
        calibration_model_dir=args.calibration_model_dir,
        calibration_backend=args.calibration_backend,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        config=config,
    )
    print(
        f"wrote {summary['selected_candidate_count']} candidates to {args.output_csv} "
        f"({summary['eligible_frame_count']} eligible; summary {args.summary_json})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
