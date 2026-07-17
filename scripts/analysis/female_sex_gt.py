"""Core contracts for conservative frame-level female-sex ground truth.

This module intentionally has no Qt, video, or SLEAP dependency.  It owns the
stable candidate input contract, click evidence, optional geometry association,
and the authoritative resumable CSV/JSONL ledger used by the GUI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCHEMA_VERSION = "1.1"
LEGACY_SCHEMA_VERSION = "1.0"
SOFTWARE_NAME = "scripts.analysis.female_sex_gt_annotator"
FRAME_ALIASES = ("frame", "center_frame", "center", "jump_frame")
STANDARD_CANDIDATE_COLUMNS = {
    "experiment",
    "candidate_id",
    "source",
    "priority",
    "score",
    "reason",
    "view_hint",
    *FRAME_ALIASES,
}
GT_STATUSES = ("confirmed", "ambiguous", "not_visible", "bad_frame")
SELECTION_STATUSES = ("sleap_linked", "free_click", "not_selected")

ANNOTATION_COLUMNS = [
    "schema_version",
    "annotation_id",
    "experiment",
    "frame",
    "time_seconds",
    "annotator",
    "gt_status",
    "sex_label",
    "side1_selection_status",
    "side1_detection_key",
    "side1_instance_index",
    "side1_click_x",
    "side1_click_y",
    "side1_mouth_x",
    "side1_mouth_y",
    "side1_tail_x",
    "side1_tail_y",
    "side1_geometry_id",
    "side1_tracklet_id",
    "side1_node_names",
    "side2_selection_status",
    "side2_detection_key",
    "side2_instance_index",
    "side2_click_x",
    "side2_click_y",
    "side2_mouth_x",
    "side2_mouth_y",
    "side2_tail_x",
    "side2_tail_y",
    "side2_geometry_id",
    "side2_tracklet_id",
    "side2_node_names",
    "geometry_id_consistency",
    "consensus_geometry_id",
    "tracklet_id_consistency",
    "consensus_tracklet_id",
    "candidate_frame",
    "candidate_ids",
    "candidate_sources",
    "candidate_metadata_json",
    "candidate_geometry_tracklet_map_json",
    "candidate_geometry_source_mode",
    "video_side1_path",
    "video_side2_path",
    "slp_side1_path",
    "slp_side2_path",
    "geometry_csv_path",
    "calibration_backend",
    "calibration_model_files_json",
    "created_at_utc",
    "updated_at_utc",
    "notes",
]

TRACKLET_ANNOTATION_COLUMNS = {
    "side1_tracklet_id",
    "side2_tracklet_id",
    "tracklet_id_consistency",
    "consensus_tracklet_id",
    "candidate_geometry_tracklet_map_json",
    "candidate_geometry_source_mode",
}
LEGACY_ANNOTATION_COLUMNS = [
    column for column in ANNOTATION_COLUMNS if column not in TRACKLET_ANNOTATION_COLUMNS
]

CANDIDATE_COLUMNS = [
    "experiment",
    "frame",
    "candidate_ids",
    "candidate_sources",
    "candidate_metadata_json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_path(path: str | Path | None) -> str:
    """Return a normalized absolute path, or an empty field for no path."""
    if path is None or not str(path).strip():
        return ""
    return os.path.normpath(str(Path(path).expanduser().resolve(strict=False)))


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    )


def _json_value(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return normalize_path(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _coerce_frame(value: Any, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or _missing(value):
        raise ValueError(f"{context} must be an integral non-negative source frame")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not numeric: {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{context} must be integral, got {value!r}")
    frame = int(numeric)
    if frame < 0:
        raise ValueError(f"{context} must be non-negative, got {frame}")
    return frame


def _unique_join(values: Iterable[Any]) -> str:
    result: list[str] = []
    for value in values:
        if _missing(value):
            continue
        item = str(value)
        if item not in result:
            result.append(item)
    return ";".join(result)


def _candidate_contributions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("candidate_metadata_json", "")
    if _missing(raw):
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise ValueError("invalid candidate_metadata_json during merge") from error
    return parsed if isinstance(parsed, list) else [parsed]


def normalize_candidate_table(
    table: pd.DataFrame,
    *,
    experiment: str,
    max_frame: int | None,
    source_name: str = "<memory>",
) -> pd.DataFrame:
    """Normalize one candidate table without modifying its source.

    Duplicate frames inside the table are merged.  Each original row remains
    represented in ``candidate_metadata_json`` with its input file/row and all
    standard and non-standard metadata.
    """
    aliases = [column for column in FRAME_ALIASES if column in table.columns]
    if not aliases:
        raise ValueError(
            f"candidate table {source_name} needs one frame column named "
            + ", ".join(FRAME_ALIASES)
        )
    if max_frame is not None and max_frame < 0:
        raise ValueError("max_frame must be non-negative")

    input_label = normalize_path(source_name) if source_name != "<memory>" else source_name
    default_source = Path(source_name).stem if source_name != "<memory>" else "candidate"
    normalized_rows: list[dict[str, Any]] = []
    extra_columns = [
        column for column in table.columns if column not in STANDARD_CANDIDATE_COLUMNS
    ]
    for offset, (_, source_row) in enumerate(table.iterrows(), start=1):
        values = [
            _coerce_frame(source_row[alias], context=f"{source_name} row {offset} {alias}")
            for alias in aliases
            if not _missing(source_row[alias])
        ]
        if not values:
            raise ValueError(f"{source_name} row {offset} has no source frame")
        if len(set(values)) != 1:
            raise ValueError(
                f"{source_name} row {offset} has conflicting frame aliases: {values}"
            )
        frame = values[0]
        if max_frame is not None and frame > max_frame:
            raise ValueError(
                f"{source_name} row {offset} frame {frame} is out of range 0..{max_frame}"
            )
        row_experiment = source_row.get("experiment", experiment)
        if _missing(row_experiment):
            row_experiment = experiment
        if str(row_experiment) != experiment:
            raise ValueError(
                f"{source_name} row {offset} is for experiment {row_experiment!r}, "
                f"expected {experiment!r}"
            )

        candidate_id = source_row.get("candidate_id", "")
        if _missing(candidate_id):
            candidate_id = f"{default_source}:{offset}"
        candidate_source = source_row.get("source", "")
        if _missing(candidate_source):
            candidate_source = default_source
        contribution: dict[str, Any] = {
            "input": input_label,
            "row": offset,
            "frame_column": aliases[0],
        }
        for column in ("priority", "score", "reason", "view_hint"):
            if column in table.columns and not _missing(source_row[column]):
                contribution[column] = _json_value(source_row[column])
        extras = {
            column: _json_value(source_row[column])
            for column in extra_columns
            if not _missing(source_row[column])
        }
        if extras:
            contribution["extra"] = extras
        normalized_rows.append(
            {
                "experiment": experiment,
                "frame": frame,
                "candidate_ids": str(candidate_id),
                "candidate_sources": str(candidate_source),
                "candidate_metadata_json": stable_json([contribution]),
            }
        )
    return merge_normalized_candidates(normalized_rows)


def merge_normalized_candidates(
    tables_or_rows: Iterable[pd.DataFrame | Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in tables_or_rows:
        if isinstance(value, pd.DataFrame):
            rows.extend(value.to_dict("records"))
        else:
            rows.append(dict(value))
    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    merged: list[dict[str, Any]] = []
    raw = pd.DataFrame(rows)
    for (experiment, frame), group in raw.groupby(
        ["experiment", "frame"], sort=True, dropna=False
    ):
        ids: list[str] = []
        sources: list[str] = []
        contributions: list[dict[str, Any]] = []
        for row in group.to_dict("records"):
            ids.extend(str(row.get("candidate_ids", "")).split(";"))
            sources.extend(str(row.get("candidate_sources", "")).split(";"))
            contributions.extend(_candidate_contributions(row))
        merged.append(
            {
                "experiment": str(experiment),
                "frame": int(frame),
                "candidate_ids": _unique_join(ids),
                "candidate_sources": _unique_join(sources),
                "candidate_metadata_json": stable_json(contributions),
            }
        )
    return pd.DataFrame(merged, columns=CANDIDATE_COLUMNS).sort_values(
        ["experiment", "frame"], kind="stable", ignore_index=True
    )


def load_and_merge_candidate_csvs(
    paths: Sequence[str | Path],
    *,
    experiment: str,
    max_frame: int | None,
) -> pd.DataFrame:
    tables = []
    for path_like in paths:
        path = Path(path_like)
        table = pd.read_csv(path, low_memory=False)
        tables.append(
            normalize_candidate_table(
                table,
                experiment=experiment,
                max_frame=max_frame,
                source_name=str(path),
            )
        )
    return merge_normalized_candidates(tables)


def manual_jump_candidate(experiment: str, frame: int) -> dict[str, Any]:
    frame = _coerce_frame(frame, context="manual jump frame")
    return {
        "experiment": experiment,
        "frame": frame,
        "candidate_ids": f"manual:{frame}",
        "candidate_sources": "manual_jump",
        "candidate_metadata_json": stable_json(
            [{"input": "manual_jump", "frame": frame}]
        ),
    }


def candidate_from_annotation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Restore the queue/manual provenance that originally led to a saved row."""
    frame_value = row.get("candidate_frame", "")
    frame = row.get("frame", 0) if _missing(frame_value) else frame_value
    frame = _coerce_frame(frame, context="saved candidate frame")
    return {
        "experiment": str(row.get("experiment", "")),
        "frame": frame,
        "candidate_ids": str(row.get("candidate_ids", "")),
        "candidate_sources": str(row.get("candidate_sources", "")),
        "candidate_metadata_json": str(
            row.get("candidate_metadata_json", stable_json([])) or stable_json([])
        ),
    }


def _parse_geometry_tracklet_map(value: Any) -> dict[str, str]:
    if _missing(value):
        return {}
    if isinstance(value, Mapping):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError("invalid geometry_tracklet_map_json") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("geometry_tracklet_map_json must contain a JSON object")
    result: dict[str, str] = {}
    for geometry_id, tracklet_id in parsed.items():
        geometry_text = str(geometry_id).strip()
        tracklet_text = str(tracklet_id).strip()
        if not geometry_text or not tracklet_text:
            raise ValueError("geometry-to-tracklet mappings cannot contain empty IDs")
        result[geometry_text] = tracklet_text
    if len(set(result.values())) != len(result):
        raise ValueError("geometry-to-tracklet mapping must be one-to-one")
    return result


def candidate_tracklet_context(
    candidate: Mapping[str, Any] | None,
) -> tuple[dict[str, str], str]:
    """Extract one unambiguous geometry-to-segment mapping from a candidate."""
    if candidate is None:
        return {}, ""
    mappings: list[dict[str, str]] = []
    source_modes: list[str] = []

    direct_mapping = candidate.get("geometry_tracklet_map_json", "")
    if not _missing(direct_mapping):
        mappings.append(_parse_geometry_tracklet_map(direct_mapping))
    direct_mode = candidate.get("geometry_source_mode", "")
    if not _missing(direct_mode):
        source_modes.append(str(direct_mode).strip())

    for contribution in _candidate_contributions(candidate):
        extra = contribution.get("extra", {})
        if not isinstance(extra, Mapping):
            continue
        mapping_value = extra.get("geometry_tracklet_map_json", "")
        if not _missing(mapping_value):
            mappings.append(_parse_geometry_tracklet_map(mapping_value))
        source_mode = extra.get("geometry_source_mode", "")
        if not _missing(source_mode):
            source_modes.append(str(source_mode).strip())

    unique_mappings: list[dict[str, str]] = []
    for mapping in mappings:
        if mapping and mapping not in unique_mappings:
            unique_mappings.append(mapping)
    if len(unique_mappings) > 1:
        raise ValueError(
            "candidate contributions disagree about geometry-to-tracklet mapping"
        )
    unique_modes = sorted({mode for mode in source_modes if mode})
    if len(unique_modes) > 1:
        raise ValueError(
            f"candidate contributions disagree about geometry source mode: {unique_modes}"
        )
    return (
        unique_mappings[0] if unique_mappings else {},
        unique_modes[0] if unique_modes else "",
    )


def deterministic_annotation_id(experiment: str, frame: int, annotator: str) -> str:
    key = f"{experiment}\x1f{int(frame)}\x1f{annotator.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medaka:female-sex-gt:{key}"))


def _point_distance(point: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(point - target))


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-12:
        return _point_distance(point, start)
    fraction = float(np.dot(point - start, direction) / length_squared)
    closest = start + np.clip(fraction, 0.0, 1.0) * direction
    return _point_distance(point, closest)


def hit_test_sleap_pose(
    instances: np.ndarray | None,
    click_xy: Sequence[float],
    gate_px: float,
) -> tuple[int | None, float]:
    """Return the closest frame-local pose, considering segments and endpoints."""
    if gate_px < 0:
        raise ValueError("gate_px must be non-negative")
    if instances is None:
        return None, math.inf
    array = np.asarray(instances, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError("instances must have shape (instances, nodes, 2)")
    click = np.asarray(click_xy, dtype=float)
    if click.shape != (2,) or not np.isfinite(click).all():
        raise ValueError("click_xy must contain two finite source-pixel coordinates")

    best_index: int | None = None
    best_distance = math.inf
    for index, instance in enumerate(array):
        visible = instance[np.all(np.isfinite(instance), axis=1)]
        if not len(visible):
            continue
        distances = [_point_distance(click, point) for point in visible]
        if len(instance) >= 2 and np.isfinite(instance[:2]).all():
            distances.append(_segment_distance(click, instance[0], instance[1]))
        distance = min(distances)
        if distance < best_distance:
            best_index, best_distance = index, distance
    if best_index is None or best_distance > float(gate_px):
        return None, best_distance
    return best_index, best_distance


def empty_selection() -> dict[str, Any]:
    return {
        "selection_status": "not_selected",
        "detection_key": "",
        "instance_index": "",
        "click_x": "",
        "click_y": "",
        "mouth_x": "",
        "mouth_y": "",
        "tail_x": "",
        "tail_y": "",
        "geometry_id": "",
        "tracklet_id": "",
        "node_names": "mouth;tail",
    }


def selection_from_click(
    *,
    experiment: str,
    view: str,
    frame: int,
    click_xy: Sequence[float],
    instances: np.ndarray | None,
    gate_px: float,
    instance_geometry_ids: Mapping[int, Any] | None = None,
    node_names: Sequence[str] = ("mouth", "tail"),
) -> dict[str, Any]:
    """Build raw, source-pixel click evidence with an optional SLEAP link."""
    click = np.asarray(click_xy, dtype=float)
    index, _distance = hit_test_sleap_pose(instances, click, gate_px)
    result = empty_selection()
    result.update(
        {
            "selection_status": "free_click" if index is None else "sleap_linked",
            "click_x": float(click[0]),
            "click_y": float(click[1]),
            "node_names": ";".join(str(name) for name in node_names),
        }
    )
    if index is None:
        return result
    result["instance_index"] = int(index)
    result["detection_key"] = f"{experiment}|{view}|{int(frame)}|{int(index)}"
    pose = np.asarray(instances, dtype=float)[index]
    for node_index, node_name in enumerate(("mouth", "tail")):
        if node_index < len(pose) and np.isfinite(pose[node_index]).all():
            result[f"{node_name}_x"] = float(pose[node_index, 0])
            result[f"{node_name}_y"] = float(pose[node_index, 1])
    if instance_geometry_ids and index in instance_geometry_ids:
        result["geometry_id"] = str(instance_geometry_ids[index])
    return result


def visible_node_centroids(instances: np.ndarray | None) -> np.ndarray:
    if instances is None:
        return np.empty((0, 2), dtype=float)
    array = np.asarray(instances, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError("instances must have shape (instances, nodes, 2)")
    visible = np.all(np.isfinite(array), axis=2)
    counts = visible.sum(axis=1)
    centers = np.full((len(array), 2), np.nan, dtype=float)
    valid = counts > 0
    centers[valid] = np.where(visible[..., None], array, 0.0).sum(axis=1)[valid]
    centers[valid] /= counts[valid, None]
    return centers


def match_geometry_to_instances(
    projected_by_geometry_id: Mapping[Any, Sequence[float]],
    instances: np.ndarray | None,
    gate_px: float,
) -> dict[int, str]:
    """Globally match projected slots to visible-node pose centroids one-to-one."""
    centers = visible_node_centroids(instances)
    ids: list[str] = []
    projected: list[np.ndarray] = []
    for geometry_id, point in projected_by_geometry_id.items():
        point_array = np.asarray(point, dtype=float)
        if point_array.shape == (2,) and np.isfinite(point_array).all():
            ids.append(str(geometry_id))
            projected.append(point_array)
    if not ids or not len(centers):
        return {}
    points = np.stack(projected)
    costs = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    finite_costs = np.where(np.isfinite(costs), costs, 1e12)
    geometry_rows, instance_columns = linear_sum_assignment(finite_costs)
    matches: dict[int, str] = {}
    for geometry_row, instance_column in zip(geometry_rows, instance_columns):
        if costs[geometry_row, instance_column] <= float(gate_px):
            matches[int(instance_column)] = ids[int(geometry_row)]
    return matches


def geometry_consistency(
    side1_geometry_id: Any,
    side2_geometry_id: Any,
) -> tuple[str, str]:
    side1 = "" if _missing(side1_geometry_id) else str(side1_geometry_id)
    side2 = "" if _missing(side2_geometry_id) else str(side2_geometry_id)
    if side1 and side2:
        return ("agree", side1) if side1 == side2 else ("disagree", "")
    if side1 or side2:
        return "one_view", side1 or side2
    return "unmapped", ""


def tracklet_consistency(
    side1_tracklet_id: Any,
    side2_tracklet_id: Any,
) -> tuple[str, str]:
    return geometry_consistency(side1_tracklet_id, side2_tracklet_id)


def _selection_fields(prefix: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_selection_status": selection.get(
            "selection_status", "not_selected"
        ),
        f"{prefix}_detection_key": selection.get("detection_key", ""),
        f"{prefix}_instance_index": selection.get("instance_index", ""),
        f"{prefix}_click_x": selection.get("click_x", ""),
        f"{prefix}_click_y": selection.get("click_y", ""),
        f"{prefix}_mouth_x": selection.get("mouth_x", ""),
        f"{prefix}_mouth_y": selection.get("mouth_y", ""),
        f"{prefix}_tail_x": selection.get("tail_x", ""),
        f"{prefix}_tail_y": selection.get("tail_y", ""),
        f"{prefix}_geometry_id": selection.get("geometry_id", ""),
        f"{prefix}_tracklet_id": selection.get("tracklet_id", ""),
        f"{prefix}_node_names": selection.get("node_names", "mouth;tail"),
    }


def make_annotation_row(
    *,
    experiment: str,
    frame: int,
    source_fps: float,
    annotator: str,
    gt_status: str,
    side1_selection: Mapping[str, Any] | None,
    side2_selection: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    video_paths: Mapping[str, str | Path],
    slp_paths: Mapping[str, str | Path],
    geometry_csv_path: str | Path | None = None,
    calibration_backend: str = "",
    calibration_model_files: Any = None,
    notes: str = "",
) -> dict[str, Any]:
    frame = _coerce_frame(frame, context="annotation frame")
    if not annotator.strip():
        raise ValueError("annotator name is required")
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    side1 = dict(side1_selection or empty_selection())
    side2 = dict(side2_selection or empty_selection())
    candidate = candidate or manual_jump_candidate(experiment, frame)
    geometry_tracklet_map, geometry_source_mode = candidate_tracklet_context(candidate)
    for side_name, selection in (("side1", side1), ("side2", side2)):
        geometry_id = str(selection.get("geometry_id", "") or "")
        mapped_tracklet = geometry_tracklet_map.get(geometry_id, "")
        existing_tracklet = str(selection.get("tracklet_id", "") or "")
        if existing_tracklet and mapped_tracklet and existing_tracklet != mapped_tracklet:
            raise ValueError(
                f"{side_name} saved tracklet {existing_tracklet!r} disagrees with "
                f"candidate mapping {mapped_tracklet!r} for geometry {geometry_id!r}"
            )
        if mapped_tracklet:
            selection["tracklet_id"] = mapped_tracklet
    consistency, consensus = geometry_consistency(
        side1.get("geometry_id", ""), side2.get("geometry_id", "")
    )
    tracklet_state, consensus_tracklet = tracklet_consistency(
        side1.get("tracklet_id", ""), side2.get("tracklet_id", "")
    )
    now = utc_now()
    row: dict[str, Any] = {column: "" for column in ANNOTATION_COLUMNS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "annotation_id": deterministic_annotation_id(experiment, frame, annotator),
            "experiment": experiment,
            "frame": frame,
            "time_seconds": frame / float(source_fps),
            "annotator": annotator.strip(),
            "gt_status": gt_status,
            "sex_label": "F" if gt_status == "confirmed" else "",
            "geometry_id_consistency": consistency,
            "consensus_geometry_id": consensus,
            "tracklet_id_consistency": tracklet_state,
            "consensus_tracklet_id": consensus_tracklet,
            "candidate_frame": candidate.get("frame", ""),
            "candidate_ids": candidate.get("candidate_ids", ""),
            "candidate_sources": candidate.get("candidate_sources", ""),
            "candidate_metadata_json": candidate.get(
                "candidate_metadata_json", stable_json([])
            ),
            "candidate_geometry_tracklet_map_json": stable_json(
                geometry_tracklet_map
            ),
            "candidate_geometry_source_mode": geometry_source_mode,
            "video_side1_path": normalize_path(video_paths.get("side1_cam")),
            "video_side2_path": normalize_path(video_paths.get("side2_cam")),
            "slp_side1_path": normalize_path(slp_paths.get("side1_cam")),
            "slp_side2_path": normalize_path(slp_paths.get("side2_cam")),
            "geometry_csv_path": normalize_path(geometry_csv_path),
            "calibration_backend": str(calibration_backend or ""),
            "calibration_model_files_json": stable_json(
                calibration_model_files if calibration_model_files is not None else {}
            ),
            "created_at_utc": now,
            "updated_at_utc": now,
            "notes": notes.strip(),
        }
    )
    row.update(_selection_fields("side1", side1))
    row.update(_selection_fields("side2", side2))
    return validate_annotation_row(row)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return "" if _missing(value) else value


def validate_annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing_columns = set(ANNOTATION_COLUMNS) - set(row)
    extra_columns = set(row) - set(ANNOTATION_COLUMNS)
    if missing_columns or extra_columns:
        raise ValueError(
            f"annotation row schema mismatch; missing={sorted(missing_columns)}, "
            f"extra={sorted(extra_columns)}"
        )
    normalized = {column: _plain(row[column]) for column in ANNOTATION_COLUMNS}
    if str(normalized["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {normalized['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    frame = _coerce_frame(normalized["frame"], context="annotation frame")
    normalized["frame"] = frame
    annotator = str(normalized["annotator"]).strip()
    if not annotator:
        raise ValueError("annotator name is required")
    expected_id = deterministic_annotation_id(
        str(normalized["experiment"]), frame, annotator
    )
    if str(normalized["annotation_id"]) != expected_id:
        raise ValueError("annotation_id does not match experiment/frame/annotator")
    status = str(normalized["gt_status"])
    if status not in GT_STATUSES:
        raise ValueError(f"gt_status must be one of {GT_STATUSES}, got {status!r}")
    for side in ("side1", "side2"):
        selection_status = str(normalized[f"{side}_selection_status"])
        if selection_status not in SELECTION_STATUSES:
            raise ValueError(
                f"{side}_selection_status must be one of {SELECTION_STATUSES}"
            )
        if selection_status != "not_selected":
            for coordinate in ("click_x", "click_y"):
                value = normalized[f"{side}_{coordinate}"]
                if _missing(value) or not math.isfinite(float(value)):
                    raise ValueError(f"{side} selection requires a finite {coordinate}")
    has_selection = any(
        normalized[f"{side}_selection_status"] != "not_selected"
        for side in ("side1", "side2")
    )
    if status == "confirmed" and not has_selection:
        raise ValueError("confirmed decisions require at least one side-view selection")
    expected_sex = "F" if status == "confirmed" else ""
    if str(normalized["sex_label"]) != expected_sex:
        raise ValueError(
            "sex_label must be 'F' only for confirmed rows and empty otherwise"
        )
    consistency, consensus = geometry_consistency(
        normalized["side1_geometry_id"], normalized["side2_geometry_id"]
    )
    if normalized["geometry_id_consistency"] != consistency:
        raise ValueError("geometry_id_consistency does not match per-view mappings")
    if str(normalized["consensus_geometry_id"]) != consensus:
        raise ValueError("consensus_geometry_id does not match per-view mappings")
    tracklet_state, consensus_tracklet = tracklet_consistency(
        normalized["side1_tracklet_id"], normalized["side2_tracklet_id"]
    )
    if normalized["tracklet_id_consistency"] != tracklet_state:
        raise ValueError("tracklet_id_consistency does not match per-view tracklets")
    if str(normalized["consensus_tracklet_id"]) != consensus_tracklet:
        raise ValueError("consensus_tracklet_id does not match per-view tracklets")
    try:
        json.loads(str(normalized["candidate_metadata_json"] or "[]"))
        json.loads(str(normalized["calibration_model_files_json"] or "{}"))
    except json.JSONDecodeError as error:
        raise ValueError("annotation JSON provenance fields must be valid JSON") from error
    geometry_tracklet_map = _parse_geometry_tracklet_map(
        normalized["candidate_geometry_tracklet_map_json"] or "{}"
    )
    for side in ("side1", "side2"):
        geometry_id = str(normalized[f"{side}_geometry_id"] or "")
        tracklet_id = str(normalized[f"{side}_tracklet_id"] or "")
        if tracklet_id and not geometry_id:
            raise ValueError(f"{side}_tracklet_id requires a geometry ID")
        expected_tracklet = geometry_tracklet_map.get(geometry_id, "")
        if expected_tracklet and tracklet_id != expected_tracklet:
            raise ValueError(
                f"{side}_tracklet_id does not match candidate geometry mapping"
            )
    return normalized


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def _upgrade_legacy_annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = {column: "" for column in ANNOTATION_COLUMNS}
    for column in LEGACY_ANNOTATION_COLUMNS:
        if column in row:
            upgraded[column] = row[column]
    if str(upgraded.get("schema_version", "")) != LEGACY_SCHEMA_VERSION:
        raise ValueError(
            f"cannot upgrade annotation schema {upgraded.get('schema_version')!r}"
        )
    upgraded["schema_version"] = SCHEMA_VERSION
    candidate = candidate_from_annotation(upgraded)
    geometry_tracklet_map, geometry_source_mode = candidate_tracklet_context(candidate)
    upgraded["candidate_geometry_tracklet_map_json"] = stable_json(
        geometry_tracklet_map
    )
    upgraded["candidate_geometry_source_mode"] = geometry_source_mode
    for side in ("side1", "side2"):
        geometry_id = str(upgraded.get(f"{side}_geometry_id", "") or "")
        upgraded[f"{side}_tracklet_id"] = geometry_tracklet_map.get(geometry_id, "")
    state, consensus = tracklet_consistency(
        upgraded["side1_tracklet_id"], upgraded["side2_tracklet_id"]
    )
    upgraded["tracklet_id_consistency"] = state
    upgraded["consensus_tracklet_id"] = consensus
    return validate_annotation_row(upgraded)


def read_annotation_ledger(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return _empty_ledger()
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as error:
        raise ValueError(f"cannot parse annotation ledger {path}: {error}") from error
    if list(table.columns) == LEGACY_ANNOTATION_COLUMNS:
        upgraded_rows = [
            _upgrade_legacy_annotation_row(row) for row in table.to_dict("records")
        ]
        table = pd.DataFrame(upgraded_rows, columns=ANNOTATION_COLUMNS)
    elif list(table.columns) != ANNOTATION_COLUMNS:
        raise ValueError(
            f"annotation ledger {path} has incompatible columns; expected the "
            "documented fixed schema"
        )
    versions = set(table["schema_version"].astype(str))
    if versions - {SCHEMA_VERSION}:
        raise ValueError(
            f"annotation ledger {path} has unsupported schema versions: {sorted(versions)}"
        )
    numeric_frames = table["frame"].map(
        lambda value: _coerce_frame(value, context="ledger frame")
    )
    canonical_keys = pd.DataFrame(
        {
            "experiment": table["experiment"].astype(str),
            "frame": numeric_frames,
            "annotator": table["annotator"].astype(str),
        }
    )
    if canonical_keys.duplicated().any():
        duplicate = canonical_keys.loc[
            canonical_keys.duplicated(keep=False)
        ].iloc[0].to_dict()
        raise ValueError(f"annotation ledger {path} contains duplicate key {duplicate}")
    for row in table.to_dict("records"):
        validate_annotation_row(row)
    return sort_annotations(table)


def sort_annotations(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _empty_ledger()
    result = table.copy()
    result["__frame"] = result["frame"].map(
        lambda value: _coerce_frame(value, context="ledger frame")
    )
    result = result.sort_values(
        ["experiment", "__frame", "annotator"], kind="stable", ignore_index=True
    ).drop(columns="__frame")
    return result[ANNOTATION_COLUMNS]


def atomic_write_annotation_ledger(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    normalized_rows = [validate_annotation_row(row) for row in table.to_dict("records")]
    normalized = sort_annotations(pd.DataFrame(normalized_rows, columns=ANNOTATION_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        normalized.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_audit_record(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stable_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["experiment"]),
        _coerce_frame(row["frame"], context="annotation key frame"),
        str(row["annotator"]),
    )


def _candidate_provenance(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "candidate_frame": _plain(row.get("candidate_frame", "")),
        "candidate_ids": _plain(row.get("candidate_ids", "")),
        "candidate_sources": _plain(row.get("candidate_sources", "")),
        "candidate_metadata_json": _plain(row.get("candidate_metadata_json", "")),
        "candidate_geometry_tracklet_map_json": _plain(
            row.get("candidate_geometry_tracklet_map_json", "")
        ),
        "candidate_geometry_source_mode": _plain(
            row.get("candidate_geometry_source_mode", "")
        ),
        "consensus_tracklet_id": _plain(row.get("consensus_tracklet_id", "")),
    }


class AnnotationLedger:
    """Atomic canonical upserts with an append-only session undo audit."""

    def __init__(self, csv_path: str | Path, audit_jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.audit_jsonl_path = Path(audit_jsonl_path)
        self.rows = read_annotation_ledger(self.csv_path)
        self._history: list[
            tuple[tuple[str, int, str], dict[str, Any] | None, dict[str, Any]]
        ] = []
        self.last_undo_key: tuple[str, int, str] | None = None

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def get(self, experiment: str, frame: int, annotator: str) -> dict[str, Any] | None:
        if self.rows.empty:
            return None
        frames = pd.to_numeric(self.rows["frame"], errors="raise").astype(int)
        match = (
            self.rows["experiment"].eq(experiment)
            & frames.eq(int(frame))
            & self.rows["annotator"].eq(annotator)
        )
        if not match.any():
            return None
        return {key: _plain(value) for key, value in self.rows.loc[match].iloc[0].to_dict().items()}

    def upsert(self, row: Mapping[str, Any]) -> dict[str, Any]:
        normalized = validate_annotation_row(row)
        key = _key(normalized)
        old = self.get(*key)
        now = utc_now()
        normalized["created_at_utc"] = (
            old["created_at_utc"] if old is not None else normalized.get("created_at_utc") or now
        )
        normalized["updated_at_utc"] = now
        normalized = validate_annotation_row(normalized)

        if old is not None:
            frames = pd.to_numeric(self.rows["frame"], errors="raise").astype(int)
            keep = ~(
                self.rows["experiment"].eq(key[0])
                & frames.eq(key[1])
                & self.rows["annotator"].eq(key[2])
            )
            self.rows = self.rows.loc[keep].copy()
        self.rows = pd.concat(
            [self.rows, pd.DataFrame([normalized], columns=ANNOTATION_COLUMNS)],
            ignore_index=True,
        )
        self.rows = sort_annotations(self.rows)
        atomic_write_annotation_ledger(self.rows, self.csv_path)
        action = "edit" if old is not None else "add"
        append_audit_record(
            self.audit_jsonl_path,
            {
                "action": action,
                "timestamp_utc": now,
                "software": SOFTWARE_NAME,
                "schema_version": SCHEMA_VERSION,
                "old_row": old,
                "new_row": normalized,
                "candidate_provenance": _candidate_provenance(normalized),
            },
        )
        self._history.append((key, old, dict(normalized)))
        return dict(normalized)

    def undo(self) -> dict[str, Any] | None:
        if not self._history:
            return None
        key, previous, _saved = self._history.pop()
        self.last_undo_key = key
        current = self.get(*key)
        frames = (
            pd.to_numeric(self.rows["frame"], errors="raise").astype(int)
            if not self.rows.empty
            else pd.Series(dtype=int)
        )
        keep = ~(
            self.rows["experiment"].eq(key[0])
            & frames.eq(key[1])
            & self.rows["annotator"].eq(key[2])
        ) if not self.rows.empty else np.ones(0, dtype=bool)
        self.rows = self.rows.loc[keep].copy()
        if previous is not None:
            self.rows = pd.concat(
                [self.rows, pd.DataFrame([previous], columns=ANNOTATION_COLUMNS)],
                ignore_index=True,
            )
        self.rows = sort_annotations(self.rows)
        atomic_write_annotation_ledger(self.rows, self.csv_path)
        now = utc_now()
        append_audit_record(
            self.audit_jsonl_path,
            {
                "action": "undo",
                "timestamp_utc": now,
                "software": SOFTWARE_NAME,
                "schema_version": SCHEMA_VERSION,
                "old_row": current,
                "new_row": previous,
                "candidate_provenance": _candidate_provenance(previous or current),
            },
        )
        return previous


def queue_status(
    candidate: Mapping[str, Any],
    annotations: pd.DataFrame,
    *,
    experiment: str,
    annotator: str,
) -> str:
    if annotations.empty:
        return "unreviewed"
    candidate_frame = int(candidate["frame"])
    numeric_candidate_frames = pd.to_numeric(
        annotations["candidate_frame"], errors="coerce"
    )
    matching = annotations[
        annotations["experiment"].eq(experiment)
        & annotations["annotator"].eq(annotator)
        & numeric_candidate_frames.eq(candidate_frame)
    ]
    if matching.empty:
        return "unreviewed"
    matching = matching.sort_values("updated_at_utc", kind="stable")
    return str(matching.iloc[-1]["gt_status"])


def latest_annotated_frame(
    annotations: pd.DataFrame, *, experiment: str, annotator: str
) -> int | None:
    if annotations.empty:
        return None
    matching = annotations[
        annotations["experiment"].eq(experiment)
        & annotations["annotator"].eq(annotator)
    ]
    if matching.empty:
        return None
    row = matching.sort_values("updated_at_utc", kind="stable").iloc[-1]
    return int(row["frame"])


def file_sha256(path: str | Path) -> str:
    """Small utility used only when a caller explicitly wants provenance hashes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
