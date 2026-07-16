"""Compare mouth, centroid, and pose-QC-filtered centroid 3D association.

The strict reviewed mouth table supplies held-out H truth. Centroid rows are
matched to it independently in each frame with a one-to-one XYZ Hungarian
assignment; the resulting table retains centroid coordinates, not mouth
coordinates. A third condition masks centroid observations whose camera/G-slot
is covered by a false-positive or node/link QC event in any camera.

All three conditions are passed unchanged to
``scripts.tracking.compare_motion_linkers`` with the same calibration range
and held-out episodes. This file does not alter either linker implementation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.tracking.compare_motion_linkers import (
    ComparisonConfig,
    compare_motion_linkers,
    validate_reference,
)
from scripts.tracking.geometry_swap_annotations import (
    CAMERA_SCOPES,
    NODE_QC_EVENT_TYPES,
    normalize_events,
)


XYZ = ("X", "Y", "Z")
POSE_QC_EVENT_TYPES = ("false_positive", *NODE_QC_EVENT_TYPES)


def _integer_frames(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{name} must contain finite integer values")
    return numeric.astype(np.int64)


def resolve_geometry_slot_map(
    values: pd.Series,
    explicit: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve source slots to G labels without using reviewed identities."""
    source = sorted({str(value).strip() for value in values.dropna()})
    if not source:
        raise ValueError("centroid track contains no geometry_slot values")
    if explicit is not None:
        mapping = {str(key).strip(): str(value).strip() for key, value in explicit.items()}
        missing = sorted(set(source) - set(mapping))
        if missing:
            raise ValueError(f"slot map is missing centroid slots: {missing}")
    elif all(value.startswith("G") and value[1:].isdigit() for value in source):
        mapping = {value: value for value in source}
    else:
        mapping = {}
        for value in source:
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(
                    "non-numeric centroid slots require an explicit slot map"
                ) from error
            if not np.isfinite(number) or number != int(number) or number < 0:
                raise ValueError(
                    "default centroid slot mapping requires non-negative integers"
                )
            mapping[value] = f"G{int(number) + 1}"
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("centroid slot map values must be one-to-one")
    return mapping


def _validate_centroid_track(
    centroid: pd.DataFrame,
    slot_map: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    required = {"frame", "geometry_slot", *XYZ}
    missing = required - set(centroid.columns)
    if missing:
        raise ValueError(
            "centroid track is missing columns: " + ", ".join(sorted(missing))
        )
    table = centroid.copy()
    table["frame"] = _integer_frames(table["frame"], "frame")
    if table["geometry_slot"].isna().any():
        raise ValueError("centroid geometry_slot cannot be missing")
    table["source_geometry_slot"] = table["geometry_slot"].astype(str).str.strip()
    resolved = resolve_geometry_slot_map(table["source_geometry_slot"], slot_map)
    table["geometry_slot"] = table["source_geometry_slot"].map(resolved)
    for column in XYZ:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table.duplicated(["frame", "geometry_slot"]).any():
        raise ValueError("centroid track has duplicate frame/geometry_slot rows")
    return (
        table.sort_values(["frame", "geometry_slot"], kind="stable").reset_index(drop=True),
        resolved,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def build_centroid_reference(
    mouth_reference: pd.DataFrame,
    centroid_track: pd.DataFrame,
    *,
    match_gate_mm: float,
    slot_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach H truth to centroid XYZ via per-frame one-to-one matching.

    Every reviewed mouth row remains in the output. Unmatched rows have missing
    centroid XYZ, making the loss visible to downstream coverage/gap metrics.
    """
    if not np.isfinite(match_gate_mm) or match_gate_mm <= 0:
        raise ValueError("match_gate_mm must be finite and positive")
    mouth = validate_reference(mouth_reference)
    centroid_input_rows = int(len(centroid_track))
    if "frame" not in centroid_track:
        raise ValueError("centroid track is missing columns: frame")
    centroid_frames = pd.to_numeric(centroid_track["frame"], errors="coerce")
    frame_min = int(mouth["frame"].min())
    frame_max = int(mouth["frame"].max())
    centroid_in_range = centroid_track.loc[
        centroid_frames.between(frame_min, frame_max)
    ].copy()
    centroid, resolved_slot_map = _validate_centroid_track(
        centroid_in_range, slot_map
    )
    output = mouth[["frame", "human_identity", "reference_row_index"]].copy()
    for column in XYZ:
        output[f"mouth_{column}"] = mouth[column].astype(float)
        output[column] = np.nan
        output[f"offset_{column}"] = np.nan
    output["geometry_slot"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["source_geometry_slot"] = pd.Series(
        pd.NA, index=output.index, dtype="string"
    )
    output["match_distance_mm"] = np.nan
    output["centroid_matched"] = False

    mouth_groups = {int(frame): group for frame, group in mouth.groupby("frame", sort=False)}
    centroid_groups = {
        int(frame): group for frame, group in centroid.groupby("frame", sort=False)
    }
    accepted = 0
    rejected_by_gate = 0
    for frame in sorted(set(mouth_groups) & set(centroid_groups)):
        mouth_frame = mouth_groups[frame]
        centroid_frame = centroid_groups[frame]
        mouth_finite = mouth_frame[
            np.isfinite(mouth_frame[list(XYZ)].to_numpy(dtype=float)).all(axis=1)
        ]
        centroid_finite = centroid_frame[
            np.isfinite(centroid_frame[list(XYZ)].to_numpy(dtype=float)).all(axis=1)
        ]
        if mouth_finite.empty or centroid_finite.empty:
            continue
        distance = np.linalg.norm(
            mouth_finite[list(XYZ)].to_numpy(dtype=float)[:, None, :]
            - centroid_finite[list(XYZ)].to_numpy(dtype=float)[None, :, :],
            axis=2,
        )
        mouth_indices, centroid_indices = linear_sum_assignment(distance)
        for mouth_index, centroid_index in zip(mouth_indices, centroid_indices):
            value = float(distance[int(mouth_index), int(centroid_index)])
            if not np.isfinite(value) or value > match_gate_mm:
                rejected_by_gate += 1
                continue
            mouth_row = mouth_finite.iloc[int(mouth_index)]
            centroid_row = centroid_finite.iloc[int(centroid_index)]
            target = int(mouth_row["reference_row_index"])
            for column in XYZ:
                centroid_value = float(centroid_row[column])
                output.loc[target, column] = centroid_value
                output.loc[target, f"offset_{column}"] = (
                    centroid_value - float(mouth_row[column])
                )
            output.loc[target, "geometry_slot"] = str(centroid_row["geometry_slot"])
            output.loc[target, "source_geometry_slot"] = str(
                centroid_row["source_geometry_slot"]
            )
            output.loc[target, "match_distance_mm"] = value
            output.loc[target, "centroid_matched"] = True
            accepted += 1

    finite_mouth = int(
        np.isfinite(mouth[list(XYZ)].to_numpy(dtype=float)).all(axis=1).sum()
    )
    matched_frames = output.loc[output["centroid_matched"], "frame"].nunique()
    complete_by_frame = output.groupby("frame")["centroid_matched"].all()
    summary = {
        "match_gate_mm": float(match_gate_mm),
        "reviewed_rows": int(len(mouth)),
        "finite_reviewed_rows": finite_mouth,
        "centroid_input_rows": centroid_input_rows,
        "centroid_rows_in_reference_range": int(len(centroid)),
        "matched_rows": accepted,
        "match_coverage_of_finite_reviewed": (
            float(accepted / finite_mouth) if finite_mouth else None
        ),
        "frames_with_any_match": int(matched_frames),
        "complete_match_frames": int(complete_by_frame.sum()),
        "rejected_hungarian_pairs_over_gate": rejected_by_gate,
        "resolved_slot_map": resolved_slot_map,
        "offset_distance_mm": _distribution(
            output["match_distance_mm"].to_numpy(dtype=float)
        ),
        "signed_axis_offsets_mm": {
            column: _distribution(output[f"offset_{column}"].to_numpy(dtype=float))
            for column in XYZ
        },
    }
    return output, summary


def apply_pose_qc_exclusions(
    centroid_reference: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    slots: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Mask centroid coordinates covered by pose-QC events in any camera."""
    table = centroid_reference.copy()
    table["pose_qc_excluded"] = False
    table["pose_qc_event_ids"] = ""
    table["pose_qc_event_types"] = ""
    normalized = normalize_events(events, slots)
    relevant = normalized[
        normalized["event_type"].isin(POSE_QC_EVENT_TYPES)
        & normalized["view_scope"].isin(CAMERA_SCOPES)
    ]
    excluded_by_type = {event_type: 0 for event_type in POSE_QC_EVENT_TYPES}
    for event in relevant.itertuples(index=False):
        mask = (
            table["geometry_slot"].astype("string").eq(str(event.slot_a))
            & table["frame"].between(int(event.frame_start), int(event.frame_end))
            & table["centroid_matched"].astype(bool)
        )
        excluded_by_type[str(event.event_type)] += int(mask.sum())
        if not mask.any():
            continue
        table.loc[mask, "pose_qc_excluded"] = True
        for column, value in (
            ("pose_qc_event_ids", str(event.event_id)),
            ("pose_qc_event_types", str(event.event_type)),
        ):
            existing = table.loc[mask, column].astype(str)
            table.loc[mask, column] = np.where(
                existing.eq(""), value, existing + ";" + value
            )
    excluded = table["pose_qc_excluded"].astype(bool)
    table.loc[excluded, list(XYZ)] = np.nan

    node_events = relevant[relevant["event_type"].isin(NODE_QC_EVENT_TYPES)]
    node_count = int(len(node_events))
    if node_count == 0:
        caveat = (
            "No node-specific bad_mouth, bad_tail, or wrong_link labels exist "
            "in the ledger; this dataset supports no conclusion about whether "
            "node-QC filtering improves association."
        )
    else:
        caveat = (
            "Node-specific labels exist, so the centroid+QC condition is "
            "measurable; effect size still depends on reviewed label coverage."
        )
    summary = {
        "pose_qc_event_types": list(POSE_QC_EVENT_TYPES),
        "relevant_events": int(len(relevant)),
        "events_by_type": {
            event_type: int(relevant["event_type"].eq(event_type).sum())
            for event_type in POSE_QC_EVENT_TYPES
        },
        "node_specific_events": node_count,
        "node_qc_conclusion_supported": node_count > 0,
        "excluded_matched_rows": int(excluded.sum()),
        "excluded_rows_by_event_type_with_overlap": excluded_by_type,
        "remaining_centroid_rows": int(
            np.isfinite(table[list(XYZ)].to_numpy(dtype=float)).all(axis=1).sum()
        ),
        "caveat": caveat,
    }
    return table, summary


def _comparison_report(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "predictions"}


def run_anchor_ablation(
    *,
    mouth_reference: pd.DataFrame,
    centroid_track: pd.DataFrame,
    events: pd.DataFrame | None,
    episodes: pd.DataFrame,
    comparison_config: ComparisonConfig,
    centroid_match_gate_mm: float = 35.0,
    slot_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run paired mouth/centroid/centroid+QC linker comparisons."""
    mouth = validate_reference(mouth_reference)
    identities = sorted(mouth["human_identity"].astype(str).unique())
    config = comparison_config
    if config.num_individuals is None:
        config = replace(config, num_individuals=len(identities))
    elif config.num_individuals != len(identities):
        raise ValueError(
            "num_individuals must equal reviewed identity count "
            f"({len(identities)})"
        )
    centroid, match_summary = build_centroid_reference(
        mouth,
        centroid_track,
        match_gate_mm=centroid_match_gate_mm,
        slot_map=slot_map,
    )
    resolved_slots = sorted(
        match_summary["resolved_slot_map"].values()  # type: ignore[union-attr]
    )
    centroid_qc, qc_summary = apply_pose_qc_exclusions(
        centroid, events, slots=resolved_slots
    )

    anchor_tables = {
        "mouth": mouth,
        "centroid": centroid,
        "centroid_pose_qc": centroid_qc,
    }
    comparisons = {
        name: compare_motion_linkers(table, config=config, episodes=episodes)
        for name, table in anchor_tables.items()
    }
    report = {
        "schema_version": "anchor_coordinate_ablation_v1",
        "comparison_config": asdict(config),
        "paired_design": (
            "All anchors use identical calibration/context and held-out "
            "episode intervals. Missing or QC-excluded centroid rows remain "
            "explicit gaps rather than being removed from the timeline."
        ),
        "centroid_matching": match_summary,
        "pose_qc": qc_summary,
        "comparisons": {
            name: _comparison_report(result)
            for name, result in comparisons.items()
        },
    }
    return {
        "report": report,
        "centroid_reference": centroid,
        "centroid_pose_qc_reference": centroid_qc,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_anchor_outputs(
    result: dict[str, object],
    *,
    output_json: Path,
    centroid_reference_csv: Path | None = None,
    centroid_pose_qc_csv: Path | None = None,
) -> dict[str, Path]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    paths = {"report_json": output_json}
    if centroid_reference_csv is not None:
        centroid_reference_csv.parent.mkdir(parents=True, exist_ok=True)
        result["centroid_reference"].to_csv(centroid_reference_csv, index=False)  # type: ignore[union-attr]
        paths["centroid_reference_csv"] = centroid_reference_csv
    if centroid_pose_qc_csv is not None:
        centroid_pose_qc_csv.parent.mkdir(parents=True, exist_ok=True)
        result["centroid_pose_qc_reference"].to_csv(  # type: ignore[union-attr]
            centroid_pose_qc_csv, index=False
        )
        paths["centroid_pose_qc_csv"] = centroid_pose_qc_csv
    report = dict(result["report"])  # type: ignore[arg-type]
    report["outputs"] = {name: str(path) for name, path in paths.items()}
    output_json.write_text(
        json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8"
    )
    return paths


def _parse_slot_map(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--slot-map expects SOURCE=G_SLOT, got {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target:
            raise ValueError(f"--slot-map expects SOURCE=G_SLOT, got {value!r}")
        mapping[source] = target
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouth-reference-csv", type=Path, required=True)
    parser.add_argument("--centroid-track-csv", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--centroid-reference-csv", type=Path)
    parser.add_argument("--centroid-pose-qc-csv", type=Path)
    parser.add_argument("--centroid-match-gate-mm", type=float, default=35.0)
    parser.add_argument("--slot-map", action="append", default=[])
    parser.add_argument("--calibration-start-frame", type=int, required=True)
    parser.add_argument("--calibration-end-frame", type=int, required=True)
    parser.add_argument("--num-individuals", type=int)
    parser.add_argument("--max-jump-mm", type=float, default=8.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--velocity-prediction-weight", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=2.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.0)
    parser.add_argument(
        "--motion-uncertainty-growth-per-frame", type=float, default=0.25
    )
    parser.add_argument("--max-motion-gap-frames", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mouth = pd.read_csv(args.mouth_reference_csv, low_memory=False)
    centroid = pd.read_csv(args.centroid_track_csv, low_memory=False)
    events = pd.read_csv(args.events_csv, keep_default_na=False)
    episodes = pd.read_csv(args.episodes_csv, low_memory=False)
    config = ComparisonConfig(
        calibration_start_frame=args.calibration_start_frame,
        calibration_end_frame=args.calibration_end_frame,
        num_individuals=args.num_individuals,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        velocity_prediction_weight=args.velocity_prediction_weight,
        flow_weight=args.flow_weight,
        acceleration_weight=args.acceleration_weight,
        motion_uncertainty_growth_per_frame=(
            args.motion_uncertainty_growth_per_frame
        ),
        max_motion_gap_frames=args.max_motion_gap_frames,
    )
    result = run_anchor_ablation(
        mouth_reference=mouth,
        centroid_track=centroid,
        events=events,
        episodes=episodes,
        comparison_config=config,
        centroid_match_gate_mm=args.centroid_match_gate_mm,
        slot_map=_parse_slot_map(args.slot_map),
    )
    paths = write_anchor_outputs(
        result,
        output_json=args.output_json,
        centroid_reference_csv=args.centroid_reference_csv,
        centroid_pose_qc_csv=args.centroid_pose_qc_csv,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
