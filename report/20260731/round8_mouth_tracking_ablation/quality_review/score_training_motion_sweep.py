from __future__ import annotations

"""Independently score the A1/A2 motion sweep on training windows only."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
RELINK_ROOT = ROOT / "training_relink"
OUTPUT_ROOT = ROOT / "quality_review"
ANNOTATIONS = Path(
    "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv"
)
CALIBRATION_DIR = Path("calibration_results/fixed_refractive_192x192x186")
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}
CONFIGS = {
    "A1": {
        "method": "A1",
        "velocity_weight": 0.0,
        "history_frames": 2,
        "direct_name": "a1_direct.csv",
        "summary_name": "a1_summary.json",
    },
    "A2_v025_h5": {
        "method": "A2",
        "velocity_weight": 0.25,
        "history_frames": 5,
        "direct_name": "a2_direct.csv",
        "summary_name": "a2_summary.json",
    },
    "A2_v050_h5": {
        "method": "A2",
        "velocity_weight": 0.5,
        "history_frames": 5,
        "direct_name": "a2_direct.csv",
        "summary_name": "a2_summary.json",
    },
    "A2_v100_h5": {
        "method": "A2",
        "velocity_weight": 1.0,
        "history_frames": 5,
        "direct_name": "a2_direct.csv",
        "summary_name": "a2_summary.json",
    },
    "A2_v050_h9": {
        "method": "A2",
        "velocity_weight": 0.5,
        "history_frames": 9,
        "direct_name": "a2_direct.csv",
        "summary_name": "a2_summary.json",
    },
    "A2_v100_h9": {
        "method": "A2",
        "velocity_weight": 1.0,
        "history_frames": 9,
        "direct_name": "a2_direct.csv",
        "summary_name": "a2_summary.json",
    },
}
CAMERA_INDEX = {"side1": 1, "side2": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_training_annotations() -> pd.DataFrame:
    table = pd.read_csv(ANNOTATIONS, low_memory=False)
    table = table[
        table.gt_status.eq("confirmed") & table.sex_label.eq("F")
    ].copy()
    allowed = np.zeros(len(table), dtype=bool)
    window_names = np.full(len(table), "", dtype=object)
    for name, (frame_start, frame_end) in WINDOWS.items():
        within = table.frame.between(frame_start, frame_end, inclusive="both")
        allowed |= within.to_numpy()
        window_names[within.to_numpy()] = name
    table = table.loc[allowed].copy()
    table["training_window"] = window_names[allowed]
    return table.sort_values(["training_window", "frame"], kind="stable")


def composite_ids(table: pd.DataFrame) -> np.ndarray:
    base = table.fish_id.astype(str)
    if "identity_segment" not in table:
        return base.to_numpy()
    segment = pd.to_numeric(table.identity_segment, errors="coerce")
    token = table.identity_segment.astype("string").fillna("missing")
    integral = segment.notna() & np.isclose(segment, np.round(segment))
    token.loc[integral] = (
        np.round(segment.loc[integral]).astype(np.int64).astype(str)
    )
    return (base + "#identity_segment=" + token.astype(str)).to_numpy()


def map_female_rows(
    annotations: pd.DataFrame,
    tracks: pd.DataFrame,
    calibration: Calibration,
    *,
    config_name: str,
    window_name: str,
    gate_px: float = 80.0,
) -> pd.DataFrame:
    tracks = tracks.copy()
    tracks["method_id"] = composite_ids(tracks)
    frames = set(int(value) for value in annotations.frame)
    by_frame = {
        int(frame): group
        for frame, group in tracks[tracks.frame.isin(frames)].groupby(
            "frame", sort=False
        )
    }
    records: list[dict[str, Any]] = []
    for annotation in annotations.itertuples(index=False):
        group = by_frame.get(int(annotation.frame))
        choices: dict[str, str] = {}
        distances: dict[str, float] = {}
        if group is not None:
            xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
            finite = np.all(np.isfinite(xyz), axis=1)
            projected = np.full((len(group), 3, 2), np.nan, dtype=float)
            if finite.any():
                projected[finite] = np.asarray(
                    calibration.project_world(xyz[finite]), dtype=float
                )
            method_ids = group.method_id.astype(str).to_numpy()
            for prefix in ("side1", "side2"):
                target = np.asarray(
                    [
                        getattr(annotation, f"{prefix}_mouth_x"),
                        getattr(annotation, f"{prefix}_mouth_y"),
                    ],
                    dtype=float,
                )
                costs = np.linalg.norm(
                    projected[:, CAMERA_INDEX[prefix], :] - target[None, :],
                    axis=1,
                )
                if np.isfinite(costs).any():
                    index = int(np.nanargmin(costs))
                    distances[prefix] = float(costs[index])
                    if distances[prefix] <= gate_px:
                        choices[prefix] = method_ids[index]
                else:
                    distances[prefix] = np.nan
        available = [
            choices[prefix]
            for prefix in ("side1", "side2")
            if prefix in choices
        ]
        if len(available) == 2 and available[0] == available[1]:
            mapping_status, identity = "agree", available[0]
        elif len(available) == 2:
            mapping_status, identity = "disagree", ""
        elif len(available) == 1:
            mapping_status, identity = "one_view", available[0]
        else:
            mapping_status, identity = "unmapped", ""
        records.append(
            {
                "config": config_name,
                "training_window": window_name,
                "annotation_id": str(annotation.annotation_id),
                "frame": int(annotation.frame),
                "mapping_status": mapping_status,
                "mapped_method_id": identity,
                "side1_distance_px": distances.get("side1", np.nan),
                "side2_distance_px": distances.get("side2", np.nan),
            }
        )
    return pd.DataFrame.from_records(records)


def track_metrics(table: pd.DataFrame, frame_start: int, frame_end: int) -> dict[str, Any]:
    expected_rows = 3 * (frame_end - frame_start + 1)
    if len(table) != expected_rows:
        raise ValueError(
            f"direct table has {len(table)} rows; expected {expected_rows}"
        )
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(xyz), axis=1)
    interpolated = (
        table.is_interpolated.fillna(False).astype(bool).to_numpy()
        if "is_interpolated" in table
        else np.zeros(len(table), dtype=bool)
    )
    direct = finite & ~interpolated
    frame_counts = (
        pd.Series(direct.astype(np.int8), index=table.frame)
        .groupby(level=0)
        .sum()
        .reindex(range(frame_start, frame_end + 1), fill_value=0)
    )
    jump_values: list[float] = []
    group_columns = ["fish_id"]
    if "identity_segment" in table:
        group_columns.append("identity_segment")
    for _key, rows in table.groupby(group_columns, sort=True, dropna=False):
        rows = rows.sort_values("frame", kind="stable")
        points = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
        frames = rows.frame.to_numpy(dtype=np.int64)
        row_direct = (
            np.all(np.isfinite(points), axis=1)
            & ~rows.is_interpolated.fillna(False).astype(bool).to_numpy()
        )
        valid = (
            (np.diff(frames) == 1)
            & row_direct[:-1]
            & row_direct[1:]
        )
        jump_values.extend(
            np.linalg.norm(np.diff(points, axis=0), axis=1)[valid]
        )
    jumps = np.asarray(jump_values, dtype=float)
    residual = pd.to_numeric(
        table.ray_residual_mm2, errors="coerce"
    ).to_numpy(dtype=float)
    return {
        "rows": int(len(table)),
        "expected_rows": expected_rows,
        "all_three_direct_frames": int((frame_counts == 3).sum()),
        "all_three_direct_fraction": float((frame_counts == 3).mean()),
        "unavailable_rows": int((~direct).sum()),
        "unavailable_row_fraction": float((~direct).mean()),
        "jump_mm": distribution(jumps),
        "jump_ge_7_5mm_count": int((jumps >= 7.5).sum()),
        "jump_gt_8mm_count": int((jumps > 8.0).sum()),
        "residual_mm2": distribution(residual[direct]),
        "_jump_values": jumps.tolist(),
        "_residual_values": residual[direct & np.isfinite(residual)].tolist(),
    }


def summarize_assignments(table: pd.DataFrame) -> dict[str, Any]:
    mapped = table[table.mapped_method_id.ne("")].copy()
    counts = Counter(mapped.mapped_method_id.astype(str))
    best_label, best_count = counts.most_common(1)[0] if counts else ("", 0)
    identities = mapped.mapped_method_id.astype(str).to_numpy()
    return {
        "eligible_annotations": int(len(table)),
        "mapped_annotations": int(len(mapped)),
        "mapping_status_counts": {
            str(key): int(value)
            for key, value in table.mapping_status.value_counts().items()
        },
        "identity_counts": dict(sorted(counts.items())),
        "best_fixed_identity": best_label,
        "best_fixed_correct": int(best_count),
        "full_eligible_accuracy": (
            float(best_count / len(table)) if len(table) else 0.0
        ),
        "selective_mapped_accuracy": (
            float(best_count / len(mapped)) if len(mapped) else 0.0
        ),
        "observed_identity_changes": int(
            np.sum(identities[1:] != identities[:-1])
            if len(identities) > 1
            else 0
        ),
    }


def main() -> None:
    annotations = load_training_annotations()
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")
    if calibration.backend_name != "refractive" or calibration.score_unit != "mm^2":
        raise ValueError("training scorer requires the fixed refractive mm^2 backend")
    assignment_parts: list[pd.DataFrame] = []
    records: dict[str, Any] = {}
    normalized_settings: dict[str, set[str]] = {
        name: set() for name in CONFIGS
    }
    for config_name, expected in CONFIGS.items():
        window_records: dict[str, Any] = {}
        pooled_jumps: list[float] = []
        pooled_residual: list[float] = []
        pooled_unavailable = 0
        pooled_rows = 0
        pooled_direct_frames = 0
        pooled_frames = 0
        total_correct = 0
        total_mapped = 0
        total_eligible = 0
        total_changes = 0
        for window_name, (frame_start, frame_end) in WINDOWS.items():
            folder = RELINK_ROOT / window_name / config_name
            summary_path = folder / expected["summary_name"]
            direct_path = folder / expected["direct_name"]
            if not summary_path.exists() or not direct_path.exists():
                raise FileNotFoundError(
                    f"incomplete sweep: missing {summary_path} or {direct_path}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            direct_hash = sha256(direct_path)
            if summary.get("status") != "completed":
                raise ValueError(f"incomplete summary for {window_name}/{config_name}")
            if summary["outputs"]["direct"]["sha256"] != direct_hash:
                raise ValueError(f"direct hash mismatch for {window_name}/{config_name}")
            for output_name, output in summary["outputs"].items():
                output_path = Path(output["path"])
                if (
                    not output_path.exists()
                    or sha256(output_path) != output["sha256"]
                ):
                    raise ValueError(
                        f"{output_name} hash mismatch for "
                        f"{window_name}/{config_name}"
                    )
            settings = dict(summary["settings"])
            if summary.get("configuration_hash") != stable_hash(settings):
                raise ValueError(
                    f"configuration hash mismatch for {window_name}/{config_name}"
                )
            if summary.get("code_sha256") != settings.get("code_sha256"):
                raise ValueError(
                    f"code manifest duplication mismatch for "
                    f"{window_name}/{config_name}"
                )
            for code_path, recorded_hash in settings["code_sha256"].items():
                actual_path = Path(code_path)
                if not actual_path.exists() or sha256(actual_path) != recorded_hash:
                    raise ValueError(
                        f"stale code hash {code_path} for "
                        f"{window_name}/{config_name}"
                    )
            if settings["method"] != expected["method"]:
                raise ValueError(f"method mismatch for {window_name}/{config_name}")
            if not np.isclose(
                float(settings["velocity_weight"]),
                float(expected["velocity_weight"]),
            ):
                raise ValueError(f"velocity mismatch for {window_name}/{config_name}")
            if int(settings["motion_history_frames"]) != int(
                expected["history_frames"]
            ):
                raise ValueError(f"history mismatch for {window_name}/{config_name}")
            geometry_path = (
                ROOT
                / "training_geometry"
                / window_name
                / "strict_mouth_selected_geometry.parquet"
            )
            if sha256(geometry_path) != settings["geometry_sha256"]:
                raise ValueError(
                    f"geometry hash mismatch for {window_name}/{config_name}"
                )
            if (
                int(settings["frame_start"]) != frame_start
                or int(settings["frame_end"]) != frame_end
            ):
                raise ValueError(
                    f"frame interval mismatch for {window_name}/{config_name}"
                )
            normalized = dict(settings)
            for key in ("geometry_sha256", "frame_start", "frame_end"):
                normalized.pop(key, None)
            normalized_settings[config_name].add(stable_hash(normalized))
            tracks = pd.read_csv(direct_path, low_memory=False)
            metrics = track_metrics(tracks, frame_start, frame_end)
            window_annotations = annotations[
                annotations.training_window.eq(window_name)
            ]
            assignments = map_female_rows(
                window_annotations,
                tracks,
                calibration,
                config_name=config_name,
                window_name=window_name,
            )
            assignment_summary = summarize_assignments(assignments)
            assignments["best_fixed_identity"] = assignment_summary[
                "best_fixed_identity"
            ]
            assignments["is_correct_fixed_identity"] = (
                assignments.mapped_method_id.eq(
                    assignment_summary["best_fixed_identity"]
                )
                & assignments.mapped_method_id.ne("")
            )
            assignment_parts.append(assignments)
            pooled_jumps.extend(metrics.pop("_jump_values"))
            pooled_residual.extend(metrics.pop("_residual_values"))
            pooled_unavailable += metrics["unavailable_rows"]
            pooled_rows += metrics["rows"]
            pooled_direct_frames += metrics["all_three_direct_frames"]
            pooled_frames += frame_end - frame_start + 1
            total_correct += assignment_summary["best_fixed_correct"]
            total_mapped += assignment_summary["mapped_annotations"]
            total_eligible += assignment_summary["eligible_annotations"]
            total_changes += assignment_summary["observed_identity_changes"]
            window_records[window_name] = {
                "interval": [frame_start, frame_end],
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "configuration_hash": summary["configuration_hash"],
                "direct_path": str(direct_path),
                "direct_sha256": direct_hash,
                "all_summary_output_hashes_verified": True,
                "track_qc": metrics,
                "female_gt": assignment_summary,
            }
        jump_array = np.asarray(pooled_jumps, dtype=float)
        records[config_name] = {
            "configuration": {
                key: value
                for key, value in expected.items()
                if key not in {"direct_name", "summary_name"}
            },
            "cross_window_settings_equal": (
                len(normalized_settings[config_name]) == 1
            ),
            "normalized_settings_sha256": sorted(
                normalized_settings[config_name]
            ),
            "training_windows": window_records,
            "aggregate": {
                "female_eligible_annotations": total_eligible,
                "female_mapped_annotations": total_mapped,
                "female_best_fixed_correct_sum_by_window": total_correct,
                "female_full_eligible_accuracy": (
                    total_correct / total_eligible if total_eligible else 0.0
                ),
                "female_selective_mapped_accuracy": (
                    total_correct / total_mapped if total_mapped else 0.0
                ),
                "female_observed_identity_changes_sum_by_window": total_changes,
                "all_three_direct_frames": pooled_direct_frames,
                "all_three_direct_fraction": pooled_direct_frames / pooled_frames,
                "unavailable_rows": pooled_unavailable,
                "unavailable_row_fraction": pooled_unavailable / pooled_rows,
                "jump_mm": distribution(jump_array),
                "jump_ge_7_5mm_count": int((jump_array >= 7.5).sum()),
                "jump_gt_8mm_count": int((jump_array > 8.0).sum()),
                "residual_mm2": distribution(pooled_residual),
            },
        }

    ranking = []
    for config_name, record in records.items():
        if record["configuration"]["method"] != "A2":
            continue
        aggregate = record["aggregate"]
        canonical = json.dumps(
            record["configuration"],
            sort_keys=True,
            separators=(",", ":"),
        )
        ranking.append(
            {
                "config": config_name,
                "female_full_eligible_accuracy": aggregate[
                    "female_full_eligible_accuracy"
                ],
                "female_observed_identity_changes": aggregate[
                    "female_observed_identity_changes_sum_by_window"
                ],
                "jump_ge_7_5mm_count": aggregate["jump_ge_7_5mm_count"],
                "unavailable_rows": aggregate["unavailable_rows"],
                "velocity_weight": record["configuration"]["velocity_weight"],
                "history_frames": record["configuration"]["history_frames"],
                "canonical_configuration": canonical,
                "_sort_key": (
                    -aggregate["female_full_eligible_accuracy"],
                    aggregate[
                        "female_observed_identity_changes_sum_by_window"
                    ],
                    aggregate["jump_ge_7_5mm_count"],
                    aggregate["unavailable_rows"],
                    record["configuration"]["velocity_weight"],
                    record["configuration"]["history_frames"],
                    canonical,
                ),
            }
        )
    ranking.sort(key=lambda row: row["_sort_key"])
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
        row.pop("_sort_key")
    winner = ranking[0]["config"]
    primary_signature_count = len(
        {
            (
                row["female_full_eligible_accuracy"],
                row["female_observed_identity_changes"],
            )
            for row in ranking
        }
    )
    a2_direct_hashes_by_window = {
        window_name: sorted(
            {
                records[config_name]["training_windows"][window_name][
                    "direct_sha256"
                ]
                for config_name in CONFIGS
                if CONFIGS[config_name]["method"] == "A2"
            }
        )
        for window_name in WINDOWS
    }
    denominator = int(len(annotations))
    selection = {
        "schema_version": 1,
        "status": "training_only_motion_configuration_selected",
        "selected_a2_config": winner,
        "selected_configuration": records[winner]["configuration"],
        "ranking": ranking,
        "tie_break_order": [
            "maximize full-eligible female accuracy, summed after independent fixed-label mapping within each window",
            "minimize observed female identity changes",
            "minimize direct adjacent jumps >=7.5 mm (false-merge/swap proxy)",
            "minimize unavailable direct rows (false-break proxy)",
            "lower velocity weight",
            "shorter motion history",
            "lexicographic canonical configuration JSON",
        ],
        "female_training_denominator": denominator,
        "a2_direct_hashes_by_training_window": a2_direct_hashes_by_window,
        "all_a2_direct_outputs_byte_identical_within_each_window": all(
            len(values) == 1
            for values in a2_direct_hashes_by_window.values()
        ),
        "female_denominator_assessment": {
            "sufficient_to_estimate_generalization": False,
            "reason": (
                f"Only {denominator} confirmed female annotations occur across "
                "the four training windows (per-window counts "
                + ", ".join(
                    f"{name}={int((annotations.training_window == name).sum())}"
                    for name in WINDOWS
                )
                + "). This can detect gross within-window changes but is too "
                "small to distinguish close configurations reliably."
            ),
            "distinct_primary_accuracy_change_signatures": primary_signature_count,
        },
        "ground_truth_limitation": (
            "Female-only ground truth cannot detect a male-to-male identity swap."
        ),
        "data_scope": (
            "Training windows only. Validation, held-out, and critical-event "
            "method outputs were not read."
        ),
    }
    audit = {
        "schema_version": 1,
        "data_scope": selection["data_scope"],
        "annotations_path": str(ANNOTATIONS),
        "annotations_sha256": sha256(ANNOTATIONS),
        "training_annotation_rows": denominator,
        "calibration_backend": calibration.backend_name,
        "calibration_score_unit": calibration.score_unit,
        "method_results": records,
        "selection": selection,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "training_motion_sweep_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    (OUTPUT_ROOT / "training_motion_sweep_selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
    )
    assignments = pd.concat(assignment_parts, ignore_index=True)
    assignments.to_csv(
        OUTPUT_ROOT / "training_motion_sweep_assignments.csv", index=False
    )
    summary_rows = []
    for config_name, record in records.items():
        summary_rows.append(
            {
                "config": config_name,
                **record["configuration"],
                **record["aggregate"],
            }
        )
    pd.json_normalize(summary_rows, sep=".").to_csv(
        OUTPUT_ROOT / "training_motion_sweep_summary.csv", index=False
    )
    print(json.dumps(selection, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
