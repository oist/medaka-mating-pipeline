"""Independently authenticate and score the frozen Round8 full-length finalists.

This scorer is deliberately separate from ``audit_and_score_postfreeze.py``:
the latter is already hash-bound by the critical quality seal. This script
refuses to inspect scientific outputs until the executor's finalized marker is
present and its exact reviewer-entered SHA-256 matches ``FINALIZED_SHA256``.
"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


REPO = Path(__file__).resolve().parents[4]
REPORT = REPO / "report/20260731/round8_mouth_tracking_ablation"
QUALITY = REPORT / "quality_review"
ROOT = REPORT / "full_length_postfreeze"
FREEZE = REPORT / "frozen_thresholds.json"
FINALIZED = ROOT / "STAGE_FINALIZED.json"
BASE_SCORER = QUALITY / "audit_and_score_postfreeze.py"
METRIC_HELPER = QUALITY / "recompute_method_metrics.py"
LEDGER = REPO / "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv"
CALIBRATION_DIR = REPO / "calibration_results/fixed_refractive_192x192x186"

FREEZE_SHA256 = "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
CRITICAL_SEAL_SHA256 = (
    "55a479e1b97a1054b7f195530d1505202d4d87935a1dd2f36b06f51e665fe83e"
)
BASE_SCORER_SHA256 = (
    "5d7e4e8f87f40228ddd185f07476951a2009e468258719aed92b7d059345ca72"
)
METRIC_HELPER_SHA256 = (
    "11e7f14d9ca1e2a12e40cf2fcf8ff4c9d6d516846dd63eed894c6eda38e69242"
)
LEDGER_SHA256 = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
CALIBRATION_BINDINGS = {
    "calibration_manifest.json": (
        "9bb7dda927c01c018c411d3b56da58f3e44379f4d4ce1491c5ab67d33e5bf105"
    ),
    "cam0_bundle_refractive.json": (
        "f6756885f0b4b2c86033f1818f3681a85084ff56f40dd478ccdc66720063106c"
    ),
    "cam1_bundle_refractive.json": (
        "393cbe50dc8d434d97e2a6ec41005abdb92e08d1691370f42a4de978c31af7f0"
    ),
    "cam2_bundle_refractive.json": (
        "87af873ff80710f51c622c9305b20a5219b3aaede3837cdd92c7964ae0cd06f8"
    ),
}
RUNNER_SHA256 = "32a8452da10f0b53ec555cef1212aa4dabbda13ba7e146f7d508deca5c0fb87d"
WRAPPED_RUNNER_CORE_SHA256 = (
    "0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818"
)
FINALIZER_SHA256 = (
    "1a223d5c446989857f2b4dd732df447cd721792b11c846bcbb37ba24c53c909b"
)
PREPARED_SHA256 = (
    "d9c14fcb973f0fa29877443802eeba643fd5b288245988414b13c53d9f1f4aa1"
)
PREAUTH_SHA256 = (
    "bd1ef6119ff770121ad65c1ce87d772190c3d637c479940045ee84a5f81f8b7b"
)
PREDECESSORS = {
    "validation": "297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706",
    "heldout": "3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152",
    "critical": "fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c",
}

FINALIZED_SHA256 = (
    "a9a8ae25cca6ef0c94dc7996cf39680017b9bb58e20ab60473bced96284e07b0"
)
COMPLETED_SHA256 = (
    "7e9e826cc48965d2b0ada1569b63d7a77b3b2972a8e20649eff15bc2de00dca0"
)

FRAME_START = 0
FRAME_END = 348599
FRAME_COUNT = 348600
EXPECTED_ROWS = 3 * FRAME_COUNT
METHODS = ("A0", "A1", "A2", "A3")
NEW_METHODS = ("A1", "A2", "A3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_inside(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"full-length artifact escapes stage root: {path}")
    return path


def authenticate_item(item: dict[str, Any]) -> Path:
    path = resolve_inside(str(item["path"]))
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    if "bytes" in item and int(item["bytes"]) != path.stat().st_size:
        raise ValueError(f"artifact byte-size mismatch: {path}")
    return path


def all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_strings(child)


def exact_scope(value: dict[str, Any]) -> bool:
    return (
        value.get("experiment") == "20251225_K_1_2_FCMC"
        and value.get("round_tag") == "round8_260729_snn031_m6"
        and value.get("inclusive_frame_interval") == [FRAME_START, FRAME_END]
        and value.get("expected_source_frames") == FRAME_COUNT
        and value.get("shared_geometry_methods") == list(NEW_METHODS)
        and value.get("new_full_length_methods") == list(NEW_METHODS)
        and value.get("unchanged_reference_only") == ["A0"]
        and value.get("forbidden_full_length_methods") == ["A4", "A5"]
        and value.get("selection_or_tuning_permitted") is False
        and value.get("settings_change_permitted") is False
    )


def authenticate_full_length() -> tuple[dict[str, Any], dict[str, Any]]:
    if FINALIZED_SHA256.startswith("__"):
        raise ValueError("reviewer has not bound the finalized marker hash")
    if not FINALIZED.is_file() or sha256(FINALIZED) != FINALIZED_SHA256:
        raise ValueError("full-length finalized marker is absent or hash-mismatched")
    if sha256(FREEZE) != FREEZE_SHA256:
        raise ValueError("freeze changed")
    if sha256(BASE_SCORER) != BASE_SCORER_SHA256:
        raise ValueError("sealed base scorer changed")
    if sha256(METRIC_HELPER) != METRIC_HELPER_SHA256:
        raise ValueError("independent metric helper changed")
    if sha256(LEDGER) != LEDGER_SHA256:
        raise ValueError("confirmed-female ledger changed")
    for filename, expected_hash in CALIBRATION_BINDINGS.items():
        calibration_path = CALIBRATION_DIR / filename
        if not calibration_path.is_file() or sha256(calibration_path) != expected_hash:
            raise ValueError(f"refractive calibration binding changed: {filename}")

    finalized = load_json(FINALIZED)
    if (
        finalized.get("schema_version")
        != "round8_bounded_postfreeze_finalized_v1"
        or finalized.get("status") != "finalized_and_fully_authenticated"
        or finalized.get("stage") != "full_length"
        or finalized.get("role") != "production_finalist_full_length_execution"
        or finalized.get("scientific_results_changed_during_finalization") is not False
        or finalized.get("selection_or_tuning_performed") is not False
        or finalized.get("heldout_or_later_stage_opened") is not False
        or finalized.get("finalizer", {}).get("sha256") != FINALIZER_SHA256
        or finalized.get("bindings", {}).get("freeze", {}).get("sha256")
        != FREEZE_SHA256
        or finalized.get("bindings", {}).get("runner", {}).get("sha256")
        != RUNNER_SHA256
        or finalized.get("bindings", {}).get("prepared_manifest", {}).get("sha256")
        != PREPARED_SHA256
        or finalized.get("bindings", {}).get("prelaunch_auth", {}).get("sha256")
        != PREAUTH_SHA256
    ):
        raise ValueError("full-length finalization identity/provenance mismatch")

    current = finalized.get("current_stage_inventory", {})
    if len(current) != finalized.get("current_stage_inventory_count"):
        raise ValueError("finalized current inventory count mismatch")
    current_paths = set()
    for path_text, metadata in current.items():
        current_paths.add(authenticate_item({"path": path_text, **metadata}))
    actual_paths = {
        path.resolve()
        for path in ROOT.rglob("*")
        if path.is_file() and path.resolve() != FINALIZED.resolve()
    }
    if current_paths != actual_paths:
        raise ValueError("finalized recursive inventory mismatch")

    final_audit = finalized.get("completed_marker_inventory_validation", {})
    if (
        final_audit.get("result") != "PASS"
        or final_audit.get("mismatches") != []
        or final_audit.get("recorded_artifact_count")
        != final_audit.get("checked_artifact_count")
        or final_audit.get("unchanged_artifact_count")
        != final_audit.get("recorded_artifact_count")
        or finalized.get("outer_launcher", {}).get("exit_code") != 0
        or finalized.get("outer_launcher", {}).get("stderr", {}).get("bytes") != 0
    ):
        raise ValueError("full-length finalization inventory/launcher check failed")

    completed_path = authenticate_item(finalized["bindings"]["completed_marker"])
    if sha256(completed_path) != COMPLETED_SHA256:
        raise ValueError("full-length completion marker hash mismatch")
    completed = load_json(completed_path)
    if (
        completed.get("schema_version")
        != "round8_full_length_postfreeze_completed_v1"
        or completed.get("status") != "completed_and_authenticated"
        or completed.get("stage") != "full_length"
        or completed.get("role") != "production_finalist_full_length_execution"
        or completed.get("freeze", {}).get("sha256") != FREEZE_SHA256
        or completed.get("runner", {}).get("sha256") != RUNNER_SHA256
        or completed.get("wrapped_audited_core", {}).get("sha256")
        != WRAPPED_RUNNER_CORE_SHA256
        or completed.get("prepared_manifest", {}).get("sha256") != PREPARED_SHA256
        or completed.get("prelaunch_auth", {}).get("sha256") != PREAUTH_SHA256
        or completed.get("selection_or_tuning_performed") is not False
        or completed.get("frozen_settings_changed") is not False
        or completed.get("new_full_length_methods") != list(NEW_METHODS)
        or completed.get("reference_only_methods") != ["A0"]
        or completed.get("excluded_methods") != ["A4", "A5"]
        or not exact_scope(completed.get("full_length_scope", {}))
    ):
        raise ValueError("full-length completion identity/scope mismatch")

    predecessor = completed.get("predecessor_finalized", {})
    if {
        name: item.get("sha256") for name, item in predecessor.items()
    } != PREDECESSORS:
        raise ValueError("predecessor finalization bindings mismatch")
    critical_quality = completed.get("critical_quality", {})
    if (
        critical_quality.get("sha256") != CRITICAL_SEAL_SHA256
        or critical_quality.get("independent_quality_verdict") != "PASS"
    ):
        raise ValueError("critical quality gate binding mismatch")
    critical_path = REPO / critical_quality["path"]
    if not critical_path.is_file() or sha256(critical_path) != CRITICAL_SEAL_SHA256:
        raise ValueError("critical quality gate changed")

    planned = completed.get("planned_commands", {})
    expected_processes = {
        "shared_geometry.fresh",
        "shared_geometry.resume",
        "A1.fresh",
        "A1.repeat",
        "A2.fresh",
        "A2.repeat",
        "A3.fresh",
        "A3.repeat",
    }
    if set(planned) != expected_processes:
        raise ValueError("full-length planned command coverage mismatch")
    forbidden_tokens = {"A4", "A5", "a4", "a5"}
    if any(token in forbidden_tokens for token in all_strings(planned)):
        raise ValueError("forbidden method appears in full-length command plan")

    records = completed.get("process_records", {})
    if set(records) != expected_processes:
        raise ValueError("full-length process record coverage mismatch")
    for item in records.values():
        record_path = authenticate_item(item)
        record = load_json(record_path)
        if (
            record.get("status") != "completed"
            or record.get("return_code") != 0
            or record.get("stderr", {}).get("bytes") != 0
            or float(item.get("wall_seconds", -1)) < 0
            or int(item.get("peak_rss_bytes", 0)) <= 0
        ):
            raise ValueError(f"failed full-length process record: {record_path}")

    summaries = completed.get("summaries", {})
    if set(summaries) != {"shared_geometry", *NEW_METHODS}:
        raise ValueError("full-length summary coverage mismatch")
    summary_objects = {}
    for key, item in summaries.items():
        path = authenticate_item(item)
        summary_objects[key] = load_json(path)
    geometry = summary_objects["shared_geometry"]
    configuration = geometry.get("configuration", {})
    if (
        geometry.get("status") != "completed"
        or configuration.get("frame_start") != FRAME_START
        or configuration.get("frame_end") != FRAME_END
        or configuration.get("block_frames") != 500
        or configuration.get("evaluation_batch_frames") != 128
        or configuration.get("num_candidates") != 6
        or configuration.get("num_individuals") != 3
        or configuration.get("anchor_node") != "mouth"
        or configuration.get("require_anchor_node") is not True
        or configuration.get("filter_fp") is not False
        or configuration.get("tail_coordinates_consumed") is not False
        or configuration.get("calibration_backend") != "refractive"
        or configuration.get("score_unit") != "mm^2"
        or geometry.get("reused_blocks") != geometry.get("total_blocks")
    ):
        raise ValueError("full-length strict-mouth geometry contract mismatch")

    determinism = completed.get("deterministic_fresh_resume_equivalence", {})
    if set(determinism) != {"shared_geometry", *NEW_METHODS}:
        raise ValueError("full-length determinism coverage mismatch")
    geometry_determinism = determinism["shared_geometry"]
    if (
        geometry_determinism.get("result") != "PASS"
        or geometry_determinism.get("fresh_scientific_contract")
        != geometry_determinism.get("resume_scientific_contract")
        or geometry_determinism.get("reused_blocks")
        != geometry_determinism.get("total_blocks")
    ):
        raise ValueError("full-length geometry fresh/resume mismatch")
    for method in NEW_METHODS:
        evidence = determinism[method]
        if (
            evidence.get("result") != "PASS"
            or evidence.get("fresh_scientific_output_sha256")
            != evidence.get("repeat_scientific_output_sha256")
        ):
            raise ValueError(f"full-length {method} fresh/repeat mismatch")

    inventory = completed.get("artifact_inventory", {})
    if len(inventory) != completed.get("artifact_inventory_count"):
        raise ValueError("full-length completion inventory count mismatch")
    for path_text, metadata in inventory.items():
        authenticate_item({"path": path_text, **metadata})
        if current.get(path_text) != metadata:
            raise ValueError(f"post-completion artifact changed: {path_text}")
    extras = {
        str(completed_path.relative_to(REPO)).replace("/", "\\"),
        str((ROOT / "DETACHED_LAUNCH_EXIT.json").relative_to(REPO)).replace(
            "/", "\\"
        ),
        str((ROOT / "detached_launcher.stdout.log").relative_to(REPO)).replace(
            "/", "\\"
        ),
        str((ROOT / "detached_launcher.stderr.log").relative_to(REPO)).replace(
            "/", "\\"
        ),
    }
    if set(current) != set(inventory) | extras:
        raise ValueError("finalized inventory is not completion inventory plus closure files")

    # Authenticate the unchanged A0 reference in place without copying or rerunning it.
    a0_artifacts = completed.get("a0_reference", {}).get("artifacts", {})
    if set(a0_artifacts) != {
        "geometry_csv",
        "identified_csv",
        "best_json",
        "best_pre_link_json",
        "reconstruction_provenance",
        "identified_projection_provenance",
        "stage3_status",
    }:
        raise ValueError("A0 reference contract incomplete")
    for item in a0_artifacts.values():
        path = Path(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256(path) != item["sha256"]
        ):
            raise ValueError(f"A0 reference authentication failed: {path}")

    return completed, summary_objects


def source_switch_count(table: pd.DataFrame) -> int | None:
    if "source_index" not in table:
        return None
    switches = 0
    for _, rows in table.groupby("method_id", sort=True):
        rows = rows.sort_values("frame", kind="stable")
        frames = rows.frame.to_numpy(np.int64)
        sources = pd.to_numeric(rows.source_index, errors="coerce").to_numpy(float)
        valid = (
            (np.diff(frames) == 1)
            & np.isfinite(sources[:-1])
            & np.isfinite(sources[1:])
        )
        switches += int((sources[:-1][valid] != sources[1:][valid]).sum())
    return switches


def status_counts(table: pd.DataFrame) -> dict[str, int]:
    result: Counter[str] = Counter()
    for column in ("identity_status", "assignment_status"):
        if column in table:
            result.update(
                {
                    f"{column}:{key}": int(value)
                    for key, value in table[column]
                    .astype("string")
                    .fillna("missing")
                    .value_counts()
                    .items()
                }
            )
    return dict(result)


def full_resource_qc(completed: dict[str, Any]) -> dict[str, Any]:
    compact = []
    rollup = {}
    for index, (record_key, record) in enumerate(
        completed["process_records"].items()
    ):
        method, purpose = record_key.split(".", 1)
        wall = float(record["wall_seconds"])
        peak = int(record["peak_rss_bytes"])
        compact.append(
            {
                "index": index,
                "record_key": record_key,
                "method": method,
                "purpose": purpose,
                "window": None,
                "wall_seconds": wall,
                "peak_rss_bytes": peak,
                "record_sha256": record["sha256"],
            }
        )
        method_rollup = rollup.setdefault(
            method,
            {
                "process_count": 0,
                "summed_wall_seconds": 0.0,
                "max_peak_rss_bytes": 0,
            },
        )
        method_rollup["process_count"] += 1
        method_rollup["summed_wall_seconds"] += wall
        method_rollup["max_peak_rss_bytes"] = max(
            method_rollup["max_peak_rss_bytes"], peak
        )
    return {
        "measurement_contract": (
            "Wall time and peak RSS are authenticated external process-runner "
            "measurements. Each full-stage key is parsed as method.purpose; "
            "fresh and repeat are separate purposes of the same method."
        ),
        "process_records": compact,
        "per_method_rollup": rollup,
        "stage_wall_seconds": float(completed["wall_seconds"]),
        "stage_peak_rss_bytes": int(completed["peak_rss_bytes_max"]),
    }


def compact_scientific(table: pd.DataFrame) -> pd.DataFrame:
    return table[
        ["frame", "method_id", "X", "Y", "Z", "is_interpolated"]
    ].sort_values(["frame", "method_id"], kind="stable").reset_index(drop=True)


def compare_compact(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    same_keys = (
        left[["frame", "method_id"]].equals(right[["frame", "method_id"]])
    )
    left_xyz = left[["X", "Y", "Z"]].to_numpy(float)
    right_xyz = right[["X", "Y", "Z"]].to_numpy(float)
    both_finite = np.isfinite(left_xyz).all(axis=1) & np.isfinite(right_xyz).all(
        axis=1
    )
    finite_mismatch = np.isfinite(left_xyz).all(axis=1) != np.isfinite(
        right_xyz
    ).all(axis=1)
    distances = np.linalg.norm(left_xyz[both_finite] - right_xyz[both_finite], axis=1)
    coordinate_equal = np.isclose(
        left_xyz, right_xyz, rtol=0, atol=0, equal_nan=True
    ).all(axis=1)
    coordinate_equal_by_frame = coordinate_equal.reshape(FRAME_COUNT, 3)
    fixed_identity_different_frames = (~coordinate_equal_by_frame).any(axis=1)
    fully_populated_both = (
        both_finite.reshape(FRAME_COUNT, 3).all(axis=1)
    )
    left_candidate_hash = pd.util.hash_pandas_object(
        left[["X", "Y", "Z", "is_interpolated"]], index=False
    ).to_numpy(np.uint64).reshape(FRAME_COUNT, 3)
    right_candidate_hash = pd.util.hash_pandas_object(
        right[["X", "Y", "Z", "is_interpolated"]], index=False
    ).to_numpy(np.uint64).reshape(FRAME_COUNT, 3)
    candidate_set_equal = (
        np.sort(left_candidate_hash, axis=1)
        == np.sort(right_candidate_hash, axis=1)
    ).all(axis=1)
    return {
        "same_frame_identity_grid": bool(same_keys),
        "rows": int(len(left)),
        "candidate_set_equal_frames": int(candidate_set_equal.sum()),
        "candidate_set_different_frames": int((~candidate_set_equal).sum()),
        "fixed_identity_coordinate_equal_rows": int(coordinate_equal.sum()),
        "fixed_identity_assignment_different_frames": int(
            fixed_identity_different_frames.sum()
        ),
        "fixed_identity_assignment_different_fully_populated_frames": int(
            (fixed_identity_different_frames & fully_populated_both).sum()
        ),
        "finite_availability_mismatch_rows": int(finite_mismatch.sum()),
        "coordinate_exact_mismatch_rows": int((~coordinate_equal).sum()),
        "finite_pair_distance_mm": {
            "n": int(len(distances)),
            "p95": float(np.quantile(distances, 0.95)) if len(distances) else None,
            "p99": float(np.quantile(distances, 0.99)) if len(distances) else None,
            "max": float(np.max(distances)) if len(distances) else None,
        },
    }


def main() -> None:
    completed, summaries = authenticate_full_length()
    base = load_module(BASE_SCORER, "sealed_postfreeze_scorer")
    helper = load_module(METRIC_HELPER, "full_length_metric_helper")
    freeze = load_json(FREEZE)
    annotations = pd.read_csv(LEDGER, low_memory=False)
    annotations = annotations[
        annotations.frame.between(FRAME_START, FRAME_END)
    ].copy()
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")

    a0_items = completed["a0_reference"]["artifacts"]
    paths = {
        "A0": Path(a0_items["identified_csv"]["path"]),
        **{
            method: Path(summaries[method]["outputs"]["direct"]["path"])
            for method in NEW_METHODS
        },
    }
    for method in NEW_METHODS:
        base.assert_settings_frozen(method, summaries[method], freeze)
        if sha256(paths[method]) != summaries[method]["outputs"]["direct"]["sha256"]:
            raise ValueError(f"{method} direct output hash mismatch")

    a0_residuals = base.a0_prelink_residuals(
        Path(a0_items["best_pre_link_json"]["path"]),
        {"full_experiment": (FRAME_START, FRAME_END)},
    )["full_experiment"]
    method_metrics = {}
    assignments_parts = []
    critical_event_parts = []
    critical_event_metrics = {}
    compact = {}
    for method in METHODS:
        table = base.load_csv_window(
            paths[method],
            FRAME_START,
            FRAME_END,
            preferred_id="final_fish_id" if method == "A0" else None,
        )
        if (
            len(table) != EXPECTED_ROWS
            or table.frame.min() != FRAME_START
            or table.frame.max() != FRAME_END
            or table.duplicated(["frame", "base_method_id"]).any()
            or not table.groupby("frame").size().eq(3).all()
        ):
            raise ValueError(f"{method} lacks the exact full-length 3xframe grid")
        track = helper.method_track_metrics(table.copy())
        jumps, direct_residuals = base.direct_values(table)
        residuals = a0_residuals if method == "A0" else direct_residuals
        assignments, female = helper.female_assignments(
            annotations,
            table,
            method=method,
            anchor="centroid" if method == "A0" else "mouth",
            calibration=calibration,
        )
        assignments["stage"] = "full_length"
        assignments_parts.append(assignments)
        critical_event = table[table.frame.between(34547, 34555)].copy()
        critical_event.insert(0, "stage", "full_length")
        critical_event.insert(1, "method", method)
        critical_event_parts.append(critical_event)
        event_jumps, event_direct_residuals = base.direct_values(critical_event)
        event_xyz = critical_event[["X", "Y", "Z"]].to_numpy(float)
        event_source_values = {}
        event_source_switches = 0
        if "source_index" in critical_event:
            for identity, rows in critical_event.groupby(
                "base_method_id", sort=True
            ):
                rows = rows.sort_values("frame", kind="stable")
                values = pd.to_numeric(
                    rows.source_index, errors="coerce"
                ).dropna().to_numpy(float)
                event_source_values[str(identity)] = sorted(
                    int(value) for value in set(values.tolist())
                )
                event_source_switches += int(
                    (values[:-1] != values[1:]).sum()
                )
        if method == "A0":
            first = 34547 * 3
            last = (34555 + 1) * 3
            event_residuals = a0_residuals[first:last]
        else:
            event_residuals = event_direct_residuals
        critical_event_metrics[method] = {
            "inclusive_frames": [34547, 34555],
            "rows": int(len(critical_event)),
            "finite_direct_rows": int(
                (
                    np.isfinite(event_xyz).all(axis=1)
                    & ~critical_event.is_interpolated.to_numpy(bool)
                ).sum()
            ),
            "unavailable_or_unresolved_rows": int(
                helper.method_track_metrics(critical_event.copy())[
                    "unavailable_or_unresolved_row_count"
                ]
            ),
            "direct_jump_mm": base.distribution(event_jumps),
            "direct_or_prelink_ray_residual_mm2": base.distribution(
                event_residuals
            ),
            "source_index_values_by_identity": event_source_values,
            "source_index_switches": event_source_switches,
            "identity_status_counts": status_counts(critical_event),
            "normalized_scientific_sha256": helper.normalized_scientific_hash(
                critical_event
            ),
        }
        method_metrics[method] = {
            "path": str(paths[method]),
            "sha256": sha256(paths[method]),
            "track_qc": track,
            "female_identity": female["mapped"],
            "direct_jump_mm": base.distribution(jumps),
            "jump_ge_7_5mm_count": int((jumps >= 7.5).sum()),
            "jump_gt_8mm_count": int((jumps > 8.0).sum()),
            "direct_or_prelink_ray_residual_mm2": base.distribution(residuals),
            "identity_status_counts": status_counts(table),
            "source_index_switch_count": source_switch_count(table),
            "normalized_scientific_sha256": helper.normalized_scientific_hash(
                table
            ),
        }
        if method in NEW_METHODS:
            compact[method] = compact_scientific(table)
        del table
        gc.collect()

    a3_audit_item = summaries["A3"]["outputs"]["assignment_audit"]
    a3_audit_path = Path(a3_audit_item["path"])
    if sha256(a3_audit_path) != a3_audit_item["sha256"]:
        raise ValueError("A3 assignment audit hash mismatch")
    a3_audit = pd.read_csv(a3_audit_path, low_memory=False)
    a3_margins = pd.to_numeric(
        a3_audit.assignment_margin, errors="coerce"
    ).to_numpy(float)
    a3_event_audit = a3_audit[a3_audit.frame.between(34547, 34555)].copy()
    a3_event_margins = pd.to_numeric(
        a3_event_audit.assignment_margin, errors="coerce"
    ).to_numpy(float)
    a3_assignment_audit = {
        "path": str(a3_audit_path),
        "sha256": a3_audit_item["sha256"],
        "rows": int(len(a3_audit)),
        "resolved_rows": int(a3_audit.resolved.fillna(False).astype(bool).sum()),
        "ambiguous_rows": int(
            a3_audit.ambiguous.fillna(False).astype(bool).sum()
        ),
        "constraint_conflict_rows": int(
            a3_audit.constraint_conflict.fillna(False).astype(bool).sum()
        ),
        "assignment_margin": base.distribution(a3_margins),
        "critical_event": {
            "inclusive_frames": [34547, 34555],
            "rows": int(len(a3_event_audit)),
            "all_resolved": bool(
                a3_event_audit.resolved.fillna(False).astype(bool).all()
            ),
            "ambiguous_rows": int(
                a3_event_audit.ambiguous.fillna(False).astype(bool).sum()
            ),
            "constraint_conflict_rows": int(
                a3_event_audit.constraint_conflict.fillna(False).astype(bool).sum()
            ),
            "assignment_margin": base.distribution(a3_event_margins),
        },
    }

    pairwise = {
        f"{left}_vs_{right}": compare_compact(compact[left], compact[right])
        for left, right in (("A1", "A2"), ("A1", "A3"), ("A2", "A3"))
    }
    all_three_identical = (
        len(
            {
                method_metrics[method]["normalized_scientific_sha256"]
                for method in NEW_METHODS
            }
        )
        == 1
    )
    resources = full_resource_qc(completed)
    output = {
        "schema_version": 1,
        "status": "full_length_independently_authenticated_and_scored",
        "stage": "full_length",
        "role": "production_finalist_full_length_qc",
        "selection_or_tuning_performed": False,
        "frozen_settings_changed": False,
        "inclusive_frame_interval": [FRAME_START, FRAME_END],
        "authentication": {
            "result": "PASS",
            "finalized_current_inventory_files": completed.get(
                "artifact_inventory_count"
            )
            + 4,
            "completion_inventory_files_unchanged": completed.get(
                "artifact_inventory_count"
            ),
            "process_records_authenticated": 8,
            "geometry_resume_passed": True,
            "finalist_fresh_repeat_checks_passed": 3,
            "a0_unchanged_reference_authenticated": True,
            "forbidden_A4_A5_absent": True,
        },
        "resources": resources,
        "methods": method_metrics,
        "cross_method": {
            "A1_A2_A3_normalized_scientific_identical": all_three_identical,
            "pairwise": pairwise,
        },
        "a3_assignment_audit": a3_assignment_audit,
        "critical_event_34547_34555": {
            "role": "descriptive_nonconfirmatory",
            "methods": critical_event_metrics,
            "female_annotation_rows": int(
                annotations.frame.between(34547, 34555).sum()
            ),
            "biological_identity_limitation": (
                "The available ledger has no female row in this interval and "
                "cannot observe a male-to-male exchange. Residual, smoothness, "
                "and source-slot continuity are not biological identity proof."
            ),
        },
        "bindings": {
            "freeze": {"path": str(FREEZE), "sha256": FREEZE_SHA256},
            "critical_quality_seal": {
                "path": str(REPORT / "quality_review/CRITICAL_QUALITY_REVIEW_COMPLETED.json"),
                "sha256": CRITICAL_SEAL_SHA256,
            },
            "full_length_finalized": {
                "path": str(FINALIZED),
                "sha256": FINALIZED_SHA256,
            },
            "full_length_completed": {
                "path": str(ROOT / "STAGE_COMPLETED.json"),
                "sha256": COMPLETED_SHA256,
            },
            "prepared_manifest": {
                "path": str(ROOT / "STAGE_PREPARED.json"),
                "sha256": PREPARED_SHA256,
            },
            "base_scorer": {
                "path": str(BASE_SCORER),
                "sha256": BASE_SCORER_SHA256,
            },
            "metric_helper": {
                "path": str(METRIC_HELPER),
                "sha256": METRIC_HELPER_SHA256,
            },
            "confirmed_female_annotation_ledger": {
                "path": str(LEDGER),
                "sha256": LEDGER_SHA256,
            },
            "refractive_calibration": {
                filename: {
                    "path": str(CALIBRATION_DIR / filename),
                    "sha256": expected_hash,
                }
                for filename, expected_hash in CALIBRATION_BINDINGS.items()
            },
        },
    }
    metrics_path = QUALITY / "full_length_independent_metrics.json"
    metrics_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assignments_path = QUALITY / "full_length_independent_female_assignments.csv"
    pd.concat(assignments_parts, ignore_index=True).to_csv(
        assignments_path, index=False
    )
    critical_event_path = QUALITY / "full_length_critical_event_rows_34547_34555.csv"
    pd.concat(critical_event_parts, ignore_index=True).to_csv(
        critical_event_path, index=False
    )
    method_rows = []
    for method, record in method_metrics.items():
        track = record["track_qc"]
        female = record["female_identity"]
        jumps = record["direct_jump_mm"]
        residuals = record["direct_or_prelink_ray_residual_mm2"]
        method_rows.append(
            {
                "method": method,
                "female_eligible": female["count"],
                "female_mapped": female["mapped_count"],
                "female_full_accuracy": female[
                    "full_eligible_best_fixed_accuracy"
                ],
                "female_selective_accuracy": female[
                    "selective_mapped_best_fixed_accuracy"
                ],
                "female_identity_changes": female["observed_identity_changes"],
                "all_three_direct_fraction": track[
                    "all_three_direct_noninterpolated_frame_share"
                ],
                "unavailable_or_unresolved_fraction": track[
                    "unavailable_or_unresolved_row_share"
                ],
                "identity_count": track["identity_count"],
                "base_identity_count": track["base_identity_count"],
                "direct_jump_p95_mm": jumps["p95"],
                "direct_jump_p99_mm": jumps["p99"],
                "direct_jump_max_mm": jumps["max"],
                "jump_ge_7_5mm_count": record["jump_ge_7_5mm_count"],
                "jump_gt_8mm_count": record["jump_gt_8mm_count"],
                "residual_p95_mm2": residuals["p95"],
                "residual_p99_mm2": residuals["p99"],
                "residual_max_mm2": residuals["max"],
                "source_index_switch_count": record["source_index_switch_count"],
            }
        )
    method_summary_path = QUALITY / "full_length_independent_method_summary.csv"
    pd.DataFrame(method_rows).to_csv(method_summary_path, index=False)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "female_assignments": str(assignments_path),
                "female_assignments_sha256": sha256(assignments_path),
                "method_summary": str(method_summary_path),
                "method_summary_sha256": sha256(method_summary_path),
                "critical_event_rows": str(critical_event_path),
                "critical_event_rows_sha256": sha256(critical_event_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
