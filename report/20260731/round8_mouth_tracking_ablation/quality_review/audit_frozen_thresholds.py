"""Recursive semantic audit of the immutable pre-validation Round8 freeze."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
REPORT = REPO / "report/20260731/round8_mouth_tracking_ablation"
FREEZE = REPORT / "frozen_thresholds.json"
OUTPUT = REPORT / "quality_review/frozen_thresholds_independent_audit.json"
EXPECTED_FREEZE_SHA256 = (
    "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
)
EXPECTED_LEDGER_SHA256 = (
    "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
)
EXPECTED_IMPLEMENTATION_PATHS = {
    "scripts\\reconstruction_3d\\calibration.py",
    "scripts\\reconstruction_3d\\candidate_cache.py",
    "scripts\\reconstruction_3d\\conservative_cross_view_paths.py",
    "scripts\\reconstruction_3d\\conservative_mouth_tracklets.py",
    "scripts\\reconstruction_3d\\cross_frame_tracking.py",
    "scripts\\reconstruction_3d\\deferred_identity_tracking.py",
    "scripts\\reconstruction_3d\\fill_missing_3d.py",
    "scripts\\reconstruction_3d\\mouth_ablation.py",
    "scripts\\reconstruction_3d\\mouth_tracklet_ablation.py",
    "scripts\\reconstruction_3d\\windowed_tracklet_core.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def authenticate(item: dict[str, str]) -> Path:
    path = resolve(item["path"])
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"path/hash mismatch: {path}")
    return path


def reject_stringified_structures(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            reject_stringified_structures(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_stringified_structures(child, f"{location}[{index}]")
    elif isinstance(value, str):
        stripped = value.strip()
        if (
            stripped.startswith("@{")
            or "System.Object[]" in stripped
            or stripped.startswith("[System.")
        ):
            raise ValueError(f"double/stringified structure at {location}")


def assert_interval(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] > value[1]
    ):
        raise ValueError(f"invalid integer interval: {name}={value!r}")
    return int(value[0]), int(value[1])


def all_cohort_intervals(cohort: dict[str, Any]) -> list[tuple[str, int, int]]:
    records: list[tuple[str, int, int]] = []
    for phase in ("training", "validation", "held_out"):
        for category in (
            "reviewed_contact_events",
            "matched_control_windows",
            "round8_risk_windows",
        ):
            for item in cohort.get(phase, {}).get(category, []):
                start, end = assert_interval(
                    item["frames"], f"{phase}.{category}.{item['name']}"
                )
                records.append((f"{phase}:{item['name']}", start, end))
    return records


def main() -> None:
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise ValueError("unexpected freeze hash")
    freeze = load(FREEZE)
    reject_stringified_structures(freeze)
    if (
        freeze.get("schema_version")
        != "round8_mouth_tracking_ablation_freeze_v3"
        or freeze.get("status")
        != "frozen_before_validation_heldout_and_postfreeze_critical_reporting"
        or freeze.get("experiment") != "20251225_K_1_2_FCMC"
        or freeze.get("round_tag") != "round8_260729_snn031_m6"
        or freeze.get("random_seed") != 20260731
        or freeze.get("inclusive_frame_intervals") is not True
    ):
        raise ValueError("freeze identity/status/seed mismatch")
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    if frozen_at.tzinfo is None:
        raise ValueError("freeze timestamp lacks timezone")

    generator_path = authenticate(freeze["generator"])
    implementation = freeze["implementation_sha256"]
    if set(implementation) != EXPECTED_IMPLEMENTATION_PATHS:
        raise ValueError("implementation file set is incomplete or expanded")
    for path_text, expected_hash in implementation.items():
        path = REPO / path_text
        if sha256(path) != expected_hash:
            raise ValueError(f"implementation changed after freeze: {path}")

    source = freeze["source_contract"]
    input_path = authenticate(source["input_provenance"])
    cohort_path = authenticate(source["cohort_protocol"])
    deviation_path = authenticate(source["protocol_deviations"])
    input_provenance = load(input_path)
    cohort = load(cohort_path)
    deviations = load(deviation_path)
    if source["source_slps"] != input_provenance["source_slps"]:
        raise ValueError("source SLP contract differs from input provenance")
    if (
        source["canonical_candidate_cache"]["policy"]
        != input_provenance["canonical_strict_mouth_candidate_caches"]["policy"]
        or source["canonical_candidate_cache"]["views"]
        != input_provenance["canonical_strict_mouth_candidate_caches"]["views"]
    ):
        raise ValueError("candidate-cache contract differs from provenance")
    cache_dir = resolve(source["canonical_candidate_cache"]["directory"])
    for view, record in source["canonical_candidate_cache"]["views"].items():
        cache_path = cache_dir / f"{view}.candidates.h5"
        if sha256(cache_path) != record["cache_sha256"]:
            raise ValueError(f"candidate cache drift: {view}")
    if freeze["frozen_methods"]["A0"]["reference_artifact_hashes"] != (
        input_provenance["a0_unchanged_reference"]
    ):
        raise ValueError("A0 reference hashes differ from provenance")

    calibration = source["calibration"]
    provenance_calibration = input_provenance["calibration"]
    if calibration != provenance_calibration:
        raise ValueError("calibration contract differs from provenance")
    if (
        calibration["resolved_backend"] != "refractive"
        or calibration["score_unit"] != "mm^2"
    ):
        raise ValueError("wrong calibration backend/unit")
    calibration_pairs = [
        ("manifest_path", "calibration_manifest_sha256"),
        ("cam0_bundle_path", "cam0_bundle_sha256"),
        ("cam1_bundle_path", "cam1_bundle_sha256"),
        ("cam2_bundle_path", "cam2_bundle_sha256"),
    ]
    for path_key, hash_key in calibration_pairs:
        if sha256(resolve(calibration[path_key])) != calibration[hash_key]:
            raise ValueError(f"calibration file drift: {path_key}")

    ledger_source = source["confirmed_female_annotation_ledger"]
    ledger_training = freeze["training_artifacts"][
        "confirmed_female_annotation_ledger"
    ]
    if (
        ledger_source != ledger_training
        or ledger_source["sha256"] != EXPECTED_LEDGER_SHA256
    ):
        raise ValueError("female annotation ledger binding mismatch")
    ledger_path = authenticate(ledger_source)

    controlled = freeze["controlled_geometry"]
    expected_controlled = {
        **input_provenance["controlled_reconstruction"],
        **input_provenance["strict_mouth_policy"],
        "calibration_backend": "refractive",
        "residual_score_unit": "mm^2",
    }
    if controlled != expected_controlled:
        raise ValueError("controlled geometry differs from input provenance")

    training = freeze["training_artifacts"]
    scales_path = authenticate(training["frozen_body_length_scales"])
    quality_paths: list[Path] = []
    for path_text, expected_hash in training[
        "quality_and_completion_sha256"
    ].items():
        path = REPO / path_text
        if sha256(path) != expected_hash:
            raise ValueError(f"training selection/completion drift: {path}")
        quality_paths.append(path)
    required_quality_suffixes = {
        "training_motion_sweep_selection.json",
        "training_motion_sweep_audit.json",
        "score_training_motion_sweep.py",
        "training_a3_sweep_audit.json",
        "score_training_a3_sweep.py",
        "training_a4_method_sweep_audit.json",
        "training_a4_method_sweep_summary.csv",
        "audit_training_a4_method_sweep.py",
        "training_a5_hint_audit.json",
        "audit_training_a5_hints.py",
        "SWEEP_COMPLETED.json",
        "TRAINING_A5_COMPLETED.json",
    }
    if not required_quality_suffixes.issubset(
        {path.name for path in quality_paths}
    ):
        raise ValueError("selection/completion/scorer replay hashes are incomplete")
    geometry_paths: list[Path] = []
    for window in (
        "reviewed_event_1",
        "reviewed_event_3",
        "control_for_event_1",
        "control_for_event_3",
    ):
        for key in ("summary", "selected_geometry"):
            geometry_paths.append(
                authenticate(training["strict_mouth_geometry"][window][key])
            )

    methods = freeze["frozen_methods"]
    if methods["A1"]["settings"] != {
        "velocity_weight": 0.0,
        "tangent_weight": 0.0,
        "acceleration_weight": 0.0,
        "uncertainty_growth_per_frame": 0.25,
        "max_motion_gap_frames": None,
        "motion_history_frames": 2,
        "max_jump_mm": 8,
        "max_jump_hard_cap_mm": 80,
        "max_gap_frames": 100,
    }:
        raise ValueError("A1 settings mismatch")
    if (
        methods["A2"]["selected_configuration"] != "A2_v025_h5"
        or methods["A2"]["settings"]
        != {
            "velocity_weight": 0.25,
            "tangent_weight": 0.0,
            "acceleration_weight": 0.0,
            "uncertainty_growth_per_frame": 0.25,
            "max_motion_gap_frames": 20,
            "motion_history_frames": 5,
            "max_jump_mm": 8,
            "max_jump_hard_cap_mm": 80,
            "max_gap_frames": 100,
        }
    ):
        raise ValueError("A2 selection/settings mismatch")
    expected_a3 = {
        "velocity_weight": 0.25,
        "tangent_weight": 0.0,
        "acceleration_weight": 0.05,
        "uncertainty_growth_per_frame": 0.25,
        "max_motion_gap_frames": 20,
        "motion_history_frames": 5,
        "max_jump_mm": 8,
        "max_jump_hard_cap_mm": 80,
        "max_gap_frames": 100,
        "beam_width": 16,
        "entry_margin": 0.5,
        "retention_margin": 2,
        "resolution_margin": 1,
        "resolution_stability_frames": 3,
        "minimum_lookahead_frames": 3,
        "maximum_defer_horizon": 100,
    }
    if (
        methods["A3"]["selected_configuration"] != "A3_a005"
        or methods["A3"]["settings"] != expected_a3
    ):
        raise ValueError("A3 selection/settings mismatch")

    a4_audit_path = REPORT / "quality_review/training_a4_method_sweep_audit.json"
    a4_audit = load(a4_audit_path)
    selected_a4_summary_path = (
        REPORT
        / "training_a4_cutoff_sweep/reviewed_event_1"
        / a4_audit["selected"]
        / "a4_summary.json"
    )
    selected_a4 = load(selected_a4_summary_path)["configuration"]
    a4_settings = methods["A4"]["settings"]
    expected_a4 = {
        key: selected_a4[key]
        for key in (
            "hypothesis_block_frames",
            "evaluation_batch_frames",
            "xyz_bounds",
            "num_candidates",
            "num_individuals",
            "anchor",
            "tail_identity_features",
            "frozen_body_scales_sha256",
            "tracklet_config",
            "group_config",
            "path_config",
            "cache_sha256",
            "calibration_file_sha256",
            "calibration",
            "code_sha256",
        )
    }
    expected_a4["body_length_px"] = selected_a4["frozen_body_length_px"]
    expected_a4["bridge_consistency"] = "forward_backward_reciprocal"
    expected_a4["retained_tracklet_count_mismatch"] = "reject_group"
    if (
        methods["A4"]["selected_configuration"] != a4_audit["selected"]
        or a4_settings != expected_a4
    ):
        raise ValueError("A4 complete replay configuration mismatch")

    a5_audit_path = REPORT / "quality_review/training_a5_hint_audit.json"
    a5_audit = load(a5_audit_path)
    if (
        methods["A5"]["hint_thresholds"] != a5_audit["hint_thresholds"]
        or methods["A5"]["comparison_to_selected_A3"]
        != a5_audit["comparison_to_selected_a3"]
        or methods["A5"]["dependencies"]
        != {"A2": "A2_v025_h5", "A3": "A3_a005", "A4": "s035_m005_g0"}
        or methods["A5"]["role"] != "exploratory_comparison_only"
    ):
        raise ValueError("A5 threshold/dependency/comparator mismatch")
    for name in ("A1", "A2", "A3"):
        if (
            methods[name]["production_finalist"] is not True
            or methods[name]["full_length_finalist_run_required"] is not True
        ):
            raise ValueError(f"{name} finalist booleans malformed")
    for name in ("A4", "A5"):
        if (
            methods[name]["production_finalist"] is not False
            or methods[name]["full_length_finalist_run_required"] is not False
        ):
            raise ValueError(f"{name} nonfinalist booleans malformed")

    policy = freeze["postfreeze_execution_policy"]
    if policy["validation"]["frames"] != cohort["validation"]:
        raise ValueError("validation cohort was not copied structurally")
    if policy["held_out"]["frames"] != cohort["held_out"]:
        raise ValueError("held-out cohort was not copied structurally")
    for key, value in cohort["critical_event_exclusion"].items():
        if policy["critical_event"].get(key) != value:
            raise ValueError(f"critical cohort field mismatch: {key}")
    if (
        policy["validation"]["role"] != "report_only"
        or policy["validation"]["may_change_frozen_settings_or_finalists"]
        is not False
        or policy["held_out"]["role"]
        != "sole_postfreeze_generalization_evidence"
        or policy["held_out"]["may_change_frozen_settings_or_finalists"]
        is not False
        or policy["critical_event"]["confirmatory"] is not False
        or policy["critical_event"]["may_change_frozen_settings_or_finalists"]
        is not False
        or policy["full_length_runs"]["required_new_methods"]
        != ["A1", "A2", "A3"]
        or policy["full_length_runs"]["reference_only"] != ["A0"]
        or policy["full_length_runs"]["bounded_exploratory_only"]
        != ["A4", "A5"]
    ):
        raise ValueError("post-freeze role/finalist policy mismatch")

    cohort_intervals = all_cohort_intervals(cohort)
    for index, (left_name, left_start, left_end) in enumerate(cohort_intervals):
        for right_name, right_start, right_end in cohort_intervals[index + 1 :]:
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(
                    f"declared cohort overlap: {left_name} and {right_name}"
                )
    exclusion_start, exclusion_end = assert_interval(
        cohort["critical_event_exclusion"]["excluded_from_all_tuning_and_validation"],
        "critical exclusion",
    )
    for name, start, end in cohort_intervals:
        if max(start, exclusion_start) <= min(end, exclusion_end):
            raise ValueError(f"cohort overlaps critical exclusion: {name}")

    superseded = freeze["supersedes_invalid_freezes"]
    if [item["sha256"] for item in superseded] != [
        "412e365b8c8fc0dc09061dc6c3ac235897d89f213a003f940c615c398ccb206e",
        "4d74438fad0d244c383ac63acd7f23575ab3869f752b2090cb2eafe9ec9b92ca",
    ]:
        raise ValueError("invalid-freeze quarantine history mismatch")
    for item in superseded:
        path = resolve(item["quarantined_path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"quarantined invalid freeze drift: {path}")

    frozen_utc = frozen_at.astimezone(timezone.utc)
    timestamp_paths = [
        generator_path,
        input_path,
        cohort_path,
        deviation_path,
        ledger_path,
        scales_path,
        *quality_paths,
        *geometry_paths,
    ]
    late = [
        str(path)
        for path in timestamp_paths
        if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) > frozen_utc
    ]
    if late:
        raise ValueError(f"training/source artifact postdates freeze: {late}")

    absence = freeze["postfreeze_root_absence_check"]
    forbidden_prefixes = tuple(absence["forbidden_name_prefixes"])
    matching_roots = sorted(
        child.name
        for child in REPORT.iterdir()
        if child.is_dir() and child.name.lower().startswith(forbidden_prefixes)
    )
    if (
        absence["result"] != "PASS"
        or absence["matching_roots"] != []
        or matching_roots
    ):
        raise ValueError(f"post-freeze roots existed before audit: {matching_roots}")

    result = {
        "schema_version": 2,
        "status": "freeze_v3_authenticated_before_postfreeze_outputs",
        "freeze": {
            "path": str(FREEZE),
            "sha256": sha256(FREEZE),
            "frozen_at": freeze["frozen_at"],
        },
        "audit_code_sha256": sha256(Path(__file__)),
        "recursive_stringified_structure_check": "PASS",
        "source_contract_structured_equality": "PASS",
        "cohort_interval_disjointness": "PASS",
        "implementation_files_authenticated": len(implementation),
        "quality_and_completion_artifacts_authenticated": len(quality_paths),
        "training_geometry_artifacts_authenticated": len(geometry_paths),
        "female_ledger_sha256": ledger_source["sha256"],
        "quarantined_invalid_freezes_authenticated": len(superseded),
        "all_checked_training_artifacts_predate_freeze": True,
        "production_finalists": ["A1", "A2", "A3"],
        "reference_only": ["A0"],
        "exploratory_only": ["A4", "A5"],
        "postfreeze_output_roots_present_at_audit": matching_roots,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
