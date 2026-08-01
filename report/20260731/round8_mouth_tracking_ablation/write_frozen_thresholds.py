"""Write the Round8 ablation configuration freeze as one atomic JSON artifact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUTPUT = REPORT / "frozen_thresholds.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("/", "\\")


def authenticated_artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": rel(path), "sha256": sha256(path)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {OUTPUT}")

    provenance_path = REPORT / "input_provenance.json"
    cohort_path = REPORT / "cohort_protocol.json"
    deviations_path = REPORT / "protocol_deviations.json"
    scales_path = REPORT / "frozen_body_length_scales.json"
    a2_selection_path = REPORT / "quality_review" / "training_motion_sweep_selection.json"
    a2_audit_path = REPORT / "quality_review" / "training_motion_sweep_audit.json"
    a3_audit_path = REPORT / "quality_review" / "training_a3_sweep_audit.json"
    a4_audit_path = REPORT / "quality_review" / "training_a4_method_sweep_audit.json"
    a5_audit_path = REPORT / "quality_review" / "training_a5_hint_audit.json"
    female_annotation_ledger_path = (
        REPO
        / "annotations"
        / "fish_sex_gt"
        / "20251225_K_1_2_FCMC"
        / "female_sex_gt.csv"
    )
    a4_selected_summary_path = (
        REPORT
        / "training_a4_cutoff_sweep"
        / "reviewed_event_1"
        / "s035_m005_g0"
        / "a4_summary.json"
    )

    provenance = load_json(provenance_path)
    cohort = load_json(cohort_path)
    deviations = load_json(deviations_path)
    scales = load_json(scales_path)
    a2_selection = load_json(a2_selection_path)
    a3_audit = load_json(a3_audit_path)
    a4_audit = load_json(a4_audit_path)
    a5_audit = load_json(a5_audit_path)
    a4_selected_summary = load_json(a4_selected_summary_path)

    assert provenance["experiment"] == "20251225_K_1_2_FCMC"
    assert provenance["round_tag"] == "round8_260729_snn031_m6"
    assert provenance["calibration"]["resolved_backend"] == "refractive"
    assert provenance["calibration"]["score_unit"] == "mm^2"
    assert a2_selection["selected_a2_config"] == "A2_v025_h5"
    assert a3_audit["selected"] == "A3_a005"
    assert a4_audit["selected"] == "s035_m005_g0"
    assert a5_audit["production_decision"]["retain_as_production_finalist"] is False
    assert a5_audit["production_decision"]["retain_for_full_length_finalist_run"] is False
    assert all(item["status"] == "recorded_and_contained" for item in deviations["deviations"])
    expected_female_ledger_sha256 = (
        "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
    )
    assert sha256(female_annotation_ledger_path) == expected_female_ledger_sha256
    assert (
        load_json(a2_audit_path)["annotations_path"]
        == rel(female_annotation_ledger_path)
    )
    assert (
        load_json(a2_audit_path)["annotations_sha256"]
        == expected_female_ledger_sha256
    )
    assert a4_audit["training_annotations"] == {
        "path": rel(female_annotation_ledger_path),
        "sha256": expected_female_ledger_sha256,
    }

    implementation_paths = [
        REPO / "scripts" / "reconstruction_3d" / "candidate_cache.py",
        REPO / "scripts" / "reconstruction_3d" / "calibration.py",
        REPO / "scripts" / "reconstruction_3d" / "cross_frame_tracking.py",
        REPO / "scripts" / "reconstruction_3d" / "fill_missing_3d.py",
        REPO / "scripts" / "reconstruction_3d" / "windowed_tracklet_core.py",
        REPO / "scripts" / "reconstruction_3d" / "mouth_ablation.py",
        REPO / "scripts" / "reconstruction_3d" / "deferred_identity_tracking.py",
        REPO / "scripts" / "reconstruction_3d" / "conservative_mouth_tracklets.py",
        REPO / "scripts" / "reconstruction_3d" / "conservative_cross_view_paths.py",
        REPO / "scripts" / "reconstruction_3d" / "mouth_tracklet_ablation.py",
    ]
    implementation = {rel(path): sha256(path) for path in implementation_paths}
    expected_implementation = {
        "scripts\\reconstruction_3d\\candidate_cache.py": "deb4e6b03abb8af1218eb36fe0c8aebe4d1c3f96002f6595e5e3be51bdc0322c",
        "scripts\\reconstruction_3d\\calibration.py": "39e642104691f4dc337135ccf8af80b21f7e64e6b6283f910ca705a661459d19",
        "scripts\\reconstruction_3d\\cross_frame_tracking.py": "db29b8251717ea32b13b54a2c2b1c077111a276a247b88ed73ba17aa41f53496",
        "scripts\\reconstruction_3d\\fill_missing_3d.py": "7a2f7fe1e0e9ab94c9541fd62e2737bc033eb902658bbac59c91f7ea5d174818",
        "scripts\\reconstruction_3d\\windowed_tracklet_core.py": "4e762136bc2131a57024c96e7558ad6361f5585a32bc5feaaf128c51aefdcd26",
        "scripts\\reconstruction_3d\\mouth_ablation.py": "d856585da1515a06067909f0b7aeead1eab2af81973a2cf097e17de33a7192e0",
        "scripts\\reconstruction_3d\\deferred_identity_tracking.py": "c08ac9071b7ab218d338a2e1a2859811a2c322bf1d93f5740e1202c88a663add",
        "scripts\\reconstruction_3d\\conservative_mouth_tracklets.py": "a454ae0b6e59cfd51bdea4b60de29438fc8c407c861e05befd80d3e1d9da2631",
        "scripts\\reconstruction_3d\\conservative_cross_view_paths.py": "4bed90ac17656c7792baf62105ea2acb8253b034846123c215e378d8d3757c8a",
        "scripts\\reconstruction_3d\\mouth_tracklet_ablation.py": "1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0",
    }
    if implementation != expected_implementation:
        raise RuntimeError("Implementation hashes changed before freeze")

    quality_paths = [
        REPORT / "quality_review" / "candidate_cache_independent_audit.json",
        REPORT / "quality_review" / "training_geometry_independent_audit.json",
        a2_selection_path,
        a2_audit_path,
        REPORT / "quality_review" / "score_training_motion_sweep.py",
        a3_audit_path,
        REPORT / "quality_review" / "score_training_a3_sweep.py",
        a4_audit_path,
        REPORT / "quality_review" / "training_a4_method_sweep_summary.csv",
        REPORT / "quality_review" / "audit_training_a4_method_sweep.py",
        a5_audit_path,
        REPORT / "quality_review" / "audit_training_a5_hints.py",
        REPORT / "training_a4_cutoff_sweep" / "SWEEP_COMPLETED.json",
        REPORT / "training_a5" / "TRAINING_A5_COMPLETED.json",
        REPORT / "finalize_training_a4_sweep.ps1",
        REPORT / "finalize_training_a5.py",
        REPORT / "run_training_a5.ps1",
    ]
    quality_artifacts = {
        rel(path): sha256(path)
        for path in quality_paths
    }

    geometry_artifacts = {}
    for name in (
        "reviewed_event_1",
        "reviewed_event_3",
        "control_for_event_1",
        "control_for_event_3",
    ):
        base = REPORT / "training_geometry" / name
        geometry_artifacts[name] = {
            "summary": authenticated_artifact(base / "geometry_summary.json"),
            "selected_geometry": authenticated_artifact(
                base / "strict_mouth_selected_geometry.parquet"
            ),
        }

    a4_selected_record = a4_audit["records"][a4_audit["selected"]]
    a4_replay = a4_selected_summary["configuration"]
    expected_a4_group_config = {
        "minimum_support_frames": 3,
        "minimum_median_mouth_confidence": 0.3,
        "maximum_median_ray_residual_mm2": 5.0,
        "support_tau_frames": 20.0,
        "support_bonus_weight": 2.0,
        "residual_weight": 0.5,
        "residual_mad_weight": 0.25,
        "confidence_weight": 0.5,
        "missing_view_penalty": 0.75,
        "reject_retained_count_mismatch": True,
    }
    expected_a4_path_config = {
        "number_of_paths": 3,
        "beam_width": 256,
        "maximum_link_gap_frames": 100,
        "max_jump_mm": 8.0,
        "max_jump_hard_cap_mm": 80.0,
        "velocity_link_weight": 1.0,
        "maximum_velocity_angle_degrees": 120.0,
        "all_views_break_penalty": 0.5,
        "segment_restart_penalty": 1.0,
        "eligible_group_drop_cost": 0.5,
    }
    assert a4_replay["group_config"] == expected_a4_group_config
    assert a4_replay["path_config"] == expected_a4_path_config
    assert a4_replay["hypothesis_block_frames"] == 250
    assert a4_replay["evaluation_batch_frames"] == 128
    for window_name in (
        "reviewed_event_1",
        "reviewed_event_3",
        "control_for_event_1",
        "control_for_event_3",
    ):
        summary_path = (
            REPORT
            / "training_a4_cutoff_sweep"
            / window_name
            / a4_audit["selected"]
            / "a4_summary.json"
        )
        window_config = load_json(summary_path)["configuration"]
        for replay_key in (
            "hypothesis_block_frames",
            "evaluation_batch_frames",
            "xyz_bounds",
            "num_candidates",
            "num_individuals",
            "anchor",
            "tail_identity_features",
            "frozen_body_scales_sha256",
            "frozen_body_length_px",
            "tracklet_config",
            "group_config",
            "path_config",
            "cache_sha256",
            "calibration_file_sha256",
            "calibration",
            "code_sha256",
        ):
            assert window_config[replay_key] == a4_replay[replay_key], (
                window_name,
                replay_key,
            )
    a5_thresholds = a5_audit["hint_thresholds"]
    common_motion = {
        "uncertainty_growth_per_frame": 0.25,
        "max_motion_gap_frames": 20,
        "motion_history_frames": 5,
        "max_jump_mm": 8,
        "max_jump_hard_cap_mm": 80,
        "max_gap_frames": 100,
    }
    deferred_settings = {
        "beam_width": 16,
        "entry_margin": 0.5,
        "retention_margin": 2,
        "resolution_margin": 1,
        "resolution_stability_frames": 3,
        "minimum_lookahead_frames": 3,
        "maximum_defer_horizon": 100,
    }

    frozen_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    forbidden_postfreeze_prefixes = (
        "validation",
        "heldout",
        "held_out",
        "postfreeze",
        "critical_postfreeze",
        "full_length",
    )
    matching_postfreeze_roots = sorted(
        path.name
        for path in REPORT.iterdir()
        if path.is_dir()
        and path.name.lower().startswith(forbidden_postfreeze_prefixes)
    )
    assert not matching_postfreeze_roots, matching_postfreeze_roots

    freeze = {
        "schema_version": "round8_mouth_tracking_ablation_freeze_v3",
        "status": "frozen_before_validation_heldout_and_postfreeze_critical_reporting",
        "frozen_at": frozen_at,
        "supersedes_invalid_freezes": [
            {
                "reason": (
                    "The v1 freeze omitted three implementation hashes and the "
                    "complete selected A4 group/path/block replay configuration."
                ),
                "sha256": "412e365b8c8fc0dc09061dc6c3ac235897d89f213a003f940c615c398ccb206e",
                "quarantined_path": (
                    "report\\20260731\\round8_mouth_tracking_ablation\\"
                    "quarantine_invalid_freeze_20260731_0414\\"
                    "frozen_thresholds_invalid_412e365b.json"
                ),
            },
            {
                "reason": (
                    "The v2 freeze omitted the direct confirmed-female "
                    "annotation-ledger path and SHA-256 binding."
                ),
                "sha256": "4d74438fad0d244c383ac63acd7f23575ab3869f752b2090cb2eafe9ec9b92ca",
                "quarantined_path": (
                    "report\\20260731\\round8_mouth_tracking_ablation\\"
                    "quarantine_invalid_freeze_20260731_0416\\"
                    "frozen_thresholds_invalid_4d74438f.json"
                ),
            },
        ],
        "experiment": provenance["experiment"],
        "round_tag": provenance["round_tag"],
        "random_seed": cohort["random_seed"],
        "inclusive_frame_intervals": True,
        "generator": authenticated_artifact(Path(__file__).resolve()),
        "source_contract": {
            "cohort_protocol": authenticated_artifact(cohort_path),
            "input_provenance": authenticated_artifact(provenance_path),
            "protocol_deviations": authenticated_artifact(deviations_path),
            "confirmed_female_annotation_ledger": authenticated_artifact(
                female_annotation_ledger_path
            ),
            "source_slps": provenance["source_slps"],
            "canonical_candidate_cache": {
                "directory": rel(
                    REPORT / "candidate_cache_strict_mouth_unfiltered_nodescores"
                ),
                "policy": provenance["canonical_strict_mouth_candidate_caches"][
                    "policy"
                ],
                "views": provenance["canonical_strict_mouth_candidate_caches"]["views"],
            },
            "calibration": provenance["calibration"],
        },
        "controlled_geometry": {
            **provenance["controlled_reconstruction"],
            **provenance["strict_mouth_policy"],
            "calibration_backend": "refractive",
            "residual_score_unit": "mm^2",
        },
        "implementation_sha256": implementation,
        "training_artifacts": {
            "strict_mouth_geometry": geometry_artifacts,
            "confirmed_female_annotation_ledger": authenticated_artifact(
                female_annotation_ledger_path
            ),
            "quality_and_completion_sha256": quality_artifacts,
            "frozen_body_length_scales": authenticated_artifact(scales_path),
        },
        "frozen_methods": {
            "A0": {
                "role": "unchanged_centroid_reference",
                "new_full_length_run_required": False,
                "production_finalist": False,
                "reference_artifact_hashes": provenance["a0_unchanged_reference"],
            },
            "A1": {
                "role": "strict_mouth_position_only_baseline",
                "production_finalist": True,
                "full_length_finalist_run_required": True,
                "settings": {
                    "velocity_weight": 0.0,
                    "tangent_weight": 0.0,
                    "acceleration_weight": 0.0,
                    "uncertainty_growth_per_frame": 0.25,
                    "max_motion_gap_frames": None,
                    "motion_history_frames": 2,
                    "max_jump_mm": 8,
                    "max_jump_hard_cap_mm": 80,
                    "max_gap_frames": 100,
                },
            },
            "A2": {
                "role": "selected_motion_aware_strict_mouth_linker",
                "selected_configuration": a2_selection["selected_a2_config"],
                "production_finalist": True,
                "full_length_finalist_run_required": True,
                "settings": {
                    "velocity_weight": 0.25,
                    "tangent_weight": 0.0,
                    "acceleration_weight": 0.0,
                    **common_motion,
                },
                "selection_limitation": a2_selection["female_denominator_assessment"],
            },
            "A3": {
                "role": "selected_deferred_ambiguity_linker",
                "selected_configuration": a3_audit["selected"],
                "production_finalist": True,
                "full_length_finalist_run_required": True,
                "settings": {
                    "velocity_weight": 0.25,
                    "tangent_weight": 0.0,
                    "acceleration_weight": 0.05,
                    **common_motion,
                    **deferred_settings,
                },
            },
            "A4": {
                "role": "bounded_postfreeze_exploratory_tracklet_method",
                "selected_configuration": a4_audit["selected"],
                "production_finalist": False,
                "full_length_finalist_run_required": False,
                "settings": {
                    "hypothesis_block_frames": a4_replay["hypothesis_block_frames"],
                    "evaluation_batch_frames": a4_replay[
                        "evaluation_batch_frames"
                    ],
                    "xyz_bounds": a4_replay["xyz_bounds"],
                    "num_candidates": a4_replay["num_candidates"],
                    "num_individuals": a4_replay["num_individuals"],
                    "anchor": a4_replay["anchor"],
                    "tail_identity_features": a4_replay[
                        "tail_identity_features"
                    ],
                    "frozen_body_scales_sha256": a4_replay[
                        "frozen_body_scales_sha256"
                    ],
                    "body_length_px": a4_replay["frozen_body_length_px"],
                    "tracklet_config": a4_replay["tracklet_config"],
                    "group_config": a4_replay["group_config"],
                    "path_config": a4_replay["path_config"],
                    "cache_sha256": a4_replay["cache_sha256"],
                    "calibration_file_sha256": a4_replay[
                        "calibration_file_sha256"
                    ],
                    "calibration": a4_replay["calibration"],
                    "code_sha256": a4_replay["code_sha256"],
                    "bridge_consistency": "forward_backward_reciprocal",
                    "retained_tracklet_count_mismatch": "reject_group",
                },
                "training_performance": a4_selected_record["pooled"],
                "nonfinalist_reason": (
                    "Purity-first exploratory method was highly selective on training "
                    "(3/15 female annotations mapped; zero all-three-direct frames)."
                ),
            },
            "A5": {
                "role": a5_audit["production_decision"]["post_freeze_role"],
                "production_finalist": False,
                "full_length_finalist_run_required": False,
                "dependencies": {
                    "A2": a2_selection["selected_a2_config"],
                    "A3": a3_audit["selected"],
                    "A4": a4_audit["selected"],
                },
                "hint_thresholds": a5_thresholds,
                "linker_settings": {
                    "velocity_weight": 0.25,
                    "tangent_weight": 0.0,
                    "acceleration_weight": 0.05,
                    **common_motion,
                    **deferred_settings,
                    "xyz_tolerance_mm": 1e-9,
                },
                "comparison_to_selected_A3": a5_audit[
                    "comparison_to_selected_a3"
                ],
                "nonfinalist_reason": a5_audit["production_decision"]["reason"],
            },
        },
        "postfreeze_execution_policy": {
            "validation": {
                "frames": cohort["validation"],
                "role": "report_only",
                "may_change_frozen_settings_or_finalists": False,
            },
            "held_out": {
                "frames": cohort["held_out"],
                "role": "sole_postfreeze_generalization_evidence",
                "may_change_frozen_settings_or_finalists": False,
            },
            "critical_event": {
                **cohort["critical_event_exclusion"],
                "role": "permanently_unblinded_descriptive_stress_evidence_only",
                "confirmatory": False,
                "may_change_frozen_settings_or_finalists": False,
            },
            "full_length_runs": {
                "required_new_methods": ["A1", "A2", "A3"],
                "reference_only": ["A0"],
                "bounded_exploratory_only": ["A4", "A5"],
                "determinism_requirement": (
                    "authenticate exact code/configuration/input hashes and verify "
                    "rerun or block-resume equivalence"
                ),
            },
        },
        "postfreeze_root_absence_check": {
            "checked_at": frozen_at,
            "report_root": rel(REPORT),
            "forbidden_name_prefixes": list(forbidden_postfreeze_prefixes),
            "matching_roots": matching_postfreeze_roots,
            "result": "PASS",
        },
        "protocol_deviations": deviations["deviations"],
        "limitations": [
            "Only 15 confirmed female annotations were available in the training windows.",
            "Female-only identity labels cannot detect male-to-male identity swaps.",
            "The critical interval was opened before freeze and is permanently descriptive.",
            "A4 and A5 were not retained as production/full-length finalists.",
        ],
    }

    assert (
        freeze["postfreeze_execution_policy"]["validation"]["frames"]
        == cohort["validation"]
    )
    assert (
        freeze["postfreeze_execution_policy"]["held_out"]["frames"]
        == cohort["held_out"]
    )
    assert (
        freeze["frozen_methods"]["A5"]["comparison_to_selected_A3"]
        == a5_audit["comparison_to_selected_a3"]
    )
    assert (
        freeze["source_contract"]["confirmed_female_annotation_ledger"]
        == {
            "path": rel(female_annotation_ledger_path),
            "sha256": expected_female_ledger_sha256,
        }
    )
    assert (
        freeze["training_artifacts"]["confirmed_female_annotation_ledger"]
        == freeze["source_contract"]["confirmed_female_annotation_ledger"]
    )
    assert (
        freeze["frozen_methods"]["A4"]["settings"]["group_config"]
        == expected_a4_group_config
    )
    assert (
        freeze["frozen_methods"]["A4"]["settings"]["path_config"]
        == expected_a4_path_config
    )

    def reject_object_display_strings(value: object) -> None:
        if isinstance(value, str):
            assert "@{" not in value
            assert "System.Object[]" not in value
        elif isinstance(value, dict):
            for nested in value.values():
                reject_object_display_strings(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_object_display_strings(nested)

    reject_object_display_strings(freeze)

    payload = (json.dumps(freeze, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=".frozen_thresholds.", suffix=".tmp", dir=REPORT
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(json.dumps({"path": rel(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
