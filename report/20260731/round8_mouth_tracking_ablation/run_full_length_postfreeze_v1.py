"""Prepare and execute the frozen Round8 full-length mouth-only finalists.

Scope is intentionally narrow:

* A0 is an authenticated unchanged reference; it is not rerun.
* One strict-mouth framewise geometry is shared by A1, A2, and A3.
* Only A1, A2, and A3 are executed.
* A4 and A5 remain bounded exploratory methods and are forbidden here.
* Geometry is rerun in resume mode and every finalist relink is repeated so
  deterministic scientific-output equivalence is checked before completion.

Preparation is allowed after critical finalization. Execution additionally
requires an independently sealed critical-quality PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.common.local_paths import load_local_paths


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "run_bounded_postfreeze_stage_v4.py"
STAGE_ROOT = HERE / "full_length_postfreeze"
QUALITY_SEAL = HERE / "quality_review" / "CRITICAL_QUALITY_REVIEW_COMPLETED.json"

CORE_SHA256 = "0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818"
FREEZE_SHA256 = "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
VALIDATION_FINALIZED_SHA256 = (
    "297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706"
)
HELDOUT_FINALIZED_SHA256 = (
    "3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152"
)
CRITICAL_FINALIZED_SHA256 = (
    "fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c"
)
CRITICAL_COMPLETED_SHA256 = (
    "e493fdcc449034c4f9dd22e3f124283fe8d3cd06d9bd50b0c46002e440367706"
)

EXPERIMENT = "20251225_K_1_2_FCMC"
ROUND_TAG = "round8_260729_snn031_m6"
FRAME_START = 0
FRAME_END = 348599
FRAME_COUNT = 348600

FULL_SCOPE = {
    "experiment": EXPERIMENT,
    "round_tag": ROUND_TAG,
    "inclusive_frame_interval": [FRAME_START, FRAME_END],
    "expected_source_frames": FRAME_COUNT,
    "shared_geometry_methods": ["A1", "A2", "A3"],
    "new_full_length_methods": ["A1", "A2", "A3"],
    "unchanged_reference_only": ["A0"],
    "forbidden_full_length_methods": ["A4", "A5"],
    "selection_or_tuning_permitted": False,
    "settings_change_permitted": False,
    "fresh_resume_design": {
        "geometry": (
            "fresh block run followed by exact resume; all block scientific "
            "fields and selected-geometry SHA-256 must match"
        ),
        "A1": "fresh relink followed by exact scientific-output hash repeat",
        "A2": "fresh relink followed by exact scientific-output hash repeat",
        "A3": "fresh relink followed by exact scientific-output hash repeat",
    },
    "resource_capture": (
        "0.1 s process-tree RSS sampling, wall time, exit code, stdout/stderr "
        "byte count and SHA-256 for every child"
    ),
}

A0_HASHES = {
    "geometry_csv": {
        "filename": f"{ROUND_TAG}_3d.csv",
        "bytes": 77243436,
        "sha256": "6f42483363dd3be4c1aa7ec758521a4f1fde72877be18b5e4dafc8eee14ffb03",
    },
    "identified_csv": {
        "filename": f"{ROUND_TAG}_3d_identified.csv",
        "bytes": 117902758,
        "sha256": "b2100d24cec8a660cff5888ceee1ade26c9990d900f1e67508e566aaab2870b6",
    },
    "best_json": {
        "filename": f"{ROUND_TAG}_best.json",
        "bytes": 338770313,
        "sha256": "ae4b727bb6679a825e3a9dac143a85fa04c41cfe620c4be875e3b0187a28f2b2",
    },
    "best_pre_link_json": {
        "filename": f"{ROUND_TAG}_best_pre_link.json",
        "bytes": 348811915,
        "sha256": "93f322114e10b2fae897f28375237c719a86964b232e3fd686ccd2561fba9068",
    },
    "reconstruction_provenance": {
        "filename": f"{ROUND_TAG}_reconstruction_provenance.json",
        "bytes": 7038,
        "sha256": "e9a895b034beb7d422bd0d9202935f7a17d3637f8e2b17cca1643e86f4d3c1ff",
    },
    "identified_projection_provenance": {
        "filename": f"{ROUND_TAG}_3d_identified_projection_provenance.json",
        "bytes": 1940,
        "sha256": "757f9808237867028b3c4f88a75687bfcdb0ce01de15019fe23279820136ee01",
    },
    "stage3_status": {
        "filename": "stage3_reconstruct_from_slp.status.json",
        "bytes": 7122,
        "sha256": "36aa6fe085bca7ec6bce0b3e25ca2f6dd531818f3bbac5d5883523d830b8b56d",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from all_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from all_strings(nested)


def load_core() -> Any:
    assert sha256(CORE_PATH) == CORE_SHA256
    spec = importlib.util.spec_from_file_location(
        "round8_postfreeze_full_core", CORE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse v4 helpers while binding authentication to this full-only runner.
    module.__file__ = str(Path(__file__).resolve())
    return module


def predecessor_bindings(core: Any) -> dict[str, Any]:
    records = {
        "validation": (
            core.REPORT / "validation_postfreeze" / "STAGE_FINALIZED.json",
            VALIDATION_FINALIZED_SHA256,
        ),
        "heldout": (
            core.REPORT / "heldout_postfreeze" / "STAGE_FINALIZED.json",
            HELDOUT_FINALIZED_SHA256,
        ),
        "critical": (
            core.REPORT
            / "critical_descriptive_postfreeze"
            / "STAGE_FINALIZED.json",
            CRITICAL_FINALIZED_SHA256,
        ),
    }
    result = {}
    for name, (path, expected) in records.items():
        assert path.is_file(), path
        assert sha256(path) == expected
        marker = load_json(path)
        assert marker["status"] == "finalized_and_fully_authenticated"
        result[name] = {"path": core.relative(path), "sha256": expected}
    return result


def a0_reference_contract() -> dict[str, Any]:
    local = load_local_paths(HERE.parents[2] / "configs" / "paths.local.toml")
    assert local is not None and local.reconstruction_output_dir is not None
    root = local.reconstruction_output_dir / EXPERIMENT / ROUND_TAG
    artifacts = {}
    for name, record in A0_HASHES.items():
        artifacts[name] = {
            "path": str(root / record["filename"]),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
    return {
        "role": "unchanged_visible_node_centroid_reference",
        "new_run": False,
        "anchor_policy": "visible-node centroid",
        "temporal_linker": "position-only cross-frame Hungarian",
        "path_resolution": (
            "reconstruction_output_dir from configs/paths.local.toml, then "
            f"{EXPERIMENT}/{ROUND_TAG}"
        ),
        "artifacts": artifacts,
        "authentication_basis": (
            "exact hashes frozen before validation and already independently "
            "audited; no A0 file is copied, modified, or rerun"
        ),
    }


def quality_gate_requirement(core: Any) -> dict[str, Any]:
    return {
        "required_before_execution": True,
        "path": core.relative(QUALITY_SEAL),
        "required_verdict": "PASS",
        "must_bind": {
            "critical_finalized_sha256": CRITICAL_FINALIZED_SHA256,
            "critical_completed_sha256": CRITICAL_COMPLETED_SHA256,
            "freeze_sha256": FREEZE_SHA256,
        },
        "selection_or_tuning_must_be_false": True,
        "settings_change_must_be_false": True,
    }


def verify_quality_gate(core: Any) -> dict[str, Any]:
    requirement = quality_gate_requirement(core)
    assert QUALITY_SEAL.is_file(), QUALITY_SEAL
    quality = load_json(QUALITY_SEAL)
    strings = set(all_strings(quality))
    assert quality["independent_quality_verdict"] == "PASS"
    assert "pass" in quality["status"].lower()
    for expected in requirement["must_bind"].values():
        assert expected in strings, expected
    assert quality.get("selection_or_tuning_performed", False) is False
    assert quality.get("frozen_settings_changed", False) is False
    assert quality.get("full_length_launched_during_review", False) is False
    return {
        "path": core.relative(QUALITY_SEAL),
        "sha256": sha256(QUALITY_SEAL),
        "status": quality["status"],
        "independent_quality_verdict": "PASS",
        "verified_required_bindings": requirement["must_bind"],
    }


def command(core: Any, *arguments: str) -> list[str]:
    return [core.PYTHON, *map(str, arguments)]


def planned_commands(core: Any) -> dict[str, list[str]]:
    geometry_root = STAGE_ROOT / "shared_geometry"
    geometry = geometry_root / "strict_mouth_selected_geometry.parquet"
    common = [
        "-m",
        "scripts.reconstruction_3d.mouth_ablation",
        "relink",
        "--geometry",
        core.relative(geometry),
        "--frame-start",
        str(FRAME_START),
        "--frame-end",
        str(FRAME_END),
        "--uncertainty-growth",
        "0.25",
        "--max-jump-mm",
        "8",
        "--max-jump-hard-cap-mm",
        "80",
        "--max-gap-frames",
        "100",
    ]
    geometry_command = command(
        core,
        "-m",
        "scripts.reconstruction_3d.mouth_ablation",
        "geometry",
        "--cache-dir",
        core.relative(core.CACHE),
        "--paths-config",
        core.relative(core.PATHS_CONFIG),
        "--output-dir",
        core.relative(geometry_root),
        "--frame-start",
        str(FRAME_START),
        "--frame-end",
        str(FRAME_END),
        "--block-frames",
        "500",
        "--evaluation-batch-frames",
        "128",
        "--xyz-bounds",
        "-20",
        "220",
    )
    method_commands = {
        "A1": command(
            core,
            *common,
            "--output-dir",
            core.relative(STAGE_ROOT / "A1"),
            "--method",
            "A1",
            "--velocity-weight",
            "0",
            "--tangent-weight",
            "0",
            "--acceleration-weight",
            "0",
            "--motion-history-frames",
            "2",
        ),
        "A2": command(
            core,
            *common,
            "--output-dir",
            core.relative(STAGE_ROOT / "A2"),
            "--method",
            "A2",
            "--velocity-weight",
            "0.25",
            "--tangent-weight",
            "0",
            "--acceleration-weight",
            "0",
            "--max-motion-gap-frames",
            "20",
            "--motion-history-frames",
            "5",
        ),
        "A3": command(
            core,
            *common,
            "--output-dir",
            core.relative(STAGE_ROOT / "A3"),
            "--method",
            "A3",
            "--velocity-weight",
            "0.25",
            "--tangent-weight",
            "0",
            "--acceleration-weight",
            "0.05",
            "--max-motion-gap-frames",
            "20",
            "--motion-history-frames",
            "5",
            "--beam-width",
            "16",
            "--entry-margin",
            "0.5",
            "--retention-margin",
            "2",
            "--resolution-margin",
            "1",
            "--resolution-stability-frames",
            "3",
            "--minimum-lookahead-frames",
            "3",
            "--maximum-defer-horizon",
            "100",
        ),
    }
    result = {
        "shared_geometry.fresh": geometry_command,
        "shared_geometry.resume": geometry_command,
    }
    for method, method_command in method_commands.items():
        result[f"{method}.fresh"] = method_command
        result[f"{method}.repeat"] = method_command
    assert len(result) == 8
    assert not any(".A4" in name or ".A5" in name for name in result)
    return result


def prepare(core: Any) -> Path:
    assert not STAGE_ROOT.exists(), STAGE_ROOT
    predecessors = predecessor_bindings(core)
    auth = core.authentication_snapshot()
    STAGE_ROOT.mkdir(parents=False)
    preauth_path = STAGE_ROOT / "prelaunch_auth.json"
    preauth = {
        **auth,
        "predecessor_finalized": predecessors,
        "critical_quality_gate_requirement": quality_gate_requirement(core),
        "a0_reference": a0_reference_contract(),
        "full_length_scope": FULL_SCOPE,
    }
    core.write_json_atomic(preauth_path, preauth)
    prepared = {
        "schema_version": "round8_full_length_postfreeze_prepared_v1",
        "status": "prepared_not_launched_quality_gate_pending",
        "prepared_at": core.now(),
        "stage": "full_length",
        "role": "production_finalist_full_length_execution",
        "full_length_scope": FULL_SCOPE,
        "freeze_sha256": FREEZE_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "wrapped_audited_core": {
            "path": core.relative(CORE_PATH),
            "sha256": CORE_SHA256,
        },
        "prelaunch_auth": {
            "path": core.relative(preauth_path),
            "sha256": sha256(preauth_path),
        },
        "predecessor_finalized": predecessors,
        "critical_quality_gate_requirement": quality_gate_requirement(core),
        "critical_quality_gate_satisfied_at_preparation": False,
        "a0_reference": a0_reference_contract(),
        "planned_commands": planned_commands(core),
        "planned_command_count": 8,
        "selection_or_tuning_permitted": False,
        "settings_change_permitted": False,
        "full_length_methods": ["A1", "A2", "A3"],
        "forbidden_methods": ["A4", "A5"],
        "execution_authorization": (
            "BLOCKED until CRITICAL_QUALITY_REVIEW_COMPLETED.json is a bound "
            "PASS and an independent auditor rechecks this exact manifest"
        ),
    }
    prepared_path = STAGE_ROOT / "STAGE_PREPARED.json"
    core.write_json_atomic(prepared_path, prepared)
    return prepared_path


def validate_geometry(
    core: Any, summary: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, Any]:
    assert summary["status"] == "completed"
    config = summary["configuration"]
    assert config["frame_start"] == FRAME_START
    assert config["frame_end"] == FRAME_END
    assert config["block_frames"] == 500
    assert config["evaluation_batch_frames"] == 128
    assert config["xyz_bounds"] == [-20.0, 220.0]
    assert config["num_candidates"] == 6
    assert config["num_individuals"] == 3
    assert config["anchor_node"] == "mouth"
    assert config["require_anchor_node"] is True
    assert config["filter_fp"] is False
    assert config["tail_coordinates_consumed"] is False
    assert config["calibration_backend"] == "refractive"
    assert config["score_unit"] == "mm^2"
    core.assert_summary_code(summary, freeze)
    core.assert_calibration(summary)
    return {
        "advertised_outputs": core.named_advertised_outputs(summary),
        "geometry_blocks": core.geometry_block_scientific_contract(
            STAGE_ROOT / "shared_geometry", summary
        ),
    }


def execute(core: Any) -> Path:
    predecessors = predecessor_bindings(core)
    quality = verify_quality_gate(core)
    prepared_path = STAGE_ROOT / "STAGE_PREPARED.json"
    prepared = load_json(prepared_path)
    assert prepared["status"] == "prepared_not_launched_quality_gate_pending"
    assert prepared["runner_sha256"] == sha256(Path(__file__))
    assert prepared["freeze_sha256"] == FREEZE_SHA256
    assert prepared["planned_commands"] == planned_commands(core)
    assert prepared["predecessor_finalized"] == predecessors
    assert prepared["full_length_methods"] == ["A1", "A2", "A3"]
    assert prepared["forbidden_methods"] == ["A4", "A5"]
    assert prepared["critical_quality_gate_satisfied_at_preparation"] is False

    preauth_path = STAGE_ROOT / "prelaunch_auth.json"
    assert prepared["prelaunch_auth"]["sha256"] == sha256(preauth_path)
    preauth = load_json(preauth_path)
    current_auth = core.authentication_snapshot()
    assert current_auth["bindings"] == preauth["bindings"]
    freeze = load_json(core.FREEZE)
    assert freeze["postfreeze_execution_policy"]["full_length_runs"] == {
        "bounded_exploratory_only": ["A4", "A5"],
        "determinism_requirement": (
            "authenticate exact code/configuration/input hashes and verify "
            "rerun or block-resume equivalence"
        ),
        "reference_only": ["A0"],
        "required_new_methods": ["A1", "A2", "A3"],
    }

    gate_path = STAGE_ROOT / "execution_gate_auth.json"
    core.write_json_atomic(
        gate_path,
        {
            "schema_version": "round8_full_length_execution_gate_v1",
            "status": "critical_quality_pass_authenticated",
            "authenticated_at": core.now(),
            "critical_quality": quality,
            "predecessor_finalized": predecessors,
            "freeze_sha256": FREEZE_SHA256,
            "runner_sha256": sha256(Path(__file__)),
            "selection_or_tuning_performed": False,
            "settings_changed": False,
        },
    )

    planned = prepared["planned_commands"]
    records: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    statuses: Counter[str] = Counter()
    started_at = core.now()
    started = time.perf_counter()

    geometry_root = STAGE_ROOT / "shared_geometry"
    geometry_summary_path = geometry_root / "geometry_summary.json"
    records["shared_geometry.fresh"] = core.run_process(
        STAGE_ROOT,
        "shared_geometry.fresh",
        planned["shared_geometry.fresh"],
        geometry_root / "geometry.fresh.stdout.log",
        geometry_root / "geometry.fresh.stderr.log",
    )
    geometry_summary = load_json(geometry_summary_path)
    geometry_before = validate_geometry(core, geometry_summary, freeze)
    assert geometry_summary["reused_blocks"] == 0

    records["shared_geometry.resume"] = core.run_process(
        STAGE_ROOT,
        "shared_geometry.resume",
        planned["shared_geometry.resume"],
        geometry_root / "geometry.resume.stdout.log",
        geometry_root / "geometry.resume.stderr.log",
    )
    geometry_summary = load_json(geometry_summary_path)
    geometry_after = validate_geometry(core, geometry_summary, freeze)
    assert geometry_after == geometry_before
    assert geometry_summary["reused_blocks"] == geometry_summary["total_blocks"]
    determinism["shared_geometry"] = {
        "result": "PASS",
        "fresh_scientific_contract": geometry_before,
        "resume_scientific_contract": geometry_after,
        "reused_blocks": geometry_summary["reused_blocks"],
        "total_blocks": geometry_summary["total_blocks"],
    }
    summaries["shared_geometry"] = {
        "path": core.relative(geometry_summary_path),
        "sha256": sha256(geometry_summary_path),
        "status": "completed",
    }
    statuses["geometry:completed"] += 1

    for method in ("A1", "A2", "A3"):
        method_root = STAGE_ROOT / method
        summary_path = method_root / f"{method.lower()}_summary.json"
        records[f"{method}.fresh"] = core.run_process(
            STAGE_ROOT,
            f"{method}.fresh",
            planned[f"{method}.fresh"],
            method_root / f"{method.lower()}.fresh.stdout.log",
            method_root / f"{method.lower()}.fresh.stderr.log",
        )
        summary = load_json(summary_path)
        assert summary["status"] == "completed"
        core.assert_relink_settings(summary, method, FRAME_START, FRAME_END)
        core.assert_summary_code(summary, freeze)
        core.assert_calibration(summary)
        fresh_outputs = core.authenticate_advertised_outputs(summary, STAGE_ROOT)

        records[f"{method}.repeat"] = core.run_process(
            STAGE_ROOT,
            f"{method}.repeat",
            planned[f"{method}.repeat"],
            method_root / f"{method.lower()}.repeat.stdout.log",
            method_root / f"{method.lower()}.repeat.stderr.log",
        )
        summary = load_json(summary_path)
        assert summary["status"] == "completed"
        core.assert_relink_settings(summary, method, FRAME_START, FRAME_END)
        core.assert_summary_code(summary, freeze)
        core.assert_calibration(summary)
        repeat_outputs = core.authenticate_advertised_outputs(summary, STAGE_ROOT)
        assert repeat_outputs == fresh_outputs
        determinism[method] = {
            "result": "PASS",
            "fresh_scientific_output_sha256": fresh_outputs,
            "repeat_scientific_output_sha256": repeat_outputs,
        }
        summaries[method] = {
            "path": core.relative(summary_path),
            "sha256": sha256(summary_path),
            "status": "completed",
            "metrics": summary["metrics"],
        }
        statuses[f"{method}:completed"] += 1

    postauth = core.authentication_snapshot()
    assert postauth["bindings"] == preauth["bindings"]
    postauth.update(
        {
            "predecessor_finalized": predecessors,
            "critical_quality": quality,
            "a0_reference": prepared["a0_reference"],
            "full_length_scope": FULL_SCOPE,
        }
    )
    postauth_path = STAGE_ROOT / "postcompletion_auth.json"
    core.write_json_atomic(postauth_path, postauth)

    inventory = {}
    for path in sorted(STAGE_ROOT.rglob("*")):
        if (
            path.is_file()
            and path.name != "STAGE_COMPLETED.json"
            and path.name
            not in {
                "detached_launcher.stdout.log",
                "detached_launcher.stderr.log",
            }
        ):
            inventory[core.relative(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    completion = {
        "schema_version": "round8_full_length_postfreeze_completed_v1",
        "status": "completed_and_authenticated",
        "stage": "full_length",
        "role": "production_finalist_full_length_execution",
        "started_at": started_at,
        "completed_at": core.now(),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes_max": max(
            record["peak_rss_bytes"] for record in records.values()
        ),
        "peak_memory_method": "maximum process-tree RSS across 0.1 s samples",
        "full_length_scope": FULL_SCOPE,
        "a0_reference": prepared["a0_reference"],
        "freeze": postauth["bindings"]["freeze"],
        "runner": postauth["bindings"]["runner"],
        "wrapped_audited_core": prepared["wrapped_audited_core"],
        "predecessor_finalized": predecessors,
        "critical_quality": quality,
        "prepared_manifest": {
            "path": core.relative(prepared_path),
            "sha256": sha256(prepared_path),
        },
        "prelaunch_auth": {
            "path": core.relative(preauth_path),
            "sha256": sha256(preauth_path),
        },
        "execution_gate_auth": {
            "path": core.relative(gate_path),
            "sha256": sha256(gate_path),
        },
        "postcompletion_auth": {
            "path": core.relative(postauth_path),
            "sha256": sha256(postauth_path),
        },
        "planned_commands": planned,
        "process_records": {
            name: {
                "path": core.relative(
                    STAGE_ROOT / "process_records" / f"{name}.json"
                ),
                "sha256": sha256(
                    STAGE_ROOT / "process_records" / f"{name}.json"
                ),
                "wall_seconds": record["wall_seconds"],
                "peak_rss_bytes": record["peak_rss_bytes"],
            }
            for name, record in records.items()
        },
        "summaries": summaries,
        "scientific_status_counts": dict(statuses),
        "deterministic_fresh_resume_equivalence": determinism,
        "artifact_inventory": inventory,
        "artifact_inventory_count": len(inventory),
        "artifact_inventory_scope": (
            "all files under full_length_postfreeze before STAGE_COMPLETED.json, "
            "except still-open detached launcher stdout/stderr"
        ),
        "outer_launcher_files_excluded_until_finalization": [
            core.relative(STAGE_ROOT / "detached_launcher.stdout.log"),
            core.relative(STAGE_ROOT / "detached_launcher.stderr.log"),
        ],
        "selection_or_tuning_performed": False,
        "frozen_settings_changed": False,
        "new_full_length_methods": ["A1", "A2", "A3"],
        "reference_only_methods": ["A0"],
        "excluded_methods": ["A4", "A5"],
    }
    completion_path = STAGE_ROOT / "STAGE_COMPLETED.json"
    core.write_json_atomic(completion_path, completion)
    return completion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core = load_core()
    result = prepare(core) if args.prepare_only else execute(core)
    print(
        json.dumps(
            {"path": core.relative(result), "sha256": sha256(result)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
