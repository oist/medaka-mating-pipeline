"""Prepare and execute one immutable bounded Round8 post-freeze stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psutil
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
FREEZE = REPORT / "frozen_thresholds.json"
FREEZE_SHA256 = "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
PYTHON = str(Path(sys.executable).resolve())
CACHE = REPORT / "candidate_cache_strict_mouth_unfiltered_nodescores"
SCALES = REPORT / "frozen_body_length_scales.json"
PATHS_CONFIG = REPO / "configs" / "paths.local.toml"
ADAPTER = REPORT / "adapt_a4_empty_selection_for_a5.py"
ADAPTER_SHA256 = "eea6877f16204f63a9fea4820e92a6696f8b24bdddfcc4fab57e2b25a5ffa95b"
REPAIR_PROTOCOL = REPORT / "HELDOUT_EMPTY_A4_SCHEMA_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA256 = (
    "ec72854059d4484b4ef8bed58b70d0c04284f0efa7771a93aeeae8b69b283070"
)
FAILED_HELDOUT = (
    REPORT / "quarantine_failed_heldout_attempt_runner1f737_20260731_0457"
)
FAILED_HELDOUT_FINALIZED = FAILED_HELDOUT / "FAILED_ATTEMPT_FINALIZED.json"
FAILED_HELDOUT_FINALIZED_SHA256 = (
    "75a506ce112b8df00b49024f039ddb6ef554637620840bf5f9a78c73204a85d5"
)

STAGES = {
    "validation": {
        "root": "validation_postfreeze",
        "role": "report_only_no_selection_or_tie_breaking",
        "windows": [
            {"name": "reviewed_event_2", "frame_start": 96612, "frame_end": 97802},
            {
                "name": "control_for_event_2",
                "frame_start": 151050,
                "frame_end": 152240,
            },
        ],
    },
    "heldout": {
        "root": "heldout_postfreeze",
        "role": "sole_postfreeze_generalization_evidence",
        "windows": [
            {
                "name": "reviewed_event_4",
                "frame_start": 273617,
                "frame_end": 278182,
            },
            {
                "name": "control_for_event_4",
                "frame_start": 57363,
                "frame_end": 61928,
            },
            {
                "name": "low_top_pair_separation_8_9_a",
                "frame_start": 123124,
                "frame_end": 123274,
            },
            {
                "name": "low_top_pair_separation_8_9_b",
                "frame_start": 136451,
                "frame_end": 136601,
            },
            {
                "name": "low_top_pair_separation_8_11",
                "frame_start": 167108,
                "frame_end": 167258,
            },
        ],
    },
    "critical": {
        "root": "critical_descriptive_postfreeze",
        "role": "permanently_unblinded_descriptive_nonconfirmatory",
        "windows": [
            {
                "name": "critical_descriptive_34000_35100",
                "frame_start": 34000,
                "frame_end": 35100,
            }
        ],
    },
}

METHOD_ROLES = {
    "A0": {
        "role": "unchanged_centroid_reference",
        "executed_in_bounded_stage": False,
        "production_finalist": False,
    },
    "A1": {
        "role": "strict_mouth_position_only_production_finalist",
        "executed_in_bounded_stage": True,
        "production_finalist": True,
    },
    "A2": {
        "role": "selected_motion_aware_production_finalist",
        "executed_in_bounded_stage": True,
        "production_finalist": True,
    },
    "A3": {
        "role": "selected_deferred_ambiguity_production_finalist",
        "executed_in_bounded_stage": True,
        "production_finalist": True,
    },
    "A4": {
        "role": "bounded_exploratory_nonfinalist",
        "executed_in_bounded_stage": True,
        "production_finalist": False,
    },
    "A5": {
        "role": "bounded_exploratory_nonfinalist",
        "executed_in_bounded_stage": True,
        "production_finalist": False,
    },
}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO)).replace("/", "\\")


def command(*arguments: str) -> list[str]:
    return [PYTHON, *map(str, arguments)]


def stage_commands(stage: str) -> dict[str, list[str]]:
    stage_spec = STAGES[stage]
    stage_root = REPORT / stage_spec["root"]
    planned: dict[str, list[str]] = {}
    for window in stage_spec["windows"]:
        name = window["name"]
        first = str(window["frame_start"])
        last = str(window["frame_end"])
        window_root = stage_root / name
        geometry_root = window_root / "geometry"
        geometry = geometry_root / "strict_mouth_selected_geometry.parquet"
        geometry_summary = geometry_root / "geometry_summary.json"
        common_relink = [
            "-m",
            "scripts.reconstruction_3d.mouth_ablation",
            "relink",
            "--geometry",
            relative(geometry),
            "--frame-start",
            first,
            "--frame-end",
            last,
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
            "-m",
            "scripts.reconstruction_3d.mouth_ablation",
            "geometry",
            "--cache-dir",
            relative(CACHE),
            "--paths-config",
            relative(PATHS_CONFIG),
            "--output-dir",
            relative(geometry_root),
            "--frame-start",
            first,
            "--frame-end",
            last,
            "--block-frames",
            "500",
            "--evaluation-batch-frames",
            "128",
            "--xyz-bounds",
            "-20",
            "220",
        )
        planned[f"{name}.geometry"] = geometry_command
        planned[f"{name}.geometry_resume"] = geometry_command
        planned[f"{name}.A1"] = command(
            *common_relink,
            "--output-dir",
            relative(window_root / "A1"),
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
        )
        planned[f"{name}.A2"] = command(
            *common_relink,
            "--output-dir",
            relative(window_root / "A2"),
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
        )
        planned[f"{name}.A3"] = command(
            *common_relink,
            "--output-dir",
            relative(window_root / "A3"),
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
        )
        a4_root = window_root / "A4"
        a4_command = command(
            "-m",
            "scripts.reconstruction_3d.mouth_tracklet_ablation",
            "run",
            "--cache-dir",
            relative(CACHE),
            "--paths-config",
            relative(PATHS_CONFIG),
            "--frozen-scales",
            relative(SCALES),
            "--output-dir",
            relative(a4_root),
            "--frame-start",
            first,
            "--frame-end",
            last,
            "--hypothesis-block-frames",
            "250",
            "--evaluation-batch-frames",
            "128",
            "--xyz-bounds",
            "-20",
            "220",
            "--separation-body-lengths",
            "0.35",
            "--ambiguity-margin-body-lengths",
            "0.05",
            "--maximum-missing-gap-frames",
            "0",
            "--minimum-mouth-score",
            "0.3",
            "--minimum-fragment-observations",
            "2",
            "--minimum-group-support",
            "3",
            "--beam-width",
            "256",
        )
        planned[f"{name}.A4"] = a4_command
        planned[f"{name}.A4_resume"] = a4_command
        planned[f"{name}.A4_schema_adapter"] = command(
            relative(ADAPTER),
            "--a4-summary",
            relative(a4_root / "a4_summary.json"),
            "--a4-selected",
            relative(a4_root / "a4_selected_observations.parquet"),
            "--output-dir",
            relative(a4_root),
        )
        a5_root = window_root / "A5"
        planned[f"{name}.A5_hints"] = command(
            "-m",
            "scripts.reconstruction_3d.mouth_tracklet_ablation",
            "hints",
            "--a4-selected",
            relative(a4_root / "a4_selected_for_a5.parquet"),
            "--a2-direct",
            relative(window_root / "A2" / "a2_direct.csv"),
            "--output-dir",
            relative(a5_root),
            "--minimum-mouth-confidence",
            "0.5",
            "--minimum-group-support",
            "3",
            "--maximum-group-median-residual",
            "1.0",
            "--maximum-group-mad-residual",
            "0.25",
            "--minimum-group-median-mouth-confidence",
            "0.5",
            "--maximum-group-assignment-cost",
            "0",
            "--minimum-mapping-votes",
            "3",
            "--minimum-mapping-purity",
            "0.9",
            "--minimum-vote-margin",
            "2",
            "--xyz-tolerance-mm",
            "1e-9",
        )
        planned[f"{name}.A5"] = command(
            "-m",
            "scripts.reconstruction_3d.mouth_tracklet_ablation",
            "a5",
            "--geometry",
            relative(geometry),
            "--geometry-summary",
            relative(geometry_summary),
            "--hints",
            relative(a5_root / "a5_hints.csv"),
            "--forced-reset-frames",
            relative(a5_root / "a5_forced_reset_frames.json"),
            "--hints-summary",
            relative(a5_root / "a5_hints_summary.json"),
            "--output-dir",
            relative(a5_root),
            "--frame-start",
            first,
            "--frame-end",
            last,
            "--velocity-weight",
            "0.25",
            "--acceleration-weight",
            "0.05",
            "--uncertainty-growth",
            "0.25",
            "--max-motion-gap-frames",
            "20",
            "--motion-history-frames",
            "5",
            "--max-jump-mm",
            "8",
            "--max-jump-hard-cap-mm",
            "80",
            "--max-gap-frames",
            "100",
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
            "--xyz-tolerance-mm",
            "1e-9",
        )
    return planned


def assert_sequence(stage: str, preparing: bool) -> dict[str, str]:
    roots = {name: REPORT / spec["root"] for name, spec in STAGES.items()}
    full_length_root = REPORT / "full_length_postfreeze"
    assert not full_length_root.exists(), full_length_root
    predecessors: dict[str, str] = {}
    if stage == "validation":
        if preparing:
            assert not any(path.exists() for path in roots.values())
        else:
            assert roots["validation"].is_dir()
            assert not roots["heldout"].exists()
            assert not roots["critical"].exists()
    elif stage == "heldout":
        validation_marker = roots["validation"] / "STAGE_FINALIZED.json"
        assert validation_marker.is_file(), validation_marker
        validation_record = load_json(validation_marker)
        assert validation_record["status"] == "finalized_and_fully_authenticated"
        predecessors["validation"] = sha256(validation_marker)
        if preparing:
            assert not roots["heldout"].exists()
        else:
            assert roots["heldout"].is_dir()
        assert not roots["critical"].exists()
    elif stage == "critical":
        for prior in ("validation", "heldout"):
            marker = roots[prior] / "STAGE_FINALIZED.json"
            assert marker.is_file(), marker
            record = load_json(marker)
            assert record["status"] == "finalized_and_fully_authenticated"
            predecessors[prior] = sha256(marker)
        if preparing:
            assert not roots["critical"].exists()
        else:
            assert roots["critical"].is_dir()
    return predecessors


def authentication_snapshot() -> dict[str, Any]:
    assert sha256(FREEZE) == FREEZE_SHA256
    assert sha256(ADAPTER) == ADAPTER_SHA256
    assert sha256(REPAIR_PROTOCOL) == REPAIR_PROTOCOL_SHA256
    assert sha256(FAILED_HELDOUT_FINALIZED) == FAILED_HELDOUT_FINALIZED_SHA256
    freeze = load_json(FREEZE)
    assert freeze["status"].startswith("frozen_before_validation")
    implementation = {}
    for relative_path, expected in freeze["implementation_sha256"].items():
        path = REPO / relative_path
        actual = sha256(path)
        assert actual == expected, (relative_path, actual, expected)
        implementation[relative_path] = actual

    cache_manifest_path = CACHE / "candidate_cache_manifest.json"
    cache_manifest = load_json(cache_manifest_path)
    caches = {}
    for view, expected_record in freeze["source_contract"][
        "canonical_candidate_cache"
    ]["views"].items():
        cache_path = Path(cache_manifest["views"][view]["cache_path"])
        actual = sha256(cache_path)
        expected = expected_record["cache_sha256"]
        assert actual == expected, (view, actual, expected)
        caches[view] = {"path": relative(cache_path), "sha256": actual}

    calibration = freeze["source_contract"]["calibration"]
    calibration_files = {
        calibration["manifest_path"]: calibration["calibration_manifest_sha256"],
        calibration["cam0_bundle_path"]: calibration["cam0_bundle_sha256"],
        calibration["cam1_bundle_path"]: calibration["cam1_bundle_sha256"],
        calibration["cam2_bundle_path"]: calibration["cam2_bundle_sha256"],
    }
    authenticated_calibration = {}
    for relative_path, expected in calibration_files.items():
        actual = sha256(REPO / relative_path)
        assert actual == expected, (relative_path, actual, expected)
        authenticated_calibration[relative_path] = actual

    ledger_record = freeze["source_contract"]["confirmed_female_annotation_ledger"]
    ledger_path = REPO / ledger_record["path"]
    assert sha256(ledger_path) == ledger_record["sha256"]
    assert sha256(SCALES) == freeze["frozen_methods"]["A4"]["settings"][
        "frozen_body_scales_sha256"
    ]
    return {
        "checked_at": now(),
        "bindings": {
            "freeze": {
                "path": relative(FREEZE),
                "sha256": FREEZE_SHA256,
                "frozen_at": freeze["frozen_at"],
            },
            "runner": {
                "path": relative(Path(__file__)),
                "sha256": sha256(Path(__file__)),
            },
            "implementation_sha256": implementation,
            "candidate_cache": caches,
            "candidate_cache_manifest": {
                "path": relative(cache_manifest_path),
                "sha256": sha256(cache_manifest_path),
            },
            "calibration_sha256": authenticated_calibration,
            "frozen_scales": {
                "path": relative(SCALES),
                "sha256": sha256(SCALES),
            },
            "confirmed_female_annotation_ledger": ledger_record,
            "postfreeze_empty_schema_adapter": {
                "path": relative(ADAPTER),
                "sha256": ADAPTER_SHA256,
            },
            "postfreeze_repair_protocol": {
                "path": relative(REPAIR_PROTOCOL),
                "sha256": REPAIR_PROTOCOL_SHA256,
            },
            "failed_heldout_attempt_finalized": {
                "path": relative(FAILED_HELDOUT_FINALIZED),
                "sha256": FAILED_HELDOUT_FINALIZED_SHA256,
            },
        },
    }


def named_advertised_outputs(
    summary: dict[str, Any],
    *,
    relocated_stage_root: Path | None = None,
) -> dict[str, str]:
    def resolve_advertised(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_file() or relocated_stage_root is None:
            return path
        parts = list(path.parts)
        index = parts.index("heldout_postfreeze")
        return relocated_stage_root.joinpath(*parts[index + 1 :])

    authenticated: dict[str, str] = {}
    if "output_path" in summary:
        path = resolve_advertised(summary["output_path"])
        actual = sha256(path)
        assert actual == summary["output_sha256"]
        authenticated["selected_geometry"] = actual
    for name, value in summary.get("outputs", {}).items():
        if not isinstance(value, dict) or "path" not in value:
            continue
        path = resolve_advertised(value["path"])
        actual = sha256(path)
        assert actual == value["sha256"]
        authenticated[name] = actual
    return authenticated


def failed_heldout_binding_and_baseline() -> dict[str, Any]:
    finalized = load_json(FAILED_HELDOUT_FINALIZED)
    assert finalized["status"] == "failed_attempt_preserved_and_authenticated"
    binding_paths = {
        "finalized_failure": FAILED_HELDOUT_FINALIZED,
        "prepared_manifest": FAILED_HELDOUT / "STAGE_PREPARED.json",
        "prelaunch_auth": FAILED_HELDOUT / "prelaunch_auth.json",
        "outer_launch": FAILED_HELDOUT / "DETACHED_LAUNCH.json",
        "outer_exit": FAILED_HELDOUT / "DETACHED_LAUNCH_EXIT.json",
        "failed_process_record": (
            FAILED_HELDOUT
            / "process_records"
            / "low_top_pair_separation_8_9_b.A5_hints.json"
        ),
        "failed_stderr": (
            FAILED_HELDOUT
            / "low_top_pair_separation_8_9_b"
            / "A5"
            / "hints.stderr.log"
        ),
        "inventory_manifest": FAILED_HELDOUT / "FAILED_ATTEMPT_MANIFEST.json",
    }
    bindings = {
        name: {"path": relative(path), "sha256": sha256(path)}
        for name, path in binding_paths.items()
    }
    baseline: dict[str, dict[str, str]] = {}
    for window in STAGES["heldout"]["windows"]:
        name = window["name"]
        root = FAILED_HELDOUT / name
        summary_specs = {
            "geometry": root / "geometry" / "geometry_summary.json",
            "A1": root / "A1" / "a1_summary.json",
            "A2": root / "A2" / "a2_summary.json",
            "A3": root / "A3" / "a3_summary.json",
            "A4": root / "A4" / "a4_summary.json",
            "A5": root / "A5" / "a5_summary.json",
        }
        for step, summary_path in summary_specs.items():
            if summary_path.is_file():
                baseline[f"{name}.{step}"] = named_advertised_outputs(
                    load_json(summary_path),
                    relocated_stage_root=FAILED_HELDOUT,
                )
        hints_summary_path = root / "A5" / "a5_hints_summary.json"
        if hints_summary_path.is_file():
            hints = load_json(hints_summary_path)
            hints_csv = root / "A5" / "a5_hints.csv"
            reset_json = root / "A5" / "a5_forced_reset_frames.json"
            assert sha256(hints_csv) == hints["outputs"]["hints_sha256"]
            assert (
                sha256(reset_json)
                == hints["outputs"]["forced_reset_frames_sha256"]
            )
            baseline[f"{name}.A5_hints"] = {
                "hints": sha256(hints_csv),
                "forced_reset_frames": sha256(reset_json),
            }
    return {
        "bindings": bindings,
        "scientific_baseline_sha256": baseline,
        "comparison_policy": (
            "fresh replacement outputs must match every overlapping scientific "
            "artifact hash; runtime summaries, process records, and logs may drift"
        ),
    }


def prepare(stage: str) -> Path:
    stage_spec = STAGES[stage]
    stage_root = REPORT / stage_spec["root"]
    predecessors = assert_sequence(stage, preparing=True)
    auth = authentication_snapshot()
    stage_root.mkdir(parents=False)
    prelaunch_path = stage_root / "prelaunch_auth.json"
    write_json_atomic(prelaunch_path, auth)
    prepared = {
        "schema_version": "round8_bounded_postfreeze_prepared_v1",
        "status": "prepared_not_launched",
        "prepared_at": now(),
        "stage": stage,
        "role": stage_spec["role"],
        "method_roles": METHOD_ROLES,
        "windows": stage_spec["windows"],
        "freeze_sha256": FREEZE_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "prelaunch_auth": {
            "path": relative(prelaunch_path),
            "sha256": sha256(prelaunch_path),
        },
        "predecessor_finalized_sha256": predecessors,
        "supersedes_failed_attempt": (
            failed_heldout_binding_and_baseline() if stage == "heldout" else None
        ),
        "planned_commands": stage_commands(stage),
        "selection_or_tuning_permitted": False,
        "full_length_launch_permitted": False,
    }
    prepared_path = stage_root / "STAGE_PREPARED.json"
    write_json_atomic(prepared_path, prepared)
    return prepared_path


def process_tree_rss(process: psutil.Process) -> int:
    total = 0
    candidates = [process]
    try:
        candidates.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for candidate in candidates:
        try:
            total += candidate.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_process(
    stage_root: Path,
    record_name: str,
    arguments: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = now()
    started = time.perf_counter()
    peak_rss = 0
    samples = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(
            arguments,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ps_process = psutil.Process(child.pid)
        while child.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(ps_process))
            samples += 1
            time.sleep(0.1)
        return_code = child.wait()
    wall_seconds = time.perf_counter() - started
    record = {
        "schema_version": "round8_postfreeze_process_v1",
        "status": "completed" if return_code == 0 else "failed",
        "record_name": record_name,
        "started_at": started_at,
        "finished_at": now(),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss,
        "peak_memory_method": "0.1 s psutil RSS sampling of process tree",
        "memory_samples": samples,
        "pid": child.pid,
        "return_code": return_code,
        "command": arguments,
        "stdout": {
            "path": relative(stdout_path),
            "bytes": stdout_path.stat().st_size,
            "sha256": sha256(stdout_path),
        },
        "stderr": {
            "path": relative(stderr_path),
            "bytes": stderr_path.stat().st_size,
            "sha256": sha256(stderr_path),
        },
    }
    record_path = stage_root / "process_records" / f"{record_name}.json"
    write_json_atomic(record_path, record)
    if return_code != 0:
        raise RuntimeError(f"{record_name} exited {return_code}: {stderr_path}")
    if stderr_path.stat().st_size:
        raise RuntimeError(f"{record_name} wrote stderr: {stderr_path}")
    return record


def assert_inside_stage(path: Path, stage_root: Path) -> None:
    path.resolve().relative_to(stage_root.resolve())


def authenticate_advertised_outputs(
    summary: dict[str, Any], stage_root: Path
) -> dict[str, str]:
    authenticated = {}
    if "output_path" in summary:
        path = Path(summary["output_path"])
        assert_inside_stage(path, stage_root)
        actual = sha256(path)
        assert actual == summary["output_sha256"]
        authenticated[relative(path)] = actual
    outputs = summary.get("outputs", {})
    for value in outputs.values():
        if not isinstance(value, dict) or "path" not in value:
            continue
        path = Path(value["path"])
        assert_inside_stage(path, stage_root)
        actual = sha256(path)
        assert actual == value["sha256"]
        authenticated[relative(path)] = actual
    return authenticated


def assert_summary_code(summary: dict[str, Any], freeze: dict[str, Any]) -> None:
    code = summary.get("code_sha256", {})
    for relative_path, advertised in code.items():
        actual = sha256(REPO / relative_path)
        assert actual == advertised, (relative_path, actual, advertised)
        if relative_path in freeze["implementation_sha256"]:
            assert advertised == freeze["implementation_sha256"][relative_path]


def assert_calibration(summary: dict[str, Any]) -> None:
    calibration = summary.get("calibration")
    if calibration:
        backend = calibration.get("resolved_backend", calibration.get("backend"))
        assert backend == "refractive"
        assert calibration["score_unit"] == "mm^2"


def assert_relink_settings(
    summary: dict[str, Any], method: str, first: int, last: int
) -> None:
    settings = summary["settings"]
    assert settings["method"] == method
    assert settings["frame_start"] == first
    assert settings["frame_end"] == last
    assert settings["max_jump_mm"] == 8.0
    assert settings["max_jump_hard_cap_mm"] == 80.0
    assert settings["max_gap_frames"] == 100
    assert settings["shift_to_positive"] is False
    assert settings["fill_ensure_non_negative"] is False
    if method == "A1":
        assert settings["velocity_weight"] == 0.0
        assert settings["tangent_weight"] == 0.0
        assert settings["acceleration_weight"] == 0.0
        assert settings["max_motion_gap_frames"] is None
        assert settings["motion_history_frames"] == 2
    elif method == "A2":
        assert settings["velocity_weight"] == 0.25
        assert settings["tangent_weight"] == 0.0
        assert settings["acceleration_weight"] == 0.0
        assert settings["max_motion_gap_frames"] == 20
        assert settings["motion_history_frames"] == 5
    elif method == "A3":
        assert settings["velocity_weight"] == 0.25
        assert settings["tangent_weight"] == 0.0
        assert settings["acceleration_weight"] == 0.05
        assert settings["max_motion_gap_frames"] == 20
        assert settings["motion_history_frames"] == 5
        expected = {
            "num_individuals": 3,
            "max_jump_mm": 8.0,
            "max_jump_hard_cap_mm": 80.0,
            "max_gap_frames": 100,
            "velocity_prediction_weight": 0.25,
            "tangent_weight": 0.0,
            "acceleration_weight": 0.05,
            "motion_uncertainty_growth_per_frame": 0.25,
            "max_motion_gap_frames": 20,
            "motion_history_frames": 5,
            "beam_width": 16,
            "entry_margin": 0.5,
            "retention_margin": 2.0,
            "resolution_margin": 1.0,
            "resolution_stability_frames": 3,
            "minimum_lookahead_frames": 3,
            "maximum_defer_horizon": 100,
        }
        assert settings["deferred"] == expected


def execute(stage: str) -> Path:
    stage_spec = STAGES[stage]
    stage_root = REPORT / stage_spec["root"]
    assert_sequence(stage, preparing=False)
    prepared_path = stage_root / "STAGE_PREPARED.json"
    prepared = load_json(prepared_path)
    assert prepared["status"] == "prepared_not_launched"
    assert prepared["freeze_sha256"] == FREEZE_SHA256
    assert prepared["runner_sha256"] == sha256(Path(__file__))
    assert prepared["planned_commands"] == stage_commands(stage)
    prelaunch = load_json(stage_root / "prelaunch_auth.json")
    launch_auth = authentication_snapshot()
    assert launch_auth["bindings"] == prelaunch["bindings"]
    freeze = load_json(FREEZE)
    a4_frozen = freeze["frozen_methods"]["A4"]["settings"]
    planned = prepared["planned_commands"]
    records: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    resume_checks: dict[str, Any] = {}
    scientific_statuses: Counter[str] = Counter()
    replacement_baseline = (
        prepared["supersedes_failed_attempt"]["scientific_baseline_sha256"]
        if stage == "heldout"
        else {}
    )
    replacement_reproducibility: dict[str, Any] = {}
    adapter_checks: dict[str, Any] = {}

    def check_replacement(step: str, actual: dict[str, str]) -> None:
        if step not in replacement_baseline:
            return
        expected = replacement_baseline[step]
        assert actual == expected, (step, actual, expected)
        replacement_reproducibility[step] = {
            "result": "PASS",
            "expected_from_quarantined_attempt": expected,
            "actual_fresh_rerun": actual,
        }

    stage_started_at = now()
    stage_started = time.perf_counter()

    for window in stage_spec["windows"]:
        name = window["name"]
        first = window["frame_start"]
        last = window["frame_end"]
        window_root = stage_root / name
        geometry_root = window_root / "geometry"

        records[f"{name}.geometry"] = run_process(
            stage_root,
            f"{name}.geometry",
            planned[f"{name}.geometry"],
            geometry_root / "geometry.stdout.log",
            geometry_root / "geometry.stderr.log",
        )
        geometry_summary_path = geometry_root / "geometry_summary.json"
        geometry_summary = load_json(geometry_summary_path)
        assert geometry_summary["status"] == "completed"
        config = geometry_summary["configuration"]
        assert config["frame_start"] == first and config["frame_end"] == last
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
        assert config["cache_sha256"] == a4_frozen["cache_sha256"]
        assert config["calibration_file_sha256"] == a4_frozen[
            "calibration_file_sha256"
        ]
        assert_summary_code(geometry_summary, freeze)
        assert_calibration(geometry_summary)
        geometry_hash_before = geometry_summary["output_sha256"]

        records[f"{name}.geometry_resume"] = run_process(
            stage_root,
            f"{name}.geometry_resume",
            planned[f"{name}.geometry_resume"],
            geometry_root / "geometry.resume.stdout.log",
            geometry_root / "geometry.resume.stderr.log",
        )
        geometry_summary = load_json(geometry_summary_path)
        assert geometry_summary["reused_blocks"] == geometry_summary["total_blocks"]
        assert geometry_summary["output_sha256"] == geometry_hash_before
        authenticate_advertised_outputs(geometry_summary, stage_root)
        check_replacement(
            f"{name}.geometry",
            named_advertised_outputs(geometry_summary),
        )
        resume_checks[f"{name}.geometry"] = {
            "result": "PASS",
            "reused_blocks": geometry_summary["reused_blocks"],
            "total_blocks": geometry_summary["total_blocks"],
            "output_sha256_before": geometry_hash_before,
            "output_sha256_after": geometry_summary["output_sha256"],
        }
        summaries[f"{name}.geometry"] = {
            "path": relative(geometry_summary_path),
            "sha256": sha256(geometry_summary_path),
            "status": geometry_summary["status"],
        }

        for method, summary_name in (
            ("A1", "a1_summary.json"),
            ("A2", "a2_summary.json"),
            ("A3", "a3_summary.json"),
        ):
            method_root = window_root / method
            records[f"{name}.{method}"] = run_process(
                stage_root,
                f"{name}.{method}",
                planned[f"{name}.{method}"],
                method_root / f"{method.lower()}.stdout.log",
                method_root / f"{method.lower()}.stderr.log",
            )
            summary_path = method_root / summary_name
            summary = load_json(summary_path)
            assert summary["status"] == "completed"
            assert_relink_settings(summary, method, first, last)
            assert_summary_code(summary, freeze)
            assert_calibration(summary)
            authenticate_advertised_outputs(summary, stage_root)
            check_replacement(
                f"{name}.{method}",
                named_advertised_outputs(summary),
            )
            summaries[f"{name}.{method}"] = {
                "path": relative(summary_path),
                "sha256": sha256(summary_path),
                "status": summary["status"],
            }
            scientific_statuses[f"{method}:{summary['status']}"] += 1

        a4_root = window_root / "A4"
        records[f"{name}.A4"] = run_process(
            stage_root,
            f"{name}.A4",
            planned[f"{name}.A4"],
            a4_root / "a4.stdout.log",
            a4_root / "a4.stderr.log",
        )
        a4_summary_path = a4_root / "a4_summary.json"
        a4_summary = load_json(a4_summary_path)
        a4_config = a4_summary["configuration"]
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
        ):
            assert a4_config[key] == a4_frozen[key], (name, key)
        assert (
            a4_config["frozen_body_length_px"]
            == a4_frozen["body_length_px"]
        )
        assert_summary_code(a4_summary, freeze)
        assert_calibration(a4_summary)
        a4_outputs_before = authenticate_advertised_outputs(a4_summary, stage_root)
        a4_status_before = a4_summary["status"]

        records[f"{name}.A4_resume"] = run_process(
            stage_root,
            f"{name}.A4_resume",
            planned[f"{name}.A4_resume"],
            a4_root / "a4.resume.stdout.log",
            a4_root / "a4.resume.stderr.log",
        )
        a4_summary = load_json(a4_summary_path)
        assert a4_summary["status"] == a4_status_before
        assert (
            a4_summary["reused_hypothesis_blocks"]
            == a4_summary["hypothesis_blocks"]
        )
        a4_outputs_after = authenticate_advertised_outputs(a4_summary, stage_root)
        assert a4_outputs_after == a4_outputs_before
        check_replacement(
            f"{name}.A4",
            named_advertised_outputs(a4_summary),
        )
        resume_checks[f"{name}.A4"] = {
            "result": "PASS",
            "reused_hypothesis_blocks": a4_summary["reused_hypothesis_blocks"],
            "hypothesis_blocks": a4_summary["hypothesis_blocks"],
            "advertised_output_sha256_before": a4_outputs_before,
            "advertised_output_sha256_after": a4_outputs_after,
        }
        summaries[f"{name}.A4"] = {
            "path": relative(a4_summary_path),
            "sha256": sha256(a4_summary_path),
            "status": a4_summary["status"],
        }
        scientific_statuses[f"A4:{a4_summary['status']}"] += 1

        records[f"{name}.A4_schema_adapter"] = run_process(
            stage_root,
            f"{name}.A4_schema_adapter",
            planned[f"{name}.A4_schema_adapter"],
            a4_root / "a4_schema_adapter.stdout.log",
            a4_root / "a4_schema_adapter.stderr.log",
        )
        adapter_summary_path = a4_root / "a4_selected_for_a5_summary.json"
        adapter_summary = load_json(adapter_summary_path)
        assert adapter_summary["status"] == "completed"
        assert adapter_summary["adapter"]["sha256"] == ADAPTER_SHA256
        assert adapter_summary["frozen_builder"]["sha256"] == (
            freeze["implementation_sha256"][
                "scripts\\reconstruction_3d\\mouth_tracklet_ablation.py"
            ]
        )
        assert adapter_summary["a4_scientific_status"] == a4_summary["status"]
        assert (
            adapter_summary["inputs"]["a4_selected"]["sha256"]
            == a4_summary["outputs"]["selected"]["sha256"]
        )
        adapted_path = Path(adapter_summary["output"]["path"])
        assert_inside_stage(adapted_path, stage_root)
        assert sha256(adapted_path) == adapter_summary["output"]["sha256"]
        assert adapter_summary["scientific_rows_added_removed_or_changed"] is False
        assert adapter_summary["identity_information_added"] is False
        assert adapter_summary["thresholds_changed"] is False
        adapted_abstention = adapter_summary["mode"] in {
            "zero_row_zero_column_abstention_schema_normalized",
            "canonical_empty_byte_copy",
        }
        if adapted_abstention:
            assert a4_summary["status"] == "insufficient_three_path_solution"
            assert adapter_summary["inputs"]["a4_selected"]["rows"] == 0
            assert adapter_summary["output"]["rows"] == 0
            assert len(adapter_summary["output"]["columns"]) == 29
        else:
            assert adapter_summary["mode"] == "nonempty_byte_identical_copy"
            assert (
                adapter_summary["output"]["sha256"]
                == adapter_summary["inputs"]["a4_selected"]["sha256"]
            )
        adapter_checks[f"{name}.A4_schema_adapter"] = {
            "result": "PASS",
            "mode": adapter_summary["mode"],
            "a4_status": a4_summary["status"],
            "source_rows": adapter_summary["inputs"]["a4_selected"]["rows"],
            "source_columns": len(
                adapter_summary["inputs"]["a4_selected"]["columns"]
            ),
            "output_rows": adapter_summary["output"]["rows"],
            "output_columns": len(adapter_summary["output"]["columns"]),
            "source_sha256": adapter_summary["inputs"]["a4_selected"]["sha256"],
            "output_sha256": adapter_summary["output"]["sha256"],
        }
        summaries[f"{name}.A4_schema_adapter"] = {
            "path": relative(adapter_summary_path),
            "sha256": sha256(adapter_summary_path),
            "status": adapter_summary["mode"],
        }

        a5_root = window_root / "A5"
        records[f"{name}.A5_hints"] = run_process(
            stage_root,
            f"{name}.A5_hints",
            planned[f"{name}.A5_hints"],
            a5_root / "hints.stdout.log",
            a5_root / "hints.stderr.log",
        )
        hints_summary_path = a5_root / "a5_hints_summary.json"
        hints_summary = load_json(hints_summary_path)
        thresholds = freeze["frozen_methods"]["A5"]["hint_thresholds"]
        for key, value in thresholds.items():
            assert hints_summary[key] == value, (name, key)
        assert_summary_code(hints_summary, freeze)
        assert sha256(a5_root / "a5_hints.csv") == hints_summary["outputs"][
            "hints_sha256"
        ]
        assert sha256(a5_root / "a5_forced_reset_frames.json") == hints_summary[
            "outputs"
        ]["forced_reset_frames_sha256"]
        hints_named = {
            "hints": sha256(a5_root / "a5_hints.csv"),
            "forced_reset_frames": sha256(
                a5_root / "a5_forced_reset_frames.json"
            ),
        }
        check_replacement(f"{name}.A5_hints", hints_named)
        if adapted_abstention:
            assert hints_summary["hint_rows"] == 0
            assert hints_summary["mapped_segments"] == 0
            assert hints_summary["forced_reset_frames"] == []
            assert load_json(a5_root / "a5_forced_reset_frames.json") == []
            a2_direct = pd.read_csv(window_root / "A2" / "a2_direct.csv")
            assert set(a2_direct.identity_segment.dropna().astype(int)) <= {0}
            adapter_checks[f"{name}.A5_zero_hint_contract"] = {
                "result": "PASS",
                "hint_rows": 0,
                "forced_reset_frames": [],
                "a2_identity_segments": sorted(
                    set(a2_direct.identity_segment.dropna().astype(int))
                ),
            }
        summaries[f"{name}.A5_hints"] = {
            "path": relative(hints_summary_path),
            "sha256": sha256(hints_summary_path),
            "status": "completed",
        }

        records[f"{name}.A5"] = run_process(
            stage_root,
            f"{name}.A5",
            planned[f"{name}.A5"],
            a5_root / "a5.stdout.log",
            a5_root / "a5.stderr.log",
        )
        a5_summary_path = a5_root / "a5_summary.json"
        a5_summary = load_json(a5_summary_path)
        assert a5_summary["status"] == "completed"
        expected_deferred = freeze["frozen_methods"]["A5"]["linker_settings"].copy()
        expected_deferred.pop("xyz_tolerance_mm")
        expected_deferred["num_individuals"] = 3
        expected_deferred["velocity_prediction_weight"] = expected_deferred.pop(
            "velocity_weight"
        )
        expected_deferred["motion_uncertainty_growth_per_frame"] = (
            expected_deferred.pop("uncertainty_growth_per_frame")
        )
        expected_deferred.pop("tangent_weight")
        expected_deferred["tangent_weight"] = 0.0
        assert (
            a5_summary["settings"]["deferred_tracking_config"]
            == expected_deferred
        )
        assert a5_summary["settings"]["xyz_tolerance_mm"] == 1e-9
        assert_summary_code(a5_summary, freeze)
        assert_calibration(a5_summary)
        authenticate_advertised_outputs(a5_summary, stage_root)
        check_replacement(
            f"{name}.A5",
            named_advertised_outputs(a5_summary),
        )
        if adapted_abstention:
            core_columns = [
                "frame",
                "fish_id",
                "source_index",
                "hypothesis_id",
                "X",
                "Y",
                "Z",
                "identity_segment",
            ]
            a3_core = pd.read_csv(
                window_root / "A3" / "a3_direct.csv"
            )[core_columns]
            a5_core = pd.read_csv(
                window_root / "A5" / "a5_direct.csv"
            )[core_columns]
            pd.testing.assert_frame_equal(
                a5_core,
                a3_core,
                check_exact=True,
                check_dtype=True,
            )
            core_bytes = a3_core.to_csv(
                index=False, lineterminator="\n"
            ).encode("utf-8")
            adapter_checks[f"{name}.A5_equals_A3_core"] = {
                "result": "PASS",
                "columns": core_columns,
                "rows": len(a3_core),
                "canonical_core_sha256": hashlib.sha256(core_bytes).hexdigest(),
            }
        summaries[f"{name}.A5"] = {
            "path": relative(a5_summary_path),
            "sha256": sha256(a5_summary_path),
            "status": a5_summary["status"],
        }
        scientific_statuses[f"A5:{a5_summary['status']}"] += 1

    post_auth = authentication_snapshot()
    assert post_auth["bindings"] == prelaunch["bindings"]
    post_auth_path = stage_root / "postcompletion_auth.json"
    write_json_atomic(post_auth_path, post_auth)

    inventory = {}
    for path in sorted(stage_root.rglob("*")):
        if (
            path.is_file()
            and path.name != "STAGE_COMPLETED.json"
            and path.name
            not in {
                "detached_launcher.stdout.log",
                "detached_launcher.stderr.log",
            }
        ):
            inventory[relative(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    stage_wall_seconds = time.perf_counter() - stage_started
    completion = {
        "schema_version": "round8_bounded_postfreeze_stage_v1",
        "status": "completed_and_authenticated",
        "stage": stage,
        "role": stage_spec["role"],
        "method_roles": METHOD_ROLES,
        "windows": stage_spec["windows"],
        "started_at": stage_started_at,
        "completed_at": now(),
        "wall_seconds": stage_wall_seconds,
        "peak_rss_bytes_max": max(
            record["peak_rss_bytes"] for record in records.values()
        ),
        "peak_memory_method": "maximum process-tree RSS across 0.1 s samples",
        "freeze": post_auth["bindings"]["freeze"],
        "runner": post_auth["bindings"]["runner"],
        "prelaunch_auth": {
            "path": relative(stage_root / "prelaunch_auth.json"),
            "sha256": sha256(stage_root / "prelaunch_auth.json"),
        },
        "postcompletion_auth": {
            "path": relative(post_auth_path),
            "sha256": sha256(post_auth_path),
        },
        "prepared_manifest": {
            "path": relative(prepared_path),
            "sha256": sha256(prepared_path),
        },
        "process_records": {
            key: {
                "path": relative(
                    stage_root / "process_records" / f"{key}.json"
                ),
                "sha256": sha256(
                    stage_root / "process_records" / f"{key}.json"
                ),
                "wall_seconds": record["wall_seconds"],
                "peak_rss_bytes": record["peak_rss_bytes"],
            }
            for key, record in records.items()
        },
        "summaries": summaries,
        "scientific_status_counts": dict(scientific_statuses),
        "resume_equivalence": resume_checks,
        "a4_to_a5_schema_adapter_checks": adapter_checks,
        "fresh_replacement_reproducibility": replacement_reproducibility,
        "superseded_failed_attempt": prepared["supersedes_failed_attempt"],
        "postfreeze_execution_repair": {
            "path": relative(REPAIR_PROTOCOL),
            "sha256": REPAIR_PROTOCOL_SHA256,
            "canonical_code_or_threshold_change": False,
            "selection_or_tuning_impact": "none",
        },
        "artifact_inventory": inventory,
        "artifact_inventory_count": len(inventory),
        "artifact_inventory_scope": (
            "all files under the stage root before STAGE_COMPLETED.json, except "
            "the still-open detached_launcher stdout/stderr logs; those closed "
            "logs are authenticated by STAGE_FINALIZED.json"
        ),
        "outer_launcher_files_excluded_until_finalization": [
            relative(stage_root / "detached_launcher.stdout.log"),
            relative(stage_root / "detached_launcher.stderr.log"),
        ],
        "selection_or_tuning_performed": False,
        "frozen_settings_changed": False,
        "a4_abstentions_and_unassigned_rows_preserved": True,
        "full_length_launched": False,
    }
    completion_path = stage_root / "STAGE_COMPLETED.json"
    write_json_atomic(completion_path, completion)
    return completion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(args.stage) if args.prepare_only else execute(args.stage)
    print(json.dumps({"path": relative(result), "sha256": sha256(result)}, indent=2))


if __name__ == "__main__":
    main()
