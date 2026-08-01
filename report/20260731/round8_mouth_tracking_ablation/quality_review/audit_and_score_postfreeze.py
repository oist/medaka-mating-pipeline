"""Authenticate and independently score completed Round8 post-freeze stages.

This script is intentionally not executed until the executor writes an atomic
``STAGE_COMPLETED.json`` for a stage. Validation is report-only, held-out is
the sole post-freeze generalization cohort, and critical results are
permanently descriptive.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
import mmap
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from scripts.common.local_paths import load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


REPO = Path(__file__).resolve().parents[4]
REPORT = REPO / "report/20260731/round8_mouth_tracking_ablation"
QUALITY = REPORT / "quality_review"
FREEZE = REPORT / "frozen_thresholds.json"
FREEZE_SHA256 = "09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d"
FINALIZED_SHA256 = {
    "validation": (
        "297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706"
    ),
    "heldout": (
        "3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152"
    ),
    "critical": (
        "fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c"
    ),
}
FINALIZER_SHA256 = {
    "validation": (
        "379e94ccfb3979f36e418cdcf304f442c4cc935640a1c72b27f95e496c50dee7"
    ),
    "heldout": (
        "c9ea5d44a57470bda06583a3f5311a9c84708cb8040b0d0233b8156c5c7a7f27"
    ),
    "critical": (
        "c9ea5d44a57470bda06583a3f5311a9c84708cb8040b0d0233b8156c5c7a7f27"
    ),
}
RUNNER_SHA256 = {
    "validation": (
        "54ec5698f84c6c65aa2e0aff226dd208445ab1db51c35a60bed8e396f3abab16"
    ),
    "heldout": (
        "0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818"
    ),
    "critical": (
        "2eb4012e428bf8951609a8c345d55fc75c74cd934b955ef5400bb2c56d72c38d"
    ),
}
PREPARED_SHA256 = {
    "validation": (
        "3fb4e79994df34c5d0d140a1ed5f24d5358970114230950d7cd9addb74eabae6"
    ),
    "heldout": (
        "72557e96b34c2582c58419bbaa1e71ae6379b89f7e8e3624a42fbc1e5c216037"
    ),
    "critical": (
        "8900912581348c4df936767c639b429100d17c3d5736d528c108679824862d6b"
    ),
}
PREAUTH_SHA256 = {
    "validation": (
        "abdfe584b902c3e904712231f5b4ba1b701c89b5d7c7cac8af0ed7870eb2c514"
    ),
    "heldout": (
        "871420dd5b55ed4d3b5b69beb91dc12035644e1c30c919199e590d0737c1f61b"
    ),
    "critical": (
        "0cfe3a3f000d964838d6c1cd0ba075a2cc6ac5c5c086644dd02441aa4100cd4c"
    ),
}
ADAPTER_SHA256 = {
    "heldout": (
        "eea6877f16204f63a9fea4820e92a6696f8b24bdddfcc4fab57e2b25a5ffa95b"
    ),
}
REPLACEMENT_PROTOCOL_SHA256 = {
    "heldout": (
        "d42a5095829f60b8ede4fa1b8f4dd6cfa490592f468641f2439a4a20d0e6f42a"
    ),
}
ADAPTER_SMOKE_SHA256 = {
    "heldout": (
        "8d4e645d28c111b5914e99237a2b22b96f559590006175a1d29318a89902f9ea"
    ),
}
LEDGER = REPO / "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv"
CALIBRATION_DIR = REPO / "calibration_results/fixed_refractive_192x192x186"
METRIC_HELPER = QUALITY / "recompute_method_metrics.py"
EXPERIMENT = "20251225_K_1_2_FCMC"
TAG = "round8_260729_snn031_m6"
METHODS = ("A0", "A1", "A2", "A3", "A4", "A5")
CRITICAL_DETAIL = (34540, 34565)
CRITICAL_EVENT = (34547, 34555)
FLOAT_TOKEN = rb"(?:-?Infinity|NaN|null|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
FRAME_ERRORS_PATTERN = re.compile(
    rb'"(\d+)":\{"groups":\{.*?\},"errors":\{"a":('
    + FLOAT_TOKEN
    + rb'),"b":('
    + FLOAT_TOKEN
    + rb'),"c":('
    + FLOAT_TOKEN
    + rb")\}",
    flags=re.DOTALL,
)


STAGE_SPECS = {
    "validation": {
        "root": "validation_postfreeze",
        "role": "report_only_no_selection_or_tie_breaking",
        "windows": {
            "reviewed_event_2": (96612, 97802),
            "control_for_event_2": (151050, 152240),
        },
        "confirmation": False,
    },
    "heldout": {
        "root": "heldout_postfreeze",
        "role": "sole_postfreeze_generalization_evidence",
        "windows": {
            "reviewed_event_4": (273617, 278182),
            "control_for_event_4": (57363, 61928),
            "low_top_pair_separation_8_9_a": (123124, 123274),
            "low_top_pair_separation_8_9_b": (136451, 136601),
            "low_top_pair_separation_8_11": (167108, 167258),
        },
        "confirmation": True,
    },
    "critical": {
        "root": "critical_descriptive_postfreeze",
        "role": "permanently_unblinded_descriptive_nonconfirmatory",
        "freeze_role": "permanently_unblinded_descriptive_stress_evidence_only",
        "windows": {"critical_descriptive_34000_35100": (34000, 35100)},
        "detail": CRITICAL_DETAIL,
        "event": CRITICAL_EVENT,
        "confirmation": False,
    },
    "full_length": {
        "root": "full_length_postfreeze",
        "role": "full_length_finalist_qc",
        "windows": {"full_experiment": (0, 348599)},
        "methods": ("A1", "A2", "A3"),
        "confirmation": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_metric_helper():
    spec = importlib.util.spec_from_file_location("postfreeze_metric_helper", METRIC_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load independent metric helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject_stringified(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            reject_stringified(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_stringified(child, f"{location}[{index}]")
    elif isinstance(value, str) and (
        value.strip().startswith("@{") or "System.Object[]" in value
    ):
        raise ValueError(f"stringified structure in manifest at {location}")


def resolve_inside(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"stage artifact escapes root: {path}")
    return path


def authenticate_item(root: Path, item: dict[str, Any]) -> Path:
    path = resolve_inside(root, str(item["path"]))
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"stage artifact hash mismatch: {path}")
    return path


def authenticate_stage(stage: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    spec = STAGE_SPECS[stage]
    root = REPORT / spec["root"]
    finalized_path = root / "STAGE_FINALIZED.json"
    if not finalized_path.is_file():
        raise FileNotFoundError(
            f"refusing to read non-finalized post-freeze stage: {finalized_path}"
        )
    expected_finalized_sha = FINALIZED_SHA256.get(stage)
    if expected_finalized_sha is None:
        raise ValueError(f"no reviewer-bound finalized marker hash for stage: {stage}")
    if sha256(finalized_path) != expected_finalized_sha:
        raise ValueError("finalized stage marker hash differs from reviewer binding")
    finalized = json.loads(finalized_path.read_text(encoding="utf-8"))
    reject_stringified(finalized)
    if (
        finalized.get("schema_version") != "round8_bounded_postfreeze_finalized_v1"
        or finalized.get("status") != "finalized_and_fully_authenticated"
        or finalized.get("stage") != stage
        or finalized.get("role") != spec["role"]
        or finalized.get("scientific_results_changed_during_finalization") is not False
        or finalized.get("selection_or_tuning_performed") is not False
        or finalized.get("heldout_or_later_stage_opened") is not False
        or finalized.get("finalizer", {}).get("sha256")
        != FINALIZER_SHA256.get(stage)
        or finalized.get("bindings", {}).get("runner", {}).get("sha256")
        != RUNNER_SHA256.get(stage)
        or finalized.get("bindings", {}).get("freeze", {}).get("sha256")
        != FREEZE_SHA256
        or finalized.get("bindings", {}).get("prepared_manifest", {}).get(
            "sha256"
        )
        != PREPARED_SHA256.get(stage)
        or finalized.get("bindings", {}).get("prelaunch_auth", {}).get(
            "sha256"
        )
        != PREAUTH_SHA256.get(stage)
    ):
        raise ValueError("finalized stage identity/provenance mismatch")

    current_inventory = finalized.get("current_stage_inventory", {})
    if (
        not isinstance(current_inventory, dict)
        or len(current_inventory) != finalized.get("current_stage_inventory_count")
    ):
        raise ValueError("finalized current inventory absent or count-mismatched")
    finalized_inventory_paths: set[Path] = set()
    for path_text, metadata in current_inventory.items():
        path = authenticate_item(root, {"path": path_text, **metadata})
        if int(metadata["bytes"]) != path.stat().st_size:
            raise ValueError(f"finalized artifact byte-size mismatch: {path}")
        finalized_inventory_paths.add(path)
    actual_finalized_paths = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != finalized_path.resolve()
    }
    if finalized_inventory_paths != actual_finalized_paths:
        raise ValueError("finalized recursive current inventory mismatch")

    finalization_audit = finalized.get("completed_marker_inventory_validation", {})
    expected_append = finalization_audit.get("expected_post_marker_append", {})
    common_finalization_bad = (
        finalization_audit.get("result") != "PASS"
        or finalization_audit.get("recorded_artifact_count")
        != finalization_audit.get("checked_artifact_count")
        or finalized.get("outer_launcher", {}).get("exit_code") != 0
        or finalized.get("outer_launcher", {}).get("stderr", {}).get("bytes") != 0
    )
    expected_stdout_path = str(
        Path(
            "report/20260731/round8_mouth_tracking_ablation/"
            f"{spec['root']}/detached_launcher.stdout.log"
        )
    ).replace("/", "\\")
    if stage == "validation":
        stage_finalization_bad = (
            finalization_audit.get("unchanged_artifact_count")
            != finalization_audit.get("recorded_artifact_count") - 1
            or expected_append.get("path") != expected_stdout_path
            or expected_append.get("recorded", {}).get("bytes") != 0
            or int(expected_append.get("actual", {}).get("bytes", 0)) <= 0
        )
    else:
        stage_finalization_bad = (
            expected_append not in ({}, None)
            or finalization_audit.get("unchanged_artifact_count")
            != finalization_audit.get("recorded_artifact_count")
        )
    if common_finalization_bad or stage_finalization_bad:
        raise ValueError("finalization race-resolution contract mismatch")

    sentinel_path = authenticate_item(
        root, finalized["bindings"]["completed_marker"]
    )
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    reject_stringified(sentinel)
    if (
        sentinel.get("schema_version") != "round8_bounded_postfreeze_stage_v1"
        or sentinel.get("status") not in {"completed_and_authenticated", "completed"}
        or sentinel.get("stage") != stage
        or sentinel.get("role") != spec["role"]
        or sentinel.get("freeze", {}).get("sha256") != FREEZE_SHA256
        or sentinel.get("frozen_settings_changed") is not False
        or sentinel.get("selection_or_tuning_performed") is not False
    ):
        raise ValueError("stage completion identity/freeze/role mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256(FREEZE) != FREEZE_SHA256:
        raise ValueError("freeze changed before post-freeze scoring")
    if sentinel.get("freeze", {}).get("frozen_at") != freeze["frozen_at"]:
        raise ValueError("stage freeze timestamp mismatch")

    expected_windows = {
        name: [start, end] for name, (start, end) in spec["windows"].items()
    }
    observed_windows = sentinel.get("windows")
    if isinstance(observed_windows, list):
        observed_windows = {
            str(item["name"]): [
                int(item.get("frame_start", item.get("frames", [None])[0])),
                int(item.get("frame_end", item.get("frames", [None, None])[1])),
            ]
            for item in observed_windows
        }
    if observed_windows != expected_windows:
        raise ValueError(f"stage window set/ranges differ from freeze: {stage}")
    for key in ("prepared_manifest", "prelaunch_auth", "postcompletion_auth"):
        authenticate_item(root, sentinel[key])
    process_records = sentinel.get("process_records", {})
    if not isinstance(process_records, dict) or not process_records:
        raise ValueError("stage process records absent or malformed")
    for item in process_records.values():
        authenticate_item(root, item)
        if float(item["wall_seconds"]) < 0 or int(item["peak_rss_bytes"]) <= 0:
            raise ValueError("invalid process resource record")

    inventory = sentinel.get("artifact_inventory", {})
    if not isinstance(inventory, dict):
        raise ValueError("stage inventory is not a path-keyed object")
    inventory_paths: set[Path] = set()
    for path_text, metadata in inventory.items():
        path = resolve_inside(root, path_text)
        if stage == "validation" and path_text == expected_append["path"]:
            if metadata != expected_append["recorded"]:
                raise ValueError("completed marker did not preserve recorded stdout state")
            if current_inventory[path_text] != expected_append["actual"]:
                raise ValueError("finalized marker did not bind appended stdout state")
        else:
            authenticate_item(root, {"path": path_text, **metadata})
            if int(metadata["bytes"]) != path.stat().st_size:
                raise ValueError(f"stage artifact byte-size mismatch: {path}")
            if current_inventory[path_text] != metadata:
                raise ValueError(f"scientific artifact changed after marker: {path}")
        if path in inventory_paths:
            raise ValueError("duplicate stage inventory path")
        inventory_paths.add(path)
    finalized_extras = {
        str(finalized["bindings"]["completed_marker"]["path"])
    }
    if stage != "validation":
        finalized_extras |= {
            str(root.relative_to(REPO) / "DETACHED_LAUNCH_EXIT.json").replace(
                "/", "\\"
            ),
            str(root.relative_to(REPO) / "detached_launcher.stderr.log").replace(
                "/", "\\"
            ),
            str(root.relative_to(REPO) / "detached_launcher.stdout.log").replace(
                "/", "\\"
            ),
        }
    if set(current_inventory) != set(inventory) | finalized_extras:
        raise ValueError("finalized inventory is not marker inventory plus marker")
    if int(sentinel.get("artifact_inventory_count", -1)) != len(inventory):
        raise ValueError("stage inventory count mismatch")
    summaries = sentinel.get("summaries", {})
    if not isinstance(summaries, dict):
        raise ValueError("stage summaries are not a key-addressed object")
    coverage_methods = tuple(
        spec.get("methods", ("geometry", "A1", "A2", "A3", "A4", "A5"))
    )
    expected_summary_keys = {
        f"{window}.{method}"
        for window in spec["windows"]
        for method in coverage_methods
    }
    observed_summary_keys: set[str] = set()
    for summary_key, item in summaries.items():
        path = authenticate_item(root, item)
        if not (
            summary_key.endswith(".A5_hints")
            or summary_key.endswith(".A4_schema_adapter")
        ):
            observed_summary_keys.add(summary_key)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for output in summary.get("outputs", {}).values():
            if isinstance(output, dict) and "path" in output and "sha256" in output:
                authenticate_item(root, output)
    if observed_summary_keys != expected_summary_keys:
        raise ValueError("stage summary method/window coverage mismatch")
    resume = sentinel.get("resume_equivalence", {})
    expected_replays = {
        f"{window}.{method}"
        for window in spec["windows"]
        for method in ("geometry", "A4")
        if method in coverage_methods
    }
    if set(resume) != expected_replays:
        raise ValueError("stage resume-equivalence coverage mismatch")
    for key, evidence in resume.items():
        if evidence.get("result") != "PASS":
            raise ValueError(f"stage deterministic/resume evidence failed: {key}")
        if key.endswith(".geometry"):
            if evidence["output_sha256_before"] != evidence["output_sha256_after"]:
                raise ValueError(f"geometry replay hash mismatch: {key}")
        else:
            if (
                evidence["advertised_output_sha256_before"]
                != evidence["advertised_output_sha256_after"]
            ):
                raise ValueError(f"A4 replay hash mismatch: {key}")
    return root, sentinel, freeze


def load_csv_window(
    path: Path,
    frame_start: int,
    frame_end: int,
    *,
    preferred_id: str | None = None,
) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0).columns)
    id_column = preferred_id or next(
        column
        for column in ("fish_id", "final_fish_id", "geometry_id", "path_id")
        if column in header
    )
    columns = ["frame", id_column, "X", "Y", "Z"]
    optional = (
        "fish_id",
        "final_fish_id",
        "geometry_id",
        "path_id",
        "is_interpolated",
        "ray_residual_mm2",
        "identity_resolved",
        "is_unresolved",
        "assignment_status",
        "identity_status",
        "identity_segment",
        "path_segment",
        "source_index",
        "hypothesis_id",
    )
    columns += [
        column for column in optional if column in header and column not in columns
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000, low_memory=False):
        chunk["frame"] = pd.to_numeric(chunk.frame, errors="raise").astype(np.int64)
        selected = chunk[chunk.frame.between(frame_start, frame_end)].copy()
        if len(selected):
            parts.append(selected)
        if len(chunk) and int(chunk.frame.min()) > frame_end:
            break
    if not parts:
        return pd.DataFrame(columns=columns + ["base_method_id", "method_id"])
    table = pd.concat(parts, ignore_index=True)
    table["base_method_id"] = table[id_column].astype(str)
    segment_column = next(
        (column for column in ("identity_segment", "path_segment") if column in table),
        None,
    )
    if segment_column is None:
        table["method_id"] = table.base_method_id
    else:
        segment = table[segment_column].astype("string").fillna("missing")
        table["method_id"] = (
            table.base_method_id + "#" + segment_column + "=" + segment.astype(str)
        )
    if "is_interpolated" not in table:
        table["is_interpolated"] = False
    else:
        table["is_interpolated"] = (
            table.is_interpolated.fillna(False).astype(bool)
        )
    return table


def direct_values(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    direct = np.isfinite(xyz).all(axis=1)
    direct &= ~table.is_interpolated.to_numpy(dtype=bool)
    residuals = np.empty(0, dtype=float)
    if "ray_residual_mm2" in table:
        values = pd.to_numeric(table.ray_residual_mm2, errors="coerce").to_numpy(float)
        residuals = values[direct & np.isfinite(values)]
    jumps: list[np.ndarray] = []
    grouping = ["method_id"]
    for _key, rows in table.groupby(grouping, sort=True, dropna=False):
        rows = rows.sort_values("frame", kind="stable")
        frames = rows.frame.to_numpy(np.int64)
        points = rows[["X", "Y", "Z"]].to_numpy(float)
        valid_rows = np.isfinite(points).all(axis=1)
        valid_rows &= ~rows.is_interpolated.to_numpy(bool)
        valid = np.diff(frames) == 1
        valid &= valid_rows[:-1] & valid_rows[1:]
        jumps.append(np.linalg.norm(np.diff(points, axis=0), axis=1)[valid])
    return (
        np.concatenate(jumps) if jumps else np.empty(0, dtype=float),
        residuals,
    )


def parse_float(value: bytes) -> float:
    return math.nan if value == b"null" else float(value)


def a0_prelink_residuals(
    path: Path, windows: dict[str, tuple[int, int]]
) -> dict[str, np.ndarray]:
    output = {name: [] for name in windows}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            start = mapped.find(b'"frames":{')
            if start < 0:
                raise ValueError("A0 pre-link JSON lacks frames object")
            for match in FRAME_ERRORS_PATTERN.finditer(mapped, start):
                frame = int(match.group(1))
                for name, (left, right) in windows.items():
                    if left <= frame <= right:
                        output[name].extend(
                            parse_float(match.group(index)) for index in (2, 3, 4)
                        )
    return {
        name: np.asarray(values, dtype=float) for name, values in output.items()
    }


def resolve_a0() -> dict[str, Path]:
    paths = load_local_paths(REPO / "configs/paths.local.toml")
    if paths is None or paths.reconstruction_output_dir is None:
        raise ValueError("cannot resolve A0 reconstruction root")
    root = paths.reconstruction_output_dir / EXPERIMENT / TAG
    resolved = {
        "identified": root / f"{TAG}_3d_identified.csv",
        "prelink": root / f"{TAG}_best_pre_link.json",
    }
    if not all(path.is_file() for path in resolved.values()):
        path_provenance = QUALITY / "critical_event_existing_geometry.json"
        provenance = json.loads(path_provenance.read_text(encoding="utf-8"))
        prelink = Path(provenance["pre_link_source"]["path"])
        root = prelink.parent
        resolved = {
            "identified": root / f"{TAG}_3d_identified.csv",
            "prelink": prelink,
        }
    if not all(path.is_file() for path in resolved.values()):
        raise FileNotFoundError(f"A0 reference files unavailable: {resolved}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    input_provenance_path = REPORT / "input_provenance.json"
    if (
        sha256(input_provenance_path)
        != freeze["source_contract"]["input_provenance"]["sha256"]
    ):
        raise ValueError("input provenance differs from freeze")
    expected = json.loads(
        input_provenance_path.read_text(encoding="utf-8")
    )["a0_unchanged_reference"]
    if (
        sha256(resolved["identified"]) != expected["identified_csv_sha256"]
        or sha256(resolved["prelink"]) != expected["best_pre_link_json_sha256"]
    ):
        raise ValueError("A0 reference artifact hash differs from freeze")
    return resolved


def summary_path_for(root: Path, window: str, method: str) -> Path:
    names = {
        "geometry": "geometry_summary.json",
        "A1": "a1_summary.json",
        "A2": "a2_summary.json",
        "A3": "a3_summary.json",
        "A4": "a4_summary.json",
        "A5": "a5_summary.json",
    }
    return root / window / method / names[method]


def direct_path_from_summary(summary: dict[str, Any]) -> Path:
    item = summary["outputs"]["direct"]
    return Path(item["path"])


def assert_settings_frozen(
    method: str, summary: dict[str, Any], freeze: dict[str, Any]
) -> None:
    frozen = freeze["frozen_methods"][method]
    if method in {"A1", "A2", "A3"}:
        settings = summary.get("settings", summary.get("configuration", {}))
        observed = (
            settings.get("deferred_tracking_config", settings.get("deferred", settings))
            if method == "A3"
            else settings
        )
        expected = frozen["settings"]
        aliases = {
            "velocity_prediction_weight": "velocity_weight",
            "uncertainty_growth": "uncertainty_growth_per_frame",
            "motion_uncertainty_growth_per_frame": (
                "uncertainty_growth_per_frame"
            ),
        }
        normalized = {
            aliases.get(key, key): value for key, value in observed.items()
            if aliases.get(key, key) in expected
        }
        if normalized != expected:
            raise ValueError(f"{method} settings drift from freeze")
    elif method == "A4":
        configuration = summary["configuration"]
        settings = frozen["settings"]
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
            if configuration[key] != settings[key]:
                raise ValueError(f"A4 frozen setting drift: {key}")
    elif method == "A5":
        observed = summary["settings"]["deferred_tracking_config"]
        expected = frozen["linker_settings"]
        aliases = {
            "velocity_prediction_weight": "velocity_weight",
            "uncertainty_growth": "uncertainty_growth_per_frame",
        }
        normalized = {
            aliases.get(key, key): value
            for key, value in observed.items()
            if aliases.get(key, key) in expected
        }
        expected_subset = {key: expected[key] for key in normalized}
        if normalized != expected_subset:
            raise ValueError("A5 linker settings drift")


def a4_extra_qc(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    outputs = summary["outputs"]
    groups = pd.read_csv(outputs["groups"]["path"], low_memory=False)
    dispositions = pd.read_csv(outputs["dispositions"]["path"], low_memory=False)
    selected_ids = set(
        dispositions.loc[
            dispositions.selected.fillna(False).astype(bool), "group_id"
        ].astype(str)
    )
    eligible = groups[groups.eligible.fillna(False).astype(bool)]
    selected = groups[groups.group_id.astype(str).isin(selected_ids)]
    bridges = []
    for view in ("top_cam", "side1_cam", "side2_cam"):
        bridges.append(pd.read_parquet(outputs[f"{view}_bridges"]["path"]))
    bridge_table = pd.concat(bridges, ignore_index=True)
    accepted = bridge_table[
        bridge_table.reciprocal_accepted.fillna(False).astype(bool)
    ]
    margins = []
    for side in ("forward", "backward"):
        margins.extend(
            pd.to_numeric(
                accepted[f"{side}_assignment_margin_body_lengths"],
                errors="coerce",
            ).tolist()
        )
    return {
        "scientific_status": summary["status"],
        "groups": int(len(groups)),
        "eligible_groups": int(len(eligible)),
        "selected_groups": int(len(selected)),
        "eligible_group_selection_fraction": (
            float(len(selected) / len(eligible)) if len(eligible) else 0.0
        ),
        "rejected_count_mismatch_groups": int(
            groups.rejection_reason.fillna("").astype(str).str.contains(
                "retained_tracklet_count_mismatch", regex=False
            ).sum()
        ),
        "false_merge_proxy_count": int(
            sum(
                int(summary["per_view_tracklet_summary"][view]["false_merge_proxy"])
                for view in ("top_cam", "side1_cam", "side2_cam")
            )
        ),
        "false_break_proxy_count": int(
            sum(
                int(summary["per_view_tracklet_summary"][view]["false_break_proxy"])
                for view in ("top_cam", "side1_cam", "side2_cam")
            )
        ),
        "fragment_count": int(
            sum(
                int(summary["per_view_tracklet_summary"][view]["fragment_count"])
                for view in ("top_cam", "side1_cam", "side2_cam")
            )
        ),
        "retained_candidate_rows": int(
            sum(
                int(
                    summary["per_view_tracklet_summary"][view][
                        "retained_candidate_rows"
                    ]
                )
                for view in ("top_cam", "side1_cam", "side2_cam")
            )
        ),
        "accepted_assignment_margin_body_lengths": distribution(margins),
    }


def a5_extra_qc(root: Path, window: str, summary: dict[str, Any]) -> dict[str, Any]:
    hint_summary_path = root / window / "A5" / "a5_hints_summary.json"
    hint_summary = json.loads(hint_summary_path.read_text(encoding="utf-8"))
    application = pd.read_csv(
        summary["outputs"]["hint_application_audit"]["path"], low_memory=False
    )
    return {
        "hint_rows": int(len(application)),
        "hints_honored": int(application.honored.fillna(False).astype(bool).sum()),
        "hint_honored_fraction": (
            float(application.honored.fillna(False).astype(bool).mean())
            if len(application)
            else 0.0
        ),
        "mapped_segments": int(hint_summary["mapped_segments"]),
        "mapping_vote_rejected": int(
            sum(
                not bool(row["accepted_by_vote"])
                for row in hint_summary["mapping_vote_audit"]
            )
        ),
        "overlap_conflicts": int(
            hint_summary["filter_counts"]["rejected_overlap_identity_segments"]
        ),
        "non_bijective_rejections": int(
            hint_summary["filter_counts"]["rejected_per_frame_non_bijective"]
        ),
        "application_status_counts": {
            f"honored={bool(honored)}|status={status}": int(len(rows))
            for (honored, status), rows in application.groupby(
                ["honored", "identity_status"], dropna=False, sort=True
            )
        },
    }


def resource_qc(sentinel: dict[str, Any]) -> dict[str, Any]:
    """Summarize authenticated external process measurements without guessing."""
    records = sentinel.get("process_records", [])
    compact: list[dict[str, Any]] = []
    method_rollup: dict[str, dict[str, Any]] = {}
    if isinstance(records, dict):
        record_items = records.items()
    else:
        record_items = ((str(index), record) for index, record in enumerate(records))
    for index, (record_key, record) in enumerate(record_items):
        key_parts = record_key.split(".", 1)
        inferred_window = key_parts[0] if len(key_parts) == 2 else None
        inferred_method = key_parts[-1].replace("_resume", "")
        method = str(record.get("method", inferred_method))
        wall = float(record["wall_seconds"])
        peak = int(record["peak_rss_bytes"])
        compact.append(
            {
                "index": index,
                "record_key": record_key,
                "window": record.get("window", inferred_window),
                "method": method,
                "purpose": record.get("purpose", record.get("label")),
                "wall_seconds": wall,
                "peak_rss_bytes": peak,
                "record_sha256": record.get("sha256"),
            }
        )
        rollup = method_rollup.setdefault(
            method,
            {
                "process_count": 0,
                "summed_wall_seconds": 0.0,
                "max_peak_rss_bytes": 0,
            },
        )
        rollup["process_count"] += 1
        rollup["summed_wall_seconds"] += wall
        rollup["max_peak_rss_bytes"] = max(rollup["max_peak_rss_bytes"], peak)
    return {
        "measurement_contract": (
            "Wall time and peak RSS are authenticated external process-runner "
            "measurements. Summed wall time is not claimed to equal stage "
            "elapsed time when processes overlap or helper processes are counted."
        ),
        "process_records": compact,
        "per_method_rollup": method_rollup,
        "stage_wall_seconds": sentinel.get(
            "stage_wall_seconds", sentinel.get("wall_seconds")
        ),
        "stage_peak_rss_bytes": sentinel.get(
            "stage_peak_rss_bytes", sentinel.get("peak_rss_bytes_max")
        ),
    }


def authenticate_heldout_replacement(sentinel: dict[str, Any]) -> dict[str, Any]:
    """Independently authenticate the schema-only held-out replacement."""
    repair = sentinel.get("postfreeze_execution_repair", {})
    if (
        repair.get("sha256") != REPLACEMENT_PROTOCOL_SHA256["heldout"]
        or repair.get("canonical_code_or_threshold_change") is not False
        or repair.get("selection_or_tuning_impact") != "none"
    ):
        raise ValueError("held-out repair protocol identity/semantics mismatch")
    protocol_path = REPO / repair["path"]
    adapter_path = REPORT / "adapt_a4_empty_selection_for_a5.py"
    smoke_path = (
        REPORT
        / "quarantine_adapter_contract_smoke_eea6877"
        / "SMOKE_COMPLETED.json"
    )
    if (
        sha256(protocol_path) != REPLACEMENT_PROTOCOL_SHA256["heldout"]
        or sha256(adapter_path) != ADAPTER_SHA256["heldout"]
        or sha256(smoke_path) != ADAPTER_SMOKE_SHA256["heldout"]
    ):
        raise ValueError("held-out adapter/protocol/smoke hash mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    reject_stringified(smoke)
    if smoke.get("status") not in {
        "completed_and_authenticated",
        "passed",
        "PASS",
    }:
        raise ValueError("held-out adapter smoke did not pass")

    superseded = sentinel.get("superseded_failed_attempt", {})
    for item in superseded.get("bindings", {}).values():
        path = (REPO / item["path"]).resolve()
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"superseded-attempt binding mismatch: {path}")
    finalized_item = superseded["bindings"]["finalized_failure"]
    failed_finalized_path = (REPO / finalized_item["path"]).resolve()
    failed_root = failed_finalized_path.parent
    failed_finalized = json.loads(
        failed_finalized_path.read_text(encoding="utf-8")
    )
    reject_stringified(failed_finalized)
    if (
        failed_finalized.get("schema_version")
        != "round8_failed_heldout_attempt_finalized_v1"
        or failed_finalized.get("status")
        != "failed_attempt_preserved_and_authenticated"
        or failed_finalized.get("critical_opened") is not False
        or failed_finalized.get("selection_or_tuning_impact") != "none"
        or failed_finalized.get("scientific_completion_marker_written") is not False
    ):
        raise ValueError("failed held-out attempt preservation semantics mismatch")
    failed_inventory = failed_finalized.get("inventory", {})
    if (
        not isinstance(failed_inventory, dict)
        or len(failed_inventory) != failed_finalized.get("inventory_count")
    ):
        raise ValueError("failed held-out finalized inventory count mismatch")
    verified_failed_paths: set[Path] = set()
    for relative, metadata in failed_inventory.items():
        path = (failed_root / relative).resolve()
        if failed_root.resolve() not in path.parents:
            raise ValueError("failed-attempt inventory path escapes quarantine")
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["bytes"])
            or sha256(path) != metadata["sha256"]
        ):
            raise ValueError(f"failed-attempt quarantine hash mismatch: {path}")
        verified_failed_paths.add(path)
    actual_failed_paths = {
        path.resolve()
        for path in failed_root.rglob("*")
        if path.is_file() and path.resolve() != failed_finalized_path
    }
    if verified_failed_paths != actual_failed_paths:
        raise ValueError("failed-attempt quarantine recursive inventory mismatch")

    manifest_item = superseded["bindings"]["inventory_manifest"]
    failed_manifest_path = (REPO / manifest_item["path"]).resolve()
    failed_manifest = json.loads(
        failed_manifest_path.read_text(encoding="utf-8")
    )
    if (
        failed_manifest.get("schema_version") != "round8_failed_stage_attempt_v1"
        or failed_manifest.get("status") != "preserved_failed_attempt"
        or failed_manifest.get("critical_opened") is not False
        or failed_manifest.get("selection_or_tuning_impact") != "none"
        or failed_manifest.get("scientific_completion_marker_written") is not False
        or len(failed_manifest.get("inventory", {}))
        != failed_manifest.get("inventory_count")
        or set(failed_inventory)
        != set(failed_manifest["inventory"]) | {"FAILED_ATTEMPT_MANIFEST.json"}
    ):
        raise ValueError("failed-attempt original manifest/finalized tree mismatch")
    for relative, metadata in failed_manifest["inventory"].items():
        if failed_inventory[relative] != metadata:
            raise ValueError("failed-attempt inventory metadata changed at finalization")

    verification = superseded.get(
        "failed_quarantine_inventory_verification", {}
    )
    if (
        verification.get("result") != "PASS"
        or verification.get("verified_path_byte_sha256_entries")
        != len(failed_inventory)
        or verification.get("only_excluded_file")
        != failed_finalized_path.name
        or verification.get("excluded_file_sha256") != sha256(failed_finalized_path)
        or verification.get("recorded_inventory_canonical_sha256")
        != stable_hash(failed_inventory)
    ):
        raise ValueError("failed-attempt quarantine verification digest mismatch")

    reproduction = sentinel.get("fresh_replacement_reproducibility", {})
    if len(reproduction) != 26:
        raise ValueError("fresh replacement reproducibility coverage mismatch")
    for key, evidence in reproduction.items():
        if evidence.get("result") != "PASS":
            raise ValueError(f"fresh replacement reproducibility failed: {key}")
        if (
            "actual_fresh_rerun" in evidence
            and evidence["actual_fresh_rerun"]
            != evidence["expected_from_quarantined_attempt"]
        ):
            raise ValueError(f"fresh replacement overlap mismatch: {key}")

    checks = sentinel.get("a4_to_a5_schema_adapter_checks", {})
    expected_check_keys = {
        f"{window}.A4_schema_adapter" for window in STAGE_SPECS["heldout"]["windows"]
    } | {
        "low_top_pair_separation_8_9_b.A5_zero_hint_contract",
        "low_top_pair_separation_8_9_b.A5_equals_A3_core",
        "low_top_pair_separation_8_11.A5_zero_hint_contract",
        "low_top_pair_separation_8_11.A5_equals_A3_core",
    }
    if set(checks) != expected_check_keys:
        raise ValueError("held-out adapter check coverage mismatch")
    if any(evidence.get("result") != "PASS" for evidence in checks.values()):
        raise ValueError("held-out adapter contract check failed")
    for window in (
        "low_top_pair_separation_8_9_b",
        "low_top_pair_separation_8_11",
    ):
        adapter = checks[f"{window}.A4_schema_adapter"]
        zero = checks[f"{window}.A5_zero_hint_contract"]
        core = checks[f"{window}.A5_equals_A3_core"]
        if (
            adapter.get("a4_status") != "insufficient_three_path_solution"
            or adapter.get("source_rows") != 0
            or adapter.get("source_columns") != 0
            or adapter.get("output_rows") != 0
            or int(adapter.get("output_columns", 0)) <= 0
            or zero.get("hint_rows") != 0
            or zero.get("forced_reset_frames") != []
            or int(core.get("rows", -1))
            != 3
            * (
                STAGE_SPECS["heldout"]["windows"][window][1]
                - STAGE_SPECS["heldout"]["windows"][window][0]
                + 1
            )
        ):
            raise ValueError(f"zero-row held-out adapter contract mismatch: {window}")
    return {
        "repair_protocol_sha256": sha256(protocol_path),
        "adapter_sha256": sha256(adapter_path),
        "adapter_smoke_sha256": sha256(smoke_path),
        "failed_attempt_finalized_sha256": sha256(failed_finalized_path),
        "failed_quarantine_authenticated_file_count": len(failed_inventory) + 1,
        "fresh_replacement_reproducibility_checks": len(reproduction),
        "schema_adapter_checks": len(checks),
        "result": "PASS",
    }


def authenticate_critical_registration(sentinel: dict[str, Any]) -> dict[str, Any]:
    authorization = sentinel.get("heldout_quality_authorization", {})
    authorization_path = (REPO / authorization.get("path", "")).resolve()
    if (
        authorization.get("sha256")
        != "022fbca3f3d0c13d7388a83c42b3b2bad60ded00de2fe89fad428943e7f213cc"
        or not authorization_path.is_file()
        or sha256(authorization_path) != authorization["sha256"]
        or authorization.get("independent_quality_verdict") != "PASS"
        or authorization.get("heldout_finalized_sha256")
        != FINALIZED_SHA256["heldout"]
        or authorization.get("freeze_sha256") != FREEZE_SHA256
    ):
        raise ValueError("critical stage lacks exact held-out quality authorization")
    quality_seal = json.loads(authorization_path.read_text(encoding="utf-8"))
    if (
        quality_seal.get("independent_quality_verdict") != "PASS"
        or quality_seal.get("selection_or_tuning_performed") is not False
        or quality_seal.get("critical_opened_during_review") is not False
        or quality_seal["bindings"]["heldout_finalized_marker"]["sha256"]
        != FINALIZED_SHA256["heldout"]
    ):
        raise ValueError("held-out quality seal semantics mismatch")

    core = sentinel.get("authorization_wrapper_core", {})
    core_path = (REPO / core.get("path", "")).resolve()
    if (
        core.get("sha256")
        != "0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818"
        or not core_path.is_file()
        or sha256(core_path) != core["sha256"]
        or sentinel.get("scientific_commands_changed_from_v4") is not False
    ):
        raise ValueError("critical authorization wrapper changed scientific core")
    registration = sentinel.get("critical_descriptive_registration", {})
    if (
        registration.get("confirmatory") is not False
        or registration.get("selection_or_tuning_permitted") is not False
        or registration.get("may_change_frozen_settings_or_finalists") is not False
        or registration.get("full_length_launch_permitted") is not False
        or registration.get("inclusive_frame_intervals") is not True
        or registration.get("execution_window") != [34000, 35100]
        or registration.get("required_detail_window") != list(CRITICAL_DETAIL)
        or registration.get("critical_event_assignment_rows")
        != list(CRITICAL_EVENT)
        or registration.get("role") != STAGE_SPECS["critical"]["role"]
    ):
        raise ValueError("critical descriptive registration mismatch")
    checks = sentinel.get("a4_to_a5_schema_adapter_checks", {})
    expected_key = "critical_descriptive_34000_35100.A4_schema_adapter"
    if set(checks) != {expected_key}:
        raise ValueError("critical adapter check coverage mismatch")
    adapter = checks[expected_key]
    if (
        adapter.get("result") != "PASS"
        or adapter.get("mode") != "nonempty_byte_identical_copy"
        or adapter.get("source_rows") != adapter.get("output_rows")
        or adapter.get("source_columns") != adapter.get("output_columns")
        or adapter.get("source_sha256") != adapter.get("output_sha256")
    ):
        raise ValueError("critical adapter changed nonempty A4 selection")
    return {
        "heldout_quality_seal_sha256": sha256(authorization_path),
        "wrapped_scientific_core_sha256": sha256(core_path),
        "confirmatory": False,
        "selection_or_tuning_performed": False,
        "execution_window": [34000, 35100],
        "detail_window": list(CRITICAL_DETAIL),
        "assignment_rows": list(CRITICAL_EVENT),
        "adapter_nonempty_byte_identical": True,
        "result": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_SPECS),
        action="append",
        required=True,
    )
    args = parser.parse_args()
    helper = load_metric_helper()
    annotations = pd.read_csv(LEDGER, low_memory=False)
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")
    a0 = resolve_a0()
    results: dict[str, Any] = {}
    assignment_parts: list[pd.DataFrame] = []
    critical_event_parts: list[pd.DataFrame] = []
    critical_event_audit_payload: dict[str, Any] | None = None

    for stage in args.stage:
        root, sentinel, freeze = authenticate_stage(stage)
        replacement_evidence = (
            authenticate_heldout_replacement(sentinel)
            if stage == "heldout"
            else None
        )
        critical_registration = (
            authenticate_critical_registration(sentinel)
            if stage == "critical"
            else None
        )
        spec = STAGE_SPECS[stage]
        methods = tuple(spec.get("methods", METHODS))
        windows = spec["windows"]
        residual_windows = dict(windows)
        if stage == "critical":
            residual_windows["__critical_detail"] = CRITICAL_DETAIL
            residual_windows["__critical_event"] = CRITICAL_EVENT
        a0_residuals = (
            a0_prelink_residuals(a0["prelink"], residual_windows)
            if "A0" in methods
            else {}
        )
        method_records: dict[str, Any] = {}
        critical_event_method_records: dict[str, Any] = {}
        for method in methods:
            pooled = Counter()
            jumps: list[float] = []
            residuals: list[float] = []
            window_records: dict[str, Any] = {}
            female_correct = 0
            female_mapped = 0
            female_eligible = 0
            female_changes = 0
            for window, (frame_start, frame_end) in windows.items():
                if method == "A0":
                    table = load_csv_window(
                        a0["identified"],
                        frame_start,
                        frame_end,
                        preferred_id="final_fish_id",
                    )
                    summary = None
                    summary_sha = None
                else:
                    summary_path = summary_path_for(root, window, method)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    assert_settings_frozen(method, summary, freeze)
                    direct_path = direct_path_from_summary(summary)
                    if sha256(direct_path) != summary["outputs"]["direct"]["sha256"]:
                        raise ValueError(f"{method} direct hash mismatch: {window}")
                    table = load_csv_window(direct_path, frame_start, frame_end)
                    summary_sha = sha256(summary_path)
                expected_rows = 3 * (frame_end - frame_start + 1)
                if (
                    len(table) != expected_rows
                    or table.duplicated(["frame", "base_method_id"]).any()
                    or not table.groupby("frame").size().eq(3).all()
                ):
                    raise ValueError(f"{method}/{window} does not have exact 3xframe grid")
                track = helper.method_track_metrics(table.copy())
                raw_jumps, raw_residuals = direct_values(table)
                jumps.extend(raw_jumps.tolist())
                if method == "A0":
                    residuals.extend(a0_residuals[window].tolist())
                else:
                    residuals.extend(raw_residuals.tolist())
                pooled.update(
                    {
                        "frames": track["frame_count"],
                        "rows": track["row_count"],
                        "all_three_direct_frames": track[
                            "all_three_direct_noninterpolated_frame_count"
                        ],
                        "nonfinite_rows": track["nonfinite_row_count"],
                        "explicit_unresolved_rows": track[
                            "explicit_identity_unresolved_row_count"
                        ],
                        "unavailable_or_unresolved_rows": track[
                            "unavailable_or_unresolved_row_count"
                        ],
                    }
                )
                stage_annotations = annotations[
                    annotations.frame.between(frame_start, frame_end)
                ]
                if len(stage_annotations):
                    assignments, identity = helper.female_assignments(
                        stage_annotations,
                        table,
                        method=method,
                        anchor="centroid" if method == "A0" else "mouth",
                        calibration=calibration,
                    )
                else:
                    assignments = pd.DataFrame(
                        columns=[
                            "frame",
                            "method",
                            "assigned_identity",
                            "assignment_distance_px",
                        ]
                    )
                    identity = {
                        "mapped": {
                            "count": 0,
                            "mapped_count": 0,
                            "identity_counts": {},
                            "best_fixed_identity": "",
                            "selective_mapped_best_fixed_accuracy": 0.0,
                            "full_eligible_best_fixed_accuracy": 0.0,
                            "observed_identity_changes": 0,
                        }
                    }
                assignments["stage"] = stage
                assignments["window"] = window
                assignment_parts.append(assignments)
                mapped = identity["mapped"]
                female_eligible += mapped["count"]
                female_mapped += mapped["mapped_count"]
                female_correct += int(
                    round(
                        mapped["full_eligible_best_fixed_accuracy"]
                        * mapped["count"]
                    )
                )
                female_changes += mapped["observed_identity_changes"]
                status_counts: dict[str, int] = {}
                for column in ("identity_status", "assignment_status"):
                    if column in table:
                        status_counts.update(
                            {
                                f"{column}:{key}": int(value)
                                for key, value in table[column]
                                .astype("string")
                                .fillna("missing")
                                .value_counts()
                                .items()
                            }
                        )
                extras: dict[str, Any] = {}
                if stage == "critical":
                    detail = table[
                        table.frame.between(*CRITICAL_DETAIL)
                    ].copy()
                    event = table[
                        table.frame.between(*CRITICAL_EVENT)
                    ].copy()
                    detail_jumps, detail_residuals = direct_values(detail)
                    extras["critical_detail"] = {
                        "frames": list(CRITICAL_DETAIL),
                        "track_qc": helper.method_track_metrics(detail.copy()),
                        "direct_jump_mm": distribution(detail_jumps),
                        "direct_ray_residual_mm2": (
                            distribution(a0_residuals["__critical_detail"])
                            if method == "A0"
                            else distribution(detail_residuals)
                        ),
                        "normalized_scientific_sha256": (
                            helper.normalized_scientific_hash(detail)
                        ),
                    }
                    event.insert(0, "stage", stage)
                    event.insert(1, "method", method)
                    event.insert(2, "window", window)
                    critical_event_parts.append(event)
                    event_jumps, event_direct_residuals = direct_values(event)
                    source_switches = 0
                    source_values: dict[str, list[Any]] = {}
                    if "source_index" in event:
                        for identity_key, rows in event.groupby(
                            "base_method_id", sort=True
                        ):
                            values = (
                                pd.to_numeric(
                                    rows.sort_values("frame").source_index,
                                    errors="coerce",
                                )
                                .dropna()
                                .tolist()
                            )
                            source_values[str(identity_key)] = sorted(set(values))
                            source_switches += sum(
                                left != right
                                for left, right in zip(values[:-1], values[1:])
                            )
                    event_xyz = event[["X", "Y", "Z"]].to_numpy(float)
                    event_residuals = (
                        a0_residuals["__critical_event"]
                        if method == "A0"
                        else event_direct_residuals
                    )
                    critical_event_method_records[method] = {
                        "inclusive_frames": list(CRITICAL_EVENT),
                        "expected_rows": 27,
                        "rows": int(len(event)),
                        "finite_direct_rows": int(
                            np.isfinite(event_xyz).all(axis=1).sum()
                        ),
                        "unavailable_or_unresolved_rows": int(
                            helper.method_track_metrics(event.copy())[
                                "unavailable_or_unresolved_row_count"
                            ]
                        ),
                        "direct_jump_mm": distribution(event_jumps),
                        "direct_or_prelink_ray_residual_mm2": distribution(
                            event_residuals
                        ),
                        "source_index_values_by_identity": source_values,
                        "source_index_switches": int(source_switches),
                        "identity_status_counts": (
                            {
                                str(key): int(value)
                                for key, value in event.identity_status.astype(
                                    "string"
                                )
                                .fillna("missing")
                                .value_counts()
                                .items()
                            }
                            if "identity_status" in event
                            else {}
                        ),
                        "normalized_scientific_sha256": (
                            helper.normalized_scientific_hash(event)
                        ),
                    }
                if method == "A4":
                    extras["tracklet"] = a4_extra_qc(root, summary)
                if method == "A5":
                    extras["hybrid"] = a5_extra_qc(root, window, summary)
                window_records[window] = {
                    "frames": [frame_start, frame_end],
                    "summary_sha256": summary_sha,
                    "track_qc": track,
                    "female_identity": mapped,
                    "identity_status_counts": status_counts,
                    "normalized_scientific_sha256": helper.normalized_scientific_hash(
                        table
                    ),
                    **extras,
                }
            jump_array = np.asarray(jumps, dtype=float)
            method_records[method] = {
                "windows": window_records,
                "pooled": {
                    **dict(pooled),
                    "female_eligible": female_eligible,
                    "female_mapped": female_mapped,
                    "female_best_fixed_correct_window_aligned": female_correct,
                    "female_full_eligible_accuracy": (
                        female_correct / female_eligible
                        if female_eligible
                        else None
                    ),
                    "female_selective_mapped_accuracy": (
                        female_correct / female_mapped if female_mapped else None
                    ),
                    "observed_female_identity_changes": female_changes,
                    "all_three_direct_fraction": (
                        pooled["all_three_direct_frames"] / pooled["frames"]
                    ),
                    "unavailable_or_unresolved_fraction": (
                        pooled["unavailable_or_unresolved_rows"] / pooled["rows"]
                    ),
                    "direct_jump_mm": distribution(jumps),
                    "jump_ge_7_5mm_count": int((jump_array >= 7.5).sum()),
                    "jump_gt_8mm_count": int((jump_array > 8.0).sum()),
                    "residual_evidence": (
                        "centroid_pre_link_json"
                        if method == "A0"
                        else "authenticated_strict_mouth_direct_csv"
                    ),
                    "direct_or_prelink_ray_residual_mm2": distribution(residuals),
                },
            }
        cross_method_checks: dict[str, Any] = {}
        if all(method in method_records for method in ("A1", "A2", "A3")):
            cross_method_checks["a1_a2_a3"] = {
                "all_windows_normalized_scientific_identical": all(
                    len(
                        {
                            method_records[method]["windows"][window][
                                "normalized_scientific_sha256"
                            ]
                            for method in ("A1", "A2", "A3")
                        }
                    )
                    == 1
                    for window in windows
                ),
                "per_window": {
                    window: {
                        method: method_records[method]["windows"][window][
                            "normalized_scientific_sha256"
                        ]
                        for method in ("A1", "A2", "A3")
                    }
                    for window in windows
                },
            }
        if all(method in method_records for method in ("A3", "A4", "A5")):
            for window in windows:
                a4_status = method_records["A4"]["windows"][window]["tracklet"][
                    "scientific_status"
                ]
                a3_hash = method_records["A3"]["windows"][window][
                    "normalized_scientific_sha256"
                ]
                a5_hash = method_records["A5"]["windows"][window][
                    "normalized_scientific_sha256"
                ]
                hint_rows = method_records["A5"]["windows"][window]["hybrid"][
                    "hint_rows"
                ]
                abstained = a4_status != "completed"
                if abstained and (hint_rows != 0 or a5_hash != a3_hash):
                    raise ValueError(
                        f"A4 abstention did not yield zero-hint A3 fallback: {window}"
                    )
                cross_method_checks[window] = {
                    "a4_scientific_status": a4_status,
                    "a4_abstained": abstained,
                    "a5_hint_rows": hint_rows,
                    "a5_equals_a3_normalized_scientific": a5_hash == a3_hash,
                    "a4_abstention_zero_hint_fallback_pass": (
                        not abstained or (hint_rows == 0 and a5_hash == a3_hash)
                    ),
                }
        if stage == "critical":
            window = next(iter(windows))
            margin_records: dict[str, Any] = {}
            for method in ("A3", "A5"):
                audit_path = (
                    root
                    / window
                    / method
                    / f"{method.lower()}_assignment_audit.csv"
                )
                audit = pd.read_csv(audit_path, low_memory=False)
                audit = audit[audit.frame.between(*CRITICAL_EVENT)].copy()
                margins = pd.to_numeric(
                    audit.assignment_margin, errors="coerce"
                ).to_numpy(float)
                margin_records[method] = {
                    "rows": int(len(audit)),
                    "all_resolved": bool(audit.resolved.fillna(False).all()),
                    "ambiguous_rows": int(
                        audit.ambiguous.fillna(False).astype(bool).sum()
                    ),
                    "constraint_conflict_rows": int(
                        audit.constraint_conflict.fillna(False).astype(bool).sum()
                    ),
                    "assignment_margin": distribution(margins),
                    "per_frame": audit[
                        [
                            "frame",
                            "best_cost",
                            "second_cost",
                            "assignment_margin",
                            "hypotheses_retained",
                            "ambiguous",
                            "constraint_conflict",
                            "resolved",
                        ]
                    ].to_dict(orient="records"),
                }
            hints_path = root / window / "A5" / "a5_hints.csv"
            hints = pd.read_csv(hints_path, low_memory=False)
            event_hint_rows = int(
                hints.frame.between(*CRITICAL_EVENT).sum()
                if "frame" in hints
                else 0
            )
            raw_audit_path = QUALITY / "critical_event_raw_audit.json"
            geometry_audit_path = (
                QUALITY / "critical_event_existing_geometry.json"
            )
            raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
            geometry_audit = json.loads(
                geometry_audit_path.read_text(encoding="utf-8")
            )
            tied_methods = ("A1", "A2", "A3", "A5")
            critical_event_audit_payload = {
                "schema_version": 1,
                "status": "critical_event_independently_scored_descriptive_only",
                "role": STAGE_SPECS["critical"]["role"],
                "confirmatory": False,
                "selection_or_tuning_performed": False,
                "may_change_frozen_settings_or_finalists": False,
                "inclusive_detail_window": list(CRITICAL_DETAIL),
                "inclusive_assignment_rows": list(CRITICAL_EVENT),
                "methods": critical_event_method_records,
                "deferred_assignment_margins": margin_records,
                "a5_hint_rows_in_assignment_interval": event_hint_rows,
                "strict_mouth_core_identical_A1_A2_A3_A5": (
                    len(
                        {
                            critical_event_method_records[method][
                                "normalized_scientific_sha256"
                            ]
                            for method in tied_methods
                        }
                    )
                    == 1
                ),
                "a0_known_geometry_swap": {
                    "gate_audit_34547_to_34548": geometry_audit[
                        "gate_audit_34547_to_34548"
                    ],
                    "linked_ID_1_group_34547": geometry_audit["linked_frames"][
                        "34547"
                    ]["groups"]["ID_1"],
                    "linked_ID_1_group_34548": geometry_audit["linked_frames"][
                        "34548"
                    ]["groups"]["ID_1"],
                    "linked_ID_3_group_34547": geometry_audit["linked_frames"][
                        "34547"
                    ]["groups"]["ID_3"],
                    "linked_ID_3_group_34548": geometry_audit["linked_frames"][
                        "34548"
                    ]["groups"]["ID_3"],
                    "interpretation": (
                        "The A0 centroid geometry linker exchanges the two "
                        "close geometry trajectories at 34547->34548 because "
                        "the straight c-to-c edge is infeasible at 8 mm."
                    ),
                },
                "raw_trigger": raw_audit["independent_values"],
                "ground_truth_limitation": (
                    "There is no female annotation in 34547-34555. The two "
                    "close fish are male under the available ledger, so "
                    "biological correctness and a male-to-male exchange are "
                    "not directly observable."
                ),
                "descriptive_interpretation": (
                    "A1/A2/A3/A5 retain stable source indices and identical "
                    "strict-mouth scientific rows across 34547-34555 while "
                    "A4 explicitly abstains. This is evidence of internal "
                    "mouth-trajectory continuity, not biological identity proof."
                ),
                "bindings": {
                    "critical_finalized": {
                        "path": str(root / "STAGE_FINALIZED.json"),
                        "sha256": sha256(root / "STAGE_FINALIZED.json"),
                    },
                    "raw_trigger_audit": {
                        "path": str(raw_audit_path),
                        "sha256": sha256(raw_audit_path),
                    },
                    "a0_geometry_audit": {
                        "path": str(geometry_audit_path),
                        "sha256": sha256(geometry_audit_path),
                    },
                },
            }
        results[stage] = {
            "role": spec["role"],
            "confirmatory_generalization": spec["confirmation"],
            "independent_execution_verdict": "PASS",
            "frozen_settings_changed": False,
            "selection_or_tuning_performed": False,
            "interpretation": (
                "sole post-freeze generalization evidence"
                if spec["confirmation"]
                else (
                    "report-only; not selection or confirmatory evidence"
                    if stage == "validation"
                    else spec["role"]
                )
            ),
            "authoritative_finalized_marker": {
                "path": str(root / "STAGE_FINALIZED.json"),
                "sha256": sha256(root / "STAGE_FINALIZED.json"),
            },
            "bound_completion_marker": {
                "path": str(root / "STAGE_COMPLETED.json"),
                "sha256": sha256(root / "STAGE_COMPLETED.json"),
            },
            "resources": resource_qc(sentinel),
            "determinism": {
                "resume_equivalence": sentinel.get("resume_equivalence"),
            },
            "cross_method_checks": cross_method_checks,
            "heldout_replacement_provenance": replacement_evidence,
            "critical_registration": critical_registration,
            "methods": method_records,
        }

    result = {
        "schema_version": 2,
        "status": "completed_postfreeze_stages_independently_authenticated_and_scored",
        "freeze": {"path": str(FREEZE), "sha256": sha256(FREEZE)},
        "scorer_code_sha256": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(METRIC_HELPER): sha256(METRIC_HELPER),
        },
        "stages": results,
        "interpretation_contract": {
            "validation": "report-only; cannot change frozen settings or finalists",
            "held_out": "sole post-freeze generalization evidence",
            "critical": "permanently unblinded descriptive stress evidence only",
            "female_ground_truth_limitation": (
                "Female-only identity labels cannot observe a male-to-male "
                "exchange; all biological identity conclusions are bounded by "
                "that unobservability."
            ),
            "residual_and_smoothness": (
                "Lower refractive residual or smoother trajectories alone are "
                "not biological identity proof."
            ),
        },
    }
    output = QUALITY / "postfreeze_independent_metrics.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    if assignment_parts:
        pd.concat(assignment_parts, ignore_index=True).to_csv(
            QUALITY / "postfreeze_independent_female_assignments.csv", index=False
        )
    if critical_event_parts:
        assignment_output = (
            QUALITY / "postfreeze_critical_event_assignments_34547_34555.csv"
        )
        pd.concat(critical_event_parts, ignore_index=True).to_csv(
            assignment_output,
            index=False,
        )
        if critical_event_audit_payload is not None:
            critical_event_audit_payload["bindings"]["exact_assignment_rows"] = {
                "path": str(assignment_output),
                "sha256": sha256(assignment_output),
            }
            (QUALITY / "postfreeze_critical_event_audit.json").write_text(
                json.dumps(
                    critical_event_audit_payload, indent=2, allow_nan=True
                ),
                encoding="utf-8",
            )
    print(json.dumps({"stages": list(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
