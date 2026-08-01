"""Independently score the A3 acceleration sweep on training windows only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
RELINK_ROOT = ROOT / "training_relink_a3"
OUTPUT_ROOT = ROOT / "quality_review"
BASE_SCORER_PATH = OUTPUT_ROOT / "score_training_motion_sweep.py"
CALIBRATION_DIR = Path("calibration_results/fixed_refractive_192x192x186")
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}
CONFIGS = {
    "A3_a000": 0.0,
    "A3_a005": 0.05,
    "A3_a010": 0.1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def load_base_scorer():
    spec = importlib.util.spec_from_file_location(
        "independent_training_motion_scorer", BASE_SCORER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the independent base scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    scorer = load_base_scorer()
    annotations = scorer.load_training_annotations()
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")
    if calibration.backend_name != "refractive" or calibration.score_unit != "mm^2":
        raise ValueError("A3 scorer requires the fixed refractive mm^2 backend")

    records: dict[str, Any] = {}
    assignment_parts: list[pd.DataFrame] = []
    for config_name, acceleration_weight in CONFIGS.items():
        pooled_jump_values: list[float] = []
        pooled_residual_values: list[float] = []
        pooled_rows = 0
        pooled_unavailable = 0
        pooled_frames = 0
        pooled_three_direct = 0
        total_eligible = 0
        total_mapped = 0
        total_correct = 0
        total_changes = 0
        window_records: dict[str, Any] = {}
        normalized_settings: set[str] = set()
        code_hash_sets: set[str] = set()
        for window_name, (frame_start, frame_end) in WINDOWS.items():
            folder = RELINK_ROOT / window_name / config_name
            summary_path = folder / "a3_summary.json"
            direct_path = folder / "a3_direct.csv"
            audit_path = folder / "a3_assignment_audit.csv"
            for required in (summary_path, direct_path, audit_path):
                if not required.exists():
                    raise FileNotFoundError(f"incomplete A3 sweep: {required}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed":
                raise ValueError(f"incomplete summary: {summary_path}")
            if summary["outputs"]["direct"]["sha256"] != sha256(direct_path):
                raise ValueError(f"direct hash mismatch: {direct_path}")
            for output_name, output in summary["outputs"].items():
                output_path = Path(output["path"])
                if (
                    not output_path.exists()
                    or sha256(output_path) != output["sha256"]
                ):
                    raise ValueError(
                        f"{output_name} hash mismatch: {summary_path}"
                    )
            if (
                "assignment_audit" not in summary["outputs"]
                or Path(summary["outputs"]["assignment_audit"]["path"]).resolve()
                != audit_path.resolve()
            ):
                raise ValueError(
                    f"assignment audit is not authenticated: {summary_path}"
                )
            settings = dict(summary["settings"])
            if stable_hash(settings) != summary["configuration_hash"]:
                raise ValueError(f"configuration hash mismatch: {summary_path}")
            if summary["code_sha256"] != settings["code_sha256"]:
                raise ValueError(f"code manifest mismatch: {summary_path}")
            for code_path, recorded_hash in settings["code_sha256"].items():
                if sha256(Path(code_path)) != recorded_hash:
                    raise ValueError(f"stale code hash in {summary_path}: {code_path}")
            if (
                settings["method"] != "A3"
                or settings["velocity_weight"] != 0.25
                or settings["motion_history_frames"] != 5
                or settings["tangent_weight"] != 0.0
                or settings["acceleration_weight"] != acceleration_weight
                or settings["max_jump_mm"] != 8.0
                or settings["max_jump_hard_cap_mm"] != 80.0
                or settings["max_gap_frames"] != 100
            ):
                raise ValueError(f"unexpected frozen settings: {summary_path}")
            normalized = dict(settings)
            for key in ("frame_start", "frame_end", "geometry_sha256"):
                normalized.pop(key)
            normalized_settings.add(stable_hash(normalized))
            code_hash_sets.add(stable_hash(settings["code_sha256"]))

            direct = pd.read_csv(direct_path)
            track_metrics = scorer.track_metrics(direct, frame_start, frame_end)
            window_annotations = annotations[
                annotations.training_window.eq(window_name)
            ]
            mapped = scorer.map_female_rows(
                window_annotations,
                direct,
                calibration,
                config_name=config_name,
                window_name=window_name,
            )
            assignment_parts.append(mapped)
            identity_metrics = scorer.summarize_assignments(mapped)
            audit = pd.read_csv(audit_path)
            margin_values = pd.to_numeric(
                audit.assignment_margin, errors="coerce"
            ).to_numpy(dtype=float)
            window_records[window_name] = {
                "direct_sha256": sha256(direct_path),
                "summary_sha256": sha256(summary_path),
                "configuration_hash": summary["configuration_hash"],
                "track_metrics": {
                    key: value
                    for key, value in track_metrics.items()
                    if not key.startswith("_")
                },
                "female_identity": identity_metrics,
                "ambiguity": {
                    "audit_rows": int(len(audit)),
                    "ambiguous_rows": int(
                        audit.ambiguous.fillna(False).astype(bool).sum()
                    ),
                    "resolved_rows": int(
                        audit.resolved.fillna(False).astype(bool).sum()
                    ),
                    "constraint_conflict_rows": int(
                        audit.constraint_conflict.fillna(False).astype(bool).sum()
                    ),
                    "assignment_margin": distribution(margin_values.tolist()),
                    "unresolved_intervals": summary["unresolved_intervals"],
                },
                "wall_seconds": float(summary["wall_seconds"]),
                "peak_rss_bytes": int(summary["peak_rss_bytes"]),
            }
            pooled_jump_values.extend(track_metrics["_jump_values"])
            pooled_residual_values.extend(track_metrics["_residual_values"])
            pooled_rows += track_metrics["rows"]
            pooled_unavailable += track_metrics["unavailable_rows"]
            pooled_frames += frame_end - frame_start + 1
            pooled_three_direct += track_metrics["all_three_direct_frames"]
            total_eligible += identity_metrics["eligible_annotations"]
            total_mapped += identity_metrics["mapped_annotations"]
            total_correct += identity_metrics["best_fixed_correct"]
            total_changes += identity_metrics["observed_identity_changes"]
        records[config_name] = {
            "acceleration_weight": acceleration_weight,
            "normalized_setting_hashes": sorted(normalized_settings),
            "code_hash_sets": sorted(code_hash_sets),
            "windows": window_records,
            "pooled": {
                "female_annotations": total_eligible,
                "female_mapped": total_mapped,
                "female_best_fixed_correct_window_aligned": total_correct,
                "female_full_eligible_accuracy": (
                    total_correct / total_eligible if total_eligible else 0.0
                ),
                "female_selective_mapped_accuracy": (
                    total_correct / total_mapped if total_mapped else 0.0
                ),
                "observed_female_identity_changes": total_changes,
                "all_three_direct_fraction": pooled_three_direct / pooled_frames,
                "unavailable_rows": pooled_unavailable,
                "unavailable_row_fraction": pooled_unavailable / pooled_rows,
                "jump_mm": distribution(pooled_jump_values),
                "jump_ge_7_5mm_count": int(
                    np.sum(np.asarray(pooled_jump_values) >= 7.5)
                ),
                "jump_gt_8mm_count": int(
                    np.sum(np.asarray(pooled_jump_values) > 8.0)
                ),
                "residual_mm2": distribution(pooled_residual_values),
            },
        }

    ranking = sorted(
        CONFIGS,
        key=lambda name: (
            -records[name]["pooled"]["female_full_eligible_accuracy"],
            records[name]["pooled"]["observed_female_identity_changes"],
            records[name]["pooled"]["unavailable_row_fraction"],
            records[name]["pooled"]["jump_ge_7_5mm_count"],
            CONFIGS[name],
            name,
        ),
    )
    eligible_ranking = [
        name for name in ranking if CONFIGS[name] in (0.05, 0.1)
    ]
    result = {
        "schema_version": 1,
        "scope": "training windows only; no validation, held-out, or critical method outputs read",
        "selection_rule": (
            "female full-eligible accuracy, observed identity changes, "
            "unavailable fraction, spike count, then lower acceleration weight"
        ),
        "records": records,
        "ranking_all_ablation_weights": ranking,
        "eligible_a3_ranking": eligible_ranking,
        "selected": eligible_ranking[0],
        "eligibility_rule": (
            "Acceleration 0 is a negative-control deferred-velocity ablation; "
            "the user-defined A3 finalist must use a low positive weight."
        ),
        "limitations": [
            "Only 15 confirmed-female annotations occur in the four training windows.",
            "Female-only labels cannot reveal male-to-male identity swaps.",
            "Temporal controls are length-matched, not behavior- or risk-matched.",
        ],
    }
    output_json = OUTPUT_ROOT / "training_a3_sweep_audit.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat(assignment_parts, ignore_index=True).to_csv(
        OUTPUT_ROOT / "training_a3_sweep_assignments.csv", index=False
    )
    summary_rows = [
        {"config": name, **records[name]["pooled"]}
        for name in CONFIGS
    ]
    flattened = []
    for row in summary_rows:
        item = {
            key: value
            for key, value in row.items()
            if not isinstance(value, dict)
        }
        for prefix in ("jump_mm", "residual_mm2"):
            item.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in row[prefix].items()
                }
            )
        flattened.append(item)
    pd.DataFrame(flattened).to_csv(
        OUTPUT_ROOT / "training_a3_sweep_summary.csv", index=False
    )
    print(
        json.dumps(
            {
                "selected": eligible_ranking[0],
                "eligible_ranking": eligible_ranking,
                "all_ablation_ranking": ranking,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
