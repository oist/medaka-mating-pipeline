"""Independent training-only audit of frozen A5 hints and hybrid outputs."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
OUTPUT_ROOT = ROOT / "quality_review"
A4_ROOT = ROOT / "training_a4_cutoff_sweep"
A2_ROOT = ROOT / "training_relink"
A5_ROOT = ROOT / "training_a5"
A5_COMPLETION = A5_ROOT / "TRAINING_A5_COMPLETED.json"
GEOMETRY_ROOT = ROOT / "training_geometry"
A4_AUDIT = OUTPUT_ROOT / "training_a4_method_sweep_audit.json"
A3_AUDIT = OUTPUT_ROOT / "training_a3_sweep_audit.json"
BASE_SCORER = OUTPUT_ROOT / "score_training_motion_sweep.py"
CALIBRATION_DIR = Path("calibration_results/fixed_refractive_192x192x186")
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}
A4_CONFIG = "s035_m005_g0"
A2_CONFIG = "A2_v025_h5"
HINT_CONFIG = {
    "minimum_mouth_confidence": 0.5,
    "minimum_group_support_frames": 3,
    "maximum_group_median_ray_residual_mm2": 1.0,
    "maximum_group_mad_ray_residual_mm2": 0.25,
    "minimum_group_median_mouth_confidence": 0.5,
    "maximum_group_assignment_cost": 0.0,
    "require_three_views": True,
    "minimum_mapping_votes": 3,
    "minimum_mapping_purity": 0.9,
    "minimum_vote_margin": 2,
    "xyz_tolerance_mm": 1e-9,
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


def load_base_scorer():
    spec = importlib.util.spec_from_file_location("a5_base_scorer", BASE_SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load training scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_tables(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
    numeric: list[str],
) -> None:
    actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(actual) != len(expected):
        raise ValueError(f"row-count mismatch: actual={len(actual)}, expected={len(expected)}")
    for column in expected.columns:
        if column not in actual:
            raise ValueError(f"missing comparison column {column}")
        if column in numeric:
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, atol=1e-10, rtol=0, equal_nan=True):
                raise ValueError(f"numeric mismatch in {column}")
        else:
            left = actual[column].astype("string").fillna("<NA>")
            right = expected[column].astype("string").fillna("<NA>")
            if not left.equals(right):
                raise ValueError(f"value mismatch in {column}")


def independently_build_hints(
    a4: pd.DataFrame, a2: pd.DataFrame
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    counts = {"selected_input_rows": int(len(a4))}
    strong = a4[a4.group_eligible.fillna(False).astype(bool)].copy()
    counts["after_group_eligible"] = len(strong)
    strong = strong[strong.group_retained_count_mismatch_fraction.eq(0)]
    counts["after_no_retained_count_mismatch"] = len(strong)
    strong = strong[strong.view_count.eq(3)]
    counts["after_view_count"] = len(strong)
    strong = strong[strong.mouth_confidence.ge(0.5)]
    counts["after_mouth_confidence"] = len(strong)
    strong = strong[strong.group_support_frames.ge(3)]
    counts["after_group_support"] = len(strong)
    strong = strong[strong.group_median_ray_residual_mm2.le(1.0)]
    counts["after_group_residual"] = len(strong)
    strong = strong[strong.group_mad_ray_residual_mm2.le(0.25)]
    counts["after_group_residual_mad"] = len(strong)
    strong = strong[strong.group_median_mouth_confidence.ge(0.5)]
    counts["after_group_median_mouth_confidence"] = len(strong)
    strong = strong[strong.group_assignment_cost.le(0.0)]
    counts["after_group_assignment_cost"] = len(strong)
    exact = strong.merge(
        a2[
            [
                "frame", "hypothesis_id", "source_index", "fish_id",
                "identity_segment", "X", "Y", "Z",
            ]
        ],
        on=["frame", "hypothesis_id"],
        how="inner",
        suffixes=("_a4", "_geometry"),
    )
    counts["after_exact_shared_geometry_match"] = len(exact)
    delta = np.linalg.norm(
        exact[["X_a4", "Y_a4", "Z_a4"]].to_numpy(float)
        - exact[["X_geometry", "Y_geometry", "Z_geometry"]].to_numpy(float),
        axis=1,
    )
    exact = exact[delta <= 1e-9].copy()
    counts["after_xyz_consistency"] = len(exact)
    exact["segment_key"] = (
        exact.path_id.astype(str) + ":" + exact.path_segment.astype(int).astype(str)
    )
    exact["a2_identity_key"] = (
        exact.fish_id.astype(str) + ":" + exact.identity_segment.astype(int).astype(str)
    )
    intervals = {
        f"{path_id}:{int(segment)}": (int(rows.frame.min()), int(rows.frame.max()))
        for (path_id, segment), rows in a4.groupby(
            ["path_id", "path_segment"], sort=True
        )
    }
    mapping: dict[str, str] = {}
    mapping_segments: dict[str, int] = {}
    vote_rows: list[dict[str, Any]] = []
    for segment_key, rows in exact.groupby("segment_key", sort=True):
        votes = rows.a2_identity_key.astype(str).value_counts()
        best_key = str(votes.index[0])
        identity, segment_text = best_key.rsplit(":", 1)
        best = int(votes.iloc[0])
        second = int(votes.iloc[1]) if len(votes) > 1 else 0
        purity = float(best / votes.sum())
        accepted = best >= 3 and purity >= 0.9 and best - second >= 2
        if accepted:
            mapping[str(segment_key)] = identity
            mapping_segments[str(segment_key)] = int(segment_text)
        vote_rows.append(
            {
                "segment_key": str(segment_key),
                "best_identity": identity,
                "best_a2_identity_segment": int(segment_text),
                "best_a2_identity_key": best_key,
                "best_votes": best,
                "second_votes": second,
                "vote_margin": best - second,
                "purity": purity,
                "accepted_by_vote": accepted,
                "accepted": accepted,
                "overlap_identity_conflict": False,
                "frame_start": intervals[str(segment_key)][0],
                "frame_end": intervals[str(segment_key)][1],
            }
        )
    conflicts: set[str] = set()
    accepted_items = sorted(mapping.items())
    for index, (left_key, identity) in enumerate(accepted_items):
        left_start, left_end = intervals[left_key]
        for right_key, right_identity in accepted_items[index + 1 :]:
            if identity != right_identity:
                continue
            right_start, right_end = intervals[right_key]
            if max(left_start, right_start) <= min(left_end, right_end):
                conflicts.update((left_key, right_key))
    for key in conflicts:
        mapping.pop(key, None)
        mapping_segments.pop(key, None)
    for row in vote_rows:
        if row["segment_key"] in conflicts:
            row["accepted"] = False
            row["overlap_identity_conflict"] = True
    counts["rejected_overlap_identity_segments"] = len(conflicts)
    exact["identity"] = exact.segment_key.map(mapping)
    exact["a2_identity_segment"] = exact.segment_key.map(mapping_segments)
    hints = exact[exact.identity.notna()].copy().sort_values(
        ["frame", "source_index", "identity"], kind="stable"
    )
    duplicate = hints.duplicated(["frame", "source_index"], keep=False)
    duplicate |= hints.duplicated(["frame", "identity"], keep=False)
    counts["rejected_per_frame_non_bijective"] = int(duplicate.sum())
    hints = hints[~duplicate].copy()
    hints["X"] = hints.X_a4
    hints["Y"] = hints.Y_a4
    hints["Z"] = hints.Z_a4
    hints = hints[
        [
            "frame", "source_index", "identity", "path_id", "path_segment",
            "segment_key", "a2_identity_segment", "hypothesis_id", "X", "Y",
            "Z", "mouth_confidence",
        ]
    ].reset_index(drop=True)
    a4_starts = a4.groupby(["path_id", "path_segment"]).frame.min().reset_index()
    a2_starts = (
        a2[a2.identity_segment.astype(int).gt(0)]
        .groupby(["fish_id", "identity_segment"])
        .frame.min()
        .reset_index()
    )
    reset_frames = sorted(
        {
            int(row.frame)
            for row in a4_starts.itertuples(index=False)
            if int(row.path_segment) > 0
        }
        | {int(row.frame) for row in a2_starts.itertuples(index=False)}
    )
    return hints, reset_frames, {
        "filter_counts": {key: int(value) for key, value in counts.items()},
        "mapping_vote_audit": vote_rows,
        "exact_candidate_matches": int(len(exact)),
        "mapped_segments": int(len(mapping)),
    }


def main() -> None:
    scorer = load_base_scorer()
    annotations = scorer.load_training_annotations()
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")
    a4_audit = json.loads(A4_AUDIT.read_text(encoding="utf-8"))
    if a4_audit["selected"] != A4_CONFIG:
        raise ValueError("A5 audit is not bound to final A4 selection")
    completion = json.loads(A5_COMPLETION.read_text(encoding="utf-8"))
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "completed_and_authenticated"
        or int(completion.get("windows", -1)) != 4
        or int(completion.get("hint_summaries", -1)) != 4
        or int(completion.get("a5_summaries", -1)) != 4
        or completion.get("all_inputs_outputs_authenticated") is not True
    ):
        raise ValueError("A5 training completion marker is incomplete")
    manifest = {
        str(item["window"]): item for item in completion["summary_manifest"]
    }
    if set(manifest) != set(WINDOWS):
        raise ValueError("A5 completion marker has wrong window set")
    for item in (
        completion["training_a5_summary_csv"],
        completion["training_a5_audit_json"],
    ):
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("A5 aggregate artifact hash mismatch")
    records: dict[str, Any] = {}
    assignments_all: list[pd.DataFrame] = []
    pooled: Counter[str] = Counter()
    pooled_application_statuses: Counter[str] = Counter()
    pooled_jumps: list[float] = []
    pooled_residuals: list[float] = []

    for window, (frame_start, frame_end) in WINDOWS.items():
        folder = A5_ROOT / window
        hint_summary_path = folder / "a5_hints_summary.json"
        run_summary_path = folder / "a5_summary.json"
        if not hint_summary_path.exists() or not run_summary_path.exists():
            raise FileNotFoundError(f"incomplete A5 training window: {window}")
        hint_summary = json.loads(hint_summary_path.read_text(encoding="utf-8"))
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if (
            sha256(hint_summary_path)
            != manifest[window]["hint_summary"]["sha256"]
            or sha256(run_summary_path)
            != manifest[window]["a5_summary"]["sha256"]
        ):
            raise ValueError(f"A5 completion summary hash mismatch: {window}")
        if run_summary["status"] != "completed":
            raise ValueError(f"A5 run not completed: {window}")
        for key, value in HINT_CONFIG.items():
            actual = hint_summary[key]
            if isinstance(value, bool):
                equal = actual is value
            else:
                equal = np.isclose(float(actual), float(value))
            if not equal:
                raise ValueError(f"non-frozen hint threshold: {window}/{key}")
        if stable_hash(HINT_CONFIG) != hint_summary["configuration_hash"]:
            raise ValueError(f"A5 hint configuration hash mismatch: {window}")

        a4_summary_path = A4_ROOT / window / A4_CONFIG / "a4_summary.json"
        a4_summary = json.loads(a4_summary_path.read_text(encoding="utf-8"))
        expected_a4_summary_hash = a4_audit["records"][A4_CONFIG]["windows"][window][
            "summary_sha256"
        ]
        if sha256(a4_summary_path) != expected_a4_summary_hash:
            raise ValueError(f"A4 summary drift: {window}")
        a4_path = Path(a4_summary["outputs"]["selected"]["path"])
        if sha256(a4_path) != a4_summary["outputs"]["selected"]["sha256"]:
            raise ValueError(f"A4 selected hash mismatch: {window}")
        a2_summary_path = A2_ROOT / window / A2_CONFIG / "a2_summary.json"
        a2_summary = json.loads(a2_summary_path.read_text(encoding="utf-8"))
        a2_path = Path(a2_summary["outputs"]["direct"]["path"])
        if sha256(a2_path) != a2_summary["outputs"]["direct"]["sha256"]:
            raise ValueError(f"A2 direct hash mismatch: {window}")
        if hint_summary["input_sha256"] != {
            "a4_selected": sha256(a4_path),
            "a2_direct": sha256(a2_path),
        }:
            raise ValueError(f"A5 hint input binding mismatch: {window}")

        hints_path = folder / "a5_hints.csv"
        reset_path = folder / "a5_forced_reset_frames.json"
        if (
            sha256(hints_path) != hint_summary["outputs"]["hints_sha256"]
            or sha256(reset_path)
            != hint_summary["outputs"]["forced_reset_frames_sha256"]
        ):
            raise ValueError(f"A5 hint outputs unauthenticated: {window}")
        a4 = pd.read_parquet(a4_path)
        a2 = pd.read_csv(a2_path, low_memory=False)
        expected_hints, expected_resets, expected_audit = independently_build_hints(
            a4, a2
        )
        hints = pd.read_csv(hints_path, low_memory=False)
        compare_tables(
            hints,
            expected_hints,
            keys=["frame", "source_index", "identity"],
            numeric=["frame", "source_index", "path_segment", "a2_identity_segment",
                     "X", "Y", "Z", "mouth_confidence"],
        )
        actual_resets = json.loads(reset_path.read_text(encoding="utf-8"))
        if actual_resets != expected_resets:
            raise ValueError(f"A5 reset-frame mismatch: {window}")
        if (
            hint_summary["filter_counts"] != expected_audit["filter_counts"]
            or hint_summary["exact_candidate_matches"]
            != expected_audit["exact_candidate_matches"]
            or hint_summary["mapped_segments"] != expected_audit["mapped_segments"]
            or hint_summary["mapping_vote_audit"]
            != expected_audit["mapping_vote_audit"]
        ):
            raise ValueError(f"A5 hint decision audit mismatch: {window}")

        if (
            run_summary["settings"]["frame_start"] != frame_start
            or run_summary["settings"]["frame_end"] != frame_end
            or run_summary["settings"]["geometry_sha256"]
            != run_summary["inputs"]["geometry"]["sha256"]
            or run_summary["settings"]["hints_sha256"] != sha256(hints_path)
            or run_summary["settings"]["hints_summary_sha256"]
            != sha256(hint_summary_path)
            or run_summary["calibration"]["backend"] != "refractive"
            or run_summary["calibration"]["score_unit"] != "mm^2"
        ):
            raise ValueError(f"A5 run control mismatch: {window}")
        deferred = run_summary["settings"]["deferred_tracking_config"]
        expected_deferred = {
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
        if deferred != expected_deferred:
            raise ValueError(f"A5 deferred settings mismatch: {window}")
        if stable_hash(run_summary["settings"]) != run_summary["configuration_hash"]:
            raise ValueError(f"A5 run configuration hash mismatch: {window}")
        for code_path, recorded_hash in run_summary["code_sha256"].items():
            if sha256(Path(code_path)) != recorded_hash:
                raise ValueError(f"A5 code drift: {window}/{code_path}")
        for section in ("inputs", "outputs"):
            for name, item in run_summary[section].items():
                if sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError(f"A5 {section} hash mismatch: {window}/{name}")

        direct_path = Path(run_summary["outputs"]["direct"]["path"])
        direct = pd.read_csv(direct_path, low_memory=False)
        expected_rows = 3 * (frame_end - frame_start + 1)
        if (
            len(direct) != expected_rows
            or direct.duplicated(["frame", "fish_id"]).any()
            or set(direct.fish_id.astype(str)) != {"ID_1", "ID_2", "ID_3"}
            or not direct.groupby("frame").size().eq(3).all()
            or set(direct.frame.astype(int))
            != set(range(frame_start, frame_end + 1))
        ):
            raise ValueError(f"invalid A5 direct grid: {window}")
        geometry_path = Path(run_summary["inputs"]["geometry"]["path"])
        geometry = pd.read_parquet(geometry_path)
        finite_direct = direct[np.isfinite(direct[["X", "Y", "Z"]]).all(axis=1)]
        matched = finite_direct.merge(
            geometry[["frame", "hypothesis_id", "X", "Y", "Z"]],
            on=["frame", "hypothesis_id"],
            how="left",
            suffixes=("_direct", "_geometry"),
            validate="many_to_one",
        )
        delta = np.linalg.norm(
            matched[["X_direct", "Y_direct", "Z_direct"]].to_numpy(float)
            - matched[["X_geometry", "Y_geometry", "Z_geometry"]].to_numpy(float),
            axis=1,
        )
        if len(matched) != len(finite_direct) or not np.all(delta <= 1e-9):
            raise ValueError(f"A5 direct output is not shared geometry: {window}")
        application = pd.read_csv(
            run_summary["outputs"]["hint_application_audit"]["path"],
            low_memory=False,
        )
        if len(application) != len(hints):
            raise ValueError(f"A5 hint application row mismatch: {window}")
        expected_application = hints[
            ["frame", "identity", "source_index", "hypothesis_id"]
        ].rename(columns={"source_index": "hint_source_index"})
        expected_application = expected_application.merge(
            direct[
                ["frame", "fish_id", "source_index", "identity_status"]
            ].rename(
                columns={
                    "fish_id": "identity",
                    "source_index": "direct_source_index",
                }
            ),
            on=["frame", "identity"],
            how="left",
            validate="one_to_one",
        )
        expected_application["honored"] = (
            expected_application.hint_source_index
            == expected_application.direct_source_index
        )
        expected_application = expected_application[
            [
                "frame", "identity", "hint_source_index",
                "direct_source_index", "identity_status", "honored",
                "hypothesis_id",
            ]
        ]
        compare_tables(
            application,
            expected_application,
            keys=["frame", "identity"],
            numeric=["frame", "hint_source_index", "direct_source_index"],
        )
        application_statuses = {
            f"honored={bool(honored)}|status={status}": int(len(rows))
            for (honored, status), rows in application.groupby(
                ["honored", "identity_status"], dropna=False, sort=True
            )
        }
        pooled_application_statuses.update(application_statuses)

        track = scorer.track_metrics(direct, frame_start, frame_end)
        pooled_jumps.extend(track.pop("_jump_values"))
        pooled_residuals.extend(track.pop("_residual_values"))
        window_annotations = annotations[annotations.training_window.eq(window)]
        assignments = scorer.map_female_rows(
            window_annotations,
            direct,
            calibration,
            config_name="A5_frozen",
            window_name=window,
        )
        identity = scorer.summarize_assignments(assignments)
        assignments_all.append(assignments)
        pooled.update(
            {
                "frames": frame_end - frame_start + 1,
                "rows": len(direct),
                "all_three_direct_frames": track["all_three_direct_frames"],
                "unavailable_rows": track["unavailable_rows"],
                "hint_rows": len(hints),
                "hints_honored": int(application.honored.sum()),
                "mapped_segments": hint_summary["mapped_segments"],
                "mapping_vote_segments": len(
                    hint_summary["mapping_vote_audit"]
                ),
                "mapping_vote_rejected": sum(
                    not bool(row["accepted_by_vote"])
                    for row in hint_summary["mapping_vote_audit"]
                ),
                "rejected_overlap_identity_segments": hint_summary["filter_counts"][
                    "rejected_overlap_identity_segments"
                ],
                "rejected_per_frame_non_bijective": hint_summary["filter_counts"][
                    "rejected_per_frame_non_bijective"
                ],
                "female_eligible": identity["eligible_annotations"],
                "female_mapped": identity["mapped_annotations"],
                "female_correct": identity["best_fixed_correct"],
                "female_changes": identity["observed_identity_changes"],
            }
        )
        records[window] = {
            "hint_summary_path": str(hint_summary_path),
            "hint_summary_sha256": sha256(hint_summary_path),
            "run_summary_path": str(run_summary_path),
            "run_summary_sha256": sha256(run_summary_path),
            "a4_summary_sha256": sha256(a4_summary_path),
            "a4_selected_sha256": sha256(a4_path),
            "a2_direct_sha256": sha256(a2_path),
            "hint_thresholds": HINT_CONFIG,
            "filter_counts": hint_summary["filter_counts"],
            "mapping_vote_audit": hint_summary["mapping_vote_audit"],
            "hint_rows": len(hints),
            "hints_honored": int(application.honored.sum()),
            "hint_application_status_counts": application_statuses,
            "forced_reset_frames": actual_resets,
            "track_qc": track,
            "female_identity": identity,
            "wall_seconds": float(
                hint_summary["wall_seconds"] + run_summary["wall_seconds"]
            ),
            "peak_rss_bytes": max(
                int(hint_summary["peak_rss_bytes"]),
                int(run_summary["peak_rss_bytes"]),
            ),
        }

    def dist(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return {
            "n": int(len(array)),
            "median": float(np.median(array)) if len(array) else None,
            "p95": float(np.quantile(array, 0.95)) if len(array) else None,
            "p99": float(np.quantile(array, 0.99)) if len(array) else None,
            "max": float(np.max(array)) if len(array) else None,
        }

    jump_array = np.asarray(pooled_jumps, dtype=float)
    result = {
        "schema_version": 1,
        "status": "training_only_a5_hint_thresholds_audited",
        "scope": "training windows only; validation, held-out, and critical outputs not read",
        "a4_selection": A4_CONFIG,
        "a4_selection_audit": {
            "path": str(A4_AUDIT),
            "sha256": sha256(A4_AUDIT),
        },
        "training_a5_completion": {
            "path": str(A5_COMPLETION),
            "sha256": sha256(A5_COMPLETION),
        },
        "hint_thresholds": HINT_CONFIG,
        "selection_code_sha256": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(BASE_SCORER): sha256(BASE_SCORER),
        },
        "records": records,
        "pooled": {
            **dict(pooled),
            "all_three_direct_fraction": (
                pooled["all_three_direct_frames"] / pooled["frames"]
            ),
            "unavailable_row_fraction": pooled["unavailable_rows"] / pooled["rows"],
            "female_full_eligible_accuracy": (
                pooled["female_correct"] / pooled["female_eligible"]
            ),
            "female_selective_mapped_accuracy": (
                pooled["female_correct"] / pooled["female_mapped"]
            ),
            "hint_row_fraction": pooled["hint_rows"] / pooled["rows"],
            "hint_honored_fraction": (
                pooled["hints_honored"] / pooled["hint_rows"]
            ),
            "hint_application_status_counts": dict(
                sorted(pooled_application_statuses.items())
            ),
            "jump_mm": dist(pooled_jumps),
            "jump_ge_7_5mm_count": int((jump_array >= 7.5).sum()),
            "jump_gt_8mm_count": int((jump_array > 8).sum()),
            "direct_residual_mm2": dist(pooled_residuals),
        },
        "limitations": [
            "A5 hint thresholds are conservative design rules, not a swept optimum.",
            "Only 15 confirmed-female training annotations are available.",
            "Female-only labels cannot expose male-to-male identity swaps.",
            "A4's no-retained-count-mismatch safeguard was critical-informed before freeze; this audit treats it as contained design provenance, not validation evidence.",
        ],
    }
    a3_audit = json.loads(A3_AUDIT.read_text(encoding="utf-8"))
    if a3_audit["selected"] != "A3_a005":
        raise ValueError("A5 comparison is not bound to selected A3")
    a3 = a3_audit["records"]["A3_a005"]["pooled"]
    result["comparison_to_selected_a3"] = {
        "selected_a3": "A3_a005",
        "a3_audit_path": str(A3_AUDIT),
        "a3_audit_sha256": sha256(A3_AUDIT),
        "metrics": {
            "female_best_fixed_correct": {
                "A3": int(a3["female_best_fixed_correct_window_aligned"]),
                "A5": int(result["pooled"]["female_correct"]),
                "eligible": int(result["pooled"]["female_eligible"]),
            },
            "female_identity_changes": {
                "A3": int(a3["observed_female_identity_changes"]),
                "A5": int(result["pooled"]["female_changes"]),
            },
            "all_three_direct_fraction": {
                "A3": float(a3["all_three_direct_fraction"]),
                "A5": float(result["pooled"]["all_three_direct_fraction"]),
            },
            "unavailable_rows": {
                "A3": int(a3["unavailable_rows"]),
                "A5": int(result["pooled"]["unavailable_rows"]),
            },
            "jumps_ge_7_5mm": {
                "A3": int(a3["jump_ge_7_5mm_count"]),
                "A5": int(result["pooled"]["jump_ge_7_5mm_count"]),
            },
            "jumps_gt_8mm": {
                "A3": int(a3["jump_gt_8mm_count"]),
                "A5": int(result["pooled"]["jump_gt_8mm_count"]),
            },
        },
    }
    result["production_decision"] = {
        "freeze_hint_thresholds_for_reproducibility": True,
        "post_freeze_role": "exploratory_comparison_only",
        "retain_as_production_finalist": False,
        "retain_for_full_length_finalist_run": False,
        "reason": (
            "Relative to selected A3 on the same training windows, A5 reduced "
            "fixed-female identity consistency from 15/15 to 6/15, introduced "
            "8 observed female identity changes, reduced all-three direct "
            "coverage from 0.997375 to 0.937582, and increased unavailable "
            "rows from 38 to 2500 without reducing the 7 jumps >=7.5 mm."
        ),
    }
    output_path = OUTPUT_ROOT / "training_a5_hint_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat(assignments_all, ignore_index=True).to_csv(
        OUTPUT_ROOT / "training_a5_female_assignments.csv", index=False
    )
    print(json.dumps({"pooled": result["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
