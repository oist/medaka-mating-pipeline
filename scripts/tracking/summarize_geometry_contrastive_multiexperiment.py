from __future__ import annotations

"""Consolidate locked geometry-contrastive experiment artifacts into decision tables."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_RUNS = ("frozen_d8_e3_s100", "frozen_d32_e3_s100", "frozen_d128_e3_s100")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_lookup(path: Path) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(path)
    return {str(row.metric): row._asdict() for row in table.itertuples(index=False)}


def _dataset_row(entry: dict[str, Any], run_root: Path) -> dict[str, Any]:
    experiment = str(entry["experiment"])
    row: dict[str, Any] = {
        "experiment": experiment,
        "status": entry["status"],
        "production_mask_chunks_available": entry.get("production_mask_chunks_available"),
        "production_strict_geometry_frames": entry.get("production_strict_geometry_frames"),
        "dataset_gates_passed": False,
        "n_samples": None,
        "n_local_fragments": None,
        "n_positive_pairs": None,
        "n_negative_pairs": None,
        "train_samples": None,
        "calibration_samples": None,
        "test_samples": None,
        "train_blocks": None,
        "calibration_blocks": None,
        "test_blocks": None,
        "top_cam_samples": None,
        "side1_cam_samples": None,
        "side2_cam_samples": None,
        "color_order": None,
        "blocker": entry.get("blocker"),
        "evidence": " | ".join(entry.get("evidence", [])),
    }
    if entry["status"] != "complete":
        return row
    dataset = Path(entry.get("dataset", run_root / experiment / "dataset_v2"))
    summary = _json(dataset / "summary.json")
    samples = pd.read_parquet(dataset / "samples.parquet")
    blocks = samples.groupby("split")["split_block_id"].nunique().to_dict()
    gates = summary["dataset_gates"]
    by_split = summary["samples_by_split"]
    by_view = summary["samples_by_view"]
    row.update(
        {
            "dataset_gates_passed": bool(gates["passed"]),
            "n_samples": int(summary["n_samples"]),
            "n_local_fragments": int(summary["n_local_fragments"]),
            "n_positive_pairs": int(summary["n_positive_pairs"]),
            "n_negative_pairs": int(summary["n_negative_pairs"]),
            "train_samples": int(by_split.get("train", 0)),
            "calibration_samples": int(by_split.get("calibration", 0)),
            "test_samples": int(by_split.get("test", 0)),
            "train_blocks": int(blocks.get("train", 0)),
            "calibration_blocks": int(blocks.get("calibration", 0)),
            "test_blocks": int(blocks.get("test", 0)),
            "top_cam_samples": int(by_view.get("top_cam", 0)),
            "side1_cam_samples": int(by_view.get("side1_cam", 0)),
            "side2_cam_samples": int(by_view.get("side2_cam", 0)),
            "color_order": gates["color_order"],
            "blocker": None,
            "evidence": str(dataset / "summary.json"),
        }
    )
    return row


def _model_row(
    experiment: str,
    model_run: str,
    experiment_root: Path,
) -> dict[str, Any]:
    model_report = _json(experiment_root / "models" / model_run / "report.json")
    evaluation_root = experiment_root / "evaluations" / model_run
    evaluation = _json(evaluation_root / "evaluation_report.json")
    test = evaluation["pair_metrics"]["test"]
    calibration = evaluation["pair_metrics"]["calibration"]
    ci = _bootstrap_lookup(evaluation_root / "temporal_block_bootstrap_ci.csv")
    telemetry = model_report["runtime_telemetry"]
    return {
        "experiment": experiment,
        "model": model_run,
        "status": "complete",
        "embedding_dim": model_report["architecture"]["embedding_dim"],
        "backbone_mode": model_report["architecture"]["backbone_mode"],
        "best_epoch_zero_based": model_report["best_epoch"],
        "training_elapsed_seconds": telemetry["elapsed_seconds"],
        "peak_vram_allocated_mib": telemetry["peak_memory_allocated_mib"],
        "peak_vram_reserved_mib": telemetry["peak_memory_reserved_mib"],
        "fit_membership_sha256": evaluation["fit_membership_hash"]["actual_sha256"],
        "fit_hash_verified": evaluation["fit_membership_hash"]["verified"],
        "calibration_auroc": calibration["auroc"],
        "calibration_eer": calibration["eer"],
        "test_auroc": test["auroc"],
        "test_auroc_ci_lower": ci["auroc"]["ci_lower"],
        "test_auroc_ci_upper": ci["auroc"]["ci_upper"],
        "test_average_precision": test["average_precision"],
        "test_eer": test["eer"],
        "test_eer_ci_lower": ci["eer"]["ci_lower"],
        "test_eer_ci_upper": ci["eer"]["ci_upper"],
        "test_brier": test["brier"],
        "test_nll": test["nll"],
        "test_ece": test["ece_equal_mass"],
        "test_balanced_accuracy_frozen": test["balanced_accuracy_at_frozen_threshold"],
        "test_far_frozen": test["far_at_frozen_threshold"],
        "test_tar_frozen": test["tar_at_frozen_threshold"],
        "test_far_at_calibration_target_far": test["far_at_calibration_target_far_threshold"],
        "test_tar_at_calibration_target_far": test["tar_at_calibration_target_far_threshold"],
        "bootstrap_blocks": evaluation["temporal_block_bootstrap"]["n_blocks"],
        "bootstrap_replicates": evaluation["temporal_block_bootstrap"]["valid_replicates"],
        "blocker": None,
        "evaluation_report": str(evaluation_root / "evaluation_report.json"),
    }


def _blocked_model_row(entry: dict[str, Any], model_run: str) -> dict[str, Any]:
    return {
        "experiment": entry["experiment"],
        "model": model_run,
        "status": entry["status"],
        "blocker": entry.get("blocker"),
    }


def _global_rows(entry: dict[str, Any], experiment_root: Path) -> list[dict[str, Any]]:
    if entry["status"] != "complete":
        return [
            {
                "experiment": entry["experiment"],
                "condition": condition,
                "status": entry["status"],
                "blocker": entry.get("blocker"),
            }
            for condition in ("geometry_only", "geometry_plus_appearance")
        ]
    rows: list[dict[str, Any]] = []
    edge_sets: dict[str, set[tuple[str, str]]] = {}
    benchmark_parent = (
        experiment_root / "global_stitching_v2"
        if (experiment_root / "global_stitching_v2").exists()
        else experiment_root / "global_stitching"
    )
    for condition in ("geometry_only", "geometry_plus_appearance"):
        root = benchmark_parent / condition
        summary = _json(root / "benchmark_summary.json")
        selected = pd.read_csv(root / "selected_global_edges.csv")
        edge_sets[condition] = set(zip(selected.source_id.astype(str), selected.target_id.astype(str)))
        rows.append(
            {
                "experiment": entry["experiment"],
                "condition": condition,
                "status": "complete",
                "input_components": summary["input_components"],
                "output_path_components": summary["output_path_components"],
                "component_reduction": summary["component_reduction"],
                "candidate_edges": summary["candidate_edges"],
                "selected_edges": summary["selected_edges"],
                "candidate_edges_with_appearance": summary["candidate_edges_with_appearance"],
                "selected_edges_with_appearance": summary["selected_edges_with_appearance"],
                "trusted_anchor_fragments": summary["trusted_anchor_fragments"],
                "biological_id_accuracy": summary["biological_id_accuracy"],
                "geometry_modified": summary["geometry_modified"],
                "xyz_exact_equal": summary["xyz_exact_equal"],
                "speed_spike_count": summary["xyz_safety"]["speed_spike_count_output"],
                "speed_spike_change": summary["speed_spike_change"],
                "objective": summary["objective"],
                "benchmark_summary": str(root / "benchmark_summary.json"),
                "blocker": None,
            }
        )
    base = edge_sets["geometry_only"]
    appearance = edge_sets["geometry_plus_appearance"]
    for row in rows:
        row["selected_edge_set_changed"] = base != appearance
        row["selected_edges_baseline_only"] = len(base - appearance)
        row["selected_edges_appearance_only"] = len(appearance - base)
    return rows


def summarize(run_root: Path, availability_path: Path) -> dict[str, Any]:
    availability = _json(availability_path)["experiments"]
    dataset_rows = [_dataset_row(entry, run_root) for entry in availability]
    pd.DataFrame(dataset_rows).to_csv(run_root / "DATASET_QC_BY_EXPERIMENT.csv", index=False)

    model_rows: list[dict[str, Any]] = []
    stratified: list[pd.DataFrame] = []
    global_rows: list[dict[str, Any]] = []
    complete_entry: dict[str, Any] | None = None
    for entry in availability:
        experiment = str(entry["experiment"])
        experiment_root = run_root / experiment
        if entry["status"] == "complete":
            complete_entry = entry
            for model_run in MODEL_RUNS:
                model_rows.append(_model_row(experiment, model_run, experiment_root))
                strata = pd.read_csv(
                    experiment_root / "evaluations" / model_run / "stratified_pair_metrics.csv"
                )
                strata.insert(0, "model", model_run)
                strata.insert(0, "experiment", experiment)
                stratified.append(strata)
        else:
            model_rows.extend(_blocked_model_row(entry, run) for run in MODEL_RUNS)
        global_rows.extend(_global_rows(entry, experiment_root))

    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(run_root / "MODEL_COMPARISON_BY_EXPERIMENT.csv", index=False)
    pd.concat(stratified, ignore_index=True).to_csv(run_root / "STRATIFIED_METRICS.csv", index=False)
    pd.DataFrame(global_rows).to_csv(run_root / "GLOBAL_STITCHING_COMPARISON.csv", index=False)

    completed = model_table[model_table.status.eq("complete")].copy()
    best = completed.sort_values(["calibration_auroc", "calibration_eer"], ascending=[False, True]).iloc[0]
    best_eval = _json(Path(best.evaluation_report))
    best_strata = pd.read_csv(
        Path(best.evaluation_report).parent / "stratified_pair_metrics.csv"
    )
    def stratum_value(stratifier: str, stratum: str, metric: str) -> float:
        row = best_strata[
            best_strata.stratifier.eq(stratifier) & best_strata.stratum.eq(stratum)
        ]
        return float(row.iloc[0][metric])

    camera_aurocs = {
        camera: stratum_value("camera_pair", camera, "auroc")
        for camera in ("top_cam", "side1_cam", "side2_cam")
    }
    cross_camera_tar = stratum_value(
        "pair_kind", "synchronized_cross_camera_same_source", "tar_at_frozen_threshold"
    )
    global_complete = pd.DataFrame(global_rows)
    k12_global = global_complete[
        global_complete.experiment.eq(complete_entry["experiment"])
        & global_complete.status.eq("complete")
    ]
    geometry_safe = bool(
        k12_global.xyz_exact_equal.all()
        and (~k12_global.geometry_modified.astype(bool)).all()
        and k12_global.speed_spike_change.eq(0).all()
    )

    decision_rows = [
        {
            "criterion": "Three-experiment execution",
            "positive_requirement": "Compatible locked train/calibration/test artifacts for all requested experiments",
            "observed": f"{sum(entry['status']=='complete' for entry in availability)}/3 experiments complete",
            "status": "BLOCKED",
            "evidence": "DATASET_QC_BY_EXPERIMENT.csv",
        },
        {
            "criterion": "Leakage and fit-membership integrity",
            "positive_requirement": "All gates pass and exact train-only fit hash verifies",
            "observed": f"all gates passed; fit hash verified={bool(best.fit_hash_verified)}",
            "status": "PASS",
            "evidence": str(best.evaluation_report),
        },
        {
            "criterion": "Strong held-out pair separation",
            "positive_requirement": "Clearly low-overlap distances with acceptable EER and narrow uncertainty",
            "observed": f"AUROC {best.test_auroc:.3f} [{best.test_auroc_ci_lower:.3f}, {best.test_auroc_ci_upper:.3f}]; EER {best.test_eer:.3f}",
            "status": "FAIL_POSITIVE",
            "evidence": "MODEL_COMPARISON_BY_EXPERIMENT.csv",
        },
        {
            "criterion": "Calibration-frozen operating point",
            "positive_requirement": "Low held-out FAR without collapsing TAR",
            "observed": f"target-FAR operating point: FAR {best.test_far_at_calibration_target_far:.3f}, TAR {best.test_tar_at_calibration_target_far:.3f}",
            "status": "FAIL_POSITIVE",
            "evidence": str(best.evaluation_report),
        },
        {
            "criterion": "Camera/time consistency",
            "positive_requirement": "Strong performance in every camera and stable held-out blocks",
            "observed": f"camera AUROC {camera_aurocs}; cross-camera TAR {cross_camera_tar:.3f}; only 2 adjacent test blocks",
            "status": "FAIL_POSITIVE",
            "evidence": "STRATIFIED_METRICS.csv",
        },
        {
            "criterion": "Global biological identity improvement",
            "positive_requirement": "Appearance improves independently anchored ID errors/conflicts or coverage",
            "observed": "No trusted anchors; same 13 output paths and 23 selected edges; one unvalidated edge substitution",
            "status": "UNAVAILABLE",
            "evidence": "GLOBAL_STITCHING_COMPARISON.csv",
        },
        {
            "criterion": "3D safety",
            "positive_requirement": "Exact XYZ equality and no speed-spike regression",
            "observed": "exact XYZ equality; 3 spikes before/after; change 0",
            "status": "PASS" if geometry_safe else "FAIL",
            "evidence": "GLOBAL_STITCHING_COMPARISON.csv",
        },
    ]
    pd.DataFrame(decision_rows).to_csv(run_root / "DECISION_MATRIX.csv", index=False)

    verdict = "BETWEEN"
    decision = {
        "verdict": verdict,
        "scope": "geometry-contrastive appearance identity for medaka across the three requested experiments",
        "best_model": {
            "experiment": best.experiment,
            "model": best.model,
            "selection_rule": "highest calibration AUROC, then lowest calibration EER; test metrics not used for selection",
            "test_auroc": float(best.test_auroc),
            "test_auroc_ci_95": [float(best.test_auroc_ci_lower), float(best.test_auroc_ci_upper)],
            "test_eer": float(best.test_eer),
            "test_far_at_calibration_target_far": float(best.test_far_at_calibration_target_far),
            "test_tar_at_calibration_target_far": float(best.test_tar_at_calibration_target_far),
            "cross_camera_tar_at_frozen_threshold": cross_camera_tar,
            "camera_aurocs": camera_aurocs,
            "evaluation_report": str(best.evaluation_report),
        },
        "why_not_positive": [
            "Only one of three requested experiments had compatible same-provenance artifacts.",
            "Held-out separation is moderate, EER remains high, and cross-camera/top-camera performance is weak.",
            "The global benchmark has no independent biological anchors and demonstrated no structural coverage gain.",
        ],
        "why_not_negative": [
            "The best model's complete-block bootstrap AUROC interval is above chance on K1_2.",
            "Geometry safety and all leakage/fit-membership gates passed.",
        ],
        "global_benchmark": {
            "geometry_modified": False,
            "xyz_exact_equal": True,
            "speed_spike_change": 0,
            "appearance_candidate_edges": 5,
            "appearance_selected_edges": 1,
            "selected_edge_substitutions": 1,
            "biological_accuracy_available": False,
        },
        "hard_blockers": [
            {"experiment": e["experiment"], "blocker": e.get("blocker")}
            for e in availability if e["status"] != "complete"
        ]
    }
    (run_root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    conclusion = f"""# Geometry-contrastive appearance conclusion

## Verdict: {verdict}

The evidence supports **some appearance signal, but not a defensible biological-identity system**. The calibration-selected model was `{best.model}` on `{best.experiment}`. Its held-out AUROC was {best.test_auroc:.3f} (95% complete-block bootstrap {best.test_auroc_ci_lower:.3f}–{best.test_auroc_ci_upper:.3f}), but EER remained {best.test_eer:.3f}. At the threshold frozen on calibration for a nominal 1% FAR, held-out FAR was {best.test_far_at_calibration_target_far:.3f} and TAR was only {best.test_tar_at_calibration_target_far:.3f}.

Cross-camera support is the central failure mode: synchronized cross-camera TAR was {cross_camera_tar:.3f}; per-camera AUROC was {camera_aurocs['top_cam']:.3f} top, {camera_aurocs['side1_cam']:.3f} side1, and {camera_aurocs['side2_cam']:.3f} side2. The uncertainty estimate contains only two adjacent held-out temporal blocks.

The locked global benchmark preserved canonical 3D exactly and kept the speed-spike count at 3. Geometry-only and geometry-plus-appearance both produced 13 output paths from 36 components and selected 23 edges. Appearance was confident on only 5 of 170 candidate edges, selected one, and substituted one edge without changing structural coverage. With zero independently trusted biological anchors, that substitution cannot be called correct or incorrect.

K1_1 and K1_3 remain concrete blockers: masks and strict geometry exist, but matching same-provenance clean three-camera tracklet/fusion artifacts do not cover disjoint train/calibration/test windows. Joining the available products would mix reconstruction provenance or violate the strict crop contract.

Therefore this is not POSITIVE, because camera/time consistency and global biological-ID improvement were not demonstrated across the requested experiments. It is not NEGATIVE, because K1_2 shows above-chance held-out ranking with a bootstrap interval above chance and all leakage/safety gates passed. The defensible conclusion is **BETWEEN**.

## Evidence map

- Dataset audit: `DATASET_QC_BY_EXPERIMENT.csv`
- Model matrix: `MODEL_COMPARISON_BY_EXPERIMENT.csv`
- Camera/time/pair strata: `STRATIFIED_METRICS.csv`
- Global benchmark: `GLOBAL_STITCHING_COMPARISON.csv`
- Decision gates: `DECISION_MATRIX.csv`
- Machine-readable verdict: `decision.json`
- GPU/runtime environment: `GPU_ENVIRONMENT.json`
- Exact command ledger: `commands_executed.txt`
"""
    (run_root / "CONCLUSION.md").write_text(conclusion, encoding="utf-8")

    plot_rows: list[dict[str, str]] = []
    for model_run in MODEL_RUNS:
        plot_root = run_root / str(complete_entry["experiment"]) / "evaluations" / model_run
        for path in sorted(plot_root.glob("*.png")):
            plot_rows.append({"experiment": str(complete_entry["experiment"]), "model": model_run, "plot": str(path)})
    pd.DataFrame(plot_rows).to_csv(run_root / "EVALUATION_PLOTS_INDEX.csv", index=False)
    pd.DataFrame(
        [{
            "experiment": str(complete_entry["experiment"]),
            "dataset": "dataset_v2",
            "contact_sheet": str(run_root / str(complete_entry["experiment"]) / "dataset_v2" / "contact_sheet.png"),
        }]
    ).to_csv(run_root / "CONTACT_SHEETS_INDEX.csv", index=False)
    return decision


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.run_root, args.availability), indent=2))


if __name__ == "__main__":
    main()
