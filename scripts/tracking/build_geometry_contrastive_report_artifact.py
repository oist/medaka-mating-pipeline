from __future__ import annotations

"""Build the canonical Data Analytics report artifact for this evidence run."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _records(path: Path, *, query: str | None = None) -> list[dict[str, Any]]:
    table = pd.read_csv(path)
    if query:
        table = table.query(query)
    return table.astype(object).where(pd.notna(table), None).to_dict("records")


def build(run_root: Path) -> dict[str, Any]:
    decision = json.loads((run_root / "decision.json").read_text(encoding="utf-8"))
    models = pd.DataFrame(_records(run_root / "MODEL_COMPARISON_BY_EXPERIMENT.csv"))
    models = models[models.status.eq("complete")].copy()
    models["model_label"] = models.embedding_dim.map(lambda value: f"{int(float(value))}-D")
    model_rows = models.astype(object).where(pd.notna(models), None).to_dict("records")

    best = decision["best_model"]
    camera_rows = [
        {"camera": camera.replace("_cam", ""), "auroc": value}
        for camera, value in best["camera_aurocs"].items()
    ]
    summary_row = [{
        "experiment_coverage": 1 / 3,
        "test_auroc": best["test_auroc"],
        "test_eer": best["test_eer"],
        "cross_camera_tar": best["cross_camera_tar_at_frozen_threshold"],
        "xyz_exact": 1.0,
    }]
    generated_at = datetime.now(timezone.utc).isoformat()
    def csv_source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
        return {
            "id": source_id,
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": description,
                "sql": f"SELECT * FROM read_csv_auto('{path}');",
                "tables_used": [path],
            },
        }

    sources = [
        {"id": "decision", "label": "Machine-readable verdict", "path": "decision.json"},
        csv_source("dataset_qc", "Dataset QC by experiment", "DATASET_QC_BY_EXPERIMENT.csv", "Read the locked per-experiment dataset audit."),
        csv_source("model_comparison", "Frozen model comparison", "MODEL_COMPARISON_BY_EXPERIMENT.csv", "Read the frozen model matrix and held-out metrics."),
        csv_source("stratified_metrics", "Pair, camera, and time strata", "STRATIFIED_METRICS.csv", "Read pair-kind, camera, and temporal-block evaluation strata."),
        csv_source("global_stitching", "Locked global-stitching comparison", "GLOBAL_STITCHING_COMPARISON.csv", "Read the geometry-only and geometry-plus-appearance benchmark results."),
        csv_source("decision_matrix", "Decision criteria", "DECISION_MATRIX.csv", "Read the evidence-to-verdict decision gates."),
        {"id": "gpu_environment", "label": "CUDA and software environment", "path": "GPU_ENVIRONMENT.json"},
        {"id": "commands", "label": "Exact command ledger", "path": "commands_executed.txt"},
    ]
    artifact: dict[str, Any] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Geometry-contrastive fish identity: evidence verdict",
            "description": "Technical evaluation across the requested medaka experiments, with leakage-safe embedding metrics and a geometry-preserving global-stitching benchmark.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "experiment_coverage",
                    "description": "Requested experiments with a complete compatible train/calibration/test evidence chain.",
                    "dataset": "summary",
                    "sourceId": "dataset_qc",
                    "metrics": [{
                        "label": "Experiment coverage",
                        "field": "experiment_coverage",
                        "format": "percent",
                        "definition": "Completed compatible evidence chains divided by the three requested experiments.",
                    }],
                },
                {
                    "id": "best_auroc",
                    "description": "Held-out pair-ranking discrimination for the calibration-selected model.",
                    "dataset": "summary",
                    "sourceId": "model_comparison",
                    "metrics": [{
                        "label": "Best held-out AUROC",
                        "field": "test_auroc",
                        "format": "number",
                        "definition": "AUROC over all locked K1_2 test must-link and cannot-link pairs; model selected on calibration only.",
                    }],
                },
                {
                    "id": "best_eer",
                    "description": "Held-out equal-error rate; lower is better.",
                    "dataset": "summary",
                    "sourceId": "model_comparison",
                    "metrics": [{
                        "label": "Held-out EER",
                        "field": "test_eer",
                        "format": "percent",
                        "definition": "Descriptive test equal-error rate for the calibration-selected 128-D model.",
                    }],
                },
                {
                    "id": "cross_camera_tar",
                    "description": "Recall of synchronized cross-camera must-links at the calibration-frozen threshold.",
                    "dataset": "summary",
                    "sourceId": "stratified_metrics",
                    "metrics": [{
                        "label": "Cross-camera TAR",
                        "field": "cross_camera_tar",
                        "format": "percent",
                        "definition": "Fraction of synchronized same-source, cross-camera positive pairs accepted at the calibration-frozen EER threshold.",
                    }],
                },
            ],
            "charts": [
                {
                    "id": "model_capacity",
                    "title": "More embedding capacity helps, but does not solve identity",
                    "subtitle": "Held-out AUROC rises from 0.679 to 0.741 under an identical frozen-backbone budget.",
                    "type": "bar",
                    "dataset": "models",
                    "sourceId": "model_comparison",
                    "encodings": {
                        "x": {"field": "model_label", "type": "ordinal", "label": "Embedding dimension"},
                        "y": {"field": "test_auroc", "type": "quantitative", "label": "Held-out AUROC", "format": "number"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "camera_gap",
                    "title": "Top-camera appearance is the weakest view",
                    "subtitle": "The 128-D model is materially less discriminative from above than from either side view.",
                    "type": "bar",
                    "dataset": "camera_metrics",
                    "sourceId": "stratified_metrics",
                    "encodings": {
                        "x": {"field": "camera", "type": "ordinal", "label": "Camera"},
                        "y": {"field": "auroc", "type": "quantitative", "label": "Held-out AUROC", "format": "number"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "experiment_status",
                    "title": "Only K1_2 has a complete compatible evidence chain",
                    "dataset": "dataset_qc",
                    "sourceId": "dataset_qc",
                    "defaultSort": {"field": "experiment", "direction": "asc"},
                    "columns": [
                        {"field": "experiment", "label": "Experiment", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "n_samples", "label": "Strict crops", "format": "number"},
                        {"field": "test_blocks", "label": "Test blocks", "format": "number"},
                        {"field": "blocker", "label": "Blocker", "type": "text"},
                    ],
                },
                {
                    "id": "model_table",
                    "title": "Frozen model matrix on the locked K1_2 split",
                    "dataset": "models",
                    "sourceId": "model_comparison",
                    "defaultSort": {"field": "test_auroc", "direction": "desc"},
                    "columns": [
                        {"field": "model_label", "label": "Embedding", "type": "text"},
                        {"field": "test_auroc", "label": "Test AUROC", "format": "number"},
                        {"field": "test_eer", "label": "Test EER", "format": "percent"},
                        {"field": "test_far_at_calibration_target_far", "label": "Test FAR at target", "format": "percent"},
                        {"field": "test_tar_at_calibration_target_far", "label": "Test TAR at target", "format": "percent"},
                        {"field": "peak_vram_allocated_mib", "label": "Peak VRAM, MiB", "format": "number"},
                    ],
                },
                {
                    "id": "global_table",
                    "title": "Appearance changes one edge, not structural coverage",
                    "dataset": "global_complete",
                    "sourceId": "global_stitching",
                    "defaultSort": {"field": "condition", "direction": "asc"},
                    "columns": [
                        {"field": "condition", "label": "Condition", "type": "text"},
                        {"field": "output_path_components", "label": "Output paths", "format": "number"},
                        {"field": "selected_edges", "label": "Selected edges", "format": "number"},
                        {"field": "selected_edges_with_appearance", "label": "Appearance edges", "format": "number"},
                        {"field": "xyz_exact_equal", "label": "XYZ exact", "type": "text"},
                        {"field": "speed_spike_change", "label": "Spike change", "format": "number"},
                    ],
                },
                {
                    "id": "decision_table",
                    "title": "Positive criteria fail despite clean safety gates",
                    "dataset": "decision_matrix",
                    "sourceId": "decision_matrix",
                    "defaultSort": {"field": "criterion", "direction": "asc"},
                    "columns": [
                        {"field": "criterion", "label": "Criterion", "type": "text"},
                        {"field": "observed", "label": "Observed", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Geometry-contrastive fish identity: evidence verdict"},
                {
                    "id": "verdict",
                    "type": "markdown",
                    "body": "## Verdict: BETWEEN\n\nK1_2 contains reproducible above-chance appearance signal, but the evidence does **not** establish robust biological identity. K1_1 and K1_3 cannot be evaluated without mixing provenance or violating the strict crop contract. The best model remains weak across cameras and the global benchmark has no trusted biological anchors.",
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["experiment_coverage", "best_auroc", "best_eer", "cross_camera_tar"]},
                {"id": "coverage_heading", "type": "markdown", "body": "## Evidence coverage is the first hard limit\n\nMasks and production geometry exist for all three experiments, but only K1_2 has matching tracklet-first, disposition, and silhouette-fusion provenance spanning disjoint splits."},
                {"id": "experiment_table", "type": "table", "tableId": "experiment_status", "layout": "full"},
                {"id": "model_heading", "type": "markdown", "body": "## Capacity improves ranking, not operational reliability\n\nAll three models use the same pretrained frozen ResNet18, seed, balanced pair sampler, CUDA budget, train-only fit hash, and calibration-frozen thresholds. Test labels do not select the model or operating point."},
                {"id": "capacity_chart", "type": "chart", "chartId": "model_capacity", "layout": "full"},
                {"id": "model_detail", "type": "table", "tableId": "model_table", "layout": "full"},
                {"id": "camera_heading", "type": "markdown", "body": "## Cross-view generalization is the central model failure\n\nThe selected 128-D model performs moderately within side views, but top-view ranking and synchronized cross-camera acceptance remain weak. Two adjacent held-out blocks also differ materially, limiting uncertainty claims."},
                {"id": "camera_chart", "type": "chart", "chartId": "camera_gap", "layout": "full"},
                {"id": "global_heading", "type": "markdown", "body": "## Global stitching is geometry-safe but biologically unvalidated\n\nAppearance is confident for only 5 of 170 candidate edges. It substitutes one selected edge while leaving path count, selected-edge count, canonical XYZ, and speed spikes unchanged. With no independently trusted anchors, the substitution is neither a confirmed correction nor a confirmed error."},
                {"id": "global_detail", "type": "table", "tableId": "global_table", "layout": "full"},
                {"id": "decision_heading", "type": "markdown", "body": "## Decision logic\n\nThe outcome is not POSITIVE because the requested cross-experiment, camera/time-consistent, anchored biological-ID improvement was not demonstrated. It is not NEGATIVE because K1_2 ranks above chance with a complete-block confidence interval above chance and all leakage and 3D-safety gates pass."},
                {"id": "decision_detail", "type": "table", "tableId": "decision_table", "layout": "full"},
                {
                    "id": "next_step",
                    "type": "markdown",
                    "body": "## What would change the verdict\n\nCreate same-provenance clean three-camera tracklet/fusion artifacts for K1_1 and K1_3, define several genuinely separated temporal blocks per split, and add independently reviewed biological anchors around crossings and occlusions. Re-run the frozen matrix first; consider fine-tuning only after cross-camera positives and anchor coverage are sufficient. A POSITIVE verdict then requires consistent camera/time performance and fewer anchored global-ID errors without any 3D regression.",
                },
                {
                    "id": "reproducibility",
                    "type": "markdown",
                    "body": "## Reproducibility and limitations\n\nThe artifact is a bounded snapshot of the CSV/JSON evidence. Exact commands, environment versions, CUDA availability, fit hashes, temporal-block bootstrap outputs, plots, contact sheet, and per-run telemetry remain in the run directory. AP and precision are optimistic because test pairs are 92.98% positive; AUROC, EER, FAR/TAR, camera strata, and global safety therefore drive the conclusion.",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "datasets": {
                "summary": summary_row,
                "dataset_qc": _records(run_root / "DATASET_QC_BY_EXPERIMENT.csv"),
                "models": model_rows,
                "camera_metrics": camera_rows,
                "global_complete": _records(run_root / "GLOBAL_STITCHING_COMPARISON.csv", query="status == 'complete'"),
                "decision_matrix": _records(run_root / "DECISION_MATRIX.csv"),
            },
            "accessIssues": [
                {"id": "k1_1_inputs", "dataset": "dataset_qc", "message": decision["hard_blockers"][0]["blocker"]},
                {"id": "k1_3_inputs", "dataset": "dataset_qc", "message": decision["hard_blockers"][1]["blocker"]},
            ],
        },
        "sources": sources,
    }
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = build(args.run_root)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "blocks": len(artifact["manifest"]["blocks"])}, indent=2))


if __name__ == "__main__":
    main()
