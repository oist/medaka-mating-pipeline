from __future__ import annotations

"""Build the canonical Data Analytics artifact for the transition benchmark."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "production_polynomial": "Production polynomial",
    "projective_dlt": "Projective DLT",
    "flat_refractive": "Flat refractive",
}
CAMERA_LABELS = {
    "cam0": "Top camera missing",
    "cam1": "Side 1 missing",
    "cam2": "Side 2 missing",
}


def percentile(series: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().quantile(quantile))


def build_report_data(benchmark_dir: Path) -> dict[str, list[dict[str, object]]]:
    events = pd.read_csv(benchmark_dir / "natural_transition_events.csv")
    ablation = pd.read_csv(benchmark_dir / "heldout_three_view_ablation.csv")
    rigid = pd.read_csv(
        benchmark_dir / "rigid_holdout" / "heldout_rigid_summary.csv"
    )
    weights = json.loads(
        (benchmark_dir / "camera_shift_weights.json").read_text(encoding="utf-8")
    )["models"]

    isolated = events.loc[
        events["split"].eq("holdout") & events["isolated"].astype(bool)
    ].copy()
    natural_model = []
    natural_camera = []
    for model, model_group in isolated.groupby("model"):
        for variant in ("raw", "corrected"):
            residual = model_group[f"{variant}_peak_trajectory_residual_mm"]
            step = model_group[f"{variant}_peak_step_excess_mm"]
            velocity = pd.concat(
                [
                    model_group[f"{variant}_entry_velocity_discontinuity_mm"],
                    model_group[f"{variant}_exit_velocity_discontinuity_mm"],
                ],
                ignore_index=True,
            )
            natural_model.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "variant": variant,
                    "variant_label": "Raw" if variant == "raw" else "Affine corrected",
                    "event_n": int(len(model_group)),
                    "trajectory_median_mm": float(residual.median()),
                    "trajectory_p95_mm": percentile(residual, 0.95),
                    "step_excess_p95_mm": percentile(step, 0.95),
                    "velocity_discontinuity_p95_mm": percentile(velocity, 0.95),
                }
            )
        for missing_view, camera_group in model_group.groupby("missing_view"):
            raw_p95 = percentile(camera_group["raw_peak_trajectory_residual_mm"], 0.95)
            corrected_p95 = percentile(
                camera_group["corrected_peak_trajectory_residual_mm"], 0.95
            )
            raw_median = float(camera_group["raw_peak_trajectory_residual_mm"].median())
            corrected_median = float(
                camera_group["corrected_peak_trajectory_residual_mm"].median()
            )
            natural_camera.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "missing_view": missing_view,
                    "missing_view_label": CAMERA_LABELS[missing_view],
                    "event_n": int(len(camera_group)),
                    "natural_raw_median_mm": raw_median,
                    "natural_corrected_median_mm": corrected_median,
                    "natural_raw_p95_mm": raw_p95,
                    "natural_corrected_p95_mm": corrected_p95,
                    "natural_p95_change_percent": 100.0 * (corrected_p95 / raw_p95 - 1.0),
                }
            )

    rigid_three = rigid.loc[rigid["variant"].eq("three_view")].copy()
    rigid_two = rigid.loc[rigid["variant"].ne("three_view")].copy()
    rigid_two["model_label"] = rigid_two["model"].map(MODEL_LABELS)
    rigid_two["missing_view_label"] = rigid_two["missing_view"].map(CAMERA_LABELS)
    rigid_raw = rigid_two.loc[rigid_two["variant"].eq("raw")].copy()

    ablation_index = ablation.set_index(["model", "missing_view", "variant"])
    rigid_index = rigid_two.set_index(["model", "missing_view", "variant"])
    detail_rows = []
    for row in natural_camera:
        model = str(row["model"])
        camera = str(row["missing_view"])
        raw_rigid = rigid_index.loc[(model, camera, "raw")]
        corrected_rigid = rigid_index.loc[(model, camera, "affine_corrected")]
        raw_ablation = ablation_index.loc[(model, camera, "raw")]
        corrected_ablation = ablation_index.loc[(model, camera, "affine_corrected")]
        coefficient = np.asarray(weights[model][camera]["coefficients"], dtype=float)
        detail_rows.append(
            {
                **row,
                "mean_shift_norm_mm": float(np.linalg.norm(coefficient[0])),
                "ablation_raw_p95_mm": float(raw_ablation["p95"]),
                "ablation_corrected_p95_mm": float(corrected_ablation["p95"]),
                "rigid_raw_kabsch_rms_mm": float(raw_rigid["kabsch_rms_mm"]),
                "rigid_corrected_kabsch_rms_mm": float(
                    corrected_rigid["kabsch_rms_mm"]
                ),
                "rigid_kabsch_change_percent": 100.0
                * (
                    float(corrected_rigid["kabsch_rms_mm"])
                    / float(raw_rigid["kabsch_rms_mm"])
                    - 1.0
                ),
            }
        )

    decision_rows = []
    natural_by_model = pd.DataFrame(natural_model)
    for model in MODEL_LABELS:
        natural_raw = natural_by_model.loc[
            natural_by_model["model"].eq(model)
            & natural_by_model["variant"].eq("raw")
        ].iloc[0]
        natural_corrected = natural_by_model.loc[
            natural_by_model["model"].eq(model)
            & natural_by_model["variant"].eq("corrected")
        ].iloc[0]
        rigid_full = rigid_three.loc[rigid_three["model"].eq(model)].iloc[0]
        raw_rigid_model = rigid_raw.loc[rigid_raw["model"].eq(model)]
        raw_ablation_model = ablation.loc[
            ablation["model"].eq(model) & ablation["variant"].eq("raw")
        ]
        decision_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "natural_event_n": int(natural_raw["event_n"]),
                "natural_raw_median_mm": float(natural_raw["trajectory_median_mm"]),
                "natural_raw_p95_mm": float(natural_raw["trajectory_p95_mm"]),
                "natural_corrected_p95_mm": float(
                    natural_corrected["trajectory_p95_mm"]
                ),
                "natural_correction_change_percent": 100.0
                * (
                    float(natural_corrected["trajectory_p95_mm"])
                    / float(natural_raw["trajectory_p95_mm"])
                    - 1.0
                ),
                "three_view_rigid_kabsch_rms_mm": float(rigid_full["kabsch_rms_mm"]),
                "two_view_rigid_kabsch_macro_mm": float(
                    raw_rigid_model["kabsch_rms_mm"].mean()
                ),
                "ablation_p95_macro_mm": float(raw_ablation_model["p95"].mean()),
            }
        )

    return {
        "natural_model": natural_model,
        "natural_camera": natural_camera,
        "rigid_raw": rigid_raw[
            [
                "model", "model_label", "missing_view", "missing_view_label",
                "n_poses", "n_markers", "kabsch_rms_mm", "kabsch_p95_mm",
                "pairwise_mae_mm", "pairwise_p95_mm", "signed_scale_bias_mm",
            ]
        ].to_dict("records"),
        "camera_detail": detail_rows,
        "decision": decision_rows,
    }


def source_objects(generated_at: str) -> list[dict[str, object]]:
    return [
        {
            "id": "round7_transition",
            "label": "Round7 K_1_2 transition benchmark",
            "query": {
                "engine": "Python",
                "id": "view_transition_benchmark_v1",
                "language": "python",
                "sql": (
                    "SELECT * FROM natural_transition_events "
                    "WHERE split = 'holdout' AND isolated = TRUE"
                ),
                "executed_at": generated_at,
                "description": (
                    "Frozen-assignment 2-view/3-view reconstruction comparison on the "
                    "full 348,600-frame round7 K_1_2 recording."
                ),
                "tables_used": [
                    "round7_260704_current_selector_position_v1_best.json",
                    "top_cam.slp", "side1_cam.slp", "side2_cam.slp",
                    "natural_transition_events.csv",
                    "heldout_three_view_ablation.csv",
                ],
                "filters": [
                    "Identity and cross-camera assignments frozen to current_selector_position_v1",
                    "Primary natural cohort is isolated 3-view to 2-view to 3-view events",
                    "Complete event window must lie in held-out 6,000-frame time blocks",
                    "Three-view anchors must pass production reprojection, tank-bound, and slot-parity checks",
                ],
                "metric_definitions": [
                    "Trajectory residual is Euclidean distance to constant-velocity interpolation between bracketing three-view anchors, in mm",
                    "Ablation error is same-frame Euclidean difference between masked two-view and full three-view reconstruction for one backend, in mm",
                    "Affine correction is trained on non-holdout clean three-view rows after counterfactual camera masking",
                ],
            },
        },
        {
            "id": "rigid_holdout",
            "label": "Untouched 13-pose rigid-object holdout",
            "query": {
                "engine": "Python",
                "id": "transition_weights_rigid_holdout_v1",
                "language": "python",
                "sql": "SELECT * FROM heldout_rigid_summary WHERE split = 'holdout'",
                "executed_at": generated_at,
                "description": (
                    "Held-out 12-ball shape and scale evaluation for three-view and "
                    "camera-masked two-view reconstruction."
                ),
                "tables_used": [
                    "rigid_object_image_coords.csv",
                    "rigid_object_geometry.csv",
                    "ball_heldout_poses.txt",
                    "heldout_rigid_summary.csv",
                ],
                "filters": ["13 untouched poses", "136 marker observations"],
                "metric_definitions": [
                    "Kabsch RMS is residual shape error after one rigid pose fit per placement, in mm",
                    "Pairwise MAE compares predicted and known marker-pair distances without pose fitting, in mm",
                ],
            },
        },
        {
            "id": "benchmark_code",
            "label": "Transition benchmark implementation",
            "query": {
                "engine": "Python",
                "id": "benchmark_view_transition_shifts_py",
                "language": "python",
                "sql": (
                    "SELECT model, missing_view, correction FROM camera_shift_weights"
                ),
                "executed_at": generated_at,
                "description": "Reproducible benchmark, correction fitting, and rigid holdout evaluation.",
                "tables_used": [
                    "benchmark_view_transition_shifts.py",
                    "evaluate_transition_weights_rigid_object.py",
                ],
            },
        },
        {
            "id": "combined_validation",
            "label": "Camera-specific transition and rigid validation join",
            "query": {
                "engine": "Python",
                "id": "camera_correction_validation_v1",
                "language": "python",
                "sql": (
                    "SELECT n.model, n.missing_view, n.transition_metrics, "
                    "r.rigid_metrics, w.correction FROM natural_transition_events AS n "
                    "JOIN heldout_rigid_summary AS r USING (model, missing_view) "
                    "JOIN camera_shift_weights AS w USING (model, missing_view) "
                    "WHERE n.split = 'holdout' AND n.isolated = TRUE"
                ),
                "executed_at": generated_at,
                "description": (
                    "Camera-level join of held-out fish continuity, rigid-object shape, "
                    "and learned affine correction diagnostics."
                ),
                "tables_used": [
                    "natural_transition_events.csv",
                    "heldout_rigid_summary.csv",
                    "camera_shift_weights.json",
                ],
            },
        },
    ]


def build_artifact(data: dict[str, list[dict[str, object]]], generated_at: str) -> dict:
    decisions = {row["model"]: row for row in data["decision"]}
    dlt = decisions["projective_dlt"]
    production = decisions["production_polynomial"]
    refractive = decisions["flat_refractive"]
    transition_reduction = (
        dlt["natural_raw_median_mm"] / production["natural_raw_median_mm"] - 1.0
    )
    rigid_reduction = (
        dlt["three_view_rigid_kabsch_rms_mm"]
        / production["three_view_rigid_kabsch_rms_mm"]
        - 1.0
    )
    sources = source_objects(generated_at)
    title = "2-view ↔ 3-view transition benchmark"
    headline = [
        {
            "dlt_transition_median_mm": dlt["natural_raw_median_mm"],
            "transition_vs_production": transition_reduction,
            "dlt_rigid_rms_mm": dlt["three_view_rigid_kabsch_rms_mm"],
            "rigid_vs_production": rigid_reduction,
            "isolated_event_n": dlt["natural_event_n"],
        }
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Technical comparison of production polynomial completion, projective "
                "DLT, and flat refractive reconstruction at missing-camera transitions."
            ),
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "dlt_transition_card",
                    "dataset": "headline",
                    "sourceId": "round7_transition",
                    "description": "Median trajectory residual for held-out isolated transitions.",
                    "metrics": [
                        {"label": "DLT median", "field": "dlt_transition_median_mm", "format": "number"},
                        {"label": "vs production", "field": "transition_vs_production", "format": "percent", "signed": True},
                    ],
                },
                {
                    "id": "dlt_rigid_card",
                    "dataset": "headline",
                    "sourceId": "rigid_holdout",
                    "description": "Three-view Kabsch RMS on the untouched rigid-object holdout.",
                    "metrics": [
                        {"label": "DLT rigid RMS", "field": "dlt_rigid_rms_mm", "format": "number"},
                        {"label": "vs production", "field": "rigid_vs_production", "format": "percent", "signed": True},
                    ],
                },
                {
                    "id": "event_count_card",
                    "dataset": "headline",
                    "sourceId": "round7_transition",
                    "description": "Held-out isolated 3-view to 2-view to 3-view events per model.",
                    "metrics": [
                        {"label": "Isolated events", "field": "isolated_event_n", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "natural_model_chart",
                    "title": "Isolated-transition trajectory residual",
                    "subtitle": "Held-out one-frame transitions; p95 distance to bracketing constant-velocity interpolation, mm",
                    "type": "bar",
                    "dataset": "natural_model",
                    "sourceId": "round7_transition",
                    "encodings": {
                        "x": {"field": "model_label", "type": "nominal", "label": "Geometry model"},
                        "y": {"field": "trajectory_p95_mm", "type": "quantitative", "label": "p95 trajectory residual", "unit": "mm"},
                        "color": {"field": "variant_label", "type": "nominal", "label": "Correction"},
                        "tooltip": [
                            {"field": "event_n", "type": "quantitative", "label": "Events"},
                            {"field": "trajectory_median_mm", "type": "quantitative", "label": "Median", "unit": "mm"},
                            {"field": "step_excess_p95_mm", "type": "quantitative", "label": "Step excess p95", "unit": "mm"},
                        ],
                    },
                    "yAxisTitle": "p95 trajectory residual (mm)",
                    "layout": "full",
                },
                {
                    "id": "rigid_two_view_chart",
                    "title": "Held-out rigid-object 2-view shape error",
                    "subtitle": "Raw variable-view reconstruction; 13 untouched poses and 136 marker observations",
                    "type": "bar",
                    "dataset": "rigid_raw",
                    "sourceId": "rigid_holdout",
                    "encodings": {
                        "x": {"field": "missing_view_label", "type": "nominal", "label": "Missing camera"},
                        "y": {"field": "kabsch_rms_mm", "type": "quantitative", "label": "Kabsch RMS", "unit": "mm"},
                        "color": {"field": "model_label", "type": "nominal", "label": "Geometry model"},
                        "tooltip": [
                            {"field": "pairwise_mae_mm", "type": "quantitative", "label": "Pairwise MAE", "unit": "mm"},
                            {"field": "kabsch_p95_mm", "type": "quantitative", "label": "Kabsch p95", "unit": "mm"},
                        ],
                    },
                    "yAxisTitle": "Kabsch RMS (mm)",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "camera_detail_table",
                    "title": "Camera-specific correction validation",
                    "subtitle": "Held-out isolated fish transitions and untouched rigid-object poses",
                    "dataset": "camera_detail",
                    "sourceId": "combined_validation",
                    "density": "comfortable",
                    "defaultSort": {"field": "natural_raw_p95_mm", "direction": "desc"},
                    "columns": [
                        {"field": "model_label", "label": "Model", "type": "text"},
                        {"field": "missing_view_label", "label": "Missing camera", "type": "text"},
                        {"field": "event_n", "label": "Events", "type": "number", "format": "number"},
                        {"field": "mean_shift_norm_mm", "label": "Mean shift", "type": "number", "format": "number", "unit": "mm"},
                        {"field": "natural_raw_p95_mm", "label": "Natural raw p95", "type": "number", "format": "number", "unit": "mm"},
                        {"field": "natural_corrected_p95_mm", "label": "Natural corrected p95", "type": "number", "format": "number", "unit": "mm"},
                        {"field": "natural_p95_change_percent", "label": "Natural p95 change", "type": "number", "format": "number", "unit": "%", "movement": True},
                        {"field": "rigid_raw_kabsch_rms_mm", "label": "Rigid raw RMS", "type": "number", "format": "number", "unit": "mm"},
                        {"field": "rigid_corrected_kabsch_rms_mm", "label": "Rigid corrected RMS", "type": "number", "format": "number", "unit": "mm"},
                        {"field": "rigid_kabsch_change_percent", "label": "Rigid RMS change", "type": "number", "format": "number", "unit": "%", "movement": True},
                    ],
                    "layout": "full",
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "**Use direct DLT rays for the next controlled runtime experiment, but do not deploy the learned affine camera weights yet.** "
                        f"On {int(dlt['natural_event_n'])} held-out isolated transitions, raw DLT reduced the median trajectory residual from "
                        f"{production['natural_raw_median_mm']:.3f} to {dlt['natural_raw_median_mm']:.3f} mm. "
                        f"The refractive model reached {refractive['natural_raw_median_mm']:.3f} mm and had the lowest raw p95 ({refractive['natural_raw_p95_mm']:.3f} mm), while DLT had the lowest median.\n\n"
                        f"DLT also won the independent three-view rigid holdout at {dlt['three_view_rigid_kabsch_rms_mm']:.3f} mm RMS, versus "
                        f"{refractive['three_view_rigid_kabsch_rms_mm']:.3f} mm for the refractive model and {production['three_view_rigid_kabsch_rms_mm']:.3f} mm for production. "
                        "The affine weights improved self-consistency and median continuity, but tail and rigid-object effects were mixed, so they are diagnostic rather than deployment-ready."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["dlt_transition_card", "dlt_rigid_card", "event_count_card"]},
                {
                    "id": "natural_finding",
                    "type": "markdown",
                    "body": (
                        "## Direct ray models reduce the typical transition shift\n\n"
                        f"Raw DLT lowered the isolated-event median by {abs(transition_reduction) * 100:.1f}% relative to production. "
                        f"At p95, production was {production['natural_raw_p95_mm']:.3f} mm, DLT {dlt['natural_raw_p95_mm']:.3f} mm, and refractive {refractive['natural_raw_p95_mm']:.3f} mm. "
                        "The chart separates raw and affine-corrected results; improvement in the middle of the distribution does not guarantee improvement in the tail."
                    ),
                },
                {"id": "natural_chart_block", "type": "chart", "chartId": "natural_model_chart"},
                {
                    "id": "rigid_finding",
                    "type": "markdown",
                    "sourceId": "rigid_holdout",
                    "body": (
                        "## The rigid holdout favors DLT for every missing camera\n\n"
                        "Raw DLT had the lowest two-view Kabsch RMS when the top, side 1, or side 2 camera was masked. "
                        "This is stronger evidence than fish-track smoothness because the object geometry is known. Missing side 1 or side 2 remains substantially harder than missing the top camera for every backend."
                    ),
                },
                {"id": "rigid_chart_block", "type": "chart", "chartId": "rigid_two_view_chart"},
                {
                    "id": "weight_finding",
                    "type": "markdown",
                    "body": (
                        "## Camera weights are useful diagnostics, not a safe correction yet\n\n"
                        f"For DLT, the aggregate natural-transition p95 changed from {dlt['natural_raw_p95_mm']:.3f} to {dlt['natural_corrected_p95_mm']:.3f} mm. "
                        f"For production it changed from {production['natural_raw_p95_mm']:.3f} to {production['natural_corrected_p95_mm']:.3f} mm, a {production['natural_correction_change_percent']:+.1f}% regression. "
                        "On rigid-object data, some camera/model corrections improved scale or shape while others worsened it. A correction that forces two-view output toward a model's three-view output can improve self-consistency without improving physical truth."
                    ),
                },
                {"id": "camera_table_block", "type": "table", "tableId": "camera_detail_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": "round7_transition",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The source recording contains 348,600 frames and three fish. Assignments were frozen to the reviewed current-selector output. "
                        "The primary natural cohort contains one-frame two-view reconstructions whose preceding and succeeding frames are QC-qualified three-view reconstructions, with the entire three-frame window inside a held-out time block.\n\n"
                        "**Trajectory residual** is the Euclidean distance from the two-view point to the midpoint of the two three-view anchors. "
                        "**Spike excess** is the largest transition step minus half the anchor-to-anchor chord. "
                        "**Counterfactual ablation error** masks one camera on a clean three-view frame and compares that two-view reconstruction with the backend's own full three-view reconstruction."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "benchmark_code",
                    "body": (
                        "## Methodology preserves assignments and separates fitting from evaluation\n\n"
                        "Cross-camera slots and fish IDs are held fixed across backends. Production reconstructs two views by synthesizing the missing pixel with its pairwise polynomial helper. DLT and refractive backends intersect only the observed rays, using one consistent two-/three-view objective.\n\n"
                        "Camera-specific affine delta models were robustly fitted after masking clean three-view rows in non-holdout time blocks. Evaluation uses every fifth 6,000-frame block and untouched rigid-object poses. Reloaded SLEAP slots reproduced 1,034,030 saved production XYZ rows with a maximum discrepancy below 10^-12 mm."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and robustness boundaries\n\n"
                        "Natural continuity is not ground truth: acceleration, turning, and an undetected identity error can increase the interpolation residual. The isolated-event design and held-out blocks reduce, but do not eliminate, those effects. "
                        "The rigid holdout validates shape and scale after pose fitting; it does not validate absolute tank position. The refractive interface fit is provisional and its final interface-refinement stage did not fully converge. "
                        "Finally, assignments were selected by production geometry, so this report isolates geometry quality rather than testing a fully DLT-scored selector."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Add native variable-view reconstruction to the runtime and test **raw DLT without affine correction** while keeping the current production selector frozen.\n"
                        "2. Repeat the same held-out transition benchmark on K_1_1 and K_1_3 before changing the default calibration.\n"
                        "3. Review the highest-residual DLT transition events to separate true turns, source-assignment mistakes, and calibration discontinuities.\n"
                        "4. If camera weights are revisited, fit them on calibration-object training poses and validate on separate rigid poses before testing fish-track smoothness."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does raw DLT preserve or improve identity assignment when DLT also scores candidate triplets?\n"
                        "- Are the largest remaining transition residuals concentrated near tank boundaries or specific fish contacts?\n"
                        "- Would independently measured dry intrinsics and pane geometry let the refractive model surpass DLT on boundary poses?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                **data,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--artifact-json", type=Path, required=True)
    parser.add_argument("--source-notes", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    data = build_report_data(args.benchmark_dir)
    artifact = build_artifact(data, generated_at)
    args.artifact_json.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_json.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    notes = {
        "audience": "technical",
        "delivery_mode": "portable_html",
        "required_structure_mapping": {
            "title": "title",
            "technical_summary": "technical_summary + headline_metrics",
            "key_findings_with_visual_evidence": [
                "natural_finding + natural_model_chart",
                "rigid_finding + rigid_two_view_chart",
                "weight_finding + camera_detail_table",
            ],
            "scope_data_metric_definitions": "scope_definitions",
            "methodology": "methodology",
            "limitations_uncertainty_robustness": "limitations",
            "recommended_next_steps": "next_steps",
            "further_questions": "further_questions",
        },
        "chart_map": [
            {
                "section": "Direct ray models reduce the typical transition shift",
                "question": "How do model and affine correction affect p95 isolated-transition residual?",
                "family": "Comparison & Ranking",
                "type": "grouped bar",
                "fields": ["model_label", "variant_label", "trajectory_p95_mm"],
                "claim": "Direct ray models reduce typical transition shift; affine tail effects differ by model.",
                "palette_policy": "hard two-root cap for raw versus corrected",
            },
            {
                "section": "The rigid holdout favors DLT for every missing camera",
                "question": "Which raw two-view backend best preserves known object shape by missing camera?",
                "family": "Comparison & Ranking",
                "type": "grouped bar",
                "fields": ["missing_view_label", "model_label", "kabsch_rms_mm"],
                "claim": "DLT has the lowest held-out Kabsch RMS for every missing camera.",
                "palette_policy": "relaxed three-category model palette",
            },
        ],
        "omissions": [
            "Longer two-view runs are secondary because constant-velocity interpolation becomes motion-confounded.",
            "Raw event-level rows are kept in CSV and omitted from the bounded report snapshot.",
            "Absolute tank-position accuracy is not claimed because the rigid holdout is pose-aligned.",
        ],
    }
    args.source_notes.write_text(
        json.dumps(notes, indent=2) + "\n", encoding="utf-8"
    )
    print(args.artifact_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
