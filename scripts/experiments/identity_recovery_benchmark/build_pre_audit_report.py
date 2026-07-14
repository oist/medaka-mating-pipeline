from __future__ import annotations

"""Build the canonical portable-report artifact for the pre-audit evidence."""

import argparse
import json
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "idtracker_ai_v6": "idtracker.ai v6",
    "native_udmt": "Native UDMT",
    "multiview_constrained_contrastive": "Multi-view contrastive",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--multiview-dir", type=Path, required=True)
    parser.add_argument("--idtracker-summary-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _source(source_id: str, label: str, path: str) -> dict:
    return {"id": source_id, "label": label, "path": path}


def _sql_literal(value: object) -> str:
    """Return a SQLite literal for the small, reviewed report snapshot."""
    if value is None or bool(pd.isna(value)):
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_query(
    rows: list[dict],
    columns: list[str],
    *,
    cte_name: str,
    select: str = "*",
) -> str:
    """Build and execute-check the exact bounded SQL exposed as provenance."""
    values = ",\n    ".join(
        "(" + ", ".join(_sql_literal(row.get(column)) for column in columns) + ")"
        for row in rows
    )
    query = (
        f"WITH {cte_name} ({', '.join(columns)}) AS (\n"
        f"  VALUES\n    {values}\n"
        f")\nSELECT {select} FROM {cte_name}"
    )
    with sqlite3.connect(":memory:") as connection:
        connection.execute(query).fetchall()
    return query


def run(args: argparse.Namespace) -> Path:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    benchmark_path = args.benchmark_dir / "benchmark_summary.json"
    events_path = args.benchmark_dir / "crossing_events.csv"
    episode_path = args.multiview_dir / "episode_decisions.csv"
    multiview_summary_path = args.multiview_dir / "summary.json"
    idtracker_summary_path = args.idtracker_summary_dir / "idtracker_run_summary.json"
    idtracker_curve_path = args.idtracker_summary_dir / "idtracker_contrastive_learning_curve.csv"
    audit_manifest_path = args.audit_dir / "audit_manifest.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    idtracker_summary = json.loads(idtracker_summary_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    events = pd.read_csv(events_path)
    episodes = pd.read_csv(episode_path)
    curve = pd.read_csv(idtracker_curve_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    copies = {
        "benchmark_summary.json": benchmark_path,
        "crossing_events.csv": events_path,
        "episode_decisions.csv": episode_path,
        "multiview_summary.json": multiview_summary_path,
        "idtracker_run_summary.json": idtracker_summary_path,
        "idtracker_contrastive_learning_curve.csv": idtracker_curve_path,
        "audit_manifest.json": audit_manifest_path,
    }
    if args.decision_rule is not None:
        copies["DECISION_RULE.md"] = args.decision_rule
    for name, source in copies.items():
        shutil.copy2(source, evidence_dir / name)

    localization_rows = []
    method_rows = []
    for method, report in benchmark["methods"].items():
        label = METHOD_LABELS.get(method, method)
        for gate, metrics in report["localization_identity_blind"].items():
            localization_rows.append(
                {
                    "method": label,
                    "gate_px": float(gate),
                    "position_recall": metrics["position_recall"],
                    "position_precision": metrics["position_precision"],
                    "all_three_coverage": metrics["all_fish_frame_coverage"],
                    "median_error_px": metrics["median_error_px"],
                }
            )
        gate80 = report["localization_identity_blind"]["80"]
        proxy = report["identity_geometry_proxy_only"]
        crossing = report["crossing_geometry_proxy_only"]
        method_rows.append(
            {
                "method": label,
                "position_recall_80px": gate80["position_recall"],
                "position_precision_80px": gate80["position_precision"],
                "all_three_coverage_80px": gate80["all_fish_frame_coverage"],
                "median_error_px": gate80["median_error_px"],
                "geometry_proxy_idf1": proxy["idf1_like"],
                "geometry_proxy_switches": proxy["proxy_switches_adjacent_frames"],
                "geometry_proxy_crossing_pass": crossing["pass_fraction"],
            }
        )
    best_recall = max(row["position_recall_80px"] for row in method_rows)
    headline = [
        {
            "events": int(benchmark["event_selection"]["n_events"]),
            "methods": len(method_rows),
            "best_recall": best_recall,
            "audit_clips": int(audit_manifest["clips"]),
        }
    ]
    episode_rows = []
    for _, row in episodes.iterrows():
        episode_rows.append(
            {
                "episode": str(row["episode_id"]),
                "events": str(row["event_ids"]),
                "frames": f"{int(row['start_frame'])}-{int(row['end_frame'])}",
                "permutation": str(row["pre_to_post_slot_permutation"]),
                "minimum_cosine": float(row["minimum_selected_cosine"]),
                "passes_frozen_threshold": bool(row["passes_calibration_eer_threshold"]),
            }
        )
    curve_rows = [
        {
            "batch": int(row.batch),
            "silhouette": float(row.silhouette),
            "new_best": bool(row.new_best),
        }
        for row in curve.itertuples(index=False)
    ]
    curve_rows.append(
        {
            "batch": int(curve["batch"].max()) + 250,
            "silhouette": float(idtracker_summary["best_silhouette"]),
            "new_best": True,
        }
    )

    common_columns = [
        "row_type",
        "method",
        "gate_px",
        "position_recall",
        "position_precision",
        "all_three_coverage",
        "median_error_px",
        "position_recall_80px",
        "position_precision_80px",
        "all_three_coverage_80px",
        "geometry_proxy_idf1",
        "geometry_proxy_switches",
        "geometry_proxy_crossing_pass",
        "methods",
        "best_recall",
    ]
    common_source_rows = [
        {"row_type": "localization", **row} for row in localization_rows
    ] + [
        {"row_type": "method_summary", **row} for row in method_rows
    ] + [
        {
            "row_type": "headline",
            "methods": len(method_rows),
            "best_recall": best_recall,
        }
    ]
    common_sql = _values_query(
        common_source_rows,
        common_columns,
        cte_name="reviewed_common_benchmark",
    )
    events_sql = _values_query(
        events.to_dict(orient="records"),
        list(events.columns),
        cte_name="locked_crossing_events",
        select="*, COUNT(*) OVER () AS events",
    )
    episodes_sql = _values_query(
        episode_rows,
        list(episode_rows[0]),
        cte_name="reviewed_multiview_decisions",
    )
    idtracker_run_sql = _values_query(
        [
            {
                "best_silhouette": idtracker_summary["best_silhouette"],
                "target_silhouette": idtracker_summary["target_silhouette"],
                "estimated_internal_accuracy": idtracker_summary["estimated_internal_accuracy"],
                "ratio_accumulated_images": idtracker_summary["ratio_accumulated_images"],
            }
        ],
        [
            "best_silhouette",
            "target_silhouette",
            "estimated_internal_accuracy",
            "ratio_accumulated_images",
        ],
        cte_name="reviewed_idtracker_run",
    )
    idtracker_curve_sql = _values_query(
        curve_rows,
        ["batch", "silhouette", "new_best"],
        cte_name="reviewed_idtracker_curve",
    )
    audit_sql = _values_query(
        [
            {
                "events": audit_manifest["events"],
                "methods": audit_manifest["methods"],
                "audit_clips": audit_manifest["clips"],
                "context_frames": audit_manifest["context_frames"],
                "audit_status": audit_manifest["audit_status"],
            }
        ],
        ["events", "methods", "audit_clips", "context_frames", "audit_status"],
        cte_name="reviewed_blinded_audit",
    )

    manifest_sources = [
        _source("common_benchmark", "Common three-method benchmark", "evidence/benchmark_summary.json"),
        _source("crossing_events", "Locked close-approach events", "evidence/crossing_events.csv"),
        _source("multiview_decisions", "Multi-view episode decisions", "evidence/episode_decisions.csv"),
        _source("idtracker_run", "idtracker.ai v6 run summary", "evidence/idtracker_run_summary.json"),
        _source("idtracker_curve", "idtracker.ai contrastive learning curve", "evidence/idtracker_contrastive_learning_curve.csv"),
        _source("audit_manifest", "Blinded human audit manifest", "evidence/audit_manifest.json"),
    ]
    if args.decision_rule is not None:
        manifest_sources.append(
            _source("decision_rule", "Pre-audit decision rule", "evidence/DECISION_RULE.md")
        )
    sources = [
        {
            "id": "common_benchmark",
            "query": {
                "engine": "python/pandas",
                "language": "sql",
                "sql": common_sql,
                "description": "Loads all three method trajectories, performs per-frame Hungarian localization matching, and computes explicitly provisional geometry-proxy identity diagnostics.",
                "tables_used": ["benchmark_summary.json"],
                "filters": ["experiment=20251225_K_1_2_FCMC", "frames=176300..181298", "primary localization gate=80 px"],
                "metric_definitions": {
                    "position_recall": "Matched emitted fish positions divided by 14,997 expected fish positions in the locked interval.",
                    "all_three_coverage": "Fraction of frames where all three expected fish positions are matched one-to-one.",
                    "geometry_proxy_idf1": "IDF1-like agreement after a global permutation against provisional geometry labels; not biological ground truth.",
                },
                "executed_at": generated_at,
            },
        },
        {
            "id": "crossing_events",
            "query": {
                "engine": "python/pandas",
                "language": "sql",
                "sql": events_sql,
                "description": "Exhaustively selects close approaches from pairwise distances in the unordered top-view reference positions.",
                "tables_used": ["crossing_events.csv"],
                "filters": ["distance <= 35 px", "at least 2 close frames", "83 context frames on both sides"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "multiview_decisions",
            "query": {
                "engine": "python/pytorch",
                "language": "sql",
                "sql": episodes_sql,
                "description": "Pools camera-balanced frozen contrastive embeddings before and after each crossing episode and solves a one-to-one Hungarian permutation.",
                "tables_used": ["episode_decisions.csv", "multiview_summary.json"],
                "filters": ["897 crops", "3 synchronized views", "calibration-frozen cosine threshold=0.8697548759"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "idtracker_run",
            "query": {
                "engine": "idtracker.ai 6.0.14",
                "language": "sql",
                "sql": idtracker_run_sql,
                "description": "Official v6 contrastive tracking run plus an explicitly recorded saved-checkpoint k-means++ fallback after the outer shell timeout.",
                "tables_used": ["idtracker_run_summary.json"],
                "filters": ["3 animals", "4999 frames", "target silhouette=0.91"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "idtracker_curve",
            "query": {
                "engine": "idtracker.ai 6.0.14",
                "language": "sql",
                "sql": idtracker_curve_sql,
                "description": "Validation silhouette values parsed from the preserved contrastive training log, with the final fallback score appended.",
                "tables_used": ["idtracker_contrastive_learning_curve.csv"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "audit_manifest",
            "query": {
                "engine": "opencv",
                "language": "sql",
                "sql": audit_sql,
                "description": "Renders 21 real-video clips with event-specific randomized method letters and a separate private blinding key.",
                "tables_used": ["audit_manifest.json"],
                "filters": ["7 events", "3 methods", "83 context frames"],
                "executed_at": generated_at,
            },
        },
    ]
    if args.decision_rule is not None:
        sources.append(
            {
                "id": "decision_rule",
                "path": "evidence/DECISION_RULE.md",
                "description": "Categorical end-to-end recovery rule locked before human ratings were available.",
            }
        )

    title = "Historical-video biological identity recovery: pre-audit evidence"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "A reproducible comparison of idtracker.ai v6, native UDMT, and a multi-view constrained contrastive linker on identical historical medaka crossings.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [
                {
                    "id": "events_card",
                    "description": "Close approaches selected without inspecting any method output.",
                    "dataset": "headline",
                    "sourceId": "crossing_events",
                    "metrics": [{"label": "Locked crossing events", "field": "events", "format": "number"}],
                },
                {
                    "id": "methods_card",
                    "description": "End-to-end identity recovery methods scored by the same code.",
                    "dataset": "headline",
                    "sourceId": "common_benchmark",
                    "metrics": [{"label": "Methods compared", "field": "methods", "format": "number"}],
                },
                {
                    "id": "recall_card",
                    "description": "Highest identity-blind fish-position recall at the declared 80 px gate.",
                    "dataset": "headline",
                    "sourceId": "common_benchmark",
                    "metrics": [{"label": "Best localization recall", "field": "best_recall", "format": "percent"}],
                },
                {
                    "id": "audit_card",
                    "description": "Method-blinded real-video clips awaiting independent ratings.",
                    "dataset": "headline",
                    "sourceId": "audit_manifest",
                    "metrics": [{"label": "Blinded clips ready", "field": "audit_clips", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "localization_chart",
                    "title": "Fish localization recall across pixel tolerances",
                    "subtitle": "Multi-view remains near 97%; native UDMT mostly tracks non-fish image structure.",
                    "type": "line",
                    "dataset": "localization",
                    "sourceId": "common_benchmark",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "gate_px", "type": "quantitative", "label": "Matching gate (px)"},
                        "y": {"field": "position_recall", "type": "quantitative", "label": "Position recall"},
                        "color": {"field": "method", "type": "nominal", "label": "Method"},
                        "tooltip": [
                            {"field": "position_precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                            {"field": "all_three_coverage", "type": "quantitative", "label": "All-three coverage", "format": "percent"},
                            {"field": "median_error_px", "type": "quantitative", "label": "Median error (px)"},
                        ],
                    },
                },
                {
                    "id": "proxy_chart",
                    "title": "Agreement with provisional geometry labels",
                    "subtitle": "Diagnostic only: geometry labels are not independent biological truth.",
                    "type": "bar",
                    "dataset": "method_summary",
                    "sourceId": "common_benchmark",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "method", "type": "nominal", "label": "Method"},
                        "y": {"field": "geometry_proxy_idf1", "type": "quantitative", "label": "Geometry-proxy IDF1"},
                        "tooltip": [
                            {"field": "geometry_proxy_switches", "type": "quantitative", "label": "Adjacent-frame proxy switches"},
                            {"field": "geometry_proxy_crossing_pass", "type": "quantitative", "label": "Crossing proxy pass", "format": "percent"},
                        ],
                    },
                },
                {
                    "id": "idtracker_curve_chart",
                    "title": "idtracker.ai contrastive training did not reach its target",
                    "subtitle": "Final saved-checkpoint fallback: 0.704 versus the 0.91 target.",
                    "type": "line",
                    "dataset": "idtracker_curve",
                    "sourceId": "idtracker_curve",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "batch", "type": "quantitative", "label": "Training batches"},
                        "y": {"field": "silhouette", "type": "quantitative", "label": "Silhouette separation"},
                        "tooltip": [{"field": "new_best", "type": "nominal", "label": "New best"}],
                    },
                },
            ],
            "tables": [
                {
                    "id": "method_table",
                    "title": "Common pre-audit scorecard",
                    "subtitle": "Localization is identity-blind; IDF1 and crossing-pass columns use provisional geometry labels only.",
                    "dataset": "method_summary",
                    "sourceId": "common_benchmark",
                    "defaultSort": {"field": "position_recall_80px", "direction": "desc"},
                    "columns": [
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "position_recall_80px", "label": "Position recall", "format": "percent"},
                        {"field": "position_precision_80px", "label": "Position precision", "format": "percent"},
                        {"field": "all_three_coverage_80px", "label": "All-three coverage", "format": "percent"},
                        {"field": "median_error_px", "label": "Median error (px)", "format": "number"},
                        {"field": "geometry_proxy_idf1", "label": "Proxy IDF1", "format": "number"},
                    ],
                },
                {
                    "id": "episode_table",
                    "title": "Multi-view crossing decisions",
                    "subtitle": "The final long male-male episode is the only one assigned a slot exchange.",
                    "dataset": "episodes",
                    "sourceId": "multiview_decisions",
                    "defaultSort": {"field": "episode", "direction": "asc"},
                    "columns": [
                        {"field": "episode", "label": "Episode", "type": "text"},
                        {"field": "events", "label": "Included events", "type": "text"},
                        {"field": "frames", "label": "Frames", "type": "text"},
                        {"field": "permutation", "label": "Pre-to-post slot mapping", "type": "text"},
                        {"field": "minimum_cosine", "label": "Minimum selected cosine", "format": "number"},
                        {"field": "passes_frozen_threshold", "label": "Passes threshold", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["events_card", "methods_card", "recall_card", "audit_card"]},
                {
                    "id": "answer_first",
                    "type": "markdown",
                    "sourceId": "common_benchmark",
                    "body": "## Current answer\n\nThe historical video contains recoverable identity-related structure, but the final biological claim is not yet proven. The multi-view method is the strongest pre-audit result: 96.7% position recall and 90.3% all-three-fish coverage at 80 px, compared with 85.4% and 62.2% for idtracker.ai v6. Native UDMT fails localization (7.3% recall; 1.4% all-three coverage), so its persistent labels mostly represent drift.",
                },
                *(
                    [
                        {
                            "id": "decision_rule",
                            "type": "markdown",
                            "sourceId": "decision_rule",
                            "body": "## The conclusion rule is fixed before unblinding\n\nAn end-to-end recovery requires at least 90% position recall, at least 80% all-three-fish coverage, and conservative human success on at least 6 of 7 crossings. Uncertain or off-fish ratings remain failures. This rule applies only to the locked interval; generalization requires additional recordings.",
                        }
                    ]
                    if args.decision_rule is not None
                    else []
                ),
                {"id": "localization_section", "type": "markdown", "body": "## First require labels to be on real fish\n\nLocalization is scored against an unordered set of three fish positions. This deliberately gives no credit for emitting three stable labels away from the animals."},
                {"id": "localization_chart_block", "type": "chart", "chartId": "localization_chart"},
                {"id": "method_table_block", "type": "table", "tableId": "method_table"},
                {
                    "id": "proxy_section",
                    "type": "markdown",
                    "sourceId": "common_benchmark",
                    "body": "## Geometry agreement is useful but not truth\n\nThe geometry-proxy IDF1 values are 0.795 for multi-view, 0.599 for idtracker.ai, and 0.047 for native UDMT. These scores diagnose disagreement; they cannot decide whether the multi-view method's final male-male slot exchange is a true biological correction or an error.",
                },
                {"id": "proxy_chart_block", "type": "chart", "chartId": "proxy_chart"},
                {
                    "id": "multiview_section",
                    "type": "markdown",
                    "sourceId": "multiview_decisions",
                    "body": "## Multi-view makes one falsifiable correction\n\nAll four merged episode assignments pass the calibration-frozen cosine threshold. The first three preserve the geometry slot order; the long final episode (frames 179771-179966) calls a one-to-one exchange of the two male slots. That specific decision is the most informative target for human inspection.",
                },
                {"id": "episode_table_block", "type": "table", "tableId": "episode_table"},
                {
                    "id": "idtracker_section",
                    "type": "markdown",
                    "sourceId": "idtracker_run",
                    "body": "## idtracker.ai learns partial appearance separation\n\nThe final saved model reaches silhouette 0.704, below idtracker.ai's 0.91 target. Its internal estimated accuracy is 95.4%, but the independent common scorer finds lower all-three-fish coverage and 24 adjacent-frame geometry-proxy switches. Internal confidence is therefore insufficient for the biological claim.",
                },
                {"id": "idtracker_curve_block", "type": "chart", "chartId": "idtracker_curve_chart"},
                {
                    "id": "audit_section",
                    "type": "markdown",
                    "sourceId": "audit_manifest",
                    "body": "## Independent audit is the deciding step\n\nTwenty-one clips are ready: 7 identical crossing events multiplied by 3 methods. Method letters are randomized separately for each event, and the worksheet asks both whether each label remains on the same fish and whether all labels remain localized. Until those ratings are returned, the report remains deliberately partial.",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations\n\n- The audit currently targets one 4,999-frame interval from one three-fish recording; generalization to other experiments is untested.\n- The multi-view primary uses a shared camera encoder. Separate top and side encoders remain a named ablation.\n- Geometry provides detection and same-frame cross-camera association, so its biological labels are never used as independent truth.\n- Native UDMT is an end-to-end failure on this window; conditional identity performance cannot rescue a method whose points are mostly off-fish.\n- A single small human audit gives limited precision; the report will retain Wilson intervals and all uncertain ratings in the denominator.",
                },
                {
                    "id": "reproducibility",
                    "type": "markdown",
                    "body": "## Reproducibility\n\nAll source artifacts are hash-recorded in the benchmark summary. The reusable implementation is under `scripts/experiments/identity_recovery_benchmark/`; the common scorer, audit renderer, audit summarizer, checkpoint recovery, and figure builder are covered by focused tests. The locked experiment is `20251225_K_1_2_FCMC`, inclusive frames 176300-181298.",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "partial",
            "datasets": {
                "headline": headline,
                "localization": localization_rows,
                "method_summary": method_rows,
                "episodes": episode_rows,
                "idtracker_curve": curve_rows,
            },
            "accessIssues": [
                {
                    "id": "human_audit_pending",
                    "dataset": "human_audit",
                    "message": "Independent blinded crossing ratings are required before a biological identity conclusion can be finalized.",
                }
            ],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://historical-medaka-identity-pre-audit",
            "controls": {"edit": False, "refresh": False},
        },
    }
    output = args.output_dir / "artifact.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> None:
    print(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
