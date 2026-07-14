from __future__ import annotations

"""Build the canonical final report after a complete blinded crossing audit."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from .build_pre_audit_report import METHOD_LABELS, _values_query, run as build_pre_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--multiview-dir", type=Path, required=True)
    parser.add_argument("--idtracker-summary-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--audit-results-dir", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_locked_decision_rule(
    benchmark: dict[str, object],
    audit_methods: list[dict[str, object]],
) -> list[dict[str, object]]:
    audit_by_method = {str(row["method"]): row for row in audit_methods}
    if set(audit_by_method) != set(benchmark["methods"]):
        raise ValueError("audit methods do not exactly match the benchmark methods")
    rows = []
    for method, report in benchmark["methods"].items():
        audit = audit_by_method[method]
        if int(audit["events_scored"]) != 7:
            raise ValueError(f"{method} does not have exactly seven scored crossing events")
        gate80 = report["localization_identity_blind"]["80"]
        recall = float(gate80["position_recall"])
        coverage = float(gate80["all_fish_frame_coverage"])
        successes = int(audit["conservative_successes"])
        localization_pass = recall >= 0.90 and coverage >= 0.80
        if localization_pass and successes >= 6:
            decision = "recovered in locked interval"
        elif localization_pass and successes >= 4:
            decision = "promising but not established"
        elif not localization_pass:
            decision = "not end-to-end recovered: localization gate failed"
        else:
            decision = "not recovered on audited crossings"
        rows.append(
            {
                "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "position_recall_80px": recall,
                "all_three_coverage_80px": coverage,
                "audit_successes": successes,
                "audited_events": int(audit["events_scored"]),
                "conservative_success_fraction": float(audit["conservative_success_fraction"]),
                "wilson_95_lower": float(audit["wilson_95_lower"]),
                "wilson_95_upper": float(audit["wilson_95_upper"]),
                "mean_confidence": float(audit["mean_confidence"]),
                "continuity_pass": int(audit["continuity_pass"]),
                "continuity_switch": int(audit["continuity_switch"]),
                "continuity_uncertain": int(audit["continuity_uncertain"]),
                "localization_fail": int(audit["localization_fail"]),
                "localization_uncertain": int(audit["localization_uncertain"]),
                "localized_events": int(audit["localized_events"]),
                "localization_gate_pass": localization_pass,
                "decision": decision,
            }
        )
    return rows


def _conclusion_text(rows: list[dict[str, object]]) -> str:
    recovered = [row for row in rows if row["decision"] == "recovered in locked interval"]
    promising = [row for row in rows if row["decision"] == "promising but not established"]
    if recovered:
        details = "; ".join(
            f"{row['method']} ({row['audit_successes']}/7 audited crossings)"
            for row in recovered
        )
        return (
            "Yes—within this locked historical interval, the predeclared end-to-end "
            f"criterion is met by {details}. This is interval-level evidence, not proof "
            "that the method generalizes to other recordings."
        )
    if promising:
        details = "; ".join(
            f"{row['method']} ({row['audit_successes']}/7 audited crossings)"
            for row in promising
        )
        return (
            "Biological identity recovery is promising but not established under the "
            f"locked rule. The localization gates were met by {details}, but the human "
            "continuity criterion was not."
        )
    return (
        "No method meets the locked end-to-end recovery criterion in this historical "
        "interval. Apparent label persistence is insufficient when localization or "
        "audited biological continuity fails."
    )


def run(args: argparse.Namespace) -> Path:
    artifact_path = build_pre_audit(
        SimpleNamespace(
            benchmark_dir=args.benchmark_dir,
            multiview_dir=args.multiview_dir,
            idtracker_summary_dir=args.idtracker_summary_dir,
            audit_dir=args.audit_dir,
            decision_rule=args.decision_rule,
            output_dir=args.output_dir,
        )
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    benchmark_path = args.benchmark_dir / "benchmark_summary.json"
    audit_json_path = args.audit_results_dir / "audit_results.json"
    audit_methods_path = args.audit_results_dir / "audit_method_summary.csv"
    audit_pairs_path = args.audit_results_dir / "audit_pairwise_comparisons.csv"
    audit_rows_path = args.audit_results_dir / "audit_unblinded_rows.csv"
    completed_worksheet_path = args.audit_dir / "AUDIT_WORKSHEET_COMPLETED.csv"
    blinding_key_path = args.audit_dir / "PRIVATE_BLINDING_KEY.csv"
    required = [
        audit_json_path,
        audit_methods_path,
        audit_pairs_path,
        audit_rows_path,
        completed_worksheet_path,
        blinding_key_path,
    ]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError(f"missing audit result files: {missing}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    if int(audit["incomplete_rows"]) != 0 or int(audit["scored_rows"]) != 21:
        raise ValueError("final report requires all 21 audit rows to be valid and scored")
    assessments = apply_locked_decision_rule(benchmark, audit["methods"])
    assessments.sort(
        key=lambda row: (
            float(row["conservative_success_fraction"]),
            float(row["position_recall_80px"]),
        ),
        reverse=True,
    )
    pairwise = [
        {
            **row,
            "method_a": METHOD_LABELS.get(str(row["method_a"]), str(row["method_a"])),
            "method_b": METHOD_LABELS.get(str(row["method_b"]), str(row["method_b"])),
        }
        for row in audit["pairwise_comparisons"]
    ]
    if len(pairwise) != 3:
        raise ValueError("final report requires all three paired method comparisons")
    conclusion = _conclusion_text(assessments)
    with audit_rows_path.open("r", encoding="utf-8", newline="") as stream:
        audit_event_rows = list(csv.DictReader(stream))
    audit_event_source_rows = [
        {
            "event_id": row["event_id"],
            "method": METHOD_LABELS.get(row["method"], row["method"]),
            "continuity": row["continuity"],
            "localization": row["localization"],
            "confidence_1_to_3": int(row["confidence_1_to_3"]),
            "conservative_success": str(row["conservative_success"]).lower() == "true",
        }
        for row in audit_event_rows
    ]
    continuity_uncertain_total = sum(
        row["continuity"] == "uncertain" for row in audit_event_source_rows
    )
    low_confidence_total = sum(
        row["confidence_1_to_3"] == 1 for row in audit_event_source_rows
    )

    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for path in required:
        shutil.copy2(path, evidence_dir / path.name)
    evidence_hashes = {path.name: _sha256(path) for path in required}
    evidence_hashes[benchmark_path.name] = _sha256(benchmark_path)
    evidence_hashes[args.decision_rule.name] = _sha256(args.decision_rule)
    decision_record = {
        "schema_version": "historical_identity_recovery_decision_v1",
        "scope": "20251225_K_1_2_FCMC frames 176300-181298 only",
        "locked_rule": {
            "minimum_position_recall_80px": 0.90,
            "minimum_all_three_coverage_80px": 0.80,
            "minimum_conservative_audit_successes_out_of_7": 6,
        },
        "conclusion": conclusion,
        "methods": assessments,
        "audit_evidence_sha256": evidence_hashes,
    }
    (evidence_dir / "FINAL_DECISION.json").write_text(
        json.dumps(decision_record, indent=2) + "\n", encoding="utf-8"
    )

    method_sql = _values_query(
        assessments,
        list(assessments[0]),
        cte_name="reviewed_final_method_assessment",
    )
    pair_sql = _values_query(
        pairwise,
        list(pairwise[0]),
        cte_name="reviewed_paired_audit_comparisons",
    )
    event_sql = _values_query(
        audit_event_source_rows,
        list(audit_event_source_rows[0]),
        cte_name="reviewed_unblinded_event_ratings",
    )
    artifact["sources"].extend(
        [
            {
                "id": "human_audit_methods",
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": method_sql,
                    "description": "Loads the complete unblinded seven-event method scorecard and applies the locked localization-plus-continuity decision rule.",
                    "tables_used": ["audit_method_summary.csv", "benchmark_summary.json"],
                    "filters": ["21/21 rows valid", "uncertain and off-fish ratings count as non-successes"],
                    "metric_definitions": {
                        "conservative_success_fraction": "Events rated both continuity=pass and localization=pass divided by all seven audited events.",
                        "decision": "At least 90% recall, 80% all-three coverage, and 6/7 conservative audit successes.",
                    },
                },
            },
            {
                "id": "human_audit_pairs",
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": pair_sql,
                    "description": "Paired event wins, losses, ties, rate differences, and two-sided exact McNemar/binomial tests.",
                    "tables_used": ["audit_pairwise_comparisons.csv"],
                    "filters": ["same seven crossing events for every method"],
                },
            },
            {
                "id": "human_audit_rows",
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": event_sql,
                    "description": "Loads all 21 unblinded event-level ratings at one row per crossing and method.",
                    "tables_used": ["audit_unblinded_rows.csv"],
                    "filters": ["21/21 rows valid", "one independent rater"],
                },
            },
        ]
    )
    artifact["manifest"]["sources"].extend(
        [
            {"id": "human_audit_methods", "label": "Complete human audit method scorecard", "path": "evidence/audit_method_summary.csv"},
            {"id": "human_audit_pairs", "label": "Paired human audit comparisons", "path": "evidence/audit_pairwise_comparisons.csv"},
            {"id": "human_audit_rows", "label": "Unblinded event-level ratings", "path": "evidence/audit_unblinded_rows.csv"},
        ]
    )

    title = "Historical-video biological identity recovery: final audited evidence"
    manifest = artifact["manifest"]
    manifest["title"] = title
    manifest["description"] = "Final locked-rule comparison of idtracker.ai v6, native UDMT, and multi-view constrained contrastive identity recovery on historical medaka crossings."
    title_block = next(block for block in manifest["blocks"] if block["id"] == "title")
    title_block["body"] = f"# {title}"
    answer = next(block for block in manifest["blocks"] if block["id"] == "answer_first")
    answer.pop("sourceId", None)
    by_key = {row["method_key"]: row for row in assessments}
    idtracker = by_key["idtracker_ai_v6"]
    multiview = by_key["multiview_constrained_contrastive"]
    udmt = by_key["native_udmt"]
    answer["body"] = (
        "## Technical summary\n\n"
        f"**Conclusion.** {conclusion}\n\n"
        f"**Human audit.** idtracker.ai v6 passed {idtracker['audit_successes']}/7 crossings, "
        f"the multi-view method passed {multiview['audit_successes']}/7, and native UDMT "
        f"passed {udmt['audit_successes']}/7. None reached the locked 6/7 criterion.\n\n"
        "**Interpretation.** The multi-view method solved fish localization best but did not "
        "preserve biological identity reliably through close contact. idtracker.ai showed "
        "more partial identity continuity, yet it also failed both the full-window localization "
        "gates and the audit threshold."
    )

    best_audit = max(float(row["conservative_success_fraction"]) for row in assessments)
    audit_card = next(card for card in manifest["cards"] if card["id"] == "audit_card")
    audit_card.update(
        {
            "description": "Highest conservative biological-continuity success across the three methods.",
            "dataset": "final_headline",
            "sourceId": "human_audit_methods",
            "metrics": [{"label": "Best audited success", "field": "best_audit_success", "format": "percent"}],
        }
    )
    manifest["charts"].append(
        {
            "id": "audit_success_chart",
            "title": "Conservative human-audit success by method",
            "subtitle": "Seven identical crossings per method; success requires localization and identity continuity.",
            "type": "bar",
            "dataset": "final_assessment",
            "sourceId": "human_audit_methods",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "conservative_success_fraction", "type": "quantitative", "label": "Conservative success"},
                "tooltip": [
                    {"field": "audit_successes", "type": "quantitative", "label": "Passed crossings"},
                    {"field": "wilson_95_lower", "type": "quantitative", "label": "95% CI lower", "format": "percent"},
                    {"field": "wilson_95_upper", "type": "quantitative", "label": "95% CI upper", "format": "percent"},
                    {"field": "decision", "type": "nominal", "label": "Locked-rule result"},
                ],
            },
        }
    )
    manifest["tables"].extend(
        [
            {
                "id": "final_assessment_table",
                "title": "Locked-rule end-to-end assessment",
                "subtitle": "All localization and human-continuity gates are shown together.",
                "dataset": "final_assessment",
                "sourceId": "human_audit_methods",
                "defaultSort": {"field": "conservative_success_fraction", "direction": "desc"},
                "columns": [
                    {"field": "method", "label": "Method", "type": "text"},
                    {"field": "position_recall_80px", "label": "Position recall", "format": "percent"},
                    {"field": "all_three_coverage_80px", "label": "All-three coverage", "format": "percent"},
                    {"field": "audit_successes", "label": "Audit passes", "format": "number"},
                    {"field": "conservative_success_fraction", "label": "Audit success", "format": "percent"},
                    {"field": "wilson_95_lower", "label": "95% CI lower", "format": "percent"},
                    {"field": "wilson_95_upper", "label": "95% CI upper", "format": "percent"},
                    {"field": "decision", "label": "Decision", "type": "text"},
                ],
            },
            {
                "id": "audit_pairwise_table",
                "title": "Paired crossing-event comparisons",
                "subtitle": "Exact tests are descriptive because only seven events were audited.",
                "dataset": "audit_pairwise",
                "sourceId": "human_audit_pairs",
                "defaultSort": {"field": "paired_success_rate_difference_a_minus_b", "direction": "desc"},
                "columns": [
                    {"field": "method_a", "label": "Method A", "type": "text"},
                    {"field": "method_b", "label": "Method B", "type": "text"},
                    {"field": "method_a_wins", "label": "A wins", "format": "number"},
                    {"field": "method_b_wins", "label": "B wins", "format": "number"},
                    {"field": "both_pass", "label": "Both pass", "format": "number"},
                    {"field": "both_fail", "label": "Both fail", "format": "number"},
                    {"field": "paired_success_rate_difference_a_minus_b", "label": "Paired difference", "format": "percent"},
                    {"field": "exact_mcnemar_p_two_sided", "label": "Exact p", "format": "number"},
                ],
            },
        ]
    )
    audit_index = next(index for index, block in enumerate(manifest["blocks"]) if block["id"] == "audit_section")
    manifest["blocks"][audit_index:audit_index + 1] = [
        {
            "id": "audit_conclusion",
            "type": "markdown",
            "sourceId": "human_audit_methods",
            "body": (
                "## No method passed the independent identity audit\n\n"
                f"idtracker.ai v6 passed {idtracker['audit_successes']}/7 crossings, the "
                f"multi-view method passed {multiview['audit_successes']}/7, and native UDMT "
                f"passed {udmt['audit_successes']}/7. The primary result keeps uncertain and "
                "off-fish ratings as non-successes; Wilson intervals remain wide because each "
                "method has only seven events."
            ),
        },
        {"id": "audit_success_chart_block", "type": "chart", "chartId": "audit_success_chart"},
        {"id": "final_assessment_table_block", "type": "table", "tableId": "final_assessment_table"},
        {
            "id": "audit_uncertainty",
            "type": "markdown",
            "sourceId": "human_audit_rows",
            "body": (
                "## The audit was cautious rather than overconfident\n\n"
                f"The rater marked identity continuity as uncertain in {continuity_uncertain_total}/21 "
                f"clips and used low confidence for {low_confidence_total}/21. These observations "
                "reduce the precision of the method ranking, but they do not rescue any method: "
                "the locked rule deliberately counts uncertainty as non-success."
            ),
        },
        {
            "id": "paired_comparison_note",
            "type": "markdown",
            "sourceId": "human_audit_pairs",
            "body": "## Paired method comparison\n\nWins and losses use the same seven biological crossing events for both methods. Exact two-sided McNemar/binomial p-values are shown as descriptive uncertainty checks, not as population-level proof.",
        },
        {"id": "audit_pairwise_table_block", "type": "table", "tableId": "audit_pairwise_table"},
    ]
    limitations = next(block for block in manifest["blocks"] if block["id"] == "limitations")
    limitations["body"] = limitations["body"].replace("The audit currently targets", "The audit targets")
    reproducibility = next(block for block in manifest["blocks"] if block["id"] == "reproducibility")
    reproducibility["body"] += " The completed worksheet, private-key unblinding output, paired comparisons, and final decision record are copied into the report evidence with SHA-256 hashes."

    scope_block = {
        "id": "scope_and_definitions",
        "type": "markdown",
        "body": (
            "## What was measured\n\n"
            "The locked scope is one 4,999-frame interval from a three-fish historical recording "
            "(`20251225_K_1_2_FCMC`, frames 176300-181298). Seven close approaches were selected "
            "without inspecting method outputs, producing 21 blinded clips. Position recall asks "
            "whether an emitted point lands near any real fish; all-three coverage asks whether all "
            "three fish are localized in the same frame. The primary audit success requires both "
            "localization=pass and identity continuity=pass for a crossing."
        ),
    }
    methodology_block = {
        "id": "methodology",
        "type": "markdown",
        "body": (
            "## Three native methods, one locked comparison\n\n"
            "idtracker.ai v6 used its native contrastive identity workflow on the cleaned top view; "
            "native UDMT used its XMem/ST-Net trajectory output; and the multi-view method used a "
            "frozen shared encoder with synchronized top and side crops plus one-to-one crossing "
            "assignments. Every method was scored on the same frames and crossing events. Geometry "
            "labels were allowed only as a diagnostic proxy, never as biological ground truth."
        ),
    }
    next_steps_block = {
        "id": "recommended_next_steps",
        "type": "markdown",
        "body": (
            "## Use the current systems as proposals, not final biological IDs\n\n"
            "1. Preserve the multi-view reconstruction for high-coverage localization, but require "
            "human confirmation at crossings before downstream social-behavior analyses use identity.\n"
            "2. For automatic recovery from these historical recordings, retrain appearance linkage "
            "with view-specific top/side encoders, segmentation-centered crops, and temporal memory, "
            "then rerun this unchanged audit protocol.\n"
            "3. Expand the audit to additional experiments and a second blinded rater before making "
            "a cohort-wide claim."
        ),
    }
    questions_block = {
        "id": "further_questions",
        "type": "markdown",
        "body": (
            "## Questions that could change the conclusion\n\n"
            "- Can view-specific encoders prevent the multi-view swaps observed after contact?\n"
            "- Does training on longer clean fragments from the same historical fish improve "
            "idtracker-style continuity without sacrificing localization coverage?\n"
            "- How stable are the rankings across recordings, fish phenotypes, and independent raters?"
        ),
    }
    blocks_by_id = {block["id"]: block for block in manifest["blocks"]}
    manifest["blocks"] = [
        blocks_by_id["title"],
        blocks_by_id["answer_first"],
        blocks_by_id["headline_metrics"],
        blocks_by_id["audit_conclusion"],
        blocks_by_id["audit_success_chart_block"],
        blocks_by_id["final_assessment_table_block"],
        blocks_by_id["audit_uncertainty"],
        blocks_by_id["paired_comparison_note"],
        blocks_by_id["audit_pairwise_table_block"],
        blocks_by_id["localization_section"],
        blocks_by_id["localization_chart_block"],
        blocks_by_id["method_table_block"],
        blocks_by_id["proxy_section"],
        blocks_by_id["proxy_chart_block"],
        scope_block,
        methodology_block,
        blocks_by_id["decision_rule"],
        blocks_by_id["multiview_section"],
        blocks_by_id["episode_table_block"],
        blocks_by_id["idtracker_section"],
        blocks_by_id["idtracker_curve_block"],
        blocks_by_id["limitations"],
        next_steps_block,
        questions_block,
        blocks_by_id["reproducibility"],
    ]

    snapshot = artifact["snapshot"]
    snapshot["status"] = "ready"
    snapshot.pop("accessIssues", None)
    snapshot["datasets"]["final_headline"] = [{"best_audit_success": best_audit}]
    snapshot["datasets"]["final_assessment"] = assessments
    snapshot["datasets"]["audit_pairwise"] = pairwise
    snapshot["datasets"]["audit_event_rows"] = audit_event_source_rows
    snapshot["datasets"]["proxy_summary"] = sorted(
        snapshot["datasets"]["method_summary"],
        key=lambda row: float(row["geometry_proxy_idf1"]),
        reverse=True,
    )
    next(chart for chart in manifest["charts"] if chart["id"] == "proxy_chart")[
        "dataset"
    ] = "proxy_summary"

    chart_map = {
        "schema_version": "identity_recovery_chart_map_v1",
        "charts": [
            {"section": "Independent human audit", "question": "Which method preserved localized biological identity across crossings?", "family": "Comparison", "type": "bar", "fields": ["method", "conservative_success_fraction"], "takeaway": "No method reached 6/7; idtracker.ai was descriptively highest at 3/7.", "palette_policy": "single-root preferred"},
            {"section": "Fish localization", "question": "Do emitted labels land on real fish across matching tolerances?", "family": "Trend over ordered tolerance", "type": "line", "fields": ["gate_px", "position_recall", "method"], "takeaway": "Multi-view localizes fish best; native UDMT mostly drifts off fish.", "palette_policy": "relaxed multi-category"},
            {"section": "Geometry proxy", "question": "How strongly do methods agree with provisional geometry labels?", "family": "Comparison", "type": "bar", "fields": ["method", "geometry_proxy_idf1"], "takeaway": "Proxy ranking disagrees with the independent human audit and is not biological truth.", "palette_policy": "single-root preferred"},
            {"section": "idtracker.ai training", "question": "Did contrastive training reach the package target?", "family": "Trend", "type": "line", "fields": ["batch", "silhouette"], "takeaway": "Training remained below the 0.91 target.", "palette_policy": "single-root preferred"},
        ],
        "qa_note": "Wilson intervals are shown in the exact-value table because the native bar chart does not encode interval whiskers.",
    }
    (evidence_dir / "CHART_MAP.json").write_text(
        json.dumps(chart_map, indent=2) + "\n", encoding="utf-8"
    )
    report_notes = {
        "audience": "technical, written for both ML/MV and general neuroscience readers",
        "required_structure_mapping": {
            "technical_summary": "answer_first",
            "key_findings": ["audit_conclusion", "localization_section", "proxy_section"],
            "scope_data_definitions": "scope_and_definitions",
            "methodology": ["methodology", "decision_rule", "multiview_section", "idtracker_section"],
            "limitations_robustness": ["audit_uncertainty", "paired_comparison_note", "limitations"],
            "recommended_next_steps": "recommended_next_steps",
            "further_questions": "further_questions",
        },
        "delivery_mode": "portable self-contained HTML",
    }
    (evidence_dir / "REPORT_NOTES.json").write_text(
        json.dumps(report_notes, indent=2) + "\n", encoding="utf-8"
    )
    artifact["package_info"]["originUrl"] = "artifact://historical-medaka-identity-final-audit"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def main(argv: list[str] | None = None) -> None:
    print(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
