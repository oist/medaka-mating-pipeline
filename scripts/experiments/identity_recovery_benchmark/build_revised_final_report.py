from __future__ import annotations

"""Build the corrected final report from separation-delimited audit encounters."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from .build_pre_audit_report import METHOD_LABELS, _values_query


AUDIT_SOURCE_IDS = {
    "audit_manifest",
    "decision_rule",
    "human_audit_methods",
    "human_audit_pairs",
    "human_audit_rows",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _primary_outcome(row: dict[str, object]) -> str:
    if bool(row["conservative_success"]):
        return "End-to-end success"
    if str(row["localization"]) != "pass":
        return "Localization failure"
    if str(row["continuity"]) == "switch":
        return "Identity switch"
    return "Uncertain continuity"


def build_encounter_assessments(
    benchmark: dict[str, object],
    audit_methods: list[dict[str, object]],
) -> list[dict[str, object]]:
    audit_by_method = {str(row["method"]): row for row in audit_methods}
    if set(audit_by_method) != set(benchmark["methods"]):
        raise ValueError("audit methods do not exactly match benchmark methods")
    event_counts = {int(row["events_scored"]) for row in audit_methods}
    if len(event_counts) != 1:
        raise ValueError("methods do not share the same encounter count")
    events = next(iter(event_counts))
    if events < 1:
        raise ValueError("revised audit has no encounters")

    rows: list[dict[str, object]] = []
    for method, benchmark_method in benchmark["methods"].items():
        audit = audit_by_method[method]
        gate80 = benchmark_method["localization_identity_blind"]["80"]
        recall = float(gate80["position_recall"])
        coverage = float(gate80["all_fish_frame_coverage"])
        successes = int(audit["conservative_successes"])
        localization_fail = int(audit["localization_fail"])
        switches = int(audit["continuity_switch"])
        if successes == events:
            decision = "end-to-end success in every audited encounter"
        elif successes > 0:
            decision = "partial end-to-end success; not established across encounters"
        elif localization_fail == events:
            decision = "no end-to-end success: localization failed in every encounter"
        elif switches == events and localization_fail == 0:
            decision = "no end-to-end success: identity switched in every encounter"
        else:
            decision = "no end-to-end success in audited encounters"
        rows.append(
            {
                "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "position_recall_80px": recall,
                "all_three_coverage_80px": coverage,
                "audit_successes": successes,
                "audited_events": events,
                "conservative_success_fraction": float(audit["conservative_success_fraction"]),
                "wilson_95_lower": float(audit["wilson_95_lower"]),
                "wilson_95_upper": float(audit["wilson_95_upper"]),
                "mean_confidence": float(audit["mean_confidence"]),
                "continuity_pass": int(audit["continuity_pass"]),
                "continuity_switch": switches,
                "continuity_uncertain": int(audit["continuity_uncertain"]),
                "localization_fail": localization_fail,
                "localization_uncertain": int(audit["localization_uncertain"]),
                "localized_events": int(audit["localized_events"]),
                "full_window_localization_gate_pass": recall >= 0.90 and coverage >= 0.80,
                "decision": decision,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--audit-results-dir", type=Path, required=True)
    parser.add_argument("--revision-notice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> Path:
    base_artifact_path = args.base_report_dir / "artifact.json"
    base_evidence = args.base_report_dir / "evidence"
    artifact = json.loads(base_artifact_path.read_text(encoding="utf-8"))
    benchmark_path = args.benchmark_dir / "benchmark_summary.json"
    audit_json_path = args.audit_results_dir / "audit_results.json"
    audit_methods_path = args.audit_results_dir / "audit_method_summary.csv"
    audit_pairs_path = args.audit_results_dir / "audit_pairwise_comparisons.csv"
    audit_rows_path = args.audit_results_dir / "audit_unblinded_rows.csv"
    worksheet_path = args.audit_dir / "AUDIT_WORKSHEET_COMPLETED.csv"
    key_path = args.audit_dir / "PRIVATE_BLINDING_KEY.csv"
    encounter_path = args.audit_dir / "encounter_events.csv"
    audit_manifest_path = args.audit_dir / "audit_manifest.json"
    codec_report_path = args.audit_dir / "codec_verification_report.json"
    required = [
        benchmark_path,
        audit_json_path,
        audit_methods_path,
        audit_pairs_path,
        audit_rows_path,
        worksheet_path,
        key_path,
        encounter_path,
        audit_manifest_path,
        codec_report_path,
        args.revision_notice,
    ]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError(f"missing revised audit evidence: {missing}")

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    encounters = _read_csv(encounter_path)
    event_count = len(encounters)
    if int(audit["incomplete_rows"]) != 0:
        raise ValueError("revised audit contains incomplete rows")
    if int(audit["scored_rows"]) != event_count * len(benchmark["methods"]):
        raise ValueError("revised audit row count does not match encounters x methods")
    assessments = build_encounter_assessments(benchmark, audit["methods"])
    assessments.sort(
        key=lambda row: (
            int(row["audit_successes"]),
            int(row["localized_events"]),
            float(row["position_recall_80px"]),
        ),
        reverse=True,
    )
    if any(int(row["audit_successes"]) != 0 for row in assessments):
        conclusion = (
            "At least one method achieved an end-to-end success in the revised audit; "
            "interpretation requires method-level review."
        )
    else:
        conclusion = (
            f"No method achieved end-to-end biological identity recovery in either of the "
            f"{event_count} separation-delimited encounters ({event_count} encounters per method)."
        )

    raw_event_rows = _read_csv(audit_rows_path)
    event_rows = [
        {
            "event_id": row["event_id"],
            "method_key": row["method"],
            "method": METHOD_LABELS.get(row["method"], row["method"]),
            "continuity": row["continuity"],
            "localization": row["localization"],
            "confidence_1_to_3": int(row["confidence_1_to_3"]),
            "conservative_success": row["conservative_success"].lower() == "true",
        }
        for row in raw_event_rows
    ]
    for row in event_rows:
        row["primary_outcome"] = _primary_outcome(row)
    outcome_order = [
        "End-to-end success",
        "Identity switch",
        "Localization failure",
        "Uncertain continuity",
    ]
    outcome_rows: list[dict[str, object]] = []
    for assessment in assessments:
        method = str(assessment["method"])
        method_rows = [row for row in event_rows if row["method"] == method]
        for order, outcome in enumerate(outcome_order):
            outcome_rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "outcome_order": order,
                    "encounters": sum(row["primary_outcome"] == outcome for row in method_rows),
                    "audited_encounters": event_count,
                }
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
        raise ValueError("expected three paired method comparisons")

    design_rows = [
        {
            "event_id": row["event_id"],
            "source_event_ids": row["source_event_ids"],
            "clip_start_frame": int(row["clip_start_frame"]),
            "clip_end_frame": int(row["clip_end_frame"]),
            "clip_frames": int(row["clip_frames"]),
            "separation_threshold_px": float(row["separation_threshold_px"]),
            "separation_hold_frames": int(row["separation_hold_frames"]),
            "separation_hold_seconds": float(row["separation_hold_seconds"]),
            "pre_min_distance_px": float(row["pre_min_distance_px"]),
            "post_min_distance_px": float(row["post_min_distance_px"]),
        }
        for row in encounters
    ]

    evidence_dir = args.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    retained_base_evidence = [
        "benchmark_summary.json",
        "crossing_events.csv",
        "multiview_summary.json",
        "episode_decisions.csv",
        "idtracker_run_summary.json",
        "idtracker_contrastive_learning_curve.csv",
    ]
    for name in retained_base_evidence:
        shutil.copy2(base_evidence / name, evidence_dir / name)
    copied_revised = [
        audit_json_path,
        audit_methods_path,
        audit_pairs_path,
        audit_rows_path,
        worksheet_path,
        key_path,
        encounter_path,
        audit_manifest_path,
        codec_report_path,
        args.revision_notice,
    ]
    for path in copied_revised:
        shutil.copy2(path, evidence_dir / path.name)
    original_rule = base_evidence / "DECISION_RULE.md"
    original_rule_copy = evidence_dir / "ORIGINAL_DECISION_RULE_SUPERSEDED.md"
    shutil.copy2(original_rule, original_rule_copy)
    evidence_hashes = {
        path.name: _sha256(path)
        for path in copied_revised
    }
    evidence_hashes[benchmark_path.name] = _sha256(benchmark_path)
    evidence_hashes[original_rule_copy.name] = _sha256(original_rule_copy)
    decision_record = {
        "schema_version": "historical_identity_recovery_revised_decision_v2",
        "scope": "20251225_K_1_2_FCMC frames 176300-181298 only",
        "audit_unit": {
            "source_close_approach_fragments": 7,
            "independent_separation_delimited_encounters": event_count,
            "separation_threshold_px": float(design_rows[0]["separation_threshold_px"]),
            "separation_hold_frames": int(design_rows[0]["separation_hold_frames"]),
            "separation_hold_seconds": float(design_rows[0]["separation_hold_seconds"]),
        },
        "original_rule_status": (
            "The pre-audit 6/7 fragment threshold is superseded because crossings 01-06 "
            "were shown, before the revised ratings were unblinded, to be one prolonged "
            "encounter. No replacement success threshold was invented after unblinding."
        ),
        "decision_basis": (
            "Descriptive encounter-level result: end-to-end success requires both "
            "localization=pass and continuity=pass; every method recorded 0/2."
        ),
        "conclusion": conclusion,
        "methods": assessments,
        "audit_evidence_sha256": evidence_hashes,
    }
    decision_path = evidence_dir / "FINAL_REVISED_DECISION.json"
    decision_path.write_text(json.dumps(decision_record, indent=2) + "\n", encoding="utf-8")

    # Replace only audit-dependent sources; preserve model and benchmark provenance.
    artifact["sources"] = [
        source for source in artifact["sources"] if source["id"] not in AUDIT_SOURCE_IDS
    ]
    artifact["manifest"]["sources"] = [
        source
        for source in artifact["manifest"]["sources"]
        if source["id"] not in AUDIT_SOURCE_IDS
    ]
    source_objects = [
        {
            "id": "revised_audit_design",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _values_query(design_rows, list(design_rows[0]), cte_name="reviewed_separation_encounters"),
                "description": "Loads the two separation-delimited audit encounters and their explicit clip bounds.",
                "tables_used": ["encounter_events.csv", "audit_manifest.json"],
                "filters": ["100 px minimum pair separation", "166 consecutive source frames at both clip ends"],
                "metric_definitions": {
                    "encounter": "One or more close-approach fragments with no sustained clear separation between them.",
                    "clear_separation": "The crossing pair remains at least 100 px apart for 166 consecutive source frames (1.0 s).",
                },
            },
        },
        {
            "id": "human_audit_methods",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _values_query(assessments, list(assessments[0]), cte_name="reviewed_encounter_method_assessment"),
                "description": "Loads the complete two-encounter method assessment and full-window localization metrics.",
                "tables_used": ["audit_method_summary.csv", "benchmark_summary.json"],
                "filters": ["6/6 ratings valid", "uncertain and off-fish ratings are non-successes"],
                "metric_definitions": {
                    "conservative_success_fraction": "Encounters rated both continuity=pass and localization=pass divided by two audited encounters.",
                    "decision": "Observed encounter-level result only; no post-hoc replacement threshold was introduced.",
                },
            },
        },
        {
            "id": "human_audit_rows",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _values_query(event_rows, list(event_rows[0]), cte_name="reviewed_encounter_ratings"),
                "description": "Loads all six unblinded encounter-level ratings.",
                "tables_used": ["audit_unblinded_rows.csv"],
                "filters": ["2 encounters", "3 methods", "one independent rater"],
            },
        },
        {
            "id": "human_audit_pairs",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _values_query(pairwise, list(pairwise[0]), cte_name="reviewed_encounter_pairwise"),
                "description": "Loads paired method ties and exact tests on the same two encounters.",
                "tables_used": ["audit_pairwise_comparisons.csv"],
                "filters": ["same two encounters for every method"],
            },
        },
        {
            "id": "revised_decision",
            "path": "evidence/FINAL_REVISED_DECISION.json",
            "description": "Corrected encounter-level conclusion, audit-unit rationale, method outcomes, and evidence hashes.",
        },
    ]
    artifact["sources"].extend(source_objects)
    artifact["manifest"]["sources"].extend(
        [
            {"id": "revised_audit_design", "label": "Separation-delimited encounter design", "path": "evidence/encounter_events.csv"},
            {"id": "human_audit_methods", "label": "Revised human-audit method scorecard", "path": "evidence/audit_method_summary.csv"},
            {"id": "human_audit_rows", "label": "Revised unblinded encounter ratings", "path": "evidence/audit_unblinded_rows.csv"},
            {"id": "human_audit_pairs", "label": "Revised paired encounter comparisons", "path": "evidence/audit_pairwise_comparisons.csv"},
            {"id": "revised_decision", "label": "Final corrected encounter-level decision", "path": "evidence/FINAL_REVISED_DECISION.json"},
        ]
    )

    manifest = artifact["manifest"]
    title = "Historical-video identity recovery: corrected encounter-level audit"
    manifest["title"] = title
    manifest["description"] = (
        "Corrected comparison of idtracker.ai v6, native UDMT, and multi-view contrastive identity recovery using separation-delimited historical encounters."
    )
    blocks = {block["id"]: block for block in manifest["blocks"]}
    blocks["title"]["body"] = f"# {title}"
    blocks["answer_first"].update(
        {
            "sourceId": "revised_decision",
            "body": (
                "## Technical summary\n\n"
                f"**Conclusion.** {conclusion}\n\n"
                "**Failure modes differ.** Multi-view localized all three fish in both encounters but visibly switched identity in both. idtracker.ai had one localization failure and one localized identity switch. Native UDMT failed localization in both encounters.\n\n"
                "**Interpretation.** These systems can propose positions or trajectories, but none currently supplies defensible biological IDs through close contact in this historical interval."
            ),
        }
    )
    blocks["audit_conclusion"].update(
        {
            "sourceId": "human_audit_methods",
            "body": (
                "## Every method failed both end-to-end encounter tests\n\n"
                "Each method recorded 0/2 conservative successes. Multi-view's two failures were identity switches despite successful localization; idtracker.ai combined one off-fish failure with one visible switch; native UDMT was off-fish in both. The bars show these mutually exclusive primary failure outcomes rather than three uninformative zero-height success bars."
            ),
        }
    )
    uncertain_total = sum(row["continuity"] == "uncertain" for row in event_rows)
    low_confidence_total = sum(row["confidence_1_to_3"] == 1 for row in event_rows)
    blocks["audit_uncertainty"].update(
        {
            "sourceId": "human_audit_rows",
            "body": (
                "## Uncertainty is concentrated in the unusable UDMT clips\n\n"
                f"Continuity was uncertain in {uncertain_total}/6 clips and confidence was low in {low_confidence_total}/6. Both uncertain-continuity ratings were native UDMT clips whose labels were also off-fish. The multi-view and idtracker.ai continuity judgments were determinate, so their observed switches do not depend on treating uncertainty as failure."
            ),
        }
    )
    blocks["paired_comparison_note"].update(
        {
            "sourceId": "human_audit_pairs",
            "body": (
                "## The corrected audit cannot rank the methods statistically\n\n"
                "All pairwise comparisons are 0 wins, 0 losses, and 2 joint end-to-end failures, giving exact p=1.0. With only two independent encounters, inferential ranking would be inappropriate; the scientifically useful result is the method-specific failure mode."
            ),
        }
    )
    blocks["scope_and_definitions"].update(
        {
            "sourceId": "revised_audit_design",
            "body": (
                "## Seven detector fragments represent two biological encounters\n\n"
                "The scope remains one 4,999-frame, three-fish historical interval. The seven automatically detected close-approach fragments were consolidated into two encounters because crossings 01-06 lacked one full second of 100-pixel pair separation between fragments. This produced six blinded clips: two encounters × three methods. Every clip starts and ends with sustained clear separation. End-to-end success requires both localization=pass and continuity=pass."
            ),
        }
    )
    blocks["methodology"].update(
        {
            "body": (
                "## Three native methods, one corrected comparison unit\n\n"
                "idtracker.ai v6 used its native contrastive identity workflow on the cleaned top view; native UDMT used its XMem/ST-Net trajectory output; and the multi-view method used a frozen shared encoder with synchronized top and side crops plus one-to-one crossing assignments. All methods were evaluated on the same two separation-delimited encounters. Geometry selected encounter boundaries and measured identity-blind localization, but its labels were never treated as biological truth."
            )
        }
    )
    blocks["decision_rule"].update(
        {
            "sourceId": "revised_decision",
            "body": (
                "## The original 6/7 rule is superseded, not silently reused\n\n"
                "The pre-audit rule assumed seven independent crossings. Longer context showed before the revised ratings were unblinded that crossings 01-06 are fragments of one prolonged encounter. Counting them independently would be pseudo-replication, so the 6/7 threshold is not applied. No replacement pass threshold was chosen after seeing the ratings: the corrected report states the direct result that every method achieved 0/2 end-to-end successes."
            ),
        }
    )
    blocks["proxy_section"].update(
        {
            "sourceId": "revised_decision",
            "body": (
                "## Geometry agreement predicts localization, not biological identity\n\n"
                "The geometry-proxy IDF1 values remain 0.795 for multi-view, 0.599 for idtracker.ai, and 0.047 for native UDMT. Yet the highest-proxy multi-view method visibly switched identity in both corrected encounters. The proxy is useful for position and association diagnostics, but it cannot certify biological continuity."
            ),
        }
    )
    blocks["multiview_section"].update(
        {
            "sourceId": "revised_decision",
            "body": (
                "## Multi-view solves localization but not the crossing identity problem\n\n"
                "The multi-view method passed localization in both encounters, consistent with its 96.7% position recall and 90.3% all-three coverage at 80 px. The rater nevertheless observed an identity switch in both long-context clips. Appearance-link similarity thresholds therefore do not yet correspond to biological identity preservation."
            ),
        }
    )
    blocks["limitations"].update(
        {
            "body": (
                "## Two clear encounters answer this interval but not generalization\n\n"
                "- The audit covers one 4,999-frame interval from one recording and only two independent encounters.\n"
                "- With 0/2 successes, the Wilson 95% upper bound is still 65.8%; the sample is too small to estimate a population success rate or rank methods.\n"
                "- Ratings come from one independent rater.\n"
                "- The multi-view primary uses a shared camera encoder; separate top and side encoders remain an untested ablation.\n"
                "- Geometry provides event boundaries and localization reference only, never biological identity truth."
            )
        }
    )
    blocks["recommended_next_steps"].update(
        {
            "body": (
                "## Treat current identities as proposals around every close encounter\n\n"
                "1. Keep multi-view reconstruction for high-coverage localization, but flag every separation-delimited encounter for identity review before social-behavior analysis.\n"
                "2. Retrain the appearance linker with view-specific top/side encoders, segmentation-centered crops, and longer temporal memory, then rerun these unchanged two encounters as regression tests.\n"
                "3. Mine additional separation-delimited encounters from other historical recordings and add a second blinded rater before estimating success rates."
            )
        }
    )
    blocks["further_questions"].update(
        {
            "body": (
                "## Questions that could change the conclusion\n\n"
                "- Can view-specific encoders prevent the two multi-view swaps observed after separation?\n"
                "- Can idtracker-style appearance training improve continuity without losing complete localization?\n"
                "- Do the same failure modes recur across more fish, recordings, and independent raters?"
            )
        }
    )
    blocks["reproducibility"].update(
        {
            "body": (
                "## Reproducibility\n\n"
                "The revised worksheet, private blinding key, encounter definitions, codec-verification report, unblinded rows, summaries, corrected decision, and source hashes are copied into the report evidence. The original 6/7 rule is retained there with an explicit superseded filename rather than erased. Reusable code is under `scripts/experiments/identity_recovery_benchmark/`."
            )
        }
    )

    cards = {card["id"]: card for card in manifest["cards"]}
    cards["events_card"].update(
        {
            "description": "Independent encounters after sustained-separation consolidation; seven detector fragments are retained as provenance.",
            "sourceId": "revised_audit_design",
            "metrics": [
                {"label": "Independent encounters", "field": "events", "format": "number"},
                {"label": "Source fragments", "field": "source_fragments", "format": "number"},
            ],
        }
    )
    cards["audit_card"].update(
        {
            "description": "Best end-to-end result across the three methods on the corrected encounter units.",
            "dataset": "final_headline",
            "sourceId": "human_audit_methods",
            "metrics": [
                {"label": "Successful encounters", "field": "best_successful_encounters", "format": "number"},
                {"label": "Audited per method", "field": "audited_encounters", "format": "number"},
            ],
        }
    )

    audit_chart = next(chart for chart in manifest["charts"] if chart["id"] == "audit_success_chart")
    audit_chart.clear()
    audit_chart.update(
        {
            "id": "audit_success_chart",
            "title": "Human-audit outcome by method",
            "subtitle": "Two separation-delimited encounters per method; each bar totals two encounters.",
            "type": "stackedBar",
            "dataset": "audit_outcomes",
            "sourceId": "human_audit_rows",
            "valueFormat": "number",
            "palette": {"kind": "categorical"},
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "encounters", "type": "quantitative", "label": "Encounters"},
                "color": {"field": "outcome", "type": "nominal", "label": "Primary outcome"},
                "tooltip": [
                    {"field": "outcome", "type": "nominal", "label": "Primary outcome"},
                    {"field": "encounters", "type": "quantitative", "label": "Encounter count"},
                ],
            },
        }
    )
    assessment_table = next(table for table in manifest["tables"] if table["id"] == "final_assessment_table")
    assessment_table.update(
        {
            "title": "Corrected encounter-level assessment",
            "subtitle": "Two independent encounters per method; full-window localization metrics provide context.",
            "defaultSort": {"field": "localized_events", "direction": "desc"},
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "position_recall_80px", "label": "Position recall", "format": "percent"},
                {"field": "all_three_coverage_80px", "label": "All-three coverage", "format": "percent"},
                {"field": "localized_events", "label": "Localized encounters", "format": "number"},
                {"field": "continuity_switch", "label": "Visible switches", "format": "number"},
                {"field": "localization_fail", "label": "Localization failures", "format": "number"},
                {"field": "audit_successes", "label": "End-to-end successes", "format": "number"},
                {"field": "wilson_95_upper", "label": "Success 95% upper", "format": "percent"},
                {"field": "decision", "label": "Observed result", "type": "text"},
            ],
        }
    )
    pair_table = next(table for table in manifest["tables"] if table["id"] == "audit_pairwise_table")
    pair_table.update(
        {
            "title": "Paired encounter comparisons",
            "subtitle": "Both encounters were joint end-to-end failures for every method pair.",
        }
    )

    snapshot = artifact["snapshot"]
    snapshot["status"] = "ready"
    snapshot.pop("accessIssues", None)
    snapshot["datasets"]["headline"][0].update(
        {"events": event_count, "source_fragments": 7, "audit_clips": event_count * 3}
    )
    snapshot["datasets"]["final_headline"] = [
        {
            "best_successful_encounters": max(int(row["audit_successes"]) for row in assessments),
            "audited_encounters": event_count,
        }
    ]
    snapshot["datasets"]["final_assessment"] = assessments
    snapshot["datasets"]["audit_pairwise"] = pairwise
    snapshot["datasets"]["audit_event_rows"] = event_rows
    snapshot["datasets"]["audit_outcomes"] = outcome_rows

    chart_map = {
        "schema_version": "identity_recovery_chart_map_v2",
        "charts": [
            {"section": "Corrected human audit", "question": "Why did each method fail end-to-end identity recovery?", "family": "Composition", "type": "stackedBar", "fields": ["method", "outcome", "encounters"], "takeaway": "All methods scored 0/2, but multi-view failed by identity switch while native UDMT failed by localization.", "palette_policy": "relaxed multi-category"},
            {"section": "Fish localization", "question": "Do emitted labels land on real fish across matching tolerances?", "family": "Trend over ordered tolerance", "type": "line", "fields": ["gate_px", "position_recall", "method"], "takeaway": "Multi-view localizes fish best; native UDMT mostly drifts off fish.", "palette_policy": "relaxed multi-category"},
            {"section": "Geometry proxy", "question": "Does geometry-label agreement predict audited biological continuity?", "family": "Comparison", "type": "bar", "fields": ["method", "geometry_proxy_idf1"], "takeaway": "The highest proxy method still switched identity in both corrected encounters.", "palette_policy": "single-root preferred"},
            {"section": "idtracker.ai training", "question": "Did contrastive training reach the package target?", "family": "Trend", "type": "line", "fields": ["batch", "silhouette"], "takeaway": "Training remained below the 0.91 target.", "palette_policy": "single-root preferred"},
        ],
        "qa_note": "The outcome chart replaces zero-height success bars and uses mutually exclusive primary outcomes; exact values remain in the assessment table.",
    }
    (evidence_dir / "CHART_MAP.json").write_text(json.dumps(chart_map, indent=2) + "\n", encoding="utf-8")
    report_notes = {
        "audience": "technical, written for both ML/MV and general neuroscience readers",
        "delivery_mode": "portable self-contained HTML",
        "correction": "The prior 21-clip report is superseded because seven detector fragments were not seven independent encounters.",
        "required_structure_mapping": {
            "technical_summary": "answer_first",
            "key_findings": ["audit_conclusion", "localization_section", "proxy_section"],
            "scope_data_definitions": "scope_and_definitions",
            "methodology": ["methodology", "decision_rule", "multiview_section", "idtracker_section"],
            "limitations_robustness": ["audit_uncertainty", "paired_comparison_note", "limitations"],
            "recommended_next_steps": "recommended_next_steps",
            "further_questions": "further_questions",
        },
    }
    (evidence_dir / "REPORT_NOTES.json").write_text(json.dumps(report_notes, indent=2) + "\n", encoding="utf-8")
    artifact["package_info"]["originUrl"] = "artifact://historical-medaka-identity-corrected-encounter-audit"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def main(argv: list[str] | None = None) -> None:
    print(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
