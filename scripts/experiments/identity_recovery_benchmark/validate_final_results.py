from __future__ import annotations

"""Independently validate the completed audit, locked decision, and evidence hashes."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import NormalDist


JOIN_COLUMNS = ("event_id", "blinded_method", "clip")
METHOD_LABELS = {
    "idtracker_ai_v6": "idtracker.ai v6",
    "native_udmt": "Native UDMT",
    "multiview_constrained_contrastive": "Multi-view contrastive",
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


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ))


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    lower, upper = max(0.0, center - half), min(1.0, center + half)
    return (0.0 if lower < 1e-15 else lower), (1.0 if 1 - upper < 1e-15 else upper)


def _exact_two_sided(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-worksheet", type=Path, required=True)
    parser.add_argument("--blinding-key", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--audit-results", type=Path, required=True)
    parser.add_argument("--final-decision", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    worksheet = _read_csv(args.completed_worksheet)
    key_rows = _read_csv(args.blinding_key)
    benchmark = _read_json(args.benchmark_summary)
    audit_results = _read_json(args.audit_results)
    final_decision = _read_json(args.final_decision)

    assert len(worksheet) == len(key_rows) == 21
    worksheet_keys = [tuple(row[column] for column in JOIN_COLUMNS) for row in worksheet]
    key_keys = [tuple(row[column] for column in JOIN_COLUMNS) for row in key_rows]
    assert len(set(worksheet_keys)) == len(worksheet_keys)
    assert len(set(key_keys)) == len(key_keys)
    assert set(worksheet_keys) == set(key_keys)
    assert all(row["continuity"] in {"pass", "switch", "uncertain"} for row in worksheet)
    assert all(row["localization"] in {"pass", "fail", "uncertain"} for row in worksheet)
    assert all(row["confidence_1_to_3"] in {"1", "2", "3"} for row in worksheet)

    key_by_join = {
        tuple(row[column] for column in JOIN_COLUMNS): row["method"] for row in key_rows
    }
    unblinded = []
    for row in worksheet:
        joined = dict(row)
        joined["method"] = key_by_join[tuple(row[column] for column in JOIN_COLUMNS)]
        joined["conservative_success"] = (
            row["continuity"] == "pass" and row["localization"] == "pass"
        )
        unblinded.append(joined)
    methods = sorted({row["method"] for row in unblinded})
    assert set(methods) == set(METHOD_LABELS)

    recomputed_methods = []
    event_scores: dict[str, dict[str, bool]] = {}
    for method in methods:
        group = [row for row in unblinded if row["method"] == method]
        assert len(group) == 7
        successes = sum(row["conservative_success"] for row in group)
        lower, upper = _wilson(successes, len(group))
        recomputed_methods.append(
            {
                "method": method,
                "events_scored": len(group),
                "conservative_successes": successes,
                "conservative_success_fraction": successes / len(group),
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "continuity": dict(Counter(row["continuity"] for row in group)),
                "localization": dict(Counter(row["localization"] for row in group)),
                "mean_confidence": sum(int(row["confidence_1_to_3"]) for row in group) / len(group),
            }
        )
        for row in group:
            event_scores.setdefault(row["event_id"], {})[method] = bool(
                row["conservative_success"]
            )
    assert len(event_scores) == 7
    assert all(set(scores) == set(methods) for scores in event_scores.values())

    reported_by_method = {row["method"]: row for row in audit_results["methods"]}
    for row in recomputed_methods:
        reported = reported_by_method[row["method"]]
        assert int(reported["events_scored"]) == row["events_scored"]
        assert int(reported["conservative_successes"]) == row["conservative_successes"]
        for field in [
            "conservative_success_fraction",
            "wilson_95_lower",
            "wilson_95_upper",
            "mean_confidence",
        ]:
            assert _close(float(reported[field]), float(row[field]))

    recomputed_pairs = []
    for method_a, method_b in combinations(methods, 2):
        pairs = [(scores[method_a], scores[method_b]) for scores in event_scores.values()]
        a_wins = sum(left and not right for left, right in pairs)
        b_wins = sum(not left and right for left, right in pairs)
        both_pass = sum(left and right for left, right in pairs)
        both_fail = sum(not left and not right for left, right in pairs)
        recomputed_pairs.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_events": len(pairs),
                "method_a_wins": a_wins,
                "method_b_wins": b_wins,
                "both_pass": both_pass,
                "both_fail": both_fail,
                "paired_success_rate_difference_a_minus_b": (a_wins - b_wins) / len(pairs),
                "exact_mcnemar_p_two_sided": _exact_two_sided(a_wins, b_wins),
            }
        )
    reported_pairs = {
        (row["method_a"], row["method_b"]): row
        for row in audit_results["pairwise_comparisons"]
    }
    for row in recomputed_pairs:
        reported = reported_pairs[(row["method_a"], row["method_b"])]
        for field in [
            "paired_events",
            "method_a_wins",
            "method_b_wins",
            "both_pass",
            "both_fail",
        ]:
            assert int(reported[field]) == int(row[field])
        for field in [
            "paired_success_rate_difference_a_minus_b",
            "exact_mcnemar_p_two_sided",
        ]:
            assert _close(float(reported[field]), float(row[field]))

    success_by_method = {
        row["method"]: int(row["conservative_successes"]) for row in recomputed_methods
    }
    decision_by_method = {row["method_key"]: row for row in final_decision["methods"]}
    rule = final_decision["locked_rule"]
    recovered_methods = []
    for method, report in benchmark["methods"].items():
        gate80 = report["localization_identity_blind"]["80"]
        recall = float(gate80["position_recall"])
        coverage = float(gate80["all_fish_frame_coverage"])
        successes = success_by_method[method]
        localization_pass = (
            recall >= float(rule["minimum_position_recall_80px"])
            and coverage >= float(rule["minimum_all_three_coverage_80px"])
        )
        if localization_pass and successes >= int(rule["minimum_conservative_audit_successes_out_of_7"]):
            expected_decision = "recovered in locked interval"
            recovered_methods.append(method)
        elif localization_pass and successes >= 4:
            expected_decision = "promising but not established"
        elif not localization_pass:
            expected_decision = "not end-to-end recovered: localization gate failed"
        else:
            expected_decision = "not recovered on audited crossings"
        actual = decision_by_method[method]
        assert actual["decision"] == expected_decision
        assert int(actual["audit_successes"]) == successes
        assert _close(float(actual["position_recall_80px"]), recall)
        assert _close(float(actual["all_three_coverage_80px"]), coverage)
    assert not recovered_methods
    expected_conclusion = (
        "No method meets the locked end-to-end recovery criterion in this historical interval. "
        "Apparent label persistence is insufficient when localization or audited biological continuity fails."
    )
    assert final_decision["conclusion"] == expected_conclusion

    path_by_name = {
        args.completed_worksheet.name: args.completed_worksheet,
        args.blinding_key.name: args.blinding_key,
        args.benchmark_summary.name: args.benchmark_summary,
        args.decision_rule.name: args.decision_rule,
    }
    for path in args.audit_results.parent.iterdir():
        if path.is_file():
            path_by_name[path.name] = path
    for name, expected_hash in final_decision["audit_evidence_sha256"].items():
        assert name in path_by_name
        assert _sha256(path_by_name[name]) == expected_hash

    report = {
        "schema_version": "historical_identity_recovery_validation_v1",
        "status": "passed",
        "checks": {
            "worksheet_rows": 21,
            "unique_join_keys": 21,
            "key_set_exact_match": True,
            "allowed_values_valid": True,
            "methods": len(methods),
            "events_per_method": 7,
            "method_metrics_recomputed": True,
            "wilson_intervals_recomputed": True,
            "paired_exact_tests_recomputed": True,
            "locked_decision_recomputed": True,
            "evidence_hashes_verified": len(final_decision["audit_evidence_sha256"]),
        },
        "recomputed_methods": recomputed_methods,
        "recomputed_pairwise": recomputed_pairs,
        "conclusion": expected_conclusion,
        "completed_worksheet_sha256": _sha256(args.completed_worksheet),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
