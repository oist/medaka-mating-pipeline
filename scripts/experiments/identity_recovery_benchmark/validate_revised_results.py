from __future__ import annotations

"""Independently validate the separation-context audit and corrected decision."""

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


JOIN_COLUMNS = ["event_id", "blinded_method", "clip"]
REQUIRED_COLUMNS = JOIN_COLUMNS + [
    "continuity",
    "localization",
    "confidence_1_to_3",
    "notes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--blank-worksheet", type=Path, required=True)
    parser.add_argument("--blinding-key", type=Path, required=True)
    parser.add_argument("--encounters", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--audit-results", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    worksheet = pd.read_csv(args.worksheet, keep_default_na=False)
    blank = pd.read_csv(args.blank_worksheet, keep_default_na=False)
    key = pd.read_csv(args.blinding_key, keep_default_na=False)
    encounters = pd.read_csv(args.encounters)
    benchmark = _read_json(args.benchmark)
    audit = _read_json(args.audit_results)
    decision = _read_json(args.decision)

    assert list(worksheet.columns) == REQUIRED_COLUMNS
    assert len(worksheet) == 6
    assert not worksheet.duplicated(JOIN_COLUMNS).any()
    assert set(map(tuple, worksheet[JOIN_COLUMNS].to_records(index=False))) == set(
        map(tuple, blank[JOIN_COLUMNS].to_records(index=False))
    )
    assert set(map(tuple, worksheet[JOIN_COLUMNS].to_records(index=False))) == set(
        map(tuple, key[JOIN_COLUMNS].to_records(index=False))
    )
    assert set(worksheet["continuity"]) <= {"pass", "switch", "uncertain"}
    assert set(worksheet["localization"]) <= {"pass", "fail", "uncertain"}
    assert set(worksheet["confidence_1_to_3"].astype(str)) <= {"1", "2", "3"}
    assert not (worksheet[["continuity", "localization", "confidence_1_to_3"]] == "").any().any()

    assert len(encounters) == 2
    source_fragments = {
        item
        for value in encounters["source_event_ids"].astype(str)
        for item in value.split(";")
    }
    assert source_fragments == {f"crossing_{index:02d}" for index in range(1, 8)}
    assert (encounters["separation_threshold_px"] == 100.0).all()
    assert (encounters["separation_hold_frames"] == 166).all()
    assert (encounters["separation_hold_seconds"] == 1.0).all()
    assert (encounters["pre_min_distance_px"] >= encounters["separation_threshold_px"]).all()
    assert (encounters["post_min_distance_px"] >= encounters["separation_threshold_px"]).all()
    assert (encounters["clip_start_frame"] < encounters["start_frame"]).all()
    assert (encounters["clip_end_frame"] > encounters["end_frame"]).all()

    merged = worksheet.merge(key, on=JOIN_COLUMNS, how="left", validate="one_to_one")
    assert merged["method"].notna().all()
    merged["success"] = (merged["continuity"] == "pass") & (merged["localization"] == "pass")
    reported = {row["method"]: row for row in audit["methods"]}
    recomputed_methods: list[dict[str, object]] = []
    for method, group in merged.groupby("method", sort=True):
        successes = int(group["success"].sum())
        total = int(len(group))
        lower, upper = _wilson(successes, total)
        expected = reported[method]
        assert total == 2
        assert successes == int(expected["conservative_successes"])
        assert math.isclose(successes / total, float(expected["conservative_success_fraction"]), abs_tol=1e-12)
        assert math.isclose(lower, float(expected["wilson_95_lower"]), abs_tol=1e-12)
        assert math.isclose(upper, float(expected["wilson_95_upper"]), abs_tol=1e-12)
        continuity = group["continuity"].value_counts().to_dict()
        localization = group["localization"].value_counts().to_dict()
        for value in ("pass", "switch", "uncertain"):
            assert int(expected[f"continuity_{value}"]) == int(continuity.get(value, 0))
        assert int(expected["localization_fail"]) == int(localization.get("fail", 0))
        assert int(expected["localization_uncertain"]) == int(localization.get("uncertain", 0))
        recomputed_methods.append(
            {
                "method": method,
                "events": total,
                "successes": successes,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "continuity": continuity,
                "localization": localization,
            }
        )

    event_scores = merged.pivot(index="event_id", columns="method", values="success")
    reported_pairs = {
        (row["method_a"], row["method_b"]): row
        for row in audit["pairwise_comparisons"]
    }
    recomputed_pairs: list[dict[str, object]] = []
    for method_a, method_b in combinations(sorted(event_scores.columns), 2):
        paired = event_scores[[method_a, method_b]].astype(bool)
        a_wins = int((paired[method_a] & ~paired[method_b]).sum())
        b_wins = int((~paired[method_a] & paired[method_b]).sum())
        both_pass = int((paired[method_a] & paired[method_b]).sum())
        both_fail = int((~paired[method_a] & ~paired[method_b]).sum())
        discordant = a_wins + b_wins
        exact_p = float(binomtest(a_wins, discordant, p=0.5).pvalue) if discordant else 1.0
        actual = reported_pairs[(method_a, method_b)]
        assert (a_wins, b_wins, both_pass, both_fail) == (
            int(actual["method_a_wins"]),
            int(actual["method_b_wins"]),
            int(actual["both_pass"]),
            int(actual["both_fail"]),
        )
        assert math.isclose(exact_p, float(actual["exact_mcnemar_p_two_sided"]), abs_tol=1e-12)
        recomputed_pairs.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "both_pass": both_pass,
                "both_fail": both_fail,
                "exact_p": exact_p,
            }
        )

    assert all(int(row["successes"]) == 0 for row in recomputed_methods)
    expected_conclusion = (
        "No method achieved end-to-end biological identity recovery in either of the "
        "2 separation-delimited encounters (2 encounters per method)."
    )
    assert decision["conclusion"] == expected_conclusion
    assert decision["audit_unit"]["independent_separation_delimited_encounters"] == 2
    assert "superseded" in decision["original_rule_status"]
    benchmark_methods = benchmark["methods"]
    for row in decision["methods"]:
        method = row["method_key"]
        gate = benchmark_methods[method]["localization_identity_blind"]["80"]
        assert math.isclose(float(row["position_recall_80px"]), float(gate["position_recall"]), abs_tol=1e-12)
        assert math.isclose(float(row["all_three_coverage_80px"]), float(gate["all_fish_frame_coverage"]), abs_tol=1e-12)
        assert int(row["audit_successes"]) == 0

    for name, expected_hash in decision["audit_evidence_sha256"].items():
        evidence_path = args.evidence_dir / name
        assert evidence_path.is_file(), name
        assert _sha256(evidence_path) == expected_hash, name

    report = {
        "schema_version": "historical_identity_recovery_revised_validation_v2",
        "status": "passed",
        "checks": {
            "worksheet_rows": len(worksheet),
            "worksheet_key_set_exact": True,
            "allowed_values_valid": True,
            "encounters": len(encounters),
            "source_fragments_reconciled": len(source_fragments),
            "sustained_separation_valid": True,
            "method_metrics_recomputed": True,
            "wilson_intervals_recomputed": True,
            "paired_exact_tests_recomputed": True,
            "localization_metrics_reconciled": True,
            "evidence_hashes_verified": len(decision["audit_evidence_sha256"]),
        },
        "recomputed_methods": recomputed_methods,
        "recomputed_pairwise": recomputed_pairs,
        "conclusion": expected_conclusion,
        "completed_worksheet_sha256": _sha256(args.worksheet),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
