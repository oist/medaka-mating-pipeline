from __future__ import annotations

"""Unblind a completed crossing audit and compute conservative method scores."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    if lower < 1e-15:
        lower = 0.0
    if 1.0 - upper < 1e-15:
        upper = 1.0
    return lower, upper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--blinding-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def _json_records(table: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in table.to_dict("records"):
        records.append(
            {
                key: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in record.items()
            }
        )
    return records


def run(args: argparse.Namespace) -> dict[str, object]:
    worksheet = pd.read_csv(args.worksheet, keep_default_na=False)
    key = pd.read_csv(args.blinding_key, keep_default_na=False)
    join = ["event_id", "blinded_method", "clip"]
    missing_columns = set(join + ["continuity", "localization", "confidence_1_to_3"]) - set(worksheet)
    if missing_columns:
        raise ValueError(f"worksheet missing {sorted(missing_columns)}")
    if key.duplicated(join).any() or worksheet.duplicated(join).any():
        raise ValueError("audit rows or blinding keys are not unique")
    worksheet_keys = set(map(tuple, worksheet[join].itertuples(index=False, name=None)))
    blinding_keys = set(map(tuple, key[join].itertuples(index=False, name=None)))
    if worksheet_keys != blinding_keys:
        missing = len(blinding_keys - worksheet_keys)
        unexpected = len(worksheet_keys - blinding_keys)
        raise ValueError(
            "worksheet and blinding key do not contain the same audit rows "
            f"(missing={missing}, unexpected={unexpected})"
        )
    merged = worksheet.merge(key, on=join, how="left", validate="one_to_one")
    if merged["method"].eq("").any() or merged["method"].isna().any():
        raise ValueError("some audit rows have no blinding-key match")
    merged["continuity"] = merged["continuity"].str.strip().str.lower()
    merged["localization"] = merged["localization"].str.strip().str.lower()
    confidence = pd.to_numeric(merged["confidence_1_to_3"], errors="coerce")
    valid_continuity = {"pass", "switch", "uncertain"}
    valid_localization = {"pass", "fail", "uncertain"}
    valid_confidence = confidence.isin([1, 2, 3])
    incomplete = (
        ~merged["continuity"].isin(valid_continuity)
        | ~merged["localization"].isin(valid_localization)
        | ~valid_confidence
    )
    if incomplete.any() and not args.allow_incomplete:
        raise ValueError(f"{int(incomplete.sum())} audit rows are incomplete or invalid")
    merged["audit_row_valid"] = ~incomplete
    merged["conservative_success"] = False
    merged.loc[~incomplete, "conservative_success"] = (
        merged.loc[~incomplete, "continuity"].eq("pass")
        & merged.loc[~incomplete, "localization"].eq("pass")
    )
    scored = merged[~incomplete].copy()
    scored["confidence_1_to_3"] = confidence[~incomplete].astype(int)
    rows = []
    for method, group in scored.groupby("method", sort=True):
        successes = int(group["conservative_success"].sum())
        total = int(len(group))
        localized = group[group["localization"].eq("pass")]
        conditional_passes = int(localized["continuity"].eq("pass").sum())
        lower, upper = wilson_interval(successes, total)
        rows.append(
            {
                "method": method,
                "events_scored": total,
                "conservative_successes": successes,
                "conservative_success_fraction": successes / total if total else np.nan,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "continuity_pass": int(group["continuity"].eq("pass").sum()),
                "continuity_switch": int(group["continuity"].eq("switch").sum()),
                "continuity_uncertain": int(group["continuity"].eq("uncertain").sum()),
                "localization_fail": int(group["localization"].eq("fail").sum()),
                "localization_uncertain": int(group["localization"].eq("uncertain").sum()),
                "mean_confidence": float(group["confidence_1_to_3"].mean()),
                "localized_events": int(len(localized)),
                "conditional_continuity_pass_fraction": (
                    conditional_passes / len(localized) if len(localized) else np.nan
                ),
            }
        )
    summary_table = pd.DataFrame(rows)
    pairwise_rows = []
    event_scores = scored.pivot(
        index="event_id",
        columns="method",
        values="conservative_success",
    )
    for method_a, method_b in combinations(sorted(event_scores.columns), 2):
        paired = event_scores[[method_a, method_b]].dropna().astype(bool)
        a_wins = int((paired[method_a] & ~paired[method_b]).sum())
        b_wins = int((~paired[method_a] & paired[method_b]).sum())
        both_pass = int((paired[method_a] & paired[method_b]).sum())
        both_fail = int((~paired[method_a] & ~paired[method_b]).sum())
        discordant = a_wins + b_wins
        exact_p = (
            float(binomtest(a_wins, discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        )
        pairwise_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "paired_events": int(len(paired)),
                "method_a_wins": a_wins,
                "method_b_wins": b_wins,
                "both_pass": both_pass,
                "both_fail": both_fail,
                "paired_success_rate_difference_a_minus_b": float(
                    paired[method_a].mean() - paired[method_b].mean()
                ),
                "exact_mcnemar_p_two_sided": exact_p,
            }
        )
    pairwise_table = pd.DataFrame(pairwise_rows)
    event_count = int(scored["event_id"].nunique())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "audit_unblinded_rows.csv", index=False)
    summary_table.to_csv(args.output_dir / "audit_method_summary.csv", index=False)
    pairwise_table.to_csv(args.output_dir / "audit_pairwise_comparisons.csv", index=False)
    report = {
        "schema_version": "blinded_crossing_audit_results_v1",
        "primary_metric": (
            "fraction of audited events rated continuity=pass and localization=pass; "
            "uncertain and off-fish clips are retained as non-successes"
        ),
        "total_rows": int(len(merged)),
        "scored_rows": int(len(scored)),
        "incomplete_rows": int(incomplete.sum()),
        "methods": _json_records(summary_table),
        "pairwise_comparisons": _json_records(pairwise_table),
        "pairwise_test": (
            "two-sided exact McNemar test implemented as an exact binomial test "
            f"on discordant event pairs; descriptive because the audit has {event_count} "
            f"event{'s' if event_count != 1 else ''}"
        ),
        "event_units": event_count,
        "limitation": (
            f"single-rater audit with {event_count} event"
            f"{'s' if event_count != 1 else ''}; add recordings and independent raters "
            "before making a population-level claim"
        ),
    }
    (args.output_dir / "audit_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
