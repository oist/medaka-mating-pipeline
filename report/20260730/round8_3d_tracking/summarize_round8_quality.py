from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_round8_quality import final_3d_metrics


EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def number(value: float | int, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def mapping_text(rows: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{row['fish_id']}→{row['final_fish_id']}" for row in rows
    )


def interval_text(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for fish_id, fish in metrics.items():
        intervals = fish["nonfinite_run_intervals"]
        if not intervals:
            continue
        rendered = ", ".join(
            f"{row['start_frame']}-{row['end_frame']}"
            f" ({row['length_frames']}f)"
            for row in intervals
        )
        parts.append(f"{fish_id}: {rendered}")
    return "; ".join(parts) if parts else "none"


def summary_row(experiment: str, result: dict[str, Any]) -> dict[str, Any]:
    prediction = result["prediction_qc"]
    views = prediction["views"]
    pre = result["pre_link_3d"]
    final = result["final_3d"]
    binding = result["identity_binding"]
    top = result["top_camera_2d_tracking"]
    steps = final["adjacent_displacement"]
    bounds = final["coordinate_sign_and_bounds"]
    residuals = pre["residuals"]
    return {
        "experiment": experiment,
        "review_status": result["quality_gate"]["status"],
        "prediction_structural_qc": result["quality_gate"][
            "prediction_structural_qc"
        ],
        "zero_detection_top_frames": views["top_cam"][
            "zero_detection_source_frames"
        ],
        "zero_detection_side1_frames": views["side1_cam"][
            "zero_detection_source_frames"
        ],
        "zero_detection_side2_frames": views["side2_cam"][
            "zero_detection_source_frames"
        ],
        "top_track_count": top["track_count"],
        "top_track_rows": top["rows"],
        "prelink_all3_valid_fraction": pre[
            "all_three_fish_valid_fraction"
        ],
        "prelink_residual_median_mm2": residuals["median"],
        "prelink_residual_p95_mm2": residuals["p95"],
        "prelink_residual_p99_mm2": residuals["p99"],
        "prelink_residual_max_mm2": residuals["max"],
        "frames_with_any_synthetic_view_fraction": pre[
            "frames_with_any_synthetic_view_fraction"
        ],
        "final_all3_finite_fraction": final[
            "all_three_finite_frame_fraction"
        ],
        "last_frame_without_all3_finite": final[
            "last_frame_without_all_three_finite"
        ],
        "post_frame5000_all3_finite_fraction": final[
            "all_three_finite_post_frame_5000"
        ]["all_three_finite_frame_fraction"],
        "final_all3_directly_observed_fraction": final[
            "all_three_directly_observed_frame_fraction"
        ],
        "interpolated_row_fraction": final["interpolated_row_fraction"],
        "adjacent_step_p95_mm": steps["step_mm_per_frame"]["p95"],
        "adjacent_step_max_mm": steps["step_mm_per_frame"]["max"],
        "adjacent_steps_ge7_5_mm": steps["steps_ge7_5_mm"],
        "adjacent_steps_gt8_mm": steps["jumps_gt8_mm"],
        "adjacent_steps_gt20_mm": steps["jumps_gt20_mm"],
        "top_cam_bound_fraction": binding["top_cam_2d_bound_fraction"],
        "bind_score_p95_px": binding["bind_score_px"]["p95"],
        "bind_score_p99_px": binding["bind_score_px"]["p99"],
        "bind_score_max_px": binding["bind_score_px"]["max"],
        "longest_geometry_only_run_frames": binding[
            "geometry_only_frame_runs"
        ]["longest_run_frames"],
        "stable_global_one_to_one_relabel": binding[
            "stable_global_one_to_one_relabel"
        ],
        "geometry_to_final_mapping": mapping_text(
            binding["geometry_fish_to_final_fish_mapping"]
        ),
        "observed_rows_with_any_negative": bounds[
            "observed_rows_with_any_negative"
        ],
        "rows_with_any_exact_zero": bounds["rows_with_any_exact_zero"],
        "rows_outside_minus20_to220_bounds": bounds[
            "rows_outside_requested_xyz_bounds_minus20_to220"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-json", required=True, type=Path)
    parser.add_argument("--windows-csv", required=True, type=Path)
    parser.add_argument("--windows-json", required=True, type=Path)
    parser.add_argument("--visual-manifest", required=True, type=Path)
    parser.add_argument("--round7-k2-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    quality = json.loads(args.quality_json.read_text(encoding="utf-8"))
    visual = json.loads(args.visual_manifest.read_text(encoding="utf-8"))
    rows = [
        summary_row(experiment, quality["experiments"][experiment])
        for experiment in EXPERIMENTS
    ]
    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    round7 = final_3d_metrics(args.round7_k2_csv)
    k2 = quality["experiments"]["20251225_K_1_2_FCMC"]["final_3d"]
    round7_delta_all3_pp = 100.0 * (
        k2["all_three_finite_frame_fraction"]
        - round7["all_three_finite_frame_fraction"]
    )
    round7_delta_step_p95_pct = 100.0 * (
        k2["adjacent_displacement"]["step_mm_per_frame"]["p95"]
        / round7["adjacent_displacement"]["step_mm_per_frame"]["p95"]
        - 1.0
    )

    lines = [
        "# Round8 SLEAP prediction and 3D tracking QC",
        "",
        f"- Tag: `{quality['tag']}`",
        f"- Quality JSON SHA-256: `{sha256(args.quality_json)}`",
        f"- Risk-window CSV SHA-256: `{sha256(args.windows_csv)}`",
        f"- Risk-window JSON SHA-256: `{sha256(args.windows_json)}`",
        "- Decision: **review**, not blanket pass. All execution/integrity "
        "checks succeeded, but biological identity at smooth contacts and "
        "crossings still requires visual review.",
        "",
        "## Cohort summary",
        "",
        "| Experiment | Pre-link all 3 | Residual p95 / p99 / max (mm²) | "
        "Final all 3 finite | All 3 directly observed | Step p95 / max (mm) | "
        "Top-ID bound | Longest geometry-only run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        result = quality["experiments"][experiment]
        pre = result["pre_link_3d"]
        final = result["final_3d"]
        bind = result["identity_binding"]
        residual = pre["residuals"]
        step = final["adjacent_displacement"]["step_mm_per_frame"]
        lines.append(
            f"| {experiment} | "
            f"{percent(pre['all_three_fish_valid_fraction'])} | "
            f"{number(residual['p95'])} / {number(residual['p99'])} / "
            f"{number(residual['max'])} | "
            f"{percent(final['all_three_finite_frame_fraction'])} | "
            f"{percent(final['all_three_directly_observed_frame_fraction'])} | "
            f"{number(step['p95'])} / {number(step['max'])} | "
            f"{percent(bind['top_cam_2d_bound_fraction'])} | "
            f"{bind['geometry_only_frame_runs']['longest_run_frames']} f |"
        )

    lines.extend(
        [
            "",
            "## Prediction integrity",
            "",
            "All 36 array children completed with exit `0:0`; all expected "
            "stderr logs were empty, no error strings were found, each view "
            "has four chunks spanning `[0, 348599]`, merged indices exactly "
            "equal the chunk union, and merged video references point to the "
            "canonical source videos. SLEAP stores only labeled frames, so "
            "`labeled + zero-detection = 348600` is the integrity invariant; "
            "the zero-detection counts below are biological-quality evidence, "
            "not merge failures.",
            "",
            "| Experiment | top zero-detection | side1 | side2 | "
            "top centroid tracks |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment']} | "
            f"{row['zero_detection_top_frames']} | "
            f"{row['zero_detection_side1_frames']} | "
            f"{row['zero_detection_side2_frames']} | "
            f"{row['top_track_count']} |"
        )

    lines.extend(
        [
            "",
            "## Final gap and continuity context",
            "",
            "Final nonfinite gaps are startup-only and are distinct from later "
            "not-directly-observed/interpolated geometry runs. After frame "
            "4834, all three experiments retain all three finite trajectories "
            "through frame 348599; the post-frame-5000 all-three-finite "
            "fraction is exactly 100% for every experiment.",
            "",
        ]
    )
    for experiment in EXPERIMENTS:
        final = quality["experiments"][experiment]["final_3d"]
        lines.append(
            f"- `{experiment}`: last frame without all three finite = "
            f"{final['last_frame_without_all_three_finite']}; nonfinite "
            f"intervals: {interval_text(final['per_fish_gap_metrics'])}."
        )

    lines.extend(
        [
            "",
            "All accepted adjacent steps are below 8 mm because the canonical "
            "linker uses an 8 mm adjacent-frame gate. Therefore zero >20 mm "
            "steps is expected bounded-continuity evidence, not an independent "
            "identity proof. The near-gate counts (≥7.5 mm) remain the useful "
            "motion-risk signal.",
            "",
            "| Experiment | ≥7.5 mm steps | >8 mm | Exact-zero rows | "
            "Observed negative rows | Out-of-bounds rows | Stable mapping |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['adjacent_steps_ge7_5_mm']} | "
            f"{row['adjacent_steps_gt8_mm']} | "
            f"{row['rows_with_any_exact_zero']} | "
            f"{row['observed_rows_with_any_negative']} | "
            f"{row['rows_outside_minus20_to220_bounds']} | "
            f"{row['geometry_to_final_mapping']} |"
        )

    lines.extend(
        [
            "",
            "The observed negative coordinates and zero exact-zero rows "
            "confirm that `--no-shift-to-positive` was honored after fixing "
            "the spline-fill clipping defect. A stable global one-to-one "
            "mapping is structural evidence only; it cannot establish "
            "biological identity through smooth contacts/crossings.",
            "",
            "## Round7 comparison",
            "",
            "Only K2 has a directly available Round7 refractive 3D CSV in the "
            "canonical reconstruction tree. Round8 K2 changes all-three-finite "
            f"coverage by {round7_delta_all3_pp:+.3f} percentage points and "
            "adjacent-step p95 by "
            f"{round7_delta_step_p95_pct:+.1f}% versus that file. K1/K3 have "
            "no Round7 reconstruction directory there. Stage-2 identity "
            "comparisons are not apples-to-apples because the older tracks "
            "were mouth-anchored and fragmented, whereas Round8 uses the "
            "default visible-node centroid.",
            "",
            f"- Round7 K2 source SHA-256: `{sha256(args.round7_k2_csv)}`",
            "",
            "## Visual-review sampling",
            "",
            "The 61-window manifest is a stratified review queue covering "
            "pre-link residual extremes, low top-camera separation, high "
            "adjacent 3D steps, long geometry-only runs, normal start/mid/end "
            "controls, all array chunk boundaries, and the K2 ambiguous-extra "
            "reference. Those are risk-ranked windows, not confirmed errors. "
            "Small rendered overlays and per-experiment visual decisions are "
            "recorded in the separate visual-QC manifest.",
            "",
            f"- Visual-QC manifest: `{args.visual_manifest}`",
            f"- Visual-QC manifest SHA-256: `{sha256(args.visual_manifest)}`",
            f"- Hashed rendered media: {len(visual['rendered_media'])}",
            "- K1 f198209: inconsistent gray candidates were excluded; final "
            "ID 6 is conservatively interpolated while IDs 13/14 remain "
            "direct and plausible. No obvious sampled identity swap.",
            "- K2 f174290–174320: gray extras are visible early and filtered; "
            "the three selected identities remain geometrically plausible.",
            "- K3 f96431–96461: the close pair preserves sampled ordering in "
            "all views and 3D, without a visible discontinuity.",
            "",
            "## Interpretation",
            "",
            "- K1 has the best binding fraction in the cohort and high finite "
            "coverage, but isolated residual extremes require review.",
            "- K2 is the weakest coverage/startup-gap case; its final "
            "trajectories become fully finite after frame 4834.",
            "- K3 has the highest residual/motion/binding risk: the largest "
            "residual maximum, highest synthetic-view exposure, lowest "
            "top-ID bound fraction, and most near-gate steps.",
            "- No residual gate was applied. "
            "`dropped_high_residual_fish_rows = 0` means no candidate was "
            "removed by an explicit residual threshold; it must not be read "
            "as evidence that every high-residual candidate was accepted into "
            "the final linked trajectory. Residual tails and the worst-frame "
            "visual windows drive review.",
            "- No labeled ground-truth pose or identity annotations were used. "
            "Prediction QC is structural/count integrity, and 3D QC measures "
            "geometry/continuity—not supervised accuracy.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "csv": str(args.output_csv),
                "csv_sha256": sha256(args.output_csv),
                "markdown": str(args.output_md),
                "markdown_sha256": sha256(args.output_md),
                "round7_k2_delta_all3_pp": round7_delta_all3_pp,
                "round7_k2_delta_step_p95_pct": round7_delta_step_p95_pct,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
