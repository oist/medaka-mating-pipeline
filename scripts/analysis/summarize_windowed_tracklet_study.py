"""Assemble final machine-readable tables for the 2026-07-17 study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ITEMS = {
    1: "overlapping temporal windows",
    2: "multiframe cross-view grouping",
    3: "explicit broken-fragment links",
    4: "conditional tail continuity",
    5: "global trajectory solver",
    6: "deferred identity decisions",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_qc(path: Path, method: str, cohort: str) -> dict[str, object]:
    table = pd.read_csv(path)
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(xyz), axis=1)
    observed = finite & ~table.get(
        "is_interpolated", pd.Series(False, index=table.index)
    ).fillna(False).to_numpy(dtype=bool)
    frame_count = int(table.frame.max() - table.frame.min() + 1)
    finite_by_frame = pd.Series(finite).groupby(table.frame).sum()
    observed_by_frame = pd.Series(observed).groupby(table.frame).sum()
    ordered = table.sort_values(["fish_id", "frame"])
    delta = ordered.groupby("fish_id", dropna=True)[["X", "Y", "Z"]].diff()
    gaps = ordered.groupby("fish_id", dropna=True).frame.diff()
    jumps = np.linalg.norm(delta.to_numpy(dtype=float), axis=1)
    consecutive = jumps[(gaps.to_numpy(dtype=float) == 1) & np.isfinite(jumps)]
    residual = pd.to_numeric(
        table.get("ray_residual_mm2", pd.Series(dtype=float)), errors="coerce"
    ).dropna().to_numpy(dtype=float)
    return {
        "method": method,
        "cohort": cohort,
        "frame_start": int(table.frame.min()),
        "frame_end": int(table.frame.max()),
        "rows": int(len(table)),
        "finite_row_share": float(finite.mean()),
        "direct_observed_row_share": float(observed.mean()),
        "all_three_finite_frame_share": float((finite_by_frame >= 3).sum() / frame_count),
        "all_three_direct_frame_share": float((observed_by_frame >= 3).sum() / frame_count),
        "jump_count": int(len(consecutive)),
        "jump_p95_mm": float(np.quantile(consecutive, 0.95)) if len(consecutive) else np.nan,
        "jump_p99_mm": float(np.quantile(consecutive, 0.99)) if len(consecutive) else np.nan,
        "jump_max_mm": float(np.max(consecutive)) if len(consecutive) else np.nan,
        "jump_over_8mm_count": int((consecutive > 8).sum()),
        "jump_over_8mm_share": float((consecutive > 8).mean()) if len(consecutive) else np.nan,
        "residual_p50_mm2": float(np.quantile(residual, 0.50)) if len(residual) else np.nan,
        "residual_p95_mm2": float(np.quantile(residual, 0.95)) if len(residual) else np.nan,
        "residual_p99_mm2": float(np.quantile(residual, 0.99)) if len(residual) else np.nan,
        "output_bytes": int(path.stat().st_size),
        "output_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("report/20260717/windowed_tracklet_stitching"),
    )
    args = parser.parse_args()
    root = args.report_root.resolve()

    baseline = load_json(root / "bench_baseline_final_32264_33263/run_summary.json")
    item1_single = load_json(root / "item1_final_single_32264_35364/run_summary.json")
    item1_resume = load_json(root / "item1_final_overlap8_32264_35364/run_summary.json")
    item1_frozen_overlap = load_json(root / "pair_item1_only_frozen_overlap_32264_33263/run_summary.json")
    item1_legacy_overlap = load_json(root / "pair_item1_only_legacy_overlap_32264_33263/run_summary.json")
    item1_item2_overlap = load_json(root / "pair_item1_item2_overlap_32264_33263/run_summary.json")
    item1_item3_overlap = load_json(root / "pair_item1_item3_overlap_32264_33263/run_summary.json")
    item1_item4_overlap = load_json(root / "pair_item1_item4_overlap_32264_33263/run_summary.json")
    frozen_frame = load_json(root / "bench_frozen_framewise_32264_33263/run_summary.json")
    frozen_multi = load_json(root / "bench_frozen_multiframe_32264_33263/run_summary.json")
    item3 = load_json(root / "bench_item3_mouth_v2_32264_33263/run_summary.json")
    item4 = load_json(root / "bench_item4_tail_v2_32264_33263/run_summary.json")
    item2_item3 = load_json(root / "pair_item2_item3_32264_33263/run_summary.json")
    item2_item4 = load_json(root / "pair_item2_item4_32264_33263/run_summary.json")
    solver_seq = load_json(root / "item5_same_graph/sequential/solver_summary.json")
    solver_beam = load_json(root / "item5_same_graph/beam32/solver_summary.json")
    solver_global = load_json(root / "item5_same_graph/global/solver_summary.json")
    immediate = load_json(root / "item6_same_graph/immediate/solver_summary.json")
    deferred = load_json(root / "item6_same_graph/deferred/solver_summary.json")
    item1_single_310 = load_json(root / "smoke_v2_single310/run_summary.json")
    item1_overlap_310 = load_json(root / "smoke_v2_overlap/run_summary.json")
    item1_item5_overlap = load_json(root / "pair_item1_item5_overlap_global_72000_72309/solver_summary.json")
    item1_item6_overlap = load_json(root / "pair_item1_item6_overlap_deferred_72000_72309/solver_summary.json")
    tail_global = load_json(root / "pair_item4_item5_global/solver_summary.json")
    tail_deferred = load_json(root / "pair_item4_item6_deferred/solver_summary.json")
    global_deferred = load_json(root / "pair_item5_item6_global_deferred/solver_summary.json")
    item2_deferred = load_json(root / "pair_item2_item6_deferred/solver_summary.json")
    bundle = load_json(root / "bundle_1_3_4_5_6_global_deferred/solver_summary.json")
    overlap_bundle_graph = load_json(root / "bundle_overlap_item1_3_4_graph_72000_72309/run_summary.json")
    overlap_bundle_global = load_json(root / "bundle_overlap_item1_3_4_5_global_72000_72309/solver_summary.json")
    overlap_bundle_global_deferred = load_json(root / "bundle_overlap_item1_3_4_5_6_global_deferred_72000_72309/solver_summary.json")
    full = load_json(root / "full_optimized_baseline_v2/run_summary.json")
    hardware = load_json(root / "hardware_benchmark.json")
    quality_json = load_json(root / "final_full_quality_comparison/comparison_summary.json")

    # Individual decisions use the strictest conclusion shared by the primary
    # implementation evidence and the independent verifier.
    individual = pd.DataFrame(
        [
            {
                "item": 1,
                "component": ITEMS[1],
                "cohort": "32264-35364",
                "baseline_runtime_seconds": item1_single["total_wall_time_seconds"],
                "item_runtime_seconds": 25.73085039999569,
                "speedup": item1_single["total_wall_time_seconds"] / 25.73085039999569,
                "time_saved_seconds": item1_single["total_wall_time_seconds"] - 25.73085039999569,
                "peak_rss_bytes": item1_resume["peak_rss_bytes"],
                "direct_rows": item1_resume["direct_rows"],
                "fragment_count": item1_resume["fragment_count"],
                "unresolved_interval_count": 0,
                "hypotheses_peak": 1,
                "deterministic": True,
                "scientific_decision": "retain execution infrastructure; reject integrated candidate",
                "evidence": "item1_final_single_32264_35364; item1_final_overlap8_32264_35364; full_optimized_baseline_v2",
            },
            {
                "item": 2,
                "component": ITEMS[2],
                "cohort": "32264-33263",
                "baseline_runtime_seconds": frozen_frame["total_wall_time_seconds"],
                "item_runtime_seconds": frozen_multi["total_wall_time_seconds"],
                "speedup": frozen_frame["total_wall_time_seconds"] / frozen_multi["total_wall_time_seconds"],
                "time_saved_seconds": frozen_frame["total_wall_time_seconds"] - frozen_multi["total_wall_time_seconds"],
                "peak_rss_bytes": frozen_multi["peak_rss_bytes"],
                "direct_rows": frozen_multi["direct_rows"],
                "fragment_count": frozen_multi["fragment_count"],
                "unresolved_interval_count": 0,
                "hypotheses_peak": 1,
                "deterministic": True,
                "scientific_decision": "reject current thresholds",
                "evidence": "bench_frozen_framewise_32264_33263; bench_frozen_multiframe_32264_33263",
            },
            {
                "item": 3,
                "component": ITEMS[3],
                "cohort": "32264-33263",
                "baseline_runtime_seconds": baseline["total_wall_time_seconds"],
                "item_runtime_seconds": item3["total_wall_time_seconds"],
                "speedup": baseline["total_wall_time_seconds"] / item3["total_wall_time_seconds"],
                "time_saved_seconds": baseline["total_wall_time_seconds"] - item3["total_wall_time_seconds"],
                "peak_rss_bytes": item3["peak_rss_bytes"],
                "direct_rows": item3["direct_rows"],
                "fragment_count": item3["fragment_count"],
                "unresolved_interval_count": 0,
                "hypotheses_peak": item3["solver"]["hypotheses_peak"],
                "deterministic": True,
                "scientific_decision": "defer; 113 restart segments are not three biological IDs",
                "evidence": "bench_item3_mouth_v2_32264_33263",
            },
            {
                "item": 4,
                "component": ITEMS[4],
                "cohort": "32264-33263",
                "baseline_runtime_seconds": item3["total_wall_time_seconds"],
                "item_runtime_seconds": item4["total_wall_time_seconds"],
                "speedup": item3["total_wall_time_seconds"] / item4["total_wall_time_seconds"],
                "time_saved_seconds": item3["total_wall_time_seconds"] - item4["total_wall_time_seconds"],
                "peak_rss_bytes": item4["peak_rss_bytes"],
                "direct_rows": item4["direct_rows"],
                "fragment_count": item4["fragment_count"],
                "unresolved_interval_count": 0,
                "hypotheses_peak": item4["solver"]["hypotheses_peak"],
                "deterministic": True,
                "scientific_decision": "defer; neutral but no selected-link benefit",
                "evidence": "bench_item3_mouth_v2_32264_33263; bench_item4_tail_v2_32264_33263",
            },
            {
                "item": 5,
                "component": ITEMS[5],
                "cohort": "72000-72309 immutable graph",
                "baseline_runtime_seconds": solver_seq["wall_time_seconds"],
                "item_runtime_seconds": solver_global["wall_time_seconds"],
                "speedup": solver_seq["wall_time_seconds"] / solver_global["wall_time_seconds"],
                "time_saved_seconds": solver_seq["wall_time_seconds"] - solver_global["wall_time_seconds"],
                "peak_rss_bytes": solver_global["peak_rss_bytes"],
                "direct_rows": solver_global["direct_rows"],
                "fragment_count": 159,
                "unresolved_interval_count": 0,
                "hypotheses_peak": solver_global["hypotheses_peak"],
                "deterministic": True,
                "scientific_decision": "defer; same optimum, slower, lex tie bug, 1000-frame infeasible",
                "evidence": "item5_same_graph; pair_item2_item5_global failure",
            },
            {
                "item": 6,
                "component": ITEMS[6],
                "cohort": "72000-72309 immutable graph",
                "baseline_runtime_seconds": immediate["wall_time_seconds"],
                "item_runtime_seconds": deferred["wall_time_seconds"],
                "speedup": immediate["wall_time_seconds"] / deferred["wall_time_seconds"],
                "time_saved_seconds": immediate["wall_time_seconds"] - deferred["wall_time_seconds"],
                "peak_rss_bytes": deferred["peak_rss_bytes"],
                "direct_rows": deferred["direct_rows"],
                "fragment_count": 159,
                "unresolved_interval_count": len(deferred["unresolved_intervals"]),
                "hypotheses_peak": deferred["hypotheses_peak"],
                "deterministic": True,
                "scientific_decision": "defer; blanking/horizon pass, policy semantics fail",
                "evidence": "item6_same_graph",
            },
        ]
    )
    individual["cpu_mean_percent_process_tree"] = [
        item1_resume["cpu_percent_mean_process_tree"],
        frozen_multi["cpu_percent_mean_process_tree"],
        item3["cpu_percent_mean_process_tree"],
        item4["cpu_percent_mean_process_tree"],
        solver_global["cpu_percent_mean_process_tree"],
        deferred["cpu_percent_mean_process_tree"],
    ]
    individual["cpu_peak_percent_process_tree"] = [
        item1_resume["cpu_percent_peak_process_tree"],
        frozen_multi["cpu_percent_peak_process_tree"],
        item3["cpu_percent_peak_process_tree"],
        item4["cpu_percent_peak_process_tree"],
        solver_global["cpu_percent_peak_process_tree"],
        deferred["cpu_percent_peak_process_tree"],
    ]
    individual["gpu_utilization"] = "not used by production path"
    individual["peak_vram_bytes"] = 0
    individual["output_size_bytes"] = [
        Path(item1_resume["output_files"]["filled_csv"]).stat().st_size,
        Path(frozen_multi["output_files"]["filled_csv"]).stat().st_size,
        Path(item3["output_files"]["filled_csv"]).stat().st_size,
        Path(item4["output_files"]["filled_csv"]).stat().st_size,
        (root / "item5_same_graph/global/direct_3d_observations.parquet").stat().st_size,
        (root / "item6_same_graph/deferred/direct_3d_observations.parquet").stat().st_size,
    ]
    individual["female_gt_evidence"] = [
        "full candidate: 300 eligible, 283 mapped, 0.42756 best-fixed accuracy, 81 observed changes",
        "1 eligible annotation in slice; inadequate identity denominator",
        "1 eligible annotation in slice; inadequate identity denominator",
        "1 eligible annotation in slice; no selected-link change",
        "0 annotations in 72000-72309",
        "0 annotations in 72000-72309",
    ]
    individual["reviewed_event_evidence"] = [
        "full candidate preserved 3/3 evaluable; historical preserved 3/4",
        "no complete reviewed event in isolation slice",
        "no complete reviewed event in isolation slice",
        "no complete reviewed event in isolation slice",
        "310-frame slice is not a complete reviewed event",
        "310-frame slice is not a complete reviewed event",
    ]
    individual["identity_and_stability_evidence"] = [
        "full candidate: 81 observed female changes; jump p95/p99 1.485/7.310 mm; 0 jumps >8 mm",
        "all-three direct frames fall 848 to 3; 0 jumps >8 mm; identity not estimable",
        "113 restart segments; jump p95/p99/max 1.272/5.421/7.966 mm/frame; 0 >8",
        "same 113 segments and selected edges as Item 3",
        "same three paths/objective/output as local and beam on feasible graph",
        "same three paths/objective; two ambiguity intervals and six blank IDs",
    ]
    individual["numerical_stability"] = [
        "single/overlap normalized output hash identical; exact seams",
        "deterministic selected groups; resource exclusivity passes",
        "edge term-sum error <=8.88e-16; deterministic objective",
        "tail-off/on selected edge and output equality",
        "recomputed objective equality; shuffled synthetic determinism passes",
        "exact 250-frame horizon synthetic passes; policy semantics fail",
    ]
    individual["advantages"] = [
        "bounded memory, parallel windows, exact resume, reusable cache",
        "robust multi-frame evidence and explicit resource set packing",
        "auditable physical link costs and explicit restart contract",
        "missing-tail neutrality and bounded secondary cost",
        "exact lower-bound/audit oracle with delayed-evidence capability",
        "can abstain instead of fabricating an ID at horizon",
    ]
    individual["disadvantages"] = [
        "parallel overhead on small slice; integrated output quality regression",
        "3.09x slower and catastrophic direct-coverage loss",
        "no exact links-off artifact in runner; output is 113 segments, not fish IDs",
        "no real decision benefit and incomplete frozen fallback/audit contract",
        "about 14x slower on real feasible graph; lex tie flaw; larger graph infeasible",
        "entry/future-support/resume semantics fail; real quality gain absent",
    ]
    individual["resource_note"] = [
        "measured process tree; full run mean CPU 72.12%, peak RSS 3.61 GB",
        "measured process tree",
        "measured process tree",
        "measured process tree",
        "single sub-second/3-second solver samples; interpret CPU/RSS cautiously",
        "sub-second sampler returned RSS 0 and CPU NaN; resources unavailable",
    ]
    individual.to_csv(root / "individual_results.csv", index=False)

    item3_no_link_control = {
        "schema_version": 1,
        "control": "exact same Item 3 fragment table with every link disabled",
        "graph_hash": load_json(
            root / "bench_item3_mouth_v2_32264_33263/frozen_graph_manifest.json"
        )["graph_hash"],
        "frame_start": 32264,
        "frame_end": 33263,
        "fragment_count": item3["fragment_count"],
        "direct_rows": item3["direct_rows"],
        "selected_link_count": 0,
        "restart_segment_count": item3["fragment_count"],
        "birth_cost": 3.0,
        "death_cost": 3.0,
        "objective": item3["fragment_count"] * 6.0,
        "runtime_seconds": None,
        "runtime_note": "analytical frozen-graph control; no solver or output serialization was required",
        "links_on_comparison": {
            "selected_link_count": len(item3["solver"]["selected_edges"]),
            "restart_segment_count": len(item3["solver"]["paths"]),
            "objective": item3["solver"]["objective"],
            "objective_change": item3["solver"]["objective"]
            - item3["fragment_count"] * 6.0,
        },
        "scientific_limitation": "fewer restart segments is not evidence of correct biological identity; only the female is labeled and male-to-male swaps remain unobservable",
    }
    (root / "item3_exact_links_off_control.json").write_text(
        json.dumps(item3_no_link_control, indent=2), encoding="utf-8"
    )

    compatibility = pd.DataFrame(
        [
            ["item1", "self", "compatible", "compatible", "compatible via item3", "compatible via item3", "compatible via item3/5"],
            ["item2", "compatible", "self", "compatible; tested", "compatible via item3; tested", "exact-three infeasible", "compatible via item3; tested"],
            ["item3", "compatible", "compatible; tested", "self", "compatible; tested", "compatible; tested", "compatible; tested"],
            ["item4", "compatible via item3", "compatible via item3; tested", "compatible; tested", "self", "compatible; tested", "compatible; tested"],
            ["item5", "compatible via item3", "exact-three infeasible", "compatible; tested", "compatible; tested", "self", "technically redundant under offline global solve; tested"],
            ["item6", "compatible via item3/5", "compatible via item3; tested", "compatible; tested", "compatible; tested", "technically redundant under offline global solve; tested", "self"],
        ],
        columns=["item", "item1", "item2", "item3", "item4", "item5", "item6"],
    )
    compatibility.to_csv(root / "compatibility_matrix.csv", index=False)

    pair_specs = [
        (1, 2, item1_item2_overlap["total_wall_time_seconds"], "pair_item1_item2_overlap_32264_33263", "synergistic runtime; mixed coverage and one unresolved seam"),
        (1, 3, item1_item3_overlap["total_wall_time_seconds"], "pair_item1_item3_overlap_32264_33263", "super-additive runtime cost; exact scientific-output equality"),
        (1, 4, item1_item4_overlap["total_wall_time_seconds"], "pair_item1_item4_overlap_32264_33263", "super-additive runtime cost; redundant scientific output"),
        (1, 5, item1_overlap_310["total_wall_time_seconds"] + item1_item5_overlap["wall_time_seconds"], "smoke_v2_overlap; pair_item1_item5_overlap_global_72000_72309", "sub-additive runtime; exact normalized output equality"),
        (1, 6, item1_overlap_310["total_wall_time_seconds"] + item1_item6_overlap["wall_time_seconds"], "smoke_v2_overlap; pair_item1_item6_overlap_deferred_72000_72309", "approximately additive runtime; exact normalized output equality"),
        (2, 3, item2_item3["total_wall_time_seconds"], "pair_item2_item3_32264_33263", "antagonistic coverage; 122 segments"),
        (2, 4, item2_item4["total_wall_time_seconds"], "pair_item2_item4_32264_33263", "redundant selected output with +0.000977 objective; antagonistic coverage"),
        (2, 5, np.nan, "pair_item2_item3_32264_33263; pair_item2_item5_global (infeasible)", "current 299-fragment graph has minimum path cover 10, so exact-three is infeasible"),
        (2, 6, item2_item3["total_wall_time_seconds"] + item2_deferred["wall_time_seconds"], "pair_item2_item3_32264_33263; pair_item2_item6_deferred", "confounded/composite: edge solution changes; 4 unresolved intervals and 744 blank IDs"),
        (3, 4, item4["total_wall_time_seconds"], "bench_item4_tail_v2_32264_33263", "redundant: identical selected edges/output"),
        (3, 5, solver_global["wall_time_seconds"], "item5_same_graph/global", "redundant optimum; global slower"),
        (3, 6, deferred["wall_time_seconds"], "item6_same_graph/deferred", "deferred intervals appear; quality unverified"),
        (4, 5, tail_global["wall_time_seconds"], "pair_item4_item5_global", "redundant: same objective/output"),
        (4, 6, tail_deferred["wall_time_seconds"], "pair_item4_item6_deferred", "redundant tail; same deferral behavior"),
        (5, 6, global_deferred["wall_time_seconds"], "pair_item5_item6_global_deferred", "implementation no-op under global solve; not scientific deferral evidence"),
    ]
    pairs = []
    for left, right, runtime, evidence, interaction in pair_specs:
        pairs.append(
            {
                "item_i": left,
                "item_j": right,
                "component_i": ITEMS[left],
                "component_j": ITEMS[right],
                "tested": True,
                "runtime_seconds": runtime,
                "observed_combined_speedup": np.nan,
                "time_interaction": interaction.split(";")[0],
                "quality_interaction": interaction,
                "evidence": evidence,
                "formula_note": "NA unless T0, Ti, Tj and Tij use the same immutable graph/cohort and timing boundary",
            }
        )
    pair_table = pd.DataFrame(pairs)
    for column in (
        "T0_seconds",
        "Ti_seconds",
        "Tj_seconds",
        "Tij_seconds",
        "Si",
        "Sj",
        "Sij",
        "delta_Ti_seconds",
        "delta_Tj_seconds",
        "delta_Tij_seconds",
        "delta_additivity_deviation_seconds",
        "speed_multiplicativity_deviation",
    ):
        pair_table[column] = np.nan
    timing_terms = {
        (1, 2): (
            frozen_frame["total_wall_time_seconds"],
            item1_frozen_overlap["total_wall_time_seconds"],
            frozen_multi["total_wall_time_seconds"],
            item1_item2_overlap["total_wall_time_seconds"],
        ),
        (1, 3): (
            baseline["total_wall_time_seconds"],
            item1_legacy_overlap["total_wall_time_seconds"],
            item3["total_wall_time_seconds"],
            item1_item3_overlap["total_wall_time_seconds"],
        ),
        (1, 4): (
            baseline["total_wall_time_seconds"],
            item1_legacy_overlap["total_wall_time_seconds"],
            item4["total_wall_time_seconds"],
            item1_item4_overlap["total_wall_time_seconds"],
        ),
        (1, 5): (
            item1_single_310["total_wall_time_seconds"],
            item1_overlap_310["total_wall_time_seconds"],
            item1_single_310["total_wall_time_seconds"] + solver_global["wall_time_seconds"],
            item1_overlap_310["total_wall_time_seconds"] + item1_item5_overlap["wall_time_seconds"],
        ),
        (1, 6): (
            item1_single_310["total_wall_time_seconds"],
            item1_overlap_310["total_wall_time_seconds"],
            item1_single_310["total_wall_time_seconds"] + deferred["wall_time_seconds"],
            item1_overlap_310["total_wall_time_seconds"] + item1_item6_overlap["wall_time_seconds"],
        ),
        (5, 6): (
            solver_seq["wall_time_seconds"],
            solver_global["wall_time_seconds"],
            deferred["wall_time_seconds"],
            global_deferred["wall_time_seconds"],
        ),
    }
    for (left, right), (t0, ti, tj, tij) in timing_terms.items():
        selector = (pair_table.item_i == left) & (pair_table.item_j == right)
        si, sj, sij = t0 / ti, t0 / tj, t0 / tij
        dti, dtj, dtij = t0 - ti, t0 - tj, t0 - tij
        pair_table.loc[selector, [
            "T0_seconds", "Ti_seconds", "Tj_seconds", "Tij_seconds",
            "Si", "Sj", "Sij", "delta_Ti_seconds", "delta_Tj_seconds",
            "delta_Tij_seconds", "delta_additivity_deviation_seconds",
            "speed_multiplicativity_deviation",
        ]] = [
            t0, ti, tj, tij, si, sj, sij, dti, dtj, dtij,
            dtij - (dti + dtj), sij - (si * sj),
        ]
        pair_table.loc[selector, "observed_combined_speedup"] = sij
        pair_table.loc[selector, "formula_note"] = (
            "same cohort/configuration family and explicit single/overlap timing boundary; one run per pair"
        )
    # Item 5+6 uses an exact same-graph solver-only boundary.
    pair_56 = (pair_table.item_i == 5) & (pair_table.item_j == 6)
    pair_table.loc[pair_56, "formula_note"] = (
        "same immutable graph and solver-only boundary; single-run HiGHS startup variance makes runtime interaction exploratory"
    )
    quality_effects = {
        (1, 2): "783 vs 840 direct rows; 514 vs 684 covered frames; 73 vs 3 all-three frames; one unresolved seam",
        (1, 3): "2819 direct rows; 113 restart segments",
        (1, 4): "2819 direct rows; 0 selected-edge/output changes versus Item 3",
        (1, 5): "3 paths, 156 edges, objective 44.371295; normalized biological output equality",
        (1, 6): "2 ambiguity intervals; 6 blank identity rows; normalized biological output equality",
        (2, 3): "840 direct rows; 299 fragments -> 122 segments",
        (2, 4): "840 direct rows; selected-edge symmetric difference 0; objective +0.000977",
        (2, 5): "infeasible exact-three path cover; no output scored",
        (2, 6): "9 intervals: 5 resolved, 3 horizon, 1 end; 744/840 identity rows blank; selected edges differ",
        (3, 4): "0 selected-edge, objective, direct-row, or jump differences",
        (3, 5): "objective/output difference 0; global slower on feasible slice",
        (3, 6): "same paths/objective; 2 intervals and 6 blank IDs",
        (4, 5): "objective/output difference 0 on tail graph",
        (4, 6): "same deferral behavior; selected tail-cost sum 0",
        (5, 6): "objective/output difference 0; unresolved interval count 0; deferral code path has no effect",
    }
    pair_table["quantitative_quality_effect"] = [
        quality_effects[(int(row.item_i), int(row.item_j))]
        for row in pair_table.itertuples()
    ]
    pair_table["time_interaction"] = (
        "not quantitatively comparable: prerequisite/cohort/timing boundary differs"
    )
    for left, right, classification in (
        (1, 2, "synergistic runtime; low confidence; quality mixed/unacceptable seam"),
        (1, 3, "super-additive runtime cost; scientific output independent"),
        (1, 4, "super-additive runtime cost; scientific output redundant"),
        (1, 5, "startup-dominated apparent sub-additivity; scientific output independent"),
        (1, 6, "approximately additive runtime cost; scientific output independent"),
    ):
        pair_table.loc[
            (pair_table.item_i == left) & (pair_table.item_j == right),
            "time_interaction",
        ] = classification
    pair_34 = (pair_table.item_i == 3) & (pair_table.item_j == 4)
    pair_table.loc[pair_34, "time_interaction"] = (
        "redundant incremental tail cost: -0.0112 s single-run difference"
    )
    pair_table.loc[pair_56, "time_interaction"] = (
        "non-additive/redundant; exploratory because single-run HiGHS startup dominates"
    )
    pair_26 = (pair_table.item_i == 2) & (pair_table.item_j == 6)
    pair_table.loc[pair_26, "time_interaction"] = (
        "not comparable: upstream completed runtime plus separate deferred solve"
    )
    pair_table["timing_classification_confidence"] = "not estimable"
    pair_table.loc[pair_table.item_i == 1, "timing_classification_confidence"] = "low; one fresh run per pair"
    pair_table.loc[pair_34, "timing_classification_confidence"] = "low; one run per mode"
    pair_table.loc[pair_56, "timing_classification_confidence"] = "low; startup variance"
    pair_table.to_csv(root / "pairwise_combination_results.csv", index=False)

    higher = pd.DataFrame(
        [
            {
                "bundle": "1+3+4",
                "cohort": "32264-33263",
                "runtime_seconds": item1_item4_overlap["total_wall_time_seconds"],
                "objective": item1_item4_overlap["solver"]["objective"],
                "direct_rows": item1_item4_overlap["direct_rows"],
                "unresolved_intervals": 0,
                "outcome": "reject/defer: overlap exercised; 113 restart segments; 6 tail-eligible edges, none selected",
                "evidence": "pair_item1_item4_overlap_32264_33263",
                "runtime_scope": "fresh end-to-end overlap graph and restart solve",
            },
            {
                "bundle": "1+3+4+5",
                "cohort": "72000-72309",
                "runtime_seconds": overlap_bundle_graph["total_wall_time_seconds"] + overlap_bundle_global["wall_time_seconds"],
                "objective": overlap_bundle_global["objective"],
                "direct_rows": overlap_bundle_global["direct_rows"],
                "unresolved_intervals": 0,
                "outcome": "defer: overlap exercised; tail has 0 eligible edges; global output identical",
                "evidence": "bundle_overlap_item1_3_4_graph_72000_72309; bundle_overlap_item1_3_4_5_global_72000_72309",
                "runtime_scope": "fresh overlap graph plus separate global solver",
            },
            {
                "bundle": "1+3+4+5+6",
                "cohort": "72000-72309",
                "runtime_seconds": overlap_bundle_graph["total_wall_time_seconds"] + overlap_bundle_global_deferred["wall_time_seconds"],
                "objective": overlap_bundle_global_deferred["objective"],
                "direct_rows": overlap_bundle_global_deferred["direct_rows"],
                "unresolved_intervals": len(overlap_bundle_global_deferred["unresolved_intervals"]),
                "outcome": "redundant: overlap exercised; global result/output identical; deferral is implementation no-op",
                "evidence": "bundle_overlap_item1_3_4_graph_72000_72309; bundle_overlap_item1_3_4_5_6_global_deferred_72000_72309",
                "runtime_scope": "fresh overlap graph plus separate global/deferred solver",
            },
            {
                "bundle": "1 only (full candidate)",
                "cohort": "0-348599",
                "runtime_seconds": full["total_wall_time_seconds"],
                "objective": np.nan,
                "direct_rows": full["direct_rows"],
                "unresolved_intervals": 0,
                "outcome": "completed but scientifically rejected versus historical strict-mouth output",
                "evidence": "full_optimized_baseline_v2; final_full_quality_comparison",
                "runtime_scope": "fresh end-to-end warm-cache full recording",
            },
        ]
    )
    higher.to_csv(root / "higher_order_bundle_results.csv", index=False)

    runtime_rows = []
    cache_manifest = load_json(root / "candidate_cache/candidate_cache_manifest.json")
    for view, metadata in cache_manifest["views"].items():
        runtime_rows.append(
            {
                "stage": f"candidate_cache_{view}",
                "cohort": "0-348599 cold decode/write",
                "wall_seconds": metadata["decode_write_seconds"],
                "peak_rss_bytes": np.nan,
                "cpu_mean_percent": np.nan,
                "cpu_peak_percent": np.nan,
                "gpu_utilization": "not sampled/cache I-O",
                "peak_vram_bytes": 0,
                "output_bytes": metadata["cache_size_bytes"],
                "notes": metadata["cache_sha256"],
            }
        )
    for workers in (1, 2, 4, 8, 16):
        summary = load_json(root / f"worker_bench/w{workers}/run_summary.json")
        runtime_rows.append(
            {
                "stage": f"overlap_worker_sweep_{workers}",
                "cohort": "32264-35364 warm local cache",
                "wall_seconds": summary["total_wall_time_seconds"],
                "peak_rss_bytes": summary["peak_rss_bytes"],
                "cpu_mean_percent": summary["cpu_percent_mean_process_tree"],
                "cpu_peak_percent": summary["cpu_percent_peak_process_tree"],
                "gpu_utilization": "not used",
                "peak_vram_bytes": 0,
                "output_bytes": (root / f"worker_bench/w{workers}/worker_bench_3d.csv").stat().st_size,
                "notes": "identical output sha a6e85f...; pre-final compact-summary optimization",
            }
        )
    runtime_rows.extend(
        [
            {
                "stage": "full_optimized_baseline_v2",
                "cohort": "0-348599 warm local cache",
                "wall_seconds": full["total_wall_time_seconds"],
                "peak_rss_bytes": full["peak_rss_bytes"],
                "cpu_mean_percent": full["cpu_percent_mean_process_tree"],
                "cpu_peak_percent": full["cpu_percent_peak_process_tree"],
                "gpu_utilization": "not used end-to-end",
                "peak_vram_bytes": 0,
                "output_bytes": (root / "full_optimized_baseline_v2/optimized_baseline_v2_3d.csv").stat().st_size,
                "notes": "35 checkpoints; 618 aliases; 0 unresolved seams",
            },
            {
                "stage": "hardware_endpoint_cpu",
                "cohort": "1,000,000 real endpoint pairs",
                "wall_seconds": hardware["cpu"]["endpoint_seconds_median"],
                "peak_rss_bytes": np.nan,
                "cpu_mean_percent": np.nan,
                "cpu_peak_percent": np.nan,
                "gpu_utilization": "not used",
                "peak_vram_bytes": 0,
                "output_bytes": 0,
                "notes": f"{hardware['cpu']['endpoint_pairs_per_second']:.3f} pairs/s",
            },
            {
                "stage": "hardware_endpoint_cuda_transfer_inclusive",
                "cohort": "1,000,000 real endpoint pairs",
                "wall_seconds": hardware["cuda"]["endpoint_transfer_seconds_median"],
                "peak_rss_bytes": np.nan,
                "cpu_mean_percent": np.nan,
                "cpu_peak_percent": np.nan,
                "gpu_utilization": "timed synchronized CUDA; utilization/VRAM sampler unavailable",
                "peak_vram_bytes": np.nan,
                "output_bytes": 0,
                "notes": f"max abs diff {hardware['cuda']['endpoint_max_abs_difference']:.3g}",
            },
        ]
    )
    pd.DataFrame(runtime_rows).to_csv(root / "runtime_breakdown.csv", index=False)

    full_methods = quality_json["methods"]
    optimized_direct = pd.read_parquet(
        root / "full_optimized_baseline_v2/direct_3d_observations.parquet",
        columns=["ray_residual_mm2", "tail_ray_residual_mm2"],
    )
    mouth_residual = optimized_direct["ray_residual_mm2"].dropna().to_numpy(dtype=float)
    tail_residual = optimized_direct["tail_ray_residual_mm2"].dropna().to_numpy(dtype=float)
    optimized_fragments = pd.read_parquet(
        root / "full_optimized_baseline_v2/fragments.parquet",
        columns=["observed_frames"],
    )["observed_frames"].to_numpy(dtype=int)
    quality_rows = []
    for name, info in full_methods.items():
        qc = info["track_qc"]
        female = info["female_gt"]
        is_optimized = name == "optimized_windowed_baseline_v2"
        quality_rows.append(
            {
                "method": name,
                "cohort": "full",
                "eligible_female_gt": female["eligible_confirmed_female_annotations"],
                "mapped_female_gt": female["mapped"]["count"],
                "mapped_best_fixed_accuracy": female["mapped"]["best_fixed_accuracy"],
                "mapped_observed_changes": female["mapped"]["observed_switch_count"],
                "strict_female_gt": female["strict_two_view"]["count"],
                "strict_best_fixed_accuracy": female["strict_two_view"]["best_fixed_accuracy"],
                "strict_observed_changes": female["strict_two_view"]["observed_switch_count"],
                "finite_row_share": qc["finite_row_share"],
                "direct_observed_row_share": qc["observed_noninterpolated_row_share"],
                "all_three_finite_frame_share": qc["all_three_finite_frame_share"],
                "all_three_direct_frame_share": qc["all_three_observed_frame_share"],
                "jump_p95_mm": qc["consecutive_jump_mm"]["p95"],
                "jump_p99_mm": qc["consecutive_jump_mm"]["p99"],
                "jump_max_mm": qc["consecutive_jump_mm"]["max"],
                "jump_over_8mm_count": qc["jump_over_8mm_count"],
                "reviewed_events_evaluable": info["mixed_event_results"]["evaluable_events"],
                "reviewed_events_preserved": info["mixed_event_results"]["preserved_events"],
                "mouth_residual_p50_mm2": float(np.quantile(mouth_residual, 0.50)) if is_optimized else np.nan,
                "mouth_residual_p95_mm2": float(np.quantile(mouth_residual, 0.95)) if is_optimized else np.nan,
                "mouth_residual_p99_mm2": float(np.quantile(mouth_residual, 0.99)) if is_optimized else np.nan,
                "tail_residual_p50_mm2": float(np.quantile(tail_residual, 0.50)) if is_optimized else np.nan,
                "tail_residual_p95_mm2": float(np.quantile(tail_residual, 0.95)) if is_optimized else np.nan,
                "tail_residual_p99_mm2": float(np.quantile(tail_residual, 0.99)) if is_optimized else np.nan,
                "residual_evidence_note": (
                    "direct refractive observations; no mouth residual hard gate"
                    if is_optimized
                    else "unavailable in historical filled CSV"
                ),
                "fragment_count": len(optimized_fragments) if is_optimized else np.nan,
                "fragment_length_p50_frames": float(np.quantile(optimized_fragments, 0.50)) if is_optimized else np.nan,
                "fragment_length_p95_frames": float(np.quantile(optimized_fragments, 0.95)) if is_optimized else np.nan,
                "fragment_length_p99_frames": float(np.quantile(optimized_fragments, 0.99)) if is_optimized else np.nan,
                "fragment_length_max_frames": int(np.max(optimized_fragments)) if is_optimized else np.nan,
                "singleton_fragment_share": float((optimized_fragments == 1).mean()) if is_optimized else np.nan,
                "unresolved_identity_intervals": 0 if is_optimized else np.nan,
                "hypotheses_peak": full["solver"]["hypotheses_peak"] if is_optimized else np.nan,
                "resource_conflicts": full["invariants"]["resource_conflicts"] if is_optimized else np.nan,
                "frames_over_three_active": full["invariants"]["frames_over_three_active"] if is_optimized else np.nan,
                "boundary_alias_candidates": full["boundary_alias_candidates"] if is_optimized else np.nan,
                "boundary_aliases_accepted": full["boundary_aliases_accepted"] if is_optimized else np.nan,
                "cross_view_link_precision": np.nan,
                "cross_view_link_precision_note": "no all-fish link truth; conflict/resource proxies only",
                "female_only_gt_limitation": "cannot detect male-to-male swaps",
            }
        )
    pd.DataFrame(quality_rows).to_csv(root / "quality_metrics.csv", index=False)

    cohorts = {
        "reviewed_1": (71000, 74100),
        "reviewed_2": (95000, 99000),
        "reviewed_3": (173000, 182000),
        "reviewed_4": (270000, 280000),
        "control_1": (32264, 35364),
        "control_2": (151050, 155050),
        "control_3": (21840, 30840),
        "control_4": (57363, 67363),
    }
    full_paths = {
        "historical_strict_mouth": Path(full_methods["historical_strict_mouth"]["path"]),
        "optimized_windowed_baseline_v2": Path(full_methods["optimized_windowed_baseline_v2"]["path"]),
    }
    cohort_rows = []
    for method, path in full_paths.items():
        table = pd.read_csv(path)
        for label, (start, end) in cohorts.items():
            subset = table[table.frame.between(start, end)]
            temporary = root / ".cohort_metric_tmp.csv"
            subset.to_csv(temporary, index=False)
            row = output_qc(temporary, method, label)
            row["frame_start"] = start
            row["frame_end"] = end
            cohort_rows.append(row)
            temporary.unlink()
    pd.DataFrame(cohort_rows).to_csv(root / "cohort_quality_metrics.csv", index=False)

    failed = {
        "schema_version": 1,
        "failed_or_interrupted_runs_not_scored_as_success": [
            {
                "run": "first full pipeline supervisor",
                "failure": "1000.5 second supervisor timeout after immutable graph creation",
                "correction": "hash-verified graph finalization, followed by final compact full rerun",
            },
            {
                "run": "dataframe graph finalizer reference",
                "failure": "600.5 and 1800.5 second supervisor timeouts before output commit",
                "correction": "array-backed exact-value-equivalent linker; final run 575.55 seconds end-to-end",
            },
            {
                "run": "Item 2 + Item 5 exact global",
                "failure": "three-path cover infeasible on 299-fragment/7306-edge multiframe graph",
                "correction": "none; recorded as current configuration incompatibility",
            },
            {
                "run": "candidate cache CLI manifest refresh",
                "failure": "sandbox could not resolve Z: source path; stopped before cache mutation",
                "correction": "manifest rehashed from local HDF5 caches only",
            },
        ],
    }
    (root / "failed_runs.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    summary = {
        "schema_version": 1,
        "experiment": "20251225_K_1_2_FCMC",
        "round_tag": "round7_260704",
        "seed": 20260717,
        "final_candidate": {
            "path": str((root / "full_optimized_baseline_v2/optimized_baseline_v2_3d.csv").resolve()),
            "sha256": sha256(root / "full_optimized_baseline_v2/optimized_baseline_v2_3d.csv"),
            "runtime_seconds": full["total_wall_time_seconds"],
            "peak_rss_bytes": full["peak_rss_bytes"],
            "decision": "scientifically rejected; retain as inspectable experimental result",
        },
        "production_recommendation": {
            "method": "historical strict-mouth full output",
            "path": full_methods["historical_strict_mouth"]["path"],
            "reason": "better female-GT accuracy, fewer observed female changes, better direct all-three coverage",
        },
        "component_decisions": {
            str(row.item): row.scientific_decision for row in individual.itertuples()
        },
        "interaction_conclusion": {
            "production_bundle": "none of the experimental scientific bundles",
            "item1_item2": "runtime-synergistic at low single-run confidence; quality mixed with unresolved seam",
            "item1_item3": "runtime super-additive cost; output-independent",
            "item1_item4": "runtime super-additive cost; output-redundant",
            "item1_item5": "startup-dominated apparent runtime sub-additivity; normalized output-independent",
            "item1_item6": "runtime approximately additive; normalized output-independent",
            "item5_item6": "startup-dominated implementation no-op; not evidence that active deferral is scientifically redundant",
            "other_runtime_formulas": "NA where T0/Ti/Tj/Tij do not share one graph/cohort/timing boundary",
        },
        "female_gt_limitation": "The GUI ledger identifies only the female and cannot independently detect male-to-male swaps.",
        "validation": load_json(root / "validation_summary.json"),
        "hardware_benchmark": hardware,
        "artifacts": {
            name: str((root / name).resolve())
            for name in (
                "individual_results.csv",
                "pairwise_combination_results.csv",
                "higher_order_bundle_results.csv",
                "runtime_breakdown.csv",
                "quality_metrics.csv",
                "cohort_quality_metrics.csv",
                "compatibility_matrix.csv",
                "artifact_manifest.json",
            )
        },
    }
    (root / "study_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    repo_root = root.resolve().parents[2]
    manifest_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and path.suffix.lower() in {".md", ".csv", ".json"}
    )
    manifest_paths.extend(
        [
            repo_root / "scripts/analysis/benchmark_windowed_tracklet_hardware.py",
            repo_root / "scripts/analysis/summarize_windowed_tracklet_study.py",
            repo_root / "scripts/reconstruction_3d/candidate_cache.py",
            repo_root / "scripts/reconstruction_3d/windowed_tracklet_core.py",
            repo_root / "scripts/reconstruction_3d/windowed_tracklet_finalize.py",
            repo_root / "scripts/reconstruction_3d/windowed_tracklet_solve.py",
            repo_root / "scripts/reconstruction_3d/windowed_tracklet_stitching.py",
            repo_root / "tests/test_candidate_cache.py",
            repo_root / "tests/test_windowed_tracklet_stitching.py",
            root / "full_optimized_baseline_v2/optimized_baseline_v2_3d.csv",
            root / "full_optimized_baseline_v2/resolved_configuration.json",
            root / "full_optimized_baseline_v2/run_summary.json",
            root / "full_optimized_baseline_v2/frozen_graph_manifest.json",
            repo_root
            / "report/20260716/mouth_tracking_ablation/full/mouth_only_3d/mouth_only_3d_3d.csv",
        ]
    )
    manifest_paths.extend(sorted((root / "candidate_cache").glob("*.h5")))
    for pattern in (
        "*/run_summary.json",
        "*/resolved_configuration.json",
        "*/frozen_graph_manifest.json",
        "*/solver_summary.json",
        "*/*/solver_summary.json",
    ):
        manifest_paths.extend(sorted(root.glob(pattern)))
    unique_paths = sorted({path.resolve() for path in manifest_paths if path.is_file()})
    manifest = {
        "schema_version": 1,
        "generated_by": "scripts.analysis.summarize_windowed_tracklet_study",
        "seed": 20260717,
        "files": [
            {
                "path": str(path.relative_to(repo_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique_paths
        ],
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
