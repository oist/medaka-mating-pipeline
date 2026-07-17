# Logs and provenance index

Every successful window directory contains `resolved_configuration.json`,
`run_summary.json`, a frozen graph manifest, content-hashed Parquet/CSV files,
and atomic `checkpoints/window_*/complete.json` records. Solver-only directories
contain `solver_summary.json`, selected edges/paths, identity audit tables, and
output hashes. Failed attempts are enumerated separately in `failed_runs.json`.

Primary provenance locations:

- Candidate cache metadata and hashes: `candidate_cache/candidate_cache_manifest.json` and the
  three `*.candidates.h5` files.
- Final full configuration/runtime/resources: `full_optimized_baseline_v2/`.
- Full quality comparison: `final_full_quality_comparison/` plus
  `quality_metrics.csv` and `cohort_quality_metrics.csv`.
- Item 1 equivalence/reuse: `item1_final_single_32264_35364/` and
  `item1_final_overlap8_32264_35364/`.
- Item 2: `bench_frozen_framewise_32264_33263/` and
  `bench_frozen_multiframe_32264_33263/`.
- Items 3/4: `bench_item3_mouth_v2_32264_33263/` and
  `bench_item4_tail_v2_32264_33263/`.
- Items 5/6: `item5_same_graph/`, `item6_same_graph/`, and pair/bundle
  directories named in `pairwise_combination_results.csv`.
- Hardware: `hardware_benchmark.json` and `runtime_breakdown.csv`.
- Final validation: `validation_summary.json` records focused and complete
  pytest results plus the final consistency checks.
- Frozen specifications and independent audit: `itemN_primary_report.md`,
  `itemN_implementation_report.md`, and `itemN_verifier_report.md` for N=1..6.
- Independent combination audits: `pair_verifier_item1_pairs.md`,
  `pair_verifier_items2_3_pairs.md`, and
  `pair_verifier_items3_6_and_bundles.md`.

The final candidate CSV SHA-256 is
`340078e8e2c6c09c8177fd717daf44a054faad116380f83f65bbbd8b34de5d0b`.
Calibration manifest SHA-256 is
`9bb7dda927c01c018c411d3b56da58f3e44379f4d4ce1491c5ab67d33e5bf105`.
Exact per-file code/cache/calibration hashes are preserved in the resolved
configuration rather than duplicated here.
