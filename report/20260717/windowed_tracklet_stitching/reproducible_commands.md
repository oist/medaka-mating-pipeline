# Reproducible commands

Run from the repository root in PowerShell. These commands use the frozen
experiment, round, seed, calibration/path configuration, and report locations.
Existing complete checkpoints are intentionally reused; use a new output
directory for a fresh timing run.

## Candidate caches

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.candidate_cache `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --output-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --num-candidates 6 --frame-count 348600
```

Do not pass `--overwrite` unless rebuilding all three caches from the source
`.slp` files is intended. The local manifest records source size/mtime and cache
SHA-256.

## Item 1 exact-current single and overlap runs

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --output-dir report\20260717\windowed_tracklet_stitching\item1_final_single_32264_35364 `
  --output-tag item1_single --frame-start 32264 --frame-end 35364 `
  --xyz-low 0 --xyz-high 200 --window-mode single --workers 1 `
  --hypothesis-source legacy --cross-view-mode framewise `
  --fragment-link-mode none --tail-continuity off `
  --trajectory-solver sequential_hungarian --identity-policy solver_only `
  --path-cover-mode restarts --max-jump-mm 8 --max-jump-hard-cap-mm 80 `
  --max-gap-frames 100 --seed 20260717

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --output-dir report\20260717\windowed_tracklet_stitching\item1_final_overlap8_32264_35364 `
  --output-tag item1_overlap8 --frame-start 32264 --frame-end 35364 `
  --xyz-low 0 --xyz-high 200 --window-mode overlap `
  --window-core-frames 500 --window-context-frames 250 --workers 8 `
  --hypothesis-source legacy --cross-view-mode framewise `
  --fragment-link-mode none --tail-continuity off `
  --trajectory-solver sequential_hungarian --identity-policy solver_only `
  --path-cover-mode restarts --max-jump-mm 8 --max-jump-hard-cap-mm 80 `
  --max-gap-frames 100 --seed 20260717
```

Rerunning the second command is the checkpoint-resume verification.

## Explicit Item 1 interaction reruns

These post-audit runs replace earlier compatibility-only evidence for pairs
1+2 through 1+6. The two controls and three 1,000-frame pairs share the exact
core/context/worker boundary below:

```powershell
$studyRoot = 'report\20260717\windowed_tracklet_stitching'
$pairCommon = @(
  '20251225_K_1_2_FCMC', '--round-tag', 'round7_260704',
  '--paths-config', 'configs\paths.local.toml',
  '--cache-dir', "$studyRoot\candidate_cache",
  '--frame-start', '32264', '--frame-end', '33263',
  '--xyz-low', '0', '--xyz-high', '200', '--window-mode', 'overlap',
  '--window-core-frames', '500', '--window-context-frames', '250',
  '--workers', '2', '--evaluation-batch-frames', '256',
  '--max-jump-mm', '8', '--max-jump-hard-cap-mm', '80',
  '--max-gap-frames', '100', '--seed', '20260717'
)

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @pairCommon --output-dir "$studyRoot\pair_item1_only_frozen_overlap_32264_33263" --output-tag pair_item1_only_frozen_overlap --hypothesis-source frozen --cross-view-mode framewise --fragment-link-mode none --tail-continuity off
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @pairCommon --output-dir "$studyRoot\pair_item1_only_legacy_overlap_32264_33263" --output-tag pair_item1_only_legacy_overlap --hypothesis-source legacy --cross-view-mode framewise --fragment-link-mode none --tail-continuity off
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @pairCommon --output-dir "$studyRoot\pair_item1_item2_overlap_32264_33263" --output-tag pair_item1_item2_overlap --hypothesis-source frozen --cross-view-mode multiframe --min-group-support 3 --min-group-valid-fraction 0.95 --max-group-residual-mm2 10 --fragment-link-mode none --tail-continuity off
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @pairCommon --output-dir "$studyRoot\pair_item1_item3_overlap_32264_33263" --output-tag pair_item1_item3_overlap --hypothesis-source legacy --cross-view-mode framewise --fragment-link-mode mouth --tail-continuity off --path-cover-mode restarts --birth-cost 3 --death-cost 3
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @pairCommon --output-dir "$studyRoot\pair_item1_item4_overlap_32264_33263" --output-tag pair_item1_item4_overlap --hypothesis-source legacy --cross-view-mode framewise --fragment-link-mode mouth --tail-continuity conditional --path-cover-mode restarts --birth-cost 3 --death-cost 3 --mouth-ambiguity-margin 0.5 --close-contact-mm 12
```

The 310-frame overlap graph and its solvers reproduce pairs 1+5 and 1+6:

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching 20251225_K_1_2_FCMC --round-tag round7_260704 --paths-config configs\paths.local.toml --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache --output-dir report\20260717\windowed_tracklet_stitching\smoke_v2_overlap --output-tag smoke --frame-start 72000 --frame-end 72309 --window-mode overlap --window-core-frames 100 --window-context-frames 100 --workers 1 --hypothesis-source legacy --cross-view-mode framewise --fragment-link-mode none --tail-continuity off --trajectory-solver sequential_hungarian --identity-policy immediate --path-cover-mode exact_three --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 --seed 20260717
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir report\20260717\windowed_tracklet_stitching\smoke_v2_overlap --output-dir report\20260717\windowed_tracklet_stitching\pair_item1_item5_overlap_global_72000_72309 --trajectory-solver global_dag --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir report\20260717\windowed_tracklet_stitching\smoke_v2_overlap --output-dir report\20260717\windowed_tracklet_stitching\pair_item1_item6_overlap_deferred_72000_72309 --trajectory-solver beam --beam-width 32 --identity-policy deferred --path-cover-mode exact_three --identity-entry-margin 0.5 --identity-retention-margin 2 --identity-resolution-margin 1 --identity-resolution-stability-frames 10 --maximum-defer-horizon 250
```

## Items 2--4 isolation and pair graphs

The shared base flags below are expanded literally in each saved
`resolved_configuration.json`.

```powershell
$studyRoot = 'report\20260717\windowed_tracklet_stitching'
$common = @(
  '20251225_K_1_2_FCMC', '--round-tag', 'round7_260704',
  '--paths-config', 'configs\paths.local.toml',
  '--cache-dir', "$studyRoot\candidate_cache",
  '--frame-start', '32264', '--frame-end', '33263',
  '--xyz-low', '0', '--xyz-high', '200', '--window-mode', 'single',
  '--workers', '1', '--max-jump-mm', '8',
  '--max-jump-hard-cap-mm', '80', '--max-gap-frames', '100',
  '--seed', '20260717'
)

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\bench_frozen_framewise_32264_33263" `
  --output-tag frozen_framewise --hypothesis-source frozen `
  --cross-view-mode framewise --fragment-link-mode none --tail-continuity off

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\bench_frozen_multiframe_32264_33263" `
  --output-tag frozen_multiframe --hypothesis-source frozen `
  --cross-view-mode multiframe --min-group-support 3 `
  --min-group-valid-fraction 0.95 --max-group-residual-mm2 10 `
  --fragment-link-mode none --tail-continuity off

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\bench_item3_mouth_v2_32264_33263" `
  --output-tag item3_mouth --hypothesis-source legacy `
  --cross-view-mode framewise --fragment-link-mode mouth --tail-continuity off `
  --path-cover-mode restarts --birth-cost 3 --death-cost 3

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\bench_item4_tail_v2_32264_33263" `
  --output-tag item4_tail --hypothesis-source legacy `
  --cross-view-mode framewise --fragment-link-mode mouth `
  --tail-continuity conditional --mouth-ambiguity-margin 0.5 `
  --close-contact-mm 12 --path-cover-mode restarts --birth-cost 3 --death-cost 3

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\pair_item2_item3_32264_33263" `
  --output-tag item2_item3 --hypothesis-source frozen `
  --cross-view-mode multiframe --min-group-support 3 `
  --min-group-valid-fraction 0.95 --max-group-residual-mm2 10 `
  --fragment-link-mode mouth --tail-continuity off --path-cover-mode restarts

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching @common `
  --output-dir "$studyRoot\pair_item2_item4_32264_33263" `
  --output-tag item2_item4 --hypothesis-source frozen `
  --cross-view-mode multiframe --min-group-support 3 `
  --min-group-valid-fraction 0.95 --max-group-residual-mm2 10 `
  --fragment-link-mode mouth --tail-continuity conditional --path-cover-mode restarts
```

## Items 5--6 and higher-order bundles

Build the exact-three-feasible mouth graph and its tail-augmented counterpart:

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --output-dir report\20260717\windowed_tracklet_stitching\smoke_v2_single310 `
  --output-tag item5_graph --frame-start 72000 --frame-end 72309 `
  --window-mode single --window-context-frames 100 --workers 1 `
  --hypothesis-source legacy --cross-view-mode framewise `
  --fragment-link-mode none --tail-continuity off `
  --trajectory-solver sequential_hungarian --identity-policy immediate `
  --path-cover-mode exact_three --max-jump-mm 8 `
  --max-jump-hard-cap-mm 80 --max-gap-frames 100 --seed 20260717

.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --output-dir report\20260717\windowed_tracklet_stitching\pair_item4_graph_72000_72309 `
  --output-tag item4_graph --frame-start 72000 --frame-end 72309 `
  --window-mode single --window-context-frames 250 --workers 1 `
  --hypothesis-source legacy --cross-view-mode framewise `
  --fragment-link-mode mouth --tail-continuity conditional `
  --trajectory-solver sequential_hungarian --identity-policy solver_only `
  --path-cover-mode restarts --max-jump-mm 8 `
  --max-jump-hard-cap-mm 80 --max-gap-frames 100 --seed 20260717
```

Run the common-graph solvers:

```powershell
$graph = 'report\20260717\windowed_tracklet_stitching\smoke_v2_single310'
$studyRoot = 'report\20260717\windowed_tracklet_stitching'
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $graph --output-dir "$studyRoot\item5_same_graph\sequential" --trajectory-solver sequential_hungarian --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $graph --output-dir "$studyRoot\item5_same_graph\beam32" --trajectory-solver beam --beam-width 32 --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $graph --output-dir "$studyRoot\item5_same_graph\global" --trajectory-solver global_dag --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $graph --output-dir "$studyRoot\item6_same_graph\immediate" --trajectory-solver beam --beam-width 32 --identity-policy immediate --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $graph --output-dir "$studyRoot\item6_same_graph\deferred" --trajectory-solver beam --beam-width 32 --identity-policy deferred --path-cover-mode exact_three --identity-entry-margin 0.5 --identity-retention-margin 2 --identity-resolution-margin 1 --identity-resolution-stability-frames 10 --maximum-defer-horizon 250

$tailGraph = 'report\20260717\windowed_tracklet_stitching\pair_item4_graph_72000_72309'
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $tailGraph --output-dir "$studyRoot\pair_item4_item5_global" --trajectory-solver global_dag --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $tailGraph --output-dir "$studyRoot\pair_item4_item6_deferred" --trajectory-solver beam --beam-width 32 --identity-policy deferred --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $tailGraph --output-dir "$studyRoot\pair_item5_item6_global_deferred" --trajectory-solver global_dag --identity-policy deferred --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir $tailGraph --output-dir "$studyRoot\bundle_1_3_4_5_6_global_deferred" --trajectory-solver global_dag --identity-policy deferred --path-cover-mode exact_three
```

The final higher-order evidence uses an explicit overlap tail graph (the older
single-window bundle above is retained only as provenance):

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching 20251225_K_1_2_FCMC --round-tag round7_260704 --paths-config configs\paths.local.toml --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache --output-dir report\20260717\windowed_tracklet_stitching\bundle_overlap_item1_3_4_graph_72000_72309 --output-tag bundle_overlap_item1_3_4_graph --frame-start 72000 --frame-end 72309 --xyz-low 0 --xyz-high 200 --window-mode overlap --window-core-frames 100 --window-context-frames 100 --workers 1 --hypothesis-source legacy --cross-view-mode framewise --fragment-link-mode mouth --tail-continuity conditional --trajectory-solver sequential_hungarian --identity-policy solver_only --path-cover-mode restarts --birth-cost 3 --death-cost 3 --mouth-ambiguity-margin 0.5 --close-contact-mm 12 --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 --seed 20260717
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir report\20260717\windowed_tracklet_stitching\bundle_overlap_item1_3_4_graph_72000_72309 --output-dir report\20260717\windowed_tracklet_stitching\bundle_overlap_item1_3_4_5_global_72000_72309 --trajectory-solver global_dag --identity-policy solver_only --path-cover-mode exact_three
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_solve --graph-dir report\20260717\windowed_tracklet_stitching\bundle_overlap_item1_3_4_graph_72000_72309 --output-dir report\20260717\windowed_tracklet_stitching\bundle_overlap_item1_3_4_5_6_global_deferred_72000_72309 --trajectory-solver global_dag --identity-policy deferred --path-cover-mode exact_three
```

The remaining Item 2 pair commands apply the same solver calls to
`pair_item2_item3_32264_33263`; exact-three infeasibility for Item 2+5 is an
expected recorded result. All 15 pair outcomes and the precise evidence
directory for each are in `pairwise_combination_results.csv`.

## Full final candidate

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.windowed_tracklet_stitching `
  20251225_K_1_2_FCMC --round-tag round7_260704 `
  --paths-config configs\paths.local.toml `
  --cache-dir report\20260717\windowed_tracklet_stitching\candidate_cache `
  --output-dir report\20260717\windowed_tracklet_stitching\full_optimized_baseline_v2 `
  --output-tag optimized_baseline_v2 --frame-start 0 --frame-end 348599 `
  --xyz-low 0 --xyz-high 200 --window-mode overlap `
  --window-core-frames 10000 --window-context-frames 250 --workers 8 `
  --evaluation-batch-frames 256 --hypothesis-source legacy `
  --cross-view-mode framewise --fragment-link-mode none --tail-continuity off `
  --trajectory-solver sequential_hungarian --identity-policy solver_only `
  --path-cover-mode restarts --max-jump-mm 8 --max-jump-hard-cap-mm 80 `
  --max-gap-frames 100 --seed 20260717
```

## Hardware benchmark, aggregation, and tests

```powershell
.venv\Scripts\python.exe -m scripts.analysis.benchmark_windowed_tracklet_hardware --fragment-file report\20260717\windowed_tracklet_stitching\full_optimized_baseline\fragments.parquet --output report\20260717\windowed_tracklet_stitching\hardware_benchmark.json --seed 20260717 --repeats 7 --endpoint-pairs 1000000 --solve-batch-size 250000
.venv\Scripts\python.exe -m scripts.analysis.summarize_windowed_tracklet_study --report-root report\20260717\windowed_tracklet_stitching
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; .venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_candidate_cache.py tests\test_windowed_tracklet_stitching.py
git diff --check
```
