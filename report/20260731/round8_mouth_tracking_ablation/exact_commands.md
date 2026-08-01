# Executor commands

All paths were resolved from `configs/paths.local.toml` and the completed
Round8 provenance. The sandbox process did not inherit the configured `Z:`
mapping, so the cache command used the equivalent OIST UNC root.

## Rejected diagnostic candidate cache

Started at `2026-07-31T01:46:22.8494633+09:00` as wrapper PID `28228` and
worker PID `36612`.

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.candidate_cache `
  20251225_K_1_2_FCMC `
  --round-tag round8_260729_snn031_m6 `
  --paths-config configs\paths.local.toml `
  --source-root \\bucket.oist.jp\bucket\StephensU\medaka\sleap_output `
  --output-dir report\20260731\round8_mouth_tracking_ablation\candidate_cache `
  --num-candidates 6 `
  --frame-count 348600
```

Logs:

- `candidate_cache/build_cache.stdout.log`
- `candidate_cache/build_cache.stderr.log`
- `candidate_cache/build_cache_resource_samples.csv`
- `candidate_cache/build_cache_resource_summary.json`

Peak memory is sampled once per second with `Get-Process`; the sample sums
`WorkingSet64` and `PeakWorkingSet64` across wrapper and worker. Wall time is
measured from the worker process start time through disappearance of both
processes. The initial two-PID monitor invocation was rejected by PowerShell's
native `-File` array parsing before writing samples. The successful monitor
sampled the memory-owning worker PID:

```powershell
pwsh -NoProfile -File `
  report\20260731\round8_mouth_tracking_ablation\monitor_processes.ps1 `
  -ProcessId 36612 `
  -CsvPath report\20260731\round8_mouth_tracking_ablation\candidate_cache\build_cache_resource_samples.csv `
  -SummaryPath report\20260731\round8_mouth_tracking_ablation\candidate_cache\build_cache_resource_summary.json `
  -StartedUnixSeconds 1785429982.849 `
  -IntervalSeconds 1
```

Completed with empty stderr. The cache build took `345.430 s` wall time. The
maximum one-second sampled working set was `1,128,710,144` bytes and the
maximum OS-reported peak working set was `1,147,113,472` bytes. Per-view
decode/write times were 112.154 s (top), 112.542 s (side1), and 111.789 s
(side2).

This first cache is retained only as rejected diagnostic provenance. It used
the legacy default `filter_fp=true`; that filter inspects tail/body span and is
therefore outside the mouth-coordinate-only A1–A5 contract.

## Superseded mouth-only candidate cache

Started at `2026-07-31T01:56:31.1338995+09:00` as wrapper PID `2716` and
worker PID `6692`.

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.candidate_cache `
  20251225_K_1_2_FCMC `
  --round-tag round8_260729_snn031_m6 `
  --paths-config configs\paths.local.toml `
  --source-root \\bucket.oist.jp\bucket\StephensU\medaka\sleap_output `
  --output-dir report\20260731\round8_mouth_tracking_ablation\candidate_cache_strict_mouth_unfiltered `
  --num-candidates 6 `
  --frame-count 348600 `
  --no-filter-fp
```

The `--no-filter-fp` switch preserves only strict finite-mouth admission and
score ordering. It disables the legacy pose-span/overlap filter because that
filter also reads tail coordinates.

Completed with empty stderr in `176.159 s` wall time. The maximum sampled
working set was `1,129,209,856` bytes and the maximum OS-reported peak working
set was `1,147,400,192` bytes. Per-view decode/write times were 56.693 s (top),
56.729 s (side1), and 57.672 s (side2).

This cache was superseded before any method run. Its coordinates are valid,
but its `mouth_score` arrays are empty because the structured SLEAP node-score
field was not extracted.

## Final canonical strict-mouth cache with node scores

Started at `2026-07-31T02:09:03.8289533+09:00` as wrapper PID `20088` and
worker PID `39752`.

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.candidate_cache `
  20251225_K_1_2_FCMC `
  --round-tag round8_260729_snn031_m6 `
  --paths-config configs\paths.local.toml `
  --source-root \\bucket.oist.jp\bucket\StephensU\medaka\sleap_output `
  --output-dir report\20260731\round8_mouth_tracking_ablation\candidate_cache_strict_mouth_unfiltered_nodescores `
  --num-candidates 6 `
  --frame-count 348600 `
  --no-filter-fp
```

The final cache completed with empty stderr. Resource monitoring sampled
worker PID `39752` once per second:

```powershell
pwsh -NoProfile -File `
  report\20260731\round8_mouth_tracking_ablation\monitor_processes.ps1 `
  -ProcessId 39752 `
  -CsvPath report\20260731\round8_mouth_tracking_ablation\candidate_cache_strict_mouth_unfiltered_nodescores\build_cache_resource_samples.csv `
  -SummaryPath report\20260731\round8_mouth_tracking_ablation\candidate_cache_strict_mouth_unfiltered_nodescores\build_cache_resource_summary.json `
  -StartedUnixSeconds 1785431343.828 `
  -IntervalSeconds 1
```

Wall time was `182.6801866 s`. Maximum sampled working set was
`1,129,820,160` bytes and maximum OS-reported peak working set was
`1,147,641,856` bytes. Per-view decode/write times were 57.222 s (top),
60.807 s (side1), and 59.523 s (side2).

The independently rehashed cache SHA-256 values are:

- top: `6d2f86e8822525e83f18c921bb2b75d0e2821b929d2dcced994dea29b2737d74`
- side1: `59d9a73895f0b0c507e5dd1c57a8503ae8ca932cdad7e2dd0a16685b7fb042be`
- side2: `1f2d639606e77cc4c1b3188e8991cdf02c2895556d225996a957f91af7832699`

The independent source audit passed: direct Round8 SLP decoding had zero
mismatches at tolerance in all three views (maximum coordinate error
`1.14e-13 px`; maximum mouth-score error `2.22e-16`). Non-score arrays were
bit-exact against the superseded strict-mouth cache.

## Baseline tests before new implementation

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp report\20260731\round8_mouth_tracking_ablation\pytest_tmp_executor `
  tests\test_candidate_cache.py `
  tests\test_cross_frame_tracking_motion.py `
  tests\test_windowed_tracklet_stitching.py
```

Result: `28 passed in 1.85s`.

## Focused implementation regression suite

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp report\20260731\round8_mouth_tracking_ablation\pytest_tmp_mouth_runner_v3 `
  tests\test_mouth_ablation.py `
  tests\test_candidate_cache.py `
  tests\test_conservative_mouth_tracklets.py `
  tests\test_deferred_identity_tracking.py `
  tests\test_cross_frame_tracking_motion.py `
  tests\test_windowed_tracklet_stitching.py
```

Result: `46 passed in 2.34s`.

This suite includes regressions for declared-view finiteness, tracklet-ID
indexing after the first geometry evaluation batch, frozen training-control
body-length scaling, recoverable temporary A4/A3 constraint conflicts, and
unshifted missing-frame filling.

The final rerun after enforcing controlled A2/A3 method isolation produced
`47 passed in 2.15s` using the same command and
`--basetemp ...\pytest_tmp_mouth_runner_v4`.

## Noncritical training-control smoke

Geometry:

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.mouth_ablation geometry `
  --cache-dir report\20260731\round8_mouth_tracking_ablation\candidate_cache_strict_mouth_unfiltered_nodescores `
  --paths-config configs\paths.local.toml `
  --output-dir report\20260731\round8_mouth_tracking_ablation\smoke_training_control_geometry `
  --frame-start 32264 --frame-end 32320 `
  --block-frames 32 --evaluation-batch-frames 32 `
  --xyz-bounds -20 220
```

A1:

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.mouth_ablation relink `
  --geometry report\20260731\round8_mouth_tracking_ablation\smoke_training_control_geometry\strict_mouth_selected_geometry.parquet `
  --output-dir report\20260731\round8_mouth_tracking_ablation\smoke_training_control_relink_a1 `
  --method A1 --frame-start 32264 --frame-end 32320 `
  --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100
```

A2 (constant velocity only):

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.mouth_ablation relink `
  --geometry report\20260731\round8_mouth_tracking_ablation\smoke_training_control_geometry\strict_mouth_selected_geometry.parquet `
  --output-dir report\20260731\round8_mouth_tracking_ablation\smoke_training_control_relink_a2 `
  --method A2 --frame-start 32264 --frame-end 32320 `
  --velocity-weight 1 --tangent-weight 0 --acceleration-weight 0 `
  --uncertainty-growth 0.25 --motion-history-frames 5 `
  --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100
```

A3 (same velocity plus low-weight acceleration and bounded deferral):

```powershell
.venv\Scripts\python.exe -m scripts.reconstruction_3d.mouth_ablation relink `
  --geometry report\20260731\round8_mouth_tracking_ablation\smoke_training_control_geometry\strict_mouth_selected_geometry.parquet `
  --output-dir report\20260731\round8_mouth_tracking_ablation\smoke_training_control_relink_a3 `
  --method A3 --frame-start 32264 --frame-end 32320 `
  --velocity-weight 1 --tangent-weight 0 --acceleration-weight 0.1 `
  --uncertainty-growth 0.25 --motion-history-frames 5 `
  --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 `
  --beam-width 16 --entry-margin 0.5 --retention-margin 2 `
  --resolution-margin 1 --resolution-stability-frames 3 `
  --minimum-lookahead-frames 3 --maximum-defer-horizon 100
```

The smoke is plumbing-only, not selection evidence. The independent audit
confirmed the cached geometry on all 57 frames against the production
extra-candidate selector: identical slot tuples/cardinality, maximum XYZ
difference `0.0 mm`, and maximum residual difference `0.0 mm^2`. See
`smoke_training_control_status.json`.

## Frozen decision boundary

The final freeze was written only after the training-control motion, A3,
tracklet-cutoff, A4, and A5 sweeps were complete. Earlier invalid freeze
attempts are retained under the two `quarantine_invalid_freeze_*` directories.
The authenticated freeze command was:

```powershell
.venv\Scripts\python.exe `
  report\20260731\round8_mouth_tracking_ablation\write_frozen_thresholds.py
```

The independent pre-launch authentication command was:

```powershell
.venv\Scripts\python.exe `
  report\20260731\round8_mouth_tracking_ablation\quality_review\audit_frozen_thresholds.py
```

The resulting `frozen_thresholds.json` SHA-256 is
`09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d`;
it was frozen at `2026-07-31T04:18:09.058553+09:00`. The independent audit
found no post-freeze output roots present at that boundary.

## Controlled post-freeze stages

Each `--prepare-only` invocation wrote the exact ordered child-command list,
input hashes, output contracts, and frozen dependency bindings into that
stage's `STAGE_PREPARED.json`. Each execution then reauthenticated that
prepared manifest before launching any child command.

Validation preparation, detached execution, and finalization:

```powershell
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage.py validation --prepare-only
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage.py validation --execute-prepared
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\finalize_bounded_postfreeze_stage.py validation
```

Authoritative validation hashes are: prepared
`3fb4e79994df34c5d0d140a1ed5f24d5358970114230950d7cd9addb74eabae6`,
runner `54ec5698f84c6c65aa2e0aff226dd208445ab1db51c35a60bed8e396f3abab16`,
prelaunch authentication
`abdfe584b902c3e904712231f5b4ba1b701c89b5d7c7cac8af0ed7870eb2c514`,
completion `7b11328f9304c9ad2a472d93125c0e108ea3e810e4a06e616b9da09128f32fa8`,
and finalization
`297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706`.

Held-out preparation, detached execution, and finalization:

```powershell
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage_v4.py heldout --prepare-only
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage_v4.py heldout --execute-prepared
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\finalize_bounded_postfreeze_stage_v2.py heldout
```

Authoritative held-out hashes are: prepared
`72557e96b34c2582c58419bbaa1e71ae6379b89f7e8e3624a42fbc1e5c216037`,
runner `0aaeac2293db434f3794b02a992ab208f4dc061162b8da017f6c85d94a21e818`,
prelaunch authentication
`871420dd5b55ed4d3b5b69beb91dc12035644e1c30c919199e590d0737c1f61b`,
completion `615b03f161db28982ee1d6b5648e79e1d4bef146b281cc911b13d4654d7da143`,
and finalization
`3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152`.

Critical descriptive preparation, detached execution, and finalization:

```powershell
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage_v5.py critical --prepare-only
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_bounded_postfreeze_stage_v5.py critical --execute-prepared
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\finalize_bounded_postfreeze_stage_v2.py critical
```

Authoritative critical-descriptive hashes are: prepared
`8900912581348c4df936767c639b429100d17c3d5736d528c108679824862d6b`,
runner `2eb4012e428bf8951609a8c345d55fc75c74cd934b955ef5400bb2c56d72c38d`,
prelaunch authentication
`0cfe3a3f000d964838d6c1cd0ba075a2cc6ac5c5c086644dd02441aa4100cd4c`,
completion `e493fdcc449034c4f9dd22e3f124283fe8d3cd06d9bd50b0c46002e440367706`,
and finalization
`fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c`.
This stage was restricted to frames 34000--35100, with detail frames
34540--34565 and assignment rows 34547--34555, and is permanently
nonconfirmatory.

Full-length preparation, detached execution, and finalization:

```powershell
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_full_length_postfreeze_v1.py --prepare-only
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\run_full_length_postfreeze_v1.py --execute-prepared
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\finalize_full_length_postfreeze_v1.py full_length
```

Authoritative full-length hashes are: prepared
`d9c14fcb973f0fa29877443802eeba643fd5b288245988414b13c53d9f1f4aa1`,
runner `32a8452da10f0b53ec555cef1212aa4dabbda13ba7e146f7d508deca5c0fb87d`,
prelaunch authentication
`bd1ef6119ff770121ad65c1ce87d772190c3d637c479940045ee84a5f81f8b7b`,
completion `7e9e826cc48965d2b0ada1569b63d7a77b3b2972a8e20649eff15bc2de00dca0`,
and finalization
`a9a8ae25cca6ef0c94dc7996cf39680017b9bb58e20ab60473bced96284e07b0`.
The run covered frames 0--348599. A0 was the unchanged reference, A1--A3
were the only new full-length methods, and A4/A5 were excluded.

## Independent scoring and final sealing

The final scorer invocations were:

```powershell
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\quality_review\audit_and_score_postfreeze.py --stage validation --stage heldout --stage critical
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\quality_review\audit_and_score_full_length.py
.venv\Scripts\python.exe report\20260731\round8_mouth_tracking_ablation\quality_review\summarize_full_length_annotation_partitions.py
```

The authoritative final quality seal is
`quality_review/FULL_LENGTH_QUALITY_REVIEW_COMPLETED.json`, SHA-256
`7e9896716a693ce81287320af1aacd6ab686f909ad4f5cc8de43af54dc03cc92`.
The independent auditor recursively rehashed all 24 seal bindings and reported
24/24 exact matches.

## Preserved execution incidents

- A validation launch was wrapped in a one-second shell timeout. The scientific
  child continued normally; the incident is authenticated by
  `validation_postfreeze/INTERRUPTED_LAUNCH_20260731_043346.json`, SHA-256
  `9047506507769e875f52af369351ef424ed3957514e536900c57ec89ae454558`.
- The first held-out attempt correctly failed when A5 received an empty,
  schema-less A4 parquet after A4 scientifically abstained. It was finalized
  as a failure and quarantined under
  `quarantine_failed_heldout_attempt_runner1f737_20260731_0457/`. The
  replacement adapter contract was quarantined and smoke-tested before the
  fresh authoritative v4 run; no failed-attempt output was reused.
- The full-length detached PowerShell transport remained attached until its
  child closed. The hash-bound launch, exit, completion, and finalization
  records establish exit code zero and the final inventory.
- The full-length scorer outlived a 14-second shell wrapper. A duplicate was
  stopped before writes, followed by clean exit-zero scoring reruns. The
  complete supervision record is
  `quality_review/full_length_scorer_execution_incident.json`, SHA-256
  `6ecf08b348f02bbc21ad87879160123809f8fd7e57259a893815f397fc71f66a`.

`FINAL_ARTIFACT_INDEX.json` is the machine-readable map from these controls to
the input provenance, training/cutoff evidence, stage manifests, completion
records, incidents, final metrics, and independent review artifacts.
