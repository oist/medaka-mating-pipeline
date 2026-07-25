# Phase 1 manual visual-QC GUI

The viewer is `scripts/sleap/benchmark_visual_qc.py`. It reads the benchmark
manifest, cached Round 7/Round 8 evidence, and raw frames. It never changes a
video, source SLP, or annotation package.

## Start with the evaluated pilot cohort

The two full-top-down match tables contain GT and prediction coordinates, so
they are sufficient for the normal review workflow:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.benchmark_visual_qc `
  --manifest report/20260724/sleap_topdown_optimization/benchmark_manifest_pilot_event_holdout_v1.csv `
  --predictions report/20260724/sleap_topdown_optimization/round7_vs_round8_top_pilot_full_matches.csv `
  --predictions report/20260724/sleap_topdown_optimization/round7_vs_round8_side_pilot_full_matches.csv
```

Validate paths, prediction coverage, and the review ledger without opening a
window by adding `--preflight`. Preflight creates the empty review CSV/JSON
pair when it does not exist.

The native prediction SLPs are also reusable. They are useful for independent
inspection or if a match table has been reduced to metrics only. For example,
the side-view caches are:

```text
predictions/round7_side_full_topdown_round8_pilot_event_holdout_45650d8ef5.predictions.slp
predictions/round8_side_full_topdown_round8_pilot_event_holdout_45650d8ef5.predictions.slp
```

Pass them as repeatable `--predictions round7=PATH` and
`--predictions round8=PATH` inputs. Top-view full-inference caches use selection
hash `edb72d3693`. Centroid-only and oracle-centered-instance caches follow the
same naming scheme and are all under `predictions/`.

Manifest ground truth is discovered from each row's `source_slp`.
`--no-manifest-ground-truth` disables discovery, and repeatable
`--ground-truth PATH` arguments override it. The cached match tables already
include GT coordinates, so the standard command does not need network SLP
access after the raw frame is available.

## Headless export

The checked Phase 1 example is the genuine side1 tail error at frame 97219:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.benchmark_visual_qc `
  --manifest report/20260724/sleap_topdown_optimization/benchmark_manifest_pilot_event_holdout_v1.csv `
  --predictions report/20260724/sleap_topdown_optimization/round7_vs_round8_side_pilot_full_matches.csv `
  --export-frame-key mf1_b1cc05dc28c0ad30f860 `
  --export-path report/20260724/sleap_topdown_optimization/visual_qc_exports/side1_frame_97219_round7_vs_round8.png
```

This exits after writing the PNG and does not import Qt. The resulting image was
visually inspected during Phase 1.

## Review workflow

The default view compares Round 7 and Round 8 side by side. Toggle mode gives
one panel more image area. Both modes show raw pixels, GT and model skeletons,
circle mouth markers, triangle tail markers, instance numbers, numerical
errors, match bases, and frame provenance.

Filter to misses, extras, merges, swaps, shared tails, partial annotations, or
manifest review candidates. Sort by largest mouth or tail error. `N`/`P` and
the arrow keys navigate; the jump box accepts an exact `frame_key`. The mouse
wheel zooms at the cursor, and the Matplotlib toolbar provides pan and reset.

Review labels are:

1. `model_error`
2. `annotation_error`
3. `matching_error`
4. `ambiguous`
5. `exclude`
6. `looks_correct`

Number keys 1-6 choose those labels and 0 clears one. `R` and `T` select Round 7
and Round 8, `C` changes comparison mode, `E` exports the current overlay, and
`Ctrl+S` saves immediately. Labels and notes autosave on edits, navigation, and
exit.

## Outputs and resume

Decisions are written atomically to:

- `report/20260724/sleap_topdown_optimization/manual_visual_qc.csv`
- `report/20260724/sleap_topdown_optimization/manual_visual_qc.json`

The JSON records input paths and hashes plus decision counts. Reusing the same
`--review-csv` resumes prior decisions. Exports go under
`visual_qc_exports/`. These are review artifacts only; source annotations stay
unchanged.
