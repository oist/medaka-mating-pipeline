# Round 8 active-learning runbook

Experiment: `20251225_K_1_2_FCMC`  
Selection seed: `20260721`  
Unit of selection: synchronized source frame (top + side1 + side2).

## Cohort

- `selection_360.csv`: authoritative 300-train / 60-locked-holdout manifest.
- `pilot_120.csv`: 100 train + 20 holdout.
- `expansion_240.csv`: 200 train + 40 holdout.
- Holdout sentinel: source frames 96,612--97,802.
- No training frame may lie within +/-166 source frames of a holdout event.

## Annotate the pilot

```powershell
.venv\Scripts\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\pilot_120_seed.pkg.slp" `
  --out "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\pilot_120_edits.pkg.slp" `
  --paths-config "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\configs\paths.local.toml" `
  --calibration-backend auto `
  --min-views 1
```

For every source frame: label all three real fish in all three views; correct
mouth and tail; delete reflection/FP instances; add missing/FN instances; and
do not skip the clean third view when only two are ambiguous. Use
Ctrl+Shift+S and return:

`\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\pilot_120_edits_full.pkg.slp`

## Build and annotate the expansion after pilot review

```powershell
.venv\Scripts\python.exe -m scripts.sleap.mine_round8_hard_frames `
  --paths-config "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\configs\paths.local.toml" `
  --output-dir "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\report\20260721\round8_active_learning" `
  --seed-output-dir "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC" `
  --reuse-selection --build-seed expansion

.venv\Scripts\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\expansion_240_seed.pkg.slp" `
  --out "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\expansion_240_edits.pkg.slp" `
  --paths-config "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\configs\paths.local.toml" `
  --calibration-backend auto `
  --min-views 1
```

Return `\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\expansion_240_edits_full.pkg.slp`.

## Finalize corrected labels without frame-level validation leakage

Pilot only:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.finalize_round8_labels `
  --corrected "\\bucket.oist.jp\bucket\StephensU\medaka\runs\round8_active_learning\20251225_K_1_2_FCMC\pilot_120_edits_full.pkg.slp" `
  --manifest "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\report\20260721\round8_active_learning\selection_360.csv" `
  --annotation-batch pilot `
  --paths-config "C:\Users\machi\Documents\GitHub\medaka-mating-pipeline\configs\paths.local.toml"
```

After expansion, pass both corrected packages and `--annotation-batch all`.
The finalizer writes explicit train/validation packages and configs; it never
changes Round 7 cumulative labels and never starts training.
