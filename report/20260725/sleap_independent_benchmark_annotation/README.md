# Independent 40-frame annotation runbook

This package is the event-disjoint manual benchmark required before a side
model can be promoted. It must never be used for training or threshold
selection.

## Manual annotation

The GUI was launched with:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.multiview_annotation_gui `
  --in report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_blank.pkg.slp `
  --out report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_labeled.pkg.slp `
  --autosave --independent --embed-output none
```

For each of the 40 synchronized source frames, label exactly three fish in
each of `top_cam`, `side1_cam`, and `side2_cam`. Every fish must have finite
`mouth` and `tail` coordinates. Do not seed from model predictions.

The selection contains 14 K_1_1 frames, 13 K_1_2 frames, and 13 K_1_3 frames.
The 120 view frames have no model predictions or prefilled keypoints.

## Freeze after annotation

After saving the manual output, run:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.prepare_independent_benchmark_labels `
  --output-dir report\20260725\sleap_independent_benchmark_annotation `
  --finalize-annotations report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_labeled.pkg.slp
```

The finalizer refuses incomplete or extra frames and requires:

- exactly 40 source frames and 120 view frames;
- exactly three fish per view;
- 360 finite mouths and 360 finite tails;
- a manifest hash matching the blank-package provenance.

It writes a separate locked SLP, a 360-row coordinate ledger, and SHA-256
provenance without modifying the manual input.

## Training-label mouth repair (separate task)

After the independent set is finished, the Round-8 training-only mouth repair
package can be opened with:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.multiview_annotation_gui `
  --in "\\bucket.oist.jp\bucket\StephensU\medaka\runs\sleap_topdown_optimization\mouth_anchor_completion_v1\mouth_anchor_completion_seed.pkg.slp" `
  --out "\\bucket.oist.jp\bucket\StephensU\medaka\runs\sleap_topdown_optimization\mouth_anchor_completion_v1\mouth_anchor_completion_manual.pkg.slp" `
  --min-views 2 --resume --autosave --independent --embed-output none
```

That package is explicitly `training_label_repair_only`: 443 synchronized
side groups, 886 side view frames, and 529 missing mouths. Its finalizer must
be run before the repaired labels are used to create any future training root.

```powershell
.venv\Scripts\python.exe -m scripts.sleap.prepare_mouth_anchor_completion_labels finalize `
  --round8-side-train-labels "\\bucket.oist.jp\bucket\StephensU\medaka\sleap_models\retrain_20260722_round8\side\train_labels.pkg.slp" `
  --prepared-manifest "\\bucket.oist.jp\bucket\StephensU\medaka\runs\sleap_topdown_optimization\mouth_anchor_completion_v1\mouth_anchor_completion_manifest.csv" `
  --gui-output "\\bucket.oist.jp\bucket\StephensU\medaka\runs\sleap_topdown_optimization\mouth_anchor_completion_v1\mouth_anchor_completion_manual.pkg.slp" `
  --output-dir "\\bucket.oist.jp\bucket\StephensU\medaka\runs\sleap_topdown_optimization\mouth_anchor_completion_completed_v1"
```
