# Independent 40-frame annotation runbook

> **Superseded selection notice (2026-07-26):** Human review showed that
> K_1_2 frames 1207 and 4626 are acquisition setup/fish-introduction footage,
> not three-fish behavior. Do not force three annotations into those images or
> freeze the v1 export. Use the versioned HITL replacement workflow in
> `report/20260726/sleap_hitl_benchmark_v2/README.md`; it replaces these rows
> with frames 19323 and 27474 while preserving the original manifest as
> provenance.

This package is the event-disjoint manual benchmark required before a side
model can be promoted. It must never be used for training or threshold
selection.

## Manual annotation

Use the simple PyQt instance annotator:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_blank.pkg.slp `
  --out report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_labeled.pkg.slp `
  --paths-config configs\paths.local.toml `
  --calibration-backend auto --min-views 1 --independent
```

For each of the 40 synchronized source frames, label exactly three fish in
each of `top_cam`, `side1_cam`, and `side2_cam`. Every fish must have finite
`mouth` and `tail` coordinates. Do not seed from model predictions.

Recommended workflow for each synchronized frame:

1. Click **Add instance (I)**. In the top pane, click one fish's **mouth**
   and then its **tail**. Repeat for all three fish; add mode remains active.
2. Repeat the mouth-then-tail pair for all three fish in side1 and side2.
   The GUI uses calibration only to suggest which existing color/slot is the
   same fish; it does not create coordinates in `--independent` mode.
3. Keep blue/orange/green on the same physical fish across the three views of
   this one synchronized frame. The colors are not sex or persistent
   biological identities and can reset on the next frame.
4. Use Left/Right to navigate and `Ctrl+S` to save immediately. A quiet
   quick-save also runs every two minutes and on normal exit. Reopening the
   same command automatically resumes from the output file.

The body arrow points from tail toward mouth. `F` is disabled for this locked
benchmark so every saved coordinate comes from a manual click.

The selection contains 14 K_1_1 frames, 13 K_1_2 frames, and 13 K_1_3 frames.
The 120 view frames have no model predictions or prefilled keypoints.

## Optional Round-8 prediction review

This is a separate human-in-the-loop review path; it does not replace or
modify the blank/manual workflow above. Keep the blank or manually edited SLP
as the authoritative `--in` package and supply the Round-8 predictions only as
a review overlay:

```powershell
.venv\Scripts\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_blank.pkg.slp `
  --candidate-slp .tmp\prediction_review\round8_candidates.slp `
  --out report\20260725\sleap_independent_benchmark_annotation\independent_annotation_40_round8_instance_review.pkg.slp `
  --paths-config configs\paths.local.toml `
  --calibration-backend auto --min-views 1 `
  --instance-review --independent
```

Use `independent_annotation_40_labeled.pkg.slp` as `--in` instead if the
separate review should retain existing manual annotations. Never use the
candidate SLP as `--in`, and keep the review output name distinct from both
the manual labels and the candidate overlay.

Instance review visits one active detection in one camera view at a time, so
it does not assume that prediction score ranks identify the same fish across
views. Left/Right moves to the previous/next detection across the 40 source
frames and 120 camera views. Press `A` to accept the active detection or `R`
to reject it. Dragging or otherwise editing a candidate accepts that detection.
Untouched candidates remain proposals and are not serialized as user labels.

`Ctrl+S` saves immediately; the same quiet two-minute and normal-exit
autosaves apply. Reopen the exact command to resume automatically from the
separate `--out` file, including prior accept/reject decisions.

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
