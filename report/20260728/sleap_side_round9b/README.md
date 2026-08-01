# Round 9b side-view split

Round 9b replaces the partially historically exposed validation split with a
deterministic, leakage-safe split drawn from the same 210 completed
annotations.

## Split

- Training annotations: 180 source frames / 360 side views.
- Primary validation: 30 source frames / 60 side views.
- Validation balance: 10 source frames from each of `K_1_[1,2,3]`.
- Validation categories:
  - 6 clean diverse controls
  - 6 close-contact or crossing frames
  - 2 count/partial/low-confidence frames
  - 4 long-body/crop-margin proxies
  - 6 Round-7-versus-Round-8 tail disagreements
  - 6 temporal/shared-tail anomalies

Every validation frame is more than 166 frames from both the sanitized
inherited training set and every newly reviewed training frame. The prior
split had 8 historically exposed validation frames; Round 9b has zero.

The locked 40-frame HITL benchmark and all 111 future-lock-eligible rows have
zero exact or guarded overlap and remain untouched.

## Cumulative training counts

Round 8 side training already contains the useful cumulative Round 7 labels.
Do not merge the Round 7 package again.

Three malformed inherited source groups are removed synchronously from both
side views:

- `20251225_K_1_2_FCMC`: frames `135885` and `281988`
- `20251225_K_1_3_FCMC`: frame `92579`

The resulting cumulative Round 9b training design contains:

- 3,118 side labeled frames
- 9,354 instances
- 8,812 visible-mouth centered-instance samples
- 2,628 strict all-mouth-visible centroid frames
- 13 hidden-mouth instances retained with finite raw coordinates and excluded
  naturally by strict mouth-anchor training

Primary validation contains 180 instances, of which 176 have visible mouth
anchors. Hidden coordinates and visibility flags remain authoritative.

## Files

- `round9b_selection_210.csv`: authoritative reassignment of all 210 reviewed
  source frames.
- `round9b_train_180.csv`: new-label training subset.
- `round9b_validation_30.csv`: primary checkpoint-selection validation subset.
- `round9b_split_report.json`: input/output hashes, quotas, visibility-aware
  counts, malformed-group removal, and guard verification.

The shared Round 9b train/validation SLPs were subsequently materialized under
`retrain_20260728_round9b`.

The requested centered-instance training condition is:

- crop size 448
- side-view rotation augmentation from -90 to +90 degrees
- scale augmentation 0.7--1.3
- exact post-affine crop-position jitter of +/-8 input pixels
- strict visible-mouth anchors
- OHKM enabled
- Round 8 production centered-instance weights as initialization

The completed run under
`retrain_20260728_round9b_ci_jitter8_ohkm` used crop 512 and rotation
augmentation from -180 to +180 degrees because the launcher incorrectly
preserved Round 8 control geometry. It is retained only as an invalid-condition
provenance artifact and must not be promoted as the requested Round 9b model.

## Reproduce

```powershell
.venv\Scripts\python.exe -m scripts.sleap.design_round9b_split `
  --selection-manifest report\20260727\sleap_side_ci_active_learning\selection_210.csv `
  --inherited-train-labels "\\bucket.oist.jp\bucket\StephensU\medaka\sleap_models\retrain_20260722_round8\side\train_labels.pkg.slp" `
  --reviewed-labels "C:\Users\machi\Downloads\annotator_bundle_uv (2) (1)\annotator_bundle_uv\side_ci_review210\output\side_ci_review_210_labeled_complete_full.pkg.slp" `
  --locked-manifest report\20260726\sleap_hitl_benchmark_v2\shared_benchmark_40_hitl_v2_manifest.csv `
  --future-locked-manifest report\20260724\sleap_topdown_optimization\future_locked_test_annotation_queue_v1.csv `
  --output-dir report\20260728\sleap_side_round9b `
  --guard-frames 166
```
