# Side centered-instance active-learning set

## Decision

Do not run full-length Round 8 inference on all six side videos for this
annotation round.

Complete Round 7 predictions already cover both side cameras for
`K_1_[1,2,3]` over frames `0--348599`. They are used for broad, inexpensive
mining. The current Round 8 models are then run only on the selected sparse
pool to remove failures that Round 8 already fixed and to supply the editable
GUI candidates. This sparse inference processed 1,260 images in 229 seconds;
full inference over six recordings would cost roughly 19 GPU-hours.

The hybrid design is important:

- Round 7 alone would bias the batch toward old-model errors.
- Full Round 8 is unnecessary for selecting a small annotation batch.
- Round 7 versus Round 8 tail disagreement is useful for centered-instance
  mining when their matched mouth anchors remain close.
- Large mouth disagreement is treated as centroid evidence, not as an isolated
  centered-instance failure.

## Final selection

The final set contains 210 synchronized source frames and both side views:

| Experiment | Train | Validation | Side images |
|---|---:|---:|---:|
| `20251225_K_1_1_FCMC` | 70 | 10 | 160 |
| `20251225_K_1_2_FCMC` | 40 | 10 | 100 |
| `20251225_K_1_3_FCMC` | 70 | 10 | 160 |
| **Total** | **180** | **30** | **420** |

At three fish per image, the expected workload is 1,260 reviewed fish
instances. The 70/40/70 training allocation deliberately counters the existing
side-label imbalance (about 250/2,090/250 labeled frames for
K_1_1/K_1_2/K_1_3).

Selection categories:

| Category | Frames |
|---|---:|
| Round 7 versus Round 8 tail disagreement | 51 |
| Close contact or crossing | 51 |
| Temporal or shared-tail anomaly | 39 |
| Long-body/crop-margin proxy | 27 |
| Count, partial, or low-confidence | 21 |
| Clean diverse control | 21 |

`long-body/crop-margin proxy` is a selection heuristic for the planned
448-pixel crop and ±8--16-pixel crop-position augmentation. It is not a direct
crop-offset sensitivity measurement.

## Leakage controls

Selection excluded:

- every locked HITL-v2 benchmark event with a ±166-frame guard;
- future locked-test-eligible events with the same guard;
- the existing Round 8 holdout and the `96612--97802` sentinel;
- existing side training labels with a local redundancy guard;
- existing side validation labels with a ±166-frame guard; and
- the previous Round 8 active-learning selection.

The new 30-frame validation subset is also separated from the new training
frames by ±166 within each experiment. Never merge rows with
`split_role=validation` into training.

## Artifacts

Small, versioned artifacts:

- `selection_210.csv`: authoritative split/category manifest.
- `selection_210_summary.json`: selection counts and scientific notes.
- `candidate_pool_630.csv`: broad Round 7 pool.
- `candidate_pool_630_round7_instances.csv`: Round 7 instance evidence for
  sparse reranking.
- `candidate_pool_630_summary.json`: cache and exclusion provenance.
- `review_bundle_audit.json`: hashes and structural/readability validation.
- `round8_deficient_views.csv`: 16 views with 16 missing Round 8 fish slots.

Large review artifacts remain outside Git:

```text
C:\Users\machi\Downloads\annotator_bundle_uv (2) (1)\annotator_bundle_uv\side_ci_review210\
```

Key validated files:

- `side_ci_review_210_blank.pkg.slp`
  - SHA-256:
    `b68d48f957fee804dd1388fab94b1f7f359ac412545d03aa8770f6fcdcd5b7f4`
  - 420/420 embedded images decode at 1200×1200 RGB
  - 0 authoritative instances
- `side_ci_review_210_round8_candidates.slp`
  - SHA-256:
    `4fc6e79dd06d7739fa9a7c35ae578525c26a8b43b76cbf6dc66daa0ea6e03c98`
  - 420 labeled frames in the same order as the blank package
  - 1,244 `PredictedInstance` candidates, capped at three per view

## Start annotation

Double-click:

```text
C:\Users\machi\Downloads\annotator_bundle_uv (2) (1)\annotator_bundle_uv\side_ci_review210\run_instance_review.bat
```

The launcher uses the updated repository annotator while preserving the user's
original standalone bundle GUI.

In instance review:

- Left/Right moves through individual fish candidates.
- Accept or correct every real fish.
- Reject reflections and false positives.
- After candidate review, add the 16 known missing fish slots and any new
  deficits caused by rejecting false positives.
- Finish with exactly three real fish in each side image.
- Every centered-instance training instance must have a finite mouth anchor;
  label the tail when visible and preserve an explicit missing tail when it is
  genuinely unobservable.

The GUI autosaves every two minutes. Use Ctrl+S for a quick save and
Ctrl+Shift+S for a fully embedded package.

The expected output is:

```text
C:\Users\machi\Downloads\annotator_bundle_uv (2) (1)\annotator_bundle_uv\side_ci_review210\output\side_ci_review_210_labeled.pkg.slp
```

After annotation, validate human-only instances and split the package by
`split_role` before adding the 180 training frames to centered-instance
training.
