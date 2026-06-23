# Per-Video ID-Appearance Model — Plan

Status: **planning / in progress** (2026-06-23). Owner: Makoto + Claude.

## Problem

The current ID-appearance model (segmentation-ID female anchor) is **trained on
K_1_2 and transferred to K_1_1 / K_1_3**. Measured behaviour:

- Within-K_1_2 female-vs-male: **0.72 crop / 0.76 tracklet** balanced acc
  (`model_female/female_classifier_report.json`).
- Unmasked `female_vs_rest` 0.828, **masked 0.697** (`eval_male/report*.json`) —
  masking *as currently done* lowered the within-video score.
- Male-vs-male: **~chance** (0.46–0.62), and the eval note itself flags the
  male labels as *"geometry-derived and NOISY (swappable at crossings)"*.
- Applied to K_1_1: female prob **0.361 / 0.327 / 0.341** — essentially chance,
  `ambiguous: true`.
- Crops are 128×128 with the fish only **~14 %** of the masked area (~48 px).

Two root causes, neither of which idtracker.ai suffers:

1. **Cross-video transfer.** idtracker.ai trains a *fresh* CNN per video on that
   video's own global fragments; it never transfers a model to a different
   fish-set. Our K_1_2→K_1_1 transfer is exactly where we hit chance.
2. **Geometry-derived labels.** Our supervised labels come from the swap-prone 3D
   `global_id` (purity is per-tracklet, blind to a swap *between* the scattered
   training windows), so different individuals get pooled under one label.

## Core idea

Resolve identity along the axis where each signal is actually reliable:

- **Cross-view, same instant** (side1 tracklet ↔ side2 tracklet): use the
  **3D-bind** — the validated part (`validate_tracklet_3d_bind`: ~6 px residual,
  99.9 % bijection). → safe to **pool side1 + side2** into one individual.
- **Cross-time / cross-fragment** (fragment @10k ↔ @300k): do **not** trust 3D
  `global_id` blindly. **Seed** from 3D, then **verify with the CNN under a
  "3 distinct IDs per fragment" bijection constraint** (idtracker.ai
  accumulation). CNN-vs-3D disagreements → flagged as likely 3D swaps.

So geometry does the cross-view glue (its strength); appearance does the
cross-time identity (idtracker.ai's strength).

## Decisions (from discussion)

- **A** — Train ID **per video / per experiment**, no transfer. **Pool side1 +
  side2** (similar side appearance of the same individuals). Test cross-video
  consistency only *afterwards*. Top view = separate optional model.
- **B** — Build training data from **co-visible global fragments**: frame ranges
  where ≥3 concrete tracklets coexist and are separated, trained as discrete
  individuals. Reflections handled (filtered via 3D triangulation; option to keep
  them as distractor classes).
- **C** — Keep **masking as a toggle/experiment** (not default); decide it with a
  held-out-recording A/B and better/tighter crops.

## Phases

### Phase 0 — Co-visible global-fragment selector (B)
`sam2_id_experiment/scripts/43_select_global_fragments.py`, reusing
`concrete_tracklets.py` + `multiview_clean_coverage.py`. Per recording, find
frame ranges with all 3 fish as separated continuous tracklets (non-hardcore),
3D-bind giving cross-view + seed cross-fragment identity; rank by #frames; write a
manifest. `--keep-reflections` optional distractor classes.

### Phase 1 — Crop extraction: masking toggle + tighter crops (C)
Extend `33_build_segid_dataset.py`: `--mask {none,soft,tight}`, crop to
**mask bbox + pad → resize** (fish ≫14 %), optional `--grayscale`.

### Phase 2 — Per-recording, side-pooled ID trainer (A + B)
`44_train_per_video_id.py` + `src/id_model/per_video_id.py`, reusing
`resnet_embedder.py` / `train_resnet.py` / constrained pair sampler. Pool
side1+side2 by 3D-bind cross-view group; train on the largest fragment then
**accumulate** (Hungarian + bijection, accept on margin, retrain). Outputs:
per-video 3-way model, per-fragment assignment, **CNN-vs-3D disagreement list**.

### Phase 3 — Evaluation incl. masking A/B (C)
Reuse `35_eval_segid_embedding.py`; add bijection certainty + #disagreements.
Run mask on/off through Phases 1–3 and compare on a **held-out recording**.

### Phase 4 — Cross-video consistency test (A, later)
`45_eval_cross_video.py`: embed each video's individuals per model; test whether
the female clusters across videos. Decides if any shared/transfer model is viable.

### Phase 5 — Integrate
Swap per-video ID into `40_female_anchor_pipeline.py`; keep "which slot is female"
as a separate **1-bit-per-video** call (gravid belly / size / spawn frame).

## De-risk check (do first)

Quantify how much of the male-male "chance" is **label noise** vs **weak
appearance**: measure 3-fish separability of the *existing* crops using the
label-noise-free self-supervised embedder, **within each window** (labels locally
clean) vs **pooled across windows**. A big within-window > pooled gap ⇒ label
noise ⇒ the per-video approach pays off. Both weak ⇒ appearance is the limiter
(masking / resolution / accumulation matter more).

## De-risk RESULT (2026-06-23) — label noise confirmed as dominant

Embedded existing K_1_2 side1 `clean_train` crops with the label-noise-free
self-supervised embedder (`model_resnet_full`); measured 3-fish / male-male
separability within each window (clean labels) vs pooled across windows.

| | 3-way ARI | male-male ARI | male-male acc |
|---|--:|--:|--:|
| within-window (avg of 4) | **0.82** | **0.67** | **0.87** |
| pooled across windows | 0.24 | 0.11 | 0.66 |

Cross-window nearest-centroid (train window A → test window B), RAW vs Hungarian-REMAP acc:
- 3-way **REMAP ≈ 1.00** between the 3 good windows (145k/152k/285k) while **RAW ≈ chance**
  (0.00–0.41). RAW≪REMAP = the 3D-bind `global_id` is **permuted between windows** —
  appearance persists perfectly, the geometry *naming* is swapped. Direct proof of the
  cross-window label swaps.
- Male-male persists in clean windows (145k↔152k REMAP 1.00) but is partial elsewhere.
- **Window 176300–181300 is a bad apple** (within & cross-window weak: 3-way REMAP 0.71,
  male-male 0.58) — heavy-interaction/occluded crops — and it's the window the production
  `model_female` leaned on.

**Conclusions:** (1) appearance is NOT the bottleneck for 3-way / female-vs-male — it
persists across time; (2) the pooled collapse is driven by cross-window label swaps, not
weak appearance — exactly what the per-video clean-label + accumulation/bijection approach
fixes; (3) drop/down-weight window 176300; (4) expect a strong female anchor and a moderate
male-male signal. The RAW-vs-REMAP alignment IS Phase 2's bijection step in miniature.

## Files

- New: `43/44/45_*.py`, `src/id_model/per_video_id.py`,
  `scripts/experiments/within_window_separability.py` (de-risk).
- Modified: `33` (mask/crop), `40` (integration).
- Tests first (per TDD rule): fragment selection, bijection assignment,
  side-pooling correspondence, masking. `code-reviewer` after each script.
