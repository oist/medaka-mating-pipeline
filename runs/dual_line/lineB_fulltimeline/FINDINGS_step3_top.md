# Step 3 — multiview-top probe (K_1_2): side models DON'T transfer to top, but top is clean where it matters

Goal 2 (multiview fusion incl. top) to shrink the 21.5% male low-confidence core (the close-contact
scenes). Two cheap probes, both decisive.

## Probe A — does the SIDE-trained female/male model transfer to TOP crops? → ❌ NO (collapse)
`step3_top_probe.py`: built top crops per geometry slot (top masks + local top video + top tracklet
points) at 3 windows (64k/194k/325k), ran the side-trained female + male models, compared the top
argmax to the validated side-based 3-fish track on side-confident frames.

| window | top female P-spread | female agree-w/side | top male P-spread | male agree-w/side |
|--------|--------------------:|--------------------:|------------------:|------------------:|
| 64000  | 0.018 | 0.68 | 0.015 | 0.69 |
| 194000 | 0.016 | 0.49 | 0.011 | 0.54 |
| 325000 | 0.048 | 0.58 | 0.008 | 0.72 |
| median | **0.018** (side ~0.91) | 0.576 | **0.011** (side ~0.52) | 0.692 |

Both models **collapse** on top crops (P-spread ~0.01–0.05 vs side 0.5–0.9); the "agreement" is
chance-level noise from near-equal probabilities, not confident calls. Top is a genuinely different
appearance distribution (top-down vs lateral); the side-only models do not generalize to it. (This is
the caveat flagged in the plan, now measured.)

## Probe B — is TOP clean where the SIDE male assignment is LOW? → ✅ YES (half the time)
`step3_top_coverage.py` (read-only, no model/video): at side-male-LOW frames, how many of the 3
geometry slots get a clean 1:1 top mask?

- **male-LOW frames: ≥3 clean 50.4%, ≥2 clean 81.6%, mean 2.29 slots**
- male-HIGH frames: ≥3 clean 67.6%, ≥2 81.6%→94.9%, mean 2.62 (baseline)

So **~half the male-low frames have all 3 fish cleanly separated on top** — a top model COULD recover
them. Potential gain: lift M1/M2 from 78.5% toward ~88–89% high-confidence (recovering ~half of the
21.5% low core, mostly the contact/aggression scenes).

## Decision
Top is worth adding, but NOT with the side models. The path is a **top-inclusive model, auto-labeled
from the validated side track** (no human labeling, per the project constraint):
1. At side-HIGH-confidence frames the F/M1/M2 slot→id map is trustworthy → label top crops by it.
2. Build a top crop dataset (`33._square_crop` on top masks), train top female + top male classifiers
   (window-holdout LOSO), probe transfer.
3. If good, run per-frame on top over the timeline → fuse into the multiview average → updated 3-fish
   track with a shrunk male low core.

Alternative: bank the validated side-only track (F 97% / M1,M2 78.5%) and proceed to production fusion
(Goal 3) / interpolation (Goal 4), treating top as a later enhancement.

Artifacts: `step3_top_probe.py`, `step3_top_coverage.py`, `step3_top/top_transfer_summary.csv`. Top
video copied local (session scratchpad).

## Step 3 EXECUTED (2026-06-24) — top-inclusive model + fusion → ✅ M1/M2 78.5% → 88.2%

**Auto-labeled top dataset** (`step3_build_top_dataset.py`): at side-HIGH-confidence frames (male_conf
high → trustworthy F/M1/M2 slot map) built top crops per slot, labelled by the side track's
individual_id (encoded F→gid1, M1→gid0, M2→gid2). 8,794 crops, 6 windows (38k–341k). No human labels.

**Top classifiers** via `39` (holdout `win_173000_176000`):
- top FEMALE: crop balacc 0.842, tracklet 1.0, **AUC 0.921**, position-only 0.596.
- top MALE:   crop balacc 0.826, tracklet 1.0, **AUC 0.909**, position-only 0.335 (chance → appearance).
Models: `model_top_female/`, `model_top_male/` (drop-in to `_segid_common.load_classifier`/`forward`).

**Per-frame validation on UNTRAINED windows** (64k/194k/325k; `step3_top_validate.py`): top models are
confident (no collapse) and agree with the side track — top MALE **0.951 agreement** (spread 0.268, up
from the side-model-on-top collapse 0.011); top female 0.777 (spread 0.215). Top is now usable.

**Fusion** (`step3_fuse.py`, side-primary gated, layered): keep the validated side male resolution
byte-identical; fill ONLY the side-low spans with the top vote (top male per-frame whole timeline,
`step3_per_frame_top.py` → `per_frame_top_male_0_348599.csv`). A blind side+top average was a wash
(+0.7pt, degraded side-high frames); the layered gate is the right design.
- **male high-confidence 0.785 → 0.882 (+0.098)**; top recovered **45.3% of the side-low core** (67,986
  fish-frames, the close-contact/aggression scenes).
- **0 side-high frames changed or degraded** (validated side track preserved exactly).
- exactly 1 F + 1 M1 + 1 M2 in all 348,600 frames. Residual 11.8% male-low = the all-views-merged core
  (Goal 4 interpolation).
**Fused track: `step2/round5_260611_3d_individual_id_fused.csv`** (adds `individual_id_fused`,
`male_conf_fused`, `indiv_conf_fused`). Final per-fish high-confidence: F 97.0% / M1 88.2% / M2 88.2%.
