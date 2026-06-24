# Dual-Line Tracking — Phase 1b Findings (K_1_2): the Phase-1a NO-GO was a crop artifact

Phase-1a concluded the appearance ID head "does NOT generalize to the untrained windows"
(collapse to female, within-frame female-vs-male P-spread 0.029) and called a **NO-GO** for
the saion fan-out. **Phase 1b overturns that.** The collapse was a *crop-distribution
mismatch*, not a timeline-generalization failure. Re-probed with crops built the production
way, the same model identifies the female on the same untrained windows at **AUC 0.982**.

Plan: [docs/dual_line_tracking_plan.md](../../../docs/dual_line_tracking_plan.md).
Phase-1a it overturns: [../phase1a/FINDINGS.md](../phase1a/FINDINGS.md).

## The confound Phase-1a missed

Phase-1a fed the female model crops built a *different way* than the model was trained:

| | scale / framing | background | FP rejection |
|---|---|---|---|
| **Training** (`33._square_crop` → `model_female_clean`) | mask **bbox resized** to 128 (fish fills frame) | **unmasked** | tracklet-linked, FPs dropped |
| **Phase-1a probe** (`crop_from_mask`) | **raw 128px box** (fish ~48px, small) | **masked to black** | none (raw Detectron2, ≤7 masks/frame) |

So Phase-1a measured *crop mismatch + window generalization* confounded together and
attributed the whole collapse to windows.

## Two experiments

**(A) Scale/mask pre-test (`p1b_scale_pretest.py`, no Detectron2).** Feed the model the SAME
in-distribution training crops it scores at AUC 1.0, but shrink the fish onto a black canvas
toward the Phase-1a look:

| crop rendering | female P | male P | separation | AUC | frac P>0.7 |
|---|---:|---:|---:|---:|---:|
| training-style (baseline) | 0.955 | 0.047 | **0.908** | 1.00 | 0.34 |
| shrunk→48px on black (~p1a fish size) | 0.93 | 0.47 | 0.46 | 0.96 | 0.56 |
| shrunk→36px on black | 0.938 | **0.856** | **0.08** | 0.86 | **0.93** |

As the fish shrinks onto black, males drift up to female and the population collapses to "93%
read female" — **reproducing Phase-1a's exact symptom** (80% P>0.7, ~0 spread) on windows the
model *aces*. Proof the model is crop-distribution-sensitive.

**(B) Matched re-probe (`p1b_prep_points.py` + `p1b_detect.py` + `p1b_score.py`).** The EXACT
Phase-1a 20-event set (untrained windows 63k–138k), raw Detectron2, but each mask is
tracklet-linked (FP/merge dropped — the QC Phase-1a skipped) and **two crops per mask** built
on the SAME mask for an airtight A/B: `prod` (mask-bbox→resize, unmasked == training) vs `p1a`
(raw 128px, masked). Then the clean female model on both. 11,371 linked crops, 2,758
multi-fish frames.

| metric | PROD (training-framed) | P1A (raw128/masked) | Phase-1a |
|---|---:|---:|---:|
| within-frame female-vs-male P-spread (median) | **0.911** | 0.025 | 0.029 |
| frac crops P(female)>0.7 | **0.338** (≈ ideal 1F:2M = 0.33) | 0.856 | 0.80 |
| global female-vs-male AUC (window-aligned) | **0.982** | — | — |
| per-event resolved (mean-P gap ≥0.3) | 0.35 | 0.0 | — |

- Same masks, same model, same untrained-window frames — only the crop framing differs:
  **0.911 vs 0.025**. The P1A column reproduces Phase-1a (0.025 ≈ 0.029; 0.856 ≈ 0.80),
  confirming the re-probe is faithful and the collapse is the crop style.
- PROD `frac P>0.7 = 0.338` is exactly the 1-female : 2-male ratio — the model is **not**
  collapsing to female, it calls ~1/3 of crops female, correctly.
- The strong, label-aware number: **global female-vs-male AUC 0.982** (female mean P 0.848,
  male 0.076) over 5,877 crops the embedder could window-align across events. The female is
  identified at near-ceiling on windows the classifier never trained on.

## Honest caveats (what 1b does NOT claim)

- **Per-event "resolved" is conservative (0.35).** It needs the female's *mean* P over the
  whole event — including the deep-merge core frames where crops are scarce and `near_centroid`
  links are noisy — to beat the second fish by 0.3. The robust discrimination signals are the
  within-frame spread (0.911) and global AUC (0.982). 13/20 events were too merge-dominated to
  window-align (no 3-fish co-visibility) — expected: the merge frame itself yields no clean
  crop for either line (irreducible; geometry/interpolation must carry it).
- **Female only.** Male-vs-male (the M1/M2 aggression question) is Phase 2; the female anchor
  is what 1b validates.
- **No across-the-board retrain was needed.** The Phase-1a "retrain across the whole timeline"
  recommendation was predicated on a generalization gap that 1b shows does not exist (for the
  female anchor). Retraining is now optional, not a blocker.

## GATE: ✅ GO for the saion Detectron2 fan-out (Phase 1 proper)

The appearance head **does** generalize across the K_1_2 timeline once crops match the training
distribution. The single required correction carried into scale-up:

> **Line B inference must build crops with `33._square_crop` (mask bbox → resize, unmasked),
> NOT `crop_from_mask` (raw 128px + black mask).** That one line is the difference between AUC
> 0.982 and ~0.5. The p1a-style builder in `phase1a/p1a_detect.py` is a latent bug; the
> phase1b builder is the corrected reference.

Saion wrapper built this session (mirrors the SLEAP HPC trio):
`scripts/hpc/{detectron2_chunk.slurm, submit_detectron2_experiment.sh, merge_detectron2_chunks.sh}`
+ the Detectron2 section in `scripts/hpc/README.md`. The only new saion dependency is a
Detectron2 conda env (`D2_PYTHON`), documented there.

## Artifacts

`p1b_prep_points.py`, `p1b_scale_pretest.py` (+ `.json`), `p1b_detect.py` (.venv-detectron2),
`p1b_score.py`, `p1b_per_crop.csv`, `p1b_per_event_prod.csv`, `p1b_summary.json`, `p1b_probe.png`,
`probe_crops_side{1,2}_cam.h5`.
