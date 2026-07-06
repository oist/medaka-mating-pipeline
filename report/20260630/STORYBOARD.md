# Poster storyboard — Medaka female mate choice via individual-ID 3D tracking

**Date:** 2026-06-30  ·  **Author:** Makoto (with Claude Code)  ·  **Folder:** `report/20260630/`

> **Working title:** *Which male does she choose?*
> **Subtitle:** Resolving three identical fish in 3D to quantify female mate choice in medaka (*Oryzias latipes*) — a multi-camera tracking pipeline distinguishing 1 female + 2 males frame-by-frame, validated against manual ethograms, across N = 3 trials.

This is a **menu**, not a final layout. Every panel below maps to a figure file in `figures/`
(generated) or `reused/` (copied real pipeline output). The PowerPoint catalog
(`medaka_poster_figures_20260630.pptx`) has **one slide per figure** with its caption and
source so you can browse and pick. A **draft assembled poster** (`draft_poster_*.pptx`) shows
one recommended arrangement.

Assumed format: **A0 landscape, 4 columns** (most common for multi-panel data posters).
Easy to switch to portrait — say the word and I re-flow the draft.

---

## The one-sentence story

> Three medaka — one female, two *visually identical* males — cannot be told apart by geometry alone at
> the moments that matter (courtship, crossings, the mating wrap). A **dual-line tracker** (3D geometry ∥
> appearance) resolves all three frame-by-frame, and the resulting individual-ID tracks show — in **all three
> trials** — that the female **associates with, is courted by, and spawns with one specific male**, with
> **courtship (not aggression)** tracking the choice. Manual BORIS ethograms agree every time.

---

## Section 1 — Background, question & setup  *(Column 1, top)*

| Panel | Figure file | What it shows / key message |
|---|---|---|
| **1.1 The question** | `figures/s1_question.png` | Schematic: 1 ♀ + 2 ♂ in a tank → *Does she choose? Is it courtship or male–male aggression that wins?* Frames the biology. |
| **1.2 Why it's hard** | `figures/s1_challenge.png` | The two males are the **same size & colour** — a human annotator (BORIS) can only score a "dyad", not *which* male. Motivates the ML. |
| **1.3 Experimental setup** | `figures/s1_setup.png` | 3 calibrated cameras (1 top + 2 side), 20×20×20 cm tank, 166 fps, ~35 min/trial, N = 3 (K_1_1/K_1_2/K_1_3). |

**Background copy (for the poster text box):**
Female mate choice is a central question in sexual selection, but quantifying it needs *per-individual*
behaviour — who courts, who is preferred, who fathers the clutch. In medaka, two males of the same strain
are visually near-identical, so manual scoring collapses them into one undifferentiated "male" category.
We built a tracking pipeline that recovers the individual identities automatically and continuously in 3D.

---

## Section 2 — Pipeline & methods  *(Column 1 bottom → Column 2)*

| Panel | Figure file | What it shows |
|---|---|---|
| **2.1 Pipeline overview** | `reused/pipeline_3d_id_tracking.png` | **Lead methods figure.** 6-stage serpentine: multi-view capture → SLEAP pose → Detectron2 masks → appearance ID → 3D reconstruction + ID-link → stable ID through crossings. |
| **2.2 Calibration accuracy** | `figures/s2_calibration_summary.png` (+ `reused/calibration_error_3d.png`) | Learned 6D→3D model: **0.90 mm mean / 1.00 mm RMSE / 2.15 mm max** over 228 points. Good for body centroids & inter-fish distance. |
| **2.3 Pose + segmentation** | `reused/detectron2_vs_sleap_top.png` | SLEAP 2-node skeleton (mouth→tail) + Detectron2 per-instance masks per view. |
| **2.4 SLEAP retraining** | `reused/sleap_round4_vs_round5_eval.png` | Round-5 model in production (centroid mOKS ≈ 0.99). |

**Methods copy:**
Acquisition (campy/Basler) → bead-cube calibration (learned polynomial 6D→3D, implicitly absorbing
water/glass refraction) → SLEAP pose on the OIST *saion* HPC → Detectron2 instance segmentation (HPC
fan-out, 3 views × 348,600 frames) → 3D reconstruction by cross-view triangulation → identity linking.

---

## Section 3 — Machine-learning tracking results  *(Column 2 bottom → Column 3)*

| Panel | Figure file | What it shows / message |
|---|---|---|
| **3.1 The dual-line method** | `figures/s3_dual_line.png` | Two independent lanes — **Line A geometry** (strong in open water, smooth-swaps at crossings) ∥ **Line B appearance** (strong at contact) — fused by scene, then interpolated. *Each line carries the frames the other fails on.* |
| **3.2 The reversal** | `figures/s3_reversal_auc.png` | **ML headline.** "Males not separable" was **label noise**, not weak appearance. Swap-correction (`window_align`): male-vs-male chance → **AUC 0.994**, female-vs-male → **1.00**; position-only baseline 0.48. |
| **3.3 Identity ≫ position (N=3)** | `figures/s3_separability_n3.png` | Corrected per-experiment classifiers (K_1_1, K_1_3): female-vs-male balanced-acc **0.86 / 0.96** vs position-only baseline **0.26 / 0.44** → it's **appearance, not location**. (Male–male is the *harder* task — correct structure.) |
| **3.4 Track quality** | `figures/s3_track_quality.png` | K_1_2 full track: 348,600 frames, exactly 1 F + 1 M1 + 1 M2 each, **98.6 % high-conf / 1.3 % interp / 0.06 % ambiguous**. |
| **3.5 Confidence ladder** | `figures/s3_confidence_ladder.png` | How the male core is recovered: side appearance **78.5 %** → + top-view fusion **88.2 %** → + interpolation **98.4 %** high-confidence. |
| **3.6 ID composition** | `figures/s3_id_composition.png` | Source breakdown of every fish-frame (high / bridged / edge / motion / ambiguous) + per-fish high-conf (F 99.0, M1/M2 98.4 %). |
| **3.7 3D trajectories** | `figures/s3_trajectory_topdown.png` | Top-down XY of F/M1/M2 through the spawn approach — IDs held through close contact (min inter-fish < 3 mm). |
| **3.8 Inter-fish distance** | `figures/s3_interfish_distance.png` | Full-timeline F–M1, F–M2, M1–M2 distance — how often the fish are in the hard contact regime. |
| **3.9 Female validation (morphology)** | `reused/gt_montage.png` | The called **female** is the deeper-bodied / **gravid** fish in every row *and across her geometry-slot swaps* → tracks biology, not a slot. |
| **3.10 Appearance separability** | `reused/segid_per_view_pca.png`, `reused/female_prob_by_tracklet.png` | Embedding/PCA: the three fish separate; per-tracklet female probability is clean. |

---

## Section 4 — Biology: female mate choice  *(Column 3 bottom → Column 4)*

| Panel | Figure file | What it shows / message |
|---|---|---|
| **4.1 Mate-choice master (per trial)** | `figures/s4_matechoice_K_1_2.png` (+ `_K_1_1`, `_K_1_3`) | 3-panel per trial: F–male distance over time · cumulative courtship-range time · preference index. Spawn marked. **Lead biology figure.** |
| **4.2 Association index (N=3)** | `figures/s4_assoc_index_n3.png` | She spends **0.61 / 0.71 / 0.80** of close-range time with the preferred male — above chance (0.5) in **every** trial. |
| **4.3 Preference sharpens at contact** | `figures/s4_sharpening_n3.png` | Preference ratio rises as the distance threshold tightens 30→8 mm (K_1_1 up to **10.8×**) — strongest exactly where it matters. |
| **4.4 Courtship M1 vs M2 (N=3)** | `figures/s4_courtship_M1M2_n3.png` | The preferred male courts **1.7–4.3×** more (e.g. K_1_1 545 vs 127 s). |
| **4.5 Courtship ≫ aggression** | `figures/s4_courtship_vs_aggression_n3.png` | Courtship : aggression time = **19–25×** in all trials → **courtship**, not male–male aggression, tracks the choice. |
| **4.6 Preference is stable in time** | `figures/s4_temporal_stability_n3.png` | Per-bin preference index across the trial — the female favours the eventual mate **from the start** (not a late switch). Spawn marked. |
| **4.7 Spawn outcome (N=3)** | `figures/s4_spawn_n3.png` | At the spawn she is **4.9–7.4 mm** (contact) from the preferred male and **>100 mm** from the other → the choice has a clear outcome. |
| **4.8 N=3 summary table** | `figures/s4_summary_table.png` | One-look table: high-conf, assoc index, courtship, sharpening, spawn, BORIS — all three trials agree. |

---

## Section 5 — Biology: ethogram cross-validation  *(Column 4 bottom)*

| Panel | Figure file | What it shows / message |
|---|---|---|
| **5.1 Behaviour ↔ distance** | `figures/s5_behavior_distance.png` (+ `reused/behavior_distance_boxplot.png`) | Independent BORIS coding lines up with geometry: baseline **40** → courtship **28** → touch **17** → wrap **8 mm**. |
| **5.2 Ethogram + distance timeline** | `reused/ethogram_distance_timeline.png` | BORIS events overlaid on the female–male distance trace (same frame clock, 166 fps). |
| **5.3 BORIS agrees on the chosen male (N=3)** | `figures/s5_boris_vs_appearance_n3.png` | Hand-coded male→female courtship is **2.5 / 12.4 / 14.7×** asymmetric toward the *same* male the appearance track calls preferred — independent confirmation in every trial. |
| **5.4 Wrap acceptance** | `figures/s5_wrap_acceptance.png` | Accepted wraps last **28 s** vs rejected **3.6 s** — matches the manual write-up. |
| **5.5 Behaviour inventory** | `figures/s5_behavior_inventory.png` | BORIS counts/time by category (courtship / competitive / mating). |
| **5.6 Spawn close-up** | `reused/spawn_closeup.png` | Trajectory zoom at the spawn frame. |
| **5.7 Cohort ethogram supplement** | `boris_cohort/medaka_boris_cohort_figures_20260630.pptx` | Optional broader manual-annotation panels from all **23 provided BORIS CSVs**: 926 parsed events, 15 spawn-coded observations, behavior repertoire, m1tf/m2tf asymmetry, wrap endpoints. |

---

## Conclusions box  *(Column 4, bottom strip)*

- A **dual-line tracker** resolves three visually identical medaka into continuous individual-ID 3D tracks
  (**98.6 % high-confidence**, gap-free, exactly 1 ♀ + 2 ♂ every frame).
- This makes **per-individual** mate choice measurable for the first time in this assay.
- **N = 3 / 3 trials:** the female **associates with, is courted by, and spawns with one specific male**;
  the preference **sharpens toward contact** and is **stable from the start**.
- **Courtship, not male–male aggression**, tracks the choice (courtship : aggression ≈ 20–25×).
- **Independently validated:** gravid-belly morphology (female), an independent embedder (M1≠M2), and
  manual **BORIS** ethograms (chosen male agrees in direction, every trial).

**Caveats to keep honest:** M1/M2 labels are arbitrary per experiment (globally consistent within a trial,
not externally GT'd per male); cross-video *same-individual* identity is unresolved (appearance embedder is
unreliable across videos); the contact "hard core" (~1 % of frames) is interpolation-bridged.

---

## Figure inventory (quick index)

**Generated (`figures/`):** s1_question, s1_challenge, s1_setup, s2_calibration_summary, s3_dual_line,
s3_reversal_auc, s3_separability_n3, s3_track_quality, s3_confidence_ladder, s3_id_composition,
s3_trajectory_topdown, s3_interfish_distance, s4_matechoice_{K_1_1,K_1_2,K_1_3}, s4_assoc_index_n3,
s4_sharpening_n3, s4_courtship_M1M2_n3, s4_courtship_vs_aggression_n3, s4_temporal_stability_n3,
s4_spawn_n3, s4_summary_table, s5_behavior_distance, s5_boris_vs_appearance_n3, s5_wrap_acceptance,
s5_behavior_inventory.

**Reused real pipeline output (`reused/`):** pipeline_3d_id_tracking, calibration_error_3d,
calibration_reprojection_2d, calibration_volume_error, detectron2_vs_sleap_top, detectron2_vs_sleap_side1,
sleap_round4_vs_round5_eval, segid_per_view_pca, female_prob_by_tracklet, segid_fish_crops_top,
tracklet_coverage_stifd, gt_montage, 3d_multiview_composite_still, ethogram_distance_timeline,
behavior_distance_boxplot, spawn_closeup, mate_choice_{K_1_1,K_1_2,K_1_3}.

**Supplement (`boris_cohort/`):** editable BORIS cohort figure deck plus normalized CSV/JSON tables
from all 23 user-provided manual ethogram exports.

**Build:** `.venv\Scripts\python.exe report\20260630\build_figures.py` then `… build_catalog.py`.
All figures regenerate from the real CSV/JSON outputs — re-runnable.
