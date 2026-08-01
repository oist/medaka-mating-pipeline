# Round8 SLEAP prediction and 3D tracking QC

- Tag: `round8_260729_snn031_m6`
- Quality JSON SHA-256: `770d7a9f0e8aa31143e935e2ceac6519aebcf5ed9d488ee03af8dedbeb51e00f`
- Risk-window CSV SHA-256: `3b392d83c1ded5c9770044dffa1fba5004577424befc084f48d2b25c747fcec7`
- Risk-window JSON SHA-256: `effd188f94352483c38d0a35e221aac7ece24615add2f3a968914006f67f9b8f`
- Decision: **review**, not blanket pass. All execution/integrity checks succeeded, but biological identity at smooth contacts and crossings still requires visual review.

## Cohort summary

| Experiment | Pre-link all 3 | Residual p95 / p99 / max (mm²) | Final all 3 finite | All 3 directly observed | Step p95 / max (mm) | Top-ID bound | Longest geometry-only run |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20251225_K_1_1_FCMC | 99.744% | 2.664 / 24.330 / 9579.575 | 99.669% | 98.846% | 1.281 / 7.998 | 97.391% | 2469 f |
| 20251225_K_1_2_FCMC | 98.949% | 3.325 / 22.412 / 6646.177 | 98.954% | 98.062% | 1.357 / 7.999 | 97.000% | 5443 f |
| 20251225_K_1_3_FCMC | 99.609% | 4.560 / 23.347 / 9623.514 | 99.780% | 98.533% | 1.603 / 8.000 | 95.469% | 2812 f |

## Prediction integrity

All 36 array children completed with exit `0:0`; all expected stderr logs were empty, no error strings were found, each view has four chunks spanning `[0, 348599]`, merged indices exactly equal the chunk union, and merged video references point to the canonical source videos. SLEAP stores only labeled frames, so `labeled + zero-detection = 348600` is the integrity invariant; the zero-detection counts below are biological-quality evidence, not merge failures.

| Experiment | top zero-detection | side1 | side2 | top centroid tracks |
|---|---:|---:|---:|---:|
| 20251225_K_1_1_FCMC | 130 | 1 | 6 | 16 |
| 20251225_K_1_2_FCMC | 158 | 9 | 7 | 15 |
| 20251225_K_1_3_FCMC | 85 | 10 | 8 | 14 |

## Final gap and continuity context

Final nonfinite gaps are startup-only and are distinct from later not-directly-observed/interpolated geometry runs. After frame 4834, all three experiments retain all three finite trajectories through frame 348599; the post-frame-5000 all-three-finite fraction is exactly 100% for every experiment.

- `20251225_K_1_1_FCMC`: last frame without all three finite = 4660; nonfinite intervals: ID_1: 135-243 (109f), 253-354 (102f), 1516-1615 (100f), 1698-1797 (100f); ID_3: 0-97 (98f), 244-591 (348f), 625-724 (100f), 864-963 (100f), 1065-1164 (100f), 4561-4660 (100f).
- `20251225_K_1_2_FCMC`: last frame without all three finite = 4834; nonfinite intervals: ID_2: 0-1208 (1209f), 1212-1350 (139f), 1374-3117 (1744f), 3376-3475 (100f), 4215-4314 (100f), 4735-4834 (100f); ID_3: 0-3344 (3345f).
- `20251225_K_1_3_FCMC`: last frame without all three finite = 1899; nonfinite intervals: ID_2: 159-337 (179f), 450-650 (201f); ID_3: 0-2 (3f), 127-245 (119f), 399-652 (254f), 1141-1240 (100f), 1242-1341 (100f), 1800-1899 (100f).

All accepted adjacent steps are below 8 mm because the canonical linker uses an 8 mm adjacent-frame gate. Therefore zero >20 mm steps is expected bounded-continuity evidence, not an independent identity proof. The near-gate counts (≥7.5 mm) remain the useful motion-risk signal.

| Experiment | ≥7.5 mm steps | >8 mm | Exact-zero rows | Observed negative rows | Out-of-bounds rows | Stable mapping |
|---|---:|---:|---:|---:|---:|---|
| 20251225_K_1_1_FCMC | 629 | 0 | 0 | 38567 | 0 | ID_1→6, ID_2→14, ID_3→13 |
| 20251225_K_1_2_FCMC | 597 | 0 | 0 | 47163 | 0 | ID_1→11, ID_2→9, ID_3→8 |
| 20251225_K_1_3_FCMC | 857 | 0 | 0 | 61045 | 0 | ID_1→11, ID_2→7, ID_3→10 |

The observed negative coordinates and zero exact-zero rows confirm that `--no-shift-to-positive` was honored after fixing the spline-fill clipping defect. A stable global one-to-one mapping is structural evidence only; it cannot establish biological identity through smooth contacts/crossings.

## Round7 comparison

Only K2 has a directly available Round7 refractive 3D CSV in the canonical reconstruction tree. Round8 K2 changes all-three-finite coverage by -0.000 percentage points and adjacent-step p95 by -4.5% versus that file. K1/K3 have no Round7 reconstruction directory there. Stage-2 identity comparisons are not apples-to-apples because the older tracks were mouth-anchored and fragmented, whereas Round8 uses the default visible-node centroid.

- Round7 K2 source SHA-256: `8ebcf795ddbeec4b01c70caba93ea1cbce6896745bc5970ecc3ddd415e6e9362`

## Visual-review sampling

The 61-window manifest is a stratified review queue covering pre-link residual extremes, low top-camera separation, high adjacent 3D steps, long geometry-only runs, normal start/mid/end controls, all array chunk boundaries, and the K2 ambiguous-extra reference. Those are risk-ranked windows, not confirmed errors. Small rendered overlays and per-experiment visual decisions are recorded in the separate visual-QC manifest.

- Visual-QC manifest: `.codex_test_tmp\round8_3d_visual_qc\round8_visual_qc_manifest.json`
- Visual-QC manifest SHA-256: `8284818e4d0464a92d71ecb64220dd125bfe6241ef5e22119d6e7f4592a5bfcf`
- Hashed rendered media: 14
- K1 f198209: inconsistent gray candidates were excluded; final ID 6 is conservatively interpolated while IDs 13/14 remain direct and plausible. No obvious sampled identity swap.
- K2 f174290–174320: gray extras are visible early and filtered; the three selected identities remain geometrically plausible.
- K3 f96431–96461: the close pair preserves sampled ordering in all views and 3D, without a visible discontinuity.

## Interpretation

- K1 has the best binding fraction in the cohort and high finite coverage, but isolated residual extremes require review.
- K2 is the weakest coverage/startup-gap case; its final trajectories become fully finite after frame 4834.
- K3 has the highest residual/motion/binding risk: the largest residual maximum, highest synthetic-view exposure, lowest top-ID bound fraction, and most near-gate steps.
- No residual gate was applied. `dropped_high_residual_fish_rows = 0` means no candidate was removed by an explicit residual threshold; it must not be read as evidence that every high-residual candidate was accepted into the final linked trajectory. Residual tails and the worst-frame visual windows drive review.
- No labeled ground-truth pose or identity annotations were used. Prediction QC is structural/count integrity, and 3D QC measures geometry/continuity—not supervised accuracy.
