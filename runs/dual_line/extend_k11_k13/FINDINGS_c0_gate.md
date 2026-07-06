# C0 — cheap gate to de-risk the K_1_1 / K_1_3 dual-line extension (before HPC)

Goal: before spending saion Detectron2 full-length compute on the two new experiments, confirm the
three Line-B ingredients transfer to them: (1) the Detectron2 **detector** produces fish masks,
(2) the **female anchor** (K_1_2 clean model + `window_align`) identifies the female cross-window,
(3) the **male-male** appearance signal exists and generalizes across windows (the K_1_2 hard part,
AUC 0.91). All done **without new inference** — from the existing sampled-window `40` artifacts plus a
few minutes of local GPU training on the already-built crops.

## Result — GREEN on all three lines, both experiments

| line | K_1_1 | K_1_3 | K_1_2 (ref) |
|------|-------|-------|-------------|
| **Detector** (Detectron2 → fish masks) | ✅ 172k pooled crops, 6 windows | ✅ pooled crops, 6 windows | ✅ full-length |
| **Female** (clean anchor, `window_align`) | P(F) **0.922**, margin **0.804**, non-ambiguous, consistent /4 windows | P(F) **0.822**, margin **0.644**, non-ambiguous, consistent /4 windows | 0.93 |
| **Male-male** (swap-corrected LOSO, window-holdout) | crop balacc **0.868**, **AUC 0.948**, tracklet 1.0 | crop balacc **0.965**, **AUC 0.990**, tracklet 1.0 | 0.936 / 0.994 |
| male position-only baseline | 0.623 (≪ CNN) | 0.571 (≈ chance) | 0.477 |

- **Female** numbers are read from the pre-existing `female_anchor_clean_report.json` (the
  `window_align` production path). The naive non-aligned `female_anchor_report.json` reads "ambiguous,
  margin ~0.02" for both — that is exactly the cross-window label-noise the `window_align` path fixes;
  the clean path resolves it (the same effect proven on K_1_2). Female occupies all 3 raw geometry
  slots across windows (K_1_1 {0,1,2,2}, K_1_3 {1,2,2,1}) → the cross-window swap, correctly aligned.
- **Male-male** was the flagged risk (K_1_2 AUC 0.91). De-risked here by mirroring K_1_2 step-1b "1a":
  `43_build_corrected_segid_dataset` (K_1_2 self-supervised embedder for alignment only) →
  `39 --mode male --female-gid 2 --split window` with a non-reference holdout window. Both clear the
  bar decisively and the position-only baseline confirms it is **appearance**, not spatial position.
  K_1_3 is the cleaner of the two; K_1_1 slightly weaker but still strong.

## Caveat
This is the **co-visible / clean-crop ceiling** (sampled windows), the same regime as K_1_2 step-1b —
not the contact/merged hard core. On K_1_2 that residual was handled downstream by top-view fusion
(males 78.5%→88.2%) + interpolation (→98.4% high); expect the same shape here. The gate's job is only
to show the appearance signal exists and generalizes *before* committing HPC — which it does.

## Verdict → GO for the saion Detectron2 full-length fan-out for K_1_1 + K_1_3
Per-experiment Line B (re-validate female on full masks, retrain male + auto-labelled top per
experiment, M1/M2 global-consistency check, stage-5 → 3-fish track) then `analyze_mate_choice` → N=3.

Artifacts (gitignored scratch): `C:/udmt_validation/segid/{K_1_1,K_1_3}/{pooled_sides_corrected,
c0_male_loso}/`. Female reports: `runs/female_anchor/20251225_{K_1_1,K_1_3}_FCMC/round5_260611/
female_anchor_clean/female_anchor_clean_report.json`.
