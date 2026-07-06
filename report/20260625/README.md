# Medaka tracking — progress report (2026-06-25)

**The story:** the 2026-06-22 deck ended with *"female identified, but the two males are NOT
separable by appearance."* That conclusion was **overturned** — it was cross-window label noise,
not weak appearance. This deck presents the dual-line method, the full 3-fish individual-ID track
for K_1_2, and the per-male **mate-choice** result it unlocks.

Decks (13 slides, 16:9):
- **`medaka_progress_report_20260625.pptx`** — English. Regenerate: `.venv\Scripts\python.exe report\20260625\build_report.py`
- **`medaka_progress_report_20260625_JA.pptx`** — Japanese (Yu Gothic). Regenerate: `.venv\Scripts\python.exe report\20260625\build_report_ja.py` (run the English build first to populate figures + media). Figures keep their English axis labels; only slide text is localized.

PNG previews: `slide_previews/` (EN), `slide_previews_ja/` (JA). Render both with
`pwsh -File report\20260625\render_previews.ps1` (PowerPoint COM → 1280×720 PNG per slide; python-pptx cannot render).

## Slides
1. Title — the reversal in one line ("The two males ARE separable")
2. Recap — where the 2026-06-22 report left off (female done, males believed indistinguishable)
3. **The reversal** — it was cross-window LABEL NOISE, not weak appearance (swap-corrected AUC bar)
4. The dual-line method — geometry (Line A) ∥ appearance (Line B), fused by scene
5. **The full 3-fish individual-ID track** for K_1_2 (98.6% high / 1.3% interp / 0.06% ambiguous)
6. How it's built — saion Detectron2 fan-out → per-frame → top fusion → interpolate
7. Validation — gravid-belly morphology GT + embedder global-consistency
8. **The payoff — per-male mate choice (K_1_2):** the female chose appearance-M1
9. Temporal dynamics — the preference is stable (not a late switch); courtship ≫ aggression (24.5×)
10. Cross-check — manual BORIS coding agrees on the per-male asymmetry
11. Extending to N=3 — K_1_1 + K_1_3 C0 gate passed; saion fan-out in progress
12. Production & reproducibility — Line B wired into the pipeline, 269 tests, PR #2
13. Summary & next steps

## Generated figures (re-runnable, built by `build_report.py`)
- `plots/fig_reversal_auc.png` — before/after swap-correction separability (male-male AUC 0.994, female AUC 0.982, vs chance / position-only baseline).
- `plots/fig_dual_line.png` — two-lane Line A ∥ Line B → fuse → interpolate schematic.
- `plots/fig_track_quality.png` — track-wide identity composition + per-fish high-confidence.

## Curated figures (copied from the analysis pipeline)
- `plots/mate_choice.png` — 3-panel: F–male distance, cumulative courtship-range time, temporal preference index
  (`runs/dual_line/lineB_fulltimeline/analysis/`).
- `plots/gt_montage.png` — gravid-belly GT montage, called F | M1 | M2 across the timeline incl. spawn
  (`runs/dual_line/lineB_fulltimeline/step2/`).

## Folders
- `plots/` — 3 generated figures + the 2 curated analysis figures.
- `images/` — 3D + multiview composite still (reused from the 06-22 deck).
- `videos/` — embedded 3D composite clip + the K_1_2 crossing montage.
- `posters/` — extracted poster frames for the videos.
- `slide_previews/`, `slide_previews_ja/` — PNG render of each slide (for quick viewing / email).

## Headline status (K_1_2)
- **Reversal:** the "two males not separable" finding was label noise. Swap-corrected (window_align):
  male-vs-male balacc 0.936 / **AUC 0.994** (position-only baseline 0.477 = chance); female-vs-male LOSO → 1.00, global female AUC 0.982.
- **Full 3-fish track:** `round5_260611_3d_individual_id_interp.csv` — 348,600 frames, exactly 1 F + 1 M1 + 1 M2 each,
  **98.6% high-confidence / 1.3% interp / 0.06% ambiguous** (F 99.0% / M1 98.4% / M2 98.4% high-conf).
- **Mate choice:** the female prefers/courts/spawns with appearance-**M1** — within 30 mm 287.3 s vs 180.7 s,
  sharpening to **1.89× at 8 mm contact**; spawn contact 7.4 mm (M1) vs 11.5 mm (M2). Preference **stable from the start**;
  courtship dominates aggression **24.5× by time**. BORIS-"m2" = appearance-"M1" (per-male asymmetry agrees).
- **Validation:** the called female is the rounder gravid fish across all her slot swaps (morphology GT);
  M1≠M2 confirmed by an independent embedder (ARI 0.83, cross-segment transfer 0.81–0.99, no global flip).
- **N=3 (in progress):** C0 gate passed (female P 0.92/0.82; male-male AUC 0.948/0.990); saion full-length
  Detectron2 fan-out submitted 2026-06-25 — masks computing. Mate-choice for K_1_1/K_1_3 follows once they land.
- **Production:** dual-line Line B is `per_frame_resolve.py` + `mate_choice_core.py`, wired into
  `40_female_anchor_pipeline` stage 5; 269 tests pass; committed + pushed → PR #2.

Sources: `runs/dual_line/lineB_fulltimeline/{analysis,step2}/` (mate-choice, GT montage, 3-fish track),
`runs/dual_line/{lineB_fulltimeline,extend_k11_k13}/FINDINGS_*.md` (numbers),
reused media from `report/20260622/{images,videos}/`.
