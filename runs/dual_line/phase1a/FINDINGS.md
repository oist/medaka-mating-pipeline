# Dual-Line Tracking — Phase 1a Probe Findings (K_1_2)

The cheap, local probe that gates the expensive full Phase-1 saion fan-out. It makes the one
measurement Phase 0 could not: at the 3D-uncertain (resolver `review`) **female-involved**
hard-core events, does appearance independently resolve the female? Plan:
[docs/dual_line_tracking_plan.md](../../../docs/dual_line_tracking_plan.md). Phase 0 gate that
sent us here: [../phase0/FINDINGS.md](../phase0/FINDINGS.md).

**Method.** Raw Detectron2 (side model, NO purity/QC filter) on the side-view frames of the
**20 longest 3D-`review` F-involved events** (frames 63k–138k courtship bout + 120k/226k;
min inter-fish 3D distance 0.29–7.93 mm), building 128² masked crops in-loop. Then the clean
female classifier (`model_female_clean`, the Phase-0 model) on every crop. "Female resolved"
at a frame = exactly one mask with P(female)>0.7 AND ≥2 masks present (segmented apart);
multiview = resolved if either side view resolves; `appearance_confident(event)` = resolved
for ≥50% of core frames. Artifacts: `p1a_per_crop.csv`, `p1a_per_event.csv`, `p1a_summary.json`,
`p1a_probe.png`, `p1a_crop_montage.png`. Detect: 3644 frames/view, ~18k masks, ~7.5 fps local.

## Result: ⛔ the appearance ID head does NOT generalize to the untrained windows → NO-GO for saion as-is

Headline 2×2 cell (all 20 events are 3D-uncertain by construction):

| | appearance-confident | appearance-uncertain |
|---|---:|---:|
| **3D-uncertain (n=20)** | 5 (25%) | 15 (75%) |

But the 25% is **not reliable discrimination** — diagnosis shows the classifier collapses:

- **80% of ALL raw crops get P(female)>0.7** (median 0.942). With 1 female : 2 males per frame a
  working classifier should call ~33% female. It calls ~80%.
- Among top-3 fish-sized masks per frame, the **median (top P − 2nd P) gap is 0.015** and the
  **median within-frame P-spread (max−min) is 0.029** — i.e. the female and the two males get
  **near-identical** P(female). The model is **not separating sexes** on these crops.
- No gradient vs contact severity: a 0.89 mm event resolves (0.65) while a 7.93 mm event does not
  (0.02). Resolution tracks *which window/pose* the model generalizes to, not inter-fish distance.
- The 5 "confident" events DO have real signal (P-spread 0.31 vs 0.02 for the other 15) — so the
  4-window model partially discriminates on a **minority (~25%)** of out-of-window events, and
  fails on the rest.
- **Crops verified valid** (`p1a_crop_montage.png`): clean, centered, background-masked single
  fish — this is **model non-generalization, not a crop/extraction bug**.

**Why.** The probe events live in frames 63k–138k (the courtship bout), which are **outside** the
classifier's 4 training windows (145/152/176/285k). Phase-0's clean LOWO 0.99 / 95%-agreement
numbers were all **within** those windows. Phase 1a is the first test on **other windows of the
same video on raw masks**, and it confirms the exact risk `per_video_id_plan.md` was built around:
**a model trained on a few windows does not transfer across the K_1_2 timeline.** It also shows
the appearance head **needs the seg-ID QC/FP-rejection in front of it** — raw Detectron2 emits up
to 7 masks/frame (only 3 fish) and the classifier cannot reject the false positives.

**Tier-2 (swap adjudication via 3D reprojection) — unavailable.** The self-validating pre-bracket
check flagged the reprojection as **uncalibrated** (13/20 `reproj_uncalibrated`): a coordinate
offset between the round5 3D track and the calibration model means the appearance-female does not
land on reprojected ID_2 even when fish are separated. So no swap-resolution claim from reproject;
fix the coordinate frame (or use appearance-continuity tracking) in Phase 2.

## GATE: ⛔ do NOT launch the full saion Phase-1 fan-out yet

Spending full-length × 3-view × 348k-frame Detectron2 + a Line-B ID pass is **not justified** while
the ID head collapses to "female" on the windows that contain the 3D-uncertain events. Phase 1a did
its job: it caught this for ~30 min of local compute instead of hours of saion + a full Phase-2 build.

**Revised next step (recommended), before any saion:**
1. **Retrain the appearance ID per `per_video_id_plan.md`** — across the *whole* K_1_2 timeline
   (window-align + pool many windows, harder negatives, or self-supervised embedder + per-window
   anchors) so it generalizes to 63k–138k, not just the 4 training windows. *(Retraining is a
   user-decision / pause trigger — not started.)*
2. Put the **seg-ID QC / FP-rejection** (tracklet-link → drop `fp_no_tracklet`/`area_spike`) in
   front of the classifier so it sees clean single-fish crops, as the production seg-ID path does.
3. **Re-run this exact Phase-1a probe** (cheap, local) on the retrained model. Green-light the
   saion fan-out only if it now resolves the 3D-uncertain F-events.
4. Separately, fix the 3D↔calibration coordinate offset to unlock Tier-2 swap adjudication.

This keeps the expensive compute gated on a model that has *demonstrated* timeline generalization.
