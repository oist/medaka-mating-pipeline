# Line B on real saion masks — first window (K_1_2, 64000-69000)

First end-to-end run of the **production Line-B identity path on the real full-length
Detectron2 masks** produced by the saion fan-out (not a one-off probe). Window 64000-69000
is in the **untrained** range (Phase-1b's generalization regime). Plan:
[docs/dual_line_tracking_plan.md](../../../docs/dual_line_tracking_plan.md).

## Pipeline (all local; masks from saion `/bucket` via `Z:`)
1. `33_build_segid_dataset` per view: link Detectron2 masks → concrete tracklets (point-in-mask),
   drop FP / merged, **production-framed crops** (`33._square_crop`, mask-bbox→resize, unmasked —
   the Phase-1b builder). side1: 12,465 clean / 51 trk (dropped 1822 FP + 693 merged); side2:
   12,861 clean / 45 trk (2291 FP + 1033 merged).
2. `34_assemble_segid_training`: pool side1+side2, attach 3D-bind `global_id` (purity 1.0 — clean
   within-window). 27,196 crops.
3. `apply_female.py`: clean female model (`model_female_clean`) → P(female) per crop.

## Result
- **Within-frame female-vs-male P-spread = 0.912** (median; mean 0.899) — matches Phase-1b's 0.911.
  Per-frame, appearance cleanly separates the female. `frac P>0.7 = 0.373` (≈ ideal 0.33 for 1F:2M).
  **The female anchor generalizes onto the production masks at an untrained window.**
- **Per-`global_id` means are muddied** (gid0 0.604 / gid2 0.440 / gid1 0.137) — NOT appearance
  weakness. The per-frame appearance-female **timeline flips between gid 0 and gid 2 at crossings**
  (gid0: 64000-66102, gid2: 66152-66650, gid0: 66651-67590, gid2: 67591-69000), while **gid 1 is
  never the female**. = a recurring within-window **3D-bind swap** (the female's geometry slot
  swaps with one male at crossings); appearance stays consistent through them.

## Why this matters
This is the dual-line thesis demonstrated on real data: **appearance is per-frame confident
(0.912) exactly where the geometry label is ambiguous (the gid0↔gid2 swap).** The swap
transitions (~66102/66150, ~67590) are candidate points where Line B resolves a Line A label
error. gid 1 = a clean male throughout; the female alternates label 0/2 with the other male.

## Next (the full Line B build)
1. Run `33`/`34` across the whole timeline (window per chunk), `window_align` across windows for a
   single global female id (the per-window female slot, like `44`).
2. Use the per-frame appearance-female to **resolve the 3D-bind swaps** (convert the `low`-confidence
   sexing spans + `resolve_hardcore_events` flags into appearance-arbitrated calls).
3. Add the male-vs-male model (Phase 2) + multiview fusion + fuse with Line A geometry.

Artifacts: `runs/dual_line/lineB_w64000/{side*_crop*, pooled/, apply_female.py, per_crop_pfemale.csv}`.
Masks: `/bucket/StephensU/medaka/detectron2_output/20251225_K_1_2_FCMC/` (= `Z:\...\detectron2_output\...`).
