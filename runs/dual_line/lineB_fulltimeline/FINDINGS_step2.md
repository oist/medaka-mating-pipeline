# Line B step 2 — per-frame swap resolution (K_1_2, whole timeline)

Step 1 anchored the female only at 8 sparse windows -> ~46% of fish-frames were `low`-confidence
(spans between windows where the female's 3D-bind slot disagrees = a swap in the gap). Step 2
computes the female's slot **per frame** from appearance and resolves them. This is Line B (appearance)
directly correcting Line A (geometry) swaps. Scripts: `step2_per_frame_female.py` (Part A),
`step2_resolve.py` (Part B).

## Method
- **Part A** (stride 10, whole 0-348599): for each sampled frame build production crops from the
  saion masks via **random-access** (read only that frame's mask group -- no full-chunk decompress)
  + the local video (sequential scan, decode every 10th frame), run the clean female model, and set
  the female's geometry slot = multiview-averaged argmax P(female). 34,814 sampled frames; **92.4%
  confident (margin>0.3), median margin 0.911**. The female occupies all 3 slots {0:12481, 1:9650,
  2:12683} -- the full swap structure at frame resolution. ~14 min/view.
- **Part B**: forward-fill the confident per-frame slots (median spacing 10 frames) to frame
  resolution; a slot change between consecutive confident anchors = a swap; re-sex the 3D track
  (female = fish whose geometry_slot == appearance slot).

## Result
- **430 female-slot swaps pinned** across the timeline (vs only 6 window samples in step 1).
- **Validation: 0.990 agreement** with the independently-trusted window-anchor sexing on its `high`
  frames -> the per-frame appearance is correct where step 1 directly measured. On `med` frames
  agreement is only **0.617**: step 1's geometry *propagation* (assume no swap between agreeing
  windows) was wrong ~38% of the time -- the female swapped out and back; appearance catches it.
- **Resolved 97.3% of the window `low` fish-frames** (473,703/486,900); **30.2% of low rows flipped
  sex** vs the step-1 nearer-window guess.
- **Final confidence: 97% high / 3% low** (per fish-frame), up from step-1's 5% high / 46% low. The
  residual 3% (31,701 fish-frames) = sustained all-views-merged spans with no confident appearance
  anchor -- the irreducible core (needs interpolation, not appearance).
- **Exactly 1 female / frame across all 348,600** (0 violations).

## What this demonstrates
The dual-line thesis end-to-end on real data: Line A (3D geometry) gives positions + a swap-prone
label; Line B (appearance) is per-frame confident (0.911 margin) and **pins the 430 swaps Line A
mislabels**, lifting the whole-track sex from 5% -> 97% high-confidence and agreeing 99% with the
trusted measurements. Caveat: the 0.990 check is appearance(per-frame) vs appearance(per-window) --
internal consistency of the female model across application modes; the biological-female grounding
comes from the model's training anchor (K_1_2 manual female). Independent per-frame GT would seal it.

Output: `step2/round5_260611_3d_identified_sexed_resolved.csv` (frame, fish_id, ..., geometry_slot,
app_female_slot, app_conf, sex_resolved), `step2/appearance_swaps.csv` (430 pinned swaps),
`step2/per_frame_female_0_348599.csv`.

## Next
(3) male-vs-male model (M1 vs M2, for the aggression question) + multiview fusion; fuse Line B sex
into the production `40_female_anchor_pipeline` / `resolve_hardcore_events`; interpolate the 3% merged
core. Optional: visual gravid-belly GT spot-check on the called female at a few pinned-swap frames.
