# Step 2 — whole-timeline male ID → full 3-fish individual-ID track (K_1_2)

Step 1 proved males are appearance-separable. Step 2 runs the swap-corrected male model per frame
across the whole timeline and resolves M1/M2 on top of the already-resolved female, producing the
**full 3-fish individual-ID track** (F + M1 + M2) — the deliverable that makes the mate-choice /
aggression question answerable. Scripts: `step2_per_frame_male.py` (Part A), `step2_resolve_male.py`
(Part B).

Naming: **M2 = M_β = the reference-window (152k) aligned-gid_2 male; M1 = M_α = aligned-gid_0.**
Arbitrary but globally consistent (one fixed appearance boundary), which is what distinguishing the
two males requires — there is no biological male anchor like the female.

## Method
- **Part A** (stride 10, whole 0–348599, both side views, ~14 min/view): same random-access-mask +
  local-video fast-path as the female step2, but the male classifier (`model_male_clean`, head →
  P(M_β)). Output `per_frame_male_0_348599.csv`: per-frame multiview-mean P(M_β) for each geometry
  slot. 34,814 frames with ≥1 fish.
- **Part B**: female slot known per frame → the 2 non-female slots are the males → M_β = argmax P(M_β)
  among them (margin = |diff|). Confident anchors (margin > 0.3) forward-filled to frame resolution;
  female/male fill conflicts (1.39%) → low + geometry fallback. Emit `individual_id ∈ {F, M1, M2}`
  per (frame, geometry_slot) + per-row confidence.

## Result
- **Full 3-fish track: 348,600 frames, exactly 1 F + 1 M1 + 1 M2 in every frame (0 violations).**
- Confident male anchors: **58.6%** of sampled frames (median margin 0.413) — males weaker than the
  female (92%), as expected.
- **Per-individual high-confidence:** F **97.0%** (unchanged from the female line), **M1 78.5%,
  M2 78.5%.** The ~21.5% low male frames = close-contact / merged spans where appearance is uncertain
  (the male hard core — larger than the female's 3% because male-male appearance is weaker). 99.5% of
  M1/M2 fish-frames still carry a finite 3D position for those spans.
- **Global α-anchoring holds (open risk resolved):** the M_β geometry slot is highly temporally
  stable — **median run 100 frames, mean 1073, 94% of the timeline in runs ≥500 frames (~3 s), 98% in
  runs ≥100 frames**, only ~324 discrete reassignments across 348k frames. A stable identity that
  reassigns only at discrete crossings, not flicker → M_β/M_α are globally coherent identities. (The
  within-window 0-flip stability from Step 1b + LOSO AUC 0.994 corroborate; a cross-timeline embedder-
  cluster check or external GT would seal it fully.)

## Metric NOT to trust
"M_β == raw-geometry slot 2" agreement (0.18 raw / 0.29 conditioning on female∉slot2) is **not a swap
rate** — it compares the appearance call (trained on `window_align`-*aligned* gid_2) against the
*raw* `tracklet_bound` slot-2 label, which differs from aligned-gid_2 by the alignment permutation.
The disagreement is dominated by that fixed offset, so it is uninterpretable here. The trustworthy
internal checks are temporal stability (above) and the Step-1b LOSO/probe.

## Step 2c — embedder global-anchoring check → ✅ GLOBALLY CONSISTENT (no flip)
Verified the M1/M2 labels (from the classifier) with the INDEPENDENT self-supervised embedder
(`model_resnet_full`): 8,509 confident M1/M2 crops from 6 well-separated timeline segments (28k→340k).
- **Global 2-means ARI 0.827, silhouette 0.814** — M1 vs M2 form 2 coherent clusters matching the
  classifier labels (vs the 3-way eval_all silhouette 0.32; 2-way swap-corrected is cleanly separable).
- **Cross-segment nearest-centroid transfer**: centroids from ANY segment classify ANY other segment's
  crops at **0.81–0.99 (off-diagonal min 0.808, median 0.965, mean 0.937)**, matrix uniform across rows
  → M1/M2 are the SAME two physical appearance identities across the whole timeline. **No global flip.**
  (Only s1/92k slightly weaker at 0.81 — a harder window, still well above chance.)
- **VERDICT: globally consistent.** The M1≠M2 distinction is a genuine, globally stable identity pair,
  validated by a model independent of the one that assigned the labels → the aggression/mate-choice
  analysis can rest on it. Script `step2c_global_anchor.py`; matrix `step2/global_anchor_transfer.csv`.

## Next
- Multiview fusion incl. top masks (Goal 2) to shrink the male low core; caveat: clean models trained
  on SIDE crops, top is a different distribution.
- Fuse Line B sex+ID into production `40_female_anchor_pipeline` / `resolve_hardcore_events` (Goal 3).
- Interpolate the residual all-views-merged core (Goal 4); optional gravid-belly GT (Goal 5).

Output: `step2/round5_260611_3d_individual_id.csv` (adds `individual_id`, `app_male_beta_slot`,
`male_conf`, `indiv_conf`), `step2/male_reassignments.csv`, `step2/per_frame_male_0_348599.csv`.
