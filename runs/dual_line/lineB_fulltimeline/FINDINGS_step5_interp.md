# Step 5 (Goal 4) — interpolate the appearance-low residual → gap-free 3-fish track

The fused track (step3) left ~11.8% male / ~3% female fish-frames `low` (no confident appearance anchor
nearby). Measuring those spans showed most are NOT actually uncertain — they're bounded by AGREEING
confident anchors, so the identity is certain. Interpolation (`step5_interpolate.py`, pure post-process,
shared `per_frame_resolve` core) classifies and fills them:

- **BRIDGED** (low run between two anchors with the SAME slot → no swap → CERTAIN): 35,455 male + 7,162
  female fish-frames upgraded from low to high.
- **EDGE** (run at the timeline start/end, one bounding anchor → fill from it): 3,515 male + 3,405 female.
- **MOTION_LOCALIZED** (low run between DISAGREEING anchors = a swap in the gap): the swap is placed at
  the frame of closest 3D approach between the two slots (the crossing). All **34 ambiguous male runs**
  (2,001 fr, median 37 fr) localized; 0 female ambiguous runs.

## Result (`step2/round5_260611_3d_individual_id_interp.csv`)
- **Female high-confidence 97% → 99.0%**; **male 88.2% → 98.4%.**
- Track-wide per fish-frame: **high 1,031,363 (98.6%) / interp 13,791 (1.3%) / ambiguous 646 (0.06%)**.
- **Irreducible ambiguous core = 646 fish-frames (0.06%)** — the simultaneous female-AND-male all-views-
  merged moments (where the independent female/male crossing-localizations collide; conflict-resolved to
  a valid assignment but flagged `ambiguous`).
- Exactly 1 F + 1 M1 + 1 M2 in all 348,600 frames.
- New columns: `id_final` (always F/M1/M2), `id_source` (high/bridged/edge/motion_localized/ambiguous),
  `id_conf` (high/interp/ambiguous).

## Interpretation
The dual-line track is now effectively complete and analysis-ready: **98.6% certain identity**, 1.3%
best-guess interpolation (edge fill or motion-placed crossing), and a 0.06% genuinely-ambiguous core
where no line has signal. The bridging is the key insight — the 11.8% male "low" was overwhelmingly
*lack of a nearby confident frame*, not actual swap ambiguity; only 4.9% of it was a real in-gap swap,
and those are motion-localizable at the crossing.

Module additions (unit-tested): `bridge_low_spans`, `localize_ambiguous_swaps`, `_runs_of` in
`per_frame_resolve.py`; tests in `tests/test_per_frame_resolve.py` (18 total; full suite 255 pass).
Caveat: motion-localized crossings and the 0.06% ambiguous core are best-guess / unresolved — external GT
(Goal 5, gravid-belly) would confirm the called identities at those frames.
