# Dual-Line Tracking Plan — Geometry (3D) + Appearance (Detectron2+ID) in parallel

**Status:** Phase 0 done (gate YES). **Phase 1a NO-GO was OVERTURNED by Phase 1b
(2026-06-23): the appearance ID head DOES generalize across the K_1_2 timeline — the Phase-1a
"collapse" was a crop-distribution artifact, not a generalization gap.** With crops built the
production way (`33._square_crop`, mask-bbox→resize, unmasked) the clean female model identifies
the female on the untrained windows (63k–138k) at **AUC 0.982** (vs ~0.5 with the Phase-1a
`crop_from_mask` raw-128/masked crops). **GATE = ✅ GO for the saion Detectron2 fan-out.** The
across-timeline retrain is now optional, not a blocker. Full 1b results +
the crop-builder fix: [runs/dual_line/phase1b/FINDINGS.md](../runs/dual_line/phase1b/FINDINGS.md);
superseded Phase-1a: [runs/dual_line/phase1a/FINDINGS.md](../runs/dual_line/phase1a/FINDINGS.md);
Phase 0: [runs/dual_line/phase0/FINDINGS.md](../runs/dual_line/phase0/FINDINGS.md). saion wrapper
built: `scripts/hpc/{detectron2_chunk.slurm,submit_detectron2_experiment.sh,merge_detectron2_chunks.sh}`
(+ README §"HPC batch Detectron2 inference"). Authored 2026-06-23. Read top to bottom.

> **CRITICAL CORRECTION carried forward:** Line B inference MUST build crops with
> `33._square_crop` (mask bbox → resize, **unmasked**), NOT `crop_from_mask` (raw 128px box +
> black mask). That one difference is AUC 0.982 vs ~0.5. The `phase1a/p1a_detect.py` builder is
> a latent bug; `phase1b/p1b_detect.py` is the corrected reference.
Background on the appearance model lives in [per_video_id_plan.md](per_video_id_plan.md);
project memory entry: `project-id-appearance-label-noise`.

## The idea (user's framing)

Run **two independent tracking lines** and **fuse them by scene-strength** (each carries
the scenes the other fails on). NOT "replace 3D" — complement it.

- **Line A — geometry:** SLEAP → 3D reconstruction → 3D-bind ID (existing, validated).
  *Strong* when fish are 3D-separated; *fails* at the 3D hard-core (fish together in
  depth) where the per-frame linker swaps identities.
- **Line B — appearance / 2D:** Detectron2 instance masks → 2D tracklets → clean ID
  classifier → per-view + multi-view-fused 2D ID. Independent of calibration/3D **at
  inference**. *Strong* when ≥1 view has clean separated masks + distinct appearance;
  *fails* at simultaneous occlusion in all views.
- **Fusion:** per-scene confidence arbitration. Appearance carries crossings where 3D
  swaps; geometry carries all-views-merged frames where appearance has no usable crop.

**The whole bet** is that the two lines fail on *different* frames. The single most
important measurement in this plan is the **failure-mode overlap** (Phase 0 / Phase 2):
complementary → fusion wins big; coincident → it doesn't.

### Constraints / principles (from the user)
- **No human labeling.** Both lines are *trained* with cross-aid but *run* independently:
  Detectron2 ← SAM2 + SLEAP prompts; ID classifier ← 3D-bind-seeded labels. Keep it that
  way. Dense-scene Detectron2 improvement uses the **3D-reproject → SAM2-prompt → per-fish
  mask** loop (`reproject_slot_points` already exists) — auto-labels, no human.
- **Compute is available on saion HPC** (GPU + large-GPU nodes). Full-length Detectron2 ×
  3 views is acceptable there; mirror the existing `scripts/hpc/` SLEAP setup.
- Detectron2 model quality (esp. dense scenes) is being assessed/improved by the user in
  parallel; SLEAP dense-scene improvement is a *separate* user workstream that feeds Line A.

## What's already established this session (don't re-derive)

Appearance ID is **strong** (this was the surprise — earlier belief was "too weak"):
- Clean swap-corrected, co-visible, leave-one-segment-out on K_1_2:
  **female-vs-male 0.99, 3-way individual 0.93, male-vs-male 0.90.**
- The old weakness (0.72 within-video, ~chance transfer) was **cross-window label noise**
  (3D-bind `global_id` permuted between scattered training windows) + one bad window
  (`win_176300_181300`), NOT weak appearance.
- Cross-video transfer of the clean model WORKS: K_1_1 P(female) 0.92 / margin 0.80,
  K_1_3 0.82 / 0.64, both consistent. (Female is *not* the largest fish — model uses real
  appearance, not a size shortcut.)
- Detection (prior UDMT test, memory `project-udmt-individual-tracking`): Detectron2-mask
  tracking **detects all 3 fish ~95–99%**, but ID **swaps/fragments at crossings** — that
  test is essentially the Line-B prototype minus the strong ID classifier.
- `multiview_clean_coverage`: ≥2 views are cleanly separated most frames; irreducible
  all-views-hard core ~**4–11%**.
- The 3D linker really does swap: K_1_1 female occupies raw 3D slot {139k→0, 204k→1,
  268k→2, 333k→2} across windows. Whole-track sexing of K_1_1 → exactly 1 female/frame
  but confidence high 60k / med 627k / **low 358k** (the `low` = swap-uncertain spans).

**All caveats:** every 0.9x number is on **clean co-visible crops** (the easy regime).
**The classifier has NOT been measured on hard/occluded/merged crops** — that is the
make-or-break unknown for Line B and the first thing Phase 0 probes.

## Key assets / paths (verified this session)

Committed in `ced9fbb` on `main`:
- `sam2_id_experiment/src/id_model/window_align.py` — co-visible select, appearance-bijection
  swap-correction (NaN/degenerate guards), `sex_track_from_window_anchors`. Tests:
  `tests/test_window_align.py` (8 pass).
- `sam2_id_experiment/scripts/43_build_corrected_segid_dataset.py` — build corrected
  co-visible dataset (drop-in for `37`).
- `sam2_id_experiment/scripts/44_apply_clean_female_anchor.py` — align target windows,
  call female, write whole-track sexed CSV with per-row `sex_confidence`.
- `sam2_id_experiment/scripts/40_female_anchor_pipeline.py` — orchestrator; defaults to the
  clean model + `--female-embedder`; stage 4 → `44` (legacy `38` behind `--legacy-anchor`).

Models / data (machine-local scratch, gitignored):
- Clean female model: `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_female_clean/model.pt` (0.994/1.000).
- Self-supervised embedder (for alignment): `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_resnet_full/model.pt`.
- Corrected dataset: `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/pooled_sides_corrected/`.
- Raw pooled (side1+side2, 4 windows, K_1_2): `Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/pooled_sides_full/`
  (crop_metadata cols incl `frame, view, window, global_id, qc_status` where qc ∈
  {`clean_train`, `reject_area_spike`}).
- Per-view existing tracklet/3D-bind: `Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind_extend_smooth/<view>.tracklet_bound.csv`
  + `hardcore_events.csv`; resolver output `.../tracklet_bind/hardcore_events_resolved.csv`.
- Target pooled crops: `runs/female_anchor/<exp>/round5_260611/pooled/{crops.h5,crop_metadata.parquet}`.
- 3D identified track: `Z:/StephensU/medaka/reconstruction_3d/round5_260611/<exp>/round5_260611_3d_identified.csv`
  (cols `frame, fish_id (ID_1/2/3), X, Y, Z`; `global_id k ↔ fish_id ID_{k+1}`, see
  `reproject_slot_points.slot_from_fish_id`).

Reusable code:
- Detectron2 inference: `sam2_id_experiment/scripts/13_run_detectron2_segmentation.py`
  (runs in `.venv-detectron2`); current model `outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/`.
- 2D tracklets: `scripts/tracking/concrete_tracklets.py` (stIFD/edge-extend gates),
  `UDMT/tracking.py` (`python -m UDMT track`), `scripts/tracking/multiview_clean_coverage.py`.
- 3D bind / swaps: `scripts/tracking/bind_tracklets_3d.py`, `resolve_hardcore_events.py`.
- 3D→2D reproject + SAM2 prompt seed: `sam2_id_experiment/src/id_model/reproject_slot_points.py`,
  `sam2_id_experiment/scripts/24_seed_prompts_from_3d.py`.
- HPC scaffolding to mirror: `scripts/hpc/` (SLEAP array jobs), `scripts/hpc/README.md`.

Environments: main `.venv` (torch), `.venv-detectron2` (Detectron2, gitignored),
`.venv-idtrackerai`. Run as `.venv\Scripts\python.exe -m ...`.
Recording geometry: K_1_2 = **348,600 frames @ 166 fps**, views `top_cam/side1_cam/side2_cam`.

## Phases

### Phase 0 — Cheap gate (no new compute; do this first)
Goal: decide whether Line B is worth scaling, and get the first failure-overlap number.
All from existing K_1_2 artifacts.

1. **Hard-crop ID diagnostic.** Run the clean model (`model_female_clean` + embedder) on
   the crops we *excluded*: `qc_status == reject_area_spike` and frames with only 1–2 fish
   present (non-co-visible). No GT, so measure **prediction confidence** (margin) and
   **within-tracklet ID stability** vs the same on clean crops. If confident/stable on
   hard crops → Line B can survive crossings. If uncertain/flickery → Line B is bounded to
   clean frames (fusion only fills, doesn't carry).
2. **Detectron2 coverage audit.** From existing window masks/crops, per view: fraction of
   frames with 3 clean separated masks vs merged/<3, stratified by scene (use the per-frame
   min-pair distance / `concrete_tracklets` clean criterion).
3. **Failure-overlap preview.** Take K_1_2 3D hard-core frames (`hardcore_events*.csv`, the
   924 events / `low`+`suspect` from the resolver) and ask: at those frames, is the
   appearance line **confident** (complementary, good) or also uncertain (coincident, bad)?
   Output a 2×2: {3D-confident, 3D-uncertain} × {appearance-confident, appearance-uncertain}.

**Gate:** proceed to Phase 1 only if (1) shows usable hard-crop confidence AND (3) shows
meaningful complementary mass (appearance confident where 3D is not).

#### Phase 0 — RESULTS (2026-06-23), all on existing K_1_2 artifacts

Artifacts in `runs/dual_line/phase0/` (per sub-step: CSV + PNG + `FINDINGS.md` section).
Female confirmed = **ID_2 / global_id 1** (margin 0.79).

1. **Hard-crop ID diagnostic → ✅ appearance survives.** Frozen clean female model + embedder
   over all 33,179 crops. On **held-out** (side2) excluded `area_spike` (merge) crops: margin
   **0.91**, **100% confident**, **92%** within-tracklet self-consistent; pooled area-spike
   agreement **95.4%** (only 4.6% confidently-wrong, absorbed by fragment-vote — within-tracklet
   agreement 95–100%). The old "bad apple" (win_176300) classifies fine per-crop (agreement
   0.95) → the prior weakness was cross-window *alignment*, not appearance. Embedding coherence
   saturates at 1.0 (uninformative; loss ~0.001). *Caveat:* only measures crops that exist; the
   fully-merged / all-views-occluded case yields no crop and isn't testable here.
2. **Detectron2 coverage → ⚠️ good in separated scenes, contact scenes unmeasurable.** seg-ID
   purity-filtered masks. **Multiview (win_176300): ≥1 view 3-clean 97.8%, all-views-hard only
   2.2%** (matches prior ~4%). BUT (a) purity filter drops ~65% of raw masks ⇒ present/span only
   0.34–0.56 (lower bound; tracking linker keeps more); (b) **all clean-mask frames have 3D dist
   >20 mm — zero coverage below 20 mm** (area-spike rejection removes near-contact crops). So the
   decisive contact-scene mask-separability is **unmeasurable from this data** → Phase 1's job.
3. **Failure-overlap 2×2 → ✅ structurally complementary.** Over all 924 hard-core events:
   **44.5% are 3D-uncertain** (`review`); **60.6% of those involve the female** ⇒ **26.9% of all
   events are 3D-uncertain AND female-addressable** (confident female model applies), + 17.5%
   male-male (needs male model, Phase 2). The *measured* 2×2 (17 in-window events) shows 0 in the
   decisive cell — **sampling artifact**: all 3 in-window review events are M-M and crops exist in
   only 4 windows. 0/17 events have a crop AT the merge (purity filter) → Line B re-IDs in the
   bracket; bracket female reads are clean (p=0.955 both sides), confirming the female as
   *bystander* even at the extreme suspect_swap f179245 (min 1.95 mm).

**GATE = YES (qualified).** (1) strong PASS, (3) structural PASS, (2) partial. Lines are
complementary, not coincident. The one number Phase 0 could not produce — does appearance resolve
a 3D-review **F-involved** event directly — is unobservable because the purity filter removed all
<20 mm crops and crops exist in only 4 windows. → **make that Phase 1's first deliverable.**

### Phase 1 — Line B detection at scale (saion)

> **Phase-1a probe FIRST (cheap, gates the full fan-out).** Before full 3-view × 348k-frame
> inference: run raw Detectron2 (NO purity filter) on the frames of the ~50 largest 3D-`review`
> **F-involved** events (a few thousand frames), build bracket crops, and directly fill the
> decisive 2×2 cell (does appearance resolve where 3D is uncertain?). Green-light the full run
> below only if it does. This is the saion / new-inference boundary.
>
> **Phase-1a EXECUTED 2026-06-23 → ⛔ NO-GO.** N=20 longest 3D-`review` F-events (frames
> 63k–138k, min 0.29–7.93 mm), raw Detectron2 (side model) + the Phase-0 female classifier.
> Result: the classifier **collapses to "female"** on these untrained-window raw crops — 80% of
> crops P(female)>0.7, female-vs-male within-frame P-spread **0.029** (0.31 on the ~25% of events
> it partly handles, 0.02 on the rest), **no gradient vs contact distance**, crops verified valid
> (not a bug). This is the `per_video_id_plan` generalization gap, measured: the 4-window model
> does not transfer across the timeline, and the head also needs seg-ID QC/FP-rejection in front
> of it (raw Detectron2 emits up to 7 masks/frame for 3 fish). Tier-2 swap-adjudication via 3D
> reproject was **uncalibrated** (coordinate offset; self-check caught it). Scripts:
> `runs/dual_line/phase1a/{p1a_detect.py (.venv-detectron2), p1a_score.py}`. Detectron2 side
> model: `outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/`; ~7.5 fps local.
>
> **Phase-1b EXECUTED 2026-06-23 → ✅ OVERTURNS the 1a NO-GO.** The 1a "revised next step"
> (retrain across the timeline) was predicated on a generalization gap that 1b shows does not
> exist. Re-ran the EXACT 1a 20-event set, but built crops the production way
> (`33._square_crop`, mask-bbox→resize, unmasked) + tracklet-linked FP-rejection, A/B'd against
> the 1a `crop_from_mask` style on the SAME masks. Result: PROD within-frame female-vs-male
> P-spread **0.911** vs P1A **0.025** (≈1a's 0.029); PROD frac P>0.7 **0.338** (≈ideal 1F:2M)
> vs P1A 0.856 (≈1a's 0.80); **global female-vs-male AUC 0.982** on the untrained windows. A
> no-Detectron2 pre-test corroborated: shrinking in-distribution crops onto black reproduces the
> collapse on windows the model aces. So 1a conflated crop-style with windows. The female anchor
> generalizes; **(1) retrain is now OPTIONAL** (not a blocker — defer unless male-vs-male needs
> it in Phase 2); **(2)** the QC/FP-rejection IS wired into the 1b builder; **(4)** the
> 3D↔calibration coord offset for Tier-2 is still open. See
> [runs/dual_line/phase1b/FINDINGS.md](../runs/dual_line/phase1b/FINDINGS.md).

**Phase 1 proper — saion fan-out (wrapper BUILT 2026-06-23, ready to submit).** Detectron2
full-length × 3 views as HPC array jobs, mirroring the SLEAP `scripts/hpc/` trio:
`detectron2_chunk.slurm` (runs `13_run_detectron2_segmentation` on one frame range),
`submit_detectron2_experiment.sh` (3 views × N chunks), `merge_detectron2_chunks.sh`
(copy mask chunks → `/bucket`); docs in `scripts/hpc/README.md`. Output: per-view
`<view>.masks_<S>-<E>.h5` chunks → `/bucket/StephensU/medaka/detectron2_output/<exp>/`, consumed
per-window by `33_build_segid_dataset`. **The only new saion dependency is a Detectron2 conda
env** (no module like SLEAP has) — pass via `D2_PYTHON`; build-once recipe in the README.
Only the SIDE Detectron2 model is trained, so `top_cam` is skipped unless `TOP_D2_MODEL` is set.
**User-gated:** submitting the fan-out (+ staging the model/videos onto `/bucket`) is the
compute boundary — paused for the user's go.

### Phase 2 — Line B identity + ceiling eval
1. **2D tracklets** from masks (reuse `UDMT/tracking.py` + `concrete_tracklets` gating;
   break at ambiguity rather than bridge).
2. **Appearance ID:** clean classifier + `window_align` accumulation/bijection → per-view
   ID; then **multi-view appearance fusion** (a fish IDed in whichever view it's clean —
   same side-pooling that already works) → the Line B track.
3. **Ceiling metrics:** per-fish ID coverage, swap rate vs the 3D-bind reference,
   **stratified by scene** (clean / near-contact / occlusion / dense cluster).
4. **Failure-mode overlap** (the decisive metric): do Line A and Line B go uncertain on the
   same frames or complementary ones?

### Phase 3 — Detectron2 dense-scene improvement (label-free)
Mine dense/crossing frames → `reproject_slot_points` (3D positions) → SAM2 prompts →
clean per-fish masks in dense scenes → add to Detectron2 training set → retrain on saion.
Report dense-scene mask-separation rate before/after. (I build the miner; user assesses/
improves the model. Parallel to the user's separate SLEAP dense-scene work feeding Line A.)

### Phase 4 — Fusion
Define a per-scene confidence for each line; build the fused track (geometry positions +
appearance ID, arbitrated by confidence). Concretely this is where Line B **resolves Line
A's hard-core swaps and the 358k `low`-confidence sexing spans** — the
`resolve_hardcore_events` integration, now with an independent appearance signal instead of
only motion continuity. Evaluate fused vs each line alone.

## Ceiling bounds ("how far can we go")
1. **Detectron2 dense separation** — improvable (Phase 3) via the SAM2+3D auto-label loop.
2. **Irreducible all-views-occluded core** — neither line has signal; needs interpolation.
   Phase 2 quantifies its true size (the `multiview_clean_coverage` hard core was ~4%).

Line B alone ≈ "everything except simultaneous all-view occlusion." Fused ceiling is
higher — *iff* Phase 0/2 show the lines complement rather than coincide.

## Open decisions for the new session
- Which experiment(s) to scale Detectron2 on first (K_1_2 has the most validated 3D ref).
- Per-frame-argmax vs fragment-vote ID assignment (use fragment-vote + bijection, like
  idtracker.ai; argmax flickers).
- Fusion arbitration rule (confidence thresholds vs learned) — defer until Phase 2 numbers.
- Whether the female-identity GT gap (no manual female label for K_1_1/K_1_3) needs a
  visual gravid-belly check before trusting fused sex.
