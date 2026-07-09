# Round7 3D-tracking + identity bake-off — DECISION

**Date:** 2026-07-08  **Experiments:** 20251225_K_1_1 / K_1_2 / K_1_3_FCMC (1 female + 2 males each)
**SLEAP source:** round7_260704 (top-down, 2-node mouth+tail, `--max_instances 6`), merged per view.
**Question:** best per-fish 3D track WITH stable biological identity (F/M1/M2), proven by
(a) candidate methods vs each other, and (b) round7 vs the validated round5 tracking.

## Recommendation

**Ship M3 (appearance / dual-line) as the production round7 track for all three experiments.**
It is the only method that produces biological F/M1/M2, it reproduces every validated round5
result, and it improves 3D coverage. M1 (geometry+bind) and M2 (SAM2 seg-bind) are structurally
incapable of naming the female — they yield arbitrary stable IDs — so they are baselines, not
candidates for the stated goal.

### Production tracks (per experiment)
| Exp | Production track (M3) |
|---|---|
| K_1_1 | `Z:\StephensU\medaka\reconstruction_3d\round7_260704\20251225_K_1_1_FCMC\round7_260704_3d_individual_id_interp_debounced.csv` |
| K_1_2 | `Z:\StephensU\medaka\reconstruction_3d\round7_260704\20251225_K_1_2_FCMC\round7_260704_3d_individual_id_interp_debounced.csv` |
| K_1_3 | `Z:\StephensU\medaka\reconstruction_3d\round7_260704\20251225_K_1_3_FCMC\round7_260704_3d_individual_id_interp_debounced.csv` |

Column `id_final ∈ {F,M1,M2}`; long format, 3 rows/frame keyed by geometry slot `fish_id ∈ {ID_1,ID_2,ID_3}`.

## Method comparison (per experiment)

Rubric: (1) female physical-position agreement vs the independent round5 gold (decisive test —
compare by position, letters are arbitrary), (2) mate-choice reproduction, (3) 3D geometry quality,
(4) ID stability / biological validity.

### M3 — appearance / dual-line  ✅ WINNER
Pipeline: round7 M1 backbone → reproject slots (`40 slot-points`) → per-frame F/M1/M2 appearance
classifiers (`45`, reusing existing Z: Detectron2 masks + per-exp segid models) → `interp` → `debounce`.

| Exp | Female vs gold (median / %<20mm) | Mate-choice assoc (gold) | Spawn nearest-male | Coverage (gold) | Ambiguous frames | 1·F+1·M1+1·M2 |
|---|---|---|---|---|---|---|
| K_1_1 | 0.18 mm / 98.8% | 0.80 (0.80) | 6.4 mm ✓ | 0.874 (0.874) | 284 | 100% |
| K_1_2 | 0.27 mm / 98.8% | 0.62 (0.61) | 7.4 mm ✓ | 0.864 (0.805) | 50 | 100% |
| K_1_3 | 0.18 mm / 99.7% | 0.71 (0.71) | 5.9 mm ✓ | 0.843 (0.851) | 8 | 100% |

M3 reproduces all three validated round5 mate-choice indices exactly, places the female at the same
physical position as the independent round5 gold (median <0.3 mm), reaches female-male contact at
every spawn, and yields perfect biological validity (exactly one F/M1/M2 every frame).

### M1 — geometry 3D-first + top-cam NN bind  (baseline; cannot name the female)
Canonical AGENTS.md recipe: `track_simple_nn` top anchor + `reconstruct_from_slp --num-candidates 6
--use-visible-node-centroid … --bind-top-id`.

| Exp | Coverage | Speed-spikes >8mm | Top-bind | Implied-female vs gold (%<20mm / median) | Biological F/M1/M2 |
|---|---|---|---|---|---|
| K_1_1 | 0.874 | 0% | 96.8% | 47% / 24 mm | none (arbitrary IDs) |
| K_1_2 | 0.864 | 0% | 96.2% | 43% / 35 mm | none |
| K_1_3 | 0.843 | 0% | 96.0% | 54% / 10 mm | none |

M1 is a clean, stable geometry backbone (0 speed-spikes, stable arbitrary IDs, ~96% top-bind) and is
the substrate M2/M3 build on — but its `final_fish_id` are arbitrary, and its best single-ID match to
the gold female agrees only 43–54% (the biological female physically swaps geometry slots at
crossings). **Geometry alone cannot hold biological identity** — the project's core thesis, confirmed
on round7.

### M2 — segmentation-assisted (local SAM2 seeded from 3D)  (baseline; cannot name the female)
Pipeline: `24_seed` (3D→top-cam prompts) → per-window SAM2 video predictor (identity = seed
`final_fish_id` via obj_id) → aggregate identity centroids → `bind_topcam_id` onto the round7 backbone.

- **Feasibility finding:** the "reuse existing Detectron2 masks" path is *infeasible* — those masks are
  identity-free and the repo has no script to attach a 3-fish reference identity to them
  (`assign_tracklet_identities` needs a labeled reference nothing produces). The only viable M2 is fresh
  **local SAM2** inference over the 25 clean seed windows/exp (~1.3 min/window). Not saion.
- **Structural ceiling:** M2's seed identities ARE the geometry `final_fish_id`, so M2 — like M1 —
  produces arbitrary stable IDs, **not** biological F/M1/M2. At best it is a swap-stabilized variant of
  M1's bind anchor over the ~7% of frames covered by clean windows.
- **Measured result (K_1_2):** the SAM2 anchor is far too sparse to matter — it covers **2,442 frames
  (0.7%)** and the bind falls back to geometry for 99.3% of frames, vs M1's top-NN anchor covering
  **96%**. Because M2's identity is the same geometry `final_fish_id`, the resulting track's implied
  female is **identical to M1**: median 35 mm vs the gold female, 43% within 20 mm, spawn 15.7 mm
  (not contact). **M2 ≈ M1, and gives the bind strictly less information than M1's dense top-NN anchor.**
  A denser M2 (stride-1, full 02→20 stitch pipeline) could at best reach ~7% coverage with the same
  geometry identity — still far below M1 and unable to name the female. M2 is strictly dominated.

| Exp | SAM2 anchor coverage | Implied-female vs gold (%<20mm / median) | Biological F/M1/M2 | vs M1 |
|---|---|---|---|---|
| K_1_2 | 0.7% (2442 fr) | 43% / 35 mm | none | **identical to M1** |
| K_1_1 | 0.7% (2478 fr) | 47% / 24 mm | none | **identical to M1** |
| K_1_3 | 0.6% (2084 fr) | 54% / 10 mm | none | **identical to M1** |

Confirmed across all three: M2's implied-female numbers are byte-for-byte the same as M1's — the
sub-1% SAM2 anchor cannot move the geometry bind, so M2 collapses onto M1. Dominated on every experiment.

## Round5 vs round7

| Exp | Coverage r5 → r7 | Ambiguous frames r5 → r7 | Mate-choice assoc r5 → r7 | Female (r7 M3 vs r5 gold) |
|---|---|---|---|---|
| K_1_1 | 0.874 → 0.874 (=) | 86 → 284 (slightly worse) | 0.80 → 0.80 | 0.18 mm median |
| K_1_2 | 0.805 → **0.864 (+6 pts)** | 462 → **50** (much better) | 0.61 → 0.62 | 0.27 mm median |
| K_1_3 | 0.851 → 0.843 (≈) | 38 → 8 (better) | 0.71 → 0.71 | 0.18 mm median |

**Verdict:** round7 is a decisive win for the hardest case (K_1_2: +6 pts coverage, ambiguous frames
462→50 — the recon_nofill round5 backbone that needed the most patching is now clean), and a wash for
K_1_1/K_1_3 (comparable; K_1_1 marginally more ambiguous, K_1_3 fewer). round7 M3 places the female at
the same physical positions as the round5 gold across all three (median <0.3 mm), so **round7 fully
reproduces round5's validated conclusions while improving the weakest experiment**. Recommend round7 as
the new production backbone.

**Naive-female-transfer test — round7 does NOT remove the round5 per-experiment retrain.** Re-running
the M3 appearance pass on the round7 K_1_1/K_1_3 spawn windows with the *naive* K_1_2 female model
(instead of each experiment's v2) places the called "female" **108 mm (K_1_1) / 120 mm (K_1_3) from the
spawning pair** at the spawn frame — the exact cross-video mislabel that broke round5. So the mislabel is
an **appearance-model limitation, not a detection/backbone one**: round7's better SLEAP improves geometry
and coverage but does not make the K_1_2 female classifier transfer to the other lines. **The per-experiment
female-anchor retrain (v2 models) + mandatory BORIS-spawn validation (female ≤~12 mm from a male at spawn)
remains required.** The production M3 tracks above already use the correct per-exp v2 models and pass this
gate (spawn contact 6.4 / 7.4 / 5.9 mm).

## Adversarial verification (independent skeptic per experiment)

All three M3 tracks: **QUALIFIED YES** — female identity survives adversarial attack.
- Female agrees with the independent round5 pipeline 98.75% / 98.8% / 99.7% within 20 mm; every
  spawn/wrap reaches female-male contact; **0 biological-validity violations**; K_1_3 explicitly clears
  the cross-video female-mislabel that broke round5.
- **Residual (eye-review queue):** ~1–2% of time in transient (<2.1 s), self-correcting female↔male
  crossing swaps. Per-exp top-10 suspect intervals recorded (see verifier reports).

**Two load-bearing caveats (apply to all experiments):**
1. **`id_conf` is not a swap-safety flag.** Swaps trade two same-position fish, leaving no speed
   signature, so swapped frames are still labeled `high`. Do not use `id_conf` to find swaps.
2. **The M1-vs-M2 *letter* is provisional.** Same-size males are not independently ground-truthable;
   M3 and the gold agree ~98% but are correlated appearance pipelines. The robust conclusions are
   "the female prefers one male and spawns with him"; *which letter* that male carries is not proven.
   (BORIS codes courtship strongly toward one male in each trial, matching the asymmetry.)

## Ranking

Per experiment and overall: **M3 ≫ M1 ≈ M2**.
- M3 is the only method that satisfies the stated goal (stable biological F/M1/M2); it reproduces
  round5 and improves coverage.
- M1 and M2 are geometry-identity baselines (arbitrary IDs). Neither can name the female; M2 adds
  local SAM2 cost for, at best, a marginally more swap-robust bind anchor over sparse clean windows.

## Reproduce
Drivers (session scratchpad): `m1_geometry.sh`, `m3_appearance.sh` (`45`→interp→debounce, reuses
existing masks + segid models), `m2_driver.py` (per-window SAM2 → aggregate → bind). Evaluation:
`qc_candidate.py`, `female_agreement.py`, `eval_candidate.sh` (wraps `analyze_mate_choice.py` +
`compare_boris_trajectory.py`). Merges via `scripts/tracking/slp_merge.merge_view_chunks` on the
`round7_260704` subdir.
