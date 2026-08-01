# K_1_2 identity track — bake-off decision (2026-06-30)

## Outcome: ship the NEW-backbone appearance track

**Final deliverable:**
`Z:/StephensU/medaka/runs/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d_individual_id_interp_debounced.csv`
(348,600 frames; `id_final ∈ {F,M1,M2}`; F 100.0% / M1 97.6% / M2 98.2% high-conf;
irreducible ambiguous 0.04%; spawn PASS; built on the 06-30 `recon_nofill` backbone with the
committed 2D-FP / entry-phantom fixes.)

## How it was produced
1. Reproject 3 backbone slots → per-view slot points (`script 40 --stop-after slot-points`).
2. Per-frame F/M1/M2 appearance resolution reusing the validated K_1_2 female/male/top Detectron2
   classifiers + the existing full-length saion masks — **no new GPU detector inference**
   (`script 45`, stride 10, sides + top fuse).
3. Gap-fill (`interp_track.py`) → teleport-debounce (`debounce_track.py`) — the exact recipe that
   produced the committed track, for an apples-to-apples comparison.

## Why NEW beats the committed OLD track (`runs/dual_line/lineB_fulltimeline/step2/…_interp_debounced.csv`)
4-agent bake-off (artifacts in this folder):
- **B1 spawn + teleports → NEW.** Both pass spawn; NEW 46 impossible (>60 mm/frame) teleports vs OLD 71
  (female 6 vs 15).
- **B4 BORIS contact → NEW.** Female 4.7 mm vs 7.6 mm from nearest male at spawn; rejected-wrap contact
  3.9 mm vs OLD's borderline 11.1 mm; both reproduce the accepted-28 s / rejected-3.6 s wrap durations.
  NEW better matches the documented baseline→courtship→touch→wrap→spawn contact progression.
- **B3 mate-choice → TIE (no regression).** NEW reproduces the published conclusion essentially exactly:
  association index 0.61 (= OLD 0.61), closer-to-preferred-male 69.9% (= 70.0%), courtship 15.2× aggression,
  female spawns in contact with the male she associates with.
- **B2 crossing stability → OLD**, but on a largely orthogonal metric (M1↔M2 slot churn 158 vs 302).
  M1/M2 are interchangeable for the *female* mate-choice question. B2's "female agreement 29.7% = chance"
  headline was a **slot-index confound** (the two backbones permute slot labels, only 22.9% index
  correspondence) — disproved below.

## Cross-validation (the decisive check)
Comparing the female by **physical 3D position** (not slot index) across the two independent
backbones + appearance pipelines, over 344,864 common frames:
**95.5% within 10 mm, 98.0% within 20 mm, median 0.27 mm.** Two independent pipelines converge on the
same physical female 98% of the time → the female anchor is genuinely cross-validated.

## Honest residual (appearance-limited; the same hard core for either track)
- **~2.0% (6,837) frames** where the two tracks place the female >20 mm apart = the genuine female-ID
  hard core (only ~25% are <15 mm dense-contact; the rest are moderate-separation residual). Eye-review
  queue = `B2_disagreements.csv` (top contacts: fr 134349, 131489, 134642) + `event_worklist.csv`.
- **Spawn-pileup male identity (M1 vs M2)** is not resolvable at the exact spawn frame (all 3 fish
  mutually <15 mm 67% of ±1 s; males 3–11 mm apart; male id_conf = interp/ambiguous). M1≠M2 is globally
  stable (prior embedder check) and the mate-choice conclusion does not hinge on the spawn-frame male letter.
- **4,416 frames (1.3%)** lack a reconstructed 3rd fish on the new backbone → kept as honest GAPS, not
  fabricated fills (OLD geometry-fills these; that can inject wrong labels).
- **M1/M2 letters are arbitrary per run** (alpha/beta boundary set by a per-run reference window); the
  NEW track's nearest-male-at-spawn is labeled M2 where OLD labeled the same physical fish M1 — a label
  flip, not a disagreement.

## Remaining human step (optional)
Eye-review the ~top crossings to confirm the female color holds, per `validation_runbook.md`
(`render_crossing_montage.py --events-csv …`). All quantitative criteria (invariant, spawn ≤12 mm,
BORIS direction + contact, mate-choice reproduction, cross-pipeline female 98%) already PASS.
