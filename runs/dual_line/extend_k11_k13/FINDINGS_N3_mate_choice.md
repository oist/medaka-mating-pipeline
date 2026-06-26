# N=3 per-male mate choice (K_1_1 + K_1_2 + K_1_3) — dual-line corrected tracks

Extends the K_1_2 dual-line result to all three 20251225 K-group experiments. **All three show a clear,
strong female mate preference, cross-validated by BORIS.**

## Result

| exp | female high-conf (interp) | assoc index (M1 share) | courtship M1 vs M2 | sharpening 30→8mm | spawn | BORIS m1tf vs m2tf |
|-----|--------------------------|------------------------|--------------------|-------------------|-------|--------------------|
| K_1_2 | 99.0% | 0.61 | 271 vs 162 s | 1.6×→1.9× | M1, 7.4 mm | 41.8 vs 103.6 s (2.5×) |
| K_1_3 | 99.8% | 0.71 | 614 vs 242 s | 2.4×→3.7× | M1, 4.9 mm | 6.7 vs 99.2 s (14.7×) |
| K_1_1 | 99.8% | 0.80 | 545 vs 127 s | 4.1×→10.8× | M1, 5.4 mm | 9.5 vs 118.2 s (12.4×) |

- **Consistent finding:** in every trial the female associates 1.6–4× more with one male, the preference
  **sharpens toward contact** (up to ~11×), and she **spawns with that preferred male** (5–7 mm contact).
- **Courtship, not aggression, tracks the choice:** courtship:aggression time ≈ 20–25× in all three.
- **BORIS agrees in direction every time:** the appearance-preferred male = the BORIS-courted/wrapped male.
  (Appearance "M1" = BORIS "m2" = the chosen mate in all three; the appearance M1/M2 labels are arbitrary
  per experiment.)
- Preference **strength varies**: K_1_2 (0.61) < K_1_3 (0.71) < K_1_1 (0.80).

## How the corrected tracks were built (K_1_1, K_1_3)

⛔ **Cross-video female-MODEL transfer FAILS.** The naive pipeline (K_1_2 `model_female_clean` applied to
K_1_1/K_1_3) confidently mislabelled a MALE as the female in BOTH experiments (it scored the real female
at P≈0.07 and a male at P≈0.9). The C0 "female P 0.82/0.92" gate was the model's confidence *in the wrong
fish*. **Tell:** the C0 "male-vs-male" LOSO was too HIGH (K_1_3 0.99, K_1_1 0.95 vs genuine male-male
~0.91) — because with the female mislabelled, the "male model" was actually separating female-from-male.

**Fix (per experiment):**
1. User identified the gravid female by eye (montage) = aligned-0 in both K_1_1 and K_1_3 (the naive call
   was aligned-2 both times — a systematic mis-transfer).
2. Retrained per-experiment `model_female_clean_v2` (female = aligned-0) + `model_male_clean_v2` (the two
   REAL males = aligned-1 vs 2). After the fix the separability has the correct structure: female-vs-male
   easy (K_1_3 0.96), male-vs-male hard (K_1_3 0.84).
3. **Validation (now mandatory): the female must land in the BORIS spawn wrap.** Corrected female-to-male
   distance at the spawn: K_1_3 0.6 mm, K_1_1 2.6 mm (was 65–245 mm when mislabelled).
4. Script 45 → interp (`interp_track.py`) → motion-continuity debounce (`debounce_track.py`) → analyze.

## Artifacts
- Tracks: `runs/dual_line/extend_k11_k13/{K_1_1,K_1_3}/round5_260611_3d_individual_id_interp_debounced.csv`.
- Analysis: `.../{K_1_1,K_1_3}/analysis/mate_choice_report.md` (+ png/csv). K_1_2:
  `runs/dual_line/lineB_fulltimeline/analysis_debounced/`.
- Models: `C:/udmt_validation/segid/{K_1_1,K_1_3}/{model_female_clean_v2,model_male_clean_v2}/`.

## Same-individuals-across-trials? → INCONCLUSIVE (appearance embedder is unreliable cross-video)
`cross_trial_identity.py` embedded each experiment's 3 fish (aligned-ids) with the frozen K_1_2
self-supervised embedder → 9 fish-centroids. They split into 3 clean cross-trial groups (match cos ~1.00,
non-match ~0.5) and **K_1_1-female ↔ K_1_3-female link consistently (both female)**. BUT the structure is
**sex-inconsistent**: that group also pairs the K_1_1/K_1_3 female with a K_1_2 *male* (aligned-2), and the
K_1_2 (gravid-GT'd) female pairs with the K_1_1/K_1_3 *males*. A fish cannot change sex across trials, so
the cross-video matching is **not reliable individual ID** (the embedder is K_1_2-trained + flagged
"uninformative, coherence saturates at 1.0" in phase-0). The per-trial female calls are trustworthy (each
is validated by SPAWNING — the strongest GT). **Verdict: cannot confirm same-individuals from appearance;
the experimental design records are the authority.** A proper cross-video re-ID model would be needed to
test the within-subject hypothesis rigorously.

## Open / next
- Same-individuals: ask the user / check recording metadata (appearance embedder can't settle it).
- Top-view fusion (as in K_1_2) could lift the K_1_1/K_1_3 male contact core further (not needed for the
  result, which rests on interp-bridged 99%+ tracks).
