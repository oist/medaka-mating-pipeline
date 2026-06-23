# Pilot findings & recommendation

Run: `20251225_K_1_2_FCMC` round5, 8 pilot windows, on existing on-disk outputs.
Reproduce: `run_benchmark --config config/experiments/tracklet_id_benchmark.example.toml`.

## What the pilot measured

| method | mean coverage | mean unknown_frac | speed_spike/k | mean reconnection |
|--------|---------------|-------------------|---------------|-------------------|
| `geometry_3d`     | 0.89 | **0.00** | 0.0 | 1.00 |
| `motion_resolver` | 0.89 | 0.27 | 0.0 | 0.73 |
| `tracklet_bind`   | 0.89 | 0.74 | 0.0 | 0.03 |
| `hybrid`          | 0.89 | 0.74 | 0.0 | 0.03 |

Inter-method (`agreement.csv`): **label_agreement = 1.00** everywhere;
**flag_agreement = 0.0–0.95** at suspect events.

## The decisive result

1. **`speed_spike` is 0.0 for every method in every window.** The established
   speed-spike QC metric is *blind* to the failure mode that matters (smooth
   identity swaps at crossings). It cannot be used to compare methods here.

2. **`label_agreement = 1.0` across all geometry-derived methods.** They share the
   3D backbone, so they make the *same* identity decision and differ only in
   `unknown_frac` — i.e. *where they admit uncertainty*. This is not four competing
   trackers; it is one tracker with three confidence-gating policies:
   - `geometry_3d` commits everywhere (hides swaps),
   - `tracklet_bind` flags the whole hard core (over-conservative: 74% unknown,
     reconnection 0.03),
   - `motion_resolver` is the **useful middle**: it auto-clears the clean passes and
     flags only the genuine residual (27% unknown, reconnection 0.73).

3. **No accuracy can be computed.** GT scoring is wired and activates, but the 3
   existing `swap` labels fall on none of the events that matter
   (`n_scored = 0`). Until labels exist on the contested events, "best method at a
   crossing" is undecidable.

## Recommendation (best method so far)

**`motion_resolver` is the best operating point today** — not because it is more
accurate (it shares geometry's labels) but because it has the best
*honesty/coverage trade-off*: it returns confident identity on ~73% of hard-event
fish while isolating the residual ~27% for review, instead of either silently
committing (`geometry_3d`) or refusing almost everything (`tracklet_bind`).

For production tracks, keep `geometry_3d` as the positional backbone and overlay
`motion_resolver` verdicts as a **confidence/flag layer** — never auto-correct.

### Failure modes
- All geometry/motion methods are **blind at true 3D dwells** (`dwell_ambiguous`:
  reconnection NaN, resolver flags but cannot decide). Motion cues separate the two
  fish by only ~0.13 rel. diff (prior result) — unresolvable from geometry alone.
- `tracklet_bind`/`hybrid` over-flag (up to 100% unknown on a suspect window),
  losing usable coverage.
- The only independent-label method (`appearance_idtracker`) is **absent** here and,
  per prior spikes, only moderately separable (silhouette 0.73 < 0.91 needed).

### Next step — targeted annotation (the binding constraint)

The harness emits `annotation_worklist.csv`. Label, in priority order:

1. **18 `suspect_swap` events** (P1) — *fish-ID-after-crossing* labels (which slot is
   which immediately after the event). ~18 examples. Discriminates
   `motion_resolver` vs `geometry_3d`: if the resolver's swap calls are right, this
   enables conservative reversible auto-correction; if wrong, it demotes the
   resolver to a pure ranker. **Highest value per label.**
2. **20 long `ambiguous` dwells** (P2) — *crossing-vs-mix* + ID-through-dwell labels.
   Tests whether *any* geometry method can recover the hard core, or whether
   appearance is mandatory there.
3. **`early_fp` window** (P3) — ~20 *false-positive / true-fish-count* frame labels
   across the single→three transition.

Total first batch ≈ **40–60 event/frame labels**, all on K_1_2. This is the minimum
that turns the GT-free panel above into a scored, decisive comparison and gates the
`appearance_idtracker` modelling step (only worth running if P1/P2 show geometry +
motion genuinely cannot resolve the contested events).
