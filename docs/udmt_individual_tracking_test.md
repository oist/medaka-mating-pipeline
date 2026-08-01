# UDMT-style individual tracking on Detectron2 segmentations — test summary

Experiment: `20251225_K_1_2_FCMC` round5, window 176300–181300 (5001 frames, ~30 s),
all 3 views. Conclusions below are **adversarially verified** (8 skeptic agents +
completeness critic + synthesis); corrections from that pass are folded in. Treat
numbers as window/recording/parameter-specific until replicated. Artifacts in
`Z:/StephensU/medaka/runs/udmt_validation/udmt_track/K_1_2/`.

## Method (what "UDMT" means here)

The in-repo `UDMT/` adapter has UDMT's *post-processing* (score / refine / repair /
crossings / resolve) but **no tracker** — the real UDMT tracker is the ST-Net/DiMP
(ResNet-50 + transformer) Linux/py3.8 stack, not runnable here. So a faithful test
required implementing UDMT's **tracking-by-segmentation core**: `UDMT/tracking.py`
links per-frame Detectron2 mask centroids into ≤3 identity tracks with a
velocity-gated, gap-tolerant greedy-Hungarian assignment (forward + backward), now
exposed as `python -m UDMT track`. Pipeline:

```
Detectron2 masks (raw, incl. reflections) -> mask centroids
  -> UDMT.tracking (motion-Hungarian, fwd+bwd)
  -> UDMT.bidirectional repair + crossing detection -> UDMT.metrics score
  -> scored vs 3D final_fish_id reprojected into the view (proxy ground truth)
```

This substitutes **motion** for ST-Net's appearance+motion. That is a **fair lower
bound on the males** (independently, males carry no transferable appearance signal —
masked male-vs-male 0.235, below the 0.303 position baseline — so UDMT's appearance
branch cannot separate them either) but **understates UDMT on the female and on
fragmentation** (DiMP's learned template could hold through short occlusions better
than centroid velocity). Magnitude of that gap is unverified (ST-Net not run).

## Results

Headline IDF1-like: top ~0.60, side1 ~0.80, side2 ~0.76. The per-fish breakdown is
the real story — single-best-track recall **understates** the tracker because it
credits only one fragment per fish:

| view | masks/fr | female (fish 15) | males (0 / 12) | tracks/fish |
|---|---|---|---|---|
| top  | 3.31 | recall 0.99, purity 1.0, 1 track | best-track 0.38 / 0.44; **pooled ~0.95 / 0.96** | 2.67 |
| side1| 3.45 | 0.96, 1 track | 0.72 / 0.71 | 1.33 |
| side2| 3.26 | **0.65, split across 2 tracks** | 0.66 / **0.98** | 1.33 |

- **All three fish are detected ~95–99% of frames — not lost.** Low male recall is
  **fragmentation** (each male split into 2–3 mostly-pure tracks), not chronic
  swapping. The unsolved problem is **re-linking fragments across crossings**.
- **Female cleanly held on top/side1, but swaps on side2**: a brief female↔male
  crossing flips her track ID and the error propagates for thousands of frames. She
  is still detected (0.974 within gate) and is >35 px from the males in 99% of side2
  frames — an ID-*linking* failure at a sparse crossing, not detection or overlap.
- **Failures are crossing-localized**: 50% (top) / 78% (side1) / 96% (side2) of ID
  switches fall within ±15 frames of a detected crossing. The crossing detector
  fires in the right places; resolution at crossings is the residual gap.
- **Residual genuine contamination** (within-male-track impurity 13–26%, concentrated
  at crossings) means males cannot be assigned one stable ID without crossing
  resolution, even after pooling fragments.

## Does UDMT bidirectional repair help? — No (verified)

**It shows no measured ID benefit; it is neutral-to-noisy, NOT a regression.** Raw
`id_switches` roughly double after repair (top 46→102, side1 31→58, side2 28→70),
which *looks* harmful — but that is an **evaluation artifact**:

- `repair_tracks_with_backward` only overwrites the **x/y of motion-anomaly rows** and
  **never changes `track_id`**. The switch metric is a per-frame position-based
  Hungarian re-match (`match_px=80`), so nudging a position can flip the matched fish
  and fabricate a "switch" with zero identity change.
- **Decisive cross-check:** restrict switches to rows repair did *not* touch →
  forward and repaired are **identical** in all 3 views (46=46, 21=21, 24=24). 100% of
  the extra switches come from the ~130 moved rows/view (0.9% of frames); reverting
  those x/y restores the counts exactly. Modal track→fish assignment is byte-identical;
  mean GT residual is flat (top 9.06→9.17 px).

**Verdict: do not credit the repair pass on this data; report forward-pass numbers
only.** (The permissive `max_anomaly_repair_distance_px=250` is the smearing mechanism.)

## Caveats (verified)

1. **3D `final_fish_id` is a proxy, not ground truth — and it can swap the males.** It
   comes from the same 3D-bind pipeline with ~924 hard-core events / ~18 likely-real
   swaps in this recording. So male numbers mean *"UDMT agrees with the 3D male label
   ~70–75% of frames,"* not *"UDMT is ~70–75% correct."* Bounding this needs the
   ~40–60 hand-labeled male frames already on the worklist.
2. **`match_px=80` biases the switch count** (two tracks can fall within gate of the
   wrong fish at a crossing). Sweep 40/60/80 before trusting absolute switch counts.
3. **Reflections confound:** frames with >3 masks are 29/41/25% (top/side1/side2) while
   `num_individuals=3` and there is no `max_area` filter; add one.
4. **N=1 window, N=1 recording, default params, bespoke (un-benchmarked) metrics.**

## Recommendation

1. **Keep 3D reconstruction as the source of identity.** Motion+segmentation tracking
   agrees with it but is scored *by* it — it cannot claim to improve it, and no
   appearance method can separate the two males.
2. **Do not run the bidirectional repair pass** here (no benefit, adds position noise);
   forward-pass output only.
3. **Actionable next lever: female-anchor fusion.** Pin fish 15 with the existing
   female classifier (`37/38`, `female_anchor.py`) to rescue the side2 female split and
   reduce a 3-body to a 2-male problem in every view — but validate the anchor *on
   side2* (it is view/gravid-dependent, ~0.70–0.83).
4. **The residual male problem is crossing re-linking, not a tracker ceiling.** Because
   failures are crossing-localized and males fragment into mostly-pure pieces, the right
   intervention is **resolve-at-crossing** (fuse 3D geometry / female anchor only inside
   crossing windows to stitch fragments) — dovetailing with the existing hard-core-event
   resolver — not a new full-sequence tracker.

**One line:** segmentation+motion tracking *detects* all three medaka ~95–99% of
frames and cleanly holds the female on top/side1, but fragments the appearance-
identical males and swaps the female on side2 at crossings; UDMT's bidirectional
repair adds no measured benefit; and because the males carry no appearance signal and
the scoring GT is itself a swap-prone 3D proxy, this is a **continuity aid to fuse
with geometry + the female anchor at crossings — not a standalone identity solution.**

Code: `UDMT/tracking.py` (+`python -m UDMT track`), `scripts/experiments/udmt_individual_tracking.py`,
tests `tests/test_udmt_tracking.py` (7). Reports/CSVs/figure under
`Z:/StephensU/medaka/runs/udmt_validation/udmt_track/K_1_2/`.
