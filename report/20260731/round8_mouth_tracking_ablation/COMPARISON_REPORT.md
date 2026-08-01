# Round8 strict-mouth identity-tracking comparison

Experiment: `20251225_K_1_2_FCMC`
Round: `round8_260729_snn031_m6`
Evaluation date: 2026-07-31 (Asia/Tokyo)
## Technical summary

Round8 reproduced the reported A0 failure mechanism at frames
`34547→34548`: the relevant top-view mouth moved only `0.2361 px`, while the
visible-node centroid moved `56.2087 px` after the tail became unavailable.
Under the fixed `8 mm` step gate, the straight A0 continuation became
infeasible and the crossed continuation remained feasible. Strict-mouth A1,
A2, and A3 avoid that local centroid-induced geometric crossover.

That diagnostic success does **not** establish a production replacement for
A0. The close pair is male, while the available biological ground truth
identifies only the female. Male-to-male correctness is therefore
unobservable. On full-history tracks, A0 has the strongest female
best-fixed consistency (`221/300`), while A1 and A2 have fewer observed
female changes (`9` versus `15`) and substantially fewer large jumps. A3
fragments into `117` segment-aware identities and is rejected.

The production decision is:

- retain A0 as the current reference/comparator;
- retain A1 as the simplest experimental mouth-only candidate for targeted
  biological validation;
- investigate A1/A2's `114,297`-frame fixed-ID divergence before considering
  A2;
- reject A3, A4, and A5 as production finalists.

## Controlled design and frozen inputs

No SLEAP inference was rerun. All variants used the same authenticated
Round8 `.slp` files, strict finite-mouth six-candidate caches, fixed-interface
refractive calibration, and reconstruction gates.

| Fixed item | Value |
|---|---|
| Cameras | `top_cam`, `side1_cam`, `side2_cam` |
| Candidate policy | six slots; strict finite mouth; no centroid, tail, or segmentation fallback |
| False-positive pose filter | disabled for the final mouth-only cache |
| Calibration | `fixed_refractive_192x192x186`; score unit `mm²` mean squared point-to-water-ray distance |
| 3D selection | 6 candidates, 3 individuals, XYZ bounds `[-20, 220]`, no positive shift |
| Identity gates | `8 mm` nominal jump, `80 mm` hard cap, `100`-frame maximum gap |
| Freeze | `frozen_thresholds.json`, SHA-256 `09579754f781e80e20b1365ded00a4bbb8e96e273260861e6c3a791d263cdc1d` |

The top, side1, and side2 strict-mouth cache SHA-256 values are,
respectively, `6d2f86e8…73d74`, `59d9a738…042be`, and
`1f2d6396…32699`. Direct SLP decoding matched the caches at the critical
event to numerical precision.

## Methods compared

| Method | Definition | Frozen status |
|---|---|---|
| A0 | Existing centroid-based Round8 geometry and identified output | Unchanged reference |
| A1 | Strict-mouth shared geometry with position-only relinking | Full-length finalist |
| A2 | A1 plus velocity prediction (`weight=0.25`, history `5`, uncertainty growth `0.25`) | Full-length finalist |
| A3 | Deferred ambiguity linker with acceleration weight `0.05`, beam width `16`, and stability-based resolution | Full-length finalist |
| A4 | Reciprocal per-view tracklets and cross-view path selection | Bounded diagnostic only |
| A5 | A3 plus conservative exact-candidate hints from A4 | Bounded diagnostic only |

The executor's expanded focused suite passed `47/47`; the quality reviewer
independently passed `43/43` checks on the final focused code snapshot.
Review-found long-gap segment semantics and A5 schema defects were corrected
before post-freeze evaluation.

## Training-only selection and cutoff sweep

Training comprised two reviewed events and two length-matched temporal
controls. Only `15` confirmed-female annotations occurred in those windows,
so training evidence was used to freeze configurations, not to claim
biological superiority.

A1 and every eligible A2/A3 setting tied at `15/15` female best-fixed
consistency with zero changes. The registered lower-complexity tie-break
selected A2 velocity weight `0.25` and A3 acceleration weight `0.05`.

The A4 sweep evaluated `18` settings across four windows (`72` method runs
and `3,528` manifested artifacts). The purity-first selection was:

- separation cutoff `0.35` body lengths;
- ambiguity margin `0.05` body lengths;
- maximum missing gap `0` frames.

It mapped only `3/15` training annotations and left `91.16%` of rows
unavailable, with no all-three-direct frame. This is a high-abstention
constraint source, not a complete reconstruction. A5 then regressed selected
A3 from `15/15` to `6/15`, introduced eight observed female changes, and
raised unavailable rows from `38` to `2,500`; it was excluded from the
full-length finalists.

## Post-freeze evidence

### Validation did not distinguish A1–A3

Across the two validation windows, A1, A2, and A3 were scientifically
identical: `2/3` female best-fixed consistency, one change, all-three-direct
fraction `0.996641`, and no jump above `8 mm`. A4 mapped none of the three
annotations; A5 did not improve the result.

### Held-out evidence is bounded and must not be mistaken for global continuity

In five isolated held-out windows, A1–A3 each achieved `9/9` female
best-fixed consistency, zero changes, and one jump at or above `7.5 mm`
versus A0's eleven. These tracks were initialized inside each window.

When the same nine annotation frames were read from tracks linked from frame
zero, A1–A3 each achieved only `6/9` with one observed change; A0 remained
`9/9` with zero changes. The isolated-window result is valid for the
registered bounded stress test, but it does not show that production-length
identity survives the intervening history.

A4 abstained in two of five held-out windows and left `88.98%` of rows
unavailable. A5 achieved only `3/9` full female consistency, five changes,
and `6.02%` unavailable rows. Both remain rejected for production.

## Critical-event audit is descriptive, not confirmatory

At the exact `34547–34555` interval, bounded A1–A3/A5 outputs were smooth,
finite, and locally stable (`27/27` direct rows, no within-event source-slot
switch, maximum step `0.6187 mm`). A4 explicitly abstained on all `27` rows.
This verifies that the strict-mouth geometry avoids the local A0
centroid-anchor failure.

The critical interval was opened before freeze during an A4 safety audit.
That deviation is recorded and contained, so the interval is permanently
descriptive. It cannot select settings or provide confirmatory evidence.

Full-history context also changes the label state hidden by bounded restart:
A1/A3 enter the event with source slots `ID_1=0, ID_2=2, ID_3=1`, while A2
enters with `ID_1=2, ID_2=0, ID_3=1`. All three remain locally stable, but
their prior global label histories differ. With no female observation in the
exact event and no male-resolving truth, none establishes the biologically
correct male labels.

## Full-length comparison

Frames `0–348599` were processed for A1–A3, with exact fresh/repeat
scientific-hash agreement. A0 remained the authenticated unchanged
reference. A4/A5 were excluded under the frozen finalist policy.

“Female best-fixed” means the number of the `300` female annotations assigned
to the most frequent single segment-aware track identity. Unmapped
annotations remain in the denominator. It is an identity-consistency
measure, not general three-fish identity accuracy.

| Method | Female best-fixed | Mapped | Changes | All-three direct | Unavailable | Identity segments |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 221/300 (73.7%) | 299/300 | 15 | 0.989544 | 0.006442 | 3 |
| A1 | 180/300 (60.0%) | 297/300 | 9 | 0.983537 | 0.009188 | 10 |
| A2 | 192/300 (64.0%) | 297/300 | 9 | 0.983537 | 0.009188 | 10 |
| A3 | 94/300 (31.3%) | 297/300 | 19 | 0.980995 | 0.015318 | 117 |

| Method | Jump p95 / p99 (mm) | Jumps ≥7.5 mm | Jumps >8 mm | Residual p95 / p99 (mm²) |
|---|---:|---:|---:|---:|
| A0 | 1.3572 / 4.8596 | 597 | 0 | 3.3247 / 22.4125 |
| A1 | 1.1225 / 2.0357 | 166 | 0 | 1.0248 / 3.0866 |
| A2 | 1.1225 / 2.0357 | 166 | 0 | 1.0248 / 3.0866 |
| A3 | 1.1197 / 2.0130 | 160 | 0 | 1.0217 / 3.0470 |

A1/A2 are smoother than A0, but smoothness and calibration residuals are not
biological identity labels. A2's apparent `12/300` advantage over A1 is not
confirmatory: their annotations differ at `40` frames, of which `39` are in
the post-hoc other-full-length partition, one is validation, and none is
held-out, training, or critical. Their candidate-coordinate sets differ in
only two frames, yet fixed-ID coordinates differ in `228,593` rows over
`114,297` frames. This is a label-path divergence requiring new truth, not a
reason to promote A2.

### Resource cost

| Process family | Fresh + repeat wall time | Peak RSS |
|---|---:|---:|
| Shared geometry fresh + resume | 1,142.09 s | 2.547 GB |
| A1 | 1,232.41 s | 2.643 GB |
| A2 | 1,382.38 s | 3.419 GB |
| A3 | 1,547.06 s | 3.396 GB |

The full stage took `5,314.74 s`; peak process-tree RSS was
`3,418,509,312` bytes. A1 is the simplest and least expensive mouth-only
finalist.

## Limitations and uncertainty

- The ledger identifies only the female. It cannot score a male-to-male
  exchange, including the exact critical close pair.
- Only `15` female annotations were available in training, `3` in
  validation, and `9` in held-out windows.
- Length-matched controls were not behavior- or risk-matched.
- The critical interval was unblinded before freeze and is descriptive only.
- Window-initialized identity can hide prior global label permutations.
- Lower jump or residual distributions show smoother geometry, not biological
  correctness.
- A0 and mouth-only methods use different anchors, so direct residual
  superiority is not an identity proof.

## Recommendation and next experiment

1. Keep A0 unchanged in production and retain its exact-hash output as the
   comparison baseline.
2. Carry A1 forward as the primary experimental strict-mouth output because
   it is the parsimonious tied mouth-only method and costs less than A2/A3.
3. Do not promote A2 until male-resolving labels explain the
   `114,297`-frame A1/A2 divergence.
4. Reject A3 for production because of `117` segments, lower coverage, and
   worse female consistency. Keep A4 only as a high-purity abstaining
   diagnostic; reject A5.
5. Acquire identity truth that distinguishes all three fish across crossings:
   persistent visual marks or another validated individual-specific signal
   spanning pre-event, event, and recovery frames. Prioritize frames where
   A1 and A2 disagree, plus the `34547→34548` close-pair event.

## Audit and reproducibility

The full-length quality seal is
`quality_review/FULL_LENGTH_QUALITY_REVIEW_COMPLETED.json`, SHA-256
`7e9896716a693ce81287320af1aacd6ab686f909ad4f5cc8de43af54dc03cc92`.
The independent auditor verified all `24/24` recursive bindings and
independently reproduced all `1,200` projected annotation assignments, the
full QC summaries, partitions, critical global slot mappings, and resource
rollups.

Supporting records:

- `QUALITY_REVIEW.md` — independent metric definitions, results, and
  recommendation.
- `INDEPENDENT_AUDIT.md` — independent provenance, command, replay, and
  conclusion audit.
- `quality_review/full_length_independent_metrics.json` — canonical
  machine-readable full-length metrics.
- `quality_review/full_length_independent_method_summary.csv` — compact
  method comparison.
- `exact_commands.md` — recorded execution commands.
- `frozen_thresholds.json`, `input_provenance.json`,
  `protocol_deviations.json` — freeze, inputs, and deviations.
