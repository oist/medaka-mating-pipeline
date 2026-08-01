# Independent quality review

Scope: `20251225_K_1_2_FCMC`, Round8
`round8_260729_snn031_m6`. This review independently checks the executor's
inputs, implementations, training-only selections, critical-event diagnosis,
and final method metrics. It does not treat residual or smoothness alone as
biological identity evidence.

## Review status

The canonical candidate cache, strict-mouth training geometry, A1/A2
long-gap semantics, A5 executable contract, final-hash A1–A4 training
sweeps, validation, held-out evaluation, permanently descriptive
critical-event evaluation, and full-length A1–A3 execution pass independent
checks. The A4 cutoff is frozen from training only, and no post-freeze result
changed settings. Full-length QC does not establish a production replacement
for A0.

The unblinded critical interval was inspected before threshold freeze during
an independent A4 safety audit. It is therefore descriptive stress evidence,
not confirmatory evidence. The held-out cohort was not opened for the
training-only selections reported below.

## Input and critical-event diagnosis

The final cache is strict finite-mouth, six-slot, and built with the legacy
tail/body false-positive filter disabled. Direct SLP decoding agreed with the
cache at the reviewed event: maximum coordinate difference
`1.14e-13 px`, maximum mouth-score difference `2.22e-16`, with no mismatched
event frames. Canonical cache SHA-256 values are:

| View | Cache SHA-256 | Round8 SLP SHA-256 |
|---|---|---|
| top | `6d2f86e8822525e83f18c921bb2b75d0e2821b929d2dcced994dea29b2737d74` | `22727b64c3ef91267e86b4125b2ac3bcd326b84bd61d699718020b5deaf7f36f` |
| side1 | `59d9a73895f0b0c507e5dd1c57a8503ae8ca932cdad7e2dd0a16685b7fb042be` | `357f2930fe2d6ae28154288a4bf1f1004bdd32c164367c058bd1454f440783a2` |
| side2 | `1f2d639606e77cc4c1b3188e8991cdf02c2895556d225996a957f91af7832699` | `10bbb2135da9a8559b2644216bcd2b32851c58644fe309846c874aa24115faa6` |

The reported trigger is independently reproduced. For the relevant top-view
raw instance:

- mouth displacement from frame 34547 to 34548 was
  `0.2360589646 px`;
- visible-node centroid displacement was `56.2087023217 px` because the
  tail became unavailable;
- both side views contained only two poses: one isolated fish and one fused
  lower close-pair pose.

For the two competing A0 geometry rows, the previous-to-current distance
matrix was:

```text
[[2.848587,  5.510631],
 [4.609890, 11.153373]] mm
```

The straight continuation included an `11.153 mm` match and violated the
fixed `8 mm` per-frame gate; the two crossed matches remained feasible and
had lower feasible cost. Geometry IDs 1 and 3 consequently swapped their
top slots, corresponding to final IDs 11 and 8. This establishes the
mechanism, but the female-only annotation set cannot establish the exact
male-to-male biological identity outcome.

Evidence:
`quality_review/critical_event_raw_audit.json`,
`quality_review/critical_event_existing_geometry.json`, and
`input_provenance.json`.

## Protocol integrity

Training, validation, held-out, and critical windows were registered in
`cohort_protocol.json`. Training selections in this review read only:

- reviewed events `72371–73027` and `174316–180515`;
- length-matched temporal controls `32264–32920` and `21840–28039`.

The controls are temporal length matches, not behavior- or risk-matched
controls. Only 15 confirmed-female annotations occur in these four training
windows (2, 1, 6, and 6); this is too sparse to distinguish close
configurations reliably and cannot reveal a male-to-male swap.

Two critical-informed A4 deviations are recorded and contained in
`protocol_deviations.json`:

1. the critical interval was used for a pre-freeze tracklet safety audit;
2. retained-tracklet count-mismatch abstention was developed after that
   unblinding.

Neither critical result is permitted to select or tie-break thresholds. The
critical cohort must remain descriptive, and held-out results are the only
post-freeze generalization evidence for those safeguards.

## Implementation checks

The expanded focused regression suite passed independently (`43 passed` on
the final focused code snapshot). Two defects found during review were fixed
before selection:

- A1/A2 previously retired and re-seeded identities after a gap longer than
  100 frames while keeping `identity_segment=0`. The actual relink regression
  now passes: post-gap rows start segment 1 and carry
  `new_segment_after_max_gap_reseed`; missing rows remain explicit.
- The first executable A5 runner indexed geometry groups as `group1...3`,
  while the shared geometry adapter emits `a,b,c`, causing `KeyError`.
  A synthetic end-to-end A5 run now passes with 9/9 hints honored, a complete
  three-row-per-frame direct grid, authenticated outputs, and the refractive
  `mm^2` calibration contract.

Evidence:
`quality_review/a1_a2_long_gap_segment_audit.json` and
`quality_review/a5_end_to_end_audit.json`.

## Training geometry

Fresh geometry under the frozen mouth runner
`d856585da1515a06067909f0b7aeead1eab2af81973a2cf097e17de33a7192e0`
passed all checks in all four windows:

- exactly three finite rows for every requested frame;
- no missing, outside-interval, or duplicate frame/hypothesis rows;
- strict mouth, no centroid fallback, no FP pose filter;
- six candidates, three selected fish, bounds `[-20, 220]`;
- fixed-interface refractive backend and residual unit `mm^2`;
- identical normalized configuration, calibration, cache, and code
  manifests across windows.

Pooled window residual extremes are not identity proof. Notably, the
control windows include long upper tails (control-1 p99 `268.291 mm^2`,
maximum `354.070 mm^2`) even though every frame has three selected rows.
These values must remain visible in final residual QC.

Evidence: `quality_review/training_geometry_independent_audit.json`.

## Training-only A1–A3 selection

All final-hash summaries and output hashes were independently authenticated.
Strict-mouth direct residuals were read from `ray_residual_mm2`; no
polynomial pixel-squared residual was mixed into these metrics.

| Method/config | Female full accuracy | Female changes | All-three direct | Unavailable rows | Jumps >=7.5 mm | Jumps >8 mm |
|---|---:|---:|---:|---:|---:|---:|
| A1 position | 15/15 | 0 | 0.997375 | 38 | 7 | 0 |
| A2, all five settings | 15/15 | 0 | 0.997375 | 38 | 7 | 0 |
| A3, acceleration 0/.05/.1 | 15/15 | 0 | 0.997375 | 38 | 7 | 0 |

All five A2 outputs were byte-identical within each training window.
`velocity_weight=0.25`, history 5 is therefore selected only by the
predeclared lower-complexity tie-break. A3 acceleration 0 is an ineligible
negative-control deferred-velocity ablation; the eligible `.05` and `.1`
settings also tied, so `.05` is selected.

Across these outputs, pooled direct adjacent jumps had p95
`1.059468 mm`, p99 `1.991226 mm`, and maximum `7.873816 mm`.
Pooled direct refractive residuals had median `0.205562 mm^2`, p95
`1.178757 mm^2`, p99 `3.249012 mm^2`, and maximum `354.070130 mm^2`.

These training ties do not demonstrate that velocity or acceleration
improves biological identity; they only define frozen configurations for
post-freeze evaluation.

Evidence:
`quality_review/training_motion_sweep_selection.json`,
`quality_review/training_motion_sweep_audit.json`, and
`quality_review/training_a3_sweep_audit.json`.

## Training-control motion scale

Final frozen body-length scales were independently reproduced exactly:
top `109.583 px`, side1 `91.119 px`, side2 `86.470 px`. On clean,
three-mouth frames separated by at least 1.5 body lengths, constant-velocity
prediction-error p99 was `0.0254`, `0.0287`, and `0.0394` body lengths for
top, side1, and side2. Clean-frame eligibility was only 13.8–17.3% of the
temporal controls, so these distributions describe separated motion, not
contact behavior.

Evidence: `quality_review/training_clean_motion_audit.json`.

## A4 cutoff and A5 threshold review

The final 18-setting by four-window A4 sweep passed independent method-level
authentication: 72 summaries, 3,528 manifested artifacts, exact block
data/status pairing and continuous frame coverage, current runner and scale
hashes, and exact three-row-per-frame direct grids. Of the 72 runs, 69
produced paths and three made a valid
`insufficient_three_path_solution` scientific abstention. Those abstentions
were retained with zero coverage/full unresolved penalty; they were not
discarded as execution failures.

The frozen A4 setting is separation `0.35` body lengths, ambiguity margin
`0.05` body lengths, and maximum missing gap `0` frames
(`s035_m005_g0`). Across the four training windows it had:

- 3/15 female annotations mapped, all three consistent with one fixed
  identity and no observed change (full accuracy `0.20`, selective accuracy
  `1.00`);
- zero reciprocal false-merge proxies and zero non-reciprocal false-break
  proxies;
- 3,638 direct selected rows, but no frame with all three paths direct and
  an unavailable-row fraction of `0.911575`;
- no jump at or above `7.5 mm`; direct-jump p95 `1.05193 mm`, p99
  `1.68754 mm`, maximum `5.29068 mm`;
- direct refractive-residual median `0.21687 mm^2`, p95 `1.51300 mm^2`,
  p99 `3.17765 mm^2`;
- retained candidate coverage `0.95962`, but only `0.27419` of eligible
  cross-view groups selected.

The closest zero-gap alternative with margin `0.10` had identical female,
direct-row, jump, and residual results but two false-break proxies, which
selected the `0.05` margin under the registered purity-first ordering. The
per-view-only rerun remains diagnostic and was not used to freeze the cutoff.
The selected A4 setting is therefore a high-abstention constraint source,
not a complete standalone reconstruction and not evidence of improvement
over A0–A3.

Evidence:
`quality_review/training_a4_method_sweep_audit.json`,
`quality_review/training_a4_method_sweep_summary.csv`, and
`training_a4_cutoff_sweep/SWEEP_COMPLETED.json`.

The frozen A5 hint rules were then independently reproduced from selected A4
and frozen A2 inputs: row and group mouth confidence at least `0.5`, group
support at least 3, group median residual at most `1.0 mm^2`, group residual
MAD at most `0.25 mm^2`, group assignment cost at most 0, three-view support,
at least 3 mapping votes, vote purity at least `0.9`, and vote margin at least
2. All 3,263 hints were exact shared-geometry candidates. All 41 segment
mappings passed the vote rule; there were no vote, overlap-identity, or
per-frame bijectivity rejections. The hints covered `7.931%` of training
rows.

Only 1,351/3,263 hints (`41.404%`) were honored by the final A5 geometry.
This is not silent override failure: 1,092 non-honored rows resolved after
lookahead to another source and 820 remained explicitly unresolved because
the tracklet constraint conflicted with the hard motion/ambiguity rules.
That behavior matches the registered hybrid policy to defer rather than
blindly force a source.

However, A5 materially regressed selected A3 on the same training windows:

| Metric | Selected A3 | A5 |
|---|---:|---:|
| Female fixed-identity correct | 15/15 | 6/15 |
| Observed female identity changes | 0 | 8 |
| All-three direct fraction | 0.997375 | 0.937582 |
| Unavailable rows | 38 | 2,500 |
| Jumps >=7.5 mm | 7 | 7 |
| Jumps >8 mm | 0 | 0 |

A5 residual p95 was `1.21070 mm^2` and p99 `3.54649 mm^2`; it did not
purchase better jump control. The hint thresholds are frozen only for
reproducible exploratory post-freeze comparison. A5 is rejected as a
production and full-length finalist.

The no-retained-count-mismatch safeguard remains a contained,
critical-informed design rule rather than validation evidence.

Evidence:
`quality_review/training_a5_hint_audit.json`,
`quality_review/training_a5_female_assignments.csv`, and
`training_a5/TRAINING_A5_COMPLETED.json`.

## Post-freeze and full-length QC

### Validation: authenticated report-only PASS

The validation stage passed independent execution and provenance review. This
is a PASS to continue the registered protocol, not a claim that any method
improved identity. The authoritative marker is
`validation_postfreeze/STAGE_FINALIZED.json`
(`297249d8cc9cdff2e0a5c850884f795e238b6911b42605e440d70397bc6fb706`).
It binds freeze `09579754...`, runner `54ec5698...`, the preserved completion
marker, and a fully re-hashed 182-file inventory. The finalization audit
independently reproduced 180 unchanged marker-inventoried artifacts and one
expected non-scientific change: detached-launcher stdout grew from zero to
194 bytes when the runner printed its final marker line. No scientific file
changed. Geometry and A4 resume replays passed for both windows with identical
advertised output hashes before and after replay.

All frozen A1--A5 settings and implementation hashes matched. There was no
selection, tuning, or tie-breaking. The two validation windows contained only
three female annotations in total, so the following accuracy evidence is
necessarily sparse:

| Method | Female fixed-ID correct | Female changes | All-three direct | Unavailable rows | Jumps >=7.5 mm | Jumps >8 mm |
|---|---:|---:|---:|---:|---:|---:|
| A0 centroid reference | 2/3 | 1 | 1.000000 | 0.000000 | 1 | 0 |
| A1 strict mouth | 2/3 | 1 | 0.996641 | 0.001120 | 1 | 0 |
| A2 selected motion | 2/3 | 1 | 0.996641 | 0.001120 | 1 | 0 |
| A3 deferred | 2/3 | 1 | 0.996641 | 0.001120 | 1 | 0 |
| A4 tracklet | 0/3 mapped | 0 | 0.000000 | 0.946823 | 0 | 0 |
| A5 hybrid | 2/3 | 1 | 0.996641 | 0.001120 | 1 | 0 |

A1, A2, and A3 had identical normalized scientific hashes in each validation
window. They therefore supply no validation evidence that motion or deferral
improved over strict-mouth position-only linking. Their pooled direct-jump
p95 was `1.23036 mm`, maximum `7.96636 mm`, with no jump above `8 mm`.
Their pooled strict-mouth refractive-residual p95 was `1.04046 mm^2`, p99
`7.11172 mm^2`, and maximum `89.23953 mm^2`.

A4 selected no frame with three direct paths and mapped none of the three
female annotations. Its zero large-jump count is dominated by abstention, not
evidence of identity quality. This independently confirms that A4 is not a
standalone reconstruction. A5 was scientifically identical to A1--A3 in the
event window; in the control it changed identity-segment provenance but did
not improve the registered female result, coverage, residual, or large-jump
outcomes. A5 remains exploratory and rejected as a production/full-length
finalist.

The pooled process-runner measurements for the two windows were:
A1 `8.50 s`/`216.7 MB`, A2 `9.14 s`/`219.7 MB`, A3
`9.91 s`/`212.0 MB`, A5 linker `10.83 s`/`213.4 MB`, and A4 including
two replay processes `57.85 s`/`308.0 MB` maximum RSS. Geometry and A5 hint
generation are recorded separately. These process sums are not stage elapsed
time when helper/replay work is included.

Validation remains report-only and cannot change frozen settings or finalist
status. It authorizes proceeding to the registered held-out stage. Only the
held-out cohort may supply post-freeze generalization evidence. Female-only
ground truth still cannot observe a male-to-male identity exchange, and
smoother or lower-residual trajectories alone are not biological identity
proof.

Evidence:
`quality_review/postfreeze_independent_metrics.json`
(`d9a4d449697c96c0d1a579a841b4fa128f805f519294acae4b4342815cf36519`),
`quality_review/postfreeze_independent_female_assignments.csv`
(`46434e15e7e038a0c290b4d526a491c06f7cdd95c6c935c87293473625e478a9`),
and independent scorer
`quality_review/audit_and_score_postfreeze.py`
(`5d7e4e8f87f40228ddd185f07476951a2009e468258719aed92b7d059345ca72`).

### Held-out: sole confirmatory generalization PASS

The held-out stage passed independent provenance and scientific-quality
review. Its authoritative finalized marker is
`heldout_postfreeze/STAGE_FINALIZED.json`
(`3b58e073c9cf6c97348e1f0c8cdbb9ca5c15d689ee222b450df1c91dc3303152`).
The review independently re-hashed all 516 finalized files, all 512
completion-inventory artifacts, ten geometry/A4 resume checks, and 50 process
records. It also authenticated all 404 files in the sealed failed-attempt
quarantine, including exact original-to-final inventory equality. All 26
overlapping scientific comparisons between the failed attempt and the fresh
whole-stage replacement matched, including geometry block parquet/status
pairs.

The operational repair was restricted to giving a typed 29-column, zero-row
schema to two valid A4 abstentions. The adapter, protocol, and smoke hashes
were independently authenticated. Nine adapter checks passed. In both
abstention windows A5 received zero hints and zero forced resets and its
scientific core was exactly A3. No canonical method code, threshold, frozen
setting, or cohort changed.

The held-out cohort contains nine female annotations (six event, three
control); the three risk windows contain none:

| Method | Female fixed-ID correct | Female changes | All-three direct | Unavailable rows | Jumps >=7.5 mm | Jumps >8 mm |
|---|---:|---:|---:|---:|---:|---:|
| A0 centroid reference | 9/9 | 0 | 1.000000 | 0.000000 | 11 | 0 |
| A1 strict mouth | 9/9 | 0 | 0.999583 | 0.000139 | 1 | 0 |
| A2 selected motion | 9/9 | 0 | 0.999583 | 0.000139 | 1 | 0 |
| A3 deferred | 9/9 | 0 | 0.999583 | 0.000139 | 1 | 0 |
| A4 tracklet | 0/9 mapped | 0 | 0.015754 | 0.889758 | 0 | 0 |
| A5 hybrid | 3/9 full; 3/8 mapped | 5 | 0.939489 | 0.060233 | 0 | 0 |

A1, A2, and A3 had identical normalized scientific outputs in all five
held-out windows. Their pooled direct-jump p95 was `1.07745 mm`, p99
`1.68992 mm`, and maximum `7.91407 mm`. Their pooled direct refractive
residual p95 was `0.96143 mm^2` and p99 `1.85714 mm^2`. They supplied no
generalization evidence that velocity or deferral improves on the simpler A1
linker.

A0 and A1--A3 all achieved 9/9 female consistency with no observed change.
A1--A3 reduced the registered large-jump count from 11 to 1, but different
anchor geometry, smoother trajectories, and lower residuals alone do not
prove biological identity improvement. Female-only labels also cannot detect
a male-to-male exchange. The held-out result therefore supports A1 as the
parsimonious tied mouth-only candidate; it does not establish biological
superiority over A0.

A4 scientifically abstained in two of five windows, mapped no female label,
and left 88.98% of rows unavailable. It remains a high-abstention diagnostic,
not a standalone production reconstruction. A5 materially failed held-out
identity and coverage gates: only 3/9 full female accuracy, five observed
female identity changes, and 6.02% unavailable rows. Its zero large-jump count
does not compensate for those regressions. A5 remains rejected as a
production/full-length finalist.

Pooled five-window process measurements were A1 `27.64 s`/`237.7 MB`, A2
`29.59 s`/`253.8 MB`, A3 `32.10 s`/`246.0 MB`, and A5 linker
`36.28 s`/`248.7 MB`. A4 including five replay processes used `186.84 s`
summed wall time and `558.5 MB` maximum RSS. Adapter, geometry, and A5 hint
generation are recorded separately.

The held-out stage is an explicit PASS to open permanently descriptive
critical-event evaluation. No selection or tuning was performed. A sealed
quality decision binds all evidence at
`quality_review/HELDOUT_QUALITY_REVIEW_COMPLETED.json`
(`022fbca3f3d0c13d7388a83c42b3b2bad60ded00de2fe89fad428943e7f213cc`).

### Critical event: permanently descriptive PASS

The critical stage passed independent authentication and scoring over the
inclusive smoke interval `34000–35100`. Its authoritative finalized marker
is `critical_descriptive_postfreeze/STAGE_FINALIZED.json`
(`fe432101a7580f4e540d3c4537ae8b1c774a2b5231d8cc0900a790e250d3149c`).
The review re-hashed its 100-file finalized inventory, verified all 96
completion-marker artifacts unchanged, authenticated ten process records,
and reproduced both geometry and A4 resume-equivalence checks. The launcher
exited zero with empty stderr.

This cohort was permanently unblinded before threshold freeze and remains
descriptive/nonconfirmatory. It did not change a setting, select or tune a
method, change finalists, or supply confirmatory generalization evidence.

| Method | Female fixed-ID correct | Female changes | All-three direct | Unavailable rows | Jumps >=7.5 mm | Jumps >8 mm |
|---|---:|---:|---:|---:|---:|---:|
| A0 centroid reference | 1/2 | 1 | 1.000000 | 0.000000 | 11 | 0 |
| A1 strict mouth | 2/2 | 0 | 0.990009 | 0.003330 | 0 | 0 |
| A2 selected motion | 2/2 | 0 | 0.990009 | 0.003330 | 0 | 0 |
| A3 deferred | 2/2 | 0 | 0.990009 | 0.003330 | 0 | 0 |
| A4 tracklet | 0/2 mapped | 0 | 0.000000 | 0.972147 | 0 | 0 |
| A5 hybrid | 2/2 | 0 | 0.990009 | 0.003330 | 0 | 0 |

A1, A2, A3, and A5 had identical normalized scientific rows over the smoke
interval. A1–A3 pooled direct jumps had maximum `6.92966 mm`, with no jump
at or above `7.5 mm`; direct refractive residual p95 was `2.89651 mm^2` and
p99 `16.50461 mm^2`. A0 had 11 jumps at or above `7.5 mm`, none above
`8 mm`, maximum `7.99054 mm`, and residual p95 `9.97512 mm^2`. These
kinematic differences describe the stress interval but do not by themselves
prove biological identity.

The inclusive detail interval `34540–34565` showed the same A1–A3/A5 tie:
all-three direct fraction `0.807692`, unavailable-row fraction `0.064103`,
maximum direct jump `0.962014 mm`, and residual p95 `4.48961 mm^2`. A0
remained fully covered there, with maximum direct jump `7.26263 mm` and
residual p95 `12.12696 mm^2`. A4 explicitly left the entire detail interval
unassigned.

The exact assignment audit over frames `34547–34555` contains 27 rows per
method. Every A1/A2/A3/A5 row was direct and finite, their scientific cores
were identical, and their source slots stayed fixed at `ID_1=0`, `ID_2=1`,
and `ID_3=2` with zero switches. A1 labeled all rows
`resolved_position_only`, A2 `resolved_velocity`, and A3/A5
`resolved_immediate`. A3/A5 assignment margins ranged from `15.02449` to
`17.71443`, with no ambiguous or constraint-conflict frame. A5 had zero hint
rows in this exact interval, so it does not provide independent hybrid
evidence. A4 explicitly marked all 27 rows `unassigned_tracklet_path`.

A0 also had 27 finite exact-event rows, but the separately reproduced
distance gate proves that its centroid geometry linker exchanged the two
close geometry trajectories at `34547→34548`. In the raw top pose, mouth
motion was only `0.23606 px` while the visible-node centroid moved
`56.20870 px` after the tail disappeared. This establishes a local
centroid-anchor failure mechanism. It does not establish biological
superiority for A1: no female annotation occurs in these nine frames, the
close pair is male under the available ledger, and male-to-male correctness
is unobservable.

A4's complete exact-event abstention confirms it is not a standalone
reconstruction. A5 remains rejected by held-out identity and coverage
results. A1–A3 remain tied; A1 is still the parsimonious candidate.

The sealed decision
`quality_review/CRITICAL_QUALITY_REVIEW_COMPLETED.json`
(`55a479e1b97a1054b7f195530d1505202d4d87935a1dd2f36b06f51e665fe83e`)
binds the freeze, held-out gate, critical finalization/completion markers,
independent scorer, combined metrics, female assignments, exact event CSV,
event audit, raw trigger audit, and A0 geometry audit. All 11 bindings
matched on independent re-hash. This is a PASS for frozen A1–A3 full-length
execution.

### Full length: execution PASS, production replacement not established

The frozen full-length stage covered frames `0–348599` and ran only A1, A2,
and A3; A0 remained an exact-hash reference and A4/A5 were excluded. The
authoritative marker is
`full_length_postfreeze/STAGE_FINALIZED.json`
(`a9a8ae25cca6ef0c94dc7996cf39680017b9bb58e20ab60473bced96284e07b0`).
Independent review re-hashed all 1,438 completion-inventory files and all
1,442 finalized pre-sidecar files. All eight process records had exit code
zero and empty stderr. The 698 geometry blocks were contiguous; the fresh
geometry and resume contracts matched exactly, as did fresh/repeat
scientific outputs for each finalist. Every finalist had the required
1,045,800-row, three-rows-per-frame grid.

The following “female correct” field is best-fixed, segment-aware identity
consistency: the number of annotations assigned to the most frequent single
track identity, with every unmapped annotation retained in the denominator.
It is not male identity accuracy.

| Method | Female best-fixed | Mapped | Changes | All-three direct | Unavailable | Identity segments |
|---|---:|---:|---:|---:|---:|---:|
| A0 centroid reference | 221/300 | 299/300 | 15 | 0.989544 | 0.006442 | 3 |
| A1 strict mouth | 180/300 | 297/300 | 9 | 0.983537 | 0.009188 | 10 |
| A2 selected motion | 192/300 | 297/300 | 9 | 0.983537 | 0.009188 | 10 |
| A3 deferred | 94/300 | 297/300 | 19 | 0.980995 | 0.015318 | 117 |

A0 has higher full-history female consistency but more observed changes.
A1/A2 reduce changes from 15 to 9 but do not keep the female on one fixed
segment as often. A3 materially fragments the output: 117 segment-aware
identities, 19 observed female changes, and only 94/300 best-fixed
consistency. This rejects A3 as a production finalist.

The bounded held-out result must not be substituted for full-history
continuity. Tracks initialized inside the five held-out windows gave 9/9 for
A1–A3. On tracks linked from frame zero, those same nine annotation frames
give 6/9 with one observed change for each A1, A2, and A3; A0 remains 9/9
with no change. The bounded result is valid for its registered isolated
generalization test, but it is not a statement that the production-length
track preserves the same identity across intervening frames.

A1 and A2 have nearly identical aggregate geometry metrics, but are not the
same identity solution. Their unordered candidate-coordinate sets match in
348,598/348,600 frames, yet fixed-ID coordinates differ in 228,593 rows over
114,297 frames, including 113,779 fully populated frames. They differ on
40/300 female annotation assignments: 39 are in the post-hoc
`other_full_length` partition, one is validation, and none is held-out,
training, or critical-descriptive. Therefore A2's descriptive 192/300 versus
A1's 180/300 global consistency does not promote A2. The large path
divergence needs male-resolving ground truth.

A3 differs from A1 in its candidate set on 5,091 frames and in fixed-ID
assignment on 320,237 frames. Its assignment audit has 348,600 rows:
341,348 resolved, 8,140 flagged ambiguous, zero constraint conflicts, and
114 new segments after unresolved intervals. Finite deferred assignment
margins have median `16.6114`, p95 `20.3913`, and maximum `80.9614`.

Full-length jump and residual QC is:

| Method | Jump p95 / p99 (mm) | >=2 mm | >=4 mm | >=6 mm | >=7.5 mm | >8 mm | Residual p95 / p99 (mm²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1.3572 / 4.8596 | 31,022 (2.986%) | 15,518 (1.493%) | 4,477 (0.431%) | 597 (0.0575%) | 0 | 3.3247 / 22.4125 |
| A1 | 1.1225 / 2.0357 | 10,757 (1.039%) | 2,524 (0.244%) | 760 (0.0734%) | 166 (0.0160%) | 0 | 1.0248 / 3.0866 |
| A2 | 1.1225 / 2.0357 | 10,758 (1.039%) | 2,523 (0.244%) | 760 (0.0734%) | 166 (0.0160%) | 0 | 1.0248 / 3.0866 |
| A3 | 1.1197 / 2.0130 | 10,440 (1.015%) | 2,432 (0.236%) | 727 (0.0707%) | 160 (0.0156%) | 0 | 1.0217 / 3.0470 |

These smoother mouth-only trajectories do not establish biological identity.
Residual maxima also remain large: `6646.18 mm²` for A0,
`5820.80 mm²` for A1/A2, and `1716.13 mm²` for A3. Lower tails do not
compensate for A3's fragmentation.

On full-history tracks at frames `34547–34555`, A1 and A3 use source slots
`ID_1=0, ID_2=2, ID_3=1`; A2 uses
`ID_1=2, ID_2=0, ID_3=1`. All have 27/27 finite direct rows and zero
within-event source-slot switches, with identical local jump and residual
distributions. A3 resolves all nine assignment frames with margins
`15.0245–17.7144`. Thus A2 is globally label-permuted relative to A1/A3
because of earlier full-history divergence; the production critical-event
outputs are not identical. With no female annotation in the exact event,
none proves the two male labels biologically correct.

Authenticated process measurements were:

| Process family | Summed fresh/repeat wall | Peak RSS |
|---|---:|---:|
| shared geometry fresh + resume | 1,142.09 s | 2.547 GB |
| A1 fresh + repeat | 1,232.41 s | 2.643 GB |
| A2 fresh + repeat | 1,382.38 s | 3.419 GB |
| A3 fresh + repeat | 1,547.06 s | 3.396 GB |

The stage wall time was `5314.74 s` and peak process-tree RSS
`3,418,509,312 bytes`. A1 is both simpler and cheaper than A2/A3. Full-length
false-bridge/false-break and retained-tracklet metrics are not applicable to
A1–A3; A4/A5 were correctly excluded under the frozen finalist policy, and
their bounded tracklet metrics remain reported above.

The quality scorer initially survived a shell-wrapper timeout, after which a
later duplicate was stopped before output creation. Clean reruns reproduced
all CSV hashes. Pre-seal review then added missing helper/ledger/calibration
bindings, fixed divergence-frame reporting, and corrected resource-key
rollups; no stage output or scientific metric definition changed. The final
incident record is
`quality_review/full_length_scorer_execution_incident.json`
(`6ecf08b348f02bbc21ad87879160123809f8fd7e57259a893815f397fc71f66a`).

Evidence:
`quality_review/full_length_independent_metrics.json`
(`0230a938d70d05465677fec1fdad46e6abf1eb13483e093a17980de800684870`),
`quality_review/full_length_independent_method_summary.csv`
(`4a8dbcc53744634e133b4625108dc40c653d0c1fe8d770b4e2b621185e0226e6`),
`quality_review/full_length_independent_female_assignments.csv`
(`2420ff3db22ffb26a788008c6eee5586e10d3e53a10282ee3f77fff3f363ed67`),
and the annotation partition JSON
(`c13861f672de73db9f7098c500c813c158069e86ba23567ea966d34e5f897d61`).
The final sealed decision is
`quality_review/FULL_LENGTH_QUALITY_REVIEW_COMPLETED.json`
(`7e9896716a693ce81287320af1aacd6ab686f909ad4f5cc8de43af54dc03cc92`);
all 24 bindings matched on recursive re-hash.

## Independent recommendation

Full-length execution and QC pass, but a production replacement for A0 is not
established.

- Retain A0 as the unchanged reference/current comparator.
- Retain A1 as the primary experimental mouth-only track. If one mouth-only
  output must be carried forward for biological validation, choose A1 under
  the preregistered parsimony rule; do not present it as a proven A0
  replacement.
- Investigate A1/A2's 114,297-frame identity-path divergence with
  male-resolving ground truth. Do not promote A2 from its non-held-out
  0.04 descriptive female-consistency difference.
- Reject A3 as a production finalist because of lower coverage, higher
  unresolved fraction, 117 segments, and worse full-history female
  consistency/changes.
- Retain A4 only as a bounded high-purity/high-abstention diagnostic
  constraint source. Reject A5 as a production/full-length finalist.

Lower residuals, fewer jump spikes, and local source-slot continuity remain
geometry/temporal evidence, not proof of biological identity.
