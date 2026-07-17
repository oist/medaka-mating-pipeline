# Item 4 primary method report: conditional tail-vector continuity

## Status

This is the primary method specification written before Item 4 implementation
or benchmark results were available. It contains no measured runtime, tail
availability, or quality claim. All result fields must be filled from saved
artifacts under `benchmark_protocol.md` and independently reproduced by the
Item 4 verifier.

## Claim under test and isolation

The claim is narrow: when two or more feasible fragment continuations are both
mouth-close and nearly tied under the Item 3 mouth-only objective, temporal
tail-vector continuity can be a secondary tie-breaker. Tail evidence must not:

- create or remove a fragment or directed candidate edge;
- relax or impose a gap, displacement, residual, or raw-resource hard gate;
- alter directly reconstructed mouth XYZ;
- decide a case that is clear under mouth-only evidence; or
- turn a missing tail into evidence for or against a candidate.

Item 4 therefore has Item 3 graph construction as minimal prerequisite
plumbing. The controlled comparison uses the exact same serialized Item 3
candidate graph, sequential Hungarian boundary matcher, birth/death costs,
anchors, and mouth-only costs:

- `conditional_tail=off`: every tail term is zero;
- `conditional_tail=on`: a capped tail term is added only to the qualifying
  ambiguity sets defined below.

Candidate graph hash equality is mandatory before comparing link decisions.
The tail augmentation receives its own hash and cannot overwrite Item 3 raw
features. Global solving and deferred identity are disabled so Items 5 and 6
receive no credit.

Female GT is evaluation-only. It identifies only the female and cannot detect
male-to-male swaps.

## Tail evidence from the same raw candidates

At each directly observed mouth sample, tail evidence must come from the exact
same `(frame, view, candidate_slot)` instances already used by that 3D mouth
group. The tail stage may not search other candidate slots or perform a new
tail-driven cross-view assignment.

### Preferred 3D representation

Triangulate the tail from the fixed mouth-selected slots when at least two views
have finite tails. A per-frame 3D body vector is valid only when:

1. the tail triangulation is finite and calibration-native;
2. tail XYZ is inside the frozen `0..200 mm` bounds on all axes;
3. refractive mean-squared point-to-ray residual is finite and at most
   `10 mm^2`;
4. `10 mm <= ||tail_xyz - mouth_xyz|| <= 60 mm`; and
5. the mouth row itself is direct rather than interpolated.

The unit vector is
`b_3d = (tail_xyz-mouth_xyz)/||tail_xyz-mouth_xyz||`. Failed conditions make
tail evidence missing with a reason code; they never reject the mouth edge.

### Multiview 2D fallback

When complete 3D evidence is unavailable for a whole ambiguity set, a
multiview 2D representation may be used without triangulating a different
tail. For each fragment and view, compute the median direct mouth-to-tail pixel
length from at least five fixed-slot samples. A per-frame 2D vector is valid
when both nodes are finite, length exceeds 1 pixel, and length is between
`0.5x` and `2.0x` that fragment/view median. Normalize it to a unit vector.

The fallback is usable only if every disputed edge in the ambiguity set has
valid endpoint summaries in the same intersection of at least two camera
views. One-view tail evidence is insufficient. 3D and 2D angles are never
mixed inside one ambiguity set.

## Temporal endpoint tail summaries

For a source fragment use at most the last 15 valid direct vectors inside its
last 30 frames. For a target use at most the first 15 inside its first 30
frames. At least three valid vectors are required at each endpoint.

The deterministic summary is the componentwise median of the unit vectors,
renormalized. If its pre-normalization norm is below `0.5`, direction is too
inconsistent and the summary is missing. For a valid 3D source/target pair:

```text
theta = degrees(arccos(clip(dot(b_source, b_target), -1, 1)))
```

For multiview 2D, compute that angle in each fixed common view and use the
median. `theta` is in degrees in `[0,180]`. There is no angular hard veto.

Every summary records representation, contributing frames/views, vector,
angular dispersion, physical or pixel-length distribution, residual
distribution, and exclusion reasons.

## Exact mouth-ambiguity eligibility

Boundary assignments are enumerated from the frozen Item 3 candidate graph
using the same one-to-one resource constraints and dummy birth/death choices.
Let `A0` be the lowest mouth-only incremental-cost assignment at a boundary.
Let `A1...Ak` be complete feasible assignments that disagree with `A0` on at
least one predecessor-to-successor mapping.

Tail is eligible for an ambiguity set only when all conditions hold:

1. At least one runner-up exists with
   `C_mouth(Ai) - C_mouth(A0) <= 0.5` dimensionless units.
2. For the fragments whose mapping differs, the minimum pairwise 3D mouth
   separation on any same directly observed frame in the inclusive
   `[boundary_frame-15, boundary_frame+15]` interval is at most `12 mm`.
3. Every disputed edge in every within-margin assignment already passed all
   Item 3 hard gates and resource constraints.
4. Complete comparable tail summaries exist for every disputed edge, using
   3D for all of them or the same intersection of at least two 2D views for all.

The `0.5` ambiguity margin matches the separately frozen Item 6 entry margin;
the `12 mm` close-contact threshold matches the existing repository's
`hardcore_mm=12.0` convention. Neither may be tuned on the four reviewed
windows.

If any disputed alternative lacks comparable tail evidence, the entire
ambiguity set is `tail_incomplete` and every tail term in that set is zero.
This complete-case rule prevents an alternative with missing tails from being
favored simply because it receives no angular penalty.

Ambiguity eligibility is always calculated from mouth-only costs. Enabling tail
cannot make a previously clear boundary become ambiguous.

## Capped tail objective

For each disputed edge `e` in a complete eligible set:

```text
C_tail(e) = (1/12) * clip(theta_e / 90 degrees, 0, 1)
```

All common, undisputed edges receive zero because their contribution would
cancel. With at most three biological continuations, the total tail addition
to any complete boundary assignment is at most `3/12 = 0.25` dimensionless
units. The tail-enabled assignment cost is:

```text
C_item4(A) = C_mouth(A) + sum(C_tail(e) for disputed e in A)
```

Thus:

- a mouth-only margin greater than `0.5` is never tail-eligible;
- even inside an eligible set, tail can alter a decision only when the effective
  disputed mouth-cost margin is less than `0.25`;
- tail continuity is a bounded penalty, never a hard veto or an unbounded
  reward; and
- missing or incomplete tail evidence produces exactly the Item 4-off cost.

Stable assignment ordering is `(total_cost, selected_edge_id_tuple)`, with
comparison tolerance `1e-12`. Tail-off and tail-on artifacts must preserve both
the original mouth-only cost and the separate tail term.

## Missing-tail policy and reason codes

Missing tail is neutral at both edge and ambiguity-set level. Required reason
codes are:

```text
tail_not_finite
fewer_than_two_tail_views
tail_triangulation_invalid
tail_outside_xyz_bounds
tail_residual_over_10_mm2
tail_length_out_of_range
insufficient_endpoint_samples
inconsistent_endpoint_direction
insufficient_2d_length_baseline
fewer_than_two_common_2d_views
representation_mismatch
incomplete_competing_set
not_mouth_ambiguous
mouths_not_within_12_mm
```

Counts and duration by reason must be reported. No missing state may be encoded
as `theta=180`, zero confidence, or another worst-case value.

## Pure proposed interface

```python
augment_graph_with_conditional_tail(
    graph,
    candidate_cache,
    *,
    enabled: bool,
    mouth_assignment_margin: float = 0.5,
    close_mouth_distance_mm: float = 12.0,
    close_support_radius_frames: int = 15,
    endpoint_support_frames: int = 30,
    endpoint_max_samples: int = 15,
    endpoint_min_samples: int = 3,
    tail_3d_residual_max_mm2: float = 10.0,
    tail_3d_length_min_mm: float = 10.0,
    tail_3d_length_max_mm: float = 60.0,
    tail_2d_length_ratio_min: float = 0.5,
    tail_2d_length_ratio_max: float = 2.0,
    tail_2d_baseline_min_samples: int = 5,
    tail_angle_scale_deg: float = 90.0,
    tail_edge_weight: float = 1.0 / 12.0,
    seed: int = 20260717,
) -> TailAugmentedGraph
```

The result includes the unchanged candidate graph hash, tail augmentation hash,
endpoint summaries, complete ambiguity table, per-edge angles and terms,
missing reasons, selected decisions, timing/resources, and deterministic
checkpoint state.

## Identical-candidate controlled tests

Every Item 4 test must first assert identical values for:

- fragment-table hash;
- directed candidate-pair list and feasibility flags;
- raw resource keys;
- every Item 3 raw feature and mouth-only term;
- birth/death choices, solver, anchors, and initial state.

Only tail summaries, tail eligibility metadata, tail term, and resulting link
choice may differ. A candidate edge added, removed, or newly vetoed by tail is
an automatic failure rather than a tail improvement.

## Deterministic synthetic cases

### Ambiguous crossing resolved by tails

Create two fish with mouths 5 mm apart at a simultaneous break. Under Item 3,
set the known-true assignment to mouth cost `2.10` and the swapped assignment
to `2.05`; both are within the 0.5 margin. Give the true two links `theta=0`
and the swapped links `theta=180` using complete 3D support. Item 4-off selects
the wrong `2.05` assignment. Item 4-on adds `2/12` to the swap and must select
the known-true `2.10` assignment. The fixture defines truth independently.

### Clear mouth evidence cannot be overridden

Use the same tails but make the swapped runner-up mouth cost `0.60` worse than
the best. The set must be `not_mouth_ambiguous`, all tail terms must be zero,
and Item 4-on must exactly match Item 4-off even when tails favor the runner-up.

### Cap prevents a moderately better mouth assignment from flipping

Create a three-fish eligible ambiguity whose best mouth assignment leads the
runner-up by `0.30`. Give the best the maximum possible tail penalty and the
runner-up zero. The total tail difference is capped at `0.25`; the mouth winner
must remain selected.

### Incomplete tails are neutral

Repeat the ambiguous crossing with one disputed target missing every tail. The
whole set must be `tail_incomplete`, Item 4-on and off costs/decisions must be
identical, and the missing alternative must not receive a relative advantage.

### Additional required cases

- Mouths farther than 12 mm disable tail despite an exact mouth-cost tie.
- Exactly 12 mm and exactly 0.5 margin are eligible; values one representable
  float above either threshold are not.
- Two-view 3D tail triangulation from the mouth-selected slots is accepted;
  a single-view tail uses no 3D evidence.
- Complete fixed-slot multiview 2D evidence in two common views provides a
  deterministic fallback when 3D is unavailable for every alternative.
- One common 2D view disables the whole ambiguity set.
- A 3D vector shorter than 10 mm, longer than 60 mm, outside bounds, or above
  10 mm^2 residual becomes missing without changing edge feasibility.
- Opposing noisy vectors whose median-vector norm is below 0.5 are missing.
- A tempting tail from a different SLEAP candidate slot is ignored.
- Simultaneous all-view breaks with all six identity permutations retain the
  same candidate assignments off and on; only complete eligible assignments
  receive capped terms.
- Reflection/extra-candidate fixtures never reuse a raw resource.
- Missing mouths remain excluded under the strict-mouth contract even when a
  tail is visible.
- Input observation, fragment, edge, and worker completion permutations produce
  byte-identical tail tables and decisions.
- A tail ambiguity crossing a 10,000-frame core boundary is identical in
  one-pass, overlapping-window, and checkpoint-resume execution.
- Tail-on preserves the exact per-frame multiset of directly observed mouth XYZ.

## Failure and determinism criteria

Any of the following invalidates a run:

- candidate graph hashes differ between tail-off and tail-on;
- tail alters edge feasibility, an Item 3 hard gate, grouping, or raw resources;
- tail is computed from a different candidate slot than the mouth;
- tail is applied when the mouth margin exceeds 0.5 or direct mouth separation
  exceeds 12 mm;
- total tail addition for one three-path boundary exceeds 0.25 by more than
  `1e-12`;
- incomplete competing-tail evidence changes a cost or link decision;
- missing evidence is treated as contradiction, maximum angle, or zero
  confidence;
- interpolation, GUI GT, reviewed-event labels, or a later manual identity is
  used to build tail evidence;
- a clear-mouth synthetic decision flips, a known ambiguous synthetic truth is
  missed, or the complete-case policy is violated;
- any directly observed mouth XYZ or raw resource assignment changes; or
- fixed-seed reruns, window resume, or input permutations change canonical
  output hashes.

## Benchmark plan and placeholders

No benchmark has yet been run. Replace placeholders only after implementation
and graph paths exist.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> augment-tail `
  --graph-dir <FROZEN_ITEM3_GRAPH_DIR> --conditional-tail off `
  --seed 20260717 --output-dir <OUT_DIR>\tail_off

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> augment-tail `
  --graph-dir <FROZEN_ITEM3_GRAPH_DIR> --candidate-cache <CACHE_DIR> `
  --conditional-tail on --mouth-assignment-margin 0.5 `
  --close-mouth-distance-mm 12 --endpoint-support-frames 30 `
  --endpoint-max-samples 15 --tail-3d-residual-max-mm2 10 `
  --tail-3d-length-min-mm 10 --tail-3d-length-max-mm 60 `
  --tail-angle-scale-deg 90 --tail-edge-weight 0.08333333333333333 `
  --seed 20260717 --output-dir <OUT_DIR>\tail_on
```

Both modes then use the same sequential boundary solver. The comparison runs on
synthetic fixtures, numerical-equivalence windows, all four reviewed windows,
and the exact frozen controls in `benchmark_protocol.md`: 32,264--35,364;
151,050--155,050; 21,840--30,840; and 57,363--67,363. A full recording is
allowed only for a justified integrated candidate.

## Measurements to fill after execution

Not yet measured:

- unchanged candidate graph hash and tail augmentation hashes;
- total boundary sets, mouth-ambiguous sets, close-contact sets, complete 3D
  sets, complete 2D fallback sets, and disabled counts by missing reason;
- tail angle, vector length, residual, dispersion, and availability
  distributions by view and cohort;
- decisions changed by tail, direction of each change, mouth margin, tail term,
  and independently reviewable correctness where available;
- synthetic tie-break accuracy and clear-mouth/cap/missing-tail safeguards;
- selected-link conflicts, fragment counts, unresolved intervals, and resource
  violations;
- eligible, mapped, and strict female-GT counts, paired common cohorts, best
  fixed female accuracy, observed changes, and reviewed-event outcomes;
- all-three direct/interpolated coverage, jump p95/p99/max, and gate violations;
- cold/warm runtime, CPU utilization, peak RSS, output size, checkpoint cost,
  and any justified GPU comparison;
- deterministic rerun hashes, numerical stability, failure cases, and verifier
  reproduction.

## Advantages

- Adds posture evidence only where mouth continuity is demonstrably ambiguous.
- Uses the exact same candidate slots, avoiding a tail-driven reassignment loop.
- Complete-case handling makes missing tails neutral rather than spuriously
  attractive.
- The 0.25 assignment cap makes tail a genuine secondary tie-breaker.
- 3D evidence is calibration-aware, while a conservative multiview 2D fallback
  preserves information when tail triangulation is incomplete.

## Disadvantages and risks

- Tail localization or mouth-tail node swaps can produce misleading direction;
  validity gates reduce but cannot eliminate this risk.
- Fish can turn sharply across a gap, making true tail vectors discontinuous.
- Complete-case neutrality reduces availability exactly in difficult occlusions.
- Fixed body-length and residual bounds may be conservative for unusual poses.
- Added tail triangulation and audit tables increase runtime and memory.
- Female-only GT cannot establish whether a changed male-to-male decision is
  correct.

## Retain criteria

Retain Item 4 for an integrated candidate only if:

1. candidate graph identity, complete-case neutrality, conditional eligibility,
   cap, fixed-slot provenance, XYZ preservation, and determinism tests all pass;
2. every required synthetic ambiguous truth is resolved correctly and every
   clear-mouth, over-cap, and missing-tail case remains unchanged;
3. there are zero raw-resource, motion-gate, or trajectory-count violations;
4. on paired real cohorts it creates no new reviewed female swap and does not
   increase observed female changes or reduce strict mapped coverage;
5. changed decisions have inspectable complete tail evidence, and any claimed
   benefit appears on frozen controls or an independently reviewed held-out
   ambiguity rather than only the four tuning-exposed events;
6. the verifier reproduces every changed decision from the same graph and
   augmentation hashes; and
7. measured full-candidate runtime and memory overhead are operationally
   acceptable.

Reject Item 4 if it changes a clear-mouth decision, benefits from missing tail,
changes graph feasibility or geometry, violates the cap, or reproducibly
worsens paired identity evidence. Defer it if no eligible complete ambiguity
sets occur, if it changes no decisions, or if apparent gains exist only in the
four reviewed windows. If it is correct but complete-tail availability is too
low for a material benefit, retain the audit implementation for research but
do not enable it in production.
