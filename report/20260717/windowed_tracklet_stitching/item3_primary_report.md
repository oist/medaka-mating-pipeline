# Item 3 primary method report: explicit fragment end-to-start links

## Status

This is the primary method specification written before Item 3 implementation
or benchmark results were available. It contains no measured runtime or quality
claim. Benchmark fields below must be filled only from saved artifacts produced
under `benchmark_protocol.md` and independently reproduced by the Item 3
verifier.

The historical four-window tracklet-first output is not evidence for Item 3.
That implementation had no explicit end-to-start fragment optimizer, and its
four independently seeded outputs were concatenated without stitching.

## Claim under test and isolation

The claim is that explicitly scoring directed links from an ending 3D
mouth fragment to a later starting fragment improves identity continuity over
leaving every break unlinked. Item 3 may change fragment association and final
biological labels. It must not change:

- cached SLEAP candidates or strict-mouth eligibility;
- multiframe cross-view grouping from Item 2 prerequisite plumbing;
- any reconstructed, directly observed XYZ sample;
- raw `(frame, view, candidate_slot)` resource use; or
- interpolation and fill policy.

The individual comparison uses one frozen fragment table and a deterministic
sequential Hungarian boundary matcher as minimal solver plumbing:

- `explicit_fragment_links=off`: the matcher receives no inter-fragment
  continuation edges, so every Item 2 fragment break remains a birth/death;
- `explicit_fragment_links=on`: the matcher receives the directed edges and
  mouth-only costs defined below.

Both modes use birth and death cost `3.0` each, stable fragment-ID ordering,
and identical anchors. Global flow, a competing beam, deferred commitment, and
tail evidence are disabled so Items 4--6 receive no credit in this experiment.
Item 5 will later compare solvers on the exact graph serialized by Item 3.

Female GT is evaluation-only. It identifies only the female and cannot detect
male-to-male swaps.

## Frozen fragment input contract

Each Item 2 3D mouth fragment must supply an inspectable row:

```text
fragment_id
frame_start, frame_end                    inclusive source frames
direct_frame_count
frame[]                                  direct observations only
mouth_xyz_mm[]                           calibration-native float64 XYZ
refractive_residual_mm2[]                mean squared point-to-ray distance
mouth_confidence[]                       [0,1] when available
raw_resource_keys[]                      frame,view,candidate_slot
window_id, ownership_core, context_range provenance only
```

Interpolated rows are forbidden in edge feature calculation. Context duplicates
from adjacent 250-frame overlaps may be retained for agreement evidence, but
only the owning window's row is a trajectory resource. Every fragment is
preflighted for finite direct mouth XYZ, monotone unique frames, and no repeated
raw resource key. Across simultaneously active fragments, one raw resource may
not be owned twice.

An edge artifact has one row per ordered candidate pair and preserves raw and
normalized values:

```text
edge_id, source_fragment_id, target_fragment_id
source_end_frame, target_start_frame, gap_frames, bridge_dt_frames
endpoint_displacement_mm, endpoint_gate_mm
forward_prediction_error_mm, backward_prediction_error_mm
motion_prediction_error_mm, velocity_sources_available
overlap_agree_frames, overlap_disagree_frames, overlap_consistency
endpoint_refractive_residual_mm2, endpoint_mouth_confidence
all normalized cost terms, total_mouth_cost, feasible, rejection_reason
```

The graph hash covers the ordered fragment table, edge table, parameters,
calibration manifest hash, candidate-cache hash, and seed.

## Directed candidate generation

For source fragment `u` and target fragment `v`, define

```text
gap_frames = v.frame_start - u.frame_end - 1
dt         = gap_frames + 1
```

Create a candidate only when all conditions hold:

1. `u != v` and `0 <= gap_frames <= 100`.
2. Owned observations do not overlap in time. Overlap context is provenance,
   not a second owned fragment.
3. Endpoint displacement is no larger than
   `g = min(8 mm/frame * dt, 80 mm)`.
4. Both fragments passed the raw-resource exclusivity preflight.

No top-K successor pruning is allowed in the primary experiment. All feasible
pairs inside a processing window and its 250-frame context are serialized.
Cross-window deduplication uses the stable `edge_id` and exact feature equality.
An edge crossing a 10,000-frame ownership-core boundary is emitted by both
contexts when available, but only one identical canonical row survives.

## Endpoint and velocity estimators

Endpoint position is the first or last directly observed 3D mouth point. For a
terminal velocity, use at most the last 15 direct samples inside the last 30
source frames; for an initial velocity, use the first 15 direct samples inside
the first 30 source frames. At least three samples spanning at least two frames
are required.

For each coordinate independently, estimate velocity as the median of all
finite pairwise slopes `(x_j-x_i)/(frame_j-frame_i)`. This deterministic
Theil--Sen-style estimate has units mm/frame. Then compute:

```text
forward_error  = ||p_v_start - (p_u_end + velocity_u * dt)||  mm
backward_error = ||p_u_end   - (p_v_start - velocity_v * dt)|| mm
```

`motion_prediction_error_mm` is the median of the available forward and
backward errors. If only one velocity is estimable, use its error. If neither
is estimable, motion evidence is missing and its cost term is exactly zero;
missing velocity never rejects an otherwise feasible edge.

## Overlap consistency

Overlap consistency uses only duplicated direct context from independently
processed neighboring windows. For every evaluable common context frame, map
both proposals back to exact raw candidate-slot resources:

- `agree`: both contexts propose the same mutually exclusive source-slot set
  for the continuation;
- `disagree`: both are present but propose contradictory source-slot sets;
- unavailable: either side lacks direct evidence; it does not enter the
  denominator.

With at least three evaluable frames,

```text
overlap_consistency = agree / (agree + disagree)
```

and its penalty is `1-overlap_consistency`. With fewer than three frames the
field is null and its cost is zero. It is evidence, not a hard veto. Within a
single non-windowed invocation this term is therefore neutral by definition.

## Mouth-only objective and units

All feature calculations remain in their declared physical units. Only the
final terms are normalized to a dimensionless edge cost. Let:

```text
d     endpoint_displacement_mm
g     min(8 * dt, 80) mm
m     motion_prediction_error_mm
c     overlap_consistency
r     median refractive residual over the terminal and initial endpoint
      supports, in mm^2
q     minimum of the two robust endpoint mouth confidences in [0,1]
```

Endpoint residual and confidence supports use the same direct samples as the
velocity supports. Each fragment value is its median. Missing residual,
confidence, motion, or overlap evidence contributes zero rather than a
penalty or veto.

The fixed primary cost is:

```text
C_position   = 1.00 * clip(d / g,                 0, 1)
C_motion     = 2.00 * clip(m / g,                 0, 1)  if available else 0
C_gap        = 0.25 * clip(gap_frames / 100,      0, 1)
C_overlap    = 1.00 * clip(1 - c,                 0, 1)  if >=3 frames else 0
C_residual   = 0.75 * clip(r / 2.5 mm^2,          0, 4)  if available else 0
C_confidence = 0.50 * clip(1 - q,                 0, 1)  if available else 0

C_mouth = sum of the six terms
```

The `2.5 mm^2` residual scale is fixed before event results and is approximately
the squared 1.563 mm p95 reported for the held-out rigid-object validation in
`calibration_results/fixed_refractive_192x192x186/calibration_manifest.json`.
It is not a polynomial pixel-squared threshold. There is no residual hard gate
in the primary Item 3 experiment; high residual receives an inspectable capped
penalty.

The edge is selected instead of a death plus birth only through the frozen
boundary matcher and its two `3.0` dummy costs. Edge generation never inspects
female annotations, reviewed-event outcomes, geometry identity names, or a
future solver result.

## Pure proposed interface

```python
build_fragment_link_graph(
    fragments,
    *,
    max_gap_frames: int = 100,
    max_jump_mm_per_frame: float = 8.0,
    max_jump_hard_cap_mm: float = 80.0,
    velocity_support_frames: int = 30,
    velocity_max_samples: int = 15,
    velocity_min_samples: int = 3,
    overlap_min_frames: int = 3,
    residual_scale_mm2: float = 2.5,
    birth_cost: float = 3.0,
    death_cost: float = 3.0,
    seed: int = 20260717,
) -> FragmentLinkGraph
```

The result must expose feasible and rejected edges, all raw features and terms,
resource-conflict diagnostics, timing/resource telemetry, the graph hash, and
checkpoint state. Resuming from a window checkpoint must reproduce the same
canonical graph as one-pass execution.

## Deterministic synthetic cases

### Velocity disambiguates a crossing

Create two source fragments ending at frame 9, one moving `+X` and one moving
`-X`, followed after a simultaneous one-frame break by two starting fragments.
Arrange endpoint distances so nearest-position matching alone prefers the
swapped pairing by 0.2 normalized cost, while constant-velocity prediction has
zero error for the true pairing and at least 4 mm error for the swap. Add a
third distant straight fragment. The explicit-link mode must recover all three
known continuations; links-off must retain six separate fragments. The fixture
defines truth independently of the algorithm.

### Simultaneous all-view break

Break all three fish in every view for ten frames. Give each true continuation
distinct endpoint velocity and valid displacement. All nine directed candidate
links may be feasible, but the deterministic one-to-one boundary assignment
must select the three true links without reusing a target or raw detection.

### Additional required cases

- Zero-gap exact continuation has zero position, motion, and gap cost.
- `gap_frames=101` is rejected exactly as `gap_exceeds_gate`.
- A one-frame bridge over 8 mm and a long bridge over 80 mm are rejected by
  their respective displacement gates.
- Missing velocity is neutral; endpoint, gap, residual, overlap, and confidence
  remain unchanged.
- Fewer than three overlap frames is neutral; three unanimous contradictions
  produce overlap cost 1.0 but do not hard-veto the edge.
- A missing mouth creates a fragment break and no mouth endpoint; tail is never
  substituted as the Item 3 anchor.
- Missing confidence and missing residual are neutral and explicitly flagged.
- A six-candidate reflection fixture preserves the true low-residual link and
  proves that no selected paths reuse a raw slot.
- Simultaneous false-positive fragments expose infeasibility or births/deaths;
  they are not silently attached to make three paths.
- Input fragment and observation row permutations produce byte-identical edge
  tables after canonical sorting.
- An edge spanning a window ownership boundary is identical after one-pass,
  two-window, checkpoint-resume, and reversed worker-completion execution.
- Enabling links changes no direct XYZ value or per-frame XYZ multiset.

## Failure and determinism criteria

Any of the following invalidates a run:

- a selected edge violates time order, the 100-frame gap, the 8 mm/frame gate,
  or the 80 mm hard cap;
- one raw detection is assigned to two fish at the same frame;
- more than three biological trajectories are active after solving;
- interpolation, GT, reviewed-event identity, or tail evidence enters a feature;
- a missing feature becomes an infinite cost, veto, or imputed worst value;
- the serialized raw term sum differs from `total_mouth_cost` by more than
  `1e-12` in float64;
- graph construction depends on input row order, worker completion order, or
  hash-map iteration;
- checkpoint/resume changes edge IDs, features, costs, or selected links;
- links-on and links-off use different fragments, candidate resources,
  calibration, or anchors; or
- any direct reconstructed XYZ coordinate changes.

Fixed-seed reruns must produce byte-identical canonical fragment and edge
tables. Timing and OS-specific telemetry files are excluded from the content
hash but not from provenance.

## Benchmark plan and placeholders

No benchmark has yet been run. The implementation owner must replace module and
path placeholders before execution and save exact commands.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> build-links `
  --fragment-dir <FROZEN_ITEM2_FRAGMENT_DIR> `
  --max-gap-frames 100 --max-jump-mm 8 --max-jump-hard-cap-mm 80 `
  --velocity-support-frames 30 --velocity-max-samples 15 `
  --overlap-min-frames 3 --residual-scale-mm2 2.5 `
  --seed 20260717 --output-dir <GRAPH_DIR>

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> solve `
  --fragment-dir <FROZEN_ITEM2_FRAGMENT_DIR> `
  --explicit-fragment-links off --solver sequential_hungarian `
  --birth-cost 3 --death-cost 3 --seed 20260717 `
  --output-dir <OUT_DIR>\links_off

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> solve `
  --graph-dir <GRAPH_DIR> --explicit-fragment-links on `
  --solver sequential_hungarian --birth-cost 3 --death-cost 3 `
  --seed 20260717 --output-dir <OUT_DIR>\links_on
```

The comparison must run on synthetic fixtures, numerical-equivalence windows,
all four reviewed windows, and the exact frozen controls in
`benchmark_protocol.md`: 32,264--35,364; 151,050--155,050;
21,840--30,840; and 57,363--67,363. A full recording is allowed only if the
individual evidence justifies it.

## Measurements to fill after execution

Not yet measured:

- fragment and candidate-edge counts, feasible/rejected counts, degree and gap
  distributions, and graph hash;
- selected link count, births/deaths avoided, fragment-length distribution,
  unresolved intervals, and exact rejection reasons;
- synthetic link precision/recall and crossing/all-view-break outcomes;
- link conflict rate and measurable cross-view link precision;
- refractive residual and every normalized cost-term distribution;
- eligible, mapped, and strict female-GT counts, paired common cohorts, best
  fixed female accuracy, observed female changes, and reviewed-event outcomes;
- all-three direct/interpolated coverage, jump p95/p99/max, and gate violations;
- cold/warm wall time, CPU utilization, peak RSS, output size, checkpoint cost,
  and any GPU measurement if separately justified;
- fixed-seed rerun hashes, numerical stability, failure cases, and verifier
  reproduction.

## Advantages

- Represents the intended missing operation: an inspectable directed decision
  between broken 3D fragments.
- Uses robust multiframe endpoint and velocity evidence instead of isolated
  frame assignments.
- Keeps every feature in its physical unit before explicit normalization.
- Preserves missing evidence as unknown rather than converting absence into a
  false contradiction.
- Produces one frozen graph that Items 4--6 can consume without attribution
  leakage.

## Disadvantages and risks

- Constant velocity is weak during rapid turns or acceleration.
- Cost weights and scales may be misspecified even when units are correct.
- Candidate count can grow substantially around simultaneous breaks and
  reflections.
- A low-cost link is not biological proof; female-only GT cannot expose
  male-to-male swaps.
- Neutral missing evidence can make poorly observed links too competitive;
  availability distributions must therefore be reported.
- Sequential Hungarian is deliberately local and may understate Item 3's value
  or make a globally inconsistent choice; that question belongs to Item 5.

## Retain criteria

Retain Item 3 for an integrated candidate only if:

1. all deterministic synthetic truths, hard gates, resource-exclusivity,
   window-resume, and XYZ-preservation tests pass;
2. synthetic link precision and recall are both 100% for the required fixtures;
3. no reviewed event that was preserved by the optimized mouth-only baseline
   becomes a female swap, and paired observed female changes do not increase;
4. direct all-three coverage does not regress and there are zero selected
   motion-gate or raw-resource violations;
5. it reduces fragment births/deaths or unresolved duration on frozen controls,
   not only on the reviewed event windows;
6. any identity gain is reproduced by the independent verifier on an identical
   graph hash; and
7. full-recording runtime and peak memory remain operationally feasible under
   the fixed worker benchmark.

Reject Item 3 if it violates a hard safeguard, is nondeterministic, changes
geometry, or reproducibly increases paired female switches. If it is correct
but quality is indistinguishable, retain the graph builder as prerequisite
plumbing for Items 4--6 but do not credit Item 3 as an independently supported
production improvement. If runtime overhead is material without a quality or
fragmentation benefit, defer production use.
