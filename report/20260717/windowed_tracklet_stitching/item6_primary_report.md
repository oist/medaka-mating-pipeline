# Item 6 primary method report: deferred identity decisions

## Status

This is the primary method specification, written before implementation results
were available. It contains no measured runtime or quality claim. The eventual
implementation and independent verifier must fill the benchmark placeholders
with saved artifacts produced under `benchmark_protocol.md`.

## Claim under test

When mouth-based fragment association has two or more nearly equivalent
identity continuations during close contact, retaining the feasible alternatives
until later directly observed separation can avoid an irreversible local ID
swap. When later evidence does not distinguish the alternatives, the method
must emit an unresolved identity interval rather than fabricate certainty.

Deferral may change fragment-to-biological-identity association. It must not
change candidate grouping, reconstructed XYZ, edge features, edge costs, or raw
detection use. Female GT is evaluation-only, identifies only the female, and
cannot detect male-to-male swaps.

## Existing code that can inform, but not implement, Item 6

- `BeamHypothesis` and `track_vector_beam` in
  `scripts/tracking/compare_top_only_identity_methods.py:62,302` demonstrate
  retained histories. That experiment always beams over 2D midpoint/body-vector
  costs and has no explicit ambiguity trigger, convergence rule, or unresolved
  output.
- `classify_event` in
  `scripts/tracking/resolve_hardcore_events.py:79` compares no-swap and swap
  motion after an event. It is a useful diagnostic counterexample but acts only
  after identities have already been committed.
- `solve_constrained_path_cover` in
  `scripts/tracking/global_fragment_stitching.py:270` is one-best optimization;
  it does not preserve or report competing identity hypotheses.

Item 6 therefore needs a new commit-policy layer over a frozen hypothesis graph.

## Isolation from Item 5

Item 5 asks which one-best path solver is preferable. Item 6 asks when a
biological identity decision becomes committed.

For the Item 6 individual experiment, both policies use the same short-beam
enumerator, graph hash, edge costs, beam width, and state transitions:

- `immediate`: at the first ambiguous boundary, keep the current best complete
  assignment and prune all competing identity permutations.
- `deferred`: keep all qualifying hypotheses until they agree after separation,
  one wins by the frozen resolution rule, or the horizon expires.

This prevents global optimization itself from being credited to deferral. In an
Item 5 x Item 6 combination, the same deferred commit policy may be supplied by
a K-best global/state-flow implementation, but those results must be reported
separately.

## Frozen input and output contract

Input is the same unit-aware fragment graph and cost table described in the
Item 5 report. It also carries mouth-only ambiguity metadata computed before
tail evidence or biological GT is consulted:

```text
boundary_id, boundary_frame, candidate_assignment_id,
mouth_only_incremental_cost, competing_mouth_distance_mm,
direct_observation_count, feasible, selected_edge_ids
```

Hypothesis state is:

```text
hypothesis_id
cumulative_cost                         dimensionless
three ordered current path endpoints
biological-label permutation            one of six permutations
selected edge ID history
ambiguity episode history
last directly observed frame per path
```

Output adds, without altering XYZ:

```text
ambiguity_id, ambiguity_start, ambiguity_end,
hypotheses_entered, peak_hypotheses, hypotheses_survived,
resolution_status, resolution_frame, winning_margin,
prune_reasons, identity_status
```

Resolved offline episodes may backfill the winning identity across their
ambiguous interval. If an episode remains unresolved, geometry rows remain in
the output but `final_fish_id` is blank for the disagreeing paths and
`identity_status="unresolved"`. It is forbidden to force the current leader at
the horizon.

## Exact ambiguity and retention policy

All costs below are dimensionless normalized graph costs. Frames are source
frames at 166 fps.

Fixed primary parameters are:

```text
beam_width                    32 hypotheses
entry_cost_margin             0.5
retention_cost_margin         2.0
resolution_cost_margin        1.0
resolution_stability_frames   10 directly observed frames
maximum_defer_horizon          250 frames (1.506 seconds)
objective_tolerance            1e-9
seed                           20260717
```

These values are frozen for the primary benchmark and must not be tuned on the
four reviewed event windows. Sensitivity checks, if performed, are exploratory
and cannot replace the frozen result.

An ambiguity episode opens only when all of the following hold:

1. At least two complete feasible three-path successor assignments exist at the
   same boundary.
2. The assignments disagree on at least one biological-label permutation.
3. The second-best mouth-only incremental cost is within 0.5 of the best.
4. The alternatives passed all resource, 100-frame gap, 8 mm/frame, and 80 mm
   hard-cap constraints.

The episode definition uses mouth-only costs. Conditional tail evidence may
alter later cumulative costs only in a separately credited Item 4 combination;
it may not decide whether the episode existed.

At each subsequent directly observed boundary, enumerate feasible successors,
accumulate the unchanged graph edge costs, merge identical states, and retain
states whose cumulative cost is within 2.0 of the current best. If more than 32
remain, stable-sort by `(cumulative_cost, selected_edge_id_tuple,
biological_permutation)` and retain 32. Record every margin and prune reason.

Resolve an episode at the first point where either:

- all surviving hypotheses agree on the identity permutation through the
  ambiguous interval; or
- the best hypothesis leads every disagreeing runner-up by at least 1.0 for 10
  consecutive directly observed frames.

Missing observations do not count toward the 10-frame stability requirement.
Interpolated XYZ, filled rows, female annotations, reviewed-event labels, and
future manual knowledge never contribute evidence.

If 250 frames elapse without resolution, close the episode as unresolved. A
later independent ambiguity starts a new episode rather than silently revising
the unresolved one.

## Pure interface

```python
resolve_identity_hypotheses(
    graph,
    *,
    policy: Literal["immediate", "deferred"],
    beam_width: int = 32,
    entry_cost_margin: float = 0.5,
    retention_cost_margin: float = 2.0,
    resolution_cost_margin: float = 1.0,
    resolution_stability_frames: int = 10,
    maximum_defer_horizon: int = 250,
    objective_tolerance: float = 1e-9,
    seed: int = 20260717,
    initial_state=None,
) -> IdentityResolutionResult
```

The result must include final path labels, every ambiguity episode, the complete
hypothesis/pruning audit table, state checkpoints at window boundaries, runtime,
resource telemetry, and deterministic content hashes.

`initial_state` is required for independently processed temporal windows. It is
serialized at the left context boundary and must reproduce byte-identical
results after resume. No hypothesis may be silently lost merely because an
episode crosses a 10,000-frame core boundary.

## Deterministic synthetic counterexamples

### Later separation reverses the local leader

Use the three-path graph from the Item 5 delayed-disambiguation test. At the
first boundary, the no-swap assignment costs 2.0 and the alternative costs 2.2,
so both enter because the margin is 0.2. Later separated fragments add 100 to
the immediate no-swap history and zero to the alternative. The immediate policy
must retain its committed wrong history; deferred must keep the 2.2 candidate
and resolve to it after 10 directly observed stable frames.

This is a solver-policy test. The synthetic identity truth is defined in the
fixture and is not inferred from output geometry.

### No informative separation

Provide two identity permutations with exactly equal costs for more than 250
frames. Deferred must report one unresolved interval, preserve both hypotheses
until the horizon, leave disagreeing biological IDs blank, and remain
deterministic. Immediate may choose the lexicographic leader but must be scored
as a forced decision.

### Additional required cases

- A one-frame near-tie followed by ten agreeing direct frames resolves at the
  exact tenth frame.
- Missing observations interrupt but do not satisfy the stability count.
- An alternative outside the 0.5 entry margin is never credited as deferred.
- More than 32 feasible histories exercise deterministic pruning; the known
  true history must remain when it is within the retention margin, otherwise
  the test exposes an inadequate cap.
- Missing tail evidence is neutral and does not close an episode.
- Reflection candidates cannot create hypotheses that reuse a raw detection.
- A simultaneous all-view break retains all feasible identity permutations.
- An episode crossing a window/checkpoint boundary is identical to a one-pass
  run after resume.
- Input row/edge order permutations produce byte-identical ambiguity tables.
- Resolved and unresolved policies preserve the exact per-frame multiset of XYZ
  values and the selected raw detection resources.

## Failure criteria

Any of the following invalidates the run:

- a decision uses GT, reviewed-event labels, interpolation, or future evidence
  outside the graph;
- hypotheses disagree while the output claims a resolved biological ID;
- horizon expiry forces a winner instead of returning unresolved;
- a retained hypothesis violates a raw-detection, time-order, or motion gate;
- the true synthetic hypothesis is pruned despite remaining inside the frozen
  retention margin and beam capacity;
- window serialization/resume changes survivors or final IDs;
- immediate and deferred runs do not start from the same graph and hypothesis
  set;
- geometry or candidate grouping differs between policies;
- fixed-seed reruns are not byte-equivalent.

## Benchmark plan and command templates

Build each frozen hypothesis graph once. Replace placeholders only after the
implementation and artifact paths exist.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> resolve-identities `
  --graph-dir <FROZEN_GRAPH_DIR> --identity-policy immediate `
  --beam-width 32 --seed 20260717 `
  --output-dir <OUT_DIR>\immediate

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> resolve-identities `
  --graph-dir <FROZEN_GRAPH_DIR> --identity-policy deferred `
  --beam-width 32 --entry-cost-margin 0.5 `
  --retention-cost-margin 2.0 --resolution-cost-margin 1.0 `
  --resolution-stability-frames 10 --maximum-defer-horizon 250 `
  --seed 20260717 --output-dir <OUT_DIR>\deferred
```

The completed report must replace placeholders with exact commands for
synthetic tests, small equivalence windows, four reviewed windows, four frozen
controls, and any justified full-recording candidate. Before comparison it must
assert equality of graph hash, candidate-assignment hash, and initial hypothesis
hash.

## Required measurements

Not yet measured:

- ambiguity episodes opened per cohort and their input reason;
- hypotheses entered, peak, pruned, and surviving per episode;
- resolved/unresolved interval count and duration;
- resolution latency in frames and seconds;
- immediate versus deferred forced-decision count;
- synthetic correct-hypothesis retention and resolution accuracy;
- wall time, process-tree CPU utilization, peak RSS, output size, and checkpoint
  overhead;
- eligible/mapped/strict female-GT counts, best fixed female accuracy, observed
  female-ID changes, paired common cohort, and reviewed-event outcomes;
- all-three direct/interpolated coverage and jump metrics;
- deterministic output hashes, numerical stability, and failure cases.

## Advantages

- Represents genuine uncertainty instead of converting a near-tie into an
  irreversible identity claim.
- Uses later directly observed evidence while preserving the original geometry
  and costs.
- Produces inspectable ambiguity intervals and hypothesis counts required by
  the protocol.
- Can avoid a swap that one-best immediate association cannot repair.

## Disadvantages and risks

- Adds state, memory, checkpoint, and audit-output cost.
- The beam cap can prune the correct history; cap sensitivity must be reported.
- Cost margins are meaningful only if upstream terms are properly normalized.
- More unresolved intervals can lower mapped GT coverage even while improving
  honesty and reducing forced errors.
- Offline backfilling after separation is not causal online tracking.
- Sparse female-only GT cannot validate male-to-male identity continuity.

## Retain criteria

Retain deferred identity for an integrated full-recording candidate only if:

1. all synthetic truths, unresolved cases, exclusivity checks, checkpoint
   checks, and deterministic reruns pass;
2. it retains the known correct synthetic hypothesis whenever that hypothesis
   is inside the frozen beam/margin contract;
3. on paired real cohorts it reduces forced decisions or observed female-ID
   changes without creating a new reviewed female swap;
4. any loss of mapped/direct identity coverage is explicitly attributable to
   unresolved intervals and is scientifically preferable to incorrect forced
   labels;
5. unresolved duration remains operationally manageable and does not consume a
   material fraction of the recording without independent evidence;
6. runtime, memory, and window-checkpoint overhead fit the full-recording
   budget; and
7. gains appear on frozen controls or a justified held-out cohort, not only the
   four reviewed windows.

Reject it if it fabricates certainty, changes geometry, violates resources,
loses the correct in-margin synthetic hypothesis, or reproducibly worsens
paired identity evidence. Defer production adoption if it is correct but leaves
too much of the recording unresolved or its runtime/resource cost is material.
