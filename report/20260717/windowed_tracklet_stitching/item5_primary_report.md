# Item 5 primary method report: global trajectory solver

## Status

This is the primary method specification, written before implementation results
were available. No runtime, resource, or tracking-quality measurement is claimed
here. All result fields and commands below must be completed from actual saved
artifacts under the frozen `benchmark_protocol.md`.

## Claim under test

Given one frozen, time-ordered fragment graph, a global three-path solver can
choose a lower-cost and more identity-consistent partition than sequential
boundary Hungarian matching, and can improve on a short beam when a locally
suboptimal boundary decision is disambiguated by later observations.

The solver may change fragment-to-trajectory association, but it must not change
the reconstructed XYZ samples, fragment observations, edge feature values, or
edge costs. Female GT is used only after solving. It identifies only the female
and cannot detect male-to-male swaps.

## Existing code that can be reused

- `scripts/tracking/global_fragment_stitching.py:26-116` provides useful
  fragment, edge, and result dataclass patterns.
- `calculate_edge_cost` at
  `scripts/tracking/global_fragment_stitching.py:136` provides an inspectable
  cost-term pattern. The new solver must consume already serialized, unit-aware
  costs rather than recomputing these legacy terms.
- `solve_constrained_path_cover` at
  `scripts/tracking/global_fragment_stitching.py:270` provides SciPy/HiGHS MILP
  plumbing and useful tests, but is not the requested solver as written. It
  permits unlabelled components, does not enforce exactly three paths, and
  rejects overlapping fragments.
- `apply_cross_frame_tracking` at
  `scripts/reconstruction_3d/cross_frame_tracking.py:168` is the current
  framewise Hungarian reference, but it cannot be used for the controlled Item
  5 comparison because it does not consume the same fragment graph.
- `track_vector_beam` at
  `scripts/tracking/compare_top_only_identity_methods.py:302` demonstrates beam
  state expansion only. Its 2D midpoint/body-vector cost is not reusable for
  the controlled comparison.

The existing structural benchmark is not a valid Round 7 cost source:
`scripts/tracking/benchmark_global_fragment_stitching.py:70-225` labels errors
as pixel-squared and square-roots them, while the fixed refractive backend used
here reports mean squared point-to-ray distance in `mm^2`.

## Frozen graph contract

All three solvers receive byte-identical node, edge, and resource tables and a
recorded graph content hash.

Required node fields are:

```text
fragment_id, frame_start, frame_end, observed_frame_count,
raw_resource_hash, directly_observed_rows, synthetic_view_rows,
residual_median_mm2, residual_mad_mm2, xyz_artifact_row_start,
xyz_artifact_row_count
```

Required edge fields are:

```text
source_id, target_id, feasible, total_cost, cost_unit,
gap_frames, endpoint_displacement_mm, prediction_error_mm,
velocity_angle_deg, overlap_consistency_mm, residual_term,
mouth_confidence_term, tail_term, tail_evidence_state,
gate_reason, deterministic_edge_id
```

`total_cost` is dimensionless after each physical feature has been normalized
by its frozen scale. Item 5 never recalculates or changes it. `tail_term` is
identically zero for the Item 5 individual experiment unless the separately
credited Item 4 flag is enabled in a combination experiment.

Before solving, the graph must pass these invariants:

1. Every edge is forward in time: `source.frame_end < target.frame_start`.
2. Every gap is at most 100 frames and respects the 8 mm/frame and 80 mm hard
   motion gates used by the edge builder.
3. Window-overlap aliases have already been canonicalized; duplicate aliases
   are not trajectory edges.
4. No two fragment nodes reuse the same `(frame, view, candidate_slot)` raw
   detection resource.
5. No more than three accepted fragment nodes overlap at any frame.
6. Each accepted fragment is assigned exactly once. The solver may not hide an
   infeasible graph by silently dropping a fragment.

Failure of an invariant is an upstream graph failure, not a solver result.

## Exact common objective

Let `V` be the accepted fragments and `E` the feasible directed edges. For every
edge `(u,v)`, let `y_uv` be one when that link is selected. For every fragment,
let `b_v` and `d_v` mark a path birth and death. All variables are binary.

The common feasible-set constraints are:

```text
b_v + sum(y_uv for u in predecessors(v)) = 1       for every v
d_v + sum(y_vw for w in successors(v)) = 1         for every v
sum(b_v for v in V) = 3
sum(d_v for v in V) = 3
```

Because the graph is a DAG, these equations partition every accepted fragment
into exactly three disjoint time-ordered paths. The primary objective is:

```text
J = sum(total_cost_uv * y_uv for (u,v) in E)
```

Birth and death counts are fixed at three and therefore do not need an
arbitrary scalar penalty. The selected graph contains exactly `|V|-3` edges.

If the graph has no exact three-path cover, the run is `infeasible`; it must not
relax the gap, motion, resource, or three-path constraints. The upstream stage
must be diagnosed and rerun with a justified correction. If genuine missing
evidence prevents biological continuity, the appropriate output is an explicit
unresolved interval in Item 6, not an out-of-gate link.

The primary optimum is the only scientific objective. When several solutions
are equal within absolute tolerance `1e-9` dimensionless cost, use a secondary
deterministic selection of the lexicographically smallest sorted tuple of
`deterministic_edge_id`. This secondary rule must not select a solution whose
primary objective is more than `1e-9` above the optimum.

## Solver interface

The implementation should expose one pure interface:

```python
solve_fragment_graph(
    graph,
    *,
    method: Literal[
        "sequential_hungarian", "beam", "global_state_flow"
    ],
    beam_width: int = 32,
    objective_tolerance: float = 1e-9,
    seed: int = 20260717,
) -> SolverResult
```

`SolverResult` must contain selected edge IDs, three ordered path lists,
objective value recomputed from the frozen edge table, feasibility/status,
runtime, explored/pruned state counts, and deterministic output hash.

### Sequential Hungarian

Process fragment boundary events in increasing frame order. At each event,
construct the complete path-endpoint to new-fragment cost matrix from the frozen
edges. Missing edges are forbidden, not assigned a large finite scientific
cost. Stable row/column ordering uses fragment IDs. Commit the minimum Hungarian
assignment immediately and never revisit it.

### Short beam

Use the same event order, assignment enumeration, forbidden edges, and costs.
Retain the 32 lowest cumulative-cost partial path covers after each boundary.
Deduplicate states with identical ordered endpoint IDs and used-fragment sets.
Sort by `(cumulative_cost, selected_edge_id_tuple)` before pruning. Beam width
32 is fixed for the primary comparison and must not be tuned on the reviewed
events.

### Global state-flow solver

Solve the binary constraints and objective above over the complete graph using
SciPy/HiGHS MILP or an exactly equivalent three-unit min-cost DAG flow after the
graph has passed the conflict-free preflight. The result must be independently
re-evaluated against the equations and objective after the solver returns.

## Deterministic synthetic counterexamples

### Delayed disambiguation defeats immediate Hungarian

Create three early fragments `A0,B0,C0`, ambiguous middle fragments `A1,B1,C1`,
and separated late fragments `A2,B2,C2`. Set all `C` links to cost 0. Let the
first boundary have local costs:

```text
A0->A1 = 1.0, A0->B1 = 1.1
B0->B1 = 1.0, B0->A1 = 1.1
```

and the second boundary have:

```text
A1->A2 = 50, A1->B2 = 0
B1->B2 = 50, B1->A2 = 0
```

Immediate Hungarian commits the locally cheaper first assignment and pays 102
over the two boundaries. Beam width at least two and the global solver retain
or select the 2.2-cost alternative. The global output must have the lowest
common objective; this test demonstrates optimization, not biological truth.

### Additional required cases

- A greedy lowest-edge choice that consumes the only valid predecessor of a
  third path.
- Three simultaneous all-view breaks with a unique valid global cover.
- Equal-cost covers to verify the lexicographic tie rule.
- Missing one feasible edge so that no three-path cover exists; all solvers must
  report infeasible rather than force an out-of-gate link.
- Fragment input order and edge input order permutations yielding byte-identical
  path tables.
- A graph with a duplicated raw detection resource rejected before solving.
- A conflict-free graph on which the MILP and min-cost-flow implementations, if
  both retained, agree on objective and selected edge IDs.

## Benchmark plan and command templates

The graph must be built once per cohort and then treated read-only for all three
commands. Replace angle-bracket placeholders only after the implementation
entry point and final graph paths exist.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> solve `
  --graph-dir <FROZEN_GRAPH_DIR> --solver sequential_hungarian `
  --seed 20260717 --output-dir <OUT_DIR>\sequential

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> solve `
  --graph-dir <FROZEN_GRAPH_DIR> --solver beam --beam-width 32 `
  --seed 20260717 --output-dir <OUT_DIR>\beam32

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> solve `
  --graph-dir <FROZEN_GRAPH_DIR> --solver global_state_flow `
  --objective-tolerance 1e-9 --seed 20260717 `
  --output-dir <OUT_DIR>\global
```

The completed report must replace these placeholders with exact reproducible
commands for unit/synthetic, equivalence, four reviewed windows, four frozen
controls, and any full-recording finalist. It must report graph hash equality
before comparing objectives or quality.

## Required measurements

Not yet measured:

- common graph node/edge counts and graph hash;
- feasibility and objective for each solver;
- objective gap to the global optimum;
- explored/pruned state counts;
- wall time, process-tree CPU utilization, peak RSS, and output size;
- direct/interpolated coverage, fragments/path, jumps, and gate violations;
- eligible/mapped/strict female-GT counts, best fixed female accuracy, observed
  female-ID changes, paired common cohort, and reviewed-event outcomes;
- deterministic rerun hashes and numerical stability;
- failure cases and timeouts.

## Advantages

- Uses later evidence without changing the graph or costs.
- Produces an auditable lower bound for local and beam solver objective gaps.
- Enforces three disjoint paths and uses every accepted fragment exactly once.
- Keeps scientific attribution clean because all solvers consume the same
  serialized decisions and terms.

## Disadvantages and risks

- Lower association cost does not prove better biological identity; the cost
  model may be misspecified.
- Exact global optimization can be slower and more memory intensive than local
  matching.
- Requiring all accepted fragments exposes upstream false-positive grouping as
  infeasibility instead of quietly discarding it. This is deliberate but may
  increase diagnostic reruns.
- Offline look-ahead is appropriate for this reconstruction but cannot be
  described as an online tracker.
- Female GT cannot reveal male-to-male swaps, so event review and synthetic
  truth remain essential.

## Retain criteria

Retain the global solver for an integrated full-recording candidate only if:

1. every graph and solver safeguard passes deterministically;
2. it is feasible on all required real cohorts without relaxing frozen gates;
3. its recomputed objective is no worse than sequential or beam on every
   identical graph;
4. paired identity and reviewed-event evidence is non-inferior, with no new
   reviewed female swap and no increase in observed female-ID changes;
5. direct all-three coverage and jump-gate metrics do not regress materially;
6. runtime and memory fit the full-recording execution budget; and
7. any claimed gain survives frozen non-event controls and is not based only on
   the four reviewed windows.

Reject it if it violates exclusivity, changes XYZ, is nondeterministic, or
improves objective while reproducibly worsening identity evidence. Defer it if
quality is indistinguishable and its runtime/resource cost is material.
