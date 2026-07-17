# Item 2 primary method report: multiframe cross-view 3D tracklet grouping

## Status

This is the primary method specification, written before implementation results
were available. It claims no measured runtime, resource use, coverage, or
tracking improvement. All result fields and command placeholders must be filled
only from saved artifacts under the frozen `benchmark_protocol.md`.

## Claim under test

Aggregating the same refractive cross-view hypotheses over multiple overlapping
frames of conservative per-view mouth tracklets can reject transient
reflection-driven or noisy one-frame matches and form more coherent 3D
fragments than one-frame grouping, without reusing a detection or forcing a
biological identity.

Item 2 ends when cross-view 3D fragments and their provenance are produced. It
does not link one 3D fragment's end to another fragment's start (Item 3), use
tail vectors (Item 4), choose among trajectory solvers (Item 5), or retain
deferred biological-ID hypotheses (Item 6). Female GT is evaluated only after
the common downstream linker. It identifies only the female and cannot detect
male-to-male swaps.

## Existing code and methodological gap

- `scripts/reconstruction_3d/triangulate_extra_candidates.py:51` already
  evaluates up to `6^3=216` full-view hypotheses per frame, performs calibrated
  two-view recovery, and records source slots and refractive scores.
- `scripts/reconstruction_3d/tracklet_first_assignment.py:143`
  `select_global_candidate_set` enforces per-frame resource exclusivity. Its
  production 48-candidate frontier at `:165-188` is a bounded heuristic and
  should remain a reference, not be assumed globally exact.
- `scripts/reconstruction_3d/multiview_tracklet_tracking.py:245`
  `build_observations` maps already-selected framewise sources to tracklets.
- `build_constrained_tracklet_graph` at
  `scripts/reconstruction_3d/multiview_tracklet_tracking.py:309` aggregates
  overlap votes after a framewise bootstrap and merges eligible pairwise edges
  with union-find. It does not aggregate the original candidate triangulation
  evidence into a multiframe tracklet-group objective.
- `scripts/tracking/concrete_tracklets.py:52` supplies the conservative
  per-view baseline tracklets. New cached tracklets should carry exact
  `(frame, view, candidate_slot)` resources rather than relying on the xy
  rematch in `load_tracklet_slot_map` at
  `multiview_tracklet_tracking.py:96`.

The optimized candidate cache and exact tracklet-to-slot plumbing are
prerequisites and receive no Item 2 scientific credit.

## Frozen input table

Build the refractive hypothesis table once and hash it before comparing
framewise and multiframe modes. Each row is one finite in-bounds hypothesis and
contains:

```text
frame, view_mask, cam0_slot, cam1_slot, cam2_slot,
cam0_tracklet_id, cam1_tracklet_id, cam2_tracklet_id,
cam0_resource, cam1_resource, cam2_resource,
X_mm, Y_mm, Z_mm, ray_residual_mm2,
mouth_confidence_cam0, mouth_confidence_cam1, mouth_confidence_cam2,
synthetic_view, deterministic_hypothesis_id
```

Strict finite mouth is required for every observed view. Missing tails are
retained in the cache but not read by Item 2. The delivered calibration is the
fixed-interface refractive backend, coordinates remain native, and all XYZ must
be inside 0--200 mm. Two-view rows keep the absent slot/resource/tracklet as
`-1` and record the synthetic view.

The framewise comparator treats each hypothesis row as a one-frame group. The
multiframe method receives the byte-identical table and aggregates rows; it may
not retriangulate, rescore, or change candidate filtering.

## Multiframe candidate-group construction

### Group key and support

Define a group key from the observed per-view tracklet IDs:

```text
(cam0_tracklet_id, cam1_tracklet_id, cam2_tracklet_id, view_mask)
```

At least two views must be observed. For a fixed key there is at most one raw
detection from each constituent tracklet at a frame. If multiple hypothesis
rows nevertheless have the same key and frame, retain the lowest residual,
breaking exact ties by `deterministic_hypothesis_id`, and record the discarded
duplicates.

Let `F_overlap(g)` be frames in which every tracklet named by the group has a
finite mouth observation. Let `F_valid(g)` be the frames represented by a
finite, in-bounds row in the frozen hypothesis table. Define:

```text
n_overlap(g)      = |F_overlap(g)|
n_support(g)      = |F_valid(g)|
valid_fraction(g) = n_support / max(n_overlap, 1)
```

The primary multiframe mode requires:

```text
n_support >= 3 frames
valid_fraction >= 0.95
```

These values match the existing minimum overlap and consistency defaults and
are frozen before reviewed-event results. A one- or two-frame reflection cannot
be promoted to a multiframe group.

### Robust refractive evidence

For the `mm^2` residual samples `e_f` of an eligible key, compute:

```text
median_g      = median(e_f)
mad_g         = median(abs(e_f - median_g))
sigma_mad_g   = 1.4826 * mad_g
robust_upper_g = median_g + 3 * sigma_mad_g
```

If the common optimized-baseline configuration has a finite
`max_ray_residual_mm2=R_max`, require `robust_upper_g <= R_max`. If it has no
finite residual gate, set `R_max=+infinity`; Item 2 must not invent or tune a
new absolute gate on the reviewed windows. Persist all four statistics and the
gate provenance.

The selected group retains only its frozen support rows. It does not fill a
frame whose triangulation was invalid, and it does not convert a two-view row
into a directly observed three-view row.

### Conflict resources

For group `g`, let `R_g` be its exact observed raw resources over its support
frames. Two candidate groups conflict when their resource sets intersect. An
additional per-frame capacity constraint permits no more than three selected
groups at a frame. These constraints cover reflections and partial views: a
synthetic missing view consumes no resource in that view.

## Exact common selection objective

Use the same set-packing solver for both modes. In framewise mode each eligible
hypothesis row is a one-frame candidate group. In multiframe mode candidates
are the eligible aggregated groups above. Let binary `z_g` select group `g`.

Constraints are:

```text
sum(z_g for g containing raw resource r) <= 1          for every resource r
sum(z_g for g observed at frame f) <= 3                for every frame f
```

The objective is lexicographic:

1. maximize selected fish-frame support
   `C = sum(n_support(g) * z_g)`;
2. at maximum `C`, maximize observed-view resources
   `V = sum(view_count(g) * n_support(g) * z_g)` so a direct three-view row is
   preferred to a two-view row when fish-frame coverage is tied;
3. at maximum `C,V`, minimize robust refractive cost
   `J = sum(n_support(g) * robust_upper_g * z_g)` in `mm^2*frame`;
4. for solutions equal within `1e-9` in `J`, choose the lexicographically
   smallest sorted tuple of deterministic group IDs.

Implement the stages as successive constrained solves or an exactly verified
integer scaling that cannot change lexicographic priority. Do not combine
coverage and residual with an arbitrary weighted sum.

The result is a set of cross-view 3D fragments. Biological identity and exactly
three full-recording trajectories are deliberately outside this objective.

## Deterministic IDs and output schema

The deterministic group ID is the content hash of its view mask, ordered
tracklet IDs, support frames, and ordered raw resources. It cannot depend on
Pandas group order, worker order, or a local integer ID.

Required outputs are:

```text
group_candidates.parquet
selected_groups.parquet
selected_group_observations.parquet
group_conflicts.parquet
discarded_duplicate_hypotheses.parquet
grouping_summary.json
```

Candidate rows record eligibility, support counts, valid fraction, residual
median/MAD/robust upper, residual gate/unit, view count, conflict degree,
selected state, and rejection reason. Observation rows preserve frozen XYZ,
residual, source slots, raw resources, and direct/synthetic-view state exactly.

## Component interface

The eventual implementation should expose a pure interface equivalent to:

```python
group_cross_view_tracklets(
    frozen_hypotheses,
    per_view_tracklets,
    *,
    mode: Literal["framewise", "multiframe"],
    min_support_frames: int = 3,
    min_valid_fraction: float = 0.95,
    residual_mad_scale: float = 1.4826,
    robust_sigma_multiplier: float = 3.0,
    max_ray_residual_mm2: float | None = None,
    max_active_groups: int = 3,
    objective_tolerance: float = 1e-9,
    seed: int = 20260717,
) -> GroupingResult
```

The function must not read female GT or tail vectors. A fixed seed is recorded
for uniform protocol, but the defined algorithm has no stochastic tie.

## Required deterministic tests

### Synthetic scientific cases

- A correct tracklet triple with moderate residual over ten frames competes
  with a lower-residual reflection match present for only two frames. The
  framewise mode may select the transient match; multiframe mode must reject it
  for insufficient support.
- One extreme residual outlier among otherwise stable frames is summarized by
  the frozen median/MAD rule without changing its raw value.
- A consistently wrong low-residual group remains selectable, demonstrating
  that multiframe persistence is not biological proof; this failure case must
  be documented rather than hidden.
- Two eligible groups sharing one `(frame, view, candidate_slot)` cannot both
  be selected even when that increases apparent coverage.
- Three compatible groups can be selected; a fourth simultaneous group is
  rejected by the capacity constraint.
- A simultaneous all-view break creates separate group fragments and is not
  linked across the gap.
- A two-view group with one missing mouth view consumes only its two real
  resources and remains marked synthetic. A tail-only instance is absent under
  the strict-mouth contract. Missing tails are neutral.
- Reflection candidates that persist for at least three frames test the limit
  of the persistence assumption.
- Input row, group, tracklet, and conflict order permutations yield identical
  selected group IDs and output hashes.

### Controlled equivalence cases

- With one-frame groups, the framewise set-packing result matches the frozen
  per-frame source selections and cardinality on deterministic small windows.
- A tracklet fixture in which each group has exactly one support frame shows
  that the only intended difference is the `min_support_frames=3` multiframe
  eligibility rule.
- Batched and scalar refractive hypothesis generation produce identical source
  slots and values within `atol=1e-9`, `rtol=1e-9` before grouping.
- Every selected observation row is byte-identical to its source row except for
  added group/provenance columns.

### Invariants

- No raw resource is selected twice.
- At most three groups are active per frame.
- Every multiframe selected group has support at least three and valid fraction
  at least 0.95.
- All saved residuals and gates are explicitly `mm^2`.
- Fixed-seed reruns and alternative input orderings are byte-equivalent.

## Isolation from other components

For the Item 2 individual experiment:

- run one bounded cohort invocation rather than crediting Item 1 overlap
  reconciliation;
- disable explicit end-to-start links;
- set every tail term to zero and do not load tail features;
- pass the selected fragments to the same sequential downstream comparator;
- force immediate decisions with no beam/deferral behavior.

In the later Item 1+2 combination, groups are built independently inside each
read window and exact raw-resource aliases reconcile duplicate overlap groups.
Any end-to-start link across a true break remains disabled unless Item 3 is also
enabled.

## Failure and determinism criteria

Item 2 fails correctness if:

- a source hypothesis is retriangulated or its XYZ/residual/source slots change;
- a detection is selected into two groups at one frame;
- more than three groups are active at a frame;
- a group below frozen support/consistency thresholds is selected;
- a pixel-squared residual or square root of a refractive residual is used;
- candidate order changes the selected set;
- an infeasible optimization is silently replaced by greedy selection; or
- tail, future identity, or female-GT information leaks into grouping.

An infeasible or timed-out solve is diagnosed and rerun only after a documented
correction. It is never counted as a completed negative result.

## Benchmark plan and command templates

Build and hash the frozen hypothesis table once per cohort, then reuse it for
both modes. Replace placeholders only after the designated implementation owner
defines the entry point.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> group `
  --hypotheses <FROZEN_HYPOTHESES> --tracklets <TRACKLET_CACHE> `
  --cross-view-grouping framewise --max-active-groups 3 `
  --seed 20260717 --output-dir <OUT_DIR>\framewise

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> group `
  --hypotheses <FROZEN_HYPOTHESES> --tracklets <TRACKLET_CACHE> `
  --cross-view-grouping multiframe --min-support-frames 3 `
  --min-valid-fraction 0.95 --residual-mad-scale 1.4826 `
  --robust-sigma-multiplier 3 --max-active-groups 3 `
  --max-ray-residual-mm2 <BASELINE_VALUE_OR_NONE> `
  --objective-tolerance 1e-9 --seed 20260717 `
  --output-dir <OUT_DIR>\multiframe
```

The completed report must save exact commands for unit/synthetic tests,
numerical-equivalence windows, four reviewed windows, four frozen controls, and
any justified full-recording candidate. Framewise and multiframe runs must
record an identical frozen-hypothesis content hash before their results are
compared.

## Required measurements

Not yet measured:

- frozen hypothesis row count, file size, and content hash;
- candidate/eligible/selected group counts and support-length distribution;
- residual median/MAD/robust-upper distributions in `mm^2`;
- duplicate-hypothesis, insufficient-support, residual-gate, conflict, and
  capacity rejection counts;
- raw-resource conflict rate and synthetic-view split;
- cross-view group precision/recall on synthetic truth and available conflict
  proxies on real data;
- fragment count and length distribution;
- wall time, process-tree CPU utilization, peak RSS, and output size;
- direct/interpolated and all-three coverage;
- jump p95/p99/max and rates above configured gates;
- eligible/mapped/strict female-GT counts, fixed female accuracy, observed
  female-ID changes, paired cohorts, and reviewed-event outcomes;
- deterministic rerun hashes, numerical stability, failures, and timeouts.

## Advantages

- Reuses evidence across frames instead of trusting isolated low residuals.
- Robust median/MAD limits the influence of a small number of localization
  outliers without deleting their provenance.
- Exact resource constraints prevent double assignment by construction.
- Tracklet-key aggregation can reduce a large per-frame hypothesis table to a
  smaller, inspectable fragment graph.
- The framewise comparator and multiframe method share the same frozen
  triangulations and optimizer family.

## Disadvantages and risks

- A persistent reflection or systematic calibration error can satisfy all
  multiframe thresholds and look more convincing than a transient true fish.
- Conservative 2D tracklet breaks can reduce support below three frames and
  discard valid short evidence.
- Set packing can be computationally expensive if candidate tracklets are
  overly fragmented or long-lived reflection groups proliferate.
- Coverage-first lexicographic selection may retain a geometrically weak but
  eligible long group; the frozen residual gate and real quality metrics must
  expose this.
- The method forms 3D fragments but does not solve identity across all-view
  breaks.
- Female GT cannot reveal male-to-male grouping errors.

## Retain criteria

Retain Item 2 for higher-order and full-recording candidates only if:

1. all numerical, resource-exclusivity, capacity, units, and determinism tests
   pass;
2. synthetic cross-view group precision improves over framewise grouping,
   including transient-reflection and crossing cases, without hiding the
   persistent-reflection failure case;
3. direct all-three coverage and jump-gate metrics are non-inferior on paired
   real cohorts;
4. fragment count/length and conflict evidence show greater temporal coherence
   rather than merely lower output coverage;
5. no new reviewed female swap or increase in observed female-ID changes occurs;
6. any claimed gain appears on frozen controls or justified held-out evidence,
   not only the four reviewed events; and
7. runtime and peak RAM fit bounded-window full-recording execution.

Reject it if it violates resource exclusivity, corrupts frozen geometry, is
nondeterministic, or improves apparent continuity by dropping substantial valid
coverage. Defer it if quality is indistinguishable and the group-selection cost
is material.
