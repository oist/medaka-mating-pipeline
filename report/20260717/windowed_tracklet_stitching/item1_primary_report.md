# Item 1 primary method report: overlapping temporal windows

## Status

This is the primary method specification, written before implementation results
were available. No runtime, memory, coverage, or tracking-quality result is
claimed here. All placeholders and pending measurements must be replaced only
from saved artifacts produced under the frozen `benchmark_protocol.md`.

## Claim under test

Bounded, independently executable windows with deterministic overlap
reconciliation can reproduce the fragment resources and associations of the
same method run as one bounded invocation, while bounding peak memory, enabling
safe parallel execution, and allowing exact checkpoint/resume.

Item 1 may remove fragments that exist only because the same observations were
processed in two adjacent windows. It may not invent an end-to-start biological
link across a real detection break; that is Item 3. It may not use tail evidence,
change a cross-view grouping cost, change a solver decision, or defer an
identity decision. Female GT is used only after reconstruction. It identifies
only the female and cannot detect male-to-male swaps.

## Existing behavior and reusable code

- `scripts/reconstruction_3d/reconstruct_from_slp.py:174` currently owns the
  recording-wide call graph and retains recording-wide dictionaries. It has no
  window checkpoint/reconciliation layer.
- `scripts/reconstruction_3d/triangulate_extra_candidates.py:171-217` already
  evaluates refractive hypotheses in bounded frame batches. This numerical
  batching can be reused inside a window, but it does not define independent
  temporal windows or boundary reconciliation.
- `scripts/reconstruction_3d/multiview_tracklet_tracking.py:287-306` provides a
  simple union-find pattern, and `build_constrained_tracklet_graph` at `:309`
  demonstrates conflict-aware deterministic merging. Window aliases require a
  separate resource-based graph.
- `scripts/reconstruction_3d/cross_frame_tracking.py:168` and the
  `last_xyz_by_identity` state in
  `scripts/reconstruction_3d/triangulate_extra_candidates.py:168,264-275,363-367`
  are ordered state. They cannot be run independently in parallel and called
  equivalent without an explicit boundary-state protocol.

The optimized candidate cache specified by `benchmark_protocol.md` is a
prerequisite and receives no Item 1 scientific credit.

## Frozen window layout

For an inclusive cohort `[F0, F1]`, core length `L=10,000`, and context
`O=250`, window `i` has:

```text
core_start_i = F0 + i * L
core_end_i   = min(F1, core_start_i + L - 1)
read_start_i = max(F0, core_start_i - O)
read_end_i   = min(F1, core_end_i + O)
```

The final short core is retained. Event and small runs may use a shorter core,
but retain 250 context frames on each available side. Adjacent full windows
therefore share 500 read frames. Core intervals are disjoint and cover every
input frame exactly once.

Only the core owner publishes a final per-frame observation. Context rows exist
only to form window-local fragments and overlap evidence. This prevents both
duplicate output frames and averaging of independently reconstructed XYZ.

## Canonical raw-resource contract

Every cached detection has a globally stable resource key:

```text
(frame, view_code, reconstruction_candidate_slot)
```

The cache content hash and source-instance mapping make that key unambiguous.
Every selected 3D observation stores its two or three sorted raw-resource keys,
direct/synthetic-view state, XYZ, refractive residual in `mm^2`, and its
window-local fragment ID. A fragment's `raw_resource_hash` is the hash of its
sorted resource rows, not of its local ID.

Preflight invariants are:

1. one raw detection resource is used by at most one selected fish at a frame;
2. no more than three selected fish are active at a frame;
3. every context/core observation is inside its recorded read interval;
4. core intervals have neither gaps nor overlap;
5. all local fragment IDs are namespaced by window ID; and
6. calibration, cache, configuration, and code hashes match across windows.

A preflight failure aborts reconciliation and is not scored as a tracking
result.

## Exact overlap-reconciliation algorithm

### 1. Build window-local fragments

Each worker receives only its read interval and runs the frozen scientific
association mode. Item 1 does not alter that mode. It emits fragment rows for
the whole read interval and marks each row as `left_context`, `core`, or
`right_context`.

### 2. Form alias evidence

For adjacent windows `i` and `i+1`, use their exact shared read interval. For
every left fragment `u` and right fragment `v`, aggregate frames where their
selected observations use the same complete raw-resource tuple. Let:

```text
n_same(u,v)       = number of shared frames with identical resource tuples
n_compared(u,v)   = number of shared frames where both fragments are observed
consistency(u,v)  = n_same / max(n_compared, 1)
xyz_delta(u,v)    = maximum absolute XYZ difference on n_same frames
```

An alias edge is eligible only when:

```text
n_same >= 3
consistency >= 0.95
xyz_delta <= atol + rtol * max(abs(XYZ_left), abs(XYZ_right))
atol = 1e-9 mm
rtol = 1e-9
```

The XYZ check detects corrupted or non-equivalent checkpoints; identical raw
resources under the same refractive calibration should not need a looser gate.
Residuals are compared under the same `atol=1e-9`, `rtol=1e-9` rule but do not
create an alias by themselves.

The alias evidence table records support, denominator, consistency, maximum XYZ
and residual difference, eligibility, and rejection reason. No proximity-only,
velocity-only, tail, appearance, or biological-ID fallback is permitted.

### 3. Select compatible aliases

Sort eligible edges by:

```text
(-n_same, -consistency, xyz_delta,
 left_window_id, left_fragment_id, right_fragment_id)
```

Merge with union-find only if the proposed component contains no pair of
fragments that use different fish resources at the same `(frame, view)`. This
all-different check allows a continuous fragment in one window to canonicalize
two artificial, temporally disjoint pieces in the other window, but forbids a
many-fish merge.

This is alias canonicalization, not trajectory association. Fragments on
opposite sides of a genuine all-view break share no observation resource and
remain distinct for Item 3 or Item 5.

### 4. Assign canonical IDs and publish cores

For every alias component, sort all member resource keys and define the
canonical fragment ID from the lexicographically smallest resource key plus a
content hash. Unaliased fragments use the same rule. Local IDs and worker
completion order never influence canonical IDs.

Publish only core-owned observation rows, replacing local fragment IDs with
canonical IDs. Sort the final table by
`(frame, canonical_fragment_id, view_code, candidate_slot)`. Window provenance
and alias evidence remain available as separate tables.

### 5. Ordered state after merge

Any method requiring persistent trajectory state runs once, after ordered core
publication, on the compact canonical fragment/observation tables. Item 1 does
not independently run or reconcile `apply_cross_frame_tracking` states. A
future stateful method may define its own checkpoint state, but it must be
tested separately and cannot be assumed equivalent.

## Checkpoint and resume contract

Each window writes into a temporary directory and atomically publishes
`complete.json` only after all content hashes have been verified. Required
checkpoint metadata are:

```text
schema_version, window_id, core_start, core_end, read_start, read_end,
candidate_cache_hashes, calibration_hash, normalized_config_hash,
code_version_or_dirty_tree_hash, seed, scientific_mode,
wall_time_seconds, peak_rss_bytes, output_files_and_hashes
```

Expected data files are compact window-local observations, fragments, resource
assignments, and an overlap-signature table. A checkpoint is reusable only when
every identity field and content hash matches. Missing `complete.json`, a hash
mismatch, a changed source mtime/size/hash, or a changed normalized
configuration forces that window to rerun. A resumed merge must be byte-equal
to an uninterrupted merge.

## Component interface

The eventual implementation should expose pure planning and reconciliation
interfaces equivalent to:

```python
plan_windows(
    start_frame: int,
    end_frame: int,
    *,
    core_frames: int = 10_000,
    context_frames: int = 250,
) -> tuple[WindowSpec, ...]

reconcile_window_outputs(
    checkpoints,
    *,
    min_alias_support_frames: int = 3,
    min_alias_consistency: float = 0.95,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    seed: int = 20260717,
) -> ReconciledOutput
```

Worker count is execution metadata, not an algorithm input. Outputs must be
identical for worker counts 1, 2, 4, 8, and 16.

## Required deterministic tests

### Planning and ownership

- Empty/invalid intervals are rejected.
- Recordings shorter than one core produce one window.
- Exact multiples and a short final core cover every frame once.
- First/last context clipping is correct.
- Core length changes do not change canonical output on equivalence fixtures.

### Reconciliation

- A continuous fragment crossing the nominal core boundary is one canonical
  fragment after exact-resource aliasing.
- A left-window fragment split into two disjoint local pieces but continuous in
  the right window is canonicalized without duplicating resources.
- Two fish close at the seam cannot be merged when their same-frame resources
  differ.
- A simultaneous all-view break across a boundary remains two fragment sets;
  Item 1 must not manufacture a biological bridge.
- A crossing centered exactly on the boundary gives the same normalized
  fragment partition as the single bounded invocation or is reported as an
  explicit unresolved seam.
- Missing mouth rows never reappear through overlap reconciliation. Missing
  tails are preserved as missing and are not consulted.
- Reflection/extra-candidate resources cannot be assigned twice.
- Equal-support alias edges resolve by the frozen lexicographic order.
- Input checkpoint order and worker completion order do not alter bytes.

### Checkpointing

- Interrupted temporary output is ignored and rerun.
- One deliberately corrupted file invalidates only its window and affected
  adjacent seam reconciliation.
- Changed configuration/cache/calibration/code hash invalidates the checkpoint.
- A mixed cached/fresh resume is byte-equal to an all-fresh run.

## Equivalence and failure criteria

On numerical-equivalence windows, compare Item 1 with one invocation of the
same scientific method. Require:

- exact core frame set and candidate-resource use;
- exact normalized fragment partition and selected source slots;
- XYZ/residual equality at `atol=1e-9`, `rtol=1e-9`;
- exact direct/interpolated flags after the common fill stage;
- zero duplicate resource assignments and at most three active fish;
- identical fixed-seed rerun hashes.

Item 1 fails correctness if it loses a core frame, duplicates a resource,
merges an all-different conflict, silently forces an unsupported seam alias,
changes output with worker order, or accepts stale/corrupt checkpoints. A seam
without sufficient exact overlap evidence is recorded as unresolved; it is not
forced and must be included in unresolved-interval metrics.

## Benchmark plan and command templates

Replace placeholders only after the designated implementation owner defines
the entry point. Both modes must receive identical cache, calibration,
scientific configuration, and frame cohorts.

```powershell
.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> run `
  --experiment 20251225_K_1_2_FCMC --sleap-round round7_260704 `
  --paths-config configs\paths.local.toml --frames <START>-<END> `
  --window-mode single --seed 20260717 `
  --output-dir <OUT_DIR>\single

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> run `
  --experiment 20251225_K_1_2_FCMC --sleap-round round7_260704 `
  --paths-config configs\paths.local.toml --frames <START>-<END> `
  --window-mode overlap --window-core-frames 10000 `
  --window-context-frames 250 --workers <1|2|4|8|16> `
  --seed 20260717 --output-dir <OUT_DIR>\overlap

.venv\Scripts\python.exe -m <WINDOWED_TRACKLET_MODULE> reconcile `
  --checkpoint-root <OUT_DIR>\overlap\checkpoints `
  --min-alias-support-frames 3 --min-alias-consistency 0.95 `
  --atol 1e-9 --rtol 1e-9 --seed 20260717 `
  --output-dir <OUT_DIR>\overlap\reconciled
```

The completed report must save exact commands for unit/synthetic cases,
equivalence windows, all reviewed and frozen-control windows, worker-count
benchmarks, resume tests, and any justified full-recording run.

## Required measurements

Not yet measured:

- number and size of planned/completed/reused/rerun checkpoints;
- per-window wall time, peak RSS, CPU utilization, and output size;
- merge time and memory;
- alias candidate/eligible/accepted/rejected counts by reason;
- unresolved seam count and duration;
- normalized single-versus-window fragment equivalence;
- worker-count speedup and efficiency, including cold/warm cache labels;
- output hashes and resume equivalence;
- direct/interpolated coverage, jump metrics, and gate violations;
- eligible/mapped/strict female-GT counts, fixed female accuracy, observed
  female-ID changes, common cohorts, and reviewed-event outcomes.

## Advantages

- Bounds live Python/array state and makes full-recording memory predictable.
- Checkpoints provide safe failure recovery and exact provenance.
- Pure numerical windows can use the available physical CPU cores.
- Exact raw-resource aliases are interpretable and do not depend on arbitrary
  local identity numbers.
- The strict isolation rule prevents window infrastructure from being credited
  for biological fragment linking.

## Disadvantages and risks

- Context duplicates computation and I/O; too many workers may contend on the
  candidate cache or saturate memory bandwidth.
- A 250-frame context is longer than the 100-frame link gate but does not
  guarantee evidence through a long all-view occlusion.
- Exact-resource reconciliation intentionally abstains when adjacent local
  methods choose different detections. A proximity fallback would confound
  Item 1 with Item 3.
- Chained aliases can propagate an incorrect local grouping unless the
  all-different conflict check is complete.
- Stateful association remains sequential after the parallel geometry stage.
- Female GT cannot expose male-to-male boundary swaps.

## Retain criteria

Retain Item 1 for integrated/full-recording processing only if:

1. all planning, exclusivity, alias, checkpoint, and determinism tests pass;
2. normalized outputs are equivalent to the single invocation on deterministic
   equivalence windows, including a seam-centered crossing;
3. no new reviewed female swap, observed female-ID change, jump-gate violation,
   or direct-coverage regression is attributable to a seam;
4. unresolved seams are explicitly represented and not silently relabelled;
5. resume output is byte-equivalent to an uninterrupted run;
6. peak RSS is materially bounded relative to the single invocation; and
7. the selected worker count improves useful wall time without unacceptable
   cache contention or overlap overhead.

Reject it if it violates raw-resource exclusivity, is worker-order dependent,
or manufactures unsupported links. Defer scientific use if it is correct but
provides no practical full-recording memory, recovery, or runtime benefit.
