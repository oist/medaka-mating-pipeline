# Windowed 3D tracklet-stitching benchmark protocol

## Status and preregistration

This protocol was fixed on 2026-07-17 before inspecting results from any new
windowed-stitching implementation.  The four reviewed close-contact windows are
confirmatory event diagnostics, not tuning data.  All random choices use seed
`20260717`, and all frame intervals are inclusive.

The experiment is `20251225_K_1_2_FCMC`, SLEAP round `round7_260704`, with
three fish, up to six candidates per view, source rate 166 fps, strict finite
mouth anchors, fixed-interface refractive calibration, calibration-native XYZ
bounds 0--200 mm, 8 mm/frame motion gate, 80 mm hard cap, and a 100-frame
maximum linking gap.

## Cohorts

1. Unit and synthetic cases: deterministic crossings, simultaneous all-view
   breaks, missing mouths, missing tails, reflections/extra candidates, and
   mutually exclusive detection assignment.
2. Numerical-equivalence windows: small deterministic slices selected solely
   for implementation checks, never used to claim identity improvement.
3. Reviewed event windows: 71,000--74,100; 95,000--99,000;
   173,000--182,000; and 270,000--280,000.
4. Non-event controls sampled before new results with NumPy
   `default_rng(20260717)`, excluding the reviewed windows and a surrounding
   buffer: 32,264--35,364; 151,050--155,050; 21,840--30,840; and
   57,363--67,363.
5. The complete 0--348,599 recording, reserved for the optimized baseline and
   final justified candidates.

The available GUI ground truth is
`annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv`.  It identifies
only the female and therefore cannot independently detect male-to-male swaps.
All comparisons must repeat this limitation.

## Fixed execution policy

- Candidate cache: one HDF5 file per view, six reconstruction slots, strict
  finite mouth, optional tail retained, chunked along frame, source file
  size/mtime and cache hash recorded.
- Default processing core: 10,000 frames with 250-frame context overlap on
  each available side.  The overlap exceeds the 100-frame link gate.
- Small/event experiments may use shorter cores but retain 250-frame overlap.
- Parallel-window worker counts tested: 1, 2, 4, 8, and 16 when the cohort is
  large enough.  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  and `NUMEXPR_NUM_THREADS` are each fixed to one inside worker processes.
- CPU measurements use process-tree wall time, sampled CPU utilization, and
  peak RSS.  GPU measurements include transfer and synchronization time,
  sampled utilization, and peak allocated/reserved VRAM.
- Each timed configuration receives one cold-cache run when feasible and at
  least three warm-cache repetitions when timing variability exceeds 5%.
  Tables report median, min/max, and median absolute deviation.
- A failed run is diagnosed and rerun after a documented correction.  Failed
  or interrupted runs are never scored as completed results.
- Every window checkpoint records configuration, input cache metadata, seed,
  calibration provenance, frame interval, wall time, resource peaks, and an
  output content hash.  Complete matching checkpoints are reused.

## Optimized-baseline equivalence

The optimized cache/window foundation is compared with the existing strict
mouth, six-candidate refractive implementation on identical frames.  Checks:

- exact candidate count, slot order, mouth coordinates, and rejected
  tail-only instances;
- identical selected source slots and cardinality;
- XYZ and refractive mean-squared point-to-ray scores within explicit
  tolerances (`atol=1e-9`, `rtol=1e-9` for float64 compute; cache float32 input
  is separately characterized and must not alter assignments);
- no detection slot used by two fish in one frame;
- deterministic byte-equivalent tabular outputs under a fixed seed; and
- end-to-end identity/coverage equality after the same linker and fill policy.

If float32 cache coordinates change any assignment, the production cache is
promoted to float64 rather than relaxing the assignment-equivalence check.

## Switchable components and isolation

All runs share one configuration schema.  The optimized baseline disables all
experimental items except prerequisite cache plumbing.

1. Windowing: single bounded invocation versus independent overlapping windows
   with overlap reconciliation/checkpoints.  The scientific association mode
   is otherwise fixed.
2. Multiframe grouping: one-frame cross-view proposals versus the same
   proposals aggregated by overlapping per-view tracklets with robust median
   residual/MAD, minimum support, and all-different conflicts.
3. Explicit fragment links: no end-to-start links versus a directed edge set
   using endpoint position, constant-velocity prediction, gap, overlap
   consistency, refractive residual, and mouth confidence.
4. Conditional tail vectors: the exact same candidate edges scored with and
   without tail-vector evidence.  Tail cost is enabled only for mouth-ambiguous
   alternatives; missing tails are neutral and tail cannot veto a clear mouth
   match.
5. Solver: sequential Hungarian, short-beam Hungarian, and a global
   time-ordered DAG/state-flow solver receive the exact same graph and costs.
6. Deferral: greedy immediate commitment versus retained competing hypotheses
   on the exact same ambiguity episodes, resolved after separation.

Prerequisite plumbing is never credited as a scientific component.

## Individual and combination matrix

Each item is tested alone against the optimized baseline.  All 15 pairs are
initially considered.  A pair may be marked technically meaningless only when
one setting eliminates the decision surface of the other; the reason and
supporting counts must be recorded in `compatibility_matrix.csv`.  All other
pairs run on identical cohorts and inputs.  Higher-order bundles are justified
from individual and pair evidence before full-recording execution.

For runtime, with baseline `T0`, item runtime `Ti`, and pair runtime `Tij`, the
report calculates `Si=T0/Ti`, `Sij=T0/Tij`, `deltaTi=T0-Ti`, deviation of
`deltaTij` from `deltaTi+deltaTj`, and deviation of `Sij` from `Si*Sj`.
Runtime and tracking-quality interactions are classified independently as
synergistic, approximately independent, sub-additive, redundant, or
antagonistic using preregistered uncertainty intervals from repeated runs.

## Quality metrics

Every applicable result records eligible/mapped/strict female-GT counts, best
fixed female-ID accuracy, observed female-ID changes, paired common cohorts,
four reviewed-event outcomes, all-three observed coverage, direct versus
interpolated coverage, jump p95/p99/max and rates above gates, fragment count
and length distribution, unresolved interval count/duration, retained
hypothesis counts, refractive residual distribution in mm^2, cross-view link
precision/conflict proxies, runtime/resources/output size, numerical stability,
determinism, advantages, disadvantages, failure cases, and a retain/reject/defer
decision.

## Independent review

The primary implementation owner and independent verifier are different
agents.  Agents use the same frame lists, annotations, seed, cache-state label,
timeout policy, metrics, timing method, and output schema.  Separate findings
are saved before reconciliation.  Material disagreements trigger a neutral
rerun.  Only the designated implementation owner edits reusable pipeline code;
review agents write non-overlapping reports.

## Execution record and protocol amendments

The preregistered cohorts, seed, calibration, strict-mouth policy, six-slot
candidate cache, and quality metrics were not changed after results were seen.
The following execution details clarify how the frozen protocol was applied:

- The primary implementation timing slice was frames 32,264--33,263, a
  preregistered subset of control 1. Exact single-versus-overlap reconciliation
  used the full control-1 interval 32,264--35,364.
- Solver isolation used frames 72,000--72,309 because that immutable real graph
  has an exact three-path cover. The 1,000-frame Item 3 graph is explicitly
  reported as infeasible for the exact-three objective, not silently relaxed.
- The worker sweep was run on an earlier, hash-equivalent implementation
  revision before compact fragment summarization. The final-code single and
  overlap runs are the authoritative direct comparison; the sweep remains a
  hardware scaling diagnostic.
- Warm-cache repetition was curtailed when it would repeat a multi-minute or
  multi-hour result without changing a scientific decision. Exact output hashes,
  deterministic reruns, and the reason are reported instead. This is a declared
  deviation from the preferred repetition count, not an unreported median.
- End-to-end CUDA utilization and peak VRAM were not sampled. The isolated CUDA
  benchmark includes synchronized kernel and host-transfer time and exact CPU
  numerical comparisons; missing utilization/VRAM fields are marked unavailable.
- A fresh overlap run took 25.7308504 s; its checkpoint-resume rerun reused all
  seven windows and retained the same output hash. The saved resume summary
  supersedes the fresh summary's per-window `reused` fields, so the fresh time is
  preserved in the verifier and aggregate tables.

Failed and interrupted attempts are enumerated in `failed_runs.json` and are
never counted as successful runs.
