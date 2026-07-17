# Independent verification of pairs (3,5)--(5,6) and higher-order bundles

Date: 2026-07-17

## Scope and overall verdict

I independently audited the current rows for pairs `(3,5)`, `(3,6)`, `(4,5)`,
`(4,6)`, and `(5,6)` in `pairwise_combination_results.csv`, and every row in
`higher_order_bundle_results.csv`. I read the cited graph, solver, direct-output,
configuration, and comparison artifacts. I recomputed file and aggregate hashes,
objectives, path-cover feasibility, output equality, ambiguity counts, runtime
formulae, and female-GT denominators. I did not rerun geometry, edit reusable
code, or alter any existing report.

The saved artifact arithmetic is sound: every cited solver objective recomputes
exactly, every solve-only graph hash matches the current three graph files, and
all reported row counts and wall times match their JSON summaries. The important
qualification is experimental rather than arithmetical:

- `(3,5)` shows that global, beam, and sequential solvers find the same optimum
  on one short feasible Item 3 graph; it does not establish a general pairwise
  quality interaction.
- `(3,6)` genuinely produces deferred-decision records, but only one of two
  episodes remains unresolved and it merely blanks six terminal identity cells.
- `(4,5)` and `(4,6)` are exact-output-neutral because the 310-frame tail graph
  has **zero tail-eligible edges**. They do not test an active tail tie-breaker.
- `(5,6)` is not a scientific deferred-decision test. The `global_dag` branch
  always returns an empty `unresolved_intervals` tuple, irrespective of
  `identity_policy`; “deferred” is therefore a no-op in this combination.
- The corrected higher-order artifacts genuinely exercise Item 1: the
  1,000-frame run has two windows and 59 accepted boundary aliases, while the
  310-frame graph has four windows and 18 accepted aliases; both have zero
  unresolved seams. Item 4 still changes no selected decision, and Item 6 is
  still bypassed by the global branch.
- The full Item 1 candidate genuinely uses 35 overlapping windows. Its
  scientific rejection versus the historical strict-mouth output is supported
  by independently recomputed female-GT, coverage, and jump metrics.

## Independent artifact checks

For solve-only artifacts, I SHA-256 hashed
`fragment_observations.parquet`, `fragments.parquet`, and
`candidate_link_edges.parquet`, then reproduced the aggregate hash using the
serialization recorded by `windowed_tracklet_solve.py`. For end-to-end artifacts,
I reproduced the compact `stable_hash` convention used by the source manifest.
All component and aggregate hashes match.

| Artifact | Frames | Verified graph hash | Recomputed objective | Direct rows | Direct Parquet SHA-256 | Wall time |
|---|---:|---|---:|---:|---|---:|
| `item5_same_graph/global` | 72000--72309 | `28306989...b1cb8d9` | 44.37129530963059 | 930 | `022b40c7...1fc4c35` | 3.090575 s |
| `item6_same_graph/deferred` | 72000--72309 | `28306989...b1cb8d9` | 44.37129530963059 | 930 | `b6a94c8b...39d0c4` | 0.215722 s |
| `pair_item4_item5_global` | 72000--72309 | `5ffb634c...e0da04` | 44.37129530963059 | 930 | `022b40c7...1fc4c35` | 0.233571 s |
| `pair_item4_item6_deferred` | 72000--72309 | `5ffb634c...e0da04` | 44.37129530963059 | 930 | `b6a94c8b...39d0c4` | 0.225609 s |
| `pair_item5_item6_global_deferred` | 72000--72309 | `28306989...b1cb8d9` | 44.37129530963059 | 930 | `022b40c7...1fc4c35` | 0.230345 s |
| `pair_item1_item4_overlap_32264_33263` | 32264--33263 | `779824ad...ad3c9` | 1677.260984288198 | 2,819 | `647a8927...72f1d` | 19.835181 s |
| `bundle_overlap_item1_3_4_graph_72000_72309` | 72000--72309 | `488a7bb8...af4733` | 62.37129530963059 | 930 | `937d3a0f...d72dd4` | 7.945997 s |
| `bundle_overlap_item1_3_4_5_global_72000_72309` | 72000--72309 | `2e2b8635...44123e` | 44.37129530963059 | 930 | `e842929e...640526` | 0.241471 s |
| `bundle_overlap_item1_3_4_5_6_global_deferred_72000_72309` | 72000--72309 | `2e2b8635...44123e` | 44.37129530963059 | 930 | `e842929e...640526` | 0.212601 s |
| `full_optimized_baseline_v2` | 0--348599 | `e2e0226e...3e5f0` | not applicable | 1,030,486 | `e11df10a...3f44d` | 575.554832 s |

Every 310-frame selected set has edge-cost sum `44.37129530963059`, 156 links,
159 path rows, and exactly three paths. The corrected overlap graph's restart
summary adds three 6.0 birth/death charges and therefore reports
`62.37129530963059`; the separate exact-three global solves correctly report
the edge sum alone. Their direct outputs have zero resource conflicts and zero
frames with more than three active identities.

For the 1,000-frame restart result, the 378 selected-edge costs sum to
`999.2609842881981`. Adding 113 segment birth/death charges of 6.0 gives
`1677.260984288198` exactly. An independent maximum-cardinality bipartite
matching on the 491-fragment DAG gives a maximum matching of 483 and hence a
minimum path cover of **8**. An exact three-path cover is therefore
topologically infeasible; this conclusion does not rely on the solver's error
message. The cost-optimal restart solution nevertheless chooses 113 paths,
because its objective trades link cost against restart cost. Those are restart
segments, not eight paths and not three biological identities.

## Pair-by-pair findings

### Pair (3,5): explicit fragment links plus global solver

**Artifact verification: PASS. Classification: PARTIAL / local redundancy
only.**

On the same `28306989...` graph, sequential Hungarian, beam-32, and global DAG
all select the same 156 edge IDs, the same three paths, objective
`44.37129530963059`, and byte-identical direct output (`022b40c7...`). The
objective/output difference of zero and “redundant optimum” wording are correct
for this graph.

The runtime comparison is a single sample: sequential 0.223005 s, beam-32
0.221155 s, and global 3.090575 s. Global is 13.86x slower than sequential in
that run. However, later nominally identical global solves complete in about
0.23 s, showing large one-time initialization/cache variance. With no repeats,
cold/warm designation, or variability, this is not a robust time-interaction
classification. Item 3 is also prerequisite graph plumbing here; there is no
comparable no-Item-3 graph on which to estimate a factorial interaction.

There are zero female-GT annotations in frames 72000--72309, so no female
accuracy, stability, or reviewed-event advantage can be credited.

### Pair (3,6): explicit fragment links plus deferred decisions

**Artifact verification: PASS. Classification: behavioral difference present;
quality unverified.**

Immediate and deferred beam runs use the same graph and choose identical paths,
edges, and objective. Deferred records two ambiguity episodes:

1. frames 72052--72055, resolved by margin with two hypotheses and a recorded
   winning margin of 1.1918498354; and
2. frames 72308--72309, unresolved because the recording slice ends, with two
   hypotheses retained.

Thus “two intervals” means two recorded episodes, not two final unresolved
intervals. Only the second remains unresolved. The direct comparison preserves
all frame, fragment, XYZ, and camera-slot values exactly. It changes six rows
(three fish times two frames) from resolved IDs to blank IDs; the deferred
output has 924 resolved and six unresolved rows. The immediate and deferred
runtimes are 0.207877 and 0.215722 s, a 0.007844 s or 3.77% increase, but the
deferred summary contains no resource samples, so only wall time is usable.

The table's “deferred intervals appear; quality unverified” conclusion is
accurate. There is no GT in this slice, and abstention alone is not evidence of
better biological identity.

### Pair (4,5): conditional tail plus global solver

**Artifact verification: PASS. Classification: neutral because Item 4 is not
active.**

The tail and non-tail source graphs have identical 159 fragments, identical
4,333 directed edge IDs, and identical numerical costs on all 4,333 edges. The
tail graph differs in audit-state labels, but has zero `tail_eligible=true`
edges and zero nonzero tail costs. The global selected edge IDs, paths,
objective, and direct output are exactly equal to the no-tail global result.

“Same objective/output” is correct, but “redundant” should be read narrowly as
**no tail decision surface in this slice**, not evidence that an active tail
tie-breaker adds nothing to global optimization. The reported 0.233571 s is one
solve-only sample with one resource sample. It is only 4.7% above the 0.223005 s
sequential sample, while another global sample on the numerically same graph
costs 3.090575 s; these unrepeated solver times do not support a robust speed
interaction.

There are zero female-GT annotations in the slice.

### Pair (4,6): conditional tail plus deferred decisions

**Artifact verification: PASS. Classification: neutral tail / same deferral
artifact.**

The two ambiguity records, 924/6 resolved/unresolved split, selected edge IDs,
paths, objective, and direct output are exactly equal to
`item6_same_graph/deferred`. The only selected-edge-table difference is tail
audit state; selected tail-cost sum is zero. The claim “same deferral behavior”
is exact for this artifact.

This tests deferral on a tail-enabled configuration but not on a tail-eligible
decision. It therefore cannot establish interaction between the two intended
algorithms. Runtime is a single 0.225609 s sample with one resource sample.
There is no female GT in the slice.

### Pair (5,6): global solver plus deferred decisions

**Artifact verification: PASS for the no-op output. Experimental claim: FAIL as
a deferred-decision combination test.**

The artifact is byte-identical to `item5_same_graph/global` for direct output,
selected-edge output, trajectory paths, and the empty ambiguity table. The
summary sets `method=global_dag` and `identity_policy=deferred`, but code
inspection shows that `solve_fragment_identities()` assigns
`unresolved_intervals=()` unconditionally in the global branch. Consequently,
`apply_fragment_identities(..., identity_policy="deferred")` has nothing to
defer. Zero unresolved intervals are guaranteed by this implementation path,
not demonstrated by a comparison of retained identity hypotheses.

The quantitative columns in the current CSV are arithmetically correct for
their inputs:

| Quantity | Recomputed value |
|---|---:|
| T0 / Ti / Tj / Tij | 0.223005 / 3.090575 / 0.215722 / 0.230345 s |
| Si / Sj / Sij | 0.072156 / 1.033761 / 0.968132 |
| delta Ti / delta Tj / delta Tij | -2.867571 / 0.007283 / -0.007341 s |
| Additivity deviation | 2.852947 s |
| Multiplicativity deviation | 0.893539 |

Those numbers do not support a scientific redundancy classification. The
global-only and global-plus-deferred branches perform the same effective solve
yet differ by 13.42x in measured wall time (3.090575 versus 0.230345 s). That is
startup/cache noise in unrepeated runs, not an Item 6 speed gain. The most
accurate classification is **Item 6 unexercised/no-op under the current global
branch**. There is no female GT in the slice.

## Higher-order bundle findings

| Bundle row | Mechanical verification | Independent classification |
|---|---|---|
| `1+3+4`, 32264--33263 | PASS: two overlap windows, 437 alias candidates, 59 accepted aliases, zero unresolved seams, 19.835181 s, objective 1677.260984288198, 2,819 direct rows, zero deferred intervals, and 113 restart paths. Against `pair_item1_item3_overlap_32264_33263`, aliases, seams, observations, fragments, selected IDs, paths, objective, and direct output are byte/table exact. Six tail edges are eligible and nonzero-cost, but none is selected. | **Reject/defer is supported and Item 1 is now genuinely exercised.** Exact-three remains infeasible (minimum path cover 8), the output contains 113 segment labels, and Item 4 changes no decision. |
| `1+3+4+5`, 72000--72309 | PASS: the source graph has four overlap windows, 228 alias candidates, 18 accepted aliases, zero unresolved seams, 4,333 edges, and zero tail-eligible edges. Fresh graph build/restart solve is 7.945997 s; the separate global solve is 0.241471 s; their sum is the reported 8.187468 s. Global objective is 44.37129530963059 with three paths and 930 rows. | **Defer is supported for lack of demonstrated benefit.** Item 1 and Item 3 are active and verified; Item 4 has no decision surface. Selected IDs, paths, objective, identity, and geometry equal the prior single-window global result. The corrected Parquet is not byte-identical to that prior file because `window_index`, `local_fragment_id`, and `overlap_agree_frames` now carry genuine overlap provenance. |
| `1+3+4+5+6`, 72000--72309 | PASS: it uses the same verified four-window graph. Graph time 7.945997 s plus global/deferred solve time 0.212601 s equals the reported 8.158598 s. Objective, all 156 selected-edge rows, all 159 path rows, the empty ambiguity table, and all 930 direct rows are byte-identical to corrected `1+3+4+5`. | **Item 1 is genuinely exercised, but Item 6 remains an implementation no-op.** Global bypasses deferred hypothesis policy, so output equality does not establish that active deferral is scientifically redundant after a global identity solver. |
| `1 only (full candidate)`, 0--348599 | PASS: 35 overlap windows, 575.554832 s, 3.613 GB peak RSS, 1,030,486 direct rows and 1,045,800 filled rows. Objective is correctly blank because fragment linking is off. | **Completed and scientifically rejected is supported.** Unlike the short bundles, Item 1 overlap execution is active. Quality is materially worse than historical strict mouth on the available female and coverage evidence. |

The revised `runtime_scope` column is accurate. `1+3+4` and the full candidate
are fresh end-to-end runs; the middle rows explicitly sum one fresh overlap
graph build with a separate solve on that graph. The tail-on overlap run is
0.086456 s (0.43%) faster than its 19.921637 s tail-off comparator, a single-run
difference consistent with noise. The corrected global solve has one resource
sample and the global/deferred solve has none. No higher-order row supplies
repeated cold/warm timing, so no additive, multiplicative, synergistic, or
antagonistic runtime interaction is validated.

Zero `unresolved_intervals` also has limited meaning across these rows. The
1,000-frame result has 113 restart segments despite zero deferred intervals.
The full baseline has 44,389 unmapped direct rows and 986,097 resolved direct
rows despite zero deferred intervals. The field counts only policy-recorded
deferral episodes; it is not a general identity-completeness metric.

## Full-candidate scientific comparison

I independently recomputed the full-candidate QC from the filled CSV and the
female metrics from `annotation_assignments.csv`. They reproduce the cited
comparison exactly:

| Metric | Historical strict mouth | Optimized Item 1 candidate |
|---|---:|---:|
| Eligible confirmed-female annotations | 300 | 300 |
| Mapped female annotations | 298 | 283 |
| Best fixed female-ID accuracy | 0.523490 | 0.427562 |
| Observed female-ID changes | 11 | 81 |
| Strict two-view count | 268 | 241 |
| Strict best fixed accuracy | 0.518657 | 0.419087 |
| Strict observed changes | 11 | 62 |
| All-three finite frame share | 0.989776 | 0.968904 |
| All-three directly observed frame share | 0.981592 | 0.894286 |
| Jump p99 (mm/frame) | 2.201934 | 7.309646 |
| Jump maximum (mm/frame) | 7.999974 | 7.999788 |
| Jumps above 8 mm | 0 | 0 |
| Preserved/evaluable reviewed events | 3/4 | 3/3 |

The paired-common cohorts lead to the same conclusion rather than a denominator
artifact. On the 283 common-mapped annotations, historical/candidate best-fixed
accuracy is 0.530035/0.427562 with 11/81 observed changes. On the 241
common-strict annotations it is 0.526971/0.419087 with 11/62 changes.

The optimized full output itself has 1,027,529 valid adjacent jumps: median
0.343561, p95 1.485437, p99 7.309646, maximum 7.999788 mm/frame, and zero above
8 mm. Its finite-row share is 0.982615 and its finite, directly observed row
share is 0.942912. These values and the output SHA-256
`340078e8e2c6c09c8177fd717daf44a054faad116380f83f65bbbd8b34de5d0b`
for the filled CSV independently match the cited result.

## Female-GT limitation

The raw ledger has 301 rows, of which 300 are eligible confirmed-female
annotations. Frames 72000--72309 contain **zero** annotations. Frames
32264--33263 contain only one confirmed female annotation, which cannot measure
switches or support a bundle-quality conclusion. Only the full comparison has a
meaningful female denominator.

Most importantly, this ledger identifies the female only. Best-fixed female-ID
accuracy and observed female-ID changes measure consistency for that one fish;
they are not three-fish identity accuracy and cannot reveal male-to-male swaps.
The full candidate's regression is sufficient to reject it on available
evidence, but neither the pair artifacts nor the full female-only comparison can
validate male identity preservation or all-fish link precision.

## Final reconciled classifications

- `(3,5)`: **locally redundant optimum; runtime and general quality interaction
  unverified**.
- `(3,6)`: **deferred behavior present; one terminal unresolved episode; quality
  unverified**.
- `(4,5)`: **neutral because tail is inactive, not evidence against an active
  tail/global interaction**.
- `(4,6)`: **neutral tail with exactly reproduced deferral behavior; active
  interaction untested**.
- `(5,6)`: **implementation no-op / invalid deferred-combination experiment**;
  output equality is real, scientific redundancy is not established.
- `1+3+4`: **reject/defer supported**, with exact-three infeasibility and 113
  non-biological restart segments independently confirmed; Item 1 overlap is
  genuinely exercised, while Item 4 changes no selected decision.
- `1+3+4+5`: **defer supported for no demonstrated benefit**; Items 1, 3, and 5
  are exercised, while Item 4 has zero eligible edges in the cited graph.
- `1+3+4+5+6`: **overlap is exercised, but deferred identity is still a global
  branch no-op**; it is not evidence that active deferral is redundant.
- `1 only (full candidate)`: **engineering completion verified; scientific
  rejection supported** versus historical strict mouth, subject to the explicit
  female-only GT limitation.
