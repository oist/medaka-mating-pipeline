# Independent verifier: pairwise rows involving Items 2 and 3

Date: 2026-07-17

Scope: rows `(2,3)`, `(2,4)`, `(2,5)`, `(2,6)`, and `(3,4)` in
`pairwise_combination_results.csv`. This was a read-only audit of the existing
JSON/Parquet artifacts and current summarization/solver code. No geometry was
rerun and no existing report was edited.

## Overall verdict

The recorded runtimes for `(2,3)`, `(2,4)`, and `(3,4)` and their arithmetic
baseline ratios are correct. The segment, direct-row, objective, and selected-
edge counts also reproduce. Several interaction labels overstate what the
artifacts establish:

- the low 840-row coverage is Item 2's existing main effect, not a demonstrated
  antagonistic interaction with Items 3 or 4;
- `(2,5)` is independently confirmed infeasible under exact-three on this graph,
  but its cited evidence directory is empty;
- `(2,6)` is not an observed end-to-end pair runtime and is scientifically
  confounded by changing the solver from sequential to beam; and
- “redundant” is supported as a quality/output statement for tail pairs, not as
  a statistically established runtime interaction.

No row has enough female ground truth to support a biological identity-quality
claim.

## Common runtime denominator

The script computes `observed_combined_speedup` as:

```text
bench_baseline_final wall time / pair wall time
```

The baseline wall time is `6.710363699996378 s`. This reproduces the populated
CSV values, but it is a baseline-relative runtime ratio, not a factorial
interaction or synergy estimate. The runs are single measurements without
repeat variance, and Item 2 changes direct coverage from 2,819 to 840 rows, so
runtime and scientific workload are not strictly exchangeable.

## Row (2,3): multiframe grouping + explicit fragment links

### Quantitative checks

| Claim | Verification |
|---|---|
| Runtime `62.23829099998693 s` | **PASS**, exact `run_summary.json` value. |
| Combined ratio `0.10781728727091475` | **PASS**, `6.710363699996378 / 62.23829099998693`. This is equivalently a 9.27495x slowdown versus the baseline. |
| Direct rows | **840**, identical to Item 2 alone and 29.80% of the 2,819-row framewise Item 3 run. |
| Fragments / candidate edges | **299 / 7,306**. |
| Selected edges / restart segments | **177 / 122**; `299 - 177 = 122`. |
| Objective | **1,185.5252262854242**, independently reproduced as selected edge costs plus `122 * (3 + 3)`. |
| Feasibility / unresolved | **Feasible**, status `optimal_boundary_state`, zero unresolved intervals, zero raw-resource conflicts, zero frames over three active paths. |

Relative to Item 2 alone (`58.05637039997964 s`), adding the Item 3 graph and
restart solve costs `4.181920600007288 s`, or 7.20%, in this one run.

### Interaction classification

**The CSV classification “antagonistic coverage; 122 segments” is only partly
supported.** The 122 restart segments are real and correctly reported. However,
the 840 direct rows are exactly the Item 2-alone count, so the coverage loss is
inherited from Item 2 rather than caused or worsened by the `(2,3)` interaction.
The links reduce 299 fragments to 122 smooth segments, but those are not three
validated biological identities. A more defensible quality label is:

> Item-2-limited coverage; Item 3 reduces fragmentation to 122 restart segments;
> biological link correctness unverified.

The `time_interaction` field should describe the observed 7.20% overhead versus
Item 2, not “antagonistic coverage.”

## Row (2,4): multiframe grouping + conditional tail continuity

### Quantitative checks

| Claim | Verification |
|---|---|
| Runtime `61.775754900008906 s` | **PASS**, exact summary value. |
| Combined ratio `0.10862455199224787` | **PASS**, exact baseline/runtime calculation; a 9.20602x baseline slowdown. |
| Direct rows / fragments / edges | **840 / 299 / 7,306**. |
| Selected edges / restart segments | **177 / 122**, exactly the same selected edge set as `(2,3)`. |
| Direct output | **Exactly equal** to `(2,3)` after loading both Parquets. |
| Item 3 edge fields | **Exactly equal**; only tail augmentation fields differ. |
| Objective | **1,185.526203502793**, exactly recomputed. It is `0.000977217369` higher than `(2,3)` because selected tail cost is nonzero. |
| Feasibility / unresolved | **Feasible**, zero unresolved intervals and invariant violations. |

Tail-on produced eight eligible 2D edges, all with nonzero tail cost. The state
counts were 6,834 `not_ambiguous`, 376 `tail_incomplete`, 88
`mouths_not_close`, and 8 `2d`. Tail altered no selected edge or direct row,
although it added `0.0009772173686428743` to the selected objective.

The pair run was `0.4625361 s` faster than `(2,3)` despite doing tail work. With
one run per mode, that 0.74% difference is ordinary run variability, not evidence
that tail improves runtime.

### Interaction classification

**“Redundant tail output” is supported if “output” means selected edges and
direct tracks; it is not true of the objective.** “Antagonistic coverage” is
again inherited entirely from Item 2. A corrected label is:

> Tail is decision/output-neutral on the Item 2 graph; objective differs by
> 0.000977; Item-2-limited coverage is unchanged.

No runtime interaction can be distinguished from noise with these single runs.

## Row (2,5): multiframe grouping + global exact-three solver

### Independent feasibility check

The cited `pair_item2_item5_global` directory exists but contains **no files**:
there is no solver summary, failure JSON, stderr log, runtime, objective, or
direct output. Thus `tested=True` and the cited immutable evidence are not
auditable as written.

Nevertheless, the claimed exact-three infeasibility is independently confirmed
from the immutable `(2,3)` fragment/edge graph without running geometry:

| Graph property | Recomputed value |
|---|---:|
| Fragments | 299 |
| Directed feasible edges | 7,306 |
| Maximum bipartite matching | 289 |
| Minimum DAG path cover | `299 - 289 = 10` |

Because even the minimum path cover needs 10 paths, a cover with exactly three
paths is impossible. Runtime, direct rows, and objective are correctly left NA.

### Interaction classification

The conclusion should be narrowed to:

> Exact-three is infeasible on the cited Item 2+3 graph (minimum path cover 10).

This does not prove that Items 2 and 5 are generally incompatible under a
restart-capable solver, different thresholds, or another graph. A small
machine-readable failure artifact should be added before describing this row as
immutably tested.

## Row (2,6): multiframe grouping + deferred identity decisions

### Quantitative checks

The CSV runtime `62.76716199997463 s` is arithmetically correct, but it is:

```text
62.23829099998693 s  completed Item 2+3 geometry + sequential solve
+0.5288709999877028 s second beam/deferred solver-only run
```

It is therefore **not an observed end-to-end Item 2+6 runtime**; it double-counts
the first sequential solve instead of replacing it with the deferred solve. The
CSV leaves the ratio blank because the generator only fills ratios when the
evidence string contains `32264`. If the same baseline formula were applied, the
arithmetic ratio would be `0.10690882758087884`, but it would still describe the
composite timing above rather than a clean pair run.

The solver-only artifact is internally consistent:

| Measure | Recomputed value |
|---|---:|
| Feasible/status | true / `optimal_boundary_state` |
| Objective | 993.6266169120354 |
| Selected edge-cost sum | 483.6266169120353 |
| Selected edges / restart paths | 214 / 85 |
| Objective reconstruction | `483.6266169120353 + 85 * 6` = 993.6266169120353 |
| Direct geometry rows | 840; geometry columns exactly equal to `(2,3)` |
| Identity-resolved / unresolved rows | 96 / 744 |
| Resource or active-path invariant failures | 0 |

The interval summary contains nine records: five resolved intervals and four
unresolved intervals. Three unresolved intervals ended at the horizon with
`frame_end - frame_start = 250`, which is **251 frames under inclusive frame
counting**. A fourth unresolved interval ended at recording end. Consequently,
“three 250-frame unresolved intervals” is incomplete and ambiguous about
duration.

### Confounding and interaction classification

This row changes both identity policy and solver method:

- `(2,3)` uses `sequential_hungarian`, selecting 177 edges and 122 paths;
- `(2,6)` uses beam width 32 plus deferred identity, selecting 214 edges and 85
  paths.

Only 143 selected edges are common; 34 are unique to sequential and 71 are
unique to beam/deferred. The lower objective and fewer segments therefore
cannot be attributed to deferral alone. No same-graph Item-2 immediate-beam
artifact is provided.

A defensible classification is:

> Extensive unresolved identity output (744/840 rows), with three horizon
> expirations plus one recording-end unresolved interval; quality and runtime
> interaction are confounded by a solver change.

Calling this simply “antagonistic” is a value judgment unsupported by biological
truth, although the loss of resolved identity coverage is quantitatively clear.

## Row (3,4): explicit links + conditional tails

### Quantitative checks

| Claim | Verification |
|---|---|
| Runtime `16.20148170000175 s` | **PASS**. |
| Combined ratio `0.41418209915921783` | **PASS**, exact baseline/runtime arithmetic. |
| Direct rows / fragments / candidate edges | **2,819 / 491 / 38,592**. |
| Selected edges / restart segments | **378 / 113**. |
| Objective | **1,677.260984288198**, exactly equal to Item 3 tail-off. |
| Feasibility / unresolved | **Feasible**, zero unresolved intervals and invariant violations. |
| Selected/direct equality | **PASS**, selected edge IDs and the full direct Parquet are exactly equal to Item 3. |

There were six tail-eligible 2D edges, but none was selected. Thus the
“identical selected edges/output” statement and redundant quality classification
are supported.

The tail-on run was only `0.0112399 s` (0.069%) faster than tail-off. With one
measurement per mode, the correct runtime conclusion is **no measurable
incremental interaction**, not a runtime benefit. The reported 0.414 ratio is
again relative to the 6.71-second baseline, not the incremental effect of Item 4
on Item 3.

## Immutable-artifact caveats

- `(2,3)` and `(2,4)` have identical fragment-observation and fragment hashes;
  their edge files differ only because Item 4 tail augmentation is stored in
  the edge table.
- The `(2,6)` solver summary cites the exact same three file hashes as `(2,3)`.
  Its displayed aggregate graph hash differs from the stitcher manifest because
  the two modules serialize the hash dictionary with different JSON separators;
  this is a hash-procedure inconsistency, not different graph content.
- `(2,5)` has no persisted failure artifact at all.

## Female-GT limitation

The female annotation file has 301 rows recording-wide, but only **one** row in
frames 32264--33263: frame 32872, `confirmed`, `F`. It is linked in only one
camera (`geometry_id_consistency=one_view`) and therefore supplies zero strict
two-view observations. The existing quality summary likewise reports one
eligible female annotation, zero strict-two-view mappings, and zero evaluable
mixed events.

None of the five pair rows contains a pair-specific mapping from its restart
segments or unresolved identities to that female annotation. These artifacts
can support runtime, geometric coverage, feasibility, fragmentation, and
unresolved-policy statements only. They cannot support female accuracy, full
three-fish identity accuracy, swap reduction, or biological quality interaction
claims.
