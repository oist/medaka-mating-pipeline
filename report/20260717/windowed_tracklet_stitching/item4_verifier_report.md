# Item 4 final independent verifier report: conditional tail vectors

Date: 2026-07-17

## Verdict

**Neutrality/safety verdict: PASS for the final v2 artifact.** Tail-on preserved
all 38,592 Item 3 edge identities and every mouth-only feature exactly, preserved
the selected 378 edges, objective, 113 restart segments, all direct rows, and all
jump metrics. Missing competing tails were complete-case neutral in both code
inspection and an independent probe.

**Full specification verdict: FAIL.** The persisted “frozen graph” hashes differ
between tail-off and tail-on because tail augmentation is stored inside the
candidate edge file. The implementation also considers only the single
runner-up assignment rather than every assignment within the 0.5 margin, and
its 2D fallback omits the required five-sample body-length baseline and
0.5x--2.0x length filter.

**Scientific acceptance verdict: FAIL / defer.** Six 2D-tail edges were eligible
but none was selected, so Item 4 changed no decision and provides no measured
identity benefit in this window.

I did not author the implementation and changed no reusable code.

## Reproduced tests and targeted probe

The shared focused suite passed **16 tests in 1.64 s** with a unique
workspace-local basetemp; the exact command and environment note are recorded in
the Item 3 verifier.

I independently repeated the former finite-tail versus missing-tail failure
case. Tail-off/on edge IDs and all Item 3 base columns were equal. Both disputed
edges became `tail_incomplete`, `tail_eligible=false`, `tail_cost=0`, and
`total_cost=mouth_cost`. The missing alternative therefore received no relative
advantage. This confirms the previous blocking defect is fixed.

## Exact off/on artifact comparison

Compared:

- `bench_item3_mouth_v2_32264_33263` (`tail=off`); and
- `bench_item4_tail_v2_32264_33263` (`tail=conditional`).

| Claim | Result | Evidence |
|---|---|---|
| Fragment observations equal | **PASS** | Identical SHA-256 `943dcd...ebe8` and equal Parquet tables. |
| Fragment tables equal | **PASS** | Identical SHA-256 `0877c5...7cd0`; 491 rows each. |
| Edge identities/pairs equal | **PASS** | 38,592 rows each; `(edge_id, source_id, target_id)` exactly equal. |
| Every Item 3 edge feature/term equal | **PASS** | All non-tail columns compare exactly, including `mouth_cost`. |
| Full edge Parquets equal | **FAIL, expected augmentation difference** | Only `tail_eligible`, `tail_state`, `tail_angle_deg`, `tail_cost`, and `total_cost` differ. |
| Frozen candidate-graph hash equal | **FAIL** | Tail-off hash `793c547c...d4a7`; tail-on hash `11f8a848...ceb1`. The manifest hashes the augmented edge file rather than a tail-independent candidate graph. |
| Selected edges equal | **PASS** | Exact same set of 378 edge IDs. |
| Objective equal | **PASS** | Both are `1677.260984288198`. |
| Restart segments equal | **PASS** | 113 paths and 113 direct `fish_id` segment labels in both. |
| Direct output equal | **PASS** | Full `direct_3d_observations.parquet` tables compare exactly; 2,819 rows each. |
| Jump metrics equal | **PASS** | Both have 2,706 transitions; rate p95 1.27211, p99 5.42100, max 7.96565 mm/frame; zero above 8. |

The hash difference violates the primary report's explicit unchanged-graph-hash
contract even though the underlying Item 3 graph is demonstrably equal when the
five augmentation columns are excluded. The correct artifact model is a shared
Item 3 graph hash plus a separate tail-augmentation hash/table.

## Tail eligibility and state audit

| Edge state in Item 4 | Count |
|---|---:|
| `not_ambiguous` | 38,211 |
| `tail_incomplete` | 362 |
| `mouths_not_close` | 13 |
| eligible `2d` | 6 |
| eligible `3d` | 0 |

All six eligible edges had nonzero tail cost. Their maximum individual cost was
0.0625595, below the fixed `1/12`, and their total over the whole edge table was
0.261185. They occurred as two alternatives at target starts 32381, 32458, and
32974. None of the six was selected. Among the 378 selected edges, states were
316 `not_ambiguous`, 60 `tail_incomplete`, and 2 `mouths_not_close`; selected
tail-cost sum was exactly zero.

Observation-level tail validation is active and inspectable:

| Observation tail state | Count |
|---|---:|
| `valid_3d` | 753 |
| `tail_residual_over_10_mm2` | 1,386 |
| `recovered_mouth_anchor` | 516 |
| `tail_outside_xyz_bounds` | 156 |
| `fewer_than_two_tail_views` | 8 |

## Claim-by-claim method verification

| Claim | Result | Evidence |
|---|---|---|
| Same mouth-selected candidate slots | **PASS** | Tail pixels and 2D vectors are read only from each selected mouth slot. |
| Strict 3D tail validity | **PASS** | Requires direct mouth, at least two views, finite triangulation, XYZ 0--200, residual <=10 mm^2, and length 10--60 mm. |
| Endpoint tail summary | **PASS** | Uses at most 15 samples in 30 frames, at least three unit vectors, median/renormalization, and pre-normalization norm >=0.5. |
| Missing-tail neutrality | **PASS** | All disputed edges are zeroed when neither complete 3D nor at least two common 2D summaries exist; independent probe passed. |
| No 2D/3D mixing in a set | **PASS** | A set uses 3D only if all disputed edges have it; otherwise it uses common 2D views or disables tail. |
| Mouth ambiguity uses complete assignments | **PARTIAL** | Best and exact second-best rectangular assignments are solved, including 6.0 birth/death dummies, but further assignments within the 0.5 margin are not enumerated. |
| Close-contact test | **PARTIAL** | Uses disputed source endpoints and target starts at <=12 mm; it does not search direct mouths throughout the specified +/-15-frame boundary support. |
| 2D fallback validity | **FAIL** | It requires three direction samples via `_direction_summary`, but not a five-sample per-fragment/view body-length baseline or the 0.5x--2.0x vector-length filter. The six eligible real edges therefore do not satisfy the full frozen fallback contract. |
| Fixed tail objective/cap | **PASS in code and artifact** | Cost is `(1/12)*clip(theta/90,0,1)` and the builder asserts <=0.25 for checked best/runner assignments. Observed maximum edge cost was 0.0625595. |
| Clear mouth cannot become ambiguous | **PASS** | Eligibility is computed from mouth-only assignment costs before adding any tail term. |
| Deterministic off/on direct result | **PASS for this run** | Selected edges and every output row are exactly equal. No worker/boundary rerun was supplied. |

## Remaining specification mismatches

1. Persist a tail-independent candidate graph and hash, then write tail
   eligibility/cost as a separately hashed augmentation. The current manifest's
   graph hash necessarily changes when tail is enabled.
2. Enumerate the best plus **every** distinct boundary assignment within 0.5,
   and apply complete-case neutrality to the union of every disputed edge. A
   third within-margin assignment can currently be omitted.
3. Implement the 2D fallback length baseline and ratio filter exactly: at least
   five direct samples per fragment/view, and accept each vector only between
   0.5x and 2.0x that baseline before forming endpoint direction summaries.
4. Evaluate the <=12 mm mouth separation within the full +/-15-frame direct
   boundary support rather than only at the two endpoint coordinates.
5. Serialize the complete ambiguity-set table and missing-reason counts. Edge
   states are inspectable, but assignment costs/margins and the full set
   membership are not persisted.

## Scientific acceptance

Item 4 passes the most important safety result in this artifact: enabling tail
did not alter a clear or incomplete case, selected edge, objective, segment, XYZ
row, or jump. It does **not** pass the retention criterion because no selected
decision changed, no synthetic/real truth improvement is demonstrated, and the
only eligible real evidence used the noncompliant 2D fallback. After correcting
the remaining mismatches, rerun the required ambiguous-tail synthetics and the
four reviewed windows. Any female-only evaluation must remain explicitly
female-only because the supplied annotation is not full three-fish identity
ground truth.
