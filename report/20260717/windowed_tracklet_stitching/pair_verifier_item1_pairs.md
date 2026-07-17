# Independent verification of revised Item 1 pairwise rows

## Final verdict: interaction tests now exist; two timing labels need correction

The post-audit artifacts supersede the earlier conclusion that pairs (1,2)
through (1,6) were only prerequisite-compatibility tests. Each revised row now
has an overlap-enabled Item 1 control and a corresponding overlap-enabled pair
on the same frame cohort and configuration family. Keeping `tested=True` is
therefore defensible for an engineering interaction test.

The four-runtime values and all derived numbers in the first five rows of
`pairwise_combination_results.csv` are arithmetically correct. The scientific
comparisons are also reproducible from the saved outputs. Two interpretation
corrections remain:

1. Rows (1,3) and (1,4) say `sub-additive runtime cost`, but their combined
   runtimes are respectively 0.4016703 s and 0.3264540 s *above* the additive
   expectation. Their correct descriptive sign is slightly **super-additive
   runtime cost**, or approximately additive given the one-run timing design.
2. Row (1,2) is not uniformly `quality antagonistic`. It has fewer direct rows,
   fewer covered frames, more blank identity rows, and one unresolved seam, but
   it also has more all-three-direct frames. `mixed coverage; unresolved seam`
   is the most accurate quality label.

The timing classifications remain exploratory. There is one observation per
cell, several single-window controls were produced by earlier code revisions,
and the (1,5) result is dominated by a large difference in one-off global-solver
startup time. These limitations do not invalidate the saved scientific-output
equality/difference checks.

## Independent formula check

For each pair, the table uses:

```text
Si   = T0 / Ti
Sj   = T0 / Tj
Sij  = T0 / Tij

delta_i  = T0 - Ti
delta_j  = T0 - Tj
delta_ij = T0 - Tij

D_add   = delta_ij - (delta_i + delta_j)
        = Ti + Tj - T0 - Tij
        = T_additive_expected - Tij

D_speed = Sij - Si*Sj
```

Thus positive `D_add` means the pair completed faster than the additive-cost
expectation, while negative `D_add` means the pair cost more. I independently
loaded the raw `run_summary.json` and `solver_summary.json` values and
recalculated every term:

| Pair | T0 (s) | Ti (s) | Tj (s) | Tij (s) | Additive expected (s) | Tij - expected (s) | D_add (s) | Sij | D_speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1+2 | 18.7657260 | 17.3984953 | 58.0563704 | 51.4815091 | 56.6891397 | -5.2076306 | +5.2076306 | 0.3645139 | +0.0158804 |
| 1+3 | 6.7103637 | 10.0176087 | 16.2127216 | 19.9216369 | 19.5199666 | +0.4016703 | -0.4016703 | 0.3368380 | +0.0595876 |
| 1+4 | 6.7103637 | 10.0176087 | 16.2014817 | 19.8351807 | 19.5087267 | +0.3264540 | -0.3264540 | 0.3383062 | +0.0608634 |
| 1+5 | 23.7890178 | 30.0738111 | 26.8795929 | 30.3006360 | 33.1643862 | -2.8637502 | +2.8637502 | 0.7850996 | +0.0850290 |
| 1+6 | 23.7890178 | 30.0738111 | 24.0047394 | 30.2867017 | 30.2895327 | -0.0028310 | +0.0028310 | 0.7854608 | +0.0015484 |

The recalculated CSV discrepancies are zero to floating-point rounding
(maximum absolute difference about 3.1e-15). `Sij` is named an observed
combined speedup in the CSV, but all five values are below 1.0 and therefore
represent a slowdown relative to `T0`, not a net speed benefit.

For (1,5) and (1,6), the timing boundary is explicitly composite:

```text
T0  = single-window source-pipeline time
Ti  = overlap-window source-pipeline time
Tj  = T0 + single-source downstream-solver time
Tij = Ti + overlap-source downstream-solver time
```

This is internally consistent, but the downstream solves were separate
processes rather than stages timed inside one end-to-end invocation.

## Pair verdicts

### (1,2): overlapping windows + multiframe grouping

**Verdict: interaction test PASS; timing sign PASS at low confidence; scientific
result is mixed coverage with an antagonistic seam failure.**

The four cells use the frozen-hypothesis 1,000-frame cohort 32264--33263:

- `bench_frozen_framewise_32264_33263` (`T0`);
- `pair_item1_only_frozen_overlap_32264_33263` (`Ti`);
- `bench_frozen_multiframe_32264_33263` (`Tj`);
- `pair_item1_item2_overlap_32264_33263` (`Tij`).

The combined run is a real two-window overlap run with core size 500 and
context 250. Its 51.4815091 s wall time is 5.2076306 s below the additive
expectation of 56.6891397 s. `synergistic runtime` is therefore the correct
descriptive sign, but not a high-confidence causal performance result.

The scientific difference is real but mixed:

| Metric | Item 2 single | Item 1+2 overlap | Difference |
|---|---:|---:|---:|
| Direct rows | 840 | 783 | -57 |
| Frames with at least one direct row | 684 | 514 | -170 |
| All-three-direct frames | 3 | 73 | +70 |
| Blank `fish_id` rows | 120 | 216 | +96 |
| Unresolved seams | 0 | 1 | +1 |

The unresolved overlap covers frames 32514--32963. Its left and right windows
contain 118 and 538 rows, with only 40 identical-resource rows, 78 left-only
rows, and 498 right-only rows. `strict_resource_equivalent=False` and
`unresolved=True` are stored in `boundary_seams.parquet`. The filled-output
hashes also differ:

```text
Item 2 single: 58cd3853e05091c62829660e230942178fb86fef5c5bf0378c87f962b1c2dd45
Item 1+2:      55ef8bedcdd70b5e62973bc343e19a2d20372361c376e245838a635a59ac088a
```

Recommended quality wording:

> mixed coverage interaction: fewer total direct rows and covered frames,
> more all-three-direct frames, more blank identities, and one unresolved seam.

### (1,3): overlapping windows + explicit broken-fragment links

**Verdict: interaction test PASS; exact normalized scientific equality PASS;
runtime label sign FAIL.**

The combined artifact is a real two-window overlap run with mouth linking. It
has 2,819 direct rows, 491 fragments, 378 selected links, 113 restart paths, and
objective 1677.260984288198. Relative to the single-window Item 3 control:

- the selected-edge list, path list, and objective are exactly equal;
- the filled CSV is byte-identical, SHA-256
  `cbbc95567c2aacaf194691b9632806f9dc0eedffa1d6f5902f5999d5d85ab35e`;
- after sorting by `(frame, source_index)`, the direct tables differ only in
  `window_index`, `local_fragment_id`, and `overlap_agree_frames`.

This validates `scientific output independent` in the limited sense of exact
saved-output equality. It does not validate the 113 restart paths as biological
fish identities.

The timing description needs the opposite sign. `Tij=19.9216369 s` is
0.4016703 s greater than the additive expectation of 19.5199666 s. Replace
`sub-additive runtime cost` with:

> slightly super-additive runtime cost (+0.4017 s versus additive expectation);
> low confidence, one observation per cell.

### (1,4): overlapping windows + conditional tail continuity

**Verdict: interaction test PASS; redundant scientific output PASS; runtime
label sign FAIL.**

The combined artifact is a real two-window run with mouth linking plus
conditional tail cost. It has the same 2,819 rows, 491 fragments, 378 selected
links, 113 paths, and objective 1677.260984288198 as the Item 3 run. Relative to
the single-window Item 4 control, the selected edges, paths, objective, and
filled CSV are exactly equal. The direct-table differences are again confined
to the three window-provenance columns above.

The overlap-enabled Item 3 and Item 4 runs are also exactly equal in selected
edges, paths, objective, and final output. This verifies that Item 4 is
scientifically redundant on this cohort.

`Tij=19.8351807 s` is 0.3264540 s greater than the additive expectation of
19.5087267 s. Replace `sub-additive runtime cost` with:

> slightly super-additive runtime cost (+0.3265 s versus additive expectation);
> low confidence, one observation per cell.

### (1,5): overlapping windows + global trajectory solver

**Verdict: interaction test PASS; normalized scientific equality PASS;
descriptive sub-additivity PASS but startup-dominated.**

`smoke_v2_overlap` is a four-window overlap run over frames 72000--72309. It
has three resolved seams, 930 rows, and the same normalized filled output as
`smoke_v2_single310`. The downstream global solve uses that saved overlap graph.

The single-source and overlap-source graph hashes differ because window
provenance differs, but the solver results are scientifically identical:

- exactly the same three path lists and 156 selected-edge IDs;
- exactly the same objective, 44.37129530963059;
- zero ambiguity intervals and zero blank identity rows in both;
- after sorting, every direct-output column is equal except `window_index`,
  `local_fragment_id`, and `overlap_agree_frames`.

The byte-level direct Parquet hashes differ for that provenance reason. The
normalized-output equality claim is correct.

The composite `Tij=30.3006360 s` is 2.8637502 s below the additive expectation.
However, almost all of that apparent interaction comes from the global solver
taking 3.0905751 s on the single-source graph but only 0.2268249 s on the
overlap-source graph despite selecting the same optimum. With one run per graph,
this is consistent with process/HiGHS startup variance and must not be presented
as algorithmic Item 1+5 synergy.

Recommended timing wording:

> descriptively sub-additive by 2.8638 s, low confidence and dominated by
> one-off global-solver startup variance; no scientific-output change.

### (1,6): overlapping windows + deferred identity decisions

**Verdict: interaction test PASS; normalized scientific equality PASS;
approximately additive runtime PASS.**

The deferred solver on the overlap graph exactly reproduces the single-source
deferred result apart from the same three window-provenance columns:

- exactly equal path and selected-edge lists;
- objective 44.37129530963059 in both;
- exactly equal ambiguity-interval tables;
- two intervals, including one unresolved at recording end;
- six blank `fish_id` rows and 924 resolved identity rows in both.

The downstream solver times are also close: 0.2157216 s for the single-source
graph and 0.2128906 s for the overlap graph. The composite runtime is only
0.0028310 s below the additive expectation. `approximately additive runtime`
is supported descriptively, subject to the one-run timing limitation.

## Provenance and inference limits

The factor settings and cohorts form sensible 2x2 comparisons, but the runtime
cells are not all from one immutable code revision:

- For (1,2), both single-window cells record core/orchestrator hashes
  `f33ce5...` / `598fd6...`, while both overlap cells record `a49e82...` /
  `3e8e72...`.
- For (1,3) and (1,4), `T0`, `Ti`, and `Tij` use `a49e82...` / `3e8e72...`,
  while the Item 3/4 single controls use `689c3d...` / `15cce5...`.
- For (1,5) and (1,6), the single and overlap source runs record different core
  hashes (`3a7a20...` and `b24dbb...`), and the separate solver summaries do not
  embed a solver-code hash.

Consequently, the signs above are correct descriptions of these saved runs, but
they are not clean causal estimates of feature interaction. Multiple repetitions
under one revision would be required for a confirmatory performance claim.

The biological denominator is also insufficient. Frames 32264--33263 contain
one confirmed-female annotation but no strict two-view evaluable annotation;
frames 72000--72309 contain none. Female-only labels cannot detect male-to-male
swaps. Therefore `exact scientific equality` here means equality of the saved
pipeline outputs, not independently verified biological identity accuracy.

## Sharing recommendation

The earlier `not tested` verdict should be retired. The revised first five rows
are ready to share as overlap-enabled engineering interaction tests after these
wording changes:

- correct (1,3) and (1,4) to slightly super-additive/approximately additive
  runtime cost;
- describe (1,2) as mixed coverage with an unresolved seam rather than uniformly
  antagonistic quality;
- retain low-confidence/startup-variance language for all runtime interactions,
  especially (1,5);
- retain the distinction between exact pipeline-output equality and biological
  identity validation.
