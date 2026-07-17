# Item 5 final independent verifier report: shared-graph trajectory solvers

## Scientific acceptance

**DEFER / not scientifically accepted for production.**

The final implementation passes the central engineering contract on synthetic
graphs and one exact-three-feasible real slice: sequential, plain beam-32, and
an exact HiGHS path-cover MILP consume one immutable graph, enforce three paths,
and recompute the same serialized-edge objective. The real slice is only 310
frames, all three methods return exactly the same solution, there is no female
GT in the slice, and the 1,000-frame graph is exact-three infeasible. The
required global lexicographic tie rule and state telemetry also still fail.

No reusable code was edited by this verifier.

## Final snapshot and reproduced tests

```text
windowed_tracklet_core.py           A49E827C0A7000A7C5E55C02CDA8A7166F9EC4E47A23366B9AA161C9BD05FEA2
windowed_tracklet_stitching.py      3E8E72CD5B143EE22D554D3D53D0C432DFE41AB9CB60F658D427CC51E4255D1F
windowed_tracklet_solve.py          CC450882611D526BC03232724423E4AAED2500B3F4D512F502E9BE59982F0DA0
windowed_tracklet_finalize.py       CC45C8BBC7284EE90C452C3E96537175DD26E494AC43D2F8BFE452C4BE54CBE1
test_windowed_tracklet_stitching.py 0B43FF2FF4B4339C41596DBEF057BB016E9321E3EB10386B7BBA31FC49D5359D
```

Focused command:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --basetemp report\20260717\windowed_tracklet_stitching\item56_verify_tmp_final_reaudit_7f1d9a `
  tests\test_windowed_tracklet_stitching.py `
  tests\test_global_fragment_stitching.py
```

Result: **28 passed in 1.81 s** (16 final windowed tests and 12 legacy-global
tests). The new tests cover delayed disambiguation, plain beam versus immediate
policy, exact horizon expiry, missing-cover rejection, row-order stability,
restart segmentation, unresolved identity blanking, and resource/identity
occupancy invariants.

## Real same-graph scope

The five Item 5/6 solve-only artifacts all read
`smoke_v2_single310/{fragment_observations,fragments,candidate_link_edges}.parquet`.
This is a **310-frame** real slice, frames **72,000--72,309**, inside but much
shorter than the frozen reviewed event 71,000--74,100.

Independent file hashing reproduced:

```text
fragment_observations.parquet c016fe1b7e9bd305dc74ddb94b9892ad957a5eef7cb290367276bd1b7d3e27f0
fragments.parquet             fce2e1a8afb053d64cf3935a6c9a09cafff9f21201dd8910ecc77cdb238586c4
candidate_link_edges.parquet  0add20c94c26686b13a485b95a77b53e89000c367b0206030c3d6de666522f99
aggregate graph hash          2830698919f091dc6018768e84178e757de471cbd2ccccc5d6683da25b1cb8d9
```

Inventory: 930 direct observations, 310 unique frames, 159 fragments, and
4,333 candidate edges. Every saved summary contains the exact hashes above.
Independent current-code reruns reproduced each summary's objective, selected
edge IDs, paths, ambiguity intervals, and direct table.

## Claim-by-claim verification

| Claim | Result | Independent evidence |
|---|---|---|
| All solvers consume one immutable graph and costs | **PASS** | All three Item 5 summaries have graph hash `283069...`; file hashes independently match. `windowed_tracklet_solve.py` hashes before reading and never rebuilds geometry. |
| Every accepted fragment is assigned exactly once to exactly three paths | **PASS** | Each result has 3 paths, 159 unique path rows, and 156 selected edges (`159 - 3`). Missing-edge synthetic graph is rejected as infeasible. |
| Raw-resource and active-identity exclusivity | **PASS** | Saved invariant counts are zero; tests reject duplicate raw slots and duplicate `(frame, fish_id)` occupancy. |
| Common objective is the sum of selected serialized edge costs | **PASS** | All methods report `44.37129530963059`; independent sums of each `selected_link_edges.total_cost` equal it exactly. |
| Sequential is an immediate one-state boundary comparator | **PASS (functional)** | `hypotheses_peak=1`, 10 states pruned, and the delayed synthetic takes the local 51.0 solution. The implementation enumerates at most three-path assignments rather than calling SciPy Hungarian, but its tested semantics match the comparator. |
| Beam-32 is isolated from Item 6 | **PASS** | `identity_policy=solver_only` retains a plain beam independently of deferral. Real peak is 2; delayed synthetic gives 1.1 versus sequential 51.0. |
| Global solver is exact rather than a capped beam | **PASS** | `global_dag` uses binary predecessor/birth and successor/death equations with exactly three births/deaths in SciPy/HiGHS. Status is HiGHS optimal; missing exact cover raises. |
| Later evidence can beat greedy | **PASS on synthetic** | Independent delayed fixture: sequential 51.0, beam-32 1.1, global 1.1, with beam/global selecting the known later-winning paths. |
| Global objective is no worse than sequential/beam on the real identical graph | **PASS, equality only** | All are 44.37129530963059 and select byte-identical path/edge lists. |
| Geometry/resources are solver-invariant | **PASS** | All Item 5 direct Parquet outputs have SHA-256 `022b40...` and 930 finite mapped rows. |
| Input-order determinism | **PASS on covered cases** | Synthetic fragment/edge shuffles preserve global selected edges and paths; current in-memory reruns reproduce the saved results. |
| Required lexicographically smallest edge-ID tie solution | **FAIL** | The secondary MILP minimizes the sum of edge ranks, not the edge-ID tuple. On an equal-cost 3x3 matching, it returned `e_AZ,e_BY,e_CX`; the required lexicographic solution is `e_AX,e_BY,e_CZ`. |
| Explored/pruned state telemetry | **FAIL** | Boundary solver summaries report `explored_states=0` although expansions occurred; the internal counter is not returned. Margin-pruned counts are also incomplete. |
| Real identity/quality improvement | **FAIL / not demonstrated** | All three paths, edges, identities, coverage, and direct hashes are identical on the only real graph. There are zero female-GT rows in frames 72,000--72,309. |
| Required frozen cohorts/full recording | **FAIL / pending** | No complete reviewed window, four full controls, or full-recording exact-three result exists. |

## Saved Item 5 measurements

| Method | Objective | Paths / edges | Peak hypotheses | Wall time | Peak RSS |
|---|---:|---:|---:|---:|---:|
| sequential | 44.371295 | 3 / 156 | 1 | 0.223 s | 187.3 MB |
| beam-32 | 44.371295 | 3 / 156 | 2 | 0.221 s | 186.9 MB |
| global MILP | 44.371295 | 3 / 156 | 0 (not a beam) | 3.091 s | 194.4 MB |

These are single short runs, not the preregistered cold/warm repeated timing
protocol. Global is about 14 times slower on this slice. One resource sample
per run is insufficient for robust CPU/RSS conclusions.

## Explicit 1,000-frame infeasibility and restart contract

I independently ran exact global path cover on the real graph at frames
**32,264--33,263** (1,000 inclusive frames), with **491 fragments and 38,592
candidate edges**. HiGHS returned:

```text
RuntimeError: exact three-path cover infeasible
HiGHS Status 8: model_status is Infeasible
```

The failure was fast (1.05 s) and correctly did not relax gates or force a
link. This is an important **PASS** for honest infeasibility, but a **FAIL for
real-scope applicability** of the exact-three Item 5 comparison.

The explicit restart contract works: sequential `path_cover_mode=restarts` on
the same graph produced 113 `SEG_*` paths, 378 selected edges, covered all 491
fragments (`378 + 113 = 491`), mapped all 2,819 direct rows, and passed resource
invariants. Its objective is 1,677.260984 including frozen birth/death penalties.
This is a transparent fallback, not the primary exact-three objective; the
global solver deliberately rejects restart mode. Restart results therefore
cannot be credited as an Item 5 global-path win or biological continuity across
breaks.

## Female-GT denominators and scientific decision

The 310-frame slice contains **zero** rows from
`annotations/fish_sex_gt/.../female_sex_gt.csv`. No eligible/mapped/strict or
paired-common female denominator can be computed, and female-only GT would not
detect male-to-male swaps even if present. No reviewed-event outcome or jump
comparison accompanies the same-graph artifacts.

Engineering acceptance is **conditional PASS** for exact path-cover mechanics,
same-graph isolation, and honest restart/infeasibility behavior. Scientific
acceptance is **DEFER** until:

1. the exact edge-ID lexicographic tie rule and explored-state telemetry pass;
2. an exact-three-feasible frozen cohort long enough to exercise real ambiguity
   is justified, or the preregistered question is explicitly reformulated for
   segment restarts;
3. full reviewed/control cohorts and repeated resource measurements are saved;
4. fixed female-GT denominators, paired common cohorts, reviewed-event checks,
   and the male-male limitation are reported; and
5. a real quality advantage is demonstrated rather than objective equality.
