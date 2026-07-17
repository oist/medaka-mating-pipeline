# Item 1 independent verifier report: overlapping temporal windows

## Verdict

**Execution correctness: PASS for the final implementation. Controlled Item 1
single/overlap equivalence and checkpoint reuse: PASS. Scientific acceptance of
the integrated full-recording candidate: FAIL / defer.**

The final code now implements the missing engineering component: bounded
window-local execution, content-addressed checkpoints, exact-resource seam
comparison, fragment aliasing with all-different protection, deterministic core
ownership, and parallel worker execution. The current full-recording artifact
completed 35 windows with all 34 seams exactly equivalent and no resource or
three-active violations.

Exact-current controlled artifacts now compare one 3,101-frame single window
with seven overlapping windows at eight workers. Both contain 9,122 observations
and 1,858 fragments and produce the same normalized filled CSV SHA-256. A second
overlap invocation reused all seven checkpoint summaries and preserved that
hash. Deliberate corruption and mixed fresh/reused resume behavior remain
implemented but untested. The broader saved worker-count sweep is deterministic
and useful, but its recorded code hashes predate the final revision.

Scientific acceptance is separate. The final full windowed baseline has much
lower direct coverage and many more observed female-ID changes than the
historical strict-mouth baseline. Exact seam equality makes a seam error an
unlikely explanation for that regression, but the integrated candidate still
fails the frozen retain criteria until the regression is attributed and fixed.

Female GT identifies only the female. It cannot detect male-to-male swaps, so
the scientific assessment below is an optimistic lower bound on identity
errors.

## Scope and evidence inspected

- `scripts/reconstruction_3d/candidate_cache.py`
- `scripts/reconstruction_3d/windowed_tracklet_core.py`
- `scripts/reconstruction_3d/windowed_tracklet_stitching.py`
- `tests/test_candidate_cache.py`
- `tests/test_windowed_tracklet_stitching.py`
- `worker_bench/` worker-count sweep
- `item1_final_single_32264_35364/`
- `item1_final_overlap8_32264_35364/`
- `bench_frozen_framewise_32264_33263/`
- `bench_frozen_multiframe_32264_33263/`
- `full_optimized_baseline_v2/`
- `final_full_quality_comparison/`
- `hardware_benchmark.json`

No reusable code was edited during this verification.

## Focused test reproduction

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
uv run pytest -q tests\test_candidate_cache.py `
  tests\test_windowed_tracklet_stitching.py `
  --basetemp .pytest_tmp_item12_final_verifier
```

Result:

```text
..................                                                       [100%]
18 passed in 1.74s
```

The Item 1-specific coverage includes exact core planning, context validation,
exact-resource fragment aliasing, preservation of a deliberate resource break,
and generic resource/capacity invariants. It does not include a checkpoint
corruption/resume unit test or an end-to-end current-revision worker-order test;
the saved exact-current all-reused checkpoint rerun supplies the resume evidence
reported below.

## Claim-by-claim result

| Claim | Result | Independent evidence |
|---|---:|---|
| Strict-mouth six-slot cache is reusable and bounded by frame slices | PASS | Cache build stores float64 mouth/tail/confidence/source-instance arrays; `load_candidate_window` reads only the requested inclusive HDF5 slice. Focused strict-mouth/slot-order test passes. |
| Cache policy and provenance invalidate incompatible work | PASS, with caveat | `cache_matches_source` checks completion, source size/mtime, view, candidate count, frame count, anchor and filter policy. Cache SHA-256 is saved in resolved configuration; cache, calibration, and code hashes feed the checkpoint configuration hash. The source SLP itself is not byte-hashed, so equal-size/equal-mtime source replacement remains a theoretical gap. |
| Each worker executes the complete window-local geometry/tracklet stage | PASS | `_run_geometry_window` loads only its context interval, creates geometry or frozen hypotheses, constructs per-view candidate tracklets/groups, builds local fragments, validates invariants, and hashes every checkpoint output. `_run_windows` uses a spawn-safe process pool and returns window-index order. |
| Adjacent windows are reconciled only by exact raw resources | PASS | `reconcile_window_fragments` requires identical frame/three-slot tuples plus XYZ/residual agreement for alias evidence; it enforces support at least 3, consistency at least 0.95, union-find all-different compatibility, canonical content IDs, and exact core ownership. Focused alias/break test passes. |
| Seam mismatches are explicit rather than silently merged | PASS | `boundary_seams.parquet` records left/right-only rows, XYZ/residual deltas, strict equivalence and `unresolved`. The final full run reports 34/34 strict-equivalent seams, zero left/right-only rows, zero coordinate/residual delta, and zero unresolved seams. |
| Checkpoint reuse preserves output | PASS on final revision | A rerun of `item1_final_overlap8_32264_35364` reports `reused=true` for all seven windows and retains the identical normalized filled CSV hash. `_checkpoint_valid` requires schema/config hash and recomputes every output SHA-256. Deliberate corruption and a mixed fresh/reused run remain untested. |
| Output is independent of worker count | PASS for saved sweep; exact-current evidence is single versus overlap-8 | Worker counts 1, 2, 4, 8, 8-repeat and 16 all produced the same graph, direct CSV and filled CSV hashes on the earlier revision. Exact-current single and overlap-8 produce the same normalized filled CSV. The broad worker sweep still predates the final revision. |
| Single invocation and overlap mode are scientifically/numerically equivalent | PASS on final revision | `item1_final_single_32264_35364` and `item1_final_overlap8_32264_35364` both have 9,122 observations and 1,858 fragments and produce filled CSV SHA-256 `a6e85f881143bbf20651baf4a46e4a3bcaf83a7925bd00bfa8fabed574e48bf9`. The overlap run has 208 accepted aliases, six exact seams and zero unresolved seams. |
| Full recording completes within practical resources | PASS | `full_optimized_baseline_v2` matches the current three implementation hashes, covers frames 0..348599 in 35 windows, finishes in 575.55 s, peaks at 3,613,310,976 bytes RSS, and records mean process-tree CPU 72.12%. |
| Integrated candidate preserves baseline scientific quality | FAIL | The final quality comparison shows large direct-coverage and female-ID stability regressions, detailed below. |

## Window reconciliation evidence

### Exact-current controlled equivalence and resume

The controlled artifacts cover frames 32264..35364 and record the current core
hash `a49e827c0a7000a7c5e55c02cda8a7166f9ec4e47a23366b9aa161c9bd05fea2`
and orchestrator hash
`3e8e72cd5b143ee22d554d3d53d0c432dfe41ab9cb60f658d427cc51e4255d1f`.

| Metric | Single | Overlap, 8 workers |
|---|---:|---:|
| Windows | 1 | 7 |
| Observations | 9,122 | 9,122 |
| Fragments | 1,858 | 1,858 |
| Accepted aliases | 0 | 208 |
| Unresolved seams | 0 | 0 |
| Fresh wall time (s) | 22.6572 | 25.7309 |
| Filled CSV SHA-256 | `a6e85f…e48bf9` | `a6e85f…e48bf9` |

All six overlap seams are strict-resource equivalent, with zero left/right-only
rows and maximum XYZ delta 0 mm. The graph/direct-parquet hashes differ because
they include window/fragment provenance; the normalized biological filled CSV
is byte-identical. On this short cohort overlap-8 is 13.6% slower than single,
so the result proves equivalence, not a performance gain.

The overlap directory was then invoked again from its checkpoints. Every one of
the seven window summaries reports `reused=true`, and the normalized filled CSV
hash remains unchanged. The externally measured resume-command wall time was
21.8 s (the rewritten summary records 19.30 s internally); this is dominated by
deterministic reconciliation and final output generation, not window geometry.
The original fresh overlap summary recorded 25.7308504 s before the resume
rewrote it.

### Full-recording seam evidence

The exact-current `full_optimized_baseline_v2` artifact records:

- 1,030,486 reconciled observations and 92,426 compact fragments;
- 3,575 seam alias candidates, 618 eligible/accepted aliases;
- 34 seams, all 34 strict-resource equivalent;
- zero left-only or right-only overlap rows;
- maximum XYZ delta 0 mm and residual delta 0 mm²;
- zero unresolved seams, resource conflicts, or frames over three active fish;
- 1,045,800 filled output rows.

Together with the controlled byte-equivalent single/overlap output, this is
strong evidence that the final window infrastructure did not introduce a
boundary mismatch in the full run.

## Worker scaling and determinism

The saved 3,101-frame, seven-window sweep used 500-frame cores and 250-frame
context. All runs produced graph hash
`59bafa6e451ce4f10ef629e7478da45fa313d86fb351ae38f7c8ac700c1acb94`,
direct CSV hash
`756c978d527149d688ea258f1b15aa42c6b05bffffd687605cc368dcb6e9a83e`,
and filled CSV hash
`a6e85f881143bbf20651baf4a46e4a3bcaf83a7925bd00bfa8fabed574e48bf9`.

| Workers | Wall time (s) | Speedup vs 1 | Peak RSS (GiB) | Mean process-tree CPU (%) |
|---:|---:|---:|---:|---:|
| 1 | 58.07 | 1.00x | 0.25 | 97.72 |
| 2 | 44.44 | 1.31x | 0.52 | 61.78 |
| 4 | 39.54 | 1.47x | 0.88 | 71.52 |
| 8 | 36.04 | 1.61x | 1.38 | 76.03 |
| 8 repeat | 40.42 | 1.44x | 1.38 | 72.21 |
| 16 | 37.58 | 1.55x | 1.38 | 76.31 |

The sweep supports worker-order determinism for its recorded revision and shows
that useful scaling saturates around 8 workers. The 8-worker repeat is 12.2%
slower than the first 8-worker run, so warm/cold storage variability is material.
Because the saved code hashes predate the final revision, 8 workers is a
reasonable operational candidate, not a final-revision scaling proof. The exact
current single/overlap-8 comparison establishes deterministic equivalence but is
too short for useful parallel speedup: fresh overlap-8 took 25.73 s versus 22.66
s for single.

## Hardware microbenchmark interpretation

`hardware_benchmark.json` is internally consistent and records seven repeats on
16 physical / 32 logical CPUs, 128 GiB RAM and an RTX 5070 Ti:

- CPU endpoint scoring: 12.53 million pairs/s;
- CPU compact solve kernel: 3.85 million solves/s;
- CUDA endpoint kernel: 2.02 billion pairs/s;
- CUDA endpoint including transfer: 116.96 million pairs/s;
- CUDA compact solve kernel: 252.45 million/s;
- maximum CPU/CUDA differences: 1.78e-15 (endpoint) and 2.13e-14 (solve).

This is a synthetic array-kernel benchmark sourced from the older
`full_optimized_baseline` fragment file. The production window pipeline remains
CPU/SciPy code, so these numbers do not demonstrate end-to-end GPU acceleration
or change the retain decision.

## Scientific acceptance

The final full quality comparison uses the current-code
`optimized_windowed_baseline_v2` and the historical strict-mouth output.

| Metric | Historical strict mouth | Optimized windowed baseline v2 |
|---|---:|---:|
| Observed non-interpolated row share | 0.98999 | 0.94291 |
| All-three-observed frame share | 0.98159 | 0.89429 |
| Jump rate above 8 mm | 0 | 0 |
| Female GT mapped / eligible | 298 / 300 | 283 / 300 |
| Mapped best-fixed female accuracy | 0.52349 | 0.42756 |
| Mapped observed female-ID changes | 11 | 81 |
| Strict two-view GT count | 268 | 241 |
| Strict observed female-ID changes | 11 | 62 |
| Evaluable / preserved reviewed mixed events | 4 / 3 | 3 / 3 |

On the paired common mapped cohort of 283 annotations, observed female-ID
changes are 11 versus 81; on the paired common strict cohort of 241, they are 11
versus 62. The optimized candidate keeps the 8 mm physical jump guard but loses
8.73 percentage points of all-three direct coverage and substantially worsens
female identity stability.

These regressions cannot automatically be blamed on Item 1: every full-run seam
was exactly equivalent, and the comparison also changes the optimized array
baseline/downstream association path. They nevertheless prevent scientific
acceptance of the integrated candidate.

## Independent retain decision

**Retain Item 1 as engineering infrastructure and accept controlled
single/overlap neutrality; defer scientific acceptance of the integrated
full-recording candidate.**

The exact-current controlled run now supplies the required normalized
single/overlap equivalence and a successful all-reused checkpoint rerun. Add a
deliberate corruption/mixed-resume test and repeat the broad worker sweep on the
final code before making a final performance claim. Separately diagnose the
full-baseline coverage and female-ID regressions; female-only GT cannot rule out
additional male-to-male errors.
