# Item 2 independent verifier report: multiframe cross-view grouping

## Verdict

**Execution correctness: PASS for the core final implementation, with artifact
and comparator caveats. Scientific acceptance of the current multiframe
configuration: FAIL.**

The blockers in the first audit are materially addressed. The final code freezes
all finite in-bounds two- and three-view hypotheses, preserves provenance for all
six candidate slots, deduplicates group/frame alternatives, computes support,
valid fraction and robust residual evidence in mm², and solves a resource- and
capacity-constrained multiframe MILP. The 8 mm physical motion gate is applied
before the same-group preference, so group evidence no longer relaxes the hard
gate.

The saved 1,000-frame framewise and multiframe runs share a byte-identical frozen
hypothesis file and satisfy resource/capacity invariants. However, those runs
record older core/orchestrator code hashes, not the exact final revision. More
importantly, the multiframe configuration selects only 840 direct rows versus
2,819 framewise and collapses all-three direct coverage from 84.8% to 0.3%.
That is a scientific rejection even though the optimizer executes correctly.

Female GT is not used by the grouping implementation. Available GT identifies
only the female and cannot detect male-to-male swaps; this 1,000-frame interval
contains only one eligible one-view female annotation and zero strict two-view
annotations, so it cannot validate identity quality.

## Scope and evidence inspected

- `scripts/reconstruction_3d/candidate_cache.py`
- `scripts/reconstruction_3d/windowed_tracklet_core.py`
- `scripts/reconstruction_3d/windowed_tracklet_stitching.py`
- `tests/test_candidate_cache.py`
- `tests/test_windowed_tracklet_stitching.py`
- `bench_frozen_framewise_32264_33263/`
- `bench_frozen_multiframe_32264_33263/`
- `full_optimized_baseline_v2/`
- `final_full_quality_comparison/`
- `hardware_benchmark.json`

No reusable code was edited during verification.

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

Relevant tests verify all-six-slot provenance, conservative ambiguity handling,
minimum multiframe support, frozen-support/resource set packing, physical
fragment breaks, and generic resource/capacity invariants. The suite still lacks
the preregistered transient/persistent reflection truth cases, input-order
permutations, explicit four-group capacity fixture, and batched-versus-scalar
hypothesis equality test.

## Claim-by-claim result

| Claim | Result | Independent evidence |
|---|---:|---|
| Both modes receive all finite in-bounds 2-/3-view hypotheses | PASS in final code | `enumerate_refractive_hypotheses` evaluates the 6³ full-view combinations plus three 6² two-view masks in bounded batches and keeps native refractive XYZ/residual rows. |
| Observed extra slots are distinct from missing views | PASS | `build_candidate_tracklet_ids` covers all six slots. Ambiguous observed candidates receive deterministic negative one-frame IDs below -1; -1 remains the missing-view sentinel. Focused extra-slot test passes. |
| Framewise and multiframe artifacts use the same frozen table | PASS for saved older revision | Both 1,000-frame checkpoints contain 79,975 hypothesis rows, file size 2,456,325 bytes, and SHA-256 `dc5100d5dd5e12ca83c3a22da7478906ddff7a129bf7da169285a8fb2b37b0e8`. Their recorded implementation hashes predate the final files. |
| Duplicate key/frame hypotheses are deterministically reduced | PASS by code inspection | Rows sort by view/tracklet key, frame, residual and hypothesis ID; the lowest row is retained and discarded rows are saved. The real run had zero duplicates. No input-permutation test exists. |
| Support and overlap consistency are enforced | PASS | Candidate groups store `n_overlap`, `n_support`, valid fraction and rejection reason; thresholds are support at least 3 and valid fraction at least 0.95. |
| Robust refractive evidence is unit-correct and gated | PASS | The code uses median + 3×1.4826×MAD and compares against `max_group_residual_mm2`; outputs record residual gate and `mm^2` unit. No square-root conversion occurs. |
| Resource reuse and more than three active groups are prohibited | PASS | The multiframe MILP includes one constraint per `(frame, view, slot)` and a capacity-three constraint per frame. Focused set-packing test passes; both real modes report zero resource conflicts and zero frames over three active outputs. |
| Selection follows support, view-count, residual priorities | PASS for multiframe; PARTIAL for common-comparator claim | Multiframe uses three staged SciPy MILPs: maximize support, then observed views, then minimize robust residual plus deterministic rank epsilon. Framewise uses the existing exact global candidate-set selector rather than the same MILP implementation, so the primary report's literal “same set-packing solver” claim is not met even though the frozen input and priorities are shared. |
| Group preference preserves the physical 8 mm gate | PASS | `build_fragments` stores unscaled physical distance, rejects distances above 8 mm before assignment, and rechecks the selected match after applying the same-group cost preference. |
| Required outputs and provenance are complete | PARTIAL | Checkpoints save hypotheses, candidate groups with selected flags, conflict rows, discarded duplicates, selected observations/fragments, hashes and complete configuration. Separate `selected_groups.parquet`, `selected_group_observations.parquet`, and `grouping_summary.json` named in the primary report are not emitted. |
| Female GT/tail/future identity cannot leak into grouping | PASS by inspection | Hypothesis/group functions consume only candidate coordinates, tracklet/resource provenance, calibration and frozen thresholds. Female GT and downstream identity are absent. |
| Multiframe improves scientific quality without material coverage loss | FAIL | The 1,000-frame artifact loses most direct coverage and has no adequate GT or reviewed-event evidence. |

## Frozen 1,000-frame evidence

Both modes use frames 32264..33263, frozen strict-mouth six-slot caches,
refractive mm² scores, support 3, valid fraction 0.95, and residual gate 10 mm².

### Group inventory

The multiframe run creates 48,804 candidate groups:

- 18 eligible and 9 selected;
- 48,515 rejected below minimum support;
- 269 rejected above the robust residual gate;
- 2 rejected below valid fraction;
- 1,723 recorded pairwise resource conflicts;
- 0 discarded duplicate hypotheses.

This is auditable evidence that the intended filters and set packing executed.
It also shows that the frozen thresholds reject nearly the entire candidate
inventory.

### Runtime and output

| Metric | Frozen framewise | Frozen multiframe |
|---|---:|---:|
| Frozen hypothesis rows | 79,975 | 79,975 |
| Window wall time (s) | 15.56 | 55.39 |
| Total pipeline wall time (s) | 18.77 | 58.06 |
| Peak RSS (bytes) | 402,718,720 | 268,546,048 |
| Direct observations | 2,819 | 840 |
| Local/reconciled fragments | 315 | 299 |
| Frames with three direct observations | 848 | 3 |
| Direct row share of 3,000 possible | 93.97% | 28.00% |
| Direct max consecutive jump (mm) | 6.81 | 7.20 |
| Direct jumps above 8 mm | 0 | 0 |
| Filled finite-row share | 96.67% | 71.30% |
| Filled all-three-finite frame share | 90.0% | 47.9% |

The smaller fragment count is not an identity-quality improvement: it is largely
explained by losing 1,979 observations. Spline filling recovers some finite rows
but cannot restore the missing direct evidence. Multiframe is also 3.1× slower
end to end on this window.

### Provenance limitation

The paired runs were produced with core hash
`f33ce5cf0abf92cdc6fe8c8c766425c949f4aa79f00bcf99d29a03b9676a1744`
and orchestrator hash
`598fd650fc504e4b5223d91d42a17621630985bf35209633dc28c5566da87551`.
The current files hash to
`a49e827c0a7000a7c5e55c02cda8a7166f9ec4e47a23366b9aa161c9bd05fea2`
and
`3e8e72cd5b143ee22d554d3d53d0c432dfe41ab9cb60f658d427cc51e4255d1f`.
Therefore the saved comparison is informative about the final algorithm family
but is not confirmatory evidence for the exact current revision.

## Full-recording and GT evidence

`full_optimized_baseline_v2` matches the current code hashes but is configured
with `hypothesis_source=legacy` and framewise grouping. It is not an Item 2
multiframe full run. `final_full_quality_comparison` therefore evaluates the
optimized legacy/framewise baseline, not the multiframe candidate, and cannot be
used to accept Item 2.

The 1,000-frame interval has one eligible confirmed-female annotation, mapped in
only one view, and zero strict two-view annotations. None of the four reviewed
mixed events lies in the interval. There is consequently no meaningful female
identity or event denominator for the frozen comparison. Even a larger female
denominator would remain blind to male-to-male swaps.

No saved artifact supplies synthetic transient-reflection precision/recall or
the persistent-reflection failure case. Those were frozen retain requirements,
not optional descriptive checks.

## Independent retain decision

**Reject the current multiframe configuration as a scientific candidate; retain
the frozen-hypothesis and constrained-grouping machinery for further work.**

Before re-evaluation, tune or redesign the group support construction so direct
coverage is not destroyed, then rerun framewise and multiframe on the exact same
final-code hypothesis hash. Add the missing transient/persistent reflection,
capacity-four, all-view-break, permutation, and batched/scalar tests. A new
candidate must preserve paired direct coverage, pass resource/capacity
invariants, and be evaluated on reviewed events and sufficient female GT while
explicitly acknowledging that female-only GT cannot reveal male-to-male swaps.
