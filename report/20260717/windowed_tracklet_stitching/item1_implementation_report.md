# Item 1 implementation report: overlapping temporal windows

## Implemented method

The implementation reads bounded inclusive slices from the strict-mouth,
six-candidate HDF5 caches, processes 10,000-frame ownership cores with 250-frame
context, and writes an atomic checkpoint per window. Reconciliation assigns each
frame to exactly one core, compares duplicated direct observations by raw
`(frame, view, candidate_slot)` resources, and accepts a boundary alias only with
at least three evaluable frames, at least 0.95 consistency, finite agreeing XYZ,
and all-different compatibility. A deterministic union-find produces canonical
fragment IDs. Scientific association costs are unchanged by the window wrapper.

Parameters are frames unless stated otherwise: core 10,000, context 250,
maximum link gap 100, motion gate 8 mm/frame capped at 80 mm, XYZ bounds 0--200
mm, seed 20260717, float64 cache coordinates, and eight workers for the full run.
The exact specification is in `item1_primary_report.md`; independent checks are
in `item1_verifier_report.md`.

## Evidence

On frames 32,264--35,364, current-code single and fresh overlap-8 runs each
produced 9,122 direct observations and 1,858 fragments with the same normalized
filled-output SHA-256 (`a6e85f...e48bf9`). Overlap reconciled 208 aliases and all
six seams with zero unresolved seams. Single took 22.6572 s; fresh overlap took
25.7309 s. A resume reused all seven checkpoints and preserved the output hash.

The pre-final worker sweep was deterministic at 1/2/4/8/16 workers and took
58.074/44.444/39.541/36.040/37.585 s. Eight workers were fastest in that sweep.
The full final run took 575.555 s, peak process-tree RSS 3.61 GB, emitted 35
checkpoints, 1,030,486 direct observations, 92,426 fragments, 3,575 candidate
aliases, 618 accepted aliases, and zero unresolved seams. CPU and output-size
details are in `runtime_breakdown.csv`.

## Quality and decision

Window execution/reconciliation and restartability pass. The integrated full
candidate does not: versus the historical strict-mouth result, mapped female-GT
accuracy fell from 0.5235 to 0.4276, observed female changes rose from 11 to 81,
and all-three-direct coverage fell from 0.9816 to 0.8943. Retain the cache,
checkpoint, bounded-memory, and reconciliation infrastructure; reject the
integrated candidate as a production scientific result. Female GT identifies
only the female and cannot reveal male-to-male swaps.

The post-audit pair reruns limit the equivalence claim further. Overlap is exact
with Items 3 and 4 and preserves normalized Item 5/6 solver outputs, but Item
1+2 produces 783 rather than 840 direct rows and one unresolved seam. Therefore
the wrapper is reusable only after each new window-local scientific mode passes
its own single/overlap equivalence test; it is not universally transparent.
