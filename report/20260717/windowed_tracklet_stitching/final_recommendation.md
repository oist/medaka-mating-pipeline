# Final recommendation

## Production decision

Keep the historical strict-mouth full output as the production result:

`report/20260716/mouth_tracking_ablation/full/mouth_only_3d/mouth_only_3d_3d.csv`

Do not promote the completed optimized full candidate:

`report/20260717/windowed_tracklet_stitching/full_optimized_baseline_v2/optimized_baseline_v2_3d.csv`

The candidate is faster (575.6 s versus the documented 15.6 min), bounded,
restartable, deterministic, and free of jumps above 8 mm, but it has lower
female-GT accuracy (0.4276 versus 0.5235), far more observed female-ID changes
(81 versus 11), and lower all-three-direct coverage (0.8943 versus 0.9816).
These are scientifically material regressions.

## Component decisions

- Retain Item 1's HDF5 cache, bounded window execution, exact seam
  reconciliation, checkpoints, and array kernels as infrastructure. Do not
  equate this with accepting the current full output.
- Reject Item 2's current multiframe support design because it is slower and
  reduces 1,000-frame all-three-direct coverage from 848 frames to three.
- Defer Item 3 until restart segments are converted into demonstrably stable
  biological identities rather than reported as 113 separate trajectories.
- Defer Item 4 until all within-margin assignments, full boundary proximity,
  compliant 2D length filtering, and a tail-independent graph hash are present.
- Use Item 5's exact global solver only as a small-graph audit oracle until the
  lexicographic tie rule, graph feasibility, telemetry, and quality benefit are
  resolved.
- Defer Item 6 until entry-margin, tenth-direct-frame resolution, complete audit,
  and hypothesis checkpoint/resume semantics match the frozen policy.

No higher-order bundle is justified for a new full run. Item 2 is antagonistic,
Items 3/4/5 are neutral or infeasible on the relevant graph, and the current
Item 6 code path is a no-op after an offline global solve while still failing
its active-deferral policy contract. This no-op is not evidence that a correct
active deferral method would be scientifically redundant.
The explicit overlap pair reruns sharpen this conclusion: Item 1+2 falls from
840 to 783 direct rows and from 684 to 514 covered frames, improves all-three
frames from 3 to 73, but creates one unresolved seam. That is mixed and still
unacceptable, not a uniform quality loss. Item 1+3, +4, +5, and +6 preserve
their normalized scientific outputs and add no quality gain.
Item 2+6 is especially poor: four unresolved intervals leave 744 of 840 identity
rows blank and the deferred beam changes the upstream edge solution.

Runtime interactions do not support a multiplicative bundle. Item 1+2 is the
only apparent runtime synergy: `Sij=0.3645` versus `Si*Sj=0.3486`, with 5.208 s
more saving than additive deltas predict, but it is a single-run estimate and
quality is mixed with an unresolved seam. Item 1+3 and 1+4 have slightly
super-additive runtime costs (0.402 s and 0.326 s beyond additive cost) with
identical scientific output; Item 1+5 is startup-dominated and Item 1+6
approximately additive. Item 5+6 is a startup-dominated implementation no-op (`Sij=0.9681`,
multiplicative deviation 0.8935), so the measured implementation effect is
redundant rather than a real acceleration. Other pairs lack a common
T0/Ti/Tj/Tij timing boundary and
are explicitly `NA`, not assigned a fabricated multiplicative score.

## Next acceptance gate

A successor should first reproduce the historical output's coverage and female
stability on identical frames while retaining Item 1 infrastructure. Then it
must pass all synthetic resource/crossing/break cases, the four reviewed windows
and four controls, deterministic resume equivalence, and full-recording QC. A
scientific promotion ultimately needs identity ground truth for all three fish:
the current female-only ledger cannot reveal male-to-male swaps.
