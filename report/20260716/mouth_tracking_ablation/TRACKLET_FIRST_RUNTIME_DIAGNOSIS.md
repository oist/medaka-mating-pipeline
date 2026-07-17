# Tracklet-first cohort and runtime clarification

## What the 27 annotations are

The `27` is the common mapped evaluation cohort available for all three
methods. It is not a separate manual annotation set and it is not a count of
tracklets.

The GUI ledger contains 300 rows with `gt_status=confirmed` and `sex_label=F`.
The tracklet-first reconstruction was completed only for four inclusive frame
windows, containing 26,104 frames in total. Exactly 27 confirmed-female GUI
rows fall in those windows, and all 27 produced a mapped method ID in all three
methods:

| Window | Confirmed-F rows | Annotation frames |
|---|---:|---|
| 71,000-74,100 | 4 | 71,288; 72,371; 73,027; 73,953 |
| 95,000-99,000 | 5 | 95,495; 96,132; 96,612; 97,802; 98,387 |
| 173,000-182,000 | 8 | 173,186; 174,316; 175,332; 175,927; 177,242; 179,435; 180,515; 181,221 |
| 270,000-280,000 | 10 | 270,937; 271,809; 273,617; 275,099; 275,753; 276,840; 277,726; 278,182; 279,165; 279,942 |

The evaluator first restricts each method to annotation frames physically
present in that method's output. A row is mapped when side1 or side2 matches a
projected method trajectory within the 80-pixel mapping gate; if both views
map, their method IDs must agree. The pairwise common cohort is the intersection
of mapped `annotation_id` values. Of these 27, 24 also pass the strict criterion:
both side views agree and both reprojection distances are at most 8 pixels.
The three non-strict rows are frames 275,099, 276,840, and 279,165.

Therefore, the three-way 27-row result is an event-window diagnostic, not an
estimate over the full 300-annotation recording.

## What is currently implemented

1. Build mouth-based 2D tracklets independently in each view. In the tested
   frame-conservative mode, a clean frame has exactly three sufficiently scored
   mouths separated by at least 0.5 body lengths. If any pair is closer, all
   active tracklets in that view terminate. Within clean runs, nearest-neighbor
   Hungarian assignment extends tracklets.
2. Run a geometry-only 3D bootstrap independently per frame. Up to six
   detections per view produce up to 216 three-view candidates; the selector
   chooses up to three mutually exclusive candidates with minimum residual.
3. Map each 2D tracklet observation back to its exact SLEAP candidate slot.
4. Build one global cross-view graph. Tracklets from different views receive an
   edge when the bootstrap repeatedly assigns them to the same 3D source
   (`>=3` overlap frames and `>=0.95` consistency). Union-find merges eligible
   edges while rejecting components known to contain different fish in the
   same frame.
5. Seed three canonical identities from one clean frame, propagate same-view
   all-different constraints, and rerun the framewise 3D candidate selector with
   tracklet identity and motion constraints.

Without optional silhouette-continuity edges, the graph has no direct edge that
scores the end of one broken within-view tracklet against the start of another.
It can bridge a break only indirectly when a continuous tracklet in another
view overlaps both fragments. If all views break together, there is no explicit
fragment-to-fragment optimization across the break.

This is a global connected-component constraint graph, not a windowed tracklet
stitcher and not an overlap-layout-consensus/de Bruijn-style DNA assembler.

## Why the full run exceeded one hour

The attempted full run was terminated after more than one hour; it did not
raise an algorithmic exception. Artifacts are written only after all global
tracklet processing returns, so the empty output directory did not mean that
the initial triangulation had not completed.

A no-plot `cProfile` run on frames 72,000-73,000 (1,001 frames) with the same
inputs took 235.4 seconds wall time. Important cumulative timings were:

| Operation | 1,001-frame profile |
|---|---:|
| Six full `.slp` loads | 114.9 s |
| Three tracklet-to-slot mapping calls | 45.3 s |
| Both 3D candidate-selection passes | 5.72 s |
| Final `build_component_hints` scan | 8.47 s |
| Cross-view graph construction | 2.72 s |
| Identified-2D row construction | 1.73 s |

The requested frame range is applied only after `sleap_io.load_slp` has
materialized the entire SLP, and the three SLPs are loaded once for coordinates
and again for slot matching. The profile constructed about 6.19 million SLEAP
instance objects across those six loads despite requesting only 1,001 frames.

The more serious full-recording cost is per-frame Python/Pandas iteration. A
linear extrapolation of `build_component_hints` alone is about 49 minutes for
348,600 frames; graph construction is about 16 minutes and the two
triangulation passes about 33 profiled minutes. These are extrapolations rather
than a completed full-run measurement, but together they explain why one hour
was insufficient. Full data also retain two large per-frame reconstruction
dictionaries, slot maps, and multiple multi-million-row observation tables,
consistent with the observed ~5.6 GB working set.

## Better match to the intended algorithm

A scalable implementation should use overlapping temporal windows and a
tracklet-level directed graph:

1. Triangulate candidate cross-view tracklet groups over several overlapping
   frames and aggregate robust residuals.
2. Create directed candidate edges from an ending 3D fragment to later starting
   fragments using endpoint position, velocity prediction, overlap consistency,
   gap duration, and mouth confidence. Use tail-vector continuity only as a
   tie-breaker near close contact.
3. Solve three disjoint time-ordered paths with min-cost flow, or use Hungarian
   boundary matching with a short beam of unresolved hypotheses. Ambiguous
   close-contact windows should defer the identity decision until fish separate.
4. Reconcile adjacent windows through their overlap and checkpoint each window.
5. Cache compact mouth candidate arrays once instead of decoding full SLP files
   for every stage.

This has a conceptual resemblance to overlap-based fragment assembly, but the
more appropriate algorithmic family is multi-target tracklet association using
a directed acyclic graph/min-cost flow. Time order and the fixed population of
three fish make it substantially simpler than general DNA assembly.
