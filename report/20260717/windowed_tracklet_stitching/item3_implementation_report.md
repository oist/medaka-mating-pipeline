# Item 3 implementation report: explicit broken-fragment links

## Implemented method

Fragments are linked by a complete directed DAG over feasible end-to-start
pairs with gap 0--100 frames and endpoint displacement no greater than
`min(8*dt, 80)` mm. Endpoint and velocity supports use at most 15 direct samples
within 30 frames; velocity is a deterministic median pairwise slope. The
dimensionless mouth cost is the sum of position (weight 1), prediction (2), gap
(0.25), overlap inconsistency (1), refractive residual scaled by 2.5 mm^2
(0.75), and mouth confidence (0.5). Missing optional evidence is neutral. Birth
and death each cost 3 in the explicit-restart fallback. No interpolated row,
tail, GT, or event label enters edge features.

## Evidence

Frames 32,264--33,263 produced 491 fragments, 38,592 feasible edges, and 378
selected links. The recomputed objective is 1,677.260984: 999.260984 selected
edge cost plus `113 * (3+3)` restart cost. The output retains all 2,819 direct
rows and has zero adjacent jumps above 8 mm, but it contains 113 trajectory
segments—not three biological fish identities. Runtime was 16.2127 s and peak
RSS 310 MB. Deterministic graph/objective/resource tests pass.

Following the independent audit, an exact same-fragment links-off control was
also serialized in `item3_exact_links_off_control.json`: 491 fragments, zero
links, 491 explicit restart segments, and objective `491 * (3+3) = 2946`.
Links reduce the frozen objective by 1,268.739016 and the segment count by 378.
This closes the link-decision attribution algebraically without pretending that
the reduced objective or segment count proves correct biological identity. The
control has no timed solver because disabling every edge requires no solve.

## Decision

Defer. The graph and accounting are useful, and the explicit restart contract
is safer than silently forcing an out-of-gate edge. It has not solved biological
identity, and the real slice does not supply enough female GT to measure an
identity benefit. Female-only GT also cannot detect male-to-male swaps.
