# Item 5 implementation report: global trajectory solver

## Implemented method

Sequential Hungarian, beam-32, and exact SciPy/HiGHS binary path cover consume
the same immutable fragment/edge table and serialized edge costs. The exact
model gives every fragment one predecessor/birth and one successor/death and,
in `exact_three` mode, enforces exactly three births and deaths. Its primary
objective is the sum of selected edge costs with tolerance 1e-9. A separate
`restarts` mode makes fragmentation explicit rather than pretending segments
are three fish.

## Evidence

On the exact-three-feasible frames 72,000--72,309 graph, sequential, beam-32,
and global all select three paths/156 edges with objective 44.371295 and the
same output. Runtimes were 0.223/0.221/3.091 s, so global was about 14x slower
without an objective or quality gain. A delayed-disambiguation synthetic does
show sequential objective 51.0 versus 1.1 for beam/global, proving the intended
look-ahead mechanism. The 1,000-frame and Item 2 graphs are exact-three
infeasible; this is correctly surfaced. Restart mode yields 113 explicit
segments and objective 1,677.260984.

The exact model and objective recomputation pass, but the intended
lexicographically smallest optimum is not guaranteed by the rank-sum tie-break,
and full explored/pruned-state telemetry is incomplete.

## Decision

Defer. Keep the exact solver as an audit oracle on small feasible graphs, not a
production default. Fix lexicographic tie-breaking, graph feasibility, and
telemetry before a larger run. The real isolation slice contains no female GT;
female-only GT would still miss male-to-male swaps.
