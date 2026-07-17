# Primary and verifier reconciliation

The six primary reports froze proposed methods and acceptance criteria before
implementation results were available. Independent verifiers then inspected
current code, ran focused tests and immutable real graphs, recalculated saved
objectives/hashes, and issued execution/scientific verdicts. No verifier edited
the reusable implementation.

| Item | Primary intent | Independent finding | Reconciled decision |
|---|---|---|---|
| 1 | Exact window equivalence, seam reconciliation, checkpoint reuse | Current single/overlap output and resume pass; full candidate quality regresses | Retain infrastructure; reject integrated scientific candidate |
| 2 | Multiframe support should stabilize cross-view grouping | Mechanics pass, literal shared-solver claim is only partial, direct coverage collapses | Reject current thresholds/design |
| 3 | Explicit, auditable end-to-start fragment links | Graph/objective/resource accounting pass; result is 113 restart segments, not three fish | Defer |
| 4 | Tail changes only genuinely ambiguous complete cases | Missing-tail neutrality passes and selected output is unchanged; eligibility/fallback/hash contract is incomplete | Defer |
| 5 | Exact global three-path optimum on the same graph | Exact MILP and recomputed objective pass on a feasible slice; no real gain, slower, lex tie flaw, larger graph infeasible | Defer/audit oracle only |
| 6 | Retain in-margin alternatives and abstain at horizon | Horizon/blanking and delayed synthetic pass; entry-margin, future-support, audit, and resume semantics fail | Defer |

The apparent tensions are specification-versus-execution qualifications, not
contradictory measurements. For example, Item 1 passes its isolated equivalence
test while the full experimental candidate fails scientific quality; Item 4
passes neutrality but fails promotion because it never changes a valid real
decision and its fallback is incomplete. Both statements are retained.

There was no material numerical disagreement requiring a neutral referee: all
verifiers reproduced the compared hashes, row/fragment counts, objectives, and
decisions from independent reads or reruns. Failed and infeasible runs are
listed in `failed_runs.json`, not treated as favorable evidence.

The first pair audit did identify an evidence-label disagreement: the original
Item 1 pair rows cited single-window downstream artifacts. Those rows were not
defended as-is. Five explicit overlap/control reruns were added and independently
re-audited; the pair table now uses those artifacts and full T0/Ti/Tj/Tij
arithmetic. This correction is why Item 1+2 is now classified as runtime-
synergistic but quality-mixed with an unresolved seam, and the other Item 1 pairs as output-
independent/redundant rather than inferred from compatibility alone.

All quality conclusions share one limitation: the available GUI ground truth
identifies only the female. It can score female mapping/stability but cannot
independently detect a swap between the two males.
