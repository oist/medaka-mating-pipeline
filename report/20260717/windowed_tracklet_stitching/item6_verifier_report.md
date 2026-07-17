# Item 6 final independent verifier report: deferred identity decisions

## Scientific acceptance

**DEFER / not scientifically accepted.**

The final implementation now plumbs immediate/deferred policy, uses the same
immutable graph, records ambiguity intervals, writes an exact 250-frame horizon,
and blanks identity while preserving direct geometry for unresolved intervals.
Synthetic later-separation and unresolved cases pass. However, the policies do
not retain the same pre-episode hypothesis set, and the claimed 10-direct-frame
resolution can count an entire future fragment at its start boundary. The only
real evidence is 310 frames with no female GT and no path/identity improvement;
the 1,000-frame graph is exact-three infeasible.

No reusable code was edited by this verifier.

## Final tests and shared real graph

The final hashes, command, and **28 passed in 1.81 s** result are recorded in
`item5_verifier_report.md`.

Both Item 6 outputs use beam width 32, exact-three mode, and the same 310-frame
graph (72,000--72,309) with aggregate hash:

```text
2830698919f091dc6018768e84178e757de471cbd2ccccc5d6683da25b1cb8d9
```

The graph contains 930 observations, 159 fragments, and 4,333 candidate edges.
Independent current-code reruns exactly reproduced both summaries and all saved
direct tables.

## Claim-by-claim verification

| Claim | Result | Independent evidence |
|---|---|---|
| CLI/config policy plumbing | **PASS** | Both the full pipeline and solve-only CLI populate `identity_policy`, entry/retention/resolution margins, stability frames, horizon, beam width, and cover mode. |
| Same immutable graph/costs for policies | **PASS** | Immediate and deferred summaries contain the same three file hashes and aggregate graph hash. Selected edges, paths, and objective are also identical on the real slice. |
| Initial arbitrary label symmetry is removed | **PASS** | First observed layer is canonically ordered left-to-right; synthetic ambiguity no longer opens solely from six global label permutations. |
| Same short-beam hypothesis enumerator before commit | **FAIL** | Immediate prunes to one state at every boundary. Deferred retains every state within the 2.0 retention margin even before an entry-qualified episode. Thus an alternative outside the 0.5 entry margin can survive only in deferred. On the real slice immediate records zero episodes while deferred records two, proving the compared ambiguity sets are not identical. |
| Entry uses mouth-only incremental cost | **PARTIAL** | The solver stores per-transition `mouth_cost` and uses a 0.5 comparison, but candidates are drawn from the policy-dependent retained state set above; the exact shared-episode isolation is not met. |
| Later separation can reverse a local leader | **PASS on synthetic** | Final test and independent probe: immediate objective 51.0, deferred 1.1; deferred retains the later-winning history and emits `resolved_by_margin`, immediate emits `forced_immediate`. |
| Retention margin and beam cap | **PASS mechanically** | Deferred filters to cumulative cost within 2.0 and caps at 32; delayed synthetic peak is at least two. Margin-pruned counts/reasons are not fully audited. |
| Resolve after a 1.0 lead for 10 consecutive direct frames | **FAIL** | At frame 72,055 the real run adds `observed_frames=80` from fragments spanning 72,055--72,134, then records the episode as resolved at 72,055 with `stable_direct_frames=83`. It uses future fragment support at the fragment start instead of resolving on the actual tenth direct frame. |
| Exact 250-frame unresolved horizon | **PASS on synthetic** | Equal-cost hypotheses produce one unresolved record whose `frame_end - frame_start == 250`, even when the next fragment boundary is later. |
| Horizon/recording-end uncertainty does not force an ID | **PASS for direct output** | Deferred real output has six blank-ID rows at frames 72,308--72,309, all marked `identity_status=unresolved`; immediate maps them. |
| Geometry and resources are unchanged between policies | **PASS for direct output** | Frame, fragment, XYZ, and three source-slot columns compare exactly. All 930 deferred XYZ rows remain finite; only six identity cells/statuses change. |
| Filled named-identity output preserves unresolved geometry | **PARTIAL / FAIL relative to primary wording** | `_write_filled_output` prevents false named interpolation by setting XYZ to NaN on unresolved frames. The direct artifact preserves unlabeled geometry, but the filled per-ID CSV cannot carry those unlabeled points. |
| Ambiguity audit is complete | **PARTIAL** | Saved interval tables contain start/end, status, hypotheses, margin, stability, and reason. There is no per-hypothesis selected-edge history, candidate assignment hash, or complete prune-reason table. |
| Hypothesis checkpoint/resume across window boundaries | **FAIL / absent** | Geometry checkpoints exist, but there is no serializable beam `initial_state` or one-pass-versus-resume hypothesis test. |
| Restart contract is explicit | **PASS, with scope caveat** | Missing exact cover can be represented as unique `SEG_*` fragments with birth/death cost; no biological identity is silently continued through a restart. This is segment output, not deferred biological identity. |
| Real identity-quality benefit | **FAIL / not demonstrated** | Both policies choose the same 3 paths, 156 edges, and objective 44.371295. Deferred only abstains on six final rows. The slice has zero female-GT annotations. |

An independent outside-entry counterexample confirms the policy-isolation
failure. I changed the first-boundary alternative from 1.1 to 1.6, placing it
0.6 above the local leader and therefore outside the frozen 0.5 entry margin.
Immediate returned the 51.0 greedy path. Deferred retained that unqualified
history anyway, later returned objective 1.6, and recorded **no ambiguity
episode**. The claimed gain therefore came from unconditional 2.0-margin beam
retention, not the preregistered deferral trigger.

## Saved Item 6 real result

| Policy | Objective | Peak | Episodes | Mapped rows | Unresolved rows | Wall |
|---|---:|---:|---:|---:|---:|---:|
| immediate | 44.371295 | 1 | 0 | 930 | 0 | 0.208 s |
| deferred | 44.371295 | 2 | 2 | 924 | 6 | 0.216 s |

Deferred episodes:

1. 72,052--72,055, `resolved_by_margin`, two hypotheses, winning margin
   1.19185, reported stability 83 direct frames. This resolution timestamp is
   invalid for the reason above.
2. 72,308--72,309, `unresolved_recording_end`, two hypotheses. The six geometry
   rows are correctly preserved and identity-blanked.

Both policies select byte-identical paths and link edges. Their direct Parquet
hashes differ only because deferred blanks the six unresolved identities:
immediate `022b40...`, deferred `b6a94c...`. Resource telemetry is not usable:
both runs completed before a sample was recorded (`resource_sample_count=0`,
peak RSS zero/CPU NaN).

## Explicit 1,000-frame scope failure

The 1,000-frame real graph at 32,264--33,263 has 491 fragments and 38,592 edges.
Independent exact-three global solving returned HiGHS infeasible in 1.05 s.
Therefore the immediate/deferred exact-three experiment has **not** been shown
to run even on this 1,000-frame control slice, much less a complete reviewed
event or recording.

Restart mode can cover the graph as 113 uniquely labelled segments and passes
resource invariants, but those segment IDs intentionally do not carry
biological identity through gaps. Item 6 cannot claim deferred continuity
across those restart boundaries without a separate, validated identity layer.

## Female-GT denominators and decision

There are **zero female-GT rows** in 72,000--72,309, so the six abstentions
cannot be scored as correct uncertainty versus avoidable coverage loss. No
eligible/mapped/strict counts, paired-common accuracy, observed female-ID
changes, reviewed-event outcomes, or male-male evidence are saved. Female-only
GT would not detect male-to-male swaps regardless.

Engineering acceptance is **PASS** for policy plumbing, exact horizon records,
honest direct-output abstention, and synthetic later-separation behavior.
Scientific acceptance remains **DEFER** until:

1. immediate and deferred begin from the same entry-qualified hypotheses, with
   alternatives outside 0.5 not retained only by deferred;
2. stability is counted frame-by-frame and the resolution frame is the actual
   tenth direct frame, not the start of a future-support fragment;
3. hypothesis audit/checkpoint-resume requirements pass;
4. an applicable longer graph is justified despite exact-three infeasibility;
5. repeated runtime/resources and all frozen real cohorts are evaluated; and
6. fixed female-GT denominators, abstention coverage, reviewed outcomes, and
   the male-male limitation support a real quality benefit.
