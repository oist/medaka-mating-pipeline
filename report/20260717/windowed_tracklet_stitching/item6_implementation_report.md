# Item 6 implementation report: deferred identity decisions

## Implemented method

The deferred policy retains up to 32 beam hypotheses within 2.0 cumulative
cost, enters ambiguity at a nominal 0.5 margin, resolves after a 1.0 lead for
ten direct frames, and blanks disagreeing IDs after a 250-frame horizon. The
immediate comparator commits one state at each boundary. Geometry rows are
retained while unresolved biological-ID assignments are blank.

## Evidence

On frames 72,000--72,309, immediate and deferred choose the same three paths,
156 edges, objective 44.371295, and nearly identical runtimes (0.2079 versus
0.2157 s). Deferred records one resolved interval at 72,052--72,055 and one
end-of-cohort unresolved interval at 72,308--72,309, blanking six rows. An exact
250-frame equal-cost synthetic passes, and a delayed synthetic demonstrates
that retained history can avoid a greedy 51.0-cost choice.

The verifier found policy-semantic failures: deferred can retain alternatives
outside the 0.5 entry margin, stability is credited from future frames inside a
fragment rather than resolving on the actual tenth direct frame, hypothesis
checkpoint/resume is absent, and the pruning audit lacks complete histories.
On the Item 2 graph, deferral is more clearly antagonistic: the beam changes the
selected-edge solution, records five resolved and four unresolved intervals,
and blanks 744 of 840 identity rows. Its reported runtime is a composite of the
completed upstream run plus a separate 0.529 s solve, not a clean end-to-end
timing comparison.

## Decision

Defer. The abstention/horizon mechanism is promising and never fabricates an ID
at expiry, but current episode semantics do not match the frozen specification
and the real slice shows no quality gain. No female GT occurs on the isolation
slice; elsewhere it can identify only female errors, not male-to-male swaps.
