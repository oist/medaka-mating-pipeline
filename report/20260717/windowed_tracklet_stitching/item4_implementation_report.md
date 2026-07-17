# Item 4 implementation report: conditional tail-vector continuity

## Implemented method

Tail evidence augments the frozen Item 3 graph only when the mouth-only best and
runner-up complete assignments are within 0.5 cost and disputed mouths are
within 12 mm. Endpoint summaries use at most 15 vectors in 30 frames and at
least three samples. Valid 3D tails require two views, XYZ 0--200 mm, residual
at most 10 mm^2, and body length 10--60 mm. Complete comparable 3D evidence is
preferred; a common-view 2D fallback is available. The capped penalty is
`(1/12)*clip(angle/90,0,1)` and incomplete tails contribute exactly zero.

## Evidence

The real 1,000-frame run took 16.2015 s and exactly matched Item 3: 378 selected
edges, objective 1,677.260984, 113 segments, 2,819 direct rows, and identical
jump metrics. Of 38,592 edges, 38,211 were not mouth-ambiguous, 362 had
incomplete tails, 13 failed proximity, six were 2D-eligible, and zero were
3D-eligible; none of the six eligible edges was selected. Missing-tail neutrality
passed independent synthetic and real comparisons.

The verifier found three specification gaps: only the best runner-up rather
than every within-margin assignment is enumerated; proximity uses endpoints
rather than the full +/-15-frame support; and the 2D fallback lacks the frozen
five-sample body-length baseline/filter. Tail augmentation also changes the
whole graph hash rather than preserving a tail-independent graph hash.

## Decision

Defer. It is neutral and safe on measured incomplete/clear cases but changes no
real decision and its only eligible evidence uses the nonconforming fallback.
Correct the four auditability/specification gaps before retesting. Female-only
GT cannot establish male-to-male identity continuity.
