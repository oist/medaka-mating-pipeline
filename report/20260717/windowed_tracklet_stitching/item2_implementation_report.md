# Item 2 implementation report: multiframe cross-view grouping

## Implemented method

The frozen alternative aggregates per-view candidate-slot tracklets before 3D
group selection. A group requires at least three supporting frames, valid
fraction at least 0.95, robust refractive residual at most 10 mm^2, strict finite
mouths, and mutually exclusive raw resources. Three lexicographic MILP stages
maximize fish-frame support, maximize observed-view support, then minimize the
robust residual with a deterministic rank epsilon. It consumes no tail, female
GT, reviewed-event label, or downstream identity.

The comparator uses the existing exact framewise candidate-set selector on the
same frozen hypotheses. Consequently, the intended common-priority comparison
is valid, but the two modes do not literally share one solver implementation;
this verifier qualification is preserved rather than hidden.

## Evidence

On frames 32,264--33,263, framewise reconstruction took 18.7657 s and produced
2,819 direct rows with 848 all-three-direct frames. Multiframe took 58.0564 s,
selected only 18 groups/9 final groups after 48,515 insufficient-support
rejections, and produced 840 direct rows with only three all-three-direct
frames. It is 3.09x slower (`speedup=0.323`) and loses most direct coverage.
The one eligible female annotation is insufficient to support an identity claim.

## Decision

Reject the current thresholds/aggregation. The mechanics and resource
exclusivity tests pass, but real coverage is catastrophically antagonistic and
pairing with Items 3, 4, or 6 does not repair it. Redesign support construction
before any full-recording use. Female-only GT cannot diagnose male-to-male swaps.
The explicit overlap pair is mixed: 783 rather than 840 direct rows and 514
rather than 684 covered frames, but 73 rather than three all-three frames; it
also has one unresolved seam. The Item 2+6 composite leaves 744/840 identity
rows blank.
