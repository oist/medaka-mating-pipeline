# Proposed side-view training follow-up

The authoritative machine-readable proposal is
`proposed_training_condition_v1.json`. Its state is `proposed_not_run`.

## Saved conditions

### Tested challenger to retain

- Centroid: sigma 1.5, mouth anchor.
- Centered instance: the unchanged evaluated control model.

This pair had the best coverage on the single pilot event. It is a challenger,
not the overall validated optimum: sigma 2.0 plus unchanged control remained
the broader three-experiment stress favorite.

### Proposed control-derived hybrid

- Centroid sigma: 1.5.
- Add independent, fully mouth-labeled centroid training instances.
- Centered-instance crop: 448 px.
- Centered-instance rotation: -90 to +90 degrees.
- Crop-position jitter: 0 px.
- OHKM: disabled.
- Missing-mouth policy: exclude only the ineligible centered-instance sample,
  while retaining other eligible instances from the same frame.
- Early-stopping patience: 40; LR plateau patience: 5.

Do not call this centered-instance recipe “control.” Changing crop, rotation,
OHKM, and mouth eligibility creates a new bundled challenger.

## Does every proposed change follow from the test?

No.

- Sigma 1.5 is supported as the pilot-coverage challenger, but sigma 2.0 won
  the broader stress comparison.
- The unchanged control centered-instance model clearly won both oracle
  comparisons.
- Crop 448 and rotation +/-90 degrees are geometrically reasonable, but their
  accuracy effects were not isolated.
- OHKM-off is not supported by the current result. It must be compared against
  an otherwise identical OHKM-on twin.
- Removing mouth-missing centered-instance samples is semantically reasonable
  for mouth-centered crops, but the test did not prove it improves accuracy.
  Filtering must be per instance, not by deleting the entire frame.
- More correctly labeled mouth anchors is a sensible data improvement.
  Missing mouths must be annotated or ignored in the loss; they must not become
  negative-background examples accidentally.

## Required comparison

Run the proposed hybrid beside:

1. sigma 2.0 centroid plus the unchanged evaluated control CI;
2. the identical hybrid with OHKM enabled.

Use a new output root and do not modify the completed six-candidate run or the
closed evaluation bundle.
