# K_1_2 strict-mouth 6 mm tracking comparison

## Outcome

The 6 mm versus 8 mm decision is **inconclusive**. The controlled reconstruction
comparison is still technically valid, but the follow-up human review found
close-contact scenes, SLEAP false negatives, and shared-tail/node instances.
Those scenes can support human female identity while failing as automatic
SLEAP/geometry-ID ground truth, so the earlier recommendation to prefer 8 mm is
withdrawn.

## Controlled settings

- Experiment: `20251225_K_1_2_FCMC`
- SLEAP source: `round7_260704`
- Anchor: finite mouth only (`--anchor-node mouth --require-anchor-node`)
- Candidates: 6 per view; individuals: 3
- Calibration: fixed-interface refractive backend
- Native XYZ bounds: `-20..220` mm
- Link hard cap: 80 mm; maximum gap: 100 frames
- The paired runs differ only in `--max-jump-mm 6` versus `8`

The tracker's default shift-to-positive step selected different arbitrary output
origins for the two gate values. A constant translation was removed before
reprojection scoring. After translation, the unlabeled 3D point sets agree in
100% of 341,127 common all-three direct frames (maximum discrepancy
`1.61e-13 mm`), confirming that this is a temporal-linking comparison.

## Results

| Metric | 8 mm paired rerun | 6 mm | Change at 6 mm |
|---|---:|---:|---:|
| Female GT mapped | 298/300 | 298/300 | none |
| Best fixed-ID accuracy | 52.35% | 44.63% | -7.72 pp |
| Observed female-ID switches | 11 | 10 | -1 |
| Strict two-view fixed-ID accuracy | 51.99% | 46.93% | -5.05 pp |
| Strict two-view switches | 11 | 10 | -1 |
| All three direct/non-interpolated frames | 98.159% | 97.885% | -0.275 pp |
| All three finite frames | 98.978% | 98.949% | -0.029 pp |
| Consecutive-jump p99 | 2.264 mm | 2.291 mm | +0.027 mm |
| Maximum consecutive jump | 8.000 mm | 5.998 mm | mechanical gate effect |
| Reviewed events preserved | 3/4 | 3/4 | none |

The lower maximum jump at 6 mm is expected by construction and is not evidence
of better tracking on its own. Conversely, the fixed-ID accuracy difference is
not sufficient evidence for 8 mm because the reviewed branch cohort is
observation-quality confounded.

## Follow-up human and motion audit

- 25/26 gate branches now have a female review row.
- Only 14/26 meet the old structural proxy of confirmed female, two
  SLEAP-linked side views, and agreeing geometry IDs; this proxy still does not
  adjudicate shared nodes or close contact.
- All pre-existing rows migrate to `evidence_quality=unreviewed`, so zero are
  silently accepted as tracker-scoring truth. The GUI now records `usable`,
  `human_only`, or `unusable` plus structured close-contact/FN/shared-node flags.
- A bidirectional robust tangent audit uses up to 15 direct observations on
  each side of a branch. Across 53 changed-track boundaries, 6 mm had the lower
  tangent angle in 32 and 8 mm in 21; tangent discontinuities above 90 degrees
  occurred in five 6 mm boundaries and six 8 mm boundaries. This mixed result
  does not select either gate.

The next comparison should use the jump threshold as an outer safety gate and
make identity assignments with multi-frame velocity/tangent evidence. The new
`--motion-history-frames` option provides a robust history estimate; a short
lookahead remains a separate validation step.

## Artifacts

- Canonical 6 mm track: `mouth_only_jump6_3d_aligned_to_jump8.csv`
- Raw shift-to-positive 6 mm track: `mouth_only_jump6_3d.csv`
- Controlled comparison: `controlled_comparison_vs_jump8_rerun_historical_frame/`
- Alignment provenance: `mouth_only_jump6_alignment.json`
- Full-session QC video: `mouth_only_jump6_qc_2x2_full_stride10.mp4`
- Midpoint visual check: `mouth_only_jump6_qc_2x2_midpoint.png`
- Review/motion audit notebook:
  `gate_branch_diagnostic/observation_quality_and_motion_audit.ipynb`

The video covers source frames 0--348,599 at stride 10. It contains 34,860
frames, is 1200x1200 H.264/yuv420p, runs for 580.998 seconds, and is 388,995,404
bytes. Geometry IDs are deliberately labeled G1/G2/G3 because only the female
has sparse biological-identity ground truth.

## Caveats

- Female GT cannot detect swaps between the two males.
- The female annotations are sparse, so observed switch counts are lower bounds.
- Motion smoothness is not identity truth when the underlying SLEAP observation
  is missing, shared, duplicated, or merged.
