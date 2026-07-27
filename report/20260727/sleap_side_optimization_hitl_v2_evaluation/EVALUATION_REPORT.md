# Locked HITL-v2 inference evaluation

Date: 2026-07-27

## Outcome

Round 8 remains the production model for both top and side views.

For the side optimization grid:

- `sigma15__control` is the best localization-focused challenger
  (`mouth p95=4.599 px`, `tail p95=6.465 px`), but it loses detection
  (`0.938`) and overall OKS (`0.611`) relative to Round 8
  (`detect=0.954`, `OKS=0.689`).
- `sigma20__control` matches Round 8's side detection (`0.954`) but has
  worse localization (`mouth/tail p95=6.175/7.681 px`) and OKS (`0.597`).
- The crop448/rot90/strict-mouth/OHKM-off challenger improves the
  centered-instance oracle tail p95 from `8.070` to `5.358 px`, but that
  advantage does not survive full top-down inference. It is not promoted.

The practical conclusion is to keep sigma 1.5 + unchanged centered-instance
control as a future localization condition, keep sigma 2.0 + control as the
detection reference, and retain Round 8 for production.

## Locked benchmark

- Source frames: 40
- View frames: 120
- Human instances: 360 (exactly 3 per view)
- Fully labeled instances: 345
- Partial instances: 15
- Embedded frames readable after reload: 120/120
- Skeletons: 1
- Tracks: 0
- Predicted instances: 0

Artifact hashes:

- Base HITL export:
  `b4f6253ba8a30106a6448cc26234081b31553e043b1661f950e13b3a702ee6da`
- Replacement annotation:
  `fe20c17da43eb9424e469193ce77b3b059d7331269bce1cdfd10323ab07ee1f7`
- Locked package:
  `58b2788c9fa7b1ddf6eddaeac23665619c40a3bd9978c55602a850b6319ccec4`
- Evaluation manifest:
  `eec7542e55f03d1963c58cca8abd4f17589c7aa60c83fc7b41de6e98ab424ba7`

The locked package and its 360-row instance ledger are stored outside git in:

`prediction_replacement2/finalized_hitl_v2/`

The checked-in evaluation manifest and provenance are exact copies of the
locked outputs.

## Side optimization gate

Component modes use the 74 side views where every mouth anchor is finite.
Full top-down evaluation uses all 80 side views.

### Centroid-only

| Condition | Detect | Merge | Mouth p95 (px) | OKS |
|---|---:|---:|---:|---:|
| sigma 2.0 | 0.973 | 0.032 | 5.638 | 0.319 |
| sigma 1.5 | 0.955 | 0.054 | 5.166 | 0.329 |

Sigma 2.0 is the better detector. Sigma 1.5 is the better localization
condition, but its additional misses are real, especially on K_1_2.

### Oracle centered-instance

| Condition | Detect | Mouth p95 (px) | Tail p95 (px) | OKS |
|---|---:|---:|---:|---:|
| unchanged control | 1.000 | 0.672 | 8.070 | 0.731 |
| crop448/rot90/jitter0/strict-mouth/OHKM-off | 1.000 | 0.619 | 5.358 | 0.733 |

The challenger substantially reduces the K_1_2 oracle tail p95
(`32.227 -> 7.072 px`). This comparison is a bundled-condition comparison,
not an OHKM-only ablation: crop, rotation, mouth filtering, and OHKM all differ,
and the OHKM-on twin did not finish cleanly.

### Full top-down

| Condition | Detect | Merge | Swap | Mouth p95 (px) | Tail p95 (px) | OKS |
|---|---:|---:|---:|---:|---:|---:|
| sigma 2.0 + control | 0.954 | 0.046 | 0.005 | 6.175 | 7.681 | 0.597 |
| sigma 1.5 + control | 0.938 | 0.062 | 0.000 | 4.599 | 6.465 | 0.611 |
| sigma 2.0 + OHKM-off bundle | 0.958 | 0.050 | 0.005 | 6.366 | 7.771 | 0.592 |
| sigma 1.5 + OHKM-off bundle | 0.942 | 0.067 | 0.000 | 5.295 | 6.866 | 0.607 |

The unchanged centered-instance control is better in both sigma pairings.
The oracle improvement therefore does not justify promoting the bundled
OHKM-off challenger.

## Round 7 versus Round 8 baseline

### End-to-end top view (40 views)

| Round | Detect | Merge | Mouth p95 (px) | Tail p95 (px) | OKS |
|---|---:|---:|---:|---:|---:|
| Round 7 | 0.958 | 0.033 | 3.690 | 9.662 | 0.751 |
| Round 8 | 0.975 | 0.025 | 3.483 | 9.328 | 0.805 |

### End-to-end side views (80 views)

| Round | Detect | Merge | Swap | Mouth p95 (px) | Tail p95 (px) | OKS |
|---|---:|---:|---:|---:|---:|---:|
| Round 7 | 0.946 | 0.050 | 0.005 | 5.587 | 7.606 | 0.599 |
| Round 8 | 0.954 | 0.037 | 0.000 | 5.231 | 6.712 | 0.689 |

Round 8 wins both production comparisons. The side improvement is especially
large on K_1_2 (`OKS 0.538 -> 0.648`; `tail p95 35.521 -> 16.936 px`).

## Interpretation of the proposed changes

- **Crop 448, rotation ±90, strict mouth filtering, OHKM-off:** useful as an
  oracle diagnostic bundle, but not promoted end-to-end. The present test
  cannot assign causality to any one member of the bundle.
- **Remove mouth-missing samples from centered-instance training:** still
  structurally sensible because mouth is the crop anchor. It was not isolated
  here, so retain it as a training-data policy rather than a proven promotion.
- **Crop-position jitter ±8/16 px:** do not add to the locked shortlist.
  The earlier diagnostic rejected these conditions, and the locked challenger
  intentionally used jitter 0.
- **Centroid confidence-map sigma:** retain both 1.5 and 2.0. Sigma 1.5 is the
  localization condition; sigma 2.0 is the detection condition.
- **More mouth-anchor centroid data:** supported as the most useful next data
  investment. Detection remains the limiting stage in full inference.
- **Plateau -40:** not isolated by this inference test; no conclusion can be
  made about that scheduler setting.

## Reproducibility

The raw evaluator JSON files contain model paths, per-experiment metrics,
per-camera metrics, partial/full-label subsets, and every match record. The
`*_matches.csv` files are the flattened match ledgers. Native prediction SLP
caches are intentionally ignored by git but remain selection-hashed locally
for exact retries.
