# SLEAP top-down optimization: Phase 1 common benchmark

Date: 2026-07-24
Scope: Phase 1 only; no training, parameter tuning, checkpoint changes, source
annotation edits, or full-length inference were performed.

## Outcome and decision

Phase 1 produced a deterministic manifest, common Round 7/Round 8 evaluator,
partial-label-aware association, reusable prediction caches, machine-readable
metrics, and a manual visual-QC GUI.

The audit found **no evaluation cohort genuinely unseen by both models**.
The Round 8 pilot event is a valid Round 8 fine-tune holdout, but six
synchronized Round 7 source frames occur in that event/guard interval, Round 8
was warm-started from Round 7, and the pilot annotations were initialized from
Round 7 predictions. It is therefore classified
`held_out_round8_but_seen_round7`, not as a common locked test.

**Do not promote Round 7 or Round 8, and do not start Phase 2 ablations, from
these numbers alone.** The results are useful for regression diagnosis and for
designing the genuinely locked benchmark that is still required.

## Complete optimization roadmap

1. Establish a common benchmark and visual-QC tooling. **Completed here.**
2. Decompose centroid versus centered-instance bottlenecks with full inference
   and oracle-mouth crops.
3. Run side-centroid parameter ablations.
4. Run centered-instance crop, augmentation, loss, and hard-keypoint ablations.
5. Optimize inference thresholds and centroid/instance model pairing.
6. Apply a fixed-clip 3D reconstruction and identity-QC promotion gate.

Top and side views remain separate throughout; the side view is the priority.

## Data and provenance audit

The complete machine-readable audit is
[`data_provenance_audit.json`](data_provenance_audit.json).

| Cohort class | Rows | Independent source frames/events | Use |
|---|---:|---:|---|
| Genuinely held out from both | 0 | 0 | Not available |
| Held out from Round 8, seen by Round 7 | 60 | 20 frames / 1 event | Round 8 fine-tune regression only |
| In-sample stress/regression | 126 | 57 events across 3 experiments | Diagnostic regression only |
| Unknown provenance | 0 | 0 | None in the manifest |

Verified source facts:

- Round 7 contains 1,295 top frames/3,885 instances and 2,590 side
  frames/7,771 instances. Its existing comparisons and merged-label stress
  evaluations are in-sample.
- Its saved component validation splits are randomized rather than event-aware.
  Top centroid and centered-instance validation have no common frames; side has
  only 23. They cannot serve as the common locked test.
- Round 8 finalization is pilot-only: 20 held-out source frames (60 view rows)
  and 93 usable training source frames (279 view rows). Strict model names and
  best epochs were verified from the runs/finalization metadata.
- The corrected expansion-240 annotation package does not exist. The
  active-learning artifact root contains the pilot seed, pilot edits, and pilot
  full package only; no larger locked set was fabricated.
- The pilot sentinel is frames 96,612-97,802, with a Round 8 training guard of
  96,446-97,968. Round 7 includes synchronized labels at 96,484, 96,805,
  96,956, 97,195, 97,375, and 97,825.
- Strict centroid eligibility is 19/20 top views and 29/40 side views; only
  8/20 synchronized source frames are eligible in all three views.

There is also strong annotation-reference bias. At a 0.01 px exact-coordinate
threshold, Round 7 predictions equal the pilot reference for 84/107 side
mouths, 83/109 side tails, 44/59 top mouths, and 38/57 top tails. Round 8 is
essentially zero on the same diagnostic. This is consistent with the pilot
annotations having been seeded from Round 7 predictions. The near-zero Round 7
median errors must not be interpreted as independent evidence of superiority.

## Deterministic benchmark

[`benchmark_manifest_phase1_v1.csv`](benchmark_manifest_phase1_v1.csv) contains
186 view rows. Selection uses manifest version `phase1-v1`, seed `20260724`,
and stable SHA-256 frame keys. Event/window IDs and a 166-frame guard prevent
nearby frames from crossing roles.

The manifest records stable frame key, experiment, view/camera, raw video,
frame index, source SLP and provenance, cohort/split/provenance class,
event bounds, difficulty, per-node visibility, instance count, three-mouth
finiteness, strict-centroid eligibility, and manual-review status.

Convenience subsets are:

- [`benchmark_manifest_pilot_event_holdout_v1.csv`](benchmark_manifest_pilot_event_holdout_v1.csv):
  60 rows from the single Round 8 holdout event.
- [`benchmark_manifest_in_sample_stress_v1.csv`](benchmark_manifest_in_sample_stress_v1.csv):
  126 rows across K_1_1, K_1_2, and K_1_3.
- [`future_locked_test_annotation_queue_v1.csv`](future_locked_test_annotation_queue_v1.csv):
  720 candidates requiring annotation/provenance screening.
- [`future_locked_test_annotation_queue_eligible_v1.csv`](future_locked_test_annotation_queue_eligible_v1.csv):
  111 currently eligible view rows = 37 synchronized frames from 21 K_1_2
  events. These are candidates, not ground truth or a locked test.

The generator is `scripts/sleap/benchmark_manifest.py`; source packages remain
unchanged.

## Matching and metric corrections

The evaluator now performs one global one-to-one Hungarian assignment per
frame. It:

- excludes missing/hidden nodes from numerical costs;
- prefers the semantic mouth anchor when jointly visible;
- otherwise uses only jointly visible nodes;
- records `both`, `mouth`, `tail`, or `unmatchable` as the match basis;
- keeps fully labeled and partial-instance denominators separate;
- records match coordinates and failure flags in the CSV/JSON evidence.

Regression cases confirm the intended behavior:

- side1 frame 97,219 retains the genuine Round 8 tail error of about 45.2 px;
- side2 frame 97,340 no longer reports an artificial approximately 109 px
  error; finite matched tail errors are about 0.19 and 5.53 px, while an
  unmatchable/missed partial instance stays explicit;
- side2 frame 97,630 is about 2.21 px rather than approximately 51 px.

Thus a partial-label matching artifact no longer masquerades as a keypoint
failure.

## Common comparison results

All values below use the exact same manifest rows for both rounds. Counts and
full medians/p90/p95/p99, PCK@5/PCK@10, swap/shared-tail numerators, per-view,
per-experiment, per-difficulty, and full/partial splits are in
[`round7_vs_round8_metrics_summary.csv`](round7_vs_round8_metrics_summary.csv)
and the corresponding JSON/match CSV files.

### Round 8 pilot event: complete top-down inference

This is a **single-event, Round 7-exposed, Round 7-seeded-reference cohort**.
It is shown for diagnosis, not promotion.

| View | Round | Precision | Recall | F1 | Missed / extra | All 3 detected | Mouth p95 px | Tail p95 px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Side | R7 | 0.982 | 0.925 | 0.953 | 9 / 2 | 31/40 | 5.87 | 7.57 |
| Side | R8 | 0.991 | 0.917 | 0.952 | 10 / 1 | 30/40 | 4.27 | 6.06 |
| Top | R7 | 0.894 | 0.983 | 0.937 | 1 / 7 | 19/20 | 4.04 | 6.24 |
| Top | R8 | 1.000 | 0.967 | 0.983 | 2 / 0 | 18/20 | 3.10 | 8.81 |

Side full-top-down fully labeled instances were 107/108 matched for R7 and
106/108 for R8. Each round matched 4/12 partial instances. The partial tail
p95 was 106.68 px for R7 and 8.68 px for R8, but those denominators are only
four and include real hard tail failures.

### Detector-only decomposition

| View | Round | Precision | Recall | F1 | Missed / extra | All 3 detected | Mouth p95 px |
|---|---|---:|---:|---:|---:|---:|---:|
| Side | R7 | 0.947 | 0.892 | 0.918 | 13 / 6 | 28/40 | 5.28 |
| Side | R8 | 0.955 | 0.883 | 0.918 | 14 / 5 | 27/40 | 4.05 |
| Top | R7 | 0.894 | 0.983 | 0.937 | 1 / 7 | 19/20 | 3.79 |
| Top | R8 | 1.000 | 0.967 | 0.983 | 2 / 0 | 18/20 | 2.80 |

Round 8's side detector has slightly better precision and mouth p95 but one
additional miss, lower recall, and lower all-three coverage. That tradeoff is
important for the side-priority optimization objective.

### Oracle-mouth centered-instance decomposition

| View | Round | Mouth p95 / PCK@5 | Tail p95 / PCK@5 | Swaps | Shared tails |
|---|---|---:|---:|---:|---:|
| Side | R7 | 0.681 / 1.000 | 6.45 / 0.941 | 0/107 | 0/40 |
| Side | R8 | 0.688 / 0.991 | 8.60 / 0.924 | 0/107 | 2/40 |
| Top | R7 | 0.711 / 1.000 | 6.21 / 0.879 | 0/57 | 1/20 |
| Top | R8 | 0.720 / 1.000 | 6.18 / 0.915 | 0/58 | 0/20 |

Oracle-mouth results make the bottleneck clearer: mouth localization is
similar, while the side Round 8 centered-instance tail distribution and two
shared-tail frames remain concerns. The 12 partial side instances have small
denominators and much heavier tail outliers; the JSON keeps those separate.

### Multi-experiment in-sample stress regression

| View | Round | Recall | Missed / extra | Mouth p95 px | Tail p95 px |
|---|---|---:|---:|---:|---:|
| Side | R7 | 0.955 | 11 / 0 | 4.58 | 10.10 |
| Side | R8 | 0.943 | 14 / 2 | 4.37 | 9.88 |
| Top | R7 | 0.902 | 13 / 0 | 3.26 | 15.32 |
| Top | R8 | 0.894 | 14 / 0 | 3.08 | 15.13 |

The aggregate difference is mixed, and the per-experiment breakdown matters:

- side K_1_2 tail p95 regresses from 10.92 px (R7) to 22.87 px (R8);
- top K_1_2 recall regresses from 0.889 to 0.844;
- top K_1_3 recall improves from 0.933 to 0.956 and tail p95 from 15.56 to
  12.48 px;
- other experiment/view changes are smaller or mixed.

These are deliberately labeled in-sample regression results and cannot measure
generalization.

No bootstrap confidence interval is reported for the pilot because it contains
one event. The 57-event stress set is training/in-sample evidence; an
inferential interval there would imply generalization that the provenance does
not support. Event-bootstrap intervals should begin with the new locked cohort.

## Reusable predictions and visual QC

Sixteen selection-hashed native `.slp` caches cover R7/R8, top/side, pilot
full/centroid/oracle modes, and full in-sample stress inference. They are under
[`predictions/`](predictions/), and all 16 were reload-tested. Matching evidence
is also saved in each `*_matches.csv`, so overlays do not require inference.
Multi-source caches use sequential video references to avoid ambiguous embedded
SLP video IDs.

The GUI is `scripts/sleap/benchmark_visual_qc.py`. It supports side-by-side and
toggle views, GT/R7/R8 skeleton overlays, numerical errors and match bases,
filters/sorts, next/previous/jump, zoom/pan, shortcuts, review labels, notes,
atomic autosave/resume, and headless PNG export.

Use [`VISUAL_QC_GUI_RUNBOOK.md`](VISUAL_QC_GUI_RUNBOOK.md). The review ledger is
[`manual_visual_qc.csv`](manual_visual_qc.csv) with provenance/counts in
[`manual_visual_qc.json`](manual_visual_qc.json). It is currently 0/60 reviewed.
The inspected example is
[`visual_qc_exports/side1_frame_97219_round7_vs_round8.png`](visual_qc_exports/side1_frame_97219_round7_vs_round8.png);
it visibly preserves the genuine approximately 45.2 px R8 tail error.

## Validation status

Focused Phase 1 tests:

```text
50 passed
```

They cover deterministic selection, event-level leakage prevention,
partial-label matching and known-case geometry, metric denominators/subsets,
strict-mouth run discovery, GUI ledger serialization/resume, and headless
rendering.

Full local suite with installed dependencies and optional third-party pytest
autoload disabled:

```text
698 passed, 4 warnings
```

The exact CI environment was also reproduced with Python 3.11.15 and
`environment/requirements.txt`:

- literal root `pytest -q`: collection stops on 78 pre-existing unreadable
  generated `report/` and `runs/` directories present in this working copy;
  these are not Phase 1 test failures and are absent from a clean checkout;
- tracked `tests/` in that same environment: 687 passed, 9 failed, 2 skipped.
  All nine failures are the known pre-existing `pyarrow` dependency gap:
  tests call Parquet I/O, but `environment/requirements.txt` omits `pyarrow`.
  No Phase 1 test failed.

The temporary Python 3.11 environment was removed after validation. Three
test-created `.pytest-tmp-phase1-*` directories could not be removed because
their Windows ACL denies the repository owner, matching the pre-existing
pytest-temp ACL issue.

## Required scientific decision

Before Phase 2, choose the breadth of the common locked test:

1. **Recommended:** reserve and manually annotate at least 20 independent,
   event-disjoint windows per experiment across K_1_1, K_1_2, and K_1_3, with
   synchronized top/side views and explicit ordinary, contact/crossing, and
   partial/occluded quotas. Use the 21 currently eligible K_1_2 events as
   candidates and mine equivalent untouched K_1_1/K_1_3 events. Freeze their
   hashes before any new training.
2. Narrower option: annotate only the 37 eligible K_1_2 source frames. This is
   faster, but the locked claim would be K_1_2-specific and could not support
   multi-experiment promotion.

The manual annotations—not model-seeded coordinates—must be reviewed in the
GUI and assigned a final include/exclude decision. Phase 2 should begin only
after that choice and freeze are complete.
