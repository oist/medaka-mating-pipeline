# Round 7 three-way identity-tracking comparison

## Overall assessment: share with caveats

Strict mouth-only 3D reconstruction is the strongest tested method. On the same
298 mapped female annotations, it reduces observed ID changes from 26 to 11 and
raises best-fixed-ID accuracy from 43.29% to 52.35% relative to the current
visible-node-centroid track. It preserves the female ID across three of the four
independently reviewed swap events; the current track preserves none.

The mouth-based 2D-tracklet-first method also preserves three of four reviewed
events, but it does not improve on mouth-only in the event windows and the full
recording implementation did not finish after more than one hour. It should not
replace the production path without a scalability redesign.

## Scope and controlled settings

- Experiment: `20251225_K_1_2_FCMC`
- SLEAP source: `round7_260704`, up to six candidates per view
- Calibration: fixed-interface refractive backend
- Geometry bounds: `-20..220` mm in calibration-native coordinates
- Temporal gates: 8 mm/frame, 80 mm hard cap, 100-frame maximum gap
- Female GT: 300 confirmed annotations from the GUI ledger
- Reprojection matching: each method was matched to the corresponding point on
  the manually selected SLEAP instance—visible-node centroid for the current
  method and mouth for both mouth methods.

The three methods were:

1. **Current centroid:** existing full Round 7 visible-node-centroid geometry and
   framewise 3D Hungarian tracking.
2. **Mouth-only 3D:** full reconstruction using only finite mouth predictions;
   tail-only predictions were rejected rather than used as fallback anchors.
3. **Mouth 2D tracklet-first:** conservative mouth tracklets built independently
   per view, broken when minimum mouth separation was below 0.5 median body
   lengths, then connected by the repository's existing multiview tracklet 3D
   component graph. This implementation is not a window-to-window fragment
   stitcher: it does not score and optimize end-to-start links between broken
   tracklets, and the four bounded outputs were concatenated without boundary
   stitching. It therefore tests the existing graph implementation, not the
   intended windowed fragment-association algorithm described subsequently.

## Headline results

| Metric | Current centroid, full | Mouth-only 3D, full | Mouth 2D tracklet-first, four windows |
|---|---:|---:|---:|
| Eligible confirmed-F annotations | 300 | 300 | 27 |
| Mapped annotations | 299 | 298 | 27 |
| Best fixed-ID accuracy | 43.14% | **52.35%** | 59.26% |
| Observed ID changes | 26 | **11** | 4 |
| Strict two-view annotations | 274 | 268 | 24 |
| Strict best fixed-ID accuracy | 44.89% | **51.87%** | 54.17% |
| Strict observed ID changes | 25 | **11** | 4 |
| All three observed, non-interpolated | 97.85% | **98.16%** | 99.82% within windows |
| Consecutive 3D jump p99 | 5.09 mm | **2.20 mm** | 1.87 mm within windows |
| Reviewed swap events preserving female ID | 0/4 | **3/4** | **3/4** |

The tracklet-first percentages are not full-recording estimates. They describe
26,104 output frames spanning the four reviewed event windows.

## Fair common-cohort checks

On the 298 annotations mapped by both full-recording methods:

| Method | Best fixed-ID accuracy | Observed ID changes |
|---|---:|---:|
| Current centroid | 43.29% | 26 |
| Mouth-only 3D | **52.35%** | **11** |

On the 267 annotations strict in both full-recording methods:

| Method | Best fixed-ID accuracy | Observed ID changes |
|---|---:|---:|
| Current centroid | 45.32% | 25 |
| Mouth-only 3D | **51.69%** | **11** |

On the 27 annotations available in all four event windows:

| Method | Best fixed-ID accuracy | Observed ID changes |
|---|---:|---:|
| Current centroid | 44.44% | 8 |
| Mouth-only 3D | **70.37%** | **2** |
| Mouth 2D tracklet-first | 59.26% | 4 |

## Reviewed event outcomes

An event is preserved when the female is assigned the same method ID at the
first and last strict annotations, irrespective of the arbitrary ID number.

| Audited segment | Current centroid | Mouth-only 3D | Mouth 2D tracklet-first |
|---|---|---|---|
| `R3D:ID_3:S000328`, 72,371→73,027 | swap | **preserved** | **preserved** |
| `R3D:ID_3:S000516`, 96,612→97,802 | swap | swap | swap |
| `R3D:ID_2:S000730`, 174,316→180,515 | swap | **preserved** | **preserved** |
| `R3D:ID_1:S000839`, 273,617→278,182 | swap | **preserved** | **preserved** |

The second event remains the common failure case. Because strict mouth-only
tracking still swaps there, this event is not explained solely by centroid
movement when tail visibility changes. It is the highest-priority scene for
dense review and Round 8 training.

## Tracklet threshold and graph validation

The legacy 1.5-body-length separation gate was not usable in the first event
window: it retained only 1.9% of top, 12.6% of side1, and 10.8% of side2
fish-frames. The predefined test used 0.5 body lengths, targeting near-touching
mouths. Full-recording retained tracklet coverage was:

- top: 76.1%, 5,169 fragments;
- side1: 80.9%, 9,789 fragments;
- side2: 85.0%, 9,129 fragments.

Across the four bounded graph runs, 99.53–100% of tracklet rows matched back to
their exact mouth candidate slots. There were zero reoptimized conflicts and one
geometry-coverage fallback frame in total. The graph inputs were therefore
internally coherent.

The full tracklet-first run exceeded one hour, consumed about 5.6 GB working
memory, and had not written a bootstrap artifact or final CSV when stopped. The
bounded windows completed in approximately 3.5–5.1 minutes each.

## Raw SLEAP visibility contract

Most predictions follow the expected mouth-anchor contract, but the saved SLPs
contain a small tail-only population:

| View | Mouth + tail | Mouth only | Tail only |
|---|---:|---:|---:|
| top | 97.78% | 1.90% | 0.32% |
| side1 | 96.70% | 2.45% | 0.85% |
| side2 | 96.37% | 2.98% | 0.65% |

The strict-mouth ablation excludes these tail-only instances. This distinction
matters because the prior loader silently fell back to the remaining visible
node when the requested mouth anchor was absent.

## Recommendation

1. Use strict mouth-only 3D reconstruction as the Round 8 tracking candidate.
   It provides the clearest identity improvement, slightly improves observed
   three-fish coverage, lowers the high-end motion distribution, and has a
   manageable full-recording runtime of 15.6 minutes.
2. Do not adopt the current full-recording tracklet-first implementation yet.
   It is slower, failed to scale, and did not outperform mouth-only in the
   reviewed windows.
3. Densely annotate the unresolved 96,612–97,802 event and mine similar
   close-contact frames for Round 8. Reserve event-level windows for validation
   rather than splitting adjacent frames between training and validation.
4. Record anchor mode, strict-anchor policy, and all reconstruction CLI settings
   in future provenance; the current provenance records calibration but not all
   tracking options.

## Validation and artifacts

- `95 passed` across concrete-tracklet, multiview-tracklet, tracklet-first,
  reconstruction integration, and related SLEAP tests.
- Final summary: `final_comparison/comparison_summary.json`
- Headline table: `final_comparison/method_summary.csv`
- Common-cohort checks: `final_comparison/pairwise_common_cohorts.csv`
- Event outcomes: `final_comparison/mixed_event_results.csv`
- Full mouth-only output: `full/mouth_only_3d/mouth_only_3d_3d.csv`
- Four-window tracklet output: `windows/mouth_tracklet_first_four_events_3d.csv`
- Node visibility audit: `round7_node_visibility_summary.json`

### Required caveats

- Female GT identifies one biological fish and cannot independently score the
  two males.
- The annotations are sparse, so observed ID changes are lower bounds.
- Tracklet-first was evaluated completely only in the four reviewed windows;
  its full-recording implementation did not finish.
- The tested tracklet-first code does not implement explicit best-pair linking
  of broken fragments across temporal windows. The intended windowed 3D
  tracklet-association method remains untested.
- The 0.5-body-length tracklet gate was selected from a threshold sensitivity
  check, not learned from an independent training cohort.
