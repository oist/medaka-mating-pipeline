# 3D Multi-Object Tracking With Identity in a 200 mm Cubic Workspace

## Purpose

This document describes a computer-vision pipeline for reconstructing and maintaining identities for three visually similar moving objects inside a 200 mm cubic workspace. The system uses synchronized multi-view video, 2D landmark predictions, instance segmentation, camera calibration, temporal linking, and identity classifiers.

The core problem is not only to estimate 3D positions. It is to maintain stable object identity over a 30-40 minute recording, including periods where objects move close together, become partially hidden in one view, merge into ambiguous image regions, or temporarily produce missing detections.

The main design goal is a production pipeline that can:

- recover 3D object coordinates from three camera views,
- assign one stable identity to each physical object,
- detect and repair identity swaps,
- use segmentation and learned identity models as supporting evidence,
- expose uncertain periods for human review,
- preserve intermediate CSV, JSON, H5, image, and video artifacts for debugging.

## Data Sources

Each recording contains three synchronized camera streams:

- `top_cam`: one camera looking from above the workspace,
- `side1_cam`: first side-view camera,
- `side2_cam`: second side-view camera.

For every frame, the current landmark detector produces up to three predicted instances per view. Each instance has two 2D landmarks. In this document these are called:

- `primary_point`: the landmark used as the preferred precise 2D anchor,
- `secondary_point`: an additional landmark that can define a short object axis when reliable.

The segmentation stack provides instance masks:

- SAM2 is used as an interactive or seeded mask-propagation backend.
- Detectron2 is trained from reviewed or pseudo-labeled masks and then used for denser inference.

The calibration system maps between image coordinates and physical coordinates:

- A rigid 3D bead object with known point coordinates is imaged by all three cameras.
- Manual or assisted 2D bead annotations are paired with known 3D coordinates.
- Polynomial regression models learn the mappings:
  - three-view 2D image coordinates to 3D coordinates,
  - 3D coordinates back to each camera view for reprojection QC and overlays.

## Coordinate And Identity Model

The pipeline separates three concepts that are easy to mix up:

- `detection_id`: an instance index emitted by a detector in one frame.
- `trajectory_id`: a temporal track identifier produced by a linker.
- `final_object_id`: the stable identity label intended to represent one physical object across the full recording.

The 3D output table is organized around:

```text
frame, trajectory_id, X, Y, Z
```

The identity-bound output adds:

```text
final_object_id, bind_source, bind_score, source_bind_key
```

The distinction matters because short fragments may belong to the same physical object. A production identity pipeline should pool evidence by stable identity, not only by temporary fragment number.

## Calibration

Calibration is treated as a learned correspondence problem rather than a pinhole-camera derivation.

The calibration input is:

```text
known 3D bead coordinates
matched 2D bead coordinates in top_cam, side1_cam, side2_cam
```

The learned calibration artifacts include:

- a model from multi-view image coordinates to workspace coordinates,
- optional pairwise helper models,
- a model from workspace coordinates back to image coordinates,
- QC reports and plots measuring 3D error and 2D reprojection error.

The polynomial approach is useful because the real setup includes optical effects and non-ideal camera geometry. Instead of explicitly modeling every optical component, the regression absorbs systematic distortion as long as the calibration object samples the workspace well.

Important calibration checks:

- calibration points should cover the full 200 mm cube,
- reprojection error should be inspected in every camera,
- held-out calibration points should not regress when calibration is refined,
- extrapolation outside the calibrated volume should be rejected or flagged.

## Baseline 3D Reconstruction

The baseline 3D reconstruction starts from SLEAP predictions in all three views.

For each frame:

1. Extract candidate 2D landmarks from each camera.
2. Enumerate valid cross-view combinations.
3. Use the calibration model to convert each three-view combination into a 3D candidate.
4. Reject candidates outside the workspace bounds.
5. Score candidates by reprojection consistency and pairwise calibration consistency.
6. Select a mutually exclusive set of three 3D candidates.
7. Link selected candidates over time into 3D trajectories.

The mutually exclusive selection is important. When a detector returns extra candidates, the reconstruction stage should not simply take the top three detections in each view. It should choose a globally consistent triplet set where each image-space detection is used at most once.

The validated production-style reconstruction uses:

- extra 2D candidates when available,
- workspace bounds to reject impossible 3D points,
- greedy mutually exclusive triplet selection for speed,
- 3D temporal linking with a maximum jump gate,
- a hard cap on gap-bridging distance so a long missing segment cannot attach to a different object.

## Temporal Linking

Temporal linking converts per-frame 3D candidates into trajectories.

The simple version uses frame-to-frame nearest-neighbor matching with a distance gate. This works well when objects are separated and detections are stable. It fails when objects approach each other because a short nearest-neighbor decision can flip identities.

The improved temporal linker uses:

- last valid position,
- estimated velocity,
- maximum jump per frame,
- hard maximum jump over a gap,
- forward and backward consistency checks,
- optional spline or linear filling after identity assignment.

The rule is conservative: gap filling should happen after identity decisions, not before. Interpolating too early can hide an identity swap and make a wrong trajectory look smooth.

## Top-View Identity Binding

The original production identity anchor is a 2D track from the top camera. After 3D trajectories are reconstructed, the 3D points are projected back into the top view and matched to the top-view 2D identity tracks.

This creates a mapping:

```text
3D trajectory_id -> top-view identity track -> final_object_id
```

A global one-to-one vote assigns stable identities. Frames where the top view is weak can still inherit identity from 3D geometry and are marked by `bind_source`.

This design keeps geometry and identity separate:

- 3D reconstruction estimates where objects are.
- 2D identity binding decides which physical object each trajectory represents.

## Segmentation-Assisted Extension

Landmark-only tracking becomes fragile when objects are close, when a detector misses one instance, or when an instance jumps to a visually similar neighbor.

To improve this, we added a segmentation-assisted identity pipeline:

1. Mine clean seed windows from high-confidence 3D identity output.
2. Project clean 3D points into a camera view.
3. Use those projected points as SAM2 prompts.
4. Propagate instance masks through a short chunk.
5. Convert masks into tracklets.
6. Render QC overlays.
7. Review masks and identity sanity in Napari.
8. Export reviewed labels.
9. Export COCO pseudo-labels.
10. Train Detectron2 on reviewed or pseudo-reviewed masks.
11. Run Detectron2 inference densely on the same camera view.
12. Link segmentation outputs into tracklets.
13. Create object crops for identity-model training and evaluation.

The segmentation system is not meant to replace 3D geometry. It supplies additional evidence:

- object outline,
- mask area and shape,
- crop content,
- persistence through partial detection failure,
- identity evidence near difficult events.

## Side-View Pipeline

The side-view pipeline generalizes seed prompt mining from one camera to any of:

```text
top_cam
side1_cam
side2_cam
```

For each requested view, clean 3D identity points are projected into that view and used to seed segmentation. This allows side-camera segmentation models to be trained from 3D-supported pseudo-labels without manually drawing all masks.

For the first side-view production pass, the output per view is:

- SAM2 mask H5,
- raw segmentation tracklets,
- QC tables,
- QC overlay videos,
- Napari review CSV,
- reviewed clean tracklets,
- COCO pseudo-label export,
- Detectron2 inference masks,
- linked segmentation tracklets,
- identity crop dataset,
- contrastive identity report.

The two side views can also be pooled for a joint identity model, but matching identities across cameras must be checked carefully. The 3D-bound identity label is the reference source for the pilot.

## Human Review

Human review is used where automatic evidence is uncertain, not as a replacement for the pipeline.

The Napari review tool displays:

- video frame,
- instance mask layer,
- tracklet ID,
- QC status,
- projected 3D identity reference points when available,
- optional identity assignment columns.

Review labels include:

```text
good
partial
uncertain
merged
missed
wrong_object
background
```

Keyboard shortcuts were added so a reviewer can move frame-by-frame and label common cases without mouse interaction. The review output is a CSV consumed by the existing QC filtering flow. Manual CSV editing is intentionally not part of the human workflow.

## Identity Models

We tested identity classification as a support layer for difficult tracking periods.

The tested identity-model ideas include:

- direct multi-class classification on clean crops,
- contrastive representation learning,
- camera-specific identity embeddings,
- pooled side-view identity embeddings,
- permutation-corrected evaluation,
- tracklet-majority evaluation,
- margin-reject evaluation.

The most useful role for the identity model is not to override every frame. It is to confirm or repair identity around critical events.

The intended repair pattern is:

```text
clean segment before event
uncertain or merged event
clean segment after event
```

If the identity model strongly agrees before and after the event, the middle segment can be marked as an identity bridge. If high-confidence identity evidence conflicts across the event, the event is flagged for review instead of being silently repaired.

This became the identity-sandwich audit:

- find clean tracklets before and after a critical point,
- compare their identity evidence,
- fill weak middle segments only when both sides agree,
- flag conflicting evidence,
- export event tables and rendered review videos.

## Motion-Aware 2D Tracking

A recent issue was zigzagging trajectories when two or three objects are close. The root cause is early 2D assignment: nearest-neighbor matching can make a wrong local decision during close contact. Once a wrong 2D identity track is used as the binding anchor, the 3D identity output can inherit the error.

We added an optional motion-aware mode to the early 2D tracker.

The motion-aware tracker uses:

- `primary_point` as the 2D coordinate,
- constant-velocity prediction,
- rolling speed,
- heading change,
- angular velocity,
- tangent acceleration,
- lateral acceleration,
- nearest-neighbor distance,
- close-contact flags,
- ambiguity margin,
- conservative bridging instead of forced assignment.

The intended behavior is:

- accept smooth fast motion when direction is continuous,
- flag high-speed lateral jumps,
- flag sharp heading reversals,
- bridge ambiguous close-contact frames,
- avoid smoothing coordinates as the primary fix.

This is important because smoothing can hide an identity swap. The first priority is correct assignment. Coordinate smoothing should only happen after identity continuity is trusted.

## Motion Parameter Diagnostics

Hard absolute thresholds are too brittle. A single velocity threshold does not work across all recordings because baseline movement, frame gaps, and detection quality vary.

To tune motion-aware tracking systematically, we added a diagnostic stage that writes time-series variables:

- speed,
- velocity components,
- heading,
- heading change,
- angular velocity,
- tangent acceleration,
- lateral acceleration,
- nearest-neighbor distance,
- close-contact state,
- rolling statistics in time windows.

The diagnostic output includes:

- `motion_timeseries.csv`,
- `motion_summary.csv`,
- `weak_thresholds.json`,
- `motion_timeseries.png`,
- `motion_distributions.png`.

The recommended thresholding strategy is weak band-pass filtering:

- very low speed: likely stationary or near-stationary,
- basal speed: normal movement,
- middle speed: evaluate with time-window context,
- high speed: allowed only if direction is continuous,
- extreme speed: likely detection or identity error unless supported by context.

For middle-range values, the tracker should rely more on temporal context:

- sustained speedup over several frames is plausible,
- one-frame lateral displacement is suspicious,
- high angular velocity near close contact is suspicious,
- high tangent acceleration with stable direction may be valid,
- high lateral acceleration plus close contact should be bridged or reviewed.

## What We Tried

The work so far tested several levels of evidence.

### Landmark-Only 2D Tracking

Initial 2D identity tracks used simple Hungarian nearest-neighbor matching. This is fast and works in sparse scenes, but it is vulnerable to close-contact swaps.

We tested anchor choices and confirmed that the coordinate choice changes downstream quality. The current motion-aware tests use the primary landmark as the precise 2D point.

### 3D Geometry Linking

3D reconstruction from the three camera views gives strong spatial constraints. It handles many cases where a single view is ambiguous.

However, geometry alone does not solve all identity issues. If the identity anchor swaps during a difficult section, the final identity assignment can still be wrong.

### Segmentation From 3D Seeds

We used clean 3D identity points to seed SAM2 masks, then converted those masks into pseudo-labels. This gave a way to bootstrap segmentation without drawing all masks manually.

The SAM2 output was useful for short chunks and for creating training data for Detectron2. Human review was needed around merged or partial masks.

### Detectron2 Expansion

Detectron2 was trained on pseudo-labeled masks and used to expand segmentation to more frames.

The motivation was to avoid relying only on sparse seed windows. Dense masks create more tracklets and more crops for identity-model training.

### Side-View Segmentation

We extended seed projection to side cameras. This tested whether side-view masks and crops could support identity evaluation, especially where the top view is ambiguous.

The side-view pipeline produced useful diagnostics, but side-camera identity must be matched through the 3D-bound identity reference before being trusted.

### Contrastive Identity Learning

Contrastive representation learning was tested as an alternative to direct classification. The goal is to learn an embedding where crops from the same object cluster together and crops from different objects separate.

This is useful because literal class labels may be permuted between runs or cameras. The evaluation therefore reports both literal label accuracy and best-remapped accuracy.

### Identity-Sandwich Repair

We implemented a repair audit that checks clean identity evidence before and after a difficult event. If both sides agree, the uncertain middle segment can be bridged. If they disagree, the event is flagged.

This matches the desired production use: identity models support the tracking system at failure points rather than replacing geometry and motion tracking everywhere.

### Motion-Aware Early Tracking

We added motion-aware diagnostics and an initial motion-aware tracker mode. The first parameters were not considered final. They revealed that parameter optimization must be data-driven.

The next tuning step is to sweep weak thresholds using time-series outputs and compare:

- identity fragmentation,
- coverage,
- high-speed jump counts,
- close-contact bridge counts,
- 3D binding quality,
- rendered review videos.

## Current Failure Modes

Known difficult cases include:

- two or three objects close together in one view,
- merged segmentation regions,
- missing landmark detections,
- extra false-positive detections,
- sudden frame-to-frame landmark jumps,
- identity swaps around crossings,
- long gaps where a stale identity could attach to the wrong object,
- segmentation masks that are partial or include background,
- camera-specific appearance differences.

These are handled by different evidence layers:

- geometry rejects impossible 3D combinations,
- temporal linking rejects impossible jumps,
- segmentation helps separate object extent,
- identity embeddings help confirm object consistency,
- review tools expose uncertain cases,
- rendered videos provide final human-facing QC.

## Acceptance Criteria

A production run should produce:

- synchronized per-view landmark inputs,
- calibration QC artifacts,
- 3D candidate and final trajectory CSVs,
- identity-bound 3D CSV,
- bind-source and bind-score diagnostics,
- segmentation masks where available,
- reviewed QC tables,
- identity crop datasets,
- identity-model evaluation reports,
- motion time-series diagnostics,
- rendered videos for difficult events.

The output should make three things inspectable:

1. Where each object was in 3D.
2. Which stable identity each object carried.
3. Why the pipeline trusted, bridged, or rejected each difficult section.

## Recommended Next Steps

The next practical step is parameter optimization for the motion-aware 2D tracker.

Recommended procedure:

1. Mine separate-close-separate windows from long recordings.
2. Compute motion time-series variables for baseline tracks.
3. Estimate weak speed bands from clean non-close frames.
4. Sweep close-contact distance, ambiguity margin, angular threshold, tangent acceleration threshold, and lateral acceleration threshold.
5. For each parameter set, run 2D tracking, 3D reconstruction, identity binding, and event-video rendering.
6. Score each run by both numeric metrics and visual review.
7. Promote only parameter sets that reduce swaps without causing excessive fragmentation.

The key rule is that uncertain frames should be explicit. A short gap with a clear bridge is better than a confident but wrong identity assignment.

