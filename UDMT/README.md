# UDMT Adapter For Medaka Identity Tracking

This directory contains a small, scriptable adapter inspired by UDMT
(`cabooster/UDMT`) for this medaka pipeline. It is not a vendored copy of the
official UDMT GUI, XMem stack, or ST-Net training code. The goal here is to
borrow the parts that fit the current SLEAP -> top-camera ID -> 3D binding
workflow:

- automatic quality scoring for candidate 2D identity tracks,
- mask-centroid localization refinement,
- smooth crossing-event detection for 2D or 3D tracks,
- bidirectional forward/backward consensus repair.

The production 3D path remains unchanged. These tools produce CSVs that can be
used as the `--top-tracks-csv` input to
`scripts.reconstruction_3d.reconstruct_from_slp --bind-top-id`.

## What Was Verified From Official UDMT

The official repository was inspected from `https://github.com/cabooster/UDMT`.
Relevant source locations in that project:

| UDMT stage | Official files observed |
| --- | --- |
| Project creation | `udmt/create_project/new.py`, `udmt/gui/tabs/create_project.py` |
| Tracking initialization | `udmt/gui/tabs/tracking_initialization.py`, `udmt/gui/tabs/xmem/interactive_demo.py` |
| Initial masks | `udmt/gui/tabs/xmem/*`, `udmt/gui/tabs/initial_masks_inference/*` |
| Training dataset creation | `udmt/gui/tabs/create_training_dataset.py`, `ST_Net/pytracking/utils/convert_to_train_set.py` |
| ST-Net training | `udmt/gui/tabs/train_network.py`, `ST_Net/ltr/run_training.py`, `ST_Net/ltr/train_settings/dimp/transformer_dimp.py` |
| ST-Net model | `ST_Net/ltr/models/tracking/dimpnet.py`, `ST_Net/ltr/models/target_classifier/transformer.py` |
| Inference and tuning | `udmt/gui/tabs/analyze_videos.py`, `ST_Net/pytracking/run_tracker.py` |
| Post-processing/video output | `ST_Net/pytracking/utils/filter_and_save.py` |
| Manual refinement | `udmt/gui/tabs/refine_tracklets.py`, `udmt/gui/tabs/udmt_refine_gui/refine_win.py` |

Verified implementation details that matter for this adapter:

- Users initialize tracking by selecting animals in the first frame.
- GUI training-dataset creation samples up to 2000 frames after resize and
  downsample settings.
- Automatic tuning uses `search_scale_range=np.arange(1.5, 3, 0.5)` and
  `target_sz_bias_range=[-0.1, 0, 0.1]` before final tracking.
- The ST-Net code builds a DiMP-style tracker with a ResNet-50 backbone,
  target-classifier feature extractor, transformer module, filter optimizer,
  and Atom IoU regressor.
- `transformer_dimp.py` trains with `LBHinge` for target classification and
  `KLRegression` in the objective dictionary, although the checked loss weight
  for `bb_ce` is zero in that settings file.
- Official installation is Linux-first with Python 3.8, PyTorch/CUDA-specific
  dependencies, XMem/SAM checkpoints, and Precise ROI pooling. Windows is
  documented as secondary/not recommended.

Anything not implemented in this local directory should be treated as an
external UDMT feature, not as a hidden dependency.

## Local Data Flow

```text
SLEAP or segmentation top-camera tracks
  -> UDMT score/select candidate CSVs
  -> UDMT refine centers with mask centroids
  -> UDMT detect smooth close-contact crossings
  -> UDMT repair suspicious rows with a backward pass, or export crossing windows for review
  -> reconstruct_from_slp --bind-top-id --top-tracks-csv <refined/repaired.csv>
```

The local CSV schema is intentionally compatible with
`scripts.reconstruction_3d.bind_topcam_id`:

```text
frame, track_id, x, y, score, node
```

Extra columns such as `identity`, `track_label`, `bind_weight`,
`identity_trajectory_id`, and UDMT diagnostic columns are preserved.

## Commands

Score one candidate:

```bash
python -m UDMT score --tracks top_cam.tracks.csv --summary top_cam.udmt_score.json
```

Rank several precomputed candidate runs:

```bash
python -m UDMT select run_a.csv run_b.csv run_c.csv \
  --summary udmt_selection.json \
  --table udmt_selection.csv
```

Refine SLEAP/NN track centers with segmentation mask centroids:

```bash
python -m UDMT refine \
  --tracks top_cam.tracks.csv \
  --masks blob_candidates.reviewed.csv \
  --output top_cam.udmt_refined.tracks.csv \
  --summary top_cam.udmt_refined.summary.json
```

Repair a forward pass using a backward pass:

```bash
python -m UDMT repair \
  --forward top_cam.forward.tracks.csv \
  --backward top_cam.backward.tracks.csv \
  --output top_cam.udmt_repaired.tracks.csv \
  --summary top_cam.udmt_repaired.summary.json \
  --votes top_cam.udmt_forward_backward_votes.csv
```

Detect smooth crossing windows in top-camera tracks:

```bash
python -m UDMT crossings \
  --tracks top_cam.tracks.csv \
  --events top_cam.udmt_crossings.csv \
  --annotated top_cam.udmt_crossing_annotated.tracks.csv \
  --id-column track_id \
  --coordinate-columns x y \
  --distance-threshold 35 \
  --min-duration 2 \
  --merge-gap 10
```

Detect smooth crossing windows in 3D geometry tracks. For the validated K_1_2
round5 3D tracks, start near 18 mm with a merge gap around 10 frames to keep
nearby fragments together:

```bash
python -m UDMT crossings \
  --tracks round5_260611_3d.csv \
  --events round5_260611_3d.udmt_crossings.csv \
  --annotated round5_260611_3d.udmt_crossing_annotated.csv \
  --id-column fish_id \
  --coordinate-columns X Y Z \
  --distance-threshold 18 \
  --min-duration 2 \
  --merge-gap 10
```

Write a tighter review subset while preserving the same high-recall detector
settings:

```bash
python -m UDMT crossings \
  --tracks round5_260611_3d.csv \
  --events round5_260611_3d.udmt_crossings.review.csv \
  --id-column fish_id \
  --coordinate-columns X Y Z \
  --distance-threshold 18 \
  --min-duration 2 \
  --merge-gap 10 \
  --review-max-min-distance 6 \
  --review-min-duration 15 \
  --review-only
```

Then bind to 3D:

```bash
.venv/Scripts/python.exe -m scripts.reconstruction_3d.reconstruct_from_slp \
  <experiment> \
  --paths-config configs/paths.local.toml \
  --sleap-subdir round4_260520 \
  --num-candidates 6 \
  --use-visible-node-centroid \
  --xyz-bounds -20 220 --no-shift-to-positive \
  --max-residual-per-fish 5000 \
  --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 \
  --num-individuals 3 \
  --bind-top-id \
  --top-tracks-csv top_cam.udmt_repaired.tracks.csv \
  --output-dir <output_dir> \
  --output-tag round4_260520
```

## Module Boundaries

- `io.py`: schema checks and identity-key normalization.
- `metrics.py`: transparent track quality metrics.
- `tuning.py`: UDMT-style candidate ranking utilities.
- `refinement.py`: mask-centroid localization refinement.
- `bidirectional.py`: crossing-event detection plus forward/backward ID mapping
  and conservative repair.
- `cli.py`: command-line wrapper.

## K_1_2 Crossing Validation Notes

The 3D crossing detector was checked on the full K_1_2 round5 track table
against 40 reviewed montage crossings. The close-contact trigger covered all
40 reviewed crossings at thresholds from 12 to 35 mm, confirming that it catches
smooth swaps that the older speed-spike trigger misses.

The remaining issue is precision: at 18 mm and merge-gap 3, the detector emits
1198 per-pair events, many of which are real but mild or brief approaches. A
larger merge gap around 10 reduces fragmentation (910 events) and the vectorized
pair-distance path makes a full-video pass run in ~3.5 s.

Review-gate finding (validated against the 40 reviewed crossings): make the gate
**distance-primary**. `--review-max-min-distance 6` alone keeps **166** candidates
and still covers **40/40** reviewed crossings; `--review-max-min-distance 3` keeps
**68** at 40/40. Adding `--review-min-duration 15` (the AND form) drops one
genuinely tight crossing — a 3-fish pile-up whose closest pair hit 2.26 mm but for
only 9 frames — giving 39/40. Brief tight passes are real swap risks, so do **not**
hard-filter the review set on duration; recommended starting settings are
`--distance-threshold 18 --merge-gap 10 --review-max-min-distance 6` (duration off,
or as a soft/secondary signal inside the resolver, not a recall gate). Treat event
counts as per-pair windows, not one-to-one matches to montage centers; three-fish
pile-ups create overlapping pair events.

## Implementation Cautions

- This adapter does not train ST-Net and does not load official UDMT
  checkpoints.
- Mask refinement is only as good as the mask table. Reject masks, tiny masks,
  and reflection masks before trusting the output.
- Speed spikes are not the right primary trigger for smooth medaka swaps.
  Use `python -m UDMT crossings` on 3D geometry tracks or top-camera tracks to
  find close-contact windows even when kinematics remain smooth. Use the
  optional review gates to triage tight, sustained candidates without changing
  the underlying high-recall event table.
- Bidirectional repair is conservative by default: it only modifies motion
  anomalies, or crossing rows when `--repair-crossings` is passed, after the
  forward/backward ID mapping is established. Normal consensus uses a tight
  same-frame gate; anomalous/crossing rows can use a larger repair gate because
  the forward point itself may be the bad observation.
- The full UDMT-style "which fish swapped" majority vote is not implemented
  here yet. For now, crossing windows are detection/review artifacts unless a
  suitable backward pass and medaka-specific vote signals are supplied.
- For round4 production 3D, keep `--num-candidates 6` and the existing
  reconstruction flags. Use this adapter only to improve the top-camera
  identity anchor.
