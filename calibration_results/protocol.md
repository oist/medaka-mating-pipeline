# Calibration Protocol for 3-Camera 3D Reconstruction

## Overview

This document describes the bead-based calibration procedure used to map
multi-camera 2D pixel coordinates to 3D world coordinates for medaka
mating behaviour tracking. The calibration trains a polynomial regression
model that takes a 6D pixel-triple input
`(cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y)` and outputs a 3D
world position `(X, Y, Z)` in millimetres.

---

## 1. Hardware Setup

### Three-camera arrangement

| Camera | View         | World axes visible | Mapping in tracking pipeline |
|--------|--------------|--------------------|------------------------------|
| cam0   | Top-down     | X, Y               | player1 in processed CSVs    |
| cam1   | Side view 1  | X, Z               | player2                      |
| cam2   | Side view 2  | Y, Z               | player3                      |

The cameras are fixed; only the calibration object moves between data
collection steps ("ways").

### Calibration object

A rigid 190 x 190 x 190 mm cube with:

- **49 coloured beads** (7 colours x 7 beads) embedded on 7 vertical
  columns at fixed positions inside the cube. Bead (X, Y) positions
  span the axis values {27.5, 50.0, 72.5, 95.0, 117.5, 140.0, 162.5}
  mm.
- **8 corner markers** (black dots) at the 4 top and 4 bottom corners
  of a 175 x 175 mm inner face. Corner (X, Y) positions are
  {7.5, 182.5} mm. Top face sits at Z = 194 mm and bottom face at
  Z = 4 mm (accounting for a 4 mm base thickness).

| Colour | Bead indices | Column X (mm) |
|--------|-------------|---------------|
| Pink   | 4 -- 10     | 27.5          |
| Yellow | 11 -- 17    | 50.0          |
| Blue   | 18 -- 24    | 72.5          |
| Black  | 25 -- 31    | 95.0          |
| Orange | 32 -- 38    | 117.5         |
| Purple | 39 -- 45    | 140.0         |
| Silver | 46 -- 52    | 162.5         |

---

## 2. Data Collection Protocol

### Per-way capture

The calibration object is placed in 4 orientations ("ways"), each
rotated approximately 90 degrees from the previous:

| Way  | Cube orientation   |
|------|--------------------|
| way1 | Reference (0 deg)  |
| way2 | ~90 deg CW         |
| way3 | ~180 deg           |
| way4 | ~270 deg CW        |

For each way, one frame is captured from all three cameras
simultaneously. The resulting images are stored in the raw calibration
folder under `cali_<date>_way<N>_pic/`.

### Why 4 ways?

Each way places the 49 beads at different physical positions in the
room, sampling a wider portion of the 3D volume than a single
orientation. This gives the merged calibration model ~196 bead
positions (4 x 49) instead of 49, which improves accuracy and reduces
bias toward any one region of the volume.

---

## 3. Annotation Workflow

### Manual annotation

Each bead and corner is manually clicked in all three camera images
using the interactive GUI:

```
python -m scripts.calibration.annotate_calibration_gui \
    --folder <calibration_raw_root> \
    --way_name way1 \
    --output <output_dir> \
    --include_bottom_corners
```

### Refinement

Existing annotations can be reviewed and refined using the `--refine`
mode, which auto-zooms to each already-placed point and provides a
wider view for any unplaced point:

```
python -m scripts.calibration.annotate_calibration_gui \
    --folder <calibration_raw_root> \
    --way_name way1 \
    --output <output_dir> \
    --include_bottom_corners \
    --refine \
    --refine_window 100
```

### Rule-based ID assignment

After annotation, `reassign_bead_ids_by_rule.py` canonicalises the
bead and corner IDs based on pixel position in each camera:

- **Beads**: sorted within each colour group according to a
  per-camera spatial rule (top-to-bottom in cam0 for way1,
  right-to-left for way2, etc.) and assigned to known world
  coordinates from the synthetic bead grid.
- **Corners**: classified into image quadrants (TL / TR / BR / BL)
  relative to their centroid in each camera, then assigned canonical
  physical coordinates `{7.5, 182.5}` mm at Z = 194 (top) or
  Z = 4 (bottom).

```
python -m scripts.calibration.reassign_bead_ids_by_rule \
    --folder <calibration_raw_root> \
    --way_name way1 \
    --annotation_dir <manual_annotation_dir> \
    --output_dir <rule_reassigned_dir>
```

---

## 4. Model Training

### Per-way calibration

Each way is trained independently as a sanity check:

```
python -m scripts.calibration.solve_intrinsics \
    --calibration_folder <calibration_raw_root> \
    --image_coords_folder <rule_reassigned_dir> \
    --no_auto_detect --way_name way1 \
    --model_types polynomial --poly_deg_range 2 3
```

The output is a `imCoords_to_XYZ.joblib` model per way.

### Merged calibration (recommended for production)

All 4 ways are merged into a single model. Before merging, a frame
alignment step rotates the world coordinates of ways 2 and 4 by 180
degrees about the cube centre (95, 95) mm so that all ways share a
common room coordinate frame:

```
python -m scripts.calibration.solve_intrinsics \
    --calibration_folder <calibration_raw_root> \
    --image_coords_folder <merged_input_dir> \
    --merge_ways --beads_only \
    --model_types polynomial --poly_deg_range 2 3
```

The merged model trains on 228 points (196 beads + 16 top corners +
16 bottom corners from 4 ways) and is the one used for downstream
tracking.

### Why the 180-degree frame alignment?

Empirical analysis revealed that the per-way bead coordinate frames
group into two families: ways 1 and 3 share one frame ("Frame A"),
and ways 2 and 4 share another ("Frame B"), related by a 180-degree
rotation about the cube centre. Without alignment, the merged model
sees contradictory pixel-to-XYZ mappings and gives ~30--70 mm error.
With the rotation applied, the error drops below 1 mm.

---

## 5. Calibration QC Results (cali_20251217)

### Per-way 3D error (57 points per way: 49 beads + 8 corners)

| Way  | Mean (mm) | RMSE (mm) | Max (mm) |
|------|-----------|-----------|----------|
| way1 | 0.60      | 0.69      | 1.42     |
| way2 | 0.57      | 0.67      | 1.23     |
| way3 | 0.65      | 0.74      | 1.56     |
| way4 | 0.61      | 0.70      | 1.56     |

### Merged model 3D error (228 points)

| Slice            | n   | Mean (mm) | RMSE (mm) | Max (mm) |
|------------------|-----|-----------|-----------|----------|
| **All points**   | 228 | **0.90**  | **1.00**  | **2.15** |
| Beads only       | 196 | 0.89      | 0.99      | 2.15     |
| Top corners      | 16  | 0.91      | 0.99      | 1.67     |
| Bottom corners   | 16  | 1.01      | 1.12      | 1.97     |

### Per-way breakdown in the merged model

| Way  | n  | Mean (mm) | Max (mm) |
|------|----|-----------|----------|
| way1 | 57 | 0.90      | 1.98     |
| way2 | 57 | 0.88      | 1.92     |
| way3 | 57 | 0.99      | 2.15     |
| way4 | 57 | 0.83      | 1.74     |

### 2D reprojection error (beads only, per camera)

Evaluated on the merged beads-only model (196 bead points):

| Camera | Mean (px) | Median (px) | Max (px) |
|--------|-----------|-------------|----------|
| cam0   | 1.96      | 1.70        | 7.94     |
| cam1   | 2.06      | 1.78        | 6.53     |
| cam2   | 1.88      | 1.64        | 7.09     |

All three cameras show uniform ~2 px mean reprojection error with no
outlier camera, indicating balanced calibration quality across views.

---

## 6. Interpreting Calibration Accuracy for Animal Tracking

### What does "0.9 mm mean error" mean for tracking?

The calibration error sets the **floor** for 3D reconstruction
accuracy. Any 3D position reconstructed from multi-camera 2D
detections will have at least ~1 mm of positional uncertainty arising
from the calibration alone.

### Practical implications

| Animal feature           | Typical size     | Calibration error relative to feature |
|--------------------------|------------------|---------------------------------------|
| Medaka body length       | ~25--35 mm       | ~3% of body length                    |
| Head-to-tail axis        | ~20 mm           | ~5%                                   |
| Inter-fish distance      | 5--100 mm        | <1% at courtship distance             |
| Fin extension            | ~2--5 mm         | ~20--50% (cannot reliably resolve)     |

**What you can trust at this accuracy:**

- **Body centroid tracking**: the ~1 mm error is well below the
  body length, so centroid positions and trajectories are reliable.
- **Inter-fish distance**: at typical courtship distances (10--100 mm),
  the ~1 mm error contributes <10% relative uncertainty. Distance
  thresholds for behavioural classification (e.g., "within 20 mm" for
  close approach) are robust.
- **Swimming speed**: frame-to-frame displacement over typical video
  intervals (30--60 fps) is much larger than the 1 mm calibration
  noise for normal swimming. Speed estimates are reliable.
- **3D heading and orientation**: coarse body orientation (which
  direction the fish faces) is resolvable. Fine angular measurements
  require sub-mm calibration.

**What is at the limit:**

- **Fine appendage tracking** (individual fin rays, mouth gape):
  features smaller than ~2 mm approach the calibration error and
  cannot be precisely localised in 3D.
- **Depth (Z) axis**: the merged model shows ~0.72 mm mean axial
  error in Z vs ~0.4 mm in X and Y, because Z is constrained by the
  two side cameras only (cam0 sees X, Y but not Z directly). If your
  analysis is sensitive to Z, keep this asymmetry in mind.

### Volume coverage and edge effects

The calibration beads span the inner cube from (27.5, 27.5, 27.5) to
(162.5, 162.5, 162.5) mm, and the 8 corner markers extend coverage
to (7.5, 7.5, 4) -- (182.5, 182.5, 194) mm. The merged model is
most accurate near the cube centre (mean ~0.95 mm at r < 80 mm from
centre) and slightly less accurate at the edges (~1.2 mm at
r > 80 mm). If your animals swim outside the calibrated volume, the
polynomial extrapolation degrades rapidly. Ensure the tracking volume
fits within the calibrated region.

---

## 7. File Inventory

### Calibration raw data (on shared drive)

```
calibration_raw/cali_20251217/
    cali_coords_template_way{1..4}.txt     # 3D coordinate templates
    cali_251217_way{1..4}_pic/             # Raw camera images
```

### Processed calibration data (in this repo)

```
calibration_results/cali_20251217_manual/
    way{1..4}/
        cali_way{w}_cam{0..2}_image_coords.txt        # Manual annotation
        rule_reassigned/
            cali_way{w}_cam{0..2}_image_coords.txt     # Rule-reassigned
            xyz_previews/                              # Visual check PNGs
            calibration_with_corners/
                models/imCoords_to_XYZ.joblib           # Per-way model
                qc/                                    # Per-way QC
            calibration_beads_only/
                models/imCoords_to_XYZ.joblib           # Per-way beads-only
                qc/                                    # Per-way beads-only QC
    merged_rule_reassigned_beads_only/
        input/                                         # Merged annotation input
        models/imCoords_to_XYZ.joblib                  # Merged model (production)
        qc/                                            # Merged QC
```

### Model used for 3D reconstruction

The reconstruction pipeline (`scripts/reconstruction_3d/calibration.py`)
loads `imCoords_to_XYZ.joblib` from a configurable model folder. Point
it to:

```
calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models/
```

This merged model was trained on 228 calibration points (196 beads +
32 corners across 4 orientations) and achieves 0.90 mm mean / 2.15 mm
max 3D error across the full calibrated volume.

---

## 8. Reproducing the Calibration

To redo the calibration from scratch on a new dataset:

```bash
# 1. Annotate each way (repeat for way1..way4, each camera prompted in sequence)
python -m scripts.calibration.annotate_calibration_gui \
    --folder <raw_root> --way_name way1 --output <way1_output> \
    --include_bottom_corners

# 2. Rule-reassign IDs (repeat for way1..way4)
python -m scripts.calibration.reassign_bead_ids_by_rule \
    --folder <raw_root> --way_name way1 \
    --annotation_dir <way1_output> --output_dir <way1_reassigned>

# 3. Copy reassigned files into a merged input folder
cp <way{1..4}_reassigned>/cali_way*_cam*_image_coords.txt <merged_input>/

# 4. Train merged model
python -m scripts.calibration.solve_intrinsics \
    --calibration_folder <raw_root> \
    --image_coords_folder <merged_input> \
    --output_folder <merged_models> \
    --results_root <merged_qc> \
    --merge_ways --model_types polynomial --poly_deg_range 2 3

# 5. QC: 2D reprojection and volume error
python -m scripts.calibration.qc_reprojection_2d --mapping_csv <merged_qc>/*.csv
python -m scripts.calibration.qc_volume_error    --mapping_csv <merged_qc>/*.csv

# 6. Preview annotation overlays
python -m scripts.calibration.preview_annotation_xyz \
    --folder <raw_root> --way_name way1 \
    --annotation_dir <way1_reassigned> --output_dir <previews>
```

---

*Protocol written 2026-04-13. Calibration dataset: cali_20251217.*
