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

The cameras are fixed to the tank; only the calibration object moves
between data collection steps ("ways").

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
- **Corners**: classified into image quadrants (TL / TR / BL / BR)
  relative to their centroid in each camera, then assigned to
  canonical tank-frame positions `{7.5, 182.5}` mm at Z = 194 (top)
  or Z = 4 (bottom). Side cameras (cam1, cam2) use separate
  top/bottom quadrant mappings because perspective reversal causes
  bottom corners to appear in the opposite vertical half of the image.

```
python -m scripts.calibration.reassign_bead_ids_by_rule \
    --folder <calibration_raw_root> \
    --way_name way1 \
    --annotation_dir <manual_annotation_dir> \
    --output_dir <rule_reassigned_dir>
```

**Important**: The automatic quadrant-based corner assignment works
reliably for way1. For way2--4, manual verification of corner IDs is
required (using `preview_annotation_xyz.py` overlays) and corrections
should be applied directly to the annotation files.

---

## 4. Coordinate Frames

### Tank frame (world frame)

All 3D coordinates in the pipeline are expressed in the **tank frame**,
which is fixed relative to the three cameras. The origin coincides with
the cube corner when it is in way1 orientation.

- Corners have fixed tank-frame positions across all ways (the same
  physical tank position, regardless of cube orientation).
- Bead positions in the template are given in the **cube frame** (i.e.,
  relative to the cube's own axes). For odd ways (1, 3) the cube frame
  approximately equals the tank frame. For even ways (2, 4) the cube
  frame is rotated ~180 degrees about the centre. The calibration
  pipeline applies `(190 - X, 190 - Y)` to even-way bead coordinates
  inside `load_calibration_data()` to convert them to tank frame.

### Why 180-degree alignment for even ways?

The per-way bead coordinate frames group into two families: ways 1 and 3
share one frame ("Frame A"), and ways 2 and 4 share another ("Frame B"),
related by a 180-degree rotation about the cube centre (95, 95) mm.
Without alignment, the merged model sees contradictory pixel-to-XYZ
mappings and gives ~30--70 mm error. With the rotation applied, the
error drops below 1 mm.

This alignment is now performed automatically inside
`load_calibration_data()` for bead points, so no explicit alignment step
is needed when merging ways.

---

## 5. Model Training

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

### Merged calibration (production baseline)

All 4 ways are merged into a single model. The frame alignment is
handled automatically by `load_calibration_data()`:

```
python -m scripts.calibration.solve_intrinsics \
    --calibration_folder <calibration_raw_root> \
    --image_coords_folder <merged_input_dir> \
    --merge_ways --beads_only \
    --model_types polynomial --poly_deg_range 2 3
```

### Rigid body refinement (recommended)

The merged calibration assumes the cube is placed at an exact position
and rotation for each way. In practice, placement is imprecise by
~1 mm translation and ~0.3 degrees rotation. The rigid refinement step
corrects for this:

```
python -m scripts.calibration.refine_calibration_rigid \
    --calibration_folder <calibration_raw_root> \
    --image_coords_folder <merged_input_dir> \
    --output_folder <refined_models> \
    --results_root <refined_qc> \
    --beads_only \
    --way_names way1 way2 way3 way4
```

The script:

1. Trains an initial polynomial model with the assumed 3D targets.
2. Predicts 3D positions for all calibration points.
3. Uses the **Kabsch algorithm** to find the optimal rigid transform
   (rotation R + translation t) per way, mapping the known cube
   geometry to the predicted tank-frame positions.
4. Computes refined 3D targets from the rigid transforms.
5. Detects annotation outliers (points with large rigid-fit residuals).
6. Retrains the model with refined targets, excluding outliers.
7. Repeats until convergence (typically 2--3 iterations).

This improves the merged model by ~35% in 3D error (0.92 -> 0.60 mm
mean on cali_20251217) and also identifies physically misplaced beads
vs. annotation errors.

---

## 6. Calibration QC Results (cali_20251217)

### Per-way 3D error (57 points per way: 49 beads + 8 corners)

| Way  | Mean (mm) | RMSE (mm) | Max (mm) |
|------|-----------|-----------|----------|
| way1 | 0.60      | 0.69      | 1.42     |
| way2 | 0.57      | 0.67      | 1.23     |
| way3 | 0.65      | 0.74      | 1.56     |
| way4 | 0.61      | 0.70      | 1.56     |

### Merged model 3D error (beads only, 196 points)

| Model variant        | n   | Mean (mm) | RMSE (mm) | Max (mm) |
|----------------------|-----|-----------|-----------|----------|
| **Assumed poses**    | 196 | 0.92      | 1.03      | 2.38     |
| **Rigid-refined**    | 196 | **0.60**  | **0.70**  | **2.26** |
| Rigid (inliers only) | 191 | 0.57      | 0.63      | 1.26     |

### Rigid-refined per-way transforms (cali_20251217)

| Way  | Rotation from assumed (deg) | Translation from assumed (mm) |
|------|-----------------------------|-------------------------------|
| way1 | (+0.10, -0.34, +0.23)       | (+1.41, -0.06, -0.57)         |
| way2 | (+0.23, +0.05, +179.72)     | (+190.77, +189.0, -0.20)      |
| way3 | (-0.32, +0.49, +0.23)       | (-0.98, -0.81, +0.88)         |
| way4 | (-0.46, +0.10, +179.81)     | (+190.38, +190.3, +1.12)      |

These confirm that the cube was placed within ~1 mm and ~0.5 degrees
of the assumed position for each way, with way2 showing the largest
Y-axis offset (~1 mm).

### Outlier beads detected

| Bead         | Ways flagged | Residual (mm) | Likely cause                |
|--------------|-------------|---------------|-----------------------------|
| beads_way1_008 | all 4     | 1.4 -- 2.2    | Physical bead misplacement  |
| beads_way1_026 | way4      | 1.4           | Annotation error            |

`beads_way1_008` (index 11, nominal position 50.0/50.0/162.5) is
flagged in all 4 ways, strongly suggesting the physical bead is
~1.5--2 mm from its template position on the cube.

### 2D reprojection error (beads only, per camera)

| Camera | Mean (px) | Median (px) | Max (px) |
|--------|-----------|-------------|----------|
| cam0   | 2.08      | 1.82        | 6.65     |
| cam1   | 2.12      | 1.88        | 6.70     |
| cam2   | 2.18      | 1.90        | 7.39     |

All three cameras show uniform ~2 px mean reprojection error with no
outlier camera, indicating balanced calibration quality across views.
The 2D reprojection floor (~2 px) reflects the annotation precision
limit, not the calibration model.

---

## 7. Interpreting Calibration Accuracy for Animal Tracking

### What does "0.6 mm mean error" mean for tracking?

The calibration error sets the **floor** for 3D reconstruction
accuracy. Any 3D position reconstructed from multi-camera 2D
detections will have at least ~0.6 mm of positional uncertainty arising
from the calibration alone.

### Practical implications

| Animal feature           | Typical size     | Calibration error relative to feature |
|--------------------------|------------------|---------------------------------------|
| Medaka body length       | ~25--35 mm       | ~2% of body length                    |
| Head-to-tail axis        | ~20 mm           | ~3%                                   |
| Inter-fish distance      | 5--100 mm        | <1% at courtship distance             |
| Fin extension            | ~2--5 mm         | ~12--30% (approaching the limit)      |

**What you can trust at this accuracy:**

- **Body centroid tracking**: the ~0.6 mm error is well below the
  body length, so centroid positions and trajectories are reliable.
- **Inter-fish distance**: at typical courtship distances (10--100 mm),
  the ~0.6 mm error contributes <6% relative uncertainty. Distance
  thresholds for behavioural classification (e.g., "within 20 mm" for
  close approach) are robust.
- **Swimming speed**: frame-to-frame displacement over typical video
  intervals (30--60 fps) is much larger than the 0.6 mm calibration
  noise for normal swimming. Speed estimates are reliable.
- **3D heading and orientation**: coarse body orientation (which
  direction the fish faces) is resolvable.

**What is at the limit:**

- **Fine appendage tracking** (individual fin rays, mouth gape):
  features smaller than ~2 mm approach the calibration error and
  cannot be precisely localised in 3D.
- **Depth (Z) axis**: the merged model shows slightly higher error
  in Z than in X and Y, because Z is constrained by the two side
  cameras only (cam0 sees X, Y but not Z directly). If your analysis
  is sensitive to Z, keep this asymmetry in mind.

### Volume coverage and edge effects

The calibration beads span the inner cube from (27.5, 27.5, 27.5) to
(162.5, 162.5, 162.5) mm, and the 8 corner markers extend coverage
to (7.5, 7.5, 4) -- (182.5, 182.5, 194) mm. The merged model is
most accurate near the cube centre and slightly less accurate at the
edges. If your animals swim outside the calibrated volume, the
polynomial extrapolation degrades rapidly. Ensure the tracking volume
fits within the calibrated region.

---

## 8. File Inventory

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
            models/imCoords_to_XYZ.joblib              # Per-way model
            qc/                                        # Per-way QC
    merged_rule_reassigned_beads_only/
        input/                                         # Merged annotation input
        models/imCoords_to_XYZ.joblib                  # Merged baseline model
        qc/                                            # Merged QC
        models_rigid/imCoords_to_XYZ.joblib            # Rigid-refined model
        qc_rigid/                                      # Rigid-refined QC
```

### Model used for 3D reconstruction

The reconstruction pipeline (`scripts/reconstruction_3d/calibration.py`)
loads `imCoords_to_XYZ.joblib` from a configurable model folder. Point
it to:

```
calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models_rigid/
```

This rigid-refined model was trained on 191 inlier bead points
(196 total minus 5 outliers, across 4 orientations) and achieves
0.60 mm mean / 0.70 mm RMSE 3D error across the calibrated volume.

---

## 9. Reproducing the Calibration

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

# 3. Verify corner IDs visually (way2-4 may need manual correction)
python -m scripts.calibration.preview_annotation_xyz \
    --folder <raw_root> --way_name way1 \
    --annotation_dir <way1_reassigned> --output_dir <previews>

# 4. Copy reassigned files into a merged input folder
cp <way{1..4}_reassigned>/cali_way*_cam*_image_coords.txt <merged_input>/

# 5. Train merged baseline model
python -m scripts.calibration.solve_intrinsics \
    --calibration_folder <raw_root> \
    --image_coords_folder <merged_input> \
    --output_folder <merged_models> \
    --results_root <merged_qc> \
    --merge_ways --beads_only \
    --model_types polynomial --poly_deg_range 2 3

# 6. Rigid body refinement (recommended)
python -m scripts.calibration.refine_calibration_rigid \
    --calibration_folder <raw_root> \
    --image_coords_folder <merged_input> \
    --output_folder <refined_models> \
    --results_root <refined_qc> \
    --beads_only

# 7. QC: 2D reprojection and volume error
python -m scripts.calibration.qc_reprojection_2d --mapping_csv <refined_qc>/*_point_mapping.csv
python -m scripts.calibration.qc_volume_error    --mapping_csv <refined_qc>/*_point_mapping.csv
```

---

*Protocol updated 2026-04-14. Calibration dataset: cali_20251217.*
