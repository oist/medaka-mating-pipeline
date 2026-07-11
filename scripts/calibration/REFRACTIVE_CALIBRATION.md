# Physics-based calibration for the 20 cm medaka tank

## Optical model

The side-camera path from the fish to the camera is

`water -> glass -> air`.

The implementation traces the reciprocal direction from a pixel into the tank:

`air -> glass -> water`.

The two faces of one pane are modelled as parallel planes separated by the
measured glass thickness. At each face the direction is updated with vector
Snell refraction. The top camera uses one air/water interface unless a cover
plate is present, in which case it uses the same two-interface model as a side
camera.

For camera `c` and pixel `u`, the model produces a water-side ray
`(o_c(u), d_c(u))`. A 3D point is the least-squares intersection of the three
rays:

```text
X* = argmin_X sum_c ||(I - d_c d_c^T)(X - o_c)||^2
```

This is an axial, non-single-viewpoint camera model. A single 2D radial warp
cannot represent the full depth-dependent effect of a flat refractive
interface.

The working implementation is in
`scripts/calibration/refractive_geometry.py`. The current 12-ball fitting and
held-out evaluation command is
`scripts/calibration/fit_refractive_rigid_object.py`.

## Parameters

| Parameter | Source | Fit policy |
|---|---|---|
| `fx, fy, cx, cy`, lens distortion | Dry ChArUco calibration, same focus/resolution | Fix in wet fit |
| Camera rotation and center in tank coordinates | Wet initialization from DLT/PnP | Fit |
| Outer pane normal and plane offset | Mechanical survey / tank CAD | Fit only a small correction with a prior |
| Glass thickness | **4 mm** for this tank | Fix |
| `n_air` | 1.0003 | Fix |
| `n_glass` | Manufacturer value, normally about 1.49-1.52 | Fix initially |
| `n_water` | 25 C recording; initialize at **1.3325** for visible light | Fix initially; optionally fit one shared scalar later |
| Rigid-object pose for each placement | Unknown | Fit six degrees of freedom per pose |
| Tank/world gauge | Surveyed wall/floor fiducials or known tank planes | Fix |

The current wet-only prototype fits several correlated quantities because dry
intrinsics and surveyed pane geometry do not yet exist. Its held-out score is
useful, but its fitted focal lengths, principal points, pane tilts, and camera
centres must not be interpreted as physical measurements.

## Data-collection protocol

### 1. Freeze the imaging configuration

- Lock camera mounts, focus, aperture, resolution, crop, and any digital image
  stabilization.
- Water temperature was **25 C**. Record the water depth before the next wet
  recording; it was not measured for the supplied sequence.
- Do not refocus between dry-intrinsic, wet-extrinsic, and fish recordings.
- Use the measured **192 x 192 mm inner footprint** (200 mm outer span minus
  two 4 mm panes). Water height is currently unknown, so absolute Z and the
  top-interface offset remain provisional until a direct height measurement or
  surveyed fixed marker anchors them.

### 2. Calibrate dry intrinsics independently

For each camera, record 40-60 sharp ChArUco/checkerboard images in air at the
same resolution and focus used for the experiment.

- Cover the full image, especially all four corners.
- Use strong board tilt in both axes and several distances.
- Fit `K` plus Brown distortion with OpenCV (`calibrateCamera` or
  `calibrateCameraExtended`).
- Reject blurred frames and board corners with large residuals.
- Select the distortion order by held-out image reprojection rather than by
  training error.
- Save parameter covariance and per-image residuals.

This step separates lens distortion from refraction. Jointly estimating both
from the wet cube makes focal length, principal point, camera-to-pane distance,
and pane tilt strongly correlated.

### 3. Measure the interfaces

For each side pane:

- Measure glass thickness at several locations with calipers. If thickness
  varies, the parallel-pane assumption is invalid and a two-plane model is
  required.
- Measure the inner/outer wall plane in the tank coordinate frame. A ChArUco
  board temporarily attached to the pane or surveyed tank-corner fiducials can
  establish the normal and offset.
- Record glass type or refractive index if available.

For the top camera, measure the water-surface plane at the acquisition water
level. Avoid meniscus-dominated wall regions during calibration, or use a flat
transparent cover whose geometry can be modelled.

### 4. Record a wet rigid-object dataset

Use the measured 12-ball object geometry and hardware-synchronized cameras.

- Retain at least 50-60 clean poses; 80 is preferable.
- Sample centre, faces, edges, and all eight corners of the usable fish volume.
- Include oblique viewing rays, not only fronto-parallel placements.
- Rotate the object substantially between placements so interface parameters
  cannot be traded against one repeated pose.
- Add poses in the currently sparse regions identified by
  `qc_rigid_object_volume.py`, especially high-X/high-Z and the low-Z corners.
- Keep entire placements, not individual balls, as the cross-validation unit.

Ball centroids should be stored with a per-observation uncertainty. Saturated,
occluded, or merged blobs should be rejected rather than assigned equal weight.

### 5. Fit by staged refractive bundle adjustment

Let `x_j` be a ball in object coordinates and `(R_p, t_p)` the unknown pose of
placement `p`. Its tank coordinate is `X_pj = R_p x_j + t_p`. For each observed
pixel, back-project through the appropriate interfaces and minimize point-to-ray
distance:

```text
r_cpj = W_cpj^(1/2) (I - d_cpj d_cpj^T) (X_pj - o_cpj)
```

Use a robust Huber/soft-L1 loss. Recommended release order:

1. Fix dry intrinsics, distortion, measured interface planes, thicknesses, and
   indices. Fit object poses plus camera extrinsics.
2. Release small interface-plane normal/offset corrections with survey priors.
3. Release one shared `n_water` only if temperature is uncertain and the data
   contain strong oblique/depth diversity.
4. Do not fit glass thickness and glass index simultaneously; measure and fix
   them.

Anchor the gauge with surveyed tank fiducials/planes. Otherwise the cameras and
all object poses can move together without changing image residuals.

The current prototype fixes the Kabsch-derived training poses. The production
version should add per-placement pose variables and sparse Jacobian structure;
`scipy.optimize.least_squares` is adequate for this dataset, while Ceres is a
natural upgrade for larger datasets.

### 6. Validate without pose leakage

Split by complete object placement and stratify test poses into central,
boundary, and deliberately extrapolative groups. Report:

- Kabsch residual to the known 12-ball shape;
- fit-free pairwise-distance error and signed scale bias;
- water-ray intersection residual;
- error as a function of position, depth, and incidence angle;
- absolute localization error on separately surveyed fixed markers;
- parameter stability across folds and plausible perturbations of thickness,
  water index, and interface plane.

The Kabsch and pairwise metrics validate shape and scale but do not validate
absolute tank position. Surveyed fixed markers are required for that final
claim.

The initial acceptance bar should be non-inferiority to the refined polynomial
on central held-out poses (current mean Kabsch residual 0.389 mm and pairwise
MAE 0.330 mm), plus a clear improvement on boundary/extrapolation poses. The
refractive model should also pass the existing fish-track speed-spike,
association, and behavioural non-regression checks.

## Current wet-only prototype result

The same 21 training poses and 10 untouched test poses were used for every
model.

| Model | Kabsch mean / RMS / P95 | Pairwise MAE / P95 | Scale bias |
|---|---:|---:|---:|
| Base polynomial | 0.592 / 0.693 / 1.291 mm | 0.539 / 1.516 mm | +0.353 mm |
| Refined polynomial | **0.389 / 0.436 / 0.804 mm** | **0.330 / 0.932 mm** | +0.179 mm |
| Projective DLT | 0.436 / 0.512 / 0.902 mm | 0.385 / 1.043 mm | +0.182 mm |
| Flat refractive prototype | 0.420 / 0.514 / 0.961 mm | 0.374 / 1.046 mm | +0.232 mm |

The explicit model converges and is competitive. It improves the low-parameter
DLT mean and pairwise MAE slightly, but it does not yet beat the refined
polynomial. The next decisive experiment is the same fit with independently
measured dry intrinsics and pane geometry, followed by boundary-focused held-out
poses.

## References

- Treibitz, Schechner, Kunz, and Singh, *Flat Refractive Geometry*, IEEE TPAMI
  34(1), 2012: https://doi.org/10.1109/TPAMI.2011.105
- Agrawal, Ramalingam, Taguchi, and Chari, *A Theory of Multi-Layer Flat
  Refractive Geometry*, CVPR 2012:
  https://www.merl.com/publications/TR2012-047
- Jordt-Sedlazeck and Koch, *Refractive Calibration of Underwater Cameras*,
  ECCV 2012: https://doi.org/10.1007/978-3-642-33715-4_61
- Seegraber, She, Woelk, and Koser, *A Calibration Tool for Refractive
  Underwater Vision*, ICCV Workshops 2025:
  https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20%26%20AAMVEM/html/Seegraber_A_Calibration_Tool_for_Refractive_Underwater_Vision_ICCVW_2025_paper.html
