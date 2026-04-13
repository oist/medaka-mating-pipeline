# Camera Calibration Theory

This repository uses a learned 2D-to-3D calibration approach rather than a
classical pinhole-camera-only workflow.

## Core idea

The calibration pipeline learns a mapping:

```text
(u0, v0, u1, v1, u2, v2) -> (X, Y, Z)
```

where:

- `(u0, v0)` comes from the top view
- `(u1, v1)` comes from side view 1
- `(u2, v2)` comes from side view 2

The learned model is stored as `imCoords_to_XYZ.joblib`.

An optional inverse model, `XYZ_to_imCoords.joblib`, can be used for
reprojection error and quality control.

## Why use a learned model?

This project is interested in recording through water and glass, where simple
pinhole assumptions are often not sufficient on their own.

Advantages of the learned approach:

- can absorb refraction-like nonlinear effects implicitly
- avoids hand-deriving a full physical optical model
- works directly from measured calibration correspondences

Tradeoffs:

- accuracy depends strongly on the coverage and quality of calibration points
- extrapolation outside the calibration volume can fail badly
- the model is less physically interpretable than classical intrinsics +
  extrinsics

## Training approach

The implementation searches over polynomial model complexity and regularization
settings, then trains a regression model that predicts 3D coordinates from
stacked 2D image coordinates.

The exact code path lives in `solve_intrinsics.py` and its helper library.

## Practical implications

To improve precision:

- collect many calibration points
- distribute them across the full volume where fish will move
- include depth variation, not just one plane
- verify reprojection quality, not just raw 3D fit quality

To reduce failure risk:

- do not trust predictions far outside the bead-covered region
- be cautious when 2D inputs are partially missing or mismatched across views
- prefer using the reprojection model when available so candidate permutations
  can be scored meaningfully

## In this repo

Main calibration artifacts:

- `imCoords_to_XYZ.joblib`
- `XYZ_to_imCoords.joblib` (optional but highly recommended)
- additional helper models for estimating one view from the other two in the
  legacy reconstruction path

The active `scripts/reconstruction_3d` path can also use those helper models
when they are present. In practice that means:

- triplet scoring can still be geometry-informed even if `XYZ_to_imCoords.joblib`
  is missing
- exactly one missing view in a triplet can be inferred from the other two
  views before 3D prediction

Main quality-control scripts:

- `qc_3d_error.py`
- `qc_reprojection_2d.py`
- `qc_3d_template.py`
