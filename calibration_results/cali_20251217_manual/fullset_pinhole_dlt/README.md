# Full-set projective DLT calibration

This directory contains the deployable form of the projective DLT model that
won the independent 13-pose rigid-object comparison in
`runs/calib_fullset/FULLSET_CALIBRATION_REPORT.md`.

## What the bundle contains

- `models/imCoords_to_XYZ.joblib`: vectorized three-ray DLT triangulation.
- `models/XYZ_to_imCoords.joblib`: the matching DLT projection model.
- `models/cam*_pinhole_dlt.json`: inspectable camera parameters.
- `models/xz_yz_to_xy.joblib` and the other two pairwise helpers: unchanged
  production helpers used only for missing-view recovery.
- `models/calibration_bundle.json`: provenance and reproduced held-out metrics.

The bundle is generated, not hand-assembled:

```powershell
uv run python -m scripts.calibration.export_dlt_calibration_bundle `
  --training-coordinates-csv runs/calib_fullset/physics_bundle/ball_training_coordinates.csv `
  --geometry-csv runs/calib_12balls/rigid_object_geometry.csv `
  --base-model-folder calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models `
  --output-dir calibration_results/cali_20251217_manual/fullset_pinhole_dlt/models `
  --all-image-coords-csv runs/calib_fullset/balls/rigid_object_image_coords.csv `
  --heldout-pose-list runs/calib_fullset/physics_bundle/ball_heldout_poses.txt
```

## Controlled round7 use

For the calibration-only comparison, freeze cross-camera assignment to the
production model and change only the reported 3D geometry:

```powershell
uv run python -m scripts.reconstruction_3d.reconstruct_from_slp <experiment> `
  --paths-config configs/paths.local.toml `
  --sleap-subdir round7_260704 `
  --model-folder calibration_results/cali_20251217_manual/fullset_pinhole_dlt/models `
  --assignment-model-folder calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models `
  --num-candidates 6 --use-visible-node-centroid `
  --xyz-bounds -20 220 --no-shift-to-positive `
  --max-residual-per-fish 5000 `
  --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 `
  --num-individuals 3
```

Omit `--assignment-model-folder` only for the separate experiment that asks
whether DLT reprojection should also select cross-camera detection triplets.

Do not change `configs/paths.local.toml` to this bundle until the frozen-
assignment comparison passes stratified windows and full runs for K_1_1,
K_1_2, and K_1_3. Round7 fish tracks are a downstream non-regression test;
absolute calibration accuracy still comes from the held-out rigid object.
