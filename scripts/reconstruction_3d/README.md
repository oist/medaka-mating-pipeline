# 3D Reconstruction

Use this stage after `scripts/tracking/` has written the three
`*_player1/2/3_processed.csv` files for a clip.

Preferred CLI:

```powershell
python -m scripts.reconstruction_3d.run_3d_reconstruction `
  --paths-config configs/paths.local.toml `
  --set_key "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
```

If your local config does not define `processed_dir` or
`calibration_model_dir`, pass them explicitly:

```powershell
python -m scripts.reconstruction_3d.run_3d_reconstruction `
  --processed_dir "D:\sleap_output\processed" `
  --model_folder "D:\calibration_model\cali_251217_way1" `
  --set_key "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
```

Data expectations:

- `player1` = top / `cam0`
- `player2` = side1 / `cam1`
- `player3` = side2 / `cam2`
- processed CSVs should include `frame`, `x`, `y`, with optional `fish_id` and `score`
- calibration model directory must contain `imCoords_to_XYZ.joblib`

Pipeline summary:

1. Load per-frame 2D points from the processed CSV files.
2. Enumerate candidate cross-view permutations.
3. Use the learned calibration model to predict 3D points.
4. Select the best candidate per frame.
5. Reassign identities across frames in 3D.
6. Optionally fill missing 3D frames with spline interpolation.
7. Export `{set_key}_3d.csv` and an optional trajectory PNG.

Main modules:

- `load_2d.py`
- `triangulate.py`
- `select_best.py`
- `cross_frame_tracking.py`
- `fill_missing_3d.py`
- `run_3d_reconstruction.py`
