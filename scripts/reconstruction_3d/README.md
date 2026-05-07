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
- if available, `XYZ_to_imCoords.joblib` improves reprojection scoring
- if available, `xy_yz_to_xz.joblib`, `xz_yz_to_xy.joblib`, and `xz_xy_to_yz.joblib`
  are used for pairwise consistency scoring and one-view recovery

Pipeline summary:

1. Load per-frame 2D points from the processed CSV files.
2. Enumerate candidate cross-view permutations.
3. Use the learned calibration model to predict 3D points.
4. Score candidates with reprojection and/or pairwise calibration consistency.
5. Select the best candidate per frame.
6. Reassign identities across frames in 3D.
7. Optionally fill missing 3D frames with spline interpolation.
8. Export `{set_key}_3d.csv` and an optional trajectory PNG.

Interactive QC with Rerun:

```powershell
python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC `
  --paths-config configs/paths.local.toml `
  --start-frame 171800 --end-frame 176799
```

For a full 30-minute run, start with diagnostics only so the `.rrd` does not
become dominated by video frames:

```powershell
python -m scripts.visualization.rerun_viewer 20251225_K_1_2_FCMC `
  --paths-config configs/paths.local.toml `
  --no-video --save full_qc.rrd
```

The viewer overlays the three SLEAP camera streams when video is enabled,
logs linked 3D tracks, and, when `<tag>_best_pre_link.json` exists, logs the
pre-link triangulated candidates plus registration-error timelines. Use the
full-run diagnostic pass to find fragile frame ranges, then rerun a short
window with video enabled for visual inspection. Relinked outputs from
`relink.py` can be opened with `--tag <experiment>_relinked`; the viewer
auto-detects `<experiment>_best_relinked.json`.

Main modules:

- `load_2d.py`
- `triangulate.py`
- `select_best.py`
- `cross_frame_tracking.py`
- `fill_missing_3d.py`
- `run_3d_reconstruction.py`
