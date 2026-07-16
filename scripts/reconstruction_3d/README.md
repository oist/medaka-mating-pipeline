# 3D Reconstruction

The canonical full-run entry point is `reconstruct_from_slp`. Future runs use
the validated fixed-interface refractive calibration by default through
`configs/paths.local.toml` and `--calibration-backend auto`:

```powershell
python -m scripts.reconstruction_3d.reconstruct_from_slp EXPERIMENT `
  --paths-config configs/paths.local.toml `
  --calibration-backend auto `
  --sleap-subdir ROUND_TAG --num-candidates 6 `
  --use-visible-node-centroid --xyz-bounds -20 220 `
  --no-shift-to-positive --num-individuals 3
```

`auto` prefers valid `cam0/1/2_bundle_refractive.json` assets and still
recognizes historical joblib-only folders. Every run prints and persists the
resolved backend, exact files, capabilities, and hypothesis-score unit. Fixed
refractive scores are in `mm^2`; polynomial registration scores are in `px^2`.
Use `--max-ray-residual-mm2` only with refractive calibration and
`--max-residual-px2` only with polynomial calibration. The old
`--max-residual-per-fish` spelling is a deprecated polynomial-only alias. Never
reuse a threshold across those units.

For an explicit legacy reproduction:

```powershell
python -m scripts.reconstruction_3d.reconstruct_from_slp EXPERIMENT `
  --paths-config configs/paths.local.toml `
  --calibration-backend polynomial `
  --model-folder calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models
```

The refractive backend intersects the two or three observed rays directly and
projects XYZ through the same flat-port geometry for top-ID binding. It never
silently synthesizes a missing view or falls back to polynomial geometry.

The older processed-CSV entry point below remains for compatibility.

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
- polynomial calibration directories must contain `imCoords_to_XYZ.joblib`;
  refractive directories contain all three `cam*_bundle_refractive.json` files
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
