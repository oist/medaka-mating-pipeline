# SLEAP Command-Line Workflow

This directory contains helper scripts for SLEAP inference, export, and
quality-control steps in a Windows / PowerShell environment.

## Main tasks

1. run SLEAP inference on top and side videos
2. export `.slp` files to CSV or HDF5
3. inspect missing detections versus occlusions across views
4. render labeled videos when needed

## Environment

Make sure SLEAP is installed in the environment you plan to use.

Examples:

```powershell
pip install "sleap[nn]"
```

or, if you prefer a tool install:

```powershell
uv tool install sleap-nn[torch] --torch-backend auto
```

## Batch inference

Use `batch_track.py` when you want to:

- pass one or more PLAYER1 videos
- automatically locate the matching PLAYER2 and PLAYER3 clips
- apply the top-view models to PLAYER1
- apply the side-view models to PLAYER2 and PLAYER3

Typical example:

```powershell
python scripts/sleap/batch_track.py `
  --top-videos "D:\behavior_clips\...\PLAYER1_....mp4" `
  --top-models "D:\sleap_top\models\centroid.model" `
  --top-models "D:\sleap_top\models\instance.model" `
  --side-models "D:\sleap_side\models\centroid.model" `
  --side-models "D:\sleap_side\models\instance.model" `
  --output-dir "D:\sleap_output" `
  --max-instances 3
```

## Batch inference across all experiments

Use `batch_predict.py` to walk every experiment folder under
`medaka_data_dir` and run top + side1 + side2 inference for each one.
Paths and model versions are read from `configs/paths.local.toml` and
`configs/sleap/model_paths.yaml`, so the command stays short and the
same across machines.

Expected per-experiment recording layout:

```
<medaka_data_dir>/<experiment>/top_cam/cam0_top.mp4
<medaka_data_dir>/<experiment>/side1_cam/cam1_side.mp4
<medaka_data_dir>/<experiment>/side2_cam/cam2_side.mp4
```

Outputs are written as:

```
<sleap_output_dir>/<experiment>/{top_cam,side1_cam,side2_cam}.slp
```

Existing outputs are skipped automatically (`--force` to re-run).

```powershell
# Preview the work plan for every experiment:
python -m scripts.sleap.batch_predict --paths-config configs/paths.local.toml --dry-run

# Run a single experiment:
python -m scripts.sleap.batch_predict `
  --paths-config configs/paths.local.toml `
  --experiments 20251220_D_1_1_FCMC

# Run everything:
python -m scripts.sleap.batch_predict --paths-config configs/paths.local.toml
```

Required `paths.local.toml` keys: `medaka_data_dir`, `sleap_models_dir`,
`sleap_output_dir`. The model-version selection (currently top `n=330`,
side `n=593`) lives in `configs/sleap/model_paths.yaml`.

## Export

Use `export_tracks.py` to export `.slp` predictions into:

- CSV
- analysis HDF5
- JSON
- SLEAP-native formats

If you need all tracks in CSV form, use `--all-tracks`.

## Multiview QC

Use `qc_tracks.py` to classify missing detections frame by frame as:

- true missing data
- likely single-view occlusion

This is especially useful when all three synchronized views are available.

## Related files

- `batch_predict.py`
- `batch_track.py`
- `export_tracks.py`
- `qc_tracks.py`
