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

- `batch_track.py`
- `export_tracks.py`
- `qc_tracks.py`
