# Acquisition (campy)

This repository uses **campy** for multi-camera synchronized acquisition.

Campy is treated as an external acquisition tool; the pipeline repo focuses on:
- keeping **configs** for a recording session under version control, and
- standardizing where recordings/metadata land on disk.

## Install campy

Follow campy’s upstream installation instructions:
https://github.com/ksseverson57/campy

In short (typical):
- create a conda env (Python 3.7+)
- install camera vendor SDK (Basler Pylon / FLIR Spinnaker) and Python bindings
- `git clone` campy and `pip install -e .`

## Configuration

- Campy uses a YAML config file.
- This repo keeps a template at [configs/acquisition/campy_config.yaml](../../configs/acquisition/campy_config.yaml).
- For Basler cameras, camera settings are commonly stored as `.pfs` files (see campy docs).

## Start recording

From the repo root:

```bash
campy-acquire ./configs/acquisition/campy_config.yaml
```

Or use the thin wrapper (adds a friendlier error if campy is missing):

```bash
python scripts/acquisition/run_campy.py --config ./configs/acquisition/campy_config.yaml
```

## Output layout (recommended)

To keep downstream processing consistent, configure campy to write to a trial directory like:

```
data/raw/<trial_id>/videos/<camera_name>/...
```

and ensure campy also writes its per-camera metadata files (e.g. frame times, metadata CSV).

## Notes

- Triggering/sync: campy supports integration with Arduino/Teensy or external triggering.
- Avoid committing raw videos to git; keep them under `data/raw/` (ignored by `.gitignore`).
