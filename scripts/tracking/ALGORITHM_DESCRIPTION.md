# Tracking Algorithm Notes

This document summarizes the current 2D tracking post-processing flow in
`scripts/tracking/`.

## Scope

The tracking stage takes SLEAP-exported CSV files for three synchronized views
and produces cleaned `*_processed.csv` files for later 3D reconstruction.

Main tasks:

1. group matching PLAYER1/2/3 CSV files into one set
2. assign stable identities across frames inside each view
3. mark obvious 2D outliers as `NaN`
4. remove likely mirror/reflection detections
5. optionally interpolate and smooth short 2D gaps
6. export processed CSV files and optional trajectory plots

## File grouping

`file_set_generator.py` scans the SLEAP export directories and builds
`file_sets.json`.

Expected directory layout:

- `top_mouth_180/` for PLAYER1 files
- `side_mouth_593/` for PLAYER2 and PLAYER3 files

Only complete sets with all three files are retained.

## Identity assignment

`id_assignment.py` performs per-view temporal identity assignment.

Current behavior:

- first frame: assign `a`, `b`, `c` in x-order
- later frames: use Hungarian assignment on distance plus optional
  motion-direction penalty
- matches beyond `max_distance` are rejected instead of being force-assigned
- fish that disappear temporarily keep their last valid position as the anchor
  for the next frame
- fallback x-order assignment is only used for fish that have never appeared
  before
- each camera is tracked on its own frame timeline, so dropped frames in one
  view do not block identity propagation in the others
- when SLEAP exports multiple node rows for one `(frame, track)`, the rows are
  collapsed to one centroid before assignment

## Outlier handling

`outlier_detection.py` currently uses two simple rules:

1. frame-gap-scaled jump rule
2. short loop-segment rule

Outliers are not deleted from the table. Their `x` and `y` coordinates are set
to `NaN`.

Mirror filtering is also handled there and writes a separate log of removed
reflection-like points when enabled.

## Missing-value filling

`missing_value_filling.py` can:

- infer clip duration from the filename
- expand trajectories to the full frame range
- fill gaps with linear or cubic interpolation
- apply optional Savitzky-Golay smoothing

This stage should generally run after outlier detection.

## Visualization

`visualization.py` renders trajectory plots for one view or all three views on
one figure.

## Main entry points

Preferred CLI:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
```

Useful flags:

- `--simple`: skip outlier and mirror filtering
- `--flow-weight`: add motion-direction cost during assignment
- `--skip-2d-interpolation`: leave missing 2D points for later 3D handling
- `--no-viz`: skip PNG outputs

Core Python entry points:

- `process_tracking.py::process_single_set`
- `process_tracking.py::process_from_csv_path`
- `process_tracking.py::process_from_file_set_key`
- `process_tracking_simple.py::process_single_set_simple`

## Limitations

- identity assignment is still per-view, not cross-view
- difficult crossings and false detections can still trigger ID swaps
- strong filtering can create too many `NaN` points for downstream 3D stages
- long gaps cannot be recovered reliably by interpolation alone
