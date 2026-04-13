# Post-CSV Pipeline Logic (Including 3D Reconstruction)

This document describes the code flow after exporting CSV files from SLEAP and
before producing final 3D trajectories. It also explains how outliers are
defined, how they are handled, and why 3D reconstruction can fail.

---

## 1. End-to-End Flow

### 1.1 Data flow from CSV to 3D

```text
SLEAP-exported CSV files
  (top_mouth_180 / side_mouth_593, *_PLAYER1/2/3.csv)
        ↓
[04_tracking / scripts/tracking]
  load + parse -> ID assignment -> outlier removal -> mirror filtering
  -> optional 2D interpolation
        ↓
*_processed.csv written for player1 / player2 / player3
        ↓
[reconstruction_3d]
  load_2d -> points_by_frame
  -> candidate generation (calibrated or no-calibration mode)
  -> best-permutation selection
  -> 3D tracking / ID assignment
        ↓
final 3D trajectories
```

- Before tracking post-processing, SLEAP exports one CSV per view containing
  `frame`, `x`, `y`, `track`, and related columns.
- The tracking stage assigns identities across frames, removes obvious bad
  detections, optionally interpolates 2D gaps, and writes
  `*_player1/2/3_processed.csv`.
- The 3D stages use the processed CSV files only. They no longer read `.slp`
  directly.

---

## 2. What Counts as an Outlier?

In this pipeline, an outlier is a 2D point on a single fish track
(`same camera + same fish_id`) that is implausible for the same fish across
adjacent frames.

### 2.1 Large single-frame jump

For a trajectory sorted by `frame`:

- compute the displacement `dist` from the previous valid point to the current
  point
- compute the frame gap `gap = frame_curr - frame_prev`
- mark the current point as an outlier if:

```text
dist > max_jump_per_frame * gap
```

Example:

- if `max_jump_per_frame = 100`, then a one-frame gap allows at most 100 pixels
- a two-frame gap allows at most 200 pixels

Interpretation:

- the same fish is unlikely to move that far in that short a time
- the point is more likely to be an ID mismatch or bad detection

### 2.2 Loop-segment outlier

This catches short segments where the track leaves a location, wanders far
away, and quickly returns to almost the same place.

The segment is marked as wrong if:

- the endpoints are at most `max_loop_frames` apart
- the endpoints are within `return_dist_thresh`
- every intermediate point is farther than `excursion_dist_thresh` from both
  endpoints

Typical case:

- frames 423 and 429 are nearly in the same place
- frames 424-428 jump to a different region

That pattern usually means the track followed the wrong target or a reflection.

### 2.3 Mirror filtering

Mirror filtering is separate from the outlier module, but it also sets points
to `NaN`.

Within one frame and one view:

- if two points have similar `y`
- and their `x` coordinates are approximately symmetric around an axis

then the pipeline keeps the point closer to the previous position of that
`fish_id` and drops the other point as a likely reflection.

Removed points are written to `*_mirror_outliers.csv`.

### 2.4 Implementation locations

- outlier detection: `scripts/tracking/outlier_detection.py`
- mirror filtering: `scripts/tracking/outlier_detection.py`
- main orchestration: `scripts/tracking/process_tracking.py`

---

## 3. From Processed CSV to 3D

### 3.1 Loading 2D points

`run_load_2d(processed_dir, set_key, use_fish_id=False)`:

- reads the three processed CSV files
- returns `points_by_frame[frame][cam]`
- `cam0`, `cam1`, `cam2` correspond to `player1`, `player2`, `player3`

Ordering behavior:

- `use_fish_id=True`: points are read in `a, b, c` order by `fish_id`
- `use_fish_id=False`: points are ordered by score (or by `x` if score is
  missing)

Important consequence:

- when `use_fish_id=False`, cross-view correspondence is not guaranteed before
  3D candidate selection

### 3.2 Candidate generation

Two modes exist:

- calibrated mode: `compute_candidates()`
- no-calibration mode: `compute_candidates_no_calib()`

Calibrated mode:

- enumerates permutations between views
- feeds the 2D triplets into the learned calibration model
- computes a 3D point and an associated cost

No-calibration mode:

- assumes `top=(X,Y)`, `side1=(X,Z)`, `side2=(Y,Z)`
- combines coordinates by averaging shared dimensions
- uses cross-view disagreement as the cost

In both modes:

- missing 2D values cause the candidate 3D point to become `NaN`
- invalid or out-of-bounds 3D points are rejected

### 3.3 Best candidate per frame

`select_best_all_frames()` picks the minimum-cost candidate for each frame,
preferably one where all three points stay inside `xyz_bounds`.

The output is:

```text
best_by_frame[frame] = {
  "groups": {"a": [...], "b": [...], "c": [...]},
  "perm_cam1": [...],
  "perm_cam2": [...],
}
```

### 3.4 3D tracking

The selected per-frame 3D groups are then associated across frames in 3D space.

Typical behavior:

- initialize IDs from the first valid frame
- match subsequent frames using Hungarian assignment on 3D distance
- preserve IDs while filling missing frames later if requested

If most frames contain only `NaN` 3D candidates, downstream tracking will also
remain mostly empty.

---

## 4. Why 3D Reconstruction Can Fail

### 4.1 Too many missing 2D points

If outlier removal or mirror filtering deletes a large fraction of the 2D
points, many frame/view combinations become `NaN`, which directly reduces the
number of valid 3D candidates.

### 4.2 Weak cross-view correspondence

If `use_fish_id=False`, the three views may not be aligned by individual before
candidate generation. This makes candidate selection much more ambiguous.

### 4.3 No valid initialization frame

If the first usable 3D frame never appears, the tracking stage cannot start
reliably and the final 3D trajectories can remain empty.

### 4.4 Bounds mismatch

If `xyz_bounds` do not match the scale of the data, valid points can be thrown
away as out of range.

### 4.5 Wrong set key or file paths

If the processed CSV files do not match the requested `set_key`, the 3D stage
can load the wrong files or no files at all.

---

## 5. Practical Checks

- Confirm that tracking post-processing already produced all three
  `*_player1/2/3_processed.csv` files.
- Inspect the processed CSV files and estimate how many `NaN` values remain in
  `x` and `y`.
- Try `use_fish_id=True` if the 2D identities are already reliable.
- Relax or disable `xyz_bounds` temporarily when debugging.
- Run the no-calibration path once to decide whether the problem is already
  present before the 3D tracker.
