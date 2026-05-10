# Tracking Post-Processing

Two separate workflows live here:

1. **Clip-based CSV pipeline** — `main.py` takes SLEAP CSV exports for
   pre-cut PLAYER1/2/3 clips and produces cleaned per-view tracks for 3D
   reconstruction. Legacy path; still supported.
2. **HPC / raw-recording pipeline** — merge per-view `.slp` chunks produced
   by `scripts/hpc/`, run frame-by-frame tracking in 2D or 3D, feed results
   into analysis. This is the active workflow for new recordings.

---

## 1. Clip-based CSV pipeline (legacy)

Run from a set key or an explicit PLAYER1 CSV:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"

python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "D:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv"
```

Useful flags:

- `--simple` — lightweight identity stabilisation without outlier removal
- `--no-viz` — skip visualization PNGs
- `--skip-2d-interpolation` — leave NaN gaps for the 3D stage
- `--flow-weight` — add motion-direction cost to the Hungarian assignment

Outputs per set key: `*_player{1,2,3}_processed.csv`, `*_after_id_assignment.png`,
`*_final.png`. See [ALGORITHM_DESCRIPTION.md](ALGORITHM_DESCRIPTION.md).

---

## 2. HPC / raw-recording pipeline

Given a batch of experiment folders under `medaka_data_dir` and SLEAP chunk
`.slp` files in `sleap_output_dir/<experiment>/` (produced by
[scripts/hpc/submit_experiment.sh](../hpc/submit_experiment.sh)), the stages
are:

```
  cam0 chunks ─┐
  cam1 chunks ─┼─► merge ─► <view>.slp + experiment_metadata.json
  cam2 chunks ─┘
                   │
                   ├─► track_simple_nn.py (2D, per view)        → tracks.csv
                   │
                   └─► reconstruct_from_slp.py (3D, all views)  → 3d.csv
```

### 2.1 Merge per-view chunks → one `.slp` per camera

```bash
python -m scripts.tracking.main --merge-chunks 20251220_D_1_1_FCMC \
    --paths-config configs/paths.local.toml
```

What it does:

- Reads `<sleap_output_dir>/20251220_D_1_1_FCMC/<view>.frames_*.slp` (4 chunks
  per view by default).
- Merges via `sleap-io`, deduplicating Video + Skeleton entries and
  rewriting the Video filename to the canonical `/bucket/.../cam*_*.mp4`
  path so downstream loads do not depend on `/work` scratch.
- Skips views whose merged `.slp` is already fresher than every chunk
  (override with `--force-merge`).
- Updates `<medaka_data_dir>/<experiment>/experiment_metadata.json` with a
  `sleap` section (chunk list, merged path, total frames per view) while
  preserving hand-authored fields (e.g. `animal`).

Outputs:

- `<sleap_output_dir>/<experiment>/{top_cam,side1_cam,side2_cam}.slp`
- `<medaka_data_dir>/<experiment>/experiment_metadata.json`

See [slp_merge.py](slp_merge.py) and [../common/experiment_metadata.py](../common/experiment_metadata.py).

### 2.2 Visualise raw predictions (sanity check)

```bash
python -m scripts.sleap.visualize_predictions 20251220_D_1_1_FCMC \
    --paths-config configs/paths.local.toml \
    --view top_cam --seconds 5 --radius 10
```

Writes `<slp>.viz.mp4` with keypoints overlaid on the source video. Supports
`--frames START-END` and `--every N` subsampling.

### 2.3 Frame-by-frame 2D tracker (per view, simple NN)

```bash
python -m scripts.tracking.track_simple_nn 20251220_D_1_1_FCMC \
    --paths-config configs/paths.local.toml \
    --view top_cam \
    --start-frame 171800 --end-frame 176799 \
    --max-distance 100 --max-tracks 3 --max-gap 500
```

Hungarian matching of each frame's detections against the previous
frame's track positions, gated by `--max-distance` (pixels). Retires a
track that has been unseen for more than `--max-gap` frames.

Outputs: `<slp>.tracks.csv` with columns `frame,track_id,x,y,score,node`.

**Limitation**: 2D-only — when a fish is occluded in one view, its ID
fragments on that view (e.g. 4 IDs over 5000 frames on a hard side view).
Use the 3D tracker below when you need consistent identities across views.

To overlay the 2D tracks on the video coloured by `track_id`:

```bash
python -m scripts.sleap.visualize_predictions 20251220_D_1_1_FCMC \
    --paths-config configs/paths.local.toml \
    --view top_cam --seconds 5 \
    --track-csv <sleap_output_dir>/20251220_D_1_1_FCMC/top_cam.tracks.csv
```

### 2.3b Blob-first top-camera ID prototype

The top camera can be used as an idtracker-style identity prior, but the
prototype should operate on fish-shaped blobs/fragments rather than per-frame
mouth crops. The workflow below mines candidate blobs, lets you curate clean
ID seeds, trains a simple fragment-level appearance model, and writes an
overlay-compatible CSV. Ambiguous fragments are left as `unknown`.

Fragment linking is a lightweight Mask-OCSORT-style tracker rather than plain
SORT: it uses centroid velocity prediction, mask IoU, bbox IoU, area stability,
Hungarian assignment, manual-ID compatibility constraints, and a short offline
tracklet stitch pass. Same-frame split blobs assigned the same manual ID are
merged into virtual training crops/masks before fitting the appearance model.

```bash
python -m scripts.tracking.top_camera_identity mine-blobs 20251225_K_1_2_FCMC \
    --paths-config configs/paths.local.toml \
    --start-frame 171300 --end-frame 173000 \
    --max-candidates-per-sleap 1 \
    --out-dir runs/top_id/20251225_K_1_2_FCMC_f171300-173000

python -m scripts.tracking.top_camera_identity review \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/blob_candidates.csv \
    --experiment 20251225_K_1_2_FCMC \
    --paths-config configs/paths.local.toml \
    --play-fps 2

python -m scripts.tracking.top_camera_identity train-id \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/blob_candidates.reviewed.csv

python -m scripts.tracking.top_camera_identity apply-id \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/blob_candidates.reviewed.csv \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/top_id_model.joblib
```

For a fresh first-pass dataset from background subtraction only, without SLEAP
proximity filtering or SAM masks:

```bash
python -m scripts.tracking.top_camera_identity mine-blobs 20251225_K_1_2_FCMC \
    --paths-config configs/paths.local.toml \
    --start-frame 171300 --end-frame 173000 \
    --pure-bgsub \
    --threshold 24 \
    --min-area 80 --max-area 8000 \
    --out-dir runs/top_id/20251225_K_1_2_FCMC_f171300-173000_pure_bgsub
```

To reduce manual review time, curate a small clean seed set first, then
bootstrap labels on the same blob candidates:

```bash
python -m scripts.tracking.top_camera_identity bootstrap-id \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/blob_candidates.reviewed.csv

python -m scripts.tracking.top_camera_identity review \
    runs/top_id/20251225_K_1_2_FCMC_f171300-173000/blob_candidates.bootstrapped.csv \
    --experiment 20251225_K_1_2_FCMC \
    --paths-config configs/paths.local.toml \
    --play-fps 2
```

Repeat `bootstrap-id -> review` as labels improve. Rows confirmed with the GUI
are marked `id_source=manual`; automatic guesses are marked
`id_source=bootstrap`, and bootstrap training ignores its own previous guesses
unless `--include-bootstrap-labels` is passed. Use `--suggest-only` if you want
predicted IDs shown in the reviewer without changing rows to `status=accept`.

Outputs include `blob_candidates.csv`, `masks/`, `crops/`,
`blob_candidates_preview.mp4`, `top_cam.blob_identity_tracks.csv`, and
`top_cam.blob_identity_fragments.csv`. To overlay the result:

```bash
python -m scripts.sleap.visualize_predictions 20251225_K_1_2_FCMC \
    --paths-config configs/paths.local.toml \
    --view top_cam --frames 171300-173000 \
    --track-csv runs/top_id/20251225_K_1_2_FCMC_f171300-173000/top_cam.blob_identity_tracks.csv \
    --track-only \
    --output runs/top_id/20251225_K_1_2_FCMC_f171300-173000/top_cam.blob_identity.viz.mp4
```

When `--track-csv` is provided, the visualizer draws the CSV track points
directly and uses nearest matching to recolor SLEAP keypoints. Use
`--track-only` to hide raw SLEAP keypoints and show only blob-ID centroids,
which is usually easier for identity QC. Without
`--track-csv`, colors are only raw per-frame SLEAP instance order and can swap.

Optional SAM/SAM2/SAM3 assistance is ingested as an external manifest with
columns `frame,mask_path,source,score`; the repo does not depend on SAM.

The reviewer opens the full top-camera frame when `--video` or `--experiment`
is provided. It overlays translucent masks, bounding boxes, and current labels.
Use click/`[`/`]`/`tab` to choose a blob, `1`/`2`/`3` to assign ID, `u` for
unknown, `r` or `d` to reject/delete a candidate, `space` for slow play/pause,
left/right arrows or `p`/`n` to move frames, `+`/`-` to change speed, mouse
wheel to zoom around the cursor, and `s` to save.
Press `f` for smart-fill mode; moving the mouse previews the connected, locally
color-similar fill region, clicking adds it to the selected mask, and `,`/`.`
shrinks/grows the smart-fill search area. Press `x` for smart-erase mode; it
uses the same preview but removes the clicked connected region from the selected
mask. Press `a` for add-mask brush mode, `e` for erase-mask brush mode, `z` to
undo the latest brush/fill/erase edit, `v` to return to selection mode, and
`,`/`.` to change brush size outside smart modes. Brush and smart edits rewrite
the candidate mask PNG and update `area`, `bbox_*`, centroid, and crop artifacts
in the reviewed CSV directory. Pressing `q` also autosaves. With the default
reviewed output path, mask edits are mirrored back to the input candidate CSV so
reopened masks and bboxes stay consistent; if `blob_candidates.reviewed.csv`
already exists, reopening from `blob_candidates.csv` will load the reviewed file.
Pass `--crop-only` if you only want the old crop-by-crop view.

### 2.4 3D-first tracker (recommended)

```bash
python -m scripts.reconstruction_3d.reconstruct_from_slp 20251220_D_1_1_FCMC \
    --paths-config configs/paths.local.toml \
    --start-frame 171800 --end-frame 176799 \
    --xyz-bounds -20 220 --no-shift-to-positive \
    --max-residual-per-fish 5000 \
    --max-jump-mm 30 --max-gap-frames 100 \
    --num-individuals 3
```

Algorithm (Dynamics-of-dominance-style):

1. **Per-frame candidate enumeration** — for each frame, enumerate every
   assignment of `num_individuals` detections from cam0, cam1, cam2
   (`N! × N!` permutations). Triangulate each triplet through the
   calibration model and score it by the triplet registration error
   (reprojection loss summed across pair-to-third models).
2. **Best candidate per frame** — pick the minimum-cost assignment whose
   points stay inside `--xyz-bounds`.
3. **Per-fish residual filter** — any fish whose individual registration
   error exceeds `--max-residual-per-fish` is marked NaN for that frame,
   even when the overall candidate wins. Drops geometrically implausible
   individual matches without discarding the whole frame.
4. **Temporal linking** — Hungarian NN over 3D positions, using each ID's
   **last valid** point (survives short NaN gaps). `--max-jump-mm` rejects
   matches whose per-frame distance × gap exceeds the threshold;
   `--max-gap-frames` retires IDs missing too long so a re-appearing fish
   can re-seed an orphaned slot.
5. **Gap fill** — linearly interpolate NaN frames inside each trajectory's
   observed range.

Output: `<output_dir>/<tag>_3d.csv` with columns `frame, fish_id, X, Y, Z`.

**Key CLI flags**

| Flag | Purpose | Example |
| --- | --- | --- |
| `--xyz-bounds LOW HIGH` | Reject triangulations outside this cube at candidate selection | `-20 220` |
| `--max-residual-per-fish` | Pixel²-sum threshold for individual reprojection error | `5000` |
| `--max-jump-mm` | Per-frame velocity gate for ID linking | `30` |
| `--max-gap-frames` | Retire IDs unseen this long (allows re-seed) | `100` |
| `--num-individuals` | Number of fish slots end-to-end | `3` |
| `--start-frame / --end-frame` | Inclusive frame range | `171800 176799` |
| `--first-frame-id-only` | Skip Hungarian NN; fix IDs from first valid frame only | — |
| `--no-fill-missing` | Leave gaps as NaN | — |
| `--no-shift-to-positive` | Do not shift origin; keep calibration-native coordinates | — |

**Picking residual and jump thresholds**

The per-fish registration error sums squared pixel residuals across the
three pair-to-third reprojections (6 terms). For medaka with the current
calibration:

| Quantile | Per-fish error (px²) |
| ---: | ---: |
| p50 | 209 |
| p75 | 332 |
| p90 | 2,059 |
| p95 | 48,068 |
| p99 | 245,126 |

Clear bimodality between "good fits" (<1k) and "broken fits" (>10k) — default
`--max-residual-per-fish 5000` rejects ~5–10% of per-fish assignments.

`--max-jump-mm 30` at 166 fps permits ≤5 m/s per-frame, well above natural
medaka swim speed (observed p99 ~2.5 mm/frame ≈ 0.4 m/s). Tighten to 15 mm
if you still see rare single-frame outliers.

**Throughput**: ~10 frames/s single-core with 36 permutations per frame;
5000 frames ≈ 9 min.

### 2.5 Related scripts

- [track_simple_nn.py](track_simple_nn.py) — 2D per-view NN tracker (§2.3)
- [slp_merge.py](slp_merge.py) — chunk merger (§2.1)
- [../sleap/visualize_predictions.py](../sleap/visualize_predictions.py) — 2D overlay viz
- [../reconstruction_3d/reconstruct_from_slp.py](../reconstruction_3d/reconstruct_from_slp.py) — 3D pipeline (§2.4)
- [../reconstruction_3d/load_2d_slp.py](../reconstruction_3d/load_2d_slp.py) — `.slp` → `points_by_frame`
- [../reconstruction_3d/triangulate.py](../reconstruction_3d/triangulate.py) — permutation scoring
- [../reconstruction_3d/cross_frame_tracking.py](../reconstruction_3d/cross_frame_tracking.py) — temporal linking
