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
