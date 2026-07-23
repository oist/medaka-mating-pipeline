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
                       │
                       └─► --bind-top-id (optional)             → 3d_identified.csv
                           uses top_cam.tracks.csv as 2D-ID anchor;
                           per-frame reprojection-NN + global Hungarian
                           1-to-1 voting attaches a stable `final_fish_id`
                           to each geometry trajectory.
```

### 2D-ID source for `--bind-top-id` (round4_260520 production pipeline)

Empirical 10-window evaluation (`runs/id_eval_batch_step2_isolation/`) showed that
**neither mouth-anchored matching nor top-K-by-score filtering improves identity
quality** on the n=6 source. The best 2D-ID tracker is the **default visible-node
centroid tracker** invoked on the round4 top-camera `.slp`:

```bash
python -m scripts.tracking.track_simple_nn <exp> \
    --slp <sleap_output_dir>/<exp>/round4_260520/top_cam.slp \
    --output <sleap_output_dir>/<exp>/round4_260520/top_cam.tracks.csv \
    --max-distance 100 --max-tracks 3 --max-gap 500
```

(no `--anchor-node`, no `--top-k-by-score` — those flags exist for experimentation
but degrade quality; the isolation study above was first run on the older
`round2_trt_n6/` source and the conclusion carries to round4 mouth+tail. See
commit log.)

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
    --calibration-backend auto \
    --sleap-subdir round4_260520 \
    --num-candidates 6 \
    --use-visible-node-centroid \
    --xyz-bounds -20 220 --no-shift-to-positive \
    --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 \
    --num-individuals 3 \
    --bind-top-id \
    --top-tracks-csv <sleap_output_dir>/20251220_D_1_1_FCMC/round4_260520/top_cam.tracks.csv
```

`--calibration-backend auto` prefers the checked-in fixed-interface refractive
assets and records the resolved backend, exact model files, capabilities, and
score unit. The round4 production run uses **`--use-visible-node-centroid`** (centroid of the
visible mouth+tail nodes — invariant to round4 mouth↔tail swaps), the ID-link
thresholds **`--max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100`**,
and **`--bind-top-id`** (with the Stage 2.3 `top_cam.tracks.csv`) to emit
`<tag>_3d_identified.csv` carrying a stable `final_fish_id`. Run
`track_simple_nn` (sect. 2.3 / the 2D-ID source box above) first to produce that
tracks CSV. Omit `--start-frame/--end-frame` for a full-length run.

> **Always pass `--num-candidates 6` for `.slp` produced with
> `--max-instances 6` inference** (the standard now). It selects the global
> mutually-exclusive candidate-set path (`triangulate_extra_candidates.py`, tqdm
> **"3D best-assign"**, typically ~400-900 frame/s after coordinates are
> loaded) and recovers
> reflection-displaced fish via `--xyz-bounds` gating. **Omitting it silently
> falls back to the default full-permutation path** (`triangulate.py`, tqdm
> **"3D candidates"**, ~10 frame/s — full length ~9.7 h), which is both far
> slower **and a different algorithm** whose results do not match the validated
> round4 numbers. Seeing "3D candidates" in the log means the flag was
> forgotten. Use `--sleap-subdir <round_tag>` when the merged `.slp` live in a
> round-tagged subfolder (`<sleap_output_dir>/<exp>/<round_tag>/<view>.slp`)
> rather than directly under `<exp>/`.

Algorithm (Dynamics-of-dominance-style):

1. **Per-frame candidate enumeration** — for each frame, enumerate every
   assignment of `num_individuals` detections from cam0, cam1, cam2
   (`N! × N!` permutations). Triangulate each triplet through the
   selected calibration backend and score it in that backend's declared unit.
   The refractive backend directly intersects observed rays, including every
   two-view missing-camera case; it does not synthesize a polynomial third view.
2. **Best candidate set per frame** — globally select the maximum-cardinality,
   minimum-cost mutually exclusive set whose points stay inside `--xyz-bounds`.
   The selector is not greedy: one locally cheap triplet cannot consume a slot
   needed by a lower-cost complete set.
3. **Per-fish residual filter** — any fish whose individual registration
   error exceeds the backend-specific gate is marked NaN for that frame. Use
   `--max-residual-px2` for polynomial `px^2` scores or
   `--max-ray-residual-mm2` for fixed-refractive `mm^2` scores,
   even when the overall candidate wins. Drops geometrically implausible
   individual matches without discarding the whole frame.
4. **Temporal linking** — Hungarian NN over 3D positions, using each ID's
   **last valid** point (survives short NaN gaps). `--max-jump-mm` rejects
   matches whose per-frame distance × gap exceeds the threshold;
   `--max-gap-frames` retires IDs missing too long so a re-appearing fish
   can re-seed an orphaned slot. Optional velocity-prediction, tangent, and
   acceleration costs are available through the `--motion-*` flags. They are
   computed for the complete frame before any fish state is updated and default
   to zero, preserving the validated position-only linker.
5. **Gap fill** — linearly interpolate NaN frames inside each trajectory's
   observed range.

Output: `<output_dir>/<tag>_3d.csv` with columns `frame, fish_id, X, Y, Z`.

### 2.4a Tracklet-first multi-view identity

The canonical N=6 path can associate concrete tracklets before final triplet
selection by passing one `--tracklets view=csv` flag for every camera. It then:

1. runs a geometry-only global bootstrap;
2. constructs cross-view equality and same-view all-different constraints;
3. propagates singleton identity domains (including two-known-implies-third);
4. globally re-solves mutually exclusive real and calibrated two-view
   hypotheses with identity and motion compatibility;
5. preserves reliable identified 2D rows when no XYZ is available; and
6. writes a disposition/reason row for every finite candidate slot.

Conflicting concrete-tracklet evidence is rejected during candidate-set search;
it is not converted directly to NaN. Real three-view sets are preferred over
two-view completion. Synthetic reconstructions use `-1` for the missing view
and retain `synthetic_views`/`recovered` provenance.

Outputs in addition to the compatibility CSV/JSON are
`*_tracklet_assignments.csv`, `*_identified_2d.csv`,
`*_detection_dispositions.csv`, `*_cross_view_edges.csv`, and
`*_tracklet_first_summary.json`.

### 2.4b Legacy post-link multi-view pilot

`multiview_tracklet_tracking` implements the stricter identity layer used after
triangulation when the best-separated camera changes by scene:

1. Build edge-extended concrete tracklets independently in every view with
   `concrete_tracklets --mode edge-extended`.
2. Match tracklet points back to the exact filtered SLEAP candidate slots.
3. Join tracklets across cameras only after repeated low-residual 3D
   co-observation. Simultaneous different-fish observations are cannot-link
   constraints.
4. Transfer component identities into the temporal 3D linker. A view that
   remains concrete carries identity across a break in another view.
5. Record cross-camera-inconsistent triplets without changing a concrete
   tracklet identity. Unresolved components are written to `*_appearance_needed.csv`.
   An optional appearance table (`view,tracklet_id,predicted_identity,confidence`)
   can label those components without changing the geometric graph.

The pilot writes `*_cross_view_edges.csv`, `*_tracklet_assignments.csv`,
`*_hint_conflicts.csv`, `*_continuity_violations.csv`, and a JSON summary in
addition to the normal 3D CSV and `best.json`. Expansion should require zero
continuity violations for labeled components and should report the coverage
cost of conservative triplet vetoes.

**Key CLI flags**

| Flag | Purpose | Example |
| --- | --- | --- |
| `--calibration-backend` | Resolve `auto`, `refractive`, or `polynomial`; provenance is persisted | `auto` |
| `--xyz-bounds LOW HIGH` | Reject triangulations outside this cube at candidate selection | `-20 220` |
| `--max-residual-px2` | Optional polynomial registration-score gate (`px^2`) | `5000` for legacy round4 reproduction |
| `--max-ray-residual-mm2` | Optional refractive mean-squared point-to-ray gate (`mm^2`) | omit unless validated |
| `--max-jump-mm` | Per-frame velocity gate for ID linking | `30` |
| `--max-gap-frames` | Retire IDs unseen this long (allows re-seed) | `100` |
| `--motion-velocity-prediction-weight` | Soft constant-velocity prediction cost (opt-in) | `0` |
| `--motion-tangent-weight` | Soft direction-discontinuity cost (opt-in) | `0` |
| `--motion-acceleration-weight` | Soft acceleration-discontinuity cost (opt-in) | `0` |
| `--max-motion-gap-frames` | Disable motion priors, but not distance gating, across longer gaps | `20` |
| `--motion-history-frames` | Robust velocity/tangent history; `2` is last-interval compatibility mode | `2` |
| `--num-individuals` | Number of fish slots end-to-end | `3` |
| `--start-frame / --end-frame` | Inclusive frame range | `171800 176799` |
| `--first-frame-id-only` | Skip Hungarian NN; fix IDs from first valid frame only | — |
| `--no-fill-missing` | Leave gaps as NaN | — |
| `--no-shift-to-positive` | Do not shift origin; keep calibration-native coordinates | — |

**Picking residual and jump thresholds**

Residual values are not portable across backends. The polynomial adapter reports
the historical summed squared pixel registration error (`px^2`). Its round4
distribution was:

| Quantile | Per-fish error (px²) |
| ---: | ---: |
| p50 | 209 |
| p75 | 332 |
| p90 | 2,059 |
| p95 | 48,068 |
| p99 | 245,126 |

Clear bimodality between "good fits" (<1k) and "broken fits" (>10k); the legacy
explicit polynomial gate `--max-residual-px2 5000` rejected ~5–10% of per-fish
assignments (`--max-residual-per-fish` remains only as its deprecated alias).
The refractive adapter reports mean squared point-to-ray
distance (`mm^2`). Do not reuse `5000`: omit the gate until an `mm^2` threshold
has been validated on tuning episodes, then record the chosen unit and value.

`--max-jump-mm` gates ID linking per-frame; pair it with `--max-jump-hard-cap-mm`
so the effective threshold at a gap of N frames is `min(max_jump_mm*N,
hard_cap)`. The round4 full-length production run uses **`--max-jump-mm 8
--max-jump-hard-cap-mm 80`** (≈1.3 m/s instantaneous at 166 fps, well above the
observed p99 ~2.5 mm/frame ≈ 0.4 m/s, while the 80 mm cap stops a long NaN gap
from binding a *different* fish onto the same ID). Loosen `--max-jump-mm` toward
30 only for sparse/low-fps clips where 8 mm fragments tracks.

**Throughput** (single-core), by path:

| Path | tqdm label | frame/s | 348k frames |
| --- | --- | ---: | ---: |
| `--num-candidates 6` (global constrained set, **recommended**) | `3D best-assign` | ~400–900 | ~10–15 min end-to-end |
| default (full 36-permutation) | `3D candidates` | ~10 | ~9.7 h |

Prefer the `--num-candidates 6` path (see callout in §2.4). Both are
single-core, so independent experiments can run in parallel. The tqdm
`frame/s` value is selector compute after candidate coordinates are loaded;
remote `.slp` loading/decoding and final output I/O are outside that rate and
explain why the full 348k-frame wall time is longer.

For conservative post-link repair of appearance-derived `F/M1/M2` labels, see
[MOTION_IDENTITY_REPAIR.md](../../sam2_id_experiment/MOTION_IDENTITY_REPAIR.md).
For source-frame human review of hidden swaps in the anonymous geometry slots,
including pair-swap events, deliberately unlabeled intervals, and conservative
clean-tracklet export, see
[GEOMETRY_SWAP_REVIEW.md](../analysis/GEOMETRY_SWAP_REVIEW.md).

### 2.5 Related scripts

- [track_simple_nn.py](track_simple_nn.py) — 2D per-view NN tracker (§2.3)
- [slp_merge.py](slp_merge.py) — chunk merger (§2.1)
- [../sleap/visualize_predictions.py](../sleap/visualize_predictions.py) — 2D overlay viz
- [../reconstruction_3d/reconstruct_from_slp.py](../reconstruction_3d/reconstruct_from_slp.py) — 3D pipeline (§2.4)
- [../reconstruction_3d/load_2d_slp.py](../reconstruction_3d/load_2d_slp.py) — `.slp` → `points_by_frame`
- [../reconstruction_3d/triangulate.py](../reconstruction_3d/triangulate.py) — permutation scoring
- [../reconstruction_3d/cross_frame_tracking.py](../reconstruction_3d/cross_frame_tracking.py) — temporal linking
