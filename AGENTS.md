# Repository Guidelines

## Project Shape

This repository is a script-first medaka mating behavior pipeline for acquisition,
calibration, SLEAP inference, 2D/3D tracking, reconstruction, behavioral analysis,
and visualization.

- Active implementation lives primarily under `scripts/`.
- `src/medaka_pipeline/` is currently a package scaffold; do not assume it is the
  canonical implementation unless the surrounding code shows otherwise.
- Prefer the newer `.slp`/HPC workflow for new tracking and reconstruction work:
  `scripts/hpc/` -> `scripts.tracking.main --merge-chunks` ->
  `scripts.reconstruction_3d.reconstruct_from_slp`.
- The older step1-step6 pipeline was retired to `archive/scripts/reconstruction/`
  (kept for back-compat). Use `scripts/reconstruction_3d/` for all new 3D work.

## Setup And Commands

- Recommended environment setup:
  `uv sync --python 3.11`
- Optional groups:
  `uv sync --extra rerun` for `scripts/visualization/rerun_viewer.py`
  and `uv sync --extra sleap` for SLEAP I/O helpers.
- Run tests with:
  `pytest -q`
- CI installs from `environment/requirements.txt` and runs on Python 3.11.

## 3D Tracking Runbook (SLEAP -> 3D)

This is the canonical, agent-agnostic recipe for going from raw recordings to
per-fish 3D tracks for one experiment. Both Claude and Codex should follow it.
Concrete `Z:`/round-tag values below are the current production values on this
machine; resolve real roots from `configs/paths.local.toml`, never hardcode them
in reusable code.

**Stage 1 - SLEAP inference (per view, full length).** Produces one
`<view>.slp` per camera under
`<sleap_output_dir>/<experiment>/<round_tag>/`.

- Local (this box, RTX 5070 Ti): use the torch path. TensorRT export is broken on
  this GPU (Blackwell `peaks_castOut` op), so always run `--mode track` (the
  default); do not propose `--mode predict`/TRT here.

  ```bash
  .venv/Scripts/python.exe -m scripts.sleap.run_round_inference \
      --round-dir   Z:/StephensU/medaka/sleap_models/retrain_20260508 \
      --round-tag   round4_260520 \
      --experiment  <experiment> \
      --data-root   Z:/StephensU/medaka/medaka_202512 \
      --output-root Z:/StephensU/medaka/sleap_output \
      --predict-batch-size 12 --max-instances 6 --continue-on-fail
  ```

  - `--max-instances 6` (> 3 fish) is intentional: extra detections let the 3D
    stage recover fish displaced by reflections (see Stage 2).
  - Batch 12 sits at ~15.5/16.3 GB VRAM on the 5070 Ti (no OOM); ~3.2 h/view at
    full length (~348k frames). sleap-nn buffers stdout, so the per-view
    `.inference.log` stays header-only until a view finishes - monitor by
    watching for the `<view>.slp` to appear, not by tailing the log.

- HPC (saion) alternative: submit via `scripts/hpc/`, then merge the per-view
  chunks into one `.slp` per camera before Stage 2:

  ```bash
  python -m scripts.tracking.main --merge-chunks <experiment> \
      --paths-config configs/paths.local.toml
  ```

**Stage 2 - top-camera 2D identity track.** A per-view NN tracker on the
top camera supplies the stable 2D-ID anchor that the bind step (Stage 3b) uses
to attach a `final_fish_id`. Track on the **default visible-node centroid**
(do **not** pass `--anchor-node`; mouth-anchoring was measured ~2x worse).

```bash
.venv/Scripts/python.exe -m scripts.tracking.track_simple_nn <experiment> \
    --paths-config configs/paths.local.toml \
    --view top_cam \
    --slp Z:/StephensU/medaka/sleap_output/<experiment>/round4_260520/top_cam.slp \
    --output Z:/StephensU/medaka/sleap_output/<experiment>/round4_260520/top_cam.tracks.csv \
    --max-distance 100 --max-tracks 3 --max-gap 500
```

**Stage 3 - 3D reconstruction + identity binding.** Consumes the three
`<view>.slp` files plus the top-cam tracks; writes `<tag>_3d.csv`
(`frame,fish_id,X,Y,Z`) **and** `<tag>_3d_identified.csv` (adds
`final_fish_id`, `bind_source`, `bind_score`, ...).

```bash
.venv/Scripts/python.exe -m scripts.reconstruction_3d.reconstruct_from_slp \
    <experiment> \
    --paths-config configs/paths.local.toml \
    --calibration-backend auto \
    --sleap-subdir round4_260520 \
    --num-candidates 6 \
    --use-visible-node-centroid \
    --xyz-bounds -20 220 --no-shift-to-positive \
    --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 \
    --num-individuals 3 \
    --bind-top-id \
    --top-tracks-csv Z:/StephensU/medaka/sleap_output/<experiment>/round4_260520/top_cam.tracks.csv \
    --output-dir Z:/StephensU/medaka/reconstruction_3d/<experiment>/round4_260520 \
    --output-tag round4_260520
```

- **Calibration policy**: `--calibration-backend auto` resolves the checked-in
  `calibration_model_dir` to the validated fixed-interface refractive backend.
  The resolved backend, score unit, and exact model files are printed and saved
  in reconstruction provenance. For a legacy reproduction, pass
  `--calibration-backend polynomial --model-folder
  calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models`.
  A residual gate is backend-specific: use `--max-residual-px2` for polynomial
  scores and `--max-ray-residual-mm2` for refractive mean squared point-to-ray
  distances. Never carry the historical polynomial value `5000 px^2` into a
  refractive run; `--max-residual-per-fish` is only a deprecated polynomial
  alias.
- **ALWAYS pass `--num-candidates 6`** when the `.slp` came from
  `--max-instances 6` inference. It is the path behind every validated round4
  metric and is ~60x faster. Why it matters:
  - With `--num-candidates 6` (> `--num-individuals 3`) the script uses the
    global mutually-exclusive candidate-set selector
    (`triangulate_extra_candidates.py`, tqdm label **"3D best-assign"**,
    typically ~400-900 frame/s). The selector rate measures candidate scoring
    after coordinates are loaded; remote `.slp` loading/decoding and output I/O
    are separate. Allow roughly 10-15 min end-to-end for 348k frames on the
    current production storage path. This recovers reflection-displaced fish
    via `--xyz-bounds` gating.
  - Omitting it falls back to the default full-permutation path
    (`triangulate.py`, tqdm label **"3D candidates"**, ~10 frame/s, ~9.7 h) -
    both far slower **and a different algorithm** that does not match the
    validated numbers. If you ever see "3D candidates" in the log, you forgot
    the flag; stop and re-run.
- **`--use-visible-node-centroid` is the round4 production choice.** It tracks
  the centroid of the visible `mouth`+`tail` nodes, which is invariant to the
  known round4 mouth<->tail swaps (see [[project-sleap-round4]]); mouth-anchoring
  is for comparison experiments only.
- **3D ID-link thresholds**: `--max-jump-mm 8 --max-jump-hard-cap-mm 80
  --max-gap-frames 100`. The hard cap is essential - it bounds the per-match
  distance so a long NaN gap cannot accept a *different* fish as the same ID
  (effective gate at gap N = `min(8*N, 80)` mm).
- **`--bind-top-id` + `--top-tracks-csv`** attach the Stage 2 top-cam IDs to the
  3D trajectories (per-frame reprojection-NN gated by `--bind-distance`, default
  60 px, then a global 1-to-1 Hungarian vote), producing the
  `final_fish_id` in `<tag>_3d_identified.csv`. Frames where the top camera is
  occluded inherit the geometry ID and are flagged `bind_source=geometry_only`.
- `--sleap-subdir <round_tag>` points the loader at the round-tagged subfolder
  (`<sleap_output_dir>/<experiment>/<round_tag>/<view>.slp`). Omit only if the
  merged `.slp` live directly under `<experiment>/`.

**QC both outputs** (per experiment):
- From the pre-link JSON (`<tag>_best_pre_link.json`): confirm calibration
  provenance and score unit, all-3-fish-valid coverage (frames where all three
  `groups` are finite), the per-fish residual distribution (`errors`, in the
  saved backend unit), and `synthetic_views` missing-camera provenance. Older
  JSONs without calibration or synthetic-view provenance must be reported as
  legacy/unknown rather than treated as an empty cohort. These are pre-fill;
  the final `_3d.csv` is spline-gap-filled.
- From `<tag>_3d.csv` / `<tag>_3d_identified.csv`: 3D speed-spike rate (large
  per-frame jumps = residual ID swaps), the `bind_source` split
  (`top_cam_2d` vs `geometry_only`), the `bind_score` distribution, and that
  `final_fish_id` is a stable 1-to-1 relabel of the geometry IDs.
See `scripts/tracking/README.md` (sect. 2.4) for threshold rationale.

**Optional Stage 4 - 3D-seeded segmentation ID anchor.** Use this when
`track_simple_nn` is the weak link around false negatives, occlusions, or
crossings. The goal is not to change the 3D geometry path; it is to produce a
better top-camera identity CSV for `--bind-top-id`.

1. Mine clean seed windows from the existing 3D output:

```bash
.venv/Scripts/python.exe -m sam2_id_experiment.scripts.24_seed_prompts_from_3d \
    --identified-3d-csv <output_dir>/<tag>_3d_identified.csv \
    --best-pre-link-json <output_dir>/<tag>_best_pre_link.json \
    --paths-config configs/paths.local.toml \
    --output-dir runs/seg_id/<experiment>/<round_tag>/seed_from_3d
```

Outputs are `clean_seed_windows.csv`, `clean_seed_points.csv`,
`sam2_seed_prompts_from_3d.yaml`, and
`sam2_seed_prompts_from_3d_summary.json`. The miner is intentionally strict:
finite 3D rows, unique `final_fish_id`, low pre-link residual, low bind score,
sufficient 3D separation, and no speed spike.

2. Use the existing `sam2_id_experiment/scripts/` workflow to run SAM2 or
Detectron2 masks, SLEAP-mask gating, QC filtering, ID assignment/stitching, and
top-track export. `top_track_export.py` now pools fragments by biological
`assigned_identity`, while preserving `identity_trajectory_id` as provenance.

3. Re-run `reconstruct_from_slp --bind-top-id` with the exported
`top_cam.seg_identity_tracks.csv`, or call `scripts.reconstruction_3d.bind_topcam_id`
directly on an existing `<tag>_3d.csv`. The bind step now writes
`source_bind_key` and tallies by that stable identity key, so short fragments
with the same biological ID count together.

## Data And Paths

- Large raw data, videos, `.slp`, `.h5`, and similar outputs live outside the repo.
- Machine-specific paths belong in `configs/paths.local.toml`, copied from
  `configs/paths.example.toml`; the local config is gitignored.
- Never hardcode drive-specific paths such as `G:` or `Z:` in reusable code.
- Pipeline stages should write inspectable CSV/JSON outputs on disk where practical.

## Calibration Notes

- The default calibration for future SLEAP→3D runs is the fixed-interface
  refractive model under
  `calibration_results/fixed_refractive_192x192x186/`. Its small manifest fixes
  camera/view order, physical geometry, score units, source hashes, and held-out
  validation metrics.
- The learned polynomial/joblib calibration remains available under
  `calibration_results/cali_20251217_manual/` for historical reproducibility.
  Select it explicitly; no backend silently falls back to another geometry.
- When editing calibration scripts, verify behavior against the checked-in QC data
  where feasible.
- For new precision refinement, prefer filming a moved rigid object with 6+
  known non-coplanar bright markers. Use
  `scripts.calibration.refine_calibration_rigid_object` with:
  - `rigid_object_geometry.csv`:
    `marker_id,object_x,object_y,object_z`
  - `rigid_object_image_coords.csv`:
    `pose_id,marker_id,cam0_x,cam0_y,cam1_x,cam1_y,cam2_x,cam2_y`
  The script fits a Kabsch pose per object placement, retrains
  `imCoords_to_XYZ.joblib`, `XYZ_to_imCoords.joblib`, and pairwise helper
  models, and writes `calibration_refinement_report.json`. Accept the new model
  only if held-out rigid-object error and downstream 3D QC metrics do not
  regress.

## Code Style

- English is the primary language for code, comments, docs, and CLI messages.
- Keep reusable logic in `scripts/`. Retired exploratory notebooks live in
  `archive/notebooks/` and one-off reproductions in `archive/scripts/debug/`.
- Preserve existing script conventions and command-line entry point patterns.
- Use structured config/loaders such as `scripts/common/local_paths.py` instead of
  ad hoc local path handling.
- Avoid unrelated refactors while making task-focused changes.

## Visualization

- `scripts/visualization/rerun_viewer.py` is the interactive Rerun viewer for
  synchronized camera streams, 2D detections, 3D trajectories, and tank wireframe.
- It requires `rerun-sdk`, available through the `rerun` optional dependency group.

## Git And Generated Files

- The worktree may contain user changes. Do not revert files you did not change.
- Large binary experiment outputs should stay out of git.
- Calibration results are intentionally committed; treat them as repository data.
