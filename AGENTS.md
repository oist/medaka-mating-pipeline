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
- `scripts/reconstruction/` is the older step1-step6 pipeline kept for
  compatibility. Prefer `scripts/reconstruction_3d/` for new 3D work.

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

**Stage 2 - 3D reconstruction + identity tracking.** Consumes the three
`<view>.slp` files and writes `<tag>_3d.csv` (`frame,fish_id,X,Y,Z`).

```bash
.venv/Scripts/python.exe -m scripts.reconstruction_3d.reconstruct_from_slp \
    <experiment> \
    --paths-config configs/paths.local.toml \
    --sleap-subdir round4_260520 \
    --num-candidates 6 \
    --xyz-bounds -20 220 --no-shift-to-positive \
    --max-residual-per-fish 5000 --max-jump-mm 30 --max-gap-frames 100 \
    --num-individuals 3 \
    --output-dir Z:/StephensU/medaka/reconstruction_3d/<experiment>/round4_260520 \
    --output-tag round4_260520
```

- **ALWAYS pass `--num-candidates 6`** when the `.slp` came from
  `--max-instances 6` inference. It is the path behind every validated round4
  metric and is ~60x faster. Why it matters:
  - With `--num-candidates 6` (> `--num-individuals 3`) the script uses the
    greedy mutually-exclusive triplet selector
    (`triangulate_extra_candidates.py`, tqdm label **"3D best-assign"**,
    ~500-660 frame/s, ~10 min for 348k frames). This recovers reflection-
    displaced fish via `--xyz-bounds` gating.
  - Omitting it falls back to the default full-permutation path
    (`triangulate.py`, tqdm label **"3D candidates"**, ~10 frame/s, ~9.7 h) -
    both far slower **and a different algorithm** that does not match the
    validated numbers. If you ever see "3D candidates" in the log, you forgot
    the flag; stop and re-run.
- `--sleap-subdir <round_tag>` points the loader at the round-tagged subfolder
  (`<sleap_output_dir>/<experiment>/<round_tag>/<view>.slp`). Omit only if the
  merged `.slp` live directly under `<experiment>/`.
- `--anchor-node` defaults to `mouth`, matching the 2-node `mouth -> tail`
  skeleton. The thresholds above are the documented production values.

**QC the result** (per experiment) from the pre-link JSON: report
all-3-fish-valid coverage (frames where all three `groups` are finite) and the
per-fish residual distribution (`errors`, px^2; healthy round4 is median <200,
frac>5000 <1%). These are pre-fill numbers; the final `_3d.csv` is
spline-gap-filled. See `scripts/tracking/README.md` (sect. 2.4) for threshold
rationale.

## Data And Paths

- Large raw data, videos, `.slp`, `.h5`, and similar outputs live outside the repo.
- Machine-specific paths belong in `configs/paths.local.toml`, copied from
  `configs/paths.example.toml`; the local config is gitignored.
- Never hardcode drive-specific paths such as `G:` or `Z:` in reusable code.
- Pipeline stages should write inspectable CSV/JSON outputs on disk where practical.

## Calibration Notes

- Production calibration uses learned polynomial regression from multi-view pixel
  coordinates to 3D world coordinates, with refraction implicitly absorbed.
- Current checked-in calibration assets are under
  `calibration_results/cali_20251217_manual/`.
- When editing calibration scripts, verify behavior against the checked-in QC data
  where feasible.

## Code Style

- English is the primary language for code, comments, docs, and CLI messages.
- Keep reusable logic in `scripts/`; notebooks are for exploration, and
  `scripts/debug/` is for one-off reproductions.
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
