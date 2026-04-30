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
