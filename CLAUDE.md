# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-camera 3D behavior tracking pipeline for medaka fish mating experiments.
Research goal: quantify female mate choice by measuring male aggression vs courtship
intensity in a 3-fish (1 female + 2 males) setup.

**Pipeline stages**: Acquisition → Calibration → SLEAP inference (HPC) → 2D/3D tracking → 3D reconstruction → Behavioral analysis → Visualization

## Repository Layout

```
scripts/                # All active logic lives here (not src/)
  acquisition/          # Multi-camera recording (campy, Basler)
  calibration/          # Bead-based camera calibration pipeline
    lib/                # Calibration core library (calibration_core, model_training)
  hpc/                  # Slurm-based SLEAP inference on OIST saion cluster
  sleap/                # SLEAP batch inference & export/visualization wrappers
  tracking/             # Two workflows: legacy clip-based CSV + new .slp-based
  reconstruction_3d/    # Preferred 3D reconstruction (incl. 3D-first tracker)
  reconstruction/       # Legacy step1-step6 reconstruction flow (kept for back-compat)
  analysis/             # Behavioral metrics (placeholders) + render_*_video.py (working)
  visualization/        # rerun_viewer.py — interactive 3D + multiview Rerun viewer
  common/               # local_paths.py, experiment_metadata.py, build_catalog.py
  debug/                # One-off issue reproductions
src/medaka_pipeline/    # Package scaffold — placeholders only, not used yet
configs/                # YAML/TOML config (acquisition, sleap, paths)
calibration_results/    # Checked-in calibration data & trained models
tests/                  # pytest unit tests
docs/                   # overview, calibration, data_layout, script_workflow, etc.
```

## Setup & Environment

```bash
# Recommended: uv (configured to pull torch from pytorch-cu130 index on win/linux)
# Default install now includes sleap-io and rerun-sdk (common-path scripts need them).
uv sync --python 3.11

# Optional extras (opt-in)
uv sync --extra gui          # napari + PyQt5 for scripts/sleap/*_annotator.py
uv sync --extra tensorrt     # tensorrt + onnx for the sleap-nn -> TRT inference path
uv sync --all-extras         # everything at once

# Alternative: pip
python -m venv .venv && pip install -e ".[dev,sleap,rerun]"
```

- Python >=3.11, <3.14
- Key deps: numpy, pandas, opencv-python, scikit-learn, scipy, torch, sleap
- Run tests: `pytest -q`
- Note: CI (`.github/workflows/ci.yml`) installs from `environment/requirements.txt` (not pyproject) and runs `pytest -q` on Ubuntu / Python 3.11.

## Data & Path Configuration

**Large data lives outside the repo** on network drives or Google Drive.
Each machine must create `configs/paths.local.toml` (gitignored) from `configs/paths.example.toml`.
Loader: `scripts/common/local_paths.py` (`load_local_paths`, `choose_path`).

Current data paths point to Google Drive (`G:\.shortcut-targets-by-id\...\OIST 1215-0122\`)
and OIST network storage (`Z:\StephensU\medaka\medaka_202512`). These differ per PC — never hardcode.

Keys expected in `paths.local.toml`:
- `data_root`, `raw_data_dir`, `sleap_output_dir`, `processed_dir`, `tracking_output_dir`
- `calibration_model_dir`, `reconstruction_output_dir`, `runs_dir`
- `medaka_data_dir` — root of per-experiment recording folders (`<exp>/{top_cam,side1_cam,side2_cam}/cam*.mp4`)
- `sleap_models_dir` — root for SLEAP models referenced by `configs/sleap/model_paths.yaml`

Each experiment folder under `medaka_data_dir` carries an `experiment_metadata.json`
(written/updated by `scripts/common/experiment_metadata.py` and the slp-merge step) that
records hand-authored fields plus a `sleap` section with chunk lists / merged paths.

## Two Tracking / Reconstruction Workflows

The repo contains two parallel workflows for going from SLEAP output to per-fish 3D tracks.
**Default to the new HPC/.slp workflow** for new recordings. The old clip-CSV workflow is
kept for back-compat with already-cut courtship clips.

### A. New workflow (raw recordings, .slp end-to-end)

```
SLEAP inference on saion (scripts/hpc/)         → per-view chunked .slp on /bucket
        │
        ▼
scripts.tracking.main --merge-chunks <exp>      → <view>.slp + experiment_metadata.json
        │
        ├─► scripts.tracking.track_simple_nn (2D, per view)         → tracks.csv
        │
        └─► scripts.reconstruction_3d.reconstruct_from_slp (3D)     → <tag>_3d.csv
```

The 3D-first tracker (`reconstruct_from_slp`) is the recommended path: it enumerates per-frame
cross-view permutations, scores triplets via reprojection through the calibration model, and
links IDs in 3D with a velocity/gap-aware Hungarian matcher. See
[scripts/tracking/README.md](scripts/tracking/README.md) §2.4 for thresholds and rationale, and
[AGENTS.md](AGENTS.md) for the full copy-pasteable inference→3D runbook.

> **Gotcha — always pass `--num-candidates 6`** to `reconstruct_from_slp` for `.slp` produced
> with `--max-instances 6` inference. It selects the fast greedy-triplet path (tqdm
> `3D best-assign`, ~10 min full length) that the validated round4 metrics use. Omitting it
> silently falls back to the default full-permutation path (tqdm `3D candidates`, ~9.7 h) —
> slower **and a different algorithm**. Use `--sleap-subdir <round_tag>` when the `.slp` live in
> a round-tagged subfolder.

### B. Legacy workflow (pre-cut clips, CSV-based)

```bash
# 2D tracking post-processing for one PLAYER1/2/3 clip set
python -m scripts.tracking.main --paths-config configs/paths.local.toml <set_key>

# Older 3D reconstruction (preferred for legacy clips)
python -m scripts.reconstruction_3d.run_3d_reconstruction \
  --paths-config configs/paths.local.toml --set_key <set_key>
```

`set_key` = clip identifier, e.g. `20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331`.
View↔player mapping: `player1`=top/cam0, `player2`=side1/cam1, `player3`=side2/cam2.

### Other entry points

```bash
# SLEAP batch inference (local; use scripts/hpc/ for cluster runs)
python scripts/sleap/batch_track.py --top-videos ... --output-dir ...

# Visualize predictions overlaid on video (sanity check; supports --track-csv)
python -m scripts.sleap.visualize_predictions <exp> --paths-config configs/paths.local.toml --view top_cam --seconds 5

# Interactive 3D + multiview Rerun viewer (requires --extra rerun)
python -m scripts.visualization.rerun_viewer <exp> --paths-config configs/paths.local.toml \
  --start-frame 171800 --end-frame 176799
```

`scripts/reconstruction/` (no `_3d`) is the older step1–step6 pipeline; prefer
`scripts/reconstruction_3d/` for new work (per `docs/script_workflow.md`).

## Calibration System

**Method**: Learned polynomial regression (6D pixel coords → 3D world coords), not classical pinhole.
This implicitly absorbs refraction from water/glass - no explicit refraction model needed.

**Calibration object**: 190x190x190 mm cube with 49 color-coded beads (7x7 grid) + 8 corners.
Data collected in 4 orientations ("ways") rotated ~90 degrees each.

**Production model**: `calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models/imCoords_to_XYZ.joblib`
- Accuracy: 0.90 mm mean / 1.00 mm RMSE / 2.15 mm max (228 points)
- Good for: body centroids (25-35 mm fish), inter-fish distances
- Not for: fine appendages (<2 mm), precise Z-axis

**Calibration workflow**:
1. `preannotate_colored_beads.py` - auto-detect bead centers
2. `annotate_calibration_gui.py` - manual refinement GUI
3. `reassign_bead_ids_by_rule.py` - canonical ID assignment
4. `solve_intrinsics.py` - train model (supports merge mode for multi-way)
5. QC scripts: `qc_3d_error.py`, `qc_reprojection_2d.py`, `qc_volume_error.py`

## HPC SLEAP Inference (OIST saion)

`scripts/hpc/` runs SLEAP from the saion login node, submitting one Slurm array job per view
(N chunks × 3 views in parallel). Output `.slp` chunks land in
`/bucket/StephensU/medaka/sleap_output/<experiment>/` and are **not merged on the cluster** —
the merge happens at analysis time via `scripts.tracking.main --merge-chunks`. Configure via
env vars (`MEDAKA_BUCKET_ROOT`, `NUM_CHUNKS`, `PARTITION`, model paths, etc.); see
[scripts/hpc/README.md](scripts/hpc/README.md) for the full table and `DRY_RUN=1` preview.

## Hardware Setup

- 3 cameras: cam0 (top, XY plane), cam1 (side, XZ plane), cam2 (side, YZ plane)
- Tank: 20x20x20 cm
- Frame rate: 166 fps
- Subjects: 1 female (F), 2 males (M1, M2) - 3 fish max

## Important Conventions

- **Language**: English is the primary language for code and docs (refactored from Japanese)
- **Scripts over notebooks**: Reusable logic goes in `scripts/`, one-off exploration in `notebooks/`, one-off bug repros in `scripts/debug/`
- **Identity tracking**: 2D tracking assigns per-view IDs (a, b, c); 3D reconstruction unifies across views
- **Frame alignment**: Ways 2 & 4 are rotated 180 degrees before merging calibration data
- **Git**: Commits co-authored with Claude. Large binary files (.mp4, .h5, .slp) are gitignored.
- **Per-stage outputs are CSV/JSON on disk**, not in-memory only — makes each stage individually re-runnable and inspectable (visidata is in dev deps for this).

## Testing

```bash
pytest -q                    # All tests
pytest tests/test_tracking_and_reconstruction_regressions.py  # Specific
```

Tests cover: tracking regressions, calibration GUI, helper backends, bead remapping, 3D reconstruction.
CI runs on GitHub Actions (Ubuntu, Python 3.11).

## Working with This Repo

- Always check `configs/paths.local.toml` exists before running pipeline scripts.
- The `src/medaka_pipeline/` package is scaffold only — all real logic is in `scripts/`.
- Calibration results are committed to git; raw video data is not.
- When modifying calibration scripts, verify against `calibration_results/cali_20251217_manual/` QC data.
- Network drive paths (Z:, G:) vary by machine — never hardcode them.
- `scripts/analysis/compute_*_metrics.py` are placeholders that raise `SystemExit("TODO: ...")`; the working analysis scripts there are the `render_*_video.py` renderers.
- For new behavior-tracking work, prefer the `.slp`-based path (HPC inference → `--merge-chunks` → `reconstruct_from_slp`) over the legacy clip-CSV path.

## Session Start Rules

- Always read [MEMORY.md](MEMORY.md) at the start of a session to load the current project status, rules, and procedures before starting work.
- When rules or procedures change, update MEMORY.md in the same turn so future sessions inherit the change.

## Work Rules

### No confirmation needed (auto-execute)

- Listing files and folders (`ls`, `find`, Glob)
- Reading existing files
- Investigation, analysis, and read-only inspection of project state

### Confirmation required (ask y/n first)

- Creating new files
- Writing to or overwriting existing files
- Installing packages or modifying dependencies

These confirmation rules apply to changes initiated by Claude. When the user explicitly asks for a specific create/edit/install in their message, that request is the confirmation — proceed without re-asking.

## Script Execution Notes

- Always run Python scripts via the project virtualenv, not the system interpreter:
  - Windows (PowerShell): `.venv\Scripts\python.exe scripts\xxx.py`
  - Unix: `.venv/bin/python scripts/xxx.py`
- Use full paths for binaries when the user's PATH may not be configured. Concrete paths on this machine are recorded in [MEMORY.md](MEMORY.md) under *Key technical info → Common commands / binary paths* (e.g. `ffmpeg`).

## ECC Agent Usage Rules

| Trigger | Agent |
|---------|-------|
| After creating or modifying a script or other source file | `code-reviewer` |
| After touching authentication, API credentials, or external-communication code | `security-reviewer` |
| Before implementing a new feature that spans multiple files | `planner` |
| Before fixing a bug or adding a new feature (write tests first) | `tdd-guide` |

Launch independent agents in parallel when their work doesn't depend on each other (see `~/.claude/rules/ecc/common/agents.md`).
