# CLAUDE.md - Medaka Mating Pipeline

## Project Overview

Multi-camera 3D behavior tracking pipeline for medaka fish mating experiments.
Research goal: quantify female mate choice by measuring male aggression vs courtship
intensity in a 3-fish (1 female + 2 males) setup.

**Pipeline stages**: Acquisition → Calibration → SLEAP inference → 2D tracking → 3D reconstruction → Behavioral analysis

## Repository Layout

```
scripts/              # All active logic lives here (not src/)
  acquisition/        # Multi-camera recording (campy, Basler)
  calibration/        # Bead-based camera calibration pipeline
    lib/              # Calibration core library (calibration_core, model_training)
  tracking/           # 2D identity tracking post-processing
  reconstruction_3d/  # 3D reconstruction from multi-view 2D tracks
  sleap/              # SLEAP inference & export wrappers
  analysis/           # Behavioral metrics (mostly TODO/placeholder)
  common/             # Shared utilities (local_paths.py)
src/medaka_pipeline/  # Package scaffold - placeholders only, not used yet
configs/              # YAML/TOML configuration templates
calibration_results/  # Checked-in calibration data & trained models
tests/                # pytest unit tests
docs/                 # Documentation (overview, calibration, data layout, etc.)
```

## Setup & Environment

```bash
# Recommended: uv (handles PyTorch CUDA automatically)
uv sync --python 3.11

# Alternative: pip
python -m venv .venv && pip install -e ".[dev]"
```

- Python >=3.11, <3.14
- Key deps: numpy, pandas, opencv-python, scikit-learn, scipy, torch, sleap
- Run tests: `pytest -q`

## Data & Path Configuration

**Large data lives outside the repo** on network drives or Google Drive.
Each machine must create `configs/paths.local.toml` (gitignored) from `configs/paths.example.toml`.

Current data paths point to Google Drive via `G:\.shortcut-targets-by-id\...\OIST 1215-0122\`.
These paths differ per PC - always use `configs/paths.local.toml` for machine-specific paths.

Expected external data structure:
```
<data_root>/
  raw/                  # Raw video files
  sleap_output/         # SLEAP .slp inference results
  processed/            # 2D tracking outputs (*_processed.csv)
  calibration_model/    # Production calibration model
  reconstruction_3d/    # 3D reconstruction outputs
  runs/                 # Pipeline run logs
```

## Key CLI Entry Points

```bash
# 2D tracking post-processing
python -m scripts.tracking.main --paths-config configs/paths.local.toml <set_key>

# 3D reconstruction
python -m scripts.reconstruction_3d.run_3d_reconstruction \
  --paths-config configs/paths.local.toml --set_key <set_key>

# SLEAP batch inference
python scripts/sleap/batch_track.py --top-videos ... --output-dir ...
```

`set_key` = clip identifier, e.g. `20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331`

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

## Hardware Setup

- 3 cameras: cam0 (top, XY plane), cam1 (side, XZ plane), cam2 (side, YZ plane)
- Tank: 20x20x20 cm
- Frame rate: 166 fps
- Subjects: 1 female (F), 2 males (M1, M2) - 3 fish max

## Important Conventions

- **Language**: English is the primary language for code and docs (refactored from Japanese)
- **Scripts over notebooks**: Reusable logic goes in `scripts/`, one-off exploration in `notebooks/`
- **Identity tracking**: 2D tracking assigns per-view IDs (a, b, c); 3D reconstruction unifies across views
- **Frame alignment**: Ways 2 & 4 are rotated 180 degrees before merging calibration data
- **Git**: Commits co-authored with Claude. Large binary files (.mp4, .h5, .slp) are gitignored.

## Testing

```bash
pytest -q                    # All tests
pytest tests/test_tracking_and_reconstruction_regressions.py  # Specific
```

Tests cover: tracking regressions, calibration GUI, helper backends, bead remapping, 3D reconstruction.
CI runs on GitHub Actions (Ubuntu, Python 3.11).

## Working with This Repo

- Always check `configs/paths.local.toml` exists before running pipeline scripts
- The `src/medaka_pipeline/` package is scaffold only - all real logic is in `scripts/`
- Calibration results are committed to git; raw video data is not
- When modifying calibration scripts, verify against `calibration_results/cali_20251217_manual/` QC data
- Network drive paths (Z:, G:) vary by machine - never hardcode them
