# Medaka Mating Pipeline (3-fish)

This repository contains the end-to-end pipeline for filming, calibration, SLEAP inference, and behavioral analysis for **three medaka** recorded during pre-mating interactions.

## Project description
This study investigates the interplay between female mate choice, male courtship behavior, and male-male aggression in medaka fish (*Oryzias latipes*).
Through video recording and quantitative behavioral analysis before mating events, we aim to dissect how multiple mating-related behaviors interact and identify which specific behavioral factors determine female mating decisions.

## Goals
1. **Female preference determinants:** quantify preference and test whether it is driven by male aggression, courtship intensity, or their interaction.
2. **Decision dynamics:** identify whether mate choice occurs early or requires observation of male-male aggression.
3. **Consistency:** assess stability of male strategies across time and female choice patterns across different male pairs.

## Pipeline overview
1. Acquisition (multi-camera, synchronized)
2. Calibration (intrinsics/extrinsics; refraction-aware if underwater)
3. SLEAP inference (2D keypoints per view; optional identity resolution)
4. 3D reconstruction (optional, if multi-view triangulation is used)
5. Behavioral feature extraction (courtship/aggression/preference metrics)
6. Statistics + reporting

## Data layout (expected)
See `docs/data_layout.md`.

## Quick start

### Environment setup (uv)

This project uses `uv` for fast, reproducible Python environments.

#### Install `uv`

If you don’t have `uv` yet, install it using the official installer:

- **Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

- **macOS/Linux (bash/zsh):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal (or open a new one) so `uv` is on your `PATH`.

More options (package managers, manual install): https://docs.astral.sh/uv/

1) Create/sync the local virtual environment (recommended Python: 3.11):

```bash
uv sync --python 3.11
```

This will create `.venv/` (if missing), install the project, and install the default dev tools.

2) (Optional) Activate the environment

VS Code can usually use `.venv` automatically without manual activation.

Activation commands by platform/shell:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux (bash/zsh):**

```bash
source .venv/bin/activate
```

3) Run tests:

```powershell
uv run pytest
```

Notes:
- The intended Python version is pinned in `.python-version`.
- `uv.lock` is checked in for reproducible dependency resolution.
- `environment/requirements.txt` is kept for reference/migration, but is currently mostly placeholders.

### VS Code / IDE setup

Most team members can simply open the repository folder in VS Code.

- The Python extension will typically auto-detect the `.venv/` environment in the workspace root.
- If it does not, run **Python: Select Interpreter** and choose the `.venv` interpreter.
- Workspace defaults are provided in `.vscode/settings.json` (pytest + `src/` on the Python path).

### Configure experiment

- Configure experiment: edit `configs/project.yaml`

### Acquisition (campy)

Acquisition is performed using **campy** (external tool): https://github.com/ksseverson57/campy

- See `scripts/acquisition/README.md` and `configs/acquisition/campy_config.yaml`.
- Typical run:

```bash
campy-acquire ./configs/acquisition/campy_config.yaml
```

### Tracking / analysis

- Run inference (HPC): `scripts/sleap/run_inference_slurm.sh`
- Compute metrics: `scripts/analysis/compute_preference_metrics.py`

## Reproducibility
- All parameters should live in `configs/`.
- Outputs should be written to `runs/<date>/<experiment_id>/...` (recommended).
