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

## Script-first workflow

If you prefer terminal-driven development over notebooks, start with
`docs/script_workflow.md`. It documents the recommended directory layout,
gitignored local path config, and the preferred CLI entry points for 2D
tracking post-processing and 3D reconstruction.

## Quick start

### Environment setup

**One-step install with uv (recommended)** — from the project root:

# 未装 uv 时先装（执行一次即可）
irm https://astral.sh/uv/install.ps1 | iex
# 关掉并重新打开终端后再执行下面

cd D:\medaka-mating-pipeline
uv sync --python 3.11

```bash
uv sync --python 3.11
```

This creates `.venv`, installs dependencies (including PyTorch CUDA 13.0 on Windows/Linux when configured for your GPU), and dev tools. See `environment/requirements.txt` for details and CPU-only options.

**Alternative: pip + venv** — from the project root:

1) Create and activate a virtual environment (example):

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the project in editable mode:

```bash
pip install -e ".[dev]"
```

3) Run tests:

```powershell
pytest
```

Notes:
- The intended Python version is in `.python-version`.
- `environment/requirements.txt` is kept for reference/migration.

### FFmpeg（视频处理，按需安装）

若流程中需要导出视频或使用 ffmpeg 做转码/剪辑，可在项目根目录执行：

```powershell
.\scripts\install-ffmpeg.ps1
```

脚本会下载 [Gyan FFmpeg](https://github.com/GyanD/codexffmpeg/releases) 到 `tools/` 并将 `bin` 加入当前用户 PATH。若遇执行策略限制，先运行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

也可用系统包管理器安装（需自行加入 PATH）：`winget install Gyan.FFmpeg`。

### VS Code / IDE setup

Most team members can simply open the repository folder in VS Code.

- The Python extension will typically auto-detect the `.venv/` environment in the workspace root.
- If it does not, run **Python: Select Interpreter** and choose the `.venv` interpreter.
- Workspace defaults are provided in `.vscode/settings.json` (pytest + `src/` on the Python path).

### Configure experiment

- Configure experiment: edit `configs/project.yaml`
- Configure local machine/data paths: copy `configs/paths.example.toml` to
  `configs/paths.local.toml` and update it for your machine. The local file is
  gitignored.

### Acquisition (campy)

Acquisition is performed using **campy** (external tool): https://github.com/ksseverson57/campy

- See `scripts/acquisition/README.md` and `configs/acquisition/campy_config.yaml`.
- Typical run:

```bash
campy-acquire ./configs/acquisition/campy_config.yaml
```

### Tracking / analysis

- Run inference (HPC): `scripts/sleap/run_inference_slurm.sh`
- Run 2D tracking post-processing: `python -m scripts.tracking.main ...`
- Run 3D reconstruction: `python -m scripts.reconstruction_3d.run_3d_reconstruction ...`
- Compute metrics: `scripts/analysis/compute_courtship_metrics.py` and `scripts/analysis/compute_aggression_metrics.py`

## Reproducibility
- All parameters should live in `configs/`.
- Outputs should be written to `runs/<date>/<experiment_id>/...` (recommended).
