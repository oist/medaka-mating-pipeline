# MEMORY — Project Handoff Document

> Read this file at the start of every session (see [CLAUDE.md](CLAUDE.md) → *Session Start Rules*).
> When project state, rules, or procedures change, update this file in the same turn.

Last updated: 2026-05-11

---

## User Profile

- **Role**: Researcher (OIST-affiliated) studying medaka fish mating behavior.
- **Goal**: Quantify female mate choice by measuring male aggression vs. courtship intensity in a 3-fish (1 female + 2 males) setup, using 3-camera 3D tracking at 166 fps.
- **Tech stack**: Windows 11, Python 3.11 managed with `uv`, virtualenv at `.venv/`. CUDA GPU available for SLEAP / TensorRT. Working drives include local SSD, network share `Z:\StephensU\medaka\`, and Google Drive `G:\.shortcut-targets-by-id\...\OIST 1215-0122\`.
- **Collaboration style**: User provides domain knowledge and validation; Claude implements. Commits are co-authored.

## Current Project Structure

Top-level layout (see [CLAUDE.md](CLAUDE.md) → *Repository Layout* for the full tree):

```
medaka-mating-pipeline/
├── scripts/                  # All active logic
│   ├── acquisition/          # Multi-camera recording (campy, Basler)
│   ├── calibration/          # Bead-based calibration + lib/
│   ├── hpc/                  # Slurm SLEAP inference on OIST saion
│   ├── sleap/                # SLEAP batch inference & visualization
│   ├── tracking/             # 2D tracking (legacy + .slp workflow)
│   ├── reconstruction_3d/    # Preferred 3D reconstruction (3D-first tracker)
│   ├── reconstruction/       # Legacy step1–step6 (back-compat only)
│   ├── analysis/             # Behavior metrics (placeholders) + renderers
│   ├── visualization/        # rerun_viewer.py
│   ├── common/               # local_paths.py, experiment_metadata.py
│   └── debug/                # One-off bug repros
├── src/medaka_pipeline/      # Package scaffold only (not used yet)
├── configs/                  # YAML/TOML configs (acquisition, sleap, paths)
├── calibration_results/      # Checked-in calibration data + trained models
├── tests/                    # pytest unit tests
├── docs/                     # Documentation
├── runs/                     # Per-run analysis outputs (gitignored data)
├── CLAUDE.md                 # Project instructions for Claude Code
└── MEMORY.md                 # This file
```

## Automation Systems In Place

- **HPC SLEAP inference** (`scripts/hpc/`): One Slurm array job per view × N chunks runs on OIST saion. Outputs land in `/bucket/StephensU/medaka/sleap_output/<experiment>/`. Merge happens locally via `scripts.tracking.main --merge-chunks`. See [scripts/hpc/README.md](scripts/hpc/README.md).
- **TensorRT-accelerated inference**: `sleap-nn export` + `predict` gives ~2.5× speedup over PyTorch; use `--device cuda` (not `cuda:0`); export FP16 engines.
- **CI**: GitHub Actions runs `pytest -q` on Ubuntu / Python 3.11 using `environment/requirements.txt`.
- **Calibration production model**: `calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models/imCoords_to_XYZ.joblib` (0.90 mm mean / 1.00 mm RMSE / 2.15 mm max over 228 points).

## Key Technical Info

### Environment variables (HPC SLEAP)

Set these when invoking `scripts/hpc/` from the saion login node (see [scripts/hpc/README.md](scripts/hpc/README.md) for the full table):

- `MEDAKA_BUCKET_ROOT` — root for SLEAP output on `/bucket`.
- `NUM_CHUNKS` — number of array tasks per view.
- `PARTITION` — Slurm partition.
- Model paths (per view) — set per the README.
- `DRY_RUN=1` — preview submission without running.

### Required local config

- `configs/paths.local.toml` (gitignored). Copy from `configs/paths.example.toml` and fill in machine-specific paths. Loader: [scripts/common/local_paths.py](scripts/common/local_paths.py).
  - Required keys: `data_root`, `raw_data_dir`, `sleap_output_dir`, `processed_dir`, `tracking_output_dir`, `calibration_model_dir`, `reconstruction_output_dir`, `runs_dir`, `medaka_data_dir`, `sleap_models_dir`.

### Common commands / binary paths

- Sync env: `uv sync --python 3.13` (optional: `--extra rerun`, `--extra sleap`).
- Run a Python script: `.venv\Scripts\python.exe -m scripts.<module> ...` (Windows PowerShell).
- Run tests: `pytest -q`.
- ffmpeg: installed under `C:\ffmpeg\` on the user's PCs, but the exact subfolder (version) varies per machine and per colleague — **do not hardcode a single path**. Discover it at the start of each session when needed:
  - PowerShell: `Get-ChildItem C:\ffmpeg -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 5`
  - Or check `PATH`: `Get-Command ffmpeg -ErrorAction SilentlyContinue`
  - If multiple versions are present, ask the user which one to use.

### Credentials / secrets

- No API keys or credentials live in this repo. Network drive access (`Z:`) relies on Windows mapped-drive auth; Google Drive (`G:`) relies on Drive for Desktop.
- HPC access (saion) uses the user's OIST SSH key — never commit private keys.

## Done / Remaining Tasks

### Completed

- HPC-based SLEAP inference pipeline with per-view chunked `.slp` output.
- TensorRT export path for ~2.5× inference speedup.
- 3D-first tracker (`scripts.reconstruction_3d.reconstruct_from_slp`) with per-frame cross-view permutation scoring and velocity/gap-aware Hungarian matching.
- Extra-candidates 3D reconstructor with cam0 recovery via pairwise model (round-2 hit 99.02% 3-fish coverage).
- Calibration model (`cali_20251217_manual`) accepted as production.
- ID consistency evaluation script: [scripts/tracking/evaluate_id_consistency.py](scripts/tracking/evaluate_id_consistency.py) (added 2026-05-11).

### Low priority / next up

- Behavioral metric scripts in [scripts/analysis/](scripts/analysis/) (`compute_*_metrics.py`) — currently `SystemExit("TODO: ...")` placeholders.
- SLEAP retraining round 2: add `tail` node, mine frames from 24 inferred experiments (see auto-memory `project_sleap_retrain_plan.md`).

---

*Reference: longer-lived per-topic notes (user profile, data paths, SLEAP failure modes, retrain plan, 3D recovery design, etc.) are stored in the auto-memory index at `~/.claude/projects/c--Users-Makoto-Hiroi-Documents-GitHub-medaka-mating-pipeline/memory/MEMORY.md` and its linked files. This `MEMORY.md` is the short, regularly-updated handoff snapshot.*
