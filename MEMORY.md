# MEMORY — Project Handoff Document

> Read this file at the start of every session (see [CLAUDE.md](CLAUDE.md) → *Session Start Rules*).
> When project state, rules, or procedures change, update this file in the same turn.

Last updated: 2026-10-26

---

## User Profile

- **Role**: Researcher (OIST-affiliated) studying medaka fish mating behavior.
- **Goal**: Quantify female mate choice by measuring male aggression vs. courtship intensity in a 3-fish (1 female + 2 males) setup, using 3-camera 3D tracking at 166 fps.
- **Tech stack**: Windows 11, Python 3.11 managed with `uv`, virtualenv at `.venv/`. CUDA GPU available for SLEAP / TensorRT. Working drives include local SSD, network share `Z:\medaka\` (current machine; `configs/paths.local.toml` still lists the older `Z:\StephensU\medaka` layout), and Google Drive `G:\.shortcut-targets-by-id\...\OIST 1215-0122\`.
- **Collaboration style**: User provides domain knowledge and validation; Claude implements. Commits are co-authored.
- **Notebook rule (2026-10-26)**: From now on the user works in Jupyter notebooks under [archive/notebooks/experiment_workflows/](archive/notebooks/experiment_workflows/) (moved from repo-root `notebook/` on 2026-10-26). Put new analysis/run notebooks there (date- and experiment-prefixed, e.g. `20260106_E_2_1_FCMC_<topic>.ipynb`), keep cell outputs saved in the `.ipynb`, and precede every code cell with a markdown cell explaining what that step does, its inputs/outputs, and its result. Reusable logic still goes in `scripts/`; notebooks call it. Notebooks must not hardcode the repo or data locations: the first code cell walks up from the cwd to find the repo root (`pyproject.toml`), and data roots come from `configs/paths.<USERNAME>.toml` (falls back to `configs/paths.local.toml`); on a new clone/machine only that TOML changes. Template: `20260106_E_2_1_FCMC_side1_sleap_inference.ipynb`.
- **Language rule (2026-10-26)**: Everything outside `archive/notebooks/experiment_workflows/` (code, comments, docs, configs, commit messages, this file) must be written in English. Content inside `archive/notebooks/experiment_workflows/` may be in Chinese.

## Current Project Structure

Top-level layout (see [CLAUDE.md](CLAUDE.md) → *Repository Layout* for the full tree):

```
medaka-mating-pipeline/
├── scripts/                  # All active logic
│   ├── acquisition/          # Multi-camera recording (campy, Basler)
│   ├── calibration/          # Bead/object calibration + lib/
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
├── sam2_id_experiment/       # SAM2/Detectron2 segmentation-assisted ID tools
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
- **Local SLEAP inference path**: torch via `sleap_nn.cli track` (`scripts/sleap/run_round_inference.py --mode track`, the default). TensorRT export is **broken on the RTX 5070 Ti** (Blackwell `peaks_castOut` op has no TRT 10.16 tactic), and ONNX adds no worthwhile speedup — so do **not** propose `--mode predict`/TRT export on this machine. See [AGENTS.md](AGENTS.md) "3D Tracking Runbook".
- **CI**: GitHub Actions runs `pytest -q` on Ubuntu / Python 3.11 using `environment/requirements.txt`.
- **Calibration production model**: `calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models/imCoords_to_XYZ.joblib` (0.90 mm mean / 1.00 mm RMSE / 2.15 mm max over 228 points).
- **2026-06-06 precision/ID upgrade**:
  - 3D-seeded SAM2 prompt mining lives in `sam2_id_experiment/src/sam2_adapter/prompt_seed_from_3d.py` with CLI `sam2_id_experiment/scripts/24_seed_prompts_from_3d.py`.
  - Segmentation top-track export now pools multiple short fragments by biological `assigned_identity` while preserving `identity_trajectory_id` provenance.
  - `scripts/reconstruction_3d/bind_topcam_id.py` now votes by stable `source_bind_key`, so fragmented segmentation IDs can replace/augment `top_cam.tracks.csv`.
  - Arbitrary moved rigid-object calibration refinement lives in `scripts/calibration/refine_calibration_rigid_object.py`.

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

- Sync env: `uv sync --python 3.11` (repo standard; optional: `--extra rerun`, `--extra sleap`).
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

### Current 3D reconstruction procedure (this machine)

- Data live at `Z:\medaka` (`medaka_202512`, `sleap_models`, `sleap_output`, `reconstruction_3d`).
- Per-experiment runbook for `20260106_E_2_1_FCMC` (current): [archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_3d_reconstruction_workflow.md](archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_3d_reconstruction_workflow.md); side1 inference notebook: [archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_side1_sleap_inference.ipynb](archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_side1_sleap_inference.ipynb). Older round8 version: [archive/notebooks/notebook](archive/notebooks/notebook).
- Current model (2026-10-26): `Z:\medaka\sleap_models\retrain_20260810_3node_reviewed` bottom-up `s14.bottomup_3node.reviewed_{top,side}_f24_ms64_paf8` (skeleton mouth/pectoral/tail). Not launchable by `run_round_inference.py`; run `.venv\Scripts\python.exe -m sleap_nn.cli track --model_paths <run_dir> --device cuda --batch_size 4 --max_instances 6`, one view at a time. Benchmarked 2026-10-26 on side1 (RTX 5070 Ti): batch 2/4/6/8 = 31/33/33/34 FPS (decode/PAF-grouping bound, ~2.9 h per full 348600-frame view); batch 8 peaks at 13.9 GB; batch 12 overflows VRAM into shared memory and drops to 1.35 FPS (~72 h) - never use it. Output tag/subdir `s14_bottomup_3node_reviewed` → `Z:\medaka\sleap_output\<exp>\<tag>\<view>.slp`; Stages 2–5 unchanged (visible-node centroid works for 3 nodes).
- Fallback model: `retrain_20260722_round8` / tag `round8_260723_strictmouth` via `scripts.sleap.run_round_inference --mode track --max-instances 6`.
- That experiment has raw videos plus incomplete round7 top-camera chunks only; 3-node side1 benchmark started 2026-10-26, no full-length 3-node outputs yet.

### Completed

- HPC-based SLEAP inference pipeline with per-view chunked `.slp` output.
- TensorRT export path for ~2.5× inference speedup.
- 3D-first tracker (`scripts.reconstruction_3d.reconstruct_from_slp`) with per-frame cross-view permutation scoring and velocity/gap-aware Hungarian matching.
- Extra-candidates 3D reconstructor with cam0 recovery via pairwise model (round-2 hit 99.02% 3-fish coverage).
- Calibration model (`cali_20251217_manual`) accepted as production.
- ID consistency evaluation script: [scripts/tracking/evaluate_id_consistency.py](scripts/tracking/evaluate_id_consistency.py) (added 2026-05-11).
- 3D-seeded segmentation-ID handoff:
  - Mine clean prompts from `<tag>_3d_identified.csv` + `<tag>_best_pre_link.json`:
    `.venv\Scripts\python.exe -m sam2_id_experiment.scripts.24_seed_prompts_from_3d --identified-3d-csv <tag>_3d_identified.csv --best-pre-link-json <tag>_best_pre_link.json --paths-config configs/paths.local.toml --output-dir runs/seg_id/<experiment>/<round_tag>/seed_from_3d`
  - Outputs: `clean_seed_windows.csv`, `clean_seed_points.csv`, `sam2_seed_prompts_from_3d.yaml`, `sam2_seed_prompts_from_3d_summary.json`.
  - Use those prompts with the existing SAM2/Detectron2 scripts, then export a bind-compatible top track CSV via `export_segmentation_top_tracks`; pass it to `reconstruct_from_slp --bind-top-id --top-tracks-csv <top_cam.seg_identity_tracks.csv>`.
- Rigid-object calibration refinement:
  - New input contract: `rigid_object_geometry.csv` with `marker_id,object_x,object_y,object_z`; `rigid_object_image_coords.csv` with `pose_id,marker_id,cam0_x,cam0_y,cam1_x,cam1_y,cam2_x,cam2_y`.
  - Run `.venv\Scripts\python.exe -m scripts.calibration.refine_calibration_rigid_object --rigid-object-geometry-csv <geometry.csv> --rigid-object-image-coords-csv <image_coords.csv> --base-model-folder <current_models> --output-folder <new_models> --results-root <qc_dir>`.
  - Writes `imCoords_to_XYZ.joblib`, `XYZ_to_imCoords.joblib`, pairwise helper models, `rigid_object_refined_points.csv`, and `calibration_refinement_report.json`.

### Low priority / next up

- Behavioral metric scripts in [scripts/analysis/](scripts/analysis/) (`compute_*_metrics.py`) — currently `SystemExit("TODO: ...")` placeholders.
- Continue SLEAP retraining with more labels from failure modes: high residual, dropped fish, speed spikes, crossings, reflections, tank edges/depth extremes, and occlusion windows. Validate by downstream 3D metrics, not only SLEAP loss.

---

*Reference: longer-lived per-topic notes (user profile, data paths, SLEAP failure modes, retrain plan, 3D recovery design, etc.) are stored in the auto-memory index at `~/.claude/projects/c--Users-Makoto-Hiroi-Documents-GitHub-medaka-mating-pipeline/memory/MEMORY.md` and its linked files. This `MEMORY.md` is the short, regularly-updated handoff snapshot.*
