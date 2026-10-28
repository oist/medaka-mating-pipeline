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
- **Notebook rule (2026-10-26)**: From now on the user works in Jupyter notebooks under [archive/notebooks/experiment_workflows/](archive/notebooks/experiment_workflows/) (moved from repo-root `notebook/` on 2026-10-26). Put new analysis/run notebooks there (date- and experiment-prefixed, e.g. `20260106_E_2_1_FCMC_<topic>.ipynb`), keep cell outputs saved in the `.ipynb`, and precede every code cell with a markdown cell explaining what that step does, its inputs/outputs, and its result. Reusable logic still goes in `scripts/`; notebooks call it. Notebooks must not hardcode the repo or data locations: the first code cell walks up from the cwd to find the repo root (`pyproject.toml`), and data roots come from `configs/paths.<USERNAME>.toml` (falls back to `configs/paths.local.toml`); on a new clone/machine only that TOML changes. Template: `batch_pipeline.ipynb`.
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
- Run tests: `pytest -q`. On this machine use `.venv\Scripts\python.exe -m pytest -q -p no:napari -p no:napari-plugin-engine -p no:npe2`: the napari pytest plugin (gui extra) imports Qt before torch, which makes every torch-importing test fail at collection with `WinError 1114` (c10.dll). CI (no napari) is unaffected.
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
- Per-experiment runbook for `20260106_E_2_1_FCMC` (current): [archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_3d_reconstruction_workflow.md](archive/notebooks/experiment_workflows/20260106_E_2_1_FCMC_3d_reconstruction_workflow.md); side1 inference notebook: [archive/notebooks/experiment_workflows/batch_pipeline.ipynb](archive/notebooks/experiment_workflows/batch_pipeline.ipynb). Older round8 version: [archive/notebooks/notebook](archive/notebooks/notebook).
- Current model (2026-10-26): `Z:\medaka\sleap_models\retrain_20260810_3node_reviewed` bottom-up `s14.bottomup_3node.reviewed_{top,side}_f24_ms64_paf8` (skeleton mouth/pectoral/tail). Not launchable by `run_round_inference.py`; run `.venv\Scripts\python.exe -m sleap_nn.cli track --model_paths <run_dir> --device cuda --batch_size 4 --max_instances 6`, one view at a time. Benchmarked 2026-10-26 on side1 (RTX 5070 Ti): batch 2/4/6/8 = 31/33/33/34 FPS (decode/PAF-grouping bound, ~2.9 h per full 348600-frame view); batch 8 peaks at 13.9 GB; batch 12 overflows VRAM into shared memory and drops to 1.35 FPS (~72 h) - never use it. Output tag/subdir `s14_bottomup_3node_reviewed` → `Z:\medaka\sleap_output\<exp>\<tag>\<view>.slp`; Stages 2–5 unchanged (visible-node centroid works for 3 nodes).
- Fallback model: `retrain_20260722_round8` / tag `round8_260723_strictmouth` via `scripts.sleap.run_round_inference --mode track --max-instances 6`.
- That experiment has raw videos plus incomplete round7 top-camera chunks only; 3-node side1 benchmark started 2026-10-26, no full-length 3-node outputs yet.

### Human review of identities / head-tail (2026-10-26, prototype)

- Requirement from the user: every frame of all 66 experiments (35 min each) is watched by a person at <= 1x and identities (F / M1 / M2 = insertion order: 1st fish in = F, 2nd = M1, 3rd = M2) plus head/tail are fixed. Review happens mainly on a laptop (no strong GPU), with packages copied to a 1 TB external SSD.
- Code: [scripts/review/](scripts/review/) (README there); driver notebook [archive/notebooks/experiment_workflows/review_pipeline.ipynb](archive/notebooks/experiment_workflows/review_pipeline.ipynb). Laptop env: `environment/requirements-review.txt` into `.venv-review` (numpy/pandas/opencv/PyQt5 only).
- Packages live in `Z:\medaka\review_packages\<exp>\{proxy,<round_tag>}`. Proxies: 800x800 H.264 (NVENC), `*.play.mp4` = every 6th source frame (frame k = source 6k, verified), `*.full.mp4` = every frame; ~12 min per view, ~35 min and 3.6 GB per experiment on the RTX 5070 Ti (~240 GB for 66). GUI render on this desktop: 8.8 ms per shown frame vs a 36 ms budget at 1x.
- Measured: a full-length proxy transcode slows concurrent local SLEAP inference 32.5 -> ~29 fps (~11%), so `scripts.review.build_packages` pauses proxies while `sleap_nn.cli track` runs (default `--when-inference pause`).
- 3D CSVs only carry the visible-node centroid; `review_data.py` recovers mouth/pectoral/tail via `<tag>_best.json` slot maps (same slot order as `load_2d_slp`), fixes head/tail (cross-view flip search + temporal Viterbi) and triangulates every node. On the 20 s test: mouth-tail length median 25.3 mm, head forward in 98.8% of fast-swimming frames.
- Ledger files per review folder: `human_events.csv` (state-from-frame events: label / flip / uncertain / bad3d), `coverage.csv` (shown intervals + speed), `session_log.csv`; `machine_risk.csv` is frozen for machine-vs-human comparison.
- Status 2026-10-26: first experiment `20260112_P_1_1_FCMC` proxies built; review data only for the 20 s test tag `s14_bottomup_3node_reviewed_test` (full-length 3D pending). Next: user tries the GUI on the laptop, then discuss.
- Status 2026-10-27: user decided to watch every frame (no triage-only shortcut) and to use `review_gui` instead of Rerun (Rerun pre-decodes every full-res frame, unusable at full length). Full-length review data built for `20260112_P_1_1_FCMC` / `s14_bottomup_3node_reviewed` (304 s; 525 risk windows, mostly `close` 317 / `missing_view` 347, covering 151,947 frames = 44% of the video); GUI opens in ~3 s. Full-length 3D also exists for P_1_3, P_1_2, P_2_1, P_2_2, P_2_3 (review data not built yet). Next: user tries the GUI on this desktop and reports what is missing.
- 2026-10-27 GUI v2 after first feedback: (a) the user's "ID2 head is on ID3" cases (frames 10230/10356) were side-view occlusions - two fish overlap along the line of sight, SLEAP puts a node on the neighbour; ID and 3D centroid were fine. Review data v2 (`DATA_VERSION 2`) flags `view_overlap` (skeletons < 20 source px) and the GUI draws those skeletons dashed with `~`; leaving the overlapped view out of node triangulation (`--drop-overlapped-view`) is off by default because on P_1_1 it barely helped (body-length outliers 0.33% -> 0.29%) and made rare large node jumps larger (p99.9 7.5 -> 9.7 mm). (b) GUI: projected 3D cross in every view, auto zoom onto the fish (Z), double-click/V picks the large view, +/- mark size, settings kept in QSettings, opens maximized. (c) The first ~1-1.5 min of each video has hands/net and is not analysed; the reviewer marks the first usable frame per experiment with S (`analysis_start` ledger event; `apply_events` -> `analysed`). P_1_1 v1 data kept in `<review folder>/v1_backup/`.
- 2026-10-27 GUI v3 (user: "finish the review GUI so later work is fast"; analysis will use orientation, tail-beat frequency and speed with the 3 existing nodes). (a) Old label model ("label from here on" per track) produced two F's when edits were made out of order or snapped onto an existing edit; replaced by static names per corrected fish + `swap` events (compose in frame order) - 1/2/3 names once, afterwards swaps; Shift+1/2/3 renames. The first session's 23 old-style events are archived as `human_events.v1_labels.csv` (it contained ID_3=F at 50910 and ID_2=F at 900; ID_2/ID_3 came within 12 mm only at 37664, 47051, 49569). (b) Review data v3: short (<8 frame) head/tail swaps undone per view in 2D and in 3D; the 1-2 frame flips the user saw came from the per-frame cross-view choice flipping one side view wrongly (e.g. frame 26519 side1). Display views oriented by the projected 3D ends. `orientation_segment` lets H flip a whole segment; remaining 3D head/tail jumps are `head_tail_jump` risks. (c) B = body nodes bad, E = edit list (go/delete; removals logged), traces of speed + nearest-fish distance, unified cross symbol, `export_reviewed.py` -> `reviewed_3d.csv` + `reviewed_summary.json` (`ready_for_analysis`). Skipping flagged frames is NOT a valid analysis strategy: node errors concentrate in close interactions (head/tail 3D missing 36% when < 10 mm apart vs 0.08% when > 40 mm), so fix first, then report a sampled error rate. v2 data kept in `v2_backup/`.
- 2026-10-27 requirement (user): the final result must have NO missing values - every body node has a 3D coordinate on every analysed frame - and the GUI must show the 3D reconstruction. Review data v4: `nodes3d_filled` + `node_source` (0 triangulated, 1 one view fitted on that view's ray at the median node-to-pectoral length, 2 body model = anchor + time-interpolated offset); before v4, P_1_1 (frames >= 14940) had mouth 0.43% / tail 0.89% / pectoral 0.09% of fish-frames missing (~80% of those seen in one view; 86% of gaps <= 5 frames). Export carries `<node>_source` so tail-beat analysis can tell measured from filled. GUI: dockable rotatable 3D window (D), missing 2D nodes drawn as projected dashed squares. Reviewer marked analysis start at frame 4152 and named ID_3=F, ID_2=M1, ID_1=M2 (2026-10-27 15:49). v3 data kept in `v3_backup/`.
- 2026-10-27 16:38 the GUI closed without a message (WER: Qt5Core.dll 0xc0000409 = PyQt aborting on an unhandled Python exception in a Qt callback) right after B at frame 22020; not reproducible offscreen. Fix: `install_error_log` (sys.excepthook + faulthandler -> `<review folder>/review_gui_errors.log`, status-bar message, GUI keeps running) and `ReviewLock` ignores locks of dead PIDs on this computer. If it recurs, read that log. User prefers to launch the GUI themselves from a command line.
- 2026-10-27 evening GUI v5 / review data v5 (user-approved design): keys 1/2/3 select the fish named F/M1/M2, 4/5/6 assign (Shift = rename); B now means "body wrong here" (user's meaning: tail short, orientation changed, centre shifted) and the run is re-estimated (mouth/tail from pectoral + length + direction, centre follows); Q/R/T per-view fixes (head/tail reversed, view left out, detections of two fish exchanged) pressed at first and last wrong frame; close co-moving intervals (`machine_intervals.csv`, < 12 mm for >= 0.5 s, user accepted these thresholds provisionally; 23 on P_1_1) with I/Home/End/[ ]/Backspace, C keyframes and Y solve (Viterbi over per-view permutations in `reconstruct.solve_interval`, cached in `<ledger>/interval_solutions/`). Edits are recomputed from `detections.npz` + calibration (`geometry.py`), at GUI start-up too (P_1_1 with the user's ~166 B runs: 13.6 s). Build adds robust two-view re-triangulation + despiking (jitter > 3 mm/frame^2: mouth 1.00->0.62%, tail 2.03->1.45%). On interval M015 (101844-107275, ID_2/ID_3 sinking) solve took 3 s, jumps > 4 mm/frame 553->138, and the identity differed from the user's 50 manual swaps on 428 frames (mostly 105263-105345) - user must check. The user had watched 50.8% of P_1_1 with 441 edits by then. v4 data kept in `v4_backup/`.
  Code-review fixes the same evening: keyframes now pin the confirmed on-screen positions (not the tracker assignment); interval solutions are keyed on keyframes + identity swaps over the interval + overlapping view swaps (stale ones ignored); despike keeps only spike peaks; edits before the first data frame are clipped; open B runs are not re-estimated; export reports `geometry_edits_applied`. Start-up recompute (~24 s for ~118 ranges) is cached in `<ledger>/geometry_cache.npz` at close (next start 0.4 s). Final P_1_1 numbers after the rebuild: M015 jumps > 4 mm/frame 854->282 (ledger copy of that evening), identity differs from the user's swaps on ~320-410 frames around 104929-105200 - point the user there.
- 2026-10-27 user: manual-correction GUI "broadly done", now improvements. Added a BORIS behaviour row (`scripts/review/behaviors.py`): BORIS CSVs live in the repo's `archive/boris/` (P group and later sets in `archive/boris/p2/`; untracked, added by the user 2026-10-27), `Image index` = top-camera source frame at 166 fps = review frame. Lanes to-F / M-M / solo, colour by normalised category (courtship / aggressive (incl. "Competitive") / mating / other), Ctrl+Left/Right jump between bouts, `--boris` overrides the search. P_1_1: 110 bouts (52 courtship + 3 aggressive towards F, 25 aggressive + 27 courtship between males). No Node.js on this PC (the dataviz palette validator cannot run); colours are the dataviz reference palette's first three dark-mode slots, documented as validated all-pairs.

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
