# Script-Driven Workflow

This repository works better as a terminal-first pipeline than as a
notebook-first project. Keep notebooks optional for quick inspection or plots,
but put reusable logic and repeatable runs behind scripts.

## Recommended layout

Keep code in the repository, and keep large data on external storage.

Suggested data layout on your external drive:

```text
<data_root>/
  raw/
  sleap_output/
  processed/
  calibration_model/
  reconstruction_3d/
  runs/
```

Use the repository for:

- `src/`: reusable library code
- `scripts/`: command-line entry points and stage wrappers
- `configs/`: committed shared config and gitignored local path config
- `scripts/debug/`: one-off issue reproductions and diagnostics

## Local path config

Copy [`configs/paths.example.toml`](../configs/paths.example.toml) to
`configs/paths.local.toml` and fill in your machine-specific directories.

The local file is gitignored, so you can keep personal drive mappings there.

## Preferred entry points

2D tracking post-processing:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331"
```

Or start from a specific PLAYER1 CSV path:

```powershell
python -m scripts.tracking.main --paths-config configs/paths.local.toml `
  "G:\...\top_mouth_180\...\PLAYER1...\csv"
```

3D reconstruction:

```powershell
python -m scripts.reconstruction_3d.run_3d_reconstruction `
  --paths-config configs/paths.local.toml `
  --set_key "20251225_K_1_2_FCMC_m1tf_Courtship_290.729-296.331"
```

If `processed_dir` or `calibration_model_dir` are not defined in the local
paths file, pass them explicitly with `--processed_dir` and `--model_folder`.

## Troubleshooting practice

- Reproduce bugs with small clips first.
- Save intermediate CSV and JSON outputs instead of debugging only in memory.
- Keep one-off reproductions in `scripts/debug/`.
- Keep production-stage scripts deterministic and config-driven.
- Prefer one clear CLI per stage over notebook-only logic.

## Notes

- Prefer `scripts/reconstruction_3d/` over the older `scripts/reconstruction/`
  flow for new work unless you specifically need the legacy path.
- Commit shared config templates, not your machine-specific path mappings.
