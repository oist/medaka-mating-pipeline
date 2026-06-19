# archive/

Retired, superseded, or one-off material kept for reference. Nothing here is
part of the active pipeline. Code was moved with `git mv` so history is
preserved — recover anything by moving it back or via `git log --follow`.

Moved during the 2026-06-20 directory cleanup.

## Contents

- `scripts/reconstruction/` — legacy step1–step6 3D reconstruction flow.
  Superseded by `scripts/reconstruction_3d/`. Kept for back-compat with
  already-cut legacy clip-CSV recordings.
- `scripts/tracking/multiview_3d_id_assignment.py` — thin wrapper around the
  legacy reconstruction steps above (geometry-based a/b/c assignment for the
  legacy clip-CSV workflow). Moved together with its dependency.
- `scripts/debug/` — one-off issue reproductions and diagnostics
  (`check_id_swap_428_431.py`, `qc_round5_*.py`, `summarize_5k_test.py`, …).
- `notebooks/notebook` — stray scratch planning notes (original Japanese/Chinese
  design jottings), no file extension.
- `logs/` — run logs from the June 2026 idtracker.ai / top-view cleaning
  experiments (`clean_topview_full*.log`, `idtrackerai*.log`). Gitignored
  (`*.log`); not tracked.

## Known dangling reference

`notebooks/05_multiview_3d_id_assignment.ipynb` still imports
`scripts.tracking.multiview_3d_id_assignment`, which now lives here. That
exploratory notebook will not run as-is until its import path is updated or the
module is restored. This is intentional — the legacy path was retired.
