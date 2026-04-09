# Debug Scripts

This directory is for one-off investigation helpers that are useful during
analysis or troubleshooting but are not part of the repeatable production
pipeline.

Guidelines:

- Keep reusable pipeline logic in `src/` or the stage-specific modules under `scripts/`.
- Keep ad hoc reproductions, sanity checks, and issue-specific diagnostics here.
- Prefer descriptive filenames tied to the problem they reproduce.
- Do not make notebooks the only place where an investigation can be rerun.

Current examples:

- `check_id_swap_428_431.py`: reproduce the Hungarian assignment around the
  known 2D ID-swap case near frames 428 to 431.
