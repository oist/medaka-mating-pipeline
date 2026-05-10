from __future__ import annotations

"""
Build a self-contained share folder for the multi-view annotator.

The annotator script imports a couple of helper modules from the rest of
the repo and reads calibration ``.joblib`` files from a network drive,
neither of which is convenient to ship to a colleague who just wants to
review and correct labels. This helper produces a tidy folder containing
everything the annotator needs:

    <out_dir>/
      pyqt_multiview_annotator.py
      annotator_helpers.py            # local_paths + Calibration inlined
      calibration/
        imCoords_to_XYZ.joblib
        XYZ_to_imCoords.joblib
        xy_yz_to_xz.joblib
        xz_yz_to_xy.joblib
        xz_xy_to_yz.joblib
      data/
        annotation_seed.pkg.slp       # input pkg.slp (full or quick)
      output/                         # default destination for saves
      paths.local.toml                # auto-pointed at the bundled paths
      pyproject.toml                  # uv sync entry point
      requirements.txt                # fallback for non-uv users
      run.bat                         # Windows launcher (uv run)
      run.sh                          # POSIX launcher (uv run)
      README.md                       # setup + usage instructions

Usage
-----
    python -m scripts.sleap.make_annotator_bundle \
        --seed C:/Users/.../annotation_seed_round3_full.pkg.slp \
        --paths-config configs/paths.local.toml \
        --out-dir C:/Users/.../colleague_bundle

Notes
-----
- The seed should ideally be a *full* (embedded-frame) pkg.slp produced by
  Ctrl+Shift+S in the annotator. With ``embed="user"`` every labelled
  frame is inside the file, so the colleague needs no video access to
  open and edit the embedded frames. (They still need video access if you
  want them to fill in *missing* views via the calibration prediction
  pathway.)
- The bundle is portable across Windows / macOS / Linux as long as
  Python 3.11+ is available.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


PYPROJECT_TOML = """\
[project]
name = "medaka-multiview-annotator"
version = "0.1.0"
description = "Tiled-view annotator for the medaka mating pipeline (colleague bundle)."
requires-python = ">=3.11,<3.14"
dependencies = [
    "numpy",
    "scipy",
    "joblib",
    "scikit-learn",
    "opencv-python",
    "sleap-io",
    "PyQt5",
    # pyqt5-qt5 5.15.18 (current latest) ships only Linux/macOS wheels.
    # Pin the Windows side to the last release that has win_amd64 wheels.
    "pyqt5-qt5==5.15.2; sys_platform == 'win32'",
]

[tool.uv]
package = false
"""

REQUIREMENTS_TXT = """\
# Fallback for non-uv users. Prefer `uv sync` which uses pyproject.toml.
numpy
scipy
joblib
scikit-learn
opencv-python
sleap-io
PyQt5
"""

RUN_BAT = """\
@echo off
REM One-shot launcher for the medaka multi-view annotator bundle.
REM Run `uv sync` once before this to install dependencies into ./.venv.

setlocal
cd /d "%~dp0"
uv run python pyqt_multiview_annotator.py ^
    --in   "data\\annotation_seed.pkg.slp" ^
    --out  "output\\edits.pkg.slp" ^
    --paths-config paths.local.toml ^
    --min-views 1 ^
    %*
endlocal
"""

RUN_SH = """\
#!/usr/bin/env bash
# One-shot launcher for the medaka multi-view annotator bundle.
# Run `uv sync` once before this to install dependencies into ./.venv.
set -euo pipefail
cd "$(dirname "$0")"
uv run python pyqt_multiview_annotator.py \\
    --in   "data/annotation_seed.pkg.slp" \\
    --out  "output/edits.pkg.slp" \\
    --paths-config paths.local.toml \\
    --min-views 1 \\
    "$@"
"""

README = """\
# Multi-view annotator (medaka mating pipeline) — colleague bundle

This folder is everything you need to review and correct the multi-view
annotations for our medaka tracking experiments. No clone, no external
paths to configure.

## Setup (once)

1. Install [`uv`](https://docs.astral.sh/uv/) (one-line install — see the
   uv docs for your OS).
2. Open this folder in a terminal and run:

   ```
   uv sync
   ```

   That creates `./.venv` and installs every dependency listed in
   `pyproject.toml`. Done in a couple of minutes.

   *(Without uv: `python -m pip install -r requirements.txt` works too,
   provided you already have Python 3.11+.)*

## Run

- **Windows**: double-click **`run.bat`**, or `./run.bat` from a terminal.
- **macOS / Linux**: `bash run.sh` (or `./run.sh` after `chmod +x`).

The annotator opens a window with three camera panes. Hotkeys are listed
in the docked Help panel on the right.

## Layout

- `data/annotation_seed.pkg.slp` — the file you'll edit.
- `output/edits.pkg.slp` — written by **`Ctrl+S`** (quick save; references
  the seed's embedded frames). Use during iterative work.
- `output/edits_full.pkg.slp` — written by **`Ctrl+Shift+S`** (fully
  self-contained, embeds every user-labeled frame). Use this when you're
  ready to send your work back.
- `calibration/` — calibration models for cross-view prediction.
  Don't edit.

## When you're done

Send `output/edits_full.pkg.slp` back. That's the only file we need.

## Hotkey cheat-sheet

| Key | Action |
|-----|--------|
| `←` / `→` | previous / next frame group |
| `=` / `-` / `0` | zoom in / out / fit |
| Mouse wheel | zoom toward cursor |
| Drag empty area | pan (the active view leads, the others follow) |
| Click (no drag) | re-center all 3 views on the click via calibration |
| `I` | toggle Add-instance mode (then click mouth, click tail) |
| `F` | re-match all fish across views via calibration; predict missing views |
| Drag a keypoint | move it; release to commit |
| Right-click a keypoint | toggle hidden ↔ visible |
| Ctrl + right-click a keypoint | menu: Swap ID, delete keypoint / instance / fish |
| `Esc` | cancel pending instance |
| `Ctrl+S` | quick save |
| `Ctrl+Shift+S` | full save (embedded frames) |
"""


# ---------------------------------------------------------------------------
# Helper module bundling
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _build_helpers_module() -> str:
    """Concatenate the contents of the two helper modules into a single
    file the annotator can import. We keep their public APIs so the
    annotator's existing ``from scripts.common.local_paths import ...``
    and ``from scripts.reconstruction_3d.calibration import Calibration``
    statements work after a small redirection at the top of the
    annotator.

    The original ``local_paths.py`` resolves relative paths against the
    repo root via ``parents[2]``. In the bundled layout the helper file
    sits at the *bundle root*, so we patch ``REPO_ROOT`` to point at the
    bundle directory — that makes ``medaka_data_dir = "data/videos"`` in
    the bundled ``paths.local.toml`` resolve to
    ``<bundle>/data/videos`` regardless of where the colleague placed
    the folder.
    """
    local_paths = _read(REPO_ROOT / "scripts" / "common" / "local_paths.py")
    calibration = _read(
        REPO_ROOT / "scripts" / "reconstruction_3d" / "calibration.py"
    )

    # In the bundled layout the helper file is at the bundle root, so
    # parents[2] would point above it; patch to parent.
    local_paths_patched = local_paths.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[2]",
        "REPO_ROOT = Path(__file__).resolve().parent",
    )
    if local_paths_patched == local_paths:
        # Defensive: if upstream changes the line we want to know.
        raise RuntimeError(
            "Couldn't find REPO_ROOT line in local_paths.py to patch. "
            "Update make_annotator_bundle.py to match."
        )

    return (
        '"""\n'
        "Bundled helper module for the multi-view annotator.\n"
        "\n"
        "Concatenated from:\n"
        "  - scripts/common/local_paths.py  (REPO_ROOT patched)\n"
        "  - scripts/reconstruction_3d/calibration.py\n"
        "\n"
        "Do not edit by hand. Regenerate via\n"
        "scripts/sleap/make_annotator_bundle.py.\n"
        '"""\n\n'
        "# ============================================================\n"
        "# local_paths.py\n"
        "# ============================================================\n\n"
        + local_paths_patched
        + "\n\n# ============================================================\n"
        "# calibration.py\n"
        "# ============================================================\n\n"
        + calibration
    )


def _patch_annotator_imports(src: str) -> str:
    """Rewrite the annotator's repo-relative imports to point at the
    inlined helper module."""
    src = src.replace(
        "from scripts.common.local_paths import choose_path, load_local_paths",
        "from annotator_helpers import choose_path, load_local_paths",
    )
    src = src.replace(
        "from scripts.reconstruction_3d.calibration import Calibration",
        "from annotator_helpers import Calibration",
    )
    # Drop the sys.path hack — everything we need is in the same dir now.
    src = src.replace(
        "REPO_ROOT = Path(__file__).resolve().parents[2]\n"
        "if str(REPO_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_ROOT))",
        "# (sys.path bootstrap not needed in the bundled layout)",
    )
    return src


# ---------------------------------------------------------------------------
# Calibration discovery
# ---------------------------------------------------------------------------

CALIBRATION_FILES = (
    "imCoords_to_XYZ.joblib",
    "XYZ_to_imCoords.joblib",
    "xy_yz_to_xz.joblib",
    "xz_yz_to_xy.joblib",
    "xz_xy_to_yz.joblib",
)


def _resolve_calibration_dir(arg: str | None, paths_config: str | None) -> Path:
    if arg:
        p = Path(arg).resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"--calibration-dir not found: {p}")
    if paths_config:
        try:
            from scripts.common.local_paths import load_local_paths
            local = load_local_paths(paths_config)
            if local and local.calibration_model_dir:
                p = Path(local.calibration_model_dir).resolve()
                if p.is_dir():
                    return p
        except Exception:
            pass
    raise SystemExit(
        "Calibration directory not provided. Pass --calibration-dir or set "
        "calibration_model_dir in --paths-config."
    )


def _build_paths_local_toml() -> str:
    return (
        "[paths]\n"
        '# Auto-generated for the colleague bundle. Pointers are relative\n'
        '# to the bundle root, so no further configuration is needed.\n\n'
        'medaka_data_dir = "data/videos"\n'
        'calibration_model_dir = "calibration"\n'
    )


# ---------------------------------------------------------------------------
# Bundle build
# ---------------------------------------------------------------------------

def build_bundle(
    *,
    seed: Path,
    out_dir: Path,
    calibration_dir: Path,
    annotator_path: Path,
    expand_missing_views: bool = False,
    medaka_data_dir: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "data" / "videos").mkdir(exist_ok=True)
    (out_dir / "output").mkdir(exist_ok=True)

    # 1. Copy / expand the seed.
    seed_dst = out_dir / "data" / "annotation_seed.pkg.slp"
    if expand_missing_views:
        if medaka_data_dir is None:
            raise SystemExit(
                "--expand-missing-views needs --medaka-data-dir or "
                "medaka_data_dir in paths.local.toml."
            )
        from scripts.sleap.expand_seed_with_missing_views import expand
        print("  expanding seed: adding empty LFs for missing views, "
              "embedding frames…")
        expand(seed, seed_dst, medaka_data_dir)
    elif seed.resolve() == seed_dst.resolve():
        # Re-running the bundler with --seed already pointing at the
        # bundle's seed — nothing to copy.
        print(f"  seed already in place at {seed_dst}; skipping copy")
    else:
        shutil.copy2(seed, seed_dst)
    print(f"  seed -> {seed_dst}  ({seed_dst.stat().st_size / 1e6:.1f} MB)")

    # 2. Calibration files.
    for name in CALIBRATION_FILES:
        src = calibration_dir / name
        if not src.exists():
            print(f"  WARNING: calibration file missing: {src}  (skipping)")
            continue
        shutil.copy2(src, out_dir / "calibration" / name)
        print(f"  copied calibration/{name}")

    # 3. Inlined helpers + patched annotator script.
    helpers_src = _build_helpers_module()
    (out_dir / "annotator_helpers.py").write_text(helpers_src, encoding="utf-8")
    print(f"  wrote annotator_helpers.py  ({len(helpers_src)} chars)")

    annotator_src = _patch_annotator_imports(_read(annotator_path))
    (out_dir / "pyqt_multiview_annotator.py").write_text(
        annotator_src, encoding="utf-8"
    )
    print(f"  wrote pyqt_multiview_annotator.py")

    # 4. paths.local.toml + ancillaries.
    (out_dir / "paths.local.toml").write_text(
        _build_paths_local_toml(), encoding="utf-8"
    )
    (out_dir / "pyproject.toml").write_text(PYPROJECT_TOML, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(REQUIREMENTS_TXT, encoding="utf-8")
    (out_dir / "run.bat").write_text(RUN_BAT, encoding="utf-8")
    sh_path = out_dir / "run.sh"
    sh_path.write_text(RUN_SH, encoding="utf-8")
    try:
        # Make sure run.sh is executable on POSIX hosts.
        sh_path.chmod(0o755)
    except OSError:
        pass
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    print()
    print(f"Bundle ready: {out_dir}")
    print("Hand the whole folder to your colleague. They run `run.bat` "
          "(Windows) or `bash run.sh` (mac/linux).")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed", type=Path, required=True,
        help="The pkg.slp to annotate (ideally the *_full.pkg.slp from "
        "Ctrl+Shift+S so all frames are embedded).",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Destination folder for the bundle.",
    )
    ap.add_argument(
        "--paths-config", default=None,
        help="paths.local.toml — used to locate calibration if "
        "--calibration-dir not given.",
    )
    ap.add_argument(
        "--calibration-dir", default=None,
        help="Override calibration model dir (containing the 5 .joblib files).",
    )
    ap.add_argument(
        "--annotator-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "sleap" / "pyqt_multiview_annotator.py",
        help="Source of the annotator script (defaults to the repo path).",
    )
    ap.add_argument(
        "--expand-missing-views",
        action="store_true",
        help=(
            "Before bundling, run expand_seed_with_missing_views on the "
            "seed so every (experiment, frame_idx) group has all 3 views "
            "embedded. Required for colleagues who don't have access to "
            "the source video drive (Z:\\)."
        ),
    )
    ap.add_argument(
        "--medaka-data-dir",
        default=None,
        help="Source video root, used only with --expand-missing-views.",
    )
    args = ap.parse_args(argv)

    if not args.seed.exists():
        ap.error(f"seed not found: {args.seed}")
    if not args.annotator_script.exists():
        ap.error(f"annotator script not found: {args.annotator_script}")

    # Repo-imports are needed to resolve calibration via paths.local.toml.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    calibration_dir = _resolve_calibration_dir(
        args.calibration_dir, args.paths_config,
    )

    medaka_data_dir: Optional[Path] = None
    if args.expand_missing_views:
        local = None
        if args.paths_config:
            try:
                from scripts.common.local_paths import load_local_paths
                local = load_local_paths(args.paths_config)
            except Exception:
                pass
        m_str = args.medaka_data_dir or (
            local.medaka_data_dir if local else None
        )
        if not m_str:
            raise SystemExit(
                "--expand-missing-views needs --medaka-data-dir or "
                "medaka_data_dir in paths.local.toml."
            )
        medaka_data_dir = Path(m_str)

    print(f"Building annotator bundle:")
    print(f"  seed                    : {args.seed}")
    print(f"  calibration_dir         : {calibration_dir}")
    print(f"  annotator_script        : {args.annotator_script}")
    print(f"  out_dir                 : {args.out_dir}")
    print(f"  expand_missing_views    : {args.expand_missing_views}")
    if medaka_data_dir is not None:
        print(f"  medaka_data_dir         : {medaka_data_dir}")
    print()

    build_bundle(
        seed=args.seed,
        out_dir=args.out_dir,
        calibration_dir=calibration_dir,
        annotator_path=args.annotator_script,
        expand_missing_views=args.expand_missing_views,
        medaka_data_dir=medaka_data_dir,
    )


if __name__ == "__main__":
    main()
