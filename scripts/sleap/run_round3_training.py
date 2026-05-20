from __future__ import annotations

"""
Sequential launcher for round-3 sleap-nn training on the medaka pipeline.

Runs 4 trainings in order (single GPU on this Windows box):
    top/centroid       (n=166, dataset under top/labels.pkg.slp)
    top/centered_instance
    side/centroid      (n=334, dataset under side/labels.pkg.slp)
    side/centered_instance

Each training's stdout+stderr is *redirected to a file* (NOT piped through a
shell — see memory/feedback_sleap_training_setup.md). Run names are stamped
with a date + round tag + dataset size to keep them distinguishable from
round-2 results that live in the same runs/ folder.

Usage:
    python -m scripts.sleap.run_round3_training \\
        --round-dir "Z:/StephensU/medaka/sleap_models/retrain_20260508" \\
        --round-tag "round3_260512"
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


VIEWS = ["top", "side"]
KINDS = ["centroid", "centered_instance"]


def _count_labels(labels_slp: Path) -> int | None:
    """Return number of labeled frames in a .pkg.slp, or None on any error."""
    try:
        import sleap_io as sio
        labels = sio.load_slp(str(labels_slp))
        return int(len(labels.labeled_frames))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not count labels in {labels_slp}: {exc}")
        return None


def run_one(
    *, config_path: Path, run_name: str, log_path: Path, dry_run: bool,
) -> int:
    # Invoke the sleap_nn CLI module via the venv interpreter rather than the
    # `sleap-nn.exe` setuptools wrapper. On Windows with Application Control,
    # newly installed/updated wrappers can trip [WinError 4551] until the IT
    # policy catches up; the python -m form bypasses that since the venv's
    # python is already trusted.
    cmd = [
        sys.executable, "-m", "sleap_nn.cli", "train",
        "--config-name", config_path.name,
        "--config-dir", str(config_path.parent),
        f"trainer_config.run_name={run_name}",
    ]
    print(f"\n--- {run_name} ---")
    print(f"cwd        : {config_path.parent}")
    print(f"config     : {config_path}")
    print(f"run_name   : {run_name}")
    print(f"log        : {log_path}")
    print(f"command    : {' '.join(cmd)}")
    if dry_run:
        print("(dry-run; not launching)")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Force UTF-8 in the child so `rich` spinner/progress chars don't crash
    # Python's default cp1252 encoder when stdout is file-redirected on Windows.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {run_name}\n# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n\n")
        f.flush()
        # File redirect (NOT a pipe) so tqdm/progress lines don't deadlock.
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd=str(config_path.parent), env=env,
        )
    dt = time.time() - t0
    print(f"exit code  : {proc.returncode}   elapsed: {dt/60:.1f} min")
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--round-dir", required=True, type=Path,
                   help="e.g. Z:/StephensU/medaka/sleap_models/retrain_20260508")
    p.add_argument("--round-tag", default="round3",
                   help="Tag injected into each run_name for disambiguation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run without launching.")
    p.add_argument("--continue-on-fail", action="store_true",
                   help="Don't stop the sequence if a training returns non-zero.")
    p.add_argument("--kinds", nargs="+", default=None, choices=KINDS,
                   help="Subset of model kinds to train (default: all).")
    p.add_argument("--views", nargs="+", default=None, choices=VIEWS,
                   help="Subset of views to train (default: all).")
    args = p.parse_args(argv)

    if not args.round_dir.is_dir():
        sys.exit(f"--round-dir not found: {args.round_dir}")

    views = list(args.views) if args.views else VIEWS
    kinds = list(args.kinds) if args.kinds else KINDS
    failures = []
    for view in views:
        labels_slp = args.round_dir / view / "labels.pkg.slp"
        n = _count_labels(labels_slp) if labels_slp.exists() else None
        n_tag = n if n is not None else "X"
        for kind in kinds:
            cfg = args.round_dir / view / "configs" / f"{kind}.yaml"
            if not cfg.exists():
                print(f"[skip] {view}/{kind}: config not found: {cfg}")
                continue
            # Avoid '=' in run_name — Hydra override parser splits on the first
            # '=' and chokes on the second one. Use '_n<N>' instead.
            run_name = f"{args.round_tag}.{kind}_n{n_tag}"
            log = args.round_dir / view / "runs" / run_name / "training_console.log"
            rc = run_one(
                config_path=cfg, run_name=run_name, log_path=log,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((view, kind, rc, log))
                if not args.continue_on_fail:
                    print(f"\nABORT: {view}/{kind} returned {rc}; stopping sequence.")
                    print(f"       see {log} for details.")
                    return rc
    print("\nDone.")
    if failures:
        print("Failed runs:")
        for view, kind, rc, log in failures:
            print(f"  {view}/{kind} rc={rc}: {log}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
