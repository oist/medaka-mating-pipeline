"""Run bottom-up SLEAP inference for many experiments, one view at a time.

For each experiment (folder names or set tokens such as ``P12``, see
``scripts.common.experiment_sets``) and each view, runs ``sleap-nn track``
with a single bottom-up model and writes

    <sleap_output_dir>/<experiment>/<round_tag>/<view>.slp

Already finished views are skipped, so the queue can be re-run after a crash.
A view is also skipped when HPC chunks (``<view>.frames_*.slp``) are present
or another queue holds its ``<view>.running`` lock. Output is written to
``<view>.partial.slp`` and renamed only after sleap-nn exits cleanly.

Example (30-second test of one experiment, all three views):

    python -m scripts.sleap.batch_bottomup_track 20260112_P_1_1_FCMC \
        --round-tag s14_bottomup_3node_reviewed_test --frames 171800-176779

Batch size 4 is deliberate: on a 16 GB RTX 5070 Ti, batch >= 4 is equally
fast (~33 FPS, decode/PAF-grouping bound) and batch 12 overflows VRAM
(1.35 FPS).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.common.experiment_sets import resolve_experiments
from scripts.common.local_paths import REPO_ROOT, load_local_paths

VIEWS = ("top_cam", "side1_cam", "side2_cam")
DEFAULT_ROUND_TAG = "s14_bottomup_3node_reviewed"
DEFAULT_TOP_MODEL = "retrain_20260810_3node_reviewed/top/s14.bottomup_3node.reviewed_top_f24_ms64_paf8"
DEFAULT_SIDE_MODEL = "retrain_20260810_3node_reviewed/side/s14.bottomup_3node.reviewed_side_f24_ms64_paf8"


def default_paths_config() -> Path:
    user_cfg = REPO_ROOT / "configs" / f"paths.{getpass.getuser()}.toml"
    return user_cfg if user_cfg.exists() else REPO_ROOT / "configs" / "paths.local.toml"


def resolve_model(value: str, models_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = models_root / path
    for name in ("training_config.yaml", "best.ckpt"):
        if not (path / name).exists():
            raise FileNotFoundError(f"model file missing: {path / name}")
    return path


def lock_is_live(lock: Path) -> bool:
    if not lock.exists():
        return False
    try:
        import psutil

        return psutil.pid_exists(json.loads(lock.read_text())["pid"])
    except Exception:
        return True  # unreadable lock: be conservative, do not double-run


def run_view(experiment: str, view: str, video: Path, model: Path, out_dir: Path, args) -> str:
    out_slp = out_dir / f"{view}.slp"
    if out_slp.exists():
        return "skip (done)"
    if any(out_dir.glob(f"{view}.frames_*.slp")):
        return "skip (HPC chunks present)"
    lock = out_dir / f"{view}.running"
    if lock_is_live(lock):
        return "skip (running elsewhere)"

    partial = out_dir / f"{view}.partial.slp"
    cmd = [
        sys.executable, "-m", "sleap_nn.cli", "track",
        "--data_path", str(video),
        "--model_paths", str(model),
        "--output_path", str(partial),
        "--device", "cuda",
        "--batch_size", str(args.batch_size),
        "--max_instances", str(args.max_instances),
    ]
    if args.frames:
        cmd += ["--frames", args.frames]
    if args.dry_run:
        print("  DRY:", " ".join(cmd))
        return "dry-run"

    out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    started = time.time()
    try:
        with open(out_dir / f"{view}.inference.log", "w", encoding="utf-8") as log:
            log.write(f"# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
    finally:
        lock.unlink(missing_ok=True)
    hours = (time.time() - started) / 3600
    if rc != 0 or not partial.exists():
        return f"FAILED (exit {rc}, {hours:.2f} h; see {view}.inference.log)"
    partial.replace(out_slp)
    return f"done ({hours:.2f} h)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="Folder names and/or set tokens (P12, K1, ALL).")
    p.add_argument("--paths-config", type=Path, default=None,
                   help="Default: configs/paths.<USERNAME>.toml, else configs/paths.local.toml.")
    p.add_argument("--round-tag", default=DEFAULT_ROUND_TAG)
    p.add_argument("--top-model", default=DEFAULT_TOP_MODEL, help="Relative to sleap_models_dir, or absolute.")
    p.add_argument("--side-model", default=DEFAULT_SIDE_MODEL, help="Relative to sleap_models_dir, or absolute.")
    p.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-instances", type=int, default=6)
    p.add_argument("--frames", default=None, help="Optional frame range for tests, e.g. 171800-176779.")
    p.add_argument("--include-part2", action="store_true", help="Also expand sets to *_FCMC_P2 folders.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = args.paths_config or default_paths_config()
    paths = load_local_paths(cfg)
    if paths is None:
        raise SystemExit(f"paths config not found: {cfg}")
    experiments = resolve_experiments(args.experiments, paths.medaka_data_dir, args.include_part2)
    models = {
        "top_cam": resolve_model(args.top_model, paths.sleap_models_dir),
        "side": resolve_model(args.side_model, paths.sleap_models_dir),
    }

    print(f"paths config : {cfg}")
    print(f"experiments  : {len(experiments)} -> {', '.join(experiments)}")
    print(f"views        : {', '.join(args.views)}   round tag: {args.round_tag}", flush=True)

    failures = 0
    for i, exp in enumerate(experiments, 1):
        out_dir = paths.sleap_output_dir / exp / args.round_tag
        for view in args.views:
            videos = sorted((paths.medaka_data_dir / exp / view).glob("cam*.mp4"))
            if len(videos) != 1:
                status = f"FAILED (expected 1 cam*.mp4, found {len(videos)})"
            else:
                model = models["top_cam" if view == "top_cam" else "side"]
                print(f"[{time.strftime('%m-%d %H:%M')}] ({i}/{len(experiments)}) {exp} {view} ...", flush=True)
                status = run_view(exp, view, videos[0], model, out_dir, args)
            failures += status.startswith("FAILED")
            print(f"[{time.strftime('%m-%d %H:%M')}] ({i}/{len(experiments)}) {exp} {view}: {status}", flush=True)

    print(f"finished; {failures} failed view(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
