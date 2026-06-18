from __future__ import annotations

"""
Reusable per-round inference launcher for the medaka pipeline.

For a given round (i.e. the per-view best.ckpt pair under
``<round_dir>/<view>/runs/<round_tag>.{centroid,centered_instance}_n<N>/``)
this script supports two inference paths:

* ``--mode track`` (default, recommended): ``sleap-nn track`` -- pure
  torch/cuda, reads the .ckpt directly. Proven to work and apples-to-apples
  with round-3.
* ``--mode predict``: ``sleap-nn export`` -> ``sleap-nn predict``. Uses
  ONNX/TRT runtime; faster in theory but the TRT engine build fails on
  this model architecture (``Could not find any implementation for node
  {ForeignNode[...peaks_castOut]}`` -- needs ~2.4 GB workspace).

Per-view stdout+stderr is redirected to a log file (NOT piped through a
shell -- mirrors run_round3_training.py: tqdm/progress lines don't deadlock
on file redirect, and the log header replicates the round-3 inference log
format so the two are diff-able).

Per-view stdout+stderr is redirected to a log file (NOT piped through a
shell -- mirrors run_round3_training.py: tqdm/progress lines don't deadlock
on file redirect, and the log header replicates the round-3 inference log
format so the two are diff-able).

Usage::

    python -m scripts.sleap.run_round_inference \
        --round-dir Z:/StephensU/medaka/sleap_models/retrain_20260508 \
        --round-tag round4_260520 \
        --experiment 20251225_K_1_2_FCMC \
        --data-root Z:/StephensU/medaka/medaka_202512 \
        --output-root Z:/StephensU/medaka/sleap_output
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

VIEW_TO_BUCKET = {"top_cam": "top", "side1_cam": "side", "side2_cam": "side"}
VIEW_TO_VIDEO_NAME = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


def _resolve_run(round_dir: Path, bucket: str, kind: str, round_tag: str) -> Path:
    """Find the run directory for the given (bucket, kind, round_tag).

    Run dirs are named ``<round_tag>.<kind>_n<N>``; the n<N> suffix varies
    per round, so we glob.
    """
    runs_dir = round_dir / bucket / "runs"
    cands = sorted(runs_dir.glob(f"{round_tag}.{kind}_n*"))
    if not cands:
        raise SystemExit(
            f"no run found at {runs_dir} matching '{round_tag}.{kind}_n*'"
        )
    if len(cands) > 1:
        print(f"[warn] multiple matches for {bucket}/{kind}; using {cands[-1].name}")
    return cands[-1]


def export_view(
    *,
    round_dir: Path,
    bucket: str,
    round_tag: str,
    export_dir: Path,
    precision: str,
    max_instances: int,
    max_batch_size: int,
    crop_size: int,
    fmt: str,
    log_path: Path,
    force: bool,
    dry_run: bool,
) -> int:
    """Export (centroid, centered_instance) pair for ``bucket`` to ``export_dir``."""
    centroid_dir = _resolve_run(round_dir, bucket, "centroid", round_tag)
    instance_dir = _resolve_run(round_dir, bucket, "centered_instance", round_tag)
    for d in (centroid_dir, instance_dir):
        if not (d / "best.ckpt").exists():
            raise SystemExit(f"checkpoint missing: {d / 'best.ckpt'}")

    expected_artefacts = {"onnx": ("model.onnx",), "tensorrt": ("model.trt",),
                          "both": ("model.onnx", "model.trt")}[fmt]
    if not force and all((export_dir / a).exists() for a in expected_artefacts):
        print(f"[{bucket} export] already present -> {export_dir}")
        return 0

    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "sleap_nn.cli", "export",
        str(centroid_dir), str(instance_dir),
        "-o", str(export_dir),
        "-f", fmt,
        "--precision", precision,
        "--max-instances", str(max_instances),
        "--max-batch-size", str(max_batch_size),
        "--crop-size", str(crop_size),
        "--device", "cuda",
    ]
    print(f"\n--- export {bucket} ({fmt}, {precision}) ---")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# export {bucket}\n# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    print(f"  exit: {proc.returncode}   elapsed: {dt/60:.1f} min")
    return int(proc.returncode)


def track_view(
    *,
    view: str,
    round_dir: Path,
    bucket: str,
    round_tag: str,
    video: Path,
    output_slp: Path,
    batch_size: int,
    max_instances: int,
    log_path: Path,
    dry_run: bool,
    frames: str | None = None,
) -> int:
    """Torch-path inference via ``sleap_nn.cli track``."""
    centroid_dir = _resolve_run(round_dir, bucket, "centroid", round_tag)
    instance_dir = _resolve_run(round_dir, bucket, "centered_instance", round_tag)
    output_slp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "sleap_nn.cli", "track",
        "--data_path", str(video),
        "--model_paths", str(centroid_dir),
        "--model_paths", str(instance_dir),
        "--output_path", str(output_slp),
        "--device", "cuda",
        "--batch_size", str(batch_size),
        "--max_instances", str(max_instances),
    ]
    # Optional frame subset (comma list "1,2,3" or range "start-end"); used for
    # quick slice tests so we don't run full-length (~348k frames) inference.
    if frames:
        cmd += ["--frames", frames]
    print(f"\n--- track {view} ---")
    print(f"  video: {video}")
    print(f"  out  : {output_slp}")
    print(f"  log  : {log_path}")
    print(f"  cmd  : {' '.join(cmd)}")
    if dry_run:
        return 0
    if not video.exists():
        raise SystemExit(f"video missing: {video}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {view}\n# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    print(f"  exit: {proc.returncode}   elapsed: {dt/60:.1f} min")
    return int(proc.returncode)


def predict_view(
    *,
    view: str,
    export_dir: Path,
    video: Path,
    output_slp: Path,
    runtime: str,
    batch_size: int,
    max_instances: int,
    log_path: Path,
    dry_run: bool,
) -> int:
    output_slp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "sleap_nn.cli", "predict",
        str(export_dir), str(video),
        "-o", str(output_slp),
        "--runtime", runtime,
        "--batch-size", str(batch_size),
        "--max-instances", str(max_instances),
        "--device", "cuda",
    ]
    print(f"\n--- predict {view} ---")
    print(f"  video: {video}")
    print(f"  out  : {output_slp}")
    print(f"  log  : {log_path}")
    print(f"  cmd  : {' '.join(cmd)}")
    if dry_run:
        return 0
    if not video.exists():
        raise SystemExit(f"video missing: {video}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {view}\n# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    print(f"  exit: {proc.returncode}   elapsed: {dt/60:.1f} min")
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round-dir", required=True, type=Path)
    p.add_argument("--round-tag", required=True)
    p.add_argument("--experiment", required=True,
                   help="Experiment folder name under --data-root.")
    p.add_argument("--data-root", required=True, type=Path,
                   help="Root containing <experiment>/{top_cam,side1_cam,side2_cam}/cam*.mp4")
    p.add_argument("--output-root", required=True, type=Path,
                   help="Root for predicted .slp output (<output-root>/<experiment>/<round-tag>/).")
    p.add_argument("--views", nargs="+", default=list(VIEW_TO_BUCKET.keys()),
                   choices=list(VIEW_TO_BUCKET.keys()))
    p.add_argument("--mode", default="track", choices=["track", "predict"],
                   help="'track' = torch path via sleap-nn track (default, proven). "
                        "'predict' = ONNX/TRT path via export+predict (faster in "
                        "theory, but TRT engine build fails on this architecture).")
    p.add_argument("--format", default="onnx",
                   choices=["onnx", "tensorrt", "both"],
                   help="Export format (used only when --mode predict).")
    p.add_argument("--precision", default="fp16",
                   choices=["fp32", "fp16", "tf32"])
    p.add_argument("--max-instances", type=int, default=6)
    p.add_argument("--max-batch-size", type=int, default=8,
                   help="Used at export time (graph compile constraint).")
    p.add_argument("--predict-batch-size", type=int, default=8)
    p.add_argument("--frames", default=None,
                   help="Optional frame subset for track mode: comma list "
                        "(e.g. 1,2,3) or a range (e.g. 171800-176799). Omit "
                        "for full-length inference.")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--runtime", default="auto",
                   choices=["auto", "onnx", "tensorrt"])
    p.add_argument("--force-export", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-fail", action="store_true")
    args = p.parse_args(argv)

    if not args.round_dir.is_dir():
        sys.exit(f"--round-dir not found: {args.round_dir}")
    exp_dir = args.data_root / args.experiment
    if not exp_dir.is_dir():
        sys.exit(f"experiment dir not found: {exp_dir}")

    output_exp_dir = args.output_root / args.experiment / args.round_tag
    output_exp_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str, int]] = []
    exports: dict[str, Path] = {}

    if args.mode == "predict":
        # Export per bucket (top, side) - one export covers both side1 and side2.
        buckets = sorted({VIEW_TO_BUCKET[v] for v in args.views})
        export_root = args.round_dir / "exports" / args.round_tag
        for bucket in buckets:
            ed = export_root / bucket
            rc = export_view(
                round_dir=args.round_dir, bucket=bucket, round_tag=args.round_tag,
                export_dir=ed, precision=args.precision,
                max_instances=args.max_instances, max_batch_size=args.max_batch_size,
                crop_size=args.crop_size, fmt=args.format,
                log_path=ed / "export.log",
                force=args.force_export, dry_run=args.dry_run,
            )
            exports[bucket] = ed
            if rc != 0:
                failures.append((bucket, "export", rc))
                if not args.continue_on_fail:
                    print(f"\nABORT: export {bucket} failed; see {ed/'export.log'}")
                    return rc

    # Inference per view.
    for view in args.views:
        bucket = VIEW_TO_BUCKET[view]
        video = exp_dir / view / VIEW_TO_VIDEO_NAME[view]
        out_slp = output_exp_dir / f"{view}.slp"
        log = output_exp_dir / f"{view}.inference.log"
        if args.mode == "track":
            rc = track_view(
                view=view, round_dir=args.round_dir, bucket=bucket,
                round_tag=args.round_tag, video=video, output_slp=out_slp,
                batch_size=args.predict_batch_size,
                max_instances=args.max_instances,
                log_path=log, dry_run=args.dry_run,
                frames=args.frames,
            )
        else:
            rc = predict_view(
                view=view, export_dir=exports[bucket], video=video,
                output_slp=out_slp, runtime=args.runtime,
                batch_size=args.predict_batch_size,
                max_instances=args.max_instances,
                log_path=log, dry_run=args.dry_run,
            )
        if rc != 0:
            failures.append((view, args.mode, rc))
            if not args.continue_on_fail:
                print(f"\nABORT: {args.mode} {view} failed; see {log}")
                return rc

    print("\nDone.")
    if failures:
        print("Failures:")
        for view, kind, rc in failures:
            print(f"  {view}/{kind} rc={rc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
