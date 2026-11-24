"""Keep review packages up to date in the background (independent of the 3D watcher).

Each pass:

1. builds ``review_data.npz`` for every experiment whose 3D reconstruction is
   finished (``batch_3d.done``) and newer than its review data - minutes each;
2. transcodes the proxies of the next experiment that has none - ~40 min each.

Review data never waits behind the long proxy queue. The process and its
ffmpeg children run at idle priority, and no proxy view is started while a
local SLEAP inference (``sleap_nn.cli track``) is running: transcoding shares
the GPU and measurably slowed inference (32.5 -> ~29 frames/s). Use
``--when-inference run`` to transcode anyway. With ``--watch N`` it re-checks every N minutes until every
experiment has proxies and review data; failures are logged and not retried
within the same run (restart after fixing).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.review.proxies import MANIFEST_NAME, build_experiment
from scripts.review.review_data import DATA_VERSION, build_review_data, calibration_folder

DONE_3D = "batch_3d.done"


def lower_own_priority() -> None:
    try:
        import psutil

        proc = psutil.Process()
        proc.nice(psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else 19)
    except Exception:  # psutil missing or not permitted: run at normal priority
        pass


def local_inference_running() -> bool:
    """True while a ``sleap_nn.cli track`` (local SLEAP inference) process is alive."""
    try:
        import psutil
    except ImportError:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if ("sleap_nn.cli" in cmd or "sleap-nn" in cmd) and " track" in cmd:
            return True
    return False


def wait_for_free_gpu(log, poll_minutes: float = 5.0) -> None:
    if local_inference_running():
        log(f"local SLEAP inference running; proxies paused (re-check every {poll_minutes:g} min)")
        while local_inference_running():
            time.sleep(poll_minutes * 60)
        log("local SLEAP inference finished; resuming proxies")


def review_data_pending(exp: str, paths, review_root: Path, tag: str) -> bool:
    """3D is done and the review data is missing, older than the 3D, or from an older ``DATA_VERSION``."""
    done = paths.reconstruction_output_dir / exp / tag / DONE_3D
    manifest = review_root / exp / tag / "review_manifest.json"
    if not done.exists():
        return False
    if not manifest.exists() or manifest.stat().st_mtime < done.stat().st_mtime:
        return True
    try:
        version = int(json.loads(manifest.read_text(encoding="utf-8")).get("data_version", 0))
    except (OSError, ValueError, TypeError):
        return True
    return version < DATA_VERSION


def source_fps(paths, exp: str) -> float:
    meta = paths.medaka_data_dir / exp / "experiment_metadata.json"
    try:
        return float(json.loads(meta.read_text(encoding="utf-8"))["views"]["top_cam"]["frame_rate"])
    except (OSError, KeyError, ValueError, TypeError):
        return 166.0


def main(argv: list[str] | None = None) -> int:
    from scripts.common.experiment_sets import resolve_experiments
    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import DEFAULT_ROUND_TAG, default_paths_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="Folder names and/or set tokens (P12, K1, ALL); order = priority.")
    p.add_argument("--round-tag", default=DEFAULT_ROUND_TAG)
    p.add_argument("--paths-config", type=Path, default=None)
    p.add_argument("--review-root", type=Path, default=None, help="Default: <data_root>/review_packages.")
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--no-proxies", action="store_true", help="Only build review data.")
    p.add_argument("--when-inference", choices=("pause", "run"), default="pause",
                   help="pause (default): start no proxy view while local SLEAP inference runs.")
    p.add_argument("--watch", type=float, default=0, metavar="MIN")
    p.add_argument("--include-part2", action="store_true")
    args = p.parse_args(argv)

    lower_own_priority()
    cfg = args.paths_config or default_paths_config()
    paths = load_local_paths(cfg)
    if paths is None:
        raise SystemExit(f"paths config not found: {cfg}")
    review_root = args.review_root or paths.data_root / "review_packages"
    exps = resolve_experiments(args.experiments, paths.medaka_data_dir, args.include_part2)
    print(f"review root  : {review_root}\nexperiments  : {len(exps)}   round tag: {args.round_tag}", flush=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%m-%d %H:%M')}] {msg}", flush=True)

    failed: dict[str, str] = {}
    while True:
        for exp in exps:
            key = f"{exp}/data"
            if key in failed or not review_data_pending(exp, paths, review_root, args.round_tag):
                continue
            recon_dir = paths.reconstruction_output_dir / exp / args.round_tag
            try:
                build_review_data(exp, args.round_tag, paths.sleap_output_dir / exp / args.round_tag, recon_dir,
                                  review_root / exp / args.round_tag,
                                  calibration_folder(recon_dir, args.round_tag, paths.calibration_model_dir),
                                  fps=source_fps(paths, exp), log=log)
                log(f"{exp}: review data done")
            except Exception as exc:
                failed[key] = repr(exc)
                log(f"{exp}: review data FAILED: {exc!r}")

        todo = [] if args.no_proxies else [
            e for e in exps if f"{e}/proxy" not in failed and not (review_root / e / "proxy" / MANIFEST_NAME).exists()]
        if todo and args.when_inference == "pause" and local_inference_running():
            log(f"local SLEAP inference running; proxies paused ({len(todo)} experiments without proxies)")
            time.sleep(min(args.watch, 5.0) * 60 if args.watch else 300)
            continue
        if todo:
            exp = todo[0]
            log(f"{exp}: proxies ({len(todo)} experiments without proxies)")
            gate = (lambda: wait_for_free_gpu(log)) if args.when_inference == "pause" else None
            try:
                status = build_experiment(exp, paths.medaka_data_dir, review_root, size=args.size, log=log,
                                          before_view=gate)
            except Exception as exc:
                status = f"FAILED: {exc!r}"
            if status.startswith("FAILED"):
                failed[f"{exp}/proxy"] = status
            log(f"{exp}: proxies {status}")
            continue  # check for new 3D results before the next long transcode

        waiting = [e for e in exps if f"{e}/data" not in failed
                   and not (review_root / e / args.round_tag / "review_manifest.json").exists()]
        if not args.watch or not waiting:
            break
        log(f"{len(waiting)} experiments waiting for 3D; next check in {args.watch:g} min")
        time.sleep(args.watch * 60)

    for key, why in failed.items():
        log(f"FAILED {key}: {why}")
    log(f"summary: {len(failed)} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
