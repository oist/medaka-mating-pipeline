"""Run medaka-3d's full chain on several experiments and build a review package for each.

One experiment goes through the steps of ``medaka3d_chain`` in order: detection cache, c27 -> c34, export
at lambda 0.767, ``build_review_data`` into ``<review_packages>/<exp>/<tag>_m3dfull``, then ``trajectory_qc``
(anomaly intervals from the trajectories and c30's joins / refusals) into the same folder. Every step is skipped
when its result exists, so an interrupted batch is resumed by starting it again. A failing experiment
does not stop the others.

Detections come from ``<sleap_output>/<exp>/<round tag>/<view>.slp``. The K_1 experiments have no such
files: medaka-3d's cache was built from its own uncapped s16 chunks, and the chain's detector rows refer
to those. ``merge_s16`` concatenates the same chunks into ``<sleap_output>/<exp>/<S16_TAG>/`` and checks the
result against the cache, so export and review data read exactly the detections the chain used.

CLI::

    python -m scripts.reconstruction_3d.medaka3d_batch 20260109_K_2_1_FCMC 20251225_K_1_1_FCMC --jobs 3

``--watch MIN`` re-checks every MIN minutes and starts an experiment only once its three
merged ``<view>.slp`` files exist (or it is a K_1 experiment with s16 chunks). Finished
experiments are skipped; a failure is not retried until the process is started again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.reconstruction_3d import medaka3d_chain as M3C

REPO_ROOT = Path(__file__).resolve().parents[2]
S14_TAG = "s14_bottomup_3node_reviewed"
#: Merged s16 chunks: same model as s14 (2026-08-10 reviewed labels), inferred without an instance cap.
S16_TAG = "s16_bottomup_3node_uncapped"
LAMBDA = 0.767


@dataclass(frozen=True)
class Source:
    """Where one experiment's detections are read from."""

    tag: str
    slp_dir: Path
    s16: bool


def choose_source(experiment: str, sleap_output_dir: Path, round_tag: str, s16_experiments) -> Source:
    """Merged per-view ``.slp`` of ``round_tag`` if complete, else the s16 chunks of a K_1 experiment."""
    slp_dir = Path(sleap_output_dir) / experiment / round_tag
    if all((slp_dir / f"{view}.slp").is_file() for view in M3C.VIEWS):
        return Source(round_tag, slp_dir, False)
    if experiment in s16_experiments:
        return Source(S16_TAG, Path(sleap_output_dir) / experiment / S16_TAG, True)
    raise FileNotFoundError(f"{experiment}: no complete {slp_dir} and no s16 chunks")


def same_detections(a, b) -> bool:
    """Same frames, in the same order, with the same points (NaN = NaN)."""
    return np.array_equal(a.frame_idx, b.frame_idx) and np.array_equal(a.points, b.points, equal_nan=True)


def merge_s16(experiment: str, slp_dir: Path, m3d: Path, log=print) -> None:
    """Write ``<slp_dir>/<view>.slp`` from the s16 chunks; a result that differs from c01's cache is deleted."""
    from scripts.tracking.slp_merge import merge_view_chunks

    paths = M3C.m3d_paths(m3d)
    import c01_s16_predictions as c01

    chunks = c01.chunk_files(paths.work_dir.parent / c01.RUNS[M3C.RUN])
    for view in M3C.VIEWS:
        out = slp_dir / f"{view}.slp"
        if out.is_file():
            continue
        started = time.time()
        frames = merge_view_chunks(chunks[(experiment, view)], out)
        merged = c01.read_pair([out], experiment, view, c01.N_FRAMES)
        if not same_detections(merged, c01.load(paths.work_dir, experiment, view, M3C.RUN)):
            out.unlink()
            raise ValueError(f"{experiment} {view}: merged chunks differ from the c01 cache")
        log(f"[batch] {experiment} {view}: merged {frames} frames, equal to the cache "
            f"({time.time() - started:.0f} s)")


def run_one(experiment: str, *, round_tag: str = S14_TAG, lam: float = LAMBDA, m3d=None, log=print) -> None:
    """All steps for one experiment; each is skipped when its result exists."""
    from scripts.common.local_paths import load_local_paths
    from scripts.review.build_packages import source_fps
    from scripts.review.review_data import build_review_data, calibration_folder
    from scripts.sleap.batch_bottomup_track import default_paths_config

    paths = load_local_paths(default_paths_config())
    m3d = M3C.medaka3d_repo(m3d)
    m3p = M3C.m3d_paths(m3d)
    import c01_s16_predictions as c01

    src = choose_source(experiment, paths.sleap_output_dir, round_tag, c01.EXPERIMENTS)
    log(f"[batch] {experiment}: detections {src.slp_dir}")
    if src.s16:
        merge_s16(experiment, src.slp_dir, m3d, log=log)
    else:
        M3C.build_detection_cache(experiment, src.slp_dir, m3d=m3d, log=log)
    M3C.run_chain(experiment, m3d=m3d, log_dir=m3p.work_dir / "medaka3d_chain_logs" / experiment, log=log)

    tag = f"{src.tag}_m3dfull"
    out_dir = paths.reconstruction_output_dir / experiment / tag
    review_dir = paths.data_root / "review_packages" / experiment / tag
    if (out_dir / f"{tag}_3d.csv").is_file():
        log(f"[batch] {experiment}: export exists")
    else:
        M3C.export_review_inputs(experiment, src.slp_dir, out_dir, tag, lam=lam, m3d=m3d, log=log)
    if (review_dir / "review_manifest.json").is_file():
        log(f"[batch] {experiment}: review package exists")
    else:
        build_review_data(experiment, tag, src.slp_dir, out_dir, review_dir,
                          calibration_folder(out_dir, tag, paths.calibration_model_dir),
                          fps=source_fps(paths, experiment))
    if (review_dir / "trajectory_qc_summary.json").is_file():
        log(f"[batch] {experiment}: trajectory QC exists")
    else:
        from scripts.reconstruction_3d.trajectory_qc import run_qc

        run_qc(review_dir, c30_summary=c30_summary_path(m3p.work_dir, experiment), lam=lam, log=log)
    log(f"[batch] {experiment}: done -> {review_dir}")


def c30_summary_path(work_dir: Path, experiment: str) -> Path:
    """medaka-3d's ``c30_summary.json`` (slot joins and refused boundaries) of the production chain."""
    return Path(work_dir) / "c30_slot_stitch" / f"{experiment}__{M3C.C30_TAG}" / "c30_summary.json"


def chain_readiness(experiment: str, *, sleap_output_dir: Path, reconstruction_output_dir: Path,
                    data_root: Path, round_tag: str, s16_experiments) -> str:
    """``done`` when the m3dfull CSV, review manifest and trajectory QC exist, ``ready`` when detections
    do, else ``waiting``."""
    slp_dir = Path(sleap_output_dir) / experiment / round_tag
    if all((slp_dir / f"{view}.slp").is_file() for view in M3C.VIEWS):
        tag = round_tag
        ready = True
    elif experiment in s16_experiments:
        tag = S16_TAG
        ready = True
    else:
        return "waiting"
    out_tag = f"{tag}_m3dfull"
    csv_ok = (Path(reconstruction_output_dir) / experiment / out_tag / f"{out_tag}_3d.csv").is_file()
    review_dir = Path(data_root) / "review_packages" / experiment / out_tag
    review_ok = all((review_dir / name).is_file() for name in ("review_manifest.json", "trajectory_qc_summary.json"))
    if ready and csv_ok and review_ok:
        return "done"
    return "ready"


def status_row(experiment: str, *, round_tag: str = S14_TAG, m3d=None) -> dict:
    """One line of progress: chain steps, export, review package."""
    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import default_paths_config

    paths = load_local_paths(default_paths_config())
    m3d = M3C.medaka3d_repo(m3d)
    M3C.m3d_paths(m3d)
    import c01_s16_predictions as c01

    src = choose_source(experiment, paths.sleap_output_dir, round_tag, c01.EXPERIMENTS)
    row = {"experiment": experiment, "detections": src.tag}
    row.update({tool: M3C.step_status(tool, out) for tool, _, out in M3C.chain_steps(experiment, m3d)})
    tag = f"{src.tag}_m3dfull"
    row["export"] = (paths.reconstruction_output_dir / experiment / tag / f"{tag}_3d.csv").is_file()
    row["review"] = (paths.data_root / "review_packages" / experiment / tag / "review_manifest.json").is_file()
    return row


def run_batch(experiments: list[str], *, jobs: int, round_tag: str, log_dir: Path) -> dict[str, int]:
    """One process per experiment, at most ``jobs`` at a time; returns the exit code of each."""
    def one(experiment: str) -> int:
        folder = log_dir / experiment
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "scripts.reconstruction_3d.medaka3d_batch", experiment, "--one",
               "--round-tag", round_tag]
        started = time.time()
        print(f"[batch] start {experiment}", flush=True)
        with open(folder / "batch.log", "a", encoding="utf-8") as handle:
            handle.write(f"\n# started {time.ctime()}\n")
            handle.flush()
            code = subprocess.call(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
        print(f"[batch] {experiment}: exit {code}, {(time.time() - started) / 60:.1f} min", flush=True)
        return code

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return dict(zip(experiments, pool.map(one, experiments)))


def watch_batch(experiments: list[str], *, jobs: int, round_tag: str, log_dir: Path, watch_min: float,
                paths, s16_experiments) -> int:
    """Start each experiment when its detections exist; re-check until none are still waiting."""
    status: dict[str, str] = {}
    while True:
        ready: list[str] = []
        for experiment in experiments:
            if status.get(experiment) in ("done", "failed"):
                continue
            state = chain_readiness(
                experiment, sleap_output_dir=paths.sleap_output_dir,
                reconstruction_output_dir=paths.reconstruction_output_dir, data_root=paths.data_root,
                round_tag=round_tag, s16_experiments=s16_experiments)
            if state == "ready":
                ready.append(experiment)
            else:
                status[experiment] = state
                print(f"[batch] {experiment}: {state}", flush=True)
        if ready:
            codes = run_batch(ready, jobs=jobs, round_tag=round_tag, log_dir=log_dir)
            for experiment, code in codes.items():
                status[experiment] = "done" if code == 0 else "failed"
        waiting = [experiment for experiment in experiments if status.get(experiment) == "waiting"]
        if watch_min <= 0 or not waiting:
            break
        print(f"[{time.strftime('%m-%d %H:%M')}] {len(waiting)} waiting; next check in {watch_min:g} min", flush=True)
        time.sleep(watch_min * 60)
    failed = [experiment for experiment in experiments if status.get(experiment) == "failed"]
    print(f"summary: {sum(status.get(e) == 'done' for e in experiments)} done, "
          f"{len(failed)} failed, {sum(status.get(e) == 'waiting' for e in experiments)} waiting", flush=True)
    return int(bool(failed))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--round-tag", default=S14_TAG)
    p.add_argument("--watch", type=float, default=0, metavar="MIN",
                   help="Re-check every MIN minutes; start an experiment only once its detections exist.")
    p.add_argument("--one", action="store_true", help="run the first experiment in this process")
    args = p.parse_args(argv)
    if args.one:
        run_one(args.experiments[0], round_tag=args.round_tag)
        return 0
    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import default_paths_config

    paths = load_local_paths(default_paths_config())
    if paths is None:
        raise SystemExit(f"paths config not found: {default_paths_config()}")
    log_dir = M3C.m3d_paths(M3C.medaka3d_repo()).work_dir / "medaka3d_chain_logs"
    if args.watch:
        import c01_s16_predictions as c01

        return watch_batch(args.experiments, jobs=args.jobs, round_tag=args.round_tag, log_dir=log_dir,
                           watch_min=args.watch, paths=paths, s16_experiments=c01.EXPERIMENTS)
    codes = run_batch(args.experiments, jobs=args.jobs, round_tag=args.round_tag, log_dir=log_dir)
    print(json.dumps(codes, indent=1), flush=True)
    return int(any(codes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
