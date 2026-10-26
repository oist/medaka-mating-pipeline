"""2D top-camera tracking + 3D reconstruction for many experiments.

For each experiment (folder names or set tokens such as ``P12``) whose three
``<view>.slp`` files exist under ``<sleap_output_dir>/<exp>/<round_tag>/``:

  1. ``scripts.tracking.track_simple_nn``   -> ``top_cam.tracks.csv``
  2. ``scripts.reconstruction_3d.reconstruct_from_slp`` with the production
     round8 parameters + ``--bind-top-id``  -> ``<tag>_3d_identified.csv``
     under ``<reconstruction_output_dir>/<exp>/<round_tag>/``

HPC runs leave per-view chunks (``<view>.frames_*.slp``) instead of merged
files; once all three views have the same number of chunks and the copy-back
finished (slurm logs present), they are merged with
``scripts.tracking.main --merge-chunks`` first.

Finished experiments are skipped. With ``--watch N`` the script re-checks
every N minutes until every experiment is done or has failed, so 3D runs as
soon as each experiment's inference lands (local queue or HPC).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts.common.experiment_sets import resolve_experiments
from scripts.common.local_paths import REPO_ROOT, load_local_paths
from scripts.sleap.batch_bottomup_track import DEFAULT_ROUND_TAG, VIEWS, default_paths_config

RECONSTRUCT_ARGS = [
    "--calibration-backend", "auto",
    "--num-candidates", "6",
    "--use-visible-node-centroid",
    "--xyz-bounds", "-20", "220", "--no-shift-to-positive",
    "--max-jump-mm", "8", "--max-jump-hard-cap-mm", "80", "--max-gap-frames", "100",
    "--num-individuals", "3",
    "--bind-top-id",
]


def hpc_chunks_ready(slp_dir: Path) -> bool:
    counts = {v: len(list(slp_dir.glob(f"{v}.frames_*.slp"))) for v in VIEWS}
    logs = slp_dir / "logs"
    return min(counts.values()) > 0 and len(set(counts.values())) == 1 and logs.is_dir() and any(logs.iterdir())


def run(cmd: list[str], log_path: Path, dry_run: bool) -> bool:
    if dry_run:
        print("  DRY:", " ".join(cmd))
        return True
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n# cmd: {' '.join(cmd)}\n# started: {time.ctime()}\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
        log.write(f"# exit {rc} at {time.ctime()}\n")
    return rc == 0


def process(exp: str, paths, cfg: Path, tag: str, dry_run: bool) -> str:
    slp_dir = paths.sleap_output_dir / exp / tag
    out_dir = paths.reconstruction_output_dir / exp / tag
    final_csv = out_dir / f"{tag}_3d_identified.csv"
    if final_csv.exists():
        return "done"

    py = [sys.executable, "-m"]
    log_path = out_dir / "batch_3d.log"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if not all((slp_dir / f"{v}.slp").exists() for v in VIEWS):
        if not hpc_chunks_ready(slp_dir):
            return "waiting for inference"
        merge = py + ["scripts.tracking.main", exp, "--merge-chunks", "--sleap-subdir", tag, "--paths-config", str(cfg)]
        if not run(merge, log_path, dry_run):
            return f"FAILED at merge (see {log_path})"

    tracks_csv = slp_dir / "top_cam.tracks.csv"
    if not tracks_csv.exists():
        track = py + [
            "scripts.tracking.track_simple_nn", exp, "--paths-config", str(cfg), "--view", "top_cam",
            "--slp", str(slp_dir / "top_cam.slp"), "--output", str(tracks_csv),
            "--max-distance", "100", "--max-tracks", "3", "--max-gap", "500",
        ]
        if not run(track, log_path, dry_run):
            return f"FAILED at track_simple_nn (see {log_path})"

    recon = py + [
        "scripts.reconstruction_3d.reconstruct_from_slp", exp, "--paths-config", str(cfg),
        "--sleap-subdir", tag, *RECONSTRUCT_ARGS,
        "--top-tracks-csv", str(tracks_csv), "--output-dir", str(out_dir), "--output-tag", tag,
    ]
    if not run(recon, log_path, dry_run):
        return f"FAILED at reconstruct_from_slp (see {log_path})"
    return "dry-run" if dry_run else "done (new)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+", help="Folder names and/or set tokens (P12, K1, ALL).")
    p.add_argument("--paths-config", type=Path, default=None,
                   help="Default: configs/paths.<USERNAME>.toml, else configs/paths.local.toml.")
    p.add_argument("--round-tag", default=DEFAULT_ROUND_TAG)
    p.add_argument("--watch", type=float, default=0, metavar="MIN",
                   help="Re-check every MIN minutes until all experiments are done or failed.")
    p.add_argument("--include-part2", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = args.paths_config or default_paths_config()
    paths = load_local_paths(cfg)
    if paths is None:
        raise SystemExit(f"paths config not found: {cfg}")
    experiments = resolve_experiments(args.experiments, paths.medaka_data_dir, args.include_part2)
    print(f"paths config : {cfg}")
    print(f"experiments  : {len(experiments)}   round tag: {args.round_tag}", flush=True)

    status: dict[str, str] = {}
    while True:
        for exp in experiments:
            if status.get(exp, "").startswith(("done", "FAILED")):
                continue
            status[exp] = process(exp, paths, cfg, args.round_tag, args.dry_run)
            print(f"[{time.strftime('%m-%d %H:%M')}] {exp}: {status[exp]}", flush=True)
        pending = [e for e in experiments if status[e] == "waiting for inference"]
        if not args.watch or args.dry_run or not pending:
            break
        print(f"[{time.strftime('%m-%d %H:%M')}] {len(pending)} waiting; next check in {args.watch:g} min", flush=True)
        time.sleep(args.watch * 60)

    failed = [e for e in experiments if status[e].startswith("FAILED")]
    print(f"summary: {sum(s.startswith('done') for s in status.values())} done, "
          f"{len(failed)} failed, {sum(s == 'waiting for inference' for s in status.values())} waiting", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
