from __future__ import annotations

"""Evaluate the controlled Phase-4 OHKM pair on the shared side benchmark.

This runner keeps the completed Phase-1/3 evaluation bundle immutable. It
compares two centroid models (sigma 1.5 and 2.0) with three centered-instance
models (the evaluated control, controlled OHKM-off, and controlled OHKM-on):

1. two centroid-only candidates;
2. three GT-mouth oracle centered-instance candidates;
3. the complete 2 x 3 full top-down pairing grid.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths
from scripts.sleap.run_topdown_optimization_evaluation import (
    DEFAULT_BENCHMARK,
    EvaluationJob,
    aggregate_results,
    artifact_record,
    build_command,
    sha256_file,
    summarize_benchmark,
    write_artifact_inventory,
    write_status,
)


DEFAULT_OUTPUT_DIR = Path(
    "report/20260725/sleap_side_optimization_phase4_ohkm_evaluation"
)


def resolve_phase4_candidate_runs(
    reference_root: Path,
    phase4_root: Path,
) -> dict[str, Path]:
    """Resolve the frozen reference models and controlled Phase-4 pair."""
    reference_runs = reference_root / "side" / "runs"
    phase4_runs = phase4_root / "side" / "runs"
    resolved = {
        "centroid_sigma15": (
            reference_runs / "side_centroid_sigma15_strictframe_p40"
        ),
        "centroid_sigma20": (
            reference_runs / "side_centroid_sigma20_strictframe_p40"
        ),
        "instance_control": (
            reference_runs / "side_centered_instance_control_p40"
        ),
        "instance_ohkmoff": (
            phase4_runs
            / "side_centered_instance_crop448_rot90_jitter0_"
            "strictmouth_ohkmoff_p40"
        ),
        "instance_ohkmon": (
            phase4_runs
            / "side_centered_instance_crop448_rot90_jitter0_"
            "strictmouth_ohkmon_p40"
        ),
    }
    missing = [path for path in resolved.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Phase-4 candidate run directories missing: {missing}"
        )
    return resolved


def build_phase4_jobs(
    candidate_runs: dict[str, Path],
) -> list[EvaluationJob]:
    """Build the two component jobs and complete 2 x 3 pairing grid."""
    centroids = {
        name: (candidate_runs[name],)
        for name in ("centroid_sigma15", "centroid_sigma20")
    }
    instances = {
        name: (candidate_runs[name],)
        for name in (
            "instance_control",
            "instance_ohkmoff",
            "instance_ohkmon",
        )
    }
    full_grid: dict[str, tuple[Path, ...]] = {}
    for centroid_name, centroid_paths in centroids.items():
        sigma = centroid_name.removeprefix("centroid_")
        for instance_name, instance_paths in instances.items():
            instance = instance_name.removeprefix("instance_")
            full_grid[f"{sigma}__{instance}"] = (
                centroid_paths[0],
                instance_paths[0],
            )
    return [
        EvaluationJob(
            name="phase4_centroid_only",
            mode="centroid_only",
            models=centroids,
            strict_anchor_eligible_only=True,
        ),
        EvaluationJob(
            name="phase4_oracle_centered_instance",
            mode="oracle_centered_instance",
            models=instances,
            strict_anchor_eligible_only=True,
        ),
        EvaluationJob(
            name="phase4_full_topdown_grid",
            mode="full_topdown",
            models=full_grid,
            strict_anchor_eligible_only=False,
        ),
    ]


def _model_artifacts(
    candidate_runs: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, run_dir in candidate_runs.items():
        checkpoint = run_dir / "best.ckpt"
        config = run_dir / "training_config.yaml"
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(
                f"Candidate is not inference-ready: {run_dir}"
            )
        artifacts[name] = {
            "run_dir": str(run_dir.resolve()),
            "best_checkpoint": str(checkpoint.resolve()),
            "best_checkpoint_sha256": sha256_file(checkpoint),
            "training_config": str(config.resolve()),
            "training_config_sha256": sha256_file(config),
        }
    return artifacts


def wait_for_training_queue(
    phase4_root: Path,
    *,
    poll_seconds: float,
    max_wait_hours: float,
) -> dict[str, Any]:
    """Wait for the restartable training queue before claiming the GPU."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if max_wait_hours <= 0:
        raise ValueError("max_wait_hours must be positive")
    status_path = phase4_root / "queue_status.json"
    deadline = time.monotonic() + max_wait_hours * 60.0 * 60.0
    last_state: str | None = None
    while True:
        payload: dict[str, Any] | None = None
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None
        if payload is not None:
            state = str(payload.get("state", "unknown"))
            if state != last_state:
                print(
                    f"Phase-4 training queue state: {state}; "
                    f"active={payload.get('active')}",
                    flush=True,
                )
                last_state = state
            if state == "complete":
                return payload
            if state in {"failed", "complete_with_failures"}:
                raise RuntimeError(
                    f"Phase-4 training queue did not complete cleanly: {payload}"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for training queue: {status_path}"
            )
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG
    )
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--phase4-root", type=Path, default=None)
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_BENCHMARK
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=(
            "phase4_centroid_only",
            "phase4_oracle_centered_instance",
            "phase4_full_topdown_grid",
        ),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument(
        "--wait-for-training",
        action="store_true",
        help=(
            "Poll <phase4-root>/queue_status.json and start evaluation only "
            "after both controlled training candidates finish cleanly."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    args = parser.parse_args(argv)

    local_paths = load_local_paths(args.paths_config)
    model_root = (
        local_paths.sleap_models_dir
        if local_paths is not None
        else None
    )
    if args.reference_root is None:
        if model_root is None:
            parser.error(
                "paths config must define paths.sleap_models_dir or pass "
                "--reference-root"
            )
        reference_root = model_root / "retrain_20260724_sideopt"
    else:
        reference_root = args.reference_root
    if args.phase4_root is None:
        if model_root is None:
            parser.error(
                "paths config must define paths.sleap_models_dir or pass "
                "--phase4-root"
            )
        phase4_root = (
            model_root / "retrain_20260725_sideopt_phase4_ohkm_pair"
        )
    else:
        phase4_root = args.phase4_root

    if args.wait_for_training:
        wait_for_training_queue(
            phase4_root,
            poll_seconds=args.poll_seconds,
            max_wait_hours=args.max_wait_hours,
        )

    benchmark = args.benchmark
    if not benchmark.is_file():
        parser.error(f"benchmark not found: {benchmark}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    candidate_runs = resolve_phase4_candidate_runs(
        reference_root, phase4_root
    )
    jobs = build_phase4_jobs(candidate_runs)
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [job for job in jobs if job.name in wanted]
    model_artifacts = _model_artifacts(candidate_runs)

    repo_root = Path(__file__).resolve().parents[2]
    evaluator_sources = [
        Path(__file__).resolve(),
        repo_root
        / "scripts"
        / "sleap"
        / "run_topdown_optimization_evaluation.py",
        repo_root / "scripts" / "sleap" / "compare_rounds_topdown.py",
        repo_root / "scripts" / "sleap" / "eval_topdown_vs_bottomup.py",
    ]
    manifest = {
        "workflow": "side_topdown_phase4_controlled_ohkm_evaluation",
        "scientific_status": "diagnostic_not_locked_generalization_test",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_summary": summarize_benchmark(benchmark),
        "reference_root": str(reference_root.resolve()),
        "phase4_root": str(phase4_root.resolve()),
        "evaluator_sources": [
            artifact_record(path, relative_to=repo_root)
            for path in evaluator_sources
        ],
        "model_artifacts": model_artifacts,
        "jobs": [
            {
                "name": job.name,
                "mode": job.mode,
                "strict_anchor_eligible_only": (
                    job.strict_anchor_eligible_only
                ),
                "models": {
                    name: [str(path.resolve()) for path in paths]
                    for name, paths in job.models.items()
                },
                "command": build_command(
                    job=job,
                    benchmark=benchmark,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    device=args.device,
                ),
            }
            for job in jobs
        ],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(
            f"Prepared {len(jobs)} Phase-4 evaluation jobs under "
            f"{output_dir}",
            flush=True,
        )
        return 0

    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    write_status(
        output_dir,
        state="running",
        active=None,
        completed=completed,
        failed=failed,
    )
    for job in jobs:
        result_path = (
            output_dir / f"side_{job.name}_shared_benchmark.json"
        )
        if result_path.is_file() and not args.force:
            print(f"[skip complete] {job.name}: {result_path}", flush=True)
            completed.append(job.name)
            continue
        write_status(
            output_dir,
            state="running",
            active=job.name,
            completed=completed,
            failed=failed,
        )
        command = build_command(
            job=job,
            benchmark=benchmark,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"\n--- {job.name} ---", flush=True)
        print(" ".join(command), flush=True)
        log_path = output_dir / f"{job.name}.console.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            result = subprocess.run(
                command,
                cwd=str(repo_root),
                env={
                    **os.environ,
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and result_path.is_file():
            completed.append(job.name)
        else:
            failed.append(
                {"job": job.name, "return_code": int(result.returncode)}
            )
            if not args.continue_on_fail:
                write_status(
                    output_dir,
                    state="failed",
                    active=None,
                    completed=completed,
                    failed=failed,
                )
                return int(result.returncode or 2)

    aggregate_results(output_dir, jobs)
    final_state = "complete" if not failed else "complete_with_failures"
    write_status(
        output_dir,
        state=final_state,
        active=None,
        completed=completed,
        failed=failed,
    )
    write_artifact_inventory(output_dir)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
