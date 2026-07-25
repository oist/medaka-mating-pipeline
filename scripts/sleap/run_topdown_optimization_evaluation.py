from __future__ import annotations

"""Run the side-view optimization candidates on the shared benchmark.

The evaluation is deliberately decomposed into:

1. centroid-only inference for sigma 2.5, 2.0, and 1.5;
2. GT-mouth oracle crops for the centered-instance control, +/-8 px, and
   +/-16 px candidates;
3. the complete 3 x 3 top-down pairing grid.

All jobs use the Phase-1 partial-label-aware matching and metric code. The
shared benchmark is diagnostic rather than a genuinely locked test; provenance
is copied into the evaluation manifest and final report.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths


DEFAULT_BENCHMARK = Path(
    "report/20260724/sleap_topdown_optimization/"
    "benchmark_manifest_phase1_v1.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "report/20260725/sleap_side_optimization_evaluation"
)


@dataclass(frozen=True)
class EvaluationJob:
    name: str
    mode: str
    models: dict[str, tuple[Path, ...]]
    strict_anchor_eligible_only: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def stage_sigma25_best(
    run_dir: Path, staging_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Stage the recovered sigma-2.5 best without mutating its training run."""
    best = run_dir / "best.ckpt"
    provenance: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "staging_dir": str(staging_dir.resolve()),
        "staged": True,
    }
    if best.is_file():
        source = best
        provenance["source_kind"] = "live_best"
    else:
        candidates = sorted(run_dir.glob("best.interrupted_*.ckpt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected one archived sigma-2.5 best checkpoint under "
                f"{run_dir}, found {[path.name for path in candidates]}"
            )
        source = candidates[0]
        provenance["source_kind"] = "archived_interrupted_best"

    import torch

    checkpoint = torch.load(
        source, map_location="cpu", weights_only=True
    )
    if not checkpoint.get("state_dict"):
        raise ValueError(f"Selected checkpoint has no state_dict: {source}")

    source_config = run_dir / "training_config.yaml"
    if not source_config.is_file():
        raise FileNotFoundError(
            f"Sigma-2.5 training config not found: {source_config}"
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_best = staging_dir / "best.ckpt"
    staged_config = staging_dir / "training_config.yaml"
    source_sha256 = sha256_file(source)
    if (
        not staged_best.is_file()
        or sha256_file(staged_best) != source_sha256
    ):
        shutil.copy2(source, staged_best)
    config_sha256 = sha256_file(source_config)
    if (
        not staged_config.is_file()
        or sha256_file(staged_config) != config_sha256
    ):
        shutil.copy2(source_config, staged_config)
    provenance.update(
        {
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": source_sha256,
            "staged_checkpoint": str(staged_best.resolve()),
            "staged_checkpoint_sha256": sha256_file(staged_best),
            "staged_training_config": str(staged_config.resolve()),
            "staged_training_config_sha256": sha256_file(staged_config),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(
                checkpoint.get("global_step", -1)
            ),
        }
    )
    return staging_dir, provenance


def resolve_candidate_runs(model_root: Path) -> dict[str, Path]:
    runs = model_root / "side" / "runs"
    names = {
        "centroid_sigma25": "side_centroid_sigma25_control_p40",
        "centroid_sigma20": "side_centroid_sigma20_strictframe_p40",
        "centroid_sigma15": "side_centroid_sigma15_strictframe_p40",
        "instance_control": "side_centered_instance_control_p40",
        "instance_jitter8": "side_centered_instance_opt_jitter8_p40",
        "instance_jitter16": "side_centered_instance_opt_jitter16_p40",
    }
    resolved = {name: runs / run_name for name, run_name in names.items()}
    missing = [path for path in resolved.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Candidate run directories missing: {missing}")
    return resolved


def build_jobs(candidate_runs: dict[str, Path]) -> list[EvaluationJob]:
    centroids = {
        name: (candidate_runs[name],)
        for name in (
            "centroid_sigma25",
            "centroid_sigma20",
            "centroid_sigma15",
        )
    }
    instances = {
        name: (candidate_runs[name],)
        for name in (
            "instance_control",
            "instance_jitter8",
            "instance_jitter16",
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
            name="centroid_only",
            mode="centroid_only",
            models=centroids,
            strict_anchor_eligible_only=True,
        ),
        EvaluationJob(
            name="oracle_centered_instance",
            mode="oracle_centered_instance",
            models=instances,
            strict_anchor_eligible_only=True,
        ),
        EvaluationJob(
            name="full_topdown_grid",
            mode="full_topdown",
            models=full_grid,
            strict_anchor_eligible_only=False,
        ),
    ]


def build_command(
    *,
    job: EvaluationJob,
    benchmark: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> list[str]:
    output_json = output_dir / f"side_{job.name}_shared_benchmark.json"
    command = [
        sys.executable,
        "-m",
        "scripts.sleap.compare_rounds_topdown",
        "--view",
        "side",
        "--manifest",
        str(benchmark.resolve()),
        "--evaluation-mode",
        job.mode,
        "--predictions-dir",
        str((output_dir / "predictions").resolve()),
        "--out-json",
        str(output_json.resolve()),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--reuse-existing-predictions",
    ]
    if job.strict_anchor_eligible_only:
        command.append("--strict-centroid-eligible-only")
    for name, paths in job.models.items():
        command.extend(
            [
                "--model",
                f"{name}={','.join(str(path.resolve()) for path in paths)}",
            ]
        )
    return command


def summarize_benchmark(benchmark: Path) -> dict[str, Any]:
    """Record benchmark counts used to interpret the evaluation scopes."""
    with benchmark.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    side_rows = [
        row
        for row in rows
        if str(row.get("view", "")).casefold().startswith("side")
    ]
    strict_rows = [
        row
        for row in side_rows
        if str(row.get("strict_centroid_eligible", "")).strip().lower()
        in {"1", "true", "yes", "y"}
    ]

    def counts(key: str) -> dict[str, int]:
        values = sorted({row.get(key, "") for row in side_rows})
        return {
            value: sum(row.get(key, "") == value for row in side_rows)
            for value in values
        }

    return {
        "all_rows": len(rows),
        "side_rows": len(side_rows),
        "side_strict_centroid_eligible_rows": len(strict_rows),
        "side_by_cohort": counts("cohort"),
        "side_by_experiment": counts("experiment"),
        "side_by_difficulty": counts("difficulty"),
    }


def write_status(
    output_dir: Path,
    *,
    state: str,
    active: str | None,
    completed: list[str],
    failed: list[dict[str, Any]],
) -> None:
    (output_dir / "evaluation_status.json").write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "state": state,
                "active": active,
                "completed": completed,
                "failed": failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


SUMMARY_METRICS = (
    "n_frames",
    "n_gt",
    "precision",
    "recall",
    "f1",
    "n_missed",
    "n_extra",
    "merge_rate",
    "all_three_detection_coverage",
    "mouth_err_median",
    "mouth_err_p90",
    "mouth_err_p95",
    "mouth_err_p99",
    "mouth_pck5",
    "tail_err_median",
    "tail_err_p90",
    "tail_err_p95",
    "tail_err_p99",
    "tail_pck5",
    "swap_rate",
    "shared_tail_rate",
)


def aggregate_results(output_dir: Path, jobs: list[EvaluationJob]) -> None:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for job in jobs:
        path = output_dir / f"side_{job.name}_shared_benchmark.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[job.name] = payload
        for candidate, result in payload["results"].items():
            scopes = [("overall", "all", result["overall"])]
            for scope_key, output_key in (
                ("per_cohort", "cohort"),
                ("per_experiment", "experiment"),
                ("per_camera", "camera"),
                ("per_difficulty", "difficulty"),
            ):
                scopes.extend(
                    (output_key, name, metrics)
                    for name, metrics in result.get(scope_key, {}).items()
                )
            for scope, subset, metrics in scopes:
                row = {
                    "evaluation_mode": job.mode,
                    "candidate": candidate,
                    "scope": scope,
                    "subset": subset,
                }
                row.update(
                    {metric: metrics.get(metric) for metric in SUMMARY_METRICS}
                )
                rows.append(row)
            for cohort, metrics in result.get("per_cohort", {}).items():
                for label_subset, subset_metrics in (
                    metrics.get("keypoint_subsets", {}).items()
                ):
                    mouth = subset_metrics.get("mouth", {})
                    tail = subset_metrics.get("tail", {})
                    rows.append(
                        {
                            "evaluation_mode": job.mode,
                            "candidate": candidate,
                            "scope": "cohort_label_subset",
                            "subset": f"{cohort}|{label_subset}",
                            "n_frames": metrics.get("n_frames"),
                            "n_gt": subset_metrics.get("n_gt"),
                            "precision": None,
                            "recall": subset_metrics.get("match_rate"),
                            "f1": None,
                            "n_missed": (
                                subset_metrics.get("n_gt", 0)
                                - subset_metrics.get("n_matched", 0)
                            ),
                            "n_extra": None,
                            "merge_rate": None,
                            "all_three_detection_coverage": None,
                            "mouth_err_median": mouth.get("median"),
                            "mouth_err_p90": mouth.get("p90"),
                            "mouth_err_p95": mouth.get("p95"),
                            "mouth_err_p99": mouth.get("p99"),
                            "mouth_pck5": mouth.get("pck5"),
                            "tail_err_median": tail.get("median"),
                            "tail_err_p90": tail.get("p90"),
                            "tail_err_p95": tail.get("p95"),
                            "tail_err_p99": tail.get("p99"),
                            "tail_pck5": tail.get("pck5"),
                            "swap_rate": None,
                            "shared_tail_rate": None,
                        }
                    )

    summary_csv = output_dir / "side_optimization_metrics_summary.csv"
    if rows:
        with summary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Side top-down optimization: shared-benchmark evaluation",
        "",
        "This is diagnostic selection/regression evidence, not a genuinely "
        "locked generalization test. The pilot is Round-8-held-out but "
        "Round-7-exposed; the stress cohort is in-sample.",
        "",
    ]
    for job in jobs:
        payload = payloads.get(job.name)
        if payload is None:
            continue
        lines.extend([f"## {job.name.replace('_', ' ')}", ""])
        cohort_names = sorted(
            {
                cohort
                for result in payload["results"].values()
                for cohort in result.get("per_cohort", {})
            }
        )
        for cohort in cohort_names:
            lines.extend(
                [
                    f"### {cohort}",
                    "",
                    "| candidate | frames | precision | recall | all 3 | "
                    "mouth p95 | tail p95 | swaps | shared tails |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for candidate, result in payload["results"].items():
                metrics = result["per_cohort"][cohort]

                def fmt(key: str) -> str:
                    value = metrics.get(key)
                    if value is None:
                        return "n/a"
                    try:
                        return f"{float(value):.4f}"
                    except (TypeError, ValueError):
                        return str(value)

                lines.append(
                    f"| {candidate} | {metrics.get('n_frames')} | "
                    f"{fmt('precision')} | {fmt('recall')} | "
                    f"{fmt('all_three_detection_coverage')} | "
                    f"{fmt('mouth_err_p95')} | {fmt('tail_err_p95')} | "
                    f"{fmt('swap_rate')} | {fmt('shared_tail_rate')} |"
                )
            lines.append("")
            lines.extend(
                [
                    "Fully labeled instances only:",
                    "",
                    "| candidate | matched / GT | mouth p95 | mouth p99 | "
                    "tail p95 | tail p99 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for candidate, result in payload["results"].items():
                fully_labeled = result["per_cohort"][cohort][
                    "keypoint_subsets"
                ]["fully_labeled"]
                mouth = fully_labeled["mouth"]
                tail = fully_labeled["tail"]

                def subset_fmt(values: dict[str, Any], key: str) -> str:
                    value = values.get(key)
                    if value is None:
                        return "n/a"
                    try:
                        return f"{float(value):.4f}"
                    except (TypeError, ValueError):
                        return str(value)

                lines.append(
                    f"| {candidate} | {fully_labeled.get('n_matched')} / "
                    f"{fully_labeled.get('n_gt')} | "
                    f"{subset_fmt(mouth, 'p95')} | "
                    f"{subset_fmt(mouth, 'p99')} | "
                    f"{subset_fmt(tail, 'p95')} | "
                    f"{subset_fmt(tail, 'p99')} |"
                )
            lines.append("")
    (output_dir / "EVALUATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_artifact_inventory(output_dir: Path) -> None:
    """Close the provenance chain over prediction caches and result files."""
    prediction_paths = sorted((output_dir / "predictions").glob("*.slp"))
    result_paths = sorted(
        {
            output_dir / "evaluation_manifest.json",
            output_dir / "evaluation_status.json",
            output_dir / "EVALUATION_REPORT.md",
            output_dir / "side_optimization_metrics_summary.csv",
            *output_dir.glob("side_*_shared_benchmark.json"),
            *output_dir.glob("side_*_shared_benchmark_matches.csv"),
        }
    )
    missing = [path for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot inventory missing evaluation artifacts: {missing}"
        )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prediction_cache_count": len(prediction_paths),
        "prediction_caches": [
            artifact_record(path, relative_to=output_dir)
            for path in prediction_paths
        ],
        "result_artifacts": [
            artifact_record(path, relative_to=output_dir)
            for path in result_paths
        ],
    }
    (output_dir / "evaluation_artifact_inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG
    )
    parser.add_argument("--model-root", type=Path, default=None)
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
            "centroid_only",
            "oracle_centered_instance",
            "full_topdown_grid",
        ),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args(argv)

    local_paths = load_local_paths(args.paths_config)
    if args.model_root is None:
        if local_paths is None or local_paths.sleap_models_dir is None:
            parser.error(
                "paths config must define paths.sleap_models_dir or pass "
                "--model-root"
            )
        model_root = (
            local_paths.sleap_models_dir / "retrain_20260724_sideopt"
        )
    else:
        model_root = args.model_root
    benchmark = args.benchmark
    if not benchmark.is_file():
        parser.error(f"benchmark not found: {benchmark}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    source_candidate_runs = resolve_candidate_runs(model_root)
    sigma25_staging_dir = output_dir / "model_staging" / "centroid_sigma25"
    staged_sigma25, recovery = stage_sigma25_best(
        source_candidate_runs["centroid_sigma25"],
        sigma25_staging_dir,
    )
    candidate_runs = dict(source_candidate_runs)
    candidate_runs["centroid_sigma25"] = staged_sigma25
    jobs = build_jobs(candidate_runs)
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [job for job in jobs if job.name in wanted]

    model_artifacts: dict[str, dict[str, Any]] = {}
    for name, run_dir in candidate_runs.items():
        best = run_dir / "best.ckpt"
        config = run_dir / "training_config.yaml"
        if not best.is_file() or not config.is_file():
            raise FileNotFoundError(
                f"Candidate is not inference-ready: {run_dir}"
            )
        model_artifacts[name] = {
            "source_run_dir": str(source_candidate_runs[name].resolve()),
            "run_dir": str(run_dir.resolve()),
            "best_checkpoint": str(best.resolve()),
            "best_checkpoint_sha256": sha256_file(best),
            "training_config": str(config.resolve()),
            "training_config_sha256": sha256_file(config),
        }

    repo_root = Path(__file__).resolve().parents[2]
    evaluator_sources = [
        Path(__file__).resolve(),
        repo_root / "scripts" / "sleap" / "compare_rounds_topdown.py",
        repo_root / "scripts" / "sleap" / "eval_topdown_vs_bottomup.py",
    ]
    evaluation_manifest = {
        "workflow": "side_topdown_shared_benchmark_evaluation",
        "scientific_status": (
            "diagnostic_not_locked_generalization_test"
        ),
        "benchmark": str(benchmark.resolve()),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_summary": summarize_benchmark(benchmark),
        "evaluator_sources": [
            artifact_record(path, relative_to=repo_root)
            for path in evaluator_sources
        ],
        "model_root": str(model_root.resolve()),
        "sigma25_checkpoint_recovery": recovery,
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
        json.dumps(evaluation_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.prepare_only:
        print(
            f"Prepared {len(jobs)} evaluation jobs under {output_dir}",
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
        log_path = output_dir / f"{job.name}.console.log"
        print(f"\n--- {job.name} ---", flush=True)
        print(" ".join(command), flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as stream:
            result = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
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
