from __future__ import annotations

"""Run a provenance-locked top-down inference-threshold grid.

The Phase-1/3 evaluator is intentionally left unchanged. This sibling runner
uses the same manifest loading, partial-label-aware matching, and reporting
logic, while forwarding separate centroid and centered-instance confidence
thresholds to the installed ``sleap_nn.inference.predict`` API.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.sleap import compare_rounds_topdown as comparison
from scripts.sleap import eval_topdown_vs_bottomup as evaluator
from scripts.sleap.run_topdown_optimization_evaluation import (
    DEFAULT_BENCHMARK,
    SUMMARY_METRICS,
    artifact_record,
    sha256_file,
    summarize_benchmark,
)


DEFAULT_OUTPUT_DIR = Path(
    "report/20260725/sleap_side_optimization_phase5_thresholds"
)
DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)


def threshold_token(value: float) -> str:
    """Return a stable compact token for a [0, 1] confidence threshold."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {value}")
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def threshold_label(
    centroid_threshold: float,
    keypoint_threshold: float,
) -> str:
    return (
        f"centroid{threshold_token(centroid_threshold)}"
        f"__keypoint{threshold_token(keypoint_threshold)}"
    )


def run_threshold_strategy(
    *,
    strategy: str,
    model_paths: list[Path],
    subset_labels,
    max_instances: int,
    batch_size: int,
    device: str,
    centroid_threshold: float,
    keypoint_threshold: float,
    output_path: Path | None = None,
    prediction_mode: str = "full_topdown",
):
    """Run top-down inference with explicit stage-specific thresholds."""
    if prediction_mode != "full_topdown":
        raise ValueError(
            "Phase-5 threshold evaluation currently requires full_topdown"
        )
    from sleap_nn.inference import predict

    print(
        f"\n--- {strategy}: centroid={centroid_threshold:g}, "
        f"keypoint={keypoint_threshold:g} ---"
    )
    output = predict(
        source=subset_labels,
        model_paths=[str(path) for path in model_paths],
        centroid_only=False,
        emit_centroid="instance",
        max_instances=max_instances,
        batch_size=batch_size,
        device=device,
        output_path=None,
        clean_empty_frames=False,
        centroid_threshold=centroid_threshold,
        keypoint_threshold=keypoint_threshold,
    )
    output = evaluator.make_prediction_labels_self_contained(output)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_prediction_slp(output, output_path)
        print(f"    cached predictions: {output_path}")
    return output


def suffixed_model_specs(
    model_specs: list[str],
    *,
    centroid_threshold: float,
    keypoint_threshold: float,
) -> list[str]:
    """Encode thresholds into candidate names to make caches collision-safe."""
    suffix = threshold_label(centroid_threshold, keypoint_threshold)
    values: list[str] = []
    for value in model_specs:
        name, separator, paths = value.partition("=")
        if not separator or not name.strip() or not paths.strip():
            raise ValueError(f"--model must be NAME=CENTROID,INSTANCE: {value}")
        values.append(f"{name.strip()}__{suffix}={paths.strip()}")
    return values


def build_single_command(
    *,
    model_specs: list[str],
    benchmark: Path,
    output_dir: Path,
    centroid_threshold: float,
    keypoint_threshold: float,
    max_instances: int,
    batch_size: int,
    device: str,
    force: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.sleap.run_phase5_threshold_evaluation",
        "--benchmark",
        str(benchmark.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--centroid-thresholds",
        str(centroid_threshold),
        "--keypoint-thresholds",
        str(keypoint_threshold),
        "--max-instances",
        str(max_instances),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--run-one",
    ]
    for model in model_specs:
        command.extend(["--model", model])
    if force:
        command.append("--force")
    return command


def _run_one(args: argparse.Namespace) -> int:
    centroid_threshold = args.centroid_thresholds[0]
    keypoint_threshold = args.keypoint_thresholds[0]
    label = threshold_label(centroid_threshold, keypoint_threshold)
    output_json = (
        args.output_dir.resolve()
        / f"side_full_topdown_threshold_{label}_shared_benchmark.json"
    )
    if output_json.is_file() and not args.force:
        print(f"[skip complete] {label}: {output_json}", flush=True)
        return 0

    original_run_strategy = comparison.E.run_strategy

    def thresholded_strategy(**kwargs):
        return run_threshold_strategy(
            **kwargs,
            centroid_threshold=centroid_threshold,
            keypoint_threshold=keypoint_threshold,
        )

    comparison.E.run_strategy = thresholded_strategy
    try:
        compare_args = [
            "--view",
            "side",
            "--evaluation-mode",
            "full_topdown",
            "--manifest",
            str(args.benchmark.resolve()),
            "--predictions-dir",
            str((args.output_dir.resolve() / "predictions")),
            "--out-json",
            str(output_json),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--max-instances",
            str(args.max_instances),
            "--reuse-existing-predictions",
        ]
        for model in suffixed_model_specs(
            args.model,
            centroid_threshold=centroid_threshold,
            keypoint_threshold=keypoint_threshold,
        ):
            compare_args.extend(["--model", model])
        return_code = comparison.main(compare_args)
        if return_code == 0 and output_json.is_file():
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            payload["phase5_inference_thresholds"] = {
                "centroid_threshold": centroid_threshold,
                "keypoint_threshold": keypoint_threshold,
                "peak_threshold": None,
            }
            payload["max_instances"] = args.max_instances
            output_json.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        return return_code
    finally:
        comparison.E.run_strategy = original_run_strategy


def aggregate_threshold_results(
    output_dir: Path,
    threshold_pairs: list[tuple[float, float]],
) -> None:
    rows: list[dict[str, Any]] = []
    for centroid_threshold, keypoint_threshold in threshold_pairs:
        label = threshold_label(centroid_threshold, keypoint_threshold)
        result_path = (
            output_dir
            / f"side_full_topdown_threshold_{label}_shared_benchmark.json"
        )
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for candidate, result in payload["results"].items():
            suffix = f"__{label}"
            model_pair = (
                candidate.removesuffix(suffix)
                if candidate.endswith(suffix)
                else candidate
            )
            scopes = [("overall", "all", result["overall"])]
            for input_key, scope in (
                ("per_cohort", "cohort"),
                ("per_experiment", "experiment"),
                ("per_camera", "camera"),
                ("per_difficulty", "difficulty"),
            ):
                scopes.extend(
                    (scope, subset, metrics)
                    for subset, metrics in result.get(input_key, {}).items()
                )
            for scope, subset, metrics in scopes:
                row = {
                    "model_pair": model_pair,
                    "centroid_threshold": centroid_threshold,
                    "keypoint_threshold": keypoint_threshold,
                    "max_instances": payload.get("max_instances"),
                    "scope": scope,
                    "subset": subset,
                }
                row.update(
                    {
                        metric: metrics.get(metric)
                        for metric in SUMMARY_METRICS
                    }
                )
                rows.append(row)

    summary_path = output_dir / "threshold_metrics_summary.csv"
    if rows:
        with summary_path.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    overall = [row for row in rows if row["scope"] == "overall"]
    overall.sort(
        key=lambda row: (
            -(float(row.get("all_three_detection_coverage") or 0.0)),
            float(row.get("n_missed") or 0.0),
            float(row.get("merge_rate") or 0.0),
            float(row.get("mouth_err_p95") or float("inf")),
            float(row.get("tail_err_p95") or float("inf")),
        )
    )
    lines = [
        "# Phase 5 side-view inference-threshold sweep",
        "",
        "This shared benchmark is diagnostic selection/regression evidence, "
        "not the locked independent test.",
        "",
        "| model pair | centroid | keypoint | all 3 | missed | merge | "
        "mouth p95 | tail p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['model_pair']} | "
            f"{float(row['centroid_threshold']):.2f} | "
            f"{float(row['keypoint_threshold']):.2f} | "
            f"{float(row.get('all_three_detection_coverage') or 0):.4f} | "
            f"{row.get('n_missed')} | "
            f"{float(row.get('merge_rate') or 0):.4f} | "
            f"{float(row.get('mouth_err_p95') or float('nan')):.3f} | "
            f"{float(row.get('tail_err_p95') or float('nan')):.3f} |"
        )
    (output_dir / "THRESHOLD_EVALUATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_artifact_inventory(output_dir: Path) -> None:
    paths = sorted(
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.name != "threshold_artifact_inventory.json"
        ]
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_count": len(paths),
        "artifacts": [
            artifact_record(path, relative_to=output_dir) for path in paths
        ],
    }
    (output_dir / "threshold_artifact_inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CENTROID_PATH,CENTERED_INSTANCE_PATH",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_BENCHMARK
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--centroid-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--keypoint-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-fail", action="store_true")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    for value in (
        *args.centroid_thresholds,
        *args.keypoint_thresholds,
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"thresholds must be between 0 and 1, got {value}")
    if args.max_instances < 1:
        parser.error("--max-instances must be positive")
    if not args.benchmark.is_file():
        parser.error(f"benchmark not found: {args.benchmark}")
    try:
        parsed_models = comparison.parse_model_specs(
            args.model, "full_topdown"
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    if args.run_one:
        if (
            len(args.centroid_thresholds) != 1
            or len(args.keypoint_thresholds) != 1
        ):
            parser.error("--run-one requires exactly one threshold per stage")
        return _run_one(args)

    threshold_pairs = [
        (centroid, keypoint)
        for centroid in args.centroid_thresholds
        for keypoint in args.keypoint_thresholds
    ]
    commands = [
        build_single_command(
            model_specs=args.model,
            benchmark=args.benchmark,
            output_dir=output_dir,
            centroid_threshold=centroid,
            keypoint_threshold=keypoint,
            max_instances=args.max_instances,
            batch_size=args.batch_size,
            device=args.device,
            force=args.force,
        )
        for centroid, keypoint in threshold_pairs
    ]
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        repo_root / "scripts" / "sleap" / "compare_rounds_topdown.py",
        repo_root / "scripts" / "sleap" / "eval_topdown_vs_bottomup.py",
    ]
    manifest = {
        "workflow": "side_topdown_phase5_inference_threshold_grid",
        "scientific_status": "diagnostic_not_locked_generalization_test",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "benchmark_summary": summarize_benchmark(args.benchmark),
        "max_instances": args.max_instances,
        "centroid_thresholds": args.centroid_thresholds,
        "keypoint_thresholds": args.keypoint_thresholds,
        "model_artifacts": {
            name: [
                {
                    "run_dir": str(path.resolve()),
                    "best_checkpoint_sha256": sha256_file(
                        path / "best.ckpt"
                    ),
                    "training_config_sha256": sha256_file(
                        path / "training_config.yaml"
                    ),
                }
                for path in paths
            ]
            for name, paths in parsed_models.items()
        },
        "evaluator_sources": [
            artifact_record(path, relative_to=repo_root)
            for path in source_paths
        ],
        "jobs": [
            {
                "centroid_threshold": centroid,
                "keypoint_threshold": keypoint,
                "label": threshold_label(centroid, keypoint),
                "command": command,
            }
            for (centroid, keypoint), command in zip(
                threshold_pairs, commands
            )
        ],
    }
    (output_dir / "threshold_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(
            f"Prepared {len(commands)} threshold jobs under {output_dir}",
            flush=True,
        )
        return 0

    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    status_path = output_dir / "threshold_evaluation_status.json"
    for (centroid, keypoint), command in zip(threshold_pairs, commands):
        label = threshold_label(centroid, keypoint)
        status_path.write_text(
            json.dumps(
                {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "state": "running",
                    "active": label,
                    "completed": completed,
                    "failed": failed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log_path = output_dir / f"{label}.console.log"
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
        if result.returncode == 0:
            completed.append(label)
        else:
            failed.append(
                {"job": label, "return_code": int(result.returncode)}
            )
            if not args.continue_on_fail:
                break

    aggregate_threshold_results(output_dir, threshold_pairs)
    final_state = "complete" if not failed else "complete_with_failures"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "state": final_state,
                "active": None,
                "completed": completed,
                "failed": failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_artifact_inventory(output_dir)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
