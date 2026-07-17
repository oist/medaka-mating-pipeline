"""Benchmark vectorizable kernels used by windowed 3D tracklet association.

This is deliberately a decision aid rather than an alternate implementation.
It measures float64 endpoint/prediction cost primitives from a real fragment
table and batched 3x3 linear solves representative of ray intersections.  Both
CUDA kernel-only and transfer-inclusive timings are reported when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timed(callable_, repeats: int, synchronize=None) -> tuple[float, list[float]]:
    callable_()
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples)), samples


def _load_endpoint_arrays(path: Path, sample_size: int) -> tuple[np.ndarray, ...]:
    columns = [
        "start_X", "start_Y", "start_Z", "end_X", "end_Y", "end_Z",
        "end_vX", "end_vY", "end_vZ", "frame_start", "frame_end",
    ]
    table = pd.read_parquet(path, columns=columns).sort_values(
        ["frame_start", "frame_end"]
    )
    if len(table) < 2:
        raise ValueError("fragment table needs at least two rows")
    source = table[["end_X", "end_Y", "end_Z"]].to_numpy(dtype=np.float64)
    target = np.roll(
        table[["start_X", "start_Y", "start_Z"]].to_numpy(dtype=np.float64),
        -1,
        axis=0,
    )
    velocity = table[["end_vX", "end_vY", "end_vZ"]].to_numpy(dtype=np.float64)
    velocity = np.nan_to_num(velocity)
    target_start = np.roll(table.frame_start.to_numpy(dtype=np.int64), -1)
    gap = np.clip(target_start - table.frame_end.to_numpy(dtype=np.int64), 1, 100)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source, target, velocity, gap = (
        value[finite] for value in (source, target, velocity, gap)
    )
    if not len(source):
        raise ValueError("no finite endpoint pairs")
    indices = np.arange(sample_size, dtype=np.int64) % len(source)
    return source[indices], target[indices], velocity[indices], gap[indices].astype(np.float64)


def _numpy_endpoint(
    source: np.ndarray,
    target: np.ndarray,
    velocity: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    endpoint = np.linalg.norm(target - source, axis=1)
    prediction = np.linalg.norm(target - (source + velocity * gap[:, None]), axis=1)
    return np.clip(endpoint / 8.0, 0.0, 4.0) + 2.0 * np.clip(
        prediction / 8.0, 0.0, 4.0
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    fragment_path = args.fragments.resolve()
    source, target, velocity, gap = _load_endpoint_arrays(
        fragment_path, args.sample_size
    )
    cpu_endpoint_result = _numpy_endpoint(source, target, velocity, gap)
    cpu_endpoint_seconds, cpu_endpoint_samples = _timed(
        lambda: _numpy_endpoint(source, target, velocity, gap), args.repeats
    )

    rng = np.random.default_rng(args.seed)
    base = rng.normal(size=(args.solve_batch_size, 3, 3))
    matrices = np.swapaxes(base, 1, 2) @ base + np.eye(3)[None] * 0.25
    rhs = rng.normal(size=(args.solve_batch_size, 3, 1))
    cpu_solve_result = np.linalg.solve(matrices, rhs)
    cpu_solve_seconds, cpu_solve_samples = _timed(
        lambda: np.linalg.solve(matrices, rhs), args.repeats
    )

    result: dict[str, object] = {
        "schema_version": "windowed_tracklet_hardware_benchmark_v1",
        "fragment_file": str(fragment_path),
        "fragment_file_sha256": _sha256(fragment_path),
        "seed": args.seed,
        "repeats": args.repeats,
        "endpoint_pairs": args.sample_size,
        "solve_batch_size": args.solve_batch_size,
        "dtype": "float64",
        "environment": {
            "platform": platform.platform(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cpu": {
            "endpoint_seconds_median": cpu_endpoint_seconds,
            "endpoint_seconds_samples": cpu_endpoint_samples,
            "endpoint_pairs_per_second": args.sample_size / cpu_endpoint_seconds,
            "solve_seconds_median": cpu_solve_seconds,
            "solve_seconds_samples": cpu_solve_samples,
            "solves_per_second": args.solve_batch_size / cpu_solve_seconds,
        },
        "cuda": {"available": bool(torch.cuda.is_available())},
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensors = [torch.as_tensor(value, device=device) for value in (source, target, velocity, gap)]
        matrix_tensor = torch.as_tensor(matrices, device=device)
        rhs_tensor = torch.as_tensor(rhs, device=device)

        def cuda_endpoint():
            source_t, target_t, velocity_t, gap_t = tensors
            endpoint = torch.linalg.vector_norm(target_t - source_t, dim=1)
            prediction = torch.linalg.vector_norm(
                target_t - (source_t + velocity_t * gap_t[:, None]), dim=1
            )
            return torch.clamp(endpoint / 8.0, 0.0, 4.0) + 2.0 * torch.clamp(
                prediction / 8.0, 0.0, 4.0
            )

        endpoint_gpu = cuda_endpoint()
        solve_gpu = torch.linalg.solve(matrix_tensor, rhs_tensor)
        torch.cuda.synchronize()
        endpoint_kernel_seconds, endpoint_kernel_samples = _timed(
            cuda_endpoint, args.repeats, torch.cuda.synchronize
        )
        solve_kernel_seconds, solve_kernel_samples = _timed(
            lambda: torch.linalg.solve(matrix_tensor, rhs_tensor),
            args.repeats,
            torch.cuda.synchronize,
        )

        def cuda_endpoint_transfer():
            local = [torch.as_tensor(value, device=device) for value in (source, target, velocity, gap)]
            source_t, target_t, velocity_t, gap_t = local
            endpoint = torch.linalg.vector_norm(target_t - source_t, dim=1)
            prediction = torch.linalg.vector_norm(
                target_t - (source_t + velocity_t * gap_t[:, None]), dim=1
            )
            output = torch.clamp(endpoint / 8.0, 0.0, 4.0) + 2.0 * torch.clamp(
                prediction / 8.0, 0.0, 4.0
            )
            output.cpu().numpy()

        endpoint_transfer_seconds, endpoint_transfer_samples = _timed(
            cuda_endpoint_transfer, args.repeats, torch.cuda.synchronize
        )
        result["cuda"] = {
            "available": True,
            "endpoint_kernel_seconds_median": endpoint_kernel_seconds,
            "endpoint_kernel_seconds_samples": endpoint_kernel_samples,
            "endpoint_kernel_pairs_per_second": args.sample_size / endpoint_kernel_seconds,
            "endpoint_transfer_seconds_median": endpoint_transfer_seconds,
            "endpoint_transfer_seconds_samples": endpoint_transfer_samples,
            "endpoint_transfer_pairs_per_second": args.sample_size / endpoint_transfer_seconds,
            "solve_kernel_seconds_median": solve_kernel_seconds,
            "solve_kernel_seconds_samples": solve_kernel_samples,
            "solve_kernel_per_second": args.solve_batch_size / solve_kernel_seconds,
            "endpoint_max_abs_difference": float(
                np.max(np.abs(endpoint_gpu.cpu().numpy() - cpu_endpoint_result))
            ),
            "solve_max_abs_difference": float(
                np.max(np.abs(solve_gpu.cpu().numpy() - cpu_solve_result))
            ),
        }
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fragments", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--sample-size", type=int, default=1_000_000)
    value.add_argument("--solve-batch-size", type=int, default=250_000)
    value.add_argument("--repeats", type=int, default=7)
    value.add_argument("--seed", type=int, default=20260717)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
