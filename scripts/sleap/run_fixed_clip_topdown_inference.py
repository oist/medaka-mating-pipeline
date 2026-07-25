from __future__ import annotations

"""Run bounded top-down checkpoint inference with independent stage thresholds.

This wrapper intentionally calls ``sleap_nn.inference.predict`` with trained
checkpoint directories. It does not accept exported ONNX/TensorRT models. The
legacy ``sleap_nn.cli track`` command installed on the production machine has
only one peak-threshold option, so it cannot represent the Phase-5/6
centroid-versus-keypoint settings faithfully.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


def _artifact_record(model_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"model_dir": str(model_dir)}
    for name in ("best.ckpt", "training_config.yaml"):
        path = model_dir / name
        if path.is_file():
            stat = path.stat()
            record[name] = {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
        else:
            record[name] = None
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_topdown_clip(
    *,
    video: Path,
    centroid_model: Path,
    centered_instance_model: Path,
    output_slp: Path,
    start_frame: int,
    end_frame: int,
    centroid_threshold: float,
    keypoint_threshold: float,
    max_instances: int,
    batch_size: int,
    device: str,
    predict_fn: Callable[..., Any] | None = None,
) -> Any:
    """Run an inclusive clip using the checkpoint-only torch inference API."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(
            f"invalid inclusive frame bounds: {start_frame}:{end_frame}"
        )
    for threshold in (centroid_threshold, keypoint_threshold):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be between 0 and 1, got {threshold}"
            )
    if max_instances < 1 or batch_size < 1:
        raise ValueError("max_instances and batch_size must be positive")
    if predict_fn is None:
        from sleap_nn.inference import predict

        predict_fn = predict

    output_slp.parent.mkdir(parents=True, exist_ok=True)
    labels = predict_fn(
        source=str(video),
        model_paths=[
            str(centroid_model),
            str(centered_instance_model),
        ],
        device=device,
        batch_size=batch_size,
        centroid_only=False,
        emit_centroid="instance",
        frames=list(range(start_frame, end_frame + 1)),
        centroid_threshold=centroid_threshold,
        keypoint_threshold=keypoint_threshold,
        max_instances=max_instances,
        integral_refinement="integral",
        output_path=str(output_slp),
        output_format="slp",
        embed="false",
        restore_source_videos=True,
        clean_empty_frames=False,
    )
    if not output_slp.is_file():
        raise RuntimeError(
            f"SLEAP inference returned without creating {output_slp}"
        )
    provenance = {
        "schema_version": "fixed_clip_topdown_inference_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inference_api": "sleap_nn.inference.predict",
        "checkpoint_only_torch_path": True,
        "exported_runtime_used": False,
        "video": str(video),
        "output_slp": str(output_slp),
        "inclusive_frame_window": {
            "start": start_frame,
            "end": end_frame,
            "count": end_frame - start_frame + 1,
        },
        "centroid_model": _artifact_record(centroid_model),
        "centered_instance_model": _artifact_record(
            centered_instance_model
        ),
        "centroid_threshold": centroid_threshold,
        "keypoint_threshold": keypoint_threshold,
        "max_instances": max_instances,
        "batch_size": batch_size,
        "device": device,
        "integral_refinement": "integral",
    }
    output_slp.with_suffix(".inference_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--centroid-model", type=Path, required=True)
    parser.add_argument(
        "--centered-instance-model", type=Path, required=True
    )
    parser.add_argument("--output-slp", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--centroid-threshold", type=float, required=True)
    parser.add_argument("--keypoint-threshold", type=float, required=True)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    for path, label in (
        (args.video, "video"),
        (args.centroid_model / "best.ckpt", "centroid checkpoint"),
        (
            args.centered_instance_model / "best.ckpt",
            "centered-instance checkpoint",
        ),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    try:
        run_topdown_clip(
            video=args.video,
            centroid_model=args.centroid_model,
            centered_instance_model=args.centered_instance_model,
            output_slp=args.output_slp,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            centroid_threshold=args.centroid_threshold,
            keypoint_threshold=args.keypoint_threshold,
            max_instances=args.max_instances,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
