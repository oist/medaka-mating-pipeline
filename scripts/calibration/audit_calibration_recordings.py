"""Audit externally triggered multi-camera calibration recordings.

Checks video/frame-time completeness and creates uniformly sampled contact
sheets for target/pattern and coverage inspection. Camera-internal timestamps
are used only for within-recording gap checks, never for cross-camera sync.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


CAMERAS = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


def _timestamp_vector(path: Path) -> np.ndarray | None:
    arrays = sorted(path.glob("*frametime*.npy"))
    if not arrays:
        return None
    values = np.asarray(np.load(arrays[0]), dtype=float)
    if values.ndim == 2:
        if values.shape[0] == 2:
            return values[1]
        if values.shape[1] == 2:
            return values[:, 1]
    return values.reshape(-1)


def _frame_indices(frame_count: int, count: int) -> np.ndarray:
    if frame_count <= 1:
        return np.array([0], dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, count, dtype=int))


def _contact_sheet(
    video: Path,
    output: Path,
    *,
    frame_count: int,
    sample_count: int,
    tile_size: int,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video))
    indices = _frame_indices(frame_count, sample_count)
    tiles: list[np.ndarray] = []
    landed_frames: list[int] = []
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        landed_frames.append(landed)
        frame = cv2.resize(frame, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (tile_size, 25), (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"f{landed}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    cap.release()
    if not tiles:
        return {"output": None, "requested_frames": indices.tolist(), "landed_frames": []}
    columns = 4
    rows = int(np.ceil(len(tiles) / columns))
    blank = np.full_like(tiles[0], 245)
    padded = tiles + [blank] * (rows * columns - len(tiles))
    sheet = np.vstack(
        [np.hstack(padded[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return {
        "output": str(output),
        "requested_frames": indices.tolist(),
        "landed_frames": landed_frames,
        "exact_seek_fraction": float(np.mean(np.asarray(landed_frames) == indices[: len(landed_frames)])),
    }


def audit_camera(
    dataset: Path,
    view: str,
    filename: str,
    output_dir: Path,
    *,
    samples: int,
    tile_size: int,
) -> dict[str, object]:
    view_dir = dataset / view
    video = view_dir / filename
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    timestamps = _timestamp_vector(view_dir)
    timing: dict[str, object] | None = None
    if timestamps is not None and len(timestamps) >= 2:
        differences = np.diff(timestamps)
        positive = differences[differences > 0]
        median = float(np.median(positive)) if len(positive) else None
        estimated_missing = (
            int(np.sum(np.maximum(np.rint(positive / median).astype(int) - 1, 0)))
            if median and median > 0
            else None
        )
        timing = {
            "array_length": int(len(timestamps)),
            "median_step": median,
            "maximum_step": float(np.max(positive)) if len(positive) else None,
            "nonpositive_steps": int(np.sum(differences <= 0)),
            "estimated_missing_triggers": estimated_missing,
            "length_minus_video_frames": int(len(timestamps) - frame_count),
        }
    contact = _contact_sheet(
        video,
        output_dir / f"{dataset.name}_{view}_contact.jpg",
        frame_count=frame_count,
        sample_count=samples,
        tile_size=tile_size,
    )
    return {
        "video": str(video),
        "file_size_bytes": int(video.stat().st_size),
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps > 0 else None,
        "image_size": [width, height],
        "within_camera_timing": timing,
        "contact_sheet": contact,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "synchronization": {
            "source": "shared external frame trigger",
            "cross_camera_internal_timestamp_comparison": False,
            "join_key": "frame index",
        },
        "datasets": {},
    }
    for dataset in args.datasets:
        print(f"Auditing {dataset}", flush=True)
        cameras = {
            view: audit_camera(
                dataset,
                view,
                filename,
                args.output_dir,
                samples=args.samples,
                tile_size=args.tile_size,
            )
            for view, filename in CAMERAS.items()
        }
        counts = [camera["frame_count"] for camera in cameras.values()]
        report["datasets"][dataset.name] = {  # type: ignore[index]
            "path": str(dataset),
            "cameras": cameras,
            "frame_count_range": [int(min(counts)), int(max(counts))],
            "frame_count_span": int(max(counts) - min(counts)),
            "same_resolution": len({tuple(camera["image_size"]) for camera in cameras.values()}) == 1,
        }
    report_path = args.output_dir / "recording_audit_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
