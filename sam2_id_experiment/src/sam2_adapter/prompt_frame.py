from __future__ import annotations

import json
from pathlib import Path


def resolve_prompt_frame_index(*, start_frame: int, frame: int | None, local_frame: int | None) -> int:
    if frame is not None and local_frame is not None:
        raise ValueError("Pass either frame or local_frame, not both.")
    if local_frame is not None:
        return int(start_frame) + int(local_frame)
    if frame is not None:
        return int(frame)
    return int(start_frame)


def export_prompt_frame(
    *,
    video_path: Path,
    output_path: Path,
    frame_index: int,
    start_frame: int,
) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    finally:
        cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    height, width = frame.shape[:2]
    meta = {
        "video_path": str(video_path),
        "output_png": str(output_path),
        "absolute_frame": int(frame_index),
        "local_frame": int(frame_index - start_frame),
        "width": int(width),
        "height": int(height),
        "yaml_prompt_template": [
            {
                "obj_id": 0,
                "frame": int(frame_index - start_frame),
                "points": [[0, 0]],
                "labels": [1],
            }
        ],
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta

