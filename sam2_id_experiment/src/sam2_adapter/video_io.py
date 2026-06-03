from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()

