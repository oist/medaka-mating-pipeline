"""Extract fish crops from a video given per-frame (x, y) keypoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class CropSet:
    """Container for extracted crops and their row-aligned index table.

    `index` has one row per image in `images`, in the same order. Columns:
        orig_row   — index into the caller's input df
        frame, track, fish_id, x, y — echoed from the input
        row        — 0..N-1, matches the first axis of `images`
    """

    images: np.ndarray  # (N, size, size) uint8, grayscale
    index: pd.DataFrame


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """
    Seek to `frame_idx` and return a grayscale frame, or None on failure.

    We issue an explicit set() on every call; cv2 backends differ in whether
    sequential reads advance the cursor, and our df is not guaranteed to be
    contiguous or in frame order.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _crop_with_padding(
    frame: np.ndarray, cx: float, cy: float, size: int
) -> np.ndarray:
    """Return a `size x size` crop centered on (cx, cy), zero-padded at edges."""
    h, w = frame.shape[:2]
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)

    out = np.zeros((size, size), dtype=np.uint8)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    out[dy0:dy1, dx0:dx1] = frame[sy0:sy1, sx0:sx1]
    return out


def extract_crops_from_video(
    video_path: str | Path,
    df: pd.DataFrame,
    size: int = 96,
    frame_offset: int = 0,
) -> CropSet:
    """
    Extract one crop per `(frame, track)` row from `video_path`.

    Args:
        video_path: Path to an mp4 readable by OpenCV.
        df: DataFrame with at least columns `frame`, `track`, `x`, `y`. A
            `fish_id` column is preserved if present.
        size: Square crop side length in pixels.
        frame_offset: Added to `df['frame']` before seeking. Use when the df
            frame indices are clip-relative but the video is a full session.

    Returns:
        A `CropSet`: `images[i]` is the grayscale crop for row `i` of `index`.
        Rows with NaN (x, y) or unreadable frames produce all-zero crops and
        are still kept (the caller decides whether to mask them).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    keep_cols = [c for c in ("frame", "track", "fish_id", "x", "y") if c in df.columns]
    # Preserve the caller's original row index as `orig_row` so downstream
    # code can write results back to the right df rows regardless of how the
    # input was indexed.
    work = df[keep_cols].copy()
    work["orig_row"] = np.arange(len(work), dtype=int)
    work = work.dropna(subset=["frame", "x", "y"]).reset_index(drop=True)
    work["frame"] = work["frame"].astype(int)

    n = len(work)
    images = np.zeros((n, size, size), dtype=np.uint8)
    cache_frame_idx: int | None = None
    cache_gray: np.ndarray | None = None

    # Sort by frame for efficient sequential seeks, but remember the target
    # slot in `images` via the post-reset row position.
    order = work["frame"].to_numpy().argsort(kind="stable")
    try:
        for pos in order:
            dst = int(pos)
            frame_idx = int(work.at[dst, "frame"]) + frame_offset
            if cache_frame_idx != frame_idx:
                gray = _read_frame(cap, frame_idx)
                cache_frame_idx = frame_idx
                cache_gray = gray
            gray = cache_gray
            if gray is None:
                continue
            images[dst] = _crop_with_padding(
                gray, float(work.at[dst, "x"]), float(work.at[dst, "y"]), size
            )
    finally:
        cap.release()

    work["row"] = np.arange(n, dtype=int)
    return CropSet(images=images, index=work)
