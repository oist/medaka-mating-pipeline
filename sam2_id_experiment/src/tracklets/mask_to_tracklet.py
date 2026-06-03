from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    bbox_from_mask,
    iter_mask_instances_h5,
)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    return (float(xs.mean()), float(ys.mean()))


def _contour_json(mask: np.ndarray) -> str:
    try:
        import cv2
    except ImportError:
        return "[]"

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    contours, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "[]"
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(int).tolist()
    return json.dumps(pts, separators=(",", ":"))


def instance_to_row(inst: MaskInstance) -> dict:
    mask = np.asarray(inst.mask, dtype=bool)
    area = int(mask.sum())
    bbox = inst.bbox if inst.bbox != (0, 0, 0, 0) else bbox_from_mask(mask)
    cx, cy = _centroid(mask)
    tracklet_id = int(inst.sam2_track_id if inst.sam2_track_id >= 0 else inst.instance_id)
    return {
        "tracklet_id": tracklet_id,
        "sam2_track_id": int(inst.sam2_track_id),
        "instance_id": int(inst.instance_id),
        "frame": int(inst.frame),
        "centroid_x": cx,
        "centroid_y": cy,
        "bbox_x0": int(bbox[0]),
        "bbox_y0": int(bbox[1]),
        "bbox_x1": int(bbox[2]),
        "bbox_y1": int(bbox[3]),
        "area": area,
        "confidence": float(inst.score),
        "contour_json": _contour_json(mask),
    }


def masks_to_instance_dataframe(mask_path: str | Path) -> pd.DataFrame:
    rows = [instance_to_row(inst) for inst in iter_mask_instances_h5(mask_path)]
    columns = [
        "tracklet_id", "sam2_track_id", "instance_id", "frame",
        "centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "area", "confidence", "contour_json",
    ]
    return pd.DataFrame(rows, columns=columns)


def add_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["centroid_speed_px_per_frame"] = pd.Series(dtype=float)
        return out
    out = df.sort_values(["tracklet_id", "frame"]).copy()
    out["prev_frame"] = out.groupby("tracklet_id")["frame"].shift(1)
    out["prev_x"] = out.groupby("tracklet_id")["centroid_x"].shift(1)
    out["prev_y"] = out.groupby("tracklet_id")["centroid_y"].shift(1)
    dt = out["frame"] - out["prev_frame"]
    dist = np.hypot(out["centroid_x"] - out["prev_x"], out["centroid_y"] - out["prev_y"])
    out["centroid_speed_px_per_frame"] = np.where(dt > 0, dist / dt, np.nan)
    return out.drop(columns=["prev_frame", "prev_x", "prev_y"])


def summarize_tracklets(instance_df: pd.DataFrame) -> pd.DataFrame:
    df = add_speed_columns(instance_df)
    if df.empty:
        return pd.DataFrame(columns=[
            "tracklet_id", "sam2_track_id", "frame_start", "frame_end", "n_frames",
            "mean_area", "area_cv", "mean_speed", "max_speed", "mean_confidence",
            "min_confidence", "frame_step", "n_missing_frames",
        ])

    rows: list[dict] = []
    for tracklet_id, group in df.groupby("tracklet_id", sort=True):
        frames = group["frame"].astype(int)
        frame_start = int(frames.min())
        frame_end = int(frames.max())
        n_frames = int(group.shape[0])
        frame_step = _infer_frame_step(frames)
        expected = ((frame_end - frame_start) // frame_step) + 1
        mean_area = float(group["area"].mean())
        area_std = float(group["area"].std(ddof=0))
        area_cv = area_std / mean_area if mean_area > 0 else math.inf
        speed = group["centroid_speed_px_per_frame"].dropna()
        rows.append({
            "tracklet_id": int(tracklet_id),
            "sam2_track_id": int(group["sam2_track_id"].iloc[0]),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "n_frames": n_frames,
            "mean_area": mean_area,
            "area_cv": float(area_cv),
            "mean_speed": float(speed.mean()) if not speed.empty else 0.0,
            "max_speed": float(speed.max()) if not speed.empty else 0.0,
            "mean_confidence": float(group["confidence"].mean()),
            "min_confidence": float(group["confidence"].min()),
            "frame_step": frame_step,
            "n_missing_frames": int(expected - n_frames),
        })
    return pd.DataFrame(rows)


def _infer_frame_step(frames: pd.Series) -> int:
    unique_frames = pd.Series(frames.astype(int).unique()).sort_values()
    diffs = unique_frames.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return int(diffs.value_counts().sort_values(ascending=False).index[0])
