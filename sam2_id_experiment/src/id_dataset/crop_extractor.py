from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


def _crop_bounds(cx: float, cy: float, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size
    return x0, y0, x1, y1


def _extract_padded(arr: np.ndarray, bounds: tuple[int, int, int, int], fill: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    h, w = arr.shape[:2]
    out_shape = (y1 - y0, x1 - x0) + arr.shape[2:]
    out = np.full(out_shape, fill, dtype=arr.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = (
        arr[src_y0:src_y1, src_x0:src_x1]
    )
    return out


def _load_masks_by_key(mask_path: str | Path) -> dict[tuple[int, int], np.ndarray]:
    masks: dict[tuple[int, int], np.ndarray] = {}
    for inst in iter_mask_instances_h5(mask_path):
        tracklet_id = int(inst.sam2_track_id if inst.sam2_track_id >= 0 else inst.instance_id)
        masks[(int(inst.frame), tracklet_id)] = np.asarray(inst.mask, dtype=bool)
    return masks


def extract_id_crops(
    *,
    video_path: str | Path,
    mask_path: str | Path,
    tracklets_qc: pd.DataFrame,
    output_dir: str | Path,
    crop_size: int,
    max_crops_per_tracklet: int,
    statuses: set[str] | None = None,
    debug_first_n: int = 0,
) -> tuple[Path, Path]:
    import cv2
    import h5py

    statuses = statuses or {"clean_train"}
    rows = tracklets_qc[tracklets_qc["qc_status"].isin(statuses)].copy()
    if "qc_segment_id" in rows.columns:
        rows = rows[rows["qc_segment_id"].astype(int) >= 0].copy()
    if rows.empty:
        raise RuntimeError("No eligible tracklet rows for crop extraction.")

    rows.sort_values(["tracklet_id", "frame"], inplace=True)
    group_col = "qc_segment_id" if "qc_segment_id" in rows.columns else "tracklet_id"
    rows = (
        rows.groupby(group_col, group_keys=False)
        .head(int(max_crops_per_tracklet))
        .sort_values(["frame", "tracklet_id"])
        .reset_index(drop=True)
    )
    masks = _load_masks_by_key(mask_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_path = out_dir / "crops.h5"
    meta_path = out_dir / "crop_metadata.parquet"
    debug_dir = out_dir / "debug_crops"
    if debug_first_n > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    crops: list[np.ndarray] = []
    meta: list[dict] = []
    try:
        current_frame = -1
        frame_bgr = None
        for row in rows.to_dict("records"):
            frame_idx = int(row["frame"])
            tracklet_id = int(row["tracklet_id"])
            mask = masks.get((frame_idx, tracklet_id))
            if mask is None:
                continue
            if current_frame != frame_idx:
                if current_frame >= 0 and frame_idx == current_frame + 1:
                    ok, frame_bgr = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame_bgr = cap.read()
                if not ok:
                    continue
                current_frame = frame_idx

            assert frame_bgr is not None
            crop_id = len(crops)
            masked = frame_bgr.copy()
            masked[~mask] = 0
            bounds = _crop_bounds(row["centroid_x"], row["centroid_y"], crop_size, frame_bgr.shape[1], frame_bgr.shape[0])
            crop = _extract_padded(masked, bounds)
            local_mask = _extract_padded(mask.astype(np.uint8) * 255, bounds)
            crops.append(crop)
            meta.append({
                "crop_id": crop_id,
                "frame": frame_idx,
                "tracklet_id": tracklet_id,
                "qc_segment_id": int(row.get("qc_segment_id", -1)),
                "sam2_track_id": int(row.get("sam2_track_id", tracklet_id)),
                "bbox_x0": int(row["bbox_x0"]),
                "bbox_y0": int(row["bbox_y0"]),
                "bbox_x1": int(row["bbox_x1"]),
                "bbox_y1": int(row["bbox_y1"]),
                "centroid_x": float(row["centroid_x"]),
                "centroid_y": float(row["centroid_y"]),
                "area": int(row["area"]),
                "qc_status": str(row["qc_status"]),
            })
            if crop_id < debug_first_n:
                cv2.imwrite(str(debug_dir / f"crop_{crop_id:06d}.png"), crop)
                cv2.imwrite(str(debug_dir / f"mask_{crop_id:06d}.png"), local_mask)
    finally:
        cap.release()

    if not crops:
        raise RuntimeError("No crops were extracted; check frame indices and mask keys.")

    with h5py.File(crops_path, "w") as h5:
        h5.create_dataset("crops", data=np.stack(crops, axis=0), compression="gzip")
    pd.DataFrame(meta).to_parquet(meta_path, index=False)
    return crops_path, meta_path
