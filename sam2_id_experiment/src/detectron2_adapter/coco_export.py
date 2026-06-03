from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


def export_sam2_masks_to_coco(
    *,
    video_path: str | Path,
    mask_path: str | Path,
    tracklets_qc_path: str | Path,
    output_dir: str | Path,
    statuses: Iterable[str] = ("clean_train",),
    max_images: int | None = None,
    frame_stride: int = 1,
    image_quality: int = 92,
) -> dict:
    import cv2

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "annotations.json"

    allowed_statuses = {str(status) for status in statuses}
    qc = pd.read_parquet(tracklets_qc_path)
    qc = qc[qc["qc_status"].astype(str).isin(allowed_statuses)].copy()
    if qc.empty:
        raise RuntimeError(f"No QC rows matched statuses: {sorted(allowed_statuses)}")

    if frame_stride > 1:
        ordered_frames = sorted(qc["frame"].astype(int).unique())
        kept_frames = set(ordered_frames[::frame_stride])
        qc = qc[qc["frame"].astype(int).isin(kept_frames)].copy()

    if max_images is not None:
        ordered_frames = sorted(qc["frame"].astype(int).unique())[: int(max_images)]
        qc = qc[qc["frame"].astype(int).isin(set(ordered_frames))].copy()

    qc_by_key = {
        (int(row["frame"]), int(row["tracklet_id"])): row
        for row in qc.to_dict("records")
    }
    masks_by_frame: dict[int, list] = {}
    for inst in iter_mask_instances_h5(mask_path):
        key = (int(inst.frame), int(inst.sam2_track_id))
        if key in qc_by_key:
            masks_by_frame.setdefault(int(inst.frame), []).append(inst)

    frames = sorted(masks_by_frame)
    if not frames:
        raise RuntimeError("No masks remained after matching QC rows to mask instances.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    try:
        for image_id, frame in enumerate(frames, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if not ok:
                continue
            height, width = image.shape[:2]
            file_name = f"frame_{int(frame):06d}.jpg"
            cv2.imwrite(str(image_dir / file_name), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(image_quality)])
            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
                "frame": int(frame),
            })
            for inst in masks_by_frame[frame]:
                row = qc_by_key[(int(inst.frame), int(inst.sam2_track_id))]
                polygons = mask_to_coco_polygons(inst.mask)
                if not polygons:
                    continue
                x0, y0, x1, y1 = [int(v) for v in inst.bbox]
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": polygons,
                    "area": int(np.asarray(inst.mask, dtype=bool).sum()),
                    "bbox": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    "iscrowd": 0,
                    "tracklet_id": int(row["tracklet_id"]),
                    "sam2_track_id": int(row["sam2_track_id"]),
                    "qc_status": str(row["qc_status"]),
                    "source": "sam2_qc_pseudolabel",
                })
                ann_id += 1
    finally:
        cap.release()

    coco = {
        "info": {
            "description": "SAM2 QC pseudo-labels for medaka instance segmentation",
            "version": "0.1",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "medaka", "supercategory": "fish"}],
        "images": images,
        "annotations": annotations,
    }
    annotation_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    summary = {
        "annotation_path": str(annotation_path),
        "image_dir": str(image_dir),
        "n_images": len(images),
        "n_annotations": len(annotations),
        "statuses": sorted(allowed_statuses),
    }
    (output_dir / "coco_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def mask_to_coco_polygons(mask: np.ndarray) -> list[list[float]]:
    import cv2

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        flat = contour.astype(float).reshape(-1).tolist()
        if len(flat) >= 6:
            polygons.append(flat)
    return polygons
