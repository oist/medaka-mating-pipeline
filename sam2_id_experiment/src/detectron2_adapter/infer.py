from __future__ import annotations

from pathlib import Path

import numpy as np

from sam2_id_experiment.src.sam2_adapter.mask_io import MaskInstance, bbox_from_mask, write_mask_instances_h5


def run_detectron2_video_inference(
    *,
    video_path: str | Path,
    config_path: str | Path,
    model_path: str | Path,
    output_masks: str | Path,
    start_frame: int,
    end_frame: int,
    frame_stride: int = 1,
    score_thresh: float = 0.5,
    device: str = "cuda",
) -> dict:
    try:
        import cv2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed in this environment. Install Detectron2 "
            "for your CUDA/PyTorch platform, then rerun this script."
        ) from exc

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = str(device)
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    instances: list[MaskInstance] = []
    image_shape: tuple[int, int] | None = None
    try:
        for frame_idx in range(int(start_frame), int(end_frame) + 1, max(1, int(frame_stride))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            image_shape = frame.shape[:2]
            outputs = predictor(frame)
            pred = outputs["instances"].to("cpu")
            masks = pred.pred_masks.numpy() if pred.has("pred_masks") else np.zeros((0, *image_shape), dtype=bool)
            scores = pred.scores.numpy() if pred.has("scores") else np.ones((len(masks),), dtype=float)
            for instance_id, mask in enumerate(masks):
                mask = np.asarray(mask, dtype=bool)
                instances.append(MaskInstance(
                    frame=int(frame_idx),
                    instance_id=int(instance_id),
                    sam2_track_id=int(instance_id),
                    mask=mask,
                    bbox=bbox_from_mask(mask),
                    score=float(scores[instance_id]),
                ))
    finally:
        cap.release()

    write_mask_instances_h5(output_masks, instances, image_shape=image_shape)
    return {
        "runner": "detectron2",
        "output_masks": str(output_masks),
        "n_instances": len(instances),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_stride": int(frame_stride),
    }


def run_detectron2_image_dir_inference(
    *,
    image_dir: str | Path,
    config_path: str | Path,
    model_path: str | Path,
    output_masks: str | Path,
    start_frame: int,
    frame_stride: int = 1,
    score_thresh: float = 0.5,
    device: str = "cuda",
    output_shape: tuple[int, int] | None = None,
) -> dict:
    try:
        import cv2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed in this environment. Install Detectron2 "
            "for your CUDA/PyTorch platform, then rerun this script."
        ) from exc

    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(Path(image_dir).glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No JPG/PNG frames found in {image_dir}")

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = str(device)
    predictor = DefaultPredictor(cfg)

    instances: list[MaskInstance] = []
    image_shape: tuple[int, int] | None = None
    for local_idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        image_shape = frame.shape[:2]
        frame_idx = int(start_frame) + int(local_idx) * max(1, int(frame_stride))
        outputs = predictor(frame)
        pred = outputs["instances"].to("cpu")
        masks = pred.pred_masks.numpy() if pred.has("pred_masks") else np.zeros((0, *image_shape), dtype=bool)
        scores = pred.scores.numpy() if pred.has("scores") else np.ones((len(masks),), dtype=float)
        for instance_id, mask in enumerate(masks):
            mask = np.asarray(mask, dtype=bool)
            if output_shape is not None and tuple(mask.shape) != tuple(output_shape):
                mask = _resize_bool_mask(mask, output_shape)
            instances.append(MaskInstance(
                frame=int(frame_idx),
                instance_id=int(instance_id),
                sam2_track_id=int(instance_id),
                mask=mask,
                bbox=bbox_from_mask(mask),
                score=float(scores[instance_id]),
            ))

    write_mask_instances_h5(output_masks, instances, image_shape=output_shape or image_shape)
    return {
        "runner": "detectron2",
        "input_images": str(image_dir),
        "output_masks": str(output_masks),
        "n_instances": len(instances),
        "start_frame": int(start_frame),
        "frame_stride": int(frame_stride),
        "n_input_images": len(image_paths),
        "output_shape": list(output_shape) if output_shape is not None else list(image_shape) if image_shape else None,
    }


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)
