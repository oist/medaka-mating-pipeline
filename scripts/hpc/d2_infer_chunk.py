"""Standalone Detectron2 chunk inference for the saion Line-B fan-out.

Self-contained: depends ONLY on the `detectron.2/0.6` venv (detectron2, cv2, h5py,
numpy) + the co-located ``mask_io.py``. It deliberately does NOT import the medaka
repo / ``sam2_id_experiment`` / ``scripts.common.local_paths`` (that chain needs
``tomllib``, i.e. py3.11+, while the saion module venv is py3.10). Output masks .h5
is byte-compatible with ``mask_io.iter_mask_instances_h5`` / ``33_build_segid_dataset``.

Masks are streamed to the .h5 via a generator (a 40k-frame chunk of full-frame masks
cannot sit in RAM), so memory stays O(1) per frame.

Run::
    python d2_infer_chunk.py --video <mp4> --model <model_final.pth> \
        --config <detectron2_config.yaml> --start S --end E [--stride 1] \
        --score-thresh 0.5 --output <view>.masks_S-E.h5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_io import MaskInstance, bbox_from_mask, write_mask_instances_h5  # noqa: E402


def make_predictor(config: str, model: str, score_thresh: float, device: str):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def instance_stream(cap, predictor, start: int, end: int, stride: int, progress_every: int, state: dict):
    """Yield MaskInstance for frames [start, end] inclusive, processing every `stride`-th.

    Sequential decode (seek once to start, then read forward) — labels each mask with its
    true frame number `f`. Lazily yields so write_mask_instances_h5 streams to disk
    (O(1) memory per frame)."""
    start, end, stride = int(start), int(end), max(1, int(stride))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    t0 = time.time()
    n = 0
    f = start
    while f <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start) % stride == 0:
            state["shape"] = frame.shape[:2]
            out = predictor(frame)["instances"].to("cpu")
            masks = out.pred_masks.numpy() if out.has("pred_masks") else np.zeros((0,) + frame.shape[:2], bool)
            scores = out.scores.numpy() if out.has("scores") else np.ones(len(masks))
            for i, m in enumerate(masks):
                m = np.asarray(m, bool)
                yield MaskInstance(frame=int(f), instance_id=int(i), sam2_track_id=int(i),
                                   mask=m, bbox=bbox_from_mask(m), score=float(scores[i]))
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[d2] {n} processed, {n / max(time.time() - t0, 1e-6):.1f} fps, at frame {f}", flush=True)
        f += 1
    state["n_frames"] = n
    state["fps"] = n / max(time.time() - t0, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    t_load = time.time()
    predictor = make_predictor(args.config, args.model, args.score_thresh, args.device)
    print(f"[d2] predictor ready in {time.time() - t_load:.0f}s; device={args.device}", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    state: dict = {"shape": shape, "n_frames": 0, "fps": 0.0}
    try:
        write_mask_instances_h5(
            tmp_path,
            instance_stream(cap, predictor, args.start, args.end, max(1, args.stride), args.progress_every, state),
            image_shape=state["shape"],
        )
    finally:
        cap.release()
    tmp_path.replace(out_path)   # atomic: a half-written chunk never looks complete
    print(f"[d2] DONE {state['n_frames']} frames, {state['fps']:.1f} fps -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
