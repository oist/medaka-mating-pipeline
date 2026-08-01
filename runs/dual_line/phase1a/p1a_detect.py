"""Phase-1a probe, stage 1 (run in .venv-detectron2).

Raw Detectron2 (NO purity filter) on the side-view frames of the largest 3D-`review`
FEMALE-involved hard-core events, building the 128x128 masked crops in-loop (frame + mask
are both in hand, so no full-frame storage / no double video I/O). Writes one crops .h5 per
view containing EVERY detected mask's crop + per-mask metadata as h5 arrays (no pandas here).

Stage 2 (p1a_score.py, main .venv) loads these, runs the female classifier, and fills the
decisive 2x2 cell Phase 0 could not measure: at a 3D-uncertain F-involved event, does
appearance confidently + continuously track the female through the merge?

Run::
    .venv-detectron2/Scripts/python.exe runs/dual_line/phase1a/p1a_detect.py \
        --max-events 2 --min-frames 100 --max-frames 170   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO))


# inlined from sam2_id_experiment/src/id_dataset/crop_extractor.py (that module imports
# pandas at top level, which is absent in .venv-detectron2). Byte-identical behaviour.
def _crop_bounds(cx: float, cy: float, size: int, width: int, height: int):
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    return x0, y0, x0 + size, y0 + size


def _extract_padded(arr: np.ndarray, bounds, fill: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    h, w = arr.shape[:2]
    out = np.full((y1 - y0, x1 - x0) + arr.shape[2:], fill, dtype=arr.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out

EVENTS_CSV = "Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind/hardcore_events_resolved.csv"
VIDEOS = {
    "side1_cam": "Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/side1_cam/cam1_side.mp4",
    "side2_cam": "Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/side2_cam/cam2_side.mp4",
}
MODEL = str(REPO / "outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/model_final.pth")
DET_CFG = str(REPO / "outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/detectron2_config.yaml")
NFRAMES_TOTAL = 348600
CROP = 128


def select_events(args) -> list[dict]:
    rows = []
    with open(EVENTS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["verdict"] != "review" or "ID_2" not in r["pair"]:
                continue
            nf = int(r["n_frames"])
            if args.min_frames and nf < args.min_frames:
                continue
            if args.max_frames and nf > args.max_frames:
                continue
            rows.append({"pair": r["pair"], "start": int(r["start"]), "end": int(r["end"]),
                         "n_frames": nf, "min_mm": float(r["min_mm"]), "reason": r["reason"]})
    rows.sort(key=lambda d: -d["n_frames"])
    return rows[: args.max_events]


def frame_list(ev: dict, bracket: int) -> tuple[np.ndarray, int, int]:
    nf = ev["n_frames"]
    cs = 1 if nf <= 200 else (2 if nf <= 500 else (4 if nf <= 1000 else 6))
    core = range(ev["start"], ev["end"] + 1, cs)
    pre = range(ev["start"] - bracket, ev["start"], 5)
    post = range(ev["end"] + 1, ev["end"] + bracket + 1, 5)
    fr = sorted({f for f in list(pre) + list(core) + list(post) if 0 <= f < NFRAMES_TOTAL})
    return np.array(fr, dtype=int), ev["start"], ev["end"]


def make_predictor(device, score_thresh):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(DET_CFG)
    cfg.MODEL.WEIGHTS = MODEL
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def crop_from_mask(frame_bgr: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    masked = frame_bgr.copy()
    masked[~mask] = 0
    h, w = frame_bgr.shape[:2]
    crop = _extract_padded(masked, _crop_bounds(cx, cy, CROP, w, h))
    return crop.astype(np.uint8), cx, cy, int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--max-events", type=int, default=25)
    ap.add_argument("--min-frames", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--bracket", type=int, default=100)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--views", nargs="+", default=["side1_cam", "side2_cam"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = select_events(args)
    for i, e in enumerate(events):
        e["event_id"] = i
    (out / "probe_events.json").write_text(json.dumps(events, indent=2))
    print(f"[detect] {len(events)} events; total core frames="
          f"{sum(e['n_frames'] for e in events)}; views={args.views}", flush=True)

    t_load = time.time()
    predictor = make_predictor(args.device, args.score_thresh)
    print(f"[detect] predictor ready in {time.time()-t_load:.0f}s (model load)", flush=True)

    for view in args.views:
        cap = cv2.VideoCapture(VIDEOS[view])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {VIDEOS[view]}")
        crops, meta_frame, meta_event, meta_cx, meta_cy, meta_area, meta_score = ([] for _ in range(7))
        per_frame_counts = {}  # (event_id, frame) -> n_masks
        t0 = time.time()
        n_frames_done = 0
        for e in events:
            fl, s, en = frame_list(e, args.bracket)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fl[0]))
            cur = int(fl[0])
            want = set(fl.tolist())
            for target in fl:
                # sequential read; only re-seek if we jumped (stride gaps)
                while cur < target:
                    cap.grab(); cur += 1
                ok, frame = cap.read(); cur += 1
                if not ok:
                    continue
                pred = predictor(frame)["instances"].to("cpu")
                masks = pred.pred_masks.numpy() if pred.has("pred_masks") else np.zeros((0,) + frame.shape[:2], bool)
                scores = pred.scores.numpy() if pred.has("scores") else np.ones(len(masks))
                per_frame_counts[(e["event_id"], int(target))] = int(len(masks))
                for mi, m in enumerate(masks):
                    res = crop_from_mask(frame, np.asarray(m, bool))
                    if res is None:
                        continue
                    cr, cx, cy, area = res
                    crops.append(cr); meta_frame.append(int(target)); meta_event.append(e["event_id"])
                    meta_cx.append(cx); meta_cy.append(cy); meta_area.append(area); meta_score.append(float(scores[mi]))
                n_frames_done += 1
        cap.release()
        fps = n_frames_done / max(time.time() - t0, 1e-6)
        path = out / f"probe_crops_{view}.h5"
        with h5py.File(path, "w") as h:
            h.create_dataset("crops", data=np.asarray(crops, np.uint8) if crops else np.zeros((0, CROP, CROP, 3), np.uint8),
                             compression="gzip", compression_opts=4)
            h.create_dataset("frame", data=np.asarray(meta_frame, np.int64))
            h.create_dataset("event_id", data=np.asarray(meta_event, np.int64))
            h.create_dataset("cx", data=np.asarray(meta_cx, np.float32))
            h.create_dataset("cy", data=np.asarray(meta_cy, np.float32))
            h.create_dataset("area", data=np.asarray(meta_area, np.int64))
            h.create_dataset("score", data=np.asarray(meta_score, np.float32))
            pf = np.array([[ev, fr, n] for (ev, fr), n in per_frame_counts.items()], np.int64)
            h.create_dataset("per_frame_counts", data=pf if len(pf) else np.zeros((0, 3), np.int64))
            h.attrs["view"] = view
        print(f"[detect] {view}: {n_frames_done} frames, {len(crops)} masks, {fps:.1f} fps -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
