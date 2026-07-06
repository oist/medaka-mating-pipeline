"""Phase-1b probe, stage 1 (run in .venv-detectron2).

Re-runs the EXACT Phase-1a 20-event set (probe_events.json), but builds crops the
PRODUCTION way and labels them, so the Phase-1a confound (crop-distribution mismatch +
no FP rejection) is removed:

  * link each raw Detectron2 mask to the concrete-tracklet points (point-in-mask, the
    ``33_build_segid_dataset`` linking) -> drop FP / merged masks (the QC the production
    seg-ID path does and Phase-1a skipped); the surviving mask carries a tracklet_id and
    its 3D-bind ``global_id`` (fish identity label);
  * build TWO crops per kept mask on the SAME mask, for an airtight A/B:
      - ``prod`` : mask bbox + 0.15 pad, resized to 128, UNMASKED  (== training framing)
      - ``p1a``  : raw 128px box around the centroid, background masked to 0 (Phase-1a)

Stage 2 (p1b_score.py, main .venv) runs the clean female model on both and compares the
female-vs-male within-frame P-spread -- the number Phase-1a reported as 0.029 (collapsed).

Run::
    .venv-detectron2/Scripts/python.exe runs/dual_line/phase1b/p1b_detect.py \
        --max-events 2 --max-frames 700   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
EVENTS_JSON = HERE.parent / "phase1a" / "probe_events.json"
VIDEOS = {
    "side1_cam": "Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/side1_cam/cam1_side.mp4",
    "side2_cam": "Z:/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/side2_cam/cam2_side.mp4",
}
REPO = HERE.parents[2]
MODEL = str(REPO / "outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/model_final.pth")
DET_CFG = str(REPO / "outputs/20251225_K_1_2_FCMC/side_detectron2_model_firstprompt/detectron2_config.yaml")
NFRAMES_TOTAL = 348600
CROP = 128


# ---- inlined pure-numpy/cv2 helpers (byte-identical to the production scripts; the
# ---- source modules import pandas, absent in .venv-detectron2) -----------------------
def link_frame(masks, points, gate_px=40.0):
    """masks: [(mask_bool, bbox)]; points: [(tracklet_id, x, y, gid)].
    Returns [(tracklet_id, gid, mask_idx, status)] for kept masks (== 33.link_frame)."""
    assigned = []
    for mi, (m, _bbox) in enumerate(masks):
        H, W = m.shape
        inside = [(tid, gid) for tid, x, y, gid in points
                  if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W and m[int(round(y)), int(round(x))]]
        if len(inside) == 1:
            assigned.append((inside[0][0], inside[0][1], mi, "point_inside"))
        elif len(inside) == 0:
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            cx, cy = xs.mean(), ys.mean()
            best, bestd = None, gate_px
            for tid, x, y, gid in points:
                d = float(np.hypot(x - cx, y - cy))
                if d < bestd:
                    best, bestd = (tid, gid), d
            if best is not None:
                assigned.append((best[0], best[1], mi, "near_centroid"))
        # len(inside) >= 2 -> merged mask: dropped (the QC Phase-1a lacked)
    return assigned


def square_crop_prod(img, bbox, size=CROP, pad_frac=0.15):
    """== 33._square_crop with mask=None: mask bbox + pad, resized to size, UNMASKED."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(w, h) * (1 + pad_frac) / 2) + 1
    H, W = img.shape[:2]
    xa, xb = max(0, int(cx - half)), min(W, int(cx + half))
    ya, yb = max(0, int(cy - half)), min(H, int(cy + half))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _crop_bounds(cx, cy, size, w, h):
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    return x0, y0, x0 + size, y0 + size


def _extract_padded(arr, bounds, fill=0):
    x0, y0, x1, y1 = bounds
    h, w = arr.shape[:2]
    out = np.full((y1 - y0, x1 - x0) + arr.shape[2:], fill, dtype=arr.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def crop_p1a(frame_bgr, mask):
    """== p1a_detect.crop_from_mask: raw 128px box around centroid, background masked to 0."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    masked = frame_bgr.copy()
    masked[~mask] = 0
    h, w = frame_bgr.shape[:2]
    return _extract_padded(masked, _crop_bounds(cx, cy, CROP, w, h)).astype(np.uint8)


def bbox_from_mask(m):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def frame_list(ev, bracket):
    nf = ev["n_frames"]
    cs = 1 if nf <= 200 else (2 if nf <= 500 else (4 if nf <= 1000 else 6))
    core = range(ev["start"], ev["end"] + 1, cs)
    pre = range(ev["start"] - bracket, ev["start"], 5)
    post = range(ev["end"] + 1, ev["end"] + bracket + 1, 5)
    fr = sorted({f for f in list(pre) + list(core) + list(post) if 0 <= f < NFRAMES_TOTAL})
    return np.array(fr, dtype=int)


def make_predictor(device, score_thresh):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(DET_CFG)
    cfg.MODEL.WEIGHTS = MODEL
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--points-dir", default=str(HERE), help="dir with points_<view>.json from p1b_prep_points.py")
    ap.add_argument("--max-events", type=int, default=0, help="0 = all events in probe_events.json")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames/view for smoke (0=all)")
    ap.add_argument("--bracket", type=int, default=100)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--gate-px", type=float, default=40.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--views", nargs="+", default=["side1_cam", "side2_cam"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads(EVENTS_JSON.read_text())
    if args.max_events:
        events = events[: args.max_events]
    ev_by_id = {e["event_id"]: e for e in events}
    print(f"[detect] {len(events)} events; "
          f"core frames={sum(e['n_frames'] for e in events)}; views={args.views}", flush=True)

    t_load = time.time()
    predictor = make_predictor(args.device, args.score_thresh)
    print(f"[detect] predictor ready in {time.time()-t_load:.0f}s", flush=True)

    for view in args.views:
        points = {int(k): [tuple(p) for p in v]
                  for k, v in json.loads((Path(args.points_dir) / f"points_{view}.json").read_text()).items()}
        cap = cv2.VideoCapture(VIDEOS[view])
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {VIDEOS[view]}")
        crops_prod, crops_p1a = [], []
        m_frame, m_event, m_tid, m_gid, m_cx, m_cy, m_area, m_score, m_status = ([] for _ in range(9))
        per_frame = []  # (event_id, frame, n_masks_raw, n_masks_linked)
        t0 = time.time()
        n_done = 0
        for e in events:
            fl = frame_list(e, args.bracket)
            if args.max_frames and n_done >= args.max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fl[0]))
            cur = int(fl[0])
            for target in fl:
                if args.max_frames and n_done >= args.max_frames:
                    break
                while cur < target:
                    cap.grab(); cur += 1
                ok, frame = cap.read(); cur += 1
                if not ok:
                    continue
                pred = predictor(frame)["instances"].to("cpu")
                pmasks = pred.pred_masks.numpy() if pred.has("pred_masks") else np.zeros((0,) + frame.shape[:2], bool)
                scores = pred.scores.numpy() if pred.has("scores") else np.ones(len(pmasks))
                masks = [(np.asarray(m, bool), bbox_from_mask(np.asarray(m, bool))) for m in pmasks]
                pts = points.get(int(target), [])
                assigned = link_frame(masks, pts, gate_px=args.gate_px)
                per_frame.append((e["event_id"], int(target), len(masks), len(assigned)))
                for tid, gid, mi, status in assigned:
                    m, bbox = masks[mi]
                    ys, xs = np.nonzero(m)
                    cprod = square_crop_prod(frame, bbox)
                    cp = crop_p1a(frame, m)
                    if cp is None:
                        continue
                    crops_prod.append(cprod); crops_p1a.append(cp)
                    m_frame.append(int(target)); m_event.append(int(e["event_id"]))
                    m_tid.append(int(tid)); m_gid.append(int(gid))
                    m_cx.append(float(xs.mean())); m_cy.append(float(ys.mean()))
                    m_area.append(int(len(xs))); m_score.append(float(scores[mi])); m_status.append(status)
                n_done += 1
        cap.release()
        fps = n_done / max(time.time() - t0, 1e-6)
        path = out / f"probe_crops_{view}.h5"
        with h5py.File(path, "w") as h:
            empty = np.zeros((0, CROP, CROP, 3), np.uint8)
            h.create_dataset("crops_prod", data=np.asarray(crops_prod, np.uint8) if crops_prod else empty,
                             compression="gzip", compression_opts=4)
            h.create_dataset("crops_p1a", data=np.asarray(crops_p1a, np.uint8) if crops_p1a else empty,
                             compression="gzip", compression_opts=4)
            h.create_dataset("frame", data=np.asarray(m_frame, np.int64))
            h.create_dataset("event_id", data=np.asarray(m_event, np.int64))
            h.create_dataset("tracklet_id", data=np.asarray(m_tid, np.int64))
            h.create_dataset("global_id", data=np.asarray(m_gid, np.int64))
            h.create_dataset("cx", data=np.asarray(m_cx, np.float32))
            h.create_dataset("cy", data=np.asarray(m_cy, np.float32))
            h.create_dataset("area", data=np.asarray(m_area, np.int64))
            h.create_dataset("score", data=np.asarray(m_score, np.float32))
            h.create_dataset("link_status", data=np.array(m_status, dtype=h5py.string_dtype()))
            h.create_dataset("per_frame", data=np.asarray(per_frame, np.int64) if per_frame else np.zeros((0, 4), np.int64))
            h.attrs["view"] = view
        print(f"[detect] {view}: {n_done} frames, {len(crops_prod)} linked crops, {fps:.1f} fps -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
