"""Step 2 (Part A): per-frame appearance female-slot across the timeline.

For sampled frames (stride S) in [start,end], build production-framed crops from the saion
masks (RANDOM-ACCESS: read only that frame's mask group -- no full-chunk decompress) + the
local video, run the clean female model, and assign the female's geometry slot per frame as
the multiview-averaged argmax P(female). The female's slot == the 3D-bind global_id she
occupies that frame (global_id k <-> fish_id ID_{k+1} <-> geometry_slot k), so this directly
exposes the within/between-window 3D-bind swaps at frame resolution -> resolves the window-
anchor `low` spans (Part B).

Output: per_frame_female_<start>_<end>.csv (frame, female_slot, p_slot0/1/2, margin, n_views).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "phase0"))
from _segid_common import FEMALE_MODEL, device, forward, load_classifier  # noqa: E402

MASKS_Z = Path(r"Z:\StephensU\medaka\detectron2_output\20251225_K_1_2_FCMC")
VIDEOS = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\b07af347-293e-4fed-a208-e52ffcd113b6\scratchpad\videos")
TB = Path(r"Z:\StephensU\medaka\runs\udmt_validation\tracklet_bind_extend_smooth")
CHUNKS = [(0, 43574), (43575, 87149), (87150, 130724), (130725, 174299),
          (174300, 217874), (217875, 261449), (261450, 305024), (305025, 348599)]
VIEWS = [("side1_cam", "cam1_side"), ("side2_cam", "cam2_side")]
CROP, GATE = 128, 40.0


def square_crop(img, bbox, size=CROP, pad_frac=0.15):
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


def link_gid(masks, points, gate=GATE):
    """masks: [(mask_bool, bbox)]; points: [(x,y,gid)]. -> [(gid, mask_idx)] (33.link_frame)."""
    out = []
    for mi, (m, _b) in enumerate(masks):
        H, W = m.shape
        inside = [g for x, y, g in points if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W and m[int(round(y)), int(round(x))]]
        if len(inside) == 1:
            out.append((inside[0], mi))
        elif len(inside) == 0:
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            cx, cy = xs.mean(), ys.mean()
            best, bd = None, gate
            for x, y, g in points:
                d = float(np.hypot(x - cx, y - cy))
                if d < bd:
                    best, bd = g, d
            if best is not None:
                out.append((best, mi))
        # >=2 inside -> merged, dropped
    return out


def load_points(view, lo, hi):
    df = pd.read_csv(TB / f"{view}.tracklet_bound.csv", usecols=["frame", "x", "y", "global_id"])
    df = df[(df.frame >= lo) & (df.frame <= hi) & df.global_id.isin([0, 1, 2])]
    pts: dict[int, list] = {}
    for f, x, y, g in zip(df.frame, df.x, df.y, df.global_id):
        pts.setdefault(int(f), []).append((float(x), float(y), int(g)))
    return pts


def read_frame_masks(h5, f):
    key = f"frame_{f:06d}"
    if key not in h5:
        return []
    g = h5[key]
    return [(np.asarray(g[ik]["mask"][...], bool), tuple(int(v) for v in g[ik]["bbox"][...]))
            for ik in g if ik.startswith("instance_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=348599)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "step2"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    clf = load_classifier(FEMALE_MODEL, dev)

    # accumulate p_female per (frame, view, gid)
    recs = {"frame": [], "view": [], "gid": [], "p": []}
    BATCH = 4096
    for view, cam in VIEWS:
        t0 = time.time()
        pts = load_points(view, args.start, args.end)
        chunks = {(cs, ce): h5py.File(MASKS_Z / f"{view}.masks_{cs}-{ce}.h5", "r") for cs, ce in CHUNKS}

        def chunk_h5(f):
            for (cs, ce), h in chunks.items():
                if cs <= f <= ce:
                    return h
            return None

        cap = cv2.VideoCapture(str(VIDEOS / view / f"{cam}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        buf_crops, buf_meta = [], []

        def flush():
            if not buf_crops:
                return
            p = forward(clf, np.asarray(buf_crops, np.uint8), dev, head=True)
            for (fr, g), pf in zip(buf_meta, p):
                recs["frame"].append(fr); recs["view"].append(view); recs["gid"].append(g); recs["p"].append(float(pf))
            buf_crops.clear(); buf_meta.clear()

        f = args.start
        n_proc = 0
        while f <= args.end:
            sampled = (f - args.start) % args.stride == 0
            if sampled:
                ok, img = cap.read()
            else:
                ok = cap.grab(); img = None
            if not ok:
                break
            if sampled and img is not None and f in pts:
                h5 = chunk_h5(f)
                if h5 is not None:
                    masks = read_frame_masks(h5, f)
                    if masks:
                        for gid, mi in link_gid(masks, pts[f]):
                            buf_crops.append(square_crop(img, masks[mi][1])); buf_meta.append((f, gid))
                            if len(buf_crops) >= BATCH:
                                flush()
                n_proc += 1
            f += 1
        flush()
        cap.release()
        for h in chunks.values():
            h.close()
        print(f"[{view}] processed {n_proc} sampled frames in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    # multiview mean p_female per (frame, gid), then per-frame female slot = argmax
    pivot = df.groupby(["frame", "gid"]).p.mean().reset_index()
    wide = pivot.pivot(index="frame", columns="gid", values="p")
    for g in (0, 1, 2):
        if g not in wide.columns:
            wide[g] = np.nan
    wide = wide[[0, 1, 2]]
    arr = wide.to_numpy()
    valid = ~np.all(np.isnan(arr), axis=1)
    filled = np.where(np.isnan(arr), -1.0, arr)
    female_slot = np.where(valid, filled.argmax(axis=1), -1)
    srt = np.sort(np.where(np.isnan(arr), -1.0, arr), axis=1)
    margin = np.where(valid & (np.sum(~np.isnan(arr), axis=1) >= 2), srt[:, -1] - srt[:, -2], np.nan)
    res = pd.DataFrame({"frame": wide.index, "female_slot": female_slot,
                        "p_slot0": arr[:, 0], "p_slot1": arr[:, 1], "p_slot2": arr[:, 2],
                        "margin": margin, "n_slots": np.sum(~np.isnan(arr), axis=1)})
    fn = out / f"per_frame_female_{args.start}_{args.end}.csv"
    res.to_csv(fn, index=False)
    n = len(res)
    print(f"\nwrote {fn}  ({n} sampled frames with >=1 fish)")
    print("female_slot counts:", {int(k): int(v) for k, v in res.female_slot.value_counts().items()})
    conf = res[res.margin > 0.3]
    print(f"frames with >=2 slots: {int((res.n_slots>=2).sum())}  | confident (margin>0.3): {len(conf)} "
          f"({len(conf)/max(n,1):.1%})  median margin {res.margin.median():.3f}")


if __name__ == "__main__":
    main()
