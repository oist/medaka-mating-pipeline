"""Step 3 (Part A-top): per-frame TOP P(M2) per geometry slot across the whole timeline.

Runs the top-trained MALE model (model_top_male; head -> P(M2)) on TOP crops, whole timeline, stride
10. Output tm_slot0/1/2 = top P(M2) per geometry slot, fused with the side per_frame_male in step3_fuse
to recover the male-low (close-contact) frames where the top view is clean but the sides merged.
(Only the male model is needed: the female slot stays anchored by the strong side resolution.)
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
import step2_per_frame_female as s2  # noqa: E402
from _segid_common import device, forward, load_classifier  # noqa: E402

TOP_MALE_MODEL = Path("Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_top_male/model.pt")
TOP_VIDEO = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\9d685e6e-fa9d-44ed-bbfb-964240c54875\scratchpad\videos\top_cam\cam0_top.mp4")
VIEW = "top_cam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=348599)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "step2"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    clf = load_classifier(TOP_MALE_MODEL, dev)

    recs = {"frame": [], "gid": [], "p": []}
    BATCH = 4096
    t0 = time.time()
    pts = s2.load_points(VIEW, args.start, args.end)
    chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{VIEW}.masks_{cs}-{ce}.h5", "r")
              for cs, ce in s2.CHUNKS if not (ce < args.start or cs > args.end)}

    def chunk_h5(f):
        for (cs, ce), h in chunks.items():
            if cs <= f <= ce:
                return h
        return None

    cap = cv2.VideoCapture(str(TOP_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    buf_crops, buf_meta = [], []

    def flush():
        if not buf_crops:
            return
        p = forward(clf, np.asarray(buf_crops, np.uint8), dev, head=True)
        for (fr, g), pf in zip(buf_meta, p):
            recs["frame"].append(fr); recs["gid"].append(g); recs["p"].append(float(pf))
        buf_crops.clear(); buf_meta.clear()

    f = args.start
    n_proc = 0
    while f <= args.end:
        sampled = (f - args.start) % args.stride == 0
        ok, img = (cap.read() if sampled else (cap.grab(), None))
        if not ok:
            break
        if sampled and img is not None and f in pts:
            h5 = chunk_h5(f)
            if h5 is not None:
                masks = s2.read_frame_masks(h5, f)
                if masks:
                    for gid, mi in s2.link_gid(masks, pts[f]):
                        buf_crops.append(s2.square_crop(img, masks[mi][1])); buf_meta.append((f, gid))
                        if len(buf_crops) >= BATCH:
                            flush()
            n_proc += 1
        f += 1
    flush()
    cap.release()
    for h in chunks.values():
        h.close()
    print(f"[{VIEW}] processed {n_proc} sampled frames in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    pivot = df.groupby(["frame", "gid"]).p.mean().reset_index()
    wide = pivot.pivot(index="frame", columns="gid", values="p")
    for g in (0, 1, 2):
        if g not in wide.columns:
            wide[g] = np.nan
    wide = wide[[0, 1, 2]]
    arr = wide.to_numpy()
    res = pd.DataFrame({"frame": wide.index,
                        "tm_slot0": arr[:, 0], "tm_slot1": arr[:, 1], "tm_slot2": arr[:, 2],
                        "n_slots": np.sum(~np.isnan(arr), axis=1)})
    fn = out / f"per_frame_top_male_{args.start}_{args.end}.csv"
    res.to_csv(fn, index=False)
    print(f"\nwrote {fn}  ({len(res)} sampled frames)")


if __name__ == "__main__":
    main()
