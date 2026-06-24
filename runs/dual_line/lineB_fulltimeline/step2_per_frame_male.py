"""Step 2 male (Part A): per-frame P(M_beta) per geometry slot across the whole timeline.

Identical fast-path to step2_per_frame_female (random-access masks + local video scan), but runs
the swap-corrected MALE classifier (model_male_clean, classes [gid_0, gid_2]; head -> P(gid_2) =
P(M_beta)). Outputs pb_slot0/1/2 = multiview-mean P(M_beta) for each geometry slot. The female slot
is resolved separately (step2 female); Part B uses pb only on the 2 NON-female slots to assign
M_alpha / M_beta and resolve male swaps.

Output: per_frame_male_<start>_<end>.csv (frame, pb_slot0, pb_slot1, pb_slot2, n_slots).
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

MALE_MODEL = Path("C:/udmt_validation/segid/K_1_2/model_male_clean/model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=348599)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "step2"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    clf = load_classifier(MALE_MODEL, dev)

    recs = {"frame": [], "view": [], "gid": [], "p": []}
    BATCH = 4096
    for view, cam in s2.VIEWS:
        t0 = time.time()
        pts = s2.load_points(view, args.start, args.end)
        chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{view}.masks_{cs}-{ce}.h5", "r")
                  for cs, ce in s2.CHUNKS if not (ce < args.start or cs > args.end)}

        def chunk_h5(f):
            for (cs, ce), h in chunks.items():
                if cs <= f <= ce:
                    return h
            return None

        cap = cv2.VideoCapture(str(s2.VIDEOS / view / f"{cam}.mp4"))
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
        print(f"[{view}] processed {n_proc} sampled frames in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    pivot = df.groupby(["frame", "gid"]).p.mean().reset_index()
    wide = pivot.pivot(index="frame", columns="gid", values="p")
    for g in (0, 1, 2):
        if g not in wide.columns:
            wide[g] = np.nan
    wide = wide[[0, 1, 2]]
    arr = wide.to_numpy()
    res = pd.DataFrame({"frame": wide.index,
                        "pb_slot0": arr[:, 0], "pb_slot1": arr[:, 1], "pb_slot2": arr[:, 2],
                        "n_slots": np.sum(~np.isnan(arr), axis=1)})
    fn = out / f"per_frame_male_{args.start}_{args.end}.csv"
    res.to_csv(fn, index=False)
    print(f"\nwrote {fn}  ({len(res)} sampled frames with >=1 fish)")


if __name__ == "__main__":
    main()
