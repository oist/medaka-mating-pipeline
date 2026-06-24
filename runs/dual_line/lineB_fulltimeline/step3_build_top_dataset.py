"""Step 3: build an AUTO-LABELED top-view crop dataset from the validated side-based 3-fish track.

No human labels (project constraint): at side-HIGH-confidence frames (male_conf == 'high', which makes
the whole F/M1/M2 slot map trustworthy), the geometry slot -> individual map is the teacher. Build TOP
crops (33._square_crop on top masks + local top video) per geometry slot and label by the side track's
individual_id, encoded as global_id so `39_train_identity_classifier` trains both classifiers directly:

    F -> global_id 1,  M1 -> global_id 0,  M2 -> global_id 2   (the window_align-aligned convention).

Then: `39 --mode female --female-gid 1` -> top female (head=P(female)); `39 --mode male --female-gid 1`
-> top male (head=P(M2)). Both drop-in compatible with _segid_common.load_classifier + forward.

Output: <out>/crops.h5 (uint8 N,128,128,3) + crop_metadata.parquet (schema 39 expects).
"""
from __future__ import annotations

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

TRACK = HERE / "step2" / "round5_260611_3d_individual_id.csv"
TOP_VIDEO = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\9d685e6e-fa9d-44ed-bbfb-964240c54875\scratchpad\videos\top_cam\cam0_top.mp4")
TB = Path(r"C:\udmt_validation\tracklet_bind_extend_smooth\top_cam.tracklet_bound.csv")
OUT = Path("C:/udmt_validation/segid/K_1_2/pooled_top_autolabeled")
VIEW = "top_cam"
WINDOWS = [(38000, 41000), (108000, 111000), (173000, 176000),
           (238000, 241000), (298000, 301000), (338000, 341000)]
STRIDE = 5
ID2GID = {"M1": 0, "F": 1, "M2": 2}


def link_gid_tid(masks, points, gate=s2.GATE):
    """points: [(x,y,gid,tid)] -> [(gid, mask_idx, tid)] (mirror of s2.link_gid, carrying tracklet id)."""
    out = []
    for mi, (m, _b) in enumerate(masks):
        H, W = m.shape
        inside = [(g, t) for x, y, g, t in points
                  if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W and m[int(round(y)), int(round(x))]]
        if len(inside) == 1:
            out.append((inside[0][0], mi, inside[0][1]))
        elif len(inside) == 0:
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            cx, cy = xs.mean(), ys.mean()
            best, bd = None, gate
            for x, y, g, t in points:
                d = float(np.hypot(x - cx, y - cy))
                if d < bd:
                    best, bd = (g, t), d
            if best is not None:
                out.append((best[0], mi, best[1]))
    return out


def load_points_tid(lo, hi):
    df = pd.read_csv(TB, usecols=["frame", "x", "y", "global_id", "tracklet_id"])
    df = df[(df.frame >= lo) & (df.frame <= hi) & df.global_id.isin([0, 1, 2])]
    pts: dict[int, list] = {}
    for f, x, y, g, t in zip(df.frame, df.x, df.y, df.global_id, df.tracklet_id):
        pts.setdefault(int(f), []).append((float(x), float(y), int(g), int(t)))
    return pts


def side_slot_map(lo, hi):
    """frame -> {gid: individual_id} for male_conf=='high' frames."""
    df = pd.read_csv(TRACK, usecols=["frame", "geometry_slot", "individual_id", "male_conf"])
    df = df[(df.frame >= lo) & (df.frame <= hi) & (df.male_conf == "high")]
    out = {}
    for fr, g in df.groupby("frame"):
        out[int(fr)] = dict(zip(g.geometry_slot.astype(int), g.individual_id))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops, rows = [], []
    chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{VIEW}.masks_{cs}-{ce}.h5", "r") for cs, ce in s2.CHUNKS}

    def chunk_h5(f):
        for (cs, ce), h in chunks.items():
            if cs <= f <= ce:
                return h
        return None

    cap = cv2.VideoCapture(str(TOP_VIDEO))
    for (lo, hi) in WINDOWS:
        t0 = time.time()
        win = f"win_{lo}_{hi}"
        pts = load_points_tid(lo, hi)
        smap = side_slot_map(lo, hi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        f = lo
        n0 = len(crops)
        while f <= hi:
            sampled = (f - lo) % STRIDE == 0
            ok, img = (cap.read() if sampled else (cap.grab(), None))
            if not ok:
                break
            if sampled and img is not None and f in pts and f in smap:
                h5 = chunk_h5(f)
                if h5 is not None:
                    masks = s2.read_frame_masks(h5, f)
                    if masks:
                        for gid, mi, tid in link_gid_tid(masks, pts[f]):
                            if gid not in smap[f]:
                                continue
                            indiv = smap[f][gid]            # F / M1 / M2 from the side track
                            mask, bbox = masks[mi]
                            crops.append(s2.square_crop(img, bbox))
                            px = [p for p in pts[f] if p[2] == gid][0]
                            rows.append({"frame": f, "view": VIEW, "window": win,
                                         "tracklet_id": int(tid), "view_tracklet": int(tid),
                                         "global_id": ID2GID[indiv], "raw_geometry_slot": int(gid),
                                         "individual_id": indiv, "global_id_purity": 1.0,
                                         "qc_status": "clean_train",
                                         "x": float(px[0]), "y": float(px[1]), "area": float(mask.sum())})
            f += 1
        print(f"{win}: +{len(crops)-n0} crops in {time.time()-t0:.0f}s", flush=True)
    cap.release()
    for h in chunks.values():
        h.close()

    meta = pd.DataFrame(rows)
    meta["crop_id"] = np.arange(len(meta), dtype=int)
    arr = np.asarray(crops, np.uint8)
    with h5py.File(OUT / "crops.h5", "w") as h:
        h.create_dataset("crops", data=arr, compression="gzip", compression_opts=4)
    meta.to_parquet(OUT / "crop_metadata.parquet", index=False)
    print(f"\nwrote {OUT}  ({len(meta)} crops)")
    print("per individual:", meta.individual_id.value_counts().to_dict())
    print("per window x individual:")
    print(meta.groupby(["window", "individual_id"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
