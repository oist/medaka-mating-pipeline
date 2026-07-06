"""Step 2c: embedder global-anchoring check for the M1/M2 assignment.

The M1/M2 per-frame labels come from the classifier (model_male_clean). This check verifies, with
an INDEPENDENT model (the self-supervised embedder model_resnet_full), that M1 and M2 are each one
GLOBALLY coherent appearance identity across the whole timeline -- not just locally stable. A global
flip (what's called M2 after frame X is physically the old M1) would survive within-window stability
but is caught here.

Method: sample confident-male frames from 6 well-separated timeline segments; build M1/M2 crops (both
side views) at their assigned geometry slots; embed; then
  (1) cross-segment nearest-centroid transfer: centroids from segment i classify segment j's crops by
      assigned label. Global-consistent => every (i,j) cell >> 0.5. Any cell < 0.5 = a possible global
      flip between those segments.
  (2) global 2-means ARI vs the M1/M2 labels + silhouette (sanity that 2 clusters exist and match).
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
from _segid_common import EMBEDDER_MODEL, device, forward, load_embedder  # noqa: E402

TRACK = HERE / "step2" / "round5_260611_3d_individual_id.csv"
SEGMENTS = [(28000, 30500), (92000, 94500), (158000, 160500),
            (224000, 226500), (290000, 292500), (338000, 340500)]
STRIDE = 5


def male_slot_map(lo, hi):
    """frame -> {'M1': slot, 'M2': slot} for confident-male frames in [lo,hi]."""
    df = pd.read_csv(TRACK, usecols=["frame", "geometry_slot", "individual_id", "male_conf"])
    df = df[(df.frame >= lo) & (df.frame <= hi) & (df.male_conf == "high") & df.individual_id.isin(["M1", "M2"])]
    out = {}
    for fr, g in df.groupby("frame"):
        m = dict(zip(g.individual_id, g.geometry_slot.astype(int)))
        if "M1" in m and "M2" in m:
            out[int(fr)] = m
    return out


def collect_crops(seg_idx, lo, hi, dev):
    smap = male_slot_map(lo, hi)
    crops, meta = [], []
    for view, cam in s2.VIEWS:
        pts = s2.load_points(view, lo, hi)
        chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{view}.masks_{cs}-{ce}.h5", "r")
                  for cs, ce in s2.CHUNKS if not (ce < lo or cs > hi)}

        def chunk_h5(f):
            for (cs, ce), h in chunks.items():
                if cs <= f <= ce:
                    return h
            return None

        cap = cv2.VideoCapture(str(s2.VIDEOS / view / f"{cam}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        f = lo
        while f <= hi:
            sampled = (f - lo) % STRIDE == 0
            ok, img = (cap.read() if sampled else (cap.grab(), None))
            if not ok:
                break
            if sampled and img is not None and f in smap and f in pts:
                h5 = chunk_h5(f)
                if h5 is not None:
                    masks = s2.read_frame_masks(h5, f)
                    if masks:
                        gid2mask = dict(s2.link_gid(masks, pts[f]))
                        for lab in ("M1", "M2"):
                            sl = smap[f][lab]
                            if sl in gid2mask:
                                crops.append(s2.square_crop(img, masks[gid2mask[sl]][1]))
                                meta.append((seg_idx, f, view, lab))
            f += 1
        cap.release()
        for h in chunks.values():
            h.close()
    return crops, meta


def main():
    dev = device()
    emb = load_embedder(EMBEDDER_MODEL, dev)
    all_crops, all_meta = [], []
    for i, (lo, hi) in enumerate(SEGMENTS):
        t0 = time.time()
        c, m = collect_crops(i, lo, hi, dev)
        all_crops.extend(c); all_meta.extend(m)
        print(f"seg {i} [{lo}-{hi}]: {len(c)} crops in {time.time()-t0:.0f}s", flush=True)

    meta = pd.DataFrame(all_meta, columns=["seg", "frame", "view", "label"])
    Z = forward(emb, np.asarray(all_crops, np.uint8), dev, head=False)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    y = (meta.label == "M2").to_numpy().astype(int)
    seg = meta.seg.to_numpy()
    print(f"\ntotal crops {len(Z)}; per label {meta.label.value_counts().to_dict()}; "
          f"per seg {meta.seg.value_counts().sort_index().to_dict()}")

    # (2) global 2-means ARI + silhouette
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Z)
    print(f"\nglobal 2-means ARI vs M1/M2 labels: {adjusted_rand_score(y, km.labels_):.3f}")
    print(f"silhouette of M1/M2 labeling:        {silhouette_score(Z, y):.3f}")

    # (1) cross-segment nearest-centroid transfer matrix
    nseg = len(SEGMENTS)
    cents = {}  # (seg, label) -> centroid
    for s_i in range(nseg):
        for lab in (0, 1):
            mask = (seg == s_i) & (y == lab)
            if mask.sum():
                v = Z[mask].mean(0); cents[(s_i, lab)] = v / (np.linalg.norm(v) + 1e-9)
    print("\ncross-segment nearest-centroid transfer (train rows -> test cols), accuracy vs assigned label:")
    hdr = "train\\test " + " ".join(f"s{j:>4}" for j in range(nseg))
    print(hdr)
    mat = np.full((nseg, nseg), np.nan)
    for i in range(nseg):
        if (i, 0) not in cents or (i, 1) not in cents:
            continue
        c0, c1 = cents[(i, 0)], cents[(i, 1)]
        row = []
        for j in range(nseg):
            mask = seg == j
            if not mask.sum():
                row.append("  -- "); continue
            pred = (Z[mask] @ c1 > Z[mask] @ c0).astype(int)  # nearest (cosine) centroid
            acc = float((pred == y[mask]).mean())
            mat[i, j] = acc
            row.append(f"{acc:5.2f}")
        print(f"   s{i:<6} " + " ".join(row))
    off = mat[~np.eye(nseg, dtype=bool)]
    off = off[~np.isnan(off)]
    print(f"\noff-diagonal (cross-segment) transfer: min {off.min():.3f}, median {np.median(off):.3f}, mean {off.mean():.3f}")
    print("VERDICT:", "GLOBALLY CONSISTENT (no flip)" if off.min() > 0.7 else
          ("POSSIBLE GLOBAL FLIP -- inspect cells < 0.5" if off.min() < 0.5 else "MOSTLY CONSISTENT (weak cells 0.5-0.7)"))

    out = HERE / "step2" / "global_anchor_transfer.csv"
    pd.DataFrame(mat, columns=[f"test_s{j}" for j in range(nseg)],
                 index=[f"train_s{i}" for i in range(nseg)]).to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
