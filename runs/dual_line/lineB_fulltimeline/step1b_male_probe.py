"""Step 1b (male de-risk): per-frame male-A-vs-male-B separation probe.

Mirror of Phase-1b for the female, but for the two MALES. The female slot is already
known per frame (step2 per_frame_female CSV); the two non-female slots are the males.
Run the swap-corrected male classifier (model_male_clean, classes [gid_0, gid_2];
forward head -> P(gid_2) == P(M_beta)) on the male slots and measure, on UNTRAINED
windows (none of {145k,152k,285k}):

  * within-frame male P-spread |P_Mbeta(slotA) - P_Mbeta(slotB)|  -- the decisive,
    GT-free metric (analog of the female's 0.911). High => the model tells the two
    males apart this frame; ~0 => collapse (Phase-1a style).
  * per-geometry-gid mean P_Mbeta  -- does appearance track the geometry male id where
    geometry is stable?
  * AUC-vs-geometry: label=(gid==2), score=P_Mbeta  -- SECONDARY, depressed by the very
    geometry swaps we want appearance to fix (so a LOWER bound on appearance quality).

Reuses step2_per_frame_female fast-path helpers (random-access masks + local video).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
import step2_per_frame_female as s2  # noqa: E402  (square_crop, link_gid, load_points, read_frame_masks, CHUNKS, VIEWS, MASKS_Z, VIDEOS)
from _segid_common import device, forward, load_classifier  # noqa: E402

SEGID = Path("C:/udmt_validation/segid/K_1_2")
MALE_MODEL = SEGID / "model_male_clean" / "model.pt"   # classes [gid_0, gid_2]; head -> P(gid_2)=P(M_beta)
PER_FRAME_FEMALE = HERE / "step2" / "per_frame_female_0_348599.csv"
FEMALE_MARGIN_THR = 0.30   # only use frames where the female slot is confident

# untrained probe windows (male model trained on 145k/152k/285k)
PROBE_WINDOWS = [(64000, 69000), (194000, 199000), (325000, 330000)]


def load_female_map(lo, hi):
    pf = pd.read_csv(PER_FRAME_FEMALE)
    pf = pf[(pf.frame >= lo) & (pf.frame <= hi) & (pf.female_slot >= 0) & (pf.margin > FEMALE_MARGIN_THR)]
    return dict(zip(pf.frame.astype(int), pf.female_slot.astype(int)))


def probe_window(start, end, stride, clf, dev):
    """Return DataFrame: frame, view, gid, p_mbeta  (per male-slot crop)."""
    recs = {"frame": [], "view": [], "gid": [], "p": []}
    BATCH = 4096
    for view, cam in s2.VIEWS:
        pts = s2.load_points(view, start, end)
        chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{view}.masks_{cs}-{ce}.h5", "r")
                  for cs, ce in s2.CHUNKS if not (ce < start or cs > end)}

        def chunk_h5(f):
            for (cs, ce), h in chunks.items():
                if cs <= f <= ce:
                    return h
            return None

        cap = cv2.VideoCapture(str(s2.VIDEOS / view / f"{cam}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        buf_crops, buf_meta = [], []

        def flush():
            if not buf_crops:
                return
            p = forward(clf, np.asarray(buf_crops, np.uint8), dev, head=True)
            for (fr, g), pf in zip(buf_meta, p):
                recs["frame"].append(fr); recs["view"].append(view); recs["gid"].append(g); recs["p"].append(float(pf))
            buf_crops.clear(); buf_meta.clear()

        f = start
        while f <= end:
            sampled = (f - start) % stride == 0
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
            f += 1
        flush()
        cap.release()
        for h in chunks.values():
            h.close()
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "step1b_male"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    clf = load_classifier(MALE_MODEL, dev)
    from sklearn.metrics import roc_auc_score

    all_pf, summary = [], []
    for (lo, hi) in PROBE_WINDOWS:
        t0 = time.time()
        fmap = load_female_map(lo, hi)
        raw = probe_window(lo, hi, args.stride, clf, dev)
        if raw.empty:
            print(f"[{lo}-{hi}] no crops", flush=True); continue
        # multiview mean P_Mbeta per (frame, gid)
        pivot = raw.groupby(["frame", "gid"]).p.mean().reset_index()
        # restrict to frames with a confident female and >=2 male slots present
        rows = []
        for fr, grp in pivot.groupby("frame"):
            if fr not in fmap:
                continue
            fs = fmap[fr]
            males = grp[grp.gid != fs]
            if males.gid.nunique() < 2:
                continue
            mg = males.set_index("gid").p
            present = sorted(mg.index.tolist())
            spread = float(abs(mg.loc[present[0]] - mg.loc[present[1]]))
            beta_slot = int(mg.idxmax())                 # argmax P(M_beta)
            alpha_slot = int(mg.idxmin())
            rows.append({"frame": int(fr), "female_slot": int(fs),
                         "alpha_slot": alpha_slot, "beta_slot": beta_slot, "spread": spread,
                         "p_g0": float(mg.get(0, np.nan)), "p_g2": float(mg.get(2, np.nan)),
                         "p_g1": float(mg.get(1, np.nan))})
        wpf = pd.DataFrame(rows)
        wpf["window"] = f"{lo}_{hi}"
        all_pf.append(wpf)

        # AUC-vs-geometry (secondary; lower bound): per male crop, label=(gid==2), score=P_Mbeta
        mc = pivot.merge(pd.Series(fmap, name="fs").rename_axis("frame").reset_index(), on="frame")
        mc = mc[(mc.gid != mc.fs) & mc.gid.isin([0, 2])]
        auc = float(roc_auc_score((mc.gid == 2).astype(int), mc.p)) if mc.gid.nunique() == 2 and len(mc) > 10 else float("nan")
        # per-gid mean P_Mbeta among male crops
        permean = mc.groupby("gid").p.mean().to_dict()

        s = {"window": f"{lo}_{hi}", "n_frames": int(len(wpf)),
             "median_spread": round(float(wpf.spread.median()), 3) if len(wpf) else None,
             "frac_spread_gt0.3": round(float((wpf.spread > 0.3).mean()), 3) if len(wpf) else None,
             "auc_vs_geom_lowerbound": round(auc, 3),
             "mean_p_mbeta_gid0": round(float(permean.get(0, np.nan)), 3),
             "mean_p_mbeta_gid2": round(float(permean.get(2, np.nan)), 3),
             "secs": round(time.time() - t0, 0)}
        summary.append(s)
        print(s, flush=True)

    if all_pf:
        pf = pd.concat(all_pf, ignore_index=True)
        pf.to_csv(out / "male_probe_per_frame.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "male_probe_summary.csv", index=False)
        print("\n=== OVERALL ===")
        print(f"frames probed: {len(pf)}")
        print(f"median within-frame male P-spread: {pf.spread.median():.3f}  "
              f"(female ref ~0.911); frac>0.3: {(pf.spread>0.3).mean():.3f}")
        print(f"wrote {out/'male_probe_per_frame.csv'} and male_probe_summary.csv")


if __name__ == "__main__":
    main()
