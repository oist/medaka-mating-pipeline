"""Step 3 (probe): does the SIDE-trained female/male model transfer to TOP crops?

The clean female + male models were trained on side crops (pooled_sides_*). Top is a different
appearance distribution (top-down vs lateral). Before adding top to the multiview average, probe
whether the side-trained models, run on TOP crops, are (a) confident/separated (not collapsed like
Phase-1a) and (b) AGREE with the validated side-based 3-fish track on side-confident frames.

For a few windows: build top crops per geometry slot (top masks + top video + top tracklet points),
run the female model -> P(female) and the male model -> P(M_beta) per slot. Compare the top argmax
calls to the side-based reference (round5_..._individual_id.csv).

  high agreement + confident => top transfers -> add top to the average (shrinks the male low core).
  collapse / chance        => need a top-inclusive model (auto-labelled from the side track, Phase 3).
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
from _segid_common import FEMALE_MODEL, device, forward, load_classifier  # noqa: E402

MALE_MODEL = Path("Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_male_clean/model.pt")
TRACK = HERE / "step2" / "round5_260611_3d_individual_id.csv"
TOP = ("top_cam", "cam0_top")
TOP_VIDEO = Path(r"C:\Users\machi\AppData\Local\Temp\claude\c--Users-machi-Documents-GitHub-medaka-mating-pipeline\9d685e6e-fa9d-44ed-bbfb-964240c54875\scratchpad\videos\top_cam\cam0_top.mp4")
WINDOWS = [(64000, 66000), (194000, 196000), (325000, 327000)]
STRIDE = 10


def side_reference(lo, hi):
    """frame -> {'F':slot,'M1':slot,'M2':slot, 'fconf':bool, 'mconf':bool}."""
    df = pd.read_csv(TRACK, usecols=["frame", "geometry_slot", "individual_id", "indiv_conf", "male_conf", "app_conf"])
    df = df[(df.frame >= lo) & (df.frame <= hi)]
    out = {}
    for fr, g in df.groupby("frame"):
        m = dict(zip(g.individual_id, g.geometry_slot.astype(int)))
        if {"F", "M1", "M2"} <= set(m):
            fconf = (g[g.individual_id == "F"].app_conf == "high").any()
            mconf = (g[g.individual_id == "M2"].male_conf == "high").any()
            out[int(fr)] = {**m, "fconf": bool(fconf), "mconf": bool(mconf)}
    return out


def probe_window(lo, hi, fclf, mclf, dev):
    """-> per (frame,gid): p_female, p_mbeta (top view)."""
    recs = {"frame": [], "gid": [], "pf": [], "pm": []}
    view, cam = TOP
    pts = s2.load_points(view, lo, hi)
    chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{view}.masks_{cs}-{ce}.h5", "r")
              for cs, ce in s2.CHUNKS if not (ce < lo or cs > hi)}

    def chunk_h5(f):
        for (cs, ce), h in chunks.items():
            if cs <= f <= ce:
                return h
        return None

    cap = cv2.VideoCapture(str(TOP_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    buf_crops, buf_meta = [], []

    def flush():
        if not buf_crops:
            return
        arr = np.asarray(buf_crops, np.uint8)
        pf = forward(fclf, arr, dev, head=True)
        pm = forward(mclf, arr, dev, head=True)
        for (fr, g), a, b in zip(buf_meta, pf, pm):
            recs["frame"].append(fr); recs["gid"].append(g); recs["pf"].append(float(a)); recs["pm"].append(float(b))
        buf_crops.clear(); buf_meta.clear()

    f = lo
    while f <= hi:
        sampled = (f - lo) % STRIDE == 0
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
                        if len(buf_crops) >= 4096:
                            flush()
        f += 1
    flush()
    cap.release()
    for h in chunks.values():
        h.close()
    return pd.DataFrame(recs)


def main():
    dev = device()
    fclf = load_classifier(FEMALE_MODEL, dev)
    mclf = load_classifier(MALE_MODEL, dev)
    rows = []
    for (lo, hi) in WINDOWS:
        t0 = time.time()
        ref = side_reference(lo, hi)
        df = probe_window(lo, hi, fclf, mclf, dev)
        if df.empty:
            print(f"[{lo}-{hi}] no top crops"); continue
        pf = df.pivot_table(index="frame", columns="gid", values="pf", aggfunc="mean")
        pm = df.pivot_table(index="frame", columns="gid", values="pm", aggfunc="mean")
        nfem = nfem_ok = nmale = nmale_ok = 0
        fspread, mspread = [], []
        for fr in pf.index:
            if fr not in ref:
                continue
            r = ref[fr]
            fp = pf.loc[fr].dropna()
            if len(fp) >= 2:
                top_f = int(fp.idxmax())
                s = fp.sort_values(ascending=False)
                fspread.append(float(s.iloc[0] - s.iloc[1]))
                if r["fconf"]:
                    nfem += 1; nfem_ok += int(top_f == r["F"])
            # males: among non-(side F) slots present on top
            mp = pm.loc[fr].dropna()
            mp = mp[[g for g in mp.index if g != r["F"]]]
            if len(mp) >= 2:
                top_m2 = int(mp.idxmax())
                s = mp.sort_values(ascending=False)
                mspread.append(float(s.iloc[0] - s.iloc[1]))
                if r["mconf"]:
                    nmale += 1; nmale_ok += int(top_m2 == r["M2"])
        rec = {"window": f"{lo}_{hi}", "n_top_frames": int(pf.shape[0]),
               "top_female_spread": round(float(np.median(fspread)), 3) if fspread else None,
               "female_agree_with_side": round(nfem_ok / nfem, 3) if nfem else None, "n_fem_cmp": nfem,
               "top_male_spread": round(float(np.median(mspread)), 3) if mspread else None,
               "male_agree_with_side": round(nmale_ok / nmale, 3) if nmale else None, "n_male_cmp": nmale,
               "secs": round(time.time() - t0, 0)}
        rows.append(rec); print(rec, flush=True)

    out = HERE / "step3_top"; out.mkdir(exist_ok=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "top_transfer_summary.csv", index=False)
    print("\n=== OVERALL ===")
    print(f"female: median agreement-with-side {summ.female_agree_with_side.dropna().median():.3f}, "
          f"median top P-spread {summ.top_female_spread.dropna().median():.3f} (side ref ~0.91)")
    print(f"male:   median agreement-with-side {summ.male_agree_with_side.dropna().median():.3f}, "
          f"median top P-spread {summ.top_male_spread.dropna().median():.3f} (side ref ~0.52)")
    print("chance ~0.5 for both; high agreement + confident spread => top transfers")


if __name__ == "__main__":
    main()
