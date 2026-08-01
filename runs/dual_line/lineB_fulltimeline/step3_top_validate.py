"""Step 3: validate the TOP models per-frame (the analog of step3_top_probe, but with the top-trained
models). Windows 64k/194k/325k are OUTSIDE the top training windows. Confident top P-spread + agreement
with the side track on side-confident frames => top is ready to fuse.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
import step3_top_probe as p3  # noqa: E402
from _segid_common import device, load_classifier  # noqa: E402

TOP_F = "Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_top_female/model.pt"
TOP_M = "Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2/model_top_male/model.pt"


def main():
    dev = device()
    fclf = load_classifier(TOP_F, dev)
    mclf = load_classifier(TOP_M, dev)
    rows = []
    for (lo, hi) in p3.WINDOWS:
        t0 = time.time()
        ref = p3.side_reference(lo, hi)
        df = p3.probe_window(lo, hi, fclf, mclf, dev)
        if df.empty:
            continue
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
                s = fp.sort_values(ascending=False)
                fspread.append(float(s.iloc[0] - s.iloc[1]))
                if r["fconf"]:
                    nfem += 1; nfem_ok += int(int(fp.idxmax()) == r["F"])
            mp = pm.loc[fr].dropna()
            mp = mp[[g for g in mp.index if g != r["F"]]]
            if len(mp) >= 2:
                s = mp.sort_values(ascending=False)
                mspread.append(float(s.iloc[0] - s.iloc[1]))
                if r["mconf"]:
                    nmale += 1; nmale_ok += int(int(mp.idxmax()) == r["M2"])
        rec = {"window": f"{lo}_{hi}",
               "top_female_spread": round(float(np.median(fspread)), 3) if fspread else None,
               "female_agree_with_side": round(nfem_ok / nfem, 3) if nfem else None, "n_fem": nfem,
               "top_male_spread": round(float(np.median(mspread)), 3) if mspread else None,
               "male_agree_with_side": round(nmale_ok / nmale, 3) if nmale else None, "n_male": nmale,
               "secs": round(time.time() - t0, 0)}
        rows.append(rec); print(rec, flush=True)
    summ = pd.DataFrame(rows)
    (HERE / "step3_top").mkdir(exist_ok=True)
    summ.to_csv(HERE / "step3_top" / "top_MODEL_validate_summary.csv", index=False)
    print("\n=== OVERALL (top models on untrained windows) ===")
    print(f"female: agreement-with-side {summ.female_agree_with_side.dropna().median():.3f}, "
          f"P-spread {summ.top_female_spread.dropna().median():.3f}  (side-model-on-top was 0.58 / 0.018)")
    print(f"male:   agreement-with-side {summ.male_agree_with_side.dropna().median():.3f}, "
          f"P-spread {summ.top_male_spread.dropna().median():.3f}  (side-model-on-top was 0.69 / 0.011)")


if __name__ == "__main__":
    main()
