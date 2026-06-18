"""QC of the full-length round5 3D reconstruction (HPC inference -> 3D).

Coverage (all-3-fish-valid, pre-fill) + triangulation residual (px^2) from the
pre_link JSON; 3D speed-spike rate from the gap-filled identified CSV. Prints the
5k-window round5 numbers and round4 references alongside for context.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"Z:/StephensU/medaka/reconstruction_3d/round5_260611")
EXPS = ["20251225_K_1_1_FCMC", "20251225_K_1_2_FCMC", "20251225_K_1_3_FCMC"]
TAG = "round5_260611"
VEL_THRESH = 30.0  # mm/frame implausible

# round5 5k-window (171800-176799) numbers, for context.
R5_5K = {"20251225_K_1_1_FCMC": (87.0, 96, 366, 0.0),
         "20251225_K_1_2_FCMC": (86.8, 127, 471, 0.0),
         "20251225_K_1_3_FCMC": (97.8, 82, 298, 0.0)}


def coverage_and_resid(pre_link: Path):
    d = json.load(open(pre_link))
    frames = d["frames"]
    items = frames.values() if isinstance(frames, dict) else frames
    n = all3 = 0
    resids = []
    for fr in items:
        n += 1
        g = fr["groups"]
        fin = [k for k in ("a", "b", "c")
               if g.get(k) is not None and all(np.isfinite(g[k]))]
        if len(fin) == 3:
            all3 += 1
        for k in fin:
            e = fr["errors"].get(k)
            if e is not None and np.isfinite(e):
                resids.append(float(e))
    return n, 100 * all3 / max(1, n), np.array(resids)


def speed_spikes(csv: Path):
    df = pd.read_csv(csv, usecols=["frame", "final_fish_id", "X", "Y", "Z"])
    steps = []
    for _, g in df.groupby("final_fish_id"):
        g = g.sort_values("frame")
        d = np.linalg.norm(np.diff(g[["X", "Y", "Z"]].to_numpy(float), axis=0), axis=1)
        steps.append(d[np.isfinite(d)])
    return np.concatenate(steps) if steps else np.array([])


hdr = (f"{'experiment':<14}{'frames':>8}{'cov%':>7}{'rMed':>6}{'rP90':>7}"
       f"{'rP99':>8}{'>5k%':>7}{'vP50':>7}{'vP95':>7}{'spike%':>8}"
       f"   | 5k-window cov/med/p90")
print(hdr)
print("-" * len(hdr))
for e in EXPS:
    d = BASE / e
    n, cov, r = coverage_and_resid(d / f"{TAG}_best_pre_link.json")
    s = speed_spikes(d / f"{TAG}_3d_identified.csv")
    med = np.median(r) if r.size else float("nan")
    p90 = np.percentile(r, 90) if r.size else float("nan")
    p99 = np.percentile(r, 99) if r.size else float("nan")
    f5k = 100 * np.mean(r > 5000) if r.size else float("nan")
    vp50 = np.percentile(s, 50) if s.size else float("nan")
    vp95 = np.percentile(s, 95) if s.size else float("nan")
    spike = 100 * np.mean(s > VEL_THRESH) if s.size else float("nan")
    c5 = R5_5K[e]
    print(f"{e[9:]:<14}{n:>8}{cov:>7.1f}{med:>6.0f}{p90:>7.0f}{p99:>8.0f}"
          f"{f5k:>7.2f}{vp50:>7.2f}{vp95:>7.2f}{spike:>8.3f}"
          f"   | {c5[0]:.0f}/{c5[1]}/{c5[2]}")
print("\nResidual px^2; vel mm/frame; spike = frac > 30 mm/frame (implausible).")
