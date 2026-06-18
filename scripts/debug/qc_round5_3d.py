"""One-off QC of the round5 5k-window 3D reconstruction.

Coverage (all-3-fish-valid, pre-fill) + triangulation residual (px^2) from the
pre_link JSON; 3D speed-spike rate from the gap-filled identified CSV.
Round-4 reference numbers (from memory; full-length, K_1_1/K_1_3 non-canonical
config) printed alongside for rough context — NOT identical-window/config.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"Z:/StephensU/medaka/reconstruction_3d/round5_5k_test")
EXPS = ["20251225_K_1_1_FCMC", "20251225_K_1_2_FCMC", "20251225_K_1_3_FCMC"]
TAG = "round5_260611_f171800-176799"
VEL_THRESH = 30.0  # mm/frame; >30 mm in 1/166 s (>5 m/s) is implausible

# Round-4 reference (memory): coverage, residual median/p90/frac>5000.
R4 = {
    "20251225_K_1_1_FCMC": (85.6, 147, 468, 0.6),   # non-canonical cfg
    "20251225_K_1_2_FCMC": (99.0, 164, 3016, 6.7),  # canonical-ish
    "20251225_K_1_3_FCMC": (81.7, 114, 450, 0.7),   # non-canonical cfg
}


def coverage_and_resid(pre_link: Path):
    d = json.load(open(pre_link))
    frames = d["frames"]
    items = frames.values() if isinstance(frames, dict) else frames
    n = 0
    all3 = 0
    resids = []
    for fr in items:
        n += 1
        g = fr["groups"]
        finite = [k for k in ("a", "b", "c")
                  if g.get(k) is not None and all(np.isfinite(g[k]))]
        if len(finite) == 3:
            all3 += 1
        for k in finite:
            e = fr["errors"].get(k)
            if e is not None and np.isfinite(e):
                resids.append(float(e))
    r = np.array(resids)
    return n, 100 * all3 / max(1, n), r


def speed_spikes(csv: Path):
    df = pd.read_csv(csv)
    steps = []
    for fid, g in df.groupby("final_fish_id"):
        g = g.sort_values("frame")
        xyz = g[["X", "Y", "Z"]].to_numpy(float)
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        d = d[np.isfinite(d)]
        steps.append(d)
    s = np.concatenate(steps) if steps else np.array([])
    return s


print(f"{'experiment':<22}{'cov%':>7}{'resid_med':>10}{'resid_p90':>10}"
      f"{'>5000%':>8}{'vel_p50':>9}{'vel_p95':>9}{'spike%':>8}"
      f"  | round4 ref (cov/med/p90/>5k%)")
print("-" * 118)
for e in EXPS:
    d = BASE / e / "round5_260611"
    n, cov, r = coverage_and_resid(d / f"{TAG}_best_pre_link.json")
    s = speed_spikes(d / f"{TAG}_3d_identified.csv")
    med = np.median(r) if r.size else float("nan")
    p90 = np.percentile(r, 90) if r.size else float("nan")
    f5k = 100 * np.mean(r > 5000) if r.size else float("nan")
    vp50 = np.percentile(s, 50) if s.size else float("nan")
    vp95 = np.percentile(s, 95) if s.size else float("nan")
    spike = 100 * np.mean(s > VEL_THRESH) if s.size else float("nan")
    r4 = R4[e]
    print(f"{e[9:]:<22}{cov:>7.1f}{med:>10.0f}{p90:>10.0f}{f5k:>8.1f}"
          f"{vp50:>9.2f}{vp95:>9.2f}{spike:>8.2f}"
          f"  | {r4[0]:.1f}/{r4[1]}/{r4[2]}/{r4[3]}")
print("\nNote: round4 ref is full-length, K_1_1/K_1_3 non-canonical config "
      "(mouth anchor, max-jump 30) — rough context, not identical window/config.")
