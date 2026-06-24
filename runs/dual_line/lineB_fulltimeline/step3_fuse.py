"""Step 3 (fusion, side-primary gated): fill the side male-low frames with the top vote and re-resolve.

Thin wrapper over ``per_frame_resolve.fuse_side_top``: keep the validated side male resolution byte-
identical where confident; ONLY where the side is low, take the TOP call if top is confident. A blind
side+top average degrades side-confident frames; the layered gate strictly adds coverage. The female
stays anchored by the side resolution (top female is weaker). Keeps K_1_2 research paths + prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402

NFRAMES = 348600
MARGIN_THR = 0.30
HIGH_DIST = 25

top = pd.read_csv(HERE / "step2" / f"per_frame_top_male_0_{NFRAMES-1}.csv")
res = pd.read_csv(HERE / "step2" / "round5_260611_3d_individual_id.csv")

female_ff = pfr.female_dense_from_track(res, NFRAMES)

# side baseline (the validated side-only resolution), dense per frame
side_beta_ff = np.full(NFRAMES, -1, dtype=int)
side_conf = np.array(["low"] * NFRAMES, dtype=object)
pf = res.drop_duplicates("frame")
side_beta_ff[pf.frame.to_numpy()] = pf.app_male_beta_slot.to_numpy().astype(int)
side_conf[pf.frame.to_numpy()] = pf.male_conf.to_numpy()

# top confident anchors (M2 among non-female slots, per frame)
top_tm = top[["tm_slot0", "tm_slot1", "tm_slot2"]].to_numpy()
tf = top.frame.to_numpy().astype(int)
tbeta = np.full(len(top), -1, dtype=int)
tmarg = np.full(len(top), np.nan)
for i in range(len(top)):
    b, mgn = pfr.beta_from_probs(top_tm[i], int(female_ff[tf[i]]))
    tbeta[i], tmarg[i] = b, mgn
taf, tabeta = pfr.confident_anchors(tf, tbeta, tmarg, MARGIN_THR)

beta_final, male_conf = pfr.fuse_side_top(side_beta_ff, side_conf, taf, tabeta, female_ff,
                                          NFRAMES, HIGH_DIST)
print(f"side-high frames: {int((side_conf=='high').sum())}; "
      f"top confident anchors: {len(taf)}")

frames = res.frame.to_numpy(); slots = res.geometry_slot.to_numpy().astype(int)
res["individual_id_fused"] = pfr.assemble_individual_labels(frames, slots, female_ff, beta_final, NFRAMES)
res["male_conf_fused"] = np.asarray(male_conf)[frames]
res["indiv_conf_fused"] = np.where(res.individual_id_fused == "F", res.app_conf, res.male_conf_fused)

males = res[res.individual_id_fused.isin(["M1", "M2"])]
old_hi = float((males.male_conf == "high").mean())
new_hi = float((males.male_conf_fused == "high").mean())
print(f"\nmale high-confidence: side-only {old_hi:.3f} -> fused {new_hi:.3f}  (+{new_hi-old_hi:.3f})")
recov = males[(males.male_conf == "low") & (males.male_conf_fused == "high")]
print(f"side-low recovered to high by top: {len(recov)} ({len(recov)/max(int((males.male_conf=='low').sum()),1):.1%} of side-low)")
degraded = males[(males.male_conf == "high") & (males.male_conf_fused == "low")]
print(f"side-high degraded to low (should be 0 with gating): {len(degraded)}")
changed = res[(res.individual_id != res.individual_id_fused) & (res.individual_id_fused != "F")
              & (res.male_conf == "high")]
print(f"fused M1/M2 differs from side-only on side-HIGH frames (should be 0): {len(changed)}")

counts = res.groupby("frame").individual_id_fused.value_counts().unstack(fill_value=0)
ok = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
print(f"frames with exactly 1 F + 1 M1 + 1 M2 (fused): {ok}/{len(counts)}")
print("\nfused indiv_conf:", {k: int(v) for k, v in res.indiv_conf_fused.value_counts().items()})

out = HERE / "step2" / "round5_260611_3d_individual_id_fused.csv"
res.to_csv(out, index=False)
print(f"\nwrote {out}")
