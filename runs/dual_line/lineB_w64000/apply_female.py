"""Apply the clean female model to the production-framed crops 33/34 built from the
real saion full-length masks (window 64000-69000, untrained range). The production-path
analogue of Phase-1b: does the female anchor separate female vs males on real masks?
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "runs" / "dual_line" / "phase0"))
sys.path.insert(0, str(REPO))
from _segid_common import FEMALE_MODEL, device, forward, load_classifier  # noqa: E402
from sam2_id_experiment.src.id_model.window_align import select_covisible_mask  # noqa: E402

P_HI = 0.7
dev = device()
ds = HERE / "pooled"
meta = pd.read_parquet(ds / "crop_metadata.parquet").reset_index(drop=True)
with h5py.File(ds / "crops.h5", "r") as h:
    crops = np.asarray(h["crops"][...], np.uint8)

valid = meta[(meta.qc_status == "clean_train") & (meta.global_id.isin([0, 1, 2]))].copy()
clf = load_classifier(FEMALE_MODEL, dev)
valid["p_female"] = forward(clf, crops[valid.index.to_numpy()], dev, head=True)

# per-global_id mean P(female) (the within-window 3D-bind id is reliable -> direct female call)
per = valid.groupby("global_id").p_female.agg(["mean", "size"]).sort_values("mean", ascending=False)
female = int(per.index[0])
sep = float(per["mean"].iloc[0] - per["mean"].iloc[1])

# within-frame female-vs-male P-spread on co-visible (all-3-present) frames
cov = select_covisible_mask(valid.view, valid.frame, valid.global_id, num_fish=3)
covdf = valid[cov]
spreads = []
for _, g in covdf.groupby(["view", "frame"]):
    pf = g.groupby("global_id").p_female.mean()
    if len(pf) >= 2:
        spreads.append(float(pf.max() - pf.min()))
spreads = np.array(spreads)

print("=== window 64000-69000 | clean crops:", len(valid), "| co-visible frames:", len(spreads), "===")
print("\nper-global_id mean P(female) [all clean crops]:")
for gid, row in per.iterrows():
    print(f"  global_id {int(gid)}: meanP={row['mean']:.3f}  n={int(row['size'])}  {'<- FEMALE' if int(gid)==female else ''}")
print(f"\nfemale global_id = {female}  | separation (top - 2nd) = {sep:.3f}")
print(f"frac crops P(female)>{P_HI} = {float((valid.p_female > P_HI).mean()):.3f}  (ideal ~0.33 for 1F:2M)")
print(f"within-frame female-vs-male P-spread: median={np.median(spreads):.3f}  mean={np.mean(spreads):.3f}")
print(f"  (Phase-1b production-framed reference was 0.911 within-frame on untrained windows)")
# per-view consistency: does each view independently call the same female?
print("\nper-view female call (consistency):")
for v, g in valid.groupby("view"):
    pv = g.groupby("global_id").p_female.mean().sort_values(ascending=False)
    print(f"  {v}: female gid={int(pv.index[0])} (P={pv.iloc[0]:.3f}), 2nd={int(pv.index[1])} (P={pv.iloc[1]:.3f})")

# Distinguish a geometry SWAP (consistent appearance, flipping label) from appearance
# ambiguity: per co-visible frame, the appearance-female = argmax mean P over the 3 fish
# (multiview-averaged). If it switches gid at a frame and stays, that's a 3D-bind swap.
fr = (valid[cov].groupby(["frame", "global_id"]).p_female.mean().reset_index()
      .pivot(index="frame", columns="global_id", values="p_female").dropna())
arg = fr.idxmax(axis=1)
frames, gids = arg.index.to_numpy(), arg.to_numpy()
runs = []
s = 0
for i in range(1, len(gids) + 1):
    if i == len(gids) or gids[i] != gids[s]:
        runs.append((int(frames[s]), int(frames[i - 1]), int(gids[s]), i - s)); s = i
runs_long = [r for r in runs if r[3] >= 20]
print(f"\nappearance-female gid timeline ({len(fr)} all-3 frames): {len(runs)} runs, "
      f"{len(runs_long)} long(>=20f); distinct females={sorted(set(gids.tolist()))}")
print("  long runs (start,end,gid,nframes):")
for r in runs_long:
    print("   ", r)
valid[["view", "frame", "global_id", "p_female"]].to_csv(HERE / "per_crop_pfemale.csv", index=False)
