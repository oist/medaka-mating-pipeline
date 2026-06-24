"""Step 2 (Part B): resolve the window-anchor `low` spans with the per-frame appearance
female-slot (Part A output), re-sex the 3D track at frame resolution, and validate against
the window-anchor sexing.

- Confident appearance anchors (margin > THR) give the female's geometry slot densely
  (~every `stride` frames). Forward-fill them to frame resolution; a slot change between
  consecutive anchors = a 3D-bind swap, localized to that <=stride-frame gap.
- Re-sex: female = the fish whose `geometry_slot` == appearance female-slot. Frames far from
  any confident anchor (sustained all-views merge) stay `low`.
- Validate: on the window-anchor `high`/`med` frames (independently trusted), does the
  appearance re-sex AGREE? Report resolution of the `low` frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NFRAMES = 348600
MARGIN_THR = 0.30          # confident per-frame female-slot
HIGH_DIST = 25             # frames: within this of a confident anchor -> high/med; else low

pf = pd.read_csv(HERE / "step2" / f"per_frame_female_0_{NFRAMES-1}.csv")
sexed = pd.read_csv(HERE / "female_anchor" / "round5_260611_3d_identified_sexed.csv")

# confident appearance anchors
anc = pf[(pf.margin > MARGIN_THR) & (pf.female_slot >= 0)].sort_values("frame")
af, asl = anc.frame.to_numpy(), anc.female_slot.to_numpy().astype(int)
print(f"confident appearance anchors: {len(af)} / {len(pf)} sampled frames "
      f"({len(af)/len(pf):.1%}); median spacing {int(np.median(np.diff(af)))} frames")

# frame-resolution female slot by nearest-preceding confident anchor; distance to nearest anchor
allf = np.arange(NFRAMES)
prev = np.searchsorted(af, allf, side="right") - 1
prev = np.clip(prev, 0, len(af) - 1)
slot_ff = asl[prev]
nearest_d = np.abs(allf - af[np.clip(np.searchsorted(af, allf), 0, len(af) - 1)])
nearest_d = np.minimum(nearest_d, np.abs(allf - af[prev]))
appconf = np.where(nearest_d <= HIGH_DIST, "high", "low")

# swaps = consecutive confident anchors that disagree (localized to the gap between them)
chg = np.where(asl[1:] != asl[:-1])[0]
swaps = [(int(af[i]), int(af[i + 1]), int(asl[i]), int(asl[i + 1])) for i in chg]
print(f"appearance-detected female-slot swaps: {len(swaps)}")

# re-sex the 3D track
sexed["app_female_slot"] = slot_ff[sexed.frame.to_numpy()]
sexed["app_conf"] = appconf[sexed.frame.to_numpy()]
sexed["sex_resolved"] = np.where(sexed.geometry_slot == sexed.app_female_slot, "F", "M")

# exactly one female per frame? (appearance slot is a single value per frame -> yes by construction)
fpf = sexed[sexed.sex_resolved == "F"].groupby("frame").size()
print(f"frames with exactly 1 resolved F: {int((fpf == 1).sum())}/{sexed.frame.nunique()} "
      f"(!=1: {int((fpf != 1).sum())})")

# validation vs window-anchor sexing on independently-trusted high/med frames
hm = sexed[sexed.sex_confidence.isin(["high", "med"])]
agree_hm = float((hm.sex == hm.sex_resolved).mean())
hi = sexed[sexed.sex_confidence == "high"]
print(f"\nvalidation vs window-anchor sexing:")
print(f"  agreement on window 'high' frames:    {float((hi.sex==hi.sex_resolved).mean()):.3f}")
print(f"  agreement on window 'high'+'med':     {agree_hm:.3f}")

# resolution of the window `low` spans
low = sexed[sexed.sex_confidence == "low"]
low_now_conf = low[low.app_conf == "high"]
print(f"\nresolution of window 'low' fish-frames ({len(low)}):")
print(f"  now appearance-confident:  {len(low_now_conf)} ({len(low_now_conf)/max(len(low),1):.1%})")
print(f"  changed sex vs window call: {int((low.sex != low.sex_resolved).sum())} "
      f"({float((low.sex != low.sex_resolved).mean()):.1%} of low rows)")

# final confidence breakdown (per fish-frame)
print("\nresolved confidence (per fish-frame):", {k: int(v) for k, v in pd.Series(appconf[sexed.frame.to_numpy()]).value_counts().items()})

out = HERE / "step2" / "round5_260611_3d_identified_sexed_resolved.csv"
sexed.to_csv(out, index=False)
pd.DataFrame(swaps, columns=["anchor_before", "anchor_after", "slot_before", "slot_after"]).to_csv(
    HERE / "step2" / "appearance_swaps.csv", index=False)
print(f"\nwrote {out}")
