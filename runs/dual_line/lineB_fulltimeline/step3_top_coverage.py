"""Step 3 (cheap gate): is the TOP view clean where the SIDE male assignment is low-confidence?

The side-trained models don't transfer to top (probe: collapsed). Training a top-inclusive model is
only worth it if top masks are SEPARATED at the male-low-confidence frames (the close-contact scenes
we'd want to recover). Read-only: link top masks to the 3 geometry slots (no model, no video) and
count how many slots get a clean 1:1 mask. Compare male-LOW vs male-HIGH frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase0"))
import step2_per_frame_female as s2  # noqa: E402

TRACK = HERE / "step2" / "round5_260611_3d_individual_id.csv"
VIEW = "top_cam"
STRIDE = 50


def clean_gids(masks, points):
    """number of distinct geometry slots that link 1:1 to a top mask (merges dropped)."""
    gids = [g for g, _mi in s2.link_gid(masks, points)]
    return len(set(gids))


def main():
    df = pd.read_csv(TRACK, usecols=["frame", "individual_id", "male_conf"])
    mrows = df[df.individual_id.isin(["M1", "M2"])]
    conf_by_frame = mrows.groupby("frame").male_conf.first()  # male_conf is per-frame
    low_frames = set(conf_by_frame[conf_by_frame == "low"].index.astype(int))
    high_frames = set(conf_by_frame[conf_by_frame == "high"].index.astype(int))

    pts_all = s2.load_points(VIEW, 0, 348599)
    chunks = {(cs, ce): h5py.File(s2.MASKS_Z / f"{VIEW}.masks_{cs}-{ce}.h5", "r") for cs, ce in s2.CHUNKS}

    def chunk_h5(f):
        for (cs, ce), h in chunks.items():
            if cs <= f <= ce:
                return h
        return None

    res = {"low": [], "high": []}
    for label, frameset in (("low", low_frames), ("high", high_frames)):
        fr_sorted = sorted(frameset)[::STRIDE]
        for f in fr_sorted:
            if f not in pts_all:
                continue
            h5 = chunk_h5(f)
            if h5 is None:
                continue
            masks = s2.read_frame_masks(h5, f)
            if not masks:
                res[label].append(0); continue
            res[label].append(clean_gids(masks, pts_all[f]))
    for h in chunks.values():
        h.close()

    print("top-view clean-slot coverage (how many of the 3 geometry slots get a 1:1 top mask):")
    for label in ("low", "high"):
        a = np.array(res[label])
        if not len(a):
            print(f"  {label}: no frames"); continue
        print(f"  male-{label} frames (n={len(a)}): "
              f">=3 clean {float((a>=3).mean()):.3f}, >=2 {float((a>=2).mean()):.3f}, "
              f"mean {a.mean():.2f}")
    aL = np.array(res["low"])
    print(f"\nKEY: of side-male-LOW frames, {float((aL>=3).mean()):.1%} have all 3 slots clean on TOP "
          f"(=> a top model could recover them); {float((aL>=2).mean()):.1%} have >=2 clean.")


if __name__ == "__main__":
    main()
