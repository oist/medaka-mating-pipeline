"""Step 5 (Goal 4) — interpolate the appearance-low residual into a gap-free 3-fish ID track.

The fused track (step3) leaves ~11.8% male / ~3% female fish-frames "low" (no confident appearance
anchor nearby = sustained all-views merge). But most of those are NOT actually uncertain:

  * BRIDGED  — a low run bounded by two confident anchors that AGREE on the slot: the identity is
               certain (no swap occurred), it merely lacked a nearby confident frame -> mark CERTAIN.
  * EDGE     — a low run at the timeline start/end with one bounding anchor -> fill from it (interp).
  * AMBIGUOUS— a low run between DISAGREEING anchors (a swap somewhere in the gap): localize the swap
               at the frame of closest 3D approach between the two slots (the crossing) -> interp.
               (No external signal there; closest-approach is the maximum-likelihood crossing.)

Output: `<...>_individual_id_interp.csv` with `id_final` (always F/M1/M2), `id_source`
(high/bridged/edge/motion_localized/ambiguous) and `id_conf` (high/interp/ambiguous). Uses the shared,
unit-tested `per_frame_resolve` core. Pure post-processing on the fused CSV (no inference).
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402

NFRAMES = 348600
FUSED = HERE / "step2" / "round5_260611_3d_individual_id_fused.csv"

track = pd.read_csv(FUSED)
fr = track.frame.to_numpy().astype(int)
sl = track.geometry_slot.to_numpy().astype(int)

# dense per-frame fields
female_ff = pfr.female_dense_from_track(track, NFRAMES, "app_female_slot")
pf = track.drop_duplicates("frame")
app_conf_f = np.array(["low"] * NFRAMES, dtype=object); app_conf_f[pf.frame.to_numpy()] = pf.app_conf.to_numpy()
beta_ff = np.full(NFRAMES, -1, dtype=int); beta_ff[pf.frame.to_numpy()] = pf.app_male_beta_slot.to_numpy().astype(int)
male_conf = np.array(["low"] * NFRAMES, dtype=object); male_conf[pf.frame.to_numpy()] = pf.male_conf_fused.to_numpy()

# 3D positions + per-frame inter-slot distances (for crossing localization)
xyz = np.full((NFRAMES, 3, 3), np.nan)
xyz[fr, sl, 0] = track.X.to_numpy(); xyz[fr, sl, 1] = track.Y.to_numpy(); xyz[fr, sl, 2] = track.Z.to_numpy()
pair_dist = {frozenset((a, b)): np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in combinations(range(3), 2)}


def interpolate(slot_ff, conf, label):
    src, runs = pfr.bridge_low_spans(slot_ff, conf, NFRAMES)
    slot_i, swaps = pfr.localize_ambiguous_swaps(slot_ff, runs, pair_dist)
    # mark which ambiguous runs got a localized crossing (finite distance) vs stayed truly ambiguous
    localized = set(swaps)
    for (s, e, kind, a, b) in runs:
        if kind != "ambiguous":
            continue
        seg = pair_dist.get(frozenset((a, b)))
        if seg is not None and np.isfinite(seg[s:e + 1]).any():
            src[s:e + 1] = "motion_localized"   # the run's crossing was placed
    n = {k: int((src == k).sum()) for k in ("high", "bridged", "edge", "motion_localized", "ambiguous")}
    print(f"[{label}] {n}  | localized {len(localized)} crossings")
    return slot_i, src, n


female_i, src_f, nf = interpolate(female_ff, app_conf_f, "female")
beta_i, src_m, nm = interpolate(beta_ff, male_conf, "male")

# keep exactly 1F+1M1+1M2 after independent female/male interpolation
beta_final, conflict = pfr.resolve_conflict_beta(beta_i, female_i)
src_m = np.where(conflict, "ambiguous", src_m)
track["id_final"] = pfr.assemble_individual_labels(fr, sl, female_i, beta_final, NFRAMES)

# per (frame, slot) provenance: female slot uses the female source, the two male slots use the male source
SRC2CONF = {"high": "high", "bridged": "high", "edge": "interp", "motion_localized": "interp", "ambiguous": "ambiguous"}
is_fem = sl == female_i[fr]
src_row = np.where(is_fem, src_f[fr], src_m[fr])
track["id_source"] = src_row
track["id_conf"] = np.vectorize(SRC2CONF.get)(src_row)

# report
counts = track.groupby("frame").id_final.value_counts().unstack(fill_value=0)
one_each = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
print(f"\nframes with exactly 1 F+1 M1+1 M2: {one_each}/{track.frame.nunique()}")
print("id_conf per fish-frame:", {k: int(v) for k, v in track.id_conf.value_counts().items()})
for ind, g in track.groupby("id_final"):
    b = g.id_conf.value_counts(normalize=True)
    print(f"  {ind}: high {b.get('high',0):.3f}  interp {b.get('interp',0):.3f}  ambiguous {b.get('ambiguous',0):.3f}")
amb = float((track.id_conf == "ambiguous").mean())
print(f"\nirreducible ambiguous core: {(track.id_conf=='ambiguous').sum()} fish-frames ({amb:.2%}) "
      f"= the all-views-merged swap interiors with no appearance OR motion signal")

out = HERE / "step2" / "round5_260611_3d_individual_id_interp.csv"
track.to_csv(out, index=False)
print(f"\nwrote {out}")
