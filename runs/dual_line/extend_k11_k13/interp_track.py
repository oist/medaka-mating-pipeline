"""Parameterized interp step (Goal 4) for the N=3 extension — gap-fill the appearance-low
residual of a script-45 3-fish track into a gap-free id_final track.

Generalises runs/dual_line/lineB_fulltimeline/step5_interpolate.py (which was hardcoded to the
K_1_2 fused CSV / male_conf_fused) to any 45 output (--in-csv) with its column names. Pure
post-processing over the shared, unit-tested per_frame_resolve core (no inference).

  BRIDGED   low run between AGREEING anchors -> certain (no swap).
  EDGE      low run at timeline start/end -> fill from the one bounding anchor.
  AMBIGUOUS low run between DISAGREEING anchors -> localise the swap at the 3D closest approach.

Run::
    .venv\\Scripts\\python.exe runs/dual_line/extend_k11_k13/interp_track.py \\
        --in-csv runs/dual_line/extend_k11_k13/K_1_3/round5_260611_3d_individual_id.csv \\
        --out-csv runs/dual_line/extend_k11_k13/K_1_3/round5_260611_3d_individual_id_interp.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402

SRC2CONF = {"high": "high", "bridged": "high", "edge": "interp", "motion_localized": "interp", "ambiguous": "ambiguous"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interpolate a 45 3-fish track into a gap-free id_final track.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--female-slot-col", default="app_female_slot")
    ap.add_argument("--female-conf-col", default="app_conf")
    ap.add_argument("--male-beta-col", default="app_male_beta_slot")
    ap.add_argument("--male-conf-col", default="male_conf")
    args = ap.parse_args(argv)

    track = pd.read_csv(args.in_csv)
    fr = track.frame.to_numpy().astype(int)
    sl = track.geometry_slot.to_numpy().astype(int)
    n = int(fr.max()) + 1

    female_ff = pfr.female_dense_from_track(track, n, args.female_slot_col)
    pf = track.drop_duplicates("frame")
    pframes = pf.frame.to_numpy().astype(int)
    app_conf_f = np.array(["low"] * n, dtype=object); app_conf_f[pframes] = pf[args.female_conf_col].to_numpy()
    beta_ff = np.full(n, -1, dtype=int); beta_ff[pframes] = pf[args.male_beta_col].to_numpy().astype(int)
    male_conf = np.array(["low"] * n, dtype=object); male_conf[pframes] = pf[args.male_conf_col].to_numpy()

    xyz = np.full((n, 3, 3), np.nan)
    xyz[fr, sl, 0] = track.X.to_numpy(); xyz[fr, sl, 1] = track.Y.to_numpy(); xyz[fr, sl, 2] = track.Z.to_numpy()
    pair_dist = {frozenset((a, b)): np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in combinations(range(3), 2)}

    def interpolate(slot_ff, conf, label):
        src, runs = pfr.bridge_low_spans(slot_ff, conf, n)
        slot_i, swaps = pfr.localize_ambiguous_swaps(slot_ff, runs, pair_dist)
        for (s, e, kind, a, b) in runs:
            if kind != "ambiguous":
                continue
            seg = pair_dist.get(frozenset((a, b)))
            if seg is not None and np.isfinite(seg[s:e + 1]).any():
                src[s:e + 1] = "motion_localized"
        ncount = {k: int((src == k).sum()) for k in ("high", "bridged", "edge", "motion_localized", "ambiguous")}
        print(f"[{label}] {ncount}  | localized {len(swaps)} crossings", flush=True)
        return slot_i, src

    female_i, src_f = interpolate(female_ff, app_conf_f, "female")
    beta_i, src_m = interpolate(beta_ff, male_conf, "male")

    beta_final, conflict = pfr.resolve_conflict_beta(beta_i, female_i)
    src_m = np.where(conflict, "ambiguous", src_m)
    track["id_final"] = pfr.assemble_individual_labels(fr, sl, female_i, beta_final, n)
    is_fem = sl == female_i[fr]
    src_row = np.where(is_fem, src_f[fr], src_m[fr])
    track["id_source"] = src_row
    track["id_conf"] = np.vectorize(SRC2CONF.get)(src_row)

    counts = track.groupby("frame").id_final.value_counts().unstack(fill_value=0)
    one_each = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
    print(f"\nframes with exactly 1 F+1 M1+1 M2: {one_each}/{track.frame.nunique()}")
    print("id_conf per fish-frame:", {k: int(v) for k, v in track.id_conf.value_counts().items()})
    for ind, g in track.groupby("id_final"):
        b = g.id_conf.value_counts(normalize=True)
        print(f"  {ind}: high {b.get('high', 0):.3f}  interp {b.get('interp', 0):.3f}  ambiguous {b.get('ambiguous', 0):.3f}")
    amb = float((track.id_conf == "ambiguous").mean())
    print(f"irreducible ambiguous core: {(track.id_conf=='ambiguous').sum()} fish-frames ({amb:.2%})")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
