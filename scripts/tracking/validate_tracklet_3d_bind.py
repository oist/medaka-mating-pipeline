from __future__ import annotations

"""Validation A: can a per-view tracklet be bound to a UNIQUE 3D fish reliably?

For each view we build concrete (high-confidence) tracklets, reproject every
reconstructed 3D fish into that view via the calibration, and ask of each tracklet:

  * residual  -- how close (px) the tracklet's detections sit to the 3D fish they
                 match (does the tracklet actually lie on a reconstructed fish?),
  * consistency-- does the tracklet stay matched to the SAME 3D fish across all its
                 frames (a "pure" tracklet) or does the best match flip (impure),
  * margin     -- how much farther the 2nd-nearest 3D fish reprojects (is the choice
                 decisive or a coin-flip?).

And per frame, a bijection check: do a view's N detections bind to N DISTINCT 3D
fish? If residuals are low, consistency high, margins decisive and bijection near
100%, then the 3D trajectory is a reliable identity backbone to stitch tracklets
onto -- the precondition for the cross-view binder.

Note the top view is partly circular (most 3D came from the top 2D bind), so the
SIDE views are the honest test: do the sides agree with the 3D the top anchored?

Run::

    python -m scripts.tracking.validate_tracklet_3d_bind \
        --slp-dir <dir> --csv-3d <..._3d_identified.csv> --model-folder <models>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.tracking.concrete_tracklets import _load_frames_data, build_concrete_tracklets

VIEWS = ["top_cam", "side1_cam", "side2_cam"]
SLOTS = ["ID_1", "ID_2", "ID_3"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate per-view tracklet -> 3D-fish binding.")
    p.add_argument("--slp-dir", required=True)
    p.add_argument("--csv-3d", required=True, help="<tag>_3d_identified.csv (per-frame ID_1/2/3 XYZ)")
    p.add_argument("--model-folder", default=None, help="calibration models dir (else paths.local.toml)")
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sep-bl", type=float, default=1.5)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--resid-max", type=float, default=20.0, help="px; clean-bind residual threshold")
    p.add_argument("--margin-min", type=float, default=10.0, help="px; clean-bind margin threshold")
    p.add_argument("--consistency-min", type=float, default=0.95)
    p.add_argument("--min-len", type=int, default=5, help="ignore tracklets shorter than this")
    p.add_argument("--output", default=None, help="optional per-tracklet metrics CSV")
    p.add_argument("--paths-config", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    slp_dir = Path(args.slp_dir)
    model_folder = args.model_folder
    if model_folder is None:
        local = load_local_paths(args.paths_config)
        model_folder = choose_path(None, local.calibration_model_dir if local else None)
    if model_folder is None:
        raise SystemExit("need --model-folder or calibration_model_dir in paths.local.toml")
    calib = Calibration(str(model_folder))

    print(f"loading 3D {args.csv_3d} ...", flush=True)
    df = pd.read_csv(args.csv_3d)
    piv = df.pivot_table(index="frame", columns="fish_id", values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    xyz = {s: np.stack([piv[("X", s)].to_numpy(), piv[("Y", s)].to_numpy(),
                        piv[("Z", s)].to_numpy()], axis=1) for s in SLOTS}

    # reproject each 3D fish into each view: proj[V] shape (3, Nframes, 2)
    proj: dict[str, np.ndarray] = {}
    for v in VIEWS:
        arr = np.full((len(SLOTS), len(frames), 2), np.nan)
        for si, s in enumerate(SLOTS):
            x = xyz[s]
            m = np.isfinite(x).all(axis=1)
            if m.any():
                arr[si, m] = calib.project_world_to_view(x[m], v)
        proj[v] = arr

    rows = []
    perframe_best: dict[str, dict[int, list[int]]] = {v: defaultdict(list) for v in VIEWS}
    for v in VIEWS:
        slp = slp_dir / f"{v}.slp"
        if not slp.exists():
            raise SystemExit(f"missing {slp}")
        print(f"tracklets on {v} ...", flush=True)
        fd, bl, _ = _load_frames_data(slp, num_fish=args.num_fish, top_k=True,
                                      start_frame=0, end_frame=None)
        tracklets, _ = build_concrete_tracklets(
            fd, num_fish=args.num_fish, body_length=bl, sep_bl=args.sep_bl,
            jump_bl=1.0, min_score=args.min_score, max_gap=args.max_gap)
        A = proj[v]
        for t in tracklets:
            if t["length"] < args.min_len:
                continue
            bests, resids, margins = [], [], []
            for f, x, y, _score in t["points"]:
                i = frame_to_idx.get(int(f))
                if i is None:
                    continue
                d = np.linalg.norm(A[:, i, :] - np.array([x, y]), axis=1)  # (3,)
                fin = np.isfinite(d)
                if fin.sum() < 1:
                    continue
                order = np.argsort(np.where(fin, d, np.inf))
                best = int(order[0])
                bests.append(best)
                resids.append(float(d[best]))
                perframe_best[v][int(f)].append(best)
                if fin.sum() >= 2:
                    margins.append(float(d[order[1]] - d[order[0]]))
            if not bests:
                continue
            ba = np.array(bests)
            modal = int(np.bincount(ba).argmax())
            rows.append({
                "view": v, "tracklet": t["id"], "len": t["length"],
                "frames_used": len(bests), "modal_fish": modal,
                "consistency": float((ba == modal).mean()),
                "mean_resid": float(np.mean(resids)),
                "median_margin": float(np.median(margins)) if margins else np.nan,
            })

    R = pd.DataFrame(rows)
    if args.output:
        R.to_csv(args.output, index=False)
        print(f"wrote per-tracklet metrics -> {args.output}")

    def report(name, sub):
        if sub.empty:
            print(f"\n[{name}] no tracklets"); return
        clean = ((sub["consistency"] >= args.consistency_min)
                 & (sub["mean_resid"] <= args.resid_max)
                 & (sub["median_margin"] >= args.margin_min))
        fw = sub["len"].to_numpy()
        cov = fw[clean.to_numpy()].sum() / max(fw.sum(), 1)
        print(f"\n[{name}] tracklets={len(sub)}  frames={int(fw.sum())}")
        print(f"  residual px      median={sub['mean_resid'].median():.1f}  "
              f"p90={sub['mean_resid'].quantile(.9):.1f}")
        print(f"  margin px        median={sub['median_margin'].median():.1f}  "
              f"p10={sub['median_margin'].quantile(.1):.1f}")
        print(f"  consistency      median={sub['consistency'].median():.3f}  "
              f">=0.99: {100*(sub['consistency']>=0.99).mean():.1f}%  "
              f">=0.95: {100*(sub['consistency']>=0.95).mean():.1f}%")
        print(f"  CLEAN-BIND tracklets: {100*clean.mean():.1f}%   "
              f"frame-weighted coverage: {100*cov:.1f}%")

    for v in VIEWS:
        report(v, R[R["view"] == v])
    report("ALL VIEWS", R)

    print("\n=== per-frame bijection (a view's N dets -> N DISTINCT 3D fish) ===")
    for v in VIEWS:
        pf = perframe_best[v]
        full = [b for b in pf.values() if len(b) == args.num_fish]
        if not full:
            print(f"  {v}: no full-coverage frames"); continue
        distinct = sum(len(set(b)) == args.num_fish for b in full)
        print(f"  {v}: {distinct}/{len(full)} full frames have distinct binding "
              f"({100*distinct/len(full):.1f}%)")


if __name__ == "__main__":
    main()
