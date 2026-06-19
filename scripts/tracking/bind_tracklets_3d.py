from __future__ import annotations

"""B: bind per-view concrete tracklets to the 3D identity backbone.

Validated precondition (validate_tracklet_3d_bind): per-view tracklets reproject
onto a unique 3D fish with ~6 px residual, ~200 px margin and ~1.0 consistency. So
the 3D trajectory is a reliable cross-view identity backbone -- EXCEPT where two
fish are genuinely together in 3D, where the 3D linker itself can swap. This tool:

  1. detects HARD-CORE EVENTS -- contiguous runs where a pair is within
     `hardcore_mm` in 3D -- and breaks the backbone there (the residual decisions);
  2. binds every per-view concrete tracklet to its modal 3D slot by reprojection
     (the single-view overlaps that the 3D sails through are stitched for free);
  3. writes per-view identity-stitched detections (each view's fragments joined
     under the 3 slot identities -- the win, especially for the side views that had
     no identity before), and a hard-core-events list flagged for review.

It does NOT resolve the swaps at hard-core events (that needs appearance / round-6
pose / human review) -- it ISOLATES them: identity is certain between events and
flagged within them, instead of silently smooth-swapped.

Run::

    python -m scripts.tracking.bind_tracklets_3d \
        --slp-dir <dir> --csv-3d <..._3d_identified.csv> --model-folder <models> \
        --out-dir runs/tracklet_bind/<exp>
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.tracking.concrete_tracklets import (
    build_concrete_tracklets, build_tracklets_edge_extended,
)
from scripts.tracking.compare_anchor_tracklets import heading_flip_rate, load_pose_frames
from scripts.tracking.compare_stifd import ifd_and_stifd, per_det_speed, smooth_by_frame

VIEWS = ["top_cam", "side1_cam", "side2_cam"]
SLOTS = ["ID_1", "ID_2", "ID_3"]


# --------------------------- pure, testable logic --------------------------- #

def pairwise_3d(xyz: dict[str, np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """xyz: slot -> (N,3). Returns pair -> (N,) Euclidean distance (NaN-propagating)."""
    out = {}
    for a, b in combinations(SLOTS, 2):
        out[(a, b)] = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    return out


def _merged_runs(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs in `mask`, merging runs separated by <= merge_gap."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p <= merge_gap:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    runs.append((s, p))
    return runs


def detect_hardcore_events(pair_dist, frames, *, hardcore_mm, merge_gap=5, min_frames=1):
    """Per pair, contiguous runs where 3D distance < hardcore_mm (the residual decisions)."""
    frames = np.asarray(frames)
    events = []
    for pair, d in pair_dist.items():
        d = np.asarray(d, dtype=float)
        mask = np.isfinite(d) & (d < hardcore_mm)
        for s, e in _merged_runs(mask, merge_gap):
            n = e - s + 1
            if n < min_frames:
                continue
            events.append({
                "pair": f"{pair[0]}|{pair[1]}",
                "start": int(frames[s]), "end": int(frames[e]),
                "n_frames": int(n), "min_mm": round(float(np.nanmin(d[s:e + 1])), 2),
            })
    events.sort(key=lambda ev: ev["start"])
    return events


def bind_tracklet_points(points, reproj, frame_to_idx):
    """points: [(frame,x,y,score)]; reproj: (3,N,2) for one view.

    Returns {slot, mean_resid, median_margin, consistency, perframe:[(frame,slot|None)]}
    or None if no frame could be matched.
    """
    bests, resids, margins, perframe = [], [], [], []
    for f, x, y, _score in points:
        i = frame_to_idx.get(int(f))
        if i is None:
            perframe.append((int(f), None)); continue
        d = np.linalg.norm(reproj[:, i, :] - np.array([x, y]), axis=1)
        fin = np.isfinite(d)
        if fin.sum() < 1:
            perframe.append((int(f), None)); continue
        order = np.argsort(np.where(fin, d, np.inf))
        best = int(order[0])
        bests.append(best); resids.append(float(d[best])); perframe.append((int(f), best))
        if fin.sum() >= 2:
            margins.append(float(d[order[1]] - d[order[0]]))
    if not bests:
        return None
    ba = np.array(bests)
    modal = int(np.bincount(ba).argmax())
    return {
        "slot": modal,
        "mean_resid": float(np.mean(resids)),
        "median_margin": float(np.median(margins)) if margins else float("nan"),
        "consistency": float((ba == modal).mean()),
        "perframe": perframe,
    }


# ------------------------------- orchestration ------------------------------ #

def _reproject_all(calib, xyz):
    n = len(next(iter(xyz.values())))
    proj = {}
    for v in VIEWS:
        arr = np.full((len(SLOTS), n, 2), np.nan)
        for si, s in enumerate(SLOTS):
            x = xyz[s]
            m = np.isfinite(x).all(axis=1)
            if m.any():
                arr[si, m] = calib.project_world_to_view(x[m], v)
        proj[v] = arr
    return proj


def _build_view_tracklets(slp, args):
    """Per-view tracklets. Positions are ALWAYS the centroid (clean low-residual
    binding to the reprojected 3D centroid). --anchor sets the IFD metric (auto =
    mouth where the mouth/tail swap rate <1%, else centroid); --gate sets the break
    criterion:
      * 'stifd'   = velocity-standardized: break when IFD / sqrt(si^2+sj^2) < tau,
                    sigma = sigma0 + smoothed per-frame displacement (close-but-slow
                    stays a tracklet, fast crossings break) -- the recommended gate.
      * 'absolute'= break when IFD < sep_bl * BL (the original constant gate).
    """
    frames_pose, bl, _ = load_pose_frames(slp, num_fish=args.num_fish, top_k=True)
    fd_centroid = {f: [(d["centroid"][0], d["centroid"][1], d["score"]) for d in ds]
                   for f, ds in frames_pose.items()}
    common = dict(num_fish=args.num_fish, body_length=bl, sep_bl=args.sep_bl,
                  jump_bl=1.0, min_score=args.min_score, max_gap=args.max_gap)
    metric, swap = args.anchor, float("nan")
    if args.anchor == "auto":
        tr_c, _ = build_concrete_tracklets(fd_centroid, **common)
        swap, _ = heading_flip_rate(frames_pose, tr_c)
        metric = "mouth" if swap < 0.01 else "centroid"
    speeds = per_det_speed(frames_pose, gate_px=bl)
    if args.gate == "extend":
        mean_sp = float(np.mean([s for sp in speeds.values() for s in sp])) if speeds else 1.0
        tracklets, _ = build_tracklets_edge_extended(
            fd_centroid, num_fish=args.num_fish, body_length=bl,
            tau=args.stifd_tau, sigma0=args.sigma0, nominal_speed=mean_sp,
            jump_bl=1.0, min_score=args.min_score, max_gap=args.max_gap,
            sep_smooth_window=args.smooth_window)
        return tracklets, bl, metric, swap
    ifd, _, sigma = ifd_and_stifd(frames_pose, speeds, metric, args.sigma0)
    if args.gate == "stifd":
        sig_s = smooth_by_frame(sigma, args.smooth_window)
        sep = {f: (ifd[f] / sig_s[f] if np.isfinite(sig_s.get(f, np.nan)) and sig_s[f] > 0
                   else np.inf) for f in ifd}
        sep_threshold = args.stifd_tau
    else:
        sep = ifd
        sep_threshold = args.sep_bl * bl
    tracklets, _ = build_concrete_tracklets(
        fd_centroid, min_sep_by_frame=sep, sep_threshold=sep_threshold, **common)
    return tracklets, bl, metric, swap


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bind per-view tracklets to the 3D backbone.")
    p.add_argument("--slp-dir", required=True)
    p.add_argument("--csv-3d", required=True)
    p.add_argument("--model-folder", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--sep-bl", type=float, default=1.5)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--anchor", choices=["auto", "centroid", "mouth", "body-axis"], default="auto",
                   help="IFD metric per view (positions stay centroid for binding); "
                        "auto = mouth where mouth/tail swap rate <1%% else centroid")
    p.add_argument("--gate", choices=["stifd", "absolute", "extend"], default="stifd",
                   help="break gate: 'stifd' = frame-level velocity-standardized IFD/sigma<tau; "
                        "'extend' = core-seed + per-tracklet stIFD edge-extension (only the close "
                        "pair breaks, uninvolved fish keeps tracking); 'absolute' = IFD < sep_bl*BL")
    p.add_argument("--stifd-tau", type=float, default=2.0, help="stIFD gate threshold (sigma units)")
    p.add_argument("--sigma0", type=float, default=2.0, help="stIFD localization-noise floor (px)")
    p.add_argument("--smooth-window", type=int, default=9, help="frames for rolling-median sigma smoothing")
    p.add_argument("--hardcore-mm", type=float, default=12.0,
                   help="3D pair distance below which identity is unresolvable (break the backbone)")
    p.add_argument("--event-merge-gap", type=int, default=5)
    p.add_argument("--resid-max", type=float, default=20.0)
    p.add_argument("--margin-min", type=float, default=10.0)
    p.add_argument("--consistency-min", type=float, default=0.95)
    p.add_argument("--paths-config", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    # 1) hard-core events (break the backbone)
    pdist = pairwise_3d(xyz)
    events = detect_hardcore_events(pdist, frames, hardcore_mm=args.hardcore_mm,
                                    merge_gap=args.event_merge_gap)
    hardcore_frames = set()
    for ev in events:
        hardcore_frames.update(range(ev["start"], ev["end"] + 1))
    pd.DataFrame(events).to_csv(out_dir / "hardcore_events.csv", index=False)

    # 2) reproject + bind per-view tracklets
    proj = _reproject_all(calib, xyz)
    summary = {"hardcore": {
        "hardcore_mm": args.hardcore_mm, "n_events": len(events),
        "frames_in_events": len(hardcore_frames),
        "frac_frames_in_events": round(len(hardcore_frames) / max(len(frames), 1), 4),
    }, "views": {}}

    for v in VIEWS:
        slp = Path(args.slp_dir) / f"{v}.slp"
        if not slp.exists():
            raise SystemExit(f"missing {slp}")
        tracklets, bl, chosen_anchor, swap = _build_view_tracklets(slp, args)
        swap_str = f" swap={100*swap:.1f}%" if swap == swap else ""
        print(f"binding {v} (gate={args.gate} anchor={chosen_anchor}{swap_str}): "
              f"{len(tracklets)} tracklets", flush=True)
        A = proj[v]
        rows = []
        n_low = 0
        n_bound = 0
        for t in tracklets:
            if t["length"] < args.min_len:
                continue
            res = bind_tracklet_points(t["points"], A, frame_to_idx)
            if res is None:
                continue
            low = (res["consistency"] < args.consistency_min
                   or res["mean_resid"] > args.resid_max
                   or not (res["median_margin"] >= args.margin_min))
            n_bound += 1
            n_low += int(low)
            for f, x, y, score in t["points"]:
                rows.append({
                    "frame": int(f), "x": x, "y": y, "score": score,
                    "global_id": res["slot"], "tracklet_id": t["id"],
                    "bind_resid": round(res["mean_resid"], 2),
                    "bind_margin": round(res["median_margin"], 1),
                    "bind_consistency": round(res["consistency"], 3),
                    "low_conf": bool(low),
                    "in_hardcore": int(f) in hardcore_frames,
                })
        out_csv = out_dir / f"{v}.tracklet_bound.csv"
        rdf = pd.DataFrame(rows).sort_values(["frame", "global_id"])
        rdf.to_csv(out_csv, index=False)
        certain = rdf[~rdf["in_hardcore"]]
        summary["views"][v] = {
            "tracklets_bound": n_bound, "low_conf_tracklets": n_low,
            "rows": len(rdf), "distinct_ids": int(rdf["global_id"].nunique()),
            "certain_rows": int(len(certain)),
            "frac_certain": round(len(certain) / max(len(rdf), 1), 4),
            "anchor": chosen_anchor,
            "gate": args.gate,
            "swap_rate": (round(swap, 4) if swap == swap else None),
        }
        print(f"  {v}: bound {n_bound} tracklets ({n_low} low-conf), "
              f"{len(rdf)} det rows -> {out_csv.name}")

    (out_dir / "bind_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(f"hard-core events: {len(events)}  "
          f"({summary['hardcore']['frac_frames_in_events']*100:.1f}% of frames in an event)")
    if events:
        durs = np.array([e["n_frames"] for e in events])
        print(f"  event duration frames: median={int(np.median(durs))}  "
              f"max={int(durs.max())}  total={int(durs.sum())}")
        from collections import Counter
        print(f"  by pair: {dict(Counter(e['pair'] for e in events))}")
    for v in VIEWS:
        s = summary["views"][v]
        print(f"  {v}: {s['tracklets_bound']} tracklets -> {s['distinct_ids']} ids, "
              f"{s['frac_certain']*100:.1f}% of det rows outside hard-core")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
