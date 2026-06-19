from __future__ import annotations

"""Auto-clear the clean passes among the binder's hard-core events.

Each hard-core event (a pair within `hardcore_mm` in 3D) is a place the 3D linker's
continuous assignment MIGHT be a swap. Most short events are clean crossings the
linker got right; only dwells / true merges are genuinely ambiguous. We separate
them by 3D motion continuity, with NO ground truth:

  * extrapolate each fish's pre-event velocity across the event gap (pred_a, pred_b),
  * no_swap_err = |post_a - pred_a| + |post_b - pred_b|   (linker kept i->i)
  * swap_err    = |post_b - pred_a| + |post_a - pred_b|   (i and j swapped)
  * CLEAR (velocity confirms the linker) iff swap_err - no_swap_err > margin AND the
    fit is good (no_swap_err < max_err);
  * SUSPECT_SWAP (velocity contradicts the linker -> likely a real error) iff
    no_swap_err - swap_err > margin;
  * else AMBIGUOUS (a near-stationary dwell fits both -> stays for review).

The margin is ABSOLUTE (mm), so a dwell where both hypotheses fit within noise is
NOT cleared. Output: resolved events with verdict + cues; the non-clear subset is
what a human (or appearance/pose) still needs to adjudicate. Read-only on 3D.

Run::

    python -m scripts.tracking.resolve_hardcore_events \
        --csv-3d <..._3d_identified.csv> \
        --events-csv <tracklet_bind/hardcore_events.csv> \
        --out <tracklet_bind/hardcore_events_resolved.csv>
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

SLOTS = ["ID_1", "ID_2", "ID_3"]


def classify_event(pre_a, vel_a, pre_b, vel_b, post_a, post_b, gap, min_3d, max_step,
                   *, margin_mm, max_err_mm, step_max_mm, near_miss_mm):
    """Return (verdict, reason, no_swap_err, swap_err). verdict in {clear, review}."""
    arrs = [pre_a, vel_a, pre_b, vel_b, post_a, post_b]
    if any(a is None or not np.isfinite(a).all() for a in arrs):
        return "review", "no_3d", float("nan"), float("nan")
    pred_a = pre_a + vel_a * gap
    pred_b = pre_b + vel_b * gap
    nse = float(np.linalg.norm(post_a - pred_a) + np.linalg.norm(post_b - pred_b))
    swe = float(np.linalg.norm(post_b - pred_a) + np.linalg.norm(post_a - pred_b))
    if np.isfinite(max_step) and max_step > step_max_mm:
        return "review", "teleport", nse, swe
    diff = swe - nse
    if diff > margin_mm and nse < max_err_mm:
        return "clear", "velocity_consistent", nse, swe
    if -diff > margin_mm and swe < max_err_mm:
        return "review", "suspect_swap", nse, swe
    if min_3d >= near_miss_mm:
        return "clear", "near_miss", nse, swe
    return "review", "ambiguous", nse, swe


def _anchor_vel(arr, i_start, w):
    """Pre-event anchor (closest finite pos before i_start) + velocity over the window."""
    lo = max(i_start - w, 0)
    seg = arr[lo:i_start]
    fin = np.isfinite(seg).all(axis=1)
    if fin.sum() < 2:
        return None, None
    idx = np.where(fin)[0]
    first, last = seg[idx[0]], seg[idx[-1]]
    span = max(idx[-1] - idx[0], 1)
    return last, (last - first) / span


def _first_finite(arr, i_start, w):
    seg = arr[i_start:i_start + w]
    for row in seg:
        if np.isfinite(row).all():
            return row
    return None


def _max_step(arrs, i0, i1):
    m = 0.0
    for arr in arrs:
        seg = arr[max(i0, 0):i1 + 1]
        if len(seg) < 2:
            continue
        steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        steps = steps[np.isfinite(steps)]
        if steps.size:
            m = max(m, float(steps.max()))
    return m


def resolve_events(events, xyz, frame_to_idx, *, window, **th):
    rows = []
    for ev in events:
        a, b = ev["pair"].split("|")
        si = frame_to_idx.get(int(ev["start"]))
        ei = frame_to_idx.get(int(ev["end"]))
        if si is None or ei is None:
            rows.append({**ev, "verdict": "review", "reason": "no_idx",
                         "no_swap_err": np.nan, "swap_err": np.nan, "max_step": np.nan})
            continue
        pre_a, vel_a = _anchor_vel(xyz[a], si, window)
        pre_b, vel_b = _anchor_vel(xyz[b], si, window)
        post_a = _first_finite(xyz[a], ei + 1, window)
        post_b = _first_finite(xyz[b], ei + 1, window)
        gap = (ei + 1) - (si - 1)
        max_step = _max_step([xyz[a], xyz[b]], si - 1, ei + 1)
        verdict, reason, nse, swe = classify_event(
            pre_a, vel_a, pre_b, vel_b, post_a, post_b, gap,
            ev["min_mm"], max_step, **th)
        rows.append({**ev, "verdict": verdict, "reason": reason,
                     "no_swap_err": round(nse, 1) if np.isfinite(nse) else nse,
                     "swap_err": round(swe, 1) if np.isfinite(swe) else swe,
                     "max_step": round(max_step, 1)})
    return rows


def build_parser():
    p = argparse.ArgumentParser(description="Auto-clear clean passes among hard-core events.")
    p.add_argument("--csv-3d", required=True)
    p.add_argument("--events-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=12, help="pre/post velocity window (frames)")
    p.add_argument("--margin-mm", type=float, default=15.0, help="decisive abs. margin between hypotheses")
    p.add_argument("--max-err-mm", type=float, default=30.0, help="best fit must beat this (~1 body length)")
    p.add_argument("--step-max-mm", type=float, default=20.0, help="per-frame 3D jump above this = teleport")
    p.add_argument("--near-miss-mm", type=float, default=10.0, help="closest approach above this = barely met")
    p.add_argument("--short-frames", type=int, default=83, help="event < this = 'short pass' (~0.5s)")
    p.add_argument("--fps", type=float, default=166.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"loading 3D {args.csv_3d} ...", flush=True)
    df = pd.read_csv(args.csv_3d)
    piv = df.pivot_table(index="frame", columns="fish_id", values=["X", "Y", "Z"])
    frames = piv.index.to_numpy()
    frame_to_idx = {int(f): i for i, f in enumerate(frames)}
    xyz = {s: np.stack([piv[("X", s)].to_numpy(), piv[("Y", s)].to_numpy(),
                        piv[("Z", s)].to_numpy()], axis=1) for s in SLOTS}

    events = pd.read_csv(args.events_csv).to_dict("records")
    th = dict(margin_mm=args.margin_mm, max_err_mm=args.max_err_mm,
              step_max_mm=args.step_max_mm, near_miss_mm=args.near_miss_mm)
    rows = resolve_events(events, xyz, frame_to_idx, window=args.window, **th)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n = len(out)
    cleared = out["verdict"] == "clear"
    short = out["n_frames"] < args.short_frames
    print(f"\n=== resolved {n} hard-core events ===")
    print(f"  CLEARED (no review needed): {cleared.sum()} ({100*cleared.mean():.1f}%)")
    print(f"  flagged for review:         {(~cleared).sum()} ({100*(~cleared).mean():.1f}%)")
    print("  by reason:", dict(Counter(out["reason"])))
    print(f"\n  short events (<{args.short_frames}fr): {short.sum()}  "
          f"-> cleared {(cleared & short).sum()} ({100*(cleared & short).mean()/max(short.mean(),1e-9):.1f}% of short)")
    print(f"  long events (>={args.short_frames}fr): {(~short).sum()}  "
          f"-> cleared {(cleared & ~short).sum()}, flagged {((~cleared) & ~short).sum()}")
    susp = out["reason"] == "suspect_swap"
    print(f"\n  SUSPECT-SWAP (velocity contradicts linker -> likely real errors): {susp.sum()}")
    if susp.any():
        cols = ["start", "end", "n_frames", "pair", "min_mm", "no_swap_err", "swap_err"]
        print(out[susp].sort_values("n_frames", ascending=False)[cols].head(12).to_string(index=False))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
