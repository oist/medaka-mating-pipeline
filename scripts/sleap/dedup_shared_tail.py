from __future__ import annotations

"""
Drop shared-tail predictions from a per-view SLEAP ``.slp``.

Failure mode (observed at close contact): two centered_instance predictions have
DISTINCT mouths but their tails both latch onto the SAME physical tail (each crop
grabs the nearer / more salient tail). The instance that got the wrong tail has
its centroid dragged toward the neighbour, which converges the two 3D positions
and triggers an ID swap.

Fix: per frame, when two instances' tails fall within ``--tail-eps`` px, keep the
tail with the highest tail-node score and **null the others' tail** (set it
missing) while KEEPING their mouth. Downstream 3D then uses the mouth alone for
that instance (visible-node centroid collapses to the mouth) instead of a tail
sitting on the wrong fish. No instances are deleted, so coverage is preserved.

The nulled frames are also written to a CSV — these are exactly the close-contact
hard cases to add to the next annotation round.

Usage::

    python -m scripts.sleap.dedup_shared_tail \
        --slp    <...>/side1_cam.slp \
        --output <...>/side1_cam.dedup.slp \
        --tail-eps 20 --report-csv <...>/side1_cam.shared_tail.csv
"""

import argparse
from pathlib import Path

import numpy as np
import sleap_io as sio


def _node_idx(skeleton, name):
    for i, n in enumerate(skeleton.nodes):
        if n.name == name:
            return i
    return None


def dedup(labels: sio.Labels, tail_eps: float, tail_name: str = "tail"):
    ti = _node_idx(labels.skeletons[0], tail_name)
    if ti is None:
        raise SystemExit(f"no '{tail_name}' node in skeleton")
    n_nulled = 0
    events = []  # (frame_idx, n_nulled_this_frame)
    for lf in labels.labeled_frames:
        # collect (instance, tail_xy, tail_score) for instances with a visible tail
        cands = []
        for inst in lf.instances:
            pts = inst.points
            if not bool(pts["visible"][ti]):
                continue
            xy = np.asarray(pts["xy"][ti], float)
            if not np.all(np.isfinite(xy)):
                continue
            cands.append((inst, xy, float(pts["score"][ti])))
        if len(cands) < 2:
            continue
        # greedy: highest tail score keeps its tail; later tails within eps of an
        # already-kept tail get nulled.
        cands.sort(key=lambda c: c[2], reverse=True)
        kept_xy = []
        nulled_here = 0
        for inst, xy, _score in cands:
            if any(np.hypot(*(xy - k)) < tail_eps for k in kept_xy):
                inst.points["xy"][ti] = [np.nan, np.nan]
                inst.points["visible"][ti] = False
                inst.points["score"][ti] = 0.0
                n_nulled += 1
                nulled_here += 1
            else:
                kept_xy.append(xy)
        if nulled_here:
            events.append((int(lf.frame_idx), nulled_here))
    return labels, n_nulled, events


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slp", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tail-eps", type=float, default=20.0,
                    help="Tails within this many px are treated as the same tail.")
    ap.add_argument("--tail-name", default="tail")
    ap.add_argument("--report-csv", type=Path, default=None,
                    help="Optional: write per-frame nulled counts (hard-case list).")
    args = ap.parse_args(argv)

    labels = sio.load_slp(str(args.slp))
    n_lf = len(labels.labeled_frames)
    labels, n_nulled, events = dedup(labels, args.tail_eps, args.tail_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(args.output))
    print(f"{args.slp.name}: {n_lf} frames, nulled {n_nulled} shared tails "
          f"in {len(events)} frames (eps={args.tail_eps}px) -> {args.output.name}")
    if args.report_csv is not None:
        import csv
        with args.report_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["frame_idx", "n_nulled"])
            w.writerows(events)
        print(f"  hard-case frames -> {args.report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
