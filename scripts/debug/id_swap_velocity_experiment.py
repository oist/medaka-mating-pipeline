from __future__ import annotations

"""
Offline ID-swap experiment (read-only on production data).

Question: does a velocity-aware 3D temporal linker reduce identity swaps versus
the production position-only linker (`apply_cross_frame_tracking`), on the
*same* pre-link candidates?

The production linker matches each frame's detections to every ID's *last valid
position*. During a fish crossing where both fish keep their heading, the two
fish sit near the same midpoint for a few frames and a position-only matcher has
~even odds of swapping them. A constant-velocity prediction (last_pos + v*gap)
separates the two by where they are *going*, which is exactly the discriminating
signal at a pass-through crossing.

This harness:
  - loads a *_best_pre_link.json (the per-frame min-cost geometry candidate),
  - runs baseline (position-only) and velocity-aware forward linkers,
  - runs forward+backward reconciliation for each (relink.py machinery),
  - reports swap-proxy metrics + an ambiguity-triaged review list.

No production files are written. Outputs go under the given --output-dir.
"""

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reconstruction_3d.cross_frame_tracking import (
    _get_points_from_groups,
    apply_cross_frame_tracking,
)
from scripts.reconstruction_3d.reconstruct_from_slp import load_best_pre_link_json
from scripts.reconstruction_3d import relink as relink_mod


# ---------------------------------------------------------------------------
# Velocity-aware forward linker (same I/O shape as apply_cross_frame_tracking).
# ---------------------------------------------------------------------------

def _id_labels_for(n: int) -> List[str]:
    return [f"ID_{i + 1}" for i in range(n)]


def _estimate_velocity(history: deque) -> np.ndarray:
    """Per-frame velocity (mm/frame) from recent (frame, pos) valid points.

    Uses the oldest and newest point in the window: robust, cheap, and avoids
    overfitting jitter. Returns zeros if <2 points.
    """
    if len(history) < 2:
        return np.zeros(3)
    f0, p0 = history[0]
    f1, p1 = history[-1]
    dt = max(1, f1 - f0)
    return (p1 - p0) / dt


def link_velocity_aware(
    best_by_frame: Dict[int, Dict],
    *,
    num_individuals: int = 3,
    max_jump_mm: float | None = None,
    max_jump_hard_cap_mm: float | None = None,
    max_gap_frames: int | None = None,
    vel_window: int = 5,
    max_speed_mm_per_frame: float = 4.0,
) -> Dict[int, Dict]:
    """Like apply_cross_frame_tracking but matches against a velocity-predicted
    position instead of the last valid position.

    Gating stays anchored to the *last valid position* (not the prediction) so a
    runaway velocity estimate cannot widen the gate; only the assignment cost
    uses the prediction. `max_speed_mm_per_frame` clamps the predicted step so a
    pre-crossing acceleration spike does not overshoot.
    """
    n = num_individuals
    id_labels = _id_labels_for(n)
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}
    groups0 = best_by_frame[frames[0]].get("groups", {})
    pts0 = _get_points_from_groups(groups0, n)
    valid = [i for i in range(n) if not np.any(np.isnan(pts0[i]))]
    valid_sorted = sorted(valid, key=lambda i: pts0[i, 0])

    last_valid: Dict[str, tuple[int, np.ndarray] | None] = {lb: None for lb in id_labels}
    history: Dict[str, deque] = {lb: deque(maxlen=vel_window) for lb in id_labels}

    new_groups_0: Dict[str, List[float]] = {}
    id_mapping_0: List[int] = [-1] * n
    for i, lb in enumerate(id_labels):
        if i < len(valid_sorted):
            src = valid_sorted[i]
            pt = pts0[src]
            new_groups_0[lb] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[lb] = (frames[0], pt.copy())
            history[lb].append((frames[0], pt.copy()))
            id_mapping_0[i] = int(src)
        else:
            new_groups_0[lb] = [np.nan, np.nan, np.nan]
    result[frames[0]] = {"groups": new_groups_0, "id_mapping": id_mapping_0,
                         "perm_cam0": list(best_by_frame[frames[0]].get("perm_cam0", [])),
                         "perm_cam1": list(best_by_frame[frames[0]].get("perm_cam1", [])),
                         "perm_cam2": list(best_by_frame[frames[0]].get("perm_cam2", []))}

    for idx in range(1, len(frames)):
        t = frames[idx]
        info = best_by_frame[t]
        pts = _get_points_from_groups(info.get("groups", {}), n)
        valid_curr = [j for j in range(n) if not np.any(np.isnan(pts[j]))]

        if max_gap_frames is not None:
            for lb in id_labels:
                lv = last_valid[lb]
                if lv is not None and (t - lv[0]) > max_gap_frames:
                    last_valid[lb] = None
                    history[lb].clear()

        cost = np.full((n, n), 1e10)
        for i, lb in enumerate(id_labels):
            lv = last_valid[lb]
            if lv is None:
                continue
            fr_prev, pt_prev = lv
            gap = max(1, t - fr_prev)
            vel = _estimate_velocity(history[lb])
            step = vel * gap
            sp = float(np.linalg.norm(step))
            if sp > max_speed_mm_per_frame * gap:
                step = step * (max_speed_mm_per_frame * gap / sp)
            pred = pt_prev + step
            threshold = None
            if max_jump_mm is not None:
                threshold = max_jump_mm * gap
            if max_jump_hard_cap_mm is not None:
                threshold = max_jump_hard_cap_mm if threshold is None else min(threshold, max_jump_hard_cap_mm)
            for j in valid_curr:
                gate_d = float(np.linalg.norm(pt_prev - pts[j]))  # gate on last valid pos
                if threshold is not None and gate_d > threshold:
                    continue
                cost[i, j] = float(np.linalg.norm(pred - pts[j]))  # cost on prediction

        row_ind, col_ind = linear_sum_assignment(cost)
        new_groups: Dict[str, List[float]] = {}
        id_mapping: List[int] = [-1] * n
        matched: set[int] = set()
        for i, lb in enumerate(id_labels):
            m = int(np.where(row_ind == i)[0][0])
            j = int(col_ind[m])
            if cost[i, j] >= 1e10:
                new_groups[lb] = [np.nan, np.nan, np.nan]
            else:
                pt = pts[j]
                new_groups[lb] = [float(pt[0]), float(pt[1]), float(pt[2])]
                last_valid[lb] = (t, pt.copy())
                history[lb].append((t, pt.copy()))
                id_mapping[i] = j
                matched.add(j)

        unmatched = [j for j in valid_curr if j not in matched]
        for i, lb in enumerate(id_labels):
            if last_valid[lb] is not None:
                continue
            if not unmatched:
                break
            j = unmatched.pop(0)
            pt = pts[j]
            new_groups[lb] = [float(pt[0]), float(pt[1]), float(pt[2])]
            last_valid[lb] = (t, pt.copy())
            history[lb].append((t, pt.copy()))
            id_mapping[i] = j

        result[t] = {"groups": new_groups, "id_mapping": id_mapping,
                     "perm_cam0": list(info.get("perm_cam0", [])),
                     "perm_cam1": list(info.get("perm_cam1", [])),
                     "perm_cam2": list(info.get("perm_cam2", []))}
    return result


# ---------------------------------------------------------------------------
# Forward+backward reconciliation, parametrized by linker (mirrors relink.py).
# ---------------------------------------------------------------------------

def relink_fb(best_by_frame, linker, *, num_individuals=3, sigma_tolerance_mm=5.0, **kw):
    n = num_individuals
    fwd = linker(best_by_frame, num_individuals=n, **kw)

    T = max(best_by_frame.keys())
    remapped = {(T - f): v for f, v in best_by_frame.items()}
    bwd_remapped = linker(remapped, num_individuals=n, **kw)
    bwd = {(T - rf): v for rf, v in bwd_remapped.items()}

    frames = sorted(fwd.keys())
    fpos = {t: relink_mod._positions_from_result(fwd[t], n) for t in frames}
    bpos = {t: relink_mod._positions_from_result(bwd[t], n) for t in frames}

    sigma = {}
    for t in frames:
        s, c = relink_mod._best_alignment(fpos[t], bpos[t], n)
        sigma[t] = None if (s is None or c > sigma_tolerance_mm) else s

    pi = tuple(range(n))
    prev_s = None
    prev_t = None
    decisions = []
    out = {}
    labels = _id_labels_for(n)
    for t in frames:
        s = sigma[t]
        if s is not None and prev_s is not None and s != prev_s and prev_t is not None:
            f_jump = float(np.linalg.norm(fpos[t] - fpos[prev_t], axis=1).mean())
            b_jump = float(np.linalg.norm(bpos[t] - bpos[prev_t], axis=1).mean())
            delta = relink_mod._compose(s, relink_mod._inverse(prev_s))
            fwd_swapped = f_jump > b_jump
            if fwd_swapped:
                pi = relink_mod._compose(pi, delta)
            decisions.append({"frame": int(t), "prev_valid_frame": int(prev_t),
                              "side_swapped": "fwd" if fwd_swapped else "bwd",
                              "fwd_jump_mm": f_jump, "bwd_jump_mm": b_jump})
        if s is not None:
            prev_s = s
            prev_t = t
        fp = fpos[t]
        g = {}
        for i, lb in enumerate(labels):
            pt = fp[pi[i]]
            g[lb] = [float(v) for v in pt] if not np.any(np.isnan(pt)) else [np.nan]*3
        out[t] = {"groups": g}
    return out, decisions


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def tracked_to_series(tracked, n):
    labels = _id_labels_for(n)
    series = {lb: [] for lb in labels}
    for t in sorted(tracked.keys()):
        g = tracked[t].get("groups", {})
        for lb in labels:
            pt = g.get(lb)
            if pt is not None and not np.any(np.isnan(np.asarray(pt, float))):
                series[lb].append((t, np.asarray(pt, float)))
    return series


def hard_spike_counts(tracked, n, thresholds=(20.0, 40.0)):
    series = tracked_to_series(tracked, n)
    out = {f">{int(T)}mm": 0 for T in thresholds}
    finite = 0
    for lb, pts in series.items():
        finite += len(pts)
        if len(pts) < 2:
            continue
        fr = np.array([p[0] for p in pts])
        pos = np.stack([p[1] for p in pts])
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        consec = np.diff(fr) == 1
        dd = d[consec]
        for T in thresholds:
            out[f">{int(T)}mm"] += int((dd > T).sum())
    out["finite_rows"] = finite
    return out


def triage_decisions(decisions, *, ambiguous_margin=0.25, trusted_jump_mm=12.0):
    """Split fwd/bwd sigma-events into confidently-resolved vs needs-review.

    needs-review if either:
      - the two sides are close to a coin flip (small relative margin), or
      - even the trusted (smaller-jump) side still teleports (>trusted_jump_mm),
        meaning neither pass produced a smooth continuation -> a real tangle.
    """
    rows = []
    for d in decisions:
        fj, bj = d["fwd_jump_mm"], d["bwd_jump_mm"]
        lo, hi = min(fj, bj), max(fj, bj)
        margin = (hi - lo) / hi if hi > 1e-9 else 0.0
        ambiguous = (margin < ambiguous_margin) or (lo > trusted_jump_mm)
        rows.append({**d, "trusted_jump_mm": lo, "margin": margin,
                     "needs_review": bool(ambiguous)})
    df = pd.DataFrame(rows)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pre_link_json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-individuals", type=int, default=3)
    ap.add_argument("--max-jump-mm", type=float, default=8.0)
    ap.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    ap.add_argument("--max-gap-frames", type=int, default=100)
    ap.add_argument("--vel-window", type=int, default=5)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_individuals
    kw = dict(max_jump_mm=args.max_jump_mm,
              max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
              max_gap_frames=args.max_gap_frames)

    print(f"Loading {args.pre_link_json} ...")
    best = load_best_pre_link_json(Path(args.pre_link_json))
    print(f"Frames: {len(best)}")

    print("\n[baseline] forward link (position-only) ...")
    base_fwd = apply_cross_frame_tracking(best, num_individuals=n, **kw)
    print("[baseline] fwd+bwd reconcile ...")
    _, base_dec = relink_fb(best, apply_cross_frame_tracking, num_individuals=n, **kw)

    print("\n[velocity] forward link (velocity-aware) ...")
    vel_fwd = link_velocity_aware(best, num_individuals=n, vel_window=args.vel_window, **kw)
    print("[velocity] fwd+bwd reconcile ...")
    _, vel_dec = relink_fb(best, lambda b, **k: link_velocity_aware(b, vel_window=args.vel_window, **k),
                           num_individuals=n, **kw)

    base_spikes = hard_spike_counts(base_fwd, n)
    vel_spikes = hard_spike_counts(vel_fwd, n)
    base_tri = triage_decisions(base_dec)
    vel_tri = triage_decisions(vel_dec)

    def summarize(name, spikes, tri):
        nrev = int(tri["needs_review"].sum()) if not tri.empty else 0
        print(f"\n=== {name} ===")
        print(f"  finite rows           : {spikes['finite_rows']}")
        print(f"  hard spikes >20mm/fr  : {spikes['>20mm']}")
        print(f"  hard spikes >40mm/fr  : {spikes['>40mm']}")
        print(f"  fwd/bwd sigma events  : {len(tri)}")
        print(f"  -> confidently resolved: {len(tri) - nrev}")
        print(f"  -> NEEDS REVIEW        : {nrev}")

    summarize("BASELINE (position-only)", base_spikes, base_tri)
    summarize("VELOCITY-AWARE", vel_spikes, vel_tri)

    vel_tri.sort_values("frame").to_csv(out_dir / "velocity_sigma_events.csv", index=False)
    base_tri.sort_values("frame").to_csv(out_dir / "baseline_sigma_events.csv", index=False)
    review = vel_tri[vel_tri["needs_review"]].sort_values("trusted_jump_mm", ascending=False)
    review.to_csv(out_dir / "velocity_review_list.csv", index=False)
    print(f"\nWrote review list ({len(review)} events) to {out_dir/'velocity_review_list.csv'}")


if __name__ == "__main__":
    main()
