"""Machine post-pass on the reviewed 3D tracks: remove the jitter left after reconstruction and review.

Runs per corrected fish (after the reviewer's swaps, flips and geometry edits, so every series is one
biological fish), on every node in 3D:

* **spikes** - a short run (<= ``max_run`` frames) where a node leaves its path and comes back. Found
  where the frame-to-frame acceleration ``|x[t+1] - 2 x[t] + x[t-1]|`` exceeds ``acc_mm``; the shortest
  run whose replacement by a cubic Hermite bridge (end points and end velocities of the good frames on
  either side) brings the acceleration back under ``acc_mm`` is replaced, provided the original
  deviates from the bridge by at least ``floor_mm`` and the frame after the run continues the motion
  before it (end mismatch <= ``return_ratio`` x that deviation: out and back, not a change of
  course). Real turns, tail beats and accelerations follow the bridge closely and are left alone.
* **jumps** - the pectoral node (the fish's position) moving more than ``max_step_mm`` in one frame
  (166 mm/s per mm; 6 mm/frame ~ 1 m/s is beyond a medaka). Mostly the frame where a corrected
  identity swap joins two tracks that were not at the same place. The step is spread over
  ``2 w`` frames (``w = ceil(step / max_step_mm)``, at most ``max_run // 2``) by a bridge; mouth and
  tail keep their offset to the pectoral, so the whole body moves.

The pectoral node is cleaned in world coordinates first; mouth and tail are then cleaned as offsets
from the cleaned pectoral, so swimming and turning of the whole fish do not look like jitter.
Replaced node-frames get ``node_source`` 5 (``NODE_BRIDGED``). The report lists every replaced run,
jumps too long to bridge, and ``jerk`` runs - accelerations above ``acc_mm`` left as they are (a real
escape or strike, or an error longer than ``max_run``). Nothing here looks at the video: a replaced
value is an estimate, not a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

NODE_BRIDGED = 5
REPORT_COLUMNS = ["fish_id", "label", "node", "kind", "frame_start", "frame_end", "size_mm", "repaired"]


@dataclass
class PostParams:
    acc_mm: float = 3.0          # mm/frame^2 (~83 m/s^2 at 166 fps); a 10 Hz tail beat of 4 mm amplitude peaks at 0.6
    floor_mm: float = 1.5        # a run is replaced only if it deviates at least this much from its bridge ...
    return_ratio: float = 0.5    # ... and the motion around it must agree within this x that deviation
    max_run: int = 12            # frames (~72 ms); longer runs are reported, not replaced
    max_step_mm: float = 6.0     # pectoral step per frame beyond which the position jumped
    passes: int = 2


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs where ``mask`` is true."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def acceleration(x: np.ndarray) -> np.ndarray:
    """``(F,)`` norm of the second difference of ``x (F, 3)`` (0 at the ends and next to NaN)."""
    a = np.zeros(len(x))
    if len(x) >= 3:
        a[1:-1] = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1)
    return np.nan_to_num(a)


def hermite_bridge(x: np.ndarray, i: int, j: int) -> np.ndarray | None:
    """Values for frames ``i..j`` from a cubic Hermite curve between ``x[i-1]`` and ``x[j+1]`` with the
    velocities just outside the run; None if the neighbours are missing."""
    if i < 1 or j + 1 >= len(x):
        return None
    p0, p1 = x[i - 1], x[j + 1]
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        return None
    m0 = p0 - x[i - 2] if i >= 2 and np.isfinite(x[i - 2]).all() else p1 - p0
    m1 = x[j + 2] - p1 if j + 2 < len(x) and np.isfinite(x[j + 2]).all() else p1 - p0
    span = j - i + 2                                    # frames from i-1 to j+1
    u = (np.arange(i, j + 1) - (i - 1)) / span
    h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
    h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    m0, m1 = m0 * span, m1 * span                       # velocities per frame -> per unit of u
    return (h00[:, None] * p0 + h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1)


def _try_run(x: np.ndarray, i: int, j: int, p: PostParams) -> tuple[np.ndarray, float] | None:
    b = hermite_bridge(x, i, j)
    if b is None:
        return None
    seg = x[i:j + 1]
    if not np.isfinite(seg).all():
        return None
    excursion = float(np.linalg.norm(seg - b, axis=1).max())
    # Out and back: the frame after the run continues the motion before it (velocity of both sides).
    v_in = x[i - 1] - x[i - 2] if i >= 2 and np.isfinite(x[i - 2]).all() else np.zeros(3)
    v_out = x[j + 2] - x[j + 1] if j + 2 < len(x) and np.isfinite(x[j + 2]).all() else v_in
    mismatch = float(np.linalg.norm(x[j + 1] - (x[i - 1] + (j - i + 2) * (v_in + v_out) / 2)))
    if excursion < p.floor_mm or mismatch > p.return_ratio * excursion:
        return None
    lo, hi = max(0, i - 3), min(len(x), j + 4)
    y = x[lo:hi].copy()
    y[i - lo:j + 1 - lo] = b
    if acceleration(y)[1:-1].max(initial=0.0) > p.acc_mm:
        return None
    return b, excursion


def despike_series(x: np.ndarray, p: PostParams) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Replace out-and-back runs of ``x (F, 3)``. Returns the series, ``(F,)`` replaced mask and
    report rows ``(kind, start, end, size_mm, repaired)`` (indices, not frames)."""
    x = np.asarray(x, float).copy()
    replaced = np.zeros(len(x), dtype=bool)
    rows: list[tuple] = []
    for _ in range(p.passes):
        jerk = acceleration(x) > p.acc_mm
        if not jerk.any():
            break
        cand = jerk | np.r_[jerk[1:], False] | np.r_[False, jerk[:-1]]
        # A plateau (node displaced for several frames) is jerky only where it starts and ends: join
        # jerky runs that together fit in one bridgeable run.
        clusters: list[list[int]] = []
        for s, e in _runs(cand):
            if clusters and e - clusters[-1][0] + 1 <= p.max_run + 2:
                clusters[-1][1] = e
            else:
                clusters.append([s, e])
        changed = False
        for s, e in clusters:
            found = None
            for length in range(1, min(p.max_run, e - s + 1) + 1):
                options = [(i, r) for i in range(s, e - length + 2) if (r := _try_run(x, i, i + length - 1, p))]
                if options:
                    i0, best = min(options, key=lambda o: o[1][1])
                    found = (i0, i0 + length - 1, best)
                    break
            if found is None:
                continue
            i, j, (b, excursion) = found
            x[i:j + 1] = b
            replaced[i:j + 1] = True
            rows.append(("spike", i, j, excursion, True))
            changed = True
        if not changed:
            break
    # Still jerky after the passes: a real fast change (escape, strike) or an error too long to bridge.
    acc = acceleration(x)
    for s, e in _runs(acc > p.acc_mm):
        rows.append(("jerk", s, e, float(acc[s:e + 1].max()), False))
    return x, replaced, rows


def dejump_series(x: np.ndarray, p: PostParams) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Spread single-frame steps longer than ``max_step_mm`` over a few frames each side."""
    x = np.asarray(x, float).copy()
    replaced = np.zeros(len(x), dtype=bool)
    rows: list[tuple] = []
    step = np.r_[np.nan_to_num(np.linalg.norm(np.diff(x, axis=0), axis=1)), 0.0]   # step[t] = x[t+1] - x[t]
    for t in np.flatnonzero(step > p.max_step_mm):
        size = float(step[t])
        w = math.ceil(size / p.max_step_mm)
        while w <= p.max_run // 2:
            i, j = t - w + 1, t + w
            b = hermite_bridge(x, i, j) if i >= 1 else None
            if b is not None:
                y = x.copy()
                y[i:j + 1] = b
                steps = np.linalg.norm(np.diff(y[i - 1:j + 2], axis=0), axis=1)
                if steps.max() <= p.max_step_mm:
                    x = y
                    replaced[i:j + 1] = True
                    rows.append(("jump", i, j, size, True))
                    break
            w += 1
        else:
            rows.append(("jump", t, t + 1, size, False))
    return x, replaced, rows


def postprocess_fish(nodes: np.ndarray, source: np.ndarray, pect: int,
                     p: PostParams | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Clean one fish. ``nodes (F, N, 3)``, ``source (F, N)``; returns cleaned copies and report rows
    (``node`` index, ``kind``, ``start``/``end`` row indices, ``size_mm``, ``repaired``)."""
    p = p or PostParams()
    nodes = np.asarray(nodes, float).copy()
    source = np.asarray(source).copy()
    report: list[dict] = []
    x, spiked, rows = despike_series(nodes[:, pect], p)
    report += [dict(node=pect, kind=k, start=s, end=e, size_mm=z, repaired=r) for k, s, e, z, r in rows]
    x, jumped, rows = dejump_series(x, p)
    report += [dict(node=pect, kind=k, start=s, end=e, size_mm=z, repaired=r) for k, s, e, z, r in rows]
    moved = spiked | jumped
    shift = x - nodes[:, pect]
    for n in range(nodes.shape[1]):
        if n == pect:
            continue
        # The body follows a corrected jump of its pectoral; the offset is then cleaned on its own.
        nodes[jumped, n] += shift[jumped]
        offset = nodes[:, n] - x
        clean, rep, rows = despike_series(offset, p)
        report += [dict(node=n, kind=k, start=s, end=e, size_mm=z, repaired=r) for k, s, e, z, r in rows]
        nodes[:, n] = x + clean
        ok = source[:, n] >= 0
        source[rep & ok, n] = NODE_BRIDGED
    nodes[:, pect] = x
    source[moved & (source[:, pect] >= 0), pect] = NODE_BRIDGED
    return nodes, source, report


def postprocess_table(table: pd.DataFrame, node_names: list[str], pect_name: str = "pectoral",
                      p: PostParams | None = None, clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every fish of a :func:`scripts.review.ledger.apply_events` table (one row per frame and
    fish, sorted by frame). The fish position ``X/Y/Z`` becomes the (cleaned) pectoral node; the
    reconstruction centroid is kept as ``centroid_X/Y/Z``. ``clean=False`` only moves the position.
    Returns the table and the report (frames)."""
    p = p or PostParams()
    out = table.copy()
    for axis in "XYZ":
        out[f"centroid_{axis}"] = table[axis]
    pect = node_names.index(pect_name)
    cols = [[f"{n}_{a}" for a in "XYZ"] for n in node_names]
    report = []
    for fid, idx in (out.groupby("fish_id", sort=False).groups.items() if clean else []):
        idx = out.loc[idx].sort_values("frame").index
        nodes = np.stack([out.loc[idx, c].to_numpy(float) for c in cols], axis=1)        # (F, N, 3)
        source = np.stack([out.loc[idx, f"{n}_source"].to_numpy() for n in node_names], axis=1)
        clean, src, rows = postprocess_fish(nodes, source, pect, p)
        for n, c in enumerate(cols):
            out.loc[idx, c] = clean[:, n]
            out.loc[idx, f"{node_names[n]}_source"] = src[:, n]
        frames = out.loc[idx, "frame"].to_numpy()
        label = str(out.loc[idx[0], "label"]) if len(idx) else ""
        for r in rows:
            report.append({"fish_id": fid, "label": label, "node": node_names[r["node"]], "kind": r["kind"],
                           "frame_start": int(frames[r["start"]]), "frame_end": int(frames[min(r["end"], len(frames) - 1)]),
                           "size_mm": round(float(r["size_mm"]), 2), "repaired": bool(r["repaired"])})
    for a, axis in enumerate("XYZ"):
        out[axis] = out[f"{pect_name}_{axis}"]
    return out, pd.DataFrame(report, columns=REPORT_COLUMNS)
