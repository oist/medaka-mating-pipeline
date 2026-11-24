"""Three-fish (female-centred) states for qc / attack detection.

One *copy* of a frame is (F, focal male, rival male); every frame has two copies (each male is
focal once), so no ordering rule between the males is needed. Helpers:

- ``trio_variables``: pair geometry of F-focal, F-rival and focal-rival plus speeds.
- ``mirror``: reflect a copy so the focal male is on one side of the female (left/right symmetry).
- ``line_rotation``: net rotation of the horizontal F -> focal line over a centred window.
- ``bin_values`` / ``level_code``: binning with an extra "undefined" bin, mixed-radix state codes.
- ``backoff_scores``: P(class | state) shrunk level by level towards coarser states.
- ``backoff_majority``: carry a state -> cluster map to states not seen in training.
- ``alarm_counts`` / ``bouts_found`` / ``threshold_for_alarm_rate``: vectorised bout bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis import pair_state4d as P4

FOCAL_VARS = ("fD", "fdz", "fthF", "fthM", "fsp")
RIVAL_VARS = ("rD", "rdz", "rthF", "rthM", "rsp")
MM_VARS = ("mD", "mthf", "mthr")
ANGLE_VARS = ("fthF", "fthM", "rthF", "rthM", "mthf", "mthr")


def _pair(nodes: np.ndarray, a: str, b: str, min_xy: float) -> tuple[np.ndarray, ...]:
    ia, ib = P4.FISH.index(a), P4.FISH.index(b)
    pa, pb = nodes[:, ia, P4.PEC], nodes[:, ib, P4.PEC]
    ha, hb = nodes[:, ia, P4.HEAD], nodes[:, ib, P4.HEAD]
    d_xy = np.linalg.norm(pb[:, :2] - pa[:, :2], axis=1)
    return (d_xy, pb[:, 2] - pa[:, 2], P4.planar_angle(ha, pa, pb, min_xy), P4.planar_angle(hb, pb, pa, min_xy),
            np.linalg.norm(pb - pa, axis=1))


def trio_variables(nodes: np.ndarray, speed: np.ndarray, focal: str, rival: str,
                   min_xy: float = 1.0) -> dict[str, np.ndarray]:
    """Variables of the copy (F, ``focal``, ``rival``).

    ``nodes`` is ``(T, 3 fish, 3 nodes, xyz)`` in ``P4.FISH`` order, ``speed`` is ``(T, 3)``.
    f* = F -> focal, r* = F -> rival: horizontal distance, height of the male above F, F's heading angle
    to the male (``thF``), the male's heading angle to F (``thM``), the male's speed. m* = focal -> rival:
    3D pectoral distance and the two heading angles. ``Fsp`` = female speed. Angles are NaN when a
    horizontal vector is shorter than ``min_xy`` (e.g. fish on top of each other); distances stay finite.
    """
    out = {}
    for pre, m in (("f", focal), ("r", rival)):
        d, dz, th_f, th_m, _d3 = _pair(nodes, "F", m, min_xy)
        out.update({f"{pre}D": d, f"{pre}dz": dz, f"{pre}thF": th_f, f"{pre}thM": th_m,
                    f"{pre}sp": speed[:, P4.FISH.index(m)]})
    _d, _dz, th_a, th_b, d3 = _pair(nodes, focal, rival, min_xy)
    out.update({"mD": d3, "mthf": th_a, "mthr": th_b, "Fsp": speed[:, P4.FISH.index("F")]})
    return out


def mirror(values: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Reflect copies with ``fthF < 0`` (all signed horizontal angles change sign) so that ``fthF >= 0``.

    A reflection through a vertical plane keeps distances, heights and speeds. Returns the new values
    and the reflected-frame mask."""
    flip = np.asarray(values["fthF"] < 0)
    out = dict(values)
    for k in ANGLE_VARS:
        if k in out:
            out[k] = np.where(flip, -values[k], values[k])
    return out, flip


def line_rotation(nodes: np.ndarray, focal: str, window: int, min_xy: float = 1.0) -> np.ndarray:
    """|net rotation| (degrees) of the horizontal line F -> ``focal`` (pectoral points, world frame) between
    ``t - window/2`` and ``t + window/2``. The angle is unwrapped inside runs where the line is defined, so
    turning one way accumulates and turning back cancels. NaN where the window leaves such a run."""
    if window < 2 or window % 2:
        raise ValueError("window must be an even number of frames >= 2")
    v = nodes[:, P4.FISH.index(focal), P4.PEC, :2] - nodes[:, P4.FISH.index("F"), P4.PEC, :2]
    ok = np.isfinite(v).all(axis=1) & (np.linalg.norm(np.nan_to_num(v), axis=1) >= min_xy)
    ang = np.arctan2(v[:, 1], v[:, 0])
    out = np.full(len(v), np.nan)
    h = window // 2
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a > window:
            u = np.unwrap(ang[a:b])
            out[a + h:b - h] = np.abs(u[window:] - u[:-window])
    return np.degrees(out)


def bin_values(x: np.ndarray, kind: str, edges=None) -> tuple[np.ndarray, int]:
    """Bin index per frame and the number of bins including a last "undefined" bin for NaN.

    ``kind``: ``"lin"`` (``edges`` ascending, first/last may be +-inf), ``"circ4"`` (signed angle in rad:
    front |a| < 45 deg, left 45-135, back > 135, right -135 to -45), ``"half4"`` (|angle| in four 45-deg bins).
    """
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    xs = np.where(fin, x, 0.0)
    if kind == "lin":
        e = np.asarray(edges, dtype=float)
        n = len(e) - 1
        b = np.clip(np.searchsorted(e, xs, side="right") - 1, 0, n - 1)
    elif kind == "circ4":
        n = 4
        deg = np.degrees(xs)
        b = np.select([np.abs(deg) < 45, (deg >= 45) & (deg < 135), np.abs(deg) >= 135], [0, 1, 2], 3)
    elif kind == "half4":
        n = 4
        b = np.clip((np.degrees(np.abs(xs)) // 45).astype(np.int64), 0, 3)
    else:
        raise ValueError(f"unknown kind {kind}")
    return np.where(fin, b, n).astype(np.int64), n + 1


def level_code(bins: dict[str, np.ndarray], widths: dict[str, int], keys) -> np.ndarray:
    """Mixed-radix state code of the variables ``keys`` (order matters only for the code value)."""
    keys = list(keys)
    if np.prod([float(widths[k]) for k in keys]) >= 2 ** 62:
        raise OverflowError("state code does not fit in int64")
    code = np.zeros(len(next(iter(bins.values()))), dtype=np.int64)
    for k in keys:
        code = code * int(widths[k]) + bins[k]
    return code


def check_nested(levels) -> None:
    """Each back-off level must use a subset of the variables of the previous (finer) level."""
    for fine, coarse in zip(levels[:-1], levels[1:]):
        if not set(coarse) <= set(fine):
            raise ValueError(f"level {list(coarse)} is not a subset of {list(fine)}")


def backoff_scores(train_codes: list[np.ndarray], y: np.ndarray, test_codes: list[np.ndarray],
                   prior: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(y = 1 | state) per frame with hierarchical shrinkage.

    ``train_codes`` / ``test_codes``: one code array per level, finest first; levels must be nested (each
    coarser state is a union of finer ones). Only train frames with ``y >= 0`` are counted. The coarsest
    level is shrunk towards the overall rate with ``prior`` pseudo-frames; every finer level towards the
    score of its parent state. Returns (train scores, test scores, finest level with at least ``prior``
    labelled train frames per test frame; ``len(levels)`` if none)."""
    if len(train_codes) != len(test_codes):
        raise ValueError("train and test need the same levels")
    y = np.asarray(y)
    lab = y >= 0
    base = float((y[lab] == 1).mean()) if lab.any() else 0.0
    n_tr = len(y)
    s = np.full(n_tr + len(test_codes[0]), base)
    used = np.full(len(test_codes[0]), len(train_codes), dtype=np.int64)
    for k in range(len(train_codes) - 1, -1, -1):
        allc = np.concatenate([train_codes[k], test_codes[k]])
        u, inv = np.unique(allc, return_inverse=True)
        n = np.bincount(inv[:n_tr][lab], minlength=len(u)).astype(float)
        pos = np.bincount(inv[:n_tr][lab], weights=(y[lab] == 1).astype(float), minlength=len(u))
        s = (pos[inv] + prior * s) / (n[inv] + prior)
        enough = n[inv[n_tr:]] >= prior
        used[enough] = k
    return s[:n_tr], s[n_tr:], used


def backoff_majority(train_levels: list[np.ndarray], train_label: np.ndarray, test_levels: list[np.ndarray],
                     weights: np.ndarray | None = None) -> np.ndarray:
    """Label of each test frame: the label of its finest-level state if that state occurs in training,
    otherwise the most frequent training label (by frames) among training frames sharing its state at the
    next coarser level, and so on; -1 if no level matches. ``train_label`` < 0 is ignored."""
    lab = np.asarray(train_label)
    w = np.ones(len(lab)) if weights is None else np.asarray(weights, float)
    out = np.full(len(test_levels[0]), -1, dtype=np.int64)
    for tr, te in zip(train_levels, test_levels):
        ok = lab >= 0
        df = pd.DataFrame({"code": tr[ok], "lab": lab[ok], "w": w[ok]})
        g = df.groupby(["code", "lab"], sort=False)["w"].sum().reset_index()
        g = g.sort_values(["code", "w", "lab"], ascending=[True, False, True]).drop_duplicates("code")
        best = pd.Series(g["lab"].to_numpy(), index=g["code"].to_numpy())
        todo = out < 0
        if not todo.any():
            break
        hit = best.reindex(te[todo]).to_numpy()
        fill = np.isfinite(hit)
        idx = np.flatnonzero(todo)[fill]
        out[idx] = hit[fill].astype(np.int64)
    return out


def _runs(mask: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    block = np.asarray(block)
    brk = np.ones(len(mask), dtype=bool)
    brk[1:] = (mask[1:] != mask[:-1]) | (block[1:] != block[:-1])
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [len(mask)]])
    keep = mask[starts] if len(mask) else np.zeros(0, bool)
    return starts[keep], ends[keep]


def alarm_counts(y: np.ndarray, pred: np.ndarray, block: np.ndarray) -> tuple[int, int]:
    """Same as ``P4.bout_precision``, vectorised: (alarm runs touching a positive frame, alarm runs touching
    any labelled frame)."""
    a, b = _runs(pred, block)
    y = np.asarray(y)
    cpos = np.concatenate([[0], np.cumsum(y == 1)])
    clab = np.concatenate([[0], np.cumsum(y >= 0)])
    counted = (clab[b] - clab[a]) > 0
    good = (cpos[b] - cpos[a]) > 0
    return int((good & counted).sum()), int(counted.sum())


def bout_table(y: np.ndarray, pred: np.ndarray, block: np.ndarray, min_overlap: float = 0.5,
               y_bout: np.ndarray | None = None) -> pd.DataFrame:
    """One row per bout inside a block: start, end (exclusive), usable frames, predicted fraction, found.

    Bouts are runs of ``y_bout == 1`` (default ``y``). Pass the labels before excluded frames were set to -1
    as ``y_bout`` so that a bout interrupted by excluded frames counts once; only its frames with ``y == 1``
    are scored, and a bout without such frames is dropped."""
    y = np.asarray(y)
    a, b = _runs((y if y_bout is None else np.asarray(y_bout)) == 1, block)
    pos = y == 1
    cpos = np.concatenate([[0], np.cumsum(pos)])
    chit = np.concatenate([[0], np.cumsum(pos & np.asarray(pred, dtype=bool))])
    usable = cpos[b] - cpos[a]
    keep = usable > 0
    a, b, usable = a[keep], b[keep], usable[keep]
    frac = (chit[b] - chit[a]) / np.maximum(usable, 1)
    return pd.DataFrame({"start": a, "end": b, "usable_frames": usable, "pred_fraction": frac,
                         "found": frac >= min_overlap})


def bouts_found(y: np.ndarray, pred: np.ndarray, block: np.ndarray, min_overlap: float = 0.5,
                y_bout: np.ndarray | None = None) -> tuple[int, int]:
    """(found, total) bouts; without ``y_bout`` the same as ``P4.bout_recall``, vectorised."""
    t = bout_table(y, pred, block, min_overlap, y_bout)
    return int(t["found"].sum()), len(t)


def alarm_curve(score: np.ndarray, y: np.ndarray, block: np.ndarray, minutes: float,
                n_grid: int = 300, y_bout: np.ndarray | None = None) -> pd.DataFrame:
    """False-alarm runs per minute and bouts found over a grid of thresholds: score quantiles of labelled
    frames from the median up, spaced logarithmically in the upper tail (rare classes need fine steps there)."""
    score = np.asarray(score, float)
    ok = (np.asarray(y) >= 0) & np.isfinite(score)
    q = 1.0 - np.logspace(np.log10(0.5), -6, n_grid)
    grid = np.unique(np.quantile(score[ok], q))
    rows = []
    for t in grid:
        pred = score >= t
        good, counted = alarm_counts(y, pred, block)
        f, n = bouts_found(y, pred, block, y_bout=y_bout)
        rows.append({"threshold": float(t), "false_alarms_per_min": (counted - good) / max(minutes, 1e-9),
                     "bouts_found": f, "bouts": n})
    return pd.DataFrame(rows)


def threshold_for_alarm_rate(score: np.ndarray, y: np.ndarray, block: np.ndarray, minutes: float,
                             target: float, n_grid: int = 300) -> float:
    """Lowest threshold on the grid of ``alarm_curve`` whose false-alarm rate is at most ``target``
    (the highest threshold if none is)."""
    return threshold_from_curve(alarm_curve(score, y, block, minutes, n_grid), target)


def threshold_from_curve(curve: pd.DataFrame, target: float) -> float:
    """Lowest threshold of an ``alarm_curve`` whose false-alarm rate is at most ``target`` (else the highest)."""
    ok = curve[curve.false_alarms_per_min <= target]
    return float(ok.threshold.min()) if len(ok) else float(curve.threshold.max())
