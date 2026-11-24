"""Four-variable pair state ``(D_xy, dz, theta_a, theta_b)`` for the pair-maneuver notebook
(``archive/notebooks/experiment_workflows/20261107_KP_pair_state4d_infomap.ipynb``).

Extends the (D_PP, theta_1, theta_2) state of O'Shaughnessy et al., PRX Life 2, 043006 (2024)
with a signed vertical separation, because in medaka the vertical pec-pec separation carries
more variance than the horizontal one.

Conventions (pair ``a -> b``):

* ``D_xy``  horizontal distance between the two pectoral points (mm).
* ``dz``    ``z_a - z_b`` of the pectoral points (mm); positive = ``a`` above ``b``.
* ``theta`` signed horizontal angle from (own pec -> other pec) to (own pec -> own head),
  counter-clockwise positive seen from above (z up); 0 = facing the other fish, +-pi = facing away.

Swapping the pair (``a <-> b``) maps ``(D_xy, dz, theta_a, theta_b)`` to
``(D_xy, -dz, theta_b, theta_a)``. With dz edges symmetric about 0 and the same theta edges for
both fish this is an exact permutation of the joint states (``swap_state_map``).

Optional speed variables (v3 notebook ``20261108_KP_pair_state6d_speed.ipynb``): ``speed_a``,
``speed_b`` = 3D speed of each pectoral point (``fish_speed``); a swap also exchanges them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FISH = ("F", "M1", "M2")
HEAD, PEC = 0, 1
VARIABLES = ("D_xy", "dz", "theta_a", "theta_b")
VARIABLES_SPEED = VARIABLES + ("speed_a", "speed_b")


# ---------------------------------------------------------------- state variables
def planar_angle(head: np.ndarray, pec: np.ndarray, other: np.ndarray, min_xy: float) -> np.ndarray:
    """Signed XY angle from (pec -> other) to (pec -> head); NaN if either XY vector is shorter than ``min_xy``."""
    h = head[:, :2] - pec[:, :2]
    r = other[:, :2] - pec[:, :2]
    ang = np.arctan2(r[:, 0] * h[:, 1] - r[:, 1] * h[:, 0], np.sum(r * h, axis=1))
    bad = ~(np.linalg.norm(h, axis=1) >= min_xy) | ~(np.linalg.norm(r, axis=1) >= min_xy)
    ang[bad] = np.nan
    return ang


def pair_variables(nodes: np.ndarray, a: str, b: str, min_xy: float = 1.0) -> np.ndarray:
    """``(T, 4)`` array of ``VARIABLES`` for the ordered pair ``a -> b``; rows with any NaN are unusable.

    ``nodes`` is ``(T, 3 fish, 3 nodes, xyz)`` in the order ``FISH`` x (head, pec, tail).
    """
    ia, ib = FISH.index(a), FISH.index(b)
    pa, pb = nodes[:, ia, PEC], nodes[:, ib, PEC]
    ha, hb = nodes[:, ia, HEAD], nodes[:, ib, HEAD]
    d_xy = np.linalg.norm(pb[:, :2] - pa[:, :2], axis=1)
    dz = pa[:, 2] - pb[:, 2]
    out = np.column_stack([d_xy, dz, planar_angle(ha, pa, pb, min_xy), planar_angle(hb, pb, pa, min_xy)])
    out[~(d_xy >= min_xy)] = np.nan
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def swap_variables(values: np.ndarray) -> np.ndarray:
    """``(D_xy, dz, theta_a, theta_b[, speed_a, speed_b])`` of ``a -> b`` as seen from ``b -> a``."""
    cols = [values[:, 0], -values[:, 1], values[:, 3], values[:, 2]]
    if values.shape[1] == 6:
        cols += [values[:, 5], values[:, 4]]
    return np.column_stack(cols)


def fish_speed(points: np.ndarray, fps: float, half_window: int) -> np.ndarray:
    """3D speed (mm/s) of one point track ``(T, 3)``.

    Central difference over ``+-half_window`` frames; where that is undefined (gap or end of the
    track) the central difference over +-1 frame, then the forward and the backward 1-frame
    difference are used. NaN only for frames without any finite neighbour.
    """
    p = np.asarray(points, dtype=float)
    T = len(p)
    out = np.full(T, np.nan)

    def diff(lo_off, hi_off):
        v = np.full(T, np.nan)
        lo = max(0, -lo_off)
        hi = T - max(0, hi_off)
        if hi <= lo:
            return v
        idx = np.arange(lo, hi)
        d = np.linalg.norm(p[idx + hi_off] - p[idx + lo_off], axis=1)
        v[idx] = d / ((hi_off - lo_off) / float(fps))
        return v

    h = max(1, int(half_window))
    for lo_off, hi_off in ((-h, h), (-1, 1), (0, 1), (-1, 0)):
        miss = ~np.isfinite(out)
        if not miss.any():
            break
        out[miss] = diff(lo_off, hi_off)[miss]
    out[~np.isfinite(p).all(axis=1)] = np.nan
    return out


# ---------------------------------------------------------------- bin edges and states
def symmetric_dz_edges(abs_dz: np.ndarray, n_bins: int) -> np.ndarray:
    """Edges symmetric about 0: inner edges at quantiles of |dz|, outer bins open (+-inf).

    ``n_bins`` must be even; each half gets ``n_bins / 2`` bins with equal shares of |dz|.
    """
    if n_bins < 2 or n_bins % 2:
        raise ValueError("n_bins for dz must be even and >= 2")
    v = np.abs(abs_dz[np.isfinite(abs_dz)])
    if len(v) == 0:
        raise ValueError("no finite dz values")
    half = n_bins // 2
    q = np.quantile(v, np.arange(1, half) / half) if half > 1 else np.zeros(0)
    return np.concatenate([[-np.inf], -q[::-1], [0.0], q, [np.inf]])


def make_edges(d_max: float, n_bins_d: int, d_overflow: bool, dz_edges: np.ndarray, n_bins_theta: int) -> list:
    """Edges of the four variables. ``d_overflow`` adds one open bin ``[d_max, inf)``."""
    d = np.linspace(0.0, float(d_max), int(n_bins_d) + 1)
    if d_overflow:
        d = np.concatenate([d, [np.inf]])
    theta = np.linspace(-np.pi, np.pi, int(n_bins_theta) + 1)
    return [d, np.asarray(dz_edges, dtype=float), theta, theta.copy()]


def digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index 0..n-1 (top edge in the last bin); outside or NaN -> -1. Works with infinite edges."""
    out = np.full(len(values), -1, dtype=np.int64)
    inside = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    out[inside] = np.clip(np.searchsorted(edges, values[inside], side="right") - 1, 0, len(edges) - 2)
    return out


def digitize_symmetric(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Like ``digitize`` for edges symmetric about 0, but bins |v| and mirrors by sign, so that
    ``bin(-v) == n - 1 - bin(v)`` holds exactly (values on an edge included; 0 counts as positive)."""
    e = np.asarray(edges, dtype=float)
    n = len(e) - 1
    if n % 2 or not np.allclose(e, -e[::-1]):
        raise ValueError("edges must be symmetric about 0 with an even number of bins")
    half = n // 2
    pos = digitize(np.abs(values), e[half:])
    out = np.where(values >= 0, half + pos, half - 1 - pos)
    out[(pos < 0) | ~np.isfinite(values)] = -1
    return out.astype(np.int64)


def encode(values: np.ndarray, edges: list) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-variable bins ``(T, 4)``, joint state (first variable fastest; -1 if any bin is -1), widths.
    ``dz`` (column 1) is binned with ``digitize_symmetric``."""
    widths = [len(e) - 1 for e in edges]
    bins = np.column_stack([digitize_symmetric(values[:, j], np.asarray(e)) if j == 1
                            else digitize(values[:, j], np.asarray(e)) for j, e in enumerate(edges)])
    state = np.zeros(len(bins), dtype=np.int64)
    scale = 1
    for j, w in enumerate(widths):
        state += np.where(bins[:, j] < 0, 0, bins[:, j]) * scale
        scale *= w
    state[(bins < 0).any(axis=1)] = -1
    return bins, state, widths


def decode(state: np.ndarray, widths: list[int]) -> np.ndarray:
    state = np.asarray(state, dtype=np.int64)
    out = np.zeros((len(state), len(widths)), dtype=np.int64)
    rest = state.copy()
    for j, w in enumerate(widths):
        out[:, j] = rest % w
        rest //= w
    return out


def joint(bins: np.ndarray, widths: list[int]) -> np.ndarray:
    state = np.zeros(len(bins), dtype=np.int64)
    scale = 1
    for j, w in enumerate(widths):
        state += bins[:, j] * scale
        scale *= w
    return state


def swap_state_map(widths: list[int], n_extra: int = 0) -> np.ndarray:
    """``map[s]`` = state of the swapped pair. Needs symmetric dz edges and equal widths for the
    a / b variables (theta, and speed when there are 6 variables). ``n_extra`` states appended after
    the joint states (e.g. a "contact" state) map to themselves."""
    pairs = [(2, 3)] + ([(4, 5)] if len(widths) >= 6 else [])
    for i, j in pairs:
        if widths[i] != widths[j]:
            raise ValueError("the a and b variables need the same number of bins")
    n = int(np.prod(widths))
    b = decode(np.arange(n), widths)
    sw = b.copy()
    sw[:, 1] = widths[1] - 1 - b[:, 1]
    for i, j in pairs:
        sw[:, i], sw[:, j] = b[:, j], b[:, i]
    return np.concatenate([joint(sw, widths), n + np.arange(int(n_extra))])


def bin_centers(edges: np.ndarray, open_width: float | None = None) -> np.ndarray:
    """Centres of the bins; an open outer bin gets its finite edge +- ``open_width / 2``
    (``open_width`` defaults to the width of the neighbouring finite bin)."""
    e = np.asarray(edges, dtype=float)
    c = 0.5 * (e[:-1] + e[1:])
    if not np.isfinite(e[0]):
        w = open_width if open_width is not None else (e[2] - e[1] if len(e) > 2 else 1.0)
        c[0] = e[1] - w / 2
    if not np.isfinite(e[-1]):
        w = open_width if open_width is not None else (e[-2] - e[-3] if len(e) > 2 else 1.0)
        c[-1] = e[-2] + w / 2
    return c


# ---------------------------------------------------------------- cluster helpers
def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    a = np.asarray(angles, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(a) & np.isfinite(w)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    return float(np.arctan2(np.sum(w[ok] * np.sin(a[ok])), np.sum(w[ok] * np.cos(a[ok]))))


def mirror_partners(labels: np.ndarray, swap_map: np.ndarray, state_frames: np.ndarray) -> pd.DataFrame:
    """For every cluster, the cluster that receives most of its frame mass after swapping a <-> b.

    ``labels[s]`` = cluster of state ``s`` (-1 = none). Columns: ``cluster, partner, overlap``
    (share of the cluster's frames whose swapped state is in ``partner``), ``mutual`` (partner's
    partner is this cluster), ``self_symmetric`` (partner == cluster) and ``merged`` (the smaller
    id of a mutual pair, else the cluster itself).
    """
    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1 if (labels >= 0).any() else 0
    partner = np.full(k, -1, dtype=np.int64)
    overlap = np.zeros(k)
    for c in range(k):
        st = np.flatnonzero(labels == c)
        w = state_frames[st].astype(float)
        target = labels[swap_map[st]]
        ok = target >= 0
        if not ok.any() or w[ok].sum() <= 0:
            continue
        mass = np.bincount(target[ok], weights=w[ok], minlength=k)
        partner[c] = int(np.argmax(mass))
        overlap[c] = float(mass[partner[c]] / max(w.sum(), 1e-12))
    mutual = np.array([partner[c] >= 0 and partner[partner[c]] == c for c in range(k)], dtype=bool)
    merged = np.array([min(c, partner[c]) if mutual[c] else c for c in range(k)], dtype=np.int64)
    return pd.DataFrame({"cluster": np.arange(k), "partner": partner, "overlap": overlap, "mutual": mutual,
                         "self_symmetric": partner == np.arange(k), "merged": merged})


# ---------------------------------------------------------------- BORIS labels per pair
def class_of(behavior: str, classes: dict) -> str | None:
    """Class name of a BORIS behaviour (case-insensitive), ``None`` if it is not mapped."""
    b = str(behavior).strip().casefold()
    for cls, names in classes.items():
        if b in {str(n).strip().casefold() for n in names}:
            return cls
    return None


def pair_frame_labels(bouts: pd.DataFrame, a: str, b: str, frames: np.ndarray, classes: dict,
                      directed: bool) -> np.ndarray:
    """One label per frame of ``frames`` for the ordered pair ``a -> b``.

    A bout belongs to the pair when ``who`` is ``X->Y`` or ``X<->Y`` with {X, Y} = {a, b}.
    Labels: ``"none"`` (no bout of the pair), ``"unknown"`` (only behaviours outside ``classes``; not the same as "none"),
    or a class name. Overlapping classes: the one listed first in ``classes`` wins.
    With ``directed`` the class gets `` a->b``, `` b->a`` or `` a<->b`` appended (roles, not names).
    """
    frames = np.asarray(frames, dtype=np.int64)
    n, f0 = len(frames), int(frames[0])
    if not np.all(np.diff(frames) == 1):
        raise ValueError("frames must be contiguous")
    order = list(classes)
    rank = np.full(n, len(order) + 1, dtype=np.int64)  # lower = higher priority
    lab = np.full(n, "none", dtype=object)
    for r in bouts.itertuples():
        who = str(r.who)
        sep = "<->" if "<->" in who else ("->" if "->" in who else None)
        if sep is None:
            continue
        x, y = (s.strip() for s in who.split(sep))
        if {x, y} != {a, b}:
            continue
        s, e = max(int(r.start) - f0, 0), min(int(r.end) - f0 + 1, n)
        if e <= s:
            continue
        cls = class_of(r.behavior, classes)
        if cls is None:
            sl = slice(s, e)
            lab[sl] = np.where(lab[sl] == "none", "unknown", lab[sl])
            continue
        name = cls
        if directed:
            name += " a<->b" if sep == "<->" else (" a->b" if x == a else " b->a")
        pr = order.index(cls)
        win = rank[s:e] > pr
        lab[s:e][win] = name
        rank[s:e][win] = pr
    return lab


def fit_label_map(clusters: np.ndarray, labels: np.ndarray, min_purity: float,
                  ignore=("none", "unknown")) -> pd.DataFrame:
    """Majority label per cluster over labelled frames (``ignore`` excluded); ``unassigned`` below ``min_purity``."""
    keep = ~np.isin(labels, list(ignore)) & (clusters >= 0)
    df = pd.DataFrame({"cluster": clusters[keep], "label": labels[keep]})
    rows = []
    for c, part in df.groupby("cluster"):
        vc = part["label"].value_counts()
        purity = float(vc.iloc[0] / vc.sum())
        rows.append({"cluster": int(c), "n_labelled": int(vc.sum()), "top_label": vc.index[0], "purity": purity,
                     "mapped_label": vc.index[0] if purity >= min_purity else "unassigned"})
    return pd.DataFrame(rows, columns=["cluster", "n_labelled", "top_label", "purity", "mapped_label"])


def enrichment(clusters: np.ndarray, labels: np.ndarray, cluster_ids, classes) -> pd.DataFrame:
    """``P(class | cluster) / P(class)`` over frames with a cluster; rows = clusters, columns = classes."""
    ok = clusters >= 0
    c, l = clusters[ok], labels[ok]
    base = {k: float(np.mean(l == k)) for k in classes}
    out = pd.DataFrame(index=list(cluster_ids), columns=list(classes), dtype=float)
    for cid in cluster_ids:
        m = c == cid
        for k in classes:
            out.loc[cid, k] = (float(np.mean(l[m] == k)) / base[k]) if (m.any() and base[k] > 0) else np.nan
    return out


# ---------------------------------------------------------------- explicit edges
def make_edges_list(d_edges, d_overflow: bool, dz_edges: np.ndarray, n_bins_theta: int,
                    speed_edges=None) -> list:
    """Like ``make_edges`` but with explicit (possibly non-uniform) ``D_xy`` edges in mm.
    With ``speed_edges`` two more variables (speed_a, speed_b) share those edges."""
    d = np.asarray(d_edges, dtype=float)
    if d[0] != 0 or np.any(np.diff(d) <= 0):
        raise ValueError("D_xy edges must start at 0 and increase")
    if d_overflow:
        d = np.concatenate([d, [np.inf]])
    theta = np.linspace(-np.pi, np.pi, int(n_bins_theta) + 1)
    out = [d, np.asarray(dz_edges, dtype=float), theta, theta.copy()]
    if speed_edges is not None:
        s = np.asarray(speed_edges, dtype=float)
        if np.any(np.diff(s) <= 0):
            raise ValueError("speed edges must increase")
        out += [s, s.copy()]
    return out


def quantile_edges(values: np.ndarray, quantiles) -> np.ndarray:
    """Edges ``[0, q_1, ..., q_k, inf]`` at the given quantiles of the finite non-negative ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise ValueError("no finite values")
    q = np.quantile(v, list(quantiles))
    return np.concatenate([[0.0], q, [np.inf]])


# ---------------------------------------------------------------- v4: trajectory smoothing and extra variables
VARIABLES_EXTRA = ("bearing_rate", "approach", "turn_a", "turn_b", "bend_a", "bend_b", "pitch_a", "pitch_b")
TAIL = 2


def smooth_tracks(nodes: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time inside each finite run of each fish; no frame is dropped.

    ``nodes`` is ``(T, fish, node, xyz)``. Runs shorter than ``window`` are left unsmoothed;
    ``window`` <= 1 returns a copy. Smoothing never crosses a NaN frame.
    """
    from scipy.signal import savgol_filter

    out = np.array(nodes, dtype=float, copy=True)
    if window <= 1:
        return out
    if window <= order or window % 2 == 0:
        raise ValueError("window must be odd and larger than the polynomial order")
    for f in range(out.shape[1]):
        ok = np.isfinite(out[:, f]).all(axis=tuple(range(1, out[:, f].ndim)))
        d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= window:
                seg = out[a:b, f].reshape(b - a, -1)
                out[a:b, f] = savgol_filter(seg, window, order, axis=0, mode="interp").reshape(out[a:b, f].shape)
    return out


def series_rate(x: np.ndarray, fps: float, half_window: int, circular: bool = False) -> np.ndarray:
    """Time derivative (units per second) of a scalar series, with the fallbacks of ``fish_speed``:
    central difference over ``+-half_window`` frames, then +-1, then forward, then backward.
    ``circular``: ``x`` is an angle in radians and differences are wrapped to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    out = np.full(T, np.nan)

    def diff(lo_off, hi_off):
        v = np.full(T, np.nan)
        lo, hi = max(0, -lo_off), T - max(0, hi_off)
        if hi <= lo:
            return v
        idx = np.arange(lo, hi)
        d = x[idx + hi_off] - x[idx + lo_off]
        if circular:
            d = np.angle(np.exp(1j * d))
        v[idx] = d / ((hi_off - lo_off) / float(fps))
        return v

    h = max(1, int(half_window))
    for lo_off, hi_off in ((-h, h), (-1, 1), (0, 1), (-1, 0)):
        miss = ~np.isfinite(out)
        if not miss.any():
            break
        out[miss] = diff(lo_off, hi_off)[miss]
    out[~np.isfinite(x)] = np.nan
    return out


def encode_specs(values: np.ndarray, edges: list, kinds: list) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """General ``encode``: per column a kind ``"lin"`` (``digitize``), ``"sym"`` (``digitize_symmetric``)
    or ``"abs"`` (``digitize`` of the absolute value). Returns bins, joint state (-1 if any bin is -1), widths."""
    if not (len(edges) == len(kinds) == values.shape[1]):
        raise ValueError("one edge array and one kind per column")
    cols = []
    for j, (e, k) in enumerate(zip(edges, kinds)):
        e = np.asarray(e, dtype=float)
        if k == "lin":
            cols.append(digitize(values[:, j], e))
        elif k == "sym":
            cols.append(digitize_symmetric(values[:, j], e))
        elif k == "abs":
            cols.append(digitize(np.abs(values[:, j]), e))
        else:
            raise ValueError(f"unknown kind {k}")
    bins = np.column_stack(cols)
    widths = [len(e) - 1 for e in edges]
    state = joint(np.where(bins < 0, 0, bins), widths)
    state[(bins < 0).any(axis=1)] = -1
    return bins, state, widths


def swap_state_map_specs(widths: list[int], flip: list[int], pairs: list[tuple[int, int]],
                         n_extra: int = 0) -> np.ndarray:
    """``map[s]`` = joint state of the swapped pair: columns in ``flip`` are mirrored (bin -> n - 1 - bin,
    for signed variables binned symmetrically), each ``(i, j)`` in ``pairs`` is exchanged (a / b variables),
    all other columns are pair quantities that do not change. Extra states map to themselves."""
    for i, j in pairs:
        if widths[i] != widths[j]:
            raise ValueError("the a and b variables need the same number of bins")
    n = int(np.prod(widths))
    b = decode(np.arange(n), widths)
    sw = b.copy()
    for i in flip:
        sw[:, i] = widths[i] - 1 - b[:, i]
    for i, j in pairs:
        sw[:, i], sw[:, j] = b[:, j], b[:, i]
    return np.concatenate([joint(sw, widths), n + np.arange(int(n_extra))])


def _xy_angle(v: np.ndarray, min_xy: float) -> np.ndarray:
    ang = np.arctan2(v[:, 1], v[:, 0])
    ang[~(np.linalg.norm(v[:, :2], axis=1) >= min_xy)] = np.nan
    return ang


def extra_variables(nodes: np.ndarray, a: str, b: str, fps: float, half_window: int,
                    min_xy: float = 1.0) -> np.ndarray:
    """``(T, 8)`` array of ``VARIABLES_EXTRA`` for the ordered pair ``a -> b``.

    * ``bearing_rate``  |rotation rate| (rad/s) of the horizontal pec -> pec line (circling); the same for ``b -> a``.
    * ``approach``      -d/dt of the 3D pec-pec distance (mm/s); positive = closing in; the same for ``b -> a``.
    * ``turn_*``        |yaw rate| (rad/s) of the fish's horizontal heading (pec -> head).
    * ``bend_*``        pi minus the head-pec-tail angle (rad); 0 = straight body.
    * ``pitch_*``       elevation of pec -> head above the horizontal (rad); positive = head up.

    Angles need horizontal vectors of at least ``min_xy`` mm (else NaN, as for theta).
    """
    ia, ib = FISH.index(a), FISH.index(b)
    pa, pb = nodes[:, ia, PEC], nodes[:, ib, PEC]
    line = pb - pa
    bearing = series_rate(_xy_angle(line, min_xy), fps, half_window, circular=True)
    approach = -series_rate(np.linalg.norm(line, axis=1), fps, half_window)
    cols = [np.abs(bearing), approach]
    turn, bend, pitch = [], [], []
    for i in (ia, ib):
        head, pec, tail = nodes[:, i, HEAD], nodes[:, i, PEC], nodes[:, i, TAIL]
        h = head - pec
        turn.append(np.abs(series_rate(_xy_angle(h, min_xy), fps, half_window, circular=True)))
        u, w = h, tail - pec
        nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.sum(u * w, axis=1) / (nu * nw)
            bend.append(np.pi - np.arccos(np.clip(cos, -1.0, 1.0)))
            pitch.append(np.arcsin(np.clip(h[:, 2] / nu, -1.0, 1.0)))
    return np.column_stack(cols + turn + bend + pitch)


# ---------------------------------------------------------------- detection of one behaviour class
def base_class(labels: np.ndarray) -> np.ndarray:
    """Drop a direction suffix: ``"att a->b"`` -> ``"att"``."""
    return np.array([str(x).split(" ")[0] for x in labels], dtype=object)


def detection_target(labels: np.ndarray, positive: str, known=("none",)) -> np.ndarray:
    """1 = ``positive``, 0 = negative (``"none"`` or any other class), -1 = excluded (``"unknown"``)."""
    b = base_class(labels)
    y = np.zeros(len(b), dtype=np.int8)
    y[b == positive] = 1
    y[b == "unknown"] = -1
    return y


def cluster_scores(clusters: np.ndarray, y: np.ndarray, n_clusters: int, prior_frames: float = 50.0) -> np.ndarray:
    """Smoothed ``P(positive | cluster)`` from frames with ``y >= 0`` and a cluster.

    Each cluster is shrunk towards the overall rate with ``prior_frames`` pseudo-frames, so that
    clusters with few labelled frames do not get extreme scores. Clusters without data get the rate.
    """
    ok = (y >= 0) & (clusters >= 0)
    base = float(y[ok].mean()) if ok.any() else 0.0
    n = np.bincount(clusters[ok], minlength=n_clusters).astype(float)
    pos = np.bincount(clusters[ok], weights=(y[ok] == 1).astype(float), minlength=n_clusters)
    den = n + prior_frames
    return np.where(den > 0, (pos + prior_frames * base) / np.where(den > 0, den, 1.0), base)


def frame_scores(clusters: np.ndarray, scores: np.ndarray, default: float) -> np.ndarray:
    out = np.full(len(clusters), np.nan)
    ok = (clusters >= 0) & (clusters < len(scores))
    out[ok] = scores[clusters[ok]]
    out[~ok] = default
    return out


def rolling_mean_blocks(x: np.ndarray, block: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average of ``x`` inside each block (NaN ignored); ``window`` <= 1 returns ``x``."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.groupby(np.asarray(block)).transform(
        lambda v: v.rolling(int(window), center=True, min_periods=1).mean()).to_numpy()


def best_f1_threshold(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold on ``score`` that maximises F1 for ``y`` (1 / 0; -1 ignored). Returns (threshold, F1)."""
    from sklearn.metrics import precision_recall_curve

    ok = (y >= 0) & np.isfinite(score)
    if not ok.any() or y[ok].max() < 1 or y[ok].min() > 0:
        return float("nan"), float("nan")
    p, r, t = precision_recall_curve(y[ok], score[ok])
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def bout_recall(y: np.ndarray, pred: np.ndarray, block: np.ndarray, min_overlap: float = 0.5) -> tuple[int, int]:
    """Bouts = runs of ``y == 1`` inside one block; a bout is found if >= ``min_overlap`` of its frames are predicted."""
    y = np.asarray(y)
    brk = np.ones(len(y), dtype=bool)
    brk[1:] = (y[1:] != y[:-1]) | (np.asarray(block)[1:] != np.asarray(block)[:-1])
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [len(y)]])
    found = total = 0
    for a, b in zip(starts, ends):
        if y[a] != 1:
            continue
        total += 1
        found += int(np.mean(pred[a:b]) >= min_overlap)
    return found, total


def with_nostate_cluster(clusters: np.ndarray, has_state: np.ndarray, n_clusters: int) -> np.ndarray:
    """Clusters with one extra id ``n_clusters`` for frames without a state (so they get their own
    fitted score instead of the overall rate). Frames with a state but no cluster stay -1."""
    out = np.asarray(clusters, dtype=np.int64).copy()
    out[~np.asarray(has_state, dtype=bool)] = int(n_clusters)
    return out


def decide_classes(scores: dict, thresholds: dict, other: str = "other") -> np.ndarray:
    """Per-frame class from several detectors: the class whose ``score / threshold`` is largest
    among those at or above their threshold; ``other`` if none is."""
    names = list(scores)
    if not names:
        raise ValueError("no detectors")
    n = len(next(iter(scores.values())))
    ratio = np.full((len(names), n), -np.inf)
    for i, c in enumerate(names):
        s = np.asarray(scores[c], dtype=float)
        t = float(thresholds[c])
        if t > 0 and np.isfinite(t):
            r = s / t
            ratio[i] = np.where(np.isfinite(r), r, -np.inf)
    best = np.argmax(ratio, axis=0)
    out = np.array(names, dtype=object)[best]
    out[ratio[best, np.arange(n)] < 1.0] = other
    return out


def bout_precision(y: np.ndarray, pred: np.ndarray, block: np.ndarray) -> tuple[int, int]:
    """Predicted bouts = runs of ``pred`` inside one block; one is correct if it overlaps a frame with
    ``y == 1``. Runs lying only on excluded frames (``y == -1``) are not counted. Returns (correct, counted)."""
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(y)
    block = np.asarray(block)
    brk = np.ones(len(pred), dtype=bool)
    brk[1:] = (pred[1:] != pred[:-1]) | (block[1:] != block[:-1])
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [len(pred)]])
    good = total = 0
    for a, b in zip(starts, ends):
        if not pred[a]:
            continue
        seg = y[a:b]
        if (seg >= 0).any():
            total += 1
            good += int((seg == 1).any())
    return good, total


def paper_tau(taus, largest_fraction) -> int:
    """tau just before the largest increase of the largest-cluster size (rule of Fig. S11a)."""
    taus = list(taus)
    f = np.asarray(largest_fraction, dtype=float)
    if len(taus) < 2:
        return int(taus[0])
    return int(taus[int(np.argmax(np.diff(f)))])


def effective_clusters(frame_counts: np.ndarray) -> float:
    """exp(entropy) of the cluster occupancy: the number of equally used clusters with the same entropy."""
    p = np.asarray(frame_counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))
