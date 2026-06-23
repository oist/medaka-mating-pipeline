from __future__ import annotations

"""Metric panel for the identity benchmark.

All methods are reduced to a common **AssignmentTable**: a long DataFrame with one
row per (frame, fish slot) the method commits to, columns::

    frame   int
    fish_id int    canonical slot 0..n-1, or -1 = unknown/mix/flagged
    X,Y,Z   float  3D position for that slot (NaN if the method is label-only)
    conf    float  method confidence in [0,1] (NaN if none)
    flag    str    '', 'hardcore', 'low_conf', 'fp', 'review', ...

Every function here is pure (numpy / pandas in, dict/float out) so it is unit
tested on synthetic tables without touching disk. GT-free metrics never look at
labels; GT-gated metrics live in ``gt_*`` and return NaN when labels are absent.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ASSIGN_COLS = ["frame", "fish_id", "X", "Y", "Z", "conf", "flag", "method"]


def _window_frames(df: pd.DataFrame, start: int, end: int) -> np.ndarray:
    return np.arange(int(start), int(end) + 1)


def coverage(df: pd.DataFrame, start: int, end: int, n_fish: int = 3) -> float:
    """Fraction of window frames where all ``n_fish`` slots carry a confident
    (non -1, finite-position) assignment."""
    w = _window_frames(df, start, end)
    if len(w) == 0:
        return float("nan")
    sub = df[(df["frame"] >= start) & (df["frame"] <= end)]
    good = sub[(sub["fish_id"] >= 0)]
    if {"X", "Y", "Z"}.issubset(good.columns) and good[["X", "Y", "Z"]].notna().any().any():
        good = good[good[["X", "Y", "Z"]].notna().all(axis=1)]
    per_frame = good.groupby("frame")["fish_id"].nunique()
    full = (per_frame >= n_fish).reindex(w, fill_value=False)
    return float(full.mean())


def unknown_frac(df: pd.DataFrame, start: int, end: int, n_fish: int = 3) -> float:
    """Fraction of expected (frame x slot) assignments that are unknown/flagged.

    Denominator is ``n_window_frames * n_fish`` (the ideal where every slot is
    confidently assigned every frame), so a method that simply omits a fish counts
    as unknown there too."""
    w = _window_frames(df, start, end)
    if len(w) == 0:
        return float("nan")
    sub = df[(df["frame"] >= start) & (df["frame"] <= end)]
    confident = sub[(sub["fish_id"] >= 0) & (sub["flag"].fillna("") == "")]
    if {"X", "Y", "Z"}.issubset(confident.columns):
        ok = confident[["X", "Y", "Z"]].notna().all(axis=1) | confident[["X", "Y", "Z"]].isna().all(axis=1)
        confident = confident[ok]
    n_confident = confident.groupby("frame")["fish_id"].nunique().clip(upper=n_fish).sum()
    return float(1.0 - n_confident / (len(w) * n_fish))


def exactly_three(df: pd.DataFrame, start: int, end: int, n_fish: int = 3) -> dict:
    sub = df[(df["frame"] >= start) & (df["frame"] <= end)]
    ids = sorted(int(i) for i in sub["fish_id"].unique() if i >= 0)
    return {"n_distinct_ids": len(ids), "exactly_three": len(ids) == n_fish, "ids": ids}


def speed_spike_rate(df: pd.DataFrame, start: int, end: int,
                     spike_mm: float = 8.0, per: int = 1000) -> dict:
    """Per-fish count of 3D jumps > ``spike_mm`` between consecutive finite frames,
    normalised per ``per`` frames. Returns mean across fish + the raw counts.

    This is the established proxy for residual ID swaps: a clean track is smooth,
    a swap or teleport spikes. NaN if the method has no 3D positions."""
    sub = df[(df["frame"] >= start) & (df["frame"] <= end)]
    sub = sub[sub["fish_id"] >= 0]
    if not {"X", "Y", "Z"}.issubset(sub.columns) or sub[["X", "Y", "Z"]].notna().sum().sum() == 0:
        return {"mean_per_k": float("nan"), "per_fish": {}}
    per_fish = {}
    for fid, g in sub.groupby("fish_id"):
        g = g.sort_values("frame")
        xyz = g[["X", "Y", "Z"]].to_numpy(dtype=float)
        fr = g["frame"].to_numpy()
        fin = np.isfinite(xyz).all(axis=1)
        xyz, fr = xyz[fin], fr[fin]
        if len(xyz) < 2:
            per_fish[int(fid)] = 0.0
            continue
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        adjacent = np.diff(fr) == 1
        n_spikes = int(np.sum((step > spike_mm) & adjacent))
        span = max(int(fr[-1] - fr[0]), 1)
        per_fish[int(fid)] = round(n_spikes * per / span, 3)
    vals = [v for v in per_fish.values()]
    return {"mean_per_k": float(np.mean(vals)) if vals else float("nan"), "per_fish": per_fish}


def reconnection_rate(df: pd.DataFrame, events: list[dict],
                      start: int, end: int, look: int = 12) -> dict:
    """Across hard events inside the window, fraction of involved fish that the
    method reconnects: it has a confident assignment for that slot both ``look``
    frames before the event and ``look`` frames after. A method that drops a fish
    through the event (or never re-acquires it) scores 0 there."""
    in_win = [e for e in events if e.get("start", -1) >= start and e.get("end", 1 << 30) <= end]
    if not in_win:
        return {"reconnection_rate": float("nan"), "n_events": 0}
    confident = df[(df["fish_id"] >= 0) & (df["flag"].fillna("") == "")]
    by_fish = {int(fid): set(g["frame"].astype(int)) for fid, g in confident.groupby("fish_id")}
    hits = total = 0
    for e in in_win:
        slots = _event_slots(e)
        es, ee = int(e["start"]), int(e["end"])
        for s in slots:
            frames = by_fish.get(s, set())
            before = any(f in frames for f in range(es - look, es))
            after = any(f in frames for f in range(ee + 1, ee + 1 + look))
            total += 1
            hits += int(before and after)
    return {"reconnection_rate": round(hits / total, 4) if total else float("nan"),
            "n_events": len(in_win), "n_fish_checks": total}


def _event_slots(e: dict) -> list[int]:
    """Map an event's ``pair`` like 'ID_1|ID_2' to canonical slots [0,1]."""
    pair = str(e.get("pair", ""))
    out = []
    for tok in pair.split("|"):
        tok = tok.strip()
        if tok.startswith("ID_") and tok[3:].isdigit():
            out.append(int(tok[3:]) - 1)
    return out


def view_agreement(view_labels: dict[str, pd.DataFrame], start: int, end: int) -> float:
    """For multi-view tracklet methods: fraction of frames where the per-view
    majority slot labels are mutually consistent (a permutation of each other).

    ``view_labels`` maps view -> DataFrame[frame, x, y, global_id]. We check, per
    frame, that each view assigns a *set* of slots and the sets agree. NaN if <2
    views."""
    if len(view_labels) < 2:
        return float("nan")
    per_view_sets = {}
    for v, d in view_labels.items():
        d = d[(d["frame"] >= start) & (d["frame"] <= end)]
        per_view_sets[v] = d.groupby("frame")["global_id"].apply(lambda s: frozenset(int(i) for i in s))
    frames = set.intersection(*[set(s.index) for s in per_view_sets.values()]) if per_view_sets else set()
    if not frames:
        return float("nan")
    agree = 0
    for f in frames:
        sets = [per_view_sets[v].loc[f] for v in per_view_sets]
        agree += int(all(s == sets[0] for s in sets))
    return round(agree / len(frames), 4)


def align_labels(df_a: pd.DataFrame, df_b: pd.DataFrame, start: int, end: int) -> dict:
    """Inter-method agreement. Restrict to frames where both methods confidently
    assign, Hungarian-align b's slot labels onto a's by co-occurrence, and report
    the per-(frame,slot) label agreement after alignment, plus the agreement on the
    *confident-vs-unknown* decision (flag_agreement)."""
    a = df_a[(df_a["frame"] >= start) & (df_a["frame"] <= end)]
    b = df_b[(df_b["frame"] >= start) & (df_b["frame"] <= end)]
    aa = a[a["fish_id"] >= 0]
    bb = b[b["fish_id"] >= 0]
    if aa.empty or bb.empty:
        return {"label_agreement": float("nan"), "flag_agreement": float("nan"), "n": 0}

    # Build a co-occurrence cost over slot labels using shared positions when 3D is
    # present, else by frame-overlap of label streams.
    a_ids = sorted(aa["fish_id"].unique())
    b_ids = sorted(bb["fish_id"].unique())
    cost = np.zeros((len(a_ids), len(b_ids)))
    a_by = {i: aa[aa["fish_id"] == i].set_index("frame") for i in a_ids}
    b_by = {j: bb[bb["fish_id"] == j].set_index("frame") for j in b_ids}
    has_xyz = {"X", "Y", "Z"}.issubset(aa.columns) and {"X", "Y", "Z"}.issubset(bb.columns) \
        and aa[["X", "Y", "Z"]].notna().any().any() and bb[["X", "Y", "Z"]].notna().any().any()
    for ia, i in enumerate(a_ids):
        for jb, j in enumerate(b_ids):
            shared = a_by[i].index.intersection(b_by[j].index)
            if len(shared) == 0:
                cost[ia, jb] = 1e6
                continue
            if has_xyz:
                da = a_by[i].loc[shared, ["X", "Y", "Z"]].to_numpy(dtype=float)
                db = b_by[j].loc[shared, ["X", "Y", "Z"]].to_numpy(dtype=float)
                d = np.linalg.norm(da - db, axis=1)
                cost[ia, jb] = float(np.nanmean(d)) if np.isfinite(d).any() else 1e6
            else:
                cost[ia, jb] = -len(shared)  # maximise overlap
    ri, ci = linear_sum_assignment(cost)
    mapping = {b_ids[c]: a_ids[r] for r, c in zip(ri, ci)}

    # remap b and compare per (frame, slot)
    bb2 = bb.copy()
    bb2["fish_id"] = bb2["fish_id"].map(mapping).fillna(-1).astype(int)
    merged = aa.merge(bb2, on=["frame", "fish_id"], how="inner", suffixes=("_a", "_b"))
    n_a = len(aa.groupby(["frame", "fish_id"]))
    label_agreement = round(len(merged) / max(n_a, 1), 4)

    # flag agreement: over the union of confidently-assigned (flag=='') slots, do
    # both methods agree a slot is trustworthy? This is the real differentiator,
    # since the geometry-derived methods share positions and only differ in *where
    # they admit uncertainty*.
    a_conf = aa[aa["flag"].fillna("") == ""]
    b_conf = bb2[bb2["flag"].fillna("") == ""]
    fa = set(map(tuple, a_conf[["frame", "fish_id"]].to_numpy()))
    fb = set(map(tuple, b_conf[["frame", "fish_id"]].to_numpy()))
    union = fa | fb
    flag_agreement = round(len(fa & fb) / max(len(union), 1), 4) if union else float("nan")
    return {"label_agreement": label_agreement, "flag_agreement": flag_agreement,
            "n": len(union), "slot_mapping": {int(k): int(v) for k, v in mapping.items()}}


def gt_swap_accuracy(method_says_swap: dict[int, bool], gt: pd.DataFrame) -> dict:
    """GT-gated: compare a method's per-event swap call against eye-checked labels.

    ``gt`` has columns center_frame and ``swap`` in {Yes,No}. ``method_says_swap``
    maps center_frame -> bool. Returns accuracy over labeled events (NaN if none)."""
    if gt is None or "swap" not in gt.columns:
        return {"accuracy": float("nan"), "n_labeled": 0}
    g = gt.dropna(subset=["swap"]).copy()
    g["swap_bool"] = g["swap"].astype(str).str.strip().str.lower().map({"yes": True, "no": False})
    g = g.dropna(subset=["swap_bool"])
    if g.empty:
        return {"accuracy": float("nan"), "n_labeled": 0}
    correct = total = 0
    for _, r in g.iterrows():
        cf = int(r["center_frame"])
        if cf in method_says_swap:
            total += 1
            correct += int(bool(method_says_swap[cf]) == bool(r["swap_bool"]))
    return {"accuracy": round(correct / total, 4) if total else float("nan"),
            "n_labeled": int(len(g)), "n_scored": total}


def window_panel(df: pd.DataFrame, window: dict, events: list[dict],
                 n_fish: int = 3, spike_mm: float = 8.0) -> dict:
    """Compute the full GT-free panel for one (method, window) AssignmentTable."""
    s, e = int(window["start"]), int(window["end"])
    return {
        "coverage": coverage(df, s, e, n_fish),
        "unknown_frac": unknown_frac(df, s, e, n_fish),
        **{f"group_{k}": v for k, v in exactly_three(df, s, e, n_fish).items()},
        "speed_spike": speed_spike_rate(df, s, e, spike_mm),
        "reconnection": reconnection_rate(df, events, s, e),
    }
