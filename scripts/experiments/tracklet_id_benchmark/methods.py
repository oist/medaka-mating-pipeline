from __future__ import annotations

"""Adapters: load each method family's output into the common AssignmentTable.

Every adapter reads an *existing on-disk* pipeline output and returns a long
DataFrame with the columns described in ``metrics.py``. No method re-runs an
expensive computation here; the harness is a scorer over artifacts that already
exist.

Honest design note surfaced by the data: ``geometry_3d``, ``tracklet_bind``,
``motion_resolver`` and ``hybrid`` all share the SAME 3D backbone positions and
the SAME canonical slots (``ID_k`` -> ``k-1``). They differ only in *where they
admit uncertainty* (which frames/slots they flag instead of committing). The only
family with independent identity labels is ``appearance_idtracker``. The benchmark
makes that explicit rather than pretending four independent identity decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = ["top_cam", "side1_cam", "side2_cam"]
SLOTS = ["ID_1", "ID_2", "ID_3"]
ASSIGN_COLS = ["frame", "fish_id", "X", "Y", "Z", "conf", "flag", "method"]


def canonical_slot(fish_id) -> int:
    """'ID_1'->0, 'ID_2'->1, ... ; pass-through for ints already 0-based."""
    s = str(fish_id)
    if s.startswith("ID_") and s[3:].isdigit():
        return int(s[3:]) - 1
    try:
        return int(float(s))
    except ValueError:
        return -1


def load_3d(csv_3d: str) -> pd.DataFrame:
    """Load the *_3d_identified.csv once, add a canonical 0-based ``slot`` column."""
    d = pd.read_csv(csv_3d)
    d["slot"] = d["fish_id"].map(canonical_slot)
    keep = ["frame", "slot", "X", "Y", "Z"]
    if "bind_score" in d.columns:
        keep.append("bind_score")
    return d[keep].copy()


def _slice(d3: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return d3[(d3["frame"] >= start) & (d3["frame"] <= end)].copy()


def geometry_3d_table(d3: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Production geometry+bind: commits all 3 slots every frame, never unknown."""
    s = _slice(d3, start, end)
    s = s[s["slot"] >= 0]
    conf = (1.0 / (1.0 + s["bind_score"])) if "bind_score" in s.columns else np.nan
    return pd.DataFrame({
        "frame": s["frame"].astype(int), "fish_id": s["slot"].astype(int),
        "X": s["X"], "Y": s["Y"], "Z": s["Z"], "conf": conf,
        "flag": "", "method": "geometry_3d",
    })[ASSIGN_COLS]


def _load_tracklet_bound(bind_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for v in VIEWS:
        p = Path(bind_dir) / f"{v}.tracklet_bound.csv"
        if p.exists():
            out[v] = pd.read_csv(p)
    return out


def tracklet_bind_table(d3: pd.DataFrame, bind_dir: str, start: int, end: int):
    """Geometry tracklet-linking: 3D positions, but flag frames the tracklet binder
    could not certify. Per (frame, slot): 'hardcore' if any view marks in_hardcore,
    'no_tracklet' if no view has a detection, 'low_conf' if all present views are
    low-conf, else ''. Returns (AssignmentTable, per-view label frames)."""
    bound = _load_tracklet_bound(bind_dir)
    geom = geometry_3d_table(d3, start, end)
    # aggregate tracklet-bound flags per (frame, global_id) across views
    agg = {}
    view_labels = {}
    for v, df in bound.items():
        d = df[(df["frame"] >= start) & (df["frame"] <= end)]
        view_labels[v] = d[["frame", "x", "y", "global_id"]].copy()
        for _, r in d.iterrows():
            key = (int(r["frame"]), int(r["global_id"]))
            cur = agg.setdefault(key, {"present": True, "hardcore": False, "low_all": True})
            cur["hardcore"] = cur["hardcore"] or bool(r.get("in_hardcore", False))
            cur["low_all"] = cur["low_all"] and bool(r.get("low_conf", False))

    def flag_for(frame, slot):
        info = agg.get((int(frame), int(slot)))
        if info is None:
            return "no_tracklet"
        if info["hardcore"]:
            return "hardcore"
        if info["low_all"]:
            return "low_conf"
        return ""

    geom = geom.copy()
    geom["flag"] = [flag_for(f, s) for f, s in zip(geom["frame"], geom["fish_id"])]
    geom["method"] = "tracklet_bind"
    return geom[ASSIGN_COLS], view_labels


def motion_resolver_table(d3: pd.DataFrame, resolved: pd.DataFrame,
                          start: int, end: int) -> pd.DataFrame:
    """Temporal/dynamics overlay: geometry positions, but flag 'review' inside any
    hard-core event the resolver did NOT clear, for the two involved slots."""
    geom = geometry_3d_table(d3, start, end).copy()
    flagged = {sl: [] for sl in range(3)}
    r = resolved[(resolved["end"] >= start) & (resolved["start"] <= end)]
    for _, ev in r.iterrows():
        verdict = str(ev.get("verdict", "review"))
        if verdict == "clear":
            continue
        slots = []
        for tok in str(ev.get("pair", "")).split("|"):
            tok = tok.strip()
            if tok.startswith("ID_") and tok[3:].isdigit():
                slots.append(int(tok[3:]) - 1)
        for sl in slots:
            flagged[sl].append((int(ev["start"]), int(ev["end"])))

    def is_flagged(frame, slot):
        for s0, e0 in flagged.get(int(slot), []):
            if s0 <= frame <= e0:
                return True
        return False

    geom["flag"] = ["review" if is_flagged(f, s) else "" for f, s in zip(geom["frame"], geom["fish_id"])]
    geom["method"] = "motion_resolver"
    return geom[ASSIGN_COLS]


def hybrid_table(d3: pd.DataFrame, bind_dir: str, resolved: pd.DataFrame,
                 start: int, end: int) -> pd.DataFrame:
    """tracklet_bind flags, but un-flag a 'hardcore' frame/slot when the resolver
    cleared the overlapping event (geometry + tracklet + temporal)."""
    tb, _ = tracklet_bind_table(d3, bind_dir, start, end)
    tb = tb.copy()
    cleared = {sl: [] for sl in range(3)}
    r = resolved[(resolved["end"] >= start) & (resolved["start"] <= end)]
    for _, ev in r.iterrows():
        if str(ev.get("verdict", "")) != "clear":
            continue
        for tok in str(ev.get("pair", "")).split("|"):
            tok = tok.strip()
            if tok.startswith("ID_") and tok[3:].isdigit():
                cleared[int(tok[3:]) - 1].append((int(ev["start"]), int(ev["end"])))

    def cleared_here(frame, slot):
        for s0, e0 in cleared.get(int(slot), []):
            if s0 <= frame <= e0:
                return True
        return False

    new_flags = []
    for f, s, fl in zip(tb["frame"], tb["fish_id"], tb["flag"]):
        new_flags.append("" if (fl == "hardcore" and cleared_here(f, s)) else fl)
    tb["flag"] = new_flags
    tb["method"] = "hybrid"
    return tb[ASSIGN_COLS]


def appearance_idtracker_table(traj_csv: str | None, start: int, end: int,
                               frame_offset: int = 0):
    """Optional appearance classifier. idtracker.ai writes trajectories.csv with
    per-frame x/y per global identity. Returns (AssignmentTable|None, reason).
    The clip is usually offset (e.g. idtracker ran on frames 90000+), so
    ``frame_offset`` maps clip row -> absolute frame."""
    if not traj_csv or not Path(traj_csv).exists():
        return None, ("unavailable: no idtracker trajectories.csv. Reproduce with the "
                      ".venv-idtrackerai env: clean_topview_by_3d.py -> idtrackerai --track, "
                      "then point appearance_traj_csv at session/trajectories/trajectories.csv")
    tr = pd.read_csv(traj_csv)
    # idtracker layout varies; expect columns x1,y1,x2,y2,... or id,frame,x,y.
    rows = []
    if {"id", "frame", "x", "y"}.issubset(tr.columns):
        tr = tr.copy()
        tr["frame"] = tr["frame"].astype(int) + frame_offset
        sub = tr[(tr["frame"] >= start) & (tr["frame"] <= end)]
        for _, r in sub.iterrows():
            rows.append({"frame": int(r["frame"]), "fish_id": int(r["id"]),
                         "X": np.nan, "Y": np.nan, "Z": np.nan,
                         "conf": r.get("id_probability", np.nan), "flag": "",
                         "method": "appearance_idtracker"})
    else:
        return None, f"unrecognised idtracker schema: {list(tr.columns)[:8]}"
    if not rows:
        return None, "idtracker file has no frames in window"
    return pd.DataFrame(rows)[ASSIGN_COLS], "ok"


def method_swap_calls(method: str, resolved: pd.DataFrame, start: int, end: int) -> dict[int, bool]:
    """Per-event swap call for GT scoring, keyed by event center frame.

    geometry_3d/hybrid commit (never call a swap -> always False). motion_resolver
    calls swap on 'suspect_swap', no-swap on confirmed clears, abstains on
    'ambiguous'. tracklet_bind abstains inside the hard core (no entries)."""
    r = resolved[(resolved["end"] >= start) & (resolved["start"] <= end)].copy()
    r["center"] = ((r["start"] + r["end"]) / 2).astype(int)
    calls: dict[int, bool] = {}
    for _, ev in r.iterrows():
        c = int(ev["center"])
        reason = str(ev.get("reason", ""))
        if method in ("geometry_3d", "hybrid"):
            calls[c] = False
        elif method == "motion_resolver":
            if reason == "suspect_swap":
                calls[c] = True
            elif reason in ("velocity_consistent", "near_miss"):
                calls[c] = False
            # 'ambiguous' -> abstain
        # tracklet_bind / appearance abstain at the event level here
    return calls
