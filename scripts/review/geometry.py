"""Reviewer geometry corrections: which detection belongs to which fish in each view, per-view
head/tail, views left out, flagged bodies - and the 3D recomputed from them.

Ledger events used here (see :mod:`scripts.review.ledger`):

* ``view_flip`` / ``view_drop`` - ``fish`` = track, ``value`` = view name, ``frame``..``frame_end``:
  head/tail of that track reversed in that view / that view left out of its triangulation;
* ``view_swap`` - ``fish`` = ``ID_a|ID_b``, ``value`` = view name, range: the two tracks' detections
  exchanged in that view;
* ``nodes_bad`` - reviewer-flagged body (short tail, wrong orientation, shifted centre): mouth and
  tail re-estimated from the pectoral node and the body direction around the flagged run;
* ``interval`` / ``interval_del`` / ``confirm`` - close co-moving intervals and reviewer keyframes;
  a solved interval (:func:`solve`) re-assigns the group's detections between its keyframes and is
  cached in ``interval_solutions/<id>.npz`` next to the ledger.

:func:`recompute` rebuilds one frame range from ``detections.npz`` with all of these applied; the GUI
calls it for edited ranges, :func:`apply_geometry` for every edited range (the export).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.ledger import PAIR_SEP, owner_series, state_series, swap_list
from scripts.review.review_data import VIEWS, FillParams, OrientationParams, unpack_detections

GEOMETRY_KINDS = ("view_flip", "view_drop", "view_swap", "nodes_bad")
SOLUTIONS_DIR = "interval_solutions"
MARGIN = 200                 # frames of context around a recomputed range
KEYFRAME_SECONDS = 0.3       # a keyframe this close to an interval end counts as confirming that end
GEOMETRY_KEYS = ("kp2d", "xyz", "proj2d", "nodes3d_filled", "node_source", "missing_node_proj2d",
                 "orientation_segment")


def view_index(name: str) -> int:
    return VIEWS.index(name) if name in VIEWS else int(name)


def event_range(row, last: int) -> tuple[int, int]:
    end = str(row["frame_end"]).strip()
    return int(row["frame"]), int(end) if end not in ("", "nan") else int(last)


# ---------------------------------------------------------------------------
# Intervals and keyframes


def effective_intervals(machine: pd.DataFrame | None, events: pd.DataFrame) -> pd.DataFrame:
    """Machine intervals with the reviewer's edits: ``interval`` rows (``value`` = id) create or move
    one, ``interval_del`` rows remove it; the most recently written row of an id wins."""
    cols = ["interval_id", "frame_start", "frame_end", "fish", "origin"]
    rows: dict[str, dict] = {}
    if machine is not None:
        for r in machine.itertuples(index=False):
            rows[str(r.interval_id)] = {"interval_id": str(r.interval_id), "frame_start": int(r.frame_start),
                                        "frame_end": int(r.frame_end), "fish": str(r.fish), "origin": "machine"}
    e = events[events["kind"].isin(["interval", "interval_del"])]
    for r in e.itertuples(index=False):
        iid = str(r.value)
        if r.kind == "interval_del":
            rows.pop(iid, None)
            continue
        origin = rows.get(iid, {}).get("origin", "reviewer")
        rows[iid] = {"interval_id": iid, "frame_start": int(r.frame), "frame_end": int(r.frame_end),
                     "fish": str(r.fish), "origin": origin if origin == "machine" else "reviewer"}
    out = pd.DataFrame(list(rows.values()), columns=cols)
    return out.sort_values("frame_start", kind="stable").reset_index(drop=True)


def confirm_value(heads: dict[str, tuple]) -> str:
    """``confirm`` event value: per track the displayed head and tail 3D positions and whether a human
    head/tail flip (H) was active then (so the machine head can be recovered later)."""
    parts = []
    for fid, item in heads.items():
        h, t = item[0], item[1]
        flipped = bool(item[2]) if len(item) > 2 else False
        if np.isfinite(h).all() and np.isfinite(t).all():
            parts.append(fid + ":" + ",".join(f"{x:.2f}" for x in (*h, *t)) + f",{int(flipped)}")
    return ";".join(parts)


def keyframes(events: pd.DataFrame) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray, bool]]]:
    """Reviewer keyframes: frame -> {track: (head xyz, tail xyz, human flip active)} as displayed."""
    out: dict[int, dict] = {}
    for r in events[events["kind"] == "confirm"].itertuples(index=False):
        heads = {}
        for part in str(r.value).split(";"):
            fid, _, nums = part.partition(":")
            vals = np.array([float(x) for x in nums.split(",")]) if nums else np.zeros(0)
            if len(vals) in (6, 7):
                heads[fid] = (vals[:3], vals[3:6], bool(len(vals) == 7 and vals[6] > 0.5))
        out[int(r.frame)] = heads
    return out


def interval_keyframes(kf: dict, a: int, b: int, fps: float) -> list[int]:
    return sorted(f for f in kf if a <= f <= b)


def interval_ready(kf: dict, a: int, b: int, fps: float) -> bool:
    """Both ends confirmed (a keyframe within ``KEYFRAME_SECONDS`` of each end)."""
    frames = interval_keyframes(kf, a, b, fps)
    tol = KEYFRAME_SECONDS * fps
    return bool(frames) and frames[0] <= a + tol and frames[-1] >= b - tol


# ---------------------------------------------------------------------------
# Detections


@dataclass
class Detections:
    """``detections.npz``: every detection of every view plus the tracker's assignment."""

    folder: Path
    det: np.ndarray = field(init=False)
    slots: np.ndarray = field(init=False)
    frames: np.ndarray = field(init=False)

    def __post_init__(self):
        with np.load(Path(self.folder) / "detections.npz") as z:
            self.det, self.slots, self.frames = z["det"], z["slots"], z["frames"]

    def points(self, t0: int, t1: int) -> np.ndarray:
        """Detections of data rows ``t0:t1`` as float pixels ``(F, V, S, N, 2)``."""
        return unpack_detections(self.det[t0:t1])


def gather(points: np.ndarray, assign: np.ndarray) -> np.ndarray:
    """Per-track keypoints ``(F, K, V, N, 2)`` from detections ``(F, V, S, N, 2)`` and slots ``(F, K, V)``."""
    n_frames, n_fish, n_views = assign.shape
    out = np.full((n_frames, n_fish, n_views) + points.shape[3:], np.nan, dtype=np.float32)
    rows = np.arange(n_frames)
    for k in range(n_fish):
        for v in range(n_views):
            s = assign[:, k, v]
            ok = s >= 0
            out[ok, k, v] = points[rows[ok], v, s[ok]]
    return out


# ---------------------------------------------------------------------------
# Interval solutions


def solution_path(ledger_dir: Path, interval_id: str) -> Path:
    return Path(ledger_dir) / SOLUTIONS_DIR / f"{interval_id}.npz"


def solution_key(events: pd.DataFrame, fish_ids: list[str], a: int, b: int, fish: str, kf: dict,
                 fps: float) -> str:
    """Everything a solution depends on: bounds, members, keyframes (frames and confirmed positions),
    the identity swaps in force over the interval and the view swaps overlapping it. Any change makes a
    cached solution stale (it is then ignored until solved again)."""
    frames = np.arange(a, b + 1)
    owner = owner_series(len(fish_ids), swap_list(events, fish_ids), frames)
    changes = np.flatnonzero(np.r_[True, (owner[1:] != owner[:-1]).any(axis=1)])
    kf_items = [[f, sorted((fid, [round(float(x), 2) for x in (*v[0], *v[1])], bool(v[2]))
                           for fid, v in kf[f].items())] for f in interval_keyframes(kf, a, b, fps)]
    vs = events[events["kind"] == "view_swap"]
    vs_rows = sorted([str(r.fish), str(r.value), int(r.frame), str(r.frame_end)] for r in vs.itertuples(index=False)
                     if int(r.frame) <= b and (str(r.frame_end).strip() in ("", "nan") or int(r.frame_end) >= a))
    payload = [a, b, fish, kf_items, [[int(c), owner[c].tolist()] for c in changes], vs_rows]
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:16]


def load_solutions(ledger_dir: Path, intervals: pd.DataFrame, events: pd.DataFrame, fish_ids: list[str],
                   fps: float) -> dict[str, dict]:
    """Cached solutions whose interval, keyframes, identity swaps and view swaps are unchanged."""
    out = {}
    kf = keyframes(events)
    for r in intervals.itertuples(index=False):
        path = solution_path(ledger_dir, r.interval_id)
        if not path.exists():
            continue
        with np.load(path) as z:
            sol = {k: z[k] for k in z.files}
        key = solution_key(events, fish_ids, int(r.frame_start), int(r.frame_end), str(r.fish), kf, fps)
        if str(sol["key"]) == key:
            out[str(r.interval_id)] = sol
    return out


# ---------------------------------------------------------------------------
# Assignment of detections to tracks


def base_assignment(dets: Detections, events: pd.DataFrame, fish_ids: list[str], t0: int, t1: int,
                    first: int) -> np.ndarray:
    """Tracker slots ``(F, K, V)`` for data rows ``t0:t1`` with the reviewer's ``view_swap`` events."""
    assign = dets.slots[t0:t1].copy()
    index = {f: k for k, f in enumerate(fish_ids)}
    for r in events[events["kind"] == "view_swap"].itertuples(index=False):
        a_id, _, b_id = str(r.fish).partition(PAIR_SEP)
        if a_id not in index or b_id not in index:
            continue
        lo, hi = event_range(r._asdict(), first + len(dets.slots) - 1)
        s, e = max(lo - first, t0), min(hi - first, t1 - 1)
        if s > e:
            continue
        v = view_index(str(r.value))
        a, b = index[a_id], index[b_id]
        assign[s - t0:e - t0 + 1, [a, b], v] = assign[s - t0:e - t0 + 1, [b, a], v]
    return assign


def apply_solutions(assign: np.ndarray, solutions: dict, source: np.ndarray, t0: int, first: int) -> np.ndarray:
    """Overwrite the group's slots with solved interval assignments (solutions are in fish space)."""
    assign = assign.copy()
    n_frames = len(assign)
    for sol in solutions.values():
        frames = sol["frames"]
        rows = frames - first - t0
        ok = (rows >= 0) & (rows < n_frames)
        if not ok.any():
            continue
        members = sol["members"]                               # corrected fish indices
        chosen = np.take_along_axis(sol["cand_slots"], sol["perm"].astype(np.int64), axis=2)   # (Fi, V, m)
        for i, c in enumerate(members):
            tracks = source[rows[ok], c]
            assign[rows[ok], tracks] = chosen[ok, :, i]
    return assign


# ---------------------------------------------------------------------------
# Recompute


@dataclass
class Geometry:
    """Everything a recompute needs: review data, detections, calibration and parameters."""

    data: dict
    dets: Detections
    cal: object
    fps: float
    orient: OrientationParams = field(default_factory=OrientationParams)
    fill: FillParams = field(default_factory=FillParams)
    robust: object = None

    @classmethod
    def from_manifest(cls, data: dict, dets: Detections, cal, manifest: dict) -> Geometry:
        """Parameters as the review data was built with (keys of other versions ignored)."""
        from dataclasses import fields

        from scripts.review.reconstruct import RobustParams

        def make(kind, key):
            given = manifest.get(key) or {}
            names = {f.name for f in fields(kind)}
            return kind(**{k: v for k, v in given.items() if k in names})

        return cls(data, dets, cal, float(manifest.get("fps", 166.0)), make(OrientationParams, "orientation_params"),
                   make(FillParams, "fill_params"), make(RobustParams, "robust_params"))

    @property
    def fish_ids(self) -> list[str]:
        return [str(f) for f in self.data["fish_ids"]]

    @property
    def first(self) -> int:
        return int(self.data["frames"][0])


def flagged_ranges(events: pd.DataFrame, kind: str, fish_ids: list[str], frames: np.ndarray) -> list[tuple[int, int]]:
    """Closed runs of a "from here on" flag (an open run to the end is left out until it is closed)."""
    out = []
    for fid in fish_ids:
        on = state_series(events, kind, fid, frames) == "1"
        if not on.any():
            continue
        d = np.diff(np.r_[0, on.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
            if e < len(frames) - 1:
                out.append((int(frames[s]), int(frames[e])))
    return out


def closed_flag_mask(events: pd.DataFrame, kind: str, fid: str, frames: np.ndarray) -> np.ndarray:
    """Frames inside closed runs of a "from here on" flag of one track (an open run to the end is not
    counted until it is closed, as in :func:`flagged_ranges`). ``frames`` must be the full data range."""
    on = state_series(events, kind, fid, frames) == "1"
    if on.any() and on[-1]:
        d = np.diff(np.r_[0, on.astype(np.int8)])
        on[np.flatnonzero(d == 1)[-1]:] = False
    return on


def edited_ranges(events: pd.DataFrame, solutions: dict, fish_ids: list[str], frames: np.ndarray) -> list[tuple[int, int]]:
    """Frame ranges whose 3D differs from the review data because of geometry edits (merged)."""
    last = int(frames[-1])
    spans = []
    for r in events[events["kind"].isin(["view_flip", "view_drop", "view_swap"])].itertuples(index=False):
        spans.append(event_range(r._asdict(), last))
    spans += flagged_ranges(events, "nodes_bad", fish_ids, frames)
    for sol in solutions.values():
        spans.append((int(sol["frames"][0]), int(sol["frames"][-1])))
    first = int(frames[0])
    spans = sorted((max(a, first), min(b, last)) for a, b in spans if b >= first and a <= last)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def recompute(geo: Geometry, events: pd.DataFrame, solutions: dict, lo: int, hi: int,
              margin: int = MARGIN) -> dict:
    """3D of frames ``lo..hi`` with every geometry edit applied; arrays for exactly those frames."""
    from scripts.review.reconstruct import (
        centroid_from_kp,
        group_fill,
        project_missing,
        project_points,
        reconstruct_tracks,
        reestimate_body,
    )

    data, first = geo.data, geo.first
    frames_all = data["frames"]
    fish_ids = geo.fish_ids
    head, tail = int(data["head_node"]), int(data["tail_node"])
    lo, hi = max(lo, first), min(hi, int(frames_all[-1]))
    t0, t1 = max(0, lo - first - margin), min(len(frames_all), hi - first + margin + 1)
    frames = frames_all[t0:t1]
    owner = owner_series(len(fish_ids), swap_list(events, fish_ids), frames)
    source = np.argsort(owner, axis=1)
    assign = apply_solutions(base_assignment(geo.dets, events, fish_ids, t0, t1, first), solutions, source, t0, first)
    kp_raw = gather(geo.dets.points(t0, t1), assign)
    kp_tri = kp_raw.copy()
    extra_flip = np.zeros(assign.shape, dtype=bool)
    index = {f: k for k, f in enumerate(fish_ids)}
    last = int(frames_all[-1])
    for r in events[events["kind"].isin(["view_drop", "view_flip"])].itertuples(index=False):
        if r.fish not in index:
            continue
        a, b = event_range(r._asdict(), last)
        s, e = max(a - first, t0) - t0, min(b - first, t1 - 1) - t0
        if s > e:
            continue
        k, v = index[r.fish], view_index(str(r.value))
        if r.kind == "view_drop":
            kp_tri[s:e + 1, k, v] = np.nan
        else:
            extra_flip[s:e + 1, k, v] ^= True
    # Reviewer keyframes pin the orientation. The confirmed head is in display terms; where a human
    # head/tail flip (H) was active when it was confirmed, the machine head is the displayed tail.
    prefer = np.full(assign.shape[:2] + (3,), np.nan)
    for f, heads in keyframes(events).items():
        t = f - first - t0
        if not 0 <= t < len(frames):
            continue
        for fid, (h, tl, flipped) in heads.items():
            if fid in index:
                prefer[t, index[fid]] = tl if flipped else h
    xyz = centroid_from_kp(geo.cal, kp_tri, head, tail)
    # Fish of a solved interval that no view pair locates move with their group mate (the tracker's
    # centroid may belong to the other fish there); everywhere else the review data's centroid fills in.
    solved = np.zeros(assign.shape[:2], dtype=bool)
    for sol in solutions.values():
        rows_s = sol["frames"] - first - t0
        ok = (rows_s >= 0) & (rows_s < len(frames))
        for c in sol["members"]:
            solved[rows_s[ok], source[rows_s[ok], c]] = True
    if solved.any():
        members_k = [k for k in range(len(fish_ids)) if solved[:, k].any()]
        grouped = group_fill(np.where(solved[..., None], xyz, np.nan), members_k)
        xyz = np.where(solved[..., None] & ~np.isfinite(xyz), grouped, xyz)
    base_xyz = data["xyz"][t0:t1].astype(float)
    xyz = np.where(np.isfinite(xyz), xyz, base_xyz)
    rec = reconstruct_tracks(geo.cal, kp_raw, kp_tri, xyz, head, tail, geo.orient, geo.fill, geo.robust,
                             extra_view_flip=extra_flip, prefer_head=prefer)
    nodes, source_n = rec["nodes3d_filled"].astype(float), rec["node_source"]
    flagged = np.stack([closed_flag_mask(events, "nodes_bad", f, frames_all)[t0:t1] for f in fish_ids], axis=1)
    for k in range(len(fish_ids)):
        if flagged[:, k].any():
            nodes[:, k], source_n[:, k] = reestimate_body(nodes[:, k], source_n[:, k], flagged[:, k], head, tail)
            mid = (nodes[:, k, head] + nodes[:, k, tail]) / 2
            xyz[flagged[:, k], k] = mid[flagged[:, k]]
    kp = rec["kp2d"]
    missing = project_missing(geo.cal, kp, nodes) if flagged.any() else rec["missing_node_proj2d"]
    # Segment ids of this window must not collide with the review data's (H flips by id).
    segments = np.where(rec["orientation_segment"] >= 0, rec["orientation_segment"] + 100_000_000 + lo, -1)
    out = {"kp2d": kp, "xyz": xyz.astype(np.float32), "proj2d": project_points(geo.cal, xyz),
           "nodes3d_filled": nodes.astype(np.float32), "node_source": source_n, "missing_node_proj2d": missing,
           "orientation_segment": segments.astype(np.int32)}
    s, e = lo - first - t0, hi - first - t0 + 1
    return {k: v[s:e] for k, v in out.items()}


def recompute_many(geo: Geometry, events: pd.DataFrame, solutions: dict, ranges: list[tuple[int, int]],
                   join_gap: int = 2 * MARGIN):
    """:func:`recompute` for many ranges: ranges closer than ``join_gap`` share one computation, but
    only the edited ranges themselves are returned. Yields ``(lo, hi, arrays)``."""
    groups: list[list[tuple[int, int]]] = []
    for a, b in sorted(ranges):
        if groups and a - groups[-1][-1][1] <= join_gap:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    for group in groups:
        lo, hi = group[0][0], max(b for _, b in group)
        part = recompute(geo, events, solutions, lo, hi)
        for a, b in group:
            yield a, b, {k: v[a - lo:b - lo + 1] for k, v in part.items()}


def solve(geo: Geometry, events: pd.DataFrame, interval: dict, ledger_dir: Path, params=None) -> dict:
    """Re-assign the interval group's detections between keyframes and cache the result."""
    from scripts.review.reconstruct import IntervalParams, solve_interval

    params = params or IntervalParams()
    data, first = geo.data, geo.first
    fish_ids = geo.fish_ids
    head, tail = int(data["head_node"]), int(data["tail_node"])
    a, b = int(interval["frame_start"]), int(interval["frame_end"])
    t0, t1 = a - first, b - first + 1
    frames = data["frames"][t0:t1]
    owner = owner_series(len(fish_ids), swap_list(events, fish_ids), frames)
    source = np.argsort(owner, axis=1)
    tracks = [fish_ids.index(f) for f in str(interval["fish"]).split(PAIR_SEP) if f in fish_ids]
    members = [int(owner[0, k]) for k in tracks]                   # corrected fish at the interval start
    assign = base_assignment(geo.dets, events, fish_ids, t0, t1, first)
    points = geo.dets.points(t0, t1)
    rows = np.arange(len(frames))
    cand_slots = np.stack([assign[rows[:, None], source[:, members], v] for v in range(len(VIEWS))], axis=1)  # (F,V,m)
    cand = np.full(cand_slots.shape + (2,), np.nan)
    for v in range(len(VIEWS)):
        for i in range(len(members)):
            s = cand_slots[:, v, i]
            ok = s >= 0
            pts = points[rows[ok], v, s[ok]]                                               # (n, N, 2)
            both = np.isfinite(pts[:, head]).all(-1) & np.isfinite(pts[:, tail]).all(-1)
            c = np.where(both[:, None], (pts[:, head] + pts[:, tail]) / 2, np.nan)
            cand[rows[ok], v, i] = c
    # Fish merged into one blob in a view (only one of the group detected there): every fish of the group
    # is under that blob, so for the decision they all "see" its centre - the view then cannot reorder
    # them and the views that separate the fish decide. The blob itself stays with the fish that had it.
    for v in range(len(VIEWS)):
        has = np.isfinite(cand[:, v]).all(axis=-1)                       # (F, m)
        one = has.sum(axis=1) == 1
        if one.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blob = np.nanmean(np.where(has[..., None], cand[:, v], np.nan), axis=1)   # (F, 2)
            fill = one[:, None] & ~has
            cand[:, v] = np.where(fill[..., None], blob[:, None, :], cand[:, v])
    # Keyframes: what the reviewer confirmed on screen - each member's centroid then (midpoint of the
    # confirmed head and tail, unaffected by head/tail flips) - not the tracker's assignment.
    kf = keyframes(events)
    kf_frames = interval_keyframes(kf, a, b, geo.fps)
    key_mask = np.isin(frames, kf_frames)
    key_pos = np.full((len(frames), len(members), 3), np.nan)
    for f in kf_frames:
        t = f - a
        for i, c in enumerate(members):
            item = kf[f].get(fish_ids[int(source[t, c])])
            if item is not None:
                key_pos[t, i] = (item[0] + item[1]) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = solve_interval(geo.cal, cand, key_mask, params, key_positions=key_pos)
    chosen = np.take_along_axis(cand_slots, out["perm"].astype(np.int64), axis=2)
    changed = (chosen != cand_slots).any(axis=(1, 2))                   # a fish really got another detection
    path = solution_path(ledger_dir, str(interval["interval_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames, members=np.array(members), cand_slots=cand_slots.astype(np.int8),
                        perm=out["perm"].astype(np.int8), changed=changed,
                        key=solution_key(events, fish_ids, a, b, str(interval["fish"]), kf, geo.fps))
    return {"frames_changed": int(changed.sum()), "frames": len(frames), "members": members}


CACHE_FILE = "geometry_cache.npz"
STATE_KINDS = GEOMETRY_KINDS + ("interval", "interval_del", "confirm", "swap")


def geometry_state_key(events: pd.DataFrame, solutions: dict, manifest: dict) -> str:
    """Hash of everything the recomputed ranges depend on: the review data build, the geometry-related
    events (and identity swaps, which interval solutions depend on) and the cached solutions."""
    e = events[events["kind"].isin(STATE_KINDS)]
    rows = [[str(x) for x in r] for r in e[["kind", "fish", "frame", "frame_end", "value"]].itertuples(index=False)]
    payload = [manifest.get("created"), manifest.get("data_version"), rows,
               sorted((k, str(v["key"])) for k, v in solutions.items())]
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:16]


def save_geometry_cache(ledger_dir: Path, key: str, data: dict, ranges: list[tuple[int, int]], first: int) -> None:
    """The recomputed rows of every edited range, so the next start-up need not recompute them."""
    idx = np.concatenate([np.arange(a, b + 1) - first for a, b in ranges]) if ranges else np.zeros(0, int)
    arrays = {f"a_{k}": np.asarray(data[k])[idx] for k in GEOMETRY_KEYS if k in data}
    tmp = Path(ledger_dir) / (CACHE_FILE + ".tmp.npz")
    np.savez_compressed(tmp, key=key, rows=idx, **arrays)
    tmp.replace(Path(ledger_dir) / CACHE_FILE)


def load_geometry_cache(ledger_dir: Path, key: str) -> tuple[np.ndarray, dict] | None:
    path = Path(ledger_dir) / CACHE_FILE
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            if str(z["key"]) != key:
                return None
            return z["rows"], {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    except (OSError, ValueError, KeyError):
        return None


def apply_geometry(geo: Geometry, events: pd.DataFrame, ledger_dir: Path, intervals: pd.DataFrame,
                   log=None) -> tuple[dict, list[tuple[int, int]]]:
    """Copy of the review data with every edited range recomputed (for the export)."""
    solutions = load_solutions(ledger_dir, intervals, events, geo.fish_ids, geo.fps)
    data = dict(geo.data)
    for key in GEOMETRY_KEYS:
        if key in data:
            data[key] = np.array(data[key], copy=True)
    ranges = edited_ranges(events, solutions, geo.fish_ids, data["frames"])
    first = geo.first
    for lo, hi, part in recompute_many(geo, events, solutions, ranges):
        for key, value in part.items():
            if key in data:
                data[key][lo - first:hi - first + 1] = value
    if log is not None:
        log(f"[geometry] recomputed {len(ranges)} edited range(s)")
    return data, ranges
