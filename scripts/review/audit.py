"""Audit: numbers for SLEAP and for old vs new 3D without watching the whole recording.

Three task kinds, all in ``<recon_dir>/audit/``:

* **SLEAP** - every frame and view is screened automatically (``slp_raw.npz`` + ``review_data.npz``) into
  suspicious groups (fewer / more than 3 instances, incomplete instance, instance too long = two fish
  joined, head-tail against the other views, view left out of triangulation as inconsistent). A random
  sample of each group is checked by a person ("real error?"), giving each group's precision and so the
  number of real errors in the whole recording; a random sample of unflagged frames checks what the
  screening misses.
* **geometry** - old and new give the same 3D in ~98% of frames (same 2D points); only the stretches where
  they differ matter for choosing one. A random sample of them is judged blind, one version at a time: A
  alone, then B alone (old / new in random order, hidden): right / small error / clear error at the key
  frame.
* **id** - every place where old's and new's identities stop matching (one of them swapped) is judged
  blind the same way, one version at a time: does every colour stay on one fish (right / wrong and stays
  wrong / wrong briefly, i.e. swaps and comes back)?

Which version is better is worked out by ``stats`` from the two separate answers.

* **lost** (added later with ``add-lost``) - in close encounters (fish < 10 mm apart) a fish often has a
  detection in fewer than two views. Where it lost a view, the raw SLEAP output is sorted automatically
  (fewer than 3 instances / a separate unassigned instance where the fish should be / one overlapping
  another fish / nothing there); a sample of each kind is checked by eye, blind to the kind: was the fish
  detected (the tracker did not use it), not detected, or fully hidden?

::

    python -m scripts.review.audit build <folder> --recon-dir <dir>     # task list (once; fixed seed)
    python -m scripts.review.audit add-lost <folder> --recon-dir <dir>  # append the lost-view items (once)
    python -m scripts.review.audit stats <folder> --recon-dir <dir>     # results (unblinded)

The GUI is :mod:`scripts.review.audit_gui`. Answers are append-only (last one per item and part wins).
Everywhere 1 = right, 2 / 3 = wrong, 0 = unsure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.recon_compare import ReconState
from scripts.review.slp_raw import RawSlp

AUDIT_DIR = "audit"
TASKS_FILE = "audit_tasks.csv"
ANSWERS_FILE = "audit_answers.csv"
FLAGS_FILE = "audit_flags_summary.json"
TASK_COLUMNS = ["item", "task", "category", "start", "end", "frame", "view", "instance", "fish", "a_is", "b_is",
                "x", "y"]           # x, y: lost items' expected fish position (source pixels of ``view``)
ANSWER_COLUMNS = ["item", "part", "answer", "note", "created"]
PARTS = ("A", "B")              # old vs new items: each version judged on its own

SLEAP_CATEGORIES = {
    "missing": "fewer than 3 instances in this view",
    "extra": "more than 3 instances in this view",
    "incomplete": "an instance without all body parts",
    "too_long": "an instance much longer than usual (two fish joined?)",
    "head_tail": "head and tail against the other two views",
    "view_outlier": "this view left out of 3D as inconsistent with the others",
}
QUESTIONS = {
    "missing": "SLEAP found fewer than 3 fish in the * view. Is a fish that can be seen there missed?\n"
               "1 = SLEAP right (nothing visible is missed, e.g. the fish is fully hidden)   2 = SLEAP wrong (missed)",
    "extra": "SLEAP found more than 3 fish in the * view.\n"
             "1 = SLEAP right (nothing false)   2 = SLEAP wrong (a false or duplicate fish)",
    "incomplete": "The ringed fish (* view) lacks a body part.\n"
                  "1 = SLEAP right (that part really cannot be seen)   2 = SLEAP wrong (the part is visible)",
    "too_long": "The ringed fish (* view) is much longer than usual.\n"
                "1 = SLEAP right (its points are on one fish)   2 = SLEAP wrong (joins two fish / points off)",
    "head_tail": "The ringed fish (* view): head and tail disagree with the other two views.\n"
                 "1 = SLEAP right in this view   2 = SLEAP wrong here (head-tail swapped / points off)",
    "view_outlier": "The ringed fish (* view) did not fit the other two views in 3D.\n"
                    "1 = SLEAP right in this view   2 = SLEAP wrong here (wrong fish / points off / swapped)",
    "lost": "Yellow ring (* view, key frame): a fish the tracker gave no detection here.\n"
            "Does every fish in the ring have its own magenta instance?\n"
            "1 = yes   2 = no, a visible fish has none of its own   3 = a fish is fully hidden",
    "clean": "Look at all three views.\n"
             "1 = SLEAP right everywhere   2 = SLEAP wrong somewhere (missed fish, false fish, point off the "
             "body, head-tail swapped, two fish joined)",
    "geometry": "Only * is shown. At the KEY FRAME, is *'s 3D body on the real fish? (other frames are only "
                "context; Home goes back)\n1 = right   2 = small error (a little off)   "
                "3 = clear error (wrong place, wrong fish, head-tail swapped)",
    "id": "Only * is shown. Watch the whole clip: does every colour stay on the same fish all the time?\n"
          "1 = right (yes)   2 = wrong (a colour moves to another fish and stays there)   "
          "3 = wrong briefly (moves and comes back soon; one frame is enough)",
}
ANSWERS = {
    "sleap": {"1": "SLEAP right", "2": "SLEAP wrong", "0": "unsure"},
    "clean": {"1": "SLEAP right", "2": "SLEAP wrong", "0": "unsure"},
    "geometry": {"1": "right", "2": "small error", "3": "clear error", "0": "unsure"},
    "id": {"1": "right", "2": "wrong", "3": "wrong briefly", "0": "unsure"},
    "lost": {"1": "detected", "2": "not detected", "3": "hidden", "0": "unsure"},
}
LOST_CATEGORIES = {                 # hidden from the person judging
    "A": "fewer than 3 raw SLEAP instances in this view",
    "B1": ">= 3 instances; a separate unassigned one where the fish should be",
    "B2": ">= 3 instances; the unassigned one there overlaps another fish's detection",
    "C": ">= 3 instances; nothing unassigned where the fish should be",
}
RANK = {"right": 0, "small error": 1, "clear error": 2, "wrong briefly": 1, "wrong": 2}


@dataclass(frozen=True)
class AuditParams:
    per_group: int = 30            # SLEAP items per suspicious group (a group with <= take_all frames: all)
    take_all: int = 50
    n_clean: int = 100             # unflagged frames
    n_geometry: int = 150
    geometry_mm: float = 5.0       # old and new differ: some node further apart than this
    geometry_merge_frames: int = 17
    geometry_pad_frames: int = 50
    id_persist_frames: int = 83    # an old->new name matching must last this long to count
    id_pad_frames: int = 166
    sleap_pad_frames: int = 40
    too_long_ratio: float = 1.6
    seed: int = 20261029
    close_mm: float = 10.0          # lost: two fish closer than this (3D position) = a close encounter
    close_merge_frames: int = 17
    lost_per_group: int = 30
    lost_min_gap_frames: int = 20   # lost items of one group at least this far apart (neighbours look alike)
    near_body: float = 0.5          # lost: an unassigned instance this near (body lengths) = "where the fish is"
    overlap_body: float = 0.25      # ... and this near another fish's detection = overlapping it


def answer_kind(task: str) -> str:
    return task if task in ("clean", "geometry", "id", "lost") else "sleap"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of ``k / n``."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]


# ---------------------------------------------------------------------------
# SLEAP screening


def sleap_flags(folder: Path, raw: RawSlp, t0: int, params: AuditParams):
    """``{category: (F, V) bool}`` from row ``t0`` on, plus ``{category: (F, V) instance slot or -1}``."""
    pts = raw.det                                                     # (F, V, S, N, 2) int16
    none = np.iinfo(np.int16).min
    found = (pts != none).all(axis=-1)                                # (F, V, S, N)
    present = found.any(axis=-1)                                      # (F, V, S)
    count = present.sum(axis=-1)
    flags, slot = {}, {}
    flags["missing"], flags["extra"] = count < 3, count > 3
    incomplete = present & ~found.all(axis=-1)
    flags["incomplete"] = incomplete.any(axis=-1)
    slot["incomplete"] = np.where(flags["incomplete"], incomplete.argmax(axis=-1), -1)
    complete = found.all(axis=-1)
    p = pts.astype(np.float32)
    length = np.zeros(present.shape, np.float32)
    for i, j in raw.edges:
        length += np.linalg.norm(p[..., i, :] - p[..., j, :], axis=-1)
    length[~complete] = np.nan
    ratio = np.full(length.shape, np.nan, np.float32)
    for v in range(length.shape[1]):
        med = np.nanmedian(length[t0:, v])
        ratio[:, v] = length[:, v] / med
    long_ = np.nan_to_num(ratio, nan=0) > params.too_long_ratio
    flags["too_long"] = long_.any(axis=-1)
    slot["too_long"] = np.where(flags["too_long"], np.nan_to_num(ratio, nan=0).argmax(axis=-1), -1)
    with np.load(Path(folder) / "review_data.npz") as z:
        flipped, left_out, kp2d = z["auto_flipped"], z["view_left_out"], z["kp2d"]
    if len(flipped) != len(pts):
        raise ValueError("review_data.npz and slp_raw.npz cover different frames")
    flags["head_tail"] = flipped.any(axis=1)                          # (F, K, V) -> (F, V)
    fish = {"head_tail": np.where(flags["head_tail"], flipped.argmax(axis=1), -1)}
    lo = np.stack([(left_out == v).any(axis=2) for v in range(pts.shape[1])], axis=-1)   # (F, K, V)
    flags["view_outlier"] = lo.any(axis=1)
    fish["view_outlier"] = np.where(flags["view_outlier"], lo.argmax(axis=1), -1)
    for cat in flags:
        flags[cat][:t0] = False
    return flags, slot, fish, kp2d


def nearest_slot(raw_pts: np.ndarray, q: np.ndarray) -> int:
    """Raw instance ``(S, N, 2)`` nearest to keypoints ``q (N, 2)`` (mean node distance), -1 if none."""
    import warnings

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # instances without any found node
        d = np.nanmean(np.linalg.norm(raw_pts - q[None], axis=-1), axis=-1)
    return int(np.nanargmin(d)) if np.isfinite(d).any() else -1


# ---------------------------------------------------------------------------
# old vs new


def _best_perm(a: np.ndarray, b: np.ndarray, pos: int):
    """Per frame the matching old track -> new track (index into permutations of the fish) by position, and
    every fish's largest node distance under that matching ``(T, L)``."""
    perms = list(itertools.permutations(range(a.shape[1])))
    cost = np.stack([np.nansum(np.linalg.norm(a[:, :, pos] - b[:, list(p), pos], axis=-1), axis=1) for p in perms])
    k = np.argmin(cost, axis=0)
    diff = np.stack([np.linalg.norm(a - b[:, list(p)], axis=-1).max(axis=-1) for p in perms])    # (P, T, L)
    return perms, k, diff[k, np.arange(len(k))]


def geometry_runs(diff: np.ndarray, params: AuditParams) -> list[tuple[int, int]]:
    mask = (np.nan_to_num(diff, nan=0) > params.geometry_mm).any(axis=1)
    runs = _runs(mask)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= params.geometry_merge_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def id_events(k: np.ndarray, params: AuditParams) -> list[tuple[int, int]]:
    """Rows ``(a, b)`` between two lasting old->new matchings that differ: one version swapped there."""
    stable = [(a, b, int(k[a])) for a, b in _runs_by_value(k) if b - a + 1 >= params.id_persist_frames]
    out = []
    for (a0, b0, p0), (a1, b1, p1) in zip(stable[:-1], stable[1:]):
        if p0 != p1:
            out.append((b0 + 1, a1))
    return out


def _runs_by_value(x: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(x)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges - 1, len(x) - 1]
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# lost views in close encounters


def close_encounters(xyz: np.ndarray, t0: int, params: AuditParams) -> list[tuple[int, int]]:
    """Rows ``(a, b)`` where the two nearest fish (3D position ``(F, K, 3)``) are closer than ``close_mm``,
    runs less than ``close_merge_frames`` apart merged."""
    pairs = list(itertools.combinations(range(xyz.shape[1]), 2))
    with np.errstate(invalid="ignore"):
        d = np.stack([np.linalg.norm(xyz[:, i] - xyz[:, j], axis=-1) for i, j in pairs], -1)
    mask = (np.nan_to_num(d, nan=np.inf) < params.close_mm).any(axis=1)
    mask[:t0] = False
    merged: list[list[int]] = []
    for a, b in _runs(mask):
        if merged and a - merged[-1][1] <= params.close_merge_frames:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def lost_views(folder: Path, raw: RawSlp, t0: int, params: AuditParams) -> pd.DataFrame:
    """Every (row, fish, view) inside a close encounter where the fish has a detection in fewer than two views
    and none in this one, sorted by the raw SLEAP output of that view (``LOST_CATEGORIES``)."""
    with np.load(Path(folder) / "review_data.npz") as z:
        present, kp2d, proj2d, xyz = z["view_present"], z["kp2d"], z["proj2d"], z["xyz"]
    n_views = present.sum(axis=-1)
    blen = []
    for v in range(present.shape[2]):
        with np.errstate(invalid="ignore"):
            ln = np.linalg.norm(kp2d[t0::50, :, v, 0] - kp2d[t0::50, :, v, -1], axis=-1)
        blen.append(float(np.nanmedian(ln)) if np.isfinite(ln).any() else 100.0)
    rows = []
    for a, b in close_encounters(xyz, t0, params):
        for t in range(a, b + 1):
            for k in np.flatnonzero(n_views[t] < 2):
                pts = raw.points(int(raw.frames[t]))
                for v in np.flatnonzero(~present[t, k]):
                    p = pts[v]                                                   # (S, N, 2)
                    inst = [s for s in range(len(p)) if np.isfinite(p[s]).all(axis=-1).any()]
                    target = proj2d[t, k, v]
                    others = [j for j in range(present.shape[1]) if j != k and present[t, j, v]]

                    def gap(s, j):
                        with np.errstate(invalid="ignore"):
                            return float(np.nanmean(np.linalg.norm(p[s] - kp2d[t, j, v], axis=-1)))

                    if not np.isfinite(target).all():
                        continue
                    if len(inst) < 3:
                        cat = "A"
                    else:
                        free = [s for s in inst if not any(gap(s, j) < 3 for j in others)]
                        cen = [np.nanmean(p[s], axis=0) for s in free]
                        dist = [float(np.linalg.norm(c - target)) / blen[v] for c in cen]
                        if not free or min(dist) >= params.near_body:
                            cat = "C"
                        else:
                            s = free[int(np.argmin(dist))]
                            ov = min([gap(s, j) for j in others] or [np.inf]) / blen[v]
                            cat = "B2" if ov < params.overlap_body else "B1"
                    rows.append(dict(t=t, fish=int(k), view=int(v), category=cat,
                                     x=float(target[0]), y=float(target[1])))
    return pd.DataFrame(rows, columns=["t", "fish", "view", "category", "x", "y"])


def _thinned(t: np.ndarray, n: int, gap: int, rng) -> list[int]:
    """Up to ``n`` random indices into ``t`` whose rows are at least ``gap`` apart."""
    out: list[int] = []
    for i in rng.permutation(len(t)):
        if all(abs(int(t[i]) - int(t[j])) >= gap for j in out):
            out.append(int(i))
            if len(out) == n:
                break
    return out


def add_lost(folder: Path, recon_dir: Path, *, params: AuditParams | None = None, log=print) -> pd.DataFrame:
    """Append the lost-view items to an existing task list (answers kept; the old files are backed up)."""
    params = params or AuditParams()
    out = audit_dir(recon_dir)
    tasks = Audit(recon_dir).tasks
    if (tasks["task"] == "lost").any():
        raise FileExistsError(f"{out / TASKS_FILE} already has lost items")
    raw = RawSlp.open(folder)
    if raw is None:
        raise FileNotFoundError(f"{Path(folder) / 'slp_raw.npz'} missing: run python -m scripts.review.slp_raw")
    summary = json.loads((out / FLAGS_FILE).read_text(encoding="utf-8"))
    frames = raw.frames
    t0 = int(summary["analysis_start"]) - int(frames[0])
    with np.load(Path(folder) / "review_data.npz") as z:
        fish_ids, xyz = [str(f) for f in z["fish_ids"]], z["xyz"]
    cases = lost_views(folder, raw, t0, params)
    rng = np.random.default_rng(params.seed + 1)
    pad = params.sleap_pad_frames
    rows = []
    for cat in LOST_CATEGORIES:
        g = cases[cases["category"] == cat].reset_index(drop=True)
        for i in _thinned(g["t"].to_numpy(), params.lost_per_group, params.lost_min_gap_frames, rng):
            r = g.iloc[i]
            t = int(r["t"])
            rows.append(dict(task="lost", category=cat, start=int(frames[max(0, t - pad)]),
                             end=int(frames[min(len(frames) - 1, t + pad)]), frame=int(frames[t]), view=int(r["view"]),
                             instance=-1, fish=fish_ids[int(r["fish"])], a_is="", b_is="",
                             x=round(float(r["x"]), 1), y=round(float(r["y"]), 1)))
        log(f"[audit] lost {cat}: {len(g)} (fish, view, frame), {sum(x['category'] == cat for x in rows)} to check")
    new = pd.DataFrame(rows).sample(frac=1, random_state=params.seed + 1).reset_index(drop=True)
    new.insert(0, "item", np.arange(len(tasks) + 1, len(tasks) + len(new) + 1))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name in (TASKS_FILE, FLAGS_FILE):
        shutil.copy2(out / name, out / f"{Path(name).stem}.before_lost_{stamp}{Path(name).suffix}")
    both = pd.concat([tasks.astype(str), new.astype(str)], ignore_index=True)
    both[TASK_COLUMNS].to_csv(out / TASKS_FILE, index=False)
    summary["lost"] = {"population": {c: int((cases["category"] == c).sum()) for c in LOST_CATEGORIES},
                       "encounters": len(close_encounters(xyz, t0, params)),
                       "params": {k: getattr(params, k) for k in ("close_mm", "close_merge_frames", "lost_per_group",
                                                                  "lost_min_gap_frames", "near_body", "overlap_body")},
                       "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / FLAGS_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[audit] {len(new)} lost items appended -> {out / TASKS_FILE} (items {len(tasks) + 1}-{len(both)})")
    return new


# ---------------------------------------------------------------------------
# build / answers / stats


def audit_dir(recon_dir: Path) -> Path:
    return Path(recon_dir) / AUDIT_DIR


def build(folder: Path, recon_dir: Path, *, old: str | None = None, new: str | None = None,
          params: AuditParams | None = None, log=print) -> pd.DataFrame:
    """Write the task list (refuses to overwrite one that has answers)."""
    params = params or AuditParams()
    out = audit_dir(recon_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / ANSWERS_FILE).exists() and len(pd.read_csv(out / ANSWERS_FILE)):
        raise FileExistsError(f"{out / ANSWERS_FILE} has answers; move it aside to build a new task list")
    rng = np.random.default_rng(params.seed)
    t_start = time.time()
    s = ReconState(folder, recon_dir)
    old = old or s.versions[1]
    new = new or s.versions[2]
    frames = s.frames
    start = s.analysis_start if s.analysis_start is not None else int(frames[0])
    t0 = start - int(frames[0])
    rows = []

    def add(task, category, a, b, frame, view=-1, instance=-1, fish="", ab=("", "")):
        rows.append(dict(task=task, category=category, start=int(frames[max(0, a)]),
                         end=int(frames[min(len(frames) - 1, b)]), frame=int(frame), view=int(view),
                         instance=int(instance), fish=fish, a_is=ab[0], b_is=ab[1], x="", y=""))

    # SLEAP
    raw = RawSlp.open(folder)
    if raw is None:
        raise FileNotFoundError(f"{Path(folder) / 'slp_raw.npz'} missing: run python -m scripts.review.slp_raw")
    flags, slot, fish, kp2d = sleap_flags(folder, raw, t0, params)
    n_rows = len(frames) - t0
    summary = {"analysis_start": start, "analysis_frames": int(n_rows), "views": 3, "groups": {}}
    pad = params.sleap_pad_frames
    for cat, mask in flags.items():
        tv = np.argwhere(mask)
        summary["groups"][cat] = {"flagged_view_frames": int(len(tv)),
                                  "flagged_frames": int(mask.any(axis=1).sum()),
                                  "share_of_view_frames": float(len(tv) / (3 * n_rows))}
        take = tv if len(tv) <= params.take_all else tv[np.sort(rng.choice(len(tv), params.per_group, replace=False))]
        for t, v in take:
            inst = int(slot[cat][t, v]) if cat in slot else -1
            if cat in fish and fish[cat][t, v] >= 0:
                inst = nearest_slot(raw.points(int(frames[t]))[v], kp2d[t, fish[cat][t, v], v])
            add("sleap", cat, t - pad, t + pad, frames[t], v, inst)
        log(f"[audit] SLEAP {cat}: {len(tv)} flagged view-frames, {len(take)} to check")
    any_flag = np.zeros(flags["missing"].shape, bool)
    for m in flags.values():
        any_flag |= m
    nviews = any_flag.sum(axis=1)
    summary["frames_flagged_in_1_view"] = int((nviews[t0:] == 1).sum())
    summary["frames_flagged_in_2plus_views"] = int((nviews[t0:] >= 2).sum())
    clean = np.flatnonzero(nviews == 0)
    clean = clean[clean >= t0]
    for t in np.sort(rng.choice(clean, min(params.n_clean, len(clean)), replace=False)):
        add("clean", "clean", t - pad, t + pad, frames[t])
    log(f"[audit] SLEAP unflagged: {len(clean)} frames, {min(params.n_clean, len(clean))} to check")

    # old vs new
    a = s.series(old, t0, len(frames), False)
    b = s.series(new, t0, len(frames), False)
    perms, k, diff = _best_perm(a, b, s.pos_node)
    runs = geometry_runs(diff, params)
    summary["geometry"] = {"runs": len(runs), "fish_frames_differ": float(np.mean(np.nan_to_num(diff) > params.geometry_mm)),
                           "frames_differ": int(sum(r1 - r0 + 1 for r0, r1 in runs))}
    pick = np.sort(rng.choice(len(runs), min(params.n_geometry, len(runs)), replace=False))
    aligned = s.series(old, t0, len(frames), True)
    for i in pick:
        r0, r1 = runs[i]
        # The key frame (the one judged): a random frame of the stretch where they really differ (merged
        # stretches have equal frames in between); the ring goes on the fish most apart there.
        dd = np.nan_to_num(diff[r0:r1 + 1], nan=0)
        mid = r0 + int(rng.choice(np.flatnonzero((dd > params.geometry_mm).any(axis=1))))
        kk = int(np.argmax(np.nan_to_num(diff[mid], nan=-1)))
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(aligned[mid, :, s.pos_node] - a[mid, kk, s.pos_node], axis=-1)
        name = s.labels[int(np.nanargmin(d))] if np.isfinite(d).any() else ""
        ab = (old, new) if rng.random() < 0.5 else (new, old)
        add("geometry", "geometry", t0 + r0 - params.geometry_pad_frames, t0 + r1 + params.geometry_pad_frames,
            frames[t0 + mid], fish=name, ab=ab)
    log(f"[audit] geometry: {len(runs)} stretches where old and new differ, {len(pick)} to judge")
    events = id_events(k, params)
    summary["id"] = {"events": len(events)}
    for e0, e1 in events:
        ab = (old, new) if rng.random() < 0.5 else (new, old)
        add("id", "id", t0 + e0 - params.id_pad_frames, t0 + e1 + params.id_pad_frames,
            frames[t0 + (e0 + e1) // 2], ab=ab)
    log(f"[audit] id: {len(events)} places where old's and new's identities stop matching (all judged)")

    tasks = pd.DataFrame(rows)
    order = {"id": 0, "geometry": 1}
    tasks["_o"] = tasks["task"].map(lambda x: order.get(x, 2))
    tasks["_r"] = rng.random(len(tasks))                            # shuffled within a task kind
    tasks = tasks.sort_values(["_o", "_r"]).drop(columns=["_o", "_r"]).reset_index(drop=True)
    tasks.insert(0, "item", np.arange(1, len(tasks) + 1))
    tasks[TASK_COLUMNS].to_csv(out / TASKS_FILE, index=False)
    summary.update(old=old, new=new, params=asdict(params), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / FLAGS_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[audit] {len(tasks)} items -> {out / TASKS_FILE} ({time.time() - t_start:.0f} s)")
    return tasks


class Audit:
    """Task list + answers of one ``audit/`` folder. Old-vs-new items have two parts (A, B), SLEAP items one."""

    def __init__(self, recon_dir: Path):
        self.dir = audit_dir(recon_dir)
        self.tasks = pd.read_csv(self.dir / TASKS_FILE, keep_default_na=False)
        for col in TASK_COLUMNS:                     # task lists written before a column existed
            if col not in self.tasks:
                self.tasks[col] = ""
        self.summary = json.loads((self.dir / FLAGS_FILE).read_text(encoding="utf-8"))
        self._task = dict(zip(self.tasks["item"].astype(int), self.tasks["task"]))
        path = self.dir / ANSWERS_FILE
        self.log = (pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists()
                    else pd.DataFrame(columns=ANSWER_COLUMNS))
        if len(self.log) and list(self.log.columns) != ANSWER_COLUMNS:
            raise ValueError(f"{path} has columns {list(self.log.columns)}, expected {ANSWER_COLUMNS}")
        self.answers: dict[tuple[int, str], tuple[str, str]] = {}
        for r in self.log.itertuples(index=False):
            self._apply(int(r.item), str(r.part), str(r.answer), str(r.note))

    def _apply(self, item: int, part: str, answer: str, note: str):
        if answer == "" and note == "":
            self.answers.pop((item, part), None)
        else:
            self.answers[(item, part)] = (answer, note)

    def parts(self, item: int) -> tuple[str, ...]:
        return PARTS if self._task[item] in ("id", "geometry") else ("",)

    def code(self, item: int, part: str = "") -> str:
        return self.answers.get((item, part), ("", ""))[0]

    def answer(self, item: int, answer: str, note: str | None = None, part: str = ""):
        if part not in self.parts(item):
            raise ValueError(f"item {item} has parts {self.parts(item)}, not {part!r}")
        kind = answer_kind(self._task[item])
        if answer != "" and answer not in ANSWERS[kind]:
            raise ValueError(f"answer {answer!r} not one of {list(ANSWERS[kind])} for {kind}")
        if note is None:
            note = self.answers.get((item, part), ("", ""))[1]
        rec = pd.DataFrame([dict(item=item, part=part, answer=answer, note=note.replace("\n", " "),
                                 created=time.strftime("%Y-%m-%d %H:%M:%S"))])
        path = self.dir / ANSWERS_FILE
        rec[ANSWER_COLUMNS].to_csv(path, mode="a", header=not path.exists(), index=False)
        self._apply(item, part, answer, note)

    def label(self, item: int, part: str = "") -> str:
        return ANSWERS[answer_kind(self._task[item])].get(self.code(item, part), "")

    def summary_text(self, item: int) -> str:
        """``A: right, B: wrong`` / ``SLEAP wrong`` for the item list."""
        parts = self.parts(item)
        if parts == ("",):
            return self.label(item)
        return ", ".join(f"{p}: {self.label(item, p) or '-'}" for p in parts)

    def done(self, item: int) -> bool:
        return all(self.code(item, p) != "" for p in self.parts(item))

    def progress(self) -> dict[str, tuple[int, int]]:
        out = {}
        for task, g in self.tasks.groupby("task", sort=False):
            out[task] = (sum(self.done(int(i)) for i in g["item"]), len(g))
        return out

    # -- results ----------------------------------------------------------------
    def stats(self) -> dict[str, pd.DataFrame]:
        t = self.tasks.copy()
        t["answer"] = [self.label(int(i)) if self.parts(int(i)) == ("",) else "" for i in t["item"]]
        out = {}
        rows = []
        n_view_frames = 3 * self.summary["analysis_frames"]
        for cat in SLEAP_CATEGORIES:
            g = t[(t["task"] == "sleap") & (t["category"] == cat)]
            judged = g[g["answer"].isin(["SLEAP right", "SLEAP wrong"])]
            k, n = int((judged["answer"] == "SLEAP wrong").sum()), len(judged)
            lo, hi = wilson(k, n)
            flagged = self.summary["groups"][cat]["flagged_view_frames"]
            rows.append(dict(group=cat, meaning=SLEAP_CATEGORIES[cat], flagged=flagged,
                             flagged_pct=100 * flagged / n_view_frames, checked=n, real=k,
                             precision=k / n if n else math.nan, precision_lo=lo, precision_hi=hi,
                             est_real_view_frames=flagged * k / n if n else math.nan,
                             est_real_pct=100 * flagged * k / n / n_view_frames if n else math.nan))
        out["sleap"] = pd.DataFrame(rows)
        g = t[t["task"] == "clean"]
        judged = g[g["answer"].isin(["SLEAP right", "SLEAP wrong"])]
        k, n = int((judged["answer"] == "SLEAP wrong").sum()), len(judged)
        lo, hi = wilson(k, n)
        out["clean"] = pd.DataFrame([dict(checked=n, with_error=k, rate=k / n if n else math.nan, rate_lo=lo,
                                          rate_hi=hi, rule_of_three_upper=3 / n if n and k == 0 else math.nan)])
        for kind in ("geometry", "id"):
            out[kind] = self.unblinded(kind)
        out["lost"] = self.lost_stats(t)
        return out

    def lost_stats(self, t: pd.DataFrame) -> pd.DataFrame:
        """Per lost group the answer shares; the ``all`` row weights each group by its size in the recording."""
        pop = self.summary.get("lost", {}).get("population", {})
        labels = [v for c, v in ANSWERS["lost"].items() if c != "0"]
        rows = []
        for cat in LOST_CATEGORIES:
            g = t[(t["task"] == "lost") & (t["category"] == cat)]
            ans = [self.label(int(i)) for i in g["item"]]
            judged = [a for a in ans if a in labels]
            row = dict(group=cat, meaning=LOST_CATEGORIES[cat], population=int(pop.get(cat, 0)), checked=len(judged),
                       unsure=ans.count("unsure"))
            for lab in labels:
                row[lab] = judged.count(lab) / len(judged) if judged else math.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        if len(df) and df["checked"].sum():
            ok = df[df["checked"] > 0]
            w = ok["population"] / ok["population"].sum()
            total = dict(group="all", meaning="weighted by group size", population=int(df["population"].sum()),
                         checked=int(df["checked"].sum()), unsure=int(df["unsure"].sum()))
            total.update({lab: float((ok[lab] * w).sum()) for lab in labels})
            df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df

    def unblinded(self, kind: str) -> pd.DataFrame:
        """One row per item: each version's answer (old / new, unblinded) and which one is better."""
        old, new = self.summary["old"], self.summary["new"]
        rows = []
        for r in self.tasks[self.tasks["task"] == kind].itertuples(index=False):
            item = int(r.item)
            got = {r.a_is: self.label(item, "A"), r.b_is: self.label(item, "B")}
            o, n = got.get(old, ""), got.get(new, "")
            if o in RANK and n in RANK:
                better = "same" if RANK[o] == RANK[n] else (old if RANK[o] < RANK[n] else new)
            else:
                better = ""
            rows.append(dict(item=item, old=o, new=n, better=better))
        return pd.DataFrame(rows, columns=["item", "old", "new", "better"])


def sign_test(old_wins: int, new_wins: int) -> float:
    """Two-sided binomial p of ``old_wins`` vs ``new_wins`` (ties left out)."""
    n = old_wins + new_wins
    if n == 0:
        return math.nan
    from scipy.stats import binomtest

    return float(binomtest(old_wins, n, 0.5).pvalue)


def print_stats(recon_dir: Path) -> None:
    a = Audit(recon_dir)
    s = a.stats()
    old, new = a.summary["old"], a.summary["new"]
    pd.set_option("display.width", 200)
    print(f"progress (items with every part answered): {a.progress()}")
    print("\n== SLEAP: suspicious groups (view-frames after the analysis start); real = SLEAP wrong")
    print(s["sleap"].round(3).to_string(index=False))
    print(f"frames flagged in 1 view: {a.summary['frames_flagged_in_1_view']}, in 2+ views: "
          f"{a.summary['frames_flagged_in_2plus_views']} (of {a.summary['analysis_frames']})")
    print("\n== SLEAP: unflagged frames")
    print(s["clean"].round(4).to_string(index=False))
    for kind in ("geometry", "id"):
        u = s[kind]
        print(f"\n== {kind}: each version on its own (unblinded)")
        for v in ("old", "new"):
            counts = u[v].replace("", "(not judged)").value_counts()
            print(f"  {a.summary[v]}: " + ", ".join(f"{k} {c}" for k, c in counts.items()))
        b = u["better"].value_counts()
        ow, nw = int(b.get(old, 0)), int(b.get(new, 0))
        print(f"  better: {old} {ow}, {new} {nw}, same {int(b.get('same', 0))}; sign test p = {sign_test(ow, nw):.3g}")
    if (a.tasks["task"] == "lost").any():
        print("\n== lost views in close encounters (population = (fish, view, frame) cases; shares of judged items)")
        print("   detected = SLEAP found the fish, the tracker did not use it; not detected = SLEAP missed a visible "
              "fish; hidden = not visible")
        print(s["lost"].round(3).to_string(index=False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["build", "add-lost", "stats"])
    p.add_argument("folder", type=Path)
    p.add_argument("--recon-dir", type=Path, required=True)
    p.add_argument("--old")
    p.add_argument("--new")
    p.add_argument("--per-group", type=int, default=AuditParams.per_group)
    p.add_argument("--n-geometry", type=int, default=AuditParams.n_geometry)
    p.add_argument("--n-clean", type=int, default=AuditParams.n_clean)
    args = p.parse_args(argv)
    if args.command == "build":
        build(args.folder, args.recon_dir, old=args.old, new=args.new,
              params=AuditParams(per_group=args.per_group, n_geometry=args.n_geometry, n_clean=args.n_clean))
    elif args.command == "add-lost":
        add_lost(args.folder, args.recon_dir)
    else:
        print_stats(args.recon_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
