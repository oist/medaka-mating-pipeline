"""Machine danger intervals and the per-frame status of an identity pass (``id_pass_gui.py``).

A reviewer watches a recording once. The machine only sets the pace and the bookkeeping: a frame is
*danger* where two fish are close or the 3D is unreliable (played slower than safe frames), and a danger
run is *high risk* where they come very close. The reviewer's identity changes and danger marks then
decide which frames can be used:

* ``WATCHED_SAFE`` - shown, not in a danger run;
* ``WATCHED_DANGER`` - shown, in a danger run in which the reviewer changed nothing;
* ``UNUSABLE`` - a danger run holding an identity change, or a range marked by hand;
* ``UNWATCHED`` - not shown yet.

P_1_1 (2026-10-30, against the full review): 15 mm with safe stretches >= 1 s leaves no swap in a safe
frame; 10 mm does (frame 183060). 8 mm separates the high-risk runs: 35 of 307, holding 13 of the 14
runs with a swap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.review.ledger import ALL_FISH, snap_frame

NOT_ANALYSED, UNWATCHED, WATCHED_SAFE, WATCHED_DANGER, UNUSABLE = -1, 0, 1, 2, 3
LEDGER_FOLDER = "id_pass"
CHANGE_MARGIN_S = 0.5     # an identity change outside every danger run makes this much around it unusable
SEGMENT_COLUMNS = ["start", "end", "seconds", "danger_frames", "names"]


@dataclass(frozen=True)
class SegmentParams:
    safe_mm: float = 15.0        # closer than this: danger
    high_risk_mm: float = 8.0    # a danger run coming this close is high risk
    min_safe_s: float = 1.0      # shorter safe stretches join the danger around them
    max_step_mm: float = 6.0     # a larger 3D step in one frame is unreliable (about 1 m/s)
    slow_pad_s: float = 0.1      # slow down this long before and after a danger run


def runs_of(mask: np.ndarray) -> np.ndarray:
    """``(n, 2)`` half-open ``[start, end)`` index runs where ``mask`` is true."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]


def pair_distances(xyz: np.ndarray) -> np.ndarray:
    """``(F, pairs)`` distance of every pair of tracks; NaN where one has no position."""
    n = xyz.shape[1]
    return np.stack([np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1) for i in range(n) for j in range(i + 1, n)], 1)


def unreliable_frames(xyz: np.ndarray, present: np.ndarray, interpolated: np.ndarray, max_step_mm: float) -> np.ndarray:
    """A fish without a position, interpolated, seen in fewer than two views, or jumping."""
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=2)
    jump = np.r_[np.zeros(1, bool), (np.nan_to_num(step) > max_step_mm).any(axis=1)]
    return ~np.isfinite(xyz).all(axis=(1, 2)) | interpolated.any(axis=1) | (present.sum(axis=2) < 2).any(axis=1) | jump


def danger_map(xyz: np.ndarray, present: np.ndarray, interpolated: np.ndarray, fps: float,
               params: SegmentParams = SegmentParams()) -> dict:
    """``danger``, ``slow`` and ``high_frames`` frame masks, the danger ``runs`` and which are ``high`` risk."""
    nearest = np.nan_to_num(pair_distances(xyz), nan=np.inf).min(axis=1)
    danger = (nearest < params.safe_mm) | unreliable_frames(xyz, present, interpolated, params.max_step_mm)
    min_safe = int(round(params.min_safe_s * fps))
    for a, b in runs_of(~danger):
        if b - a < min_safe:
            danger[a:b] = True
    runs = runs_of(danger)
    high = np.array([nearest[a:b].min() < params.high_risk_mm for a, b in runs], dtype=bool)
    slow, high_frames = np.zeros(len(danger), bool), np.zeros(len(danger), bool)
    pad = int(round(params.slow_pad_s * fps))
    for (a, b), is_high in zip(runs, high):
        slow[max(0, a - pad):b + pad] = True
        high_frames[a:b] = is_high
    return {"danger": danger, "slow": slow, "high_frames": high_frames, "runs": runs, "high": high, "nearest": nearest}


def place_change(frame: int, first: int, distance: np.ndarray, runs: np.ndarray, not_before: int, window: int) -> int:
    """Frame an identity change entered at ``frame`` belongs to.

    The key comes after the reaction time, so the change goes back to the closest approach of the two
    tracks in the latest danger run of the last ``window`` frames, else of the whole window. It never
    goes back to or past ``not_before``: an earlier change touching either track does not commute with it.
    """
    lo = max(frame - window, not_before)
    frames = np.arange(len(distance)) + first
    t, t_lo = frame - first, lo - first
    near = np.flatnonzero((runs[:, 0] <= t) & (runs[:, 1] - 1 > t_lo)) if len(runs) else np.zeros(0, int)
    if len(near):
        a, b = runs[near[-1]]
        lo, frame = max(lo, first + a - 1), min(frame, first + b - 1)
    return snap_frame(frames, distance, lo, frame)


def manual_ranges(events: pd.DataFrame) -> list[tuple[int, int]]:
    """Inclusive frame ranges marked as danger by hand."""
    e = events[events["kind"] == "danger"]
    return [(int(a), int(b)) for a, b in zip(e["frame"], e["frame_end"])]


def pose_ranges(events: pd.DataFrame) -> list[tuple[int, int, list[str]]]:
    """Inclusive frame ranges marked as wrong pose by hand, with their tracks."""
    e = events[events["kind"] == "pose_bad"]
    return [(int(g["frame"].iloc[0]), int(g["frame_end"].iloc[0]), list(g["fish"]))
            for _group, g in e.groupby("group", sort=False)]


def frame_status(first: int, n: int, start: int | None, danger: np.ndarray, runs: np.ndarray,
                 watched: list[tuple[int, int]], changes: list[int], manual: list[tuple[int, int]],
                 fps: float) -> np.ndarray:
    """``(n,)`` status of every frame; nothing before the analysis start counts."""
    status = np.full(n, UNWATCHED, np.int8)
    for a, b in watched:
        lo, hi = max(a - first, 0), min(b - first + 1, n)
        if hi > lo:
            status[lo:hi] = np.where(danger[lo:hi], WATCHED_DANGER, WATCHED_SAFE)
    margin = int(round(CHANGE_MARGIN_S * fps))
    for f in changes:
        t = f - first
        hit = np.flatnonzero((runs[:, 0] <= t) & (t < runs[:, 1])) if len(runs) else []
        if len(hit):
            a, b = runs[hit[0]]
            status[a:b] = UNUSABLE
        else:
            status[max(0, t - margin):max(0, t + margin + 1)] = UNUSABLE
    for a, b in manual:
        status[max(0, a - first):max(0, b - first + 1)] = UNUSABLE
    if start is not None:
        status[:max(0, start - first)] = NOT_ANALYSED
    return status


def usable_segments(status: np.ndarray, first: int, fps: float, names_at) -> pd.DataFrame:
    """Runs of watched, usable frames; ``names_at(t)`` gives the names on the tracks at index ``t``."""
    rows = []
    for a, b in runs_of(np.isin(status, (WATCHED_SAFE, WATCHED_DANGER))):
        rows.append({"start": first + int(a), "end": first + int(b) - 1, "seconds": round((b - a) / fps, 2),
                     "danger_frames": int((status[a:b] == WATCHED_DANGER).sum()), "names": names_at(int(a))})
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def summary(status: np.ndarray, segments: pd.DataFrame, fps: float) -> dict:
    """Shares of the analysed frames and the lengths of the usable segments."""
    analysed = max(1, int((status != NOT_ANALYSED).sum()))
    share = lambda *codes: round(100 * float(np.isin(status, codes).sum()) / analysed, 2)  # noqa: E731
    seconds = segments["seconds"] if len(segments) else pd.Series([0.0])
    return {"analysed_seconds": round(analysed / fps, 1),
            "watched_pct": share(WATCHED_SAFE, WATCHED_DANGER, UNUSABLE),
            "usable_pct": share(WATCHED_SAFE, WATCHED_DANGER),
            "usable_safe_pct": share(WATCHED_SAFE),
            "usable_danger_pct": share(WATCHED_DANGER),
            "unusable_pct": share(UNUSABLE),
            "segments": len(segments),
            "median_segment_s": round(float(seconds.median()), 1),
            "longest_segment_s": round(float(seconds.max()), 1)}


def play_seconds(frames_safe: int, frames_slow: int, fps: float, speed: float, slow_speed: float) -> float:
    """Playback time of that many frames at the two speeds."""
    return frames_safe / (fps * speed) + frames_slow / (fps * min(speed, slow_speed))


__all__ = ["ALL_FISH", "SegmentParams", "danger_map", "frame_status", "manual_ranges", "place_change",
           "play_seconds", "pose_ranges", "runs_of", "summary", "usable_segments"]
