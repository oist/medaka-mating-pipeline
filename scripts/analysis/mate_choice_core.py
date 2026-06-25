"""Shared, tested core for per-male mate-choice / courtship / aggression metrics.

Single source of truth for the behavioural quantities that the dual-line 3-fish
individual-ID track unlocks (female pinned per-frame + the two males distinguished,
so behaviour is attributable to a SPECIFIC male). Consumed by:

  * ``scripts/analysis/analyze_mate_choice.py``      — the integrated per-experiment report,
  * ``scripts/analysis/compute_courtship_metrics.py`` — courtship/association CLI,
  * ``scripts/analysis/compute_aggression_metrics.py``— male-male aggression CLI.

Pure NumPy/pandas (no torch/cv2); every function is unit-tested in
``tests/test_mate_choice_core.py``. Distances are in mm, speeds in mm/s, time in s.
Input is the interp 3-fish track (cols ``frame``, ``id_final`` ∈ {F,M1,M2}, ``X,Y,Z``).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 166.0
FISH = ("F", "M1", "M2")


@dataclass
class BehaviourParams:
    """Thresholds shared by all three CLIs (one place to keep them consistent)."""
    fps: float = FPS
    court_mm: float = 30.0          # female-male courtship/association range
    contact_mm: float = 12.0        # contact (close to an explicit display / spawn)
    agg_mm: float = 25.0            # male-male close range for an aggressive approach
    agg_speed: float = 150.0        # mm/s speed spike for an aggressive lunge
    min_bout: int = 33              # frames (~0.2 s @166fps) for a sustained bout
    n_bins: int = 10                # temporal-dynamics bins over the timeline

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- loading
def load_individual_track(path, id_col: str = "id_final") -> tuple[dict[str, np.ndarray], int]:
    """Load the 3-fish track into per-fish XYZ arrays indexed by absolute frame.

    Returns ``(pos, n)`` where ``pos[f]`` is an ``(n, 3)`` float array (NaN where the
    fish has no finite 3D position that frame) for ``f`` in {F, M1, M2}, and ``n`` is
    ``max(frame) + 1``. Missing fish/frames are NaN-filled so downstream distance/speed
    functions propagate NaN rather than crash.
    """
    t = pd.read_csv(path, usecols=["frame", id_col, "X", "Y", "Z"])
    return track_from_frame(t, id_col=id_col)


def track_from_frame(t: pd.DataFrame, id_col: str = "id_final") -> tuple[dict[str, np.ndarray], int]:
    """Same as :func:`load_individual_track` but from an in-memory DataFrame (testable).

    ``aggfunc="first"`` is deterministic: a well-formed track has exactly one row per
    (frame, id_final) so it is a no-op, but a malformed track with a duplicate row keeps
    one position rather than silently averaging two (which would fabricate a 3D point the
    fish was never at).
    """
    piv = t.pivot_table(index="frame", columns=id_col, values=["X", "Y", "Z"], aggfunc="first")
    n = int(piv.index.max()) + 1
    piv = piv.reindex(range(n))
    pos = {}
    for f in FISH:
        if (("X", f) in piv.columns):
            pos[f] = np.c_[piv["X"][f].to_numpy(), piv["Y"][f].to_numpy(), piv["Z"][f].to_numpy()]
        else:
            pos[f] = np.full((n, 3), np.nan)
    return pos, n


# --------------------------------------------------------------------- geometry helpers
def pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean distance, NaN where either position is missing."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    bad = np.isnan(a).any(1) | np.isnan(b).any(1)
    d = np.sqrt(np.sum((a - b) ** 2, axis=1))
    return np.where(bad, np.nan, d)


def speed(a: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Per-frame speed (mm/s); first frame and any NaN-bounded step are NaN."""
    a = np.asarray(a, float)
    v = np.full(len(a), np.nan)
    if len(a) > 1:
        step = np.sqrt(np.sum(np.diff(a, axis=0) ** 2, axis=1)) * fps
        v[1:] = step
    return v


def runs_over(mask, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs of length >= ``min_len`` -> list of (start, end) inclusive.

    NaN entries in ``mask`` count as False (a missing frame breaks a run).
    """
    idx = np.where(np.nan_to_num(np.asarray(mask, float), nan=0.0).astype(bool))[0]
    if not len(idx):
        return []
    out, splits = [], np.where(np.diff(idx) > 1)[0] + 1
    for r in np.split(idx, splits):
        if len(r) >= min_len:
            out.append((int(r[0]), int(r[-1])))
    return out


def _bout_seconds(bouts: list[tuple[int, int]], fps: float = FPS) -> float:
    return sum(e - s + 1 for s, e in bouts) / fps


def _safe_nanmean(x) -> float:
    """np.nanmean without the 'Mean of empty slice' RuntimeWarning on all-NaN input."""
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _safe_nanmedian(x) -> float:
    """np.nanmedian without the 'All-NaN slice' RuntimeWarning on all-NaN input."""
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


# --------------------------------------------------------------- distances bundle
@dataclass
class Distances:
    dFM1: np.ndarray
    dFM2: np.ndarray
    dMM: np.ndarray
    spdM1: np.ndarray
    spdM2: np.ndarray
    n: int


def compute_distances(pos: dict[str, np.ndarray], fps: float = FPS) -> Distances:
    F, M1, M2 = pos["F"], pos["M1"], pos["M2"]
    return Distances(
        dFM1=pairwise_distance(F, M1),
        dFM2=pairwise_distance(F, M2),
        dMM=pairwise_distance(M1, M2),
        spdM1=speed(M1, fps),
        spdM2=speed(M2, fps),
        n=len(F),
    )


# --------------------------------------------------------------- association / mate choice
def association_metrics(d: Distances, p: BehaviourParams) -> dict:
    """Female mate-choice association: time near each male, preference index, sharpening."""
    dFM1, dFM2 = d.dFM1, d.dFM2
    valid = ~np.isnan(dFM1) & ~np.isnan(dFM2)
    t1 = float(np.nansum(dFM1 < p.court_mm) / p.fps)
    t2 = float(np.nansum(dFM2 < p.court_mm) / p.fps)
    assoc_index = t1 / (t1 + t2) if (t1 + t2) else float("nan")  # >0.5 => prefers M1
    close_any = valid & (np.fmin(dFM1, dFM2) < p.court_mm)
    pref_M1 = int(np.nansum(close_any & (dFM1 <= dFM2)))
    pref_M2 = int(np.nansum(close_any & (dFM2 < dFM1)))
    closer_M1_frac = float(np.nanmean((valid & (dFM1 < dFM2)).astype(float))) if valid.any() else float("nan")

    sharpening = {}
    for thr in (30, 20, 12, 8):
        a = float(np.nansum(dFM1 < thr) / p.fps)
        b = float(np.nansum(dFM2 < thr) / p.fps)
        sharpening[thr] = round(max(a, b) / max(min(a, b), 1e-6), 3)

    return {
        "time_near_M1_s": round(t1, 2),
        "time_near_M2_s": round(t2, 2),
        "association_index_M1": round(assoc_index, 4),
        "preferred_male": "M1" if (assoc_index > 0.5) else "M2",
        "closer_to_M1_frac": round(closer_M1_frac, 4),
        "pref_frames_M1": pref_M1,
        "pref_frames_M2": pref_M2,
        "sharpening_ratio_by_mm": sharpening,
        "mean_d_F_M1_mm": round(_safe_nanmean(dFM1), 2),
        "mean_d_F_M2_mm": round(_safe_nanmean(dFM2), 2),
        "mean_d_M1_M2_mm": round(_safe_nanmean(d.dMM), 2),
        "valid_FM_frames": int(valid.sum()),
    }


# ------------------------------------------------------------------- courtship bouts
def courtship_bouts(d: Distances, p: BehaviourParams) -> dict:
    """Per-male sustained-proximity (courtship) bouts."""
    b1 = runs_over(d.dFM1 < p.court_mm, p.min_bout)
    b2 = runs_over(d.dFM2 < p.court_mm, p.min_bout)
    # round to 1 dp (the display precision): rounding to 2 dp then formatting :.1f hits a
    # banker's-rounding boundary that flips e.g. 271.253 -> 271.25 -> 271.2 (vs the gold 271.3).
    return {
        "M1": {"n_bouts": len(b1), "total_s": round(_bout_seconds(b1, p.fps), 1), "bouts": b1},
        "M2": {"n_bouts": len(b2), "total_s": round(_bout_seconds(b2, p.fps), 1), "bouts": b2},
    }


# ----------------------------------------------------------------- aggression bouts
def aggression_bouts(d: Distances, p: BehaviourParams) -> list[dict]:
    """Male-male close+fast (aggressive-approach) bouts with the faster initiator.

    A bout is a sustained run where the males are within ``agg_mm`` AND at least one
    exceeds ``agg_speed``. ``initiator`` is the male with the higher mean speed over the
    bout (the lunging fish). Uses a shorter min length than courtship (lunges are brief).
    """
    agg_mask = (d.dMM < p.agg_mm) & (np.fmax(d.spdM1, d.spdM2) > p.agg_speed)
    runs = runs_over(agg_mask, max(3, p.min_bout // 3))
    out = []
    for s, e in runs:
        m1 = float(np.nanmean(d.spdM1[s:e + 1]))
        m2 = float(np.nanmean(d.spdM2[s:e + 1]))
        out.append({
            "start": s, "end": e, "dur_s": round((e - s + 1) / p.fps, 3),
            "min_d_mm": round(float(np.nanmin(d.dMM[s:e + 1])), 2),
            "mean_spd_M1": round(m1, 1), "mean_spd_M2": round(m2, 1),
            # strict '>' (a tie attributes to M2), matching the original analyze_mate_choice
            "initiator": "M1" if m1 > m2 else "M2",
        })
    return out


def aggression_summary(bouts: list[dict]) -> dict:
    n = len(bouts)
    init_M1 = sum(1 for b in bouts if b["initiator"] == "M1")
    return {
        "n_bouts": n,
        "total_s": round(sum(b["dur_s"] for b in bouts), 2),
        "initiator_M1": init_M1,
        "initiator_M2": n - init_M1,
        "initiator_M1_frac": round(init_M1 / n, 3) if n else float("nan"),
    }


# ----------------------------------------------------------------- temporal dynamics
def temporal_preference(d: Distances, p: BehaviourParams) -> tuple[pd.DataFrame, dict]:
    """Windowed female preference over the timeline — does it BUILD or SWITCH?

    Splits the timeline into ``n_bins`` equal frame bins; per bin computes the
    association index (M1 share of female courtship-range time). Returns a per-bin
    table plus a summary: early-vs-late preference, linear trend slope, and the number
    of bins where the preferred male flips (a "switch").
    """
    n = d.n
    edges = np.linspace(0, n, p.n_bins + 1).astype(int)
    rows = []
    for b in range(p.n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        seg1, seg2 = d.dFM1[lo:hi], d.dFM2[lo:hi]
        t1 = float(np.nansum(seg1 < p.court_mm) / p.fps)
        t2 = float(np.nansum(seg2 < p.court_mm) / p.fps)
        idx = t1 / (t1 + t2) if (t1 + t2) else np.nan
        rows.append({
            "bin": b, "frame_lo": int(lo), "frame_hi": int(hi),
            "t_start_s": round(lo / p.fps, 1), "t_end_s": round(hi / p.fps, 1),
            "court_M1_s": round(t1, 2), "court_M2_s": round(t2, 2),
            "pref_index_M1": round(idx, 4) if np.isfinite(idx) else np.nan,
        })
    table = pd.DataFrame(rows)

    valid = table.dropna(subset=["pref_index_M1"])
    summary: dict = {"n_bins_with_data": int(len(valid))}
    if len(valid) >= 2:
        x = valid["bin"].to_numpy(float)
        y = valid["pref_index_M1"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        third = max(1, len(valid) // 3)
        early = float(valid["pref_index_M1"].iloc[:third].mean())
        late = float(valid["pref_index_M1"].iloc[-third:].mean())
        chosen = "M1" if y.mean() > 0.5 else "M2"
        # a switch = a bin on the opposite side of 0.5 from the overall choice
        opposite = ((y > 0.5) != (y.mean() > 0.5))
        switches = int(opposite.sum())
        if abs(late - early) < 0.08:
            trend = "stable"
        elif (chosen == "M1" and late > early) or (chosen == "M2" and late < early):
            trend = "builds"        # preference toward the chosen male strengthens
        else:
            trend = "weakens"
        summary.update({
            "early_pref_index_M1": round(early, 4),
            "late_pref_index_M1": round(late, 4),
            "slope_per_bin": round(slope, 4),
            "overall_preferred_male": chosen,
            "early_predicts_choice": bool((early > 0.5) == (chosen == "M1")),
            "n_switch_bins": switches,
            "trend": trend,
        })
    return table, summary


# ----------------------------------------------------------------- spawn attribution
def spawn_attribution(d: Distances, p: BehaviourParams, spawn_frame: int | None) -> dict | None:
    """Which male the female is in contact with at the spawn (the choice outcome)."""
    if spawn_frame is None or spawn_frame < 0 or spawn_frame >= d.n:
        return None
    w = slice(max(0, spawn_frame - p.min_bout), min(d.n, spawn_frame + p.min_bout))
    s1 = _safe_nanmedian(d.dFM1[w])
    s2 = _safe_nanmedian(d.dFM2[w])
    # guard the all-NaN window (female/males occluded right at the contact moment): do NOT
    # silently default to a male — report spawn_male=None so the caller flags it.
    if np.isnan(s1) and np.isnan(s2):
        male, near = None, float("nan")
    elif np.isnan(s2):
        male, near = "M1", s1
    elif np.isnan(s1):
        male, near = "M2", s2
    else:
        male = "M1" if s1 <= s2 else "M2"
        near = min(s1, s2)
    return {
        "spawn_frame": int(spawn_frame),
        "spawn_time_s": round(spawn_frame / p.fps, 1),
        "median_d_F_M1_mm": round(s1, 2),
        "median_d_F_M2_mm": round(s2, 2),
        "spawn_male": male,
        "contact": bool(np.isfinite(near) and near < p.contact_mm),
    }


# --------------------------------------------------------------------- unified bouts
def bout_table(d: Distances, p: BehaviourParams, courtship=None, aggression=None) -> pd.DataFrame:
    """Unified per-bout breakdown: courtship (per male) + male-male aggression, time-sorted.

    ``courtship``/``aggression`` may be passed in (the dict from :func:`courtship_bouts` and
    the list from :func:`aggression_bouts`) to avoid recomputing them when the caller already has.
    """
    rows = []
    cb = courtship if courtship is not None else courtship_bouts(d, p)
    agg = aggression if aggression is not None else aggression_bouts(d, p)
    for male in ("M1", "M2"):
        for s, e in cb[male]["bouts"]:
            rows.append({"type": "courtship", "participants": f"F-{male}", "start": s, "end": e,
                         "dur_s": round((e - s + 1) / p.fps, 3), "initiator": "", "min_d_mm": round(float(np.nanmin((d.dFM1 if male == "M1" else d.dFM2)[s:e + 1])), 2)})
    for b in agg:
        rows.append({"type": "aggression", "participants": "M1-M2", "start": b["start"], "end": b["end"],
                     "dur_s": b["dur_s"], "initiator": b["initiator"], "min_d_mm": b["min_d_mm"]})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("start").reset_index(drop=True)
    return table


def behaviour_breakdown(d: Distances, p: BehaviourParams, courtship=None, aggression=None) -> dict:
    """Top-level courtship-vs-aggression summary tallies."""
    cb = courtship if courtship is not None else courtship_bouts(d, p)
    agg = aggression if aggression is not None else aggression_bouts(d, p)
    court_total = cb["M1"]["total_s"] + cb["M2"]["total_s"]
    agg_total = round(sum(b["dur_s"] for b in agg), 2)
    return {
        "courtship_total_s": round(court_total, 2),
        "courtship_M1_s": cb["M1"]["total_s"], "courtship_M2_s": cb["M2"]["total_s"],
        "aggression_total_s": agg_total,
        "courtship_to_aggression_ratio": round(court_total / agg_total, 3) if agg_total else float("inf"),
        "n_courtship_bouts": cb["M1"]["n_bouts"] + cb["M2"]["n_bouts"],
        "n_aggression_bouts": len(agg),
    }
