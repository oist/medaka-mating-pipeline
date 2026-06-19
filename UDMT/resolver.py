from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from UDMT.io import require_columns

"""Read-only crossing resolver: score detector crossing events with
medaka-native cues and emit a provisional "which fish is more anomalous" vote
with a confidence margin.

This is deliberately a *diagnostic*, not a repairer: it never edits tracks and
never touches the production 3D path. Each crossing gets either a low-margin
``review`` flag (the honest default for ambiguous smooth swaps) or a high-margin
``suspect:<fish>`` provisional call for downstream review. Applying a correction
is a separate, later, gated step.

v1 uses the two cleanly-available, fish_id-keyed cues:
- 3D speed jerk over the crossing window (velocity std, max/mean step ratio) —
  higher = more anomalous;
- top-camera bind reprojection distance (`bind_score` from the
  *_3d_identified.csv, in px) — higher = worse match = more anomalous.

Documented extension cues (not yet wired, need extra plumbing / slot mapping):
2D top-cam speed + SLEAP confidence (via `source_top_id`), per-fish reprojection
residual (from `*_best_pre_link.json`), and mask quality.
"""


@dataclass(frozen=True)
class ResolverConfig:
    id_column: str = "fish_id"
    coordinate_columns: tuple[str, ...] = ("X", "Y", "Z")
    window_pad_frames: int = 30
    min_points_for_speed: int = 4
    margin_threshold: float = 0.34
    w_velstd: float = 1.0
    w_maxmean: float = 1.0
    w_bind: float = 1.5
    w_residual: float = 2.0
    residual_match_gate: float = 5.0
    review_only: bool = False


@dataclass(frozen=True)
class ResolverSummary:
    crossings_scored: int
    suspect_calls: int
    review_flags: int
    undecided_no_cues: int
    margin_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_crossings(
    events: pd.DataFrame,
    tracks_3d: pd.DataFrame,
    identified: pd.DataFrame | None = None,
    pre_link: dict | str | None = None,
    config: ResolverConfig | None = None,
) -> tuple[pd.DataFrame, ResolverSummary]:
    """Score crossing events read-only; return (decisions_df, summary).

    ``pre_link`` may be a loaded ``*_best_pre_link.json`` dict or a path to one;
    when given, a per-fish 3D reprojection-residual cue is added (slots a/b/c are
    mapped to fish_id by nearest 3D position per frame).
    """

    cfg = config or ResolverConfig()
    require_columns(
        events,
        ("event_id", "track_id_a", "track_id_b", "start_frame", "end_frame",
         "min_frame", "min_distance", "duration_frames"),
        label="events",
    )
    xyz_index = _per_fish_xyz(tracks_3d, cfg)
    bind_index = _per_fish_bind(identified, cfg) if identified is not None else {}

    ev = events
    if cfg.review_only and "review_candidate" in ev.columns:
        ev = ev[ev["review_candidate"].astype(bool)]
    ev_rows = list(ev.itertuples(index=False))

    # Build the per-fish residual index only over the frames we actually score.
    residual_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if pre_link is not None and ev_rows:
        needed: set[int] = set()
        for e in ev_rows:
            lo = int(e.start_frame) - int(cfg.window_pad_frames)
            hi = int(e.end_frame) + int(cfg.window_pad_frames)
            needed.update(range(lo, hi + 1))
        pre_link_obj = _load_pre_link(pre_link)
        residual_index = _per_fish_residual(pre_link_obj, xyz_index, needed, cfg)

    rows: list[dict] = []
    for e in ev_rows:
        lo = int(e.start_frame) - int(cfg.window_pad_frames)
        hi = int(e.end_frame) + int(cfg.window_pad_frames)
        a, b = str(e.track_id_a), str(e.track_id_b)

        velstd_a, maxmean_a = _speed_cues(_window_xyz(xyz_index, a, lo, hi), cfg)
        velstd_b, maxmean_b = _speed_cues(_window_xyz(xyz_index, b, lo, hi), cfg)
        bind_a = _window_mean(bind_index, a, lo, hi)
        bind_b = _window_mean(bind_index, b, lo, hi)
        resid_a = _window_max_finite(residual_index, a, lo, hi)
        resid_b = _window_max_finite(residual_index, b, lo, hi)

        # Signed cue scores in [-1, 1]; +1 => fish A looks MORE anomalous.
        s_velstd = _signed(velstd_a, velstd_b)          # higher std = anomalous
        s_maxmean = _signed(maxmean_a, maxmean_b)        # higher ratio = anomalous
        s_bind = _signed(bind_a, bind_b)                 # bind_score = reproj distance (px): higher = worse match = anomalous
        s_residual = _signed(resid_a, resid_b)           # higher 3D reproj residual = worse triangulation = anomalous

        weighted = [
            (cfg.w_velstd, s_velstd),
            (cfg.w_maxmean, s_maxmean),
            (cfg.w_bind, s_bind),
            (cfg.w_residual, s_residual),
        ]
        num = sum(w * s for w, s in weighted if np.isfinite(s))
        den = sum(w for w, s in weighted if np.isfinite(s))
        n_cues = sum(1 for _, s in weighted if np.isfinite(s))
        net = float(num / den) if den > 0 else float("nan")
        margin = abs(net) if np.isfinite(net) else float("nan")

        if not np.isfinite(net):
            decision, suspect = "review", ""
        elif margin >= float(cfg.margin_threshold):
            suspect = a if net > 0 else b
            decision = f"suspect:{suspect}"
        else:
            decision, suspect = "review", ""

        rows.append({
            "event_id": e.event_id,
            "track_id_a": a,
            "track_id_b": b,
            "min_frame": int(e.min_frame),
            "min_distance": float(e.min_distance),
            "duration_frames": int(e.duration_frames),
            "review_candidate": bool(getattr(e, "review_candidate", False)),
            "velstd_a": velstd_a, "velstd_b": velstd_b,
            "maxmean_a": maxmean_a, "maxmean_b": maxmean_b,
            "bind_a": bind_a, "bind_b": bind_b,
            "resid_a": resid_a, "resid_b": resid_b,
            "n_cues": int(n_cues),
            "net_vote": net,
            "margin": margin,
            "suspect_fish": suspect,
            "decision": decision,
        })

    decisions = pd.DataFrame(rows)
    suspects = int(decisions["decision"].str.startswith("suspect:").sum()) if not decisions.empty else 0
    undecided = int((decisions["n_cues"] == 0).sum()) if not decisions.empty else 0
    reviews = int(len(decisions) - suspects)
    summary = ResolverSummary(
        crossings_scored=int(len(decisions)),
        suspect_calls=suspects,
        review_flags=reviews,
        undecided_no_cues=undecided,
        margin_threshold=float(cfg.margin_threshold),
    )
    return decisions, summary


def _signed(a: float, b: float) -> float:
    """Normalized (a - b) in [-1, 1]; +1 means a >> b. NaN if either is NaN."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    denom = abs(a) + abs(b)
    return 0.0 if denom <= 1e-12 else float((a - b) / denom)


def _speed_cues(xyz: np.ndarray, cfg: ResolverConfig) -> tuple[float, float]:
    if len(xyz) < int(cfg.min_points_for_speed):
        return float("nan"), float("nan")
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if len(steps) == 0:
        return float("nan"), float("nan")
    mean = float(steps.mean())
    velstd = float(steps.std())
    maxmean = float(steps.max() / mean) if mean > 1e-9 else float("nan")
    return velstd, maxmean


def _per_fish_xyz(tracks_3d: pd.DataFrame, cfg: ResolverConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    require_columns(tracks_3d, ("frame", cfg.id_column, *cfg.coordinate_columns), label="tracks_3d")
    df = tracks_3d.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    for col in cfg.coordinate_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frame", *cfg.coordinate_columns]).sort_values("frame")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fish, g in df.groupby(cfg.id_column, sort=False):
        out[str(fish)] = (
            g["frame"].to_numpy(dtype=int),
            g[list(cfg.coordinate_columns)].to_numpy(dtype=float),
        )
    return out


def _per_fish_bind(identified: pd.DataFrame, cfg: ResolverConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if "bind_score" not in identified.columns:
        return {}
    require_columns(identified, ("frame", cfg.id_column), label="identified")
    df = identified.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["bind_score"] = pd.to_numeric(df["bind_score"], errors="coerce")
    df = df.dropna(subset=["frame"]).sort_values("frame")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fish, g in df.groupby(cfg.id_column, sort=False):
        out[str(fish)] = (g["frame"].to_numpy(dtype=int), g["bind_score"].to_numpy(dtype=float))
    return out


def _window_xyz(index: dict, fish: str, lo: int, hi: int) -> np.ndarray:
    item = index.get(fish)
    if item is None:
        return np.empty((0, 3), dtype=float)
    frames, xyz = item
    i0 = int(np.searchsorted(frames, lo, side="left"))
    i1 = int(np.searchsorted(frames, hi, side="right"))
    return xyz[i0:i1]


def _window_mean(index: dict, fish: str, lo: int, hi: int) -> float:
    item = index.get(fish)
    if item is None:
        return float("nan")
    frames, values = item
    i0 = int(np.searchsorted(frames, lo, side="left"))
    i1 = int(np.searchsorted(frames, hi, side="right"))
    window = values[i0:i1]
    finite = window[np.isfinite(window)]
    return float(finite.mean()) if finite.size else float("nan")


def _load_pre_link(pre_link: dict | str | Path) -> dict:
    if isinstance(pre_link, dict):
        return pre_link
    with open(pre_link, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pos_at(xyz_index: dict, fish: str, frame: int) -> np.ndarray | None:
    item = xyz_index.get(fish)
    if item is None:
        return None
    frames, xyz = item
    i = int(np.searchsorted(frames, frame, side="left"))
    if i < len(frames) and int(frames[i]) == int(frame):
        return xyz[i]
    return None


def _per_fish_residual(
    pre_link_obj: dict,
    xyz_index: dict,
    needed_frames: set[int],
    cfg: ResolverConfig,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map per-frame triangulation-slot reprojection errors to fish_id by nearest
    3D position, restricted to ``needed_frames``. Returns {fish_id: (frames, residuals)}."""

    frames_payload = pre_link_obj.get("frames", {})
    fishes = list(xyz_index.keys())
    acc_frames: dict[str, list[int]] = defaultdict(list)
    acc_resid: dict[str, list[float]] = defaultdict(list)

    for frame in sorted(needed_frames):
        rec = frames_payload.get(str(frame))
        if rec is None:
            continue
        groups = rec.get("groups", {}) or {}
        errors = rec.get("errors", {}) or {}
        pos_by_fish = {f: p for f in fishes if (p := _pos_at(xyz_index, f, frame)) is not None}
        if not pos_by_fish:
            continue
        for slot, xyz in groups.items():
            arr = np.asarray(xyz, dtype=float)
            if arr.size < 3 or not np.all(np.isfinite(arr)):
                continue
            best, best_d = None, float(cfg.residual_match_gate)
            for fish, p in pos_by_fish.items():
                d = float(np.linalg.norm(arr[:3] - p[:3]))
                if d <= best_d:
                    best_d, best = d, fish
            if best is None:
                continue
            try:
                res = float(errors.get(slot, float("nan")))
            except (TypeError, ValueError):
                res = float("nan")
            acc_frames[best].append(int(frame))
            acc_resid[best].append(res)

    return {
        fish: (np.asarray(acc_frames[fish], dtype=int), np.asarray(acc_resid[fish], dtype=float))
        for fish in acc_frames
    }


def _window_max_finite(index: dict, fish: str, lo: int, hi: int) -> float:
    item = index.get(fish)
    if item is None:
        return float("nan")
    frames, values = item
    i0 = int(np.searchsorted(frames, lo, side="left"))
    i1 = int(np.searchsorted(frames, hi, side="right"))
    window = values[i0:i1]
    finite = window[np.isfinite(window)]
    return float(finite.max()) if finite.size else float("nan")
