from __future__ import annotations

"""
Active-learning frame-mining + annotation-seed tool for the next SLEAP retraining
round on the medaka mating pipeline.

What it does
------------
1. Walks every inferred experiment under ``sleap_output_dir`` and, per camera
   view, streams the merged ``.slp`` predictions into a per-frame feature
   parquet file (``training_frame_mining/<exp>/<view>.parquet``).
2. Computes per-view active-learning signals (S1 stationary, S2 mouth/tail
   flip, S5 score dispersion) and cross-view signals (S3 count mismatch,
   S4 reprojection residual where available).
3. Selects ``--target-count`` (default 500) candidate frames with stratified
   sampling across experiment / view / time bucket.
4. Builds an annotation seed ``pkg.slp`` (frames embedded) with a two-node
   ``mouth -> tail`` skeleton. ``mouth`` keeps the predicted position;
   ``tail`` is seeded a few pixels below it for the annotator to drag.

Outputs (relative to ``--out-dir``)
-----------------------------------
- ``mined_frames.csv``         — every candidate row (full diagnostic table).
- ``annotation_seed.pkg.slp``  — SLEAP labels project, frames embedded.
- ``mining_report.txt``        — signal frequency + diversification summary.
- ``training_frame_mining/<exp>/<view>.parquet`` — per-view feature cache.

CLI
---
    python -m scripts.sleap.mine_training_frames \
        --paths-config configs/paths.local.toml \
        --target-count 500

For a smoke run on one experiment::

    python -m scripts.sleap.mine_training_frames \
        --paths-config configs/paths.local.toml \
        --experiments 20251225_K_1_2_FCMC \
        --target-count 80
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths


VIEWS = ("top_cam", "side1_cam", "side2_cam")

DEFAULT_FPS = 166.0
DEFAULT_SKIP_SECONDS = 5 * 60.0
DEFAULT_TARGET_COUNT = 500
DEFAULT_TIME_BINS = 30
DEFAULT_MIN_INTER_PICK_FRAMES = 200
DEFAULT_NN_GATE_PX = 50.0
DEFAULT_STATIONARY_WINDOW = 100
DEFAULT_STATIONARY_STD_PX = 1.0
DEFAULT_TAIL_SEED_OFFSET_PX = 8.0  # tail placed close to mouth (BL unknown).
DEFAULT_S2_LO_FRAC = 0.6
DEFAULT_S2_HI_FRAC = 1.5
DEFAULT_S4_RESIDUAL = 5000.0
DEFAULT_S5_DISP = 0.3
DEFAULT_S5_MIN_SCORE = 0.6

SIGNAL_WEIGHTS: dict[str, float] = {
    "S1": 1.0,   # stationary instance (reflection)
    "S2": 1.5,   # ~body-length jump (mouth/tail flip)
    "S3": 2.0,   # cross-view count mismatch (max-3 cap dropping)
    "S4": 1.5,   # high reprojection residual
    "S5": 1.0,   # score dispersion
    "S6": 2.0,   # weak 3D/ID result (NaN XYZ or geometry-only top bind)
    "S7": 1.0,   # high top-camera reprojection bind distance
}


# ---------------------------------------------------------------------------
# Per-view feature extraction (streamed, parquet-cached).
# ---------------------------------------------------------------------------

def _per_view_cache_path(cache_root: Path, experiment: str, view: str) -> Path:
    return cache_root / experiment / f"{view}.parquet"


def _slp_path_for(
    slp_root: Path,
    experiment: str,
    view: str,
    *,
    sleap_subdir: str | None = None,
) -> Path:
    base = slp_root / experiment
    if sleap_subdir:
        base = base / sleap_subdir
    return base / f"{view}.slp"


def _is_cache_fresh(cache_path: Path, slp_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        return cache_path.stat().st_mtime >= slp_path.stat().st_mtime
    except OSError:
        return False


def _instance_xy_score(inst) -> tuple[float, float, float]:
    """Centroid (x, y) over visible nodes plus the instance score."""
    pts = inst.numpy()
    mask = np.isfinite(pts).all(axis=1)
    score = getattr(inst, "score", None)
    score_f = float(score) if score is not None else float("nan")
    if not mask.any():
        return float("nan"), float("nan"), score_f
    cx, cy = pts[mask].mean(axis=0)
    return float(cx), float(cy), score_f


def _simple_nn_track(
    rows: list[dict],
    *,
    gate_px: float,
    max_gap_frames: int = 30,
) -> list[int]:
    """
    Greedy nearest-neighbour tracker over per-instance rows already grouped by
    frame index. Returns a track-id list aligned with ``rows``. Each row must
    have ``frame``, ``x``, ``y`` keys.
    """
    track_ids: list[int] = [-1] * len(rows)
    if not rows:
        return track_ids

    # Group row indices by frame (preserving file order).
    by_frame: dict[int, list[int]] = {}
    for idx, r in enumerate(rows):
        by_frame.setdefault(int(r["frame"]), []).append(idx)
    frame_order = sorted(by_frame.keys())

    next_track = 0
    # active: list of dicts {track, last_frame, last_x, last_y}
    active: list[dict] = []

    for f in frame_order:
        idxs = by_frame[f]
        # Drop tracks that have aged out.
        active = [t for t in active if (f - t["last_frame"]) <= max_gap_frames]

        if active and idxs:
            # Greedy: repeatedly take the closest (track, instance) pair under gate.
            inst_xy = [(rows[i]["x"], rows[i]["y"]) for i in idxs]
            unmatched_inst = set(range(len(idxs)))
            unmatched_trk = set(range(len(active)))
            cost: list[tuple[float, int, int]] = []
            for ti, t in enumerate(active):
                for ii, (x, y) in enumerate(inst_xy):
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    d = math.hypot(x - t["last_x"], y - t["last_y"])
                    if d <= gate_px * max(1, f - t["last_frame"]):
                        cost.append((d, ti, ii))
            cost.sort()
            for _d, ti, ii in cost:
                if ti in unmatched_trk and ii in unmatched_inst:
                    unmatched_trk.discard(ti)
                    unmatched_inst.discard(ii)
                    t = active[ti]
                    track_ids[idxs[ii]] = t["track"]
                    t["last_frame"] = f
                    t["last_x"], t["last_y"] = inst_xy[ii]
            # New tracks for unmatched detections.
            for ii in sorted(unmatched_inst):
                x, y = inst_xy[ii]
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                track_ids[idxs[ii]] = next_track
                active.append({
                    "track": next_track,
                    "last_frame": f,
                    "last_x": x,
                    "last_y": y,
                })
                next_track += 1
        else:
            for ii in idxs:
                x, y = rows[ii]["x"], rows[ii]["y"]
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                track_ids[ii] = next_track
                active.append({
                    "track": next_track,
                    "last_frame": f,
                    "last_x": x,
                    "last_y": y,
                })
                next_track += 1

    return track_ids


def _build_view_features(
    slp_path: Path,
    *,
    skip_frame: int,
    nn_gate_px: float,
    stationary_window: int,
    stationary_std_px: float,
    s2_lo_frac: float,
    s2_hi_frac: float,
    s5_disp: float,
    s5_min_score: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Stream a merged per-view ``.slp`` into a per-frame feature dataframe.

    Returns
    -------
    df : pd.DataFrame
        One row per (frame). Columns:
            frame, n_inst, max_score, min_score,
            x_first, y_first, score_first,
            s1, s2, s5
    meta : dict
        Per-view metadata: BL_px estimate, total frames, video filename.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required. `uv sync --extra sleap` or `pip install sleap-io`."
        ) from exc

    labels = sio.load_slp(str(slp_path))
    video_filename = (
        labels.videos[0].filename if labels.videos else None
    )

    inst_rows: list[dict] = []
    frame_summaries: dict[int, dict] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        if fi < skip_frame:
            continue
        scores: list[float] = []
        per_frame: list[dict] = []
        for inst in lf.instances:
            x, y, sc = _instance_xy_score(inst)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            scores.append(sc if math.isfinite(sc) else 0.0)
            per_frame.append({"frame": fi, "x": x, "y": y, "score": sc})
        if not per_frame:
            continue
        # Sort by descending score so "first" instance is the top-1.
        per_frame.sort(
            key=lambda r: -(r["score"] if math.isfinite(r["score"]) else -1.0)
        )
        for r in per_frame:
            inst_rows.append(r)
        frame_summaries[fi] = {
            "n_inst": len(per_frame),
            "max_score": max(scores) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "first_x": per_frame[0]["x"],
            "first_y": per_frame[0]["y"],
            "first_score": per_frame[0]["score"],
        }

    if not inst_rows:
        empty = pd.DataFrame(
            columns=[
                "frame", "n_inst", "max_score", "min_score",
                "x_first", "y_first", "score_first",
                "s1", "s2", "s5",
            ]
        )
        return empty, {
            "video_filename": video_filename,
            "bl_px_est": float("nan"),
            "n_frames": 0,
            "n_instances": 0,
        }

    track_ids = _simple_nn_track(inst_rows, gate_px=nn_gate_px)
    inst_df = pd.DataFrame(inst_rows)
    inst_df["track"] = track_ids
    inst_df = inst_df[inst_df["track"] >= 0].copy()

    # Per-track step length (consecutive frames only).
    inst_df.sort_values(["track", "frame"], inplace=True)
    inst_df["dx"] = inst_df.groupby("track")["x"].diff()
    inst_df["dy"] = inst_df.groupby("track")["y"].diff()
    inst_df["df"] = inst_df.groupby("track")["frame"].diff()
    valid_step = inst_df["df"] == 1
    inst_df["step_px"] = np.where(
        valid_step, np.hypot(inst_df["dx"], inst_df["dy"]), np.nan
    )

    # BL estimate: 99.5th percentile of step length. Most consecutive steps
    # are tiny (fish slow/stationary), so a low percentile under-estimates BL
    # by an order of magnitude. The 99.5th percentile lives in the heavy tail
    # where genuine mouth<->tail flips and large repositioning steps live.
    # We also stash some lower percentiles in `meta` for inspection.
    finite_steps = inst_df["step_px"].dropna()
    bl_pct = {}
    if len(finite_steps) > 100:
        for q in (50, 90, 95, 99, 99.5, 99.9):
            bl_pct[q] = float(np.percentile(finite_steps, q))
        bl_px_est = bl_pct[99.5]
    else:
        bl_px_est = float("nan")
    if math.isnan(bl_px_est) or bl_px_est <= 0:
        bl_px_est = 30.0  # safe default; signals will be weak but not crash.
    s2_lo = s2_lo_frac * bl_px_est
    s2_hi = s2_hi_frac * bl_px_est

    # Rolling per-track position std over `stationary_window` frames (rows-based;
    # gaps of >1 frame slightly distort but the threshold is loose).
    inst_df["roll_x_std"] = (
        inst_df.groupby("track")["x"]
        .rolling(stationary_window, min_periods=stationary_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    inst_df["roll_y_std"] = (
        inst_df.groupby("track")["y"]
        .rolling(stationary_window, min_periods=stationary_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    inst_df["roll_std"] = np.hypot(
        inst_df["roll_x_std"].fillna(np.inf),
        inst_df["roll_y_std"].fillna(np.inf),
    )

    inst_df["s1_inst"] = (inst_df["roll_std"] < stationary_std_px).fillna(False)
    # S2: step within [lo, hi] AND track has a valid step both directions
    # (i.e. previous step was finite — which it is by construction here — and
    # the next adjacency is also tight). Use a forward-shift to require continuity.
    inst_df["next_df"] = inst_df.groupby("track")["frame"].diff(-1).abs()
    s2_band = (inst_df["step_px"] >= s2_lo) & (inst_df["step_px"] <= s2_hi)
    s2_continuous = inst_df["next_df"] == 1
    inst_df["s2_inst"] = (s2_band & s2_continuous).fillna(False)

    # Reduce per-instance flags up to per-frame.
    s1_per_frame = inst_df.groupby("frame")["s1_inst"].any()
    s2_per_frame = inst_df.groupby("frame")["s2_inst"].any()

    sf = pd.DataFrame.from_dict(frame_summaries, orient="index")
    sf.index.name = "frame"
    sf.reset_index(inplace=True)
    sf["s1"] = sf["frame"].map(s1_per_frame).fillna(False).astype(bool)
    sf["s2"] = sf["frame"].map(s2_per_frame).fillna(False).astype(bool)
    sf["s5"] = (
        (sf["max_score"] - sf["min_score"] > s5_disp)
        & (sf["min_score"] < s5_min_score)
        & (sf["n_inst"] >= 2)
    ).fillna(False).astype(bool)

    sf.rename(
        columns={"first_x": "x_first", "first_y": "y_first", "first_score": "score_first"},
        inplace=True,
    )
    sf = sf[[
        "frame", "n_inst", "max_score", "min_score",
        "x_first", "y_first", "score_first",
        "s1", "s2", "s5",
    ]].sort_values("frame").reset_index(drop=True)

    return sf, {
        "video_filename": video_filename,
        "bl_px_est": bl_px_est,
        "step_pct": bl_pct,
        "n_frames": int(sf.shape[0]),
        "n_instances": int(inst_df.shape[0]),
    }


def build_features_for_experiment(
    experiment: str,
    *,
    slp_root: Path,
    sleap_subdir: str | None = None,
    cache_root: Path,
    skip_frame: int,
    nn_gate_px: float = DEFAULT_NN_GATE_PX,
    stationary_window: int = DEFAULT_STATIONARY_WINDOW,
    stationary_std_px: float = DEFAULT_STATIONARY_STD_PX,
    s2_lo_frac: float = DEFAULT_S2_LO_FRAC,
    s2_hi_frac: float = DEFAULT_S2_HI_FRAC,
    s5_disp: float = DEFAULT_S5_DISP,
    s5_min_score: float = DEFAULT_S5_MIN_SCORE,
    force: bool = False,
    log: bool = True,
) -> dict[str, dict]:
    """
    Build (or load from cache) per-view feature dataframes for one experiment.

    Returns a mapping ``view -> {"df": DataFrame, "meta": dict}``.
    """
    out: dict[str, dict] = {}
    for view in VIEWS:
        slp_path = _slp_path_for(
            slp_root, experiment, view, sleap_subdir=sleap_subdir
        )
        if not slp_path.exists():
            if log:
                print(f"  [{experiment}/{view}] missing .slp, skip")
            continue
        cache_path = _per_view_cache_path(cache_root, experiment, view)
        meta_path = cache_path.with_suffix(".meta.json")
        if (not force) and _is_cache_fresh(cache_path, slp_path) and meta_path.exists():
            df = pd.read_parquet(cache_path)
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if log:
                print(
                    f"  [{experiment}/{view}] cached: "
                    f"{len(df)} frames, BL~{meta.get('bl_px_est'):.1f} px"
                )
            out[view] = {"df": df, "meta": meta}
            continue

        t0 = time.time()
        df, meta = _build_view_features(
            slp_path,
            skip_frame=skip_frame,
            nn_gate_px=nn_gate_px,
            stationary_window=stationary_window,
            stationary_std_px=stationary_std_px,
            s2_lo_frac=s2_lo_frac,
            s2_hi_frac=s2_hi_frac,
            s5_disp=s5_disp,
            s5_min_score=s5_min_score,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        if log:
            print(
                f"  [{experiment}/{view}] built in {time.time() - t0:.1f}s: "
                f"{len(df)} frames, BL~{meta.get('bl_px_est'):.1f} px"
            )
        out[view] = {"df": df, "meta": meta}
    return out


# ---------------------------------------------------------------------------
# Cross-view aggregation + signal fusion.
# ---------------------------------------------------------------------------

def _load_pre_link_residuals(
    pre_link_json_path: Path,
    *,
    threshold: float = DEFAULT_S4_RESIDUAL,
) -> set[int]:
    """
    Return the set of frame indices in which any per-fish reprojection error
    in the pre-link JSON exceeds ``threshold``. Frames with all-None errors
    (e.g. survived but had NaN error) are skipped.
    """
    if not pre_link_json_path.exists():
        return set()
    with pre_link_json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    flagged: set[int] = set()
    for f_str, info in payload.get("frames", {}).items():
        errors = info.get("errors", {}) or {}
        for v in errors.values():
            if v is not None and v > threshold:
                flagged.add(int(f_str))
                break
    return flagged


def _load_identified_failure_frames(
    identified_csv_path: Path,
    *,
    bind_score_quantile: float = 0.95,
) -> tuple[set[int], set[int], dict]:
    """
    Return (S6 frames, S7 frames, metadata) from a *_3d_identified.csv.

    S6 is intentionally broad: any frame with NaN XYZ or a geometry-only row
    tells us the SLEAP/triangulation/bind stack was under stress.
    S7 focuses on top-camera binds whose reprojection distance is in the
    high tail of accepted matches.
    """
    if not identified_csv_path.exists():
        return set(), set(), {}

    usecols = lambda c: c in {
        "frame", "X", "Y", "Z", "bind_source", "bind_score",
    }
    df = pd.read_csv(identified_csv_path, usecols=usecols)
    s6_frames: set[int] = set()
    if {"X", "Y", "Z"}.issubset(df.columns):
        xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)
        bad_xyz = ~np.isfinite(xyz).all(axis=1)
        s6_frames.update(df.loc[bad_xyz, "frame"].astype(int).tolist())
    if "bind_source" in df.columns:
        geom_only = df["bind_source"].astype(str) == "geometry_only"
        s6_frames.update(df.loc[geom_only, "frame"].astype(int).tolist())

    s7_frames: set[int] = set()
    threshold = float("nan")
    if "bind_score" in df.columns:
        scores = pd.to_numeric(df["bind_score"], errors="coerce").dropna()
        if not scores.empty:
            threshold = float(scores.quantile(bind_score_quantile))
            high_bind = pd.to_numeric(df["bind_score"], errors="coerce") >= threshold
            s7_frames.update(df.loc[high_bind, "frame"].astype(int).tolist())

    meta = {
        "identified_csv": str(identified_csv_path),
        "s6_frames": len(s6_frames),
        "s7_frames": len(s7_frames),
        "s7_bind_score_threshold": threshold,
        "s7_bind_score_quantile": bind_score_quantile,
    }
    return s6_frames, s7_frames, meta


def aggregate_experiment(
    experiment: str,
    per_view: dict[str, dict],
    *,
    pre_link_json_path: Path | None = None,
    identified_csv_path: Path | None = None,
    bind_score_quantile: float = 0.95,
    fps: float,
    skip_frame: int,
    time_bins: int,
) -> pd.DataFrame:
    """
    Join per-view dataframes for one experiment into a single per-(view, frame)
    candidate table with all signals, suspicion score, and time bin.
    """
    s4_frames: set[int] = set()
    if pre_link_json_path is not None:
        s4_frames = _load_pre_link_residuals(pre_link_json_path)
    s6_frames: set[int] = set()
    s7_frames: set[int] = set()
    if identified_csv_path is not None:
        s6_frames, s7_frames, _ = _load_identified_failure_frames(
            identified_csv_path,
            bind_score_quantile=bind_score_quantile,
        )

    # Per-frame n_inst counts across views (for S3).
    n_inst_per_view: dict[str, dict[int, int]] = {}
    for view, payload in per_view.items():
        df = payload["df"]
        if df.empty:
            n_inst_per_view[view] = {}
        else:
            n_inst_per_view[view] = dict(zip(df["frame"].astype(int), df["n_inst"].astype(int)))

    union_frames: set[int] = set()
    for d in n_inst_per_view.values():
        union_frames.update(d.keys())

    # Estimate experiment max_frame (used for time-bin edges).
    max_frame = max(union_frames) if union_frames else skip_frame + 1

    rows: list[dict] = []
    for view, payload in per_view.items():
        df = payload["df"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            f = int(r["frame"])
            n_top = n_inst_per_view.get("top_cam", {}).get(f, 0)
            n_s1 = n_inst_per_view.get("side1_cam", {}).get(f, 0)
            n_s2 = n_inst_per_view.get("side2_cam", {}).get(f, 0)
            n_max = max(n_top, n_s1, n_s2)
            n_min_active = min(
                n for n in (n_top, n_s1, n_s2) if n > 0
            ) if any(n > 0 for n in (n_top, n_s1, n_s2)) else 0
            s3_gap = n_max - n_min_active if n_min_active else 0
            s3 = s3_gap >= 1
            s4 = f in s4_frames
            s6 = f in s6_frames
            s7 = f in s7_frames

            signals: list[str] = []
            if bool(r["s1"]):
                signals.append("S1")
            if bool(r["s2"]):
                signals.append("S2")
            if s3:
                signals.append("S3")
            if s4:
                signals.append("S4")
            if bool(r["s5"]):
                signals.append("S5")
            if s6:
                signals.append("S6")
            if s7:
                signals.append("S7")

            score = sum(SIGNAL_WEIGHTS[s] for s in signals)
            # Time bin: linear partition over (skip_frame, max_frame).
            span = max(1, max_frame - skip_frame)
            tb = int(min(time_bins - 1, max(0, ((f - skip_frame) * time_bins) // span)))

            rows.append({
                "experiment": experiment,
                "view": view,
                "frame": f,
                "n_inst": int(r["n_inst"]),
                "n_top": n_top,
                "n_side1": n_s1,
                "n_side2": n_s2,
                "max_score": float(r["max_score"]),
                "min_score": float(r["min_score"]),
                "x_first": float(r["x_first"]),
                "y_first": float(r["y_first"]),
                "score_first": float(r["score_first"]),
                "s3_gap": s3_gap,
                "signals_fired": ";".join(signals),
                "n_signals": len(signals),
                "suspicion_score": score,
                "time_bin": tb,
                "time_sec": (f - skip_frame) / fps,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selection: stratified greedy across (experiment, view, time_bin).
# ---------------------------------------------------------------------------

def apply_force_include(
    selected: pd.DataFrame,
    force_include_csv: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Force ``picked = True`` for every (experiment, view, frame) listed in
    ``force_include_csv`` (columns: experiment, view, frame). Rows already in
    the candidate table are flipped in place; rows that are absent (e.g. a
    crossing frame that produced no mined signal) are appended as minimal
    ``forced`` rows so ``build_annotation_seed_slp`` still reads their
    predictions from the source .slp.

    A boolean ``forced`` column marks provenance. Returns (updated_df, stats).
    """
    req = pd.read_csv(force_include_csv)
    missing_cols = {"experiment", "view", "frame"} - set(req.columns)
    if missing_cols:
        raise ValueError(
            f"--force-include-csv {force_include_csv} missing columns: "
            f"{sorted(missing_cols)}"
        )
    req = req[["experiment", "view", "frame"]].copy()
    req["experiment"] = req["experiment"].astype(str)
    req["view"] = req["view"].astype(str)
    req["frame"] = req["frame"].astype(int)
    bad_view = sorted(set(req["view"]) - set(VIEWS))
    if bad_view:
        raise ValueError(
            f"--force-include-csv has unknown view(s) {bad_view}; "
            f"expected one of {list(VIEWS)}"
        )
    req = req.drop_duplicates(ignore_index=True)

    out = selected.copy()
    if "forced" not in out.columns:
        out["forced"] = False
    # Match on the (experiment, view, frame) key.
    key_cols = ["experiment", "view", "frame"]
    existing_keys = set(map(tuple, out[key_cols].astype(
        {"experiment": str, "view": str, "frame": int}).itertuples(
        index=False, name=None)))
    new_rows = []
    n_flipped = 0
    for exp, view, frame in req.itertuples(index=False, name=None):
        if (exp, view, frame) in existing_keys:
            mask = (
                (out["experiment"].astype(str) == exp)
                & (out["view"].astype(str) == view)
                & (out["frame"].astype(int) == frame)
            )
            n_flipped += int((~out.loc[mask, "picked"]).sum())
            out.loc[mask, "picked"] = True
            out.loc[mask, "forced"] = True
        else:
            row = {c: np.nan for c in out.columns}
            row.update({"experiment": exp, "view": view, "frame": frame,
                        "picked": True, "forced": True})
            if "n_signals" in out.columns:
                row["n_signals"] = 0
            if "signals_fired" in out.columns:
                row["signals_fired"] = ""
            if "time_bin" in out.columns:
                # Sentinel: keeps time_bin integer (no NaN -> no float coercion,
                # which would crash write_report's {tb:02d}); groups as "forced".
                row["time_bin"] = -1
            new_rows.append(row)
    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    stats = {
        "requested": int(len(req)),
        "flipped_existing": n_flipped,
        "appended_missing": int(len(new_rows)),
    }
    return out, stats


def select_candidates(
    df: pd.DataFrame,
    *,
    target_count: int,
    min_inter_pick_frames: int,
    time_bins: int,
) -> pd.DataFrame:
    """
    Pick ``target_count`` rows balanced across experiments, views, and time
    bins, filtering to rows where at least one signal fired. Adds a ``picked``
    boolean column to a copy of the input dataframe.

    Algorithm (three-axis round-robin, signal-weighted):
        1. Drop rows with ``n_signals == 0``.
        2. Sort each (experiment, view, time_bin) bucket by suspicion_score desc.
        3. Iterate rounds. Within each round, iterate every (time_bin, exp,
           view) cell exactly once and try one pick from it. ``time_bin`` is
           the outermost cycle, so the first ``len(time_bins) × len(ev_pairs)``
           picks each land in a distinct (time_bin, exp, view) cell. Round 1
           takes the second-best in each cell, and so on.
        4. Stop when ``target_count`` is reached or no cell yields more picks.
    """
    full = df.copy()
    full["picked"] = False
    cand = full[full["n_signals"] > 0].copy()
    if cand.empty:
        return full

    # Sort within each bucket so rank=0 is the most suspicious.
    cand.sort_values(
        ["experiment", "view", "time_bin", "suspicion_score", "n_signals"],
        ascending=[True, True, True, False, False],
        inplace=True,
    )
    cand["bucket_rank"] = cand.groupby(
        ["experiment", "view", "time_bin"], sort=False
    ).cumcount()

    bucket_queues: dict[tuple[str, str, int], list[int]] = {}
    for (exp, view, tb), grp in cand.groupby(
        ["experiment", "view", "time_bin"], sort=False
    ):
        bucket_queues[(str(exp), str(view), int(tb))] = list(grp.index)

    ev_keys: list[tuple[str, str]] = sorted(
        {(k[0], k[1]) for k in bucket_queues.keys()}
    )
    tb_values: list[int] = sorted({k[2] for k in bucket_queues.keys()})
    n_evs = len(ev_keys)
    n_tbs = len(tb_values)

    # Diagonal stepper: at step k, advance both axes simultaneously
    # (tb_idx = k % n_tbs, ev_idx = k % n_evs). Because gcd(n_tbs, n_evs)
    # is usually small, this visits ~ lcm(n_tbs, n_evs) distinct cells before
    # repeating, while keeping per-tb and per-ev pick counts balanced (after
    # K steps each tb gets ceil(K/n_tbs) picks and each ev gets ceil(K/n_evs)).
    picked_idx: list[int] = []
    picked_frames_per_view: dict[tuple[str, str], list[int]] = {}
    rank_cursor: dict[tuple[str, str, int], int] = {k: 0 for k in bucket_queues}
    visited_steps_since_progress = 0
    max_idle_steps = 4 * max(n_evs, n_tbs)
    step = 0

    while len(picked_idx) < target_count:
        ev = ev_keys[step % n_evs]
        tb = tb_values[step % n_tbs]
        step += 1
        key = (ev[0], ev[1], tb)
        queue = bucket_queues.get(key)
        if not queue:
            visited_steps_since_progress += 1
            if visited_steps_since_progress > max_idle_steps:
                break
            continue
        cur = rank_cursor[key]
        progressed = False
        while cur < len(queue):
            row_idx = queue[cur]
            row = cand.loc[row_idx]
            f = int(row["frame"])
            recent = picked_frames_per_view.get(ev, [])
            if any(abs(f - p) < min_inter_pick_frames for p in recent):
                cur += 1
                continue
            picked_idx.append(row_idx)
            picked_frames_per_view.setdefault(ev, []).append(f)
            rank_cursor[key] = cur + 1
            progressed = True
            break
        if not progressed:
            rank_cursor[key] = cur
            visited_steps_since_progress += 1
            if visited_steps_since_progress > max_idle_steps:
                break
        else:
            visited_steps_since_progress = 0

    full.loc[picked_idx, "picked"] = True
    return full


# ---------------------------------------------------------------------------
# Annotation seed pkg.slp builder.
# ---------------------------------------------------------------------------

def _resolve_video_path(stored_filename: str) -> str | None:
    """
    .slp files reference videos via UNC/forward-slash bucket paths. Try the
    stored filename first; if it's a UNC ``//bucket.oist.jp/...`` or
    ``\\\\bucket.oist.jp\\...`` path that doesn't resolve, fall back to a few
    common rewrites.
    """
    candidates: list[str] = [stored_filename]
    norm = stored_filename.replace("\\", "/")
    candidates.append(norm)
    # Try mapped Z: drive: '//bucket.oist.jp/bucket/StephensU/medaka/...' ->
    # 'Z:/StephensU/medaka/...'
    if norm.startswith("//bucket.oist.jp/bucket/StephensU/medaka/"):
        candidates.append(
            "Z:/" + norm[len("//bucket.oist.jp/bucket/StephensU/medaka/"):]
        )
    if norm.startswith("//bucket.oist.jp/bucket/"):
        candidates.append(
            "Z:/" + norm[len("//bucket.oist.jp/bucket/StephensU/"):]
        )
    for c in candidates:
        try:
            if Path(c).exists():
                return c
        except OSError:
            continue
    return None


def build_annotation_seed_slp(
    picks_df: pd.DataFrame,
    *,
    slp_root: Path,
    sleap_subdir: str | None = None,
    out_path: Path,
    tail_offset_xy: tuple[float, float] = (0.0, DEFAULT_TAIL_SEED_OFFSET_PX),
    embed: str | bool = "user",
    log: bool = True,
) -> dict:
    """
    Assemble a SLEAP labels project containing the picked frames with a
    ``mouth -> tail`` skeleton.

    If the source predictions already contain a finite ``tail`` node (as in
    round4 mouth+tail inference), preserve it. If the source has only
    ``mouth`` or the tail is missing/NaN for an instance, seed tail next to
    mouth as a hidden fallback for the annotator.

    Returns a summary dict with frame counts and the output path.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required to write the seed .slp. `uv sync --extra sleap`."
        ) from exc

    skel = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])

    # Cache one Labels object per source .slp so we read each disk file once.
    src_cache: dict[Path, "sio.Labels"] = {}
    video_cache: dict[str, "sio.Video"] = {}

    seed_lfs: list["sio.LabeledFrame"] = []
    n_with_inst = 0
    n_skipped = 0
    n_tail_from_source = 0
    n_tail_seeded = 0

    picks = picks_df[picks_df["picked"]].copy()
    picks.sort_values(["experiment", "view", "frame"], inplace=True)

    for (exp, view), group in picks.groupby(["experiment", "view"], sort=False):
        slp_path = _slp_path_for(
            slp_root, exp, view, sleap_subdir=sleap_subdir
        )
        if not slp_path.exists():
            n_skipped += len(group)
            continue
        if slp_path not in src_cache:
            src_cache[slp_path] = sio.load_slp(str(slp_path))
        src = src_cache[slp_path]
        src_nodes = [n.name for n in src.skeletons[0].nodes] if src.skeletons else []
        mouth_idx = src_nodes.index("mouth") if "mouth" in src_nodes else 0
        tail_idx = src_nodes.index("tail") if "tail" in src_nodes else None
        # Map frame_idx -> LabeledFrame in the source .slp for O(1) lookup.
        frame_index = {int(lf.frame_idx): lf for lf in src.labeled_frames}

        # Resolve the source video to a usable path if we want frame embedding.
        src_video = src.videos[0] if src.videos else None
        if src_video is None:
            n_skipped += len(group)
            continue
        stored_name = src_video.filename
        resolved = _resolve_video_path(stored_name)
        cache_key = stored_name if resolved is None else resolved
        if cache_key not in video_cache:
            try:
                if resolved is not None:
                    video_cache[cache_key] = sio.load_video(resolved)
                else:
                    video_cache[cache_key] = src_video
            except Exception as exc:  # noqa: BLE001
                if log:
                    print(
                        f"  [{exp}/{view}] could not open video for embedding "
                        f"({stored_name}): {exc}; using stored reference."
                    )
                video_cache[cache_key] = src_video
        target_video = video_cache[cache_key]

        for _, row in group.iterrows():
            f = int(row["frame"])
            src_lf = frame_index.get(f)
            if src_lf is None:
                n_skipped += 1
                continue
            new_instances: list = []
            for inst in src_lf.instances:
                pts = inst.numpy()
                if mouth_idx >= len(pts):
                    continue
                mouth = pts[mouth_idx]
                if not np.isfinite(mouth).all():
                    mask = np.isfinite(pts).all(axis=1)
                    if not mask.any():
                        continue
                    mouth = pts[mask][0]
                mx, my = float(mouth[0]), float(mouth[1])

                tail = None
                if tail_idx is not None and tail_idx < len(pts):
                    cand_tail = pts[tail_idx]
                    if np.isfinite(cand_tail).all():
                        tail = cand_tail
                if tail is not None:
                    tx, ty = float(tail[0]), float(tail[1])
                    tail_visible = True
                    n_tail_from_source += 1
                else:
                    tx = mx + tail_offset_xy[0]
                    ty = my + tail_offset_xy[1]
                    tail_visible = False
                    n_tail_seeded += 1
                new_inst = sio.Instance.from_numpy(
                    points_data=np.array(
                        [
                            [float(mx), float(my), True],
                            [tx, ty, tail_visible],
                        ],
                        dtype=float,
                    ),
                    skeleton=skel,
                )
                new_instances.append(new_inst)
            if not new_instances:
                n_skipped += 1
                continue
            seed_lf = sio.LabeledFrame(
                video=target_video,
                frame_idx=f,
                instances=new_instances,
            )
            seed_lfs.append(seed_lf)
            n_with_inst += 1

    # Mirror every labeled frame as a SuggestionFrame so sleap-label's
    # "next suggestion" navigation walks through the picked set in order.
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        for lf in seed_lfs
    ]

    labels = sio.Labels(
        skeletons=[skel],
        videos=list({id(v): v for v in (lf.video for lf in seed_lfs)}.values()),
        labeled_frames=seed_lfs,
        suggestions=suggestions,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        print(
            f"  Saving {n_with_inst} labeled frames to {out_path} "
            f"(embed={embed!r}, skipped={n_skipped})"
        )
    try:
        sio.save_slp(labels, str(out_path), embed=embed)
        embedded = bool(embed)
    except Exception as exc:  # noqa: BLE001
        if log:
            print(
                f"  Embedding failed ({exc}); retrying without embedding "
                "(annotator will need access to the original videos)."
            )
        sio.save_slp(labels, str(out_path), embed=False)
        embedded = False
    return {
        "out_path": str(out_path),
        "n_frames": n_with_inst,
        "n_skipped": n_skipped,
        "n_tail_from_source": n_tail_from_source,
        "n_tail_seeded": n_tail_seeded,
        "embedded": embedded,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def _signal_frequency(df: pd.DataFrame) -> dict[str, int]:
    counts = {s: 0 for s in SIGNAL_WEIGHTS}
    for sig_str in df["signals_fired"]:
        if not sig_str:
            continue
        for s in str(sig_str).split(";"):
            if s in counts:
                counts[s] += 1
    return counts


def write_report(
    df: pd.DataFrame,
    picks: pd.DataFrame,
    out_path: Path,
    *,
    bl_estimates: dict[tuple[str, str], float],
) -> None:
    lines: list[str] = []
    lines.append("# Frame-mining summary")
    lines.append(f"Total candidate rows         : {len(df):,}")
    lines.append(f"Rows with any signal fired   : {(df['n_signals'] > 0).sum():,}")
    lines.append(f"Frames picked                : {int(picks['picked'].sum()):,}")
    lines.append("")
    lines.append("## Signal frequency (over all candidate rows)")
    for s, c in _signal_frequency(df).items():
        lines.append(f"  {s}: {c:,}")
    lines.append("")
    lines.append("## Signal frequency (picked rows)")
    for s, c in _signal_frequency(picks[picks["picked"]]).items():
        lines.append(f"  {s}: {c:,}")
    lines.append("")
    lines.append("## Per-experiment picks")
    pe = picks[picks["picked"]].groupby("experiment").size().sort_index()
    for exp, n in pe.items():
        lines.append(f"  {exp}: {n}")
    lines.append("")
    lines.append("## Per-view picks")
    pv = picks[picks["picked"]].groupby("view").size().sort_index()
    for v, n in pv.items():
        lines.append(f"  {v}: {n}")
    lines.append("")
    lines.append("## Per-time-bin picks (across experiments)")
    pt = picks[picks["picked"]].groupby("time_bin").size().sort_index()
    for tb, n in pt.items():
        if pd.isna(tb) or int(tb) < 0:
            lines.append(f"  bin forced/unbinned: {n}")
        else:
            lines.append(f"  bin {int(tb):02d}: {n}")
    lines.append("")
    lines.append("## BL_px estimates (99.5th pct step length)")
    for (exp, view), bl in sorted(bl_estimates.items()):
        lines.append(f"  {exp}/{view}: {bl:.1f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def discover_experiments(slp_root: Path) -> list[str]:
    """List experiment folders that contain at least one merged view .slp."""
    exps: list[str] = []
    for child in sorted(slp_root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / f"{v}.slp").exists() for v in VIEWS):
            exps.append(child.name)
    return exps


def discover_experiments_for_subdir(slp_root: Path, sleap_subdir: str | None) -> list[str]:
    """List experiments containing merged view .slp files in an optional subdir."""
    if not sleap_subdir:
        return discover_experiments(slp_root)
    exps: list[str] = []
    for child in sorted(slp_root.iterdir()):
        if not child.is_dir():
            continue
        base = child / sleap_subdir
        if any((base / f"{v}.slp").exists() for v in VIEWS):
            exps.append(child.name)
    return exps


def run_pipeline(
    *,
    slp_root: Path,
    sleap_subdir: str | None,
    cache_root: Path,
    out_dir: Path,
    pre_link_dir: Path | None,
    reconstruction_subdir: str | None = None,
    pre_link_json: Path | None,
    identified_csv: Path | None,
    force_include_csv: Path | None = None,
    bind_score_quantile: float,
    experiments: list[str],
    fps: float,
    skip_frame: int,
    target_count: int,
    time_bins: int,
    min_inter_pick_frames: int,
    nn_gate_px: float,
    stationary_window: int,
    stationary_std_px: float,
    s2_lo_frac: float,
    s2_hi_frac: float,
    s5_disp: float,
    s5_min_score: float,
    tail_offset_px: float,
    force_features: bool,
    skip_seed: bool,
    embed: str | bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mining frames from {len(experiments)} experiment(s).")
    all_rows: list[pd.DataFrame] = []
    bl_estimates: dict[tuple[str, str], float] = {}

    for exp in experiments:
        per_view = build_features_for_experiment(
            exp,
            slp_root=slp_root,
            sleap_subdir=sleap_subdir,
            cache_root=cache_root,
            skip_frame=skip_frame,
            nn_gate_px=nn_gate_px,
            stationary_window=stationary_window,
            stationary_std_px=stationary_std_px,
            s2_lo_frac=s2_lo_frac,
            s2_hi_frac=s2_hi_frac,
            s5_disp=s5_disp,
            s5_min_score=s5_min_score,
            force=force_features,
        )
        if not per_view:
            continue
        for view, payload in per_view.items():
            bl_estimates[(exp, view)] = payload["meta"].get("bl_px_est", float("nan"))

        pre_link_json_path = None
        if pre_link_json is not None:
            pre_link_json_path = pre_link_json
        elif reconstruction_subdir and pre_link_dir is not None:
            cand = (pre_link_dir / reconstruction_subdir / exp
                    / f"{reconstruction_subdir}_best_pre_link.json")
            if cand.exists():
                pre_link_json_path = cand
        elif pre_link_dir is not None:
            cand = pre_link_dir / f"{exp}_best_pre_link.json"
            if cand.exists():
                pre_link_json_path = cand
        identified_csv_path = None
        if identified_csv is not None:
            identified_csv_path = identified_csv
        elif reconstruction_subdir and pre_link_dir is not None:
            cand = (pre_link_dir / reconstruction_subdir / exp
                    / f"{reconstruction_subdir}_3d_identified.csv")
            if cand.exists():
                identified_csv_path = cand
        df = aggregate_experiment(
            exp,
            per_view,
            pre_link_json_path=pre_link_json_path,
            identified_csv_path=identified_csv_path,
            bind_score_quantile=bind_score_quantile,
            fps=fps,
            skip_frame=skip_frame,
            time_bins=time_bins,
        )
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No candidate rows produced — nothing to select from.")

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total candidate rows: {len(combined):,}")
    print(f"Rows with any signal: {(combined['n_signals'] > 0).sum():,}")

    selected = select_candidates(
        combined,
        target_count=target_count,
        min_inter_pick_frames=min_inter_pick_frames,
        time_bins=time_bins,
    )
    n_signal_picked = int(selected["picked"].sum())
    print(f"Frames picked (signal-based): {n_signal_picked} / target {target_count}")

    if force_include_csv is not None:
        selected, fi_stats = apply_force_include(selected, force_include_csv)
        print(
            f"Force-include: {fi_stats['requested']} requested "
            f"({fi_stats['flipped_existing']} flipped from candidates, "
            f"{fi_stats['appended_missing']} appended as forced rows)."
        )
    n_picked = int(selected["picked"].sum())
    print(f"Frames picked (total): {n_picked}")

    # Full table (~21M rows): parquet only. CSV would be ~3.6 GB and is
    # rarely browsed in full. Parquet compresses the same data to ~100 MB
    # and reads ~30x faster in pandas.
    parquet_path = out_dir / "mined_frames.parquet"
    selected.to_parquet(parquet_path, index=False, compression="zstd")
    print(f"Wrote {parquet_path}")

    # Picked-frames CSV: small, human-inspectable, useful in visidata.
    picked_csv_path = out_dir / "picked_frames.csv"
    selected[selected["picked"]].to_csv(picked_csv_path, index=False)
    print(f"Wrote {picked_csv_path}")

    seed_summary = None
    if not skip_seed:
        seed_path = out_dir / "annotation_seed.pkg.slp"
        seed_summary = build_annotation_seed_slp(
            selected,
            slp_root=slp_root,
            sleap_subdir=sleap_subdir,
            out_path=seed_path,
            tail_offset_xy=(0.0, tail_offset_px),
            embed=embed,
        )

    report_path = out_dir / "mining_report.txt"
    write_report(combined, selected, report_path, bl_estimates=bl_estimates)
    print(f"Wrote {report_path}")

    return {
        "parquet_path": str(parquet_path),
        "picked_csv_path": str(picked_csv_path),
        "report_path": str(report_path),
        "n_candidates": int(len(combined)),
        "n_picked": n_picked,
        "seed": seed_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Mine SLEAP training frames for the next retraining round and "
            "build a two-node mouth->tail annotation seed pkg.slp."
        )
    )
    p.add_argument("--paths-config", default=None)
    p.add_argument(
        "--sleap-output-dir", default=None,
        help="Override sleap_output_dir (where merged per-view .slp files live).",
    )
    p.add_argument(
        "--sleap-subdir", default=None,
        help=(
            "Optional subfolder inside each experiment, e.g. round4_260520. "
            "Use this when mining from a retraining round under "
            "<sleap_output_dir>/<experiment>/<round>/."
        ),
    )
    p.add_argument(
        "--reconstruction-dir", default=None,
        help=(
            "Override reconstruction_output_dir; used to find "
            "<exp>_best_pre_link.json for signal S4."
        ),
    )
    p.add_argument(
        "--reconstruction-subdir", default=None,
        help=(
            "Round-tag subfolder under reconstruction_output_dir for the "
            "round-tagged layout <recon_dir>/<subdir>/<exp>/<subdir>_*. When "
            "set, S4/S6/S7 per-experiment files are discovered as "
            "<subdir>_best_pre_link.json and <subdir>_3d_identified.csv inside "
            "each experiment folder (e.g. --reconstruction-subdir round5_260611)."
        ),
    )
    p.add_argument(
        "--pre-link-json", default=None,
        help="Explicit *_best_pre_link.json for S4 residual signal.",
    )
    p.add_argument(
        "--identified-csv", default=None,
        help=(
            "Explicit *_3d_identified.csv for S6/S7 signals "
            "(NaN/geometry-only/high-bind-distance frames)."
        ),
    )
    p.add_argument(
        "--bind-score-quantile", type=float, default=0.95,
        help="Quantile of accepted bind_score used for S7 high-bind signal.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: <sleap_output_dir>/training_frame_mining).",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="Per-view feature cache (default: <out-dir>/cache).",
    )
    p.add_argument(
        "--experiments", nargs="*", default=None,
        help="Limit to these experiment folder names. Default: all under sleap_output_dir.",
    )
    p.add_argument(
        "--target-count", type=int, default=DEFAULT_TARGET_COUNT,
        help=f"Number of frames to pick (default: {DEFAULT_TARGET_COUNT}).",
    )
    p.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Recording frame rate (default: {DEFAULT_FPS}).",
    )
    p.add_argument(
        "--skip-seconds", type=float, default=DEFAULT_SKIP_SECONDS,
        help=(
            "Skip the first N seconds of each video (default: 300 — animals "
            "not yet placed)."
        ),
    )
    p.add_argument(
        "--time-bins", type=int, default=DEFAULT_TIME_BINS,
        help=f"Number of equal-time buckets per video (default: {DEFAULT_TIME_BINS}).",
    )
    p.add_argument(
        "--min-inter-pick-frames", type=int, default=DEFAULT_MIN_INTER_PICK_FRAMES,
        help=(
            "Within (experiment, view), require picks to be at least this far "
            f"apart in frames (default: {DEFAULT_MIN_INTER_PICK_FRAMES})."
        ),
    )
    p.add_argument("--nn-gate-px", type=float, default=DEFAULT_NN_GATE_PX)
    p.add_argument("--stationary-window", type=int, default=DEFAULT_STATIONARY_WINDOW)
    p.add_argument("--stationary-std-px", type=float, default=DEFAULT_STATIONARY_STD_PX)
    p.add_argument("--s2-lo-frac", type=float, default=DEFAULT_S2_LO_FRAC)
    p.add_argument("--s2-hi-frac", type=float, default=DEFAULT_S2_HI_FRAC)
    p.add_argument("--s5-disp", type=float, default=DEFAULT_S5_DISP)
    p.add_argument("--s5-min-score", type=float, default=DEFAULT_S5_MIN_SCORE)
    p.add_argument(
        "--tail-offset-px", type=float, default=DEFAULT_TAIL_SEED_OFFSET_PX,
        help=(
            f"Pixels to seed `tail` below `mouth` in image coords "
            f"(default: {DEFAULT_TAIL_SEED_OFFSET_PX})."
        ),
    )
    p.add_argument(
        "--force-features", action="store_true",
        help="Rebuild per-view feature parquets even if cached and fresh.",
    )
    p.add_argument(
        "--skip-seed", action="store_true",
        help="Don't build annotation_seed.pkg.slp (just emit mined_frames.csv).",
    )
    p.add_argument(
        "--force-include-csv", default=None,
        help=(
            "CSV with columns experiment,view,frame (view in "
            "{top_cam,side1_cam,side2_cam}) whose rows are FORCE-picked into the "
            "seed regardless of signal-based selection. Use to guarantee "
            "hand-reviewed hard cases (e.g. crossing events) are annotated. "
            "Frames absent from the candidate table are appended as forced rows; "
            "the seed builder still reads their predictions from the source .slp."
        ),
    )
    p.add_argument(
        "--embed", default="user",
        help=(
            "sleap-io save_slp embed mode (default: 'user' — embeds picked "
            "frames so the .slp is portable). Pass 'source' to keep video "
            "references only, or 'all' to also embed suggestion frames."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local.sleap_output_dir if local else None,
    )
    if sleap_output_dir is None:
        parser.error(
            "sleap_output_dir is required (paths.local.toml or --sleap-output-dir)."
        )
    slp_root = Path(sleap_output_dir)

    reconstruction_dir = choose_path(
        args.reconstruction_dir,
        local.reconstruction_output_dir if local else None,
    )
    pre_link_dir = Path(reconstruction_dir) if reconstruction_dir else None

    out_dir = Path(args.out_dir) if args.out_dir else slp_root / "training_frame_mining"
    if args.cache_dir:
        cache_root = Path(args.cache_dir)
    else:
        cache_suffix = args.sleap_subdir or "root"
        cache_root = out_dir / "cache" / cache_suffix

    if args.experiments:
        experiments = list(args.experiments)
    else:
        experiments = discover_experiments_for_subdir(slp_root, args.sleap_subdir)
    if not experiments:
        parser.error(f"No experiments with merged .slp under {slp_root}")

    skip_frame = int(round(args.skip_seconds * args.fps))

    embed_arg: str | bool
    if str(args.embed).lower() in {"false", "0", "none"}:
        embed_arg = False
    else:
        embed_arg = args.embed

    run_pipeline(
        slp_root=slp_root,
        sleap_subdir=args.sleap_subdir,
        cache_root=cache_root,
        out_dir=out_dir,
        pre_link_dir=pre_link_dir,
        reconstruction_subdir=args.reconstruction_subdir,
        pre_link_json=Path(args.pre_link_json) if args.pre_link_json else None,
        identified_csv=Path(args.identified_csv) if args.identified_csv else None,
        force_include_csv=Path(args.force_include_csv) if args.force_include_csv else None,
        bind_score_quantile=args.bind_score_quantile,
        experiments=experiments,
        fps=args.fps,
        skip_frame=skip_frame,
        target_count=args.target_count,
        time_bins=args.time_bins,
        min_inter_pick_frames=args.min_inter_pick_frames,
        nn_gate_px=args.nn_gate_px,
        stationary_window=args.stationary_window,
        stationary_std_px=args.stationary_std_px,
        s2_lo_frac=args.s2_lo_frac,
        s2_hi_frac=args.s2_hi_frac,
        s5_disp=args.s5_disp,
        s5_min_score=args.s5_min_score,
        tail_offset_px=args.tail_offset_px,
        force_features=args.force_features,
        skip_seed=args.skip_seed,
        embed=embed_arg,
    )


if __name__ == "__main__":
    main()
