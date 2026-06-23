from __future__ import annotations

"""Tracking-by-segmentation core for the UDMT adapter.

The official UDMT links per-individual masks across frames with an ST-Net
(TrDiMP) appearance+motion tracker (Linux/py3.8/checkpoint stack, not runnable
here). This module supplies the *motion* half of that idea so the adapter can be
exercised end-to-end on Detectron2 segmentations: it links per-frame mask
centroids into up to ``num_individuals`` identity tracks with a velocity-gated,
gap-tolerant greedy-Hungarian assignment (a SORT-style tracker on segmentation
detections), forward and backward.

This is deliberately appearance-free. On medaka that is a *fair* proxy for UDMT's
ceiling: the two males were shown to carry no usable appearance signal (see
``docs/detectron2_appearance_classifier_test.md``), and UDMT's tracker is
appearance-light/centroid-based, so the achievable individual-ID is bounded by
motion + segmentation quality -- exactly what this measures. The output schema is
the adapter's ``frame, track_id, x, y, score`` so it feeds ``UDMT repair`` /
``crossings`` / ``score`` directly.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MASK_COLUMNS = ("frame", "x", "y")


@dataclass(frozen=True)
class TrackingConfig:
    num_individuals: int = 3
    max_jump_px: float = 60.0          # per-frame gate; effective gate = min(max_jump*gap, hard_cap)
    max_jump_hard_cap_px: float = 400.0
    max_gap_frames: int = 30           # keep a track alive this long through missed detections
    min_area: float = 20.0
    max_area: float | None = None
    velocity_blend: float = 0.5        # EMA on velocity for prediction
    min_track_len: int = 5             # drop ephemeral tracks below this

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Track:
    track_id: int
    last_frame: int
    xy: np.ndarray
    vel: np.ndarray
    rows: list  # (frame, x, y, score)


def _prepare_masks(masks: pd.DataFrame, cfg: TrackingConfig) -> pd.DataFrame:
    out = masks.copy()
    if "centroid_x" in out.columns and "x" not in out.columns:
        out = out.rename(columns={"centroid_x": "x", "centroid_y": "y"})
    missing = set(MASK_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"masks missing columns: {sorted(missing)}")
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    for c in ("x", "y"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["frame", "x", "y"])
    out["frame"] = out["frame"].astype(int)
    if "area" in out.columns:
        area = pd.to_numeric(out["area"], errors="coerce").fillna(0.0)
        out = out[area >= float(cfg.min_area)]
        if cfg.max_area is not None:
            out = out[area <= float(cfg.max_area)]
    if "score" not in out.columns:
        out["score"] = 1.0
    return out.sort_values("frame").reset_index(drop=True)


def _run_tracker(df: pd.DataFrame, cfg: TrackingConfig) -> list[_Track]:
    """Single SORT-style pass. Returns every track ever created (live + retired)."""
    live: list[_Track] = []
    done: list[_Track] = []
    next_id = 0
    for frame, g in df.groupby("frame", sort=True):
        frame = int(frame)
        det_xy = g[["x", "y"]].to_numpy(dtype=float)
        det_score = g["score"].to_numpy(dtype=float)
        n_det = len(det_xy)

        # retire tracks idle longer than the gap tolerance (frees identity slots)
        keep = []
        for t in live:
            (keep if (frame - t.last_frame) <= cfg.max_gap_frames else done).append(t)
        live = keep

        matched_det: set[int] = set()
        if live and n_det:
            preds = np.array([t.xy + t.vel * (frame - t.last_frame) for t in live])
            cost = np.linalg.norm(preds[:, None, :] - det_xy[None, :, :], axis=2)
            gates = np.array([
                min(cfg.max_jump_px * max(1, frame - t.last_frame), cfg.max_jump_hard_cap_px)
                for t in live
            ])
            gated = np.where(cost > gates[:, None], 1e9, cost)
            ri, ci = linear_sum_assignment(gated)
            for r, c in zip(ri, ci):
                if gated[r, c] >= 1e9:
                    continue
                t = live[r]
                new_xy = det_xy[c]
                gap = max(1, frame - t.last_frame)
                t.vel = (1 - cfg.velocity_blend) * t.vel + cfg.velocity_blend * (new_xy - t.xy) / gap
                t.xy = new_xy
                t.last_frame = frame
                t.rows.append((frame, float(new_xy[0]), float(new_xy[1]), float(det_score[c])))
                matched_det.add(c)

        # spawn new tracks for unmatched detections while an identity slot is free
        # (highest-score detection first, so reflections lose to real fish).
        for c in sorted((c for c in range(n_det) if c not in matched_det),
                        key=lambda c: -det_score[c]):
            if len(live) >= cfg.num_individuals:
                break
            live.append(_Track(track_id=next_id, last_frame=frame, xy=det_xy[c].copy(),
                               vel=np.zeros(2),
                               rows=[(frame, float(det_xy[c][0]), float(det_xy[c][1]),
                                      float(det_score[c]))]))
            next_id += 1

    done.extend(live)
    return done


def track_masks_by_centroid(masks: pd.DataFrame, config: TrackingConfig | None = None) -> pd.DataFrame:
    """Link mask centroids into up to ``num_individuals`` tracks (forward in time).

    Returns ``frame, track_id, x, y, score``. Tracks shorter than
    ``min_track_len`` are dropped.
    """
    cfg = config or TrackingConfig()
    df = _prepare_masks(masks, cfg)
    if df.empty:
        return pd.DataFrame(columns=["frame", "track_id", "x", "y", "score"])
    rows = []
    for t in _run_tracker(df, cfg):
        if len(t.rows) < cfg.min_track_len:
            continue
        for f, x, y, s in t.rows:
            rows.append({"frame": f, "track_id": t.track_id, "x": x, "y": y, "score": s})
    out = pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "score"])
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def track_forward_backward(masks: pd.DataFrame, config: TrackingConfig | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward and backward (time-reversed) mask-centroid tracks. Backward frames
    are flipped so the output is in real frame order with its own track ids."""
    cfg = config or TrackingConfig()
    fwd = track_masks_by_centroid(masks, cfg)
    rev = masks.copy()
    rev["frame"] = -pd.to_numeric(rev["frame"], errors="coerce").astype(int)
    bwd = track_masks_by_centroid(rev, cfg)
    bwd["frame"] = -bwd["frame"]
    bwd = bwd.sort_values(["frame", "track_id"]).reset_index(drop=True)
    return fwd, bwd
