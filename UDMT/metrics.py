from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from UDMT.io import finite_xy_mask, normalize_tracks


@dataclass(frozen=True)
class TrackQualityConfig:
    """Heuristic quality thresholds for top-camera identity tracks."""

    num_individuals: int = 3
    max_speed_px_per_frame: float = 80.0
    max_speed_ratio: float = 4.0
    close_contact_distance_px: float = 35.0
    rolling_window: int = 5


@dataclass(frozen=True)
class TrackQualityReport:
    rows: int
    frames: int
    tracks: int
    coverage_fraction: float
    missing_fraction: float
    frame_count_error: float
    duplicate_frame_track_fraction: float
    jump_rate: float
    speed_ratio_spike_rate: float
    close_contact_frame_fraction: float
    mean_score: float | None
    quality_score: float
    cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def score_track_table(
    tracks: pd.DataFrame,
    config: TrackQualityConfig | None = None,
) -> TrackQualityReport:
    """Score a tidy top-camera track table.

    The score is intentionally transparent rather than model-specific. It
    rewards full 3-fish coverage and penalizes duplicate IDs, missing rows,
    speed spikes, and close-contact ambiguity. Higher `quality_score` is better.
    """

    cfg = config or TrackQualityConfig()
    df = normalize_tracks(tracks)
    finite = df.loc[finite_xy_mask(df)].copy()

    rows = int(len(finite))
    frames = int(finite["frame"].nunique()) if rows else 0
    tracks_seen = int(finite["track_id"].nunique()) if rows else 0
    expected = frames * int(cfg.num_individuals) if cfg.num_individuals > 0 else rows
    coverage = float(rows / expected) if expected else 0.0
    missing = float(max(0.0, 1.0 - min(coverage, 1.0)))

    if rows and cfg.num_individuals > 0:
        counts = finite.groupby("frame").size()
        frame_count_error = float(
            np.mean(np.abs(counts.to_numpy(dtype=float) - cfg.num_individuals))
            / max(cfg.num_individuals, 1)
        )
    else:
        frame_count_error = 0.0

    duplicate_count = int(finite.duplicated(["frame", "track_id"], keep=False).sum())
    duplicate_fraction = float(duplicate_count / max(rows, 1))

    speeds = _per_track_speeds(finite, rolling_window=cfg.rolling_window)
    if speeds.empty:
        jump_rate = 0.0
        speed_ratio_spike_rate = 0.0
    else:
        jump_rate = float((speeds["speed_px_per_frame"] > cfg.max_speed_px_per_frame).mean())
        speed_ratio_spike_rate = float((speeds["speed_ratio"] > cfg.max_speed_ratio).mean())

    close_contact_fraction = _close_contact_frame_fraction(
        finite,
        threshold_px=cfg.close_contact_distance_px,
    )
    mean_score = None
    score_penalty = 0.0
    if "score" in finite.columns and finite["score"].notna().any():
        mean_score = float(finite["score"].mean())
        score_penalty = max(0.0, 1.0 - mean_score)

    cost = (
        3.0 * missing
        + 2.0 * frame_count_error
        + 5.0 * duplicate_fraction
        + 4.0 * jump_rate
        + 3.0 * speed_ratio_spike_rate
        + 0.5 * close_contact_fraction
        + 1.0 * score_penalty
    )
    quality = float(1.0 / (1.0 + cost))
    return TrackQualityReport(
        rows=rows,
        frames=frames,
        tracks=tracks_seen,
        coverage_fraction=coverage,
        missing_fraction=missing,
        frame_count_error=frame_count_error,
        duplicate_frame_track_fraction=duplicate_fraction,
        jump_rate=jump_rate,
        speed_ratio_spike_rate=speed_ratio_spike_rate,
        close_contact_frame_fraction=close_contact_fraction,
        mean_score=mean_score,
        quality_score=quality,
        cost=float(cost),
    )


def _per_track_speeds(df: pd.DataFrame, *, rolling_window: int) -> pd.DataFrame:
    rows: list[dict] = []
    for track_id, group in df.sort_values("frame").groupby("track_id", sort=False):
        prev_speeds: list[float] = []
        records = group[["frame", "x", "y"]].to_records(index=False)
        for prev, curr in zip(records, records[1:]):
            gap = max(1, int(curr.frame) - int(prev.frame))
            speed = float(np.hypot(curr.x - prev.x, curr.y - prev.y) / gap)
            baseline = float(np.nanmedian(prev_speeds[-rolling_window:])) if prev_speeds else speed
            if not np.isfinite(baseline) or baseline <= 1e-6:
                ratio = 1.0 if speed <= 1e-6 else np.inf
            else:
                ratio = speed / baseline
            rows.append(
                {
                    "track_id": track_id,
                    "frame": int(curr.frame),
                    "speed_px_per_frame": speed,
                    "speed_ratio": float(ratio),
                }
            )
            prev_speeds.append(speed)
    return pd.DataFrame(rows)


def _close_contact_frame_fraction(df: pd.DataFrame, *, threshold_px: float) -> float:
    if df.empty:
        return 0.0
    close_frames = 0
    for _, group in df.groupby("frame", sort=False):
        if len(group) < 2:
            continue
        xy = group[["x", "y"]].to_numpy(dtype=float)
        dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        if bool((dist <= threshold_px).any()):
            close_frames += 1
    return float(close_frames / max(df["frame"].nunique(), 1))
