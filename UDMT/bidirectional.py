from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from UDMT.io import normalize_tracks, require_columns


@dataclass(frozen=True)
class BidirectionalConfig:
    """Settings for forward/backward top-camera trajectory consensus."""

    max_pair_distance_px: float = 35.0
    max_anomaly_repair_distance_px: float = 250.0
    max_speed_px_per_frame: float = 80.0
    max_speed_ratio: float = 4.0
    rolling_window: int = 5
    min_mapping_votes: int = 3
    repair_anomalies_only: bool = True
    repair_crossings: bool = False
    crossing_distance_px: float = 35.0
    crossing_min_duration_frames: int = 1
    consensus_blend: float = 1.0
    drop_unrepaired_anomalies: bool = False


@dataclass(frozen=True)
class CrossingConfig:
    """Settings for smooth crossing-event detection.

    `coordinate_columns` may be `("x", "y")` for top-camera tracks or
    `("X", "Y", "Z")` for 3D geometry tracks. Distances are measured in the
    units of those columns.
    """

    id_column: str = "track_id"
    coordinate_columns: tuple[str, ...] = ("x", "y")
    distance_threshold: float = 35.0
    min_duration_frames: int = 1
    merge_gap_frames: int = 0
    dtw_window_frames: int = 20
    dtw_threshold: float | None = None
    normalize_dtw: bool = True


@dataclass(frozen=True)
class BidirectionalSummary:
    input_rows: int
    output_rows: int
    anomalous_rows: int
    repaired_rows: int
    dropped_rows: int
    mapped_tracks: int
    max_pair_distance_px: float

    def to_dict(self) -> dict:
        return asdict(self)


def detect_crossing_events(
    tracks: pd.DataFrame,
    config: CrossingConfig | None = None,
) -> pd.DataFrame:
    """Detect close-contact crossing events independent of speed spikes.

    This is the UDMT-relevant trigger for smooth medaka swaps: two trajectories
    can exchange biological identity while each positional path remains smooth.
    The detector therefore looks for pairwise close-contact runs, with optional
    DTW filtering on the pair's local trajectory window.
    """

    cfg = config or CrossingConfig()
    df = _normalize_coordinate_tracks(tracks, cfg)
    pair_rows = _pairwise_distances_by_frame(df, cfg)
    if pair_rows.empty:
        return _empty_crossing_events()

    events: list[dict] = []
    event_id = 0
    for pair_key, group in pair_rows.groupby("pair_key", sort=False):
        group = group.sort_values("frame").reset_index(drop=True)
        close = group[group["distance"] <= float(cfg.distance_threshold)].copy()
        if close.empty:
            continue
        for segment in _contiguous_segments(close, max_gap_frames=cfg.merge_gap_frames):
            if len(segment) < int(cfg.min_duration_frames):
                continue
            min_idx = int(segment["distance"].idxmin())
            min_row = segment.loc[min_idx]
            dtw_distance = _event_dtw_distance(df, min_row, cfg)
            if cfg.dtw_threshold is not None:
                if dtw_distance is None or dtw_distance > float(cfg.dtw_threshold):
                    continue
            events.append(
                {
                    "event_id": event_id,
                    "track_id_a": str(min_row["track_id_a"]),
                    "track_id_b": str(min_row["track_id_b"]),
                    "start_frame": int(segment["frame"].min()),
                    "end_frame": int(segment["frame"].max()),
                    "duration_frames": int(len(segment)),
                    "min_frame": int(min_row["frame"]),
                    "min_distance": float(min_row["distance"]),
                    "distance_threshold": float(cfg.distance_threshold),
                    "dtw_distance": np.nan if dtw_distance is None else float(dtw_distance),
                    "coordinate_columns": ",".join(cfg.coordinate_columns),
                    "id_column": cfg.id_column,
                }
            )
            event_id += 1
    if not events:
        return _empty_crossing_events()
    return pd.DataFrame(events)


def annotate_crossing_rows(
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    config: CrossingConfig | None = None,
) -> pd.DataFrame:
    """Annotate track rows that fall inside detected crossing windows."""

    cfg = config or CrossingConfig()
    require_columns(tracks, ("frame", cfg.id_column), label="tracks")
    out = tracks.copy()
    out["udmt_crossing_candidate"] = False
    out["udmt_crossing_event_ids"] = ""
    if events.empty:
        return out

    event_ids_by_row: dict[int, list[str]] = defaultdict(list)
    frames = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
    ids = out[cfg.id_column].astype(str)
    for event in events.itertuples(index=False):
        participants = {str(event.track_id_a), str(event.track_id_b)}
        mask = (
            frames.ge(int(event.start_frame))
            & frames.le(int(event.end_frame))
            & ids.isin(participants)
        )
        for idx in out.index[mask.fillna(False)]:
            event_ids_by_row[int(idx)].append(str(event.event_id))

    for idx, event_ids in event_ids_by_row.items():
        out.at[idx, "udmt_crossing_candidate"] = True
        out.at[idx, "udmt_crossing_event_ids"] = ";".join(event_ids)
    return out


def detect_motion_anomalies(
    tracks: pd.DataFrame,
    config: BidirectionalConfig | None = None,
) -> pd.DataFrame:
    """Annotate per-track speed spikes that are likely ID-switch points."""

    cfg = config or BidirectionalConfig()
    df = normalize_tracks(tracks)
    df["udmt_speed_px_per_frame"] = np.nan
    df["udmt_speed_ratio"] = np.nan
    df["udmt_motion_anomaly"] = False

    for _, group in df.sort_values("frame").groupby("track_id", sort=False):
        prev_speeds: list[float] = []
        records = list(group[["frame", "x", "y"]].itertuples(index=True, name="Row"))
        for prev, curr in zip(records, records[1:]):
            gap = max(1, int(curr.frame) - int(prev.frame))
            speed = float(np.hypot(curr.x - prev.x, curr.y - prev.y) / gap)
            baseline = float(np.nanmedian(prev_speeds[-cfg.rolling_window:])) if prev_speeds else speed
            if not np.isfinite(baseline) or baseline <= 1e-6:
                ratio = 1.0 if speed <= 1e-6 else np.inf
            else:
                ratio = speed / baseline
            anomaly = bool(
                speed > float(cfg.max_speed_px_per_frame)
                or ratio > float(cfg.max_speed_ratio)
            )
            df.at[curr.Index, "udmt_speed_px_per_frame"] = speed
            df.at[curr.Index, "udmt_speed_ratio"] = float(ratio)
            df.at[curr.Index, "udmt_motion_anomaly"] = anomaly
            prev_speeds.append(speed)
    return df


def repair_tracks_with_backward(
    forward_tracks: pd.DataFrame,
    backward_tracks: pd.DataFrame,
    config: BidirectionalConfig | None = None,
) -> tuple[pd.DataFrame, BidirectionalSummary, pd.DataFrame]:
    """Repair forward tracks when a backward pass gives a local consensus.

    The function first maps forward IDs to backward IDs by nearest same-frame
    votes. It then uses the mapped backward coordinate to replace only motion
    anomalies by default. This conservative mode is meant for top-camera anchor
    CSVs that will later feed `bind_topcam_id`.
    """

    cfg = config or BidirectionalConfig()
    forward = detect_motion_anomalies(forward_tracks, cfg)
    if cfg.repair_crossings:
        crossing_cfg = CrossingConfig(
            id_column="track_id",
            coordinate_columns=("x", "y"),
            distance_threshold=cfg.crossing_distance_px,
            min_duration_frames=cfg.crossing_min_duration_frames,
        )
        events = detect_crossing_events(forward, crossing_cfg)
        forward = annotate_crossing_rows(forward, events, crossing_cfg)
    else:
        forward["udmt_crossing_candidate"] = False
        forward["udmt_crossing_event_ids"] = ""
    backward = normalize_tracks(backward_tracks, label="backward_tracks")
    mapping, vote_table = _map_backward_track_ids(forward, backward, cfg)

    repaired = forward.copy()
    repaired["udmt_backward_track_id"] = ""
    repaired["udmt_backward_distance_px"] = np.nan
    repaired["udmt_repair_action"] = "forward"
    repaired["pre_repair_x"] = repaired["x"].astype(float)
    repaired["pre_repair_y"] = repaired["y"].astype(float)

    backward_lookup = {
        (int(row.frame), row.track_id): row
        for row in backward.itertuples(index=False)
    }

    repaired_rows = 0
    drop_indices: set[int] = set()
    alpha = float(np.clip(cfg.consensus_blend, 0.0, 1.0))
    for row in repaired.itertuples(index=True):
        mapped_id = mapping.get(row.track_id)
        if mapped_id is None:
            if _needs_backward_repair(row, cfg) and cfg.drop_unrepaired_anomalies:
                drop_indices.add(row.Index)
            continue
        backward_row = backward_lookup.get((int(row.frame), mapped_id))
        if backward_row is None:
            if _needs_backward_repair(row, cfg) and cfg.drop_unrepaired_anomalies:
                drop_indices.add(row.Index)
            continue

        distance = float(np.hypot(row.x - backward_row.x, row.y - backward_row.y))
        repaired.at[row.Index, "udmt_backward_track_id"] = str(mapped_id)
        repaired.at[row.Index, "udmt_backward_distance_px"] = distance

        anomaly = bool(row.udmt_motion_anomaly)
        crossing = bool(row.udmt_crossing_candidate)
        allowed_distance = (
            float(cfg.max_anomaly_repair_distance_px)
            if anomaly or crossing
            else float(cfg.max_pair_distance_px)
        )
        if distance > allowed_distance:
            if anomaly or crossing:
                repaired.at[row.Index, "udmt_repair_action"] = (
                    "crossing_unrepaired" if crossing else "motion_anomaly_unrepaired"
                )
                if cfg.drop_unrepaired_anomalies:
                    drop_indices.add(row.Index)
            continue

        should_repair = (not cfg.repair_anomalies_only) or anomaly or crossing
        if should_repair:
            repaired.at[row.Index, "x"] = (1.0 - alpha) * float(row.x) + alpha * float(backward_row.x)
            repaired.at[row.Index, "y"] = (1.0 - alpha) * float(row.y) + alpha * float(backward_row.y)
            repaired.at[row.Index, "udmt_repair_action"] = "backward_consensus"
            repaired_rows += 1
        else:
            repaired.at[row.Index, "udmt_repair_action"] = "forward_consensus"

    if drop_indices:
        repaired = repaired.drop(index=sorted(drop_indices))

    summary = BidirectionalSummary(
        input_rows=int(len(forward)),
        output_rows=int(len(repaired)),
        anomalous_rows=int(forward["udmt_motion_anomaly"].sum()),
        repaired_rows=int(repaired_rows),
        dropped_rows=int(len(drop_indices)),
        mapped_tracks=int(len(mapping)),
        max_pair_distance_px=float(cfg.max_pair_distance_px),
    )
    return repaired.sort_values(["frame", "track_id"]).reset_index(drop=True), summary, vote_table


def _needs_backward_repair(row, cfg: BidirectionalConfig) -> bool:
    return bool(row.udmt_motion_anomaly) or (
        bool(getattr(row, "udmt_crossing_candidate", False)) and cfg.repair_crossings
    )


def _map_backward_track_ids(
    forward: pd.DataFrame,
    backward: pd.DataFrame,
    cfg: BidirectionalConfig,
) -> tuple[dict[object, object], pd.DataFrame]:
    votes: Counter[tuple[object, object]] = Counter()
    for frame, fgroup in forward.groupby("frame", sort=False):
        bgroup = backward[backward["frame"] == frame]
        if fgroup.empty or bgroup.empty:
            continue
        fxy = fgroup[["x", "y"]].to_numpy(dtype=float)
        bxy = bgroup[["x", "y"]].to_numpy(dtype=float)
        cost = np.linalg.norm(fxy[:, None, :] - bxy[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for ri, ci in zip(rows, cols):
            if cost[ri, ci] <= float(cfg.max_pair_distance_px):
                votes[(fgroup.iloc[ri]["track_id"], bgroup.iloc[ci]["track_id"])] += 1

    if not votes:
        return {}, pd.DataFrame(columns=["forward_track_id", "backward_track_id", "votes"])

    vote_rows = [
        {"forward_track_id": fwd, "backward_track_id": bwd, "votes": count}
        for (fwd, bwd), count in votes.items()
    ]
    vote_table = pd.DataFrame(vote_rows)
    fwd_ids = list(vote_table["forward_track_id"].drop_duplicates())
    bwd_ids = list(vote_table["backward_track_id"].drop_duplicates())
    matrix = np.zeros((len(fwd_ids), len(bwd_ids)), dtype=float)
    fwd_index = {value: i for i, value in enumerate(fwd_ids)}
    bwd_index = {value: i for i, value in enumerate(bwd_ids)}
    for item in vote_table.itertuples(index=False):
        matrix[fwd_index[item.forward_track_id], bwd_index[item.backward_track_id]] = float(item.votes)

    rows, cols = linear_sum_assignment(-matrix)
    mapping: dict[object, object] = {}
    for ri, ci in zip(rows, cols):
        if matrix[ri, ci] >= int(cfg.min_mapping_votes):
            mapping[fwd_ids[ri]] = bwd_ids[ci]
    vote_table = vote_table.sort_values(["forward_track_id", "votes"], ascending=[True, False])
    return mapping, vote_table.reset_index(drop=True)


def _normalize_coordinate_tracks(tracks: pd.DataFrame, cfg: CrossingConfig) -> pd.DataFrame:
    require_columns(
        tracks,
        ("frame", cfg.id_column, *cfg.coordinate_columns),
        label="tracks",
    )
    out = tracks.copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out = out.dropna(subset=["frame"])
    out["frame"] = out["frame"].astype(int)
    out["_track_key"] = out[cfg.id_column].astype(str)
    for col in cfg.coordinate_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=list(cfg.coordinate_columns))
    finite = np.isfinite(out[list(cfg.coordinate_columns)].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[finite].copy()
    return out.sort_values(["frame", "_track_key"]).reset_index(drop=True)


def _pairwise_distances_by_frame(df: pd.DataFrame, cfg: CrossingConfig) -> pd.DataFrame:
    rows: list[dict] = []
    coord_cols = list(cfg.coordinate_columns)
    for frame, group in df.groupby("frame", sort=False):
        if len(group) < 2:
            continue
        ids = group["_track_key"].to_numpy(dtype=object)
        coords = group[coord_cols].to_numpy(dtype=float)
        for i in range(len(group) - 1):
            for j in range(i + 1, len(group)):
                a = str(ids[i])
                b = str(ids[j])
                if a > b:
                    a, b = b, a
                    ci, cj = coords[j], coords[i]
                else:
                    ci, cj = coords[i], coords[j]
                rows.append(
                    {
                        "frame": int(frame),
                        "track_id_a": a,
                        "track_id_b": b,
                        "pair_key": f"{a}||{b}",
                        "distance": float(np.linalg.norm(ci - cj)),
                    }
                )
    return pd.DataFrame(rows)


def _contiguous_segments(close: pd.DataFrame, *, max_gap_frames: int) -> list[pd.DataFrame]:
    segments: list[pd.DataFrame] = []
    start = 0
    frames = close["frame"].to_numpy(dtype=int)
    allowed_gap = int(max_gap_frames) + 1
    for idx in range(1, len(close)):
        if frames[idx] - frames[idx - 1] > allowed_gap:
            segments.append(close.iloc[start:idx].copy())
            start = idx
    segments.append(close.iloc[start:].copy())
    return segments


def _event_dtw_distance(
    df: pd.DataFrame,
    event_row: pd.Series,
    cfg: CrossingConfig,
) -> float | None:
    if cfg.dtw_threshold is None:
        return None
    half = max(1, int(cfg.dtw_window_frames) // 2)
    start = int(event_row["frame"]) - half
    end = int(event_row["frame"]) + half
    a = _trajectory_window(df, str(event_row["track_id_a"]), start, end, cfg)
    b = _trajectory_window(df, str(event_row["track_id_b"]), start, end, cfg)
    if len(a) < 2 or len(b) < 2:
        return None
    if cfg.normalize_dtw:
        joined = np.vstack([a, b])
        scale = np.nanmax(joined, axis=0) - np.nanmin(joined, axis=0)
        scale[scale <= 1e-9] = 1.0
        origin = np.nanmin(joined, axis=0)
        a = (a - origin) / scale
        b = (b - origin) / scale
    return _dtw_distance(a, b)


def _trajectory_window(
    df: pd.DataFrame,
    track_id: str,
    start_frame: int,
    end_frame: int,
    cfg: CrossingConfig,
) -> np.ndarray:
    group = df[
        (df["_track_key"] == track_id)
        & (df["frame"] >= int(start_frame))
        & (df["frame"] <= int(end_frame))
    ].sort_values("frame")
    return group[list(cfg.coordinate_columns)].to_numpy(dtype=float)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            step = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[len(a), len(b)] / max(len(a), len(b), 1))


def _empty_crossing_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "track_id_a",
            "track_id_b",
            "start_frame",
            "end_frame",
            "duration_frames",
            "min_frame",
            "min_distance",
            "distance_threshold",
            "dtw_distance",
            "coordinate_columns",
            "id_column",
        ]
    )
