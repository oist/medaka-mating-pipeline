from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from UDMT.io import normalize_tracks


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
    consensus_blend: float = 1.0
    drop_unrepaired_anomalies: bool = False


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
            if bool(row.udmt_motion_anomaly) and cfg.drop_unrepaired_anomalies:
                drop_indices.add(row.Index)
            continue
        backward_row = backward_lookup.get((int(row.frame), mapped_id))
        if backward_row is None:
            if bool(row.udmt_motion_anomaly) and cfg.drop_unrepaired_anomalies:
                drop_indices.add(row.Index)
            continue

        distance = float(np.hypot(row.x - backward_row.x, row.y - backward_row.y))
        repaired.at[row.Index, "udmt_backward_track_id"] = str(mapped_id)
        repaired.at[row.Index, "udmt_backward_distance_px"] = distance

        anomaly = bool(row.udmt_motion_anomaly)
        allowed_distance = (
            float(cfg.max_anomaly_repair_distance_px)
            if anomaly
            else float(cfg.max_pair_distance_px)
        )
        if distance > allowed_distance:
            if anomaly:
                repaired.at[row.Index, "udmt_repair_action"] = "motion_anomaly_unrepaired"
                if cfg.drop_unrepaired_anomalies:
                    drop_indices.add(row.Index)
            continue

        should_repair = (not cfg.repair_anomalies_only) or anomaly
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
