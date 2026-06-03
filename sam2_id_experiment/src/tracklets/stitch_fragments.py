from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def stitch_identity_fragments(
    *,
    tracklets_qc: pd.DataFrame,
    assignments: pd.DataFrame,
    max_gap_frames: int = 300,
    max_speed_px_per_frame: float = 5.0,
    max_assignment_distance: float = 0.35,
    min_assignment_margin: float = 0.02,
    output_tracks_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    output_json_path: str | Path | None = None,
) -> dict:
    assigned = _eligible_assignments(
        assignments,
        max_assignment_distance=max_assignment_distance,
        min_assignment_margin=min_assignment_margin,
    )
    if assigned.empty:
        tracks = _empty_tracks(tracklets_qc)
        summary = _empty_summary()
        stitch_events: list[dict] = []
    else:
        tracks = tracklets_qc.merge(
            assigned,
            on="tracklet_id",
            how="inner",
            suffixes=("", "_assignment"),
        ).copy()
        tracks.sort_values(["assigned_identity", "frame", "tracklet_id"], inplace=True)
        fragment_summaries = _fragment_summaries(tracks)
        stitched_fragments, stitch_events = _stitch_fragment_summaries(
            fragment_summaries,
            max_gap_frames=max_gap_frames,
            max_speed_px_per_frame=max_speed_px_per_frame,
        )
        tracks = tracks.merge(stitched_fragments, on="tracklet_id", how="left")
        summary = _trajectory_summary(tracks, stitch_events)

    if output_tracks_path is not None:
        path = Path(output_tracks_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_parquet(path, index=False)
    if output_summary_path is not None:
        path = Path(output_summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(path, index=False)

    report = {
        "n_input_rows": int(len(tracklets_qc)),
        "n_assigned_fragments": int(assigned["tracklet_id"].nunique()) if not assigned.empty else 0,
        "n_track_rows": int(len(tracks)),
        "n_identity_trajectories": int(summary["identity_trajectory_id"].nunique()) if not summary.empty else 0,
        "identity_counts": {
            str(k): int(v) for k, v in tracks["assigned_identity"].value_counts().to_dict().items()
        } if not tracks.empty else {},
        "stitch_events": stitch_events,
        "max_gap_frames": int(max_gap_frames),
        "max_speed_px_per_frame": float(max_speed_px_per_frame),
        "max_assignment_distance": float(max_assignment_distance),
        "min_assignment_margin": float(min_assignment_margin),
    }
    if output_json_path is not None:
        path = Path(output_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _eligible_assignments(
    assignments: pd.DataFrame,
    *,
    max_assignment_distance: float,
    min_assignment_margin: float,
) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(columns=["tracklet_id", "assigned_identity"])
    out = assignments.copy()
    out = out[out["assigned_identity"].fillna("").astype(str).ne("unassigned")]
    if "assignment_status" in out.columns:
        out = out[out["assignment_status"].fillna("").astype(str).eq("assigned")]
    if "match_distance" in out.columns:
        out = out[out["match_distance"].astype(float) <= float(max_assignment_distance)]
    if "match_margin" in out.columns:
        out = out[out["match_margin"].astype(float) >= float(min_assignment_margin)]
    keep_cols = [
        col for col in [
            "tracklet_id",
            "assigned_identity",
            "assignment_status",
            "match_distance",
            "match_margin",
            "reference_tracklet_id",
        ]
        if col in out.columns
    ]
    return out[keep_cols].copy()


def _fragment_summaries(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for tracklet_id, group in tracks.groupby("tracklet_id", sort=True):
        ordered = group.sort_values("frame")
        clean = ordered[ordered["qc_status"].fillna("").astype(str).eq("clean_train")]
        endpoint_rows = clean if not clean.empty else ordered
        first = endpoint_rows.iloc[0]
        last = endpoint_rows.iloc[-1]
        rows.append({
            "tracklet_id": int(tracklet_id),
            "assigned_identity": str(ordered["assigned_identity"].iloc[0]),
            "frame_start": int(ordered["frame"].min()),
            "frame_end": int(ordered["frame"].max()),
            "endpoint_frame_start": int(first["frame"]),
            "endpoint_frame_end": int(last["frame"]),
            "start_x": float(first["centroid_x"]),
            "start_y": float(first["centroid_y"]),
            "end_x": float(last["centroid_x"]),
            "end_y": float(last["centroid_y"]),
            "n_rows": int(len(ordered)),
            "n_clean_rows": int(ordered["qc_status"].fillna("").astype(str).eq("clean_train").sum()),
        })
    return pd.DataFrame(rows)


def _stitch_fragment_summaries(
    fragments: pd.DataFrame,
    *,
    max_gap_frames: int,
    max_speed_px_per_frame: float,
) -> tuple[pd.DataFrame, list[dict]]:
    if fragments.empty:
        return pd.DataFrame(columns=["tracklet_id", "identity_trajectory_id", "stitch_status"]), []

    rows: list[dict] = []
    events: list[dict] = []
    for identity, group in fragments.sort_values(["assigned_identity", "endpoint_frame_start"]).groupby("assigned_identity"):
        segment_idx = 0
        prev: dict | None = None
        for frag in group.to_dict("records"):
            status = "trajectory_start"
            gap = None
            distance = None
            speed = None
            if prev is not None:
                gap = int(frag["endpoint_frame_start"] - prev["endpoint_frame_end"])
                distance = float(np.hypot(float(frag["start_x"]) - float(prev["end_x"]), float(frag["start_y"]) - float(prev["end_y"])))
                speed = distance / max(1, gap)
                if gap <= 0:
                    segment_idx += 1
                    status = "new_segment_overlap"
                elif gap > max_gap_frames:
                    segment_idx += 1
                    status = "new_segment_gap"
                elif speed > max_speed_px_per_frame:
                    segment_idx += 1
                    status = "new_segment_jump"
                else:
                    status = "stitched"
                    events.append({
                        "assigned_identity": str(identity),
                        "from_tracklet_id": int(prev["tracklet_id"]),
                        "to_tracklet_id": int(frag["tracklet_id"]),
                        "gap_frames": int(gap),
                        "endpoint_distance_px": float(distance),
                        "speed_px_per_frame": float(speed),
                    })
            trajectory_id = f"{identity}_seg{segment_idx}"
            rows.append({
                "tracklet_id": int(frag["tracklet_id"]),
                "identity_trajectory_id": trajectory_id,
                "stitch_status": status,
                "stitch_gap_frames": int(gap) if gap is not None else 0,
                "stitch_endpoint_distance_px": float(distance) if distance is not None else 0.0,
                "stitch_speed_px_per_frame": float(speed) if speed is not None else 0.0,
            })
            prev = frag
    return pd.DataFrame(rows), events


def _trajectory_summary(tracks: pd.DataFrame, stitch_events: list[dict]) -> pd.DataFrame:
    if tracks.empty:
        return _empty_summary()
    frame_step = _infer_frame_step(tracks["frame"])
    rows: list[dict] = []
    stitched_by_to = {int(event["to_tracklet_id"]): event for event in stitch_events}
    for trajectory_id, group in tracks.groupby("identity_trajectory_id", sort=True):
        frames = group["frame"].astype(int)
        expected = ((int(frames.max()) - int(frames.min())) // frame_step) + 1
        tracklet_ids = sorted(int(v) for v in group["tracklet_id"].unique())
        rows.append({
            "identity_trajectory_id": str(trajectory_id),
            "assigned_identity": str(group["assigned_identity"].iloc[0]),
            "frame_start": int(frames.min()),
            "frame_end": int(frames.max()),
            "n_rows": int(len(group)),
            "n_unique_frames": int(frames.nunique()),
            "n_clean_rows": int(group["qc_status"].fillna("").astype(str).eq("clean_train").sum()),
            "n_tracklets": int(len(tracklet_ids)),
            "tracklet_ids": ",".join(str(v) for v in tracklet_ids),
            "frame_step": int(frame_step),
            "n_missing_sampled_frames": int(max(0, expected - frames.nunique())),
            "n_stitches": int(sum(1 for tid in tracklet_ids if tid in stitched_by_to)),
        })
    return pd.DataFrame(rows)


def _empty_tracks(tracklets_qc: pd.DataFrame) -> pd.DataFrame:
    out = tracklets_qc.iloc[0:0].copy()
    for col in [
        "assigned_identity",
        "identity_trajectory_id",
        "stitch_status",
        "stitch_gap_frames",
        "stitch_endpoint_distance_px",
        "stitch_speed_px_per_frame",
    ]:
        out[col] = []
    return out


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "identity_trajectory_id",
        "assigned_identity",
        "frame_start",
        "frame_end",
        "n_rows",
        "n_unique_frames",
        "n_clean_rows",
        "n_tracklets",
        "tracklet_ids",
        "frame_step",
        "n_missing_sampled_frames",
        "n_stitches",
    ])


def _infer_frame_step(frames: pd.Series) -> int:
    unique_frames = pd.Series(frames.astype(int).unique()).sort_values()
    diffs = unique_frames.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return int(diffs.value_counts().sort_values(ascending=False).index[0])
