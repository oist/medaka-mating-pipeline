from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_HYBRID_TRACKING_CONFIG = {
    "anchor_node": "mouth",
    "num_individuals": 3,
    "max_points_per_frame": 3,
    "max_sleap_distance_px_per_frame": 12.0,
    "max_sleap_gap_frames": 500,
    "segmentation_match_distance_px": 90.0,
    "segmentation_anchor_statuses": ["clean_train", "usable_inference", "manual_review"],
    "segmentation_bonus": 80.0,
    "segmentation_conflict_penalty": 220.0,
    "motion_bind_weight": 0.35,
    "segmentation_bind_weight": 1.0,
    "manual_review_bind_weight": 0.15,
    "bootstrap_initial_sleap": True,
    "bootstrap_bind_weight": 0.05,
    "reject_large_segmentation_anchor_jumps": True,
    "max_segmentation_anchor_jump_px_per_frame": 40.0,
    "max_segmentation_anchor_hard_jump_px": 250.0,
    "close_contact_distance_px": 35.0,
}


@dataclass
class TrackState:
    label: str
    track_id: int
    last_frame: int | None = None
    last_pos: tuple[float, float] | None = None
    prev_frame: int | None = None
    prev_pos: tuple[float, float] | None = None


def track_sleap_with_segmentation_identity(
    *,
    sleap_points: pd.DataFrame,
    segmentation_tracks: pd.DataFrame,
    config: Mapping | None = None,
    output_csv: str | Path | None = None,
    events_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = dict(DEFAULT_HYBRID_TRACKING_CONFIG)
    cfg.update(dict(config or {}))
    points = _normalize_sleap_points(sleap_points)
    anchors = _normalize_segmentation_anchors(
        segmentation_tracks,
        allowed_statuses=set(str(v) for v in cfg["segmentation_anchor_statuses"]),
    )
    labels = _identity_labels(anchors, num_individuals=int(cfg["num_individuals"]))
    states = {label: TrackState(label=label, track_id=i) for i, label in enumerate(labels)}

    rows: list[dict] = []
    events: list[dict] = []
    if points.empty:
        out = pd.DataFrame()
        event_df = pd.DataFrame()
        summary = _summary(out, event_df, labels=labels)
        _write_outputs(out, event_df, summary, output_csv, events_csv, summary_json)
        return out, event_df, summary

    points_by_frame = {int(frame): group.copy() for frame, group in points.groupby("frame", sort=True)}
    anchors_by_frame = {int(frame): group.copy() for frame, group in anchors.groupby("frame", sort=True)} if not anchors.empty else {}

    for frame in sorted(points_by_frame):
        detections = _frame_detections(
            points_by_frame[frame],
            max_points_per_frame=int(cfg["max_points_per_frame"]),
        )
        detections = _add_detection_spacing(
            detections,
            close_contact_distance_px=float(cfg["close_contact_distance_px"]),
        )
        frame_anchors = anchors_by_frame.get(frame, _empty_anchors())
        if detections.empty:
            continue
        cost, diagnostics = _cost_matrix(states, detections, frame_anchors, frame=frame, cfg=cfg)
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_labels: set[str] = set()
        assigned_detections: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            label = labels[int(r)]
            if not _accept_match(states[label], detections.iloc[int(c)], diagnostics[(int(r), int(c))], frame=frame, cfg=cfg):
                continue
            det = detections.iloc[int(c)]
            diag = diagnostics[(int(r), int(c))]
            assigned_labels.add(label)
            assigned_detections.add(int(c))
            state = states[label]
            source = "segmentation_anchor" if diag["same_anchor"] else "sleap_motion"
            bind_weight = _bind_weight(source=source, anchor_status=diag["anchor_qc_status"], cfg=cfg)
            rows.append({
                "frame": int(frame),
                "track_id": int(state.track_id),
                "x": float(det["x"]),
                "y": float(det["y"]),
                "score": float(det["score"]) if pd.notna(det["score"]) else np.nan,
                "node": str(det.get("node", "sleap_point") or "sleap_point"),
                "track_label": str(label),
                "identity": str(label),
                "identity_trajectory_id": str(diag["identity_trajectory_id"] or label),
                "source_tracklet_id": int(diag["source_tracklet_id"]),
                "qc_status": str(diag["anchor_qc_status"] or ""),
                "bind_weight": float(bind_weight),
                "tracking_source": source,
                "segmentation_distance_px": float(diag["same_anchor_distance"]),
                "motion_distance_px": float(diag["motion_distance"]),
                "nearest_neighbor_distance_px": float(det.get("nearest_neighbor_distance_px", np.inf)),
                "close_contact": bool(det.get("close_contact", False)),
            })
            if diag["conflict_anchor"]:
                events.append(_event(frame, label, "segmentation_conflict", diagnostics=diag))
            if diag.get("rejected_same_anchor"):
                events.append(_event(frame, label, "rejected_segmentation_anchor_large_jump", diagnostics=diag))
            if bool(det.get("close_contact", False)):
                events.append(_event(frame, label, "close_contact_top_view", diagnostics=diag))
            large_motion = diag["motion_distance"] > float(cfg["max_sleap_distance_px_per_frame"]) * max(1, int(frame) - int(state.last_frame or frame))
            if large_motion:
                events.append(_event(frame, label, "large_motion", diagnostics=diag))
            if source == "segmentation_anchor" and large_motion:
                # A segmentation anchor can intentionally correct an identity after
                # a crossing. Do not carry the old label's velocity through that
                # correction, or the following SLEAP-only frames will be rejected.
                state.prev_frame = None
                state.prev_pos = None
            else:
                state.prev_frame = state.last_frame
                state.prev_pos = state.last_pos
            state.last_frame = int(frame)
            state.last_pos = (float(det["x"]), float(det["y"]))

        if (
            bool(cfg.get("bootstrap_initial_sleap", True))
            and not assigned_labels
            and all(state.last_pos is None for state in states.values())
        ):
            bootstrap = detections.sort_values("x").reset_index(drop=True)
            for idx, label in enumerate(labels[:len(bootstrap)]):
                det = bootstrap.iloc[idx]
                state = states[label]
                assigned_labels.add(label)
                assigned_detections.add(int(idx))
                rows.append({
                    "frame": int(frame),
                    "track_id": int(state.track_id),
                    "x": float(det["x"]),
                    "y": float(det["y"]),
                    "score": float(det["score"]) if pd.notna(det["score"]) else np.nan,
                    "node": str(det.get("node", "sleap_point") or "sleap_point"),
                    "track_label": str(label),
                    "identity": str(label),
                    "identity_trajectory_id": str(label),
                    "source_tracklet_id": -1,
                    "qc_status": "",
                    "bind_weight": float(cfg["bootstrap_bind_weight"]),
                    "tracking_source": "sleap_bootstrap",
                    "segmentation_distance_px": np.inf,
                    "motion_distance_px": np.inf,
                    "nearest_neighbor_distance_px": float(det.get("nearest_neighbor_distance_px", np.inf)),
                    "close_contact": bool(det.get("close_contact", False)),
                })
                state.last_frame = int(frame)
                state.last_pos = (float(det["x"]), float(det["y"]))

        for label in labels:
            if label not in assigned_labels and states[label].last_frame is not None:
                if int(frame) - int(states[label].last_frame) <= int(cfg["max_sleap_gap_frames"]):
                    events.append(_event(frame, label, "missing_sleap_detection", diagnostics={}))
        for det_idx, det in detections.iterrows():
            if int(det_idx) not in assigned_detections:
                events.append({
                    "frame": int(frame),
                    "track_label": "",
                    "event_type": "unassigned_sleap_detection",
                    "x": float(det["x"]),
                    "y": float(det["y"]),
                    "detail": "",
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["frame", "track_id"], inplace=True)
    event_df = pd.DataFrame(events)
    summary = _summary(out, event_df, labels=labels)
    _write_outputs(out, event_df, summary, output_csv, events_csv, summary_json)
    return out, event_df, summary


def _normalize_sleap_points(points: pd.DataFrame) -> pd.DataFrame:
    if points is None or points.empty:
        return pd.DataFrame(columns=["frame", "sleap_index", "x", "y", "score", "node"])
    out = points.copy()
    required = {"frame", "x", "y"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"SLEAP points missing columns: {', '.join(sorted(missing))}")
    if "sleap_index" not in out.columns:
        out["sleap_index"] = out.groupby("frame").cumcount()
    if "score" not in out.columns:
        out["score"] = np.nan
    if "node" not in out.columns:
        out["node"] = "sleap_point"
    out = out[np.isfinite(out["x"].astype(float)) & np.isfinite(out["y"].astype(float))].copy()
    return out[["frame", "sleap_index", "x", "y", "score", "node"]]


def _normalize_segmentation_anchors(tracks: pd.DataFrame, *, allowed_statuses: set[str]) -> pd.DataFrame:
    if tracks is None or tracks.empty:
        return _empty_anchors()
    out = tracks.copy()
    required = {"frame", "centroid_x", "centroid_y"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Segmentation tracks missing columns: {', '.join(sorted(missing))}")
    if "assigned_identity" not in out.columns:
        out["assigned_identity"] = out.get("identity_trajectory_id", "")
    if "identity_trajectory_id" not in out.columns:
        out["identity_trajectory_id"] = out["assigned_identity"]
    if "tracklet_id" not in out.columns:
        out["tracklet_id"] = -1
    if "qc_status" not in out.columns:
        out["qc_status"] = "clean_train"
    out = out[out["qc_status"].fillna("").astype(str).isin(allowed_statuses)].copy()
    out = out[out["assigned_identity"].fillna("").astype(str).ne("unassigned")].copy()
    out = out[np.isfinite(out["centroid_x"].astype(float)) & np.isfinite(out["centroid_y"].astype(float))].copy()
    return out[["frame", "centroid_x", "centroid_y", "assigned_identity", "identity_trajectory_id", "tracklet_id", "qc_status"]]


def _empty_anchors() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "frame", "centroid_x", "centroid_y", "assigned_identity",
        "identity_trajectory_id", "tracklet_id", "qc_status",
    ])


def _identity_labels(anchors: pd.DataFrame, *, num_individuals: int) -> list[str]:
    labels = sorted(str(v) for v in anchors["assigned_identity"].dropna().unique()) if not anchors.empty else []
    for idx in range(max(0, int(num_individuals) - len(labels))):
        label = f"unassigned_{idx}"
        if label not in labels:
            labels.append(label)
    return labels[: int(num_individuals)]


def _frame_detections(points: pd.DataFrame, *, max_points_per_frame: int) -> pd.DataFrame:
    out = points.copy()
    score = pd.to_numeric(out["score"], errors="coerce").fillna(-np.inf)
    out = out.assign(_score_sort=score).sort_values(["_score_sort", "x"], ascending=[False, True])
    out = out.head(max_points_per_frame).drop(columns=["_score_sort"]).reset_index(drop=True)
    return out


def _add_detection_spacing(detections: pd.DataFrame, *, close_contact_distance_px: float) -> pd.DataFrame:
    out = detections.copy()
    if out.empty:
        out["nearest_neighbor_distance_px"] = []
        out["close_contact"] = []
        return out
    coords = out[["x", "y"]].astype(float).to_numpy()
    nearest: list[float] = []
    for i, xy in enumerate(coords):
        if len(coords) <= 1:
            nearest.append(float("inf"))
            continue
        d = np.hypot(coords[:, 0] - xy[0], coords[:, 1] - xy[1])
        d[i] = np.inf
        nearest.append(float(np.min(d)))
    out["nearest_neighbor_distance_px"] = nearest
    out["close_contact"] = out["nearest_neighbor_distance_px"].astype(float) <= float(close_contact_distance_px)
    return out


def _cost_matrix(
    states: dict[str, TrackState],
    detections: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    frame: int,
    cfg: Mapping,
) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    labels = list(states)
    cost = np.full((len(labels), len(detections)), 1e6, dtype=float)
    diagnostics: dict[tuple[int, int], dict] = {}
    for r, label in enumerate(labels):
        state = states[label]
        for c, det in detections.iterrows():
            motion_distance = _motion_distance(state, det, frame=frame)
            same_anchor, conflict_anchor = _anchor_diagnostics(label, det, anchors, cfg=cfg)
            rejected_same_anchor = False
            if same_anchor is not None and _reject_large_anchor_jump(
                state, motion_distance, frame=frame, cfg=cfg,
            ):
                rejected_same_anchor = True
                same_anchor = None
            if state.last_pos is None and same_anchor is None:
                base_cost = 1e5
            else:
                base_cost = motion_distance if np.isfinite(motion_distance) else 1e5
            if same_anchor is not None:
                base_cost = min(base_cost, float(same_anchor["distance_px"])) - float(cfg["segmentation_bonus"])
            if (
                conflict_anchor is not None
                and motion_distance > float(cfg["max_sleap_distance_px_per_frame"])
            ):
                base_cost += float(cfg["segmentation_conflict_penalty"])
            cost[r, int(c)] = base_cost
            diagnostics[(r, int(c))] = {
                "same_anchor": same_anchor is not None,
                "rejected_same_anchor": rejected_same_anchor,
                "conflict_anchor": conflict_anchor is not None,
                "same_anchor_distance": float(same_anchor["distance_px"]) if same_anchor is not None else np.inf,
                "conflict_anchor_distance": float(conflict_anchor["distance_px"]) if conflict_anchor is not None else np.inf,
                "motion_distance": float(motion_distance),
                "identity_trajectory_id": same_anchor["identity_trajectory_id"] if same_anchor is not None else "",
                "source_tracklet_id": int(same_anchor["tracklet_id"]) if same_anchor is not None else -1,
                "anchor_qc_status": same_anchor["qc_status"] if same_anchor is not None else "",
            }
    return cost, diagnostics


def _reject_large_anchor_jump(
    state: TrackState,
    motion_distance: float,
    *,
    frame: int,
    cfg: Mapping,
) -> bool:
    if not bool(cfg.get("reject_large_segmentation_anchor_jumps", True)):
        return False
    if state.last_pos is None or state.last_frame is None:
        return False
    if not np.isfinite(motion_distance):
        return False
    gap = max(1, int(frame) - int(state.last_frame))
    per_frame_limit = float(cfg.get("max_segmentation_anchor_jump_px_per_frame", cfg["max_sleap_distance_px_per_frame"]))
    hard_limit = float(cfg.get("max_segmentation_anchor_hard_jump_px", np.inf))
    limit = min(per_frame_limit * gap, hard_limit)
    return float(motion_distance) > limit


def _motion_distance(state: TrackState, det: pd.Series, *, frame: int) -> float:
    if state.last_pos is None or state.last_frame is None:
        return np.inf
    pred = _predict_position(state, frame=frame)
    return float(np.hypot(float(det["x"]) - pred[0], float(det["y"]) - pred[1]))


def _predict_position(state: TrackState, *, frame: int) -> tuple[float, float]:
    assert state.last_pos is not None
    if state.prev_pos is None or state.prev_frame is None or state.last_frame is None:
        return state.last_pos
    dt = max(1, int(state.last_frame) - int(state.prev_frame))
    gap = max(1, int(frame) - int(state.last_frame))
    vx = (state.last_pos[0] - state.prev_pos[0]) / dt
    vy = (state.last_pos[1] - state.prev_pos[1]) / dt
    return (state.last_pos[0] + vx * gap, state.last_pos[1] + vy * gap)


def _anchor_diagnostics(label: str, det: pd.Series, anchors: pd.DataFrame, *, cfg: Mapping) -> tuple[dict | None, dict | None]:
    if anchors.empty:
        return None, None
    rows: list[dict] = []
    for anchor in anchors.to_dict("records"):
        dist = float(np.hypot(float(det["x"]) - float(anchor["centroid_x"]), float(det["y"]) - float(anchor["centroid_y"])))
        if dist <= float(cfg["segmentation_match_distance_px"]):
            row = dict(anchor)
            row["distance_px"] = dist
            rows.append(row)
    if not rows:
        return None, None
    same = [row for row in rows if str(row["assigned_identity"]) == str(label)]
    other = [row for row in rows if str(row["assigned_identity"]) != str(label)]
    same_best = min(same, key=lambda row: float(row["distance_px"])) if same else None
    other_best = min(other, key=lambda row: float(row["distance_px"])) if other else None
    return same_best, other_best


def _accept_match(state: TrackState, det: pd.Series, diag: dict, *, frame: int, cfg: Mapping) -> bool:
    if diag["same_anchor"]:
        return True
    if state.last_pos is None or state.last_frame is None:
        return False
    gap = max(1, int(frame) - int(state.last_frame))
    return float(diag["motion_distance"]) <= float(cfg["max_sleap_distance_px_per_frame"]) * gap


def _bind_weight(*, source: str, anchor_status: str, cfg: Mapping) -> float:
    if source == "sleap_bootstrap":
        return float(cfg["bootstrap_bind_weight"])
    if source != "segmentation_anchor":
        return float(cfg["motion_bind_weight"])
    if str(anchor_status) == "manual_review":
        return float(cfg["manual_review_bind_weight"])
    return float(cfg["segmentation_bind_weight"])


def _event(frame: int, label: str, event_type: str, *, diagnostics: dict) -> dict:
    return {
        "frame": int(frame),
        "track_label": str(label),
        "event_type": str(event_type),
        "x": np.nan,
        "y": np.nan,
        "detail": json.dumps(diagnostics, default=str, separators=(",", ":")) if diagnostics else "",
    }


def _summary(tracks: pd.DataFrame, events: pd.DataFrame, *, labels: list[str]) -> dict:
    return {
        "n_rows": int(len(tracks)),
        "n_labels": int(len(labels)),
        "labels": labels,
        "frame_start": int(tracks["frame"].min()) if not tracks.empty else None,
        "frame_end": int(tracks["frame"].max()) if not tracks.empty else None,
        "n_unique_frames": int(tracks["frame"].nunique()) if not tracks.empty else 0,
        "rows_per_frame_distribution": {
            str(k): int(v)
            for k, v in tracks.groupby("frame").size().value_counts().sort_index().to_dict().items()
        } if not tracks.empty else {},
        "rows_by_label": {str(k): int(v) for k, v in tracks["track_label"].value_counts().to_dict().items()} if not tracks.empty else {},
        "tracking_source_counts": {str(k): int(v) for k, v in tracks["tracking_source"].value_counts().to_dict().items()} if not tracks.empty else {},
        "event_counts": {str(k): int(v) for k, v in events["event_type"].value_counts().to_dict().items()} if not events.empty else {},
    }


def _write_outputs(
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    summary: dict,
    output_csv: str | Path | None,
    events_csv: str | Path | None,
    summary_json: str | Path | None,
) -> None:
    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(path, index=False)
    if events_csv is not None:
        path = Path(events_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(path, index=False)
    if summary_json is not None:
        path = Path(summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
