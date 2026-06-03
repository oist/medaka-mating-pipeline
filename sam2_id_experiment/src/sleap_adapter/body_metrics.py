from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BODY_METRIC_COLUMNS = [
    "frame",
    "track_id",
    "identity",
    "track_label",
    "identity_trajectory_id",
    "mouth_x",
    "mouth_y",
    "tail_x",
    "tail_y",
    "body_vector_x",
    "body_vector_y",
    "heading_angle_rad",
    "heading_angle_smooth_rad",
    "body_length_px",
    "source_sleap_index",
    "mouth_match_distance_px",
    "body_qc_status",
    "body_qc_reason",
]


def load_sleap_body_points(
    slp_path: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    mouth_node: str = "mouth",
    tail_node: str = "tail",
) -> pd.DataFrame:
    """Load per-instance mouth/tail coordinates from a SLEAP `.slp` file."""
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required to load .slp files. Run `uv sync --extra sleap` "
            "or use an environment with sleap-io installed."
        ) from exc

    labels = sio.load_slp(str(slp_path))
    node_names = [node.name for node in labels.skeletons[0].nodes] if labels.skeletons else []
    if mouth_node not in node_names:
        raise ValueError(f"SLEAP skeleton has no mouth node {mouth_node!r}; nodes={node_names}")
    if tail_node not in node_names:
        raise ValueError(f"SLEAP skeleton has no tail node {tail_node!r}; nodes={node_names}")
    mouth_idx = node_names.index(mouth_node)
    tail_idx = node_names.index(tail_node)

    rows: list[dict] = []
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if start_frame is not None and frame < int(start_frame):
            continue
        if end_frame is not None and frame > int(end_frame):
            continue
        for instance_index, inst in enumerate(lf.instances):
            pts = inst.numpy()
            mouth = pts[mouth_idx] if len(pts) > mouth_idx else (np.nan, np.nan)
            tail = pts[tail_idx] if len(pts) > tail_idx else (np.nan, np.nan)
            score = getattr(inst, "score", None)
            rows.append({
                "frame": frame,
                "sleap_index": int(instance_index),
                "mouth_x": float(mouth[0]) if np.isfinite(mouth).all() else np.nan,
                "mouth_y": float(mouth[1]) if np.isfinite(mouth).all() else np.nan,
                "tail_x": float(tail[0]) if np.isfinite(tail).all() else np.nan,
                "tail_y": float(tail[1]) if np.isfinite(tail).all() else np.nan,
                "score": float(score) if score is not None else np.nan,
            })
    return pd.DataFrame(
        rows,
        columns=["frame", "sleap_index", "mouth_x", "mouth_y", "tail_x", "tail_y", "score"],
    )


def compute_body_metrics_for_tracks(
    *,
    tracks: pd.DataFrame,
    body_points: pd.DataFrame,
    max_mouth_match_distance_px: float = 15.0,
    min_body_length_px: float = 2.0,
    max_body_length_px: float = 250.0,
    tail_near_other_mouth_px: float = 5.0,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """Attach tail/body-axis metrics to already identified mouth tracks.

    This function never changes identity columns. A failed tail match produces
    missing body fields while keeping the mouth trajectory row.
    """
    if tracks is None or tracks.empty:
        return pd.DataFrame(columns=BODY_METRIC_COLUMNS)
    _require_columns(tracks, {"frame", "track_id", "x", "y"}, name="tracks")
    _require_columns(body_points, {"frame", "sleap_index", "mouth_x", "mouth_y", "tail_x", "tail_y"}, name="body_points")

    track_rows = tracks.copy()
    body_by_frame = {
        int(frame): group.copy()
        for frame, group in body_points.groupby("frame", sort=False)
    }
    mouth_by_frame = {
        int(frame): group[["x", "y"]].astype(float).to_numpy()
        for frame, group in track_rows.groupby("frame", sort=False)
    }

    rows: list[dict] = []
    for row in track_rows.itertuples(index=False):
        frame = int(getattr(row, "frame"))
        mouth_x = float(getattr(row, "x"))
        mouth_y = float(getattr(row, "y"))
        match = _nearest_body_point(
            body_by_frame.get(frame),
            mouth_x=mouth_x,
            mouth_y=mouth_y,
            max_distance_px=float(max_mouth_match_distance_px),
        )
        out = _base_output_row(row, mouth_x=mouth_x, mouth_y=mouth_y)
        if match is None:
            out.update(_missing_body_fields("no_sleap_instance_match"))
            rows.append(out)
            continue

        tail_x = float(match["tail_x"])
        tail_y = float(match["tail_y"])
        source_sleap_index = int(match["sleap_index"])
        match_dist = float(match["_mouth_distance_px"])
        if not np.isfinite([tail_x, tail_y]).all():
            out.update(_missing_body_fields("missing_tail", source_sleap_index, match_dist))
            rows.append(out)
            continue

        body_vector_x = mouth_x - tail_x
        body_vector_y = mouth_y - tail_y
        body_length = float(np.hypot(body_vector_x, body_vector_y))
        heading = float(np.arctan2(body_vector_y, body_vector_x))
        reasons = []
        status = "good"
        if body_length < float(min_body_length_px) or body_length > float(max_body_length_px):
            reasons.append("implausible_body_length")
            status = "manual_review"
        if _tail_near_other_mouth(
            tail_x=tail_x,
            tail_y=tail_y,
            mouths=mouth_by_frame.get(frame),
            own_mouth=(mouth_x, mouth_y),
            threshold=float(tail_near_other_mouth_px),
        ):
            reasons.append("tail_near_other_mouth")
            status = "manual_review"
        out.update({
            "tail_x": tail_x,
            "tail_y": tail_y,
            "body_vector_x": body_vector_x,
            "body_vector_y": body_vector_y,
            "heading_angle_rad": heading,
            "heading_angle_smooth_rad": np.nan,
            "body_length_px": body_length,
            "source_sleap_index": source_sleap_index,
            "mouth_match_distance_px": match_dist,
            "body_qc_status": status,
            "body_qc_reason": ";".join(reasons) if reasons else "passed_body_metric_rules",
        })
        rows.append(out)

    out_df = pd.DataFrame(rows, columns=BODY_METRIC_COLUMNS)
    out_df["heading_angle_smooth_rad"] = _smooth_heading_by_identity(
        out_df,
        window=int(smooth_window),
    )
    return out_df


def body_metrics_summary(metrics: pd.DataFrame) -> dict:
    if metrics is None or metrics.empty:
        return {
            "n_rows": 0,
            "n_frames": 0,
            "body_qc_status_counts": {},
            "body_qc_reason_counts": {},
        }
    return {
        "n_rows": int(len(metrics)),
        "n_frames": int(metrics["frame"].nunique()),
        "body_qc_status_counts": {
            str(k): int(v) for k, v in metrics["body_qc_status"].value_counts(dropna=False).items()
        },
        "body_qc_reason_counts": {
            str(k): int(v) for k, v in metrics["body_qc_reason"].value_counts(dropna=False).items()
        },
        "mean_body_length_px": float(metrics["body_length_px"].mean(skipna=True)),
        "median_body_length_px": float(metrics["body_length_px"].median(skipna=True)),
    }


def _nearest_body_point(
    candidates: pd.DataFrame | None,
    *,
    mouth_x: float,
    mouth_y: float,
    max_distance_px: float,
) -> pd.Series | None:
    if candidates is None or candidates.empty:
        return None
    valid = candidates[np.isfinite(candidates["mouth_x"].astype(float)) & np.isfinite(candidates["mouth_y"].astype(float))].copy()
    if valid.empty:
        return None
    dist = np.hypot(valid["mouth_x"].astype(float) - mouth_x, valid["mouth_y"].astype(float) - mouth_y)
    idx = dist.idxmin()
    best_dist = float(dist.loc[idx])
    if best_dist > float(max_distance_px):
        return None
    best = valid.loc[idx].copy()
    best["_mouth_distance_px"] = best_dist
    return best


def _base_output_row(row, *, mouth_x: float, mouth_y: float) -> dict:
    return {
        "frame": int(getattr(row, "frame")),
        "track_id": int(getattr(row, "track_id")),
        "identity": str(getattr(row, "identity", "")),
        "track_label": str(getattr(row, "track_label", getattr(row, "identity", ""))),
        "identity_trajectory_id": str(getattr(row, "identity_trajectory_id", "")),
        "mouth_x": float(mouth_x),
        "mouth_y": float(mouth_y),
    }


def _missing_body_fields(
    reason: str,
    source_sleap_index: int = -1,
    mouth_match_distance_px: float = np.inf,
) -> dict:
    return {
        "tail_x": np.nan,
        "tail_y": np.nan,
        "body_vector_x": np.nan,
        "body_vector_y": np.nan,
        "heading_angle_rad": np.nan,
        "heading_angle_smooth_rad": np.nan,
        "body_length_px": np.nan,
        "source_sleap_index": int(source_sleap_index),
        "mouth_match_distance_px": float(mouth_match_distance_px),
        "body_qc_status": "missing",
        "body_qc_reason": reason,
    }


def _tail_near_other_mouth(
    *,
    tail_x: float,
    tail_y: float,
    mouths: np.ndarray | None,
    own_mouth: tuple[float, float],
    threshold: float,
) -> bool:
    if mouths is None or len(mouths) <= 1 or threshold <= 0:
        return False
    d = np.hypot(mouths[:, 0] - tail_x, mouths[:, 1] - tail_y)
    own_d = np.hypot(mouths[:, 0] - own_mouth[0], mouths[:, 1] - own_mouth[1])
    other = own_d > 1e-6
    return bool(np.any(d[other] <= threshold))


def _smooth_heading_by_identity(metrics: pd.DataFrame, *, window: int) -> pd.Series:
    if metrics.empty or window <= 1:
        return metrics["heading_angle_rad"]
    key_col = _identity_key_column(metrics)
    out = pd.Series(np.nan, index=metrics.index, dtype=float)
    for _key, group in metrics.groupby(key_col, sort=False):
        angles = group["heading_angle_rad"].astype(float)
        sin_roll = np.sin(angles).rolling(window, center=True, min_periods=1).mean()
        cos_roll = np.cos(angles).rolling(window, center=True, min_periods=1).mean()
        smoothed = np.arctan2(sin_roll, cos_roll)
        out.loc[group.index] = smoothed
    return out


def _identity_key_column(df: pd.DataFrame) -> str:
    for col in ["identity", "track_label", "track_id"]:
        if col in df.columns:
            return col
    return "frame"


def _require_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}")
