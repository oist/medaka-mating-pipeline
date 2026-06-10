from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UNASSIGNED_LABELS = {"", "nan", "none", "null", "unassigned"}


@dataclass(frozen=True)
class SandwichConfig:
    max_gap_frames: int = 500
    max_speed_px_per_frame: float = 12.0
    min_crops: int = 5
    min_prediction_purity: float = 0.75
    min_median_margin: float = 0.05
    max_bridge_deviation_px: float = 80.0
    prediction_col: str = "accepted_predicted_identity"
    fallback_prediction_col: str = "predicted_identity"
    current_identity_col: str | None = "reference_identity"


def run_identity_sandwich_audit(
    *,
    tracklets: pd.DataFrame,
    crop_predictions: pd.DataFrame,
    config: SandwichConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or SandwichConfig()
    tracklet_summary = summarize_tracklets(tracklets)
    evidence = summarize_identity_evidence(crop_predictions, config=cfg)
    repaired = build_repair_table(tracklet_summary, evidence, config=cfg)
    events = find_sandwich_events(tracklet_summary, repaired, tracklets=tracklets, config=cfg)
    repaired = apply_sandwich_repairs(repaired, events, config=cfg)
    summary = summarize_audit(repaired, events)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tracklet_summary.to_parquet(out / "tracklet_motion_summary.parquet", index=False)
        evidence.to_parquet(out / "tracklet_identity_evidence.parquet", index=False)
        events.to_csv(out / "sandwich_events.csv", index=False)
        repaired.to_csv(out / "tracklet_identity_repair.csv", index=False)
        (out / "identity_sandwich_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "summary": summary,
        "tracklet_summary": tracklet_summary,
        "identity_evidence": evidence,
        "sandwich_events": events,
        "tracklet_identity_repair": repaired,
    }


def summarize_tracklets(tracklets: pd.DataFrame) -> pd.DataFrame:
    required = {"tracklet_id", "frame", "centroid_x", "centroid_y"}
    missing = required - set(tracklets.columns)
    if missing:
        raise ValueError(f"Tracklets missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for tracklet_id, group in tracklets.groupby("tracklet_id", sort=True):
        ordered = group.sort_values("frame").copy()
        if "qc_status" in ordered:
            clean = ordered[ordered["qc_status"].fillna("").astype(str).eq("clean_train")]
        else:
            clean = ordered
        endpoints = clean if not clean.empty else ordered
        first = endpoints.iloc[0]
        last = endpoints.iloc[-1]
        qc_counts = (
            ordered["qc_status"].fillna("").astype(str).value_counts().to_dict()
            if "qc_status" in ordered
            else {}
        )
        rows.append(
            {
                "tracklet_id": int(tracklet_id),
                "frame_start": int(ordered["frame"].min()),
                "frame_end": int(ordered["frame"].max()),
                "endpoint_frame_start": int(first["frame"]),
                "endpoint_frame_end": int(last["frame"]),
                "start_x": float(first["centroid_x"]),
                "start_y": float(first["centroid_y"]),
                "end_x": float(last["centroid_x"]),
                "end_y": float(last["centroid_y"]),
                "n_rows": int(len(ordered)),
                "n_clean_rows": int(qc_counts.get("clean_train", 0)),
                "n_manual_rows": int(qc_counts.get("manual_review", 0)),
                "n_reject_rows": int(qc_counts.get("reject", 0)),
                "dominant_qc_status": str(max(qc_counts, key=qc_counts.get)) if qc_counts else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_identity_evidence(crop_predictions: pd.DataFrame, *, config: SandwichConfig) -> pd.DataFrame:
    if crop_predictions.empty:
        return _empty_evidence()
    required = {"tracklet_id", "frame"}
    missing = required - set(crop_predictions.columns)
    if missing:
        raise ValueError(f"Crop predictions missing required columns: {sorted(missing)}")

    pred_col = config.prediction_col if config.prediction_col in crop_predictions.columns else ""
    fallback_col = (
        config.fallback_prediction_col
        if config.fallback_prediction_col in crop_predictions.columns
        else pred_col
    )
    if not pred_col and not fallback_col:
        raise ValueError("Crop predictions need a predicted identity column.")

    rows: list[dict[str, Any]] = []
    for tracklet_id, group in crop_predictions.groupby("tracklet_id", sort=True):
        labels = _usable_labels(group[pred_col]) if pred_col else pd.Series(dtype=str)
        if labels.empty and fallback_col:
            labels = _usable_labels(group[fallback_col])
        label, purity = _majority_label(labels)
        current_label = ""
        current_purity = 0.0
        if config.current_identity_col and config.current_identity_col in group.columns:
            current_label, current_purity = _majority_label(_usable_labels(group[config.current_identity_col]))
        margins = (
            pd.to_numeric(group["embedding_margin"], errors="coerce").dropna()
            if "embedding_margin" in group.columns
            else pd.Series(dtype=float)
        )
        median_margin = float(margins.median()) if not margins.empty else np.nan
        rows.append(
            {
                "tracklet_id": int(tracklet_id),
                "model_identity": str(label),
                "model_identity_purity": float(purity),
                "n_identity_crops": int(len(labels)),
                "median_embedding_margin": median_margin,
                "current_identity": str(current_label),
                "current_identity_purity": float(current_purity),
                "prediction_frame_min": int(group["frame"].min()),
                "prediction_frame_max": int(group["frame"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["has_high_confidence_id"] = (
        out["model_identity"].fillna("").astype(str).map(_is_assigned_label)
        & (out["n_identity_crops"].astype(int) >= int(config.min_crops))
        & (out["model_identity_purity"].astype(float) >= float(config.min_prediction_purity))
        & (
            out["median_embedding_margin"].isna()
            | (out["median_embedding_margin"].astype(float) >= float(config.min_median_margin))
        )
    )
    return out


def build_repair_table(
    tracklet_summary: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    config: SandwichConfig,
) -> pd.DataFrame:
    out = tracklet_summary.merge(evidence, on="tracklet_id", how="left")
    defaults: dict[str, Any] = {
        "model_identity": "",
        "model_identity_purity": 0.0,
        "n_identity_crops": 0,
        "median_embedding_margin": np.nan,
        "current_identity": "",
        "current_identity_purity": 0.0,
        "has_high_confidence_id": False,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    out["repaired_identity"] = np.where(out["has_high_confidence_id"], out["model_identity"], "")
    out["repair_action"] = np.where(out["has_high_confidence_id"], "model_confirmed", "no_high_confidence_model_id")
    out["repair_supporting_event_id"] = ""
    current_assigned = out["current_identity"].fillna("").astype(str).map(_is_assigned_label)
    model_assigned = out["model_identity"].fillna("").astype(str).map(_is_assigned_label)
    out["model_disagrees_with_current"] = (
        current_assigned
        & model_assigned
        & out["has_high_confidence_id"].astype(bool)
        & out["current_identity"].astype(str).ne(out["model_identity"].astype(str))
    )
    out.loc[out["model_disagrees_with_current"], "repair_action"] = "model_disagrees_current"
    return out


def find_sandwich_events(
    tracklet_summary: pd.DataFrame,
    repair_table: pd.DataFrame,
    *,
    tracklets: pd.DataFrame,
    config: SandwichConfig,
) -> pd.DataFrame:
    high = repair_table[repair_table["has_high_confidence_id"].astype(bool)].copy()
    if high.empty:
        return _empty_events()

    critical_frames = _critical_frames(tracklets)
    rows: list[dict[str, Any]] = []
    event_id = 0
    for before in high.sort_values("endpoint_frame_end").to_dict("records"):
        after_candidates = high[
            (high["endpoint_frame_start"].astype(int) > int(before["endpoint_frame_end"]))
            & (
                high["endpoint_frame_start"].astype(int) - int(before["endpoint_frame_end"])
                <= int(config.max_gap_frames)
            )
            & high["tracklet_id"].astype(int).ne(int(before["tracklet_id"]))
        ].copy()
        for after in after_candidates.to_dict("records"):
            gap = int(after["endpoint_frame_start"]) - int(before["endpoint_frame_end"])
            distance = float(
                np.hypot(float(after["start_x"]) - float(before["end_x"]), float(after["start_y"]) - float(before["end_y"]))
            )
            speed = distance / max(1, gap)
            same_identity = str(before["model_identity"]) == str(after["model_identity"])
            critical_count = _count_between(
                critical_frames,
                start=int(before["endpoint_frame_end"]),
                end=int(after["endpoint_frame_start"]),
            )
            geometry_ok = speed <= float(config.max_speed_px_per_frame)
            if same_identity and geometry_ok:
                decision = "confirmed_identity_bridge"
            elif geometry_ok:
                decision = "possible_identity_swap"
            else:
                decision = "geometry_reject"
            rows.append(
                {
                    "event_id": f"bridge_{event_id:05d}",
                    "before_tracklet_id": int(before["tracklet_id"]),
                    "after_tracklet_id": int(after["tracklet_id"]),
                    "frame_start": int(before["endpoint_frame_end"]),
                    "frame_end": int(after["endpoint_frame_start"]),
                    "gap_frames": int(gap),
                    "endpoint_distance_px": float(distance),
                    "speed_px_per_frame": float(speed),
                    "before_identity": str(before["model_identity"]),
                    "after_identity": str(after["model_identity"]),
                    "same_identity": bool(same_identity),
                    "critical_frame_count": int(critical_count),
                    "decision": decision,
                }
            )
            event_id += 1
    if not rows:
        return _empty_events()
    out = pd.DataFrame(rows)
    return out.sort_values(["frame_start", "gap_frames", "event_id"]).reset_index(drop=True)


def apply_sandwich_repairs(
    repair_table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: SandwichConfig,
) -> pd.DataFrame:
    out = repair_table.copy()
    if events.empty:
        return out
    bridge_events = events[events["decision"].eq("confirmed_identity_bridge")].copy()
    for event in bridge_events.sort_values(["gap_frames", "event_id"]).to_dict("records"):
        label = str(event["before_identity"])
        start = int(event["frame_start"])
        end = int(event["frame_end"])
        candidates = out[
            (out["frame_start"].astype(int) >= start)
            & (out["frame_end"].astype(int) <= end)
            & (~out["has_high_confidence_id"].astype(bool))
            & (
                out["dominant_qc_status"].fillna("").astype(str).ne("clean_train")
                | (out["n_clean_rows"].astype(int) < out["n_rows"].astype(int))
            )
        ].copy()
        before_row = out.loc[out["tracklet_id"].eq(int(event["before_tracklet_id"]))].iloc[0]
        after_row = out.loc[out["tracklet_id"].eq(int(event["after_tracklet_id"]))].iloc[0]
        for idx, row in candidates.iterrows():
            speed_to_before = _endpoint_speed_between(before_row, row)
            speed_to_after = _endpoint_speed_between(row, after_row)
            bridge_deviation = _bridge_deviation_px(before_row, row, after_row)
            if (
                speed_to_before <= float(config.max_speed_px_per_frame)
                and speed_to_after <= float(config.max_speed_px_per_frame)
                and bridge_deviation <= float(config.max_bridge_deviation_px)
            ):
                out.loc[idx, "repaired_identity"] = label
                out.loc[idx, "repair_action"] = "sandwich_filled"
                out.loc[idx, "repair_supporting_event_id"] = str(event["event_id"])

    conflict_ids = set()
    for event in events[events["decision"].eq("possible_identity_swap")].to_dict("records"):
        conflict_ids.add(int(event["before_tracklet_id"]))
        conflict_ids.add(int(event["after_tracklet_id"]))
    if conflict_ids:
        mask = out["tracklet_id"].astype(int).isin(conflict_ids) & out["repair_action"].eq("model_confirmed")
        out.loc[mask, "repair_action"] = "needs_review_possible_swap"
    return out


def summarize_audit(repair_table: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_tracklets": int(len(repair_table)),
        "n_high_confidence_model_tracklets": int(repair_table["has_high_confidence_id"].astype(bool).sum())
        if "has_high_confidence_id" in repair_table
        else 0,
        "n_model_current_disagreements": int(repair_table["model_disagrees_with_current"].astype(bool).sum())
        if "model_disagrees_with_current" in repair_table
        else 0,
        "repair_action_counts": {
            str(k): int(v) for k, v in repair_table["repair_action"].value_counts().to_dict().items()
        }
        if "repair_action" in repair_table
        else {},
        "n_sandwich_events": int(len(events)),
        "sandwich_decision_counts": {
            str(k): int(v) for k, v in events["decision"].value_counts().to_dict().items()
        }
        if not events.empty and "decision" in events
        else {},
    }


def _usable_labels(values: pd.Series) -> pd.Series:
    labels = values.fillna("").astype(str)
    return labels[labels.map(_is_assigned_label)]


def _is_assigned_label(value: Any) -> bool:
    return str(value).strip().lower() not in UNASSIGNED_LABELS


def _majority_label(labels: pd.Series) -> tuple[str, float]:
    if labels.empty:
        return "", 0.0
    counts = labels.astype(str).value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / len(labels))


def _critical_frames(tracklets: pd.DataFrame) -> set[int]:
    if tracklets.empty:
        return set()
    flags = pd.Series(False, index=tracklets.index)
    if "qc_status" in tracklets:
        flags |= ~tracklets["qc_status"].fillna("").astype(str).eq("clean_train")
    if "auto_qc_flag" in tracklets:
        flags |= tracklets["auto_qc_flag"].fillna(False).astype(bool)
    if "expected_object_count_match" in tracklets:
        flags |= ~tracklets["expected_object_count_match"].fillna(True).astype(bool)
    return set(int(v) for v in tracklets.loc[flags, "frame"].dropna().astype(int).unique())


def _count_between(frames: set[int], *, start: int, end: int) -> int:
    if not frames:
        return 0
    return sum(1 for frame in frames if int(start) <= int(frame) <= int(end))


def _endpoint_speed_between(left: pd.Series, right: pd.Series) -> float:
    gap = max(1, int(right["endpoint_frame_start"]) - int(left["endpoint_frame_end"]))
    dist = float(np.hypot(float(right["start_x"]) - float(left["end_x"]), float(right["start_y"]) - float(left["end_y"])))
    return dist / gap


def _bridge_deviation_px(before: pd.Series, middle: pd.Series, after: pd.Series) -> float:
    x0 = float(before["end_x"])
    y0 = float(before["end_y"])
    x1 = float(after["start_x"])
    y1 = float(after["start_y"])
    mx = (float(middle["start_x"]) + float(middle["end_x"])) / 2.0
    my = (float(middle["start_y"]) + float(middle["end_y"])) / 2.0
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return float(np.hypot(mx - x0, my - y0))
    t = max(0.0, min(1.0, ((mx - x0) * vx + (my - y0) * vy) / denom))
    px = x0 + t * vx
    py = y0 + t * vy
    return float(np.hypot(mx - px, my - py))


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "tracklet_id",
            "model_identity",
            "model_identity_purity",
            "n_identity_crops",
            "median_embedding_margin",
            "current_identity",
            "current_identity_purity",
            "prediction_frame_min",
            "prediction_frame_max",
            "has_high_confidence_id",
        ]
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "before_tracklet_id",
            "after_tracklet_id",
            "frame_start",
            "frame_end",
            "gap_frames",
            "endpoint_distance_px",
            "speed_px_per_frame",
            "before_identity",
            "after_identity",
            "same_identity",
            "critical_frame_count",
            "decision",
        ]
    )
