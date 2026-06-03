from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from sam2_id_experiment.src.tracklets.qc_metrics import compute_tracklet_metrics


QC_STATUSES = {"clean_train", "usable_inference", "manual_review", "reject"}
REJECT_REVIEW_LABELS = {"missed", "merged", "wrong_object", "background"}
REVIEW_REVIEW_LABELS = {"partial", "uncertain"}


def classify_tracklets(
    instance_df: pd.DataFrame,
    qc_config: Mapping,
    *,
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    crossing_distance = float(qc_config.get("crossing_distance_px", 30))
    qc_dict = dict(qc_config)
    per_frame, summary = compute_tracklet_metrics(
        instance_df,
        crossing_distance_px=crossing_distance,
        qc_config=qc_dict,
        image_shape=image_shape,
    )
    if summary.empty:
        summary["qc_status"] = pd.Series(dtype=str)
        summary["qc_reason"] = pd.Series(dtype=str)
        return per_frame, summary

    min_len = int(qc_config.get("min_tracklet_length_frames", 100))
    min_area = float(qc_config.get("min_mask_area", 50))
    max_area = float(qc_config.get("max_mask_area", 100000))
    max_area_cv = float(qc_config.get("max_area_cv", 0.35))
    max_speed = float(qc_config.get("max_centroid_speed_px_per_frame", 80))
    min_conf = float(qc_config.get("min_confidence", 0.8))
    max_missing = int(qc_config.get("max_missing_frames", 0))
    reject_near_crossing = bool(qc_config.get("reject_near_crossing", True))

    statuses: list[str] = []
    reasons: list[str] = []
    tracklet_level_review_flags: list[bool] = []
    for row in summary.to_dict("records"):
        hard_reasons: list[str] = []
        review_reasons: list[str] = []
        tracklet_level_review = False

        if row["mean_area"] < min_area:
            hard_reasons.append("area_too_small")
        if row["mean_area"] > max_area:
            hard_reasons.append("area_too_large")
        if row["mean_confidence"] < min_conf:
            hard_reasons.append("low_confidence")
        if row["n_frames"] < max(2, min_len // 5):
            hard_reasons.append("very_short_tracklet")
        if _has_any_reason(row, REJECT_REVIEW_LABELS, prefix="review_"):
            hard_reasons.append("human_review_reject")

        inference_reasons: list[str] = []
        if row["n_frames"] < min_len:
            inference_reasons.append("short_tracklet")
        if row["area_cv"] > max_area_cv:
            review_reasons.append("unstable_area")
            tracklet_level_review = True
        if row["max_speed"] > max_speed:
            review_reasons.append("centroid_jump")
            tracklet_level_review = True
        if row["n_missing_frames"] > max_missing:
            inference_reasons.append("missing_frames")
        if reject_near_crossing and bool(row["has_crossing_risk"]):
            review_reasons.append("near_crossing")
        auto_reasons = _summary_auto_reasons(row)
        if auto_reasons:
            review_reasons.extend(auto_reasons)
        if _has_any_reason(row, REVIEW_REVIEW_LABELS, prefix="review_"):
            review_reasons.append("human_review_uncertain")

        if hard_reasons:
            statuses.append("reject")
            reasons.append(";".join(hard_reasons + review_reasons + inference_reasons))
        elif review_reasons:
            statuses.append("manual_review")
            reasons.append(";".join(review_reasons + inference_reasons))
        elif inference_reasons:
            statuses.append("usable_inference")
            reasons.append(";".join(inference_reasons))
        else:
            statuses.append("clean_train")
            reasons.append("passed_clean_train_rules")
        tracklet_level_review_flags.append(tracklet_level_review)

    summary = summary.copy()
    summary["qc_status"] = statuses
    summary["qc_reason"] = reasons
    summary["has_tracklet_level_review"] = tracklet_level_review_flags
    per_frame = per_frame.merge(
        summary[["tracklet_id", "qc_status", "qc_reason", "has_tracklet_level_review"]],
        on="tracklet_id",
        how="left",
        suffixes=("", "_tracklet"),
    )
    per_frame.rename(
        columns={
            "qc_status": "tracklet_qc_status",
            "qc_reason": "tracklet_qc_reason",
        },
        inplace=True,
    )
    row_statuses: list[str] = []
    row_reasons: list[str] = []
    for row in per_frame.to_dict("records"):
        status, reason = _classify_frame_row(row)
        row_statuses.append(status)
        row_reasons.append(reason)
    per_frame["qc_status"] = row_statuses
    per_frame["qc_reason"] = row_reasons
    per_frame["qc_segment_id"] = _assign_clean_segments(per_frame)
    return per_frame, summary


def _summary_auto_reasons(row: dict) -> list[str]:
    reasons = str(row.get("auto_qc_reasons", "") or "")
    if not reasons:
        return []
    return [f"auto_{r}" for r in reasons.split(";") if r]


def _has_any_reason(row: dict, labels: set[str], *, prefix: str) -> bool:
    reasons = str(row.get("auto_qc_reasons", "") or "")
    return any(f"{prefix}{label}" in reasons.split(";") for label in labels)


def _classify_frame_row(row: dict) -> tuple[str, str]:
    tracklet_status = str(row.get("tracklet_qc_status", "manual_review"))
    tracklet_reason = str(row.get("tracklet_qc_reason", "") or "")
    auto_reason = str(row.get("auto_qc_reason", "") or "")
    auto_parts = [r for r in auto_reason.split(";") if r]

    if tracklet_status == "reject":
        return "reject", tracklet_reason
    if any(r in {f"review_{label}" for label in REJECT_REVIEW_LABELS} for r in auto_parts):
        return "reject", auto_reason
    if auto_parts:
        return "manual_review", auto_reason
    if bool(row.get("has_tracklet_level_review", False)):
        return "manual_review", tracklet_reason
    if tracklet_status == "usable_inference":
        return "usable_inference", tracklet_reason
    return "clean_train", "passed_clean_frame_rules"


def _assign_clean_segments(per_frame: pd.DataFrame) -> pd.Series:
    segment_ids = pd.Series(-1, index=per_frame.index, dtype=int)
    next_segment = 0
    for _tracklet_id, group in per_frame.sort_values(["tracklet_id", "frame"]).groupby("tracklet_id"):
        frame_step = _infer_frame_step(group["frame"])
        last_frame: int | None = None
        current_segment: int | None = None
        for idx, row in group.iterrows():
            frame = int(row["frame"])
            if str(row["qc_status"]) != "clean_train":
                current_segment = None
                last_frame = frame
                continue
            if current_segment is None or last_frame is None or frame != last_frame + frame_step:
                current_segment = next_segment
                next_segment += 1
            segment_ids.loc[idx] = current_segment
            last_frame = frame
    return segment_ids


def _infer_frame_step(frames: pd.Series) -> int:
    unique_frames = pd.Series(frames.astype(int).unique()).sort_values()
    diffs = unique_frames.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return int(diffs.value_counts().sort_values(ascending=False).index[0])
