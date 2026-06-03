from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from sam2_id_experiment.src.tracklets.mask_to_tracklet import add_speed_columns, summarize_tracklets


def add_nearest_neighbor_distance(instance_df: pd.DataFrame) -> pd.DataFrame:
    if instance_df.empty:
        out = instance_df.copy()
        out["nearest_neighbor_distance_px"] = pd.Series(dtype=float)
        return out

    out = instance_df.copy()
    out["nearest_neighbor_distance_px"] = np.nan
    for _frame, group in out.groupby("frame", sort=False):
        idxs = list(group.index)
        if len(idxs) < 2:
            continue
        xy = group[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        for i, idx in enumerate(idxs):
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            d[i] = np.inf
            out.loc[idx, "nearest_neighbor_distance_px"] = float(np.nanmin(d))
    return out


def compute_tracklet_metrics(
    instance_df: pd.DataFrame,
    *,
    crossing_distance_px: float,
    qc_config: dict | None = None,
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_frame = compute_mask_qc_frames(
        instance_df,
        qc_config=qc_config or {},
        image_shape=image_shape,
    )
    summary = summarize_tracklets(per_frame)
    if summary.empty:
        summary["min_neighbor_distance_px"] = pd.Series(dtype=float)
        summary["has_crossing_risk"] = pd.Series(dtype=bool)
        summary["n_auto_qc_flagged_frames"] = pd.Series(dtype=int)
        summary["auto_qc_reasons"] = pd.Series(dtype=str)
        return per_frame, summary

    nn = (
        per_frame.groupby("tracklet_id")["nearest_neighbor_distance_px"]
        .min()
        .rename("min_neighbor_distance_px")
        .reset_index()
    )
    summary = summary.merge(nn, on="tracklet_id", how="left")
    summary["has_crossing_risk"] = (
        summary["min_neighbor_distance_px"].notna()
        & (summary["min_neighbor_distance_px"] < float(crossing_distance_px))
    )
    flag_counts = (
        per_frame.groupby("tracklet_id")["auto_qc_flag"]
        .sum()
        .astype(int)
        .rename("n_auto_qc_flagged_frames")
        .reset_index()
    )
    reason_rows = []
    for tracklet_id, group in per_frame.groupby("tracklet_id"):
        reasons: set[str] = set()
        for reason in group["auto_qc_reason"].fillna("").astype(str):
            reasons.update(r for r in reason.split(";") if r)
        reason_rows.append({
            "tracklet_id": int(tracklet_id),
            "auto_qc_reasons": ";".join(sorted(reasons)),
        })
    summary = summary.merge(flag_counts, on="tracklet_id", how="left")
    summary = summary.merge(pd.DataFrame(reason_rows), on="tracklet_id", how="left")
    summary["n_auto_qc_flagged_frames"] = summary["n_auto_qc_flagged_frames"].fillna(0).astype(int)
    summary["auto_qc_reasons"] = summary["auto_qc_reasons"].fillna("")
    return per_frame, summary


def compute_mask_qc_frames(
    instance_df: pd.DataFrame,
    *,
    qc_config: dict,
    image_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Compute one-row-per-mask automatic QC measurements and flags."""
    if instance_df.empty:
        out = instance_df.copy()
        for col, dtype in _mask_qc_empty_columns().items():
            out[col] = pd.Series(dtype=dtype)
        return out

    out = add_speed_columns(add_nearest_neighbor_distance(instance_df)).copy()
    out.sort_values(["tracklet_id", "frame"], inplace=True)

    prev_area = out.groupby("tracklet_id")["area"].shift(1)
    area = out["area"].astype(float)
    smaller = np.minimum(area, prev_area.astype(float))
    larger = np.maximum(area, prev_area.astype(float))
    out["area_ratio_prev"] = np.where((smaller > 0) & np.isfinite(smaller), larger / smaller, np.nan)

    out["object_count_frame"] = out.groupby("frame")["tracklet_id"].transform("count").astype(int)
    expected_count = int(qc_config.get("expected_object_count") or qc_config.get("n_animals_expected") or 0)
    if expected_count > 0:
        out["expected_object_count_match"] = out["object_count_frame"] == expected_count
    else:
        out["expected_object_count_match"] = True

    out["bbox_touches_image_edge"] = _bbox_touches_edge(
        out,
        image_shape=image_shape,
        edge_margin_px=int(qc_config.get("edge_margin_px", 5)),
        enabled=bool(qc_config.get("flag_bbox_touching_edge", True)),
    )
    out["overlap_fraction_max"] = _max_overlap_fraction(out, image_shape=image_shape)
    if "review_label" not in out.columns:
        out["review_label"] = ""
    out["review_label"] = out["review_label"].fillna("").astype(str)

    out["auto_qc_reason"] = [
        _auto_qc_reason(row, qc_config=qc_config)
        for row in out.to_dict("records")
    ]
    out["auto_qc_flag"] = out["auto_qc_reason"].astype(str) != ""
    return out


def _mask_qc_empty_columns() -> dict[str, str]:
    return {
        "area_ratio_prev": "float",
        "object_count_frame": "int",
        "expected_object_count_match": "bool",
        "bbox_touches_image_edge": "bool",
        "overlap_fraction_max": "float",
        "auto_qc_flag": "bool",
        "auto_qc_reason": "str",
        "review_label": "str",
    }


def _bbox_touches_edge(
    df: pd.DataFrame,
    *,
    image_shape: tuple[int, int] | None,
    edge_margin_px: int,
    enabled: bool,
) -> pd.Series:
    if not enabled or image_shape is None:
        return pd.Series(False, index=df.index, dtype=bool)
    height, width = int(image_shape[0]), int(image_shape[1])
    margin = max(0, int(edge_margin_px))
    return (
        (df["bbox_x0"] <= margin)
        | (df["bbox_y0"] <= margin)
        | (df["bbox_x1"] >= width - margin)
        | (df["bbox_y1"] >= height - margin)
    ).astype(bool)


def _max_overlap_fraction(df: pd.DataFrame, *, image_shape: tuple[int, int] | None) -> pd.Series:
    overlap = pd.Series(0.0, index=df.index, dtype=float)
    for _frame, group in df.groupby("frame", sort=False):
        idxs = list(group.index)
        if len(idxs) < 2:
            continue
        masks = [_row_mask(row, image_shape=image_shape) for _idx, row in group.iterrows()]
        for i, idx_i in enumerate(idxs):
            max_frac = 0.0
            area_i = float(group.iloc[i]["area"])
            for j, _idx_j in enumerate(idxs):
                if i == j:
                    continue
                frac = _pair_overlap_fraction(group.iloc[i], group.iloc[j], masks[i], masks[j], area_i)
                max_frac = max(max_frac, frac)
            overlap.loc[idx_i] = max_frac
    return overlap


def _row_mask(row: pd.Series, *, image_shape: tuple[int, int] | None) -> np.ndarray | None:
    contour = str(row.get("contour_json", "") or "")
    if not contour or contour == "[]":
        return None
    try:
        pts = np.asarray(json.loads(contour), dtype=np.int32)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None
    if image_shape is None:
        x0, y0, x1, y1 = (int(row["bbox_x0"]), int(row["bbox_y0"]), int(row["bbox_x1"]), int(row["bbox_y1"]))
        width = max(1, x1 - x0 + 2)
        height = max(1, y1 - y0 + 2)
        shifted = pts - np.asarray([[x0, y0]], dtype=np.int32)
        shape = (height, width)
    else:
        shifted = pts
        shape = (int(image_shape[0]), int(image_shape[1]))
    try:
        import cv2
    except ImportError:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def _pair_overlap_fraction(
    row_a: pd.Series,
    row_b: pd.Series,
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    area_a: float,
) -> float:
    if area_a <= 0 or not math.isfinite(area_a):
        return 0.0
    if mask_a is not None and mask_b is not None and mask_a.shape == mask_b.shape:
        return float(np.logical_and(mask_a, mask_b).sum()) / area_a
    return _bbox_intersection_area(row_a, row_b) / area_a


def _bbox_intersection_area(row_a: pd.Series, row_b: pd.Series) -> float:
    x0 = max(float(row_a["bbox_x0"]), float(row_b["bbox_x0"]))
    y0 = max(float(row_a["bbox_y0"]), float(row_b["bbox_y0"]))
    x1 = min(float(row_a["bbox_x1"]), float(row_b["bbox_x1"]))
    y1 = min(float(row_a["bbox_y1"]), float(row_b["bbox_y1"]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _auto_qc_reason(row: dict, *, qc_config: dict) -> str:
    reasons: list[str] = []
    max_area_ratio = float(qc_config.get("max_area_ratio_change", 2.0))
    max_overlap = float(qc_config.get("max_overlap_fraction", 0.05))
    crossing_distance = float(qc_config.get("crossing_distance_px", 30))
    area_ratio = row.get("area_ratio_prev")
    if area_ratio is not None and np.isfinite(area_ratio) and float(area_ratio) > max_area_ratio:
        reasons.append("area_ratio_change")
    if bool(row.get("bbox_touches_image_edge", False)):
        reasons.append("bbox_touches_image_edge")
    if not bool(row.get("expected_object_count_match", True)):
        reasons.append("object_count_mismatch")
    overlap = row.get("overlap_fraction_max")
    if overlap is not None and np.isfinite(overlap) and float(overlap) > max_overlap:
        reasons.append("mask_overlap")
    neighbor = row.get("nearest_neighbor_distance_px")
    if neighbor is not None and np.isfinite(neighbor) and float(neighbor) < crossing_distance:
        reasons.append("near_crossing")

    review_label = str(row.get("review_label", "") or "").strip().lower()
    if review_label in {"partial", "uncertain"}:
        reasons.append(f"review_{review_label}")
    elif review_label in {"missed", "merged", "wrong_object", "background"}:
        reasons.append(f"review_{review_label}")
    return ";".join(reasons)


def rejected_intervals_from_qc(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["tracklet_id", "frame_start", "frame_end", "qc_status", "qc_reason"])
    rejected = summary_df[summary_df["qc_status"] != "clean_train"].copy()
    return rejected[["tracklet_id", "frame_start", "frame_end", "qc_status", "qc_reason"]]
