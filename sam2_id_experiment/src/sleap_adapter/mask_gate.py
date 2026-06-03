from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    bbox_from_mask,
    iter_mask_instances_h5,
    write_mask_instances_h5,
)


DEFAULT_SLEAP_GATE_CONFIG = {
    "enabled": True,
    "view": "top_cam",
    "max_distance_to_mask_px": 20,
    "max_distance_to_centroid_px": 120,
    "min_sleap_score": 0.0,
    "require_one_mask_per_sleap_point": True,
    "reject_without_sleap_match": True,
}


def filter_masks_by_sleap_points_h5(
    *,
    mask_path: str | Path,
    sleap_points: pd.DataFrame,
    output_masks: str | Path,
    output_table: str | Path | None = None,
    output_csv: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: Mapping | None = None,
    include_manual_review: bool = False,
) -> dict:
    cfg = dict(DEFAULT_SLEAP_GATE_CONFIG)
    cfg.update(dict(config or {}))
    instances = list(iter_mask_instances_h5(mask_path))
    table = match_masks_to_sleap_points(instances, sleap_points, cfg)
    keep_statuses = {"keep", "manual_review"} if include_manual_review else {"keep"}
    keep_keys = {
        (int(row.frame), int(row.instance_id), int(row.original_track_id))
        for row in table.itertuples()
        if str(row.coincidence_status) in keep_statuses
    }
    filtered = [
        inst for inst in instances
        if (int(inst.frame), int(inst.instance_id), int(inst.sam2_track_id)) in keep_keys
    ]
    image_shape = tuple(np.asarray(instances[0].mask).shape[:2]) if instances else None
    write_mask_instances_h5(output_masks, filtered, image_shape=image_shape)

    if output_table is not None:
        path = Path(output_table)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    summary = coincidence_summary(table, n_input_masks=len(instances), n_output_masks=len(filtered))
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def match_masks_to_sleap_points(
    instances: list[MaskInstance],
    sleap_points: pd.DataFrame,
    config: Mapping | None = None,
) -> pd.DataFrame:
    cfg = dict(DEFAULT_SLEAP_GATE_CONFIG)
    cfg.update(dict(config or {}))
    points = _normalize_sleap_points(sleap_points, min_score=float(cfg.get("min_sleap_score", 0.0)))
    points_by_frame = {
        int(frame): group.reset_index(drop=True)
        for frame, group in points.groupby("frame", sort=False)
    }

    rows: list[dict] = []
    candidate_point_rows_by_mask: dict[int, list[int]] = {}
    for row_index, inst in enumerate(instances):
        mask = np.asarray(inst.mask, dtype=bool)
        centroid = _centroid(mask)
        frame_points = points_by_frame.get(int(inst.frame), _empty_points())
        point_matches = [
            _point_match(mask, centroid, point)
            for point in frame_points.to_dict("records")
        ]
        candidates = [
            match for match in point_matches
            if bool(match["point_inside_mask"])
            or float(match["distance_to_mask_px"]) <= float(cfg["max_distance_to_mask_px"])
        ]
        candidate_point_rows_by_mask[row_index] = [int(match["sleap_index"]) for match in candidates]
        best = _best_match(candidates)
        rows.append(_coincidence_row(
            inst=inst,
            row_index=row_index,
            centroid=centroid,
            best=best,
            n_candidate_sleap_points=len(candidates),
            n_sleap_points_in_frame=len(frame_points),
            cfg=cfg,
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_table()

    out["n_masks_for_matched_sleap_point"] = 0
    valid = out["matched_sleap_index"].astype(int) >= 0
    if valid.any():
        counts = (
            out.loc[valid]
            .groupby(["frame", "matched_sleap_index"])["instance_id"]
            .transform("count")
        )
        out.loc[valid, "n_masks_for_matched_sleap_point"] = counts.astype(int)

    statuses: list[str] = []
    reasons: list[str] = []
    for row in out.to_dict("records"):
        status, reason = _final_status(row, cfg)
        statuses.append(status)
        reasons.append(reason)
    out["coincidence_status"] = statuses
    out["coincidence_reason"] = reasons
    return out


def coincidence_summary(table: pd.DataFrame, *, n_input_masks: int, n_output_masks: int) -> dict:
    if table.empty:
        return {
            "n_input_masks": int(n_input_masks),
            "n_output_masks": int(n_output_masks),
            "status_counts": {},
            "reason_counts": {},
            "frames_with_less_than_three_kept_masks": 0,
            "frames_with_more_than_three_kept_masks": 0,
            "many_masks_per_sleap_point": 0,
            "many_sleap_points_per_mask": 0,
        }
    kept = table[table["coincidence_status"] == "keep"]
    kept_counts = kept.groupby("frame").size() if not kept.empty else pd.Series(dtype=int)
    reason_counts: dict[str, int] = {}
    for reason in table["coincidence_reason"].fillna("").astype(str):
        for part in reason.split(";"):
            if part:
                reason_counts[part] = reason_counts.get(part, 0) + 1
    return {
        "n_input_masks": int(n_input_masks),
        "n_output_masks": int(n_output_masks),
        "status_counts": {str(k): int(v) for k, v in table["coincidence_status"].value_counts().to_dict().items()},
        "reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
        "frames_with_less_than_three_kept_masks": int((kept_counts < 3).sum()),
        "frames_with_more_than_three_kept_masks": int((kept_counts > 3).sum()),
        "many_masks_per_sleap_point": int((table["n_masks_for_matched_sleap_point"] > 1).sum()),
        "many_sleap_points_per_mask": int((table["n_candidate_sleap_points"] > 1).sum()),
    }


def _normalize_sleap_points(points: pd.DataFrame, *, min_score: float) -> pd.DataFrame:
    if points is None or points.empty:
        return _empty_points()
    out = points.rename(columns={"sleap_x": "x", "sleap_y": "y", "matched_sleap_score": "score"}).copy()
    required = {"frame", "x", "y"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"SLEAP points missing columns: {', '.join(sorted(missing))}")
    if "sleap_index" not in out.columns:
        out["sleap_index"] = out.groupby("frame").cumcount()
    if "score" not in out.columns:
        out["score"] = np.nan
    if "node" not in out.columns:
        out["node"] = ""
    score = out["score"].astype(float)
    score_ok = score.isna() | (score >= float(min_score))
    out = out[score_ok & np.isfinite(out["x"].astype(float)) & np.isfinite(out["y"].astype(float))].copy()
    return out[["frame", "sleap_index", "x", "y", "score", "node"]]


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame(columns=["frame", "sleap_index", "x", "y", "score", "node"])


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "frame", "instance_id", "original_track_id",
        "matched_sleap_index", "matched_sleap_x", "matched_sleap_y", "matched_sleap_score",
        "point_inside_mask", "distance_to_mask_px", "distance_to_centroid_px",
        "n_candidate_sleap_points", "n_sleap_points_in_frame", "n_masks_for_matched_sleap_point",
        "coincidence_status", "coincidence_reason",
    ])


def _point_match(mask: np.ndarray, centroid: tuple[float, float], point: dict) -> dict:
    x = float(point["x"])
    y = float(point["y"])
    xi = int(round(x))
    yi = int(round(y))
    inside = 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])
    dist_mask = 0.0 if inside else _distance_to_mask(mask, x=x, y=y)
    dist_centroid = float(np.hypot(x - centroid[0], y - centroid[1])) if np.isfinite(centroid[0]) else np.inf
    return {
        "sleap_index": int(point["sleap_index"]),
        "x": x,
        "y": y,
        "score": float(point["score"]) if pd.notna(point["score"]) else np.nan,
        "point_inside_mask": inside,
        "distance_to_mask_px": dist_mask,
        "distance_to_centroid_px": dist_centroid,
    }


def _distance_to_mask(mask: np.ndarray, *, x: float, y: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.inf
    return float(np.sqrt(np.min((xs.astype(float) - x) ** 2 + (ys.astype(float) - y) ** 2)))


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (np.nan, np.nan)
    return (float(xs.mean()), float(ys.mean()))


def _best_match(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda match: (
            0 if bool(match["point_inside_mask"]) else 1,
            float(match["distance_to_mask_px"]),
            float(match["distance_to_centroid_px"]),
            -float(match["score"]) if np.isfinite(match["score"]) else 0.0,
        ),
    )[0]


def _coincidence_row(
    *,
    inst: MaskInstance,
    row_index: int,
    centroid: tuple[float, float],
    best: dict | None,
    n_candidate_sleap_points: int,
    n_sleap_points_in_frame: int,
    cfg: Mapping,
) -> dict:
    if best is None:
        best = {
            "sleap_index": -1,
            "x": np.nan,
            "y": np.nan,
            "score": np.nan,
            "point_inside_mask": False,
            "distance_to_mask_px": np.inf,
            "distance_to_centroid_px": np.inf,
        }
    return {
        "frame": int(inst.frame),
        "instance_id": int(inst.instance_id),
        "original_track_id": int(inst.sam2_track_id),
        "row_index": int(row_index),
        "matched_sleap_index": int(best["sleap_index"]),
        "matched_sleap_x": float(best["x"]),
        "matched_sleap_y": float(best["y"]),
        "matched_sleap_score": float(best["score"]) if np.isfinite(best["score"]) else np.nan,
        "point_inside_mask": bool(best["point_inside_mask"]),
        "distance_to_mask_px": float(best["distance_to_mask_px"]),
        "distance_to_centroid_px": float(best["distance_to_centroid_px"]),
        "n_candidate_sleap_points": int(n_candidate_sleap_points),
        "n_sleap_points_in_frame": int(n_sleap_points_in_frame),
        "n_masks_for_matched_sleap_point": 0,
        "coincidence_status": "reject" if bool(cfg.get("reject_without_sleap_match", True)) else "manual_review",
        "coincidence_reason": "no_sleap_match",
    }


def _final_status(row: dict, cfg: Mapping) -> tuple[str, str]:
    reasons: list[str] = []
    matched = int(row.get("matched_sleap_index", -1)) >= 0
    if not matched:
        reasons.append("no_sleap_match")
        return ("reject" if bool(cfg.get("reject_without_sleap_match", True)) else "manual_review", ";".join(reasons))

    if int(row.get("n_candidate_sleap_points", 0)) > 1:
        reasons.append("multiple_sleap_points_for_mask")
    if bool(cfg.get("require_one_mask_per_sleap_point", True)) and int(row.get("n_masks_for_matched_sleap_point", 0)) > 1:
        reasons.append("multiple_masks_for_sleap_point")
    if float(row.get("distance_to_centroid_px", np.inf)) > float(cfg.get("max_distance_to_centroid_px", np.inf)):
        reasons.append("centroid_too_far")

    if reasons:
        return "manual_review", ";".join(reasons)
    if bool(row.get("point_inside_mask", False)):
        return "keep", "point_inside_mask"
    return "keep", "near_mask"
