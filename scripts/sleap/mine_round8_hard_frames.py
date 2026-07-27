"""Mine and package synchronized hard frames for Round 8 SLEAP annotation.

This workflow is intentionally experiment-focused. It combines cached Round 7
per-view predictions, strict-mouth direct 3D geometry, contact intervals, and
manual FP/FN evidence into one synchronized source-frame table. Selection is
deterministic, category-exclusive, event-aware, and reserves the known
96,612--97,802 difficult event as a locked 60-frame holdout sentinel.

Large feature caches and ``pkg.slp`` seeds are written below the configured
``runs_dir``. Only small CSV/JSON/runbook artifacts are written to ``report/``.
The script never launches a GUI and never modifies a human annotation ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.analysis.contact_episode_gt import (
    VIEWS,
    file_provenance,
    prepare_tracking_ifd,
    read_annotation_ledger,
    stable_json,
    utc_now,
    validate_false_negative_detections,
    validate_false_positive_detections,
)
from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, REPO_ROOT, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.sleap.mine_training_frames import _simple_nn_track
from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
    full_save_path,
)


EXPERIMENT = "20251225_K_1_2_FCMC"
ROUND7_TAG = "round7_260704"
ROUND7_MODEL_DIR = "retrain_20260704"
RANDOM_SEED = 20260721
FEATURE_CACHE_VERSION = 3
SENTINEL_START = 96_612
SENTINEL_END = 97_802

CATEGORY_QUOTAS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("close_multi_view_ambiguity", 150),
        ("manual_fp_or_fn", 60),
        ("isolated_xy_outlier", 90),
        ("close_single_view_ambiguity", 30),
        ("close_clean_control", 30),
    ]
)

SELECTION_COLUMNS = [
    "experiment",
    "frame",
    "category",
    "category_score",
    "ambiguous_view_count",
    "ambiguous_views",
    "ambiguity_signals",
    "minimum_ifd_mm",
    "maximum_xy_residual_px",
    "affected_ids",
    "manual_fp_views",
    "manual_fn_views",
    "top_cam_instance_count",
    "side1_cam_instance_count",
    "side2_cam_instance_count",
    "top_cam_plausible_count",
    "side1_cam_plausible_count",
    "side2_cam_plausible_count",
    "contact_episode_ids",
    "event_group_id",
    "event_start_frame",
    "event_end_frame",
    "existing_label_overlap",
    "split",
    "annotation_batch",
    "selection_reason",
]


@dataclass(frozen=True)
class FeatureConfig:
    plausible_score: float = 0.20
    low_score: float = 0.50
    score_dispersion: float = 0.20
    mouth_close_px: float = 25.0
    association_gate_px: float = 80.0
    association_margin_px: float = 15.0
    nn_gate_px: float = 80.0
    residual_floor_px: float = 15.0
    residual_mad_multiplier: float = 6.0
    residual_window_frames: int = 333


@dataclass(frozen=True)
class SelectionConfig:
    random_seed: int = RANDOM_SEED
    time_bins: int = 30
    min_spacing_frames: int = 20
    event_merge_gap_frames: int = 166
    contact_context_frames: int = 166
    close_ifd_mm: float = 16.0
    guard_band_frames: int = 166
    sentinel_start: int = SENTINEL_START
    sentinel_end: int = SENTINEL_END
    holdout_frames: int = 60
    train_frames: int = 300


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _node_score(inst: Any, index: int) -> float:
    points = getattr(inst, "points", None)
    if points is None or index >= len(points):
        return float("nan")
    try:
        return float(points[index]["score"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def isolated_residuals(
    observations: pd.DataFrame,
    *,
    local_window_frames: int = 333,
    residual_floor_px: float = 15.0,
    mad_multiplier: float = 6.0,
) -> pd.DataFrame:
    """Compute exact-consecutive midpoint residuals for tracked observations.

    Input columns are ``frame, track_id, x, y``. A gap at either neighbor
    leaves the residual missing; it is never converted into a large jump.
    """
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations are missing columns: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    for track_id, group in observations.groupby("track_id", sort=True):
        group = group.sort_values("frame", kind="stable").copy()
        frame = group["frame"].to_numpy(dtype=np.int64)
        xy = group[["x", "y"]].to_numpy(dtype=float)
        residual = np.full(len(group), np.nan, dtype=float)
        if len(group) >= 3:
            exact = (frame[1:-1] - frame[:-2] == 1) & (frame[2:] - frame[1:-1] == 1)
            finite = (
                np.isfinite(xy[:-2]).all(axis=1)
                & np.isfinite(xy[1:-1]).all(axis=1)
                & np.isfinite(xy[2:]).all(axis=1)
            )
            valid = exact & finite
            midpoint = 0.5 * (xy[:-2] + xy[2:])
            values = np.linalg.norm(xy[1:-1] - midpoint, axis=1)
            residual[1:-1] = np.where(valid, values, np.nan)
        series = pd.Series(residual, index=group.index, dtype=float)
        minimum_periods = max(5, min(31, int(local_window_frames) // 4))
        local_median = series.rolling(
            int(local_window_frames), center=True, min_periods=minimum_periods
        ).median()

        # Pandas implements rolling median in compiled code. Center each
        # observation by its rolling median and take a second rolling median
        # over the absolute deviations. This is the scalable rolling-MAD
        # estimator for full-length (millions of detections) inference files;
        # a Python callback per window is prohibitively slow.
        absolute_deviation = (series - local_median).abs()
        local_mad = absolute_deviation.rolling(
            int(local_window_frames), center=True, min_periods=minimum_periods
        ).median()
        threshold = np.maximum(
            float(residual_floor_px),
            local_median.to_numpy(dtype=float)
            + float(mad_multiplier) * local_mad.to_numpy(dtype=float),
        )
        threshold = np.where(np.isfinite(threshold), threshold, float(residual_floor_px))
        result = pd.DataFrame(
            {
                "frame": frame,
                "track_id": str(track_id),
                "residual_px": residual,
                "residual_threshold_px": threshold,
                "is_isolated_outlier": np.isfinite(residual) & (residual > threshold),
            }
        )
        rows.append(result)
    if not rows:
        return pd.DataFrame(
            columns=[
                "frame", "track_id", "residual_px", "residual_threshold_px",
                "is_isolated_outlier",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def count_ambiguous_views(frame_table: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic ambiguity counts/names from ``<view>_ambiguous`` flags."""
    table = frame_table.copy()
    columns = [f"{view}_ambiguous" for view in VIEWS]
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"frame table is missing ambiguity columns: {sorted(missing)}")
    matrix = table[columns].fillna(False).to_numpy(dtype=bool)
    table["ambiguous_view_count"] = matrix.sum(axis=1).astype(np.int8)
    table["ambiguous_views"] = [
        ";".join(view for view, flag in zip(VIEWS, flags, strict=True) if flag)
        for flags in matrix
    ]
    return table


def _extract_view_features_object(
    slp_path: Path,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        import sleap_io as sio
    except ImportError as error:  # pragma: no cover - deployment guard
        raise RuntimeError("sleap-io is required for Round 8 feature mining") from error

    labels = sio.load_slp(str(slp_path))
    if not labels.videos:
        raise ValueError(f"{slp_path} contains no video")
    source_video = labels.videos[0]
    source_path = _video_source_path(source_video)
    node_names = [node.name for node in labels.skeletons[0].nodes]
    if "mouth" not in node_names or "tail" not in node_names:
        raise ValueError(f"{slp_path} skeleton must contain mouth and tail, got {node_names}")
    mouth_index = node_names.index("mouth")
    tail_index = node_names.index("tail")

    instance_rows: list[dict[str, Any]] = []
    frame_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    maximum_lf_frame = -1
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        maximum_lf_frame = max(maximum_lf_frame, frame)
        for instance_index, inst in enumerate(lf.instances):
            points = np.asarray(inst.numpy(), dtype=float)
            finite_nodes = np.isfinite(points).all(axis=1) if len(points) else np.zeros(0, bool)
            if not finite_nodes.any():
                continue
            visible_points = points[finite_nodes]
            centroid = visible_points.mean(axis=0)
            mouth = points[mouth_index] if mouth_index < len(points) else np.full(2, np.nan)
            tail = points[tail_index] if tail_index < len(points) else np.full(2, np.nan)
            score = getattr(inst, "score", np.nan)
            score = float(score) if score is not None else float("nan")
            plausible = not np.isfinite(score) or score >= config.plausible_score
            assoc = mouth if np.isfinite(mouth).all() else centroid
            record = {
                "frame": frame,
                "instance_index": int(instance_index),
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "assoc_x": float(assoc[0]),
                "assoc_y": float(assoc[1]),
                "mouth_x": float(mouth[0]),
                "mouth_y": float(mouth[1]),
                "tail_x": float(tail[0]),
                "tail_y": float(tail[1]),
                "mouth_score": _node_score(inst, mouth_index),
                "tail_score": _node_score(inst, tail_index),
                "instance_score": score,
                "visible_node_count": int(finite_nodes.sum()),
                "plausible": bool(plausible),
            }
            instance_rows.append(record)
            frame_records[frame].append(record)

    try:
        video_frames = int(source_video.shape[0])
    except (AttributeError, TypeError, ValueError):
        video_frames = maximum_lf_frame + 1
    n_frames = max(video_frames, maximum_lf_frame + 1)
    instances = pd.DataFrame(instance_rows)
    if instances.empty:
        instances = pd.DataFrame(
            columns=[
                "frame", "instance_index", "x", "y", "assoc_x", "assoc_y",
                "mouth_x", "mouth_y", "tail_x", "tail_y", "mouth_score",
                "tail_score", "instance_score", "visible_node_count", "plausible",
            ]
        )

    track_input = [
        {"frame": int(row.frame), "x": float(row.x), "y": float(row.y)}
        for row in instances.loc[instances["plausible"].astype(bool)].itertuples()
    ]
    track_ids = _simple_nn_track(track_input, gate_px=config.nn_gate_px, max_gap_frames=1)
    instances["track_id"] = ""
    plausible_indices = instances.index[instances["plausible"].astype(bool)].tolist()
    for index, track_id in zip(plausible_indices, track_ids, strict=True):
        instances.at[index, "track_id"] = f"track_{int(track_id)}" if track_id >= 0 else ""

    tracked = instances.loc[instances["track_id"].astype(str).ne("")].copy()
    residuals = isolated_residuals(
        tracked[["frame", "track_id", "x", "y"]],
        local_window_frames=config.residual_window_frames,
        residual_floor_px=config.residual_floor_px,
        mad_multiplier=config.residual_mad_multiplier,
    )
    if residuals.empty:
        instances["residual_px"] = np.nan
        instances["residual_threshold_px"] = np.nan
        instances["is_isolated_outlier"] = False
    else:
        # A track has at most one observation per frame. A vectorized join is
        # materially faster than millions of MultiIndex scalar lookups on a
        # full-length inference file while preserving the original row order.
        residual_columns = [
            "frame",
            "track_id",
            "residual_px",
            "residual_threshold_px",
            "is_isolated_outlier",
        ]
        instances = instances.merge(
            residuals[residual_columns],
            on=["frame", "track_id"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        instances["is_isolated_outlier"] = (
            instances["is_isolated_outlier"].fillna(False).astype(bool)
        )

    residual_summary: dict[int, dict[str, Any]] = {}
    for frame, group in instances.groupby("frame", sort=False):
        outliers = group.loc[group["is_isolated_outlier"].astype(bool)]
        finite_residual = group["residual_px"].to_numpy(dtype=float)
        finite_residual = finite_residual[np.isfinite(finite_residual)]
        residual_summary[int(frame)] = {
            "maximum_xy_residual_px": (
                float(finite_residual.max()) if len(finite_residual) else float("nan")
            ),
            "outlier_count": int(len(outliers)),
            "affected_track_ids": ";".join(
                sorted(set(outliers["track_id"].astype(str)))
            ),
        }

    summary_rows: list[dict[str, Any]] = []
    for frame in range(n_frames):
        records = frame_records.get(frame, [])
        plausible = [record for record in records if record["plausible"]]
        scores = np.asarray(
            [record["instance_score"] for record in plausible if np.isfinite(record["instance_score"])],
            dtype=float,
        )
        mouths = np.asarray(
            [
                [record["mouth_x"], record["mouth_y"]]
                for record in plausible
                if np.isfinite([record["mouth_x"], record["mouth_y"]]).all()
            ],
            dtype=float,
        )
        min_mouth_distance = float("nan")
        if len(mouths) >= 2:
            distances = [
                float(np.linalg.norm(mouths[first] - mouths[second]))
                for first in range(len(mouths))
                for second in range(first + 1, len(mouths))
            ]
            min_mouth_distance = min(distances)
        residual = residual_summary.get(frame, {})
        summary_rows.append(
            {
                "frame": frame,
                "instance_count": len(records),
                "plausible_count": len(plausible),
                "partial_instance_count": sum(record["visible_node_count"] < 2 for record in plausible),
                "minimum_mouth_distance_px": min_mouth_distance,
                "minimum_score": float(scores.min()) if len(scores) else float("nan"),
                "maximum_score": float(scores.max()) if len(scores) else float("nan"),
                "score_std": float(scores.std()) if len(scores) else float("nan"),
                "score_range": float(scores.max() - scores.min()) if len(scores) else float("nan"),
                "maximum_xy_residual_px": residual.get("maximum_xy_residual_px", float("nan")),
                "outlier_count": int(residual.get("outlier_count", 0)),
                "affected_track_ids": str(residual.get("affected_track_ids", "")),
            }
        )
    frames = pd.DataFrame(summary_rows)
    meta = {
        "cache_version": FEATURE_CACHE_VERSION,
        "source_slp": _fingerprint(slp_path),
        "source_video": source_path,
        "node_names": node_names,
        "n_frames": n_frames,
        "n_instances": int(len(instances)),
        "feature_config": asdict(config),
    }
    return frames, instances, meta


class _FastSlpPathUnavailable(RuntimeError):
    """The SLP does not satisfy the uniform predicted-instance fast path."""


def _extract_view_features_hdf5(
    slp_path: Path,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract uniform two-node prediction SLPs without Python object inflation.

    Full-length Round 7 files contain about one million predicted instances per
    view. Their native HDF5 tables are regular (predicted instances only, one
    fixed mouth/tail point span per instance), so validating and reading those
    arrays directly is both exact and orders of magnitude faster than creating
    a Python object for every instance. Irregular/user-label SLPs fall back to
    :func:`_extract_view_features_object`.
    """
    try:
        import h5py
        import sleap_io as sio
    except ImportError as error:  # pragma: no cover - deployment guard
        raise RuntimeError("h5py and sleap-io are required for Round 8 mining") from error

    labels = sio.load_slp(str(slp_path), open_videos=False, lazy=True)
    try:
        if not labels.videos or len(labels.skeletons) != 1:
            raise _FastSlpPathUnavailable("missing video or unique skeleton metadata")
        source_path = _video_source_path(labels.videos[0])
        try:
            video_frame_count = int(labels.videos[0].shape[0])
        except (AttributeError, TypeError, ValueError):
            video_frame_count = 0
        node_names = [node.name for node in labels.skeletons[0].nodes]
    finally:
        labels.close()
    if node_names != ["mouth", "tail"]:
        raise _FastSlpPathUnavailable(
            f"fast path requires mouth/tail node order, got {node_names}"
        )

    with h5py.File(slp_path, "r") as store:
        required = {"frames", "instances", "pred_points"}
        if not required.issubset(store):
            raise _FastSlpPathUnavailable("missing native SLP prediction tables")
        raw_frames = store["frames"][:]
        raw_instances = store["instances"][:]
        raw_points = store["pred_points"][:]

    if not len(raw_frames):
        raise _FastSlpPathUnavailable("SLP contains no labeled-frame rows")
    instance_types = np.asarray(raw_instances["instance_type"], dtype=np.uint8)
    point_starts = np.asarray(raw_instances["point_id_start"], dtype=np.int64)
    point_ends = np.asarray(raw_instances["point_id_end"], dtype=np.int64)
    point_counts = point_ends - point_starts
    frame_starts = np.asarray(raw_frames["instance_id_start"], dtype=np.int64)
    frame_ends = np.asarray(raw_frames["instance_id_end"], dtype=np.int64)
    frame_counts = frame_ends - frame_starts
    expected_starts = np.r_[0, np.cumsum(frame_counts[:-1], dtype=np.int64)]
    if (
        np.any(instance_types != 1)
        or np.any(point_counts != len(node_names))
        or len(raw_instances) != int(frame_counts.sum())
        or not np.array_equal(frame_starts, expected_starts)
        or (len(raw_instances) and int(point_ends.max()) > len(raw_points))
    ):
        raise _FastSlpPathUnavailable(
            "mixed instance types or non-uniform frame/point ranges"
        )

    instance_frames = np.repeat(
        np.asarray(raw_frames["frame_idx"], dtype=np.int64), frame_counts
    )
    repeated_frame_starts = np.repeat(frame_starts, frame_counts)
    instance_index = np.arange(len(raw_instances), dtype=np.int64) - repeated_frame_starts
    point_index = point_starts[:, None] + np.arange(len(node_names), dtype=np.int64)
    x = np.asarray(raw_points["x"], dtype=float)[point_index]
    y = np.asarray(raw_points["y"], dtype=float)[point_index]
    point_scores = np.asarray(raw_points["score"], dtype=float)[point_index]
    finite_nodes = np.isfinite(x) & np.isfinite(y)
    visible_count = finite_nodes.sum(axis=1).astype(np.int8)
    if np.any(visible_count == 0):
        keep = visible_count > 0
        instance_frames = instance_frames[keep]
        instance_index = instance_index[keep]
        x = x[keep]
        y = y[keep]
        point_scores = point_scores[keep]
        finite_nodes = finite_nodes[keep]
        visible_count = visible_count[keep]
        raw_instances = raw_instances[keep]

    safe_x = np.where(finite_nodes, x, 0.0)
    safe_y = np.where(finite_nodes, y, 0.0)
    centroid_x = safe_x.sum(axis=1) / visible_count
    centroid_y = safe_y.sum(axis=1) / visible_count
    mouth_finite = finite_nodes[:, 0]
    mouth_x = np.where(mouth_finite, x[:, 0], np.nan)
    mouth_y = np.where(mouth_finite, y[:, 0], np.nan)
    tail_x = np.where(finite_nodes[:, 1], x[:, 1], np.nan)
    tail_y = np.where(finite_nodes[:, 1], y[:, 1], np.nan)
    association_x = np.where(mouth_finite, mouth_x, centroid_x)
    association_y = np.where(mouth_finite, mouth_y, centroid_y)
    instance_score = np.asarray(raw_instances["score"], dtype=float)
    plausible = ~np.isfinite(instance_score) | (instance_score >= config.plausible_score)

    instances = pd.DataFrame(
        {
            "frame": instance_frames,
            "instance_index": instance_index,
            "x": centroid_x,
            "y": centroid_y,
            "assoc_x": association_x,
            "assoc_y": association_y,
            "mouth_x": mouth_x,
            "mouth_y": mouth_y,
            "tail_x": tail_x,
            "tail_y": tail_y,
            "mouth_score": point_scores[:, 0],
            "tail_score": point_scores[:, 1],
            "instance_score": instance_score,
            "visible_node_count": visible_count,
            "plausible": plausible,
        }
    )

    track_input = [
        {"frame": int(row.frame), "x": float(row.x), "y": float(row.y)}
        for row in instances.loc[instances["plausible"]].itertuples()
    ]
    track_ids = _simple_nn_track(
        track_input, gate_px=config.nn_gate_px, max_gap_frames=1
    )
    instances["track_id"] = ""
    plausible_indices = instances.index[instances["plausible"]].to_numpy()
    instances.loc[plausible_indices, "track_id"] = [
        f"track_{int(track_id)}" if track_id >= 0 else "" for track_id in track_ids
    ]
    tracked = instances.loc[instances["track_id"].ne("")]
    residuals = isolated_residuals(
        tracked[["frame", "track_id", "x", "y"]],
        local_window_frames=config.residual_window_frames,
        residual_floor_px=config.residual_floor_px,
        mad_multiplier=config.residual_mad_multiplier,
    )
    if residuals.empty:
        instances["residual_px"] = np.nan
        instances["residual_threshold_px"] = np.nan
        instances["is_isolated_outlier"] = False
    else:
        instances = instances.merge(
            residuals,
            on=["frame", "track_id"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        instances["is_isolated_outlier"] = (
            instances["is_isolated_outlier"].fillna(False).astype(bool)
        )

    n_frames = max(
        video_frame_count,
        int(np.asarray(raw_frames["frame_idx"], dtype=np.int64).max()) + 1,
    )
    instance_count = np.bincount(instances["frame"], minlength=n_frames)
    plausible_mask = instances["plausible"].to_numpy(dtype=bool)
    plausible_frames = instances.loc[plausible_mask, "frame"].to_numpy(dtype=np.int64)
    plausible_count = np.bincount(plausible_frames, minlength=n_frames)
    partial_mask = plausible_mask & instances["visible_node_count"].to_numpy(dtype=int).__lt__(2)
    partial_count = np.bincount(
        instances.loc[partial_mask, "frame"].to_numpy(dtype=np.int64),
        minlength=n_frames,
    )

    score_minimum = np.full(n_frames, np.nan, dtype=float)
    score_maximum = np.full(n_frames, np.nan, dtype=float)
    score_std = np.full(n_frames, np.nan, dtype=float)
    score = instances.loc[plausible_mask, "instance_score"].to_numpy(dtype=float)
    score_frame = plausible_frames
    finite_score = np.isfinite(score)
    if finite_score.any():
        finite_frame = score_frame[finite_score]
        finite_value = score[finite_score]
        score_minimum[:] = np.inf
        score_maximum[:] = -np.inf
        np.minimum.at(score_minimum, finite_frame, finite_value)
        np.maximum.at(score_maximum, finite_frame, finite_value)
        score_n = np.bincount(finite_frame, minlength=n_frames)
        score_sum = np.bincount(finite_frame, weights=finite_value, minlength=n_frames)
        score_sq = np.bincount(
            finite_frame, weights=finite_value**2, minlength=n_frames
        )
        valid_score = score_n > 0
        score_std[valid_score] = np.sqrt(
            np.maximum(
                0.0,
                score_sq[valid_score] / score_n[valid_score]
                - (score_sum[valid_score] / score_n[valid_score]) ** 2,
            )
        )
        score_minimum[~valid_score] = np.nan
        score_maximum[~valid_score] = np.nan

    maximum_instances = int(instances["instance_index"].max()) + 1 if len(instances) else 0
    mouth_grid_x = np.full((n_frames, maximum_instances), np.nan, dtype=float)
    mouth_grid_y = np.full_like(mouth_grid_x, np.nan)
    mouth_rows = instances.loc[
        plausible_mask
        & np.isfinite(instances["mouth_x"].to_numpy(dtype=float))
        & np.isfinite(instances["mouth_y"].to_numpy(dtype=float))
    ]
    mouth_grid_x[
        mouth_rows["frame"].to_numpy(dtype=np.int64),
        mouth_rows["instance_index"].to_numpy(dtype=np.int64),
    ] = mouth_rows["mouth_x"].to_numpy(dtype=float)
    mouth_grid_y[
        mouth_rows["frame"].to_numpy(dtype=np.int64),
        mouth_rows["instance_index"].to_numpy(dtype=np.int64),
    ] = mouth_rows["mouth_y"].to_numpy(dtype=float)
    minimum_mouth_distance = np.full(n_frames, np.inf, dtype=float)
    for first in range(maximum_instances):
        for second in range(first + 1, maximum_instances):
            distance = np.hypot(
                mouth_grid_x[:, first] - mouth_grid_x[:, second],
                mouth_grid_y[:, first] - mouth_grid_y[:, second],
            )
            # ``maximum_instances`` is global to the video. Frames with fewer
            # slots therefore contain NaNs for some later pairs. ``np.minimum``
            # would let the first absent pair poison an otherwise finite
            # distance; ``fmin`` keeps the finite value.
            minimum_mouth_distance = np.fmin(minimum_mouth_distance, distance)
    minimum_mouth_distance[~np.isfinite(minimum_mouth_distance)] = np.nan

    residual = instances["residual_px"].to_numpy(dtype=float)
    maximum_residual = np.full(n_frames, -np.inf, dtype=float)
    finite_residual = np.isfinite(residual)
    np.maximum.at(
        maximum_residual,
        instances.loc[finite_residual, "frame"].to_numpy(dtype=np.int64),
        residual[finite_residual],
    )
    maximum_residual[~np.isfinite(maximum_residual)] = np.nan
    outlier_mask = instances["is_isolated_outlier"].to_numpy(dtype=bool)
    outlier_count = np.bincount(
        instances.loc[outlier_mask, "frame"].to_numpy(dtype=np.int64),
        minlength=n_frames,
    )
    affected = np.full(n_frames, "", dtype=object)
    if outlier_mask.any():
        for frame, group in instances.loc[outlier_mask].groupby("frame", sort=False):
            affected[int(frame)] = ";".join(sorted(set(group["track_id"].astype(str))))

    frames = pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype=np.int64),
            "instance_count": instance_count.astype(np.int16),
            "plausible_count": plausible_count.astype(np.int16),
            "partial_instance_count": partial_count.astype(np.int16),
            "minimum_mouth_distance_px": minimum_mouth_distance,
            "minimum_score": score_minimum,
            "maximum_score": score_maximum,
            "score_std": score_std,
            "score_range": score_maximum - score_minimum,
            "maximum_xy_residual_px": maximum_residual,
            "outlier_count": outlier_count.astype(np.int16),
            "affected_track_ids": affected,
        }
    )
    meta = {
        "cache_version": FEATURE_CACHE_VERSION,
        "source_slp": _fingerprint(slp_path),
        "source_video": source_path,
        "node_names": node_names,
        "n_frames": n_frames,
        "n_instances": int(len(instances)),
        "feature_config": asdict(config),
        "extraction_backend": "validated_native_hdf5",
    }
    return frames, instances, meta


def _extract_view_features(
    slp_path: Path,
    *,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        return _extract_view_features_hdf5(slp_path, config=config)
    except _FastSlpPathUnavailable as error:
        print(f"[{slp_path.name}] native-array fast path unavailable ({error}); using object fallback", flush=True)
        frames, instances, meta = _extract_view_features_object(
            slp_path, config=config
        )
        meta["extraction_backend"] = "sleap_io_object_fallback"
        return frames, instances, meta


def load_or_build_view_features(
    slp_path: Path,
    *,
    cache_dir: Path,
    view: str,
    config: FeatureConfig,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load a provenance-validated parquet cache or stream one Round 7 SLP."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_path = cache_dir / f"{view}.frames.parquet"
    instances_path = cache_dir / f"{view}.instances.parquet"
    meta_path = cache_dir / f"{view}.meta.json"
    expected = {
        "cache_version": FEATURE_CACHE_VERSION,
        "source_slp": _fingerprint(slp_path),
        "feature_config": asdict(config),
    }
    if not force and frames_path.is_file() and instances_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(key) == value for key, value in expected.items()):
            return pd.read_parquet(frames_path), pd.read_parquet(instances_path), meta
    frames, instances, meta = _extract_view_features(slp_path, config=config)
    frames.to_parquet(frames_path, index=False)
    instances.to_parquet(instances_path, index=False)
    _write_json(meta_path, meta)
    return frames, instances, meta


def add_geometry_ambiguity(
    frames: pd.DataFrame,
    instances: pd.DataFrame,
    *,
    projected_geometry: np.ndarray,
    geometry_frames: np.ndarray,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Add association competition/margin and combined ambiguity signals."""
    table = frames.copy()
    geometry_frames = np.asarray(geometry_frames, dtype=np.int64)
    table_frames = table["frame"].to_numpy(dtype=np.int64)
    geometry_position = np.searchsorted(geometry_frames, table_frames)
    geometry_in_range = geometry_position < len(geometry_frames)
    geometry_match = np.zeros(len(table), dtype=bool)
    geometry_match[geometry_in_range] = (
        geometry_frames[geometry_position[geometry_in_range]]
        == table_frames[geometry_in_range]
    )
    geometry = np.full((len(table), 3, 2), np.nan, dtype=float)
    geometry[geometry_match] = np.asarray(projected_geometry, dtype=float)[
        geometry_position[geometry_match]
    ]
    geometry_valid = np.isfinite(geometry).all(axis=2)

    plausible = instances.loc[instances["plausible"].astype(bool)].copy()
    maximum_instances = (
        int(plausible["instance_index"].max()) + 1 if len(plausible) else 0
    )
    observed = np.full((len(table), maximum_instances, 2), np.nan, dtype=float)
    if len(plausible) and maximum_instances:
        detection_frames = plausible["frame"].to_numpy(dtype=np.int64)
        row_positions = np.searchsorted(table_frames, detection_frames)
        valid_rows = row_positions < len(table_frames)
        valid_rows[valid_rows] &= (
            table_frames[row_positions[valid_rows]] == detection_frames[valid_rows]
        )
        association = plausible[["assoc_x", "assoc_y"]].to_numpy(dtype=float)
        valid_rows &= np.isfinite(association).all(axis=1)
        observed[
            row_positions[valid_rows],
            plausible.loc[valid_rows, "instance_index"].to_numpy(dtype=np.int64),
        ] = association[valid_rows]

    competition = np.zeros(len(table), dtype=bool)
    small_margin = np.zeros(len(table), dtype=bool)
    if maximum_instances:
        distances = np.linalg.norm(
            geometry[:, :, None, :] - observed[:, None, :, :], axis=3
        )
        distances[~np.isfinite(distances)] = np.inf
        nearest_indices = np.argmin(distances, axis=2)
        nearest_distances = np.take_along_axis(
            distances, nearest_indices[:, :, None], axis=2
        )[:, :, 0]
        gated = geometry_valid & (nearest_distances <= config.association_gate_px)
        for first in range(3):
            for second in range(first + 1, 3):
                competition |= (
                    gated[:, first]
                    & gated[:, second]
                    & (nearest_indices[:, first] == nearest_indices[:, second])
                )
        if maximum_instances >= 2:
            nearest_two = np.partition(distances, 1, axis=2)[:, :, :2]
            first_distance = nearest_two[:, :, 0]
            second_distance = nearest_two[:, :, 1]
            valid_pair = np.isfinite(first_distance) & np.isfinite(second_distance)
            margins = np.full_like(first_distance, np.inf)
            margins[valid_pair] = second_distance[valid_pair] - first_distance[valid_pair]
            small_margin = np.any(
                geometry_valid
                & valid_pair
                & (first_distance <= config.association_gate_px)
                & (margins < config.association_margin_px),
                axis=1,
            )

    table["geometry_competition"] = competition
    table["small_association_margin"] = small_margin
    table["probable_fn"] = table["plausible_count"].astype(int) < 3
    table["probable_fp"] = table["plausible_count"].astype(int) > 3
    table["missing_node"] = table["partial_instance_count"].astype(int) > 0
    table["mouths_close"] = (
        pd.to_numeric(table["minimum_mouth_distance_px"], errors="coerce")
        < config.mouth_close_px
    ).fillna(False)
    table["low_confidence"] = (
        pd.to_numeric(table["minimum_score"], errors="coerce") < config.low_score
    ).fillna(False)
    table["dispersed_confidence"] = (
        pd.to_numeric(table["score_range"], errors="coerce") > config.score_dispersion
    ).fillna(False)
    signal_columns = [
        "probable_fn", "probable_fp", "missing_node", "geometry_competition",
        "small_association_margin", "mouths_close", "low_confidence",
        "dispersed_confidence",
    ]
    table["ambiguity_signal_count"] = table[signal_columns].fillna(False).sum(axis=1).astype(np.int8)
    table["ambiguity_signals"] = [
        ";".join(name for name in signal_columns if bool(row[name]))
        for row in table[signal_columns].to_dict("records")
    ]
    table["ambiguous"] = table["ambiguity_signal_count"].astype(int) > 0
    return table


def ingest_manual_evidence(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return one row per source frame with saved manual FP/FN view evidence."""
    evidence: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"fp": set(), "fn": set(), "episode": set()}
    )
    for row in ledger.to_dict("records"):
        episode = str(row.get("episode_id", ""))
        for item in validate_false_positive_detections(
            row.get("false_positive_detections_json", "")
        ):
            frame = int(item["frame"])
            evidence[frame]["fp"].add(str(item["view"]))
            if episode:
                evidence[frame]["episode"].add(episode)
        for item in validate_false_negative_detections(
            row.get("false_negative_detections_json", "")
        ):
            frame = int(item["frame"])
            evidence[frame]["fn"].add(str(item["view"]))
            if episode:
                evidence[frame]["episode"].add(episode)
    return pd.DataFrame(
        [
            {
                "frame": frame,
                "manual_fp_views": ";".join(view for view in VIEWS if view in value["fp"]),
                "manual_fn_views": ";".join(view for view in VIEWS if view in value["fn"]),
                "manual_episode_ids": ";".join(sorted(value["episode"])),
            }
            for frame, value in sorted(evidence.items())
        ],
        columns=["frame", "manual_fp_views", "manual_fn_views", "manual_episode_ids"],
    )


def merge_contact_events(
    candidates: pd.DataFrame,
    *,
    merge_gap_frames: int = 166,
) -> pd.DataFrame:
    """Merge nearby contact candidates into stable physical event groups."""
    required = {"contact_start_frame", "contact_end_frame", "episode_id"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"contact candidates are missing columns: {sorted(missing)}")
    ordered = candidates.copy()
    ordered["contact_start_frame"] = pd.to_numeric(ordered["contact_start_frame"]).astype(int)
    ordered["contact_end_frame"] = pd.to_numeric(ordered["contact_end_frame"]).astype(int)
    ordered = ordered.sort_values(["contact_start_frame", "contact_end_frame"], kind="stable")
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in ordered.to_dict("records"):
        start = int(row["contact_start_frame"])
        end = int(row["contact_end_frame"])
        episode = str(row["episode_id"])
        if current is None or start > int(current["event_end_frame"]) + int(merge_gap_frames):
            if current is not None:
                events.append(current)
            current = {
                "event_start_frame": start,
                "event_end_frame": end,
                "episode_ids": [episode],
            }
        else:
            current["event_end_frame"] = max(int(current["event_end_frame"]), end)
            current["episode_ids"].append(episode)
    if current is not None:
        events.append(current)
    for event in events:
        start = int(event["event_start_frame"])
        end = int(event["event_end_frame"])
        digest = hashlib.sha1(f"{start}:{end}".encode("utf-8")).hexdigest()[:10]
        event["event_group_id"] = f"contact_{start}_{end}_{digest}"
        event["contact_episode_ids"] = ";".join(sorted(set(event.pop("episode_ids"))))
    return pd.DataFrame(
        events,
        columns=[
            "event_group_id", "event_start_frame", "event_end_frame",
            "contact_episode_ids",
        ],
    )


def assign_event_groups(
    frames: Sequence[int],
    contact_events: pd.DataFrame,
    *,
    context_frames: int = 166,
    orphan_gap_frames: int = 20,
    sentinel_start: int = SENTINEL_START,
    sentinel_end: int = SENTINEL_END,
) -> pd.DataFrame:
    """Map candidates to physical events and cluster non-contact orphan frames."""
    unique_frames = sorted({int(frame) for frame in frames})
    events = contact_events.sort_values("event_start_frame", kind="stable").to_dict("records")
    assigned: dict[int, dict[str, Any]] = {}
    assignment_distance: dict[int, tuple[int, int]] = {}
    frame_set = set(unique_frames)
    # Contact events are short relative to a full recording. Iterating each
    # event's local context range is linear in covered context, not
    # frames-by-events.
    for event in events:
        start = int(event["event_start_frame"])
        end = int(event["event_end_frame"])
        for frame in range(start - context_frames, end + context_frames + 1):
            if frame not in frame_set or sentinel_start <= frame <= sentinel_end:
                continue
            distance = 0 if start <= frame <= end else min(abs(frame - start), abs(frame - end))
            key = (distance, start)
            if frame not in assignment_distance or key < assignment_distance[frame]:
                assignment_distance[frame] = key
                assigned[frame] = dict(event)

    sentinel_episode_ids = ";".join(
        sorted(
            {
                episode
                for event in events
                if int(event["event_start_frame"]) <= sentinel_end
                and int(event["event_end_frame"]) >= sentinel_start
                for episode in str(event["contact_episode_ids"]).split(";")
                if episode
            }
        )
    )
    sentinel_value = {
        "event_group_id": f"holdout_sentinel_{sentinel_start}_{sentinel_end}",
        "event_start_frame": sentinel_start,
        "event_end_frame": sentinel_end,
        "contact_episode_ids": sentinel_episode_ids,
    }
    for frame in unique_frames:
        if sentinel_start <= frame <= sentinel_end:
            assigned[frame] = sentinel_value

    orphan_frames = [frame for frame in unique_frames if frame not in assigned]
    clusters: list[list[int]] = []
    for frame in orphan_frames:
        if not clusters or frame - clusters[-1][-1] > orphan_gap_frames:
            clusters.append([frame])
        else:
            clusters[-1].append(frame)
    for cluster in clusters:
        start, end = cluster[0], cluster[-1]
        digest = hashlib.sha1(f"orphan:{start}:{end}".encode("utf-8")).hexdigest()[:10]
        value = {
            "event_group_id": f"orphan_{start}_{end}_{digest}",
            "event_start_frame": start,
            "event_end_frame": end,
            "contact_episode_ids": "",
        }
        for frame in cluster:
            assigned[frame] = value
    return pd.DataFrame(
        [{"frame": frame, **assigned[frame]} for frame in unique_frames]
    )


def _split_join(values: Iterable[str]) -> str:
    return ";".join(sorted({part for value in values for part in str(value).split(";") if part}))


def aggregate_synchronized_features(
    per_view: Mapping[str, pd.DataFrame],
    *,
    tracking: Any,
    contact_candidates: pd.DataFrame,
    manual_evidence: pd.DataFrame,
    config: SelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine view tables, 3D IFD, manual evidence, and contact events."""
    frame_min = max(int(table["frame"].min()) for table in per_view.values())
    frame_max = min(int(table["frame"].max()) for table in per_view.values())
    aggregate = pd.DataFrame({"frame": np.arange(frame_min, frame_max + 1, dtype=np.int64)})
    for view in VIEWS:
        table = per_view[view].copy()
        rename = {column: f"{view}_{column}" for column in table.columns if column != "frame"}
        aggregate = aggregate.merge(table.rename(columns=rename), on="frame", how="left", validate="one_to_one")
    aggregate = count_ambiguous_views(
        aggregate.rename(
            columns={f"{view}_ambiguous": f"{view}_ambiguous" for view in VIEWS}
        )
    )
    track_position = aggregate["frame"].to_numpy(dtype=np.int64) - int(tracking.frame_min)
    in_range = (track_position >= 0) & (track_position < len(tracking.frames))
    minimum_ifd = np.full(len(aggregate), np.nan, dtype=float)
    minimum_ifd[in_range] = tracking.minimum_direct[track_position[in_range]]
    aggregate["minimum_ifd_mm"] = minimum_ifd

    aggregate = aggregate.merge(manual_evidence, on="frame", how="left")
    for column in ("manual_fp_views", "manual_fn_views", "manual_episode_ids"):
        aggregate[column] = aggregate[column].fillna("").astype(str)

    events = merge_contact_events(
        contact_candidates, merge_gap_frames=config.event_merge_gap_frames
    )
    event_assignment = assign_event_groups(
        aggregate["frame"],
        events,
        context_frames=config.contact_context_frames,
        sentinel_start=config.sentinel_start,
        sentinel_end=config.sentinel_end,
    )
    aggregate = aggregate.merge(event_assignment, on="frame", how="left", validate="one_to_one")
    aggregate["contact_episode_ids"] = [
        _split_join((event_ids, manual_ids))
        for event_ids, manual_ids in zip(
            aggregate["contact_episode_ids"], aggregate["manual_episode_ids"], strict=True
        )
    ]
    aggregate["inside_contact_interval"] = (
        aggregate["frame"].between(
            aggregate["event_start_frame"], aggregate["event_end_frame"]
        )
        & aggregate["event_group_id"].astype(str).str.startswith("contact_")
    ) | aggregate["event_group_id"].astype(str).str.startswith("holdout_sentinel_")
    aggregate["near_contact_context"] = aggregate["event_group_id"].astype(str).str.startswith(
        "contact_"
    )
    aggregate["direct_close"] = aggregate["minimum_ifd_mm"].le(config.close_ifd_mm).fillna(False)
    aggregate["close_contact_context"] = (
        aggregate["direct_close"]
        | aggregate["inside_contact_interval"]
        | aggregate["near_contact_context"]
    )
    aggregate["maximum_xy_residual_px"] = aggregate[
        [f"{view}_maximum_xy_residual_px" for view in VIEWS]
    ].max(axis=1, skipna=True)
    aggregate["outlier_view_count"] = aggregate[
        [f"{view}_outlier_count" for view in VIEWS]
    ].fillna(0).gt(0).sum(axis=1).astype(np.int8)
    aggregate["affected_ids"] = [
        _split_join(values)
        for values in aggregate[
            [f"{view}_affected_track_ids" for view in VIEWS]
        ].fillna("").itertuples(index=False, name=None)
    ]
    aggregate["probable_fn_views"] = [
        ";".join(
            view for view in VIEWS if bool(aggregate.at[index, f"{view}_probable_fn"])
        )
        for index in aggregate.index
    ]
    aggregate["ambiguity_signals"] = [
        ";".join(
            f"{view}:{signal}"
            for view in VIEWS
            for signal in str(aggregate.at[index, f"{view}_ambiguity_signals"]).split(";")
            if signal
        )
        for index in aggregate.index
    ]
    return aggregate, events


def assign_exclusive_categories(
    features: pd.DataFrame,
    *,
    close_ifd_mm: float = 16.0,
) -> pd.DataFrame:
    """Assign at most one requested Round 8 category and a transparent score."""
    table = features.copy()
    categories: list[str] = []
    scores: list[float] = []
    reasons: list[str] = []
    for row in table.to_dict("records"):
        manual = bool(row.get("manual_fp_views") or row.get("manual_fn_views"))
        mined_fn = bool(row.get("probable_fn_views"))
        ambiguous_count = int(row.get("ambiguous_view_count", 0))
        close_context = bool(row.get("close_contact_context", False))
        direct_close = bool(row.get("direct_close", False))
        outlier_views = int(row.get("outlier_view_count", 0))
        residual = float(row.get("maximum_xy_residual_px", np.nan))
        ifd = float(row.get("minimum_ifd_mm", np.nan))
        ambiguity_signal_count = sum(
            int(row.get(f"{view}_ambiguity_signal_count", 0) or 0) for view in VIEWS
        )
        category = ""
        score = float("nan")
        reason = ""
        if manual:
            category = "manual_fp_or_fn"
            score = 1_000_000.0 + 1000.0 * len(
                set((str(row.get("manual_fp_views", "")) + ";" + str(row.get("manual_fn_views", ""))).strip(";").split(";"))
            )
            reason = "saved manual FP/FN evidence; retain real fish and verify deletion/addition"
        elif close_context and ambiguous_count >= 2:
            category = "close_multi_view_ambiguity"
            score = 10_000.0 * ambiguous_count + 100.0 * ambiguity_signal_count
            if np.isfinite(ifd):
                score += max(0.0, close_ifd_mm - ifd)
            reason = f"close/contact-context frame ambiguous in {ambiguous_count} views"
        elif mined_fn:
            category = "manual_fp_or_fn"
            missing_views = len(str(row.get("probable_fn_views", "")).split(";"))
            score = 10_000.0 + 1000.0 * missing_views + 10.0 * ambiguity_signal_count
            reason = f"mined probable FN in {row.get('probable_fn_views')}"
        elif outlier_views > 0:
            category = "isolated_xy_outlier"
            score = (residual if np.isfinite(residual) else 0.0) + 1000.0 * outlier_views
            score += 10.0 * len(str(row.get("affected_ids", "")).split(";"))
            reason = f"isolated exact-consecutive midpoint residual in {outlier_views} views"
        elif close_context and ambiguous_count == 1:
            category = "close_single_view_ambiguity"
            score = 1000.0 + 100.0 * ambiguity_signal_count
            if np.isfinite(ifd):
                score += max(0.0, close_ifd_mm - ifd)
            reason = "close/contact-context frame ambiguous in exactly one view"
        elif direct_close and ambiguous_count == 0:
            valid_counts = all(int(row.get(f"{view}_plausible_count", 0)) == 3 for view in VIEWS)
            no_outlier = outlier_views == 0 and (not np.isfinite(residual) or residual < 15.0)
            if valid_counts and no_outlier:
                category = "close_clean_control"
                score = max(0.0, close_ifd_mm - ifd) if np.isfinite(ifd) else 0.0
                reason = "direct-close clean control with three plausible fish in every view"
        categories.append(category)
        scores.append(score)
        reasons.append(reason)
    table["category"] = categories
    table["category_score"] = scores
    table["selection_reason"] = reasons
    return table


def existing_label_index(
    label_paths: Sequence[Path],
    *,
    experiment: str,
) -> set[tuple[str, int]]:
    """Return ``(view, frame)`` keys already present in cumulative labels."""
    try:
        import sleap_io as sio
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("sleap-io is required to inspect cumulative labels") from error
    result: set[tuple[str, int]] = set()
    for path in label_paths:
        labels = sio.load_slp(str(path))
        for lf in labels.labeled_frames:
            source = _video_source_path(lf.video)
            if experiment_from_video(source) != experiment:
                continue
            view = classify_view(source)
            if view is not None:
                result.add((view, int(lf.frame_idx)))
    return result


def add_existing_label_overlap(
    features: pd.DataFrame,
    label_index: set[tuple[str, int]],
) -> pd.DataFrame:
    table = features.copy()
    table["existing_label_overlap"] = [
        ";".join(view for view in VIEWS if (view, int(frame)) in label_index)
        for frame in table["frame"]
    ]
    return table


def _deterministic_jitter(frame: int, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{int(frame)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


def _diversity_order(
    candidates: pd.DataFrame,
    *,
    seed: int,
    time_bins: int,
) -> list[int]:
    """Return candidate indices in score-first temporal round-robin order."""
    if candidates.empty:
        return []
    table = candidates.copy()
    frame_min = int(table["frame"].min())
    frame_max = int(table["frame"].max())
    span = max(1, frame_max - frame_min + 1)
    table["_bin"] = np.minimum(
        int(time_bins) - 1,
        ((table["frame"].astype(int) - frame_min) * int(time_bins) // span).astype(int),
    )
    table["_jitter"] = [
        _deterministic_jitter(int(frame), int(seed)) for frame in table["frame"]
    ]
    queues: dict[int, list[int]] = {}
    for bin_index, group in table.groupby("_bin", sort=True):
        queues[int(bin_index)] = group.sort_values(
            ["category_score", "_jitter", "frame"],
            ascending=[False, False, True],
            kind="stable",
        ).index.tolist()
    order: list[int] = []
    bins = sorted(queues)
    while any(queues[bin_index] for bin_index in bins):
        for bin_index in bins:
            if queues[bin_index]:
                order.append(queues[bin_index].pop(0))
    return order


def _spacing_ok(
    row: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    min_spacing_frames: int,
    dense_event_ids: set[str],
) -> bool:
    frame = int(row["frame"])
    event = str(row.get("event_group_id", ""))
    for prior in selected:
        prior_frame = int(prior["frame"])
        if abs(frame - prior_frame) >= int(min_spacing_frames):
            continue
        prior_event = str(prior.get("event_group_id", ""))
        if event and event == prior_event and event in dense_event_ids:
            continue
        return False
    return True


def _pick_rows(
    pool: pd.DataFrame,
    count: int,
    *,
    seed: int,
    time_bins: int,
    min_spacing_frames: int,
    selected: list[dict[str, Any]],
    dense_event_ids: set[str],
    forced_frames: set[int] | None = None,
    category_caps: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select rows deterministically, honoring force/caps and temporal spacing."""
    if count <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    already = {int(row["frame"]) for row in selected}
    category_counts = Counter(str(row["category"]) for row in selected)
    forced_frames = {int(frame) for frame in (forced_frames or set())}

    def accept(index: int, *, forced: bool = False) -> None:
        if len(chosen) >= count:
            return
        row = pool.loc[index].to_dict()
        frame = int(row["frame"])
        if frame in already:
            return
        category = str(row["category"])
        if category_caps is not None and category_counts[category] >= int(category_caps.get(category, 0)):
            return
        if not forced and not _spacing_ok(
            row,
            [*selected, *chosen],
            min_spacing_frames=min_spacing_frames,
            dense_event_ids=dense_event_ids,
        ):
            return
        chosen.append(row)
        already.add(frame)
        category_counts[category] += 1

    forced_pool = pool.loc[pool["frame"].astype(int).isin(forced_frames)]
    for index in _diversity_order(forced_pool, seed=seed, time_bins=time_bins):
        accept(index, forced=True)
    for index in _diversity_order(pool, seed=seed, time_bins=time_bins):
        accept(index)
        if len(chosen) >= count:
            break
    # Quotas are hard requirements. If ordinary diversity spacing exhausts a
    # category, fill deterministically at 1-frame spacing and report this in
    # the selection report.
    if len(chosen) < count:
        for index in _diversity_order(pool, seed=seed + 1, time_bins=time_bins):
            accept(index, forced=True)
            if len(chosen) >= count:
                break
    return chosen


def select_with_quotas_and_locked_holdout(
    candidates: pd.DataFrame,
    *,
    quotas: Mapping[str, int] = CATEGORY_QUOTAS,
    config: SelectionConfig = SelectionConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select 360 rows, reserving exactly 60 sentinel frames for holdout."""
    pool = candidates.loc[candidates["category"].astype(str).ne("")].copy()
    pool = pool.drop_duplicates("frame", keep="first")
    sentinel_id = f"holdout_sentinel_{config.sentinel_start}_{config.sentinel_end}"
    sentinel_pool = pool.loc[
        pool["frame"].between(config.sentinel_start, config.sentinel_end)
    ]
    if len(sentinel_pool) < config.holdout_frames:
        raise ValueError(
            f"sentinel range has only {len(sentinel_pool)} categorized candidates; "
            f"need {config.holdout_frames}"
        )
    selected: list[dict[str, Any]] = _pick_rows(
        sentinel_pool,
        config.holdout_frames,
        seed=config.random_seed,
        time_bins=min(20, config.time_bins),
        min_spacing_frames=1,
        selected=[],
        dense_event_ids={sentinel_id},
        category_caps=quotas,
    )
    if len(selected) != config.holdout_frames:
        raise ValueError("could not fill the locked holdout within category quotas")
    for row in selected:
        row["split"] = "holdout"

    selected_counts = Counter(str(row["category"]) for row in selected)
    train_pool = pool.loc[
        ~pool["frame"].between(
            config.sentinel_start - config.guard_band_frames,
            config.sentinel_end + config.guard_band_frames,
        )
    ].copy()
    forced_manual_frames = set(
        train_pool.loc[
            train_pool["manual_fp_views"].astype(str).ne("")
            | train_pool["manual_fn_views"].astype(str).ne(""),
            "frame",
        ].astype(int)
    )
    requested_remaining = {
        category: max(0, int(target) - selected_counts[category])
        for category, target in quotas.items()
    }
    spacing_relaxed_categories: list[str] = []
    initial_actual: dict[str, int] = {}
    for category, needed in requested_remaining.items():
        category_pool = train_pool.loc[train_pool["category"].eq(category)]
        before = len(selected)
        picked = _pick_rows(
            category_pool,
            needed,
            seed=config.random_seed + list(quotas).index(category) + 10,
            time_bins=config.time_bins,
            min_spacing_frames=config.min_spacing_frames,
            selected=selected,
            dense_event_ids=set(),
            forced_frames=forced_manual_frames if category == "manual_fp_or_fn" else set(),
        )
        if len(picked) < needed:
            spacing_relaxed_categories.append(category)
        for row in picked:
            row["split"] = "train"
        selected.extend(picked)
        initial_actual[category] = len(selected) - before

    redistribution: list[dict[str, Any]] = []
    category_actual = Counter(str(row["category"]) for row in selected)
    deficits = {
        category: max(0, int(target) - category_actual[category])
        for category, target in quotas.items()
    }
    deficit_total = sum(deficits.values())
    if deficit_total:
        unselected_pool = train_pool.loc[
            ~train_pool["frame"].astype(int).isin({int(row["frame"]) for row in selected})
        ].copy()
        recipients = [
            "close_multi_view_ambiguity",
            "isolated_xy_outlier",
            "manual_fp_or_fn",
            "close_single_view_ambiguity",
            "close_clean_control",
        ]
        for source_category, deficit in deficits.items():
            remaining = int(deficit)
            for recipient in recipients:
                if remaining <= 0:
                    break
                recipient_pool = unselected_pool.loc[unselected_pool["category"].eq(recipient)]
                picked = _pick_rows(
                    recipient_pool,
                    remaining,
                    seed=config.random_seed + 100 + recipients.index(recipient),
                    time_bins=config.time_bins,
                    min_spacing_frames=config.min_spacing_frames,
                    selected=selected,
                    dense_event_ids=set(),
                )
                for row in picked:
                    row["split"] = "train"
                if picked:
                    selected.extend(picked)
                    used_frames = {int(row["frame"]) for row in picked}
                    unselected_pool = unselected_pool.loc[
                        ~unselected_pool["frame"].astype(int).isin(used_frames)
                    ]
                    redistribution.append(
                        {
                            "from_category": source_category,
                            "to_category": recipient,
                            "count": len(picked),
                        }
                    )
                    remaining -= len(picked)
            if remaining:
                raise ValueError(
                    f"unable to redistribute {remaining} missing {source_category} frames"
                )

    result = pd.DataFrame(selected).drop_duplicates("frame", keep="first")
    holdout = result.loc[result["split"].eq("holdout")]
    train = result.loc[result["split"].eq("train")]
    if len(result) != sum(quotas.values()) or len(train) != config.train_frames or len(holdout) != config.holdout_frames:
        raise ValueError(
            f"selection count mismatch: total={len(result)}, train={len(train)}, holdout={len(holdout)}"
        )
    manual_source_frames = set(
        pool.loc[pool["manual_fp_views"].astype(str).ne(""), "frame"].astype(int)
    )
    missing_manual = sorted(manual_source_frames - set(result["frame"].astype(int)))
    if missing_manual:
        raise ValueError(f"manual FP source frames were not selected: {missing_manual}")
    report = {
        "requested_quotas": {key: int(value) for key, value in quotas.items()},
        "actual_category_counts": {
            key: int(value) for key, value in result["category"].value_counts().to_dict().items()
        },
        "initial_train_picks": initial_actual,
        "redistribution": redistribution,
        "spacing_relaxed_categories": spacing_relaxed_categories,
        "manual_fp_source_frames": sorted(manual_source_frames),
        "manual_fp_missing_from_selection": missing_manual,
    }
    return result.sort_values("frame", kind="stable").reset_index(drop=True), report


def validate_event_split(
    selection: pd.DataFrame,
    *,
    guard_band_frames: int = 166,
) -> dict[str, Any]:
    """Validate counts, event exclusivity, and train/holdout guard bands."""
    train = selection.loc[selection["split"].eq("train")]
    holdout = selection.loc[selection["split"].eq("holdout")]
    train_events = set(train["event_group_id"].astype(str))
    holdout_events = set(holdout["event_group_id"].astype(str))
    shared_events = sorted(train_events & holdout_events)
    violations: list[dict[str, Any]] = []
    for event_id, group in holdout.groupby("event_group_id", sort=True):
        event_start = int(group["event_start_frame"].min())
        event_end = int(group["event_end_frame"].max())
        bad = train.loc[
            train["frame"].between(
                event_start - int(guard_band_frames),
                event_end + int(guard_band_frames),
            )
        ]
        for frame in bad["frame"].astype(int):
            violations.append(
                {"holdout_event": str(event_id), "train_frame": int(frame)}
            )
    result = {
        "train_count": int(len(train)),
        "holdout_count": int(len(holdout)),
        "shared_event_groups": shared_events,
        "guard_band_frames": int(guard_band_frames),
        "guard_band_violations": violations,
        "passed": not shared_events and not violations,
    }
    if not result["passed"]:
        raise ValueError(f"train/holdout leakage detected: {result}")
    return result


def assign_annotation_batches(
    selection: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign exactly 100 train + 20 holdout to pilot; remainder expansion."""
    table = selection.copy()
    table["annotation_batch"] = "expansion"
    for split, count in (("train", 100), ("holdout", 20)):
        subset = table.loc[table["split"].eq(split)]
        order = _diversity_order(subset, seed=seed + (0 if split == "train" else 1), time_bins=20)
        pilot_indices = order[:count]
        if len(pilot_indices) != count:
            raise ValueError(f"cannot assign {count} {split} frames to pilot")
        table.loc[pilot_indices, "annotation_batch"] = "pilot"
    counts = table.groupby(["annotation_batch", "split"]).size().to_dict()
    expected = {("pilot", "train"): 100, ("pilot", "holdout"): 20, ("expansion", "train"): 200, ("expansion", "holdout"): 40}
    if counts != expected:
        raise ValueError(f"annotation batch counts mismatch: {counts}")
    return table


def _prediction_to_user_instance(inst: Any, source_node_names: Sequence[str], skeleton: Any) -> Any | None:
    import sleap_io as sio

    points = np.asarray(inst.numpy(), dtype=float)
    converted = np.full((2, 3), np.nan, dtype=float)
    converted[:, 2] = 0.0
    finite_count = 0
    for target_index, node_name in enumerate(("mouth", "tail")):
        if node_name not in source_node_names:
            continue
        source_index = source_node_names.index(node_name)
        if source_index >= len(points) or not np.isfinite(points[source_index]).all():
            continue
        converted[target_index, :2] = points[source_index]
        converted[target_index, 2] = 1.0
        finite_count += 1
    if finite_count == 0:
        return None
    return sio.Instance.from_numpy(points_data=converted, skeleton=skeleton)


def validate_annotation_seed(
    labels_or_path: Any,
    manifest: pd.DataFrame,
    *,
    experiment: str,
) -> dict[str, Any]:
    """Validate exact synchronized LF coverage, node order, partials, and empties."""
    import sleap_io as sio

    labels = sio.load_slp(str(labels_or_path)) if isinstance(labels_or_path, (str, Path)) else labels_or_path
    if len(labels.skeletons) != 1:
        raise ValueError("Round 8 seed must contain exactly one skeleton")
    node_names = [node.name for node in labels.skeletons[0].nodes]
    if node_names != ["mouth", "tail"]:
        raise ValueError(f"seed node order must be ['mouth', 'tail'], got {node_names}")
    expected_frames = [int(frame) for frame in manifest["frame"]]
    expected = {(experiment, frame, view) for frame in expected_frames for view in VIEWS}
    expected_suggestion_order = [
        (experiment, frame, view) for frame in expected_frames for view in VIEWS
    ]
    observed: dict[tuple[str, int, str], Any] = {}
    partial_instances = 0
    instance_count = 0
    empty_lfs = 0
    for lf in labels.labeled_frames:
        source = _video_source_path(lf.video)
        view = classify_view(source)
        exp = experiment_from_video(source)
        if view is None or exp is None:
            raise ValueError(f"cannot recover experiment/view from seed video: {source}")
        key = (exp, int(lf.frame_idx), view)
        if key in observed:
            raise ValueError(f"duplicate labeled frame in seed: {key}")
        observed[key] = lf
        if not lf.instances:
            empty_lfs += 1
        for inst in lf.instances:
            instance_count += 1
            finite = np.isfinite(np.asarray(inst.numpy(), dtype=float)).all(axis=1)
            if not finite.any():
                raise ValueError(f"seed contains an all-missing instance at {key}")
            if int(finite.sum()) == 1:
                partial_instances += 1
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"seed coverage mismatch; missing={missing[:10]}, extra={extra[:10]}")
    if len(labels.suggestions) != len(expected):
        raise ValueError(
            f"seed needs one suggestion per view image; got {len(labels.suggestions)} for {len(expected)}"
        )
    suggestion_order = [
        (
            experiment_from_video(_video_source_path(suggestion.video)),
            int(suggestion.frame_idx),
            classify_view(_video_source_path(suggestion.video)),
        )
        for suggestion in labels.suggestions
    ]
    if suggestion_order != expected_suggestion_order:
        raise ValueError("seed suggestions do not preserve manifest/frame/view order")
    return {
        "source_frame_count": len(expected_frames),
        "labeled_frame_count": len(observed),
        "view_image_count": len(expected),
        "empty_labeled_frames": empty_lfs,
        "instance_count": instance_count,
        "complete_instances": instance_count - partial_instances,
        "partial_instances": partial_instances,
        "node_names": node_names,
        "suggestion_order_matches_manifest": True,
    }


def build_new_only_annotation_seed(
    manifest: pd.DataFrame,
    *,
    experiment: str,
    slp_paths: Mapping[str, Path],
    output_path: Path,
    selection_manifest_path: Path,
    raw_data_root: Path | None = None,
) -> dict[str, Any]:
    """Build an embedded seed containing only selected synchronized frames."""
    import sleap_io as sio

    skeleton = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])
    loaded: dict[str, Any] = {}
    frame_indices: dict[str, dict[int, Any]] = {}
    source_nodes: dict[str, list[str]] = {}
    videos: dict[str, Any] = {}
    raw_video_paths: dict[str, Path] = {}
    selected_frames = [int(frame) for frame in manifest["frame"]]
    for view in VIEWS:
        labels = sio.load_slp(str(slp_paths[view]), open_videos=False, lazy=True)
        if not labels.videos or not labels.skeletons:
            raise ValueError(f"{slp_paths[view]} is missing video/skeleton metadata")
        loaded[view] = labels
        stored_video = labels.videos[0]
        stored_source = Path(_video_source_path(stored_video))
        source_candidates = [stored_source]
        if raw_data_root is not None:
            source_candidates.append(
                Path(raw_data_root) / experiment / view / stored_source.name
            )
        raw_video = next((path for path in source_candidates if path.is_file()), None)
        if raw_video is None:
            raise FileNotFoundError(
                f"cannot resolve raw video for {view}; checked "
                f"{[str(path) for path in source_candidates]}"
            )
        raw_video_paths[view] = raw_video
        videos[view] = sio.Video.from_filename(str(raw_video))
        selected_lfs = labels.find(
            stored_video, frame_idx=selected_frames, return_new=False
        )
        frame_indices[view] = {int(lf.frame_idx): lf for lf in selected_lfs}
        source_nodes[view] = [node.name for node in labels.skeletons[0].nodes]

    labeled_frames: list[Any] = []
    suggestions: list[Any] = []
    conversion_counts = Counter()
    selection_records: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        frame = int(row["frame"])
        selection_records.append(
            {
                key: row.get(key, "")
                for key in (
                    "frame", "category", "event_group_id", "split",
                    "annotation_batch", "selection_reason",
                )
            }
        )
        for view in VIEWS:
            source_lf = frame_indices[view].get(frame)
            user_instances: list[Any] = []
            if source_lf is not None:
                for inst in source_lf.instances:
                    converted = _prediction_to_user_instance(
                        inst, source_nodes[view], skeleton
                    )
                    if converted is None:
                        conversion_counts["all_missing_predictions_skipped"] += 1
                        continue
                    finite = np.isfinite(converted.numpy()).all(axis=1)
                    conversion_counts[
                        "partial_instances" if int(finite.sum()) == 1 else "complete_instances"
                    ] += 1
                    user_instances.append(converted)
            if not user_instances:
                conversion_counts["empty_labeled_frames"] += 1
            lf = sio.LabeledFrame(
                video=videos[view], frame_idx=frame, instances=user_instances
            )
            labeled_frames.append(lf)
            suggestions.append(sio.SuggestionFrame(video=videos[view], frame_idx=frame))

    for labels in loaded.values():
        labels.close()

    provenance = {
        "workflow": "round8_active_learning_new_only_seed",
        "generated_at_utc": utc_now(),
        "experiment": experiment,
        "selection_manifest": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": hashlib.sha256(
            selection_manifest_path.read_bytes()
        ).hexdigest(),
        "source_slps": {view: _fingerprint(path) for view, path in slp_paths.items()},
        "raw_videos": {view: _fingerprint(path) for view, path in raw_video_paths.items()},
        "selection": selection_records,
        "instructions": [
            "label all three real fish in every view",
            "correct mouth and tail",
            "delete FP/reflection instances",
            "add missing/FN instances",
            "use all three views even if only two are ambiguous",
            "save a fully embedded edits_full.pkg.slp",
        ],
    }
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=[videos[view] for view in VIEWS],
        labeled_frames=labeled_frames,
        suggestions=suggestions,
        provenance=provenance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(output_path), embed="all")
    validation = validate_annotation_seed(
        output_path, manifest, experiment=experiment
    )
    return {
        "path": str(output_path.resolve()),
        "size_bytes": int(output_path.stat().st_size),
        "conversion_counts": dict(conversion_counts),
        "validation": validation,
    }


def _selection_manifest(selection: pd.DataFrame, *, experiment: str) -> pd.DataFrame:
    table = selection.copy()
    table["experiment"] = experiment
    for view in VIEWS:
        for suffix in ("instance_count", "plausible_count"):
            column = f"{view}_{suffix}"
            values = table[column] if column in table else pd.Series(0, index=table.index)
            table[column] = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    for column in SELECTION_COLUMNS:
        if column not in table:
            table[column] = ""
    table["frame"] = pd.to_numeric(table["frame"]).astype(int)
    table["category_score"] = pd.to_numeric(table["category_score"], errors="coerce")
    return table[SELECTION_COLUMNS].sort_values("frame", kind="stable").reset_index(drop=True)


def _resolve_required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is not configured")
    return Path(value)


def _write_runbook(
    path: Path,
    *,
    experiment: str,
    paths_config: Path,
    report_dir: Path,
    seed_dir: Path,
    pilot_seed: Path,
    expansion_seed: Path,
) -> None:
    pilot_out = seed_dir / "pilot_120_edits.pkg.slp"
    expansion_out = seed_dir / "expansion_240_edits.pkg.slp"
    text = f"""# Round 8 active-learning runbook

Experiment: `{experiment}`  
Selection seed: `{RANDOM_SEED}`  
Unit of selection: synchronized source frame (top + side1 + side2).

## Cohort

- `selection_360.csv`: authoritative 300-train / 60-locked-holdout manifest.
- `pilot_120.csv`: 100 train + 20 holdout.
- `expansion_240.csv`: 200 train + 40 holdout.
- Holdout sentinel: source frames {SENTINEL_START:,}--{SENTINEL_END:,}.
- No training frame may lie within +/-166 source frames of a holdout event.

## Annotate the pilot

```powershell
.venv\\Scripts\\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in "{pilot_seed}" `
  --out "{pilot_out}" `
  --paths-config "{paths_config}" `
  --calibration-backend auto `
  --min-views 1
```

For every source frame: label all three real fish in all three views; correct
mouth and tail; delete reflection/FP instances; add missing/FN instances; and
do not skip the clean third view when only two are ambiguous. Use
Ctrl+Shift+S and return:

`{full_save_path(pilot_out)}`

## Build and annotate the expansion after pilot review

```powershell
.venv\\Scripts\\python.exe -m scripts.sleap.mine_round8_hard_frames `
  --paths-config "{paths_config}" `
  --output-dir "{report_dir}" `
  --seed-output-dir "{seed_dir}" `
  --reuse-selection --build-seed expansion

.venv\\Scripts\\python.exe -m scripts.sleap.pyqt_multiview_annotator `
  --in "{expansion_seed}" `
  --out "{expansion_out}" `
  --paths-config "{paths_config}" `
  --calibration-backend auto `
  --min-views 1
```

Return `{full_save_path(expansion_out)}`.

## Finalize corrected labels without frame-level validation leakage

Pilot only:

```powershell
.venv\\Scripts\\python.exe -m scripts.sleap.finalize_round8_labels `
  --corrected "{full_save_path(pilot_out)}" `
  --manifest "{report_dir / 'selection_360.csv'}" `
  --annotation-batch pilot `
  --paths-config "{paths_config}"
```

After expansion, pass both corrected packages and `--annotation-batch all`.
The finalizer writes explicit train/validation packages and configs; it never
changes Round 7 cumulative labels and never starts training.
"""
    path.write_text(text, encoding="utf-8")


def _build_requested_seeds(
    build_seed: str,
    *,
    selection: pd.DataFrame,
    experiment: str,
    slp_paths: Mapping[str, Path],
    report_dir: Path,
    seed_dir: Path,
    raw_data_root: Path | None,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    batches = (
        ("pilot", "pilot_120.csv", "pilot_120_seed.pkg.slp"),
        ("expansion", "expansion_240.csv", "expansion_240_seed.pkg.slp"),
    )
    for batch, manifest_name, seed_name in batches:
        if build_seed not in {batch, "both"}:
            continue
        manifest = selection.loc[selection["annotation_batch"].eq(batch)].copy()
        manifest_path = report_dir / manifest_name
        outputs[batch] = build_new_only_annotation_seed(
            manifest,
            experiment=experiment,
            slp_paths=slp_paths,
            output_path=seed_dir / seed_name,
            selection_manifest_path=manifest_path,
            raw_data_root=raw_data_root,
        )
    return outputs


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    local = load_local_paths(args.paths_config)
    if local is None:
        raise ValueError(f"paths config not found: {args.paths_config}")
    experiment = str(args.experiment)
    report_dir = Path(args.output_dir or (REPO_ROOT / "report" / "20260721" / "round8_active_learning"))
    slp_dir = Path(
        args.slp_dir
        or (_resolve_required(local.sleap_output_dir, "sleap_output_dir") / experiment / args.sleap_subdir)
    )
    models_dir = _resolve_required(local.sleap_models_dir, "sleap_models_dir")
    top_labels = Path(args.round7_top_labels or (models_dir / ROUND7_MODEL_DIR / "top" / "labels.pkg.slp"))
    side_labels = Path(args.round7_side_labels or (models_dir / ROUND7_MODEL_DIR / "side" / "labels.pkg.slp"))
    tracking_csv = Path(
        args.tracking_csv
        or (
            REPO_ROOT
            / "report"
            / "20260718"
            / "k12_jump8_strict_mouth_rerun"
            / "mouth_only_jump8_rerun_3d_aligned_to_historical_frame.csv"
        )
    )
    annotation_dir = REPO_ROOT / "annotations" / "contact_episode_gt" / experiment
    ledger_path = Path(args.contact_ledger or (annotation_dir / "contact_episode_gt.csv"))
    contact_candidates_path = Path(
        args.contact_candidates or (annotation_dir / "contact_candidates_jump8.csv")
    )
    calibration_dir = Path(
        args.calibration_model_dir
        or _resolve_required(local.calibration_model_dir, "calibration_model_dir")
    )
    seed_dir = Path(
        args.seed_output_dir
        or (_resolve_required(local.runs_dir, "runs_dir") / "round8_active_learning" / experiment)
    )
    cache_dir = Path(args.cache_dir or (seed_dir / "feature_cache"))
    raw_data_root = local.medaka_data_dir or local.raw_data_dir
    slp_paths = {view: slp_dir / f"{view}.slp" for view in VIEWS}

    required_paths = {
        **{f"round7_{view}_slp": path for view, path in slp_paths.items()},
        "round7_top_labels": top_labels,
        "round7_side_labels": side_labels,
    }
    if not args.reuse_selection:
        required_paths.update(
            {
                "strict_mouth_3d": tracking_csv,
                "contact_ledger": ledger_path,
                "contact_candidates": contact_candidates_path,
                "calibration_model_dir": calibration_dir,
            }
        )
    missing_paths = {name: str(path) for name, path in required_paths.items() if not path.exists()}
    if missing_paths:
        raise FileNotFoundError(f"Round 8 source paths are missing: {missing_paths}")
    report_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    selection_path = report_dir / "selection_360.csv"
    if args.reuse_selection:
        if not selection_path.is_file():
            raise FileNotFoundError(f"--reuse-selection needs {selection_path}")
        selection = pd.read_csv(selection_path, keep_default_na=False)
        if len(selection) != 360:
            raise ValueError(f"saved selection must have 360 rows, got {len(selection)}")
        seed_outputs = _build_requested_seeds(
            args.build_seed,
            selection=selection,
            experiment=experiment,
            slp_paths=slp_paths,
            report_dir=report_dir,
            seed_dir=seed_dir,
            raw_data_root=raw_data_root,
        )
        for batch, seed_name in (
            ("pilot", "pilot_120_seed.pkg.slp"),
            ("expansion", "expansion_240_seed.pkg.slp"),
        ):
            existing_seed = seed_dir / seed_name
            if batch in seed_outputs or not existing_seed.is_file():
                continue
            batch_manifest = selection.loc[
                selection["annotation_batch"].astype(str).eq(batch)
            ]
            seed_outputs[batch] = {
                "path": str(existing_seed.resolve()),
                "size_bytes": int(existing_seed.stat().st_size),
                "validation": validate_annotation_seed(
                    existing_seed, batch_manifest, experiment=experiment
                ),
                "reused_existing_seed": True,
            }
        report_path = report_dir / "selection_report.json"
        if report_path.is_file():
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            saved_report["seed_outputs"] = seed_outputs
            saved_report.setdefault("artifacts", {})["seed_output_dir"] = str(
                seed_dir.resolve()
            )
            _write_json(report_path, saved_report)
        _write_runbook(
            report_dir / "RUNBOOK.md",
            experiment=experiment,
            paths_config=Path(args.paths_config).resolve(),
            report_dir=report_dir.resolve(),
            seed_dir=seed_dir.resolve(),
            pilot_seed=(seed_dir / "pilot_120_seed.pkg.slp").resolve(),
            expansion_seed=(seed_dir / "expansion_240_seed.pkg.slp").resolve(),
        )
        return {
            "mode": "reuse_selection",
            "selection_path": str(selection_path),
            "seed_outputs": seed_outputs,
        }

    feature_config = FeatureConfig(
        plausible_score=args.plausible_score,
        low_score=args.low_score,
        score_dispersion=args.score_dispersion,
        mouth_close_px=args.mouth_close_px,
        association_gate_px=args.association_gate_px,
        association_margin_px=args.association_margin_px,
        nn_gate_px=args.nn_gate_px,
        residual_floor_px=args.residual_floor_px,
        residual_mad_multiplier=args.residual_mad_multiplier,
        residual_window_frames=args.residual_window_frames,
    )
    selection_config = SelectionConfig(
        random_seed=args.random_seed,
        time_bins=args.time_bins,
        min_spacing_frames=args.min_spacing_frames,
        event_merge_gap_frames=args.event_merge_gap_frames,
        contact_context_frames=args.contact_context_frames,
        close_ifd_mm=args.close_ifd_mm,
        guard_band_frames=args.guard_band_frames,
        sentinel_start=args.sentinel_start,
        sentinel_end=args.sentinel_end,
    )

    ledger_before = ledger_path.read_bytes()
    ledger = read_annotation_ledger(ledger_path)
    ledger_after = ledger_path.read_bytes()
    if ledger_before != ledger_after:
        raise RuntimeError("contact ledger changed during read-only in-memory migration")
    manual_evidence = ingest_manual_evidence(ledger)
    contact_candidates = pd.read_csv(contact_candidates_path, keep_default_na=False)
    tracking, tracking_cache_hit = prepare_tracking_ifd(
        tracking_csv,
        method="jump8",
        cache_path=cache_dir / "strict_mouth_jump8_ifd.npz",
    )
    calibration = Calibration(calibration_dir, backend=args.calibration_backend)
    direct_xyz = np.asarray(tracking.xyz, dtype=float).copy()
    direct_xyz[np.asarray(tracking.interpolated, dtype=bool)] = np.nan
    projected = calibration.project_world(direct_xyz)

    per_view: dict[str, pd.DataFrame] = {}
    cache_provenance: dict[str, Any] = {
        "cache_dir": str(cache_dir.resolve()),
        "cache_version": FEATURE_CACHE_VERSION,
        "tracking_cache_hit": bool(tracking_cache_hit),
        "views": {},
    }
    for view_index, view in enumerate(VIEWS):
        print(f"[{view}] loading/building Round 7 feature cache...", flush=True)
        frame_features, instance_features, meta = load_or_build_view_features(
            slp_paths[view],
            cache_dir=cache_dir,
            view=view,
            config=feature_config,
            force=args.force_cache,
        )
        frame_features = add_geometry_ambiguity(
            frame_features,
            instance_features,
            projected_geometry=projected[:, :, view_index, :],
            geometry_frames=tracking.frames,
            config=feature_config,
        )
        per_view[view] = frame_features
        cache_provenance["views"][view] = meta
        print(
            f"[{view}] {len(frame_features):,} frames; "
            f"{int(frame_features['ambiguous'].sum()):,} ambiguous",
            flush=True,
        )
        del instance_features

    aggregate, contact_events = aggregate_synchronized_features(
        per_view,
        tracking=tracking,
        contact_candidates=contact_candidates,
        manual_evidence=manual_evidence,
        config=selection_config,
    )
    categorized = assign_exclusive_categories(
        aggregate, close_ifd_mm=selection_config.close_ifd_mm
    )
    label_index = existing_label_index(
        [top_labels, side_labels], experiment=experiment
    )
    categorized = add_existing_label_overlap(categorized, label_index)
    selection, quota_report = select_with_quotas_and_locked_holdout(
        categorized,
        quotas=CATEGORY_QUOTAS,
        config=selection_config,
    )
    selection = assign_annotation_batches(selection, seed=args.random_seed)
    leakage = validate_event_split(
        selection, guard_band_frames=selection_config.guard_band_frames
    )
    manifest = _selection_manifest(selection, experiment=experiment)
    pilot = manifest.loc[manifest["annotation_batch"].eq("pilot")].copy()
    expansion = manifest.loc[manifest["annotation_batch"].eq("expansion")].copy()
    manifest.to_csv(selection_path, index=False, lineterminator="\n")
    pilot.to_csv(report_dir / "pilot_120.csv", index=False, lineterminator="\n")
    expansion.to_csv(report_dir / "expansion_240.csv", index=False, lineterminator="\n")
    _write_json(report_dir / "feature_cache_provenance.json", cache_provenance)

    manual_fp_frames = set(
        manual_evidence.loc[manual_evidence["manual_fp_views"].astype(str).ne(""), "frame"].astype(int)
    )
    selected_frames = set(manifest["frame"].astype(int))
    overlap = manifest["existing_label_overlap"].astype(str).ne("")
    report: dict[str, Any] = {
        "workflow": "round8_active_learning",
        "generated_at_utc": utc_now(),
        "experiment": experiment,
        "random_seed": args.random_seed,
        "source_paths": {name: _fingerprint(path) if path.is_file() else {"path": str(path.resolve())} for name, path in required_paths.items()},
        "feature_config": asdict(feature_config),
        "selection_config": asdict(selection_config),
        "calibration": calibration.provenance,
        "contact_ledger": {
            "rows_on_disk": int(len(ledger)),
            "disk_schema_version": "1.1",
            "in_memory_schema_version": str(ledger["schema_version"].iloc[0]) if len(ledger) else "",
            "rewritten": False,
        },
        "candidate_counts": {
            "by_category": {
                key: int(value)
                for key, value in categorized.loc[categorized["category"].astype(str).ne(""), "category"].value_counts().to_dict().items()
            },
            "by_ambiguous_view_count": {
                str(key): int(value)
                for key, value in categorized["ambiguous_view_count"].value_counts().sort_index().to_dict().items()
            },
            "ambiguous_by_view": {
                view: int(categorized[f"{view}_ambiguous"].fillna(False).sum()) for view in VIEWS
            },
            "probable_fn_by_view": {
                view: int(categorized[f"{view}_probable_fn"].fillna(False).sum()) for view in VIEWS
            },
            "isolated_outlier_by_view": {
                view: int(categorized[f"{view}_outlier_count"].fillna(0).gt(0).sum()) for view in VIEWS
            },
        },
        "quota_selection": quota_report,
        "selection_counts": {
            "total": int(len(manifest)),
            "by_category": {key: int(value) for key, value in manifest["category"].value_counts().to_dict().items()},
            "by_split": {key: int(value) for key, value in manifest["split"].value_counts().to_dict().items()},
            "by_batch_split": {
                f"{batch}|{split}": int(value)
                for (batch, split), value in manifest.groupby(["annotation_batch", "split"]).size().to_dict().items()
            },
        },
        "manual_fp_coverage": {
            "source_frame_count": len(manual_fp_frames),
            "source_frames": sorted(manual_fp_frames),
            "represented_count": len(manual_fp_frames & selected_frames),
            "excluded_frames": sorted(manual_fp_frames - selected_frames),
        },
        "existing_label_overlap": {
            "selected_source_frames_with_any_overlap": int(overlap.sum()),
            "selected_source_frames_without_overlap": int((~overlap).sum()),
            "by_views": dict(Counter(manifest.loc[overlap, "existing_label_overlap"].astype(str))),
        },
        "leakage_checks": leakage,
        "contact_event_count": int(len(contact_events)),
        "artifacts": {
            "selection_360": str(selection_path.resolve()),
            "pilot_120": str((report_dir / "pilot_120.csv").resolve()),
            "expansion_240": str((report_dir / "expansion_240.csv").resolve()),
            "feature_cache_provenance": str((report_dir / "feature_cache_provenance.json").resolve()),
            "large_cache_dir": str(cache_dir.resolve()),
        },
    }
    _write_json(report_dir / "selection_report.json", report)

    seed_outputs = _build_requested_seeds(
        args.build_seed,
        selection=manifest,
        experiment=experiment,
        slp_paths=slp_paths,
        report_dir=report_dir,
        seed_dir=seed_dir,
        raw_data_root=raw_data_root,
    )
    report["seed_outputs"] = seed_outputs
    report["artifacts"]["seed_output_dir"] = str(seed_dir.resolve())
    _write_json(report_dir / "selection_report.json", report)
    _write_runbook(
        report_dir / "RUNBOOK.md",
        experiment=experiment,
        paths_config=Path(args.paths_config).resolve(),
        report_dir=report_dir.resolve(),
        seed_dir=seed_dir.resolve(),
        pilot_seed=(seed_dir / "pilot_120_seed.pkg.slp").resolve(),
        expansion_seed=(seed_dir / "expansion_240_seed.pkg.slp").resolve(),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--sleap-subdir", default=ROUND7_TAG)
    parser.add_argument("--slp-dir", type=Path)
    parser.add_argument("--round7-top-labels", type=Path)
    parser.add_argument("--round7-side-labels", type=Path)
    parser.add_argument("--tracking-csv", type=Path)
    parser.add_argument("--contact-ledger", type=Path)
    parser.add_argument("--contact-candidates", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--calibration-backend", choices=("auto", "refractive", "polynomial"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--reuse-selection", action="store_true")
    parser.add_argument("--build-seed", choices=("none", "pilot", "expansion", "both"), default="pilot")

    parser.add_argument("--plausible-score", type=float, default=FeatureConfig.plausible_score)
    parser.add_argument("--low-score", type=float, default=FeatureConfig.low_score)
    parser.add_argument("--score-dispersion", type=float, default=FeatureConfig.score_dispersion)
    parser.add_argument("--mouth-close-px", type=float, default=FeatureConfig.mouth_close_px)
    parser.add_argument("--association-gate-px", type=float, default=FeatureConfig.association_gate_px)
    parser.add_argument("--association-margin-px", type=float, default=FeatureConfig.association_margin_px)
    parser.add_argument("--nn-gate-px", type=float, default=FeatureConfig.nn_gate_px)
    parser.add_argument("--residual-floor-px", type=float, default=FeatureConfig.residual_floor_px)
    parser.add_argument("--residual-mad-multiplier", type=float, default=FeatureConfig.residual_mad_multiplier)
    parser.add_argument("--residual-window-frames", type=int, default=FeatureConfig.residual_window_frames)

    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--time-bins", type=int, default=SelectionConfig.time_bins)
    parser.add_argument("--min-spacing-frames", type=int, default=SelectionConfig.min_spacing_frames)
    parser.add_argument("--event-merge-gap-frames", type=int, default=SelectionConfig.event_merge_gap_frames)
    parser.add_argument("--contact-context-frames", type=int, default=SelectionConfig.contact_context_frames)
    parser.add_argument("--close-ifd-mm", type=float, default=SelectionConfig.close_ifd_mm)
    parser.add_argument("--guard-band-frames", type=int, default=SelectionConfig.guard_band_frames)
    parser.add_argument("--sentinel-start", type=int, default=SENTINEL_START)
    parser.add_argument("--sentinel-end", type=int, default=SENTINEL_END)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.time_bins <= 0 or args.min_spacing_frames < 1:
        parser.error("time bins must be positive and spacing at least one frame")
    if args.sentinel_end < args.sentinel_start:
        parser.error("sentinel end must not precede its start")
    report = run_pipeline(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
