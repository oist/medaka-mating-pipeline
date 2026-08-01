from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
XYZ = ("X", "Y", "Z")
EXPECTED_FRAMES = 348_600
EXPECTED_FISH = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(
    path: Path, *, hash_file: bool = True, allow_empty: bool = False
) -> dict[str, Any]:
    if not path.is_file() or (path.stat().st_size <= 0 and not allow_empty):
        qualifier = "Missing artifact" if allow_empty else "Missing or empty artifact"
        raise FileNotFoundError(f"{qualifier}: {path}")
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    if hash_file:
        record["sha256"] = sha256(path)
    return record


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def true_run_lengths(mask: np.ndarray) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return (ends - starts).astype(int).tolist()


def run_summary(mask: np.ndarray) -> dict[str, Any]:
    lengths = true_run_lengths(mask)
    return {
        "row_or_frame_count": int(np.asarray(mask, dtype=bool).sum()),
        "run_count": len(lengths),
        "longest_run_frames": max(lengths, default=0),
        "p95_run_frames": (
            float(np.percentile(lengths, 95)) if lengths else 0.0
        ),
    }


def true_run_intervals(
    mask: np.ndarray, frames: np.ndarray
) -> list[dict[str, int]]:
    values = np.asarray(mask, dtype=bool)
    frame_values = np.asarray(frames, dtype=np.int64)
    if values.size != frame_values.size:
        raise ValueError("mask and frames must have the same length")
    if not values.size:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return [
        {
            "start_frame": int(frame_values[start]),
            "end_frame": int(frame_values[end]),
            "length_frames": int(end - start + 1),
        }
        for start, end in zip(starts, ends)
    ]


def compact_prediction_qc(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Inference QC is not passed: {path}")
    views: dict[str, Any] = {}
    for view in VIEWS:
        source = payload["views"][view]
        merged = source["merged"]
        views[view] = {
            "chunk_count": source["chunk_count"],
            "declared_chunk_coverage_inclusive": source[
                "declared_chunk_coverage_inclusive"
            ],
            "merged_actual_indices_equal_chunk_union": source[
                "merged_actual_indices_equal_chunk_union"
            ],
            "merged_path": source["merged_path"],
            "merged_size_bytes": merged["size_bytes"],
            "merged_sha256": merged["sha256"],
            "labeled_frames": merged["labeled_frames"],
            "expected_source_frames": EXPECTED_FRAMES,
            "zero_detection_source_frames": (
                EXPECTED_FRAMES - merged["labeled_frames"]
            ),
            "missing_unlabeled_source_indices": source[
                "merged_missing_unlabeled_source_indices"
            ],
            "total_instances": merged["total_instances"],
            "max_instances_per_labeled_frame": merged[
                "max_instances_per_labeled_frame"
            ],
            "instances_per_labeled_frame_distribution": merged[
                "instances_per_labeled_frame_distribution"
            ],
            "canonical_video_references": merged["video_references"],
            "expected_canonical_source_video": source["source_video"],
            "skeleton_count": merged["skeleton_count"],
        }
    scheduler = payload["scheduler"]["tasks"]
    return {
        "source_report": file_record(path),
        "status": payload["status"],
        "array_ids": payload["scheduler"]["array_ids"],
        "scheduler_completed_exit_zero": all(
            row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
            for row in scheduler
        ),
        "scheduler_task_count": len(scheduler),
        "nonempty_stderr": payload["logs"]["nonempty_stderr"],
        "error_hits": payload["logs"]["error_hits"],
        "scope": (
            "Structural/integrity QC only: scheduler completion, chunk union, "
            "frame coverage, canonical references, and prediction-count "
            "distributions. This is not labeled pose-accuracy evaluation."
        ),
        "views": views,
    }


def top_track_metrics(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    required = {"frame", "track_id", "x", "y", "node"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = pd.to_numeric(table["frame"], errors="raise").astype(np.int64)
    counts = Counter(table.groupby("frame").size().astype(int).tolist())
    per_track: dict[str, Any] = {}
    all_step_per_frame: list[float] = []
    for track_id, group in table.groupby("track_id", sort=True):
        ordered = group.sort_values("frame", kind="stable")
        frames = ordered["frame"].to_numpy(np.int64)
        xy = ordered[["x", "y"]].to_numpy(float)
        gaps = np.diff(frames)
        displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        valid = (gaps > 0) & np.isfinite(displacement)
        step_per_frame = displacement[valid] / gaps[valid]
        all_step_per_frame.extend(step_per_frame.tolist())
        per_track[str(track_id)] = {
            "rows": int(len(ordered)),
            "frame_min": int(frames.min()),
            "frame_max": int(frames.max()),
            "adjacent_observation_gaps_gt_1": int(np.sum(gaps > 1)),
            "longest_observation_gap_frames": (
                int(np.max(gaps)) if gaps.size else 0
            ),
            "step_px_per_frame": quantiles(step_per_frame),
        }
    steps = np.asarray(all_step_per_frame, dtype=float)
    return {
        "artifact": file_record(path),
        "rows": int(len(table)),
        "unique_frames": int(frame.nunique()),
        "missing_source_frames": int(EXPECTED_FRAMES - frame.nunique()),
        "frame_min": int(frame.min()),
        "frame_max": int(frame.max()),
        "track_ids": sorted(str(value) for value in table["track_id"].unique()),
        "track_count": int(table["track_id"].nunique()),
        "node_distribution": {
            str(key): int(value)
            for key, value in table["node"].value_counts(dropna=False).items()
        },
        "centroid_only_coordinates": set(
            table["node"].dropna().astype(str).unique()
        )
        == {"centroid"},
        "tracks_below_bind_minimum_500_rows": int(
            (table.groupby("track_id").size() < 500).sum()
        ),
        "detections_per_observed_frame_distribution": {
            str(key): value for key, value in sorted(counts.items())
        },
        "step_px_per_frame": quantiles(steps),
        "step_denominator": int(steps.size),
        "steps_gt80_px_per_frame": int(np.sum(steps > 80.0)),
        "step_rate_gt80_px_per_frame": (
            float(np.mean(steps > 80.0)) if steps.size else None
        ),
        "steps_gt_100_px_per_frame": int(np.sum(steps > 100.0)),
        "step_rate_gt100_px_per_frame": (
            float(np.mean(steps > 100.0)) if steps.size else None
        ),
        "per_track": per_track,
    }


def pre_link_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames: dict[str, Any] = payload["frames"]
    labels = payload.get("src_labels") or ["a", "b", "c"]
    residuals: list[float] = []
    residuals_three_view: list[float] = []
    residuals_synthetic_view: list[float] = []
    residuals_by_fish: dict[str, list[float]] = {label: [] for label in labels}
    residual_tail_thresholds = (10.0, 25.0, 100.0, 1000.0)
    residual_tail_counts = {
        threshold: {"all": 0, "three_view": 0, "synthetic_view": 0}
        for threshold in residual_tail_thresholds
    }
    highest_residual_rows: list[dict[str, Any]] = []
    all_three_valid = 0
    all_three_finite_error = 0
    any_synthetic = 0
    any_recovered = 0
    synthetic_view_counts: Counter[int] = Counter()
    frames_by_backend: Counter[str] = Counter()
    frames_by_score_unit: Counter[str] = Counter()
    dropped_count = 0
    selection_diagnostic_keys: Counter[str] = Counter()

    for frame, info in frames.items():
        valid_fish = 0
        finite_errors = 0
        synthetic = [int(value) for value in info.get("synthetic_views", [])]
        for slot, label in enumerate(labels):
            xyz = np.asarray(info["groups"].get(label), dtype=float)
            if xyz.shape == (3,) and np.isfinite(xyz).all():
                valid_fish += 1
            error = info["errors"].get(label)
            if error is not None and np.isfinite(float(error)):
                value = float(error)
                synthetic_view = (
                    synthetic[slot] if slot < len(synthetic) else -1
                )
                observation_mode = (
                    "synthetic_view" if synthetic_view >= 0 else "three_view"
                )
                residuals.append(value)
                residuals_by_fish[label].append(value)
                (
                    residuals_synthetic_view
                    if observation_mode == "synthetic_view"
                    else residuals_three_view
                ).append(value)
                for threshold in residual_tail_thresholds:
                    if value > threshold:
                        residual_tail_counts[threshold]["all"] += 1
                        residual_tail_counts[threshold][observation_mode] += 1
                highest_residual_rows.append(
                    {
                        "frame": int(frame),
                        "source_fish": str(label),
                        "residual_mm2": value,
                        "observation_mode": observation_mode,
                        "synthetic_view_index": (
                            synthetic_view if synthetic_view >= 0 else None
                        ),
                    }
                )
                finite_errors += 1
        all_three_valid += int(valid_fish == EXPECTED_FISH)
        all_three_finite_error += int(finite_errors == EXPECTED_FISH)
        finite_synthetic = [value for value in synthetic if value >= 0]
        any_synthetic += int(bool(finite_synthetic))
        synthetic_view_counts.update(finite_synthetic)
        any_recovered += int(any(bool(value) for value in info.get("recovered", [])))
        dropped_count += len(info.get("dropped_high_residual", []))
        frames_by_backend[str(info.get("triangulation_backend"))] += 1
        frames_by_score_unit[str(info.get("score_unit"))] += 1
        selection_diagnostic_keys.update(
            (info.get("selection_diagnostics") or {}).keys()
        )

    n_frames = len(frames)
    finite_residual_count = len(residuals)
    calibration = payload.get("calibration", {})
    return {
        "artifact": file_record(path),
        "num_individuals": int(payload["num_individuals"]),
        "src_labels": labels,
        "frame_count": n_frames,
        "all_three_fish_valid_frames": all_three_valid,
        "all_three_fish_valid_fraction": all_three_valid / max(n_frames, 1),
        "all_three_finite_residual_frames": all_three_finite_error,
        "residual_score_unit": calibration.get("assignment_scoring", {}).get(
            "score_unit"
        ),
        "residuals": quantiles(residuals),
        "residuals_by_observation_mode": {
            "three_view": quantiles(residuals_three_view),
            "synthetic_view": quantiles(residuals_synthetic_view),
        },
        "residual_tail_counts_strictly_greater_than_mm2": {
            str(threshold): {
                **counts,
                "fraction_of_all_finite_residuals": (
                    counts["all"] / finite_residual_count
                    if finite_residual_count
                    else None
                ),
                "synthetic_share_of_tail": (
                    counts["synthetic_view"] / counts["all"]
                    if counts["all"]
                    else None
                ),
            }
            for threshold, counts in residual_tail_counts.items()
        },
        "highest_residual_rows": sorted(
            highest_residual_rows,
            key=lambda row: row["residual_mm2"],
            reverse=True,
        )[:20],
        "residuals_by_source_fish": {
            label: quantiles(values)
            for label, values in residuals_by_fish.items()
        },
        "dropped_high_residual_fish_rows": dropped_count,
        "residual_gate_interpretation": (
            "No residual gate was applied when score_gate.value is null. A "
            "zero dropped_high_residual count therefore does not mean the "
            "residual tail is absent; use the explicit tail counts and worst "
            "frames above."
        ),
        "frames_with_any_synthetic_view": any_synthetic,
        "frames_with_any_synthetic_view_fraction": any_synthetic
        / max(n_frames, 1),
        "synthetic_view_counts": {
            str(key): value for key, value in sorted(synthetic_view_counts.items())
        },
        "frames_with_any_recovered_fish": any_recovered,
        "per_frame_triangulation_backend": dict(frames_by_backend),
        "per_frame_score_unit": dict(frames_by_score_unit),
        "selection_diagnostic_key_occurrences": dict(
            selection_diagnostic_keys
        ),
        "calibration": calibration,
    }


def adjacent_step_metrics(
    table: pd.DataFrame, *, label_col: str
) -> dict[str, Any]:
    steps: list[float] = []
    for _label, group in table.groupby(label_col, sort=False):
        ordered = group.sort_values("frame", kind="stable")
        frames = ordered["frame"].to_numpy(np.int64)
        xyz = ordered[list(XYZ)].to_numpy(float)
        displacement = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        valid = (np.diff(frames) == 1) & np.isfinite(displacement)
        steps.extend(displacement[valid].tolist())
    values = np.asarray(steps, dtype=float)
    return {
        "adjacent_finite_steps": int(values.size),
        "step_mm_per_frame": quantiles(values),
        "jumps_gt8_mm": int(np.sum(values > 8.0)),
        "jump_rate_gt8_mm": (
            float(np.mean(values > 8.0)) if values.size else None
        ),
        "jumps_gt20_mm": int(np.sum(values > 20.0)),
        "jump_rate_gt20_mm": (
            float(np.mean(values > 20.0)) if values.size else None
        ),
        "steps_ge7_5_mm": int(np.sum(values >= 7.5)),
        "step_rate_ge7_5_mm": (
            float(np.mean(values >= 7.5)) if values.size else None
        ),
    }


def final_3d_metrics(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    required = {"frame", "fish_id", *XYZ}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    finite = np.isfinite(table[list(XYZ)].to_numpy(float)).all(axis=1)
    xyz = table[list(XYZ)].to_numpy(float)
    component_finite = np.isfinite(xyz)
    negative = component_finite & (xyz < 0)
    exact_zero = component_finite & (xyz == 0)
    is_interpolated = (
        table["is_interpolated"].astype(bool).to_numpy()
        if "is_interpolated" in table
        else np.zeros(len(table), dtype=bool)
    )
    observed = finite & ~is_interpolated
    frame_sizes = table.groupby("frame").size()
    frame_unique_ids = table.groupby("frame")["fish_id"].nunique()
    finite_by_frame = pd.Series(finite, index=table.index).groupby(
        table["frame"]
    ).sum()
    observed_by_frame = pd.Series(observed, index=table.index).groupby(
        table["frame"]
    ).sum()
    all_three_finite_by_frame = finite_by_frame == EXPECTED_FISH
    post_5000 = all_three_finite_by_frame.index >= 5000
    nonfinite_frame_numbers = all_three_finite_by_frame.index[
        ~all_three_finite_by_frame
    ].to_numpy(np.int64)
    per_fish_gap_metrics: dict[str, Any] = {}
    for fish_id, group in table.groupby("fish_id", sort=True):
        ordered = group.sort_values("frame", kind="stable")
        group_frames = ordered["frame"].to_numpy(np.int64)
        group_xyz = ordered[list(XYZ)].to_numpy(float)
        group_finite = np.isfinite(group_xyz).all(axis=1)
        group_interpolated = (
            ordered["is_interpolated"].astype(bool).to_numpy()
            if "is_interpolated" in ordered
            else np.zeros(len(ordered), dtype=bool)
        )
        per_fish_gap_metrics[str(fish_id)] = {
            "nonfinite_runs": run_summary(~group_finite),
            "nonfinite_run_intervals": true_run_intervals(
                ~group_finite, group_frames
            ),
            "last_nonfinite_frame": (
                int(group_frames[~group_finite].max())
                if (~group_finite).any()
                else None
            ),
            "not_directly_observed_geometry_runs": run_summary(
                ~(group_finite & ~group_interpolated)
            ),
            "interpolated_rows": int(group_interpolated.sum()),
            "finite_rows": int(group_finite.sum()),
        }
    return {
        "artifact": file_record(path),
        "rows": int(len(table)),
        "frame_min": int(table["frame"].min()),
        "frame_max": int(table["frame"].max()),
        "unique_frames": int(table["frame"].nunique()),
        "fish_ids": sorted(str(value) for value in table["fish_id"].unique()),
        "fish_id_count": int(table["fish_id"].nunique()),
        "duplicate_frame_fish_rows": int(
            table.duplicated(["frame", "fish_id"]).sum()
        ),
        "frames_with_exactly_three_rows": int((frame_sizes == 3).sum()),
        "frames_with_three_unique_fish_ids": int(
            (frame_unique_ids == 3).sum()
        ),
        "finite_xyz_rows": int(finite.sum()),
        "finite_xyz_row_fraction": float(finite.mean()),
        "all_three_finite_frames": int((finite_by_frame == 3).sum()),
        "all_three_finite_frame_fraction": float(
            all_three_finite_by_frame.mean()
        ),
        "last_frame_without_all_three_finite": (
            int(nonfinite_frame_numbers.max())
            if nonfinite_frame_numbers.size
            else None
        ),
        "all_three_finite_post_frame_5000": {
            "frame_start_inclusive": 5000,
            "frame_end_inclusive": int(all_three_finite_by_frame.index.max()),
            "frame_count": int(post_5000.sum()),
            "all_three_finite_frames": int(
                all_three_finite_by_frame[post_5000].sum()
            ),
            "all_three_finite_frame_fraction": float(
                all_three_finite_by_frame[post_5000].mean()
            ),
        },
        "all_three_directly_observed_frames": int(
            (observed_by_frame == 3).sum()
        ),
        "all_three_directly_observed_frame_fraction": float(
            (observed_by_frame == 3).mean()
        ),
        "interpolated_rows": int(is_interpolated.sum()),
        "interpolated_row_fraction": float(is_interpolated.mean()),
        "coordinate_sign_and_bounds": {
            "rows_with_any_exact_zero": int(exact_zero.any(axis=1).sum()),
            "observed_rows_with_any_exact_zero": int(
                exact_zero[observed].any(axis=1).sum()
            ),
            "axis_exact_zero_counts": {
                axis: int(value)
                for axis, value in zip(XYZ, exact_zero.sum(axis=0))
            },
            "rows_with_any_negative": int(negative.any(axis=1).sum()),
            "observed_rows_with_any_negative": int(
                negative[observed].any(axis=1).sum()
            ),
            "axis_negative_counts": {
                axis: int(value)
                for axis, value in zip(XYZ, negative.sum(axis=0))
            },
            "observed_axis_negative_counts": {
                axis: int(value)
                for axis, value in zip(XYZ, negative[observed].sum(axis=0))
            },
            "axis_minima_mm": {
                axis: float(value)
                for axis, value in zip(XYZ, np.nanmin(xyz, axis=0))
            },
            "axis_maxima_mm": {
                axis: float(value)
                for axis, value in zip(XYZ, np.nanmax(xyz, axis=0))
            },
            "rows_outside_requested_xyz_bounds_minus20_to220": int(
                (
                    component_finite
                    & ((xyz < -20.0) | (xyz > 220.0))
                ).any(axis=1).sum()
            ),
        },
        "per_fish_gap_metrics": per_fish_gap_metrics,
        "final_nonfinite_gap_interpretation": (
            "The final nonfinite gaps are evaluated separately from "
            "not-directly-observed/interpolated geometry runs. A 100% "
            "post-frame-5000 all-three-finite fraction establishes that final "
            "nonfinite gaps are startup-only; it does not imply that every "
            "later row is directly observed rather than interpolated."
        ),
        "adjacent_displacement": adjacent_step_metrics(
            table, label_col="fish_id"
        ),
    }


def binding_metrics(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    required = {
        "frame",
        "fish_id",
        "final_fish_id",
        "bind_source",
        "bind_score",
        *XYZ,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    source_counts = table["bind_source"].fillna("missing").value_counts()
    bound_scores = pd.to_numeric(
        table.loc[table["bind_source"] == "top_cam_2d", "bind_score"],
        errors="coerce",
    ).to_numpy(float)
    fish_to_final = (
        table.groupby("fish_id")["final_fish_id"]
        .nunique(dropna=False)
        .astype(int)
    )
    final_to_fish = (
        table.groupby("final_fish_id")["fish_id"]
        .nunique(dropna=False)
        .astype(int)
    )
    mapping = (
        table[["fish_id", "final_fish_id"]]
        .drop_duplicates()
        .sort_values(["fish_id", "final_fish_id"], kind="stable")
    )
    geometry_only_by_frame = (
        table["bind_source"].eq("geometry_only").groupby(table["frame"]).any()
    )
    return {
        "artifact": file_record(path),
        "rows": int(len(table)),
        "bind_source_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "top_cam_2d_bound_fraction": float(
            (table["bind_source"] == "top_cam_2d").mean()
        ),
        "bind_score_px": quantiles(bound_scores),
        "geometry_fish_to_final_fish_mapping": mapping.to_dict("records"),
        "each_geometry_id_has_one_final_id": bool((fish_to_final == 1).all()),
        "each_final_id_has_one_geometry_id": bool((final_to_fish == 1).all()),
        "stable_global_one_to_one_relabel": bool(
            (fish_to_final == 1).all()
            and (final_to_fish == 1).all()
            and len(fish_to_final) == EXPECTED_FISH
            and len(final_to_fish) == EXPECTED_FISH
        ),
        "frames_with_any_geometry_only": int(geometry_only_by_frame.sum()),
        "geometry_only_frame_runs": run_summary(
            geometry_only_by_frame.to_numpy()
        ),
        "mapping_interpretation": (
            "A stable global one-to-one mapping is structural continuity "
            "evidence. It does not prove biological identity correctness "
            "through smooth contacts/crossings."
        ),
    }


def stage_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "completed" or payload.get("exit_code") != 0:
        raise RuntimeError(f"Stage status is not completed/0: {path}")
    return {
        "artifact": file_record(path),
        "status": payload["status"],
        "exit_code": payload["exit_code"],
        "started_at": payload["started_at"],
        "finished_at": payload["finished_at"],
        "executable": payload["executable"],
        "working_directory": payload["working_directory"],
        "argv": payload["argv"],
        "pipeline_source": payload.get("pipeline_source"),
        "pipeline_sources": payload.get("pipeline_sources"),
        "input_slps": payload.get("input_slps"),
        "input_slp": payload.get("input_slp"),
        "input_top_tracks_csv": payload.get("input_top_tracks_csv"),
        "calibration_bundle": payload.get("calibration_bundle"),
        "outputs": payload.get("outputs"),
        "output_qc": payload.get("output_qc"),
        "stdout_log": file_record(Path(payload["stdout_log"])),
        "stderr_log": file_record(
            Path(payload["stderr_log"]), allow_empty=True
        ),
    }


def experiment_quality(
    experiment: str,
    *,
    sleap_root: Path,
    reconstruction_root: Path,
    prediction_qc_dir: Path,
    tag: str,
) -> dict[str, Any]:
    sleap_dir = sleap_root / experiment / tag
    output_dir = reconstruction_root / experiment / tag
    prediction_qc = (
        prediction_qc_dir / f"{tag}_{experiment}_qc.json"
    )
    track_csv = sleap_dir / "top_cam.tracks.csv"
    pre_link = output_dir / f"{tag}_best_pre_link.json"
    csv3d = output_dir / f"{tag}_3d.csv"
    identified = output_dir / f"{tag}_3d_identified.csv"
    projection_provenance = (
        output_dir / f"{tag}_3d_identified_projection_provenance.json"
    )
    reconstruction_provenance = (
        output_dir / f"{tag}_reconstruction_provenance.json"
    )

    result = {
        "prediction_qc": compact_prediction_qc(prediction_qc),
        "stage2_status": stage_status(
            output_dir / "stage2_track_simple_nn.status.json"
        ),
        "top_camera_2d_tracking": top_track_metrics(track_csv),
        "stage3_status": stage_status(
            output_dir / "stage3_reconstruct_from_slp.status.json"
        ),
        "pre_link_3d": pre_link_metrics(pre_link),
        "final_3d": final_3d_metrics(csv3d),
        "identity_binding": binding_metrics(identified),
        "reconstruction_provenance": {
            "artifact": file_record(reconstruction_provenance),
            "content": json.loads(
                reconstruction_provenance.read_text(encoding="utf-8")
            ),
        },
        "projection_provenance": {
            "artifact": file_record(projection_provenance),
            "content": json.loads(
                projection_provenance.read_text(encoding="utf-8")
            ),
        },
    }

    prediction_ok = bool(
        result["prediction_qc"]["scheduler_completed_exit_zero"]
        and result["prediction_qc"]["scheduler_task_count"] == 12
        and not result["prediction_qc"]["nonempty_stderr"]
        and not result["prediction_qc"]["error_hits"]
        and all(
            view["chunk_count"] == 4
            and view["max_instances_per_labeled_frame"] <= 6
            and view["skeleton_count"] == 1
            and (
                view["labeled_frames"]
                + view["zero_detection_source_frames"]
                == EXPECTED_FRAMES
            )
            and view["zero_detection_source_frames"]
            == view["missing_unlabeled_source_indices"]
            and view["declared_chunk_coverage_inclusive"]
            == [0, EXPECTED_FRAMES - 1]
            and view["merged_actual_indices_equal_chunk_union"]
            and view["canonical_video_references"]
            == [view["expected_canonical_source_video"]]
            for view in result["prediction_qc"]["views"].values()
        )
    )
    pre_link_ok = (
        result["pre_link_3d"]["frame_count"] == EXPECTED_FRAMES
        and result["pre_link_3d"]["num_individuals"] == EXPECTED_FISH
        and result["pre_link_3d"]["residual_score_unit"] == "mm^2"
        and result["pre_link_3d"]["calibration"]
        .get("triangulation", {})
        .get("backend")
        == "refractive"
    )
    final = result["final_3d"]
    final_ok = (
        final["rows"] == EXPECTED_FRAMES * EXPECTED_FISH
        and final["unique_frames"] == EXPECTED_FRAMES
        and final["fish_id_count"] == EXPECTED_FISH
        and final["duplicate_frame_fish_rows"] == 0
        and final["frames_with_exactly_three_rows"] == EXPECTED_FRAMES
        and final["frames_with_three_unique_fish_ids"] == EXPECTED_FRAMES
        and final["finite_xyz_row_fraction"] == 1.0
    )
    binding_ok = result["identity_binding"]["stable_global_one_to_one_relabel"]
    severe_spikes = final["adjacent_displacement"]["jumps_gt20_mm"]
    structural_checks_passed = bool(
        prediction_ok
        and pre_link_ok
        and final_ok
        and binding_ok
        and severe_spikes == 0
    )
    result["quality_gate"] = {
        "prediction_structural_qc": prediction_ok,
        "pre_link_provenance_and_shape": pre_link_ok,
        "final_3d_shape_and_finiteness": final_ok,
        "stable_global_one_to_one_relabel": binding_ok,
        "severe_adjacent_speed_spikes_gt20_mm": severe_spikes,
        "adjacent_steps_ge7_5_mm": final["adjacent_displacement"][
            "steps_ge7_5_mm"
        ],
        "structural_checks_passed": structural_checks_passed,
        "status": "review",
        "status_interpretation": (
            "Review is mandatory even when structural checks pass. The "
            "configured 8 mm adjacent-link gate makes zero >20 mm accepted "
            "steps expected; it is not independent proof of identity quality."
        ),
        "identity_caveat": (
            "A stable one-to-one relabel and bounded/zero over-threshold "
            "adjacent steps characterize bounded geometric continuity. They do "
            "not establish biological identity correctness at smooth contacts "
            "or crossings."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleap-root", required=True, type=Path)
    parser.add_argument("--reconstruction-root", required=True, type=Path)
    parser.add_argument("--prediction-qc-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "tag": args.tag,
        "expected_source_frames": EXPECTED_FRAMES,
        "expected_fish": EXPECTED_FISH,
        "speed_spike_thresholds_mm_per_adjacent_frame": [8.0, 20.0],
        "experiments": {},
    }
    for experiment in EXPERIMENTS:
        print(f"[quality] {experiment}", flush=True)
        report["experiments"][experiment] = experiment_quality(
            experiment,
            sleap_root=args.sleap_root,
            reconstruction_root=args.reconstruction_root,
            prediction_qc_dir=args.prediction_qc_dir,
            tag=args.tag,
        )
    report["overall_status"] = "review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "overall_status": report["overall_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
