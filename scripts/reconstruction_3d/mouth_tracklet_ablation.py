from __future__ import annotations

"""Bounded A4 tracklet-first runner and conservative A5 hint exporter."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from scripts.common.local_paths import load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.candidate_cache import (
    CandidateWindow,
    VIEW_NAMES,
    load_candidate_window,
)
from scripts.reconstruction_3d.conservative_cross_view_paths import (
    CrossViewGroupConfig,
    ThreePathConfig,
    build_cross_view_groups,
    solve_three_identity_paths,
)
from scripts.reconstruction_3d.conservative_mouth_tracklets import (
    ConservativeTrackletConfig,
    build_reciprocal_tracklets,
    frozen_body_length_px_from_training_controls,
)
from scripts.reconstruction_3d.mouth_ablation import (
    SOURCE_LABELS,
    _PeakMemory,
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _canonical_cache_metadata,
    _code_hashes,
    _git_state,
    _sha256,
    _stable_hash,
    _tracked_to_direct,
    _trajectory_metrics,
    _write_filled_unshifted,
    selected_geometry_to_best,
)
from scripts.reconstruction_3d.deferred_identity_tracking import (
    DeferredTrackingConfig,
    apply_deferred_identity_tracking,
)
from scripts.reconstruction_3d.windowed_tracklet_core import (
    enumerate_refractive_hypotheses,
)


SCHEMA_VERSION = "mouth_tracklet_ablation_v1"


def _slice_window(
    window: CandidateWindow, frame_start: int, frame_end: int
) -> CandidateWindow:
    left = int(frame_start - window.frame_start)
    right = int(frame_end - window.frame_start + 1)
    if left < 0 or right > len(window.mouth_xy) or right <= left:
        raise ValueError("slice lies outside candidate window")
    return CandidateWindow(
        frame_start=frame_start,
        frame_end=frame_end,
        mouth_xy=window.mouth_xy[left:right],
        tail_xy=window.tail_xy[left:right],
        instance_score=window.instance_score[left:right],
        mouth_score=window.mouth_score[left:right],
        tail_score=window.tail_score[left:right],
        visibility=window.visibility[left:right],
        source_instance=window.source_instance[left:right],
    )


def freeze_training_control_body_scales(
    *,
    cache_dir: Path,
    training_control_intervals: Iterable[tuple[int, int]],
    output_path: Path,
    command: Iterable[str] = (),
) -> dict[str, object]:
    intervals = [(int(start), int(end)) for start, end in training_control_intervals]
    if not intervals or any(start < 0 or end < start for start, end in intervals):
        raise ValueError("valid training-control intervals are required")
    started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    started = time.perf_counter()
    with _PeakMemory() as memory:
        cache_paths, cache_metadata = _canonical_cache_metadata(cache_dir)
        scales = {}
        for view in VIEW_NAMES:
            windows = [
                load_candidate_window(cache_paths[view], start, end)
                for start, end in intervals
            ]
            scales[view] = frozen_body_length_px_from_training_controls(
                windows
            )
        code = _code_hashes(
            [
                Path(__file__),
                Path(__file__).with_name("candidate_cache.py"),
                Path(__file__).with_name("conservative_mouth_tracklets.py"),
            ]
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "frozen_training_control_scale",
            "created_at": started_at.isoformat(),
            "training_control_intervals": [list(values) for values in intervals],
            "eligible_data": "training length-matched temporal controls only",
            "tail_usage": (
                "mouth-to-tail distance is used only for this per-view "
                "normalization scale; tail never enters admission or identity"
            ),
            "body_length_px": scales,
            "cache_sha256": {
                view: metadata["cache_sha256"]
                for view, metadata in cache_metadata.items()
            },
            "code_sha256": code,
            "command": list(command),
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": memory.peak_rss_bytes,
            "peak_memory_method": (
                "0.1 s psutil RSS sampling of process tree around complete action"
            ),
            "git": _git_state(),
        }
        payload["configuration_hash"] = _stable_hash(
            {
                key: payload[key]
                for key in (
                    "training_control_intervals",
                    "body_length_px",
                    "cache_sha256",
                    "code_sha256",
                )
            }
        )
        _atomic_json(output_path, payload)
    return payload


def _admissible_retained_hypotheses(table: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(table), dtype=bool)
    masks = table.view_mask.to_numpy(dtype=int)
    for view_index, view in enumerate(VIEW_NAMES):
        declared = (masks & (1 << view_index)) != 0
        keep &= ~declared | table[f"tracklet_{view}"].ge(0).to_numpy()
    return table.loc[keep].reset_index(drop=True)


def _a4_direct(
    selected: pd.DataFrame, *, frame_start: int, frame_end: int
) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [
            range(frame_start, frame_end + 1),
            ("PATH_1", "PATH_2", "PATH_3"),
        ],
        names=["frame", "fish_id"],
    ).to_frame(index=False)
    if selected.empty:
        direct = grid
        for column in ("X", "Y", "Z", "ray_residual_mm2"):
            direct[column] = np.nan
        direct["identity_segment"] = -1
        direct["hypothesis_id"] = ""
        direct["group_id"] = ""
        direct["identity_status"] = "unassigned_tracklet_path"
        direct["is_interpolated"] = False
        return direct
    observations = selected.copy()
    observations["fish_id"] = observations.path_id.astype(str)
    observations["identity_segment"] = observations.path_segment.astype(int)
    keep_columns = [
        "frame", "fish_id", "identity_segment", "X", "Y", "Z",
        "ray_residual_mm2", "hypothesis_id", "group_id",
        "mouth_confidence", "view_count", "view_mask",
        "slot_cam0", "slot_cam1", "slot_cam2",
    ]
    direct = grid.merge(observations[keep_columns], on=["frame", "fish_id"], how="left")
    segment_intervals = (
        observations.groupby(["fish_id", "identity_segment"])
        .frame.agg(["min", "max"])
        .reset_index()
    )
    direct["identity_status"] = "unassigned_tracklet_path"
    for row in segment_intervals.itertuples(index=False):
        within = (
            direct.fish_id.eq(row.fish_id)
            & direct.frame.between(int(row.min), int(row.max))
            & direct.identity_segment.isna()
        )
        direct.loc[within, "identity_segment"] = int(row.identity_segment)
        direct.loc[within, "identity_status"] = "unresolved_tracklet_gap"
    finite = np.all(
        np.isfinite(direct[["X", "Y", "Z"]].to_numpy(dtype=float)), axis=1
    )
    direct.loc[finite, "identity_status"] = "resolved_tracklet_path_segment"
    direct["identity_segment"] = direct.identity_segment.fillna(-1).astype(int)
    direct["hypothesis_id"] = direct.hypothesis_id.fillna("")
    direct["group_id"] = direct.group_id.fillna("")
    direct["is_interpolated"] = False
    return direct.sort_values(["frame", "fish_id"])[
        [
            "frame", "fish_id", "identity_segment", "X", "Y", "Z",
            "ray_residual_mm2", "hypothesis_id", "group_id",
            "identity_status", "is_interpolated",
            "mouth_confidence", "view_count", "view_mask",
            "slot_cam0", "slot_cam1", "slot_cam2",
        ]
    ]


def _a4_filled(direct: pd.DataFrame, *, max_gap_frames: int = 100) -> pd.DataFrame:
    output = direct.copy()
    output["is_interpolated"] = False
    for (fish_id, segment), rows in direct[
        direct.identity_segment.ge(0)
    ].groupby(["fish_id", "identity_segment"], sort=True):
        finite_rows = rows[
            np.all(
                np.isfinite(rows[["X", "Y", "Z"]].to_numpy(dtype=float)),
                axis=1,
            )
        ]
        if finite_rows.empty:
            continue
        first, last = int(rows.frame.min()), int(rows.frame.max())
        from scripts.reconstruction_3d.fill_missing_3d import (
            ensure_all_frames_and_fill_spline,
        )
        filled = ensure_all_frames_and_fill_spline(
            {"segment": finite_rows[["frame", "X", "Y", "Z"]]},
            list(range(first, last + 1)),
            max_gap_frames=max_gap_frames,
            ensure_non_negative=False,
        )["segment"]
        key = output.fish_id.eq(fish_id) & output.frame.between(first, last)
        indexed = filled.set_index("frame")
        values = indexed.loc[output.loc[key, "frame"], ["X", "Y", "Z"]].to_numpy()
        was_finite = np.all(
            np.isfinite(output.loc[key, ["X", "Y", "Z"]].to_numpy(dtype=float)),
            axis=1,
        )
        now_finite = np.all(np.isfinite(values), axis=1)
        output.loc[key, ["X", "Y", "Z"]] = values
        interpolated = ~was_finite & now_finite
        key_indices = output.index[key]
        output.loc[key_indices[interpolated], "is_interpolated"] = True
        output.loc[
            key_indices[interpolated], "identity_status"
        ] = "interpolated_within_tracklet_segment"
    return output


def run_a4_window(
    *,
    cache_dir: Path,
    calibration_model_dir: Path,
    frozen_scales_path: Path,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
    tracklet_config: ConservativeTrackletConfig,
    group_config: CrossViewGroupConfig,
    path_config: ThreePathConfig,
    hypothesis_block_frames: int = 250,
    evaluation_batch_frames: int = 128,
    xyz_bounds: tuple[float, float] = (-20.0, 220.0),
    command: Iterable[str] = (),
) -> dict[str, object]:
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("invalid inclusive frame interval")
    if hypothesis_block_frames < 1 or evaluation_batch_frames < 1:
        raise ValueError("block sizes must be positive")
    tracklet_config.validate()
    group_config.validate()
    path_config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    started = time.perf_counter()
    with _PeakMemory() as memory:
        cache_paths, cache_metadata = _canonical_cache_metadata(cache_dir)
        scales_payload = json.loads(frozen_scales_path.read_text(encoding="utf-8"))
        scales = {
            view: float(scales_payload["body_length_px"][view])
            for view in VIEW_NAMES
        }
        expected_cache = scales_payload.get("cache_sha256", {})
        actual_cache = {
            view: metadata["cache_sha256"]
            for view, metadata in cache_metadata.items()
        }
        if expected_cache != actual_cache:
            raise ValueError("frozen body scales are bound to different caches")
        windows = {
            view: load_candidate_window(path, frame_start, frame_end)
            for view, path in cache_paths.items()
        }
        tracklets = {
            view: build_reciprocal_tracklets(
                windows[view],
                config=tracklet_config,
                frozen_body_length_px=scales[view],
            )
            for view in VIEW_NAMES
        }
        calibration = Calibration(
            str(calibration_model_dir), backend="refractive"
        )
        if calibration.backend_name != "refractive":
            raise ValueError("A4 requires the resolved refractive backend")
        if calibration.score_unit != "mm^2":
            raise ValueError("A4 requires refractive mm^2 residual scores")
        calibration_file_sha256 = {
            str(path.relative_to(calibration_model_dir)): _sha256(path)
            for path in sorted(calibration_model_dir.glob("*.json"))
            if path.is_file()
        }
        code = _code_hashes(
            [
                Path(__file__),
                Path(__file__).with_name("candidate_cache.py"),
                Path(__file__).with_name("conservative_mouth_tracklets.py"),
                Path(__file__).with_name("conservative_cross_view_paths.py"),
                Path(__file__).with_name("windowed_tracklet_core.py"),
                Path(__file__).with_name("calibration.py"),
            ]
        )
        configuration = {
            "schema_version": SCHEMA_VERSION,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "hypothesis_block_frames": hypothesis_block_frames,
            "evaluation_batch_frames": evaluation_batch_frames,
            "xyz_bounds": list(map(float, xyz_bounds)),
            "num_candidates": 6,
            "num_individuals": 3,
            "anchor": "strict_finite_mouth",
            "tail_identity_features": False,
            "frozen_body_scales_path": str(frozen_scales_path.resolve()),
            "frozen_body_scales_sha256": _sha256(frozen_scales_path),
            "frozen_body_length_px": scales,
            "tracklet_config": asdict(tracklet_config),
            "group_config": asdict(group_config),
            "path_config": asdict(path_config),
            "cache_sha256": actual_cache,
            "calibration_file_sha256": calibration_file_sha256,
            "calibration": {
                "requested_backend": "refractive",
                "resolved_backend": calibration.backend_name,
                "score_unit": calibration.score_unit,
                "provenance": calibration.provenance,
                "file_sha256": calibration_file_sha256,
            },
            "code_sha256": code,
        }
        configuration_hash = _stable_hash(configuration)
        per_view_outputs: dict[str, Path] = {}
        for view, result in tracklets.items():
            view_dir = output_dir / "per_view_tracklets"
            tracklets_path = view_dir / f"{view}.tracklets.parquet"
            dispositions_path = view_dir / f"{view}.dispositions.parquet"
            bridges_path = view_dir / f"{view}.bridges.parquet"
            view_summary_path = view_dir / f"{view}.summary.json"
            _atomic_parquet(tracklets_path, result.tracklets)
            _atomic_parquet(dispositions_path, result.dispositions)
            _atomic_parquet(bridges_path, result.bridges)
            _atomic_json(view_summary_path, result.summary)
            per_view_outputs.update(
                {
                    f"{view}_tracklets": tracklets_path,
                    f"{view}_dispositions": dispositions_path,
                    f"{view}_bridges": bridges_path,
                    f"{view}_summary": view_summary_path,
                }
            )

        blocks_dir = output_dir / "hypothesis_blocks"
        block_tables: list[pd.DataFrame] = []
        hypothesis_block_outputs: dict[str, Path] = {}
        reused_blocks = 0
        for block_start in range(
            frame_start, frame_end + 1, hypothesis_block_frames
        ):
            block_end = min(
                frame_end, block_start + hypothesis_block_frames - 1
            )
            stem = f"admissible_f{block_start:06d}_{block_end:06d}"
            table_path = blocks_dir / f"{stem}.parquet"
            status_path = blocks_dir / f"{stem}.status.json"
            block_hash = _stable_hash(
                {
                    "configuration_hash": configuration_hash,
                    "frame_start": block_start,
                    "frame_end": block_end,
                }
            )
            reuse = False
            if table_path.exists() and status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                reuse = bool(
                    status.get("status") == "completed"
                    and status.get("block_configuration_hash") == block_hash
                    and status.get("output_sha256") == _sha256(table_path)
                )
            if reuse:
                reused_blocks += 1
                admissible = pd.read_parquet(table_path)
            else:
                block_windows = {
                    view: _slice_window(window, block_start, block_end)
                    for view, window in windows.items()
                }
                left = block_start - frame_start
                right = block_end - frame_start + 1
                block_ids = {
                    view: result.tracklet_ids[left:right]
                    for view, result in tracklets.items()
                }
                hypotheses = enumerate_refractive_hypotheses(
                    block_windows,
                    block_ids,
                    calibration,
                    xyz_bounds=xyz_bounds,
                    batch_frames=evaluation_batch_frames,
                    include_two_view=True,
                )
                admissible = _admissible_retained_hypotheses(hypotheses)
                _atomic_parquet(table_path, admissible)
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "block_configuration_hash": block_hash,
                        "frame_start": block_start,
                        "frame_end": block_end,
                        "admissible_rows": int(len(admissible)),
                        "output_sha256": _sha256(table_path),
                    },
                )
            hypothesis_block_outputs[f"{stem}_data"] = table_path
            hypothesis_block_outputs[f"{stem}_status"] = status_path
            block_tables.append(admissible)
        admissible_hypotheses = (
            pd.concat(block_tables, ignore_index=True)
            if block_tables
            else pd.DataFrame()
        )
        groups, group_observations = build_cross_view_groups(
            admissible_hypotheses,
            windows,
            {
                view: result.tracklet_ids
                for view, result in tracklets.items()
            },
            config=group_config,
        )
        path_result = solve_three_identity_paths(
            groups,
            group_observations,
            config=path_config,
            group_config=group_config,
        )
        outputs = {
            "admissible_hypotheses": output_dir / "a4_admissible_hypotheses.parquet",
            "group_observations": output_dir / "a4_group_observations.parquet",
            "groups": output_dir / "a4_groups.csv",
            "paths": output_dir / "a4_paths.csv",
            "links": output_dir / "a4_links.csv",
            "dispositions": output_dir / "a4_group_dispositions.csv",
            "selected": output_dir / "a4_selected_observations.parquet",
            "direct": output_dir / "a4_direct.csv",
            "filled": output_dir / "a4_filled.csv",
        }
        outputs.update(per_view_outputs)
        outputs.update(hypothesis_block_outputs)
        _atomic_parquet(outputs["admissible_hypotheses"], admissible_hypotheses)
        _atomic_parquet(outputs["group_observations"], group_observations)
        _atomic_csv(outputs["groups"], path_result.groups)
        _atomic_csv(outputs["paths"], path_result.paths)
        _atomic_csv(outputs["links"], path_result.links)
        _atomic_csv(outputs["dispositions"], path_result.dispositions)
        _atomic_parquet(outputs["selected"], path_result.selected_observations)
        a4_direct = _a4_direct(
            path_result.selected_observations,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        _atomic_csv(outputs["direct"], a4_direct)
        _atomic_csv(outputs["filled"], _a4_filled(a4_direct))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": path_result.summary.get("status"),
        "configuration_hash": configuration_hash,
        "configuration": configuration,
        "command": list(command),
        "started_at": started_at.isoformat(),
        "finished_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_memory_method": (
            "0.1 s psutil RSS sampling of process tree around complete action"
        ),
        "reused_hypothesis_blocks": reused_blocks,
        "hypothesis_blocks": len(block_tables),
        "admissible_hypothesis_rows": int(len(admissible_hypotheses)),
        "per_view_tracklet_summary": {
            view: result.summary for view, result in tracklets.items()
        },
        "path_summary": path_result.summary,
        "calibration": configuration["calibration"],
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in outputs.items()
        },
        "code_sha256": code,
        "git": _git_state(),
    }
    _atomic_json(output_dir / "a4_summary.json", summary)
    return summary


def build_a5_hints(
    a4_selected: pd.DataFrame,
    a2_direct: pd.DataFrame,
    *,
    minimum_mouth_confidence: float = 0.5,
    minimum_group_support_frames: int = 3,
    maximum_group_median_ray_residual_mm2: float = 1.0,
    maximum_group_mad_ray_residual_mm2: float = 0.25,
    minimum_group_median_mouth_confidence: float = 0.5,
    maximum_group_assignment_cost: float = 0.0,
    require_three_views: bool = True,
    minimum_mapping_votes: int = 3,
    minimum_mapping_purity: float = 0.9,
    minimum_vote_margin: int = 2,
    xyz_tolerance_mm: float = 1e-9,
) -> tuple[pd.DataFrame, list[int], dict[str, object]]:
    """Map exact high-confidence A4 rows into segment-bounded A5 hints."""
    required = {
        "frame", "path_id", "path_segment", "hypothesis_id", "X", "Y", "Z",
        "mouth_confidence", "view_count", "group_eligible",
        "group_support_frames", "group_median_ray_residual_mm2",
        "group_mad_ray_residual_mm2",
        "group_median_mouth_confidence",
        "group_assignment_cost",
        "group_retained_count_mismatch_fraction",
    }
    if missing := sorted(required - set(a4_selected.columns)):
        raise ValueError(f"A4 selected table lacks {missing}")
    a2_required = {
        "frame", "fish_id", "identity_segment", "source_index",
        "hypothesis_id", "X", "Y", "Z"
    }
    if missing := sorted(a2_required - set(a2_direct.columns)):
        raise ValueError(f"A2 direct table lacks {missing}")
    counts = {"selected_input_rows": int(len(a4_selected))}
    strong = a4_selected.copy()
    strong = strong[strong.group_eligible.astype(bool)]
    counts["after_group_eligible"] = int(len(strong))
    strong = strong[
        strong.group_retained_count_mismatch_fraction.eq(0)
    ]
    counts["after_no_retained_count_mismatch"] = int(len(strong))
    if require_three_views:
        strong = strong[strong.view_count.eq(3)]
    counts["after_view_count"] = int(len(strong))
    strong = strong[
        strong.mouth_confidence >= minimum_mouth_confidence
    ]
    counts["after_mouth_confidence"] = int(len(strong))
    strong = strong[
        strong.group_support_frames >= minimum_group_support_frames
    ]
    counts["after_group_support"] = int(len(strong))
    strong = strong[
        strong.group_median_ray_residual_mm2
        <= maximum_group_median_ray_residual_mm2
    ]
    counts["after_group_residual"] = int(len(strong))
    strong = strong[
        strong.group_mad_ray_residual_mm2
        <= maximum_group_mad_ray_residual_mm2
    ]
    counts["after_group_residual_mad"] = int(len(strong))
    strong = strong[
        strong.group_median_mouth_confidence
        >= minimum_group_median_mouth_confidence
    ]
    counts["after_group_median_mouth_confidence"] = int(len(strong))
    strong = strong[
        strong.group_assignment_cost <= maximum_group_assignment_cost
    ]
    counts["after_group_assignment_cost"] = int(len(strong))
    exact = strong.merge(
        a2_direct[
            [
                "frame", "hypothesis_id", "source_index", "fish_id",
                "identity_segment", "X", "Y", "Z",
            ]
        ],
        on=["frame", "hypothesis_id"],
        how="inner",
        suffixes=("_a4", "_geometry"),
    )
    counts["after_exact_shared_geometry_match"] = int(len(exact))
    xyz_delta = np.linalg.norm(
        exact[["X_a4", "Y_a4", "Z_a4"]].to_numpy(dtype=float)
        - exact[["X_geometry", "Y_geometry", "Z_geometry"]].to_numpy(dtype=float),
        axis=1,
    )
    exact = exact[xyz_delta <= xyz_tolerance_mm].copy()
    counts["after_xyz_consistency"] = int(len(exact))
    exact["segment_key"] = (
        exact.path_id.astype(str)
        + ":"
        + exact.path_segment.astype(str)
    )
    exact["a2_identity_key"] = (
        exact.fish_id.astype(str)
        + ":"
        + exact.identity_segment.astype(int).astype(str)
    )
    mapping: dict[str, str] = {}
    mapping_a2_segments: dict[str, int] = {}
    segment_intervals = {
        f"{path_id}:{int(path_segment)}": (
            int(rows.frame.min()),
            int(rows.frame.max()),
        )
        for (path_id, path_segment), rows in a4_selected.groupby(
            ["path_id", "path_segment"], sort=True
        )
    }
    vote_audit: list[dict[str, object]] = []
    for segment_key, rows in exact.groupby("segment_key", sort=True):
        votes = rows.a2_identity_key.astype(str).value_counts()
        best_identity_key = str(votes.index[0])
        best_identity, best_a2_segment_text = best_identity_key.rsplit(":", 1)
        best_a2_segment = int(best_a2_segment_text)
        best_votes = int(votes.iloc[0])
        second_votes = int(votes.iloc[1]) if len(votes) > 1 else 0
        purity = float(best_votes / votes.sum())
        margin = best_votes - second_votes
        accepted = bool(
            best_votes >= minimum_mapping_votes
            and purity >= minimum_mapping_purity
            and margin >= minimum_vote_margin
        )
        if accepted:
            mapping[str(segment_key)] = best_identity
            mapping_a2_segments[str(segment_key)] = best_a2_segment
        vote_audit.append(
            {
                "segment_key": str(segment_key),
                "best_identity": best_identity,
                "best_a2_identity_segment": best_a2_segment,
                "best_a2_identity_key": best_identity_key,
                "best_votes": best_votes,
                "second_votes": second_votes,
                "vote_margin": margin,
                "purity": purity,
                "accepted_by_vote": accepted,
                "accepted": accepted,
                "overlap_identity_conflict": False,
                "frame_start": segment_intervals[str(segment_key)][0],
                "frame_end": segment_intervals[str(segment_key)][1],
            }
        )
    conflict_keys: set[str] = set()
    accepted_items = sorted(mapping.items())
    for index, (left_key, identity) in enumerate(accepted_items):
        left_start, left_end = segment_intervals[left_key]
        for right_key, right_identity in accepted_items[index + 1 :]:
            if identity != right_identity:
                continue
            right_start, right_end = segment_intervals[right_key]
            if max(left_start, right_start) <= min(left_end, right_end):
                conflict_keys.update((left_key, right_key))
    for key in conflict_keys:
        mapping.pop(key, None)
        mapping_a2_segments.pop(key, None)
    for audit in vote_audit:
        if str(audit["segment_key"]) in conflict_keys:
            audit["overlap_identity_conflict"] = True
            audit["accepted"] = False
    counts["rejected_overlap_identity_segments"] = len(conflict_keys)
    exact["identity"] = exact.segment_key.map(mapping)
    exact["a2_identity_segment"] = exact.segment_key.map(mapping_a2_segments)
    hints = exact[exact.identity.notna()].copy()
    hints = hints.sort_values(
        ["frame", "source_index", "identity"]
    )
    duplicate_source = hints.duplicated(
        ["frame", "source_index"], keep=False
    )
    duplicate_identity = hints.duplicated(
        ["frame", "identity"], keep=False
    )
    counts["rejected_per_frame_non_bijective"] = int(
        (duplicate_source | duplicate_identity).sum()
    )
    hints = hints[~duplicate_source & ~duplicate_identity]
    hints["X"] = hints.X_a4
    hints["Y"] = hints.Y_a4
    hints["Z"] = hints.Z_a4
    a4_segment_starts = (
        a4_selected.groupby(["path_id", "path_segment"]).frame.min().reset_index()
    )
    a2_segment_starts = (
        a2_direct[a2_direct.identity_segment.astype(int).gt(0)]
        .groupby(["fish_id", "identity_segment"])
        .frame.min()
        .reset_index()
    )
    reset_frames = sorted(
        {
            int(row.frame)
            for row in a4_segment_starts.itertuples(index=False)
            if int(row.path_segment) > 0
        }
        | {
            int(row.frame)
            for row in a2_segment_starts.itertuples(index=False)
        }
    )
    columns = [
        "frame", "source_index", "identity", "path_id", "path_segment",
        "segment_key", "a2_identity_segment", "hypothesis_id", "X", "Y", "Z",
        "mouth_confidence",
    ]
    hints = hints[columns].reset_index(drop=True)
    summary = {
        "schema_version": 1,
        "minimum_mouth_confidence": minimum_mouth_confidence,
        "minimum_group_support_frames": minimum_group_support_frames,
        "maximum_group_median_ray_residual_mm2": (
            maximum_group_median_ray_residual_mm2
        ),
        "maximum_group_mad_ray_residual_mm2": (
            maximum_group_mad_ray_residual_mm2
        ),
        "minimum_group_median_mouth_confidence": (
            minimum_group_median_mouth_confidence
        ),
        "maximum_group_assignment_cost": maximum_group_assignment_cost,
        "require_three_views": require_three_views,
        "minimum_mapping_votes": minimum_mapping_votes,
        "minimum_mapping_purity": minimum_mapping_purity,
        "minimum_vote_margin": minimum_vote_margin,
        "xyz_tolerance_mm": xyz_tolerance_mm,
        "filter_counts": counts,
        "exact_candidate_matches": int(len(exact)),
        "hint_rows": int(len(hints)),
        "mapped_segments": int(len(mapping)),
        "mapping_vote_audit": vote_audit,
        "forced_reset_frames": reset_frames,
        "forced_reset_policy": (
            "union of A4 path-segment restarts and frozen-A2 "
            "identity-segment starts"
        ),
        "continuity_policy": (
            "path_id:path_segment is mapped by one-to-one vote against frozen "
            "A2 identities; path_id alone is never treated as continuous "
            "across an A4 segment restart. Accepted segment mappings to the "
            "same identity are rejected if their full A4 intervals overlap."
        ),
    }
    summary["configuration_hash"] = _stable_hash(
        {
            key: summary[key]
            for key in (
                "minimum_mouth_confidence",
                "minimum_group_support_frames",
                "maximum_group_median_ray_residual_mm2",
                "maximum_group_mad_ray_residual_mm2",
                "minimum_group_median_mouth_confidence",
                "maximum_group_assignment_cost",
                "require_three_views",
                "minimum_mapping_votes",
                "minimum_mapping_purity",
                "minimum_vote_margin",
                "xyz_tolerance_mm",
            )
        }
    )
    return hints, reset_frames, summary


def run_a5_window(
    *,
    geometry_path: Path,
    geometry_summary_path: Path,
    hints_path: Path,
    forced_reset_frames_path: Path,
    hints_summary_path: Path,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
    config: DeferredTrackingConfig,
    command: Iterable[str] = (),
    xyz_tolerance_mm: float = 1e-9,
) -> dict[str, object]:
    """Run A3's frozen deferred linker with exact A4-derived constraints."""
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("invalid inclusive frame interval")
    if not np.isfinite(xyz_tolerance_mm) or xyz_tolerance_mm < 0:
        raise ValueError("xyz_tolerance_mm must be finite and non-negative")
    config.validate()
    started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _PeakMemory() as memory:
        geometry_summary = json.loads(
            geometry_summary_path.read_text(encoding="utf-8")
        )
        if geometry_summary.get("output_sha256") != _sha256(geometry_path):
            raise ValueError("geometry summary does not authenticate geometry input")
        calibration = geometry_summary.get("calibration", {})
        if "refractive" not in str(calibration.get("backend", "")).lower():
            raise ValueError("A5 requires fixed-interface refractive geometry")
        if calibration.get("score_unit") != "mm^2":
            raise ValueError("A5 requires refractive mm^2 residual scores")

        hints_summary = json.loads(hints_summary_path.read_text(encoding="utf-8"))
        expected_hint_hash = hints_summary.get("outputs", {}).get("hints_sha256")
        expected_reset_hash = hints_summary.get("outputs", {}).get(
            "forced_reset_frames_sha256"
        )
        if expected_hint_hash != _sha256(hints_path):
            raise ValueError("hint summary does not authenticate hints input")
        if expected_reset_hash != _sha256(forced_reset_frames_path):
            raise ValueError("hint summary does not authenticate reset input")

        selected = pd.read_parquet(geometry_path)
        selected = selected[
            selected.frame.between(frame_start, frame_end)
        ].copy()
        if selected.empty:
            raise ValueError("geometry table has no rows in requested interval")
        best_observed = selected_geometry_to_best(selected)
        empty_frame = {
            "groups": {
                source: [np.nan, np.nan, np.nan]
                for source in SOURCE_LABELS
            },
            "perm_cam0": [-1, -1, -1],
            "perm_cam1": [-1, -1, -1],
            "perm_cam2": [-1, -1, -1],
            "errors": [np.nan, np.nan, np.nan],
            "recovered": [False, False, False],
            "synthetic_views": [-1, -1, -1],
            "hypothesis_ids": ["", "", ""],
            "score_unit": "mm^2",
        }
        best = {
            frame: best_observed.get(frame, dict(empty_frame))
            for frame in range(frame_start, frame_end + 1)
        }

        hints = pd.read_csv(hints_path)
        required_hint_columns = {
            "frame", "source_index", "identity", "hypothesis_id", "X", "Y", "Z",
        }
        if missing := sorted(required_hint_columns - set(hints.columns)):
            raise ValueError(f"A5 hints lack {missing}")
        hints = hints[hints.frame.between(frame_start, frame_end)].copy()
        hints["frame"] = hints.frame.astype(int)
        hints["source_index"] = hints.source_index.astype(int)
        allowed_identities = {"ID_1", "ID_2", "ID_3"}
        if not set(hints.identity.astype(str)).issubset(allowed_identities):
            raise ValueError("A5 hints contain an unknown identity")
        if not hints.source_index.between(0, 2).all():
            raise ValueError("A5 hints contain an invalid source index")
        if hints.duplicated(["frame", "source_index"]).any():
            raise ValueError("A5 hints are not source-bijective within frame")
        if hints.duplicated(["frame", "identity"]).any():
            raise ValueError("A5 hints are not identity-bijective within frame")
        for row in hints.itertuples(index=False):
            frame_source = best[int(row.frame)]
            source_index = int(row.source_index)
            hypothesis_ids = frame_source.get("hypothesis_ids", [])
            if (
                source_index >= len(hypothesis_ids)
                or str(hypothesis_ids[source_index]) != str(row.hypothesis_id)
            ):
                raise ValueError("A5 hint is not an exact shared-geometry candidate")
            point = np.asarray(
                frame_source["groups"][SOURCE_LABELS[source_index]], dtype=float
            )
            hint_point = np.asarray([row.X, row.Y, row.Z], dtype=float)
            if (
                not np.all(np.isfinite(point))
                or not np.all(np.isfinite(hint_point))
                or np.linalg.norm(point - hint_point) > xyz_tolerance_mm
            ):
                raise ValueError("A5 hint XYZ differs from shared geometry")

        reset_payload = json.loads(
            forced_reset_frames_path.read_text(encoding="utf-8")
        )
        if not isinstance(reset_payload, list):
            raise ValueError("forced reset input must be a JSON list")
        reset_frames = sorted(
            {
                int(frame)
                for frame in reset_payload
                if frame_start <= int(frame) <= frame_end
            }
        )
        source_hints = {
            int(frame): {
                int(row.source_index): str(row.identity)
                for row in rows.itertuples(index=False)
            }
            for frame, rows in hints.groupby("frame", sort=True)
        }
        deferred = apply_deferred_identity_tracking(
            best,
            config=config,
            source_id_hints_by_frame=source_hints,
            forced_reset_frames=reset_frames,
        )
        direct = _tracked_to_direct(deferred.tracked_by_frame, best)
        direct_path = output_dir / "a5_direct.csv"
        filled_path = output_dir / "a5_filled.csv"
        assignment_path = output_dir / "a5_assignment_audit.csv"
        interval_path = output_dir / "a5_unresolved_intervals.json"
        hint_audit_path = output_dir / "a5_hint_application_audit.csv"
        _atomic_csv(direct_path, direct)
        _write_filled_unshifted(
            direct,
            output_path=filled_path,
            frame_start=frame_start,
            frame_end=frame_end,
            max_gap_frames=config.max_gap_frames,
        )
        assignment_audit = deferred.assignment_audit.copy()
        _atomic_csv(assignment_path, assignment_audit)
        _atomic_json(interval_path, list(deferred.unresolved_intervals))
        direct_lookup = direct.set_index(["frame", "fish_id"])
        hint_audit_records: list[dict[str, object]] = []
        for row in hints.itertuples(index=False):
            direct_row = direct_lookup.loc[(int(row.frame), str(row.identity))]
            actual_source = int(direct_row.source_index)
            hint_audit_records.append(
                {
                    "frame": int(row.frame),
                    "identity": str(row.identity),
                    "hint_source_index": int(row.source_index),
                    "direct_source_index": actual_source,
                    "identity_status": str(direct_row.identity_status),
                    "honored": actual_source == int(row.source_index),
                    "hypothesis_id": str(row.hypothesis_id),
                }
            )
        hint_audit = pd.DataFrame.from_records(
            hint_audit_records,
            columns=[
                "frame", "identity", "hint_source_index",
                "direct_source_index", "identity_status", "honored",
                "hypothesis_id",
            ],
        )
        _atomic_csv(hint_audit_path, hint_audit)

        code = _code_hashes(
            [
                Path(__file__),
                Path(__file__).with_name("mouth_ablation.py"),
                Path(__file__).with_name("deferred_identity_tracking.py"),
                Path(__file__).with_name("cross_frame_tracking.py"),
                Path(__file__).with_name("fill_missing_3d.py"),
            ]
        )
        settings = {
            "schema_version": SCHEMA_VERSION,
            "method": "A5",
            "frame_start": frame_start,
            "frame_end": frame_end,
            "deferred_tracking_config": asdict(config),
            "xyz_tolerance_mm": xyz_tolerance_mm,
            "shift_to_positive": False,
            "fill_ensure_non_negative": False,
            "hint_contract": (
                "exact hypothesis and XYZ constraints mapped to frozen A2 "
                "identity; source and identity are one-to-one per frame"
            ),
            "geometry_sha256": _sha256(geometry_path),
            "geometry_summary_sha256": _sha256(geometry_summary_path),
            "hints_sha256": _sha256(hints_path),
            "forced_reset_frames_sha256": _sha256(
                forced_reset_frames_path
            ),
            "hints_summary_sha256": _sha256(hints_summary_path),
            "code_sha256": code,
        }
        output_paths = {
            "direct": direct_path,
            "filled": filled_path,
            "assignment_audit": assignment_path,
            "unresolved_intervals": interval_path,
            "hint_application_audit": hint_audit_path,
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "configuration_hash": _stable_hash(settings),
        "settings": settings,
        "command": list(command),
        "started_at": started_at.isoformat(),
        "finished_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_memory_method": (
            "0.1 s psutil RSS sampling of process tree around complete action"
        ),
        "metrics": _trajectory_metrics(direct),
        "hint_rows": int(len(hints)),
        "hints_honored": int(hint_audit.honored.sum()) if len(hint_audit) else 0,
        "forced_reset_frames": reset_frames,
        "unresolved_intervals": list(deferred.unresolved_intervals),
        "calibration": calibration,
        "inputs": {
            "geometry": {
                "path": str(geometry_path.resolve()),
                "sha256": _sha256(geometry_path),
            },
            "geometry_summary": {
                "path": str(geometry_summary_path.resolve()),
                "sha256": _sha256(geometry_summary_path),
            },
            "hints": {
                "path": str(hints_path.resolve()),
                "sha256": _sha256(hints_path),
            },
            "forced_reset_frames": {
                "path": str(forced_reset_frames_path.resolve()),
                "sha256": _sha256(forced_reset_frames_path),
            },
            "hints_summary": {
                "path": str(hints_summary_path.resolve()),
                "sha256": _sha256(hints_summary_path),
            },
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in output_paths.items()
        },
        "code_sha256": code,
        "git": _git_state(),
    }
    _atomic_json(output_dir / "a5_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze-scales")
    freeze.add_argument("--cache-dir", type=Path, required=True)
    freeze.add_argument(
        "--training-control", nargs=2, type=int, action="append", required=True
    )
    freeze.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--calibration-model-dir", type=Path, default=None)
    run.add_argument("--paths-config", default="configs/paths.local.toml")
    run.add_argument("--frozen-scales", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--frame-start", type=int, required=True)
    run.add_argument("--frame-end", type=int, required=True)
    run.add_argument("--hypothesis-block-frames", type=int, default=250)
    run.add_argument("--evaluation-batch-frames", type=int, default=128)
    run.add_argument("--xyz-bounds", nargs=2, type=float, default=(-20, 220))
    run.add_argument("--separation-body-lengths", type=float, default=0.5)
    run.add_argument("--ambiguity-margin-body-lengths", type=float, default=0.1)
    run.add_argument("--maximum-missing-gap-frames", type=int, default=2)
    run.add_argument("--minimum-mouth-score", type=float, default=0.3)
    run.add_argument("--minimum-fragment-observations", type=int, default=2)
    run.add_argument("--minimum-group-support", type=int, default=3)
    run.add_argument("--beam-width", type=int, default=256)

    hints = sub.add_parser("hints")
    hints.add_argument("--a4-selected", type=Path, required=True)
    hints.add_argument("--a2-direct", type=Path, required=True)
    hints.add_argument("--output-dir", type=Path, required=True)
    hints.add_argument("--minimum-mouth-confidence", type=float, default=0.5)
    hints.add_argument("--minimum-group-support", type=int, default=3)
    hints.add_argument("--maximum-group-median-residual", type=float, default=1.0)
    hints.add_argument("--maximum-group-mad-residual", type=float, default=0.25)
    hints.add_argument(
        "--minimum-group-median-mouth-confidence", type=float, default=0.5
    )
    hints.add_argument("--maximum-group-assignment-cost", type=float, default=0.0)
    hints.add_argument("--minimum-mapping-votes", type=int, default=3)
    hints.add_argument("--minimum-mapping-purity", type=float, default=0.9)
    hints.add_argument("--minimum-vote-margin", type=int, default=2)
    hints.add_argument("--xyz-tolerance-mm", type=float, default=1e-9)

    a5 = sub.add_parser("a5")
    a5.add_argument("--geometry", type=Path, required=True)
    a5.add_argument("--geometry-summary", type=Path, required=True)
    a5.add_argument("--hints", type=Path, required=True)
    a5.add_argument("--forced-reset-frames", type=Path, required=True)
    a5.add_argument("--hints-summary", type=Path, required=True)
    a5.add_argument("--output-dir", type=Path, required=True)
    a5.add_argument("--frame-start", type=int, required=True)
    a5.add_argument("--frame-end", type=int, required=True)
    a5.add_argument("--velocity-weight", type=float, required=True)
    a5.add_argument("--acceleration-weight", type=float, required=True)
    a5.add_argument("--uncertainty-growth", type=float, default=0.25)
    a5.add_argument("--max-motion-gap-frames", type=int, default=20)
    a5.add_argument("--motion-history-frames", type=int, default=5)
    a5.add_argument("--max-jump-mm", type=float, default=8.0)
    a5.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    a5.add_argument("--max-gap-frames", type=int, default=100)
    a5.add_argument("--beam-width", type=int, default=16)
    a5.add_argument("--entry-margin", type=float, default=0.5)
    a5.add_argument("--retention-margin", type=float, default=2.0)
    a5.add_argument("--resolution-margin", type=float, default=1.0)
    a5.add_argument("--resolution-stability-frames", type=int, default=3)
    a5.add_argument("--minimum-lookahead-frames", type=int, default=3)
    a5.add_argument("--maximum-defer-horizon", type=int, default=100)
    a5.add_argument("--xyz-tolerance-mm", type=float, default=1e-9)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    values = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(values)
    command = [
        sys.executable, "-m",
        "scripts.reconstruction_3d.mouth_tracklet_ablation",
        *values,
    ]
    if args.action == "freeze-scales":
        result = freeze_training_control_body_scales(
            cache_dir=args.cache_dir,
            training_control_intervals=args.training_control,
            output_path=args.output,
            command=command,
        )
    elif args.action == "run":
        model_dir = args.calibration_model_dir
        if model_dir is None:
            paths = load_local_paths(args.paths_config)
            if paths is None or paths.calibration_model_dir is None:
                raise ValueError("calibration model directory is not configured")
            model_dir = Path(paths.calibration_model_dir)
        result = run_a4_window(
            cache_dir=args.cache_dir,
            calibration_model_dir=model_dir,
            frozen_scales_path=args.frozen_scales,
            output_dir=args.output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            tracklet_config=ConservativeTrackletConfig(
                separation_body_lengths=args.separation_body_lengths,
                assignment_ambiguity_margin_body_lengths=(
                    args.ambiguity_margin_body_lengths
                ),
                maximum_missing_gap_frames=args.maximum_missing_gap_frames,
                minimum_mouth_score=args.minimum_mouth_score,
                minimum_fragment_observations=args.minimum_fragment_observations,
            ),
            group_config=CrossViewGroupConfig(
                minimum_support_frames=args.minimum_group_support,
            ),
            path_config=ThreePathConfig(beam_width=args.beam_width),
            hypothesis_block_frames=args.hypothesis_block_frames,
            evaluation_batch_frames=args.evaluation_batch_frames,
            xyz_bounds=tuple(args.xyz_bounds),
            command=command,
        )
    elif args.action == "hints":
        started_at = pd.Timestamp.now(tz="Asia/Tokyo")
        started = time.perf_counter()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with _PeakMemory() as memory:
            hints, reset_frames, summary = build_a5_hints(
                pd.read_parquet(args.a4_selected),
                pd.read_csv(args.a2_direct),
                minimum_mouth_confidence=args.minimum_mouth_confidence,
                minimum_group_support_frames=args.minimum_group_support,
                maximum_group_median_ray_residual_mm2=(
                    args.maximum_group_median_residual
                ),
                maximum_group_mad_ray_residual_mm2=(
                    args.maximum_group_mad_residual
                ),
                minimum_group_median_mouth_confidence=(
                    args.minimum_group_median_mouth_confidence
                ),
                maximum_group_assignment_cost=(
                    args.maximum_group_assignment_cost
                ),
                minimum_mapping_votes=args.minimum_mapping_votes,
                minimum_mapping_purity=args.minimum_mapping_purity,
                minimum_vote_margin=args.minimum_vote_margin,
                xyz_tolerance_mm=args.xyz_tolerance_mm,
            )
            _atomic_csv(args.output_dir / "a5_hints.csv", hints)
            _atomic_json(
                args.output_dir / "a5_forced_reset_frames.json", reset_frames
            )
        summary["outputs"] = {
            "hints_sha256": _sha256(args.output_dir / "a5_hints.csv"),
            "forced_reset_frames_sha256": _sha256(
                args.output_dir / "a5_forced_reset_frames.json"
            ),
        }
        summary["input_sha256"] = {
            "a4_selected": _sha256(args.a4_selected),
            "a2_direct": _sha256(args.a2_direct),
        }
        summary["code_sha256"] = _code_hashes(
            [
                Path(__file__),
                Path(__file__).with_name(
                    "conservative_cross_view_paths.py"
                ),
            ]
        )
        summary["command"] = command
        summary["created_at"] = pd.Timestamp.now(
            tz="Asia/Tokyo"
        ).isoformat()
        summary["started_at"] = started_at.isoformat()
        summary["finished_at"] = pd.Timestamp.now(
            tz="Asia/Tokyo"
        ).isoformat()
        summary["wall_seconds"] = time.perf_counter() - started
        summary["peak_rss_bytes"] = memory.peak_rss_bytes
        summary["peak_memory_method"] = (
            "0.1 s psutil RSS sampling of process tree around complete action"
        )
        summary["git"] = _git_state()
        _atomic_json(args.output_dir / "a5_hints_summary.json", summary)
        result = summary
    else:
        result = run_a5_window(
            geometry_path=args.geometry,
            geometry_summary_path=args.geometry_summary,
            hints_path=args.hints,
            forced_reset_frames_path=args.forced_reset_frames,
            hints_summary_path=args.hints_summary,
            output_dir=args.output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            config=DeferredTrackingConfig(
                max_jump_mm=args.max_jump_mm,
                max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
                max_gap_frames=args.max_gap_frames,
                velocity_prediction_weight=args.velocity_weight,
                tangent_weight=0.0,
                acceleration_weight=args.acceleration_weight,
                motion_uncertainty_growth_per_frame=args.uncertainty_growth,
                max_motion_gap_frames=args.max_motion_gap_frames,
                motion_history_frames=args.motion_history_frames,
                beam_width=args.beam_width,
                entry_margin=args.entry_margin,
                retention_margin=args.retention_margin,
                resolution_margin=args.resolution_margin,
                resolution_stability_frames=args.resolution_stability_frames,
                minimum_lookahead_frames=args.minimum_lookahead_frames,
                maximum_defer_horizon=args.maximum_defer_horizon,
            ),
            command=command,
            xyz_tolerance_mm=args.xyz_tolerance_mm,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
