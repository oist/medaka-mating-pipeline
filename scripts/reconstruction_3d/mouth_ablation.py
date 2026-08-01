from __future__ import annotations

"""Cached strict-mouth geometry and controlled temporal identity ablations.

The geometry action consumes only canonical candidate caches with
``filter_fp=false``. It never decodes SLP files and never reads tail coordinates.
Framewise mutually-exclusive geometry is checkpointed in bounded blocks.
Temporal relinking is a separate action so A1/A2/A3 share byte-identical
geometry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import psutil
import h5py

from scripts.common.local_paths import load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.candidate_cache import (
    VIEW_NAMES,
    load_candidate_window,
    read_cache_metadata,
)
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
)
from scripts.reconstruction_3d.deferred_identity_tracking import (
    DeferredTrackingConfig,
    apply_deferred_identity_tracking,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.windowed_tracklet_core import (
    _select_framewise_hypotheses,
    enumerate_refractive_hypotheses,
)


SCHEMA_VERSION = "mouth_ablation_v1"
SOURCE_LABELS = ("a", "b", "c")
IDENTITY_LABELS = ("ID_1", "ID_2", "ID_3")


class _PeakMemory:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            processes = [process]
            try:
                processes.extend(process.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            total = 0
            for item in processes:
                try:
                    total += int(item.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_rss_bytes = max(self.peak_rss_bytes, total)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _code_hashes(paths: Iterable[Path]) -> dict[str, str]:
    root = Path.cwd().resolve()
    records = {}
    for path in paths:
        resolved = path.resolve()
        try:
            key = str(resolved.relative_to(root))
        except ValueError:
            key = str(resolved)
        records[key] = _sha256(resolved)
    return dict(sorted(records.items()))


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def _distribution(values: Iterable[float]) -> dict[str, object]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return {
        "n": int(len(array)),
        "median": float(np.median(array)) if len(array) else None,
        "p95": float(np.quantile(array, 0.95)) if len(array) else None,
        "p99": float(np.quantile(array, 0.99)) if len(array) else None,
        "max": float(np.max(array)) if len(array) else None,
    }


def _canonical_cache_metadata(cache_dir: Path) -> tuple[dict[str, Path], dict]:
    paths = {
        view: (cache_dir / f"{view}.candidates.h5").resolve()
        for view in VIEW_NAMES
    }
    metadata = {view: read_cache_metadata(path) for view, path in paths.items()}
    for view, record in metadata.items():
        actual_cache_sha256 = _sha256(paths[view])
        if actual_cache_sha256 != record.get("cache_sha256"):
            raise ValueError(
                f"cache {view} content hash does not match its manifest: "
                f"{actual_cache_sha256} != {record.get('cache_sha256')}"
            )
        required = {
            "complete": 1,
            "filter_fp": 0,
            "require_anchor_node": 1,
            "anchor_node": "mouth",
            "num_candidates": 6,
            "coordinate_dtype": "float64",
            "frame_count": 348600,
        }
        for name, expected in required.items():
            actual = record.get(name)
            if str(actual) != str(expected):
                raise ValueError(
                    f"cache {view} violates canonical policy: "
                    f"{name}={actual!r}, expected {expected!r}"
                )
        with h5py.File(paths[view], "r") as cache:
            mouth = cache["mouth_xy"]
            mouth_score = cache["mouth_score"]
            block = 8192
            finite_mouth_count = 0
            finite_score_count = 0
            for start in range(0, mouth.shape[0], block):
                stop = min(mouth.shape[0], start + block)
                finite_mouth = np.all(
                    np.isfinite(mouth[start:stop]), axis=2
                )
                finite_score = np.isfinite(mouth_score[start:stop])
                if np.any(finite_mouth & ~finite_score):
                    raise ValueError(
                        f"cache {view} has finite mouth coordinates without "
                        "a finite structured mouth-node score"
                    )
                if np.any(~finite_mouth & finite_score):
                    raise ValueError(
                        f"cache {view} has a mouth score without finite mouth "
                        "coordinates"
                    )
                finite_mouth_count += int(finite_mouth.sum())
                finite_score_count += int(finite_score.sum())
        record["finite_mouth_coordinates"] = finite_mouth_count
        record["finite_mouth_scores"] = finite_score_count
    return paths, metadata


def selected_geometry_to_best(selected: pd.DataFrame) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for frame, group in selected.groupby("frame", sort=True):
        ordered = group.sort_values(
            ["view_count", "ray_residual_mm2", "hypothesis_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        groups: dict[str, list[float]] = {}
        permutations_by_view = [[], [], []]
        errors = []
        recovered = []
        synthetic_views = []
        hypothesis_ids = []
        for source_index, source_label in enumerate(SOURCE_LABELS):
            if source_index >= len(ordered):
                groups[source_label] = [np.nan, np.nan, np.nan]
                for values in permutations_by_view:
                    values.append(-1)
                errors.append(np.nan)
                recovered.append(False)
                synthetic_views.append(-1)
                hypothesis_ids.append("")
                continue
            row = ordered.iloc[source_index]
            groups[source_label] = [
                float(row.X),
                float(row.Y),
                float(row.Z),
            ]
            for view_index, values in enumerate(permutations_by_view):
                values.append(int(row[f"slot_cam{view_index}"]))
            errors.append(float(row.ray_residual_mm2))
            recovered.append(int(row.view_count) < 3)
            synthetic_views.append(
                next(
                    (
                        view_index
                        for view_index in range(3)
                        if int(row[f"slot_cam{view_index}"]) < 0
                    ),
                    -1,
                )
            )
            hypothesis_ids.append(str(row.hypothesis_id))
        best[int(frame)] = {
            "groups": groups,
            "perm_cam0": permutations_by_view[0],
            "perm_cam1": permutations_by_view[1],
            "perm_cam2": permutations_by_view[2],
            "errors": errors,
            "recovered": recovered,
            "synthetic_views": synthetic_views,
            "hypothesis_ids": hypothesis_ids,
            "score_unit": "mm^2",
        }
    return best


def _build_geometry_block(
    *,
    cache_paths: Mapping[str, Path],
    calibration: Calibration,
    frame_start: int,
    frame_end: int,
    xyz_bounds: tuple[float, float],
    evaluation_batch_frames: int,
) -> pd.DataFrame:
    windows = {
        view: load_candidate_window(path, frame_start, frame_end)
        for view, path in cache_paths.items()
    }
    tracklet_ids = {
        view: np.full(window.mouth_xy.shape[:2], -1, dtype=np.int64)
        for view, window in windows.items()
    }
    hypotheses = enumerate_refractive_hypotheses(
        windows,
        tracklet_ids,
        calibration,
        xyz_bounds=xyz_bounds,
        batch_frames=evaluation_batch_frames,
        include_two_view=True,
    )
    if hypotheses.empty:
        return hypotheses
    return _select_framewise_hypotheses(hypotheses).sort_values(
        ["frame", "view_count", "ray_residual_mm2", "hypothesis_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def build_cached_geometry(
    *,
    cache_dir: Path,
    calibration_model_dir: Path,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
    block_frames: int = 500,
    evaluation_batch_frames: int = 128,
    xyz_bounds: tuple[float, float] = (-20.0, 220.0),
    command: Iterable[str] = (),
) -> dict[str, object]:
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("invalid inclusive frame interval")
    if block_frames < 1 or evaluation_batch_frames < 1:
        raise ValueError("block sizes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths, cache_metadata = _canonical_cache_metadata(cache_dir)
    calibration = Calibration(str(calibration_model_dir), backend="refractive")
    if calibration.score_unit != "mm^2":
        raise ValueError(f"expected refractive mm^2, got {calibration.score_unit}")
    calibration_file_sha256 = {
        str(path.relative_to(calibration_model_dir)): _sha256(path)
        for path in sorted(calibration_model_dir.glob("*.json"))
        if path.is_file()
    }
    if not calibration_file_sha256:
        raise ValueError("no refractive calibration JSON files found")
    code = _code_hashes(
        [
            Path(__file__),
            Path(__file__).with_name("candidate_cache.py"),
            Path(__file__).with_name("windowed_tracklet_core.py"),
            Path(__file__).with_name("tracklet_first_assignment.py"),
            Path(__file__).with_name("calibration.py"),
        ]
    )
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "action": "geometry",
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "block_frames": int(block_frames),
        "evaluation_batch_frames": int(evaluation_batch_frames),
        "xyz_bounds": list(map(float, xyz_bounds)),
        "num_candidates": 6,
        "num_individuals": 3,
        "anchor_node": "mouth",
        "require_anchor_node": True,
        "filter_fp": False,
        "visible_node_centroid": False,
        "tail_coordinates_consumed": False,
        "calibration_backend": calibration.backend_name,
        "score_unit": calibration.score_unit,
        "calibration_provenance": calibration.provenance,
        "calibration_file_sha256": calibration_file_sha256,
        "cache_sha256": {
            view: record["cache_sha256"]
            for view, record in cache_metadata.items()
        },
        "code_sha256": code,
    }
    configuration_hash = _stable_hash(configuration)
    blocks_dir = output_dir / "geometry_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    started = time.perf_counter()
    block_tables: list[pd.DataFrame] = []
    reused_blocks = 0
    with _PeakMemory() as memory:
        for block_start in range(frame_start, frame_end + 1, block_frames):
            block_end = min(frame_end, block_start + block_frames - 1)
            stem = f"geometry_f{block_start:06d}_{block_end:06d}"
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
                block_table = pd.read_parquet(table_path)
            else:
                block_started = time.perf_counter()
                block_table = _build_geometry_block(
                    cache_paths=cache_paths,
                    calibration=calibration,
                    frame_start=block_start,
                    frame_end=block_end,
                    xyz_bounds=xyz_bounds,
                    evaluation_batch_frames=evaluation_batch_frames,
                )
                _atomic_parquet(table_path, block_table)
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "block_configuration_hash": block_hash,
                        "frame_start": block_start,
                        "frame_end": block_end,
                        "selected_rows": int(len(block_table)),
                        "wall_seconds": time.perf_counter() - block_started,
                        "output_sha256": _sha256(table_path),
                    },
                )
            block_tables.append(block_table)
    selected = (
        pd.concat(block_tables, ignore_index=True).sort_values(
            ["frame", "view_count", "ray_residual_mm2", "hypothesis_id"],
            ascending=[True, False, True, True],
        )
        if block_tables
        else pd.DataFrame()
    )
    geometry_path = output_dir / "strict_mouth_selected_geometry.parquet"
    _atomic_parquet(geometry_path, selected)
    counts = selected.groupby("frame").size() if len(selected) else pd.Series(dtype=int)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "configuration_hash": configuration_hash,
        "configuration": configuration,
        "command": list(command),
        "started_at": started_at.isoformat(),
        "finished_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_memory_method": "0.1 s psutil RSS sampling of process tree",
        "reused_blocks": reused_blocks,
        "total_blocks": len(block_tables),
        "selected_rows": int(len(selected)),
        "frames_with_geometry": int(counts.size),
        "frames_with_three_direct": int((counts == 3).sum()),
        "all_three_direct_fraction": (
            float((counts == 3).sum() / (frame_end - frame_start + 1))
        ),
        "ray_residual_mm2": _distribution(
            selected.ray_residual_mm2 if len(selected) else []
        ),
        "output_path": str(geometry_path.resolve()),
        "output_sha256": _sha256(geometry_path),
        "cache_metadata": cache_metadata,
        "calibration": {
            "backend": calibration.backend_name,
            "score_unit": calibration.score_unit,
            "provenance": calibration.provenance,
            "file_sha256": calibration_file_sha256,
        },
        "code_sha256": code,
        "git": _git_state(),
    }
    _atomic_json(output_dir / "geometry_summary.json", summary)
    return summary


def _tracked_to_direct(
    tracked: Mapping[int, Mapping[str, object]],
    best: Mapping[int, Mapping[str, object]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame, frame_result in sorted(tracked.items()):
        mapping = list(frame_result.get("id_mapping", [-1, -1, -1]))
        status = str(frame_result.get("identity_status", "resolved"))
        segment = int(frame_result.get("identity_segment", 0))
        statuses_by_identity = frame_result.get("identity_status_by_id", {})
        segments_by_identity = frame_result.get("identity_segments", {})
        source = best[int(frame)]
        errors = list(source.get("errors", []))
        hypotheses = list(source.get("hypothesis_ids", []))
        synthetic = list(frame_result.get("synthetic_views", [-1, -1, -1]))
        for identity_index, identity in enumerate(IDENTITY_LABELS):
            point = frame_result["groups"].get(
                identity, [np.nan, np.nan, np.nan]
            )
            source_index = (
                int(mapping[identity_index])
                if identity_index < len(mapping)
                else -1
            )
            identity_status = str(
                statuses_by_identity.get(identity, status)
            )
            if (
                source_index < 0
                or not np.all(np.isfinite(np.asarray(point, dtype=float)))
            ):
                identity_status = (
                    identity_status
                    if identity_status.startswith("unresolved")
                    else "unresolved_unassigned_identity"
                )
            records.append(
                {
                    "frame": int(frame),
                    "fish_id": identity,
                    "X": float(point[0]),
                    "Y": float(point[1]),
                    "Z": float(point[2]),
                    "identity_status": identity_status,
                    "identity_segment": int(
                        segments_by_identity.get(identity, segment)
                    ),
                    "source_index": source_index,
                    "ray_residual_mm2": (
                        float(errors[source_index])
                        if 0 <= source_index < len(errors)
                        else np.nan
                    ),
                    "hypothesis_id": (
                        str(hypotheses[source_index])
                        if 0 <= source_index < len(hypotheses)
                        else ""
                    ),
                    "synthetic_view": (
                        int(synthetic[identity_index])
                        if identity_index < len(synthetic)
                        else -1
                    ),
                    "is_interpolated": False,
                }
            )
    return pd.DataFrame.from_records(records)


def _annotate_position_tracking_segments(
    tracked: Mapping[int, Mapping[str, object]],
    *,
    max_gap_frames: int,
    default_status: str,
) -> None:
    """Expose each ID reseed after the linker's stale-state retirement."""
    identities = tuple(IDENTITY_LABELS)
    segments = {identity: 0 for identity in identities}
    last_finite: dict[str, int | None] = {
        identity: None for identity in identities
    }
    ever_finite = {identity: False for identity in identities}
    for frame, result in sorted(tracked.items()):
        statuses: dict[str, str] = {}
        for identity in identities:
            point = np.asarray(
                result.get("groups", {}).get(
                    identity, [np.nan, np.nan, np.nan]
                ),
                dtype=float,
            )
            finite = bool(np.all(np.isfinite(point)))
            reseeded = bool(
                finite
                and ever_finite[identity]
                and last_finite[identity] is not None
                and int(frame) - int(last_finite[identity]) > max_gap_frames
            )
            if reseeded:
                segments[identity] += 1
            if finite:
                last_finite[identity] = int(frame)
                ever_finite[identity] = True
            statuses[identity] = (
                "unresolved_missing_geometry"
                if not finite
                else (
                    "new_segment_after_max_gap_reseed"
                    if reseeded
                    else default_status
                )
            )
        result["identity_segments"] = dict(segments)
        result["identity_status_by_id"] = statuses


def _write_filled_unshifted(
    direct: pd.DataFrame,
    *,
    output_path: Path,
    frame_start: int,
    frame_end: int,
    max_gap_frames: int,
) -> pd.DataFrame:
    trajectories = {}
    for identity in IDENTITY_LABELS:
        table = direct[direct.fish_id.eq(identity)][
            ["frame", "X", "Y", "Z"]
        ].copy()
        finite = np.all(
            np.isfinite(table[["X", "Y", "Z"]].to_numpy(dtype=float)),
            axis=1,
        )
        # The shared filler treats every listed frame as observed. Remove
        # explicit NaN geometry rows first so bounded internal gaps are
        # actually eligible for interpolation.
        trajectories[identity] = table.loc[finite].copy()
    filled = ensure_all_frames_and_fill_spline(
        trajectories,
        list(range(frame_start, frame_end + 1)),
        max_gap_frames=max_gap_frames,
        ensure_non_negative=False,
    )
    direct_keys = set(
        zip(
            direct.loc[
                np.all(
                    np.isfinite(direct[["X", "Y", "Z"]].to_numpy(dtype=float)),
                    axis=1,
                ),
                "frame",
            ].astype(int),
            direct.loc[
                np.all(
                    np.isfinite(direct[["X", "Y", "Z"]].to_numpy(dtype=float)),
                    axis=1,
                ),
                "fish_id",
            ].astype(str),
            strict=True,
        )
    )
    parts = []
    for identity, table in filled.items():
        part = table.copy()
        part["fish_id"] = identity
        part["is_interpolated"] = [
            bool(
                np.all(np.isfinite([row.X, row.Y, row.Z]))
                and (int(row.frame), identity) not in direct_keys
            )
            for row in part.itertuples(index=False)
        ]
        parts.append(part)
    output = pd.concat(parts, ignore_index=True).sort_values(["frame", "fish_id"])
    metadata = direct[
        ["frame", "fish_id", "identity_status", "identity_segment"]
    ]
    output = output.merge(metadata, on=["frame", "fish_id"], how="left")
    unresolved = output.identity_status.astype(str).str.startswith(
        "unresolved"
    )
    output.loc[unresolved, ["X", "Y", "Z"]] = np.nan
    output.loc[unresolved, "is_interpolated"] = False
    _atomic_csv(output_path, output)
    return output


def _trajectory_metrics(direct: pd.DataFrame) -> dict[str, object]:
    finite = np.all(
        np.isfinite(direct[["X", "Y", "Z"]].to_numpy(dtype=float)), axis=1
    )
    frame_complete = (
        direct.assign(_finite=finite).groupby("frame")._finite.sum().eq(3)
    )
    steps: list[float] = []
    for (_identity, _segment), group in direct[finite].groupby(
        ["fish_id", "identity_segment"], sort=True
    ):
        group = group.sort_values("frame")
        frames = group.frame.to_numpy(dtype=int)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        adjacent = np.diff(frames) == 1
        if np.any(adjacent):
            steps.extend(np.linalg.norm(np.diff(xyz, axis=0)[adjacent], axis=1))
    step_array = np.asarray(steps, dtype=float)
    return {
        "rows": int(len(direct)),
        "frames": int(direct.frame.nunique()),
        "all_three_direct_fraction": float(frame_complete.mean()),
        "unresolved_row_fraction": float(
            direct.identity_status.astype(str).str.startswith(
                "unresolved"
            ).mean()
        ),
        "ray_residual_mm2": _distribution(direct.ray_residual_mm2),
        "adjacent_jump_mm": _distribution(step_array),
        "adjacent_steps_gt8_mm": int((step_array > 8.0).sum()),
        "adjacent_steps_gt20_mm": int((step_array > 20.0).sum()),
        "identity_segments": int(
            direct[["fish_id", "identity_segment"]].drop_duplicates().shape[0]
        ),
    }


def relink_geometry(
    *,
    geometry_path: Path,
    output_dir: Path,
    method: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
    velocity_weight: float = 0.0,
    tangent_weight: float = 0.0,
    acceleration_weight: float = 0.0,
    uncertainty_growth: float = 0.25,
    max_motion_gap_frames: int | None = None,
    motion_history_frames: int = 5,
    max_jump_mm: float = 8.0,
    max_jump_hard_cap_mm: float = 80.0,
    max_gap_frames: int = 100,
    deferred_config: DeferredTrackingConfig | None = None,
    command: Iterable[str] = (),
) -> dict[str, object]:
    method = method.upper()
    if method not in {"A1", "A2", "A3"}:
        raise ValueError("method must be A1, A2, or A3")
    if method in {"A2", "A3"} and tangent_weight != 0:
        raise ValueError(
            f"{method} is a controlled velocity method; tangent_weight must be 0"
        )
    if method == "A2" and acceleration_weight != 0:
        raise ValueError(
            "A2 is robust constant-velocity only; acceleration_weight must be 0"
        )
    selected = pd.read_parquet(geometry_path)
    if selected.empty:
        raise ValueError("geometry table is empty")
    best = selected_geometry_to_best(selected)
    frame_start = min(best) if frame_start is None else int(frame_start)
    frame_end = max(best) if frame_end is None else int(frame_end)
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError("invalid inclusive frame interval")
    if min(best) < frame_start or max(best) > frame_end:
        raise ValueError("selected geometry lies outside requested frame interval")
    empty_frame = {
        "groups": {
            source: [np.nan, np.nan, np.nan] for source in SOURCE_LABELS
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
        frame: best.get(frame, dict(empty_frame))
        for frame in range(frame_start, frame_end + 1)
    }
    started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_audit = pd.DataFrame()
    unresolved_intervals: tuple[Mapping[str, object], ...] = ()
    with _PeakMemory() as memory:
        if method in {"A1", "A2"}:
            tracked = apply_cross_frame_tracking(
                best,
                num_individuals=3,
                max_jump_mm=max_jump_mm,
                max_jump_hard_cap_mm=max_jump_hard_cap_mm,
                max_gap_frames=max_gap_frames,
                velocity_prediction_weight=(
                    0.0 if method == "A1" else velocity_weight
                ),
                flow_weight=0.0 if method == "A1" else tangent_weight,
                acceleration_weight=(
                    0.0 if method == "A1" else acceleration_weight
                ),
                motion_uncertainty_growth_per_frame=uncertainty_growth,
                max_motion_gap_frames=max_motion_gap_frames,
                motion_history_frames=(
                    2 if method == "A1" else motion_history_frames
                ),
                source_hints_override_motion_gate=False,
            )
            for result in tracked.values():
                result["identity_status"] = "resolved_position_only" if method == "A1" else "resolved_velocity"
                result["identity_segment"] = 0
            _annotate_position_tracking_segments(
                tracked,
                max_gap_frames=max_gap_frames,
                default_status=(
                    "resolved_position_only"
                    if method == "A1"
                    else "resolved_velocity"
                ),
            )
        else:
            config = deferred_config or DeferredTrackingConfig(
                max_jump_mm=max_jump_mm,
                max_jump_hard_cap_mm=max_jump_hard_cap_mm,
                max_gap_frames=max_gap_frames,
                velocity_prediction_weight=velocity_weight,
                tangent_weight=tangent_weight,
                acceleration_weight=acceleration_weight,
                motion_uncertainty_growth_per_frame=uncertainty_growth,
                max_motion_gap_frames=max_motion_gap_frames,
                motion_history_frames=motion_history_frames,
            )
            deferred = apply_deferred_identity_tracking(best, config=config)
            tracked = deferred.tracked_by_frame
            assignment_audit = deferred.assignment_audit
            unresolved_intervals = deferred.unresolved_intervals
    direct = _tracked_to_direct(tracked, best)
    direct_path = output_dir / f"{method.lower()}_direct.csv"
    filled_path = output_dir / f"{method.lower()}_filled.csv"
    assignment_path = output_dir / f"{method.lower()}_assignment_audit.csv"
    intervals_path = output_dir / f"{method.lower()}_unresolved_intervals.json"
    _atomic_csv(direct_path, direct)
    _write_filled_unshifted(
        direct,
        output_path=filled_path,
        frame_start=frame_start,
        frame_end=frame_end,
        max_gap_frames=max_gap_frames,
    )
    if method == "A3":
        _atomic_csv(assignment_path, assignment_audit)
        _atomic_json(intervals_path, list(unresolved_intervals))
    code = _code_hashes(
        [
            Path(__file__),
            Path(__file__).with_name("cross_frame_tracking.py"),
            Path(__file__).with_name("deferred_identity_tracking.py"),
            Path(__file__).with_name("fill_missing_3d.py"),
        ]
    )
    settings = {
        "method": method,
        "geometry_sha256": _sha256(geometry_path),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "max_jump_mm": max_jump_mm,
        "max_jump_hard_cap_mm": max_jump_hard_cap_mm,
        "max_gap_frames": max_gap_frames,
        "velocity_weight": 0.0 if method == "A1" else velocity_weight,
        "tangent_weight": 0.0 if method == "A1" else tangent_weight,
        "acceleration_weight": 0.0 if method == "A1" else acceleration_weight,
        "uncertainty_growth": uncertainty_growth,
        "max_motion_gap_frames": max_motion_gap_frames,
        "motion_history_frames": 2 if method == "A1" else motion_history_frames,
        "shift_to_positive": False,
        "fill_ensure_non_negative": False,
        "deferred": (
            (deferred_config or DeferredTrackingConfig()).__dict__
            if method == "A3"
            else None
        ),
        "code_sha256": code,
    }
    output_manifest = {
        "direct": {
            "path": str(direct_path.resolve()),
            "sha256": _sha256(direct_path),
        },
        "filled": {
            "path": str(filled_path.resolve()),
            "sha256": _sha256(filled_path),
        },
    }
    if method == "A3":
        output_manifest.update(
            {
                "assignment_audit": {
                    "path": str(assignment_path.resolve()),
                    "sha256": _sha256(assignment_path),
                },
                "unresolved_intervals": {
                    "path": str(intervals_path.resolve()),
                    "sha256": _sha256(intervals_path),
                },
            }
        )
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
        "peak_memory_method": "0.1 s psutil RSS sampling of process tree",
        "metrics": _trajectory_metrics(direct),
        "unresolved_intervals": list(unresolved_intervals),
        "outputs": output_manifest,
        "code_sha256": code,
        "git": _git_state(),
    }
    _atomic_json(output_dir / f"{method.lower()}_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    geometry = subparsers.add_parser("geometry")
    geometry.add_argument("--cache-dir", type=Path, required=True)
    geometry.add_argument("--calibration-model-dir", type=Path, default=None)
    geometry.add_argument("--paths-config", default="configs/paths.local.toml")
    geometry.add_argument("--output-dir", type=Path, required=True)
    geometry.add_argument("--frame-start", type=int, required=True)
    geometry.add_argument("--frame-end", type=int, required=True)
    geometry.add_argument("--block-frames", type=int, default=500)
    geometry.add_argument("--evaluation-batch-frames", type=int, default=128)
    geometry.add_argument("--xyz-bounds", nargs=2, type=float, default=(-20.0, 220.0))

    relink = subparsers.add_parser("relink")
    relink.add_argument("--geometry", type=Path, required=True)
    relink.add_argument("--output-dir", type=Path, required=True)
    relink.add_argument("--method", choices=("A1", "A2", "A3"), required=True)
    relink.add_argument("--frame-start", type=int, required=True)
    relink.add_argument("--frame-end", type=int, required=True)
    relink.add_argument("--velocity-weight", type=float, default=0.0)
    relink.add_argument("--tangent-weight", type=float, default=0.0)
    relink.add_argument("--acceleration-weight", type=float, default=0.0)
    relink.add_argument("--uncertainty-growth", type=float, default=0.25)
    relink.add_argument("--max-motion-gap-frames", type=int, default=None)
    relink.add_argument("--motion-history-frames", type=int, default=5)
    relink.add_argument("--max-jump-mm", type=float, default=8.0)
    relink.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    relink.add_argument("--max-gap-frames", type=int, default=100)
    relink.add_argument("--beam-width", type=int, default=16)
    relink.add_argument("--entry-margin", type=float, default=0.5)
    relink.add_argument("--retention-margin", type=float, default=2.0)
    relink.add_argument("--resolution-margin", type=float, default=1.0)
    relink.add_argument("--resolution-stability-frames", type=int, default=3)
    relink.add_argument("--minimum-lookahead-frames", type=int, default=3)
    relink.add_argument("--maximum-defer-horizon", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    values = list(argv) if argv is not None else None
    args = build_parser().parse_args(values)
    command = [
        sys.executable,
        "-m",
        "scripts.reconstruction_3d.mouth_ablation",
        *(values or sys.argv[1:]),
    ]
    if args.action == "geometry":
        model_dir = args.calibration_model_dir
        if model_dir is None:
            paths = load_local_paths(args.paths_config)
            if paths is None or paths.calibration_model_dir is None:
                raise ValueError("calibration model directory is not configured")
            model_dir = Path(paths.calibration_model_dir)
        summary = build_cached_geometry(
            cache_dir=args.cache_dir,
            calibration_model_dir=model_dir,
            output_dir=args.output_dir,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            block_frames=args.block_frames,
            evaluation_batch_frames=args.evaluation_batch_frames,
            xyz_bounds=tuple(args.xyz_bounds),
            command=command,
        )
    else:
        deferred_config = (
            DeferredTrackingConfig(
                max_jump_mm=args.max_jump_mm,
                max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
                max_gap_frames=args.max_gap_frames,
                velocity_prediction_weight=args.velocity_weight,
                tangent_weight=args.tangent_weight,
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
            )
            if args.method == "A3"
            else None
        )
        summary = relink_geometry(
            geometry_path=args.geometry,
            output_dir=args.output_dir,
            method=args.method,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            velocity_weight=args.velocity_weight,
            tangent_weight=args.tangent_weight,
            acceleration_weight=args.acceleration_weight,
            uncertainty_growth=args.uncertainty_growth,
            max_motion_gap_frames=args.max_motion_gap_frames,
            motion_history_frames=args.motion_history_frames,
            max_jump_mm=args.max_jump_mm,
            max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
            max_gap_frames=args.max_gap_frames,
            deferred_config=deferred_config,
            command=command,
        )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
