from __future__ import annotations

"""Run cached, overlapping-window mouth-based 3D tracklet association.

The entry point is intentionally artifact-first: every numerical window is a
resumable checkpoint; overlap evidence, cross-view groups, fragments, link
edges, hypotheses, direct observations, and filled output remain inspectable.
Machine paths are resolved from the local paths configuration or explicit CLI
arguments and are never embedded in reusable defaults.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import psutil

from scripts.common.local_paths import load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.candidate_cache import (
    VIEW_NAMES,
    load_candidate_window,
    read_cache_metadata,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)
from scripts.reconstruction_3d.windowed_tracklet_core import (
    LinkGates,
    LinkWeights,
    SolverConfig,
    WindowSpec,
    apply_fragment_identities,
    assert_tracking_invariants,
    best_to_observations,
    build_candidate_tracklet_ids,
    build_fragment_link_edges,
    build_fragments,
    candidate_windows_to_points,
    configuration_dict,
    enumerate_refractive_hypotheses,
    group_cross_view_tracklets,
    hypotheses_to_observations,
    make_overlapping_windows,
    reconcile_window_fragments,
    solve_fragment_identities,
    stable_hash,
    summarize_fragments,
    summarize_fragments_compact,
    track_observations_hungarian,
)


SCHEMA_VERSION = "windowed_tracklet_stitching_v2"


def _sha256(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


class _ResourceSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_rss = 0
        self.cpu_samples: list[float] = []

    def _sample(self) -> None:
        self.process.cpu_percent(None)
        while not self.stop_event.wait(self.interval_seconds):
            processes = [self.process]
            try:
                processes.extend(self.process.children(recursive=True))
            except psutil.Error:
                pass
            rss = 0
            cpu = 0.0
            for process in processes:
                try:
                    rss += int(process.memory_info().rss)
                    cpu += float(process.cpu_percent(None))
                except psutil.Error:
                    continue
            self.peak_rss = max(self.peak_rss, rss)
            self.cpu_samples.append(cpu)

    def __enter__(self) -> "_ResourceSampler":
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def summary(self) -> dict[str, float | int]:
        return {
            "peak_rss_bytes": int(self.peak_rss),
            "cpu_percent_mean_process_tree": (
                float(np.mean(self.cpu_samples)) if self.cpu_samples else np.nan
            ),
            "cpu_percent_peak_process_tree": (
                float(np.max(self.cpu_samples)) if self.cpu_samples else np.nan
            ),
            "resource_sample_count": int(len(self.cpu_samples)),
        }


@contextmanager
def _single_threaded_math() -> Iterable[None]:
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=1):
                yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _git_provenance(repo_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=repo_root, text=True
        )
        return {"commit": commit, "dirty": bool(status.strip()), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def _tree_content_provenance(paths: Iterable[Path]) -> dict[str, str]:
    """Hash the exact reusable inputs that can change checkpoint semantics."""
    values: dict[str, str] = {}
    for path in sorted({path.resolve() for path in paths}, key=str):
        if path.is_file():
            values[str(path)] = _sha256(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    values[str(child.resolve())] = _sha256(child)
    return values


def _checkpoint_valid(directory: Path, expected_hash: str) -> bool:
    complete_path = directory / "complete.json"
    if not complete_path.exists():
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        complete.get("schema_version") == SCHEMA_VERSION
        and complete.get("configuration_hash") == expected_hash
    ):
        return False
    files = complete.get("output_files", {})
    if not isinstance(files, dict) or not files:
        return False
    return all(
        (directory / name).is_file()
        and _sha256(directory / name) == expected_sha
        for name, expected_sha in files.items()
    )


def _run_geometry_window(task: Mapping[str, object]) -> dict[str, object]:
    """Spawn-safe worker for one complete window-local association stage."""
    spec = dict(task["spec"])
    output_dir = Path(str(task["output_dir"]))
    configuration_hash = str(task["configuration_hash"])
    if _checkpoint_valid(output_dir, configuration_hash):
        complete = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
        return {**complete, "reused": True}
    if output_dir.exists():
        stale = output_dir.with_name(
            output_dir.name + f".stale_{time.time_ns()}"
        )
        os.replace(output_dir, stale)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with _ResourceSampler() as resources, _single_threaded_math():
        windows = {
            view: load_candidate_window(
                Path(str(task["cache_paths"][view])),
                int(spec["context_start"]),
                int(spec["context_end"]),
            )
            for view in VIEW_NAMES
        }
        calibration = Calibration(
            str(task["calibration_model_dir"]), backend="refractive"
        )
        hypothesis_source = str(task["hypothesis_source"])
        output_tables: dict[str, pd.DataFrame] = {}
        tracklet_summaries: dict[str, object] = {}
        hypothesis_rows = 0
        if hypothesis_source == "legacy":
            points = candidate_windows_to_points(windows)
            best = compute_best_assignments_per_frame(
                calibration,
                points,
                xyz_bounds=(float(task["xyz_low"]), float(task["xyz_high"])),
                num_individuals=3,
                num_candidates=6,
                evaluation_batch_frames=int(task["evaluation_batch_frames"]),
            )
            observations = best_to_observations(
                best,
                windows,
                calibration,
                xyz_bounds=(float(task["xyz_low"]), float(task["xyz_high"])),
            )
            observations["cross_view_group_id"] = (
                "fw_" + observations.observation_id.astype(str)
            )
            observations["cross_view_group_support"] = 1
            observations["cross_view_group_trusted"] = False
        elif hypothesis_source == "frozen":
            tracklet_ids: dict[str, np.ndarray] = {}
            intervals: dict[tuple[int, int], tuple[int, int]] = {}
            for view_index, view in enumerate(VIEW_NAMES):
                ids, tracklet_rows, tracklet_summary = build_candidate_tracklet_ids(
                    windows[view]
                )
                tracklet_ids[view] = ids
                tracklet_summaries[view] = tracklet_summary
                output_tables[f"tracklets_{view}.parquet"] = tracklet_rows
                for tracklet_id, group in tracklet_rows.groupby("tracklet_id"):
                    intervals[(view_index, int(tracklet_id))] = (
                        int(group.frame.min()), int(group.frame.max())
                    )
            hypotheses = enumerate_refractive_hypotheses(
                windows,
                tracklet_ids,
                calibration,
                xyz_bounds=(float(task["xyz_low"]), float(task["xyz_high"])),
                batch_frames=int(task["evaluation_batch_frames"]),
                include_two_view=True,
            )
            hypothesis_rows = int(len(hypotheses))
            selected, group_candidates, conflicts, duplicates = group_cross_view_tracklets(
                hypotheses,
                mode=str(task["cross_view_mode"]),
                min_support_frames=int(task["min_group_support"]),
                min_valid_fraction=float(task["min_group_valid_fraction"]),
                max_ray_residual_mm2=float(task["max_group_residual_mm2"]),
                tracklet_intervals=intervals,
            )
            observations = hypotheses_to_observations(
                selected,
                windows,
                calibration,
                xyz_bounds=(float(task["xyz_low"]), float(task["xyz_high"])),
            )
            output_tables.update(
                {
                    "hypotheses.parquet": hypotheses,
                    "cross_view_group_candidates.parquet": group_candidates,
                    "cross_view_conflicts.parquet": conflicts,
                    "discarded_hypothesis_duplicates.parquet": duplicates,
                }
            )
        else:
            raise ValueError("hypothesis_source must be legacy or frozen")
        observations["window_index"] = int(spec["index"])
        observations["window_region"] = np.select(
            [
                observations["frame"] < int(spec["core_start"]),
                observations["frame"] > int(spec["core_end"]),
            ],
            ["left_context", "right_context"],
            default="core",
        )
        assert_tracking_invariants(observations)
        fragment_observations, fragments = build_fragments(
            observations,
            max_jump_mm=float(task["max_jump_mm"]),
            close_contact_mm=float(task["close_contact_mm"]),
        )
        prefix = f"w{int(spec['index']):04d}:"
        fragment_observations["fragment_id"] = (
            prefix + fragment_observations.fragment_id.astype(str)
        )
        fragments["fragment_id"] = prefix + fragments.fragment_id.astype(str)
        output_tables["observations.parquet"] = observations
        output_tables["fragment_observations.parquet"] = fragment_observations
        output_tables["fragments.parquet"] = fragments
        output_hashes: dict[str, str] = {}
        for name, table in output_tables.items():
            _atomic_parquet(table, output_dir / name)
            output_hashes[name] = _sha256(output_dir / name)
        _atomic_json(output_dir / "tracklet_summaries.json", tracklet_summaries)
        output_hashes["tracklet_summaries.json"] = _sha256(
            output_dir / "tracklet_summaries.json"
        )
    wall = float(time.perf_counter() - started)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": configuration_hash,
        "window": spec,
        "wall_time_seconds": wall,
        **resources.summary(),
        "observation_rows": int(len(observations)),
        "hypothesis_rows": hypothesis_rows,
        "fragment_count": int(len(fragments)),
        "tracklet_summaries": tracklet_summaries,
        "output_files": output_hashes,
        "reused": False,
    }
    _atomic_json(output_dir / "complete.json", complete)
    return complete


def _run_windows(
    tasks: list[dict[str, object]], workers: int
) -> list[dict[str, object]]:
    if workers <= 1:
        return [_run_geometry_window(task) for task in tasks]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_geometry_window, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: int(result["window"]["index"]))


def _reconcile_observations(
    checkpoint_dirs: list[Path], specs: list[Mapping[str, int]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = [
        pd.read_parquet(directory / "fragment_observations.parquet")
        for directory in checkpoint_dirs
    ]
    window_specs = [
        WindowSpec(
            index=int(spec["index"]),
            core_start=int(spec["core_start"]),
            core_end=int(spec["core_end"]),
            context_start=int(spec["context_start"]),
            context_end=int(spec["context_end"]),
        )
        for spec in specs
    ]
    merged, aliases, seams = reconcile_window_fragments(tables, window_specs)
    assert_tracking_invariants(merged)
    return merged, aliases, seams


def _write_filled_output(
    direct: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
    max_gap_frames: int,
) -> pd.DataFrame:
    trajectories: dict[str, pd.DataFrame] = {}
    direct_keys: set[tuple[int, str]] = set()
    for identity in ("ID_1", "ID_2", "ID_3"):
        rows = direct[direct.fish_id == identity][["frame", "X", "Y", "Z"]].copy()
        trajectories[identity] = rows
        direct_keys.update((int(frame), identity) for frame in rows.frame)
    filled = ensure_all_frames_and_fill_spline(
        trajectories,
        list(range(frame_start, frame_end + 1)),
        max_gap_frames=max_gap_frames,
    )
    parts = []
    for identity, table in filled.items():
        part = table.copy()
        part["fish_id"] = identity
        finite = np.isfinite(part[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)
        part["is_interpolated"] = [
            bool(is_finite and (int(frame), identity) not in direct_keys)
            for frame, is_finite in zip(part.frame, finite, strict=True)
        ]
        parts.append(part)
    output = pd.concat(parts, ignore_index=True).sort_values(["frame", "fish_id"])
    unresolved_frames = set(
        direct.loc[
            direct.get("identity_status", pd.Series("resolved", index=direct.index)).eq(
                "unresolved"
            ),
            "frame",
        ].astype(int)
    )
    output["identity_status"] = np.where(
        output.frame.astype(int).isin(unresolved_frames),
        "unresolved",
        np.where(
            np.isfinite(output[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1),
            "resolved",
            "missing",
        ),
    )
    if unresolved_frames:
        mask = output.frame.astype(int).isin(unresolved_frames)
        output.loc[mask, ["X", "Y", "Z"]] = np.nan
        output.loc[mask, "is_interpolated"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    output.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    return output


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        view: (args.cache_dir / f"{view}.candidates.h5").resolve()
        for view in VIEW_NAMES
    }
    missing = [str(path) for path in cache_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing candidate caches: " + ", ".join(missing))
    cache_metadata = {
        view: read_cache_metadata(path) for view, path in cache_paths.items()
    }
    calibration = Calibration(str(args.calibration_model_dir), backend="refractive")
    calibration_provenance = calibration.provenance
    if calibration.score_unit != "mm^2":
        raise RuntimeError(f"expected refractive mm^2 scores, got {calibration.score_unit}")

    gates = LinkGates(
        max_gap_frames=args.max_gap_frames,
        max_jump_mm_per_frame=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
    )
    weights = LinkWeights(
        residual_scale_mm2=args.residual_scale_mm2,
        mouth_ambiguity_margin=args.mouth_ambiguity_margin,
    )
    solver = SolverConfig(
        method=args.trajectory_solver,
        beam_width=args.beam_width,
        birth_cost=args.birth_cost,
        death_cost=args.death_cost,
        ambiguity_margin=args.identity_entry_margin,
        retention_margin=args.identity_retention_margin,
        resolution_margin=args.identity_resolution_margin,
        resolution_stability_frames=args.identity_resolution_stability_frames,
        deferred_horizon_frames=args.maximum_defer_horizon,
        identity_policy=args.identity_policy,
        path_cover_mode=args.path_cover_mode,
    )
    repo_root = Path(__file__).resolve().parents[2]
    code_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("windowed_tracklet_core.py").resolve(),
        Path(__file__).with_name("candidate_cache.py").resolve(),
        Path(__file__).with_name("triangulate_extra_candidates.py").resolve(),
        Path(__file__).with_name("calibration.py").resolve(),
    ]
    code_provenance = _tree_content_provenance(code_files)
    calibration_hashes = _tree_content_provenance([args.calibration_model_dir])
    if args.cross_view_mode == "multiframe" and args.hypothesis_source != "frozen":
        raise ValueError("multiframe cross-view mode requires --hypothesis-source frozen")
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "experiment": args.experiment,
        "round_tag": args.round_tag,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "window_mode": args.window_mode,
        "window_core_frames": args.window_core_frames,
        "window_context_frames": args.window_context_frames,
        "hypothesis_source": args.hypothesis_source,
        "cross_view_mode": args.cross_view_mode,
        "min_group_support": args.min_group_support,
        "min_group_valid_fraction": args.min_group_valid_fraction,
        "max_group_residual_mm2": args.max_group_residual_mm2,
        "fragment_link_mode": args.fragment_link_mode,
        "tail_continuity": args.tail_continuity,
        "trajectory_solver": args.trajectory_solver,
        "identity_policy": args.identity_policy,
        "seed": args.seed,
        "xyz_bounds_mm": [args.xyz_low, args.xyz_high],
        **configuration_dict(gates, weights, solver),
        "cache_metadata": cache_metadata,
        "calibration_provenance": calibration_provenance,
        "calibration_file_hashes": calibration_hashes,
        "code_file_hashes": code_provenance,
    }
    configuration_hash = stable_hash(configuration)
    _atomic_json(output_dir / "resolved_configuration.json", configuration)

    window_configuration = {
        key: configuration[key]
        for key in (
            "schema_version", "experiment", "round_tag", "frame_start", "frame_end",
            "window_mode", "window_core_frames", "window_context_frames",
            "hypothesis_source", "cross_view_mode", "min_group_support",
            "min_group_valid_fraction", "max_group_residual_mm2", "seed",
            "xyz_bounds_mm", "cache_metadata", "calibration_provenance",
            "calibration_file_hashes", "code_file_hashes",
        )
    }
    window_configuration["fragment_settings"] = {
        "max_jump_mm": args.max_jump_mm,
        "close_contact_mm": args.close_contact_mm,
    }
    window_configuration_hash = stable_hash(window_configuration)

    if args.window_mode == "single":
        specs = make_overlapping_windows(
            args.frame_start,
            args.frame_end,
            core_size=args.frame_end - args.frame_start + 1,
            overlap=args.window_context_frames,
            required_context=args.max_gap_frames,
        )
    else:
        specs = make_overlapping_windows(
            args.frame_start,
            args.frame_end,
            core_size=args.window_core_frames,
            overlap=args.window_context_frames,
            required_context=args.max_gap_frames,
        )
    checkpoint_root = output_dir / "checkpoints"
    tasks: list[dict[str, object]] = []
    checkpoint_dirs: list[Path] = []
    for spec in specs:
        directory = checkpoint_root / f"window_{spec.index:04d}_{spec.core_start}_{spec.core_end}"
        checkpoint_dirs.append(directory)
        tasks.append(
            {
                "spec": spec.__dict__,
                "output_dir": str(directory),
                "configuration_hash": window_configuration_hash,
                "cache_paths": {view: str(path) for view, path in cache_paths.items()},
                "calibration_model_dir": str(args.calibration_model_dir),
                "xyz_low": args.xyz_low,
                "xyz_high": args.xyz_high,
                "evaluation_batch_frames": args.evaluation_batch_frames,
                "hypothesis_source": args.hypothesis_source,
                "cross_view_mode": args.cross_view_mode,
                "min_group_support": args.min_group_support,
                "min_group_valid_fraction": args.min_group_valid_fraction,
                "max_group_residual_mm2": args.max_group_residual_mm2,
                "max_jump_mm": args.max_jump_mm,
                "close_contact_mm": args.close_contact_mm,
            }
        )
    started = time.perf_counter()
    with _ResourceSampler() as resources:
        window_results = _run_windows(tasks, args.workers)
        observations, aliases, seams = _reconcile_observations(
            checkpoint_dirs, [spec.__dict__ for spec in specs]
        )
        _atomic_parquet(observations, output_dir / "reconciled_observations.parquet")
        _atomic_parquet(aliases, output_dir / "boundary_aliases.parquet")
        _atomic_parquet(seams, output_dir / "boundary_seams.parquet")
        fragment_observations = observations
        fragments = (
            summarize_fragments(fragment_observations)
            if args.fragment_link_mode == "mouth"
            else summarize_fragments_compact(fragment_observations)
        )
        _atomic_parquet(fragment_observations, output_dir / "fragment_observations.parquet")
        _atomic_parquet(fragments, output_dir / "fragments.parquet")
        all_edges = (
            build_fragment_link_edges(
                fragments,
                gates=gates,
                weights=weights,
                tail_mode=args.tail_continuity,
            )
            if args.fragment_link_mode == "mouth"
            else pd.DataFrame(
                columns=[
                    "edge_id", "source_id", "target_id", "source_end",
                    "target_start", "gap_frames", "total_cost",
                ]
            )
        )
        _atomic_parquet(all_edges, output_dir / "candidate_link_edges.parquet")
        graph_files = {
            name: _sha256(output_dir / name)
            for name in (
                "fragment_observations.parquet",
                "fragments.parquet",
                "candidate_link_edges.parquet",
            )
        }
        graph_hash = stable_hash(graph_files)
        _atomic_json(
            output_dir / "frozen_graph_manifest.json",
            {"graph_hash": graph_hash, "files": graph_files},
        )
        if args.fragment_link_mode == "none":
            solver_edges = all_edges.iloc[0:0].copy()
            solver_result = None
            direct = track_observations_hungarian(
                fragment_observations,
                max_jump_mm=args.max_jump_mm,
                max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
                max_gap_frames=args.max_gap_frames,
            )
        else:
            solver_edges = all_edges
            solver_result = solve_fragment_identities(fragments, solver_edges, config=solver)
            direct = apply_fragment_identities(
                fragment_observations,
                solver_result,
                identity_policy=args.identity_policy,
            )
        invariants = assert_tracking_invariants(direct)
        _atomic_parquet(direct, output_dir / "direct_3d_observations.parquet")
        direct_csv = direct[[
            "frame", "fish_id", "identity_status", "X", "Y", "Z",
            "ray_residual_mm2", "fragment_id",
        ]].copy()
        direct_csv["is_interpolated"] = False
        direct_csv.to_csv(output_dir / "direct_3d_observations.csv", index=False)
        if args.fragment_link_mode == "mouth" and args.path_cover_mode == "restarts":
            filled = direct_csv.copy()
            temporary = output_dir / f"{args.output_tag}_3d.csv.partial"
            filled.to_csv(temporary, index=False)
            os.replace(temporary, output_dir / f"{args.output_tag}_3d.csv")
        else:
            filled = _write_filled_output(
                direct,
                output_dir / f"{args.output_tag}_3d.csv",
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                max_gap_frames=args.max_gap_frames,
            )
    total_wall = float(time.perf_counter() - started)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": configuration_hash,
        "experiment": args.experiment,
        "round_tag": args.round_tag,
        "output_tag": args.output_tag,
        "frames": [args.frame_start, args.frame_end],
        "windows": window_results,
        "total_wall_time_seconds": total_wall,
        **resources.summary(),
        "tracklet_summaries": {
            str(result["window"]["index"]): result.get("tracklet_summaries", {})
            for result in window_results
        },
        "observation_rows": int(len(observations)),
        "boundary_alias_candidates": int(len(aliases)),
        "boundary_aliases_accepted": int(aliases.get("accepted", pd.Series(dtype=bool)).sum()),
        "unresolved_seams": int(seams.get("unresolved", pd.Series(dtype=bool)).sum()),
        "fragment_count": int(len(fragments)),
        "link_edge_count": int(len(all_edges)),
        "frozen_graph_hash": graph_hash,
        "frozen_graph_file_hashes": graph_files,
        "solver_edge_count": int(len(solver_edges)),
        "solver": {
            "method": solver_result.method if solver_result else "baseline_hungarian",
            "objective": solver_result.objective if solver_result else None,
            "hypotheses_peak": solver_result.hypotheses_peak if solver_result else 1,
            "pruned_states": solver_result.pruned_states if solver_result else 0,
            "deterministic_tie_count": solver_result.deterministic_tie_count if solver_result else 0,
            "unresolved_intervals": list(solver_result.unresolved_intervals) if solver_result else [],
            "selected_edges": list(solver_result.selected_edges) if solver_result else [],
            "paths": [list(path) for path in solver_result.paths] if solver_result else [],
            "status": solver_result.status if solver_result else "baseline",
        },
        "invariants": invariants,
        "direct_rows": int(len(direct)),
        "filled_rows": int(len(filled)),
        "output_files": {
            "direct_parquet": str((output_dir / "direct_3d_observations.parquet").resolve()),
            "filled_csv": str((output_dir / f"{args.output_tag}_3d.csv").resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(),
            "git": _git_provenance(repo_root),
        },
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--round-tag", required=True)
    parser.add_argument("--paths-config", default="configs/paths.local.toml")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tag", default="windowed_tracklet")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=348599)
    parser.add_argument("--xyz-low", type=float, default=0.0)
    parser.add_argument("--xyz-high", type=float, default=200.0)
    parser.add_argument("--window-mode", choices=("single", "overlap"), default="overlap")
    parser.add_argument("--window-core-frames", type=int, default=10000)
    parser.add_argument("--window-context-frames", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluation-batch-frames", type=int, default=256)
    parser.add_argument(
        "--hypothesis-source",
        choices=("legacy", "frozen"),
        default="legacy",
        help="legacy exact framewise selector or frozen all-hypothesis table",
    )
    parser.add_argument("--cross-view-mode", choices=("framewise", "multiframe"), default="framewise")
    parser.add_argument("--min-group-support", type=int, default=3)
    parser.add_argument("--min-group-valid-fraction", type=float, default=0.95)
    parser.add_argument("--max-group-residual-mm2", type=float, default=10.0)
    parser.add_argument("--fragment-link-mode", choices=("none", "mouth"), default="none")
    parser.add_argument("--tail-continuity", choices=("off", "conditional"), default="off")
    parser.add_argument(
        "--trajectory-solver",
        choices=("sequential_hungarian", "beam", "global_dag"),
        default="sequential_hungarian",
    )
    parser.add_argument(
        "--identity-policy",
        choices=("solver_only", "immediate", "deferred"),
        default="solver_only",
    )
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--birth-cost", type=float, default=3.0)
    parser.add_argument("--death-cost", type=float, default=3.0)
    parser.add_argument(
        "--path-cover-mode",
        choices=("restarts", "exact_three"),
        default="restarts",
    )
    parser.add_argument("--identity-entry-margin", type=float, default=0.5)
    parser.add_argument("--identity-retention-margin", type=float, default=2.0)
    parser.add_argument("--identity-resolution-margin", type=float, default=1.0)
    parser.add_argument("--identity-resolution-stability-frames", type=int, default=10)
    parser.add_argument("--maximum-defer-horizon", type=int, default=250)
    parser.add_argument("--max-jump-mm", type=float, default=8.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--close-contact-mm", type=float, default=12.0)
    parser.add_argument("--residual-scale-mm2", type=float, default=2.5)
    parser.add_argument("--mouth-ambiguity-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.calibration_model_dir is None:
        paths = load_local_paths(args.paths_config)
        if paths is None or paths.calibration_model_dir is None:
            raise ValueError(
                "calibration_model_dir is missing; configure it or pass "
                "--calibration-model-dir"
            )
        args.calibration_model_dir = paths.calibration_model_dir
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
