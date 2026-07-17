"""Finalize a checkpointed windowed baseline from its immutable graph files."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import pandas as pd

from scripts.reconstruction_3d.windowed_tracklet_core import (
    assert_tracking_invariants,
    stable_hash,
    track_observations_hungarian,
)
from scripts.reconstruction_3d.windowed_tracklet_stitching import (
    SCHEMA_VERSION,
    _ResourceSampler,
    _atomic_json,
    _atomic_parquet,
    _git_provenance,
    _sha256,
    _write_filled_output,
)


GRAPH_FILES = (
    "fragment_observations.parquet",
    "fragments.parquet",
    "candidate_link_edges.parquet",
)


def finalize(
    output_dir: Path,
    *,
    output_tag_override: str | None,
    recovery_reason: str,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    configuration_path = output_dir / "resolved_configuration.json"
    graph_manifest_path = output_dir / "frozen_graph_manifest.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if configuration.get("fragment_link_mode") != "none":
        raise ValueError("graph finalizer currently supports fragment_link_mode=none")
    actual_hashes = {name: _sha256(output_dir / name) for name in GRAPH_FILES}
    if actual_hashes != manifest.get("files"):
        raise ValueError("immutable graph file hash mismatch")
    if stable_hash(actual_hashes) != manifest.get("graph_hash"):
        raise ValueError("immutable graph aggregate hash mismatch")

    observations = pd.read_parquet(output_dir / "fragment_observations.parquet")
    fragments = pd.read_parquet(output_dir / "fragments.parquet")
    edges = pd.read_parquet(output_dir / "candidate_link_edges.parquet")
    aliases = pd.read_parquet(output_dir / "boundary_aliases.parquet")
    seams = pd.read_parquet(output_dir / "boundary_seams.parquet")
    started = time.perf_counter()
    with _ResourceSampler() as resources:
        direct = track_observations_hungarian(
            observations,
            max_jump_mm=float(configuration["LinkGates"]["max_jump_mm_per_frame"]),
            max_jump_hard_cap_mm=float(
                configuration["LinkGates"]["max_jump_hard_cap_mm"]
            ),
            max_gap_frames=int(configuration["LinkGates"]["max_gap_frames"]),
        )
        invariants = assert_tracking_invariants(direct)
        _atomic_parquet(direct, output_dir / "direct_3d_observations.parquet")
        direct_csv = direct[
            [
                "frame", "fish_id", "identity_status", "X", "Y", "Z",
                "ray_residual_mm2", "fragment_id",
            ]
        ].copy()
        direct_csv["is_interpolated"] = False
        direct_csv_path = output_dir / "direct_3d_observations.csv"
        partial = direct_csv_path.with_suffix(".csv.partial")
        direct_csv.to_csv(partial, index=False)
        os.replace(partial, direct_csv_path)
        output_tag = str(
            output_tag_override
            or configuration.get("output_tag", "windowed_tracklet")
        )
        filled = _write_filled_output(
            direct,
            output_dir / f"{output_tag}_3d.csv",
            frame_start=int(configuration["frame_start"]),
            frame_end=int(configuration["frame_end"]),
            max_gap_frames=int(configuration["LinkGates"]["max_gap_frames"]),
        )
    finalization_wall = float(time.perf_counter() - started)

    checkpoint_summaries = []
    for path in sorted((output_dir / "checkpoints").glob("*/complete.json")):
        checkpoint_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    pre_finalization_seconds = max(
        0.0,
        graph_manifest_path.stat().st_mtime - configuration_path.stat().st_mtime,
    )
    finalizer_path = Path(__file__).resolve()
    direct_linker_path = finalizer_path.with_name("windowed_tracklet_core.py")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "recovered_complete_from_immutable_graph",
        "recovery_reason": recovery_reason,
        "configuration_hash": stable_hash(configuration),
        "experiment": configuration["experiment"],
        "round_tag": configuration["round_tag"],
        "output_tag": output_tag,
        "frames": [configuration["frame_start"], configuration["frame_end"]],
        "windows": checkpoint_summaries,
        "window_count": len(checkpoint_summaries),
        "pre_finalization_wall_from_file_timestamps_seconds": pre_finalization_seconds,
        "finalization_wall_time_seconds": finalization_wall,
        "total_observed_wall_from_file_timestamps_seconds": (
            pre_finalization_seconds + finalization_wall
        ),
        **resources.summary(),
        "observation_rows": int(len(observations)),
        "boundary_alias_candidates": int(len(aliases)),
        "boundary_aliases_accepted": int(
            aliases.get("accepted", pd.Series(dtype=bool)).sum()
        ),
        "unresolved_seams": int(
            seams.get("unresolved", pd.Series(dtype=bool)).sum()
        ),
        "fragment_count": int(len(fragments)),
        "link_edge_count": int(len(edges)),
        "frozen_graph_hash": manifest["graph_hash"],
        "frozen_graph_file_hashes": actual_hashes,
        "graph_hash_verified": True,
        "solver": {
            "method": "baseline_hungarian",
            "objective": None,
            "hypotheses_peak": 1,
            "pruned_states": 0,
            "deterministic_tie_count": 0,
            "unresolved_intervals": [],
            "selected_edges": [],
            "paths": [],
            "status": "baseline",
        },
        "invariants": invariants,
        "direct_rows": int(len(direct)),
        "filled_rows": int(len(filled)),
        "output_files": {
            "direct_parquet": str(
                (output_dir / "direct_3d_observations.parquet").resolve()
            ),
            "filled_csv": str((output_dir / f"{output_tag}_3d.csv").resolve()),
        },
        "finalizer": {
            "path": str(finalizer_path),
            "sha256": _sha256(finalizer_path),
            "direct_linker_path": str(direct_linker_path),
            "direct_linker_sha256": _sha256(direct_linker_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(),
            "git": _git_provenance(Path.cwd()),
        },
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument(
        "--recovery-reason",
        default="supervisor timeout after immutable graph creation",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.output_dir,
                output_tag_override=args.output_tag,
                recovery_reason=args.recovery_reason,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
