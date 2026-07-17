from __future__ import annotations

"""Solve one immutable windowed fragment graph without rebuilding geometry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable

import pandas as pd

from scripts.reconstruction_3d.windowed_tracklet_core import (
    SolverConfig,
    apply_fragment_identities,
    assert_tracking_invariants,
    solve_fragment_identities,
)
from scripts.reconstruction_3d.windowed_tracklet_stitching import _ResourceSampler


REQUIRED_GRAPH_FILES = (
    "fragment_observations.parquet",
    "fragments.parquet",
    "candidate_link_edges.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def solve_frozen_graph(args: argparse.Namespace) -> dict[str, object]:
    graph_dir = args.graph_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_GRAPH_FILES if not (graph_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen graph lacks {missing}: {graph_dir}")
    graph_hashes = {name: _sha256(graph_dir / name) for name in REQUIRED_GRAPH_FILES}
    graph_hash = hashlib.sha256(
        json.dumps(graph_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    observations = pd.read_parquet(graph_dir / "fragment_observations.parquet")
    fragments = pd.read_parquet(graph_dir / "fragments.parquet")
    edges = pd.read_parquet(graph_dir / "candidate_link_edges.parquet")
    config = SolverConfig(
        method=args.trajectory_solver,
        beam_width=args.beam_width,
        identity_policy=args.identity_policy,
        ambiguity_margin=args.identity_entry_margin,
        retention_margin=args.identity_retention_margin,
        resolution_margin=args.identity_resolution_margin,
        resolution_stability_frames=args.identity_resolution_stability_frames,
        deferred_horizon_frames=args.maximum_defer_horizon,
        path_cover_mode=args.path_cover_mode,
    )
    started = time.perf_counter()
    with _ResourceSampler() as resources:
        result = solve_fragment_identities(fragments, edges, config=config)
        direct = apply_fragment_identities(
            observations, result, identity_policy=args.identity_policy
        )
        invariants = assert_tracking_invariants(direct)
        _atomic_parquet(direct, output_dir / "direct_3d_observations.parquet")
        path_rows = [
            {
                "path_index": path_index,
                "fish_id": f"ID_{path_index + 1}",
                "path_order": order,
                "fragment_id": fragment_id,
            }
            for path_index, path in enumerate(result.paths)
            for order, fragment_id in enumerate(path)
        ]
        _atomic_parquet(pd.DataFrame(path_rows), output_dir / "trajectory_paths.parquet")
        selected = edges[edges.edge_id.astype(str).isin(result.selected_edges)].copy()
        _atomic_parquet(selected, output_dir / "selected_link_edges.parquet")
        _atomic_parquet(
            pd.DataFrame.from_records(result.unresolved_intervals),
            output_dir / "identity_ambiguity_intervals.parquet",
        )
    summary = {
        "graph_dir": str(graph_dir),
        "graph_file_hashes": graph_hashes,
        "graph_hash": graph_hash,
        "solver_config": config.__dict__,
        "method": result.method,
        "status": result.status,
        "feasible": result.feasible,
        "objective": result.objective,
        "selected_edges": list(result.selected_edges),
        "paths": [list(path) for path in result.paths],
        "hypotheses_peak": result.hypotheses_peak,
        "pruned_states": result.pruned_states,
        "explored_states": result.explored_states,
        "deterministic_tie_count": result.deterministic_tie_count,
        "unresolved_intervals": list(result.unresolved_intervals),
        "wall_time_seconds": float(time.perf_counter() - started),
        **resources.summary(),
        "invariants": invariants,
        "direct_rows": int(len(direct)),
        "direct_output_sha256": _sha256(output_dir / "direct_3d_observations.parquet"),
    }
    _atomic_json(output_dir / "solver_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-solver",
        choices=("sequential_hungarian", "beam", "global_dag"),
        required=True,
    )
    parser.add_argument(
        "--identity-policy",
        choices=("solver_only", "immediate", "deferred"),
        default="solver_only",
    )
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument(
        "--path-cover-mode",
        choices=("exact_three", "restarts"),
        default="exact_three",
    )
    parser.add_argument("--identity-entry-margin", type=float, default=0.5)
    parser.add_argument("--identity-retention-margin", type=float, default=2.0)
    parser.add_argument("--identity-resolution-margin", type=float, default=1.0)
    parser.add_argument("--identity-resolution-stability-frames", type=int, default=10)
    parser.add_argument("--maximum-defer-horizon", type=int, default=250)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    print(json.dumps(solve_frozen_graph(args), indent=2, default=str))


if __name__ == "__main__":
    main()
