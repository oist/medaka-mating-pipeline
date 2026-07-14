from __future__ import annotations

"""Stage a frame-locked, provenance-namespaced global-stitching benchmark.

The source tracklet-first artifacts are read-only.  Component identifiers are
namespaced to match geometry-contrastive sample provenance, while node spans are
recomputed from dispositions inside the requested inclusive frame interval.
Canonical XYZ values are filtered, never transformed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _require(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(table.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(sorted(missing))}")


def _namespace(value: object, prefix: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("component_id must be non-empty")
    marker = f"{prefix}::"
    return text if text.startswith(marker) else marker + text


def prepare_locked_inputs(
    *,
    assignments_path: Path,
    dispositions_path: Path,
    best_pre_link_path: Path,
    canonical_3d_path: Path,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
    namespace: str,
    cross_view_edges_path: Path | None = None,
) -> dict[str, object]:
    if frame_end < frame_start:
        raise ValueError("frame_end must be >= frame_start")
    output_dir.mkdir(parents=True, exist_ok=False)

    assignments = pd.read_csv(assignments_path)
    dispositions = pd.read_csv(dispositions_path)
    canonical = pd.read_csv(canonical_3d_path, float_precision="round_trip")
    _require(assignments, {"node_id", "component_id", "view"}, "assignments")
    _require(dispositions, {"frame", "tracklet_node"}, "dispositions")
    _require(canonical, {"frame", "X", "Y", "Z"}, "canonical 3D")

    dispositions["frame"] = pd.to_numeric(dispositions["frame"], errors="raise").astype(int)
    locked_dispositions = dispositions[
        dispositions.frame.between(frame_start, frame_end, inclusive="both")
    ].copy()
    if locked_dispositions.empty:
        raise ValueError("locked disposition interval is empty")
    locked_dispositions["tracklet_node"] = locked_dispositions.tracklet_node.astype(str)

    spans = (
        locked_dispositions.groupby("tracklet_node", as_index=False)
        .agg(frame_start=("frame", "min"), frame_end=("frame", "max"), n_rows=("frame", "size"))
        .rename(columns={"tracklet_node": "node_id"})
    )
    source_columns = [column for column in assignments.columns if column not in {"frame_start", "frame_end", "n_rows"}]
    locked_assignments = assignments[source_columns].merge(spans, on="node_id", how="inner", validate="one_to_one")
    if locked_assignments.empty:
        raise ValueError("no assignment nodes overlap the locked dispositions")
    locked_assignments["component_id"] = locked_assignments.component_id.map(
        lambda value: _namespace(value, namespace)
    )

    best_payload = json.loads(best_pre_link_path.read_text(encoding="utf-8"))
    best_frames = best_payload.get("frames", best_payload)
    locked_best = {
        str(frame): value
        for frame, value in best_frames.items()
        if frame_start <= int(frame) <= frame_end
    }
    if not locked_best:
        raise ValueError("best-pre-link interval is empty")
    if "frames" in best_payload:
        locked_best_payload = {**best_payload, "frames": locked_best}
    else:
        locked_best_payload = locked_best

    canonical["frame"] = pd.to_numeric(canonical["frame"], errors="raise").astype(int)
    locked_canonical = canonical[
        canonical.frame.between(frame_start, frame_end, inclusive="both")
    ].copy()
    if locked_canonical.empty:
        raise ValueError("canonical 3D interval is empty")

    outputs = {
        "assignments": output_dir / "tracklet_assignments.csv",
        "dispositions": output_dir / "detection_dispositions.csv",
        "best_pre_link": output_dir / "best_pre_link.json",
        "canonical_3d": output_dir / "canonical_3d.csv",
    }
    locked_assignments.to_csv(outputs["assignments"], index=False)
    locked_dispositions.to_csv(outputs["dispositions"], index=False)
    outputs["best_pre_link"].write_text(
        json.dumps(locked_best_payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    locked_canonical.to_csv(outputs["canonical_3d"], index=False, float_format="%.17g")
    written_canonical = pd.read_csv(outputs["canonical_3d"], float_precision="round_trip")
    if len(written_canonical) != len(locked_canonical) or not np.array_equal(
        locked_canonical[["X", "Y", "Z"]].to_numpy(float),
        written_canonical[["X", "Y", "Z"]].to_numpy(float),
        equal_nan=True,
    ):
        raise RuntimeError("canonical XYZ changed while staging the locked interval")

    edge_count = 0
    if cross_view_edges_path is not None:
        edges = pd.read_csv(cross_view_edges_path)
        _require(edges, {"node_a", "node_b"}, "cross-view edges")
        valid_nodes = set(locked_assignments.node_id.astype(str))
        locked_edges = edges[
            edges.node_a.astype(str).isin(valid_nodes)
            & edges.node_b.astype(str).isin(valid_nodes)
        ].copy()
        outputs["cross_view_edges"] = output_dir / "cross_view_edges.csv"
        locked_edges.to_csv(outputs["cross_view_edges"], index=False)
        edge_count = int(len(locked_edges))

    summary: dict[str, object] = {
        "schema_version": "locked_global_stitching_inputs_v1",
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "namespace": namespace,
        "source_paths": {
            "assignments": str(assignments_path),
            "dispositions": str(dispositions_path),
            "best_pre_link": str(best_pre_link_path),
            "canonical_3d": str(canonical_3d_path),
            "cross_view_edges": str(cross_view_edges_path) if cross_view_edges_path else None,
        },
        "counts": {
            "assignment_nodes": int(len(locked_assignments)),
            "components": int(locked_assignments.component_id.nunique()),
            "disposition_rows": int(len(locked_dispositions)),
            "best_frames": int(len(locked_best)),
            "canonical_3d_rows": int(len(locked_canonical)),
            "cross_view_edges": edge_count,
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "xyz_transform": "none; rows filtered by frame only",
        "xyz_exact_equal": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--dispositions", type=Path, required=True)
    parser.add_argument("--best-pre-link", type=Path, required=True)
    parser.add_argument("--canonical-3d", type=Path, required=True)
    parser.add_argument("--cross-view-edges", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = prepare_locked_inputs(
        assignments_path=args.assignments,
        dispositions_path=args.dispositions,
        best_pre_link_path=args.best_pre_link,
        canonical_3d_path=args.canonical_3d,
        cross_view_edges_path=args.cross_view_edges,
        output_dir=args.output_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        namespace=args.namespace,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
