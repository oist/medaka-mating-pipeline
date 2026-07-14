from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.tracking.prepare_locked_global_stitching_inputs import prepare_locked_inputs


def test_prepare_locked_inputs_namespaces_filters_and_preserves_xyz(tmp_path):
    assignments = pd.DataFrame(
        [
            {"node_id": "top:0", "view": "top_cam", "component_id": "top:0", "frame_start": 8, "frame_end": 12, "n_rows": 5},
            {"node_id": "side:0", "view": "side1_cam", "component_id": "side:0", "frame_start": 20, "frame_end": 21, "n_rows": 2},
        ]
    )
    dispositions = pd.DataFrame(
        [
            {"frame": 9, "tracklet_node": "top:0"},
            {"frame": 10, "tracklet_node": "top:0"},
            {"frame": 11, "tracklet_node": "top:0"},
            {"frame": 20, "tracklet_node": "side:0"},
        ]
    )
    value = float(np.nextafter(0.1234567890123456, 1.0))
    canonical = pd.DataFrame(
        [
            {"frame": 9, "fish_id": 0, "X": value, "Y": 2.0, "Z": 3.0},
            {"frame": 10, "fish_id": 0, "X": 4.0, "Y": 5.0, "Z": 6.0},
            {"frame": 20, "fish_id": 0, "X": 7.0, "Y": 8.0, "Z": 9.0},
        ]
    )
    edges = pd.DataFrame(
        [{"node_a": "top:0", "node_b": "side:0"}, {"node_a": "top:0", "node_b": "top:0"}]
    )
    best = {str(frame): {"groups": {}, "errors": {}} for frame in (9, 10, 20)}
    paths = {
        "assignments": tmp_path / "assignments.csv",
        "dispositions": tmp_path / "dispositions.csv",
        "canonical": tmp_path / "canonical.csv",
        "edges": tmp_path / "edges.csv",
        "best": tmp_path / "best.json",
    }
    assignments.to_csv(paths["assignments"], index=False)
    dispositions.to_csv(paths["dispositions"], index=False)
    canonical.to_csv(paths["canonical"], index=False, float_format="%.17g")
    edges.to_csv(paths["edges"], index=False)
    paths["best"].write_text(json.dumps(best), encoding="utf-8")

    output = tmp_path / "locked"
    summary = prepare_locked_inputs(
        assignments_path=paths["assignments"],
        dispositions_path=paths["dispositions"],
        best_pre_link_path=paths["best"],
        canonical_3d_path=paths["canonical"],
        cross_view_edges_path=paths["edges"],
        output_dir=output,
        frame_start=9,
        frame_end=11,
        namespace="heldout",
    )

    staged_assignments = pd.read_csv(output / "tracklet_assignments.csv")
    staged_xyz = pd.read_csv(output / "canonical_3d.csv", float_precision="round_trip")
    source_xyz = pd.read_csv(paths["canonical"], float_precision="round_trip").query("9 <= frame <= 11")
    assert staged_assignments.component_id.tolist() == ["heldout::top:0"]
    assert staged_assignments.frame_start.tolist() == [9]
    assert staged_assignments.frame_end.tolist() == [11]
    assert staged_assignments.n_rows.tolist() == [3]
    assert np.array_equal(
        source_xyz[["X", "Y", "Z"]].to_numpy(float),
        staged_xyz[["X", "Y", "Z"]].to_numpy(float),
        equal_nan=True,
    )
    assert summary["xyz_exact_equal"] is True
    assert summary["counts"]["cross_view_edges"] == 1
