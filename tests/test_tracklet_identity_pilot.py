from __future__ import annotations

import json

import pandas as pd

from scripts.tracking.tracklet_identity_pilot import (
    PilotConfig,
    assign_methods,
    build_edges,
    build_nodes,
    build_parser,
    expand_per_frame,
    load_events,
    run,
    summarize_metrics,
)


def _synthetic_tracklets() -> pd.DataFrame:
    rows = []
    specs = [
        ("a1", 0, 0.0, "ID_1"),
        ("a2", 8, 8.0, "ID_1"),
        ("b1", 0, 50.0, "ID_2"),
        ("b2", 8, 50.0, "ID_2"),
        ("c1", 0, 100.0, "ID_3"),
        ("c2", 8, 100.0, "ID_3"),
    ]
    for tracklet_id, start, y, identity in specs:
        global_id = int(identity.split("_")[1]) - 1
        for frame in range(start, start + 5):
            rows.append(
                {
                    "frame": frame,
                    "view": "top_cam",
                    "tracklet_id": tracklet_id,
                    "x": float(frame),
                    "y": y,
                    "score": 0.95,
                    "global_id": global_id,
                    "bind_consistency": 1.0,
                    "bind_resid": 2.0,
                    "bind_margin": 40.0,
                    "reference_identity": identity,
                    "source_path": "synthetic.csv",
                }
            )
    return pd.DataFrame(rows)


def _synthetic_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "cross_000",
                "kind": "manual_crossing",
                "pair": "ID_1|ID_2",
                "start": 5,
                "end": 7,
                "n_frames": 3,
                "min_distance": 5.0,
                "source": "test",
            }
        ]
    )


def _synthetic_appearance() -> pd.DataFrame:
    rows = []
    for tracklet_id, identity in {
        "a1": "ID_1",
        "a2": "ID_1",
        "b1": "ID_2",
        "b2": "ID_2",
        "c1": "ID_3",
        "c2": "ID_3",
    }.items():
        rows.append(
            {
                "view": "top_cam",
                "tracklet_id": tracklet_id,
                "predicted_identity": identity,
                "confidence": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_graph_scores_connect_same_identity_tracklets():
    cfg = PilotConfig(max_gap_frames=10, max_speed_px_per_frame=20.0)
    nodes = build_nodes(_synthetic_tracklets(), _synthetic_events(), _synthetic_appearance())
    edges = build_edges(nodes, _synthetic_events(), cfg)

    edge = edges[(edges["node_a"] == "top_cam:a1") & (edges["node_b"] == "top_cam:a2")].iloc[0]
    assert edge["geometry_same"]
    assert edge["appearance_same"]
    assert edge["geometry_score"] > 0.5
    assert edge["hybrid_score"] > 0.5

    cross_edge = edges[(edges["node_a"] == "top_cam:a1") & (edges["node_b"] == "top_cam:b2")].iloc[0]
    assert cross_edge["geometry_conflict"]
    assert cross_edge["hybrid_score"] < edge["hybrid_score"]


def test_method_assignments_and_metrics_are_written_from_pure_tables():
    cfg = PilotConfig(max_gap_frames=10, max_speed_px_per_frame=20.0)
    tracklets = _synthetic_tracklets()
    events = _synthetic_events()
    nodes = build_nodes(tracklets, events, _synthetic_appearance())
    edges = build_edges(nodes, events, cfg)
    assignments, selected = assign_methods(nodes, edges, cfg)
    per_frame = expand_per_frame(tracklets, assignments)
    metrics, event_reconnections, reference_metrics = summarize_metrics(
        nodes, assignments, per_frame, selected, events, cfg
    )

    hybrid = assignments[assignments["method"] == "hybrid"].set_index("node_id")
    assert hybrid.loc["top_cam:a1", "assigned_identity"] == "ID_1"
    assert hybrid.loc["top_cam:a2", "assigned_identity"] == "ID_1"
    assert "hybrid" in set(metrics["method"])
    assert not event_reconnections.empty
    assert not reference_metrics.empty
    assert reference_metrics.set_index("method").loc["hybrid", "best_remapped_tracklet_accuracy"] == 1.0


def test_cli_pilot_writes_reproducible_outputs(tmp_path):
    tracklets = _synthetic_tracklets()
    appearance = _synthetic_appearance()
    events = _synthetic_events()
    tracklets_csv = tmp_path / "top_cam.tracklet_bound.csv"
    appearance_csv = tmp_path / "appearance.csv"
    events_csv = tmp_path / "events.csv"
    out_dir = tmp_path / "pilot"
    tracklets.to_csv(tracklets_csv, index=False)
    appearance.to_csv(appearance_csv, index=False)
    events.to_csv(events_csv, index=False)

    args = build_parser().parse_args(
        [
            "--tracklets",
            f"top_cam={tracklets_csv}",
            "--appearance",
            str(appearance_csv),
            "--events-csv",
            str(events_csv),
            "--windows",
            "0-12",
            "--out-dir",
            str(out_dir),
            "--max-gap-frames",
            "10",
            "--max-speed-px-per-frame",
            "20",
        ]
    )
    summary = run(args)

    assert summary["n_nodes"] == 6
    assert (out_dir / "tracklet_nodes.csv").exists()
    assert (out_dir / "tracklet_edges.csv").exists()
    assert (out_dir / "method_assignments.csv").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "recommendation.md").exists()
    saved_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["best_method_so_far"] in {
        "geometry_only",
        "appearance_only",
        "hybrid",
        "udmt_inspired",
        "temporal_dynamics",
    }


def test_event_loader_accepts_start_frame_columns(tmp_path):
    events_csv = tmp_path / "events.csv"
    pd.DataFrame(
        [{"start_frame": 10, "end_frame": 20, "track_id_a": "A", "track_id_b": "B"}]
    ).to_csv(events_csv, index=False)
    events = load_events(events_csv, None, PilotConfig())
    assert list(events[["start", "end"]].iloc[0]) == [10, 20]
    assert events.iloc[0]["n_frames"] == 11
