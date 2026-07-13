import pandas as pd
import numpy as np

from scripts.reconstruction_3d.multiview_tracklet_tracking import (
    apply_targeted_appearance_fallback,
    build_identified_2d_rows,
    build_component_hints,
    build_constrained_tracklet_graph,
    continuity_vetoes,
    load_silhouette_link_edges,
    label_components_from_tracked_geometry,
    match_partial_tracks_to_slots,
    veto_conflicting_sources,
)


def test_temporal_geometry_votes_align_to_canonical_seed_labels():
    rows = []
    for frame in range(4):
        rows.extend(
            [
                {
                    "frame": frame,
                    "source_idx": 0,
                    "view": "top_cam",
                    "node_id": "top:a",
                    "geometry_usable": True,
                },
                {
                    "frame": frame,
                    "source_idx": 1,
                    "view": "top_cam",
                    "node_id": "top:b",
                    "geometry_usable": True,
                },
            ]
        )
    observations = pd.DataFrame(rows)
    components = {"top:a": "ca", "top:b": "cb"}
    # Raw geometry ID_1 follows source 1 and raw ID_2 follows source 0.
    tracked = {frame: {"id_mapping": [1, 0, 2]} for frame in range(4)}
    labels, decisions = label_components_from_tracked_geometry(
        observations,
        components,
        tracked,
        {"ca": "ID_1", "cb": "ID_2"},
        alignment_frame=0,
        min_frames=3,
    )
    assert labels == {"ca": "ID_1", "cb": "ID_2"}
    assert decisions.accepted.all()


def test_temporal_geometry_votes_reject_component_with_mixed_id():
    observations = pd.DataFrame(
        [
            {
                "frame": frame,
                "source_idx": 0 if frame < 3 else 1,
                "view": "top_cam",
                "node_id": "top:mixed",
                "geometry_usable": True,
            }
            for frame in range(4)
        ]
        + [
            {
                "frame": 0,
                "source_idx": 0,
                "view": "side1_cam",
                "node_id": "seed:a",
                "geometry_usable": True,
            },
            {
                "frame": 0,
                "source_idx": 1,
                "view": "side1_cam",
                "node_id": "seed:b",
                "geometry_usable": True,
            },
        ]
    )
    tracked = {frame: {"id_mapping": [0, 1, 2]} for frame in range(4)}
    labels, decisions = label_components_from_tracked_geometry(
        observations,
        {"top:mixed": "mixed", "seed:a": "ca", "seed:b": "cb"},
        tracked,
        {"ca": "ID_1", "cb": "ID_2"},
        alignment_frame=0,
        min_frames=3,
    )
    assert "mixed" not in labels
    mixed = decisions[decisions.component_id.eq("mixed")].iloc[0]
    assert not bool(mixed.accepted)
    assert mixed.reason == "mixed_temporal_geometry_vote"


def _observations():
    rows = []
    for frame in range(10):
        top_a = "top:a1" if frame < 5 else "top:a2"
        for source, nodes in {
            0: [("top_cam", top_a), ("side1_cam", "side:s1")],
            1: [("top_cam", "top:b"), ("side1_cam", "side:s2")],
            2: [("top_cam", "top:c"), ("side1_cam", "side:s3")],
        }.items():
            for view, node in nodes:
                rows.append({
                    "frame": frame,
                    "source_idx": source,
                    "view": view,
                    "node_id": node,
                    "residual": 10.0,
                    "geometry_usable": True,
                    "hardcore": False,
                })
    return pd.DataFrame(rows)


def test_cross_view_spanning_tracklet_transfers_identity_across_view_break():
    obs = _observations()
    nodes = sorted(obs.node_id.unique())
    component, edges = build_constrained_tracklet_graph(
        obs, nodes, min_overlap_frames=3, consistency_min=0.95
    )

    # side:s1 spans the top-view break, so both top fragments become one fish.
    assert component["top:a1"] == component["side:s1"]
    assert component["top:a2"] == component["side:s1"]
    assert component["top:a1"] != component["top:b"]
    assert int(edges["accepted"].sum()) >= 4

    labels = {
        component["top:a1"]: "ID_1",
        component["top:b"]: "ID_2",
        component["top:c"]: "ID_3",
    }
    hints, conflicts = build_component_hints(obs, component, labels)
    assert conflicts.empty
    assert hints[0] == {0: "ID_1", 1: "ID_2", 2: "ID_3"}
    assert hints[9] == {0: "ID_1", 1: "ID_2", 2: "ID_3"}


def test_different_simultaneous_fish_are_never_merged():
    obs = _observations()
    # Add misleading same-source support between tracklets that are normally
    # observed as different fish. The cannot-link evidence must win.
    fake = []
    for frame in range(3):
        fake.extend([
            {"frame": 100 + frame, "source_idx": 0, "view": "top_cam", "node_id": "top:a1", "residual": 1.0, "geometry_usable": True, "hardcore": False},
            {"frame": 100 + frame, "source_idx": 0, "view": "side1_cam", "node_id": "side:s2", "residual": 1.0, "geometry_usable": True, "hardcore": False},
        ])
    obs = pd.concat([obs, pd.DataFrame(fake)], ignore_index=True)
    nodes = sorted(obs.node_id.unique())
    component, edges = build_constrained_tracklet_graph(
        obs, nodes, min_overlap_frames=3, consistency_min=0.2
    )
    assert component["top:a1"] != component["side:s2"]
    edge = edges[
        ((edges.node_a == "side:s2") & (edges.node_b == "top:a1"))
        | ((edges.node_b == "side:s2") & (edges.node_a == "top:a1"))
    ].iloc[0]
    assert not bool(edge.accepted)


def test_partial_slot_matching_preserves_slot_indices():
    slots = np.array([[10, 10], [50, 10], [90, 10], [500, 500]], float)
    tracks = np.array([[89, 10], [11, 10]], float)
    matched = match_partial_tracks_to_slots(
        slots, tracks, ["node_c", "node_a"], match_distance_px=3
    )
    assert matched == {2: "node_c", 0: "node_a"}


def test_conflicting_component_votes_veto_the_selected_3d_triplet():
    obs = pd.DataFrame([
        {"frame": 5, "source_idx": 0, "view": "top_cam", "node_id": "top:a", "view_quality": 2.0},
        {"frame": 5, "source_idx": 0, "view": "side1_cam", "node_id": "side:b", "view_quality": 0.5},
    ])
    component = {"top:a": "ca", "side:b": "cb"}
    labels = {"ca": "ID_1", "cb": "ID_2"}
    hints, conflicts = build_component_hints(
        obs, component, labels, num_individuals=3
    )
    assert hints == {}
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["best_view"] == "top_cam"

    best = {5: {"groups": {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}}}
    vetoed = veto_conflicting_sources(best, conflicts)
    assert np.isnan(vetoed[5]["groups"]["a"]).all()
    assert vetoed[5]["groups"]["b"] == [4, 5, 6]


def test_continuity_veto_keeps_modal_identity_and_flags_minority_rows():
    obs = pd.DataFrame([
        {"frame": 0, "source_idx": 0, "view": "top_cam", "node_id": "top:a"},
        {"frame": 1, "source_idx": 0, "view": "top_cam", "node_id": "top:a"},
        {"frame": 2, "source_idx": 1, "view": "top_cam", "node_id": "top:a"},
    ])
    tracked = {
        0: {"id_mapping": [0, 1, 2]},
        1: {"id_mapping": [0, 1, 2]},
        2: {"id_mapping": [0, 1, 2]},
    }
    vetoes = continuity_vetoes(obs, tracked)
    assert len(vetoes) == 1
    assert int(vetoes.iloc[0].frame) == 2
    assert vetoes.iloc[0].best_identity == "ID_1"
    protected = continuity_vetoes(
        obs, tracked, protected_hints={2: {1: "ID_2"}}
    )
    assert protected.empty


def test_identified_2d_rows_survive_missing_3d(tmp_path):
    path = tmp_path / "top.csv"
    pd.DataFrame([
        {"frame": 1, "tracklet_id": 5, "x": 10, "y": 20, "score": 0.9},
        {"frame": 2, "tracklet_id": 5, "x": 11, "y": 20, "score": 0.9},
    ]).to_csv(path, index=False)
    observations = pd.DataFrame([
        {"frame": 1, "node_id": "top_cam:5", "geometry_usable": True},
    ])
    rows = build_identified_2d_rows(
        {"top_cam": path},
        {"top_cam:5": "component_5"},
        {"component_5": "ID_1"},
        {"component_5": "one_to_one_exclusion"},
        observations,
    )
    assert rows.observation_status.tolist() == [
        "assigned_2d_valid_3d", "assigned_2d_only"
    ]
    assert rows.reason_code.tolist() == [
        "assigned_three_view", "partial_2d_identity_only"
    ]
    windowed = build_identified_2d_rows(
        {"top_cam": path},
        {"top_cam:5": "component_5"},
        {"component_5": "ID_1"},
        {"component_5": "one_to_one_exclusion"},
        observations,
        start_frame=1,
        end_frame=1,
    )
    assert windowed.frame.tolist() == [1]


def test_appearance_is_only_fallback_and_cannot_override_locked_domain():
    appearance = pd.DataFrame([
        {"node_id": "top:locked", "predicted_identity": "ID_2", "confidence": 0.99},
        {"node_id": "top:open", "predicted_identity": "ID_3", "confidence": 0.95},
    ])
    labels, confidence, _ = apply_targeted_appearance_fallback(
        appearance,
        {"top:locked": "locked", "top:open": "open"},
        {"locked": {"ID_1"}, "open": {"ID_2", "ID_3"}},
    )
    assert "locked" not in labels
    assert labels["open"] == "ID_3"
    assert confidence["open"] == 0.95


def _sequential_3d_observations(second_start_x: float) -> pd.DataFrame:
    rows = []
    for frame, node, x in (
        (0, "top_cam:101", 0.0),
        (1, "top_cam:101", 1.0),
        (3, "top_cam:102", second_start_x),
        (4, "top_cam:102", second_start_x + 1.0),
    ):
        rows.append(
            {
                "frame": frame,
                "source_idx": 0,
                "view": "top_cam",
                "node_id": node,
                "X": x,
                "Y": 0.0,
                "Z": 0.0,
                "residual": 1.0,
                "geometry_usable": True,
                "hardcore": False,
            }
        )
    return pd.DataFrame(rows)


def _soft_silhouette_edge() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node_a": "top_cam:101",
                "node_b": "top_cam:102",
                "support_rows": 4,
                "support_weight": 0.85,
                "trusted_mask_segment_id": 7,
                "gap_frames": 1,
                "edge_source": "silhouette_continuity",
            }
        ]
    )


def test_silhouette_soft_edge_links_only_after_compatible_3d_validation():
    obs = _sequential_3d_observations(second_start_x=3.0)
    component, edges = build_constrained_tracklet_graph(
        obs,
        ["top_cam:101", "top_cam:102"],
        silhouette_edges=_soft_silhouette_edge(),
    )
    assert component["top_cam:101"] == component["top_cam:102"]
    edge = edges[edges.edge_source == "silhouette_continuity"].iloc[0]
    assert bool(edge.accepted)
    assert edge.reason == "silhouette_continuity_soft_edge"
    assert edge.silhouette_3d_validation == "compatible"


def test_silhouette_soft_edge_rejects_incompatible_3d_jump():
    obs = _sequential_3d_observations(second_start_x=100.0)
    component, edges = build_constrained_tracklet_graph(
        obs,
        ["top_cam:101", "top_cam:102"],
        silhouette_edges=_soft_silhouette_edge(),
    )
    assert component["top_cam:101"] != component["top_cam:102"]
    edge = edges[edges.edge_source == "silhouette_continuity"].iloc[0]
    assert not bool(edge.accepted)
    assert float(edge.silhouette_3d_distance_mm) > 8.0


def test_load_silhouette_edges_filters_and_maps_fused_ids(tmp_path):
    path = tmp_path / "links.csv"
    pd.DataFrame(
        [
            {
                "trusted_mask_segment_id": 7,
                "left_tracklet_id": 101,
                "right_tracklet_id": 102,
                "gap_frames": 1,
                "temporal_overlap_frames": 0,
                "silhouette_link_status": "candidate",
                "support_rows": 4,
                "mean_silhouette_weight": 0.85,
            },
            {
                "trusted_mask_segment_id": 8,
                "left_tracklet_id": 201,
                "right_tracklet_id": 202,
                "gap_frames": 1,
                "temporal_overlap_frames": 0,
                "silhouette_link_status": "candidate",
                "support_rows": 4,
                "mean_silhouette_weight": 0.20,
            },
        ]
    ).to_csv(path, index=False)
    edges = load_silhouette_link_edges("side1_cam", path)
    assert edges[["node_a", "node_b"]].to_dict("records") == [
        {"node_a": "side1_cam:101", "node_b": "side1_cam:102"}
    ]
