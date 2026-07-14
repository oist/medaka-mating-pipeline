import numpy as np
import pandas as pd
import pytest

from scripts.tracking.benchmark_global_fragment_stitching import (
    _assert_exact_xyz_equal,
    _write_and_verify_xyz_copy,
)

from scripts.tracking.global_fragment_stitching import (
    CostGates,
    EdgeFeatures,
    Fragment,
    calculate_edge_cost,
    solve_constrained_path_cover,
)


def _edge(source, target, *, gap=0, displacement=1.0, motion=1.0, **kwargs):
    return calculate_edge_cost(
        EdgeFeatures(
            source_id=source,
            target_id=target,
            gap_frames=gap,
            endpoint_displacement_mm=displacement,
            motion_prediction_error_mm=motion,
            **kwargs,
        )
    )


def test_unknown_or_missing_segmentation_is_neutral_not_contradiction():
    edge = calculate_edge_cost(
        EdgeFeatures(source_id="a", target_id="b", gap_frames=2)
    )
    assert edge.feasible
    assert edge.terms["silhouette_continuity"] == 0.0
    assert edge.terms["mask_overlap"] == 0.0


def test_only_strong_one_to_one_mask_inconsistency_vetoes():
    edge = calculate_edge_cost(
        EdgeFeatures(
            source_id="a",
            target_id="b",
            gap_frames=0,
            silhouette_state="contradiction",
            strong_one_to_one_contradiction=True,
        )
    )
    assert not edge.feasible
    assert edge.reason == "strong_one_to_one_mask_contradiction"


def test_unvalidated_contradiction_state_is_treated_as_unknown():
    edge = calculate_edge_cost(
        EdgeFeatures(
            source_id="a",
            target_id="b",
            gap_frames=0,
            silhouette_state="contradiction",
            strong_one_to_one_contradiction=False,
        )
    )
    assert edge.feasible
    assert edge.terms["silhouette_continuity"] == 0.0
    assert edge.terms["mask_overlap"] == 0.0


def test_motion_prediction_and_angle_raise_link_cost():
    good = calculate_edge_cost(
        EdgeFeatures(
            source_id="a", target_id="b", gap_frames=0,
            endpoint_displacement_mm=1.0,
            motion_prediction_error_mm=0.5,
            velocity_angle_deg=5.0,
        )
    )
    bad = calculate_edge_cost(
        EdgeFeatures(
            source_id="a", target_id="c", gap_frames=0,
            endpoint_displacement_mm=6.0,
            motion_prediction_error_mm=10.0,
            velocity_angle_deg=120.0,
        )
    )
    assert good.feasible and bad.feasible
    assert good.total_cost < bad.total_cost


def test_long_missing_detection_gap_links_when_motion_is_plausible():
    fragments = [
        Fragment("a", 0, 2, "ID_1"),
        Fragment("b", 10, 12),
    ]
    edge = _edge("a", "b", gap=7, displacement=5.0, motion=1.0)
    result = solve_constrained_path_cover(fragments, [edge], break_cost=4.0)
    assert [(e.source_id, e.target_id) for e in result.selected_edges] == [("a", "b")]
    assert result.fragment_identities["b"] == "ID_1"


def test_hard_jump_cap_rejects_gap_scaled_identity_link():
    edge = calculate_edge_cost(
        EdgeFeatures(
            source_id="a", target_id="b", gap_frames=50,
            endpoint_displacement_mm=81.0,
        ),
        gates=CostGates(max_gap_frames=100, max_jump_hard_cap_mm=80.0),
    )
    assert not edge.feasible
    assert edge.reason == "endpoint_displacement_exceeds_gate"


def test_multiple_trusted_anchors_propagate_without_geometry_id_labels():
    fragments = [
        Fragment("seed_early", 0, 1, "ID_2"),
        Fragment("middle", 3, 4),
        Fragment("seed_late", 6, 7, "ID_2"),
    ]
    result = solve_constrained_path_cover(
        fragments,
        [_edge("seed_early", "middle"), _edge("middle", "seed_late")],
        break_cost=4.0,
    )
    assert result.fragment_identities == {
        "middle": "ID_2",
        "seed_early": "ID_2",
        "seed_late": "ID_2",
    }
    assert len(result.selected_edges) == 2


def test_conflicting_biological_anchors_cannot_share_a_path():
    fragments = [
        Fragment("a", 0, 1, "ID_1"),
        Fragment("middle", 3, 4),
        Fragment("b", 6, 7, "ID_2"),
    ]
    result = solve_constrained_path_cover(
        fragments,
        [_edge("a", "middle"), _edge("middle", "b")],
        break_cost=4.0,
    )
    assert len(result.selected_edges) == 1
    assert not (
        result.component_by_fragment["a"]
        == result.component_by_fragment["b"]
    )


def test_temporal_exclusivity_blocks_two_overlapping_fragments_for_one_fish():
    fragments = [
        Fragment("left_anchor", 0, 0, "ID_1"),
        Fragment("upper", 2, 3),
        Fragment("lower", 2, 3),
        Fragment("right_anchor", 5, 5, "ID_1"),
    ]
    edges = [
        _edge("left_anchor", "upper"),
        _edge("lower", "right_anchor"),
    ]
    result = solve_constrained_path_cover(fragments, edges, break_cost=4.0)
    assert len(result.selected_edges) == 1
    assert sum(
        result.fragment_identities.get(fragment) == "ID_1"
        for fragment in ("upper", "lower")
    ) == 1


def test_crossing_uses_global_motion_cost_without_swapping_identity_labels():
    fragments = [
        Fragment("id1_before", 0, 2, "ID_1"),
        Fragment("id2_before", 0, 2, "ID_2"),
        Fragment("path_left", 4, 6),
        Fragment("path_right", 4, 6),
    ]
    edges = [
        _edge("id1_before", "path_left", motion=0.5),
        _edge("id1_before", "path_right", motion=8.0),
        _edge("id2_before", "path_left", motion=8.0),
        _edge("id2_before", "path_right", motion=0.5),
    ]
    result = solve_constrained_path_cover(fragments, edges, break_cost=4.0)
    selected = {(edge.source_id, edge.target_id) for edge in result.selected_edges}
    assert selected == {
        ("id1_before", "path_left"),
        ("id2_before", "path_right"),
    }
    assert result.fragment_identities["path_left"] == "ID_1"
    assert result.fragment_identities["path_right"] == "ID_2"


def test_simultaneous_duplicate_trusted_anchor_is_rejected_before_solving():
    with pytest.raises(ValueError, match="simultaneous trusted anchors"):
        solve_constrained_path_cover(
            [Fragment("a", 0, 3, "ID_3"), Fragment("b", 2, 4, "ID_3")],
            [],
        )


def test_xyz_safety_copy_is_exact_and_deliberate_perturbation_fails(tmp_path):
    source = tmp_path / "tracks.csv"
    table = pd.DataFrame(
        {
            "frame": [0, 1, 0, 1],
            "fish_id": [0, 0, 1, 1],
            "X": [1.0, 1.25, np.nan, 3.0],
            "Y": [2.0, 2.25, 4.0, 4.5],
            "Z": [5.0, 5.25, 6.0, 6.5],
        }
    )
    table.to_csv(source, index=False)
    result = _write_and_verify_xyz_copy(
        source,
        tmp_path / "copy.csv",
        speed_spike_threshold_mm=8.0,
    )
    assert result["xyz_exact_equal"] is True
    perturbed = table.copy()
    perturbed.loc[0, "X"] += 1e-6
    with pytest.raises(RuntimeError, match="XYZ safety failure"):
        _assert_exact_xyz_equal(table, perturbed)
