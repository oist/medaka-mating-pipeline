from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.reconstruction_3d.candidate_cache import CandidateWindow
from scripts.reconstruction_3d.windowed_tracklet_core import (
    LinkGates,
    LinkWeights,
    SolverConfig,
    SolverResult,
    WindowSpec,
    annotate_cross_view_groups,
    apply_fragment_identities,
    assert_tracking_invariants,
    build_candidate_tracklet_ids,
    build_fragment_link_edges,
    build_fragments,
    build_legacy_tracklet_ids,
    group_cross_view_tracklets,
    make_overlapping_windows,
    reconcile_window_fragments,
    solve_fragment_identities,
    summarize_fragments_compact,
    track_observations_hungarian,
)


def _window(mouth, tail=None, score=None, start=0):
    mouth = np.asarray(mouth, dtype=float)
    if tail is None:
        tail = mouth + np.array([0.0, 20.0])
    tail = np.asarray(tail, dtype=float)
    if score is None:
        score = np.ones(mouth.shape[:2], dtype=float)
    score = np.asarray(score, dtype=float)
    visibility = np.isfinite(mouth).all(axis=2).astype(np.uint8)
    visibility |= 2 * np.isfinite(tail).all(axis=2).astype(np.uint8)
    return CandidateWindow(
        frame_start=start,
        frame_end=start + len(mouth) - 1,
        mouth_xy=mouth,
        tail_xy=tail,
        instance_score=score,
        mouth_score=score,
        tail_score=score,
        visibility=visibility,
        source_instance=np.tile(np.arange(mouth.shape[1]), (len(mouth), 1)),
    )


def _observation(frame, source, xyz, slots=(0, 0, 0), group="g", trusted=True):
    return {
        "observation_id": f"f{frame}_s{source}",
        "frame": frame,
        "source_index": source,
        "X": xyz[0],
        "Y": xyz[1],
        "Z": xyz[2],
        "ray_residual_mm2": 0.1,
        "residual_unit": "mm^2",
        "slot_cam0": slots[0],
        "slot_cam1": slots[1],
        "slot_cam2": slots[2],
        "mouth_confidence": 0.9,
        "view_count": 3,
        "recovered": False,
        "missing_view": -1,
        "tail_X": xyz[0] + 10.0,
        "tail_Y": xyz[1],
        "tail_Z": xyz[2],
        "tail_ray_residual_mm2": 0.1,
        "tail_vector_X": 10.0,
        "tail_vector_Y": 0.0,
        "tail_vector_Z": 0.0,
        "cross_view_group_id": group,
        "cross_view_group_support": 3,
        "cross_view_group_trusted": trusted,
    }


def test_window_plan_has_exact_core_ownership_and_context():
    windows = make_overlapping_windows(0, 24999, core_size=10000, overlap=250)
    core = [frame for window in windows for frame in range(window.core_start, window.core_end + 1)]
    assert core == list(range(25000))
    assert windows[1].context_start == 9750
    assert windows[1].context_end == 20249
    with pytest.raises(ValueError, match="smaller than required"):
        make_overlapping_windows(0, 10, core_size=5, overlap=99)


def test_dense_legacy_tracklets_keep_exact_candidate_slots():
    mouth = np.array(
        [
            [[0, 0], [100, 0], [200, 0]],
            [[1, 0], [101, 0], [201, 0]],
            [[2, 0], [102, 0], [202, 0]],
        ],
        dtype=float,
    )
    ids, rows, summary = build_legacy_tracklet_ids(_window(mouth))
    assert ids.shape == (3, 3)
    assert np.all(ids[:, 0] == ids[0, 0])
    assert len(rows) == 9
    assert summary["n_fragments"] == 3


def test_all_candidate_tracklets_preserve_extra_slots_and_ambiguity():
    mouth = np.full((3, 6, 2), np.nan)
    mouth[:, 5] = [[500, 0], [501, 0], [502, 0]]
    mouth[:, 0] = [[0, 0], [1, 0], [2, 0]]
    mouth[:, 1] = [[1, 0], [2, 0], [3, 0]]  # deliberately too close to slot 0
    ids, rows, summary = build_candidate_tracklet_ids(_window(mouth))
    assert np.all(ids[:, 5] >= 0)
    assert np.all(ids[:, 0] == -1)
    assert np.all(ids[:, 1] == -1)
    assert set(rows.candidate_slot) == {5}
    assert summary["num_candidate_slots"] == 6


def test_multiframe_group_requires_support_and_uses_exact_tracklet_slots():
    observations = pd.DataFrame(
        [_observation(frame, 0, [frame, 0, 0]) for frame in range(3)]
    )
    tracklets = {
        view: np.zeros((3, 1), dtype=np.int32)
        for view in ("top_cam", "side1_cam", "side2_cam")
    }
    grouped, groups = annotate_cross_view_groups(
        observations, tracklets, cache_frame_start=0, mode="multiframe", min_support_frames=3
    )
    assert grouped.cross_view_group_trusted.all()
    assert groups.iloc[0].support_frames == 3
    grouped2, _ = annotate_cross_view_groups(
        observations.iloc[:2], tracklets, cache_frame_start=0, mode="multiframe", min_support_frames=3
    )
    assert not grouped2.cross_view_group_trusted.any()


def test_multiframe_set_packing_uses_frozen_support_and_resources():
    rows = []
    for frame in range(3):
        rows.extend(
            [
                {
                    "frame": frame, "view_mask": 7, "slot_cam0": 0,
                    "slot_cam1": 0, "slot_cam2": 0, "tracklet_top_cam": 10,
                    "tracklet_side1_cam": 20, "tracklet_side2_cam": 30,
                    "X": frame, "Y": 0.0, "Z": 0.0,
                    "ray_residual_mm2": 0.2, "hypothesis_id": f"true_{frame}",
                    "view_count": 3,
                },
                {
                    "frame": frame, "view_mask": 7, "slot_cam0": 0,
                    "slot_cam1": 1, "slot_cam2": 1, "tracklet_top_cam": 10,
                    "tracklet_side1_cam": 21, "tracklet_side2_cam": 31,
                    "X": frame, "Y": 1.0, "Z": 0.0,
                    "ray_residual_mm2": 0.01, "hypothesis_id": f"conflict_{frame}",
                    "view_count": 3,
                },
            ]
        )
    selected, candidates, conflicts, duplicates = group_cross_view_tracklets(
        pd.DataFrame(rows),
        mode="multiframe",
        tracklet_intervals={
            (0, 10): (0, 2), (1, 20): (0, 2), (2, 30): (0, 2),
            (1, 21): (0, 2), (2, 31): (0, 2),
        },
    )
    assert len(candidates[candidates.selected]) == 1
    assert len(selected) == 3
    assert not conflicts.empty
    assert duplicates.empty
    assert not selected.duplicated(["frame", "slot_cam0"]).any()


def test_fragments_break_for_close_contact_and_link_after_separation():
    records = []
    for frame, separation in [(0, 30.0), (1, 5.0), (2, 30.0)]:
        records.extend(
            [
                _observation(frame, 0, [0.0, 0.0, frame], slots=(0, 0, 0), group="a"),
                _observation(frame, 1, [separation, 0.0, frame], slots=(1, 1, 1), group="b"),
            ]
        )
    obs, fragments = build_fragments(pd.DataFrame(records), close_contact_mm=12.0)
    assert fragments.fragment_id.nunique() >= 4
    assert_tracking_invariants(obs)


def _fragment(
    fragment_id,
    start,
    end,
    start_xyz,
    end_xyz=None,
    velocity=(0.0, 0.0, 0.0),
    tail=(10.0, 0.0, 0.0),
):
    end_xyz = start_xyz if end_xyz is None else end_xyz
    return {
        "fragment_id": fragment_id,
        "frame_start": start,
        "frame_end": end,
        "observed_frames": end - start + 1,
        "observation_count": end - start + 1,
        "start_X": start_xyz[0],
        "start_Y": start_xyz[1],
        "start_Z": start_xyz[2],
        "end_X": end_xyz[0],
        "end_Y": end_xyz[1],
        "end_Z": end_xyz[2],
        "start_vX": velocity[0],
        "start_vY": velocity[1],
        "start_vZ": velocity[2],
        "end_vX": velocity[0],
        "end_vY": velocity[1],
        "end_vZ": velocity[2],
        "residual_median_mm2": 0.1,
        "mouth_confidence_median": 0.9,
        "start_residual_median_mm2": 0.1,
        "end_residual_median_mm2": 0.1,
        "start_mouth_confidence_median": 0.9,
        "end_mouth_confidence_median": 0.9,
        "trusted_group_ids": "",
        "tail_available_frames": 3,
        "tail_vector_X": tail[0],
        "tail_vector_Y": tail[1],
        "tail_vector_Z": tail[2],
        "start_tail3d_X": tail[0],
        "start_tail3d_Y": tail[1],
        "start_tail3d_Z": tail[2],
        "end_tail3d_X": tail[0],
        "end_tail3d_Y": tail[1],
        "end_tail3d_Z": tail[2],
        "overlap_agree_frames": 0,
        "overlap_disagree_frames": 0,
        **{
            f"{endpoint}_tail2d_cam{view}_{axis}": (
                tail[0] if axis == "x" else tail[1]
            )
            for endpoint in ("start", "end")
            for view in range(3)
            for axis in ("x", "y")
        },
    }


def test_fragment_edge_gap_gates_and_conditional_tail_policy():
    fragments = pd.DataFrame(
        [
            _fragment("a", 0, 2, (0, 0, 0)),
            _fragment("near", 3, 5, (1, 0, 0)),
            _fragment("ambiguous", 3, 5, (1.1, 0, 0), tail=(-10, 0, 0)),
            _fragment("gap100", 103, 105, (8, 0, 0)),
            _fragment("gap101", 104, 106, (8, 0, 0)),
        ]
    )
    edges = build_fragment_link_edges(
        fragments,
        gates=LinkGates(max_gap_frames=100),
        weights=LinkWeights(mouth_ambiguity_margin=0.5),
        tail_mode="conditional",
    )
    pairs = set(zip(edges.source_id, edges.target_id))
    assert ("a", "gap100") in pairs
    assert ("a", "gap101") not in pairs
    event = edges[(edges.source_id == "a") & (edges.target_start == 3)]
    assert event.tail_eligible.all()
    assert (event.tail_cost <= 0.25).all()
    # Tail is not eligible when there is no competing mouth-near alternative.
    singleton = build_fragment_link_edges(
        fragments[fragments.fragment_id.isin(["near", "gap100"])],
        tail_mode="conditional",
    )
    assert len(singleton) == 1
    assert not singleton.iloc[0].tail_eligible


def _solver_graph():
    # One middle fragment leaves two latent endpoints.  Greedy uses A0 because
    # it is locally cheaper; global uses B0 so A0 remains available for A2.
    fragments = [
        _fragment("A0", 0, 1, (0, 0, 0)),
        _fragment("B0", 0, 1, (10, 0, 0)),
        _fragment("C0", 0, 1, (20, 0, 0)),
        _fragment("X1", 10, 11, (5, 0, 0)),
        _fragment("A2", 20, 21, (0, 0, 0)),
        _fragment("C2", 20, 21, (20, 0, 0)),
    ]
    edges = []
    for source, target, cost in [
        ("A0", "X1", 1.0),
        ("B0", "X1", 1.1),
        ("A0", "A2", 0.0),
        ("C0", "C2", 0.0),
        ("B0", "A2", 50.0),
        ("X1", "A2", 50.0),
    ]:
        edges.append({"source_id": source, "target_id": target, "total_cost": cost})
    return pd.DataFrame(fragments), pd.DataFrame(edges)


def test_global_dag_beats_immediate_on_delayed_disambiguation():
    fragments, edges = _solver_graph()
    immediate = solve_fragment_identities(
        fragments, edges, config=SolverConfig(method="sequential_hungarian")
    )
    global_result = solve_fragment_identities(
        fragments, edges, config=SolverConfig(method="global_dag")
    )
    assert global_result.objective < immediate.objective
    assert len(global_result.paths) == 3
    assert len(global_result.selected_edges) == len(fragments) - 3
    assert global_result.feasible


def test_deferred_beam_reverses_local_leader_on_same_graph():
    fragments, edges = _solver_graph()
    immediate = solve_fragment_identities(
        fragments,
        edges,
        config=SolverConfig(method="beam", beam_width=32, identity_policy="immediate"),
    )
    deferred = solve_fragment_identities(
        fragments,
        edges,
        config=SolverConfig(method="beam", beam_width=32, identity_policy="deferred"),
    )
    assert deferred.objective < immediate.objective
    assert deferred.hypotheses_peak >= 2
    assert any(
        interval.get("resolution_status") == "resolved_by_margin"
        for interval in deferred.unresolved_intervals
    )
    assert any(
        interval.get("resolution_status") == "forced_immediate"
        for interval in immediate.unresolved_intervals
    )


def test_deferred_equal_hypotheses_expire_unresolved_at_exact_horizon():
    fragments = []
    for label, x in (("A", 0), ("B", 10), ("C", 20)):
        for layer, frame in ((0, 0), (1, 10), (2, 300)):
            fragments.append(_fragment(f"{label}{layer}", frame, frame + 1, (x, 0, 0)))
    edges = []
    for source in ("A0", "B0"):
        for target in ("A1", "B1"):
            edges.append({"source_id": source, "target_id": target, "total_cost": 0.0})
    edges.append({"source_id": "C0", "target_id": "C1", "total_cost": 0.0})
    for source in ("A1", "B1"):
        for target in ("A2", "B2"):
            edges.append({"source_id": source, "target_id": target, "total_cost": 0.0})
    edges.append({"source_id": "C1", "target_id": "C2", "total_cost": 0.0})
    result = solve_fragment_identities(
        pd.DataFrame(fragments),
        pd.DataFrame(edges),
        config=SolverConfig(method="beam", beam_width=32, identity_policy="deferred"),
    )
    unresolved = [row for row in result.unresolved_intervals if not row["resolved"]]
    assert len(unresolved) == 1
    assert unresolved[0]["frame_end"] - unresolved[0]["frame_start"] == 250


def test_exact_solver_rejects_missing_three_path_cover_and_is_order_stable():
    fragments, edges = _solver_graph()
    expected = solve_fragment_identities(
        fragments, edges, config=SolverConfig(method="global_dag")
    )
    shuffled = solve_fragment_identities(
        fragments.sample(frac=1, random_state=7),
        edges.sample(frac=1, random_state=8),
        config=SolverConfig(method="global_dag"),
    )
    assert shuffled.selected_edges == expected.selected_edges
    assert shuffled.paths == expected.paths
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_fragment_identities(
            fragments,
            edges.iloc[0:0],
            config=SolverConfig(method="global_dag"),
        )
    restarted = solve_fragment_identities(
        fragments,
        edges.iloc[0:0],
        config=SolverConfig(
            method="sequential_hungarian", path_cover_mode="restarts"
        ),
    )
    assert len(restarted.paths) == len(fragments)
    assert len(set(restarted.fragment_identities.values())) == len(fragments)
    assert restarted.objective > 0


def test_overlap_reconciliation_aliases_exact_resources_but_not_breaks():
    left = pd.DataFrame(
        [
            {**_observation(frame, 0, [frame, 0, 0]), "fragment_id": "left"}
            for frame in range(0, 8)
        ]
    )
    right = pd.DataFrame(
        [
            {**_observation(frame, 0, [frame, 0, 0]), "fragment_id": "right"}
            for frame in range(2, 10)
        ]
    )
    specs = [
        WindowSpec(0, 0, 4, 0, 8),
        WindowSpec(1, 5, 9, 1, 9),
    ]
    merged, aliases, seams = reconcile_window_fragments([left, right], specs)
    assert merged.fragment_id.nunique() == 1
    assert aliases.accepted.any()
    assert seams.strict_resource_equivalent.all()
    broken_right = right.copy()
    broken_right.loc[broken_right.frame >= 5, "fragment_id"] = "right_after_break"
    broken_right.loc[broken_right.frame >= 5, ["slot_cam0", "slot_cam1", "slot_cam2"]] = 1
    merged_broken, _, _ = reconcile_window_fragments([left, broken_right], specs)
    assert merged_broken.fragment_id.nunique() == 2


def test_deferred_unresolved_rows_keep_geometry_but_blank_identity():
    observations = pd.DataFrame(
        [{**_observation(frame, 0, [frame, 0, 0]), "fragment_id": "f"} for frame in range(3)]
    )
    result = SolverResult(
        {"f": "ID_1"}, 0.0, 2, 0,
        ({"frame_start": 1, "frame_end": 2, "resolved": False},),
        0, "beam",
    )
    output = apply_fragment_identities(observations, result, identity_policy="deferred")
    assert output.loc[output.frame == 0, "fish_id"].notna().all()
    assert output.loc[output.frame >= 1, "fish_id"].isna().all()
    assert np.isfinite(output[["X", "Y", "Z"]]).all().all()


def test_resource_reuse_is_rejected():
    observations = pd.DataFrame(
        [
            _observation(0, 0, [0, 0, 0], slots=(0, 0, 0)),
            _observation(0, 1, [1, 1, 1], slots=(0, 1, 1)),
        ]
    )
    with pytest.raises(AssertionError, match="resource conflicts"):
        assert_tracking_invariants(observations)
    duplicate_identity = pd.DataFrame(
        [
            {**_observation(0, 0, [0, 0, 0], slots=(0, 0, 0)), "fish_id": "ID_1"},
            {**_observation(0, 1, [1, 1, 1], slots=(1, 1, 1)), "fish_id": "ID_1"},
        ]
    )
    with pytest.raises(AssertionError, match="multiple rows"):
        assert_tracking_invariants(duplicate_identity)


def test_compact_fragment_summary_preserves_inventory_and_endpoints():
    observations = pd.DataFrame(
        [
            {
                **_observation(frame, source, xyz),
                "fragment_id": fragment,
                "overlap_agree_frames": agree,
                "overlap_disagree_frames": 0,
            }
            for frame, source, xyz, fragment, agree in (
                (0, 0, [0, 1, 2], "a", 3),
                (1, 0, [1, 2, 3], "a", 3),
                (4, 0, [8, 9, 10], "b", 0),
            )
        ]
    )
    compact = summarize_fragments_compact(observations).set_index("fragment_id")
    assert compact.loc["a", "frame_start"] == 0
    assert compact.loc["a", "frame_end"] == 1
    assert compact.loc["a", "observation_count"] == 2
    assert compact.loc["a", "start_X"] == 0
    assert compact.loc["a", "end_X"] == 1
    assert compact.loc["a", "overlap_agree_frames"] == 3


def test_array_baseline_linker_keeps_simultaneous_assignment_and_gates():
    observations = pd.DataFrame(
        [
            _observation(0, 0, [0, 0, 0]),
            _observation(0, 1, [10, 0, 0]),
            _observation(0, 2, [20, 0, 0]),
            _observation(1, 0, [9, 0, 0]),
            _observation(1, 1, [1, 0, 0]),
            _observation(1, 2, [21, 0, 0]),
            _observation(2, 0, [100, 0, 0]),
        ]
    )
    tracked = track_observations_hungarian(
        observations,
        max_jump_mm=8,
        max_jump_hard_cap_mm=80,
        max_gap_frames=100,
    )
    assert tracked.loc[tracked.frame == 0, "fish_id"].tolist() == ["ID_1", "ID_2", "ID_3"]
    assert tracked.loc[tracked.frame == 1, "fish_id"].tolist() == ["ID_2", "ID_1", "ID_3"]
    assert tracked.loc[tracked.frame == 2, "fish_id"].isna().all()
