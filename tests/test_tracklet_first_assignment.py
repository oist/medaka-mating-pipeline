import numpy as np

from scripts.reconstruction_3d.tracklet_first_assignment import (
    DISPOSITION_REASONS,
    ReconstructionCandidate,
    build_detection_dispositions,
    candidate_identity,
    propagate_identity_domains,
    select_global_candidate_set,
)


def _candidate(slots, residual, identity=None, **kwargs):
    return ReconstructionCandidate(
        slots=slots,
        xyz=(float(slots[0]), float(slots[1]), float(slots[2])),
        residual=float(residual),
        identity=identity,
        **kwargs,
    )


def test_global_selection_beats_residual_greedy_counterexample():
    # The individually cheapest proposal consumes one different resource from
    # each member of the only complete three-fish solution.
    candidates = [
        _candidate((0, 0, 0), 1.0),
        _candidate((0, 1, 1), 2.0),
        _candidate((1, 0, 2), 2.0),
        _candidate((2, 2, 0), 2.0),
    ]
    result = select_global_candidate_set(
        candidates, max_selected=3, max_states=None
    )
    assert len(result.selected) == 3
    assert {candidate.slots for candidate in result.selected} == {
        (0, 1, 1), (1, 0, 2), (2, 2, 0)
    }


def test_conflicting_low_residual_hypothesis_triggers_alternative():
    candidates = [
        _candidate((0, 0, 0), 1.0, identity_conflict=True),
        _candidate((0, 1, 1), 2.0, identity="ID_1"),
        _candidate((1, 0, 2), 2.0, identity="ID_2"),
        _candidate((2, 2, 0), 2.0, identity="ID_3"),
    ]
    result = select_global_candidate_set(candidates, max_states=None)
    assert len(result.selected) == 3
    assert result.rejected_identity_conflicts == 1
    assert all(not candidate.identity_conflict for candidate in result.selected)


def test_two_known_tracklets_force_top_cam_5_to_id_1():
    resolved, sources, domains = propagate_identity_domains(
        ["top_cam:2", "top_cam:4", "top_cam:5"],
        locked_identities={"top_cam:2": "ID_3", "top_cam:4": "ID_2"},
        simultaneous_groups=[("top_cam:2", "top_cam:4", "top_cam:5")],
    )
    assert resolved["top_cam:5"] == "ID_1"
    assert domains["top_cam:5"] == {"ID_1"}
    assert sources["top_cam:5"] == "one_to_one_exclusion"


def test_locked_tracklet_identity_never_changes():
    resolved, _, domains = propagate_identity_domains(
        ["a", "b", "c"],
        locked_identities={"a": "ID_1", "b": "ID_1"},
        simultaneous_groups=[("a", "b", "c")],
    )
    assert resolved["a"] == "ID_1"
    assert resolved["b"] == "ID_1"
    assert domains["a"] == {"ID_1"}
    assert domains["b"] == {"ID_1"}


def test_candidate_tracklet_consensus_detects_cross_view_conflict():
    identity, conflict, evidence = candidate_identity(
        (0, 1, -1),
        slot_nodes=[{0: "top:5"}, {1: "side:9"}, {}],
        node_identities={"top:5": "ID_1", "side:9": "ID_2"},
    )
    assert identity is None
    assert conflict
    assert evidence == ("ID_1", "ID_2")


def test_2d_identity_is_retained_when_xyz_is_unavailable_and_all_rows_reasoned():
    points = {
        10: {
            "cam0": [(10.0, 20.0), (30.0, 40.0)],
            "cam1": [(11.0, 21.0), (np.nan, np.nan)],
            "cam2": [(12.0, 22.0), (np.nan, np.nan)],
        }
    }
    best = {
        10: {
            "perm_cam0": [0], "perm_cam1": [0], "perm_cam2": [0],
            "recovered": [False], "synthetic_views": [-1],
            "source_id_hints": {0: "ID_2"},
        }
    }
    rows = build_detection_dispositions(
        points,
        best,
        slot_nodes_by_frame={
            10: {
                "cam0": {0: "top:known", 1: "top:5"},
                "cam1": {0: "side1:known"},
                "cam2": {0: "side2:known"},
            }
        },
        node_identities={
            "top:known": "ID_2", "side1:known": "ID_2",
            "side2:known": "ID_2", "top:5": "ID_1",
        },
    )
    assert len(rows) == 4
    top5 = next(row for row in rows if row["tracklet_node"] == "top:5")
    assert top5["final_fish_id"] == "ID_1"
    assert top5["disposition"] == "assigned_2d_only"
    assert top5["primary_reason"] == "partial_2d_identity_only"
    assert all(
        row["primary_reason"] in DISPOSITION_REASONS
        or row["primary_reason"] in {"assigned_three_view", "assigned_two_view_completed"}
        for row in rows
    )
