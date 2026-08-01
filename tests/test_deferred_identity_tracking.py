import numpy as np

from scripts.reconstruction_3d.deferred_identity_tracking import (
    DeferredTrackingConfig,
    apply_deferred_identity_tracking,
)


def _frame(*xs: float) -> dict:
    return {
        "groups": {
            chr(ord("a") + index): [float(x), 0.0, 0.0]
            for index, x in enumerate(xs)
        },
        "perm_cam0": list(range(len(xs))),
        "perm_cam1": list(range(len(xs))),
        "perm_cam2": list(range(len(xs))),
    }


def test_lookahead_resolves_crossing_after_separation():
    best = {
        0: _frame(0, 10, 30),
        1: _frame(4, 6, 30),
        2: _frame(8, 2, 30),
        3: _frame(12, -2, 30),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            entry_margin=5.0,
            retention_margin=20.0,
            resolution_margin=0.5,
            resolution_stability_frames=1,
            minimum_lookahead_frames=2,
            maximum_defer_horizon=10,
            velocity_prediction_weight=1.0,
            acceleration_weight=0.0,
        ),
    )

    assert result.unresolved_intervals == ()
    assert result.tracked_by_frame[2]["groups"]["ID_1"][0] == 8.0
    assert result.tracked_by_frame[2]["identity_status"] == "resolved_after_lookahead"
    assert result.assignment_audit.ambiguous.any()


def test_equal_hypotheses_expire_as_explicitly_unresolved():
    best = {
        0: _frame(0, 10, 30),
        1: _frame(5, 5, 30),
        2: _frame(5, 5, 30),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            entry_margin=1.0,
            retention_margin=10.0,
            resolution_margin=1.0,
            resolution_stability_frames=1,
            minimum_lookahead_frames=2,
            maximum_defer_horizon=2,
            acceleration_weight=0.0,
        ),
    )

    assert result.unresolved_intervals[0]["frame_start"] == 1
    assert result.unresolved_intervals[0]["frame_end"] == 2
    assert result.tracked_by_frame[1]["identity_status"] == "unresolved"
    assert all(
        np.isnan(point).all()
        for point in result.tracked_by_frame[1]["groups"].values()
    )
    assert "unresolved_geometry" in result.tracked_by_frame[1]


def test_timeout_does_not_commit_hidden_winner_to_later_identity():
    best = {
        0: _frame(0, 10, 30),
        1: _frame(5, 5, 30),
        2: _frame(5, 5, 30),
        3: _frame(1, 9, 30),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            entry_margin=1.0,
            retention_margin=10.0,
            resolution_margin=1.0,
            resolution_stability_frames=1,
            minimum_lookahead_frames=2,
            maximum_defer_horizon=2,
            acceleration_weight=0.0,
        ),
    )

    assert result.tracked_by_frame[1]["identity_status"] == "unresolved"
    assert result.tracked_by_frame[3]["identity_status"] == (
        "new_segment_after_unresolved"
    )
    assert result.tracked_by_frame[3]["identity_segment"] == 1


def test_tracklet_hint_never_overrides_hard_motion_gate():
    best = {
        0: _frame(0, 20, 40),
        1: _frame(1, 21, 41),
        2: _frame(22, 2, 42),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            max_jump_mm=8.0,
            max_jump_hard_cap_mm=8.0,
            entry_margin=0.1,
            maximum_defer_horizon=1,
            acceleration_weight=0.0,
        ),
        source_id_hints_by_frame={2: {0: "ID_1"}},
    )

    assert result.tracked_by_frame[2]["identity_status"] == "unresolved"
    assert result.unresolved_intervals[0]["reason"] == "tracklet_motion_gate_conflict"
    assert result.assignment_audit.iloc[-1].constraint_conflict


def test_temporary_tracklet_conflict_can_resolve_after_hint_disappears():
    best = {
        0: _frame(0, 20, 40),
        1: _frame(1, 21, 41),
        2: _frame(22, 2, 42),
        3: _frame(3, 23, 43),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            max_jump_mm=8.0,
            max_jump_hard_cap_mm=8.0,
            entry_margin=0.1,
            retention_margin=20.0,
            resolution_margin=0.1,
            resolution_stability_frames=1,
            minimum_lookahead_frames=2,
            maximum_defer_horizon=5,
            acceleration_weight=0.0,
        ),
        source_id_hints_by_frame={2: {0: "ID_1"}},
    )

    assert result.unresolved_intervals == ()
    audit = result.assignment_audit.set_index("frame")
    assert bool(audit.loc[2, "constraint_conflict"])
    assert not bool(audit.loc[3, "constraint_conflict"])
    assert result.tracked_by_frame[2]["identity_status"] == (
        "resolved_after_lookahead"
    )


def test_long_all_missing_gap_restarts_identity_segment():
    best = {
        0: _frame(0, 20, 40),
        1: _frame(),
        2: _frame(),
        3: _frame(),
        4: _frame(4, 24, 44),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            max_gap_frames=2,
            maximum_defer_horizon=10,
        ),
    )

    assert all(
        result.tracked_by_frame[frame]["identity_status"]
        == "unresolved_insufficient_geometry"
        for frame in (1, 2, 3)
    )
    assert result.tracked_by_frame[4]["identity_status"] == (
        "new_segment_after_unresolved"
    )
    assert result.tracked_by_frame[4]["identity_segment"] == 1


def test_long_partial_geometry_gap_cannot_claim_false_continuity():
    best = {
        0: _frame(0, 20, 40),
        1: _frame(1, 21),
        2: _frame(2, 22),
        3: _frame(3, 23),
        4: _frame(4, 24, 44),
    }
    result = apply_deferred_identity_tracking(
        best,
        config=DeferredTrackingConfig(
            max_gap_frames=2,
            maximum_defer_horizon=10,
        ),
    )

    assert result.tracked_by_frame[1]["identity_status"] == (
        "unresolved_insufficient_geometry"
    )
    assert result.tracked_by_frame[4]["identity_segment"] == 1
    assert result.tracked_by_frame[4]["identity_status"] == (
        "new_segment_after_unresolved"
    )


def test_forced_a4_segment_restart_resets_a5_identity_state():
    best = {
        0: _frame(0, 20, 40),
        1: _frame(1, 21, 41),
        2: _frame(2, 22, 42),
    }
    result = apply_deferred_identity_tracking(
        best,
        forced_reset_frames={2},
    )

    assert result.tracked_by_frame[2]["identity_status"] == (
        "new_segment_after_unresolved"
    )
    assert result.tracked_by_frame[2]["identity_segment"] == 1
    assert result.configuration["forced_reset_frames"] == [2]


def test_deferred_tracking_is_deterministic():
    best = {
        0: _frame(0, 10, 30),
        1: _frame(4, 6, 30),
        2: _frame(8, 2, 30),
        3: _frame(12, -2, 30),
    }
    config = DeferredTrackingConfig(
        entry_margin=5.0,
        retention_margin=20.0,
        resolution_stability_frames=1,
        minimum_lookahead_frames=2,
        acceleration_weight=0.0,
    )
    left = apply_deferred_identity_tracking(best, config=config)
    right = apply_deferred_identity_tracking(best, config=config)

    assert left.tracked_by_frame == right.tracked_by_frame
    assert left.assignment_audit.to_csv(index=False) == right.assignment_audit.to_csv(
        index=False
    )
