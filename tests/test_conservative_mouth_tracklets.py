import numpy as np
from dataclasses import replace

from scripts.reconstruction_3d.candidate_cache import CandidateWindow
from scripts.reconstruction_3d.conservative_mouth_tracklets import (
    ConservativeTrackletConfig,
    build_reciprocal_tracklets,
    clean_motion_distributions,
    frozen_body_length_px_from_training_controls,
)


def _window(points_by_frame: list[list[tuple[float, float]]]) -> CandidateWindow:
    frame_count = len(points_by_frame)
    slots = max(len(points) for points in points_by_frame)
    mouth = np.full((frame_count, slots, 2), np.nan)
    tail = np.full_like(mouth, np.nan)
    score = np.full((frame_count, slots), np.nan)
    for frame, points in enumerate(points_by_frame):
        for slot, point in enumerate(points):
            mouth[frame, slot] = point
            tail[frame, slot] = np.asarray(point) + np.array([10.0, 0.0])
            score[frame, slot] = 0.9
    return CandidateWindow(
        frame_start=0,
        frame_end=frame_count - 1,
        mouth_xy=mouth,
        tail_xy=tail,
        instance_score=score.copy(),
        mouth_score=score.copy(),
        tail_score=score.copy(),
        visibility=np.where(np.isfinite(score), 3, 0).astype(np.uint8),
        source_instance=np.where(np.isfinite(score), 0, -1).astype(np.int16),
    )


def test_reciprocal_tracklets_bridge_a_short_gap_without_global_six_ids():
    window = _window(
        [
            [(0, 0), (40, 0), (80, 0), (130, 0)],
            [(1, 0), (41, 0), (81, 0), (131, 0)],
            [],
            [(3, 0), (43, 0), (83, 0), (133, 0)],
        ]
    )
    result = build_reciprocal_tracklets(
        window,
        config=ConservativeTrackletConfig(
            separation_body_lengths=0.5,
            maximum_missing_gap_frames=2,
            base_prediction_gate_body_lengths=0.5,
        ),
    )

    assert result.summary["fragment_count"] == 4
    assert result.tracklets.tracklet_id.nunique() == 4
    assert result.summary["reciprocal_accepted_bridges"] == 8
    assert result.bridges.loc[
        result.bridges.gap_frames.eq(1), "reciprocal_accepted"
    ].all()


def test_close_mouths_are_unassigned_instead_of_forced():
    window = _window(
        [
            [(0, 0), (40, 0), (80, 0)],
            [(1, 0), (2, 0), (81, 0)],
            [(2, 0), (38, 0), (82, 0)],
        ]
    )
    result = build_reciprocal_tracklets(
        window,
        config=ConservativeTrackletConfig(
            separation_body_lengths=0.5,
            minimum_fragment_observations=1,
        ),
    )

    close = result.dispositions[
        result.dispositions.frame.eq(1)
        & result.dispositions.candidate_slot.isin([0, 1])
    ]
    assert set(close.disposition) == {"mouth_proximity_ambiguous"}
    assert (close.tracklet_id == -1).all()


def test_low_global_assignment_margin_breaks_instead_of_merging():
    window = _window(
        [
            [(0, 0), (100, 0)],
            [(49, 0), (51, 0)],
        ]
    )
    result = build_reciprocal_tracklets(
        window,
        config=ConservativeTrackletConfig(
            separation_body_lengths=0.0,
            assignment_ambiguity_margin_body_lengths=0.5,
            base_prediction_gate_body_lengths=10.0,
            maximum_displacement_body_lengths_per_frame=10.0,
            minimum_fragment_observations=1,
        ),
    )

    assert not result.bridges.reciprocal_accepted.any()
    assert result.summary["fragment_count"] == 4


def test_clean_motion_distribution_reports_body_normalized_quantiles():
    window = _window(
        [
            [(0, 0), (40, 0), (80, 0)],
            [(1, 0), (41, 0), (81, 0)],
            [(2, 0), (42, 0), (82, 0)],
            [(3, 0), (43, 0), (83, 0)],
        ]
    )
    summary = clean_motion_distributions(
        window, minimum_separation_body_lengths=1.5
    )

    assert summary["body_length_px"] == 10.0
    assert summary["displacement"]["n"] == 9
    assert summary["displacement"]["median_body_lengths"] == 0.1
    assert summary["constant_velocity_prediction_error"]["p99_body_lengths"] == 0.0


def test_tracklet_admission_uses_mouth_score_not_pose_instance_score():
    window = _window(
        [
            [(0, 0), (40, 0), (80, 0)],
            [(1, 0), (41, 0), (81, 0)],
        ]
    )
    window.instance_score[:, :] = 0.01
    window.mouth_score[:, :] = 0.9
    result = build_reciprocal_tracklets(
        window,
        config=ConservativeTrackletConfig(
            minimum_mouth_score=0.5,
            minimum_fragment_observations=1,
        ),
    )

    assert set(result.dispositions.disposition) == {"retained_tracklet"}
    assert "instance_score" not in result.tracklets.columns


def test_frozen_training_control_scale_overrides_window_tail_scale():
    training_control = _window(
        [[(0, 0), (40, 0), (80, 0)], [(1, 0), (41, 0), (81, 0)]]
    )
    frozen_scale = frozen_body_length_px_from_training_controls(
        [training_control]
    )
    evaluation = _window(
        [[(0, 0), (40, 0), (80, 0)], [(1, 0), (41, 0), (81, 0)]]
    )
    evaluation = replace(
        evaluation,
        tail_xy=evaluation.mouth_xy + np.array([100.0, 0.0]),
    )

    result = build_reciprocal_tracklets(
        evaluation,
        frozen_body_length_px=frozen_scale,
        config=ConservativeTrackletConfig(minimum_fragment_observations=1),
    )

    assert frozen_scale == 10.0
    assert result.summary["body_length_px"] == 10.0
    assert result.summary["body_length_source"] == "training_control_frozen"
