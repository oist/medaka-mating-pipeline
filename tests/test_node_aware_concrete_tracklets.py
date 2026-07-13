from __future__ import annotations

import numpy as np

from scripts.tracking.node_aware_concrete_tracklets import (
    Detection,
    PRESETS,
    build_node_aware_tracklets,
)


REFERENCE = {
    "body_length_px": 10.0,
    "median_log_length": float(np.log(10.0)),
    "mad_log_length": 0.1,
    "reference_count": 1,
}


def _det(frame: int, raw_index: int, x: float, y: float = 0.0, angle_deg: float = 0.0) -> Detection:
    direction = np.asarray([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))]) * 5.0
    centroid = np.asarray([x, y], dtype=float)
    mouth = centroid + direction
    tail = centroid - direction
    return Detection(
        frame=frame,
        raw_index=raw_index,
        mouth=mouth,
        tail=tail,
        centroid=centroid,
        body_vector=mouth - tail,
        body_length=10.0,
        score=1.0,
    )


def test_node_aware_tracklets_keep_three_smooth_fish():
    frames = {
        frame: [_det(frame, 0, frame), _det(frame, 1, 100 + frame), _det(frame, 2, 300 + frame)]
        for frame in range(20)
    }
    rows, _, stats = build_node_aware_tracklets(
        frames, config=PRESETS["balanced"], reference=REFERENCE
    )
    assert stats["n_fragments"] == 3
    assert stats["coverage_frac_of_fish_frames"] == 1.0
    assert sorted(rows.groupby("tracklet_id").size()) == [20, 20, 20]


def test_close_pair_breaks_but_isolated_fish_continues():
    frames = {}
    for frame in range(20):
        second_x = 4.0 if 8 <= frame <= 10 else 100.0
        frames[frame] = [
            _det(frame, 0, 0.0),
            _det(frame, 1, second_x),
            _det(frame, 2, 300.0),
        ]
    rows, qc, stats = build_node_aware_tracklets(
        frames, config=PRESETS["balanced"], reference=REFERENCE
    )
    assert any(rows.groupby("tracklet_id").size() == 20)
    assert stats["frames_close_segments"] >= 3
    assert qc.loc[qc.frame.between(8, 10), "break_reasons"].str.contains("close_body_segments").any()


def test_tangent_body_part_mislink_breaks_tracklet():
    frames = {
        frame: [
            _det(frame, 0, frame, angle_deg=90.0 if frame == 6 else 0.0),
            _det(frame, 1, 100 + frame),
            _det(frame, 2, 300 + frame),
        ]
        for frame in range(14)
    }
    _, qc, stats = build_node_aware_tracklets(
        frames, config=PRESETS["balanced"], reference=REFERENCE
    )
    assert stats["n_fragments"] > 3
    assert any(
        reason in qc.loc[qc.frame == 6, "break_reasons"].iloc[0]
        for reason in ("body_vector_turn", "endpoint_prediction_jump")
    )


def test_false_negative_preserves_two_observed_tracklets():
    frames = {}
    for frame in range(15):
        rows = [_det(frame, 0, frame), _det(frame, 1, 100 + frame)]
        if frame != 7:
            rows.append(_det(frame, 2, 300 + frame))
        frames[frame] = rows
    observations, _, stats = build_node_aware_tracklets(
        frames, config=PRESETS["balanced"], reference=REFERENCE
    )
    lengths = observations.groupby("tracklet_id").size().tolist()
    assert sum(length == 15 for length in lengths) == 2
    assert stats["frames_raw_count_lt_3"] == 1
