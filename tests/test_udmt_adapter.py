from __future__ import annotations

import pandas as pd

from UDMT.bidirectional import (
    BidirectionalConfig,
    CrossingConfig,
    annotate_crossing_rows,
    detect_crossing_events,
    detect_motion_anomalies,
    repair_tracks_with_backward,
)
from UDMT.metrics import TrackQualityConfig, score_track_table
from UDMT.refinement import RefinementConfig, refine_tracks_with_masks
from UDMT.resolver import ResolverConfig, resolve_crossings


def _smooth_tracks() -> pd.DataFrame:
    rows = []
    for frame in range(6):
        for track_id, y in enumerate([10.0, 40.0, 70.0]):
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "x": 10.0 + frame * 2.0,
                    "y": y,
                    "score": 0.95,
                }
            )
    return pd.DataFrame(rows)


def test_track_quality_penalizes_large_identity_jump():
    good = _smooth_tracks()
    bad = good.copy()
    bad.loc[(bad["frame"] == 3) & (bad["track_id"] == 1), "x"] = 500.0

    cfg = TrackQualityConfig(num_individuals=3, max_speed_px_per_frame=80.0)
    good_report = score_track_table(good, cfg)
    bad_report = score_track_table(bad, cfg)

    assert good_report.quality_score > bad_report.quality_score
    assert bad_report.jump_rate > 0


def test_refine_tracks_with_masks_moves_centers_inside_gate():
    tracks = pd.DataFrame(
        [
            {"frame": 0, "track_id": 0, "x": 10.0, "y": 10.0, "identity": "ID_1"},
            {"frame": 0, "track_id": 1, "x": 80.0, "y": 10.0, "identity": "ID_2"},
        ]
    )
    masks = pd.DataFrame(
        [
            {
                "frame": 0,
                "candidate_id": "near",
                "centroid_x": 12.0,
                "centroid_y": 11.0,
                "area": 100,
                "identity": "ID_1",
                "quality_score": 1.0,
            },
            {
                "frame": 0,
                "candidate_id": "far",
                "centroid_x": 150.0,
                "centroid_y": 150.0,
                "area": 100,
                "identity": "ID_2",
                "quality_score": 1.0,
            },
        ]
    )

    refined, summary = refine_tracks_with_masks(
        tracks,
        masks,
        RefinementConfig(max_distance_px=20.0, strict_identity_match=True),
    )

    row0 = refined[refined["track_id"] == 0].iloc[0]
    row1 = refined[refined["track_id"] == 1].iloc[0]
    assert row0["udmt_refined"]
    assert row0["x"] == 12.0
    assert row0["y"] == 11.0
    assert not row1["udmt_refined"]
    assert summary.refined_rows == 1


def test_bidirectional_repair_uses_backward_consensus_on_anomaly():
    forward = pd.DataFrame(
        [
            {"frame": 0, "track_id": 0, "x": 0.0, "y": 0.0},
            {"frame": 1, "track_id": 0, "x": 10.0, "y": 0.0},
            {"frame": 2, "track_id": 0, "x": 200.0, "y": 0.0},
            {"frame": 3, "track_id": 0, "x": 210.0, "y": 0.0},
        ]
    )
    backward = pd.DataFrame(
        [
            {"frame": 0, "track_id": "b0", "x": 0.0, "y": 0.0},
            {"frame": 1, "track_id": "b0", "x": 10.0, "y": 0.0},
            {"frame": 2, "track_id": "b0", "x": 20.0, "y": 0.0},
            {"frame": 3, "track_id": "b0", "x": 30.0, "y": 0.0},
        ]
    )

    repaired, summary, votes = repair_tracks_with_backward(
        forward,
        backward,
        BidirectionalConfig(
            max_pair_distance_px=30.0,
            max_speed_px_per_frame=50.0,
            min_mapping_votes=1,
            consensus_blend=1.0,
        ),
    )

    repaired_frame2 = repaired[repaired["frame"] == 2].iloc[0]
    assert repaired_frame2["x"] == 20.0
    assert repaired_frame2["udmt_repair_action"] == "backward_consensus"
    assert summary.repaired_rows >= 1
    assert int(votes["votes"].max()) >= 1


def test_crossing_detector_catches_smooth_swap_without_speed_spike():
    rows = []
    for frame in range(11):
        rows.append({"frame": frame, "track_id": 0, "x": float(frame), "y": 0.0})
        rows.append({"frame": frame, "track_id": 1, "x": float(10 - frame), "y": 0.0})
    tracks = pd.DataFrame(rows)

    motion = detect_motion_anomalies(
        tracks,
        BidirectionalConfig(max_speed_px_per_frame=50.0, max_speed_ratio=4.0),
    )
    events = detect_crossing_events(
        tracks,
        CrossingConfig(distance_threshold=2.1, min_duration_frames=3),
    )
    annotated = annotate_crossing_rows(
        tracks,
        events,
        CrossingConfig(distance_threshold=2.1, min_duration_frames=3),
    )

    assert not motion["udmt_motion_anomaly"].any()
    assert len(events) == 1
    event = events.iloc[0]
    assert int(event["start_frame"]) == 4
    assert int(event["end_frame"]) == 6
    assert int(event["min_frame"]) == 5
    assert annotated["udmt_crossing_candidate"].sum() == 6


def test_crossing_detector_supports_3d_geometry_tracks():
    rows = []
    for frame in range(11):
        rows.append(
            {
                "frame": frame,
                "fish_id": "A",
                "X": float(frame),
                "Y": 0.0,
                "Z": 5.0,
            }
        )
        rows.append(
            {
                "frame": frame,
                "fish_id": "B",
                "X": float(10 - frame),
                "Y": 0.0,
                "Z": 5.0,
            }
        )
    tracks_3d = pd.DataFrame(rows)

    events = detect_crossing_events(
        tracks_3d,
        CrossingConfig(
            id_column="fish_id",
            coordinate_columns=("X", "Y", "Z"),
            distance_threshold=2.1,
            min_duration_frames=3,
        ),
    )

    assert len(events) == 1
    assert events.iloc[0]["track_id_a"] == "A"
    assert events.iloc[0]["track_id_b"] == "B"
    assert events.iloc[0]["coordinate_columns"] == "X,Y,Z"


def test_crossing_review_gate_marks_tight_sustained_events_only():
    rows = []
    for frame in range(11):
        rows.append({"frame": frame, "track_id": "tight_a", "x": float(frame), "y": 0.0})
        rows.append({"frame": frame, "track_id": "tight_b", "x": float(10 - frame), "y": 0.0})
        rows.append({"frame": frame, "track_id": "mild_a", "x": 100.0 + frame, "y": 0.0})
        rows.append({"frame": frame, "track_id": "mild_b", "x": 107.0 + frame, "y": 0.0})
    tracks = pd.DataFrame(rows)

    events = detect_crossing_events(
        tracks,
        CrossingConfig(
            distance_threshold=8.0,
            min_duration_frames=3,
            review_max_min_distance=2.0,
            review_min_duration_frames=3,
        ),
    )

    assert len(events) == 2
    review_events = events[events["review_candidate"]]
    assert len(review_events) == 1
    assert review_events.iloc[0]["track_id_a"] == "tight_a"
    assert review_events.iloc[0]["track_id_b"] == "tight_b"


def test_resolver_suspects_anomalous_fish_and_reviews_ambiguous():
    frames = list(range(70, 151))
    rows = []
    for f in frames:
        # A: smooth except a single big jerk at frame 110 (high velocity std/ratio)
        ax = 0.1 * f + (50.0 if f >= 110 else 0.0)
        rows.append({"frame": f, "fish_id": "A", "X": ax, "Y": 0.0, "Z": 0.0})
        rows.append({"frame": f, "fish_id": "B", "X": 0.1 * f, "Y": 5.0, "Z": 0.0})
        # C, D: both smooth, parallel -> ambiguous pair
        rows.append({"frame": f, "fish_id": "C", "X": 0.1 * f, "Y": 10.0, "Z": 0.0})
        rows.append({"frame": f, "fish_id": "D", "X": 0.1 * f, "Y": 11.0, "Z": 0.0})
    tracks_3d = pd.DataFrame(rows)
    identified = pd.DataFrame(
        {"frame": f, "fish_id": fid, "bind_score": bs}
        for f in frames
        for fid, bs in (("A", 30.0), ("B", 3.0), ("C", 5.0), ("D", 5.0))
    )
    events = pd.DataFrame([
        {"event_id": 0, "track_id_a": "A", "track_id_b": "B", "start_frame": 100,
         "end_frame": 120, "min_frame": 110, "min_distance": 2.0,
         "duration_frames": 21, "review_candidate": True},
        {"event_id": 1, "track_id_a": "C", "track_id_b": "D", "start_frame": 100,
         "end_frame": 120, "min_frame": 110, "min_distance": 1.0,
         "duration_frames": 21, "review_candidate": True},
    ])

    decisions, summary = resolve_crossings(
        events, tracks_3d, identified=identified, config=ResolverConfig()
    )
    d = decisions.set_index("event_id")
    assert d.loc[0, "decision"] == "suspect:A"
    assert d.loc[0, "margin"] >= ResolverConfig().margin_threshold
    assert d.loc[1, "decision"] == "review"
    assert summary.suspect_calls == 1
    assert summary.review_flags == 1
    # read-only: input event table is untouched
    assert "decision" not in events.columns


def test_resolver_residual_cue_maps_slots_to_fish_by_position():
    frames = list(range(70, 151))
    rows = []
    pre_link = {"frames": {}}
    for f in frames:
        a_pos = [0.1 * f, 0.0, 0.0]
        b_pos = [0.1 * f, 5.0, 0.0]
        rows.append({"frame": f, "fish_id": "A", "X": a_pos[0], "Y": a_pos[1], "Z": a_pos[2]})
        rows.append({"frame": f, "fish_id": "B", "X": b_pos[0], "Y": b_pos[1], "Z": b_pos[2]})
        # slot "a" sits on fish A (Y=0), slot "b" on fish B (Y=5); B has high residual
        pre_link["frames"][str(f)] = {
            "groups": {"a": a_pos, "b": b_pos},
            "errors": {"a": 1.0, "b": 50.0},
        }
    tracks_3d = pd.DataFrame(rows)
    events = pd.DataFrame([
        {"event_id": 0, "track_id_a": "A", "track_id_b": "B", "start_frame": 100,
         "end_frame": 120, "min_frame": 110, "min_distance": 2.0,
         "duration_frames": 21, "review_candidate": True},
    ])

    decisions, _ = resolve_crossings(events, tracks_3d, pre_link=pre_link, config=ResolverConfig())
    row = decisions.iloc[0]
    # residual correctly attributed by position: A low, B high
    assert abs(row["resid_a"] - 1.0) < 1e-6
    assert abs(row["resid_b"] - 50.0) < 1e-6
    assert row["decision"] == "suspect:B"
