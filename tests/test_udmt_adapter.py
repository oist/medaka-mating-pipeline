from __future__ import annotations

import pandas as pd

from UDMT.bidirectional import BidirectionalConfig, repair_tracks_with_backward
from UDMT.metrics import TrackQualityConfig, score_track_table
from UDMT.refinement import RefinementConfig, refine_tracks_with_masks


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
