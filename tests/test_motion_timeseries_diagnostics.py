from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.tracking.motion_timeseries_diagnostics import (
    compute_motion_timeseries,
    summarize_motion_timeseries,
)


def test_motion_timeseries_computes_speed_heading_and_close_contact():
    tracks = pd.DataFrame(
        [
            {"frame": 0, "track_id": 0, "x": 0.0, "y": 0.0},
            {"frame": 1, "track_id": 0, "x": 1.0, "y": 0.0},
            {"frame": 2, "track_id": 0, "x": 2.0, "y": 0.0},
            {"frame": 0, "track_id": 1, "x": 100.0, "y": 0.0},
            {"frame": 1, "track_id": 1, "x": 102.0, "y": 0.0},
            {"frame": 2, "track_id": 1, "x": 3.0, "y": 0.0},
        ]
    )

    out = compute_motion_timeseries(tracks, close_contact_distance_px=5.0, windows=(3,))

    t0 = out[(out["track_id"] == 0) & (out["frame"] == 2)].iloc[0]
    assert np.isclose(t0["speed_px_per_frame"], 1.0)
    assert np.isclose(t0["heading_change_rad"], 0.0)
    assert bool(t0["close_contact"])
    assert np.isclose(t0["nearest_neighbor_distance_px"], 1.0)


def test_motion_timeseries_flags_lateral_turn_and_summarizes():
    tracks = pd.DataFrame(
        [
            {"frame": 0, "track_id": 0, "x": 0.0, "y": 0.0},
            {"frame": 1, "track_id": 0, "x": 1.0, "y": 0.0},
            {"frame": 2, "track_id": 0, "x": 1.0, "y": 1.0},
            {"frame": 3, "track_id": 0, "x": 1.0, "y": 2.0},
        ]
    )

    out = compute_motion_timeseries(tracks, windows=(3,))
    turn = out[out["frame"] == 2].iloc[0]

    assert turn["heading_change_rad"] > 1.5
    assert turn["lateral_accel_px_per_frame2"] > 0.9

    summary = summarize_motion_timeseries(out)
    assert "speed_px_per_frame" in set(summary["metric"])
    assert "q99" in summary.columns
