from __future__ import annotations

import numpy as np

from scripts.tracking.bind_tracklets_3d import (
    _merged_runs, bind_tracklet_points, detect_hardcore_events, pairwise_3d,
)


def test_pairwise_3d_distances():
    xyz = {
        "ID_1": np.array([[0.0, 0.0, 0.0]]),
        "ID_2": np.array([[3.0, 4.0, 0.0]]),
        "ID_3": np.array([[0.0, 0.0, 12.0]]),
    }
    pd_ = pairwise_3d(xyz)
    assert abs(pd_[("ID_1", "ID_2")][0] - 5.0) < 1e-9
    assert abs(pd_[("ID_1", "ID_3")][0] - 12.0) < 1e-9
    assert abs(pd_[("ID_2", "ID_3")][0] - 13.0) < 1e-9  # sqrt(9+16+144)


def test_merged_runs_merges_small_gaps():
    mask = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert _merged_runs(mask, merge_gap=1) == [(1, 2), (4, 5), (9, 9)]
    assert _merged_runs(mask, merge_gap=2) == [(1, 5), (9, 9)]  # idx 2->4 gap 2 merges
    assert _merged_runs(np.zeros(5, dtype=bool), merge_gap=3) == []


def test_detect_hardcore_events_finds_close_window():
    frames = np.arange(20)
    d = np.full(20, 100.0)
    d[8:13] = 5.0
    pdist = {
        ("ID_1", "ID_2"): d,
        ("ID_1", "ID_3"): np.full(20, 100.0),
        ("ID_2", "ID_3"): np.full(20, 100.0),
    }
    ev = detect_hardcore_events(pdist, frames, hardcore_mm=12.0)
    assert len(ev) == 1
    assert ev[0]["start"] == 8 and ev[0]["end"] == 12 and ev[0]["n_frames"] == 5
    assert ev[0]["pair"] == "ID_1|ID_2"
    assert ev[0]["min_mm"] == 5.0


def test_bind_tracklet_points_picks_nearest_slot_with_margin():
    N = 10
    reproj = np.zeros((3, N, 2))
    reproj[0, :, 0] = 0.0
    reproj[1, :, 0] = 100.0
    reproj[2, :, 0] = 200.0
    f2i = {f: f for f in range(N)}
    points = [(f, 101.0, 0.0, 1.0) for f in range(N)]
    res = bind_tracklet_points(points, reproj, f2i)
    assert res["slot"] == 1                 # nearest to x=101
    assert res["consistency"] == 1.0
    assert res["mean_resid"] < 2.0
    assert res["median_margin"] > 90.0      # runner-up (slot2 at 200) ~99 px away
