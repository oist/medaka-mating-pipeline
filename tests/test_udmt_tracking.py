"""Unit tests for the UDMT tracking-by-segmentation core (mask-centroid tracker)."""

import numpy as np
import pandas as pd

from UDMT.tracking import (
    TrackingConfig, track_forward_backward, track_masks_by_centroid,
)


def _masks(rows):
    return pd.DataFrame(rows, columns=["frame", "x", "y", "area", "score"])


def test_two_separated_fish_stable():
    # fish A moves right along y=10, fish B moves right along y=200; far apart
    rows = []
    for f in range(40):
        rows.append((f, 10 + f, 10.0, 500, 0.9))
        rows.append((f, 10 + f, 200.0, 500, 0.9))
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(num_individuals=3))
    assert out["track_id"].nunique() == 2
    # each track stays on its y-lane (no swaps)
    for _, g in out.groupby("track_id"):
        assert g["y"].std() < 1.0


def test_num_individuals_cap():
    # 5 detections per frame but cap = 3 -> at most 3 tracks
    rows = []
    for f in range(20):
        for k, yy in enumerate([10, 60, 110, 160, 210]):
            rows.append((f, 10 + f, float(yy), 500, 0.9))
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(num_individuals=3, max_jump_px=80))
    assert out["track_id"].nunique() <= 3


def test_gap_tolerance_keeps_identity():
    # one fish, missing for 3 frames, should remain a single track
    rows = []
    for f in range(30):
        if 10 <= f < 13:
            continue
        rows.append((f, 5 + 2 * f, 50.0, 500, 0.9))
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(num_individuals=3, max_gap_frames=10))
    assert out["track_id"].nunique() == 1


def test_reflection_does_not_steal_track():
    # a real fish + a far jumpy reflection (low score). With a tight jump gate the
    # reflection cannot capture the fish's track once the fish track exists.
    rows = []
    for f in range(30):
        rows.append((f, 100.0, 100.0 + 0.1 * f, 500, 0.95))   # near-stationary fish
        rows.append((f, 100.0 + (f % 2) * 500, 800.0, 80, 0.3))  # flickering reflection
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(num_individuals=3, max_jump_px=40,
                                                               max_jump_hard_cap_px=80))
    fish = out[out.y < 400]
    # the fish lane is one continuous track
    assert fish["track_id"].nunique() == 1


def test_min_track_len_drops_short():
    rows = [(f, 10.0 + f, 10.0, 500, 0.9) for f in range(3)]
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(min_track_len=5))
    assert out.empty


def test_crossing_can_swap_motion_only():
    # two fish on crossing diagonals meet at the center; a motion-only tracker is
    # expected to be ABLE to swap here (documents the known failure mode).
    rows = []
    for f in range(21):
        rows.append((f, float(f), float(f), 500, 0.9))          # down-right
        rows.append((f, float(f), float(20 - f), 500, 0.9))     # up-right
    out = track_masks_by_centroid(_masks(rows), TrackingConfig(num_individuals=2, max_jump_px=10))
    # tracker produces 2 tracks; whether it swaps is data-dependent, but it must
    # not crash and must keep <=2 identities.
    assert 1 <= out["track_id"].nunique() <= 2


def test_forward_backward_real_frame_order():
    rows = []
    for f in range(20):
        rows.append((f, 10 + f, 10.0, 500, 0.9))
        rows.append((f, 10 + f, 200.0, 500, 0.9))
    fwd, bwd = track_forward_backward(_masks(rows), TrackingConfig(num_individuals=3))
    assert fwd["frame"].min() == 0 and bwd["frame"].min() == 0
    assert set(["frame", "track_id", "x", "y", "score"]).issubset(bwd.columns)
    assert bwd["track_id"].nunique() == 2
