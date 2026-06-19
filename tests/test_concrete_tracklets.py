from __future__ import annotations

from scripts.tracking.concrete_tracklets import (
    build_concrete_tracklets, build_tracklets_edge_extended,
)
from scripts.tracking.compare_anchor_tracklets import seg_seg_dist


def test_edge_extension_keeps_isolated_fish_when_a_pair_is_close():
    fd = {f: [(0.0, 0.0, 1.0), (3.0, 0.0, 1.0), (500.0, 0.0, 1.0)] for f in range(20)}
    # frame-level gate: a pair is always within sep -> no clean frame at all -> 0 tracklets
    _, st_frame = build_concrete_tracklets(fd, num_fish=3, body_length=10.0, sep_bl=1.5)
    assert st_frame["n_fragments"] == 0
    # edge-extension: the isolated fish is individually unambiguous -> one full-length tracklet
    tr, st_ext = build_tracklets_edge_extended(fd, num_fish=3, body_length=10.0,
                                               tau=2.0, sigma0=2.0, nominal_speed=0.5)
    assert st_ext["n_fragments"] == 1
    assert tr[0]["length"] == 20


def test_edge_extension_tracks_separated_then_breaks_on_approach():
    fd = {}
    for f in range(10):
        fd[f] = [(0.0, 0.0, 1.0), (40.0, 0.0, 1.0)]        # separated
    for f in range(10, 15):
        fd[f] = [(0.0, 0.0, 1.0), (4.0, 0.0, 1.0)]         # close -> ambiguous
    for f in range(15, 25):
        fd[f] = [(0.0, 0.0, 1.0), (40.0, 0.0, 1.0)]        # separated again
    _, st = build_tracklets_edge_extended(fd, num_fish=2, body_length=10.0,
                                          tau=2.0, sigma0=2.0, nominal_speed=0.5)
    assert st["n_fragments"] >= 2
    assert st["coverage_frac_of_fish_frames"] > 0.5


def test_edge_extension_pairwise_break_keeps_third_fish():
    # A static; B drifts in to ~6 px around f=10-11 then back out; C far & static.
    fd = {}
    for f in range(22):
        d = abs(40.0 - 3.4 * min(f, 21 - f))      # 40 -> ~6 -> 40
        fd[f] = [(0.0, 0.0, 1.0), (d, 0.0, 1.0), (300.0, 0.0, 1.0)]
    tr, _ = build_tracklets_edge_extended(fd, num_fish=3, body_length=10.0,
                                          tau=2.0, sigma0=2.0, nominal_speed=0.5)
    # the uninvolved fish C is tracked continuously across the whole window (the close
    # pair A/B break around the approach; this also exercises the rival-velocity lookup
    # when both members of the pair break in the same frame).
    assert any(t["length"] == 22 for t in tr)


def test_seg_seg_dist_parallel_crossing_degenerate_and_tshape():
    assert abs(seg_seg_dist([0, 0], [10, 0], [0, 5], [10, 5]) - 5.0) < 1e-9   # parallel
    assert seg_seg_dist([0, 0], [10, 10], [0, 10], [10, 0]) == 0.0            # crossing
    assert abs(seg_seg_dist([0, 0], [0, 0], [3, 0], [3, 0]) - 3.0) < 1e-9     # point-point
    # T-shape: centroids close (0,0)&(0,10) but bodies 5 apart
    assert abs(seg_seg_dist([0, -5], [0, 5], [-5, 10], [5, 10]) - 5.0) < 1e-9


def test_stifd_slow_close_confident_fast_close_breaks():
    import numpy as np
    from scripts.tracking.compare_stifd import per_det_speed, ifd_and_stifd

    def det(c):
        c = np.array(c, float)
        return {"centroid": c, "mouth": None, "tail": None, "score": 1.0, "seg": (c, c)}

    # both scenarios hold the two fish a constant 20 px apart; only the speed differs
    slow = {0: [det([0, 0]), det([20, 0])], 1: [det([1, 0]), det([21, 0])],
            2: [det([2, 0]), det([22, 0])]}                      # ~1 px/frame
    fast = {0: [det([0, 0]), det([20, 0])], 1: [det([15, 0]), det([35, 0])],
            2: [det([30, 0]), det([50, 0])]}                     # ~15 px/frame
    _, st_slow, _ = ifd_and_stifd(slow, per_det_speed(slow, 100), "centroid", 2.0)
    _, st_fast, _ = ifd_and_stifd(fast, per_det_speed(fast, 100), "centroid", 2.0)
    assert st_slow[2] > 4.0      # close but slow -> stays confident (well above a 2-3 sigma gate)
    assert st_fast[2] < 1.5      # equally close but fast -> breaks
    assert st_slow[2] > st_fast[2]


def test_sep_threshold_overrides_sep_bl():
    fd = {f: [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0)] for f in range(4)}   # 5 px apart
    # sep_bl*BL would be 1.5*10=15 (>5 -> not clean); explicit sep_threshold=3 (<5 -> clean)
    _, st = build_concrete_tracklets(fd, num_fish=2, body_length=10.0, sep_bl=1.5,
                                     sep_threshold=3.0)
    assert st["clean_frames"] == 4


def test_min_sep_by_frame_overrides_separation():
    # centroids far apart (default would be clean) but supplied separation says close
    fd = {f: [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)] for f in range(5)}
    _, st = build_concrete_tracklets(
        fd, num_fish=2, body_length=10.0, sep_bl=1.5,
        min_sep_by_frame={f: 5.0 for f in range(5)})
    assert st["clean_frames"] == 0
    assert st["n_fragments"] == 0


def test_breaks_at_close_contact_and_resumes_as_fresh_tracklets():
    fd = {}
    for f in range(10):              # well separated (dist 100 >= 1.5*BL=15)
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    for f in range(10, 15):         # close contact (dist 4 < 15) -> ambiguous
        fd[f] = [(48.0, 0.0, 1.0), (52.0, 0.0, 1.0)]
    for f in range(15, 25):         # separated again
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    tr, st = build_concrete_tracklets(
        fd, num_fish=2, body_length=10.0, sep_bl=1.5, jump_bl=1.0, min_score=0.3,
    )
    assert st["n_fragments"] == 4                 # 2 per clean run, 2 runs
    assert st["clean_frames"] == 20
    assert st["break_events"] == 1
    assert sorted(t["length"] for t in tr) == [10, 10, 10, 10]
    assert st["coverage_frac_of_fish_frames"] == 0.8   # 40 covered / 50 fish-frames


def test_max_gap_bridges_dropouts_but_strict_breaks():
    fd = {}
    for f in range(5):                      # separated, clean
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    fd[5] = [(0.0, 0.0, 1.0)]               # one-frame total dropout (not close contact)
    for f in range(6, 10):                  # separated, clean again
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    common = dict(num_fish=2, body_length=10.0, sep_bl=1.5, jump_bl=1.0, min_score=0.3)
    _, strict = build_concrete_tracklets(fd, max_gap=0, **common)
    _, bridged = build_concrete_tracklets(fd, max_gap=2, **common)
    assert strict["n_fragments"] == 4       # dropout breaks both tracklets -> restart
    assert bridged["n_fragments"] == 2      # the dropout is bridged
    assert bridged["bridged_gaps"] >= 1


def test_max_gap_does_not_bridge_close_contact():
    fd = {}
    for f in range(5):
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    for f in range(5, 8):                    # close contact (a crossing) within the gap window
        fd[f] = [(48.0, 0.0, 1.0), (52.0, 0.0, 1.0)]
    for f in range(8, 13):
        fd[f] = [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)]
    _, st = build_concrete_tracklets(
        fd, num_fish=2, body_length=10.0, sep_bl=1.5, jump_bl=1.0, min_score=0.3, max_gap=5,
    )
    assert st["n_fragments"] == 4           # crossing breaks even though max_gap=5
    assert st["bridged_gaps"] == 0


def test_wrong_count_and_low_score_break_tracking():
    fd = {
        0: [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)],
        1: [(0.0, 0.0, 1.0)],                       # missed fish (count != num_fish)
        2: [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)],
        3: [(0.0, 0.0, 0.1), (100.0, 0.0, 1.0)],    # low score
        4: [(0.0, 0.0, 1.0), (100.0, 0.0, 1.0)],
    }
    _, st = build_concrete_tracklets(
        fd, num_fish=2, body_length=10.0, sep_bl=1.5, jump_bl=1.0, min_score=0.3,
    )
    assert st["clean_frames"] == 3       # frames 0, 2, 4
    # frames 0, 2, 4 are isolated clean frames -> 2 tracklets each = 6 fragments
    assert st["n_fragments"] == 6
