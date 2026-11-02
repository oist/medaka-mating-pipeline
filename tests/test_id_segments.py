import numpy as np
import pandas as pd

from scripts.review import id_segments as S

FPS = 10.0
P = S.SegmentParams(safe_mm=15, high_risk_mm=8, min_safe_s=1.0, max_step_mm=6, slow_pad_s=0.1)


def _xyz(n, gaps):
    """Three far-apart still fish; ``gaps`` = {(a, b): distance of tracks 0 and 1}."""
    xyz = np.zeros((n, 3, 3), np.float32)
    xyz[:, 1, 0], xyz[:, 2, 0] = 100, 300
    for (a, b), d in gaps.items():
        xyz[a:b, 1, 0] = d
    return xyz


def _map(xyz, **kw):
    n = len(xyz)
    present = kw.pop("present", np.ones((n, 3, 3), bool))
    return S.danger_map(xyz, present, np.zeros((n, 3), bool), FPS, P)


def test_runs_are_half_open():
    assert S.runs_of(np.array([0, 1, 1, 0, 1], bool)).tolist() == [[1, 3], [4, 5]]


def test_close_fish_make_a_danger_run_and_very_close_a_slow_one():
    # Steps stay under max_step_mm: a larger one is a jump, which is danger on its own.
    xyz = _xyz(100, {})
    xyz[:, 1, 0] = 16
    xyz[30:40, 1, 0] = 12
    xyz[60:70, 1, 0] = 12
    xyz[63:67, 1, 0] = 7
    m = _map(xyz)
    assert m["runs"].tolist() == [[30, 40], [60, 70]]
    assert m["high"].tolist() == [False, True]
    assert m["slow"][29] and m["slow"][35] and m["slow"][70] and not m["slow"][20]
    assert m["high_frames"][65] and not m["high_frames"][35]


def test_short_safe_stretch_joins_the_danger_around_it():
    xyz = _xyz(100, {})
    xyz[:, 1, 0] = 16
    xyz[30:40, 1, 0] = 12
    xyz[45:55, 1, 0] = 12      # 5 frames (0.5 s) apart, shorter than 1 s
    assert _map(xyz)["runs"].tolist() == [[30, 55]]


def test_unreliable_frames_are_danger():
    xyz = _xyz(100, {})
    xyz[:, 1, 0] = 20
    present = np.ones((100, 3, 3), bool)
    present[50, 2, 1:] = False         # seen in one view
    xyz[20, 2] = np.nan                # no position
    assert _map(xyz, present=present)["runs"].tolist() == [[20, 21], [50, 51]]


def test_change_goes_back_to_the_closest_approach_of_the_latest_danger_run():
    d = np.full(100, 50.0)
    d[20:30] = 12
    d[25] = 4
    d[60:70] = 12
    d[64] = 6
    runs = np.array([[20, 30], [60, 70]])
    assert S.place_change(75, 0, d, runs, -1, 40) == 64
    # Offset first frame: answers are frames, not indices.
    assert S.place_change(1075, 1000, d, runs, 999, 40) == 1064


def test_change_never_reaches_past_an_earlier_change():
    d = np.full(100, 50.0)
    d[60:70] = 12
    d[62] = 3
    d[68] = 9
    assert S.place_change(75, 0, d, np.array([[60, 70]]), 65, 40) == 68


def test_change_without_a_danger_run_uses_the_window():
    d = np.full(100, 50.0)
    d[72] = 30
    assert S.place_change(75, 0, d, np.zeros((0, 2), int), -1, 10) == 72


def test_status_keeps_a_run_without_change_and_drops_a_run_with_one():
    danger = np.zeros(100, bool)
    runs = np.array([[20, 30], [60, 70]])
    for a, b in runs:
        danger[a:b] = True
    st = S.frame_status(1000, 100, 1005, danger, runs, [(1000, 1079)], [1065], [(1040, 1044)], FPS)
    assert (st[:5] == S.NOT_ANALYSED).all()
    assert st[10] == S.WATCHED_SAFE and st[25] == S.WATCHED_DANGER
    assert (st[60:70] == S.UNUSABLE).all() and (st[40:45] == S.UNUSABLE).all()
    assert st[85] == S.UNWATCHED


def test_change_outside_a_run_blocks_only_around_it():
    st = S.frame_status(0, 100, None, np.zeros(100, bool), np.zeros((0, 2), int), [(0, 99)], [50], [], FPS)
    assert (st[45:56] == S.UNUSABLE).all() and st[44] == S.WATCHED_SAFE and st[56] == S.WATCHED_SAFE


def test_segments_and_summary():
    st = np.full(100, S.WATCHED_SAFE, np.int8)
    st[:10] = S.NOT_ANALYSED
    st[40:50] = S.UNUSABLE
    st[70:80] = S.WATCHED_DANGER
    st[90:] = S.UNWATCHED
    seg = S.usable_segments(st, 1000, FPS, lambda t: f"names@{t}")
    assert seg[["start", "end"]].values.tolist() == [[1010, 1039], [1050, 1089]]
    assert seg["seconds"].tolist() == [3.0, 4.0] and seg["danger_frames"].tolist() == [0, 10]
    assert seg["names"].tolist() == ["names@10", "names@50"]
    s = S.summary(st, seg, FPS)
    assert s["usable_pct"] == round(100 * 70 / 90, 2) and s["unusable_pct"] == round(100 * 10 / 90, 2)
    assert s["segments"] == 2 and s["longest_segment_s"] == 4.0


def test_manual_ranges_read_the_ledger_columns():
    e = pd.DataFrame({"kind": ["danger", "swap"], "frame": ["5", "9"], "frame_end": ["8", ""]})
    assert S.manual_ranges(e) == [(5, 8)]


def test_play_seconds():
    assert S.play_seconds(200, 100, 100.0, 2.0, 1.0) == 2.0
