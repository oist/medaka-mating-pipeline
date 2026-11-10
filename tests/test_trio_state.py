import numpy as np

from scripts.analysis import pair_state4d as P4
from scripts.analysis import trio_state as T


def _fish(pec, heading_deg, length=5.0):
    pec = np.asarray(pec, float)
    h = np.radians(heading_deg)
    d = np.array([np.cos(h), np.sin(h), 0.0])
    return np.stack([pec + length * d, pec, pec - 2 * length * d])


def _frame(f, m1, m2):
    return np.stack([f, m1, m2])[None]


def test_trio_variables_geometry():
    # F at the origin facing +x; M1 10 mm ahead and 5 mm higher, facing F; M2 20 mm to F's left, facing +x
    nodes = _frame(_fish([0, 0, 0], 0), _fish([10, 0, 5], 180), _fish([0, 20, 0], 0))
    v = T.trio_variables(nodes, np.array([[1.0, 2.0, 3.0]]), "M1", "M2")
    assert np.isclose(v["fD"][0], 10) and np.isclose(v["fdz"][0], 5)
    assert np.isclose(v["fthF"][0], 0) and np.isclose(abs(v["fthM"][0]), 0, atol=1e-9)
    assert np.isclose(v["rD"][0], 20) and np.isclose(abs(v["rthF"][0]), np.pi / 2)
    assert np.isclose(v["mD"][0], np.linalg.norm([10, -20, 5]))
    assert v["fsp"][0] == 2 and v["rsp"][0] == 3 and v["Fsp"][0] == 1


def test_angles_undefined_when_fish_overlap_but_distances_kept():
    nodes = _frame(_fish([0, 0, 0], 0), _fish([0, 0, 4], 90), _fish([30, 0, 0], 0))
    v = T.trio_variables(nodes, np.ones((1, 3)), "M1", "M2")
    assert np.isnan(v["fthF"][0]) and np.isclose(v["fD"][0], 0) and np.isclose(v["fdz"][0], 4)
    b, w = T.bin_values(v["fthF"], "circ4")
    assert b[0] == 4 and w == 5


def test_mirror_reflects_left_copies_only():
    a = np.array([0.5, -0.5])
    v = {"fthF": a, "fthM": np.array([0.1, 0.2]), "rthF": np.array([1.0, 1.0]), "fD": np.array([3.0, 3.0])}
    m, flip = T.mirror(v)
    assert flip.tolist() == [False, True]
    assert np.allclose(m["fthF"], [0.5, 0.5]) and np.allclose(m["fthM"], [0.1, -0.2])
    assert np.allclose(m["rthF"], [1.0, -1.0]) and np.allclose(m["fD"], [3.0, 3.0])


def test_mirror_matches_reflected_geometry():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 20, (50, 3, 3, 3))
    ref = pts.copy()
    ref[..., 1] *= -1
    v = T.trio_variables(pts, np.ones((50, 3)), "M1", "M2")
    w = T.trio_variables(ref, np.ones((50, 3)), "M1", "M2")
    for k in T.ANGLE_VARS:
        assert np.allclose(v[k], -w[k])


def test_line_rotation_constant_circling():
    # M1 circles F at 2 deg/frame; window 30 frames -> 60 deg
    t = np.arange(200)
    ang = np.radians(2.0 * t)
    nodes = np.zeros((200, 3, 3, 3))
    nodes[:, 1, P4.PEC, 0] = 20 * np.cos(ang)
    nodes[:, 1, P4.PEC, 1] = 20 * np.sin(ang)
    nodes[:, 2, P4.PEC, 0] = 50
    r = T.line_rotation(nodes, "M1", 30)
    assert np.isnan(r[:15]).all() and np.isnan(r[-15:]).all()
    assert np.allclose(r[15:-15], 60.0)


def test_line_rotation_back_and_forth_cancels():
    t = np.arange(100)
    ang = np.radians(20 * np.sin(2 * np.pi * t / 20))   # oscillation with period 20 frames
    nodes = np.zeros((100, 3, 3, 3))
    nodes[:, 1, P4.PEC, 0] = 20 * np.cos(ang)
    nodes[:, 1, P4.PEC, 1] = 20 * np.sin(ang)
    r = T.line_rotation(nodes, "M1", 20)
    assert np.nanmax(r) < 1e-6


def test_bin_values_kinds():
    b, w = T.bin_values(np.array([5.0, 15.0, 100.0, np.nan]), "lin", [0, 10, 20, np.inf])
    assert b.tolist() == [0, 1, 2, 3] and w == 4
    b, _ = T.bin_values(np.radians([0, 90, 180, -90, -179]), "circ4")
    assert b.tolist() == [0, 1, 2, 3, 2]
    b, _ = T.bin_values(np.radians([10, 50, 100, 170, -170]), "half4")
    assert b.tolist() == [0, 1, 2, 3, 3]


def test_level_code_unique():
    bins = {"a": np.array([0, 1, 0, 1]), "b": np.array([0, 0, 2, 2])}
    c = T.level_code(bins, {"a": 2, "b": 3}, ["a", "b"])
    assert len(np.unique(c)) == 4


def test_backoff_uses_parent_for_unseen_states():
    # fine level: state = (a, b); coarse level: a. a=0 rows are all positive, a=1 rows negative.
    a = np.repeat([0, 1], 500)
    b = np.tile([0, 1], 500)
    y = (a == 0).astype(np.int8)
    fine, coarse = a * 10 + b, a
    te_fine, te_coarse = np.array([0 * 10 + 7, 1 * 10 + 7, 0]), np.array([0, 1, 0])   # two unseen fine states
    s_tr, s_te, used = T.backoff_scores([fine, coarse], y, [te_fine, te_coarse], prior=50)
    assert s_te[0] > 0.9 and s_te[1] < 0.1      # unseen fine states take the parent's rate
    assert used.tolist() == [1, 1, 0]
    assert np.allclose(s_tr[a == 0], s_tr[0])


def test_backoff_single_level_equals_cluster_scores():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 30, 2000)
    y = (rng.random(2000) < 0.1).astype(np.int8)
    y[rng.random(2000) < 0.1] = -1
    s_tr, _s, _u = T.backoff_scores([c], y, [c[:5]], prior=50)
    ref = P4.cluster_scores(c, y, 30, 50)
    assert np.allclose(s_tr, ref[c])


def test_check_nested():
    T.check_nested([["a", "b"], ["a"]])
    try:
        T.check_nested([["a"], ["b"]])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_backoff_majority():
    tr_fine = np.array([1, 1, 2, 3, 3, 3])
    tr_coarse = np.array([10, 10, 10, 20, 20, 20])
    lab = np.array([5, 5, 6, 7, 7, 8])
    out = T.backoff_majority([tr_fine, tr_coarse], lab, [np.array([1, 9, 99]), np.array([10, 20, 30])])
    assert out.tolist() == [5, 7, -1]


def test_alarm_and_bout_counts_match_pair_state4d():
    rng = np.random.default_rng(2)
    y = np.zeros(5000, np.int8)
    for s in rng.integers(0, 4900, 40):
        y[s:s + rng.integers(5, 60)] = 1
    y[rng.random(5000) < 0.05] = -1
    pred = rng.random(5000) < 0.2
    block = np.repeat([0, 1, 2], [2000, 2000, 1000])
    assert T.alarm_counts(y, pred, block) == P4.bout_precision(y, pred, block)
    assert T.bouts_found(y, pred, block) == P4.bout_recall(y, pred, block)


def test_bout_interrupted_by_excluded_frames_counts_once():
    # BORIS bouts before exclusion: frames 2-9 and 14-17; frames 5-6 and all of 14-17 are excluded (-1)
    y_bout = np.zeros(20, np.int8)
    y_bout[2:10] = 1
    y_bout[14:18] = 1
    y = y_bout.copy()
    y[5:7] = -1
    y[14:18] = -1
    pred = np.zeros(20, bool)
    pred[2:5] = True                       # 3 of the 6 usable frames of the first bout
    block = np.zeros(20, int)
    assert T.bouts_found(y, pred, block) == (1, 2)                 # old count: the split bout is two bouts
    assert T.bouts_found(y, pred, block, y_bout=y_bout) == (1, 1)  # one bout, half of its usable frames hit
    t = T.bout_table(y, pred, block, y_bout=y_bout)
    assert t.usable_frames.tolist() == [6] and np.isclose(t.pred_fraction.iloc[0], 0.5)
    c = T.alarm_curve(pred.astype(float), y, block, minutes=1.0, y_bout=y_bout)
    assert (c.bouts == 1).all()


def test_threshold_for_alarm_rate():
    rng = np.random.default_rng(3)
    y = np.zeros(20000, np.int8)
    score = rng.random(20000)
    block = np.zeros(20000, int)
    t = T.threshold_for_alarm_rate(score, y, block, minutes=10.0, target=5.0)
    good, counted = T.alarm_counts(y, score >= t, block)
    assert (counted - good) / 10.0 <= 5.0
