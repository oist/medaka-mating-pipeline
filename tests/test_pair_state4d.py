import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pair_state4d as P


def _nodes(f_head, f_pec, m1_head, m1_pec):
    nodes = np.full((1, 3, 3, 3), np.nan)
    nodes[0, 0, 0], nodes[0, 0, 1] = f_head, f_pec
    nodes[0, 1, 0], nodes[0, 1, 1] = m1_head, m1_pec
    return nodes


def test_pair_variables_conventions():
    # F at origin facing +x (towards M1); M1 at x=10, 5 mm lower, facing +y (left of its pec->F line? check sign)
    nodes = _nodes([1, 0, 0], [0, 0, 0], [10, 1, 0], [10, 0, -5])
    v = P.pair_variables(nodes, "F", "M1")[0]
    assert v[0] == pytest.approx(10.0)
    assert v[1] == pytest.approx(5.0)  # F above M1
    assert v[2] == pytest.approx(0.0)  # F faces M1
    # M1: line to F is -x, head is +y: rotating -x to +y is clockwise seen from above -> negative
    assert v[3] == pytest.approx(-np.pi / 2)


def test_pair_variables_invalid_when_too_close_or_nan():
    nodes = _nodes([1, 0, 0], [0, 0, 0], [0.5, 1, 0], [0.5, 0, 0])
    assert np.isnan(P.pair_variables(nodes, "F", "M1", min_xy=1.0)).all()
    nodes = _nodes([1, 0, 0], [0, 0, 0], [np.nan, 1, 0], [10, 0, 0])
    assert np.isnan(P.pair_variables(nodes, "F", "M1")).all()


def test_swap_variables_matches_reversed_pair():
    rng = np.random.default_rng(0)
    nodes = rng.normal(size=(50, 3, 3, 3)) * 20
    ab = P.pair_variables(nodes, "M1", "M2")
    ba = P.pair_variables(nodes, "M2", "M1")
    np.testing.assert_allclose(P.swap_variables(ab), ba, equal_nan=True)


def test_symmetric_dz_edges():
    e = P.symmetric_dz_edges(np.arange(1, 101, dtype=float), 4)
    assert len(e) == 5 and np.isinf(e[0]) and np.isinf(e[-1])
    np.testing.assert_allclose(e[1:-1], [-e[3], 0.0, e[3]])
    with pytest.raises(ValueError):
        P.symmetric_dz_edges(np.ones(5), 3)


def test_swap_state_map_is_exact_on_data():
    rng = np.random.default_rng(1)
    nodes = rng.normal(size=(400, 3, 3, 3)) * 30
    ab = P.pair_variables(nodes, "M1", "M2")
    ok = np.isfinite(ab).all(axis=1)
    dz_edges = P.symmetric_dz_edges(ab[ok, 1], 6)
    edges = P.make_edges(60, 4, True, dz_edges, 8)
    _, s_ab, widths = P.encode(ab, edges)
    _, s_ba, _ = P.encode(P.pair_variables(nodes, "M2", "M1"), edges)
    m = P.swap_state_map(widths)
    good = (s_ab >= 0) & (s_ba >= 0)
    assert good.sum() > 300
    np.testing.assert_array_equal(m[s_ab[good]], s_ba[good])
    np.testing.assert_array_equal(m[m], np.arange(len(m)))  # involution


def test_encode_overflow_bin():
    edges = P.make_edges(10, 2, True, np.array([-np.inf, 0, np.inf]), 2)
    vals = np.array([[3.0, 1.0, 0.1, -0.1], [50.0, -1.0, 0.1, 0.1], [np.nan, 0, 0, 0]])
    bins, state, widths = P.encode(vals, edges)
    assert widths == [3, 2, 2, 2]
    assert bins[1, 0] == 2 and state[2] == -1
    np.testing.assert_array_equal(P.decode(state[:2], widths), bins[:2])


def test_mirror_partners():
    # 4 states; swap pairs 0<->1 and 2<->3 (3 is self-symmetric with itself via cluster)
    swap = np.array([1, 0, 3, 2])
    labels = np.array([0, 1, 2, 2])
    t = P.mirror_partners(labels, swap, np.array([5, 5, 3, 3]))
    assert list(t["partner"]) == [1, 0, 2]
    assert list(t["mutual"]) == [True, True, True]
    assert list(t["self_symmetric"]) == [False, False, True]
    assert list(t["merged"]) == [0, 0, 2]


def test_pair_frame_labels_direction_and_priority():
    bouts = pd.DataFrame({
        "start": [10, 12, 30, 40, 50],
        "end": [20, 14, 35, 42, 55],
        "who": ["M1->M2", "M2->M1", "M1<->M2", "M1->F", "M2->M1"],
        "behavior": ["follow", "Chase", "parallel circle", "Courtship", "sneak"],
    })
    classes = {"aggression": ["chase"], "parallel_circling": ["parallel circle"], "approach_follow": ["follow"]}
    frames = np.arange(5, 65)
    lab = P.pair_frame_labels(bouts, "M1", "M2", frames, classes, directed=True)
    at = lambda f: lab[f - 5]
    assert at(5) == "none"
    assert at(11) == "approach_follow a->b"
    assert at(13) == "aggression b->a"  # higher priority than follow
    assert at(31) == "parallel_circling a<->b"
    assert at(41) == "none"  # M1->F is not this pair
    assert at(52) == "unknown"
    lab2 = P.pair_frame_labels(bouts, "M2", "M1", frames, classes, directed=True)
    assert lab2[13 - 5] == "aggression a->b"


def test_fit_label_map_and_enrichment():
    clusters = np.array([0, 0, 0, 1, 1, -1])
    labels = np.array(["x", "x", "y", "none", "y", "x"], dtype=object)
    m = P.fit_label_map(clusters, labels, 0.6)
    assert m.set_index("cluster").loc[0, "mapped_label"] == "x"
    assert m.set_index("cluster").loc[1, "mapped_label"] == "y"
    e = P.enrichment(clusters, labels, [0, 1], ["x", "y"])
    assert e.loc[0, "x"] == pytest.approx((2 / 3) / (2 / 5))


def test_circular_mean():
    assert P.circular_mean(np.array([np.pi - 0.1, -np.pi + 0.1])) == pytest.approx(np.pi, abs=1e-9) or \
        P.circular_mean(np.array([np.pi - 0.1, -np.pi + 0.1])) == pytest.approx(-np.pi, abs=1e-9)


def test_detection_target_and_scores():
    labels = np.array(["qc", "none", "att a->b", "unknown", "wrap", "qc"], dtype=object)
    y = P.detection_target(labels, "att")
    assert list(y) == [0, 0, 1, -1, 0, 0]
    yq = P.detection_target(labels, "qc")
    clusters = np.array([0, 0, 1, 1, 1, -1])
    s = P.cluster_scores(clusters, yq, 3, prior_frames=0.0)
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(0.0)
    s2 = P.cluster_scores(clusters, yq, 3, prior_frames=1e9)
    assert s2[0] == pytest.approx(0.25) and s2[2] == pytest.approx(0.25)
    fs = P.frame_scores(clusters, s, default=0.3)
    assert fs[-1] == pytest.approx(0.3)


def test_rolling_mean_blocks_respects_blocks():
    x = np.array([0, 0, 3, 3, 9, np.nan])
    block = np.array([0, 0, 0, 1, 1, 1])
    out = P.rolling_mean_blocks(x, block, 3)
    assert out[2] == pytest.approx(1.5)   # mean of 0, 3 (next frame is another block)
    assert out[5] == pytest.approx(9.0)


def test_best_f1_and_bout_recall():
    score = np.array([0.1, 0.2, 0.8, 0.9, 0.15])
    y = np.array([0, 0, 1, 1, -1])
    thr, f1 = P.best_f1_threshold(score, y)
    assert f1 == pytest.approx(1.0) and 0.2 < thr <= 0.8
    pred = score >= thr
    assert P.bout_recall(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 0]), np.zeros(5)) == (1, 2)


def test_paper_tau_and_effective_clusters():
    assert P.paper_tau([1, 5, 10, 20], [0.1, 0.12, 0.15, 0.6]) == 10
    assert P.effective_clusters(np.array([5, 5, 5, 5])) == pytest.approx(4.0)


def test_make_edges_list():
    e = P.make_edges_list([0, 10, 30], True, np.array([-np.inf, 0, np.inf]), 4)
    assert list(e[0][:3]) == [0, 10, 30] and np.isinf(e[0][-1]) and len(e[2]) == 5
    with pytest.raises(ValueError):
        P.make_edges_list([5, 10], False, np.array([-np.inf, 0, np.inf]), 4)


def test_fish_speed_central_and_fallback():
    t = np.arange(10, dtype=float)
    pts = np.column_stack([2.0 * t, np.zeros(10), np.zeros(10)])   # 2 mm per frame
    pts[5] = np.nan
    v = P.fish_speed(pts, fps=10.0, half_window=2)
    assert np.isnan(v[5])
    np.testing.assert_allclose(v[np.isfinite(v)], 20.0)   # every other frame falls back to a shorter difference
    assert np.isfinite(v).sum() == 9


def test_swap_with_speed_and_extra_state():
    rng = np.random.default_rng(3)
    nodes = rng.normal(size=(300, 3, 3, 3)) * 30
    ab = P.pair_variables(nodes, "M1", "M2")
    ba = P.pair_variables(nodes, "M2", "M1")
    sp = rng.uniform(0, 200, size=(300, 2))
    ab6 = np.column_stack([ab, sp])
    ba6 = np.column_stack([ba, sp[:, ::-1]])
    np.testing.assert_allclose(P.swap_variables(ab6), ba6, equal_nan=True)
    ok = np.isfinite(ab).all(axis=1)
    dz_edges = P.symmetric_dz_edges(ab[ok, 1], 4)
    edges = P.make_edges_list([0, 20, 40], True, dz_edges, 8, P.quantile_edges(sp.ravel(), [0.5, 0.9]))
    _, s_ab, widths = P.encode(ab6, edges)
    _, s_ba, _ = P.encode(ba6, edges)
    m = P.swap_state_map(widths, n_extra=1)
    n = int(np.prod(widths))
    assert len(m) == n + 1 and m[n] == n
    good = (s_ab >= 0) & (s_ba >= 0)
    np.testing.assert_array_equal(m[s_ab[good]], s_ba[good])
    np.testing.assert_array_equal(m[m], np.arange(len(m)))


def test_quantile_edges():
    e = P.quantile_edges(np.array([1.0, 2, 3, 4, np.nan]), [0.5])
    assert e[0] == 0 and np.isinf(e[-1]) and e[1] == pytest.approx(2.5)


def test_nostate_cluster_and_decide_classes():
    cl = P.with_nostate_cluster(np.array([0, -1, 1, -1]), np.array([True, False, True, True]), 2)
    assert list(cl) == [0, 2, 1, -1]
    out = P.decide_classes({"qc": np.array([0.5, 0.1, 0.4, np.nan]), "att": np.array([0.01, 0.03, 0.5, 0.0])},
                           {"qc": 0.4, "att": 0.02})
    assert list(out) == ["qc", "att", "att", "other"]


def test_bout_precision():
    y = np.array([0, 1, 1, 0, 0, -1, -1, 0, 0])
    pred = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    assert P.bout_precision(y, pred, np.zeros(9)) == (1, 2)   # run on excluded frames not counted


def test_smooth_tracks_keeps_every_frame():
    rng = np.random.default_rng(0)
    nodes = np.cumsum(rng.normal(size=(40, 3, 3, 3)), axis=0)
    nodes[10, 0] = np.nan          # fish 0: gap splits runs of 10 and 29 frames
    nodes[:5, 1] = np.nan          # fish 1: run of 35 frames
    out = P.smooth_tracks(nodes, 11)
    assert np.array_equal(np.isfinite(out), np.isfinite(nodes))     # nothing dropped, nothing invented
    assert np.allclose(out[:10, 0], nodes[:10, 0])                  # run shorter than the window: unchanged
    assert not np.allclose(out[11:, 0], nodes[11:, 0])              # long run: smoothed
    line = np.linspace(0, 1, 40)[:, None, None, None] * np.ones((40, 3, 3, 3))
    assert np.allclose(P.smooth_tracks(line, 11), line)            # a straight line is kept
    assert np.allclose(P.smooth_tracks(nodes, 1)[np.isfinite(nodes)], nodes[np.isfinite(nodes)])


def test_series_rate_wraps_angles():
    t = np.arange(20) / 10.0
    ang = np.angle(np.exp(1j * (3.0 + 2.0 * t)))   # crosses +-pi
    r = P.series_rate(ang, 10.0, 2, circular=True)
    assert np.allclose(r, 2.0)
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    r = P.series_rate(x, 1.0, 1)
    assert np.isnan(r[2]) and r[0] == pytest.approx(1.0) and r[4] == pytest.approx(1.0)


def _circling(n=50, fps=10.0):
    # F still at the origin facing +x; M1 circles it at radius 10 mm, 1 rad/s, while sinking 1 mm/s
    t = np.arange(n) / fps
    nodes = np.full((n, 3, 3, 3), np.nan)
    nodes[:, 0, 1] = 0.0
    nodes[:, 0, 0] = [3.0, 0.0, 0.0]
    nodes[:, 0, 2] = [-3.0, 0.0, 0.0]
    pec = np.column_stack([10 * np.cos(t), 10 * np.sin(t), -t])
    tangent = np.column_stack([-np.sin(t), np.cos(t), np.zeros(n)])
    nodes[:, 1, 1] = pec
    nodes[:, 1, 0] = pec + 3 * tangent + [0, 0, 1.0]   # head up
    nodes[:, 1, 2] = pec - 3 * tangent
    return nodes, fps


def test_extra_variables_circling():
    nodes, fps = _circling()
    v = P.extra_variables(nodes, "F", "M1", fps, 3)
    names = list(P.VARIABLES_EXTRA)
    mid = slice(5, -5)
    assert np.allclose(v[mid, names.index("bearing_rate")], 1.0, atol=1e-3)
    assert np.allclose(v[mid, names.index("turn_b")], 1.0, atol=1e-3)      # M1 heading turns with the circle
    assert np.allclose(v[:, names.index("turn_a")], 0.0)                    # F does not turn
    assert np.allclose(v[:, names.index("bend_a")], 0.0, atol=1e-7)         # straight bodies
    assert np.all(v[:, names.index("pitch_b")] > 0) and np.allclose(v[:, names.index("pitch_a")], 0.0)
    d = np.sqrt(100 + (np.arange(len(nodes)) / fps) ** 2)                  # distance grows: retreating
    assert np.all(v[:, names.index("approach")] < 0)
    assert np.allclose(v[mid, names.index("approach")], -np.gradient(d, 1 / fps)[mid], atol=1e-3)
    w = P.extra_variables(nodes, "M1", "F", fps, 3)                         # swap: pair values equal, fish swapped
    assert np.allclose(w[:, :2], v[:, :2], equal_nan=True)
    assert np.allclose(w[:, [2, 4, 6]], v[:, [3, 5, 7]], equal_nan=True)


def test_extra_variables_bend():
    nodes, fps = _circling(n=5)
    nodes[:, 0, 2] = [0.0, -3.0, 0.0]          # F tail at a right angle
    v = P.extra_variables(nodes, "F", "M1", fps, 1)
    assert np.allclose(v[:, list(P.VARIABLES_EXTRA).index("bend_a")], np.pi / 2)


def test_encode_specs_and_swap_map_specs():
    # columns: D (lin), dz (sym 2 bins), theta_a, theta_b (abs 2 bins), approach (lin, pair), turn_a, turn_b (lin)
    edges = [np.array([0, 10, np.inf]), np.array([-np.inf, 0, np.inf]),
             np.array([0, np.pi / 2, np.pi]), np.array([0, np.pi / 2, np.pi]),
             np.array([-np.inf, 0, np.inf]), np.array([0, 1, np.inf]), np.array([0, 1, np.inf])]
    kinds = ["lin", "sym", "abs", "abs", "lin", "lin", "lin"]
    v = np.array([[5.0, 2.0, -0.3, 3.0, 4.0, 0.5, 2.0],
                  [20.0, -2.0, 3.0, -0.3, 4.0, 2.0, 0.5],      # the same frame seen from b -> a
                  [5.0, np.nan, 0.1, 0.1, 1.0, 0.1, 0.1]])
    bins, state, widths = P.encode_specs(v, edges, kinds)
    assert bins[0].tolist() == [0, 1, 0, 1, 1, 0, 1]
    assert state[2] == -1
    swap = P.swap_state_map_specs(widths, flip=[1], pairs=[(2, 3), (5, 6)], n_extra=1)
    b_sw = bins[1].copy()
    b_sw[0] = bins[0][0]                                        # same D in a real swap
    s_sw = P.joint(b_sw[None, :], widths)[0]
    assert swap[state[0]] == s_sw
    assert np.array_equal(swap[swap], np.arange(len(swap)))      # an involution
    assert swap[-1] == len(swap) - 1                             # extra state fixed
