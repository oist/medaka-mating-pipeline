import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pose_states as S


def test_param_label_and_stem():
    assert S.param_label({"tau": 7, "bins": 10, "pcs": [1, 2, 3]}) == "tau7-bins10-pcs1_2_3"
    assert S.param_label({}) == "base"
    assert S.param_label({"norm_bl": True, "x": None, "y": False}) == "norm_bl"
    stem = S.file_stem("5", "transition_block", "FM1", {"tau": 7, "bins": 10, "pcs": [1, 2, 3]}, "K")
    assert stem == "05_transition_block__FM1__tau7-bins10-pcs1_2_3__K"
    assert S.split_stem(stem) == {"step": "05", "name": "transition_block", "unit": "FM1",
                                  "params": "tau7-bins10-pcs1_2_3", "set": "K"}
    assert S.parse_label("tau7-norm_bl-sg7o2") == {"tau": "7", "norm_bl": "", "sg": "7o2"}
    assert S.split_stem(S.file_stem("05b", "clusters", "triple", None, "KP"))["step"] == "05b"


def _random_nodes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(100, 30, size=(n, 3, 3, 3))


def test_align_pair_properties_and_feature_roundtrip():
    nodes = _random_nodes()
    points = S.pair_points(nodes, "FM1")
    pose, valid = S.align_pair(points, 1.0)
    assert valid.all()
    assert np.max(np.abs(pose[:, [1, 4], 1])) < 1e-9          # both pec y' = 0
    assert np.max(np.abs(pose[:, 1, 0] + pose[:, 4, 0])) < 1e-9  # pec x' opposite
    assert np.all(pose[:, 1, 0] < 0)                           # fish 1 on -X'
    feat = S.pair_features(pose)
    assert feat.shape == (200, 14)
    assert np.allclose(S.pair_pose_from_features(feat), pose)
    # rotation keeps distances
    d0 = np.linalg.norm(points[:, 0] - points[:, 3], axis=1)
    d1 = np.linalg.norm(pose[:, 0] - pose[:, 3], axis=1)
    assert np.allclose(d0, d1)
    names = S.pair_feature_names("FM1")
    assert names[0] == "F_head_x'" and names[-2:] == ["F_pec_x'", "F_pec_z'"] and len(names) == 14


def test_fit_pca_sign_and_scores():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 4)) @ np.diag([5, 2, 1, 0.5])
    model = S.fit_pca(x, standardize=False, fix_sign=True)
    comps = model["components"]
    for j in range(comps.shape[1]):
        assert comps[np.argmax(np.abs(comps[:, j])), j] > 0
    assert np.isclose(model["explained"].sum(), 1.0)
    scores = S.pca_scores(x, model)
    assert np.allclose(scores.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(np.var(scores, axis=0, ddof=1), model["eigvals"])


def test_acf_ar1_and_segments():
    rng = np.random.default_rng(2)
    phi = 0.9
    x = np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = phi * x[t - 1] + rng.normal()
    z = (x - x.mean()) / x.std()
    seg = np.zeros(len(z), dtype=int)
    seg[10000:] = 1
    curve, weight = S.acf_segments(z, seg, 30)
    assert np.isclose(curve[0, 0], 1.0)
    assert abs(curve[5, 0] - phi ** 5) < 0.05
    tau = S.time_to_e(np.arange(31) / 1.0, curve[:, 0])
    assert abs(tau - (-1 / np.log(phi))) < 1.0


def test_digitize_joint_transitions():
    edges = np.linspace(0, 1, 11)
    b = S.digitize(np.array([0.0, 0.05, 1.0, 1.01, np.nan]), edges)
    assert b.tolist() == [0, 0, 9, -1, -1]
    bins = np.array([[1, 2, 3], [9, 9, 9], [-1, 0, 0]])
    st = S.joint_state(bins, [10, 10, 10])
    assert st.tolist() == [321, 999, -1]
    assert S.decode_state(st[:2], [10, 10, 10]).tolist() == bins[:2].tolist()
    states = np.array([0, 1, 1, 2, 0, 0])
    seg = np.array([0, 0, 0, -1, 1, 1])
    counts, n = S.transition_counts(states, seg, 1, 3)
    assert n == 3 and counts[0, 1] == 1 and counts[1, 1] == 1 and counts[0, 0] == 1
    prob = S.row_normalize(counts)
    rows = np.asarray(prob.sum(axis=1)).ravel()
    assert np.allclose(rows[[0, 1]], 1) and rows[2] == 0
    labels = np.array([0, 0, 1])
    assert S.within_fraction(counts, labels) == 1.0


def test_segment_ids_split_at_video_boundary():
    valid = np.array([1, 1, 1, 0, 1, 1], bool)
    video = np.array([0, 0, 1, 1, 1, 1])
    assert S.segment_ids(valid, video).tolist() == [0, 0, 1, -1, 2, 2]


def test_save_fig_writes_index_and_thumb(tmp_path):
    pytest.importorskip("PIL")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = S.RunLayout(tmp_path / "run", dpi=50)
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    ax.set_title("hello")
    p = layout.save_fig(fig, "1", "cum_var", "FM1", None, "K", verbose=False)
    assert p.name == "01_cum_var__FM1__base__K.png"
    assert (layout.thumbs / p.name).exists()
    idx = layout.read_index()
    assert idx.loc[0, "title"] == "hello" and idx.loc[0, "step"] == "01"
    fig, ax = plt.subplots()
    layout.save_fig(fig, "1", "cum_var", "FM1", None, "K", verbose=False)  # same file twice -> one row
    assert len(layout.read_index()) == 1
    assert isinstance(layout.read_index(), pd.DataFrame)
