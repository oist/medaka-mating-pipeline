import numpy as np
import pytest

from scripts.analysis import pair_hmm as H


def test_raw_features_and_contact_fill():
    v = np.array([[9.0, -2.0, 0.0, np.pi / 2, 99.0, 0.0, 0.99, -3.0, 0.0, 1.0, 0.1, 0.2, 0.3, -0.3],
                  [0.5, 1.0, np.nan, np.nan, 10.0, 10.0, np.nan, 2.0, np.nan, np.nan, 0.1, 0.1, 0.0, 0.0]])
    f = H.raw_features(v)
    assert f.shape == (2, len(H.FEATURES))
    names = list(H.FEATURES)
    assert f[0, names.index("log1p_D_xy")] == pytest.approx(np.log(10.0))
    assert f[0, names.index("slog_dz")] == pytest.approx(-np.log(3.0))
    assert f[0, names.index("cos_theta_a")] == pytest.approx(1.0)
    assert f[0, names.index("sin_theta_b")] == pytest.approx(1.0)
    assert f[0, names.index("log_speed_a")] == pytest.approx(2.0)
    assert f[0, names.index("log_bearing_rate")] == pytest.approx(0.0)
    assert np.isnan(f[1, names.index("cos_theta_a")])
    g = H.fill_contact(f, np.array([False, True]), {n: -1.5 for n in H.ANGLE_RATE_FEATURES})
    assert np.isfinite(g).all()
    assert g[1, names.index("sin_theta_a")] == 0.0 and g[1, names.index("log_turn_b")] == -1.5
    assert np.array_equal(g[0], f[0])


def test_runs_split_on_block():
    s, e = H.runs(np.array([1, 1, 0, 1, 1, 1], bool), np.array([0, 0, 0, 0, 1, 1]))
    assert s.tolist() == [0, 3, 4] and e.tolist() == [2, 4, 6]


def test_fit_and_decode_two_state_hmm():
    pytest.importorskip("hmmlearn")
    rng = np.random.default_rng(1)
    z = np.repeat(np.tile([0, 1], 20), 50)                     # 40 dwells of 50 frames
    X = rng.normal(size=(len(z), 2)) * 0.3 + np.where(z[:, None] == 1, 3.0, 0.0)
    model = H.fit_hmm(X, np.array([len(X)]), 2, seed=0, n_iter=30)
    ok = np.ones(len(X), bool)
    ok[100] = False
    st = H.decode_frames(model, X, ok, np.zeros(len(X)))
    assert st[100] == -1
    agree = np.mean((st[ok] == z[ok]))
    assert max(agree, 1 - agree) > 0.99
    assert H.n_parameters(model) == 1 + 2 + 4 + 6


def _toy(seed=2):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.tile([0, 1, 2], 15), 40)
    centres = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    X = centres[z] + rng.normal(size=(len(z), 2)) * 0.4
    return X, z


def test_torch_hmm_recovers_states_and_cuts_pieces():
    pytest.importorskip("torch")
    X, z = _toy()
    lengths = np.array([700, len(X) - 700])
    m = H.TorchGaussianHMM(3, seed=0, window=128, batch_pieces=4).fit(X, lengths, n_iter=40)
    assert np.all(np.diff(m.history) > -1e-6 * len(X))          # EM never decreases the likelihood
    st = m.predict(X, lengths)
    # the best label permutation matches the truth almost everywhere
    from itertools import permutations
    best = max(np.mean(np.array(p)[st] == z) for p in permutations(range(3)))
    assert best > 0.99
    assert np.allclose(m.transmat_.sum(1), 1.0)
    assert H.n_parameters(m) == 2 + 6 + 6 + 9


def test_torch_hmm_likelihood_matches_hmmlearn():
    pytest.importorskip("torch")
    hmm = pytest.importorskip("hmmlearn.hmm")
    X, _z = _toy(3)
    m = H.TorchGaussianHMM(3, seed=0, window=10_000).fit(X, [len(X)], n_iter=15)
    ref = hmm.GaussianHMM(3, covariance_type="full", init_params="", params="")
    ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = m.startprob_, m.transmat_, m.means_, m.covars_
    assert m.score(X, [len(X)]) == pytest.approx(ref.score(X, [len(X)]), rel=1e-8)
    assert np.array_equal(m.predict(X, [len(X)]), ref.predict(X, [len(X)]))
