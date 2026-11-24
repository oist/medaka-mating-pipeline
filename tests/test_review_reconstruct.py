"""Reconstruction helpers of the review tools, on a fake orthographic three-camera rig."""

import numpy as np

from scripts.review.reconstruct import (
    IntervalParams,
    RobustParams,
    close_intervals,
    despike_nodes,
    group_fill,
    reestimate_body,
    robust_nodes,
    solve_interval,
)


class FakeCal:
    """Top camera sees (x, y), side 1 (x, z), side 2 (y, z), 2 px per mm."""

    @staticmethod
    def project_world(points):
        pts = np.asarray(points, float)
        return np.stack([pts[..., [0, 1]], pts[..., [0, 2]], pts[..., [1, 2]]], axis=-2) * 2.0

    @staticmethod
    def evaluate_hypotheses(obs):
        obs = np.asarray(obs, float)
        lead = obs.shape[:-2]
        xyz, score = FakeCal._evaluate(obs.reshape(-1, 3, 2))
        return xyz.reshape(lead + (3,)), score.reshape(lead)

    @staticmethod
    def _evaluate(obs):
        obs = obs / 2.0                                                      # (n, 3, 2)
        est = {0: [obs[:, 0, 0], obs[:, 1, 0]], 1: [obs[:, 0, 1], obs[:, 2, 0]], 2: [obs[:, 1, 1], obs[:, 2, 1]]}
        xyz = np.full((len(obs), 3), np.nan)
        score = np.full(len(obs), np.inf)
        spread = np.zeros(len(obs))
        n_ok = np.zeros(len(obs), int)
        for axis, values in est.items():
            vals = np.stack(values, axis=1)
            with np.errstate(invalid="ignore"):
                xyz[:, axis] = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=1)
            both = np.isfinite(vals).all(axis=1)
            spread += np.where(both, (vals[:, 0] - vals[:, 1]) ** 2, 0.0)
            n_ok += np.isfinite(vals).any(axis=1)
        seen = np.isfinite(obs).all(axis=-1).sum(axis=1)
        ok = (seen >= 2) & (n_ok == 3)
        xyz[~ok] = np.nan
        score[ok] = spread[ok]
        return xyz, score


def test_despike_flags_one_and_two_frame_spikes_not_fast_smooth_motion():
    rng = np.random.default_rng(0)
    t = np.arange(200)
    # Swimming with a 10 Hz tail beat (166 fps) and a fast turn around frame 150.
    track = np.stack([t * 0.5, 3 * np.sin(2 * np.pi * 10 * t / 166), np.zeros(200)], axis=1)
    track[140:160, 2] = np.cumsum(np.full(20, 1.5))
    track[160:, 2] = 30.0
    track += rng.normal(0, 0.15, track.shape)
    nodes = track[:, None, None, :].copy()
    nodes[60, 0, 0] += [0, 0, 8.0]
    nodes[100:102, 0, 0] += [6.0, 0, 0]
    spikes = despike_nodes(nodes, RobustParams())
    assert list(np.flatnonzero(spikes[:, 0, 0])) == [60, 100, 101]


def test_robust_nodes_drops_the_view_that_disagrees():
    cal = FakeCal()
    t = np.arange(40)
    truth = np.stack([10 + 0.5 * t, 20 + 0.2 * t, 5 + 0.1 * t], axis=1)                   # (F, 3)
    obs = cal.project_world(truth)[:, None, None]                                           # (F, 1, 1, V, 2)
    obs = obs.copy()
    obs[20, 0, 0, 1] += [0, 30.0]                                                           # side 1 wrong in z
    xyz, res = cal.evaluate_hypotheses(obs.reshape(-1, 3, 2))
    nodes, resid = xyz.reshape(40, 1, 1, 3), res.reshape(40, 1, 1)
    fixed, fixed_res, dropped = robust_nodes(cal, obs, nodes, resid, RobustParams())
    assert dropped[20, 0, 0] == 1
    assert np.linalg.norm(fixed[20, 0, 0] - truth[20]) < 0.5 < np.linalg.norm(nodes[20, 0, 0] - truth[20])
    assert (dropped[np.arange(40) != 20] == -1).all()


def test_reestimate_body_rebuilds_mouth_and_tail_from_the_pectoral_node():
    n = 30
    pect = np.stack([np.arange(n) * 0.5, np.zeros(n), np.zeros(n)], axis=1)
    nodes = np.stack([pect + [9, 0, 0], pect, pect - [16, 0, 0]], axis=1)
    bad = nodes.copy()
    bad[10:15, 2] = pect[10:15] - [5, 3, 0]                   # tail drawn too short and sideways
    flagged = np.zeros(n, bool)
    flagged[10:15] = True
    fixed, source = reestimate_body(bad, np.zeros((n, 3), np.int8), flagged, 0, 2)
    assert np.allclose(fixed[10:15], nodes[10:15], atol=1e-6)
    assert (source[10:15, [0, 2]] == 4).all() and (source[~flagged] == 0).all()


def test_group_fill_moves_a_hidden_fish_with_its_mate():
    n = 20
    a = np.stack([np.zeros(n), np.zeros(n), 100 - 2.0 * np.arange(n)], axis=1)   # sinking
    xyz = np.stack([a, a + [6, 0, 0]], axis=1)
    hidden = xyz.copy()
    hidden[5:15, 1] = np.nan
    out = group_fill(hidden, [0, 1])
    assert np.allclose(out[5:15, 1], xyz[5:15, 1])


def test_close_intervals_merges_pairs_into_groups():
    fps = 10.0
    n = 100
    xyz = np.zeros((n, 3, 3))
    xyz[:, 1, 0] = 50
    xyz[:, 2, 0] = 100
    xyz[20:40, 1, 0] = 5                  # fish 0 and 1 close for 2 s
    xyz[35:50, 2, 0] = 8                  # fish 2 joins at 3.5 s
    xyz[70:72, 2, 0] = 1                  # too short
    df = close_intervals(np.arange(n), xyz, ["ID_1", "ID_2", "ID_3"], fps, IntervalParams(close_mm=12.0))
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["frame_start"], row["frame_end"], row["fish"]) == (20, 49, "ID_1|ID_2|ID_3")


def test_solve_interval_undoes_flickering_swaps_between_keyframes():
    cal = FakeCal()
    n = 60
    t = np.arange(n)
    f0 = np.stack([20 + 0.1 * t, np.full(n, 30.0), 80 - 0.5 * t], axis=1)              # sinking together
    f1 = f0 + [4.0, 3.0, 2.5]                     # apart in every view (side 2 sees y, z)
    proj = cal.project_world(np.stack([f0, f1], axis=1))                                 # (F, 2, V, 2)
    cand = np.moveaxis(proj, 1, 2).copy()                                                # (F, V, m=2, 2)
    # The tracker swapped the two fish in side 1 on frames 20-24 and in all views on 40-44.
    cand += np.random.default_rng(1).normal(0, 0.3, cand.shape)                          # SLEAP jitter
    truth = cand.copy()
    cand[20:25, 1] = cand[20:25, 1, ::-1]
    cand[40:45] = cand[40:45, :, ::-1]
    keys = np.zeros(n, bool)
    keys[[0, n - 1]] = True
    out = solve_interval(cal, cand, keys, IntervalParams())
    fixed = np.take_along_axis(cand, out["perm"][..., None].astype(int), axis=2)
    assert np.allclose(fixed, truth)
    assert out["changed"][20:25].all() and out["changed"][40:45].all() and not out["changed"][:20].any()
