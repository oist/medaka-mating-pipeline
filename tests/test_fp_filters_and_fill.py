"""Tests for the geometric 2D FP filter (load_2d_slp) and the gap-capped, provenance-
marked 3D fill (fill_missing_3d) — the entry-phantom root-cause fixes."""
import numpy as np
import pandas as pd

from scripts.reconstruction_3d.load_2d_slp import _drop_fp_instances, _body_span, _pair_overlap
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    ensure_all_frames_and_fill_spline,
)


class FakeInst:
    """Minimal stand-in for a sleap_io instance: .numpy() (n_nodes,2) + .score."""
    def __init__(self, mouth, tail, score):
        self._pts = np.array([mouth, tail], dtype=float)
        self.score = score

    def numpy(self):
        return self._pts


REAL = lambda s=0.9: FakeInst((100, 100), (100, 200), s)          # span 100, vertical


# ---------------------------------------------------------------- geometry helpers
def test_body_span_and_overlap_basics():
    assert _body_span(np.array([[100, 100], [100, 200]])) == 100.0
    assert _body_span(np.array([[150, 150], [152, 151]])) < 5.0          # collapsed
    # aligned duplicate ~0 overlap; a far instance ~large
    a = np.array([[100, 100], [100, 200]]); b = np.array([[101, 101], [101, 201]])
    assert _pair_overlap(a, b) < 2.0
    # flipped duplicate (mouth<->tail swapped) also detected
    bf = np.array([[100, 200], [100, 100]])
    assert _pair_overlap(a, bf) < 1e-6
    far = np.array([[300, 100], [300, 200]])
    assert _pair_overlap(a, far) > 100.0


# ---------------------------------------------------------------- collapsed
def test_collapsed_dropped_when_full_present():
    """frame-1009 analog: a degenerate FP sitting on a real fish is dropped."""
    real = REAL()
    fp = FakeInst((150, 150), (152, 151), 0.2)          # span ~2px
    kept = _drop_fp_instances([real, fp], min_span_px=15.0, dup_dist_px=12.0)
    assert kept == [real]


def test_single_node_partial_kept_with_full_present():
    """A real fish with only its mouth localized (tail NaN) must NOT be treated as
    collapsed — the 3rd fish in the top view is frequently mouth-only."""
    real_a = REAL(0.9)
    real_b = FakeInst((300, 100), (300, 200), 0.9)
    partial = FakeInst((1084, 884), (np.nan, np.nan), 0.91)   # mouth-only, distinct spot
    kept = _drop_fp_instances([real_a, real_b, partial], min_span_px=15.0, dup_dist_px=12.0)
    assert partial in kept and len(kept) == 3


def test_lone_collapsed_instance_kept():
    """A genuinely head-on lone fish (short span, no other instance) must survive."""
    fp = FakeInst((150, 150), (152, 151), 0.8)
    kept = _drop_fp_instances([fp], min_span_px=15.0, dup_dist_px=12.0)
    assert kept == [fp]                                  # n<=1 short-circuit


def test_all_collapsed_no_full_keeps_some():
    """If no full-length instance exists, do not nuke every instance."""
    a = FakeInst((150, 150), (152, 151), 0.3)
    b = FakeInst((400, 400), (402, 401), 0.3)
    kept = _drop_fp_instances([a, b], min_span_px=15.0, dup_dist_px=12.0)
    assert len(kept) == 2                                # neither dropped (no full present, not overlapping)


# ---------------------------------------------------------------- duplicate / flip
def test_aligned_duplicate_drops_lower_score():
    real = REAL(0.9)
    dup = FakeInst((101, 101), (101, 201), 0.4)          # same fish, lower score
    kept = _drop_fp_instances([real, dup], min_span_px=15.0, dup_dist_px=12.0)
    assert kept == [real]


def test_flipped_duplicate_drops_lower_score():
    real = REAL(0.9)
    flip = FakeInst((100, 200), (100, 100), 0.5)         # head/tail swapped, lower score
    kept = _drop_fp_instances([real, flip], min_span_px=15.0, dup_dist_px=12.0)
    assert kept == [real]


def test_two_distinct_fish_both_kept():
    """The safety case: two real, full-length, separated fish are both preserved."""
    a = REAL(0.9)
    b = FakeInst((300, 100), (300, 200), 0.85)
    kept = _drop_fp_instances([a, b], min_span_px=15.0, dup_dist_px=12.0)
    assert len(kept) == 2


# ---------------------------------------------------------------- fill gap-cap
def _df(frames, xyz):
    a = np.array(xyz, float)
    return pd.DataFrame({"frame": frames, "X": a[:, 0], "Y": a[:, 1], "Z": a[:, 2]})


def test_fill_does_not_bridge_large_gap():
    """A seed then a long silence then a real point: the gap stays NaN (no phantom)."""
    dfs = {"ID_1": _df([0, 500], [[0, 0, 0], [100, 100, 100]])}
    all_frames = list(range(0, 501))
    out = ensure_all_frames_and_fill_spline(dfs, all_frames, max_gap_frames=100)["ID_1"]
    mid = out[out.frame == 250].iloc[0]
    assert not np.isfinite(mid.X)                        # gap 500 > cap 100 -> NOT bridged
    assert bool(out[out.frame == 0].iloc[0].X == 0.0)    # endpoints preserved


def test_fill_bridges_small_gap_and_marks_interpolated():
    dfs = {"ID_1": _df([0, 50], [[0, 0, 0], [50, 50, 50]])}
    all_frames = list(range(0, 51))
    out = ensure_all_frames_and_fill_spline(dfs, all_frames, max_gap_frames=100)["ID_1"]
    mid = out[out.frame == 25].iloc[0]
    assert np.isfinite(mid.X) and abs(mid.X - 25.0) < 1e-6   # gap 50 <= cap -> linearly filled
    assert bool(mid.is_interpolated) is True
    # observed endpoints are NOT flagged interpolated
    assert bool(out[out.frame == 0].iloc[0].is_interpolated) is False
    assert bool(out[out.frame == 50].iloc[0].is_interpolated) is False


def test_fill_uncapped_legacy_bridges_everything():
    dfs = {"ID_1": _df([0, 500], [[0, 0, 0], [100, 100, 100]])}
    out = ensure_all_frames_and_fill_spline(dfs, list(range(0, 501)), max_gap_frames=None)["ID_1"]
    assert np.isfinite(out[out.frame == 250].iloc[0].X)     # None = no cap (back-compat)


def test_fill_can_preserve_observed_and_interpolated_negative_coordinates():
    dfs = {
        "ID_1": _df(
            [0, 2],
            [[-2.0, -4.0, -6.0], [-4.0, -8.0, -10.0]],
        )
    }
    out = ensure_all_frames_and_fill_spline(
        dfs,
        [0, 1, 2],
        max_gap_frames=100,
        ensure_non_negative=False,
    )["ID_1"]

    np.testing.assert_allclose(
        out[["X", "Y", "Z"]].to_numpy(),
        [
            [-2.0, -4.0, -6.0],
            [-3.0, -6.0, -8.0],
            [-4.0, -8.0, -10.0],
        ],
    )
    assert out["is_interpolated"].tolist() == [False, True, False]


def test_shift_to_positive_then_fill_remains_nonnegative():
    best = {
        0: {"groups": {"ID_1": [-4.0, -8.0, -10.0]}},
        2: {"groups": {"ID_1": [-2.0, -4.0, -6.0]}},
    }
    shifted = best_by_frame_to_dfs(
        best,
        shift_to_positive=True,
        ensure_non_negative=False,
        num_individuals=1,
    )
    out = ensure_all_frames_and_fill_spline(
        shifted,
        [0, 1, 2],
        max_gap_frames=100,
        ensure_non_negative=True,
    )["ID_1"]

    assert np.isfinite(out[["X", "Y", "Z"]].to_numpy()).all()
    assert (out[["X", "Y", "Z"]].to_numpy() >= 0).all()
    assert out["is_interpolated"].tolist() == [False, True, False]
