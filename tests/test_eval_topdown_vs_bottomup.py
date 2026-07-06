"""Unit tests for scripts/sleap/eval_topdown_vs_bottomup.py.

Exercises the pure matching / metric helpers on synthetic instances -- no
model, no GPU, no sleap_nn import -- so CI can run them. The module is written
so that everything imported here is numpy/scipy-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.sleap.eval_topdown_vs_bottomup import (
    MetricAccumulator,
    accumulate_frame,
    centroid_cost_matrix,
    count_merged_gt,
    frame_hash,
    hungarian_match,
    instance_centroid,
    is_mouth_tail_swap,
    min_pairwise_centroid_distance,
    node_error,
    select_eval_indices,
)

# Skeleton is [mouth, tail].
MOUTH_IDX, TAIL_IDX = 0, 1


def fish(mouth, tail):
    """Build a (2, 2) instance array from mouth + tail xy."""
    return np.array([mouth, tail], dtype=float)


# ---------------------------------------------------------------------------
# centroid + basic geometry
# ---------------------------------------------------------------------------


def test_instance_centroid_mean_of_visible_nodes():
    c = instance_centroid(fish([0, 0], [10, 0]))
    assert np.allclose(c, [5.0, 0.0])


def test_instance_centroid_ignores_nan_node():
    c = instance_centroid(fish([4, 6], [np.nan, np.nan]))
    assert np.allclose(c, [4.0, 6.0])


def test_instance_centroid_all_nan_is_nan():
    c = instance_centroid(fish([np.nan, np.nan], [np.nan, np.nan]))
    assert np.all(np.isnan(c))


def test_node_error_and_missing():
    assert node_error([0, 0], [3, 4]) == pytest.approx(5.0)
    assert node_error([np.nan, np.nan], [3, 4]) is None
    assert node_error([1, 1], [np.nan, 2]) is None


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_hungarian_matches_nearest_and_gates():
    gt = [fish([0, 0], [10, 0]), fish([100, 100], [110, 100])]
    # pr[0] near gt[0]; pr[1] far from everything (beyond gate).
    pr = [fish([1, 0], [11, 0]), fish([500, 500], [510, 500])]
    cost = centroid_cost_matrix(gt, pr)
    matches, un_gt, un_pr = hungarian_match(cost, gate=50.0)
    assert len(matches) == 1
    gi, pj, d = matches[0]
    assert (gi, pj) == (0, 0)
    assert d < 2.0
    assert un_gt == [1]
    assert un_pr == [1]


def test_hungarian_one_to_one():
    gt = [fish([0, 0], [10, 0]), fish([50, 0], [60, 0])]
    pr = [fish([49, 0], [59, 0]), fish([1, 0], [11, 0])]
    cost = centroid_cost_matrix(gt, pr)
    matches, un_gt, un_pr = hungarian_match(cost, gate=30.0)
    pairs = {(gi, pj) for gi, pj, _ in matches}
    assert pairs == {(0, 1), (1, 0)}
    assert not un_gt and not un_pr


def test_empty_predictions_all_missed():
    gt = [fish([0, 0], [10, 0])]
    cost = centroid_cost_matrix(gt, [])
    matches, un_gt, un_pr = hungarian_match(cost, gate=50.0)
    assert matches == []
    assert un_gt == [0]
    assert un_pr == []


# ---------------------------------------------------------------------------
# mouth<->tail swap
# ---------------------------------------------------------------------------


def test_swap_detected_when_nodes_flipped():
    gt_m, gt_t = [0, 0], [20, 0]
    # prediction with mouth where the tail is and vice versa.
    swap = is_mouth_tail_swap(gt_m, gt_t, pr_mouth=[20, 0], pr_tail=[0, 0])
    assert swap is True


def test_no_swap_when_aligned():
    gt_m, gt_t = [0, 0], [20, 0]
    swap = is_mouth_tail_swap(gt_m, gt_t, pr_mouth=[1, 0], pr_tail=[19, 0])
    assert swap is False


def test_swap_undecidable_with_missing_point():
    swap = is_mouth_tail_swap([0, 0], [20, 0], [np.nan, np.nan], [0, 0])
    assert swap is None


# ---------------------------------------------------------------------------
# close-contact merge
# ---------------------------------------------------------------------------


def test_merge_counts_two_gt_on_one_pred():
    # two GT fish very close; a single prediction sits between them.
    gt = [fish([0, 0], [10, 0]), fish([4, 0], [14, 0])]
    pr = [fish([2, 0], [12, 0])]
    merged = count_merged_gt(gt, pr, gate=50.0)
    assert merged == 1  # one GT beyond the first collapsed onto the shared pred


def test_no_merge_when_each_gt_has_own_pred():
    gt = [fish([0, 0], [10, 0]), fish([100, 0], [110, 0])]
    pr = [fish([1, 0], [11, 0]), fish([101, 0], [111, 0])]
    assert count_merged_gt(gt, pr, gate=50.0) == 0


def test_min_pairwise_distance():
    gt = [fish([0, 0], [10, 0]), fish([40, 0], [50, 0])]
    # centroids at (5,0) and (45,0) -> 40 apart.
    assert min_pairwise_centroid_distance(gt) == pytest.approx(40.0)
    assert np.isinf(min_pairwise_centroid_distance([fish([0, 0], [10, 0])]))


# ---------------------------------------------------------------------------
# deterministic subset selection
# ---------------------------------------------------------------------------


def test_frame_hash_is_deterministic():
    assert frame_hash(2, 100) == frame_hash(2, 100)
    assert frame_hash(2, 100) != frame_hash(2, 101)


def test_select_eval_indices_deterministic_and_capped():
    keys = [(0, i) for i in range(50)]
    a = select_eval_indices(keys, limit=10)
    b = select_eval_indices(keys, limit=10)
    assert a == b
    assert len(a) == 10
    assert a == sorted(a)
    # subset property: the capped selection is contained in the uncapped order.
    full = select_eval_indices(keys, limit=None)
    assert set(a).issubset(set(full))
    assert len(full) == 50


def test_select_eval_indices_none_keeps_all():
    keys = [(1, i) for i in range(7)]
    assert select_eval_indices(keys, None) == list(range(7))


# ---------------------------------------------------------------------------
# end-to-end accumulation over synthetic frames
# ---------------------------------------------------------------------------


def test_accumulate_frame_perfect_prediction():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [10, 0]), fish([100, 0], [110, 0])]
    pr = [fish([0, 0], [10, 0]), fish([100, 0], [110, 0])]
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50.0)
    s = acc.summary()
    assert s["n_gt"] == 2
    assert s["n_matched"] == 2
    assert s["n_missed"] == 0
    assert s["n_extra"] == 0
    assert s["swap_rate"] == 0.0
    assert s["mouth_err_mean"] == pytest.approx(0.0)
    assert s["merge_rate"] == 0.0


def test_accumulate_frame_missed_and_extra():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [10, 0])]
    pr = [fish([300, 300], [310, 300])]  # far away -> not a match
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50.0)
    s = acc.summary()
    assert s["n_missed"] == 1
    assert s["n_extra"] == 1
    assert s["n_matched"] == 0
    assert s["miss_rate"] == 1.0


def test_accumulate_frame_swap_and_error():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [20, 0])]
    pr = [fish([20, 0], [0, 0])]  # swapped nodes, same centroid -> matches
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50.0)
    s = acc.summary()
    assert s["n_matched"] == 1
    assert s["swap_rate"] == 1.0
    assert s["mouth_err_mean"] == pytest.approx(20.0)


def test_accumulate_frame_merge_rate():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [10, 0]), fish([4, 0], [14, 0])]
    pr = [fish([2, 0], [12, 0])]  # one pred for two close GT
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50.0)
    s = acc.summary()
    assert s["n_merged_gt"] == 1
    assert s["merge_rate"] == pytest.approx(0.5)
    # Hungarian is one-to-one so exactly one of the two GT is left unmatched.
    assert s["n_missed"] == 1


def test_optional_oks_fn_used_when_provided():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [10, 0])]
    pr = [fish([0, 0], [10, 0])]

    def fake_oks(g, p):
        return np.array([[0.87]])

    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50.0, oks_fn=fake_oks)
    assert acc.summary()["oks_mean"] == pytest.approx(0.87)
