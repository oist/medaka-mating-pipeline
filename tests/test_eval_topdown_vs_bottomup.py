"""Unit tests for scripts/sleap/eval_topdown_vs_bottomup.py.

Exercises the pure matching / metric helpers on synthetic instances -- no
model, no GPU, no sleap_nn import -- so CI can run them. The module is written
so that everything imported here is numpy/scipy-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.sleap.eval_topdown_vs_bottomup import (
    MetricAccumulator,
    accumulate_frame,
    centroid_arrays_as_mouth_instances,
    centroid_cost_matrix,
    count_merged_gt,
    frame_hash,
    hungarian_match,
    instance_centroid,
    is_mouth_tail_swap,
    manifest_positions,
    match_instances_partial,
    min_pairwise_centroid_distance,
    node_error,
    normalize_difficulty,
    partial_label_cost_matrix,
    run_strategy,
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


def test_hungarian_gate_is_part_of_global_assignment():
    # An unconstrained rectangular solve chooses (0,0)=1 and (1,1)=101, then
    # drops the latter at a 100 px gate. The gated global optimum retains two
    # feasible matches: (0,1)=99 and (1,0)=2.
    cost = np.array([[1.0, 99.0], [2.0, 101.0]])
    matches, un_gt, un_pr = hungarian_match(cost, gate=100.0)
    assert {(gi, pj) for gi, pj, _ in matches} == {(0, 1), (1, 0)}
    assert not un_gt and not un_pr


def test_empty_predictions_all_missed():
    gt = [fish([0, 0], [10, 0])]
    cost = centroid_cost_matrix(gt, [])
    matches, un_gt, un_pr = hungarian_match(cost, gate=50.0)
    assert matches == []
    assert un_gt == [0]
    assert un_pr == []


def test_partial_cost_marks_no_joint_node_unmatchable():
    gt = [fish([0, 0], [np.nan, np.nan])]
    pr = [fish([np.nan, np.nan], [10, 0])]
    cost, basis, joint = partial_label_cost_matrix(
        gt, pr, MOUTH_IDX, TAIL_IDX
    )
    assert np.isinf(cost[0, 0])
    assert basis[0, 0] == "unmatchable"
    assert joint[0][0] == ()
    result = match_instances_partial(gt, pr, MOUTH_IDX, TAIL_IDX, gate=150)
    assert result.matches == ()
    assert result.unmatchable_gt == (0,)


def test_centroid_only_output_becomes_mouth_partial_instance():
    converted = centroid_arrays_as_mouth_instances(
        [np.array([[12.0, 34.0]])], mouth_idx=0, n_nodes=2
    )
    assert np.allclose(converted[0][0], [12.0, 34.0])
    assert np.all(np.isnan(converted[0][1]))


def test_run_strategy_uses_sleap_nn_inference_predict(monkeypatch):
    import scripts.sleap.eval_topdown_vs_bottomup as evaluator
    import sleap_nn.inference

    calls = {}
    sentinel = object()

    def fake_predict(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(sleap_nn.inference, "predict", fake_predict)
    monkeypatch.setattr(
        evaluator,
        "make_prediction_labels_self_contained",
        lambda labels: labels,
    )
    output = run_strategy(
        strategy="oracle-test",
        model_paths=[Path("models/centered")],
        subset_labels="labels",
        max_instances=3,
        batch_size=2,
        device="cpu",
        prediction_mode="oracle_centered_instance",
    )
    assert output is sentinel
    assert calls["source"] == "labels"
    assert calls["model_paths"] == [str(Path("models/centered"))]
    assert calls["centroid_only"] is False
    assert calls["clean_empty_frames"] is False


def test_side2_97340_partial_mouths_avoid_artificial_109px_error():
    # Synthetic analogue: both GT fish have only mouth annotations. Long tails
    # make prediction centroids cross, reproducing the old ~109 px association
    # artifact even though both predicted mouths are close to their GT mouths.
    gt = [
        fish([0, 0], [np.nan, np.nan]),
        fish([109, 0], [np.nan, np.nan]),
    ]
    pr = [
        fish([1, 0], [220, 0]),
        fish([108, 0], [-110, 0]),
    ]
    legacy, _, _ = hungarian_match(centroid_cost_matrix(gt, pr), gate=150)
    assert {(gi, pj) for gi, pj, _ in legacy} == {(0, 1), (1, 0)}

    result = match_instances_partial(gt, pr, MOUTH_IDX, TAIL_IDX, gate=150)
    assert {(m.gt_index, m.pred_index) for m in result.matches} == {
        (0, 0),
        (1, 1),
    }
    assert {m.basis for m in result.matches} == {"mouth"}
    mouth_errors = [
        node_error(gt[m.gt_index][MOUTH_IDX], pr[m.pred_index][MOUTH_IDX])
        for m in result.matches
    ]
    assert mouth_errors == pytest.approx([1.0, 1.0])


def test_side2_97630_partial_mouths_avoid_artificial_51px_error():
    # Same structure at the smaller separation observed in frame 97630.
    gt = [
        fish([0, 0], [np.nan, np.nan]),
        fish([51, 0], [np.nan, np.nan]),
    ]
    pr = [
        fish([1, 0], [104, 0]),
        fish([50, 0], [-52, 0]),
    ]
    result = match_instances_partial(gt, pr, MOUTH_IDX, TAIL_IDX, gate=100)
    assert {(m.gt_index, m.pred_index) for m in result.matches} == {
        (0, 0),
        (1, 1),
    }
    assert max(
        node_error(gt[m.gt_index][0], pr[m.pred_index][0])
        for m in result.matches
    ) == pytest.approx(1.0)


def test_side1_97219_retains_genuine_tail_error():
    gt = [
        fish([0, 0], [20, 0]),
        fish([100, 0], [120, 0]),
    ]
    pr = [
        fish([1, 0], [65.2, 0]),  # correct mouth, genuine 45.2 px tail error
        fish([101, 0], [121, 0]),
    ]
    acc = MetricAccumulator()
    result = accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=150)
    assert {(m.gt_index, m.pred_index) for m in result.matches} == {
        (0, 0),
        (1, 1),
    }
    assert {m.basis for m in result.matches} == {"both"}
    assert max(acc.tail_errs) == pytest.approx(45.2)


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
    assert s["precision_numerator"] == 2
    assert s["precision_denominator"] == 2
    assert s["recall_numerator"] == 2
    assert s["recall_denominator"] == 2
    assert s["f1"] == pytest.approx(1.0)


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


def test_metric_denominators_and_full_partial_subsets():
    acc = MetricAccumulator()
    gt = [
        fish([0, 0], [10, 0]),  # fully labeled
        fish([100, 0], [np.nan, np.nan]),  # partial: mouth only
        fish([200, 0], [210, 0]),  # missed
    ]
    pr = [
        fish([3, 0], [14, 0]),
        fish([102, 0], [160, 0]),
        fish([500, 0], [510, 0]),  # extra
    ]
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50)
    summary = acc.summary()

    assert summary["precision_numerator"] == 2
    assert summary["precision_denominator"] == 3
    assert summary["recall_numerator"] == 2
    assert summary["recall_denominator"] == 3
    assert summary["all_three_detection_numerator"] == 0
    assert summary["all_three_detection_denominator"] == 1

    full = summary["keypoint_subsets"]["fully_labeled"]
    partial = summary["keypoint_subsets"]["partial"]
    assert full["n_gt"] == 2
    assert full["n_matched"] == 1
    assert full["mouth"]["n"] == 1
    assert full["tail"]["n"] == 1
    assert full["mouth"]["pck5_denominator"] == 1
    assert full["mouth"]["pck5_numerator"] == 1
    assert partial["n_gt"] == 1
    assert partial["n_matched"] == 1
    assert partial["mouth"]["n"] == 1
    assert partial["tail"]["n"] == 0
    assert summary["match_basis_counts"] == {"both": 1, "mouth": 1}


def test_shared_tail_rate_has_explicit_frame_denominator():
    acc = MetricAccumulator()
    gt = [fish([0, 0], [10, 0]), fish([100, 0], [110, 0])]
    pr = [fish([0, 0], [50, 0]), fish([100, 0], [52, 0])]
    accumulate_frame(acc, gt, pr, MOUTH_IDX, TAIL_IDX, gate=50)
    summary = acc.summary()
    assert summary["shared_tail_numerator"] == 1
    assert summary["shared_tail_denominator"] == 1
    assert summary["shared_tail_rate"] == 1.0


class _FakeLabeledFrame:
    def __init__(self, frame_idx):
        self.frame_idx = frame_idx


class _FakeLabels:
    def __init__(self, frame_indices):
        self.labeled_frames = [_FakeLabeledFrame(i) for i in frame_indices]


def test_manifest_positions_preserves_order_and_validates_frame_index():
    labels = _FakeLabels([10, 20, 30])
    rows = [
        {
            "frame_key": "b",
            "source_labeled_frame_index": "2",
            "frame_idx": "30",
        },
        {
            "frame_key": "a",
            "source_labeled_frame_index": "0",
            "frame_idx": "10",
        },
    ]
    positions, resolved = manifest_positions(labels, rows)
    assert positions == [2, 0]
    assert [row["frame_key"] for row in resolved] == ["b", "a"]

    bad = [{**rows[0], "frame_idx": "31"}]
    with pytest.raises(ValueError, match="expected 31"):
        manifest_positions(labels, bad)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"difficulty": "ordinary", "node_visibility": "full"}, "ordinary"),
        ({"difficulty": "close-contact/crossing"}, "close_contact"),
        ({"difficulty": "ordinary", "node_visibility": "mouth_only"}, "partial"),
        ({"category": "partial/occluded"}, "partial"),
        (
            {
                "difficulty": "close_contact_or_crossing",
                "node_visibility": (
                    "fish1[mouth:visible,tail:visible];"
                    "fish2[mouth:visible,tail:visible]"
                ),
            },
            "close_contact",
        ),
        (
            {
                "difficulty": "ordinary",
                "node_visibility": "fish1[mouth:hidden,tail:visible]",
            },
            "partial",
        ),
    ],
)
def test_normalize_difficulty(row, expected):
    assert normalize_difficulty(row) == expected


def test_round_dir_overrides_do_not_require_mapped_model_root():
    from scripts.sleap.compare_rounds_topdown import resolve_round_specs

    specs = resolve_round_specs(
        paths_config=Path("definitely_missing_paths_config.toml"),
        round_dir_overrides=[
            r"round7=\\server\share\retrain_20260704",
            r"round8=\\server\share\retrain_20260722_round8",
        ],
    )
    assert specs["round7"][0] == Path(r"\\server\share\retrain_20260704")
    assert specs["round8"][1] == "round8_260723_strictmouth"


def test_explicit_candidate_model_specs_validate_mode_arity(tmp_path):
    from scripts.sleap.compare_rounds_topdown import parse_model_specs

    centroid = tmp_path / "centroid"
    centered = tmp_path / "centered"
    centroid.mkdir()
    centered.mkdir()
    (centroid / "best.ckpt").touch()
    (centered / "best.ckpt").touch()

    component = parse_model_specs(
        [f"sigma20={centroid}"], "centroid_only"
    )
    assert component == {"sigma20": [centroid]}

    full = parse_model_specs(
        [f"sigma20_jitter8={centroid},{centered}"], "full_topdown"
    )
    assert full == {"sigma20_jitter8": [centroid, centered]}

    with pytest.raises(ValueError, match="requires 2 path"):
        parse_model_specs([f"broken={centroid}"], "full_topdown")


def test_explicit_candidate_model_specs_require_best_checkpoint(tmp_path):
    from scripts.sleap.compare_rounds_topdown import parse_model_specs

    model = tmp_path / "incomplete"
    model.mkdir()

    with pytest.raises(ValueError, match="no best.ckpt"):
        parse_model_specs([f"incomplete={model}"], "centroid_only")
