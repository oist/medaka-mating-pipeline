"""Tests for the strict same-frame constrained pair sampler (self-supervised IDs)."""

import pandas as pd

from sam2_id_experiment.src.id_dataset.pair_sampler import sample_constrained_pairs


def _meta():
    # two tracklets co-occur on frames 0..4 (different fish), tracklet 1 continues 5..9
    rows = []
    cid = 0
    for f in range(5):
        rows.append({"crop_id": cid, "frame": f, "tracklet_id": 1, "qc_status": "clean_train",
                     "area_ratio_prev": 1.05}); cid += 1
        rows.append({"crop_id": cid, "frame": f, "tracklet_id": 2, "qc_status": "clean_train",
                     "area_ratio_prev": 1.02}); cid += 1
    for f in range(5, 10):
        rows.append({"crop_id": cid, "frame": f, "tracklet_id": 1, "qc_status": "clean_train",
                     "area_ratio_prev": 1.03}); cid += 1
    return pd.DataFrame(rows)


def _tracklet_of(meta, crop_id):
    return int(meta.loc[meta.crop_id == crop_id, "tracklet_id"].iloc[0])


def _frame_of(meta, crop_id):
    return int(meta.loc[meta.crop_id == crop_id, "frame"].iloc[0])


def test_positives_same_tracklet_negatives_same_frame():
    meta = _meta()
    pairs = sample_constrained_pairs(meta, {"random_seed": 0, "positive_pairs_per_tracklet": 100})
    assert not pairs.empty
    pos = pairs[pairs.label == 1]
    neg = pairs[pairs.label == 0]
    # every positive pair: same tracklet
    for _, r in pos.iterrows():
        assert _tracklet_of(meta, r.crop_a_id) == _tracklet_of(meta, r.crop_b_id)
    # every negative pair: SAME frame, DIFFERENT tracklet (the exact cannot-link)
    for _, r in neg.iterrows():
        assert _frame_of(meta, r.crop_a_id) == _frame_of(meta, r.crop_b_id)
        assert _tracklet_of(meta, r.crop_a_id) != _tracklet_of(meta, r.crop_b_id)


def test_no_negative_outside_shared_frame():
    # tracklet 1 frame 9 and tracklet 2 frame 0 must NEVER be a negative (no shared frame)
    meta = _meta()
    pairs = sample_constrained_pairs(meta, {"random_seed": 1})
    neg = pairs[pairs.label == 0]
    # there is no frame where tracklet1's late crops coexist with tracklet2
    late_t1 = set(meta[(meta.tracklet_id == 1) & (meta.frame >= 5)]["crop_id"])
    t2 = set(meta[meta.tracklet_id == 2]["crop_id"])
    for _, r in neg.iterrows():
        assert not ({r.crop_a_id, r.crop_b_id} & late_t1 and {r.crop_a_id, r.crop_b_id} & t2)


def test_area_spike_guard_drops_crops():
    meta = _meta()
    meta.loc[meta.crop_id == 0, "area_ratio_prev"] = 5.0  # spike
    pairs = sample_constrained_pairs(meta, {"random_seed": 0, "max_area_ratio": 2.0})
    used = set(pairs.crop_a_id) | set(pairs.crop_b_id)
    assert 0 not in used


def test_empty_without_clean_train():
    meta = _meta()
    meta["qc_status"] = "reject"
    pairs = sample_constrained_pairs(meta, {})
    assert pairs.empty


def test_negative_cap():
    meta = _meta()
    pairs = sample_constrained_pairs(meta, {"random_seed": 0, "negative_pairs_per_frame": 1,
                                            "max_negative_pairs": 3})
    assert (pairs.label == 0).sum() <= 3
