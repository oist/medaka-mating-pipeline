"""Unit tests for the pure per-frame resolution/fusion core (sam2_id_experiment.src.id_model.
per_frame_resolve) — the production form of the dual-line Line-B step2/step3 logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sam2_id_experiment.src.id_model import per_frame_resolve as pfr


def test_forward_fill_slot_basic():
    slot_ff, conf = pfr.forward_fill_slot([10, 100], [0, 1], n_frames=120, high_dist=25)
    assert slot_ff[5] == 0 and slot_ff[50] == 0          # nearest preceding anchor (frame 10 -> slot 0)
    assert slot_ff[100] == 1 and slot_ff[119] == 1
    assert conf[10] == "high" and conf[100] == "high"
    assert conf[50] == "low"                              # 40 frames from frame-10, 50 from frame-100


def test_forward_fill_slot_empty():
    slot_ff, conf = pfr.forward_fill_slot([], [], n_frames=5, high_dist=25)
    assert (slot_ff == -1).all() and (conf == "low").all()


def test_confident_anchors_filters_and_sorts():
    f, s = pfr.confident_anchors([20, 0, 10], [1, 0, -1], [0.9, 0.9, 0.9], margin_thr=0.3)
    assert list(f) == [0, 20] and list(s) == [0, 1]      # slot -1 dropped, sorted by frame
    f2, _ = pfr.confident_anchors([0, 1], [0, 1], [0.1, 0.9], margin_thr=0.3)
    assert list(f2) == [1]                                # low-margin dropped


def test_female_slot_from_probs():
    assert pfr.female_slot_from_probs([0.9, 0.1, 0.3]) == (0, pytest.approx(0.6))
    fs, m = pfr.female_slot_from_probs([np.nan, 0.2, 0.8])
    assert fs == 2 and m == pytest.approx(0.6)
    assert pfr.female_slot_from_probs([np.nan, np.nan, np.nan]) == (-1, pytest.approx(np.nan, nan_ok=True))


def test_beta_from_probs():
    assert pfr.beta_from_probs([0.8, 0.1, 0.3], female_slot=1) == (0, pytest.approx(0.5))
    assert pfr.beta_from_probs([0.1, 0.1, 0.3], female_slot=0) == (2, pytest.approx(0.2))
    b, m = pfr.beta_from_probs([0.8, np.nan, 0.3], female_slot=2)  # only one male crop present
    assert b == -1 and np.isnan(m)
    assert pfr.beta_from_probs([0.8, 0.1, 0.3], female_slot=-1) == (-1, pytest.approx(np.nan, nan_ok=True))


def test_resolve_conflict_beta_and_alpha():
    beta_final, conflict = pfr.resolve_conflict_beta([2, 1, 0], [2, 0, 1])
    assert list(conflict) == [True, False, False]
    assert list(beta_final) == [0, 1, 0]                  # frame0: female in slot2 -> geo fallback gid_0
    alpha = pfr.alpha_from([2, 0, 1], beta_final)
    assert list(alpha) == [1, 2, 2]
    # every frame: female, beta, alpha are the 3 distinct slots
    for f, b, a in zip([2, 0, 1], beta_final, alpha):
        assert {int(f), int(b), int(a)} == {0, 1, 2}


def test_resolve_male_track():
    female_ff = np.full(25, 1, dtype=int)                 # female always slot 1 -> males in {0,2}
    pfm = pd.DataFrame({"frame": [0, 10, 20],
                        "pb_slot0": [0.8, 0.2, 0.1],      # M2 in slot0 @0, slot2 @10/20
                        "pb_slot1": [0.0, 0.0, 0.0],
                        "pb_slot2": [0.2, 0.7, 0.9]})
    out = pfr.resolve_male_track(pfm, female_ff, n_frames=25, margin_thr=0.3, high_dist=25)
    assert out["beta_final"][0] == 0 and out["beta_final"][10] == 2
    assert list(out["anchor_frames"]) == [0, 10, 20]
    # alpha is always the remaining non-female slot
    for f in (0, 10, 20):
        assert {1, int(out["beta_final"][f]), int(out["alpha_final"][f])} == {0, 1, 2}
    assert out["male_conf"][0] == "high"


def test_fuse_side_top_layered():
    n = 60
    female_ff = np.full(n, 1, dtype=int)
    side_beta = np.zeros(n, dtype=int)                    # side says M2 = slot 0 everywhere
    side_conf = np.array(["high"] * 30 + ["low"] * 30, dtype=object)
    # top confident anchor at frame 32 (in the side-low span) says M2 = slot 2
    beta_final, male_conf = pfr.fuse_side_top(side_beta, side_conf, [32], [2], female_ff,
                                              n_frames=n, high_dist=25)
    assert beta_final[10] == 0 and male_conf[10] == "high"   # side-high frame untouched
    assert beta_final[32] == 2 and male_conf[32] == "high"   # side-low frame filled by top
    assert male_conf[5] == "high"                            # still side-high
    # a side-low frame > high_dist from the top anchor stays low + falls back to side beta
    assert male_conf[59] == "low" and beta_final[59] == 0    # |59-32| = 27 > 25


def test_assemble_individual_labels_one_of_each():
    female_ff = np.array([1, 1], dtype=int)
    beta_final = np.array([0, 2], dtype=int)
    frames = np.array([0, 0, 0, 1, 1, 1])
    slots = np.array([0, 1, 2, 0, 1, 2])
    labels = pfr.assemble_individual_labels(frames, slots, female_ff, beta_final, n_frames=2)
    assert list(labels[:3]) == ["M2", "F", "M1"]
    assert list(labels[3:]) == ["M1", "F", "M2"]
    for fr in (0, 1):
        sub = labels[frames == fr]
        assert sorted(sub) == ["F", "M1", "M2"]


def test_female_dense_from_track():
    track = pd.DataFrame({"frame": [0, 0, 0, 1, 1, 1],
                          "app_female_slot": [2, 2, 2, 0, 0, 0]})
    ff = pfr.female_dense_from_track(track, n_frames=2)
    assert ff[0] == 2 and ff[1] == 0


def test_bridge_low_spans_classifies_runs():
    # high 0-2, low 3-5, high 6-8, low 9-11, high 12-14, low 15-17 (edge at end)
    conf = np.array(["high"] * 3 + ["low"] * 3 + ["high"] * 3 + ["low"] * 3 + ["high"] * 3 + ["low"] * 3, dtype=object)
    slot_ff = np.array([0] * 12 + [1] * 6, dtype=int)   # anchors: slot0 up to frame 11, slot1 from 12
    id_source, runs = pfr.bridge_low_spans(slot_ff, conf, n_frames=18)
    assert id_source[0] == "high"
    assert id_source[4] == "bridged"      # run 3-5 between slot0 anchors
    assert id_source[10] == "ambiguous"   # run 9-11 between slot0 (@8) and slot1 (@12) -> swap
    assert id_source[16] == "edge"        # run 15-17 has no confident anchor after
    kinds = {(s, e): k for (s, e, k, _b, _a) in runs}
    assert kinds[(3, 5)] == "bridged" and kinds[(9, 11)] == "ambiguous" and kinds[(15, 17)] == "edge"


def test_localize_ambiguous_swaps_at_closest_approach():
    conf = np.array(["high"] * 3 + ["low"] * 3 + ["high"] * 3 + ["low"] * 3 + ["high"] * 3 + ["low"] * 3, dtype=object)
    slot_ff = np.array([0] * 12 + [1] * 6, dtype=int)
    _id, runs = pfr.bridge_low_spans(slot_ff, conf, n_frames=18)
    dist = np.full(18, 10.0)
    dist[10] = 1.0                         # slots 0 and 1 cross at frame 10
    out, swaps = pfr.localize_ambiguous_swaps(slot_ff, runs, {frozenset((0, 1)): dist})
    assert swaps == [10]
    assert out[9] == 0 and out[10] == 1 and out[11] == 1   # transition placed at the crossing
    assert out[4] == 0                     # bridged run untouched


def _toy_id_track(n=20):
    """M1=slot0, M2=slot1, F=slot2 for frames [0,9]; then M1/M2 SWAP slots for [10,19]."""
    rows = []
    for f in range(n):
        m1, m2 = (0, 1) if f < 10 else (1, 0)
        lab = {m1: "M1", m2: "M2", 2: "F"}
        for s in (0, 1, 2):
            rows.append({"frame": f, "geometry_slot": s, "individual_id": lab[s], "male_conf": "high"})
    return pd.DataFrame(rows)


def test_appearance_swap_across_event_detects_swap():
    track = _toy_id_track()
    label_fs, conf = pfr.dense_label_and_conf(track, n_frames=20)
    # event over the crossing on slots 0 and 1: before {0:M1,1:M2}, after {0:M2,1:M1} -> SWAP
    v, before, after = pfr.appearance_swap_across_event(0, 1, start=9, end=10, label_fs=label_fs, conf=conf)
    assert v == "swap" and before == ("M1", "M2") and after == ("M2", "M1")


def test_appearance_swap_across_event_no_swap():
    track = _toy_id_track()
    label_fs, conf = pfr.dense_label_and_conf(track, n_frames=20)
    # an event fully inside the first regime (no crossing) -> identities unchanged
    v, before, after = pfr.appearance_swap_across_event(0, 1, start=3, end=5, label_fs=label_fs, conf=conf)
    assert v == "no_swap" and before == ("M1", "M2") and after == ("M1", "M2")


def test_resolve_female_track():
    pff = pd.DataFrame({"frame": [0, 10, 20],
                        "p_slot0": [0.9, 0.1, 0.1],   # female slot0 @0, slot2 @10/20
                        "p_slot1": [0.1, 0.2, 0.2],
                        "p_slot2": [0.2, 0.8, 0.9]})
    out = pfr.resolve_female_track(pff, n_frames=25, margin_thr=0.3, high_dist=25)
    assert out["female_ff"][0] == 0 and out["female_ff"][10] == 2 and out["female_ff"][20] == 2
    assert out["app_conf"][0] == "high"
    assert list(out["anchor_slots"]) == [0, 2, 2]


def test_top_male_anchors():
    female_ff = np.full(25, 1, dtype=int)              # female slot1 -> males {0,2}
    top = pd.DataFrame({"frame": [0, 10],
                        "tm_slot0": [0.8, 0.2],        # M2 slot0 @0, slot2 @10
                        "tm_slot1": [0.0, 0.0],
                        "tm_slot2": [0.2, 0.7]})
    taf, tabeta = pfr.top_male_anchors(top, female_ff, margin_thr=0.3)
    assert list(taf) == [0, 10] and list(tabeta) == [0, 2]


def test_dense_label_and_conf_tolerates_float_frame():
    # a float 'frame' column (e.g. from a CSV with NaN rows round-tripped) must not crash the indexing
    track = _toy_id_track(6)
    track["frame"] = track["frame"].astype(float)
    label_fs, conf = pfr.dense_label_and_conf(track, n_frames=6)
    assert label_fs[0, 2] == "F" and conf[0] == "high"


def test_appearance_swap_uncertain_when_no_confident_neighbour():
    track = _toy_id_track()
    label_fs, conf = pfr.dense_label_and_conf(track, n_frames=20)
    conf[:] = "low"                                   # nothing confident nearby
    v, b, a = pfr.appearance_swap_across_event(0, 1, start=9, end=10, label_fs=label_fs, conf=conf)
    assert v == "uncertain" and b is None and a is None
