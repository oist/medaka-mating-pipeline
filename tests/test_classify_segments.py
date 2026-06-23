"""Unit tests for the tracklet segment classifier (pure logic)."""

import numpy as np

from scripts.tracking.classify_segments import (
    classify_segment, run_length, velocity_cosine,
)


def test_run_length():
    assert run_length([0, 0, 0, 1, 1, 2]) == [(0, 3), (1, 2), (2, 1)]
    assert run_length([]) == []


def test_confident_single_id():
    slots = [1] * 100
    margins = [50.0] * 100
    resids = [4.0] * 100
    in_hc = [False] * 100
    r = classify_segment(slots, margins, resids, in_hc, motion_consistency=0.8)
    assert r["label"] == "ID_1" and r["modal_slot"] == 1
    assert r["p1"] == 1.0 and r["reason"] == "confident_single"


def test_mix_segregated_runs():
    # first half clearly fish 0, second half clearly fish 2 -> a crossing merge
    slots = [0] * 50 + [2] * 50
    margins = [40.0] * 100
    resids = [5.0] * 100
    in_hc = [False] * 100
    r = classify_segment(slots, margins, resids, in_hc, mix_p2_min=0.3, min_run=5)
    assert r["label"] == "mix" and r["p2"] >= 0.3


def test_crossing_in_hardcore():
    # noisy slot, mostly inside a hard-core event -> crossing
    slots = [0, 1, 0, 2, 1, 0, 2, 1, 0, 1]
    margins = [2.0] * 10
    resids = [8.0] * 10
    in_hc = [True] * 10
    r = classify_segment(slots, margins, resids, in_hc, consistency_min=0.9, hc_frac_max=0.5)
    assert r["label"] == "crossing"


def test_unknown_thin_margin():
    slots = [1] * 100
    margins = [2.0] * 100      # thin margin -> not confident
    resids = [4.0] * 100
    in_hc = [False] * 100
    r = classify_segment(slots, margins, resids, in_hc, margin_min=10.0)
    assert r["label"] == "unknown" and r["reason"] == "thin_margin_or_resid"


def test_dynamics_veto():
    # confident single by geometry, but moves OPPOSITE to its slot -> unknown
    slots = [1] * 100
    margins = [50.0] * 100
    resids = [4.0] * 100
    in_hc = [False] * 100
    r = classify_segment(slots, margins, resids, in_hc, motion_consistency=-0.9,
                         motion_min=-0.2)
    assert r["label"] == "unknown" and r["reason"] == "motion_inconsistent"


def test_appearance_veto_optional():
    slots = [1] * 100
    margins = [50.0] * 100
    resids = [4.0] * 100
    in_hc = [False] * 100
    # appearance disagrees confidently; only vetoes when use_appearance=True
    off = classify_segment(slots, margins, resids, in_hc, motion_consistency=0.8,
                           app_id=2, app_purity=0.9, use_appearance=False)
    assert off["label"] == "ID_1"
    on = classify_segment(slots, margins, resids, in_hc, motion_consistency=0.8,
                          app_id=2, app_purity=0.9, use_appearance=True)
    assert on["label"] == "unknown" and on["reason"] == "appearance_conflict"
    # appearance agreeing strengthens the reason string
    agree = classify_segment(slots, margins, resids, in_hc, motion_consistency=0.8,
                             app_id=1, app_purity=0.9, use_appearance=True)
    assert agree["label"] == "ID_1" and "appearance" in agree["reason"]


def test_velocity_cosine():
    # parallel motion -> cosine ~1
    t = np.cumsum(np.ones((20, 2)), axis=0)
    s = t + 100.0  # same velocity, offset position
    assert velocity_cosine(t, s) > 0.99
    # opposite motion -> cosine ~ -1
    assert velocity_cosine(t, -t) < -0.99


def test_empty_segment():
    r = classify_segment([], [], [], [])
    assert r["label"] == "unknown" and r["n_frames"] == 0
