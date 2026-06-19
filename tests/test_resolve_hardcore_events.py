from __future__ import annotations

import numpy as np

from scripts.tracking.resolve_hardcore_events import classify_event

TH = dict(margin_mm=15.0, max_err_mm=30.0, step_max_mm=20.0, near_miss_mm=10.0)


def _v(x, y, z):
    return np.array([float(x), float(y), float(z)])


def test_clean_crossing_clears():
    # a moves +x, b moves -x; linker continues each through (post matches velocity)
    v, r, nse, swe = classify_event(
        _v(-10, 0, 0), _v(2, 0, 0), _v(10, 0, 0), _v(-2, 0, 0),
        _v(11, 0, 0), _v(-11, 0, 0), gap=10, min_3d=6.0, max_step=2.0, **TH)
    assert v == "clear" and r == "velocity_consistent"
    assert nse < swe


def test_swap_is_flagged_as_suspect():
    # same kinematics, but the linker's post assignment is swapped
    v, r, nse, swe = classify_event(
        _v(-10, 0, 0), _v(2, 0, 0), _v(10, 0, 0), _v(-2, 0, 0),
        _v(-11, 0, 0), _v(11, 0, 0), gap=10, min_3d=6.0, max_step=2.0, **TH)
    assert v == "review" and r == "suspect_swap"
    assert swe < nse


def test_stationary_dwell_is_ambiguous_not_cleared():
    # both nearly stationary and close -> both hypotheses fit within noise
    v, r, _, _ = classify_event(
        _v(0, 0, 0), _v(0, 0, 0), _v(3, 0, 0), _v(0, 0, 0),
        _v(0, 0, 0), _v(3, 0, 0), gap=10, min_3d=3.0, max_step=0.5, **TH)
    assert v == "review" and r == "ambiguous"


def test_teleport_is_flagged():
    v, r, _, _ = classify_event(
        _v(-10, 0, 0), _v(2, 0, 0), _v(10, 0, 0), _v(-2, 0, 0),
        _v(11, 0, 0), _v(-11, 0, 0), gap=10, min_3d=6.0, max_step=99.0, **TH)
    assert v == "review" and r == "teleport"


def test_nan_positions_flagged_no_3d():
    v, r, _, _ = classify_event(
        None, None, _v(10, 0, 0), _v(-2, 0, 0),
        _v(11, 0, 0), _v(-11, 0, 0), gap=10, min_3d=6.0, max_step=2.0, **TH)
    assert v == "review" and r == "no_3d"


def test_near_miss_clears_when_barely_approached():
    # ambiguous velocity (small margin) BUT they never got within near_miss_mm
    v, r, _, _ = classify_event(
        _v(0, 0, 0), _v(0, 0, 0), _v(3, 0, 0), _v(0, 0, 0),
        _v(0, 0, 0), _v(3, 0, 0), gap=10, min_3d=11.0, max_step=0.5, **TH)
    assert v == "clear" and r == "near_miss"
