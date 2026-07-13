from __future__ import annotations

import numpy as np

from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


class _GeometryForward:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return values[:, [0, 2, 4]]


class _ScoringForward:
    def predict(self, values):
        return np.zeros((len(values), 3), dtype=float)


class _ZeroProjection:
    def predict(self, values):
        return np.zeros((len(values), 6), dtype=float)


class _MustNotRun:
    def predict(self, values):
        raise AssertionError("geometry reverse model must not score assignments")


class _CalibrationStub:
    def __init__(self, forward, reverse):
        self._model_im2xyz = forward
        self._model_xyz2im = reverse
        self._model_xy_from_xz_yz = None


def test_extra_candidate_scoring_can_be_frozen_to_a_second_calibration() -> None:
    geometry = _CalibrationStub(_GeometryForward(), _MustNotRun())
    scoring = _CalibrationStub(_ScoringForward(), _ZeroProjection())
    points = {
        0: {
            "cam0": [(1.0, 0.0), (20.0, 0.0)],
            "cam1": [(2.0, 0.0), (30.0, 0.0)],
            "cam2": [(3.0, 0.0), (40.0, 0.0)],
        }
    }
    best = compute_best_assignments_per_frame(
        geometry,
        points,
        scoring_cal=scoring,
        num_individuals=1,
        num_candidates=2,
        recover_missing_cam0=False,
    )
    assert best[0]["perm_cam0"] == [0]
    assert best[0]["perm_cam1"] == [0]
    assert best[0]["perm_cam2"] == [0]
    assert best[0]["groups"]["a"] == [1.0, 2.0, 3.0]
