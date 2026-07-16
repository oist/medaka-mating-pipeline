from __future__ import annotations

import numpy as np

from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


class _CalibrationStub:
    backend_name = "stub"
    score_unit = "px^2"

    def __init__(self, *, scoring: bool):
        self.scoring = scoring

    def triangulate(self, observations):
        if self.scoring:
            raise AssertionError("scoring backend must not supply delivered XYZ")
        observations = np.asarray(observations, dtype=float)
        return observations[..., :, 0]

    def score_hypotheses(self, observations, *, profile):
        if not self.scoring:
            raise AssertionError("geometry backend must not score assignments")
        assert profile == "production_fast"
        observations = np.asarray(observations, dtype=float)
        return np.nansum(observations**2, axis=(-2, -1))


def test_extra_candidate_scoring_can_be_frozen_to_a_second_calibration() -> None:
    geometry = _CalibrationStub(scoring=False)
    scoring = _CalibrationStub(scoring=True)
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
