import numpy as np
import pytest

from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


class _ZeroBackend:
    backend_name = "polynomial"
    score_unit = "px^2"

    def evaluate_hypotheses(self, observations, *, profile):
        observations = np.asarray(observations, dtype=float)
        leading = observations.shape[:-2]
        enough = np.isfinite(observations).all(axis=-1).sum(axis=-1) >= 2
        xyz = np.zeros((*leading, 3), dtype=float)
        scores = np.zeros(leading, dtype=float)
        xyz[~enough] = np.nan
        scores[~enough] = np.inf
        return xyz, scores

    def triangulate(self, observations):
        return self.evaluate_hypotheses(
            observations, profile="production_fast"
        )[0]

    def score_hypotheses(self, observations, *, profile):
        return self.evaluate_hypotheses(observations, profile=profile)[1]


@pytest.mark.parametrize("missing_view", [0, 1, 2])
def test_six_candidate_path_completes_any_missing_camera(missing_view):
    cal = _ZeroBackend()
    cameras = {}
    for view in range(3):
        count = 2 if view == missing_view else 3
        cameras[f"cam{view}"] = [
            (float(slot + 1), float(view + 1)) for slot in range(count)
        ]
    best = compute_best_assignments_per_frame(
        cal,
        {7: cameras},
        num_individuals=3,
        num_candidates=3,
        max_hypothesis_score=1000,
        partial_view_penalty_score_units=1_000_000,
        global_max_states=None,
    )[7]
    assert all(np.isfinite(best["groups"][label]).all() for label in ("a", "b", "c"))
    assert best["synthetic_views"].count(missing_view) == 1
    assert best["missing_views"] == best["synthetic_views"]
    synthetic_source = best["synthetic_views"].index(missing_view)
    assert best[f"perm_cam{missing_view}"][synthetic_source] == -1
