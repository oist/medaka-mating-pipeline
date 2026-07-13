from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


class _Zeros:
    def __init__(self, width):
        self.width = width

    def predict(self, values):
        return np.zeros((len(values), self.width), dtype=float)


@pytest.mark.parametrize("missing_view", [0, 1, 2])
def test_six_candidate_path_completes_any_missing_camera(missing_view):
    helpers = [_Zeros(2), _Zeros(2), _Zeros(2)]
    cal = SimpleNamespace(
        _model_im2xyz=_Zeros(3),
        _model_xyz2im=_Zeros(6),
        _model_xy_from_xz_yz=helpers[0] if missing_view == 0 else None,
        _model_xz_from_xy_yz=helpers[1] if missing_view == 1 else None,
        _model_yz_from_xz_xy=helpers[2] if missing_view == 2 else None,
    )
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
        max_residual_per_fish=1000,
        partial_view_penalty=1_000_000,
        global_max_states=None,
    )[7]
    assert all(np.isfinite(best["groups"][label]).all() for label in ("a", "b", "c"))
    assert best["synthetic_views"].count(missing_view) == 1
    synthetic_source = best["synthetic_views"].index(missing_view)
    assert best[f"perm_cam{missing_view}"][synthetic_source] == -1
