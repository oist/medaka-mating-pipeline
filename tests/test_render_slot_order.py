"""Regression tests for best.json slot parity in the 2D/3D video renderer."""

import numpy as np

from scripts.visualization.render_3d_skeleton_video import (
    _instances_in_reconstruction_slot_order,
)


class FakeInst:
    def __init__(self, mouth, tail, score):
        self._pts = np.asarray([mouth, tail], dtype=float)
        self.score = score

    def numpy(self):
        return self._pts


def _ordered(insts, **kwargs):
    return _instances_in_reconstruction_slot_order(
        insts,
        node_names=["mouth", "tail"],
        **kwargs,
    )


def test_renderer_drops_same_collapsed_candidate_as_reconstruction():
    left = FakeInst((10, 10), (30, 10), 0.80)
    collapsed = FakeInst((31, 10), (32, 10), 0.99)
    right = FakeInst((50, 10), (70, 10), 0.70)

    # The high-score collapsed detection must not become slot 0. best.json was
    # built after this candidate was removed by reconstruction ingestion.
    assert _ordered([left, collapsed, right]) == [left, right]


def test_renderer_uses_score_then_centroid_x_tie_break():
    right = FakeInst((80, 10), (100, 10), 0.75)
    left = FakeInst((10, 10), (30, 10), 0.75)
    lower_score = FakeInst((40, 10), (60, 10), 0.70)

    assert _ordered([right, lower_score, left]) == [left, right, lower_score]


def test_renderer_skips_instance_without_finite_tracking_point():
    invisible = FakeInst((np.nan, np.nan), (np.nan, np.nan), 0.99)
    visible = FakeInst((10, 10), (30, 10), 0.50)

    assert _ordered([invisible, visible]) == [visible]
