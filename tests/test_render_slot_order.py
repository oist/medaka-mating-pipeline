"""Regression tests for best.json slot parity in the 2D/3D video renderer."""

import numpy as np

from scripts.visualization.render_3d_skeleton_video import (
    _instances_in_reconstruction_slot_order,
    build_triangulation,
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


class FakeCal:
    """Two-or-three-view triangulator stand-in: mean of the finite rows, z=0."""

    def compute_XYZ_from_imageCoord_triplet(self, triplet):
        seen = triplet[np.isfinite(triplet).all(axis=1)]
        if len(seen) < 2:
            raise ValueError("need two views")
        return np.array([*seen.mean(axis=0), 0.0])


def test_fish_occluded_in_one_view_keeps_all_nodes():
    # ID_1 is missing from side2 (two-view completion in best.json); its three
    # nodes must still be triangulated from top + side1, not dropped.
    two_nodes = np.array([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])  # 1 instance, 3 nodes
    slp = {"top_cam": {7: two_nodes}, "side1_cam": {7: two_nodes + 1}, "side2_cam": {7: np.empty((0, 3, 2))}}
    best = {7: {"cam0_to_id": ["ID_1"], "cam1_to_id": ["ID_1"], "cam2_to_id": [],
                "groups": {"ID_1": [9.0, 9.0, 9.0]}}}

    out = build_triangulation(slp, best, FakeCal(), ["ID_1"], ("mouth", "pectoral", "tail"), [7])

    assert np.allclose(out[0, 0, :, 0], [0.5, 2.5, 4.5])  # real skeleton, not the centroid fallback


def test_single_view_falls_back_to_reconstructed_point():
    one = np.array([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    slp = {"top_cam": {7: one}, "side1_cam": {7: np.empty((0, 3, 2))}, "side2_cam": {7: np.empty((0, 3, 2))}}
    best = {7: {"cam0_to_id": ["ID_1"], "cam1_to_id": [], "cam2_to_id": [],
                "groups": {"ID_1": [9.0, 8.0, 7.0]}}}

    out = build_triangulation(slp, best, FakeCal(), ["ID_1"], ("mouth", "pectoral", "tail"), [7])

    assert np.allclose(out[0, 0], [[9.0, 8.0, 7.0]] * 3)
