"""Geometry of the 14-coordinate pair pose used before PCA."""

import numpy as np

from scripts.analysis.microstate import align_pair, reconstruct_pose


def _pair(yaw_deg=0.0, shift=(0.0, 0.0, 0.0)):
    """Two straight fish, pectorals 40 mm apart. Optional yaw about z and translation."""
    a = np.array([[10.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-20.0, 0.0, -1.0]])
    b = np.array([[50.0, 0.0, 1.0], [40.0, 0.0, 0.0], [20.0, 0.0, -1.0]])
    points = np.concatenate([a, b])[None]
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = points.copy()
    rot[..., 0] = c * points[..., 0] - s * points[..., 1]
    rot[..., 1] = s * points[..., 0] + c * points[..., 1]
    return rot + np.asarray(shift)


def test_alignment_removes_translation_and_yaw():
    feat, valid = align_pair(_pair(40.0, (100.0, -30.0, 5.0)))
    assert valid[0]
    pose = reconstruct_pose(feat)[0]
    assert np.allclose(pose.mean(axis=0), 0, atol=1e-8)
    reference = pose[4, :2] - pose[1, :2]
    assert reference[0] > 0
    assert abs(reference[1]) < 1e-8
    plain, _ = align_pair(_pair())
    assert np.allclose(feat, plain, atol=1e-8)


def test_short_reference_is_invalid():
    points = np.zeros((1, 6, 3))
    points[0, 3:, 0] = 0.2
    points[0, :, 2] = 1.0
    feat, valid = align_pair(points, min_xy=1.0)
    assert not valid[0]
    assert np.isnan(feat).all()
