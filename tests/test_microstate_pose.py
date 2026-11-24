"""Geometry of the 14-coordinate pair pose used before PCA."""

import numpy as np

from scripts.analysis.microstate import align_pair, reconstruct_pose, recentre_pec_mid


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


def test_pec_mid_origin():
    points = _pair(40.0, (100.0, -30.0, 5.0))
    points[0, 4, 2] = points[0, 1, 2] + 6.0   # b's pectoral 6 mm above a's: dz = -6
    feat, valid = align_pair(points, origin="pec_mid")
    assert valid[0]
    # pectoral_b sits at (D_xy / 2, 0, -dz / 2) with dz = z_a - z_b
    d_xy = np.linalg.norm(points[0, 4, :2] - points[0, 1, :2])
    assert np.isclose(feat[0, 12], d_xy / 2)
    assert np.isclose(feat[0, 13], 3.0)
    # mouth_a relative to the pectoral midpoint, before yaw: (10, 0, 1) - (20, 0, 3) -> x' = -10, z' = -2
    assert np.allclose(feat[0, [0, 2]], [-10.0, -2.0])


def test_recentre_pec_mid_matches_direct_alignment():
    rng = np.random.default_rng(0)
    points = rng.normal(scale=20.0, size=(50, 6, 3))
    centroid, valid = align_pair(points)
    direct, valid2 = align_pair(points, origin="pec_mid")
    assert (valid == valid2).all()
    assert np.allclose(recentre_pec_mid(centroid[valid]), direct[valid], atol=1e-9)


def test_reconstruct_pec_mid_is_rigid_copy():
    points = _pair(40.0, (100.0, -30.0, 5.0))
    points[0, 4, 2] += 6.0
    feat, _ = align_pair(points, origin="pec_mid")
    pose = reconstruct_pose(feat, origin="pec_mid")[0]
    assert np.allclose(pose[1] + pose[4], 0, atol=1e-9)   # pectoral midpoint at the origin
    dist = lambda p: np.linalg.norm(p[:, None] - p[None], axis=-1)
    assert np.allclose(dist(pose), dist(points[0]), atol=1e-9)   # only rotated and shifted


def test_short_reference_is_invalid():
    points = np.zeros((1, 6, 3))
    points[0, 3:, 0] = 0.2
    points[0, :, 2] = 1.0
    feat, valid = align_pair(points, min_xy=1.0)
    assert not valid[0]
    assert np.isnan(feat).all()
