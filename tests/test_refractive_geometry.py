from __future__ import annotations

import numpy as np

from scripts.calibration.refractive_geometry import (
    FlatPortCamera,
    decompose_projection,
    fit_projection_dlt,
    refract_vectors,
    triangulate_rays,
)


def test_parallel_flat_port_preserves_normal_incidence() -> None:
    camera = FlatPortCamera(
        name="test",
        fx=1000.0,
        fy=1000.0,
        cx=600.0,
        cy=600.0,
        k1=0.0,
        k2=0.0,
        rotation_vector=[0.0, 0.0, 0.0],
        camera_center=[0.0, 0.0, 0.0],
        interface_normal=[0.0, 0.0, 1.0],
        interface_point=[0.0, 0.0, 10.0],
        glass_thickness_mm=5.0,
        layer_count=2,
    )
    origin, direction = camera.backproject(np.array([[600.0, 600.0]]))
    assert np.allclose(origin[0], [0.0, 0.0, 15.0], atol=1e-10)
    assert np.allclose(direction[0], [0.0, 0.0, 1.0], atol=1e-10)
    residual = camera.point_to_ray_residuals(
        np.array([[600.0, 600.0]]), np.array([[0.0, 0.0, 100.0]])
    )
    assert np.linalg.norm(residual) < 1e-10


def test_vector_snell_matches_scalar_angle_law() -> None:
    incident_angle = np.deg2rad(35.0)
    incident = np.array([[np.sin(incident_angle), 0.0, np.cos(incident_angle)]])
    transmitted = refract_vectors(incident, np.array([0.0, 0.0, 1.0]), 1.0, 1.333)[0]
    transmitted_angle = np.arctan2(transmitted[0], transmitted[2])
    expected = np.arcsin(np.sin(incident_angle) / 1.333)
    assert np.isclose(transmitted_angle, expected, atol=1e-12)


def test_triangulate_rays_recovers_exact_intersection() -> None:
    point = np.array([25.0, 40.0, 70.0])
    origins = np.array([[-100.0, 0.0, 0.0], [200.0, 30.0, 10.0], [20.0, 220.0, 80.0]])
    directions = point - origins
    reconstructed, ray_rms = triangulate_rays(origins, directions)
    assert np.allclose(reconstructed, point, atol=1e-10)
    assert ray_rms < 1e-10


def test_normalized_dlt_recovers_pinhole_rays() -> None:
    intrinsic = np.array([[1300.0, 0.0, 610.0], [0.0, 1250.0, 590.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    center = np.array([100.0, 100.0, -400.0])
    projection = intrinsic @ np.column_stack((rotation, -rotation @ center))
    rng = np.random.default_rng(4)
    world = rng.uniform([5.0, 5.0, 5.0], [195.0, 195.0, 195.0], size=(40, 3))
    homogeneous = np.column_stack((world, np.ones(len(world))))
    image_h = (projection @ homogeneous.T).T
    image = image_h[:, :2] / image_h[:, 2:]

    fitted = fit_projection_dlt(world, image)
    fitted_intrinsic, fitted_rotation, fitted_center = decompose_projection(fitted)
    assert np.allclose(fitted_intrinsic, intrinsic, atol=1e-7)
    assert np.allclose(fitted_rotation, rotation, atol=1e-9)
    assert np.allclose(fitted_center, center, atol=1e-8)
