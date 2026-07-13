from __future__ import annotations

import numpy as np

from scripts.calibration.dlt_models import DLTProjectionModel, DLTTriangulationModel
from scripts.calibration.refractive_geometry import FlatPortCamera


def _camera(name: str, center: tuple[float, float, float], rotation_vector) -> FlatPortCamera:
    return FlatPortCamera(
        name=name,
        fx=1000.0,
        fy=1000.0,
        cx=600.0,
        cy=600.0,
        k1=0.0,
        k2=0.0,
        rotation_vector=list(rotation_vector),
        camera_center=list(center),
        interface_normal=[0.0, 0.0, 1.0],
        interface_point=[0.0, 0.0, 0.0],
        glass_thickness_mm=0.0,
        layer_count=0,
    )


def test_dlt_projection_and_triangulation_round_trip() -> None:
    cameras = [
        _camera("cam0", (0.0, 0.0, -500.0), (0.0, 0.0, 0.0)),
        _camera("cam1", (-500.0, 0.0, 0.0), (0.0, np.pi / 2.0, 0.0)),
        _camera("cam2", (0.0, -500.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ]
    points = np.array([[10.0, 20.0, 30.0], [-25.0, 5.0, 60.0]])
    image = DLTProjectionModel(cameras).predict(points)
    reconstructed = DLTTriangulationModel(cameras).predict(image)
    np.testing.assert_allclose(reconstructed, points, atol=1e-8)


def test_dlt_models_preserve_nan_rows() -> None:
    cameras = [
        _camera("cam0", (0.0, 0.0, -500.0), (0.0, 0.0, 0.0)),
        _camera("cam1", (-500.0, 0.0, 0.0), (0.0, np.pi / 2.0, 0.0)),
        _camera("cam2", (0.0, -500.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ]
    projected = DLTProjectionModel(cameras).predict([[np.nan, 0.0, 0.0]])
    triangulated = DLTTriangulationModel(cameras).predict([[np.nan] * 6])
    assert np.isnan(projected).all()
    assert np.isnan(triangulated).all()
