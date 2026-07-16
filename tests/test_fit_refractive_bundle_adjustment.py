from __future__ import annotations

import numpy as np

from scripts.calibration.fit_refractive_bundle_adjustment import (
    PoseState,
    summarize_bundle_ray_residuals,
)
from scripts.calibration.refractive_geometry import FlatPortCamera


def _pinhole_camera(name: str) -> FlatPortCamera:
    return FlatPortCamera(
        name=name,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        k1=0.0,
        k2=0.0,
        rotation_vector=[0.0, 0.0, 0.0],
        camera_center=[0.0, 0.0, 0.0],
        interface_normal=[0.0, 0.0, 1.0],
        interface_point=[0.0, 0.0, 1.0],
        glass_thickness_mm=4.0,
        layer_count=0,
    )


def test_summarize_bundle_ray_residuals_reports_exact_rays() -> None:
    cameras = [_pinhole_camera(f"cam{index}") for index in range(3)]
    object_points = np.array(
        [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 2.0, 10.0]]
    )
    pixels = np.column_stack(
        (
            50.0 + 100.0 * object_points[:, 0] / object_points[:, 2],
            50.0 + 100.0 * object_points[:, 1] / object_points[:, 2],
        )
    )
    observation_data = (
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
        object_points,
        pixels,
        np.ones(3),
        ["ball:test"],
    )
    poses = {
        "ball:test": PoseState(
            rotation_vector=np.zeros(3), translation=np.zeros(3)
        )
    }

    summary = summarize_bundle_ray_residuals(cameras, poses, observation_data)

    assert summary["all"]["n"] == 3
    assert summary["all"]["maximum_mm"] < 1e-12
    assert set(summary["by_camera"]) == {"top_cam", "side1_cam", "side2_cam"}
