from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
from joblib import dump
import numpy as np
import pytest

from scripts.calibration.refractive_geometry import FlatPortCamera, triangulate_rays
from scripts.reconstruction_3d.calibration import (
    Calibration,
    CalibrationCapabilityError,
)


class _OrthographicImageToWorld:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return np.column_stack(
            (values[:, 0], values[:, 1], np.mean(values[:, [3, 5]], axis=1))
        )


class _OrthographicWorldToImage:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        x, y, z = values.T
        return np.column_stack((x, y, x, z, y, z))


class _XZFromXYYZ:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((values[:, 0], values[:, 3]))


class _XYFromXZYZ:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((values[:, 0], values[:, 2]))


class _YZFromXZXY:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((values[:, 3], values[:, 1]))


def _rotation_vector(rows: list[list[float]]) -> list[float]:
    return cv2.Rodrigues(np.asarray(rows, dtype=float))[0].reshape(3).tolist()


def _synthetic_cameras() -> tuple[FlatPortCamera, FlatPortCamera, FlatPortCamera]:
    common = dict(
        fx=1000.0,
        fy=1000.0,
        cx=600.0,
        cy=600.0,
        k1=0.0,
        k2=0.0,
        glass_thickness_mm=4.0,
        index_air=1.0003,
        index_glass=1.52,
        index_water=1.3325,
    )
    top = FlatPortCamera(
        name="cam0",
        rotation_vector=_rotation_vector(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
        camera_center=[100.0, 100.0, 350.0],
        interface_normal=[0.0, 0.0, -1.0],
        interface_point=[0.0, 0.0, 200.0],
        layer_count=1,
        **common,
    )
    side1 = FlatPortCamera(
        name="cam1",
        rotation_vector=_rotation_vector(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        ),
        camera_center=[100.0, -350.0, 100.0],
        interface_normal=[0.0, 1.0, 0.0],
        interface_point=[0.0, -4.0, 0.0],
        layer_count=2,
        **common,
    )
    side2 = FlatPortCamera(
        name="cam2",
        rotation_vector=_rotation_vector(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        camera_center=[-350.0, 100.0, 100.0],
        interface_normal=[1.0, 0.0, 0.0],
        interface_point=[-4.0, 0.0, 0.0],
        layer_count=2,
        **common,
    )
    return top, side1, side2


def _write_refractive_bundle(
    root: Path,
    cameras: tuple[FlatPortCamera, FlatPortCamera, FlatPortCamera] | None = None,
) -> None:
    for camera in cameras or _synthetic_cameras():
        camera.to_json(root / f"{camera.name}_bundle_refractive.json")


def _write_polynomial_bundle(root: Path, *, projection: bool = True) -> None:
    dump(_OrthographicImageToWorld(), root / "imCoords_to_XYZ.joblib")
    dump(_XZFromXYYZ(), root / "xy_yz_to_xz.joblib")
    dump(_XYFromXZYZ(), root / "xz_yz_to_xy.joblib")
    dump(_YZFromXZXY(), root / "xz_xy_to_yz.joblib")
    if projection:
        dump(_OrthographicWorldToImage(), root / "XYZ_to_imCoords.joblib")


def test_json_loading_and_auto_detection_prefers_refractive(tmp_path: Path) -> None:
    _write_polynomial_bundle(tmp_path)
    _write_refractive_bundle(tmp_path)

    automatic = Calibration(tmp_path, backend="auto")
    polynomial = Calibration(tmp_path, backend="polynomial")

    assert automatic.backend_name == "refractive"
    assert automatic.score_unit == "mm^2"
    assert automatic.provenance["requested_backend"] == "auto"
    assert list(automatic.provenance["model_files"]) == ["cam0", "cam1", "cam2"]
    assert automatic.capabilities["direct_two_view_triangulation"]
    assert polynomial.backend_name == "polynomial"
    assert polynomial.score_unit == "px^2"
    assert not polynomial.capabilities["direct_two_view_triangulation"]


def test_incomplete_refractive_bundle_falls_back_only_in_auto(tmp_path: Path) -> None:
    _write_polynomial_bundle(tmp_path)
    camera = _synthetic_cameras()[0]
    camera.to_json(tmp_path / "cam0_bundle_refractive.json")

    automatic = Calibration(tmp_path)
    assert automatic.backend_name == "polynomial"
    assert "incomplete refractive bundle" in automatic.provenance[
        "auto_detection_notes"
    ][0]
    with pytest.raises(FileNotFoundError, match="requires all three"):
        Calibration(tmp_path, backend="refractive")


def test_invalid_camera_order_is_rejected(tmp_path: Path) -> None:
    cameras = list(_synthetic_cameras())
    cameras[0] = replace(cameras[0], name="wrong")
    # Preserve the expected filename while corrupting its declared camera name.
    cameras[0].to_json(tmp_path / "cam0_bundle_refractive.json")
    cameras[1].to_json(tmp_path / "cam1_bundle_refractive.json")
    cameras[2].to_json(tmp_path / "cam2_bundle_refractive.json")
    with pytest.raises(ValueError, match="expected 'cam0'"):
        Calibration(tmp_path, backend="refractive")


def test_exact_noisy_and_all_two_view_ray_intersections(tmp_path: Path) -> None:
    _write_refractive_bundle(tmp_path)
    calibration = Calibration(tmp_path)
    world = np.array(
        [[25.0, 30.0, 40.0], [100.0, 120.0, 80.0], [175.0, 165.0, 150.0]]
    )
    pixels = calibration.project_world(world)
    reconstructed, scores = calibration.evaluate_hypotheses(pixels)

    assert np.allclose(reconstructed, world, atol=5e-6)
    assert np.all(scores < 1e-10)
    rng = np.random.default_rng(7)
    noisy = pixels + rng.normal(0.0, 0.2, size=pixels.shape)
    noisy_xyz, noisy_scores = calibration.evaluate_hypotheses(noisy)
    assert np.all(np.isfinite(noisy_xyz))
    assert np.all(noisy_scores > 0.0)

    for missing_camera in range(3):
        partial = pixels.copy()
        partial[:, missing_camera] = np.nan
        partial_xyz, partial_scores = calibration.evaluate_hypotheses(partial)
        assert np.allclose(partial_xyz, world, atol=6e-6)
        assert np.all(partial_scores < 1e-10)


def test_scalar_batch_agreement_projection_roundtrip_and_profiles(tmp_path: Path) -> None:
    _write_refractive_bundle(tmp_path)
    calibration = Calibration(tmp_path)
    world = np.array([[35.0, 45.0, 55.0], [155.0, 145.0, 135.0]])
    pixels = calibration.project_world(world)
    batch_xyz, batch_scores = calibration.evaluate_hypotheses(pixels)

    for index in range(len(world)):
        scalar_xyz, scalar_score = calibration.evaluate_hypotheses(pixels[index])
        assert scalar_xyz.shape == (3,)
        assert np.asarray(scalar_score).shape == ()
        assert np.allclose(scalar_xyz, batch_xyz[index])
        assert np.isclose(scalar_score, batch_scores[index])
    assert np.allclose(batch_xyz, world, atol=5e-6)
    assert np.allclose(
        calibration.score_hypotheses(pixels, profile="production_fast"),
        calibration.score_hypotheses(pixels, profile="legacy_combined"),
    )
    definition = calibration.score_profiles["production_fast"]["definition"]
    assert "mean_squared_perpendicular_distance" in definition


def test_insufficient_nan_and_parallel_rays_fail_closed(tmp_path: Path) -> None:
    cameras = _synthetic_cameras()
    parallel = tuple(
        replace(cameras[0], name=f"cam{index}") for index in range(3)
    )
    _write_refractive_bundle(tmp_path, parallel)
    calibration = Calibration(tmp_path)
    observations = np.full((3, 3, 2), np.nan)
    observations[0, 0] = [600.0, 600.0]  # one view only
    observations[1, :2] = [600.0, 600.0]  # exactly parallel rays
    observations[2] = [600.0, 600.0]  # three coincident parallel rays

    xyz, scores = calibration.evaluate_hypotheses(observations)
    assert np.all(np.isnan(xyz))
    assert np.all(np.isinf(scores))


def test_near_parallel_ray_system_is_rejected_by_condition_guard() -> None:
    angle = 1.0e-6
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    directions = np.array(
        [[1.0, 0.0, 0.0], [np.cos(angle), np.sin(angle), 0.0]]
    )
    with pytest.raises(ValueError, match="rank deficient|ill-conditioned"):
        triangulate_rays(origins, directions)


def test_polynomial_batch_compatibility_two_view_and_score_profiles(tmp_path: Path) -> None:
    _write_polynomial_bundle(tmp_path)
    calibration = Calibration(tmp_path, backend="polynomial")
    observations = np.array(
        [
            [[10.0, 20.0], [11.0, 30.0], [21.0, 31.0]],
            [[10.0, 20.0], [np.nan, np.nan], [20.0, 30.0]],
        ]
    )
    xyz, production_scores = calibration.evaluate_hypotheses(
        observations, profile="production_fast"
    )
    legacy_scores = calibration.score_hypotheses(
        observations, profile="legacy_combined"
    )

    expected_features = np.array(
        [[10.0, 20.0, 11.0, 30.0, 21.0, 31.0], [10.0, 20.0, 10.0, 30.0, 20.0, 30.0]]
    )
    expected_xyz = _OrthographicImageToWorld().predict(expected_features)
    expected_projection = _OrthographicWorldToImage().predict(expected_xyz)
    expected_fast = np.sum((expected_features - expected_projection) ** 2, axis=1)
    assert np.allclose(xyz, expected_xyz)
    assert np.allclose(production_scores, expected_fast)
    assert np.all(legacy_scores >= production_scores)
    assert legacy_scores[0] > production_scores[0]
    assert calibration.compute_triplet_registration_error(observations[0]) == pytest.approx(
        legacy_scores[0]
    )
    assert calibration.provenance["score_profiles"]["production_fast"][
        "definition"
    ] == "sum_squared_reprojection_error_over_3_cameras"


def test_polynomial_projection_capability_error_is_explicit(tmp_path: Path) -> None:
    _write_polynomial_bundle(tmp_path, projection=False)
    calibration = Calibration(tmp_path, backend="polynomial")
    assert not calibration.capabilities["projection"]
    assert calibration.capabilities["hypothesis_scoring"]
    with pytest.raises(CalibrationCapabilityError, match="XYZ_to_imCoords.joblib"):
        calibration.project_world(np.zeros((1, 3)))
