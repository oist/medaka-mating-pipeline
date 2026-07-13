from __future__ import annotations

"""Scikit-learn-style prediction wrappers for a three-camera DLT calibration.

The reconstruction pipeline stores calibration estimators with joblib and only
requires a ``predict`` method.  These wrappers make the projective DLT cameras
from the full-set calibration usable by the existing fast, batched
``--num-candidates 6`` path without changing that pipeline's data contract.
"""

from collections.abc import Sequence

import numpy as np

from scripts.calibration.refractive_geometry import FlatPortCamera


def _as_cameras(cameras: Sequence[FlatPortCamera]) -> tuple[FlatPortCamera, ...]:
    cameras = tuple(cameras)
    if len(cameras) != 3:
        raise ValueError(f"expected exactly three cameras, got {len(cameras)}")
    if any(camera.layer_count != 0 for camera in cameras):
        raise ValueError("DLT joblib wrappers require pinhole cameras (layer_count=0)")
    return cameras


class DLTTriangulationModel:
    """Batch three-view pixels to 3D using least-squares ray intersection."""

    def __init__(self, cameras: Sequence[FlatPortCamera]) -> None:
        self.cameras = _as_cameras(cameras)

    def predict(self, image_coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(image_coordinates, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"expected image coordinates with shape (N, 6), got {values.shape}")

        output = np.full((len(values), 3), np.nan, dtype=float)
        finite = np.all(np.isfinite(values), axis=1)
        if not finite.any():
            return output

        selected = values[finite].reshape(-1, 3, 2)
        origins = []
        directions = []
        for camera_index, camera in enumerate(self.cameras):
            origin, direction = camera.backproject(selected[:, camera_index, :])
            origins.append(origin)
            directions.append(direction)
        origins_array = np.stack(origins, axis=1)
        directions_array = np.stack(directions, axis=1)

        identity = np.eye(3, dtype=float)
        projectors = (
            identity[None, None, :, :]
            - directions_array[:, :, :, None] * directions_array[:, :, None, :]
        )
        system = np.sum(projectors, axis=1)
        rhs = np.sum(
            projectors @ origins_array[:, :, :, None], axis=1
        ).squeeze(-1)
        try:
            # NumPy 2 treats a 2D right-hand side as a matrix rather than a
            # stack of vectors, so keep an explicit final singleton axis.
            xyz = np.linalg.solve(system, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            xyz = np.einsum("nij,nj->ni", np.linalg.pinv(system), rhs)
        output[finite] = xyz
        return output


class DLTProjectionModel:
    """Batch 3D world points to flattened cam0/cam1/cam2 pixels."""

    def __init__(self, cameras: Sequence[FlatPortCamera]) -> None:
        self.cameras = _as_cameras(cameras)

    def predict(self, world_coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(world_coordinates, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"expected world coordinates with shape (N, 3), got {values.shape}")

        output = np.full((len(values), 6), np.nan, dtype=float)
        finite = np.all(np.isfinite(values), axis=1)
        if not finite.any():
            return output

        points = values[finite]
        projected = []
        for camera in self.cameras:
            center = np.asarray(camera.camera_center, dtype=float)
            camera_xyz = (points - center) @ camera.rotation_matrix.T
            normalized = camera_xyz[:, :2] / camera_xyz[:, 2:3]
            pixels = np.column_stack(
                (
                    camera.fx * normalized[:, 0] + camera.cx,
                    camera.fy * normalized[:, 1] + camera.cy,
                )
            )
            projected.append(pixels)
        output[finite] = np.stack(projected, axis=1).reshape(-1, 6)
        return output
