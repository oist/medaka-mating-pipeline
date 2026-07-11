"""Flat-interface refractive camera geometry.

The camera is in air. A top camera may use one planar air-to-water interface;
side cameras use two parallel interfaces (air-to-glass and glass-to-water).
Pixels are back-projected through the layers with vector Snell refraction and
3D points are reconstructed as the least-squares intersection of water rays.

This module intentionally keeps the optical model small. It supports radial
lens distortion (k1, k2), but assumes parallel faces for a glass pane. The
production calibration protocol should measure glass thickness and dry-camera
intrinsics rather than asking the underwater fit to identify them jointly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Sequence

import cv2
import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


EPSILON = 1e-12


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < EPSILON):
        raise ValueError("cannot normalize a zero-length vector")
    return vectors / norms


def refract_vectors(
    incident: np.ndarray,
    normal: np.ndarray,
    index_from: float,
    index_to: float,
) -> np.ndarray:
    """Refract directions through an interface using vector Snell's law.

    ``normal`` points from the incident medium into the transmitted medium and
    incident directions must point toward the interface, so ``incident·normal``
    is positive. Total internal reflection is reported as ``ValueError``.
    """
    incident = normalize_vectors(incident)
    normal = normalize_vectors(np.asarray(normal, dtype=float).reshape(1, 3))[0]
    cosine_incident = incident @ normal
    if np.any(cosine_incident <= 0):
        raise ValueError("incident ray points away from the refractive interface")
    eta = float(index_from) / float(index_to)
    tangential = incident - cosine_incident[:, None] * normal
    transmitted_tangential = eta * tangential
    normal_squared = 1.0 - np.sum(transmitted_tangential**2, axis=1)
    if np.any(normal_squared < -1e-10):
        raise ValueError("total internal reflection")
    transmitted_normal = np.sqrt(np.maximum(normal_squared, 0.0))
    transmitted = transmitted_tangential + transmitted_normal[:, None] * normal
    return normalize_vectors(transmitted)


def _rotation_matrix_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rotation_vector, dtype=float).reshape(3))[0]


def _rotation_vector_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rotation_matrix, dtype=float))[0].reshape(3)


def _normal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize_vectors(np.asarray(normal, dtype=float).reshape(1, 3))[0]
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = normalize_vectors(np.cross(normal, reference).reshape(1, 3))[0]
    second = normalize_vectors(np.cross(normal, first).reshape(1, 3))[0]
    return first, second


def _undistort_normalized(
    image_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    *,
    iterations: int = 6,
) -> np.ndarray:
    distorted = np.column_stack(
        ((image_points[:, 0] - cx) / fx, (image_points[:, 1] - cy) / fy)
    )
    undistorted = distorted.copy()
    for _ in range(iterations):
        radius_squared = np.sum(undistorted**2, axis=1)
        scale = 1.0 + k1 * radius_squared + k2 * radius_squared**2
        undistorted = distorted / scale[:, None]
    return undistorted


@dataclass
class FlatPortCamera:
    """Pinhole camera behind zero, one, or two parallel planar interfaces."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    rotation_vector: list[float]
    camera_center: list[float]
    interface_normal: list[float]
    interface_point: list[float]
    glass_thickness_mm: float
    index_air: float = 1.0003
    index_glass: float = 1.52
    # Fresh water near 25 C in the visible spectrum.  Wavelength dependence is
    # smaller than the current mechanical/interface uncertainty.
    index_water: float = 1.3325
    layer_count: int = 2
    # Optional OpenCV Brown/rational terms.  Kept at the end so older JSON
    # camera files that contain only k1/k2 remain valid.
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    @property
    def rotation_matrix(self) -> np.ndarray:
        """World-to-camera rotation matrix."""
        return _rotation_matrix_from_vector(np.asarray(self.rotation_vector))

    def backproject(self, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return water-side ray origins and unit directions for pixels."""
        image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
        intrinsic = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        distortion = np.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6],
            dtype=float,
        )
        normalized = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2), intrinsic, distortion
        ).reshape(-1, 2)
        camera_directions = normalize_vectors(
            np.column_stack((normalized, np.ones(len(normalized))))
        )
        world_directions = normalize_vectors(camera_directions @ self.rotation_matrix)
        center = np.asarray(self.camera_center, dtype=float)
        origins = np.repeat(center[None, :], len(image_points), axis=0)
        if self.layer_count == 0:
            return origins, world_directions

        normal = normalize_vectors(np.asarray(self.interface_normal).reshape(1, 3))[0]
        interface_point = np.asarray(self.interface_point, dtype=float)
        denominator = world_directions @ normal
        numerator = float((interface_point - center) @ normal)
        distance = numerator / denominator
        if np.any(distance <= 0) or np.any(denominator <= 0):
            raise ValueError(f"{self.name}: rays do not hit the forward interface")
        origins = center + distance[:, None] * world_directions

        if self.layer_count == 1:
            water_directions = refract_vectors(
                world_directions, normal, self.index_air, self.index_water
            )
            return origins, water_directions

        glass_directions = refract_vectors(
            world_directions, normal, self.index_air, self.index_glass
        )
        distance_in_glass = self.glass_thickness_mm / (glass_directions @ normal)
        origins = origins + distance_in_glass[:, None] * glass_directions
        water_directions = refract_vectors(
            glass_directions, normal, self.index_glass, self.index_water
        )
        return origins, water_directions

    def point_to_ray_residuals(
        self, image_points: np.ndarray, world_points: np.ndarray
    ) -> np.ndarray:
        origins, directions = self.backproject(image_points)
        offsets = np.asarray(world_points, dtype=float) - origins
        along_ray = np.sum(offsets * directions, axis=1)
        closest = origins + along_ray[:, None] * directions
        return np.asarray(world_points, dtype=float) - closest

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "FlatPortCamera":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def triangulate_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares closest point to two or more 3D rays."""
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    directions = normalize_vectors(np.asarray(directions, dtype=float).reshape(-1, 3))
    if len(origins) < 2:
        raise ValueError("at least two rays are required")
    identity = np.eye(3)
    projectors = identity[None, :, :] - directions[:, :, None] * directions[:, None, :]
    system = np.sum(projectors, axis=0)
    right_hand_side = np.sum(projectors @ origins[:, :, None], axis=0).reshape(3)
    point = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
    distances = np.linalg.norm((projectors @ (point - origins)[:, :, None]).squeeze(-1), axis=1)
    return point, float(np.sqrt(np.mean(distances**2)))


def triangulate_pixels(
    cameras: Sequence[FlatPortCamera], image_points: np.ndarray
) -> tuple[np.ndarray, float]:
    image_points = np.asarray(image_points, dtype=float).reshape(len(cameras), 2)
    origins = []
    directions = []
    for camera, point in zip(cameras, image_points, strict=True):
        origin, direction = camera.backproject(point.reshape(1, 2))
        origins.append(origin[0])
        directions.append(direction[0])
    return triangulate_rays(np.asarray(origins), np.asarray(directions))


def fit_projection_dlt(world_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """Fit a normalized 3x4 projective camera matrix."""
    world_points = np.asarray(world_points, dtype=float).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(world_points) < 6 or len(world_points) != len(image_points):
        raise ValueError("DLT requires at least six matched 3D/2D points")

    world_center = world_points.mean(axis=0)
    world_distance = np.mean(np.linalg.norm(world_points - world_center, axis=1))
    world_scale = np.sqrt(3.0) / world_distance
    world_transform = np.eye(4)
    world_transform[:3, :3] *= world_scale
    world_transform[:3, 3] = -world_scale * world_center

    image_center = image_points.mean(axis=0)
    image_distance = np.mean(np.linalg.norm(image_points - image_center, axis=1))
    image_scale = np.sqrt(2.0) / image_distance
    image_transform = np.array(
        [
            [image_scale, 0.0, -image_scale * image_center[0]],
            [0.0, image_scale, -image_scale * image_center[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    world_h = np.column_stack((world_points, np.ones(len(world_points))))
    normalized_world = (world_transform @ world_h.T).T
    image_h = np.column_stack((image_points, np.ones(len(image_points))))
    normalized_image = (image_transform @ image_h.T).T
    rows = []
    for point, pixel in zip(normalized_world, normalized_image, strict=True):
        u, v = pixel[:2]
        rows.append(np.hstack((point, np.zeros(4), -u * point)))
        rows.append(np.hstack((np.zeros(4), point, -v * point)))
    design = np.asarray(rows)
    _, _, right = np.linalg.svd(design)
    normalized_projection = right[-1].reshape(3, 4)
    projection = np.linalg.inv(image_transform) @ normalized_projection @ world_transform
    return projection / np.linalg.norm(projection[2, :3])


def decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return K, world-to-camera R, and camera center from a DLT matrix."""
    projection = np.asarray(projection, dtype=float).reshape(3, 4)
    intrinsic, rotation = rq(projection[:, :3])
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0] = 1.0
    correction = np.diag(signs)
    intrinsic = intrinsic @ correction
    rotation = correction @ rotation
    if np.linalg.det(rotation) < 0:
        intrinsic = -intrinsic
        rotation = -rotation
    intrinsic = intrinsic / intrinsic[2, 2]
    center = -np.linalg.solve(projection[:, :3], projection[:, 3])
    return intrinsic, rotation, center


def _ray_box_entry(
    origin: np.ndarray, direction: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    inverse = 1.0 / np.where(np.abs(direction) < EPSILON, np.sign(direction) * EPSILON + EPSILON, direction)
    first = (bounds[:, 0] - origin) * inverse
    second = (bounds[:, 1] - origin) * inverse
    near = float(np.max(np.minimum(first, second)))
    far = float(np.min(np.maximum(first, second)))
    if far <= max(near, 0.0):
        raise ValueError("camera-to-tank direction does not intersect tank bounds")
    distance = near if near > 0 else far
    return origin + distance * direction


@dataclass
class FitDiagnostics:
    initial_rms_mm: float
    final_rms_mm: float
    cost: float
    optimality: float
    nfev: int
    success: bool
    message: str


def initialize_flat_port_camera(
    name: str,
    world_points: np.ndarray,
    image_points: np.ndarray,
    *,
    tank_bounds: np.ndarray,
    layer_count: int,
    glass_thickness_mm: float,
) -> FlatPortCamera:
    projection = fit_projection_dlt(world_points, image_points)
    intrinsic, rotation, center = decompose_projection(projection)
    tank_center = np.asarray(tank_bounds, dtype=float).mean(axis=1)
    normal = normalize_vectors((tank_center - center).reshape(1, 3))[0]
    interface_point = _ray_box_entry(center, normal, np.asarray(tank_bounds, dtype=float))
    return FlatPortCamera(
        name=name,
        fx=float(intrinsic[0, 0]),
        fy=float(intrinsic[1, 1]),
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        k1=0.0,
        k2=0.0,
        rotation_vector=_rotation_vector_from_matrix(rotation).tolist(),
        camera_center=center.tolist(),
        interface_normal=normal.tolist(),
        interface_point=interface_point.tolist(),
        glass_thickness_mm=float(glass_thickness_mm),
        layer_count=int(layer_count),
    )


def fit_flat_port_camera(
    initial: FlatPortCamera,
    world_points: np.ndarray,
    image_points: np.ndarray,
    *,
    fit_distortion: bool = True,
    max_nfev: int = 1500,
) -> tuple[FlatPortCamera, FitDiagnostics]:
    """Fit a flat-port camera by minimizing point-to-water-ray distance."""
    world_points = np.asarray(world_points, dtype=float).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    initial_normal = normalize_vectors(np.asarray(initial.interface_normal).reshape(1, 3))[0]
    basis_first, basis_second = _normal_basis(initial_normal)
    initial_interface_point = np.asarray(initial.interface_point, dtype=float)
    initial_center = np.asarray(initial.camera_center, dtype=float)
    initial_rotation = np.asarray(initial.rotation_vector, dtype=float)

    parameter0 = np.array(
        [
            np.log(initial.fx),
            np.log(initial.fy),
            initial.cx,
            initial.cy,
            initial.k1,
            initial.k2,
            *initial_rotation,
            *initial_center,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    def unpack(parameters: np.ndarray) -> FlatPortCamera:
        normal = normalize_vectors(
            (initial_normal + parameters[12] * basis_first + parameters[13] * basis_second).reshape(1, 3)
        )[0]
        interface_point = initial_interface_point + parameters[14] * initial_normal
        return FlatPortCamera(
            name=initial.name,
            fx=float(np.exp(parameters[0])),
            fy=float(np.exp(parameters[1])),
            cx=float(parameters[2]),
            cy=float(parameters[3]),
            k1=float(parameters[4]) if fit_distortion else 0.0,
            k2=float(parameters[5]) if fit_distortion else 0.0,
            rotation_vector=parameters[6:9].tolist(),
            camera_center=parameters[9:12].tolist(),
            interface_normal=normal.tolist(),
            interface_point=interface_point.tolist(),
            glass_thickness_mm=initial.glass_thickness_mm,
            index_air=initial.index_air,
            index_glass=initial.index_glass,
            index_water=initial.index_water,
            layer_count=initial.layer_count,
            p1=initial.p1,
            p2=initial.p2,
            k3=initial.k3,
            k4=initial.k4,
            k5=initial.k5,
            k6=initial.k6,
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        camera = unpack(parameters)
        try:
            geometric = camera.point_to_ray_residuals(image_points, world_points).ravel()
        except ValueError:
            geometric = np.full(world_points.size, 1e4)
        # Weak priors stabilize the otherwise correlated wet fit. They are not
        # substitutes for dry intrinsics and measured interface geometry.
        priors = np.array(
            [
                (parameters[0] - parameter0[0]) / 0.35,
                (parameters[1] - parameter0[1]) / 0.35,
                (parameters[2] - parameter0[2]) / 180.0,
                (parameters[3] - parameter0[3]) / 180.0,
                parameters[4] / 0.35,
                parameters[5] / 0.35,
                *(parameters[9:12] - parameter0[9:12]) / 250.0,
                parameters[12] / 0.35,
                parameters[13] / 0.35,
                parameters[14] / 40.0,
            ]
        )
        if not fit_distortion:
            priors = np.delete(priors, [4, 5])
        return np.concatenate((geometric, priors))

    lower = np.array(
        [
            np.log(150.0), np.log(150.0), -300.0, -300.0, -1.0, -1.0,
            -10.0, -10.0, -10.0,
            *(initial_center - 800.0),
            -0.8, -0.8, -100.0,
        ]
    )
    upper = np.array(
        [
            np.log(5000.0), np.log(5000.0), 1500.0, 1500.0, 1.0, 1.0,
            10.0, 10.0, 10.0,
            *(initial_center + 800.0),
            0.8, 0.8, 100.0,
        ]
    )
    initial_geometric = initial.point_to_ray_residuals(image_points, world_points)
    result = least_squares(
        residuals,
        parameter0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )
    fitted = unpack(result.x)
    final_geometric = fitted.point_to_ray_residuals(image_points, world_points)
    diagnostics = FitDiagnostics(
        initial_rms_mm=float(np.sqrt(np.mean(np.sum(initial_geometric**2, axis=1)))),
        final_rms_mm=float(np.sqrt(np.mean(np.sum(final_geometric**2, axis=1)))),
        cost=float(result.cost),
        optimality=float(result.optimality),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
    return fitted, diagnostics


def fit_flat_port_camera_fixed_intrinsics(
    initial: FlatPortCamera,
    world_points: np.ndarray,
    image_points: np.ndarray,
    *,
    fit_interface_correction: bool = True,
    normal_prior_sigma: float = 0.025,
    offset_prior_sigma_mm: float = 2.0,
    center_prior_sigma_mm: float = 250.0,
    max_nfev: int = 1500,
) -> tuple[FlatPortCamera, FitDiagnostics]:
    """Fit extrinsics and small interface corrections with dry intrinsics fixed.

    The interface normal/point in ``initial`` should come from the tank survey.
    Two tangent-plane normal corrections and one signed plane-offset correction
    are regularized around that measurement.  Set
    ``fit_interface_correction=False`` for the first staged solve.
    """
    world_points = np.asarray(world_points, dtype=float).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    initial_normal = normalize_vectors(np.asarray(initial.interface_normal).reshape(1, 3))[0]
    basis_first, basis_second = _normal_basis(initial_normal)
    initial_interface_point = np.asarray(initial.interface_point, dtype=float)
    initial_center = np.asarray(initial.camera_center, dtype=float)
    parameter0 = np.array(
        [*np.asarray(initial.rotation_vector, dtype=float), *initial_center, 0.0, 0.0, 0.0]
    )

    def unpack(parameters: np.ndarray) -> FlatPortCamera:
        if fit_interface_correction:
            normal = normalize_vectors(
                (
                    initial_normal
                    + parameters[6] * basis_first
                    + parameters[7] * basis_second
                ).reshape(1, 3)
            )[0]
            interface_point = initial_interface_point + parameters[8] * initial_normal
        else:
            normal = initial_normal
            interface_point = initial_interface_point
        return FlatPortCamera(
            name=initial.name,
            fx=initial.fx,
            fy=initial.fy,
            cx=initial.cx,
            cy=initial.cy,
            k1=initial.k1,
            k2=initial.k2,
            rotation_vector=parameters[:3].tolist(),
            camera_center=parameters[3:6].tolist(),
            interface_normal=normal.tolist(),
            interface_point=interface_point.tolist(),
            glass_thickness_mm=initial.glass_thickness_mm,
            index_air=initial.index_air,
            index_glass=initial.index_glass,
            index_water=initial.index_water,
            layer_count=initial.layer_count,
            p1=initial.p1,
            p2=initial.p2,
            k3=initial.k3,
            k4=initial.k4,
            k5=initial.k5,
            k6=initial.k6,
        )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        camera = unpack(parameters)
        try:
            geometric = camera.point_to_ray_residuals(image_points, world_points).ravel()
        except ValueError:
            geometric = np.full(world_points.size, 1e4)
        priors = list((parameters[3:6] - initial_center) / center_prior_sigma_mm)
        if fit_interface_correction:
            priors.extend(
                [
                    parameters[6] / normal_prior_sigma,
                    parameters[7] / normal_prior_sigma,
                    parameters[8] / offset_prior_sigma_mm,
                ]
            )
        return np.concatenate((geometric, np.asarray(priors)))

    lower = np.array(
        [
            -10.0,
            -10.0,
            -10.0,
            *(initial_center - 800.0),
            -0.2,
            -0.2,
            -50.0,
        ]
    )
    upper = np.array(
        [
            10.0,
            10.0,
            10.0,
            *(initial_center + 800.0),
            0.2,
            0.2,
            50.0,
        ]
    )
    initial_geometric = initial.point_to_ray_residuals(image_points, world_points)
    result = least_squares(
        residuals,
        parameter0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )
    fitted = unpack(result.x)
    final_geometric = fitted.point_to_ray_residuals(image_points, world_points)
    diagnostics = FitDiagnostics(
        initial_rms_mm=float(np.sqrt(np.mean(np.sum(initial_geometric**2, axis=1)))),
        final_rms_mm=float(np.sqrt(np.mean(np.sum(final_geometric**2, axis=1)))),
        cost=float(result.cost),
        optimality=float(result.optimality),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
    return fitted, diagnostics
