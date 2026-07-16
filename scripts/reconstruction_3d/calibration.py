"""Public calibration backends for 3D reconstruction.

The :class:`Calibration` facade resolves either the historical polynomial
joblib calibration or a three-camera flat-port refractive calibration.  Both
backends expose the same batch-first API:

``triangulate(observations[..., 3, 2]) -> xyz[..., 3]``
    Reconstruct from any two or three finite camera rows.

``score_hypotheses(observations, profile=...) -> scores[...]``
    Score in the backend's declared ``score_unit``.

``evaluate_hypotheses(observations, profile=...) -> (xyz, scores)``
    Fused triangulation and scoring for production candidate enumeration.

``project_world(xyz[..., 3]) -> pixels[..., 3, 2]``
    Project world points into cam0/cam1/cam2 order.

Missing cameras are represented by a complete ``NaN`` row.  Fewer than two
finite views, invalid optics, and degenerate ray systems return ``NaN`` XYZ and
an infinite score instead of a false finite reconstruction.

Conventions:
- ``cam0`` = top / XY
- ``cam1`` = side1 / XZ
- ``cam2`` = side2 / YZ
- polynomial feature order is
  ``[cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y]``
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Union
import warnings

import numpy as np
from joblib import load

from scripts.calibration.refractive_geometry import (
    DEFAULT_MAX_RAY_SYSTEM_CONDITION,
    EPSILON,
    FlatPortCamera,
)


CAMERA_NAMES = ("cam0", "cam1", "cam2")
REFRACTIVE_FILENAMES = tuple(
    f"{camera}_bundle_refractive.json" for camera in CAMERA_NAMES
)
SCORE_PROFILES = ("production_fast", "legacy_combined")


class CalibrationCapabilityError(RuntimeError):
    """Raised when a selected backend cannot provide a requested operation."""


def _observation_batch(
    observations: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(observations, dtype=float)
    if values.shape[-2:] != (3, 2):
        raise ValueError(
            "observations must have trailing shape (3, 2) in cam0/cam1/cam2 "
            f"order, got {values.shape}"
        )
    return values.reshape(-1, 3, 2), values.shape[:-2]


def _world_batch(world_points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(world_points, dtype=float)
    if values.shape[-1:] != (3,):
        raise ValueError(
            f"world_points must have trailing shape (3,), got {values.shape}"
        )
    return values.reshape(-1, 3), values.shape[:-1]


def _validate_profile(profile: str) -> str:
    profile = str(profile)
    if profile not in SCORE_PROFILES:
        raise ValueError(
            f"unknown score profile {profile!r}; expected one of {SCORE_PROFILES}"
        )
    return profile


class _PolynomialBackend:
    backend_name = "polynomial"
    score_unit = "px^2"

    def __init__(self, model_folder: Path) -> None:
        self.model_folder = model_folder
        self.model_dir = self._resolve_model_dir(model_folder)
        path_im2xyz = self.model_dir / "imCoords_to_XYZ.joblib"
        if not path_im2xyz.exists():
            raise FileNotFoundError(f"Could not find imCoords_to_XYZ.joblib: {path_im2xyz}")

        self.model_im2xyz = load(path_im2xyz)
        path_xyz2im = self.model_dir / "XYZ_to_imCoords.joblib"
        self.model_xyz2im = load(path_xyz2im) if path_xyz2im.exists() else None
        helper_paths = {
            "missing_cam1": self.model_dir / "xy_yz_to_xz.joblib",
            "missing_cam0": self.model_dir / "xz_yz_to_xy.joblib",
            "missing_cam2": self.model_dir / "xz_xy_to_yz.joblib",
        }
        self.model_xz_from_xy_yz = self._load_optional(helper_paths["missing_cam1"])
        self.model_xy_from_xz_yz = self._load_optional(helper_paths["missing_cam0"])
        self.model_yz_from_xz_xy = self._load_optional(helper_paths["missing_cam2"])
        self._helpers = (
            self.model_xy_from_xz_yz,
            self.model_xz_from_xy_yz,
            self.model_yz_from_xz_xy,
        )
        helper_available = tuple(model is not None for model in self._helpers)
        has_pairwise = any(helper_available)

        default_definition = (
            "sum_squared_reprojection_error_over_3_cameras"
            if self.model_xyz2im is not None
            else "sum_squared_pairwise_prediction_error"
        )
        files: dict[str, Any] = {
            "image_to_world": str(path_im2xyz.resolve()),
            "world_to_image": (
                str(path_xyz2im.resolve()) if path_xyz2im.exists() else None
            ),
            "pairwise_helpers": {
                key: str(path.resolve()) if path.exists() else None
                for key, path in helper_paths.items()
            },
        }
        self.provenance = {
            "backend": self.backend_name,
            "model_folder": str(model_folder.resolve()),
            "model_dir": str(self.model_dir.resolve()),
            "model_files": files,
            "camera_order": list(CAMERA_NAMES),
            "two_view_method": "pairwise_helper_synthetic_view",
            "score_unit": self.score_unit,
            "default_score_profile": "production_fast",
            "score_profiles": {
                "production_fast": {
                    "unit": self.score_unit,
                    "definition": default_definition,
                    "note": (
                        "Matches validated N^3 production scoring. Pairwise SSE is "
                        "used only when the reverse projection model is absent."
                    ),
                },
                "legacy_combined": {
                    "unit": self.score_unit,
                    "definition": (
                        "sum_of_available_reprojection_and_pairwise_prediction_sse"
                    ),
                    "note": "Matches the historical permutation-path scalar scorer.",
                },
            },
        }
        self.capabilities = {
            "triangulation": True,
            "three_view_triangulation": True,
            "two_view_triangulation": all(helper_available),
            "direct_two_view_triangulation": False,
            "projection": self.model_xyz2im is not None,
            "hypothesis_scoring": self.model_xyz2im is not None or has_pairwise,
            "two_view_missing_cam0": helper_available[0],
            "two_view_missing_cam1": helper_available[1],
            "two_view_missing_cam2": helper_available[2],
        }
        self.capability_errors = {
            "projection": "XYZ_to_imCoords.joblib is missing",
            "hypothesis_scoring": (
                "XYZ_to_imCoords.joblib and all pairwise scoring helpers are missing"
            ),
            "two_view_triangulation": (
                "one or more pairwise missing-view helper joblibs are missing"
            ),
            "two_view_missing_cam0": "xz_yz_to_xy.joblib is missing",
            "two_view_missing_cam1": "xy_yz_to_xz.joblib is missing",
            "two_view_missing_cam2": "xz_xy_to_yz.joblib is missing",
        }

    @staticmethod
    def _resolve_model_dir(model_folder: Path) -> Path:
        nested = model_folder / "python_calibration_models"
        if (model_folder / "imCoords_to_XYZ.joblib").exists():
            return model_folder
        if nested.is_dir():
            return nested
        return model_folder

    @staticmethod
    def _load_optional(path: Path):
        return load(path) if path.exists() else None

    def complete(self, flat: np.ndarray) -> np.ndarray:
        completed = np.asarray(flat, dtype=float).reshape(-1, 3, 2).copy()
        finite_views = np.all(np.isfinite(completed), axis=2)
        view_count = np.sum(finite_views, axis=1)
        for missing, model in enumerate(self._helpers):
            if model is None:
                continue
            mask = (view_count == 2) & ~finite_views[:, missing]
            if not np.any(mask):
                continue
            if missing == 0:
                features = np.hstack((completed[mask, 1], completed[mask, 2]))
            elif missing == 1:
                features = np.hstack((completed[mask, 0], completed[mask, 2]))
            else:
                # xz_xy_to_yz.joblib is the one helper whose feature order is
                # not ascending camera order.
                features = np.hstack((completed[mask, 1], completed[mask, 0]))
            completed[mask, missing] = np.asarray(
                model.predict(features), dtype=float
            ).reshape(-1, 2)
        return completed

    def triangulate_flat(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        completed = self.complete(flat)
        valid = np.all(np.isfinite(completed), axis=(1, 2))
        xyz = np.full((len(completed), 3), np.nan, dtype=float)
        if np.any(valid):
            predicted = np.asarray(
                self.model_im2xyz.predict(completed[valid].reshape(-1, 6)),
                dtype=float,
            ).reshape(-1, 3)
            predicted[~np.all(np.isfinite(predicted), axis=1)] = np.nan
            xyz[valid] = predicted
        return xyz, completed

    def _reprojection_sse(
        self, completed: np.ndarray, xyz: np.ndarray
    ) -> np.ndarray:
        scores = np.full(len(completed), np.inf, dtype=float)
        if self.model_xyz2im is None:
            return scores
        valid = np.all(np.isfinite(completed), axis=(1, 2)) & np.all(
            np.isfinite(xyz), axis=1
        )
        if np.any(valid):
            reprojection = np.asarray(
                self.model_xyz2im.predict(xyz[valid]), dtype=float
            ).reshape(-1, 6)
            difference = completed[valid].reshape(-1, 6) - reprojection
            values = np.sum(difference**2, axis=1)
            values[~np.isfinite(values)] = np.inf
            scores[valid] = values
        return scores

    def _pairwise_sse(self, completed: np.ndarray) -> np.ndarray:
        scores = np.full(len(completed), np.inf, dtype=float)
        valid = np.all(np.isfinite(completed), axis=(1, 2))
        if not np.any(valid):
            return scores
        selected = completed[valid]
        values = np.zeros(len(selected), dtype=float)
        used = False
        if self.model_xz_from_xy_yz is not None:
            predicted = np.asarray(
                self.model_xz_from_xy_yz.predict(
                    np.hstack((selected[:, 0], selected[:, 2]))
                ),
                dtype=float,
            ).reshape(-1, 2)
            values += np.sum((selected[:, 1] - predicted) ** 2, axis=1)
            used = True
        if self.model_xy_from_xz_yz is not None:
            predicted = np.asarray(
                self.model_xy_from_xz_yz.predict(
                    np.hstack((selected[:, 1], selected[:, 2]))
                ),
                dtype=float,
            ).reshape(-1, 2)
            values += np.sum((selected[:, 0] - predicted) ** 2, axis=1)
            used = True
        if self.model_yz_from_xz_xy is not None:
            predicted = np.asarray(
                self.model_yz_from_xz_xy.predict(
                    np.hstack((selected[:, 1], selected[:, 0]))
                ),
                dtype=float,
            ).reshape(-1, 2)
            values += np.sum((selected[:, 2] - predicted) ** 2, axis=1)
            used = True
        if used:
            values[~np.isfinite(values)] = np.inf
            scores[valid] = values
        return scores

    def score_flat(
        self,
        completed: np.ndarray,
        xyz: np.ndarray,
        *,
        profile: str,
    ) -> np.ndarray:
        profile = _validate_profile(profile)
        reprojection = self._reprojection_sse(completed, xyz)
        pairwise = self._pairwise_sse(completed)
        if profile == "production_fast":
            return reprojection if self.model_xyz2im is not None else pairwise

        scores = np.zeros(len(completed), dtype=float)
        available = np.zeros(len(completed), dtype=bool)
        for component in (reprojection, pairwise):
            finite = np.isfinite(component)
            scores[finite] += component[finite]
            available |= finite
        scores[~available] = np.inf
        return scores

    def evaluate_flat(
        self, flat: np.ndarray, *, profile: str
    ) -> tuple[np.ndarray, np.ndarray]:
        xyz, completed = self.triangulate_flat(flat)
        return xyz, self.score_flat(completed, xyz, profile=profile)

    def project_flat(self, flat_world: np.ndarray) -> np.ndarray:
        if self.model_xyz2im is None:
            raise CalibrationCapabilityError(
                "polynomial backend cannot project: XYZ_to_imCoords.joblib is missing"
            )
        projected = np.full((len(flat_world), 3, 2), np.nan, dtype=float)
        valid = np.all(np.isfinite(flat_world), axis=1)
        if np.any(valid):
            values = np.asarray(
                self.model_xyz2im.predict(flat_world[valid]), dtype=float
            ).reshape(-1, 3, 2)
            values[~np.all(np.isfinite(values), axis=2)] = np.nan
            projected[valid] = values
        return projected


class _RefractiveBackend:
    backend_name = "refractive"
    score_unit = "mm^2"

    def __init__(
        self,
        model_folder: Path,
        paths: tuple[Path, Path, Path],
        *,
        max_condition_number: float = DEFAULT_MAX_RAY_SYSTEM_CONDITION,
    ) -> None:
        self.model_folder = model_folder
        self.model_dir = model_folder
        self.max_condition_number = float(max_condition_number)
        if not np.isfinite(self.max_condition_number) or self.max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than one")
        cameras = []
        for expected_name, path in zip(CAMERA_NAMES, paths, strict=True):
            try:
                camera = FlatPortCamera.from_json(path)
            except Exception as exc:
                raise ValueError(f"Invalid refractive camera JSON {path}: {exc}") from exc
            self._validate_camera(camera, expected_name, path)
            cameras.append(camera)
        self.cameras = tuple(cameras)
        score_definition = (
            "mean_squared_perpendicular_distance_from_solved_xyz_to_each_"
            "finite_observed_water_ray"
        )
        self.provenance = {
            "backend": self.backend_name,
            "model_folder": str(model_folder.resolve()),
            "model_files": {
                camera: str(path.resolve())
                for camera, path in zip(CAMERA_NAMES, paths, strict=True)
            },
            "camera_order": list(CAMERA_NAMES),
            "two_view_method": "direct_refracted_ray_intersection",
            "score_unit": self.score_unit,
            "default_score_profile": "production_fast",
            "score_profiles": {
                profile: {"unit": self.score_unit, "definition": score_definition}
                for profile in SCORE_PROFILES
            },
            "ray_system_guard": {
                "minimum_rank": 3,
                "maximum_condition_number": self.max_condition_number,
                "failure_xyz": "NaN",
                "failure_score": "infinity",
            },
        }
        self.capabilities = {
            "triangulation": True,
            "three_view_triangulation": True,
            "two_view_triangulation": True,
            "direct_two_view_triangulation": True,
            "projection": True,
            "hypothesis_scoring": True,
            "two_view_missing_cam0": True,
            "two_view_missing_cam1": True,
            "two_view_missing_cam2": True,
        }
        self.capability_errors: dict[str, str] = {}

    @staticmethod
    def _validate_camera(
        camera: FlatPortCamera, expected_name: str, path: Path
    ) -> None:
        if camera.name != expected_name:
            raise ValueError(
                f"{path} declares camera name {camera.name!r}; expected {expected_name!r}"
            )
        numeric = np.asarray(
            [
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.k1,
                camera.k2,
                *camera.rotation_vector,
                *camera.camera_center,
                *camera.interface_normal,
                *camera.interface_point,
                camera.glass_thickness_mm,
                camera.index_air,
                camera.index_glass,
                camera.index_water,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{path} contains non-finite camera parameters")
        if camera.fx <= 0 or camera.fy <= 0:
            raise ValueError(f"{path} must declare positive focal lengths")
        if camera.layer_count not in (0, 1, 2):
            raise ValueError(f"{path} has unsupported layer_count={camera.layer_count}")
        if min(camera.index_air, camera.index_glass, camera.index_water) <= 0:
            raise ValueError(f"{path} must declare positive refractive indices")
        if np.linalg.norm(np.asarray(camera.interface_normal, dtype=float)) <= EPSILON:
            raise ValueError(f"{path} has a zero interface normal")
        # Exercise the optical axis during loading so auto-detection does not
        # select a syntactically valid but geometrically unusable bundle.
        camera.backproject(np.array([[camera.cx, camera.cy]], dtype=float))

    @staticmethod
    def _safe_unique_backprojection(
        camera: FlatPortCamera, pixels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        unique, inverse = np.unique(pixels, axis=0, return_inverse=True)
        unique_origins = np.full((len(unique), 3), np.nan, dtype=float)
        unique_directions = np.full((len(unique), 3), np.nan, dtype=float)
        try:
            origins, directions = camera.backproject(unique)
            unique_origins[:] = origins
            unique_directions[:] = directions
        except ValueError:
            # One out-of-domain pixel must not poison otherwise valid rows.
            for index, pixel in enumerate(unique):
                try:
                    origin, direction = camera.backproject(pixel.reshape(1, 2))
                except ValueError:
                    continue
                unique_origins[index] = origin[0]
                unique_directions[index] = direction[0]
        return unique_origins[inverse], unique_directions[inverse]

    def _backproject_flat(
        self, flat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite_views = np.all(np.isfinite(flat), axis=2)
        origins = np.full((len(flat), 3, 3), np.nan, dtype=float)
        directions = np.full((len(flat), 3, 3), np.nan, dtype=float)
        for camera_index, camera in enumerate(self.cameras):
            mask = finite_views[:, camera_index]
            if not np.any(mask):
                continue
            selected_origins, selected_directions = self._safe_unique_backprojection(
                camera, flat[mask, camera_index]
            )
            origins[mask, camera_index] = selected_origins
            directions[mask, camera_index] = selected_directions
        ray_valid = finite_views & np.all(np.isfinite(origins), axis=2) & np.all(
            np.isfinite(directions), axis=2
        )
        return origins, directions, ray_valid

    def _solve_and_score(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        ray_valid: np.ndarray,
        *,
        xyz_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = np.sum(ray_valid, axis=1)
        safe_directions = np.where(ray_valid[:, :, None], directions, 0.0)
        identity = np.eye(3, dtype=float)
        projectors = (
            identity[None, None, :, :]
            - safe_directions[:, :, :, None] * safe_directions[:, :, None, :]
        )
        projectors *= ray_valid[:, :, None, None]
        system = np.sum(projectors, axis=1)
        eigenvalues = np.linalg.eigvalsh(system)
        condition = np.full(len(system), np.inf, dtype=float)
        positive = eigenvalues[:, 0] > EPSILON
        condition[positive] = eigenvalues[positive, -1] / eigenvalues[positive, 0]
        well_conditioned = (
            (count >= 2)
            & positive
            & np.isfinite(condition)
            & (condition <= self.max_condition_number)
        )

        xyz = np.full((len(system), 3), np.nan, dtype=float)
        if xyz_override is None:
            safe_origins = np.where(ray_valid[:, :, None], origins, 0.0)
            right_hand_side = np.sum(
                np.matmul(projectors, safe_origins[:, :, :, None]).squeeze(-1),
                axis=1,
            )
            if np.any(well_conditioned):
                try:
                    xyz[well_conditioned] = np.linalg.solve(
                        system[well_conditioned],
                        right_hand_side[well_conditioned, :, None],
                    ).squeeze(-1)
                except np.linalg.LinAlgError:
                    for row in np.where(well_conditioned)[0]:
                        try:
                            xyz[row] = np.linalg.solve(system[row], right_hand_side[row])
                        except np.linalg.LinAlgError:
                            well_conditioned[row] = False
        else:
            override = np.asarray(xyz_override, dtype=float).reshape(-1, 3)
            if len(override) != len(system):
                raise ValueError("xyz leading shape must match observations leading shape")
            use = well_conditioned & np.all(np.isfinite(override), axis=1)
            xyz[use] = override[use]
            well_conditioned &= use

        scores = np.full(len(system), np.inf, dtype=float)
        score_rows = well_conditioned & np.all(np.isfinite(xyz), axis=1)
        if np.any(score_rows):
            safe_origins = np.where(ray_valid[:, :, None], origins, 0.0)
            offsets = xyz[:, None, :] - safe_origins
            perpendicular = np.matmul(projectors, offsets[:, :, :, None]).squeeze(-1)
            squared_distance = np.sum(perpendicular**2, axis=2)
            squared_distance *= ray_valid
            scores[score_rows] = (
                np.sum(squared_distance[score_rows], axis=1) / count[score_rows]
            )
        scores[~np.isfinite(scores)] = np.inf
        xyz[~score_rows] = np.nan
        return xyz, scores

    def evaluate_flat(
        self, flat: np.ndarray, *, profile: str
    ) -> tuple[np.ndarray, np.ndarray]:
        _validate_profile(profile)  # Both names resolve to the same native score.
        origins, directions, ray_valid = self._backproject_flat(flat)
        return self._solve_and_score(origins, directions, ray_valid)

    def score_flat(
        self,
        flat: np.ndarray,
        xyz: np.ndarray | None,
        *,
        profile: str,
    ) -> np.ndarray:
        _validate_profile(profile)
        origins, directions, ray_valid = self._backproject_flat(flat)
        if xyz is None:
            _, scores = self._solve_and_score(origins, directions, ray_valid)
        else:
            _, scores = self._solve_and_score(
                origins, directions, ray_valid, xyz_override=xyz
            )
        return scores

    def project_flat(self, flat_world: np.ndarray) -> np.ndarray:
        return np.stack(
            [camera.project(flat_world) for camera in self.cameras], axis=1
        )


class Calibration:
    """Resolve and expose a polynomial or refractive calibration backend.

    Parameters
    ----------
    model_folder:
        Directory containing either a complete cam0/cam1/cam2 refractive JSON
        bundle or the historical joblib model files.
    backend:
        ``"auto"`` prefers a valid complete refractive bundle, then falls back
        to polynomial assets.  Explicit ``"refractive"`` and ``"polynomial"``
        never fall back to another geometry.
    """

    def __init__(
        self,
        model_folder: Union[str, Path],
        backend: str = "auto",
        *,
        max_ray_system_condition: float = DEFAULT_MAX_RAY_SYSTEM_CONDITION,
    ) -> None:
        self.model_folder = Path(model_folder)
        if not self.model_folder.is_dir():
            raise FileNotFoundError(
                f"Calibration directory does not exist: {self.model_folder}"
            )
        requested = str(backend).lower()
        if requested not in ("auto", "refractive", "polynomial"):
            raise ValueError(
                "backend must be one of 'auto', 'refractive', or 'polynomial', "
                f"got {backend!r}"
            )

        refractive_paths = tuple(
            self.model_folder / filename for filename in REFRACTIVE_FILENAMES
        )
        present_refractive = tuple(path.exists() for path in refractive_paths)
        auto_notes: list[str] = []
        resolved_backend: _PolynomialBackend | _RefractiveBackend
        if requested in ("auto", "refractive") and all(present_refractive):
            try:
                resolved_backend = _RefractiveBackend(
                    self.model_folder,
                    refractive_paths,  # type: ignore[arg-type]
                    max_condition_number=max_ray_system_condition,
                )
            except (ValueError, OSError) as exc:
                if requested == "refractive":
                    raise
                auto_notes.append(f"Rejected refractive bundle: {exc}")
                resolved_backend = self._load_polynomial_after_auto(auto_notes)
        elif requested == "refractive":
            missing = [
                str(path) for path, present in zip(refractive_paths, present_refractive)
                if not present
            ]
            raise FileNotFoundError(
                "Refractive backend requires all three camera JSON files; missing: "
                + ", ".join(missing)
            )
        elif requested == "auto":
            if any(present_refractive):
                missing_names = [
                    path.name
                    for path, present in zip(refractive_paths, present_refractive)
                    if not present
                ]
                auto_notes.append(
                    "Ignored incomplete refractive bundle; missing "
                    + ", ".join(missing_names)
                )
            resolved_backend = self._load_polynomial_after_auto(auto_notes)
        else:
            resolved_backend = _PolynomialBackend(self.model_folder)

        self._backend = resolved_backend
        self.model_dir = resolved_backend.model_dir
        self.backend_name = resolved_backend.backend_name
        self.score_unit = resolved_backend.score_unit
        self.capabilities = dict(resolved_backend.capabilities)
        self.capability_errors = dict(resolved_backend.capability_errors)
        self.provenance = deepcopy(resolved_backend.provenance)
        self.provenance["requested_backend"] = requested
        self.provenance["resolved_backend"] = self.backend_name
        if auto_notes:
            self.provenance["auto_detection_notes"] = auto_notes

        # Compatibility attributes remain available for old analysis scripts.
        # Production triangulation must use the public methods above.
        polynomial = (
            resolved_backend if isinstance(resolved_backend, _PolynomialBackend) else None
        )
        self._model_im2xyz = polynomial.model_im2xyz if polynomial else None
        self._model_xyz2im = polynomial.model_xyz2im if polynomial else None
        self._model_xz_from_xy_yz = (
            polynomial.model_xz_from_xy_yz if polynomial else None
        )
        self._model_xy_from_xz_yz = (
            polynomial.model_xy_from_xz_yz if polynomial else None
        )
        self._model_yz_from_xz_xy = (
            polynomial.model_yz_from_xz_xy if polynomial else None
        )
        self.has_registration_model = self.capabilities["projection"]
        self.has_pairwise_models = bool(
            polynomial is not None and all(model is not None for model in polynomial._helpers)
        )
        self.has_any_pairwise_model = bool(
            polynomial is not None and any(model is not None for model in polynomial._helpers)
        )
        self.has_geometric_scoring = self.capabilities["hypothesis_scoring"]

        if polynomial is not None and not self.has_registration_model:
            if self.has_any_pairwise_model:
                warnings.warn(
                    (
                        f"XYZ_to_imCoords.joblib is missing in {self.model_dir}. "
                        "Production scoring explicitly uses pairwise pixel SSE."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    (
                        f"XYZ_to_imCoords.joblib is missing in {self.model_dir}, "
                        "and no pairwise helpers are available; hypothesis scoring "
                        "is unsupported."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _load_polynomial_after_auto(
        self, notes: list[str]
    ) -> _PolynomialBackend:
        try:
            return _PolynomialBackend(self.model_folder)
        except FileNotFoundError as exc:
            details = f" Auto-detection notes: {'; '.join(notes)}" if notes else ""
            raise FileNotFoundError(
                f"No valid refractive or polynomial calibration in {self.model_folder}."
                f"{details} Polynomial error: {exc}"
            ) from exc

    @property
    def score_profiles(self) -> Mapping[str, Mapping[str, Any]]:
        return deepcopy(self.provenance["score_profiles"])

    def require_capability(self, capability: str) -> None:
        if capability not in self.capabilities:
            raise KeyError(
                f"unknown calibration capability {capability!r}; expected one of "
                f"{sorted(self.capabilities)}"
            )
        if self.capabilities[capability]:
            return
        reason = self.capability_errors.get(capability, "capability is unavailable")
        raise CalibrationCapabilityError(
            f"{self.backend_name} calibration does not support {capability}: {reason}"
        )

    def evaluate_hypotheses(
        self,
        observations: np.ndarray,
        *,
        profile: str = "production_fast",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fused batched triangulation and native-unit hypothesis scoring."""
        profile = _validate_profile(profile)
        flat, leading_shape = _observation_batch(observations)
        xyz, scores = self._backend.evaluate_flat(flat, profile=profile)
        return xyz.reshape(leading_shape + (3,)), scores.reshape(leading_shape)

    def triangulate(self, observations: np.ndarray) -> np.ndarray:
        """Triangulate two or three finite camera observations in a batch."""
        flat, leading_shape = _observation_batch(observations)
        if isinstance(self._backend, _PolynomialBackend):
            xyz, _ = self._backend.triangulate_flat(flat)
        else:
            xyz, _ = self._backend.evaluate_flat(
                flat, profile="production_fast"
            )
        return xyz.reshape(leading_shape + (3,))

    def score_hypotheses(
        self,
        observations: np.ndarray,
        xyz: np.ndarray | None = None,
        *,
        profile: str = "production_fast",
    ) -> np.ndarray:
        """Return scores in :attr:`score_unit` under an explicit profile.

        Refractive scores are the mean squared perpendicular distance from the
        solved/provided XYZ to every finite observed water ray (``mm^2``).
        ``production_fast`` and ``legacy_combined`` intentionally resolve to
        that same native physical definition.

        Polynomial ``production_fast`` is the validated N^3 reprojection SSE;
        ``legacy_combined`` adds every available pairwise-helper SSE, matching
        the historical permutation scorer.  See :attr:`score_profiles` for the
        exact definition resolved by the loaded asset set.
        """
        profile = _validate_profile(profile)
        flat, leading_shape = _observation_batch(observations)
        flat_xyz = None
        if xyz is not None:
            flat_xyz, xyz_leading = _world_batch(xyz)
            if xyz_leading != leading_shape:
                raise ValueError(
                    "xyz leading shape must match observations leading shape, "
                    f"got {xyz_leading} and {leading_shape}"
                )

        if isinstance(self._backend, _PolynomialBackend):
            if flat_xyz is None:
                computed_xyz, completed = self._backend.triangulate_flat(flat)
            else:
                computed_xyz, completed = flat_xyz, self._backend.complete(flat)
            scores = self._backend.score_flat(
                completed, computed_xyz, profile=profile
            )
        else:
            scores = self._backend.score_flat(
                flat, flat_xyz, profile=profile
            )
        return scores.reshape(leading_shape)

    def project_world(self, world_points: np.ndarray) -> np.ndarray:
        """Project batched XYZ to all cameras, returning trailing ``(3, 2)``."""
        self.require_capability("projection")
        flat, leading_shape = _world_batch(world_points)
        projected = self._backend.project_flat(flat)
        return projected.reshape(leading_shape + (3, 2))

    # ------------------------------------------------------------------
    # Historical scalar API.  These aliases preserve old scripts while all
    # production triangulators migrate to the public batch methods above.

    @staticmethod
    def _as_triplet(triplet: np.ndarray) -> np.ndarray:
        triplet = np.asarray(triplet, dtype=float)
        if triplet.shape != (3, 2):
            raise ValueError(f"triplet must have shape (3, 2), got {triplet.shape}")
        return triplet

    def compute_XYZ_from_imageCoord_triplet(self, triplet: np.ndarray) -> np.ndarray:
        return np.asarray(self.triangulate(self._as_triplet(triplet)), dtype=float)

    def _require_polynomial_helper(self) -> _PolynomialBackend:
        if not isinstance(self._backend, _PolynomialBackend):
            raise CalibrationCapabilityError(
                "pairwise synthetic-view helpers exist only in the polynomial backend"
            )
        return self._backend

    def compute_XZ_imcoords_from_XY_YZ(
        self, xy: np.ndarray, yz: np.ndarray
    ) -> np.ndarray:
        backend = self._require_polynomial_helper()
        if backend.model_xz_from_xy_yz is None:
            raise RuntimeError("xy_yz_to_xz.joblib is required to infer cam1 from cam0+cam2")
        features = np.hstack((np.asarray(xy, float), np.asarray(yz, float))).reshape(1, 4)
        return np.asarray(backend.model_xz_from_xy_yz.predict(features)).ravel()

    def compute_XY_imcoords_from_XZ_YZ(
        self, xz: np.ndarray, yz: np.ndarray
    ) -> np.ndarray:
        backend = self._require_polynomial_helper()
        if backend.model_xy_from_xz_yz is None:
            raise RuntimeError("xz_yz_to_xy.joblib is required to infer cam0 from cam1+cam2")
        features = np.hstack((np.asarray(xz, float), np.asarray(yz, float))).reshape(1, 4)
        return np.asarray(backend.model_xy_from_xz_yz.predict(features)).ravel()

    def compute_YZ_imcoords_from_XZ_XY(
        self, xz: np.ndarray, xy: np.ndarray
    ) -> np.ndarray:
        backend = self._require_polynomial_helper()
        if backend.model_yz_from_xz_xy is None:
            raise RuntimeError("xz_xy_to_yz.joblib is required to infer cam2 from cam1+cam0")
        features = np.hstack((np.asarray(xz, float), np.asarray(xy, float))).reshape(1, 4)
        return np.asarray(backend.model_yz_from_xz_xy.predict(features)).ravel()

    def complete_triplet(self, triplet: np.ndarray) -> np.ndarray:
        triplet = self._as_triplet(triplet)
        if isinstance(self._backend, _PolynomialBackend):
            return self._backend.complete(triplet.reshape(1, 3, 2))[0]
        return triplet.copy()

    def compute_pairwise_consistency_error(self, triplet: np.ndarray) -> float:
        triplet = self._as_triplet(triplet)
        if not isinstance(self._backend, _PolynomialBackend):
            return np.inf
        completed = self._backend.complete(triplet.reshape(1, 3, 2))
        return float(self._backend._pairwise_sse(completed)[0])

    def compute_reprojection_error(self, triplet: np.ndarray) -> float:
        triplet = self._as_triplet(triplet)
        if isinstance(self._backend, _PolynomialBackend):
            xyz, completed = self._backend.triangulate_flat(triplet.reshape(1, 3, 2))
            return float(self._backend._reprojection_sse(completed, xyz)[0])
        return float(np.asarray(self.score_hypotheses(triplet)))

    def compute_triplet_registration_error(self, triplet: np.ndarray) -> float:
        return float(
            np.asarray(
                self.score_hypotheses(
                    self._as_triplet(triplet), profile="legacy_combined"
                )
            )
        )

    def compute_imCoords_from_XYZ(self, xyz: np.ndarray) -> np.ndarray:
        values = np.asarray(xyz, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, 3)
        return self.project_world(values)

    def project_world_to_view(self, xyz: np.ndarray, view: str) -> np.ndarray:
        view_to_cam = {"top_cam": 0, "side1_cam": 1, "side2_cam": 2}
        if view not in view_to_cam:
            raise ValueError(f"unknown view {view!r}; expected one of {list(view_to_cam)}")
        all_views = self.project_world(xyz)
        if all_views.ndim == 2:
            all_views = all_views.reshape(1, 3, 2)
        return all_views[:, view_to_cam[view], :]
