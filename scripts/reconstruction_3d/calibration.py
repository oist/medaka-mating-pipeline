"""
Calibration wrapper for 3D reconstruction.

Loads `imCoords_to_XYZ.joblib`, the optional `XYZ_to_imCoords.joblib`, and the
optional pairwise helper models, then
provides:

- 2D -> 3D prediction
- 3D -> 2D reprojection
- triplet registration error
- one-view recovery from the other two views when helper models are present

Conventions:
- `cam0` = top / XY
- `cam1` = side1 / XZ
- `cam2` = side2 / YZ
- model input order:
  `[cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y]`
"""

from pathlib import Path
from typing import Union
import warnings

import numpy as np
from joblib import load


class Calibration:
    """Wrap calibration joblib models for 2D/3D conversion and scoring."""

    def __init__(self, model_folder: Union[str, Path]) -> None:
        self.model_folder = Path(model_folder)
        if not self.model_folder.is_dir():
            raise FileNotFoundError(f"Calibration directory does not exist: {self.model_folder}")

        self.model_dir = self._resolve_model_dir(self.model_folder)

        path_im2xyz = self.model_dir / "imCoords_to_XYZ.joblib"
        if not path_im2xyz.exists():
            raise FileNotFoundError(f"Could not find imCoords_to_XYZ.joblib: {path_im2xyz}")

        self._model_im2xyz = load(path_im2xyz)

        path_xyz2im = self.model_dir / "XYZ_to_imCoords.joblib"
        self.has_registration_model = path_xyz2im.exists()
        self._model_xyz2im = load(path_xyz2im) if self.has_registration_model else None

        self._model_xz_from_xy_yz = self._load_optional_model("xy_yz_to_xz.joblib")
        self._model_xy_from_xz_yz = self._load_optional_model("xz_yz_to_xy.joblib")
        self._model_yz_from_xz_xy = self._load_optional_model("xz_xy_to_yz.joblib")

        self.has_pairwise_models = all(
            model is not None
            for model in (
                self._model_xz_from_xy_yz,
                self._model_xy_from_xz_yz,
                self._model_yz_from_xz_xy,
            )
        )
        self.has_any_pairwise_model = any(
            model is not None
            for model in (
                self._model_xz_from_xy_yz,
                self._model_xy_from_xz_yz,
                self._model_yz_from_xz_xy,
            )
        )
        self.has_geometric_scoring = self.has_registration_model or self.has_any_pairwise_model

        if not self.has_registration_model and not self.has_any_pairwise_model:
            warnings.warn(
                (
                    f"{path_xyz2im.name} is missing in {self.model_dir}. "
                    "Three-view permutation scoring will rely on input order "
                    "instead of reprojection error."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        elif not self.has_registration_model and self.has_any_pairwise_model:
            warnings.warn(
                (
                    f"{path_xyz2im.name} is missing in {self.model_dir}. "
                    "Falling back to pairwise calibration models for triplet scoring."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    def _resolve_model_dir(self, model_folder: Path) -> Path:
        """Support both flat model folders and `python_calibration_models/`."""
        nested = model_folder / "python_calibration_models"
        if (model_folder / "imCoords_to_XYZ.joblib").exists():
            return model_folder
        if nested.is_dir():
            return nested
        return model_folder

    def _load_optional_model(self, filename: str):
        path = self.model_dir / filename
        return load(path) if path.exists() else None

    def _as_triplet(self, triplet: np.ndarray) -> np.ndarray:
        triplet = np.asarray(triplet, dtype=float)
        if triplet.shape != (3, 2):
            raise ValueError(f"triplet must have shape (3, 2), got {triplet.shape}")
        return triplet

    def compute_XYZ_from_imageCoord_triplet(self, triplet: np.ndarray) -> np.ndarray:
        """
        Predict one 3D point from a three-view 2D triplet.

        `triplet` must have shape `(3, 2)` ordered as
        `cam0_xy, cam1_xy, cam2_xy`.
        """
        triplet = self._as_triplet(triplet)
        # Flatten in cam0/cam1/cam2 order.
        feat = triplet.ravel().reshape(1, -1)
        xyz = self._model_im2xyz.predict(feat)
        return xyz.ravel()

    def compute_XZ_imcoords_from_XY_YZ(
        self, xy: np.ndarray, yz: np.ndarray
    ) -> np.ndarray:
        """Infer XZ-view image coordinates from XY and YZ views."""
        if self._model_xz_from_xy_yz is None:
            raise RuntimeError("xy_yz_to_xz.joblib is required to infer cam1 from cam0+cam2")
        inp = np.hstack((np.asarray(xy, dtype=float), np.asarray(yz, dtype=float))).reshape(1, 4)
        return np.asarray(self._model_xz_from_xy_yz.predict(inp)).ravel()

    def compute_XY_imcoords_from_XZ_YZ(
        self, xz: np.ndarray, yz: np.ndarray
    ) -> np.ndarray:
        """Infer XY-view image coordinates from XZ and YZ views."""
        if self._model_xy_from_xz_yz is None:
            raise RuntimeError("xz_yz_to_xy.joblib is required to infer cam0 from cam1+cam2")
        inp = np.hstack((np.asarray(xz, dtype=float), np.asarray(yz, dtype=float))).reshape(1, 4)
        return np.asarray(self._model_xy_from_xz_yz.predict(inp)).ravel()

    def compute_YZ_imcoords_from_XZ_XY(
        self, xz: np.ndarray, xy: np.ndarray
    ) -> np.ndarray:
        """Infer YZ-view image coordinates from XZ and XY views."""
        if self._model_yz_from_xz_xy is None:
            raise RuntimeError("xz_xy_to_yz.joblib is required to infer cam2 from cam1+cam0")
        inp = np.hstack((np.asarray(xz, dtype=float), np.asarray(xy, dtype=float))).reshape(1, 4)
        return np.asarray(self._model_yz_from_xz_xy.predict(inp)).ravel()

    def complete_triplet(self, triplet: np.ndarray) -> np.ndarray:
        """
        Fill exactly one missing camera view from the other two when helper
        calibration models are available.
        """
        triplet = self._as_triplet(triplet).copy()
        valid_rows = np.all(np.isfinite(triplet), axis=1)
        missing_rows = np.where(~valid_rows)[0]
        if len(missing_rows) != 1:
            return triplet

        missing = int(missing_rows[0])
        try:
            if missing == 0 and valid_rows[1] and valid_rows[2]:
                triplet[0] = self.compute_XY_imcoords_from_XZ_YZ(triplet[1], triplet[2])
            elif missing == 1 and valid_rows[0] and valid_rows[2]:
                triplet[1] = self.compute_XZ_imcoords_from_XY_YZ(triplet[0], triplet[2])
            elif missing == 2 and valid_rows[0] and valid_rows[1]:
                triplet[2] = self.compute_YZ_imcoords_from_XZ_XY(triplet[1], triplet[0])
        except RuntimeError:
            return triplet

        return triplet

    def compute_pairwise_consistency_error(self, triplet: np.ndarray) -> float:
        """
        Score a full triplet using the auxiliary pairwise calibration models.
        """
        triplet = self._as_triplet(triplet)
        if np.any(np.isnan(triplet)):
            return np.inf

        errors = []
        if self._model_xz_from_xy_yz is not None:
            pred_xz = self.compute_XZ_imcoords_from_XY_YZ(triplet[0], triplet[2])
            errors.append(float(np.sum((triplet[1] - pred_xz) ** 2)))
        if self._model_xy_from_xz_yz is not None:
            pred_xy = self.compute_XY_imcoords_from_XZ_YZ(triplet[1], triplet[2])
            errors.append(float(np.sum((triplet[0] - pred_xy) ** 2)))
        if self._model_yz_from_xz_xy is not None:
            pred_yz = self.compute_YZ_imcoords_from_XZ_XY(triplet[1], triplet[0])
            errors.append(float(np.sum((triplet[2] - pred_yz) ** 2)))

        if not errors:
            return np.inf
        return float(sum(errors))

    def compute_reprojection_error(self, triplet: np.ndarray) -> float:
        """Score a full triplet by projecting the inferred 3D point back to 2D."""
        triplet = self._as_triplet(triplet)
        if self._model_xyz2im is None:
            return np.inf
        if np.any(np.isnan(triplet)):
            return np.inf
        xyz = self.compute_XYZ_from_imageCoord_triplet(triplet)
        reproj = self._model_xyz2im.predict(xyz.reshape(1, -1))
        if reproj.ndim == 2 and reproj.size == 6:
            reproj = reproj.reshape(3, 2)
        else:
            reproj = np.asarray(reproj).reshape(-1, 3, 2)[0]
        diff = triplet - reproj
        return float(np.sum(diff ** 2))

    def compute_triplet_registration_error(self, triplet: np.ndarray) -> float:
        """
        Compute geometric consistency error for a 2D triplet.

        The scorer combines whichever calibration artifacts are available:

        - reprojection error via `XYZ_to_imCoords.joblib`
        - pairwise consistency error via the helper view-to-view models

        If exactly one view is missing and a helper model can infer it, the
        triplet is completed first. If no scoring model is available, the
        method returns `np.inf`.
        """
        triplet = self.complete_triplet(triplet)
        if np.any(np.isnan(triplet)):
            return np.inf

        errors = []
        reproj_error = self.compute_reprojection_error(triplet)
        if np.isfinite(reproj_error):
            errors.append(reproj_error)
        pairwise_error = self.compute_pairwise_consistency_error(triplet)
        if np.isfinite(pairwise_error):
            errors.append(pairwise_error)

        if not errors:
            return np.inf
        return float(sum(errors))

    def compute_imCoords_from_XYZ(self, xyz: np.ndarray) -> np.ndarray:
        """
        Reproject 3D coordinates into all three camera views.
        """
        if self._model_xyz2im is None:
            raise RuntimeError("XYZ_to_imCoords.joblib is required for 3D->2D projection")
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        out = self._model_xyz2im.predict(xyz)
        if out.ndim == 2 and out.shape[1] == 6:
            return out.reshape(-1, 3, 2)
        return out
