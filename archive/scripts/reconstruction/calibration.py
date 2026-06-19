"""
Calibration wrapper for the legacy reconstruction pipeline.

Conventions:

- `cam0` = top / XY
- `cam1` = side1 / XZ
- `cam2` = side2 / YZ
- image coordinate order:
  `[xy_x, xy_y, xz_x, xz_y, yz_x, yz_y]`
"""

import os
from pathlib import Path

import numpy as np
from joblib import load


class Calibration:
    """Load calibration models for 3D reconstruction and registration scoring."""

    def __init__(self, calibration_folder_path: str):
        self.calibration_folder_path = Path(calibration_folder_path)
        self.python_calibration_folder = self.calibration_folder_path / "python_calibration_models"
        if not self.python_calibration_folder.exists():
            self.python_calibration_folder = self.calibration_folder_path

        self._load_models()

    def _load_models(self):
        """Load the required joblib models."""
        self.xyz_getter = load(self.python_calibration_folder / "imCoords_to_XYZ.joblib")
        self.xz_getter = load(self.python_calibration_folder / "xy_yz_to_xz.joblib")
        self.xy_getter = load(self.python_calibration_folder / "xz_yz_to_xy.joblib")
        self.yz_getter = load(self.python_calibration_folder / "xz_xy_to_yz.joblib")
        xyz_to_im_path = self.python_calibration_folder / "XYZ_to_imCoords.joblib"
        self._imcoords_getter = load(xyz_to_im_path) if xyz_to_im_path.exists() else None

    def has_imcoords_from_XYZ(self) -> bool:
        """Return True when the optional 3D->2D reprojection model is available."""
        return self._imcoords_getter is not None

    def compute_imCoords_from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """Predict three-view 2D image coordinates from one or more 3D points."""
        if self._imcoords_getter is None:
            raise RuntimeError("XYZ_to_imCoords.joblib is required for 3D->2D projection")
        single = XYZ.ndim == 1
        if single:
            XYZ = XYZ.reshape(1, 3)
        flat = self._imcoords_getter.predict(XYZ)
        out = flat.reshape(-1, 3, 2)
        return out[0] if single else out

    def compute_XYZ_from_imageCoord_triplet(self, imCoords: np.ndarray) -> np.ndarray:
        """Predict one 3D point from a full three-view 2D triplet."""
        flat = imCoords.reshape(-1, 6)
        return self.xyz_getter.predict(flat).ravel()

    def compute_XZ_imcoords_from_XY_YZ(self, xy_imCoord: np.ndarray, yz_imCoord: np.ndarray) -> np.ndarray:
        """Infer XZ-view image coordinates from XY and YZ views."""
        inp = np.hstack((xy_imCoord, yz_imCoord)).reshape(1, 4)
        return self.xz_getter.predict(inp).ravel()

    def compute_XY_imcoords_from_XZ_YZ(self, xz_imCoord: np.ndarray, yz_imCoord: np.ndarray) -> np.ndarray:
        """Infer XY-view image coordinates from XZ and YZ views."""
        inp = np.hstack((xz_imCoord, yz_imCoord)).reshape(1, 4)
        return self.xy_getter.predict(inp).ravel()

    def compute_YZ_imcoords_from_XZ_XY(self, xz_imCoord: np.ndarray, xy_imCoord: np.ndarray) -> np.ndarray:
        """Infer YZ-view image coordinates from XZ and XY views."""
        inp = np.hstack((xz_imCoord, xy_imCoord)).reshape(1, 4)
        return self.yz_getter.predict(inp).ravel()

    def compute_point_correspondence_error(
        self,
        camIdxs: list[int],
        imCoords_cam1: np.ndarray,
        imCoords_cam2: np.ndarray,
    ) -> float:
        """
        Compute cross-camera correspondence error by deriving the third view
        from two observed views and checking three-view consistency.
        """
        if np.any(np.isnan(imCoords_cam1)) or np.any(np.isnan(imCoords_cam2)):
            return np.nan

        if camIdxs == [0, 1]:
            xy, xz = imCoords_cam1, imCoords_cam2
            yz = self.compute_YZ_imcoords_from_XZ_XY(xz, xy)
            proposed = np.vstack((xz, xy, yz))
        elif camIdxs == [0, 2]:
            xy, yz = imCoords_cam1, imCoords_cam2
            xz = self.compute_XZ_imcoords_from_XY_YZ(xy, yz)
            proposed = np.vstack((xz, xy, yz))
        elif camIdxs == [1, 2]:
            xz, yz = imCoords_cam1, imCoords_cam2
            xy = self.compute_XY_imcoords_from_XZ_YZ(xz, yz)
            proposed = np.vstack((xz, xy, yz))
        else:
            raise ValueError(f"camIdxs must be one of [0,1], [0,2], or [1,2]; got {camIdxs}")

        derived_xz = self.compute_XZ_imcoords_from_XY_YZ(proposed[1], proposed[2])
        img_der_xz = np.vstack((derived_xz, proposed[1], proposed[2]))
        err_xz = np.linalg.norm(proposed - img_der_xz)

        derived_xy = self.compute_XY_imcoords_from_XZ_YZ(proposed[0], proposed[2])
        img_der_xy = np.vstack((proposed[0], derived_xy, proposed[2]))
        err_xy = np.linalg.norm(proposed - img_der_xy)

        derived_yz = self.compute_YZ_imcoords_from_XZ_XY(proposed[0], proposed[1])
        img_der_yz = np.vstack((proposed[0], proposed[1], derived_yz))
        err_yz = np.linalg.norm(proposed - img_der_yz)

        return float(err_xz + err_xy + err_yz)

    def compute_triplet_registration_error(self, imCoords: np.ndarray) -> float:
        """Compute registration error for one complete three-view 2D triplet."""
        if np.any(np.isnan(imCoords)):
            return np.nan
        err_01 = self.compute_point_correspondence_error([0, 1], imCoords[0], imCoords[1])
        err_02 = self.compute_point_correspondence_error([0, 2], imCoords[0], imCoords[2])
        err_12 = self.compute_point_correspondence_error([1, 2], imCoords[1], imCoords[2])
        errs = [e for e in (err_01, err_02, err_12) if not np.isnan(e)]
        if not errs:
            return np.nan
        return float(sum(errs))
