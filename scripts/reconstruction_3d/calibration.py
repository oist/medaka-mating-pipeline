"""
Calibration wrapper for 3D reconstruction.

Loads `imCoords_to_XYZ.joblib` and the optional `XYZ_to_imCoords.joblib`, then
provides:

- 2D -> 3D prediction
- 3D -> 2D reprojection
- triplet registration error

Conventions:
- `cam0` = top / XY
- `cam1` = side1 / XZ
- `cam2` = side2 / YZ
- model input order:
  `[cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y]`
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from joblib import load


class Calibration:
    """Wrap calibration joblib models for 2D/3D conversion and scoring."""

    def __init__(self, model_folder: Union[str, Path]) -> None:
        self.model_folder = Path(model_folder)
        if not self.model_folder.is_dir():
            raise FileNotFoundError(f"Calibration directory does not exist: {self.model_folder}")

        path_im2xyz = self.model_folder / "imCoords_to_XYZ.joblib"
        if not path_im2xyz.exists():
            raise FileNotFoundError(f"Could not find imCoords_to_XYZ.joblib: {path_im2xyz}")

        self._model_im2xyz = load(path_im2xyz)

        path_xyz2im = self.model_folder / "XYZ_to_imCoords.joblib"
        self._model_xyz2im = load(path_xyz2im) if path_xyz2im.exists() else None

    def compute_XYZ_from_imageCoord_triplet(self, triplet: np.ndarray) -> np.ndarray:
        """
        Predict one 3D point from a three-view 2D triplet.

        `triplet` must have shape `(3, 2)` ordered as
        `cam0_xy, cam1_xy, cam2_xy`.
        """
        triplet = np.asarray(triplet, dtype=float)
        if triplet.shape != (3, 2):
            raise ValueError(f"triplet must have shape (3, 2), got {triplet.shape}")
        # Flatten in cam0/cam1/cam2 order.
        feat = triplet.ravel().reshape(1, -1)
        xyz = self._model_im2xyz.predict(feat)
        return xyz.ravel()

    def compute_triplet_registration_error(self, triplet: np.ndarray) -> float:
        """
        Compute reprojection error for a full 2D triplet.

        The error is the sum of squared pixel differences between the original
        triplet and the reprojected triplet. If `XYZ_to_imCoords.joblib` is not
        available, the method returns `0.0`.
        """
        if self._model_xyz2im is None:
            return 0.0
        triplet = np.asarray(triplet, dtype=float)
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

    def compute_XY_imcoords_from_XZ_YZ(
        self, xz: np.ndarray, yz: np.ndarray
    ) -> np.ndarray:
        """Placeholder for deriving XY from XZ and YZ via a 3D intermediate."""
        raise NotImplementedError(
            "imCoords_to_XYZ alone cannot derive XY directly from XZ+YZ. "
            "Use a 3D intermediate plus projection, or provide an auxiliary model."
        )
