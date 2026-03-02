"""
3D 重建用校准封装：加载 imCoords_to_XYZ.joblib（及可选 XYZ_to_imCoords.joblib），
提供 2D↔3D 与配准误差接口。

约定：cam0=top/XY, cam1=side1/XZ, cam2=side2/YZ。
模型输入顺序：[cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y]。
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from joblib import load


class Calibration:
    """封装校准模型目录中的 joblib 模型，用于 2D→3D 与配准误差。"""

    def __init__(self, model_folder: Union[str, Path]) -> None:
        self.model_folder = Path(model_folder)
        if not self.model_folder.is_dir():
            raise FileNotFoundError(f"校准目录不存在: {self.model_folder}")

        path_im2xyz = self.model_folder / "imCoords_to_XYZ.joblib"
        if not path_im2xyz.exists():
            raise FileNotFoundError(f"未找到 imCoords_to_XYZ.joblib: {path_im2xyz}")

        self._model_im2xyz = load(path_im2xyz)

        path_xyz2im = self.model_folder / "XYZ_to_imCoords.joblib"
        self._model_xyz2im = load(path_xyz2im) if path_xyz2im.exists() else None

    def compute_XYZ_from_imageCoord_triplet(self, triplet: np.ndarray) -> np.ndarray:
        """
        从三视角 2D 坐标计算 3D 点。

        triplet: shape (3, 2)，行依次为 cam0_xy, cam1_xy, cam2_xy。
        返回: shape (3,) 的 [X, Y, Z]。
        """
        triplet = np.asarray(triplet, dtype=float)
        if triplet.shape != (3, 2):
            raise ValueError(f"triplet 需为 (3, 2)，当前 {triplet.shape}")
        # 顺序: cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y
        feat = triplet.ravel().reshape(1, -1)
        xyz = self._model_im2xyz.predict(feat)
        return xyz.ravel()

    def compute_triplet_registration_error(self, triplet: np.ndarray) -> float:
        """
        三视角 2D 点与反投影 2D 的配准误差（像素距离平方和）。
        若未提供 XYZ_to_imCoords.joblib，则返回 0。
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
        从 3D 坐标计算三视角 2D 图像坐标。

        xyz: shape (n, 3)。
        返回: shape (n, 3, 2)，[i, cam, 0/1] 为第 i 点在第 cam 视角的 (u, v)。
        """
        if self._model_xyz2im is None:
            raise RuntimeError("未找到 XYZ_to_imCoords.joblib，无法计算 2D 投影")
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
        """从 cam1(XZ) 与 cam2(YZ) 的 2D 推导 cam0(XY) 的 2D。用 3D 中转。"""
        # 需要两条射线交于一点：这里用两视角 2D 反推 3D 再投影到 XY。
        # 简化：用当前 im2xyz 需要三视角，所以若缺 cam0 无法直接算。本管线中由 triangulate 得到 3D 后再 project。
        # 若校准目录有 xy_yz_to_xz 等单独模型可在此加载；此处用 3D 中转需先有 3D。
        # 保守：若没有 xyz2im 则无法实现，返回 NaN
        raise NotImplementedError(
            "仅用 imCoords_to_XYZ 无法从 XZ+YZ 直接得 XY；请使用 fill_missing 时用两视角 2D 先算 3D 再投影，或提供辅助模型"
        )
