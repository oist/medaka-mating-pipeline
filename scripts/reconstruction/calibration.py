"""
校准模块：加载 joblib 模型并计算 3D 坐标与配准误差

与 detect_board 约定一致：
- imCoords 顺序: cam0(top/XY), cam1(side1/XZ), cam2(side2/YZ) -> [xy_x, xy_y, xz_x, xz_y, yz_x, yz_y]
- 训练数据来自 cali_way*_cam0/1/2_image_coords.txt，cam0=top, cam1=side1, cam2=side2
"""

import os
from pathlib import Path

import numpy as np
from joblib import load


class Calibration:
    """加载校准模型，用于 3D 重建与配准误差计算"""

    def __init__(self, calibration_folder_path: str):
        """
        Args:
            calibration_folder_path: 校准模型目录，需包含:
                - imCoords_to_XYZ.joblib
                - xy_yz_to_xz.joblib
                - xz_yz_to_xy.joblib
                - xz_xy_to_yz.joblib
        """
        self.calibration_folder_path = Path(calibration_folder_path)
        self.python_calibration_folder = self.calibration_folder_path / "python_calibration_models"
        if not self.python_calibration_folder.exists():
            self.python_calibration_folder = self.calibration_folder_path

        self._load_models()

    def _load_models(self):
        """加载 joblib 模型"""
        self.xyz_getter = load(self.python_calibration_folder / "imCoords_to_XYZ.joblib")
        self.xz_getter = load(self.python_calibration_folder / "xy_yz_to_xz.joblib")
        self.xy_getter = load(self.python_calibration_folder / "xz_yz_to_xy.joblib")
        self.yz_getter = load(self.python_calibration_folder / "xz_xy_to_yz.joblib")

    def compute_XYZ_from_imageCoord_triplet(self, imCoords: np.ndarray) -> np.ndarray:
        """
        从三相机 2D 坐标预测 3D XYZ。

        Args:
            imCoords: shape (3, 2)，顺序 [cam0(top/XY), cam1(side1/XZ), cam2(side2/YZ)]

        Returns:
            XYZ: shape (3,)
        """
        flat = imCoords.reshape(-1, 6)
        return self.xyz_getter.predict(flat).ravel()

    def compute_XZ_imcoords_from_XY_YZ(self, xy_imCoord: np.ndarray, yz_imCoord: np.ndarray) -> np.ndarray:
        """从 XY 和 YZ 视图推导 XZ 视图坐标"""
        inp = np.hstack((xy_imCoord, yz_imCoord)).reshape(1, 4)
        return self.xz_getter.predict(inp).ravel()

    def compute_XY_imcoords_from_XZ_YZ(self, xz_imCoord: np.ndarray, yz_imCoord: np.ndarray) -> np.ndarray:
        """从 XZ 和 YZ 视图推导 XY 视图坐标"""
        inp = np.hstack((xz_imCoord, yz_imCoord)).reshape(1, 4)
        return self.xy_getter.predict(inp).ravel()

    def compute_YZ_imcoords_from_XZ_XY(self, xz_imCoord: np.ndarray, xy_imCoord: np.ndarray) -> np.ndarray:
        """从 XZ 和 XY 视图推导 YZ 视图坐标"""
        inp = np.hstack((xz_imCoord, xy_imCoord)).reshape(1, 4)
        return self.yz_getter.predict(inp).ravel()

    def compute_point_correspondence_error(
        self,
        camIdxs: list[int],
        imCoords_cam1: np.ndarray,
        imCoords_cam2: np.ndarray,
    ) -> float:
        """
        计算跨相机配准误差（参考 Dynamics_of_dominance tracking.py）。

        用两个相机的 2D 坐标推导第三相机坐标，检查三视图自洽性。
        误差 = 三个推导误差之和。

        Args:
            camIdxs: [0,1], [0,2], 或 [1,2]，表示 imCoords_cam1/cam2 来自哪两个相机
            imCoords_cam1: 相机1的 (x,y)
            imCoords_cam2: 相机2的 (x,y)

        Returns:
            标量误差，NaN 表示无效
        """
        if np.any(np.isnan(imCoords_cam1)) or np.any(np.isnan(imCoords_cam2)):
            return np.nan

        # 辅助模型：xy_yz_to_xz(XY,YZ)->XZ, xz_yz_to_xy(XZ,YZ)->XY, xz_xy_to_yz(XZ,XY)->YZ
        # camIdxs [0,1]: 有 XY 和 XZ，求 YZ = xz_xy_to_yz(XZ, XY)
        # camIdxs [0,2]: 有 XY 和 YZ，求 XZ = xy_yz_to_xz(XY, YZ)
        # camIdxs [1,2]: 有 XZ 和 YZ，求 XY = xz_yz_to_xy(XZ, YZ)
        if camIdxs == [0, 1]:
            xy, xz = imCoords_cam1, imCoords_cam2
            yz = self.compute_YZ_imcoords_from_XZ_XY(xz, xy)
            proposed = np.vstack((xz, xy, yz))  # [XZ, XY, YZ] 供后续误差计算
        elif camIdxs == [0, 2]:
            xy, yz = imCoords_cam1, imCoords_cam2
            xz = self.compute_XZ_imcoords_from_XY_YZ(xy, yz)
            proposed = np.vstack((xz, xy, yz))
        elif camIdxs == [1, 2]:
            xz, yz = imCoords_cam1, imCoords_cam2
            xy = self.compute_XY_imcoords_from_XZ_YZ(xz, yz)
            proposed = np.vstack((xz, xy, yz))
        else:
            raise ValueError(f"camIdxs 必须为 [0,1], [0,2] 或 [1,2]，得到 {camIdxs}")

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
        """
        给定完整三相机 2D 坐标 triplet，计算配准误差。

        Args:
            imCoords: shape (3, 2)，[cam0, cam1, cam2]

        Returns:
            标量误差
        """
        if np.any(np.isnan(imCoords)):
            return np.nan
        err_01 = self.compute_point_correspondence_error([0, 1], imCoords[0], imCoords[1])
        return err_01
