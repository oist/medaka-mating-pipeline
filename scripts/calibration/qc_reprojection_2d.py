"""
Step 5b: 2D 重投影误差 QC（各相机视角下的像素误差）

校准 QC 应看的是「2D 视角的差」：把 3D（模板或预测）投影回各相机图像，
与手动标注的 2D 坐标比较，得到像素级重投影误差（含畸变在内的图像空间一致性）。

本脚本做法：
1. 用 point_mapping 中的 (X,Y,Z) 与 (cam*_x, cam*_y) 训练每相机的 3D→2D 回归
   （等价于学「该相机把 3D 投到图像平面」的映射，含畸变，因为标注在原始图像上）。
2. 用预测 3D (pred_X, pred_Y, pred_Z) 经上述模型得到「重投影 2D」。
3. 重投影误差 = ||手动 2D - 重投影 2D||（像素），分相机统计并可视化。

用法:
    python scripts/calibration/qc_reprojection_2d.py --mapping_csv "path/to/point_mapping.csv" --output "path/to/reprojection_qc.png"
"""
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump

# First 4 points = corners (vertices); rest = beads (7 per column)
N_CORNERS = 4
BEADS_PER_COLUMN = 7
BEAD_COLUMN_COLOR_ORDER = ["pink", "yellow", "blue", "black", "orange", "purple", "silver"]


def get_point_color(i: int) -> str:
    """Corners (0-3) = green; beads (4+) = column color by index."""
    if i < N_CORNERS:
        return "green"
    bead_idx = i - N_CORNERS
    col_idx = bead_idx // BEADS_PER_COLUMN
    if col_idx >= len(BEAD_COLUMN_COLOR_ORDER):
        return BEAD_COLUMN_COLOR_ORDER[-1]
    return BEAD_COLUMN_COLOR_ORDER[col_idx]


def load_mapping_csv(path: str) -> pd.DataFrame:
    """读取 point_mapping CSV，需包含 X,Y,Z, pred_*, cam*_x, cam*_y。"""
    df = pd.read_csv(path)
    required = ["X", "Y", "Z", "pred_X", "pred_Y", "pred_Z",
                "cam0_x", "cam0_y", "cam1_x", "cam1_y", "cam2_x", "cam2_y"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {path}")
    return df


def train_xyz_to_uv(xyz: np.ndarray, uv: np.ndarray, degree: int = 2, alpha: float = 1e-3):
    """训练 3D→2D 回归：(N,3) xyz -> (N,2) uv。"""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("reg", MultiOutputRegressor(Ridge(alpha=alpha))),
    ])
    pipe.fit(xyz, uv)
    return pipe


def train_xyz_to_imcoords(
    df: pd.DataFrame,
    degree: int = 2,
    alpha: float = 1e-3,
) -> Pipeline:
    """
    训练单模型 3D→三相机 2D：(N,3) XYZ -> (N,6) [cam0_x, cam0_y, cam1_x, cam1_y, cam2_x, cam2_y]。
    用于保存为 XYZ_to_imCoords.joblib，供 Calibration.compute_imCoords_from_XYZ 与 2D 叠加使用。
    """
    xyz = df[["X", "Y", "Z"]].to_numpy(float)
    cols = ["cam0_x", "cam0_y", "cam1_x", "cam1_y", "cam2_x", "cam2_y"]
    imcoords = df[cols].to_numpy(float)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("reg", MultiOutputRegressor(Ridge(alpha=alpha))),
    ])
    pipe.fit(xyz, imcoords)
    return pipe


def reprojection_errors_per_camera(
    df: pd.DataFrame,
    degree: int = 2,
    alpha: float = 1e-3,
) -> Tuple[dict, dict, dict]:
    """
    为每个相机训练 3D→2D，并计算重投影误差（像素）。
    使用模板 3D (X,Y,Z) 与手动 2D 训练；用预测 3D 做重投影与手动 2D 比较。

    Returns
    -------
    models : dict
        {"cam0": pipeline, "cam1": ..., "cam2": ...}
    errors : dict
        {"cam0": array of shape (n_points,), "cam1": ..., "cam2": ...}
        每个元素为各点的 ||手动2D - 重投影2D||（像素）
    reproj_uv : dict
        {"cam0": (N,2), "cam1": ..., "cam2": ...} 预测 3D 重投影到各相机的 2D 坐标
    """
    xyz_tpl = df[["X", "Y", "Z"]].to_numpy(float)
    xyz_pred = df[["pred_X", "pred_Y", "pred_Z"]].to_numpy(float)

    models = {}
    errors = {}
    reproj_uv = {}

    for i, cam in enumerate(["cam0", "cam1", "cam2"]):
        u_col, v_col = f"{cam}_x", f"{cam}_y"
        uv = df[[u_col, v_col]].to_numpy(float)
        # 用模板 3D + 手动 2D 训练（学的是「真实 3D → 该相机图像」的映射，含畸变）
        pipe = train_xyz_to_uv(xyz_tpl, uv, degree=degree, alpha=alpha)
        models[cam] = pipe
        # 预测 3D 重投影到该相机
        uv_reproj = pipe.predict(xyz_pred)
        reproj_uv[cam] = uv_reproj
        diff = uv - uv_reproj
        err = np.linalg.norm(diff, axis=1)
        errors[cam] = err

    return models, errors, reproj_uv


def plot_reprojection_2d_views(
    df: pd.DataFrame,
    reproj_uv: dict,
    models: dict,
    out_path: str,
    draw_segments: bool = True,
    xy_max: float = 1200.0,
) -> None:
    """
    三张 2D 小图 = Step 4 小图 2(XY)、3(XZ)、4(YZ) 仅替换数据源为各相机 2D。
    Step 4 原点 (0,0) 在图中位置：XY 右下、XZ 左下、YZ 右下。像素 y 向下，故对数据做变换，
    使投影原点在 5b 里也落在右下、左下、右下。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_pts = len(df)
    colors = [get_point_color(i) for i in range(n_pts)]
    y_max = xy_max
    x_max = xy_max
    axis_min, axis_max = -10.0, 1250.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # -------------------------------------------------------------------------
    # 小图 1：Step 4 XY → 原点在右下。像素 y 向下，故用 (x, y_max - y) 使图像下方→图下方；
    # invert_xaxis 使 0 在右 → 原点在右下。
    # -------------------------------------------------------------------------
    ax_xy = axes[0]
    # 原始像素（cam0）
    tx = df["cam0_x"].to_numpy(float)
    ty = df["cam0_y"].to_numpy(float)
    px = reproj_uv["cam0"][:, 0]
    py = reproj_uv["cam0"][:, 1]
    # 原点：图中 (0,0) 位置 = Step 4 的右下（invert_x 后 0 在右、0 在下）
    template_xy_x = x_max - tx
    template_xy_y = y_max - ty
    pred_xy_x = x_max - px
    pred_xy_y = y_max - py
    origin_xy_x = 0.0
    origin_xy_y = 0.0
    if draw_segments:
        for j in range(n_pts):
            ax_xy.plot(
                [template_xy_x[j], pred_xy_x[j]],
                [template_xy_y[j], pred_xy_y[j]],
                color="gray", alpha=0.5, linewidth=0.8, zorder=0,
            )
    ax_xy.set_title("cam0- top - XY view")
    ax_xy.scatter(
        template_xy_x, template_xy_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=1.2, zorder=2,
    )
    ax_xy.scatter(
        pred_xy_x, pred_xy_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=0.2, zorder=2,
    )
    ax_xy.scatter([origin_xy_x], [origin_xy_y], c="red", s=80, marker="*", zorder=1000)
    ax_xy.set_xlabel("x (pixels)")
    ax_xy.set_ylabel("y (pixels)")
    ax_xy.set_xlim(axis_min, axis_max)
    ax_xy.set_ylim(axis_min, axis_max)
    ax_xy.invert_xaxis()

    # -------------------------------------------------------------------------
    # 小图 2：Step 4 XZ → 原点在左下。不翻 x（用 tx）避免左右颠倒；y 用 (y_max - y) 使原点到左下。
    # -------------------------------------------------------------------------
    ax_xz = axes[1]
    tx = df["cam1_x"].to_numpy(float)
    ty = df["cam1_y"].to_numpy(float)
    px = reproj_uv["cam1"][:, 0]
    py = reproj_uv["cam1"][:, 1]
    # 原点：图中 (0,0) 位置 = Step 4 的左下（不 invert 时 0 在左、0 在下）
    template_xz_x = tx
    template_xz_y = y_max - ty
    pred_xz_x = px
    pred_xz_y = y_max - py
    origin_xz_x = 0.0
    origin_xz_y = 0.0
    if draw_segments:
        for j in range(n_pts):
            ax_xz.plot(
                [template_xz_x[j], pred_xz_x[j]],
                [template_xz_y[j], pred_xz_y[j]],
                color="gray", alpha=0.5, linewidth=0.8, zorder=0,
            )
    ax_xz.set_title("cam1- side1 - XZ view")
    ax_xz.scatter(
        template_xz_x, template_xz_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=1.2, zorder=2,
    )
    ax_xz.scatter(
        pred_xz_x, pred_xz_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=0.2, zorder=2,
    )
    ax_xz.scatter([origin_xz_x], [origin_xz_y], c="red", s=80, marker="*", zorder=1000)
    ax_xz.set_xlabel("x (pixels)")
    ax_xz.set_ylabel("z (pixels)")
    ax_xz.set_xlim(axis_min, axis_max)
    ax_xz.set_ylim(axis_min, axis_max)

    # -------------------------------------------------------------------------
    # 小图 3：Step 4 YZ → 原点在右下。同小图 1：(x, y_max - y) + invert_xaxis。
    # -------------------------------------------------------------------------
    ax_yz = axes[2]
    tx = df["cam2_x"].to_numpy(float)
    ty = df["cam2_y"].to_numpy(float)
    px = reproj_uv["cam2"][:, 0]
    py = reproj_uv["cam2"][:, 1]
    # 原点：图中 (0,0) 位置 = Step 4 的右下（invert_x 后 0 在右、0 在下）
    template_yz_x = x_max - tx
    template_yz_y = y_max - ty
    pred_yz_x = x_max - px
    pred_yz_y = y_max - py
    origin_yz_x = 0.0
    origin_yz_y = 0.0
    if draw_segments:
        for j in range(n_pts):
            ax_yz.plot(
                [template_yz_x[j], pred_yz_x[j]],
                [template_yz_y[j], pred_yz_y[j]],
                color="gray", alpha=0.5, linewidth=0.8, zorder=0,
            )
    ax_yz.set_title("cam2- side2 - YZ view")
    ax_yz.scatter(
        template_yz_x, template_yz_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=1.2, zorder=2,
    )
    ax_yz.scatter(
        pred_yz_x, pred_yz_y,
        c=colors[:n_pts], s=15, alpha=0.8,
        edgecolors="k", linewidths=0.2, zorder=2,
    )
    ax_yz.scatter([origin_yz_x], [origin_yz_y], c="red", s=80, marker="*", zorder=1000)
    ax_yz.set_xlabel("y (pixels)")
    ax_yz.set_ylabel("z (pixels)")
    ax_yz.set_xlim(axis_min, axis_max)
    ax_yz.set_ylim(axis_min, axis_max)
    ax_yz.invert_xaxis()

    # 图例单独画一次：manual / predicted 用粉色，Origin = 像素 (0,0)
    manual_proxy = mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor="hotpink",
        markeredgecolor="k", markeredgewidth=1.2, markersize=8, label="manual 2D",
    )
    pred_proxy = mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor="hotpink",
        markeredgecolor="k", markeredgewidth=0.2, markersize=8, label="predicted 2D",
    )
    origin_proxy = mlines.Line2D(
        [], [], color="none", marker="*", markerfacecolor="red",
        markersize=12, label="Origin (pixel 0,0)",
    )
    fig.legend(
        handles=[manual_proxy, pred_proxy, origin_proxy],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        fontsize=10,
    )

    fig.suptitle("2D manual vs predicted values", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5b: 2D reprojection QC (pixel error per camera).",
    )
    parser.add_argument(
        "--mapping_csv",
        type=str,
        required=True,
        help="Path to point_mapping.csv (with X,Y,Z, pred_*, cam*_x, cam*_y).",
    )
    parser.add_argument(
        "--poly_degree",
        type=int,
        default=2,
        help="Polynomial degree for 3D→2D regression (default: 2).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Ridge alpha for 3D→2D regression (default: 1e-3).",
    )
    parser.add_argument(
        "--output_views",
        type=str,
        default=None,
        help="Optional path for the 3-panel figure (manual vs reprojected 2D per camera). Default: <output_dir>/<base>_reprojection_2d_views.png",
    )
    parser.add_argument(
        "--no_segments",
        action="store_true",
        help="Do not draw segments between manual and reprojected points in the views figure.",
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Train and save XYZ_to_imCoords.joblib to calibration folder (use with --calibration_folder).",
    )
    parser.add_argument(
        "--calibration_folder",
        type=str,
        default=None,
        help="Calibration folder (python_calibration_models or parent). Used with --save_models to write XYZ_to_imCoords.joblib.",
    )
    args = parser.parse_args()

    mapping_csv = os.path.normpath(args.mapping_csv)
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(mapping_csv)

    df = load_mapping_csv(mapping_csv)

    if args.save_models and args.calibration_folder:
        cal_dir = os.path.normpath(args.calibration_folder)
        out_dir = os.path.join(cal_dir, "python_calibration_models")
        if not os.path.isdir(out_dir):
            out_dir = cal_dir
        os.makedirs(out_dir, exist_ok=True)
        pipe = train_xyz_to_imcoords(df, degree=args.poly_degree, alpha=args.alpha)
        path = os.path.join(out_dir, "XYZ_to_imCoords.joblib")
        dump(pipe, path)
        print(f"Saved XYZ_to_imCoords.joblib to {path}")
    models, errors, reproj_uv = reprojection_errors_per_camera(
        df, degree=args.poly_degree, alpha=args.alpha
    )

    # 输出 3 张小图：每相机 2D 手动 vs 重投影（类似 Step 4 的 2/3/4 小图）
    out_views = args.output_views
    if out_views is None:
        base_dir = os.path.dirname(mapping_csv)
        base_name = os.path.splitext(os.path.basename(mapping_csv))[0]
        out_views = os.path.join(base_dir, f"{base_name}_reprojection_2d_views.png")
    plot_reprojection_2d_views(
        df, reproj_uv, models, out_views, draw_segments=not args.no_segments
    )
    print(f"Views figure (manual vs reprojected 2D) saved to: {out_views}")

    print("2D reprojection QC (predicted 3D → 2D vs manual 2D):")
    for cam in ["cam0", "cam1", "cam2"]:
        e = errors[cam]
        print(f"  {cam}: mean={np.mean(e):.2f} px, median={np.median(e):.2f} px, std={np.std(e):.2f} px, max={np.max(e):.2f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
