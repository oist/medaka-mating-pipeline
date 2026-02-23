import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mapping_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 point_mapping CSV，返回 (template_3d, pred_3d)."""
    df = pd.read_csv(path)
    required_cols = ["X", "Y", "Z", "pred_X", "pred_Y", "pred_Z"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {path}")

    template = df[["X", "Y", "Z"]].to_numpy(float)
    pred = df[["pred_X", "pred_Y", "pred_Z"]].to_numpy(float)
    return template, pred


def plot_error_3d(template: np.ndarray, pred: np.ndarray, out_path: str) -> None:
    """
    小图 1（单独一张图）：3D 模板点，颜色 = 模板与预测的差 ||ΔXYZ|| (mm)。
    点的位置是 3D 模板 (X,Y,Z)，颜色表示 pred - template 的欧氏距离。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    err = pred - template
    en = np.linalg.norm(err, axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    sc3d = ax3d.scatter(
        template[:, 0], template[:, 1], template[:, 2],
        c=en, cmap="viridis", s=35, alpha=0.9,
    )
    ax3d.set_title("||ΔXYZ|| (mm) — 3D template, color = error magnitude")
    ax3d.invert_yaxis()
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    fig.colorbar(sc3d, ax=ax3d, shrink=0.6, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_xy_three(template: np.ndarray, pred: np.ndarray, out_path: str) -> None:
    """
    小图 2、3、4 画在一张大图上：固定 XY 视角，三块分别为 |ΔX|、|ΔY|、|ΔZ| (mm)。
    点的位置是模板的 (X, Y)，颜色分别是 X/Y/Z 方向误差的绝对值。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    err = pred - template
    ex, ey, ez = err[:, 0], err[:, 1], err[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = ["|ΔX| (mm)", "|ΔY| (mm)", "|ΔZ| (mm)"]
    errs = [np.abs(ex), np.abs(ey), np.abs(ez)]

    for ax, title, e in zip(axes, titles, errs):
        sc = ax.scatter(
            template[:, 0], template[:, 1],
            c=e, cmap="coolwarm", s=30, alpha=0.9,
        )
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-10, 220)
        ax.set_ylim(-10, 220)
        ax.invert_xaxis()
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("Fixed XY view: X / Y / Z error components", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5: Visualize per-point calibration error (X/Y/Z).",
    )
    parser.add_argument(
        "--mapping_csv",
        type=str,
        required=True,
        help="Path to <calib_name>_point_mapping.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path for 3D figure. XY figure will be <path_stem>_xy.png",
    )
    args = parser.parse_args()

    mapping_csv = os.path.normpath(args.mapping_csv)
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(mapping_csv)

    template, pred = load_mapping_csv(mapping_csv)

    base_dir = os.path.dirname(mapping_csv)
    base_name = os.path.splitext(os.path.basename(mapping_csv))[0]
    if args.output:
        out_3d = os.path.normpath(args.output)
        base = os.path.splitext(out_3d)[0]
        out_xy = base + "_xy.png"
    else:
        out_3d = os.path.join(base_dir, f"{base_name}_error_3d.png")
        out_xy = os.path.join(base_dir, f"{base_name}_error_xy.png")

    print(f"Mapping CSV: {mapping_csv}")
    print(f"Saving Step 5 小图1 (3D) to: {out_3d}")
    plot_error_3d(template, pred, out_3d)
    print(f"Saving Step 5 小图234 (XY 视角 |ΔX|,|ΔY|,|ΔZ|) to: {out_xy}")
    plot_error_xy_three(template, pred, out_xy)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

