"""
Step 4: 3D 标定质量可视化

支持多种输入：
- 模板 .txt 文件（way1/way2/way3/way4）：按列着色（Pink/Yellow/Blue/Black/Orange/Purple/Silver）
- point_mapping CSV：模板 vs 模型预测
- 可同时叠加多个模板或多种预测结果

用法示例：
    # 仅可视化模板（way1）
    python scripts/calibration/qc_3d_template.py --template_txt "H:/calibration_raw/cali_20251217/cali_coords_template_way1.txt" --output qc_way1.png

    # 可视化 point_mapping（模板 vs 预测）
    python scripts/calibration/qc_3d_template.py --mapping_csv "path/to/point_mapping.csv" --template_txt "path/to/cali_coords_template_way1.txt"

    # 多 way 模板一起显示
    python scripts/calibration/qc_3d_template.py --template_dir "H:/calibration_raw/cali_20251217" --ways way1 way2 way3 way4 --output qc_all_ways.png

    # 多个 mapping CSV 叠加（不同模型预测）
    python scripts/calibration/qc_3d_template.py --mapping_csv a.csv --additional_mappings b.csv c.csv --output qc_compare.png
"""
import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# 与 Step 5b 一致的珠子颜色：前 4 点为顶点绿色，其余按列 Pink/Yellow/Blue/Black/Orange/Purple/Silver
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


# 兼容两种调用方式：
# - 作为包模块: python -m scripts.calibration.qc_3d_template
# - 直接脚本路径: python scripts/calibration/qc_3d_template.py
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from visualize_coords_from_template import (  # type: ignore
    get_colors_from_template,
)


def load_template_txt(path: str) -> Tuple[np.ndarray, List[str]]:
    """从模板 .txt 加载 3D 坐标及颜色。返回 (coords Nx3, colors)."""
    return get_colors_from_template(path)


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


def find_template_in_dir(template_dir: str, way: str) -> Optional[str]:
    """在目录中查找指定 way 的模板文件。"""
    way = way.lower().replace("way", "way")
    patterns = [
        os.path.join(template_dir, f"cali_coords_template_{way}.txt"),
        os.path.join(template_dir, f"cali_coords_template_{way.upper()}.txt"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(template_dir, f"*coords*template*{way}*.txt"))
    return matches[0] if matches else None


def plot_qc_3d(
    template: np.ndarray,
    pred: Optional[np.ndarray] = None,
    colors: Optional[List[str]] = None,
    template_path: Optional[str] = None,
    way: str = "way1",
    out_path: str = "qc_3d.png",
    template_name: Optional[str] = None,
) -> None:
    """生成 4 小图：3D + XY + XZ + YZ，模板 vs 预测（若提供）。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if colors is None:
        colors = [get_point_color(i) for i in range(len(template))]

    name = template_name or "Template"
    pts = pred if pred is not None else template

    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 3D 视图（template 与 pred 同色，template 加黑框区分）
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_title(f"{name} - 3D view (template vs predicted)")
    ax3d.scatter(
        template[:, 0], template[:, 1], template[:, 2],
        c=colors[: len(template)], s=20, alpha=0.8, edgecolors="k", linewidths=1.2,
    )
    ax3d.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=colors[: len(pts)], s=30, alpha=0.8, edgecolors="k", linewidths=0.2,
    )
    ax3d.scatter([0], [0], [0], c="red", s=120, marker="*", zorder=1000)
    ax3d.invert_yaxis()
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")

    # XY 视角（template 与 pred 同色，template 加黑框）
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.set_title(f"{name} - XY view")
    ax_xy.scatter(template[:, 0], template[:, 1], c=colors[: len(template)], s=12, alpha=0.8, edgecolors="k", linewidths=1.2)
    ax_xy.scatter(pts[:, 0], pts[:, 1], c=colors[: len(pts)], s=15, alpha=0.8, edgecolors="k", linewidths=0.2)
    ax_xy.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.invert_xaxis()

    # XZ 视角（template 与 pred 同色，template 加黑框）
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_xz.set_title(f"{name} - XZ view")
    ax_xz.scatter(template[:, 0], template[:, 2], c=colors[: len(template)], s=12, alpha=0.8, edgecolors="k", linewidths=1.2)
    ax_xz.scatter(pts[:, 0], pts[:, 2], c=colors[: len(pts)], s=15, alpha=0.8, edgecolors="k", linewidths=0.2)
    ax_xz.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")

    # YZ 视角（template 与 pred 同色，template 加黑框）
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_yz.set_title(f"{name} - YZ view")
    ax_yz.scatter(template[:, 1], template[:, 2], c=colors[: len(template)], s=12, alpha=0.8, edgecolors="k", linewidths=1.2)
    ax_yz.scatter(pts[:, 1], pts[:, 2], c=colors[: len(pts)], s=15, alpha=0.8, edgecolors="k", linewidths=0.2)
    ax_yz.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.invert_xaxis()

    # 图例只画一次：template = 同色 + 黑框，predicted = 同色 + 细边
    template_proxy = mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor="green",
        markeredgecolor="k", markeredgewidth=1.2, markersize=8, label="template",
    )
    pred_proxy = mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor="green",
        markeredgecolor="k", markeredgewidth=0.2, markersize=8,
        label="predicted" if pred is not None else "template (colored)",
    )
    origin_proxy = mlines.Line2D(
        [], [], color="none", marker="*", markerfacecolor="red",
        markersize=12, label="Origin",
    )
    fig.legend(
        handles=[template_proxy, pred_proxy, origin_proxy],
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02),
        frameon=True, fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_qc_3d_multi(
    layers: List[Tuple[np.ndarray, Optional[np.ndarray], List[str], str]],
    out_path: str,
    template_name: Optional[str] = None,
) -> None:
    """叠加多个数据层（如多个 way 模板、多个预测）绘制。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    name = template_name or "Template"

    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 3D
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.set_title(f"{name} - 3D view (multiple)")
    for i, (template, pred, colors, label) in enumerate(layers):
        pts = pred if pred is not None else template
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[: len(pts)], s=25, alpha=0.6, edgecolors="k", linewidths=0.2)
    ax3d.scatter([0], [0], [0], c="red", s=120, marker="*", zorder=1000)
    ax3d.invert_yaxis()
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")

    # XY
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.set_title(f"{name} - XY view (multiple)")
    for i, (template, pred, colors, label) in enumerate(layers):
        pts = pred if pred is not None else template
        ax_xy.scatter(pts[:, 0], pts[:, 1], c=colors[: len(pts)], s=12, alpha=0.6, edgecolors="k", linewidths=0.2)
    ax_xy.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_xy.invert_xaxis()
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")

    # XZ
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_xz.set_title(f"{name} - XZ view (multiple)")
    for i, (template, pred, colors, label) in enumerate(layers):
        pts = pred if pred is not None else template
        ax_xz.scatter(pts[:, 0], pts[:, 2], c=colors[: len(pts)], s=12, alpha=0.6, edgecolors="k", linewidths=0.2)
    ax_xz.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")

    # YZ
    ax_yz = fig.add_subplot(gs[1, 1])
    ax_yz.set_title(f"{name} - YZ view (multiple)")
    for i, (template, pred, colors, label) in enumerate(layers):
        pts = pred if pred is not None else template
        ax_yz.scatter(pts[:, 1], pts[:, 2], c=colors[: len(pts)], s=12, alpha=0.6, edgecolors="k", linewidths=0.2)
    ax_yz.scatter([0], [0], c="red", s=80, marker="*", zorder=1000)
    ax_yz.invert_xaxis()
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")

    # 图例只画一次（与 5b 一致）
    points_proxy = mlines.Line2D(
        [], [], color="none", marker="o", markerfacecolor="green",
        markeredgecolor="k", markeredgewidth=0.2, markersize=8, label="points",
    )
    origin_proxy = mlines.Line2D(
        [], [], color="none", marker="*", markerfacecolor="red",
        markersize=12, label="Origin",
    )
    fig.legend(
        handles=[points_proxy, origin_proxy],
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02),
        frameon=True, fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4: Visualize calibration object in 3D (template vs predicted, support multi-way).",
    )
    parser.add_argument(
        "--mapping_csv",
        type=str,
        default=None,
        help="Path to point_mapping.csv (template + pred).",
    )
    parser.add_argument(
        "--additional_mappings",
        type=str,
        default=None,
        help="Comma-separated paths to extra mapping CSVs (e.g. different model predictions).",
    )
    parser.add_argument(
        "--template_txt",
        type=str,
        default=None,
        help="Path to cali_coords_template_wayN.txt for color mapping (or template-only mode).",
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        default=None,
        help="Directory containing cali_coords_template_way1.txt etc. Use with --ways.",
    )
    parser.add_argument(
        "--ways",
        type=str,
        nargs="+",
        default=None,
        help="Way names to load from template_dir, e.g. way1 way2 way3 way4.",
    )
    parser.add_argument(
        "--way",
        type=str,
        default="way1",
        help="When using mapping_csv without template_txt, infer colors by this way (way1/way2/way3/way4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    layers: List[Tuple[np.ndarray, Optional[np.ndarray], List[str], str]] = []
    out_path = args.output

    # 模式 1: 多个 way 模板（仅 template_dir + ways）
    if args.template_dir and args.ways and not args.mapping_csv:
        template_dir = os.path.normpath(args.template_dir)
        template_name = None
        for w in args.ways:
            path = find_template_in_dir(template_dir, w)
            if path:
                coords, _ = load_template_txt(path)
                colors = [get_point_color(i) for i in range(len(coords))]
                layers.append((coords, None, colors, w))
                if template_name is None:
                    template_name = os.path.splitext(os.path.basename(path))[0]
        if not layers:
            raise FileNotFoundError(f"No template files found in {template_dir} for {args.ways}")
        if not out_path:
            out_path = os.path.join(template_dir, "qc_3d_all_ways.png")
        plot_qc_3d_multi(layers, out_path, template_name=template_name)
        print(f"Saved multi-way figure to: {out_path}")
        return 0

    # 模式 2: 仅 template_txt（无 mapping）
    if args.template_txt and not args.mapping_csv:
        path = os.path.normpath(args.template_txt)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        coords, _ = load_template_txt(path)
        if not out_path:
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(os.path.dirname(path) or ".", f"{base}_qc_3d.png")
        template_name = os.path.splitext(os.path.basename(path))[0]
        plot_qc_3d(coords, pred=None, colors=None, out_path=out_path, template_name=template_name)
        print(f"Saved template-only figure to: {out_path}")
        return 0

    # 模式 3: mapping_csv（含可选 additional_mappings）
    if not args.mapping_csv:
        parser.error("Need --mapping_csv, or --template_txt for template-only, or --template_dir + --ways")
    mapping_csv = os.path.normpath(args.mapping_csv)
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(mapping_csv)

    template, pred = load_mapping_csv(mapping_csv)
    template_path = os.path.normpath(args.template_txt) if args.template_txt else None
    colors = [get_point_color(i) for i in range(len(template))]

    # 模板名字：优先用 template_txt 文件名，否则从 mapping_csv 推断
    template_name = None
    if template_path:
        template_name = os.path.splitext(os.path.basename(template_path))[0]
    else:
        base = os.path.splitext(os.path.basename(mapping_csv))[0]
        template_name = base.replace("_point_mapping", "")

    # 单个 mapping
    if not args.additional_mappings:
        if not out_path:
            base_dir = os.path.dirname(mapping_csv)
            base_name = os.path.splitext(os.path.basename(mapping_csv))[0]
            out_path = os.path.join(base_dir, f"{base_name}_3d_qc.png")
        plot_qc_3d(
            template,
            pred,
            colors,
            template_path=template_path,
            way=args.way,
            out_path=out_path,
            template_name=template_name,
        )
        print(f"Saved 3D QC figure to: {out_path}")
        return 0

    # 多个 mapping 叠加
    layers.append((template, pred, colors, os.path.basename(mapping_csv)))
    for p in args.additional_mappings.strip().split(","):
        p = p.strip()
        if not p or not os.path.exists(p):
            continue
        t, pr = load_mapping_csv(p)
        c = [get_point_color(i) for i in range(len(t))]
        layers.append((t, pr, c, os.path.basename(p)))
    if not out_path:
        base_dir = os.path.dirname(mapping_csv)
        base_name = os.path.splitext(os.path.basename(mapping_csv))[0]
        out_path = os.path.join(base_dir, f"{base_name}_3d_qc_compare.png")
    plot_qc_3d_multi(layers, out_path, template_name=template_name)
    print(f"Saved compare figure to: {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
