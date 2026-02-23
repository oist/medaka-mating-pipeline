"""
Generate point_mapping CSV using trained calibration model
==========================================================

根据已训练好的 imCoords_to_XYZ 模型，对手动标注的 2D 坐标进行 3D 预测，
并生成 point_mapping.csv（供 Step 4 / Step 5 可视化使用）。

用法示例（单个 way）：
    python scripts/calibration/generate_point_mapping.py ^
        --calibration_folder "H:\\calibration_raw\\cali_20251217" ^
        --image_coords_folder "H:\\calibration_marking\\cali_251217_way1" ^
        --model_folder "H:\\calibration_model\\cali_251217_way1" ^
        --way_name way1 ^
        --results_root "H:\\calibration_qc\\cali_251217_way1"
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
from joblib import load

# 添加 lib 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from lib.calibration_core import (  # type: ignore
    load_calibration_data,
    _ensure_project_paths,
    _save_point_mapping_and_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate <calib_name>_point_mapping.csv using a trained imCoords_to_XYZ model.\n"
            "This script is intended to be run AFTER Step 3 (solve_intrinsics.py)."
        )
    )
    parser.add_argument(
        "--calibration_folder",
        type=str,
        required=True,
        help="Calibration raw folder containing 3D template (same as Step 3).",
    )
    parser.add_argument(
        "--image_coords_folder",
        type=str,
        required=True,
        help="Folder containing *_image_coords.txt (manual 2D annotations).",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Folder containing imCoords_to_XYZ.joblib (output of Step 3).",
    )
    parser.add_argument(
        "--way_name",
        type=str,
        default="way1",
        help="Way name (e.g. way1 / way2 / way3 / way4). Used to select template & image coords.",
    )
    parser.add_argument(
        "--beads_only",
        action="store_true",
        help="Use only BEADS_WAY* points (ignore CORNERS/PILLARS/RED_LINES), consistent with Step 3 if used.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Output root for QC CSV/figures. "
            "If omitted, will be inferred from calibration_folder (same logic as solve_intrinsics)."
        ),
    )
    args = parser.parse_args()

    calibration_folder = os.path.normpath(args.calibration_folder)
    image_coords_folder = os.path.normpath(args.image_coords_folder)
    model_folder = os.path.normpath(args.model_folder)
    results_root: Optional[str] = (
        os.path.normpath(args.results_root) if args.results_root is not None else None
    )

    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"model_folder not found: {model_folder}")

    model_path = os.path.join(model_folder, "imCoords_to_XYZ.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"imCoords_to_XYZ.joblib not found in {model_folder}")

    print("Calibration folder:", calibration_folder)
    print("Image coords folder:", image_coords_folder)
    print("Model folder:", model_folder)
    print("Way name:", args.way_name)

    # 加载 3D 模板与 2D 标注（顺序与训练一致）
    template_3d, image_coords, _, _ = load_calibration_data(
        calibration_folder=calibration_folder,
        image_coords_folder=image_coords_folder,
        filename_keyword=args.way_name,
        way_name=args.way_name,
        beads_only=args.beads_only,
    )

    if template_3d.size == 0 or image_coords.size == 0:
        raise ValueError("Empty template_3d or image_coords; please check input folders and way_name.")

    # 加载模型并进行 3D 预测
    print("\nLoading model from:", model_path)
    model = load(model_path)

    n_points = template_3d.shape[0]
    X_feat = image_coords.swapaxes(0, 1).reshape(n_points, -1)
    print(f"Predicting 3D coordinates for {n_points} points ...")
    pred_3d = model.predict(X_feat)

    # 确定 results_root & calib_name
    paths = _ensure_project_paths(
        calibration_folder,
        output_folder=model_folder,
        results_root_override=results_root,
    )
    os.makedirs(paths["results_root"], exist_ok=True)

    # 保存 point_mapping.csv
    print("\nSaving point_mapping CSV ...")
    _save_point_mapping_and_figures(
        template_3d,
        image_coords,
        pred_3d,
        paths["results_root"],
        paths["calib_name"],
    )

    csv_path = os.path.join(
        paths["results_root"], f"{paths['calib_name']}_point_mapping.csv"
    )
    print("\nDone.")
    print("point_mapping CSV saved to:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

