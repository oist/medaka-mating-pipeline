"""
Train Calibration Models (Solve Intrinsics)
============================================

训练相机校准模型，用于3D重建。
支持批量处理多个文件夹。

用法:
    # 单个文件夹
    python scripts/calibration/solve_intrinsics.py --calibration_folder "D:\\calibration_raw\\cali_251217_way1_pic" --image_coords_folder "D:\\calibration_results\\cali_20251217"

    # 多个文件夹（逗号分隔）
    python scripts/calibration/solve_intrinsics.py --calibration_folder "D:\\calibration_raw\\cali_251217_way3_pic,D:\\calibration_raw\\cali_251217_way4_pic" --image_coords_folder "D:\\calibration_results\\cali_way3,D:\\calibration_results\\cali_way4"

输出:
    models/calibration/<calib_name>/
        imCoords_to_XYZ.joblib  - 6个图像坐标 -> 3D XYZ
        xy_yz_to_xz.joblib      - cam1+cam2 -> cam0
        xz_xy_to_yz.joblib      - cam0+cam1 -> cam2
        xz_yz_to_xy.joblib      - cam0+cam2 -> cam1
"""

import argparse
import os
import sys
import glob
import re
import numpy as np
from typing import List

# 添加lib目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from lib.calibration_core import run_calibration_pipeline


def parse_paths(path_str: str) -> List[str]:
    """解析逗号分隔的路径字符串"""
    if not path_str:
        return []
    paths = [p.strip() for p in path_str.split(',')]
    return [p for p in paths if p]


def auto_detect_ways(calibration_folder: str, image_coords_folder: str) -> List[tuple]:
    """
    自动检测way并匹配对应的文件夹。
    
    返回: List[(calib_folder, coords_folder, way_name)]
    """
    results = []
    
    # 在 image_coords_folder 中查找所有 way 的标注文件
    coord_files = glob.glob(os.path.join(image_coords_folder, "*_image_coords.txt"))
    ways_found = set()
    
    for coord_file in coord_files:
        basename = os.path.basename(coord_file)
        # 匹配格式: cali_way1_cam0_image_coords.txt
        way_match = re.search(r'way(\d+)', basename, re.IGNORECASE)
        if way_match:
            way_num = way_match.group(1)
            way_name = f"way{way_num}"
            ways_found.add((way_name, int(way_num)))
    
    if not ways_found:
        raise ValueError(f"No way files found in {image_coords_folder}")
    
    # 按way数字排序
    ways_sorted = sorted(ways_found, key=lambda x: x[1])
    
    # 为每个way查找对应的calibration_folder
    for way_name, way_num in ways_sorted:
        # 尝试多种可能的calibration_folder路径
        possible_calib_folders = [
            os.path.join(calibration_folder, f"cali_*_way{way_num}"),
            os.path.join(calibration_folder, f"cali_*_way{way_num}_pic"),
            os.path.join(calibration_folder, f"*way{way_num}*"),
            calibration_folder,  # 如果只有一个文件夹，直接使用
        ]
        
        calib_folder = None
        for pattern in possible_calib_folders:
            matches = glob.glob(pattern)
            if matches:
                calib_folder = matches[0]
                break
        
        if not calib_folder:
            # 如果找不到，使用原始calibration_folder（模板文件可能在父目录）
            calib_folder = calibration_folder
        
        results.append((calib_folder, image_coords_folder, way_name))
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train calibration models for 3D reconstruction (supports batch processing)"
    )
    parser.add_argument(
        "--calibration_folder", type=str, required=True,
        help="Path to calibration folder(s) containing 3D coordinates template, comma-separated for multiple"
    )
    parser.add_argument(
        "--image_coords_folder", type=str, required=True,
        help="Folder(s) containing manually annotated image coordinate txt files, comma-separated for multiple"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="Output folder(s) for models, comma-separated. Default: models/calibration/<calib_name>/"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Root folder(s) for training-time CSV/figures (QC 输出)，逗号分隔；不指定则按 calibration_folder 自动推断",
    )
    parser.add_argument(
        "--alpha_range", type=float, nargs="+",
        default=[1e-10, 1e-7, 1e-6, 1e-4, 1e-5, 1e-3, 0.001, 0.01, 0.1, 1],
        help="Alpha values for Ridge regression grid search"
    )
    parser.add_argument(
        "--poly_deg_range", type=int, nargs="+",
        default=[1, 2, 3, 4],
        help="Polynomial degrees for grid search"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=3,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--train_fraction", type=float, default=0.9,
        help="Fraction of data for training (for validation)"
    )
    parser.add_argument(
        "--num_permutations", type=int, default=100,
        help="Number of train-test splits for validation"
    )
    parser.add_argument(
        "--beads_only", action="store_true",
        help="Use only BEADS_WAY* points for training (ignore CORNERS, PILLARS, RED_LINES)"
    )
    parser.add_argument(
        "--model_types", type=str, nargs="+", default=None,
        help="Model types to try: polynomial, random_forest, svr, mlp. Default: try all types"
    )
    parser.add_argument(
        "--merge_ways", action="store_true",
        help="Merge multiple ways into one unified model (assumes all ways use the same 3D coordinate system)"
    )
    parser.add_argument(
        "--no_auto_detect", action="store_true",
        help="Disable auto-detection of ways; treat the given folders as a single calibration set"
    )
    parser.add_argument(
        "--way_name", type=str, default=None,
        help="When using --no_auto_detect, specify which way (e.g. way1, way2, way3, way4). "
             "Script will look for cali_<way>_cam*_image_coords.txt in image_coords_folder. "
             "Required when --no_auto_detect is set and folder name does not imply the way."
    )
    parser.add_argument(
        "--way_names", type=str, nargs="+", default=None,
        help="When --merge_ways is used, specify way names to merge (e.g., way1 way2 way3 way4). "
             "If not specified, will auto-detect from image_coords_folder"
    )
    args = parser.parse_args()

    # 解析路径
    calib_folders = parse_paths(args.calibration_folder)
    coords_folders = parse_paths(args.image_coords_folder)
    output_folders = parse_paths(args.output_folder) if args.output_folder else []
    results_roots = parse_paths(args.results_root) if args.results_root else []
    
    # 如果使用合并模式，特殊处理
    if args.merge_ways:
        if len(calib_folders) != 1 or len(coords_folders) != 1:
            print("[ERROR] --merge_ways mode requires exactly one calibration_folder and one image_coords_folder")
            return
        
        print("=" * 70)
        print("Camera Calibration Pipeline - MERGE MODE")
        print("=" * 70)
        print("Merging multiple ways into one unified model")
        print("=" * 70)
        
        way_names = args.way_names
        if way_names is None:
            # 自动检测way
            coord_files = glob.glob(os.path.join(coords_folders[0], "*_image_coords.txt"))
            ways_found = set()
            for coord_file in coord_files:
                basename = os.path.basename(coord_file)
                way_match = re.search(r'way(\d+)', basename, re.IGNORECASE)
                if way_match:
                    way_num = way_match.group(1)
                    way_name = f"way{way_num}"
                    ways_found.add((way_name, int(way_num)))
            
            if not ways_found:
                print(f"[ERROR] No way files found in {coords_folders[0]}")
                return
            
            way_names = [w[0] for w in sorted(ways_found, key=lambda x: x[1])]
            print(f"Auto-detected ways: {', '.join(way_names)}")
        else:
            print(f"Using specified ways: {', '.join(way_names)}")
        
        try:
            # 合并模式：加载所有way的数据并合并
            from lib.calibration_core import load_calibration_data
            
            print(f"Loading and merging data from {len(way_names)} way(s)...")
            all_measured_coords = []
            all_image_coords = []
            
            # 使用第一个way的模板（所有way使用相同的3D坐标系统）
            template_3d_first, _, _, _ = load_calibration_data(
                calibration_folder=calib_folders[0],
                image_coords_folder=coords_folders[0],
                filename_keyword=way_names[0],
                way_name=way_names[0],
                beads_only=args.beads_only,
            )
            
            # 为每个way加载2D坐标
            for way_name in way_names:
                print(f"  Loading {way_name}...")
                try:
                    _, image_coords_way, _, _ = load_calibration_data(
                        calibration_folder=calib_folders[0],
                        image_coords_folder=coords_folders[0],
                        filename_keyword=way_name,
                        way_name=way_name,
                        beads_only=args.beads_only,
                    )
                    
                    # 合并数据（使用相同的3D坐标）
                    num_points = image_coords_way.shape[1]
                    all_measured_coords.extend(template_3d_first[:num_points])
                    # 转置为 (num_points, num_cams, 2) 然后添加到列表
                    for i in range(num_points):
                        cam_coords = [image_coords_way[j, i, :] for j in range(3)]
                        all_image_coords.append(cam_coords)
                    
                    print(f"    {way_name}: added {num_points} points")
                except Exception as e:
                    print(f"    [WARN] {way_name}: failed to load, skipping - {e}")
                    continue
            
            if not all_measured_coords:
                raise ValueError("No valid data loaded from any way")
            
            # 转换为numpy数组
            import numpy as np
            measured_coords = np.array(all_measured_coords)
            matched_image_coords = np.array(all_image_coords)  # (total_n, num_cams, 2)
            image_coords = np.transpose(matched_image_coords, (1, 0, 2))  # (num_cams, total_n, 2)
            
            print(f"\nMerged data: {len(measured_coords)} total points from {len(way_names)} way(s)")
            
            # 使用合并后的数据训练模型（需要手动调用训练流程）
            # 为了简化，我们创建一个临时函数来直接训练
            from lib.model_training import train_calibration_model, save_all_models, prepare_model_data
            from lib.calibration_core import _ensure_project_paths, _save_training_summary, _save_point_mapping_and_figures
            
            # 准备训练数据
            model_data = prepare_model_data(image_coords, measured_coords)
            
            # 训练模型
            trained_models = {}
            summary_rows = []
            all_models_results_dict = {}
            
            for model_name, (X_data, Y_data) in model_data.items():
                valid_mask = (~np.isnan(X_data).any(axis=1)) & (~np.isnan(Y_data).any(axis=1))
                X_clean = X_data[valid_mask]
                Y_clean = Y_data[valid_mask]
                
                if X_clean.shape[0] == 0:
                    continue
                
                print(f"\n  Training {model_name} with {X_clean.shape[0]} samples...")
                model, best_params, all_models_results, val_results = train_calibration_model(
                    X_clean, Y_clean,
                    alpha_range=args.alpha_range,
                    poly_deg_range=args.poly_deg_range,
                    cv_folds=args.cv_folds,
                    train_fraction=args.train_fraction,
                    num_permutations=args.num_permutations,
                    model_types=args.model_types,
                )
                
                trained_models[model_name] = model
                all_models_results_dict[model_name] = all_models_results
                
                sorted_results = sorted(all_models_results, key=lambda x: x['cv_mean_error'])
                summary_row = {
                    "model_name": model_name,
                    "num_samples": int(X_clean.shape[0]),
                    "num_features": int(X_clean.shape[1]),
                    "output_dim": int(Y_clean.shape[1]),
                    "best_model_type": best_params.get("model_type", "polynomial"),
                    "cv_mean_error": float(sorted_results[0]["cv_mean_error"]) if sorted_results else 0.0,
                    "cv_std_error": float(sorted_results[0]["cv_std_error"]) if sorted_results else 0.0,
                    "mean_train_error": float(val_results["mean_train_error"]),
                    "std_train_error": float(val_results["std_train_error"]),
                    "mean_test_error": float(val_results["mean_test_error"]),
                    "std_test_error": float(val_results["std_test_error"]),
                }
                if "alpha" in best_params:
                    summary_row["best_alpha"] = best_params["alpha"]
                if "degree" in best_params:
                    summary_row["best_degree"] = best_params["degree"]
                summary_rows.append(summary_row)
            
            # 保存模型
            paths = _ensure_project_paths(calib_folders[0], output_folders[0] if output_folders else None)
            os.makedirs(paths["model_output_folder"], exist_ok=True)
            save_all_models(trained_models, paths["model_output_folder"])
            
            # 保存摘要
            os.makedirs(paths["results_root"], exist_ok=True)
            way_suffix = "_".join(way_names)
            summary_path = os.path.join(paths["results_root"], f"{paths['calib_name']}_merged_{way_suffix}_summary.csv")
            _save_training_summary(summary_rows, summary_path)
            
            print(f"\nMerged model saved to: {paths['model_output_folder']}")
        except Exception as e:
            print(f"[ERROR] Failed to process merged calibration: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("Merged calibration completed!")
        print("=" * 70)
        return
    
    # 智能匹配：如果只有一个calibration_folder和一个image_coords_folder
    # 自动检测way并匹配（除非显式关闭）
    way_configs = None
    if (
        len(calib_folders) == 1
        and len(coords_folders) == 1
        and not args.merge_ways
        and not args.no_auto_detect
    ):
        print("[INFO] Auto-detecting ways from image coords folder...")
        try:
            way_configs = auto_detect_ways(calib_folders[0], coords_folders[0])
            way_names = [w[2] for w in way_configs]
            print(f"[INFO] Detected ways: {', '.join(way_names)}")
            
            # 重新组织数据
            calib_folders = [w[0] for w in way_configs]
            coords_folders = [w[1] for w in way_configs]
            
            # 处理output_folder
            if len(output_folders) == 1:
                # 如果只提供一个output_folder，为每个way创建子文件夹
                base_output = output_folders[0]
                output_folders = [os.path.join(base_output, f"cali_{way_name}") for way_name in way_names]
                print(f"[INFO] Auto-generating output folders:")
                for way_name, out_folder in zip(way_names, output_folders):
                    print(f"  {way_name}: {out_folder}")
            elif len(output_folders) > 1:
                # 如果提供多个output_folder，检查数量是否匹配
                if len(way_configs) != len(output_folders):
                    print(f"[ERROR] Detected {len(way_configs)} way(s) but {len(output_folders)} output folder(s) provided")
                    print(f"  Detected ways: {way_names}")
                    return
            else:
                # 如果没有提供output_folder，使用默认值（None，让calibration_core自动生成）
                output_folders = [None] * len(way_configs)
        except Exception as e:
            print(f"[ERROR] Failed to auto-detect ways: {e}")
            return
    else:
        # 标准模式：要求数量匹配
        if len(calib_folders) != len(coords_folders):
            print(f"[ERROR] Number of calibration folders ({len(calib_folders)}) must match image coords folders ({len(coords_folders)})")
            return
        
        if args.output_folder and len(output_folders) != len(calib_folders):
            print(f"[ERROR] Number of output folders ({len(output_folders)}) must match calibration folders ({len(calib_folders)})")
            return
        if args.results_root and len(results_roots) != len(calib_folders):
            print(f"[ERROR] Number of results_root paths ({len(results_roots)}) must match calibration folders ({len(calib_folders)})")
            return
        
        if not args.output_folder:
            output_folders = [None] * len(calib_folders)
    
    print("=" * 70)
    print("Camera Calibration Pipeline")
    print("=" * 70)
    print(f"Processing {len(calib_folders)} calibration set(s)...")
    print("=" * 70)

    # 处理每个文件夹
    for i, (calib_folder, coords_folder, output_folder) in enumerate(zip(calib_folders, coords_folders, output_folders)):
        print(f"\n{'#' * 70}")
        print(f"# [{i+1}/{len(calib_folders)}] {os.path.basename(calib_folder)}")
        print(f"{'#' * 70}")
        print(f"Calibration folder: {calib_folder}")
        print(f"Image coords folder: {coords_folder}")
        if output_folder:
            print(f"Output folder: {output_folder}")
        
        # 如果使用了自动检测，提取 way 名称；否则可用 --way_name 指定
        filename_keyword = None
        if way_configs is not None:
            way_name = way_configs[i][2]
            filename_keyword = way_name
            print(f"Using filename keyword: {filename_keyword}")
        elif args.way_name:
            filename_keyword = args.way_name
            print(f"Using filename keyword (--way_name): {filename_keyword}")
        
        try:
            results_root = results_roots[i] if results_roots else None
            run_calibration_pipeline(
                calibration_folder=calib_folder,
                alpha_range=args.alpha_range,
                poly_deg_range=args.poly_deg_range,
                cv_folds=args.cv_folds,
                train_fraction=args.train_fraction,
                num_permutations=args.num_permutations,
                output_folder=output_folder,
                image_coords_folder=coords_folder,
                filename_keyword=filename_keyword,
                beads_only=args.beads_only,
                model_types=args.model_types,
                results_root=results_root,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {calib_folder}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("All calibrations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
