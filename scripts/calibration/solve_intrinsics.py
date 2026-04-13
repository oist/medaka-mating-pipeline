"""
Train calibration models for 3D reconstruction.

Supports:
- single-folder training
- batch processing over multiple folders
- auto-detection of `way1/way2/...`
- merged-way training
"""

import argparse
import glob
import os
import re
import sys
from typing import List

import numpy as np

try:
    from .lib.calibration_core import (
        _ensure_project_paths,
        _save_training_summary,
        load_calibration_data,
        run_calibration_pipeline,
    )
    from .lib.model_training import prepare_model_data, save_all_models, train_calibration_model
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.calibration_core import (  # type: ignore
        _ensure_project_paths,
        _save_training_summary,
        load_calibration_data,
        run_calibration_pipeline,
    )
    from lib.model_training import prepare_model_data, save_all_models, train_calibration_model  # type: ignore


def parse_paths(path_str: str) -> List[str]:
    """Parse comma-separated path strings."""
    if not path_str:
        return []
    paths = [p.strip() for p in path_str.split(",")]
    return [p for p in paths if p]


def auto_detect_ways(calibration_folder: str, image_coords_folder: str) -> List[tuple]:
    """
    Auto-detect available way files and match them to calibration folders.

    Returns:
        `[(calib_folder, coords_folder, way_name), ...]`
    """
    results = []

    coord_files = glob.glob(os.path.join(image_coords_folder, "*_image_coords.txt"))
    ways_found = set()

    for coord_file in coord_files:
        basename = os.path.basename(coord_file)
        way_match = re.search(r"way(\d+)", basename, re.IGNORECASE)
        if way_match:
            way_num = way_match.group(1)
            way_name = f"way{way_num}"
            ways_found.add((way_name, int(way_num)))

    if not ways_found:
        raise ValueError(f"No way files found in {image_coords_folder}")

    ways_sorted = sorted(ways_found, key=lambda x: x[1])

    for way_name, way_num in ways_sorted:
        possible_calib_folders = [
            os.path.join(calibration_folder, f"cali_*_way{way_num}"),
            os.path.join(calibration_folder, f"cali_*_way{way_num}_pic"),
            os.path.join(calibration_folder, f"*way{way_num}*"),
            calibration_folder,
        ]

        calib_folder = None
        for pattern in possible_calib_folders:
            matches = glob.glob(pattern)
            if matches:
                calib_folder = matches[0]
                break

        if not calib_folder:
            calib_folder = calibration_folder

        results.append((calib_folder, image_coords_folder, way_name))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train calibration models for 3D reconstruction (supports batch processing)"
    )
    parser.add_argument(
        "--calibration_folder", type=str, required=True,
        help="Calibration folder(s) containing the 3D coordinate template; comma-separated for multiple"
    )
    parser.add_argument(
        "--image_coords_folder", type=str, required=True,
        help="Folder(s) containing manually annotated image-coordinate txt files; comma-separated for multiple"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="Model output folder(s), comma-separated. Default: models/calibration/<calib_name>/"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Root folder(s) for training-time CSVs and figures; comma-separated. If omitted, infer from calibration_folder.",
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
        help="Fraction of data reserved for training in each validation split"
    )
    parser.add_argument(
        "--num_permutations", type=int, default=100,
        help="Number of train/test splits for validation"
    )
    parser.add_argument(
        "--beads_only", action="store_true",
        help="Use only BEADS_WAY* points and ignore corners/pillars/red lines"
    )
    parser.add_argument(
        "--model_types", type=str, nargs="+", default=None,
        help="Model types to try: polynomial, random_forest, svr, mlp. Default: try all"
    )
    parser.add_argument(
        "--merge_ways", action="store_true",
        help="Merge multiple ways into one unified model (assumes a shared 3D coordinate system)"
    )
    parser.add_argument(
        "--no_auto_detect", action="store_true",
        help="Disable auto-detection of ways and treat the folders as one calibration set"
    )
    parser.add_argument(
        "--way_name", type=str, default=None,
        help="When using --no_auto_detect, specify the way name (for example way1 or way2)"
    )
    parser.add_argument(
        "--way_names", type=str, nargs="+", default=None,
        help="When using --merge_ways, explicitly list the way names to merge"
    )
    args = parser.parse_args()

    calib_folders = parse_paths(args.calibration_folder)
    coords_folders = parse_paths(args.image_coords_folder)
    output_folders = parse_paths(args.output_folder) if args.output_folder else []
    results_roots = parse_paths(args.results_root) if args.results_root else []

    if args.merge_ways:
        if len(calib_folders) != 1 or len(coords_folders) != 1:
            print("[ERROR] --merge_ways requires exactly one calibration_folder and one image_coords_folder")
            return

        print("=" * 70)
        print("Camera Calibration Pipeline - MERGE MODE")
        print("=" * 70)
        print("Merging multiple ways into one unified model")
        print("=" * 70)

        way_names = args.way_names
        if way_names is None:
            coord_files = glob.glob(os.path.join(coords_folders[0], "*_image_coords.txt"))
            ways_found = set()
            for coord_file in coord_files:
                basename = os.path.basename(coord_file)
                way_match = re.search(r"way(\d+)", basename, re.IGNORECASE)
                if way_match:
                    way_num = way_match.group(1)
                    ways_found.add((f"way{way_num}", int(way_num)))

            if not ways_found:
                print(f"[ERROR] No way files found in {coords_folders[0]}")
                return

            way_names = [w[0] for w in sorted(ways_found, key=lambda x: x[1])]
            print(f"Auto-detected ways: {', '.join(way_names)}")
        else:
            print(f"Using specified ways: {', '.join(way_names)}")

        try:
            print(f"Loading and merging data from {len(way_names)} way(s)...")
            all_measured_coords = []
            all_image_coords = []

            # load_calibration_data now returns all coordinates in the tank
            # frame (corners via pixel-position matching, beads via 180-degree
            # alignment for even ways).  No additional alignment needed here.

            for way_name in way_names:
                print(f"  Loading {way_name}...")
                try:
                    template_3d_way, image_coords_way, labels_way, _ = load_calibration_data(
                        calibration_folder=calib_folders[0],
                        image_coords_folder=coords_folders[0],
                        filename_keyword=way_name,
                        way_name=way_name,
                        beads_only=args.beads_only,
                    )

                    num_points = image_coords_way.shape[1]
                    all_measured_coords.extend(
                        np.asarray(template_3d_way[:num_points], dtype=float)
                    )
                    for i in range(num_points):
                        cam_coords = [image_coords_way[j, i, :] for j in range(3)]
                        all_image_coords.append(cam_coords)

                    print(f"    {way_name}: added {num_points} points")
                except Exception as e:
                    print(f"    [WARN] {way_name}: failed to load, skipping - {e}")
                    continue

            if not all_measured_coords:
                raise ValueError("No valid data loaded from any way")

            measured_coords = np.array(all_measured_coords)
            matched_image_coords = np.array(all_image_coords)
            image_coords = np.transpose(matched_image_coords, (1, 0, 2))

            print(f"\nMerged data: {len(measured_coords)} total points from {len(way_names)} way(s)")

            model_data = prepare_model_data(image_coords, measured_coords)

            trained_models = {}
            summary_rows = []

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

                sorted_results = sorted(all_models_results, key=lambda x: x["cv_mean_error"])
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

            paths = _ensure_project_paths(calib_folders[0], output_folders[0] if output_folders else None)
            os.makedirs(paths["model_output_folder"], exist_ok=True)
            save_all_models(trained_models, paths["model_output_folder"])

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

    way_configs = None
    if (
        len(calib_folders) == 1
        and len(coords_folders) == 1
        and not args.merge_ways
        and not args.no_auto_detect
    ):
        print("[INFO] Auto-detecting ways from image coordinates folder...")
        try:
            way_configs = auto_detect_ways(calib_folders[0], coords_folders[0])
            way_names = [w[2] for w in way_configs]
            print(f"[INFO] Detected ways: {', '.join(way_names)}")

            calib_folders = [w[0] for w in way_configs]
            coords_folders = [w[1] for w in way_configs]

            if len(output_folders) == 1:
                base_output = output_folders[0]
                output_folders = [os.path.join(base_output, f"cali_{way_name}") for way_name in way_names]
                print("[INFO] Auto-generating output folders:")
                for way_name, out_folder in zip(way_names, output_folders):
                    print(f"  {way_name}: {out_folder}")
            elif len(output_folders) > 1:
                if len(way_configs) != len(output_folders):
                    print(f"[ERROR] Detected {len(way_configs)} way(s) but {len(output_folders)} output folder(s) were provided")
                    print(f"  Detected ways: {way_names}")
                    return
            else:
                output_folders = [None] * len(way_configs)
        except Exception as e:
            print(f"[ERROR] Failed to auto-detect ways: {e}")
            return
    else:
        if len(calib_folders) != len(coords_folders):
            print(f"[ERROR] Number of calibration folders ({len(calib_folders)}) must match image coordinate folders ({len(coords_folders)})")
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

    for i, (calib_folder, coords_folder, output_folder) in enumerate(zip(calib_folders, coords_folders, output_folders)):
        print(f"\n{'#' * 70}")
        print(f"# [{i+1}/{len(calib_folders)}] {os.path.basename(calib_folder)}")
        print(f"{'#' * 70}")
        print(f"Calibration folder: {calib_folder}")
        print(f"Image coordinates folder: {coords_folder}")
        if output_folder:
            print(f"Output folder: {output_folder}")

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
