"""Fit and evaluate a flat-interface refractive model on the 12-ball dataset.

The current 21-pose rigid-fit targets provide the training world coordinates.
All models are evaluated on the same 10 held-out poses using cube-shape Kabsch
residual and fit-free pairwise-distance error. The refractive camera uses one
air/water interface for cam0 and an air/glass/water flat pane for cam1/cam2.

This is a wet-data prototype: dry intrinsics and measured pane planes are not
available, so intrinsics/interface geometry are weakly regularized around a
projective DLT initialization. Those fitted physical parameters must not be
treated as metrology; the held-out reconstruction score is still meaningful.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.calibration.refractive_geometry import (
    FlatPortCamera,
    fit_flat_port_camera,
    fit_flat_port_camera_fixed_intrinsics,
    initialize_flat_port_camera,
    triangulate_pixels,
)
from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration


VIEW_COLUMNS = [f"cam{camera}_{axis}" for camera in range(3) for axis in ("x", "y")]


def _geometry_lookup(path: Path) -> dict[int, np.ndarray]:
    geometry = pd.read_csv(path)
    return {
        int(row.marker_id): np.array([row.object_x, row.object_y, row.object_z], dtype=float)
        for row in geometry.itertuples(index=False)
    }


def _heldout_coordinates(path: Path, pose_list: Path) -> pd.DataFrame:
    coords = pd.read_csv(path)
    coords["pose_id"] = coords["pose_id"].astype(str)
    poses = {
        line.strip()
        for line in pose_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    heldout = coords[coords["pose_id"].isin(poses)].copy()
    if heldout["pose_id"].nunique() != len(poses):
        missing = sorted(poses.difference(heldout["pose_id"]))
        raise ValueError(f"held-out pose IDs absent from image-coordinate CSV: {missing}")
    return heldout


def _regression_predictor(model_folder: Path) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    calibration = Calibration(model_folder)

    def predict(image_points: np.ndarray) -> tuple[np.ndarray, float]:
        flat = np.asarray(image_points, dtype=float).reshape(1, 6)
        point = np.asarray(calibration._model_im2xyz.predict(flat), dtype=float)[0]
        return point, np.nan

    return predict


def _ray_predictor(cameras: Sequence[FlatPortCamera]) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    def predict(image_points: np.ndarray) -> tuple[np.ndarray, float]:
        return triangulate_pixels(cameras, image_points)

    return predict


def evaluate_predictor(
    heldout: pd.DataFrame,
    geometry: dict[int, np.ndarray],
    predictor: Callable[[np.ndarray], tuple[np.ndarray, float]],
    *,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_rows = []
    pair_rows = []
    for pose_id, group in heldout.groupby("pose_id", sort=True):
        object_points = np.asarray([geometry[int(marker)] for marker in group["marker_id"]])
        predicted = []
        ray_rms = []
        for row in group.itertuples(index=False):
            image = np.array(
                [
                    [row.cam0_x, row.cam0_y],
                    [row.cam1_x, row.cam1_y],
                    [row.cam2_x, row.cam2_y],
                ],
                dtype=float,
            )
            point, ray_error = predictor(image)
            predicted.append(point)
            ray_rms.append(ray_error)
        predicted_array = np.asarray(predicted)
        rotation, translation = kabsch_rigid_fit(object_points, predicted_array)
        reference = apply_rigid(object_points, rotation, translation)
        residual_vectors = predicted_array - reference
        residuals = np.linalg.norm(residual_vectors, axis=1)
        for index, marker_id in enumerate(group["marker_id"].astype(int)):
            marker_rows.append(
                {
                    "model": model_name,
                    "pose_id": str(pose_id),
                    "marker_id": int(marker_id),
                    "X": float(reference[index, 0]),
                    "Y": float(reference[index, 1]),
                    "Z": float(reference[index, 2]),
                    "pred_X": float(predicted_array[index, 0]),
                    "pred_Y": float(predicted_array[index, 1]),
                    "pred_Z": float(predicted_array[index, 2]),
                    "residual_X_mm": float(residual_vectors[index, 0]),
                    "residual_Y_mm": float(residual_vectors[index, 1]),
                    "residual_Z_mm": float(residual_vectors[index, 2]),
                    "kabsch_residual_mm": float(residuals[index]),
                    "ray_intersection_rms_mm": float(ray_rms[index]),
                }
            )
        for first in range(len(object_points)):
            for second in range(first + 1, len(object_points)):
                true_distance = float(np.linalg.norm(object_points[first] - object_points[second]))
                predicted_distance = float(
                    np.linalg.norm(predicted_array[first] - predicted_array[second])
                )
                pair_rows.append(
                    {
                        "model": model_name,
                        "pose_id": str(pose_id),
                        "marker_id_first": int(group.iloc[first]["marker_id"]),
                        "marker_id_second": int(group.iloc[second]["marker_id"]),
                        "true_distance_mm": true_distance,
                        "predicted_distance_mm": predicted_distance,
                        "signed_distance_error_mm": predicted_distance - true_distance,
                        "absolute_distance_error_mm": abs(predicted_distance - true_distance),
                    }
                )
    return pd.DataFrame(marker_rows), pd.DataFrame(pair_rows)


def summarize_model(markers: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object]:
    residual = markers["kabsch_residual_mm"].to_numpy(dtype=float)
    distance = pairs["absolute_distance_error_mm"].to_numpy(dtype=float)
    signed = pairs["signed_distance_error_mm"].to_numpy(dtype=float)
    ray = markers["ray_intersection_rms_mm"].dropna().to_numpy(dtype=float)
    per_pose = []
    for pose_id, group in markers.groupby("pose_id"):
        pose_pairs = pairs[pairs["pose_id"] == pose_id]
        per_pose.append(
            {
                "pose_id": str(pose_id),
                "kabsch_rms_mm": float(
                    np.sqrt(np.mean(group["kabsch_residual_mm"].to_numpy() ** 2))
                ),
                "pairwise_mae_mm": float(pose_pairs["absolute_distance_error_mm"].mean()),
            }
        )
    return {
        "n_poses": int(markers["pose_id"].nunique()),
        "n_markers": int(len(markers)),
        "kabsch_residual_mm": {
            "mean": float(np.mean(residual)),
            "rms": float(np.sqrt(np.mean(residual**2))),
            "median": float(np.median(residual)),
            "p95": float(np.percentile(residual, 95)),
            "maximum": float(np.max(residual)),
        },
        "pairwise_distance_error_mm": {
            "mae": float(np.mean(distance)),
            "rms": float(np.sqrt(np.mean(distance**2))),
            "p95": float(np.percentile(distance, 95)),
            "maximum": float(np.max(distance)),
            "signed_bias": float(np.mean(signed)),
        },
        "ray_intersection_rms_mm": (
            {
                "mean": float(np.mean(ray)),
                "median": float(np.median(ray)),
                "p95": float(np.percentile(ray, 95)),
                "maximum": float(np.max(ray)),
            }
            if len(ray)
            else None
        ),
        "per_pose": per_pose,
    }


def plot_comparison(
    markers: pd.DataFrame,
    pairs: pd.DataFrame,
    summaries: dict[str, dict[str, object]],
    output_path: Path,
    *,
    n_training_poses: int | None = None,
    n_heldout_poses: int | None = None,
    dry_intrinsics_fixed: bool = False,
) -> None:
    colors = {
        "base_regression": "#8C8C8C",
        "refined_regression": "#009E73",
        "pinhole_dlt": "#56B4E9",
        "flat_refractive": "#D55E00",
    }
    labels = {
        "base_regression": "Base polynomial",
        "refined_regression": "Refined polynomial",
        "pinhole_dlt": "Projective DLT",
        "flat_refractive": "Flat refractive",
    }
    order = list(labels)
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    for model in order:
        values = np.sort(markers.loc[markers["model"] == model, "kabsch_residual_mm"])
        cumulative = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(values, cumulative, color=colors[model], linewidth=2.3, label=labels[model])
    ax.set_xlabel("Held-out marker Kabsch residual (mm)")
    ax.set_ylabel("Markers <= residual (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Cube-shape residual CDF")

    ax = fig.add_subplot(grid[0, 1])
    for model in order:
        values = np.sort(pairs.loc[pairs["model"] == model, "absolute_distance_error_mm"])
        cumulative = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(values, cumulative, color=colors[model], linewidth=2.3, label=labels[model])
    ax.set_xlabel("Held-out pairwise-distance absolute error (mm)")
    ax.set_ylabel("Marker pairs <= error (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Fit-free metric-ruler CDF")

    ax = fig.add_subplot(grid[1, 0])
    pose_ids = sorted(markers["pose_id"].unique(), key=lambda value: int(value.lstrip("f")))
    x = np.arange(len(pose_ids))
    for model in order:
        values = {
            row["pose_id"]: row["kabsch_rms_mm"]
            for row in summaries[model]["per_pose"]
        }
        ax.plot(x, [values[pose] for pose in pose_ids], marker="o", color=colors[model], label=labels[model])
    ax.set_xticks(x, pose_ids, rotation=45)
    ax.set_ylabel("Per-pose Kabsch RMS (mm)")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Generalization across held-out placements")

    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    lines = [
        "MODEL                       Kabsch mean/RMS/P95     pairwise MAE/P95     bias",
        "-" * 84,
    ]
    for model in order:
        shape = summaries[model]["kabsch_residual_mm"]
        distance = summaries[model]["pairwise_distance_error_mm"]
        lines.append(
            f"{labels[model]:<27} "
            f"{shape['mean']:.3f}/{shape['rms']:.3f}/{shape['p95']:.3f} mm      "
            f"{distance['mae']:.3f}/{distance['p95']:.3f} mm       "
            f"{distance['signed_bias']:+.3f} mm"
        )
    refractive = summaries["flat_refractive"]["ray_intersection_rms_mm"]
    guardrail = (
        [
            "Dry intrinsics are fixed and the side-pane geometry is surveyed/regularized.",
            "The unmeasured top water height and absolute Z gauge remain provisional.",
        ]
        if dry_intrinsics_fixed
        else [
            "Important: the wet-only fit regularizes intrinsics and interface geometry around DLT.",
            "Use held-out reconstruction scores here; do not interpret fitted optical parameters",
            "as physical measurements until dry intrinsics and pane geometry are supplied.",
        ]
    )
    lines.extend(
        [
            "",
            f"Refractive ray-intersection RMS: median {refractive['median']:.3f} mm, "
            f"P95 {refractive['p95']:.3f} mm.",
            "",
            *guardrail,
        ]
    )
    ax.text(
        0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F6F4EF", "edgecolor": "#C7C0B2"},
    )

    split_text = (
        f"Same {n_training_poses} training poses and {n_heldout_poses} held-out poses for every model"
        if n_training_poses is not None and n_heldout_poses is not None
        else "Identical whole-pose train/test split for every model"
    )
    fig.suptitle(
        "Held-out 12-ball head-to-head: regression versus explicit flat refraction\n"
        + split_text,
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-points-csv", type=Path, required=True)
    parser.add_argument("--image-coords-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--heldout-pose-list", type=Path, required=True)
    parser.add_argument("--base-regression-model", type=Path, required=True)
    parser.add_argument("--refined-regression-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tank-bounds", type=float, nargs=6,
        default=(0, 192, 0, 192, 0, 192),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--glass-thickness-mm", type=float, default=4.0)
    parser.add_argument(
        "--intrinsics-dir",
        type=Path,
        default=None,
        help="directory containing intrinsics_{top_cam,side1_cam,side2_cam}.npz; fixes dry K/distortion",
    )
    parser.add_argument(
        "--water-height-mm",
        type=float,
        default=192.0,
        help="provisional top air/water plane height; measure this for absolute Z",
    )
    parser.add_argument("--no-interface-correction", action="store_true")
    parser.add_argument("--no-fit-distortion", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=1500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.asarray(args.tank_bounds, dtype=float).reshape(3, 2)
    training = pd.read_csv(args.training_points_csv)
    if {"refined_X", "refined_Y", "refined_Z"}.issubset(training.columns):
        world_columns = ["refined_X", "refined_Y", "refined_Z"]
    else:
        world_columns = ["X", "Y", "Z"]
    world_points = training[world_columns].to_numpy(dtype=float)
    heldout = _heldout_coordinates(args.image_coords_csv, args.heldout_pose_list)
    geometry = _geometry_lookup(args.geometry_csv)

    pinhole_cameras = []
    refractive_cameras = []
    fit_diagnostics = {}
    camera_views = ("top_cam", "side1_cam", "side2_cam")
    for camera_index in range(3):
        image_points = training[[f"cam{camera_index}_x", f"cam{camera_index}_y"]].to_numpy(dtype=float)
        pinhole = initialize_flat_port_camera(
            f"cam{camera_index}",
            world_points,
            image_points,
            tank_bounds=bounds,
            layer_count=0,
            glass_thickness_mm=args.glass_thickness_mm,
        )
        pinhole_cameras.append(pinhole)
        initial_refractive = initialize_flat_port_camera(
            f"cam{camera_index}",
            world_points,
            image_points,
            tank_bounds=bounds,
            layer_count=1 if camera_index == 0 else 2,
            glass_thickness_mm=args.glass_thickness_mm,
        )
        if args.intrinsics_dir is None:
            fitted, diagnostics = fit_flat_port_camera(
                initial_refractive,
                world_points,
                image_points,
                fit_distortion=not args.no_fit_distortion,
                max_nfev=args.max_nfev,
            )
            fit_diagnostics[f"cam{camera_index}"] = asdict(diagnostics)
        else:
            intrinsic_path = args.intrinsics_dir / f"intrinsics_{camera_views[camera_index]}.npz"
            intrinsic_data = np.load(intrinsic_path)
            k = np.asarray(intrinsic_data["K"], dtype=float)
            d = np.pad(np.asarray(intrinsic_data["distortion"], dtype=float).reshape(-1), (0, 8))[:8]
            if camera_index == 0:
                normal = np.array([0.0, 0.0, -1.0])
                interface_point = np.array([0.0, 0.0, args.water_height_mm])
                layer_count = 1
            elif camera_index == 1:
                # Inner water plane y=0; outer air/glass plane is 4 mm toward cam1.
                normal = np.array([0.0, 1.0, 0.0])
                interface_point = np.array([0.0, -args.glass_thickness_mm, 0.0])
                layer_count = 2
            else:
                # Inner water plane x=192; outer plane is 4 mm toward cam2.
                normal = np.array([-1.0, 0.0, 0.0])
                interface_point = np.array([192.0 + args.glass_thickness_mm, 0.0, 0.0])
                layer_count = 2
            physical_initial = FlatPortCamera(
                name=f"cam{camera_index}",
                fx=float(k[0, 0]),
                fy=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                k1=float(d[0]),
                k2=float(d[1]),
                rotation_vector=initial_refractive.rotation_vector,
                camera_center=initial_refractive.camera_center,
                interface_normal=normal.tolist(),
                interface_point=interface_point.tolist(),
                glass_thickness_mm=args.glass_thickness_mm,
                index_water=1.3325,
                layer_count=layer_count,
                p1=float(d[2]),
                p2=float(d[3]),
                k3=float(d[4]),
                k4=float(d[5]),
                k5=float(d[6]),
                k6=float(d[7]),
            )
            stage1, diagnostics1 = fit_flat_port_camera_fixed_intrinsics(
                physical_initial,
                world_points,
                image_points,
                fit_interface_correction=False,
                max_nfev=args.max_nfev,
            )
            if args.no_interface_correction:
                fitted = stage1
                diagnostics = diagnostics1
                fit_diagnostics[f"cam{camera_index}"] = {"stage1_fixed_interface": asdict(diagnostics1)}
            else:
                fitted, diagnostics2 = fit_flat_port_camera_fixed_intrinsics(
                    stage1,
                    world_points,
                    image_points,
                    fit_interface_correction=True,
                    offset_prior_sigma_mm=25.0 if camera_index == 0 else 2.0,
                    max_nfev=args.max_nfev,
                )
                diagnostics = diagnostics2
                fit_diagnostics[f"cam{camera_index}"] = {
                    "stage1_fixed_interface": asdict(diagnostics1),
                    "stage2_interface_correction": asdict(diagnostics2),
                }
        refractive_cameras.append(fitted)
        fitted.to_json(args.output_dir / f"cam{camera_index}_flat_refractive.json")

    predictors = {
        "base_regression": _regression_predictor(args.base_regression_model),
        "refined_regression": _regression_predictor(args.refined_regression_model),
        "pinhole_dlt": _ray_predictor(pinhole_cameras),
        "flat_refractive": _ray_predictor(refractive_cameras),
    }
    marker_tables = []
    pair_tables = []
    summaries = {}
    for model_name, predictor in predictors.items():
        markers, pairs = evaluate_predictor(
            heldout, geometry, predictor, model_name=model_name
        )
        marker_tables.append(markers)
        pair_tables.append(pairs)
        summaries[model_name] = summarize_model(markers, pairs)
    all_markers = pd.concat(marker_tables, ignore_index=True)
    all_pairs = pd.concat(pair_tables, ignore_index=True)
    marker_path = args.output_dir / "heldout_marker_errors.csv"
    pair_path = args.output_dir / "heldout_pairwise_errors.csv"
    all_markers.to_csv(marker_path, index=False)
    all_pairs.to_csv(pair_path, index=False)

    figure_path = args.output_dir / "refractive_vs_regression_heldout.png"
    plot_comparison(
        all_markers,
        all_pairs,
        summaries,
        figure_path,
        n_training_poses=int(training["pose_id"].astype(str).nunique()) if "pose_id" in training else None,
        n_heldout_poses=int(heldout["pose_id"].nunique()),
        dry_intrinsics_fixed=args.intrinsics_dir is not None,
    )
    report = {
        "data_split": {
            "n_training_poses": int(training["pose_id"].astype(str).nunique()) if "pose_id" in training else None,
            "n_training_markers": int(len(training)),
            "n_heldout_poses": int(heldout["pose_id"].nunique()),
            "n_heldout_markers": int(len(heldout)),
        },
        "interface_model": {
            "cam0": "one planar air-to-water interface",
            "cam1": "parallel air-to-glass and glass-to-water interfaces",
            "cam2": "parallel air-to-glass and glass-to-water interfaces",
            "index_air": refractive_cameras[0].index_air,
            "index_glass": refractive_cameras[0].index_glass,
            "index_water": refractive_cameras[0].index_water,
            "assumed_glass_thickness_mm": args.glass_thickness_mm,
            "dry_intrinsics_fixed": args.intrinsics_dir is not None,
            "intrinsics_dir": str(args.intrinsics_dir) if args.intrinsics_dir else None,
            "fit_radial_distortion": args.intrinsics_dir is None and not args.no_fit_distortion,
            "inner_footprint_mm": [192.0, 192.0],
            "provisional_water_height_mm": args.water_height_mm,
            "temperature_c": 25.0,
        },
        "fit_diagnostics": fit_diagnostics,
        "models": summaries,
        "fitted_refractive_cameras": {
            camera.name: asdict(camera) for camera in refractive_cameras
        },
        "interpretation_guardrail": (
            "The top-interface height and absolute Z gauge remain provisional because water height "
            "was not measured. Shape/scale held-out errors are valid, but absolute tank localization "
            "requires measured water height or surveyed fixed markers."
            if args.intrinsics_dir is not None
            else "This wet-only prototype fits correlated intrinsic/extrinsic/interface parameters. "
            "Its held-out reconstruction error is valid for comparison, but the fitted physical "
            "parameters are not metrology until dry intrinsics and measured pane planes are used."
        ),
        "outputs": {
            "figure": str(figure_path),
            "marker_errors": str(marker_path),
            "pairwise_errors": str(pair_path),
        },
    }
    report_path = args.output_dir / "refractive_vs_regression_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
