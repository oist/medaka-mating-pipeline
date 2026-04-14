"""
Refine merged calibration using rigid body constraints.

The calibration cube is a rigid object whose bead positions are known
precisely in the cube's own coordinate frame.  For each orientation
("way"), the cube is placed in the tank at an approximately-known pose.
This script refines those poses by:

1. Training an initial polynomial model with the assumed 3D targets.
2. Predicting 3D positions for every calibration point.
3. Fitting a rigid transform (R, t) per way via the Kabsch algorithm:
   cube-frame geometry -> predicted tank-frame positions.
4. Computing refined 3D targets from the rigid transforms.
5. Detecting annotation outliers (large residual after rigid fit).
6. Retraining the model with refined targets, excluding outliers.
7. Repeating steps 2-6 until convergence.

This corrects for imprecise cube-placement assumptions and identifies
annotation errors in a principled way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .lib.calibration_core import (
        _ensure_project_paths,
        _save_point_mapping_and_figures,
        _save_training_summary,
        load_calibration_data,
    )
    from .lib.model_training import (
        prepare_model_data,
        save_all_models,
        train_calibration_model,
    )
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.calibration_core import (  # type: ignore
        _ensure_project_paths,
        _save_point_mapping_and_figures,
        _save_training_summary,
        load_calibration_data,
    )
    from lib.model_training import (  # type: ignore
        prepare_model_data,
        save_all_models,
        train_calibration_model,
    )


# ---------------------------------------------------------------------------
# Rigid-body geometry helpers
# ---------------------------------------------------------------------------

def kabsch_rigid_fit(
    P: np.ndarray,
    Q: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm: find R, t minimising sum_i w_i ||R p_i + t - q_i||^2.

    Parameters
    ----------
    P : (N, 3) source points (cube frame).
    Q : (N, 3) target points (tank frame / predicted).
    weights : (N,) optional per-point weights.

    Returns
    -------
    R : (3, 3) proper rotation matrix.
    t : (3,) translation vector.
    """
    if weights is None:
        weights = np.ones(len(P))
    w = weights / weights.sum()

    p_mean = (w[:, None] * P).sum(axis=0)
    q_mean = (w[:, None] * Q).sum(axis=0)

    Pc = P - p_mean
    Qc = Q - q_mean

    H = (w[:, None] * Pc).T @ Qc
    U, _S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    return R, t


def apply_rigid(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform: Q = R @ P^T + t for each row of P."""
    return (R @ P.T).T + t


def rotation_to_euler_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (XYZ convention) in degrees from a rotation matrix."""
    # Manual extraction avoids a scipy dependency here.
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return (np.degrees(x), np.degrees(y), np.degrees(z))


def _tank_to_cube_frame_beads(xyz: np.ndarray, way_num: int) -> np.ndarray:
    """Reverse the tank-frame conversion to recover cube-frame bead positions.

    ``load_calibration_data`` applies ``(190-X, 190-Y)`` for even ways.
    This reverses that transform.  Only valid for **bead** points.
    """
    cube = xyz.copy()
    if way_num % 2 == 0:
        cube[:, 0] = 190.0 - cube[:, 0]
        cube[:, 1] = 190.0 - cube[:, 1]
    return cube


# The 8 physical corners of the calibration cube in its own frame.
CUBE_FRAME_CORNERS = np.array([
    [7.5, 7.5, 194.0],
    [7.5, 182.5, 194.0],
    [182.5, 7.5, 194.0],
    [182.5, 182.5, 194.0],
    [7.5, 7.5, 4.0],
    [7.5, 182.5, 4.0],
    [182.5, 7.5, 4.0],
    [182.5, 182.5, 4.0],
])


def _assign_corner_cube_frames(
    corner_tank_xyz: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Assign cube-frame positions to corners using a bead-derived rigid transform.

    Projects every known cube-frame corner into the tank frame using
    (R, t) and matches each annotated corner to its nearest projection.
    """
    expected_tank = apply_rigid(CUBE_FRAME_CORNERS, R, t)
    cube_frame = np.zeros_like(corner_tank_xyz)
    for i, tank_pos in enumerate(corner_tank_xyz):
        dists = np.linalg.norm(expected_tank - tank_pos, axis=1)
        cube_frame[i] = CUBE_FRAME_CORNERS[np.argmin(dists)]
    return cube_frame


# ---------------------------------------------------------------------------
# Main refinement routine
# ---------------------------------------------------------------------------

def run_rigid_refinement(
    calibration_folder: str,
    image_coords_folder: str,
    way_names: List[str],
    *,
    beads_only: bool = True,
    outlier_sigma: float = 2.5,
    max_iterations: int = 3,
    convergence_tol_mm: float = 0.01,
    output_folder: str | None = None,
    results_root: str | None = None,
    alpha_range: List[float] | None = None,
    poly_deg_range: List[int] | None = None,
    num_permutations: int = 100,
    model_types: List[str] | None = None,
) -> dict:
    """Run the iterative rigid-body refinement pipeline.

    Returns a summary dict with per-way transforms, outlier list, and
    before/after error metrics.
    """
    if alpha_range is None:
        alpha_range = [1e-10, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1]
    if poly_deg_range is None:
        poly_deg_range = [1, 2, 3, 4]

    training_kwargs = dict(
        alpha_range=alpha_range,
        poly_deg_range=poly_deg_range,
        cv_folds=3,
        train_fraction=0.9,
        num_permutations=num_permutations,
        model_types=model_types,
    )
    fast_training_kwargs = {**training_kwargs}
    fast_training_kwargs["num_permutations"] = 20
    fast_training_kwargs["model_types"] = ["polynomial"]

    # ------------------------------------------------------------------
    # Phase 1: Load per-way data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Rigid Body Calibration Refinement")
    print("=" * 70)

    per_way: Dict[str, dict] = {}
    for way_name in way_names:
        way_num = int(re.search(r"(\d+)", way_name).group(1))
        template_3d, image_coords, labels, meta = load_calibration_data(
            calibration_folder=calibration_folder,
            image_coords_folder=image_coords_folder,
            filename_keyword=way_name,
            way_name=way_name,
            beads_only=beads_only,
        )
        # Build per-point bead/corner mask
        is_bead = np.array([l.startswith("BEAD:") for l in labels])
        # Cube-frame positions: reliable for beads via the 180-deg reversal;
        # corners need the bead-derived rigid transform (filled in later).
        cube_frame = np.full_like(template_3d, np.nan)
        cube_frame[is_bead] = _tank_to_cube_frame_beads(
            template_3d[is_bead], way_num,
        )
        n_beads = int(is_bead.sum())
        n_corners = int((~is_bead).sum())
        per_way[way_name] = dict(
            way_num=way_num,
            template_3d=template_3d,
            cube_frame=cube_frame,
            image_coords=image_coords,
            labels=labels,
            is_bead=is_bead,
            n_points=template_3d.shape[0],
        )
        label = f"{n_beads} beads"
        if n_corners:
            label += f" + {n_corners} corners"
        print(f"  {way_name}: {label}")

    # Build merged arrays with way-slice bookkeeping
    all_cube_parts, all_target_parts, all_pixel_parts = [], [], []
    all_is_bead_parts: List[np.ndarray] = []
    all_labels: List[str] = []
    all_image_coords_parts = []
    way_slices: Dict[str, slice] = {}
    offset = 0

    for way_name in way_names:
        wd = per_way[way_name]
        n = wd["n_points"]
        all_cube_parts.append(wd["cube_frame"])
        all_target_parts.append(wd["template_3d"])
        all_pixel_parts.append(
            wd["image_coords"].swapaxes(0, 1).reshape(n, -1)
        )
        all_is_bead_parts.append(wd["is_bead"])
        all_image_coords_parts.append(wd["image_coords"])
        all_labels.extend(wd["labels"])
        way_slices[way_name] = slice(offset, offset + n)
        offset += n

    all_cube = np.vstack(all_cube_parts)
    all_targets_original = np.vstack(all_target_parts)
    all_pixels = np.vstack(all_pixel_parts)
    all_is_bead = np.concatenate(all_is_bead_parts)
    all_image_coords = np.concatenate(all_image_coords_parts, axis=1)
    N_total = len(all_targets_original)
    n_beads_total = int(all_is_bead.sum())
    n_corners_total = N_total - n_beads_total

    summary_line = f"Total: {N_total} points ({n_beads_total} beads"
    if n_corners_total:
        summary_line += f" + {n_corners_total} corners"
    summary_line += f") from {len(way_names)} way(s)"
    print(f"\n{summary_line}")

    # ------------------------------------------------------------------
    # Phase 2: Iterative refinement
    # ------------------------------------------------------------------
    current_targets = all_targets_original.copy()
    inlier_mask = np.ones(N_total, dtype=bool)
    per_way_transforms: Dict[str, dict] = {}
    outlier_indices: List[int] = []

    for iteration in range(max_iterations):
        print(f"\n{'-' * 50}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print(f"{'-' * 50}")

        # Train model on inliers with current targets
        X_train = all_pixels[inlier_mask]
        Y_train = current_targets[inlier_mask]
        n_inliers = int(inlier_mask.sum())
        print(f"Training on {n_inliers}/{N_total} inlier points ...")

        use_fast = iteration < max_iterations - 1
        kwargs = fast_training_kwargs if use_fast else training_kwargs
        model, best_params, _all_results, val_results = train_calibration_model(
            X_train, Y_train, **kwargs,
        )
        print(f"  Best model: {best_params}")
        print(
            f"  CV error: {val_results['mean_test_error']:.3f}"
            f" +/- {val_results['std_test_error']:.3f} mm"
        )

        # Predict for ALL points (including outliers)
        pred_3d = np.asarray(model.predict(all_pixels), dtype=float)

        # Per-way Kabsch fit (beads only for fitting, then apply to all)
        refined_targets = np.zeros_like(current_targets)
        all_residuals = np.zeros(N_total)

        for way_name in way_names:
            s = way_slices[way_name]
            way_cube = all_cube[s]
            way_pred = pred_3d[s]
            way_inlier = inlier_mask[s]
            way_bead = all_is_bead[s]

            # Kabsch fit uses bead inliers only (cube-frame is reliable)
            bead_inlier = way_bead & way_inlier
            if bead_inlier.sum() < 6:
                print(f"  [WARN] {way_name}: only {bead_inlier.sum()} bead inliers, skipping")
                refined_targets[s] = current_targets[s]
                continue

            R, t = kabsch_rigid_fit(way_cube[bead_inlier], way_pred[bead_inlier])

            # Assign cube-frame positions to corners using the bead-derived
            # rigid transform, then update all_cube so subsequent iterations
            # and the final target computation use the correct values.
            corner_mask_way = ~way_bead
            if corner_mask_way.any():
                corner_tank = current_targets[s][corner_mask_way]
                corner_cube = _assign_corner_cube_frames(corner_tank, R, t)
                all_cube[s][corner_mask_way] = corner_cube
                way_cube = all_cube[s]  # refresh after update

            way_refined = apply_rigid(way_cube, R, t)
            refined_targets[s] = way_refined

            residuals = np.linalg.norm(way_pred - way_refined, axis=1)
            all_residuals[s] = residuals

            euler = rotation_to_euler_deg(R)
            per_way_transforms[way_name] = dict(
                R=R.tolist(),
                t=t.tolist(),
                euler_xyz_deg=list(euler),
            )
            n_way = int(way_bead.sum())
            n_corners_way = int(corner_mask_way.sum())
            pts_desc = f"{n_way} beads"
            if n_corners_way:
                pts_desc += f" + {n_corners_way} corners"
            print(f"  {way_name} ({pts_desc}):")
            print(f"    Rotation (deg): ({euler[0]:+.2f}, {euler[1]:+.2f}, {euler[2]:+.2f})")
            print(f"    Translation (mm): ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f})")
            inlier_res = residuals[way_inlier]
            print(f"    Residual (inliers): mean={inlier_res.mean():.3f}, max={inlier_res.max():.3f} mm")

        # Outlier detection
        inlier_residuals = all_residuals[inlier_mask]
        mu = np.mean(inlier_residuals)
        sigma = np.std(inlier_residuals)
        threshold = mu + outlier_sigma * sigma

        new_outlier_mask = all_residuals > threshold
        newly_flagged = np.where(new_outlier_mask & inlier_mask)[0]
        if len(newly_flagged) > 0:
            print(f"\n  New outliers (residual > {threshold:.2f} mm):")
            for idx in newly_flagged:
                way = _idx_to_way(idx, way_slices, way_names)
                print(f"    [{idx}] {way} | {all_labels[idx]} | residual={all_residuals[idx]:.2f} mm")

        inlier_mask = ~new_outlier_mask
        outlier_indices = np.where(new_outlier_mask)[0].tolist()

        # Convergence check
        target_change = np.mean(np.linalg.norm(refined_targets - current_targets, axis=1))
        print(f"\n  Mean target shift: {target_change:.4f} mm")
        print(f"  Outliers: {new_outlier_mask.sum()}/{N_total}")

        current_targets = refined_targets.copy()

        if target_change < convergence_tol_mm and iteration > 0:
            print(f"  Converged (shift < {convergence_tol_mm} mm)")
            break

    # ------------------------------------------------------------------
    # Phase 3: Final model (all 4 sub-models) on refined targets
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("Final model training (all sub-models)")
    print(f"{'=' * 50}")

    X_final = all_pixels[inlier_mask]
    Y_final = current_targets[inlier_mask]
    img_final = all_image_coords[:, inlier_mask, :]
    n_final = int(inlier_mask.sum())
    print(f"Training on {n_final}/{N_total} points (refined targets, outliers excluded)")

    model_data = prepare_model_data(img_final, Y_final)
    trained_models: Dict[str, object] = {}
    summary_rows: List[dict] = []

    for model_name, (X_data, Y_data) in model_data.items():
        valid = (~np.isnan(X_data).any(axis=1)) & (~np.isnan(Y_data).any(axis=1))
        X_clean, Y_clean = X_data[valid], Y_data[valid]
        if len(X_clean) == 0:
            continue
        print(f"  Training {model_name} ({X_clean.shape[0]} samples) ...")
        mdl, bp, all_res, vr = train_calibration_model(X_clean, Y_clean, **training_kwargs)
        trained_models[model_name] = mdl
        best_res = min(all_res, key=lambda r: r["cv_mean_error"])
        row = dict(
            model_name=model_name,
            num_samples=int(len(X_clean)),
            num_features=int(X_clean.shape[1]),
            output_dim=int(Y_clean.shape[1]),
            best_model_type=bp.get("model_type", "polynomial"),
            cv_mean_error=float(best_res["cv_mean_error"]),
            cv_std_error=float(best_res["cv_std_error"]),
            mean_train_error=float(vr["mean_train_error"]),
            std_train_error=float(vr["std_train_error"]),
            mean_test_error=float(vr["mean_test_error"]),
            std_test_error=float(vr["std_test_error"]),
        )
        if "alpha" in bp:
            row["best_alpha"] = float(bp["alpha"])
        if "degree" in bp:
            row["best_degree"] = int(bp["degree"])
        summary_rows.append(row)

    # ------------------------------------------------------------------
    # Phase 4: Save outputs
    # ------------------------------------------------------------------
    paths = _ensure_project_paths(
        calibration_folder,
        output_folder=output_folder,
        results_root_override=results_root,
    )
    os.makedirs(paths["model_output_folder"], exist_ok=True)
    os.makedirs(paths["results_root"], exist_ok=True)

    save_all_models(trained_models, paths["model_output_folder"])
    print(f"\nModels saved to: {paths['model_output_folder']}")

    way_suffix = "_".join(way_names)
    summary_path = os.path.join(
        paths["results_root"],
        f"{paths['calib_name']}_rigid_{way_suffix}_training_summary.csv",
    )
    _save_training_summary(summary_rows, summary_path)

    # Point-mapping CSV using ALL points (inliers + outliers) with refined targets
    if "imCoords_to_XYZ" in trained_models:
        pred_all = np.asarray(
            trained_models["imCoords_to_XYZ"].predict(all_pixels), dtype=float,
        )
        mapping_csv = _save_point_mapping_and_figures(
            current_targets,
            all_image_coords,
            pred_all,
            paths["results_root"],
            paths["calib_name"] + "_rigid",
            point_labels=all_labels,
        )
        print(f"Point mapping CSV: {mapping_csv}")

        # Error summary
        err_all = np.linalg.norm(pred_all - current_targets, axis=1)
        err_inlier = err_all[inlier_mask]
        print(f"\n  All points:    mean={err_all.mean():.3f}, rmse={np.sqrt(np.mean(err_all**2)):.3f}, max={err_all.max():.3f} mm")
        print(f"  Inliers only:  mean={err_inlier.mean():.3f}, rmse={np.sqrt(np.mean(err_inlier**2)):.3f}, max={err_inlier.max():.3f} mm")

    # Save refinement report
    report = dict(
        way_names=way_names,
        n_total=N_total,
        n_inliers=int(inlier_mask.sum()),
        n_outliers=int((~inlier_mask).sum()),
        outlier_sigma=outlier_sigma,
        iterations_run=iteration + 1,
        per_way_transforms=per_way_transforms,
        outliers=[
            dict(
                index=int(i),
                way=_idx_to_way(i, way_slices, way_names),
                label=all_labels[i],
                residual_mm=float(all_residuals[i]),
            )
            for i in outlier_indices
        ],
        original_error=dict(
            mean=float(np.mean(np.linalg.norm(
                all_targets_original - np.asarray(
                    trained_models["imCoords_to_XYZ"].predict(all_pixels), dtype=float,
                ) if "imCoords_to_XYZ" in trained_models else all_targets_original,
                axis=1,
            ))),
        ),
    )
    report_path = os.path.join(
        paths["results_root"],
        f"{paths['calib_name']}_rigid_refinement_report.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Refinement report: {report_path}")

    print(f"\n{'=' * 70}")
    print("Rigid body refinement complete!")
    print(f"{'=' * 70}")
    return report


def _idx_to_way(idx: int, way_slices: Dict[str, slice], way_names: List[str]) -> str:
    for way_name in way_names:
        s = way_slices[way_name]
        if s.start <= idx < s.stop:
            return way_name
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine merged calibration using rigid body constraints",
    )
    parser.add_argument(
        "--calibration_folder", type=str, required=True,
        help="Calibration folder containing the 3D coordinate template",
    )
    parser.add_argument(
        "--image_coords_folder", type=str, required=True,
        help="Folder containing manually annotated image-coordinate txt files",
    )
    parser.add_argument(
        "--output_folder", type=str, default=None,
        help="Model output folder. Default: models/calibration/<calib_name>/",
    )
    parser.add_argument(
        "--results_root", type=str, default=None,
        help="Root folder for QC CSVs and figures. Default: infer from calibration_folder.",
    )
    parser.add_argument(
        "--way_names", type=str, nargs="+", default=["way1", "way2", "way3", "way4"],
        help="Way names to include (default: way1 way2 way3 way4)",
    )
    parser.add_argument(
        "--beads_only", action="store_true",
        help="Use only bead points (exclude corners)",
    )
    parser.add_argument(
        "--outlier_sigma", type=float, default=2.5,
        help="Outlier threshold in standard deviations of rigid-fit residual (default: 2.5)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=3,
        help="Maximum number of refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--convergence_tol", type=float, default=0.01,
        help="Convergence tolerance in mm for mean target shift (default: 0.01)",
    )
    parser.add_argument(
        "--alpha_range", type=float, nargs="+",
        default=[1e-10, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1],
        help="Alpha values for Ridge regression grid search",
    )
    parser.add_argument(
        "--poly_deg_range", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Polynomial degrees for grid search",
    )
    parser.add_argument(
        "--num_permutations", type=int, default=100,
        help="Number of train/test splits for validation (default: 100)",
    )
    parser.add_argument(
        "--model_types", type=str, nargs="+", default=None,
        help="Model types to try (default: all)",
    )
    args = parser.parse_args()

    run_rigid_refinement(
        calibration_folder=args.calibration_folder,
        image_coords_folder=args.image_coords_folder,
        way_names=args.way_names,
        beads_only=args.beads_only,
        outlier_sigma=args.outlier_sigma,
        max_iterations=args.max_iterations,
        convergence_tol_mm=args.convergence_tol,
        output_folder=args.output_folder,
        results_root=args.results_root,
        alpha_range=args.alpha_range,
        poly_deg_range=args.poly_deg_range,
        num_permutations=args.num_permutations,
        model_types=args.model_types,
    )


if __name__ == "__main__":
    main()
