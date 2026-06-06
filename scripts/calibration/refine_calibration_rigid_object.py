from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.calibration.lib.calibration_core import (
    _save_point_mapping_and_figures,
    _save_training_summary,
)
from scripts.calibration.lib.model_training import (
    prepare_model_data,
    save_all_models,
    train_calibration_model,
)
from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration


VIEW_COLUMNS = ["cam0_x", "cam0_y", "cam1_x", "cam1_y", "cam2_x", "cam2_y"]


def run_rigid_object_refinement(
    *,
    rigid_object_geometry_csv: str | Path,
    rigid_object_image_coords_csv: str | Path,
    base_model_folder: str | Path,
    output_folder: str | Path,
    results_root: str | Path,
    min_markers_per_pose: int = 4,
    outlier_threshold_mm: float | None = None,
    holdout_fraction: float = 0.2,
    alpha_range: Iterable[float] = (1e-6, 1e-4, 1e-3, 1e-2),
    poly_deg_range: Iterable[int] = (1, 2, 3),
    train_fraction: float = 0.9,
    num_permutations: int = 20,
    model_types: Iterable[str] | None = ("polynomial",),
) -> dict:
    """Refine calibration targets from arbitrary moved rigid-object poses.

    The object-local marker coordinates are known. The current calibration
    supplies provisional world coordinates for each observed marker; Kabsch
    fitting then projects those noisy coordinates back onto a physically rigid
    pose before retraining the standard calibration models.
    """
    geometry = load_rigid_object_geometry(rigid_object_geometry_csv)
    observations = load_rigid_object_image_coords(rigid_object_image_coords_csv)
    merged = observations.merge(geometry, on="marker_id", how="inner")
    if merged.empty:
        raise ValueError("No image-coordinate rows matched rigid object marker IDs")
    object_xyz = geometry[["object_x", "object_y", "object_z"]].to_numpy(float)
    rank = int(np.linalg.matrix_rank(object_xyz - object_xyz.mean(axis=0)))
    if rank < 2:
        raise ValueError("Rigid object geometry is degenerate; use non-collinear markers")

    calib = Calibration(base_model_folder)
    refined = refine_observed_rigid_object_targets(
        merged,
        calib=calib,
        min_markers_per_pose=min_markers_per_pose,
        outlier_threshold_mm=outlier_threshold_mm,
    )
    if refined.empty:
        raise ValueError("No rigid-object pose had enough valid markers")

    refined = _assign_pose_holdout(refined, holdout_fraction=holdout_fraction)
    train_rows = refined[refined["pose_inlier"] & ~refined["heldout_pose"]].copy()
    if train_rows.empty:
        raise ValueError("No inlier training rows remained after holdout split")

    image_coords = _image_coords_array(train_rows)
    targets = train_rows[["refined_X", "refined_Y", "refined_Z"]].to_numpy(float)
    trained_models, summary_rows = train_rigid_object_models(
        image_coords,
        targets,
        alpha_range=alpha_range,
        poly_deg_range=poly_deg_range,
        train_fraction=train_fraction,
        num_permutations=num_permutations,
        model_types=model_types,
    )

    output_folder = Path(output_folder)
    results_root = Path(results_root)
    output_folder.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    save_all_models(trained_models, output_folder)
    _save_training_summary(
        summary_rows,
        str(results_root / "rigid_object_training_summary.csv"),
    )

    mapping_csv = ""
    if "imCoords_to_XYZ" in trained_models:
        pred_3d = np.asarray(
            trained_models["imCoords_to_XYZ"].predict(
                image_coords.swapaxes(0, 1).reshape(targets.shape[0], -1)
            ),
            dtype=float,
        )
        labels = [
            f"{row.pose_id}:{row.marker_id}"
            for row in train_rows.itertuples(index=False)
        ]
        mapping_csv = _save_point_mapping_and_figures(
            targets,
            image_coords,
            pred_3d,
            str(results_root),
            "rigid_object",
            point_labels=labels,
        )

    refined_path = results_root / "rigid_object_refined_points.csv"
    refined.to_csv(refined_path, index=False)
    report = _refinement_report(
        refined,
        geometry_rank=rank,
        output_folder=output_folder,
        results_root=results_root,
        mapping_csv=mapping_csv,
    )
    report_path = results_root / "calibration_refinement_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    report["refined_points_csv"] = str(refined_path)
    return report


def load_rigid_object_geometry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    marker_col = _first_existing(df, ["marker_id", "label", "point_id"])
    x_col = _first_existing(df, ["object_x", "x", "X"])
    y_col = _first_existing(df, ["object_y", "y", "Y"])
    z_col = _first_existing(df, ["object_z", "z", "Z"])
    out = df[[marker_col, x_col, y_col, z_col]].copy()
    out.columns = ["marker_id", "object_x", "object_y", "object_z"]
    out["marker_id"] = out["marker_id"].astype(str)
    for col in ["object_x", "object_y", "object_z"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["object_x", "object_y", "object_z"]).drop_duplicates("marker_id")
    if len(out) < 3:
        raise ValueError("Rigid object geometry needs at least 3 valid markers")
    return out.reset_index(drop=True)


def load_rigid_object_image_coords(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pose_id", "marker_id", *VIEW_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rigid object image coords missing columns: {', '.join(sorted(missing))}")
    out = df[["pose_id", "marker_id", *VIEW_COLUMNS]].copy()
    out["pose_id"] = out["pose_id"].astype(str)
    out["marker_id"] = out["marker_id"].astype(str)
    for col in VIEW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=VIEW_COLUMNS).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rigid object image-coordinate rows")
    return out


def refine_observed_rigid_object_targets(
    observations: pd.DataFrame,
    *,
    calib: Calibration,
    min_markers_per_pose: int = 4,
    outlier_threshold_mm: float | None = None,
) -> pd.DataFrame:
    work = observations.copy()
    features = work[VIEW_COLUMNS].to_numpy(float)
    predicted = np.asarray(calib._model_im2xyz.predict(features), dtype=float)
    work[["initial_X", "initial_Y", "initial_Z"]] = predicted

    refined_parts: list[pd.DataFrame] = []
    for pose_id, group in work.groupby("pose_id", sort=True):
        pose = group.copy()
        if len(pose) < int(min_markers_per_pose):
            continue
        pose = _fit_pose_rows(pose, pose_id=str(pose_id), outlier_threshold_mm=outlier_threshold_mm)
        if not pose.empty:
            refined_parts.append(pose)
    if not refined_parts:
        return pd.DataFrame()
    return pd.concat(refined_parts, ignore_index=True)


def train_rigid_object_models(
    image_coords: np.ndarray,
    targets: np.ndarray,
    *,
    alpha_range: Iterable[float],
    poly_deg_range: Iterable[int],
    train_fraction: float,
    num_permutations: int,
    model_types: Iterable[str] | None,
) -> tuple[dict[str, object], list[dict]]:
    model_data = prepare_model_data(image_coords, targets)
    flat_image = image_coords.swapaxes(0, 1).reshape(targets.shape[0], -1)
    model_data["XYZ_to_imCoords"] = (targets, flat_image)

    trained_models: dict[str, object] = {}
    summary_rows: list[dict] = []
    for model_name, (X_data, Y_data) in model_data.items():
        valid = (~np.isnan(X_data).any(axis=1)) & (~np.isnan(Y_data).any(axis=1))
        X_clean, Y_clean = X_data[valid], Y_data[valid]
        if len(X_clean) < 3:
            continue
        model, best_params, all_results, val_results = train_calibration_model(
            X_clean,
            Y_clean,
            alpha_range=alpha_range,
            poly_deg_range=poly_deg_range,
            cv_folds=3,
            train_fraction=train_fraction,
            num_permutations=num_permutations,
            model_types=model_types,
        )
        trained_models[model_name] = model
        best = min(all_results, key=lambda row: row["cv_mean_error"])
        row = {
            "model_name": model_name,
            "num_samples": int(len(X_clean)),
            "num_features": int(X_clean.shape[1]),
            "output_dim": int(Y_clean.shape[1]),
            "best_model_type": best_params.get("model_type", "polynomial"),
            "cv_mean_error": float(best["cv_mean_error"]),
            "cv_std_error": float(best["cv_std_error"]),
            "mean_train_error": float(val_results["mean_train_error"]),
            "std_train_error": float(val_results["std_train_error"]),
            "mean_test_error": float(val_results["mean_test_error"]),
            "std_test_error": float(val_results["std_test_error"]),
        }
        if "alpha" in best_params:
            row["best_alpha"] = float(best_params["alpha"])
        if "degree" in best_params:
            row["best_degree"] = int(best_params["degree"])
        summary_rows.append(row)
    return trained_models, summary_rows


def _fit_pose_rows(
    pose: pd.DataFrame,
    *,
    pose_id: str,
    outlier_threshold_mm: float | None,
) -> pd.DataFrame:
    object_xyz = pose[["object_x", "object_y", "object_z"]].to_numpy(float)
    initial_xyz = pose[["initial_X", "initial_Y", "initial_Z"]].to_numpy(float)
    if len(pose) < 3:
        return pd.DataFrame()
    R, t = kabsch_rigid_fit(object_xyz, initial_xyz)
    refined = apply_rigid(object_xyz, R, t)
    residual = np.linalg.norm(refined - initial_xyz, axis=1)
    inlier = np.ones(len(pose), dtype=bool)
    if outlier_threshold_mm is not None:
        inlier = residual <= float(outlier_threshold_mm)
        if int(inlier.sum()) >= 3 and int(inlier.sum()) < len(pose):
            R, t = kabsch_rigid_fit(object_xyz[inlier], initial_xyz[inlier])
            refined = apply_rigid(object_xyz, R, t)
            residual = np.linalg.norm(refined - initial_xyz, axis=1)
            inlier = residual <= float(outlier_threshold_mm)

    out = pose.copy()
    out["pose_id"] = pose_id
    out[["refined_X", "refined_Y", "refined_Z"]] = refined
    out["pose_residual_mm"] = residual
    out["pose_inlier"] = inlier
    out["pose_rotation_det"] = float(np.linalg.det(R))
    out["pose_translation_x"] = float(t[0])
    out["pose_translation_y"] = float(t[1])
    out["pose_translation_z"] = float(t[2])
    return out


def _image_coords_array(rows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            rows[["cam0_x", "cam0_y"]].to_numpy(float),
            rows[["cam1_x", "cam1_y"]].to_numpy(float),
            rows[["cam2_x", "cam2_y"]].to_numpy(float),
        ],
        axis=0,
    )


def _assign_pose_holdout(rows: pd.DataFrame, *, holdout_fraction: float) -> pd.DataFrame:
    out = rows.copy()
    poses = sorted(out["pose_id"].astype(str).unique())
    n_holdout = int(round(len(poses) * float(holdout_fraction)))
    if n_holdout <= 0 or n_holdout >= len(poses):
        heldout = set()
    else:
        heldout = set(poses[-n_holdout:])
    out["heldout_pose"] = out["pose_id"].astype(str).isin(heldout)
    return out


def _refinement_report(
    rows: pd.DataFrame,
    *,
    geometry_rank: int,
    output_folder: Path,
    results_root: Path,
    mapping_csv: str,
) -> dict:
    inliers = rows[rows["pose_inlier"]]
    heldout = rows[rows["heldout_pose"]]
    return {
        "n_rows": int(len(rows)),
        "n_inliers": int(len(inliers)),
        "n_poses": int(rows["pose_id"].nunique()),
        "n_heldout_rows": int(len(heldout)),
        "geometry_rank": int(geometry_rank),
        "mean_pose_residual_mm": float(rows["pose_residual_mm"].mean()),
        "max_pose_residual_mm": float(rows["pose_residual_mm"].max()),
        "mean_inlier_pose_residual_mm": float(inliers["pose_residual_mm"].mean()) if not inliers.empty else None,
        "mean_heldout_pose_residual_mm": float(heldout["pose_residual_mm"].mean()) if not heldout.empty else None,
        "output_folder": str(output_folder),
        "results_root": str(results_root),
        "mapping_csv": mapping_csv,
    }


def _first_existing(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"CSV missing one of columns: {', '.join(names)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine calibration from arbitrary moved rigid-object marker poses."
    )
    parser.add_argument("--rigid-object-geometry-csv", required=True, type=Path)
    parser.add_argument("--rigid-object-image-coords-csv", required=True, type=Path)
    parser.add_argument("--base-model-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--min-markers-per-pose", type=int, default=4)
    parser.add_argument("--outlier-threshold-mm", type=float, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--alpha-range", type=float, nargs="+", default=[1e-6, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--poly-deg-range", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--num-permutations", type=int, default=20)
    parser.add_argument("--model-types", nargs="+", default=["polynomial"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_rigid_object_refinement(
        rigid_object_geometry_csv=args.rigid_object_geometry_csv,
        rigid_object_image_coords_csv=args.rigid_object_image_coords_csv,
        base_model_folder=args.base_model_folder,
        output_folder=args.output_folder,
        results_root=args.results_root,
        min_markers_per_pose=args.min_markers_per_pose,
        outlier_threshold_mm=args.outlier_threshold_mm,
        holdout_fraction=args.holdout_fraction,
        alpha_range=args.alpha_range,
        poly_deg_range=args.poly_deg_range,
        train_fraction=args.train_fraction,
        num_permutations=args.num_permutations,
        model_types=args.model_types,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
