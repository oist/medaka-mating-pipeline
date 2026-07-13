from __future__ import annotations

"""Export the full-set projective DLT comparator as a reconstruction bundle.

The full-set calibration fits DLT cameras as the initialization for the
refractive bundle, but historically only wrote the refractive JSON files.  This
command deterministically rebuilds those same DLT cameras from the saved
training poses and writes the two joblib estimators consumed by
``reconstruct_from_slp``.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from scripts.calibration.dlt_models import DLTProjectionModel, DLTTriangulationModel
from scripts.calibration.fit_refractive_bundle_adjustment import _ball_pose_initializations
from scripts.calibration.fit_refractive_rigid_object import evaluate_predictor, summarize_model
from scripts.calibration.refractive_geometry import initialize_flat_port_camera
from scripts.reconstruction_3d.calibration import Calibration


def _geometry_lookup(path: Path) -> dict[int, np.ndarray]:
    geometry = pd.read_csv(path)
    required = {"marker_id", "object_x", "object_y", "object_z"}
    missing = sorted(required.difference(geometry.columns))
    if missing:
        raise ValueError(f"geometry CSV is missing columns: {missing}")
    return {
        int(row.marker_id): np.array([row.object_x, row.object_y, row.object_z], dtype=float)
        for row in geometry.itertuples(index=False)
    }


def build_dlt_cameras(
    training_coordinates: pd.DataFrame,
    geometry: dict[int, np.ndarray],
    base_calibration: Calibration,
    *,
    tank_bounds: np.ndarray,
):
    _states, targets = _ball_pose_initializations(
        training_coordinates, geometry, base_calibration
    )
    world = targets[["X", "Y", "Z"]].to_numpy(float)
    cameras = []
    for index in range(3):
        image = targets[[f"cam{index}_x", f"cam{index}_y"]].to_numpy(float)
        cameras.append(
            initialize_flat_port_camera(
                f"cam{index}",
                world,
                image,
                tank_bounds=tank_bounds,
                layer_count=0,
                glass_thickness_mm=0.0,
            )
        )
    return cameras


def _predictor(model: DLTTriangulationModel):
    def predict(image_points: np.ndarray) -> tuple[np.ndarray, float]:
        xyz = model.predict(np.asarray(image_points, dtype=float).reshape(1, 6))[0]
        return xyz, float("nan")

    return predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-coordinates-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--base-model-folder", type=Path, required=True)
    parser.add_argument(
        "--registration-model-folder",
        type=Path,
        default=None,
        help=(
            "Optional bundle whose XYZ_to_imCoords.joblib is retained for candidate "
            "assignment scoring. Omit to use the internally coherent DLT projection."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-span-mm", type=float, default=192.0)
    parser.add_argument("--water-height-mm", type=float, default=192.0)
    parser.add_argument("--all-image-coords-csv", type=Path, default=None)
    parser.add_argument("--heldout-pose-list", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.all_image_coords_csv is None) != (args.heldout_pose_list is None):
        raise ValueError(
            "--all-image-coords-csv and --heldout-pose-list must be provided together"
        )

    training = pd.read_csv(args.training_coordinates_csv)
    training["pose_id"] = training["pose_id"].astype(str)
    geometry = _geometry_lookup(args.geometry_csv)
    base = Calibration(args.base_model_folder)
    bounds = np.array(
        [[0.0, args.inner_span_mm], [0.0, args.inner_span_mm], [0.0, args.water_height_mm]],
        dtype=float,
    )
    cameras = build_dlt_cameras(training, geometry, base, tank_bounds=bounds)
    triangulation = DLTTriangulationModel(cameras)
    projection = DLTProjectionModel(cameras)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(triangulation, args.output_dir / "imCoords_to_XYZ.joblib")
    registration_source = "pinhole_dlt"
    if args.registration_model_folder is None:
        joblib.dump(projection, args.output_dir / "XYZ_to_imCoords.joblib")
    else:
        registration_calibration = Calibration(args.registration_model_folder)
        registration_path = registration_calibration.model_dir / "XYZ_to_imCoords.joblib"
        if not registration_path.exists():
            raise FileNotFoundError(
                f"registration model is missing XYZ_to_imCoords.joblib: {registration_path}"
            )
        shutil.copy2(registration_path, args.output_dir / "XYZ_to_imCoords.joblib")
        registration_source = str(args.registration_model_folder)
    pairwise_helpers = (
        "xy_yz_to_xz.joblib",
        "xz_yz_to_xy.joblib",
        "xz_xy_to_yz.joblib",
    )
    copied_helpers = []
    for filename in pairwise_helpers:
        source = base.model_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)
            copied_helpers.append(filename)
    for camera in cameras:
        camera.to_json(args.output_dir / f"{camera.name}_pinhole_dlt.json")

    evaluation = None
    if args.all_image_coords_csv is not None:
        all_coords = pd.read_csv(args.all_image_coords_csv)
        all_coords["pose_id"] = all_coords["pose_id"].astype(str)
        heldout_ids = {
            line.strip()
            for line in args.heldout_pose_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        heldout = all_coords[all_coords["pose_id"].isin(heldout_ids)].copy()
        markers, pairs = evaluate_predictor(
            heldout, geometry, _predictor(triangulation), model_name="pinhole_dlt"
        )
        evaluation = summarize_model(markers, pairs)
        markers.to_csv(args.output_dir / "heldout_marker_errors.csv", index=False)
        pairs.to_csv(args.output_dir / "heldout_pairwise_errors.csv", index=False)

    metadata = {
        "model_type": "projective_dlt",
        "camera_count": 3,
        "training_pose_count": int(training["pose_id"].nunique()),
        "training_marker_observations": int(len(training)),
        "tank_bounds_mm": bounds.tolist(),
        "source_training_coordinates": str(args.training_coordinates_csv),
        "source_geometry": str(args.geometry_csv),
        "source_base_model": str(args.base_model_folder),
        "assignment_registration_model": registration_source,
        "unchanged_pairwise_helpers": copied_helpers,
        "cameras": {camera.name: asdict(camera) for camera in cameras},
        "heldout_evaluation": evaluation,
        "scope": (
            "Held-out rigid-object metrics validate shape/scale. Round7 fish data can "
            "test downstream consistency and assignment robustness, but cannot provide "
            "absolute 3D accuracy without fish-position ground truth."
        ),
    }
    (args.output_dir / "calibration_bundle.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
