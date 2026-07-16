"""Joint refractive bundle adjustment for wet checkerboard and 12-ball data.

Dry intrinsics are fixed.  The first stage fits camera extrinsics and one rigid
pose per wet object placement with surveyed interface planes fixed.  The second
stage releases small plane-normal/offset corrections under priors.  A complete
12-ball pose split is held out for a head-to-head comparison with the existing
regression calibrations.

Because water height was not measured, the top-interface height and absolute Z
gauge are provisional.  One training cube pose is held fixed solely to remove
the bundle gauge; this does not enter held-out shape/scale scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from scripts.calibration.fit_refractive_rigid_object import (
    _geometry_lookup,
    _ray_predictor,
    _regression_predictor,
    evaluate_predictor,
    plot_comparison,
    summarize_model,
)
from scripts.calibration.refractive_geometry import (
    FlatPortCamera,
    fit_flat_port_camera_fixed_intrinsics,
    initialize_flat_port_camera,
    normalize_vectors,
    triangulate_pixels,
)
from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration


CAMERA_VIEWS = ("top_cam", "side1_cam", "side2_cam")
CAMERA_INDEX = {name: index for index, name in enumerate(CAMERA_VIEWS)}


@dataclass
class PoseState:
    rotation_vector: np.ndarray
    translation: np.ndarray


@dataclass
class BundleDiagnostics:
    success: bool
    message: str
    nfev: int
    cost: float
    optimality: float
    geometric_rms_mm: float


def summarize_bundle_ray_residuals(
    cameras: Sequence[FlatPortCamera],
    poses: dict[str, PoseState],
    observation_data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
    ],
) -> dict[str, object]:
    """Summarize unweighted point-to-ray distances for the fitted bundle."""

    camera_index, observation_pose, object_points, pixels, _weights, pose_names = (
        observation_data
    )
    world = np.empty_like(object_points)
    pose_kind = np.empty(len(object_points), dtype=object)
    for pose_index, name in enumerate(pose_names):
        mask = observation_pose == pose_index
        rotation = cv2.Rodrigues(poses[name].rotation_vector)[0]
        world[mask] = (rotation @ object_points[mask].T).T + poses[name].translation
        pose_kind[mask] = name.split(":", 1)[0]

    distances = np.empty(len(object_points), dtype=float)
    for index, camera in enumerate(cameras):
        mask = camera_index == index
        residual = camera.point_to_ray_residuals(pixels[mask], world[mask])
        distances[mask] = np.linalg.norm(residual, axis=1)

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        values = distances[mask]
        return {
            "n": int(len(values)),
            "mean_mm": float(np.mean(values)),
            "rms_mm": float(np.sqrt(np.mean(values**2))),
            "median_mm": float(np.median(values)),
            "p95_mm": float(np.percentile(values, 95)),
            "maximum_mm": float(np.max(values)),
        }

    by_camera: dict[str, object] = {}
    for index, view in enumerate(CAMERA_VIEWS):
        camera_mask = camera_index == index
        by_camera[view] = {
            "all": summarize(camera_mask),
            **{
                kind: summarize(camera_mask & (pose_kind == kind))
                for kind in ("ball", "board")
                if np.any(camera_mask & (pose_kind == kind))
            },
        }
    return {"all": summarize(np.ones(len(distances), dtype=bool)), "by_camera": by_camera}


def _normal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize_vectors(np.asarray(normal).reshape(1, 3))[0]
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = normalize_vectors(np.cross(normal, reference).reshape(1, 3))[0]
    second = normalize_vectors(np.cross(normal, first).reshape(1, 3))[0]
    return first, second


def _distortion_fields(values: np.ndarray) -> dict[str, float]:
    d = np.pad(np.asarray(values, dtype=float).reshape(-1), (0, 8))[:8]
    return {
        "k1": float(d[0]),
        "k2": float(d[1]),
        "p1": float(d[2]),
        "p2": float(d[3]),
        "k3": float(d[4]),
        "k4": float(d[5]),
        "k5": float(d[6]),
        "k6": float(d[7]),
    }


def _physical_camera(
    index: int,
    seed: FlatPortCamera,
    intrinsics_dir: Path,
    *,
    inner_span_mm: float,
    water_height_mm: float,
    glass_thickness_mm: float,
    index_glass: float,
    index_water: float,
) -> FlatPortCamera:
    data = np.load(intrinsics_dir / f"intrinsics_{CAMERA_VIEWS[index]}.npz")
    k = np.asarray(data["K"], dtype=float)
    if index == 0:
        normal = [0.0, 0.0, -1.0]
        point = [0.0, 0.0, water_height_mm]
        layers = 1
    elif index == 1:
        normal = [0.0, 1.0, 0.0]
        point = [0.0, -glass_thickness_mm, 0.0]
        layers = 2
    else:
        normal = [-1.0, 0.0, 0.0]
        point = [inner_span_mm + glass_thickness_mm, 0.0, 0.0]
        layers = 2
    return FlatPortCamera(
        name=f"cam{index}",
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        rotation_vector=seed.rotation_vector,
        camera_center=seed.camera_center,
        interface_normal=normal,
        interface_point=point,
        glass_thickness_mm=glass_thickness_mm,
        index_glass=index_glass,
        index_water=index_water,
        layer_count=layers,
        **_distortion_fields(data["distortion"]),
    )


def _split_pose_ids(values: Sequence[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(set(map(str, values)))
    heldout = {pose for index, pose in enumerate(ordered) if index % 5 == 2}
    return set(ordered).difference(heldout), heldout


def _split_ball_pose_clusters(
    values: Sequence[str], cluster_gap_frames: int
) -> tuple[set[str], set[str], int]:
    """Keep temporally adjacent cube frames on the same side of the split."""
    ids = sorted(set(map(str, values)), key=lambda value: int(value.removeprefix("f")))
    clusters: list[list[str]] = []
    for pose_id in ids:
        frame = int(pose_id.removeprefix("f"))
        if not clusters or frame - int(clusters[-1][-1].removeprefix("f")) > cluster_gap_frames:
            clusters.append([pose_id])
        else:
            clusters[-1].append(pose_id)
    heldout = {
        pose_id
        for cluster_index, cluster in enumerate(clusters)
        if cluster_index % 5 == 2
        for pose_id in cluster
    }
    return set(ids).difference(heldout), heldout, len(clusters)


def _ball_pose_initializations(
    coordinates: pd.DataFrame,
    geometry: dict[int, np.ndarray],
    calibration: Calibration,
) -> tuple[dict[str, PoseState], pd.DataFrame]:
    states: dict[str, PoseState] = {}
    target_rows: list[dict[str, float | str | int]] = []
    for pose_id, group in coordinates.groupby("pose_id", sort=True):
        object_points = np.asarray([geometry[int(marker)] for marker in group["marker_id"]])
        images = group[[f"cam{i}_{axis}" for i in range(3) for axis in ("x", "y")]].to_numpy(float)
        predicted = np.asarray(calibration._model_im2xyz.predict(images), dtype=float)
        rotation, translation = kabsch_rigid_fit(object_points, predicted)
        states[f"ball:{pose_id}"] = PoseState(cv2.Rodrigues(rotation)[0].reshape(3), translation)
        targets = apply_rigid(object_points, rotation, translation)
        for row, target in zip(group.itertuples(index=False), targets, strict=True):
            target_rows.append(
                {
                    "pose_id": str(pose_id),
                    "marker_id": int(row.marker_id),
                    "X": float(target[0]),
                    "Y": float(target[1]),
                    "Z": float(target[2]),
                    **{f"cam{i}_{axis}": float(getattr(row, f"cam{i}_{axis}")) for i in range(3) for axis in ("x", "y")},
                }
            )
    return states, pd.DataFrame(target_rows)


def _checkerboard_pose_initializations(
    corners: pd.DataFrame, cameras: Sequence[FlatPortCamera]
) -> dict[str, PoseState]:
    states: dict[str, PoseState] = {}
    for pose_id, pose in corners.groupby("pose_id", sort=True):
        object_points: list[np.ndarray] = []
        triangulated: list[np.ndarray] = []
        for _, corner in pose.groupby("corner_id", sort=True):
            if len(corner) < 2:
                continue
            indices = [CAMERA_INDEX[str(view)] for view in corner["camera"]]
            pixels = corner[["x", "y"]].to_numpy(float)
            try:
                point, _ = triangulate_pixels([cameras[i] for i in indices], pixels)
            except ValueError:
                continue
            first = corner.iloc[0]
            object_points.append(np.array([first.board_x_mm, first.board_y_mm, first.board_z_mm]))
            triangulated.append(point)
        if len(object_points) < 6:
            continue
        rotation, translation = kabsch_rigid_fit(np.asarray(object_points), np.asarray(triangulated))
        states[f"board:{pose_id}"] = PoseState(cv2.Rodrigues(rotation)[0].reshape(3), translation)
    return states


def _observations(
    ball: pd.DataFrame,
    board: pd.DataFrame | None,
    geometry: dict[int, np.ndarray],
    valid_pose_keys: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    camera_indices: list[int] = []
    pose_keys: list[str] = []
    object_points: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    weights: list[float] = []
    for row in ball.itertuples(index=False):
        key = f"ball:{row.pose_id}"
        if key not in valid_pose_keys:
            continue
        for index in range(3):
            camera_indices.append(index)
            pose_keys.append(key)
            object_points.append(geometry[int(row.marker_id)])
            pixels.append(np.array([getattr(row, f"cam{index}_x"), getattr(row, f"cam{index}_y")]))
            weights.append(1.0)
    if board is not None:
        for row in board.itertuples(index=False):
            key = f"board:{row.pose_id}"
            if key not in valid_pose_keys:
                continue
            camera_indices.append(CAMERA_INDEX[str(row.camera)])
            pose_keys.append(key)
            object_points.append(np.array([row.board_x_mm, row.board_y_mm, row.board_z_mm]))
            pixels.append(np.array([row.x, row.y]))
            weights.append(0.55)
    pose_names = sorted(set(pose_keys))
    pose_index = {key: index for index, key in enumerate(pose_names)}
    return (
        np.asarray(camera_indices, dtype=int),
        np.asarray([pose_index[key] for key in pose_keys], dtype=int),
        np.asarray(object_points, dtype=float),
        np.asarray(pixels, dtype=float),
        np.asarray(weights, dtype=float),
        pose_names,
    )


def fit_bundle_stage(
    cameras: Sequence[FlatPortCamera],
    poses: dict[str, PoseState],
    observation_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
    *,
    anchor_pose: str,
    release_interface: bool,
    max_nfev: int,
) -> tuple[list[FlatPortCamera], dict[str, PoseState], BundleDiagnostics]:
    camera_index, observation_pose, object_points, pixels, weights, pose_names = observation_data
    camera_width = 9 if release_interface else 6
    pose_names_free = [name for name in pose_names if name != anchor_pose]
    pose_parameter_index = {name: 3 * camera_width + 6 * i for i, name in enumerate(pose_names_free)}
    parameter0: list[float] = []
    for camera in cameras:
        parameter0.extend(camera.rotation_vector)
        parameter0.extend(camera.camera_center)
        if release_interface:
            parameter0.extend([0.0, 0.0, 0.0])
    for name in pose_names_free:
        parameter0.extend(poses[name].rotation_vector.tolist())
        parameter0.extend(poses[name].translation.tolist())
    parameter0_array = np.asarray(parameter0, dtype=float)
    initial_centers = [np.asarray(camera.camera_center, dtype=float) for camera in cameras]
    initial_normals = [normalize_vectors(np.asarray(camera.interface_normal).reshape(1, 3))[0] for camera in cameras]
    normal_bases = [_normal_basis(normal) for normal in initial_normals]
    initial_points = [np.asarray(camera.interface_point, dtype=float) for camera in cameras]

    def unpack(parameters: np.ndarray) -> tuple[list[FlatPortCamera], dict[str, PoseState]]:
        output_cameras: list[FlatPortCamera] = []
        for index, initial in enumerate(cameras):
            start = index * camera_width
            if release_interface:
                first, second = normal_bases[index]
                normal = normalize_vectors(
                    (initial_normals[index] + parameters[start + 6] * first + parameters[start + 7] * second).reshape(1, 3)
                )[0]
                point = initial_points[index] + parameters[start + 8] * initial_normals[index]
            else:
                normal = initial_normals[index]
                point = initial_points[index]
            values = asdict(initial)
            values.update(
                rotation_vector=parameters[start : start + 3].tolist(),
                camera_center=parameters[start + 3 : start + 6].tolist(),
                interface_normal=normal.tolist(),
                interface_point=point.tolist(),
            )
            output_cameras.append(FlatPortCamera(**values))
        output_poses = {anchor_pose: poses[anchor_pose]}
        for name in pose_names_free:
            start = pose_parameter_index[name]
            output_poses[name] = PoseState(parameters[start : start + 3], parameters[start + 3 : start + 6])
        return output_cameras, output_poses

    def geometric(parameters: np.ndarray) -> np.ndarray:
        current_cameras, current_poses = unpack(parameters)
        world = np.empty_like(object_points)
        for pose_index, name in enumerate(pose_names):
            mask = observation_pose == pose_index
            rotation = cv2.Rodrigues(current_poses[name].rotation_vector)[0]
            world[mask] = (rotation @ object_points[mask].T).T + current_poses[name].translation
        origins = np.empty_like(world)
        directions = np.empty_like(world)
        for index, camera in enumerate(current_cameras):
            mask = camera_index == index
            origins[mask], directions[mask] = camera.backproject(pixels[mask])
        offsets = world - origins
        closest = origins + np.sum(offsets * directions, axis=1)[:, None] * directions
        return ((world - closest) * np.sqrt(weights)[:, None]).ravel()

    def residuals(parameters: np.ndarray) -> np.ndarray:
        try:
            result = geometric(parameters)
        except ValueError:
            result = np.full(object_points.size, 1e4)
        priors: list[float] = []
        for index in range(3):
            start = index * camera_width
            priors.extend(((parameters[start + 3 : start + 6] - initial_centers[index]) / 250.0).tolist())
            if release_interface:
                priors.extend(
                    [
                        parameters[start + 6] / 0.025,
                        parameters[start + 7] / 0.025,
                        parameters[start + 8] / (25.0 if index == 0 else 2.0),
                    ]
                )
        return np.concatenate((result, np.asarray(priors)))

    n_geometric = object_points.size
    n_priors = 3 * (6 if release_interface else 3)
    sparsity = lil_matrix((n_geometric + n_priors, len(parameter0_array)), dtype=int)
    for obs_index, (cam, pose_index) in enumerate(zip(camera_index, observation_pose, strict=True)):
        rows = slice(3 * obs_index, 3 * obs_index + 3)
        camera_start = int(cam) * camera_width
        sparsity[rows, camera_start : camera_start + camera_width] = 1
        pose_name = pose_names[int(pose_index)]
        if pose_name != anchor_pose:
            pose_start = pose_parameter_index[pose_name]
            sparsity[rows, pose_start : pose_start + 6] = 1
    prior_row = n_geometric
    for index in range(3):
        start = index * camera_width
        sparsity[prior_row : prior_row + 3, start + 3 : start + 6] = 1
        prior_row += 3
        if release_interface:
            sparsity[prior_row : prior_row + 3, start + 6 : start + 9] = 1
            prior_row += 3
    result = least_squares(
        residuals,
        parameter0_array,
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=1,
    )
    fitted_cameras, fitted_poses = unpack(result.x)
    geo = geometric(result.x).reshape(-1, 3)
    diagnostics = BundleDiagnostics(
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        cost=float(result.cost),
        optimality=float(result.optimality),
        geometric_rms_mm=float(np.sqrt(np.mean(np.sum(geo * geo, axis=1)))),
    )
    return fitted_cameras, fitted_poses, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-image-coords-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument(
        "--wet-checkerboard-corners-csv",
        type=Path,
        nargs="+",
        default=None,
        help="one or more selected wet-checkerboard corner CSV files",
    )
    parser.add_argument("--intrinsics-dir", type=Path, required=True)
    parser.add_argument("--base-regression-model", type=Path, required=True)
    parser.add_argument("--refined-regression-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-span-mm", type=float, default=192.0)
    parser.add_argument("--water-height-mm", type=float, default=192.0)
    parser.add_argument("--glass-thickness-mm", type=float, default=4.0)
    parser.add_argument("--index-glass", type=float, default=1.52)
    parser.add_argument("--index-water", type=float, default=1.3325)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--ball-cluster-gap-frames", type=int, default=120)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument(
        "--physical-geometry-measured",
        action="store_true",
        help="record that tank span, water height, and glass thickness were measured",
    )
    parser.add_argument(
        "--interface-mode",
        choices=("fixed", "refine"),
        default="refine",
        help=(
            "fixed keeps the measured interface planes unchanged; refine runs the "
            "legacy soft-prior interface-correction stage"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_lookup(args.geometry_csv)
    ball = pd.read_csv(args.ball_image_coords_csv)
    ball["pose_id"] = ball["pose_id"].astype(str)
    ball_train_ids, ball_test_ids, ball_cluster_count = _split_ball_pose_clusters(
        ball["pose_id"].tolist(), args.ball_cluster_gap_frames
    )
    ball_train = ball[ball["pose_id"].isin(ball_train_ids)].copy()
    ball_test = ball[ball["pose_id"].isin(ball_test_ids)].copy()
    ball_train_path = args.output_dir / "ball_training_coordinates.csv"
    ball_heldout_path = args.output_dir / "ball_heldout_poses.txt"
    ball_train.to_csv(ball_train_path, index=False)
    ball_heldout_path.write_text("\n".join(sorted(ball_test_ids)) + "\n", encoding="utf-8")
    base_calibration = Calibration(args.base_regression_model)
    ball_states, ball_targets = _ball_pose_initializations(ball_train, geometry, base_calibration)
    bounds = np.array([[0.0, args.inner_span_mm], [0.0, args.inner_span_mm], [0.0, args.water_height_mm]])
    initial_cameras: list[FlatPortCamera] = []
    pinhole_cameras: list[FlatPortCamera] = []
    target_world = ball_targets[["X", "Y", "Z"]].to_numpy(float)
    for index in range(3):
        image = ball_targets[[f"cam{index}_x", f"cam{index}_y"]].to_numpy(float)
        seed = initialize_flat_port_camera(
            f"cam{index}", target_world, image, tank_bounds=bounds, layer_count=0,
            glass_thickness_mm=args.glass_thickness_mm,
        )
        pinhole_cameras.append(seed)
        physical = _physical_camera(
            index, seed, args.intrinsics_dir,
            inner_span_mm=args.inner_span_mm,
            water_height_mm=args.water_height_mm,
            glass_thickness_mm=args.glass_thickness_mm,
            index_glass=args.index_glass,
            index_water=args.index_water,
        )
        fitted, _ = fit_flat_port_camera_fixed_intrinsics(
            physical, target_world, image, fit_interface_correction=False, max_nfev=300
        )
        initial_cameras.append(fitted)

    board: pd.DataFrame | None = None
    board_states: dict[str, PoseState] = {}
    board_train_ids: set[str] = set()
    board_test_ids: set[str] = set()
    if args.wet_checkerboard_corners_csv is not None:
        board = pd.concat(
            [pd.read_csv(path) for path in args.wet_checkerboard_corners_csv],
            ignore_index=True,
        )
        board["pose_id"] = board["pose_id"].astype(str)
        board_train_ids, board_test_ids = _split_pose_ids(board["pose_id"].tolist())
        board_states = _checkerboard_pose_initializations(
            board[board["pose_id"].isin(board_train_ids)], initial_cameras
        )
    poses = {**ball_states, **board_states}
    valid_keys = set(poses)
    if not valid_keys:
        raise RuntimeError("No training object poses could be initialized")
    observation_data = _observations(
        ball_train,
        board[board["pose_id"].isin(board_train_ids)] if board is not None else None,
        geometry,
        valid_keys,
    )
    anchor_pose = sorted(ball_states)[0]
    stage1_cameras, stage1_poses, stage1 = fit_bundle_stage(
        initial_cameras, poses, observation_data,
        anchor_pose=anchor_pose, release_interface=False, max_nfev=args.max_nfev,
    )
    if args.interface_mode == "refine":
        final_cameras, final_poses, stage2 = fit_bundle_stage(
            stage1_cameras, stage1_poses, observation_data,
            anchor_pose=anchor_pose, release_interface=True, max_nfev=args.max_nfev,
        )
    else:
        final_cameras = stage1_cameras
        final_poses = stage1_poses
        stage2 = None
    for camera in final_cameras:
        camera.to_json(args.output_dir / f"{camera.name}_bundle_refractive.json")

    predictors = {
        "base_regression": _regression_predictor(args.base_regression_model),
        "refined_regression": _regression_predictor(args.refined_regression_model),
        "pinhole_dlt": _ray_predictor(pinhole_cameras),
        "flat_refractive": _ray_predictor(final_cameras),
    }
    marker_tables = []
    pair_tables = []
    summaries: dict[str, dict[str, object]] = {}
    for name, predictor in predictors.items():
        markers, pairs = evaluate_predictor(ball_test, geometry, predictor, model_name=name)
        marker_tables.append(markers)
        pair_tables.append(pairs)
        summaries[name] = summarize_model(markers, pairs)
    markers = pd.concat(marker_tables, ignore_index=True)
    pairs = pd.concat(pair_tables, ignore_index=True)
    marker_path = args.output_dir / "heldout_marker_errors.csv"
    pair_path = args.output_dir / "heldout_pairwise_errors.csv"
    figure_path = args.output_dir / "bundle_refractive_vs_regression.png"
    markers.to_csv(marker_path, index=False)
    pairs.to_csv(pair_path, index=False)
    plot_comparison(
        markers,
        pairs,
        summaries,
        figure_path,
        n_training_poses=len(ball_train_ids),
        n_heldout_poses=len(ball_test_ids),
        dry_intrinsics_fixed=True,
    )
    report = {
        "physical_inputs": {
            "inner_span_mm": args.inner_span_mm,
            "glass_thickness_mm": args.glass_thickness_mm,
            "temperature_c": args.temperature_c,
            "index_water": args.index_water,
            "index_glass_assumed": args.index_glass,
            "water_height_mm": args.water_height_mm,
            "water_height_mm_provisional": (
                None if args.physical_geometry_measured else args.water_height_mm
            ),
            "physical_geometry_measured": bool(args.physical_geometry_measured),
            "interface_mode": args.interface_mode,
        },
        "synchronization": {
            "source": "shared external frame trigger",
            "join_key": "frame index",
            "camera_internal_timestamps_used_cross_camera": False,
        },
        "split": {
            "ball_training_poses": len(ball_train_ids),
            "ball_heldout_poses": len(ball_test_ids),
            "ball_independent_time_clusters": ball_cluster_count,
            "ball_cluster_gap_frames": args.ball_cluster_gap_frames,
            "board_training_poses_requested": len(board_train_ids),
            "board_training_poses_initialized": len(board_states),
            "board_heldout_poses": len(board_test_ids),
            "gauge_anchor_pose": anchor_pose,
        },
        "bundle": {
            "observations": int(len(observation_data[0])),
            "object_poses": int(len(observation_data[-1])),
            "stage1_fixed_interfaces": asdict(stage1),
            "stage2_interface_corrections": (
                asdict(stage2) if stage2 is not None else {"status": "disabled"}
            ),
            "final_training_ray_residuals": summarize_bundle_ray_residuals(
                final_cameras, final_poses, observation_data
            ),
        },
        "heldout_models": summaries,
        "fitted_cameras": {camera.name: asdict(camera) for camera in final_cameras},
        "guardrail": (
            "Tank span, water height, and glass thickness are measured and the interfaces are fixed. "
            "Held-out cube shape/scale metrics are valid; one training cube pose still fixes the "
            "world gauge, so absolute surveyed-marker accuracy is not claimed."
            if args.physical_geometry_measured
            else
            "Held-out cube shape/scale metrics are valid. Absolute Z remains provisional because "
            "water height was not measured, and one training cube pose fixes the otherwise free gauge."
        ),
        "outputs": {
            "figure": str(figure_path),
            "marker_errors": str(marker_path),
            "pairwise_errors": str(pair_path),
            "ball_training_coordinates": str(ball_train_path),
            "ball_heldout_pose_list": str(ball_heldout_path),
        },
    }
    report_path = args.output_dir / "bundle_adjustment_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
