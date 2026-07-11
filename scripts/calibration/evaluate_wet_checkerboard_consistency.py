"""Evaluate held-out wet-checkerboard ray consistency and image coverage.

The camera/refraction model is fixed.  For each held-out checkerboard pose, only
that board's six-degree-of-freedom rigid pose is fitted.  The reported distance
is the 3D point-to-refracted-ray distance, so it measures multi-view geometric
consistency rather than ordinary in-air pixel reprojection error.
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
from scipy.spatial import ConvexHull, QhullError

from scripts.calibration.fit_refractive_bundle_adjustment import (
    CAMERA_INDEX,
    _checkerboard_pose_initializations,
    _split_pose_ids,
)
from scripts.calibration.refractive_geometry import FlatPortCamera


@dataclass
class DistanceSummary:
    n: int
    mean_mm: float
    rms_mm: float
    median_mm: float
    p95_mm: float
    maximum_mm: float


def _summarize(values: np.ndarray) -> DistanceSummary:
    values = np.asarray(values, dtype=float)
    return DistanceSummary(
        n=int(len(values)),
        mean_mm=float(np.mean(values)),
        rms_mm=float(np.sqrt(np.mean(values**2))),
        median_mm=float(np.median(values)),
        p95_mm=float(np.percentile(values, 95)),
        maximum_mm=float(np.max(values)),
    )


def _fit_pose(group: pd.DataFrame, cameras: Sequence[FlatPortCamera]) -> tuple[np.ndarray, dict[str, object]]:
    pose_id = str(group["pose_id"].iloc[0])
    initial = _checkerboard_pose_initializations(group, cameras).get(f"board:{pose_id}")
    if initial is None:
        raise ValueError(f"Could not initialize held-out board pose {pose_id}")

    object_points = group[["board_x_mm", "board_y_mm", "board_z_mm"]].to_numpy(float)
    pixels = group[["x", "y"]].to_numpy(float)
    camera_indices = np.asarray([CAMERA_INDEX[str(value)] for value in group["camera"]], dtype=int)
    origins = np.empty_like(object_points)
    directions = np.empty_like(object_points)
    for index, camera in enumerate(cameras):
        mask = camera_indices == index
        if not np.any(mask):
            continue
        origins[mask], directions[mask] = camera.backproject(pixels[mask])

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation = cv2.Rodrigues(parameters[:3])[0]
        world = (rotation @ object_points.T).T + parameters[3:]
        offsets = world - origins
        closest = origins + np.sum(offsets * directions, axis=1)[:, None] * directions
        return (world - closest).ravel()

    parameter0 = np.concatenate((initial.rotation_vector, initial.translation))
    result = least_squares(
        residual,
        parameter0,
        method="trf",
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=250,
    )
    distances = np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1)
    return distances, {
        "pose_id": pose_id,
        "n_rays": int(len(distances)),
        "rms_mm": float(np.sqrt(np.mean(distances**2))),
        "p95_mm": float(np.percentile(distances, 95)),
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }


def _image_coverage(data: pd.DataFrame, image_size: tuple[int, int]) -> dict[str, object]:
    width, height = image_size
    rows: dict[str, object] = {}
    centroids = data.groupby(["pose_id", "camera"], as_index=False)[["x", "y"]].mean()
    for camera, points in centroids.groupby("camera"):
        xy = points[["x", "y"]].to_numpy(float)
        hull_fraction = 0.0
        if len(xy) >= 3:
            try:
                hull_fraction = float(ConvexHull(xy).volume / (width * height))
            except QhullError:
                pass
        rows[str(camera)] = {
            "n_pose_centroids": int(len(xy)),
            "centroid_x_range_px": [float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))],
            "centroid_y_range_px": [float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))],
            "centroid_x_span_fraction": float(np.ptp(xy[:, 0]) / width),
            "centroid_y_span_fraction": float(np.ptp(xy[:, 1]) / height),
            "centroid_convex_hull_fraction_of_image": hull_fraction,
        }
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--camera-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--image-height", type=int, default=1200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cameras = [
        FlatPortCamera.from_json(args.camera_model_dir / f"cam{index}_bundle_refractive.json")
        for index in range(3)
    ]
    report: dict[str, object] = {
        "metric_definition": (
            "3D point-to-refracted-ray distance after fitting only a rigid pose for each held-out board; "
            "camera and interface parameters remain fixed"
        ),
        "datasets": {},
    }
    for label, csv_path in args.dataset:
        data = pd.read_csv(csv_path)
        data["pose_id"] = data["pose_id"].astype(str)
        _, heldout_ids = _split_pose_ids(data["pose_id"].tolist())
        heldout = data[data["pose_id"].isin(heldout_ids)].copy()
        all_distances: list[np.ndarray] = []
        per_pose: list[dict[str, object]] = []
        failures: list[str] = []
        for pose_id, group in heldout.groupby("pose_id", sort=True):
            try:
                distances, details = _fit_pose(group, cameras)
            except (ValueError, np.linalg.LinAlgError) as error:
                failures.append(f"{pose_id}: {error}")
                continue
            all_distances.append(distances)
            per_pose.append(details)
        if not all_distances:
            raise RuntimeError(f"No held-out poses could be evaluated for {label}")
        report["datasets"][label] = {
            "selected_poses": int(data["pose_id"].nunique()),
            "heldout_poses_requested": int(len(heldout_ids)),
            "heldout_poses_evaluated": int(len(per_pose)),
            "heldout_ray_distances": asdict(_summarize(np.concatenate(all_distances))),
            "per_pose": per_pose,
            "failures": failures,
            "selected_pose_image_coverage": _image_coverage(
                data, (args.image_width, args.image_height)
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
