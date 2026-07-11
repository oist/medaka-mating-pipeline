"""Plot fitted camera positions around the sampled calibration volume.

The figure combines the 12-ball training volume, interpolation hull, priority
gaps, held-out residual distribution, and the three refractive-camera centres.
Camera positions are plotted in the same provisional tank coordinate frame as
the calibration points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from scripts.calibration.qc_rigid_object_volume import (
    _bounds_array,
    _draw_tank_wireframe,
    _geometry_lookup,
    reconstruct_rigid_poses,
)
from scripts.calibration.refractive_geometry import FlatPortCamera


CAMERA_STYLE = {
    "cam0": {"label": "Top camera", "color": "#2563A6", "marker": "^"},
    "cam1": {"label": "Side 1 camera", "color": "#D99024", "marker": "<"},
    "cam2": {"label": "Side 2 camera", "color": "#C45A7A", "marker": ">"},
}
TRAINING_COLOR = "#D99024"
HULL_COLOR = "#F2C66D"
RESIDUAL_COLOR = "#167D8D"
INK = "#262626"
GRID = "#D9D9D9"


def _camera_models(directory: Path) -> list[FlatPortCamera]:
    cameras = []
    for index in range(3):
        path = directory / f"cam{index}_bundle_refractive.json"
        if not path.exists():
            raise FileNotFoundError(path)
        cameras.append(FlatPortCamera.from_json(path))
    return cameras


def _camera_direction(camera: FlatPortCamera) -> np.ndarray:
    direction = np.array([0.0, 0.0, 1.0]) @ camera.rotation_matrix
    return direction / np.linalg.norm(direction)


def _set_3d_limits(ax: plt.Axes, bounds: np.ndarray, camera_centres: np.ndarray) -> None:
    all_points = np.vstack((bounds[:, 0], bounds[:, 1], camera_centres))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.06, 15.0)
    ax.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    ax.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    ax.set_zlim(minimum[2] - min(padding[2], 10.0), maximum[2] + padding[2])
    ax.set_box_aspect(maximum - minimum)


def _draw_3d_context(
    ax: plt.Axes,
    training: np.ndarray,
    cameras: Sequence[FlatPortCamera],
    gaps: Sequence[dict[str, float]],
    bounds: np.ndarray,
) -> None:
    ax.scatter(
        training[:, 0], training[:, 1], training[:, 2],
        s=9, color=TRAINING_COLOR, alpha=0.58, depthshade=False,
    )
    if len(training) >= 4:
        hull = ConvexHull(training)
        ax.add_collection3d(
            Poly3DCollection(
                training[hull.simplices], facecolor=HULL_COLOR,
                edgecolor="none", alpha=0.07,
            )
        )
    for gap in gaps[:8]:
        ax.scatter(
            gap["X_mm"], gap["Y_mm"], gap["Z_mm"], marker="x",
            s=46, color=INK, linewidths=1.5, depthshade=False,
        )
    _draw_tank_wireframe(ax, bounds, color="#3A3A3A")
    tank_centre = bounds.mean(axis=1)
    centres = []
    for camera in cameras:
        centre = np.asarray(camera.camera_center, dtype=float)
        centres.append(centre)
        style = CAMERA_STYLE[camera.name]
        ax.scatter(
            *centre, s=115, marker=style["marker"], color=style["color"],
            edgecolor="white", linewidth=1.0, depthshade=False,
        )
        direction = _camera_direction(camera)
        ax.quiver(
            *centre, *direction, length=90.0, normalize=True,
            color=style["color"], linewidth=2.0, arrow_length_ratio=0.18,
        )
        ax.plot(
            [centre[0], tank_centre[0]],
            [centre[1], tank_centre[1]],
            [centre[2], tank_centre[2]],
            linestyle="--", color=style["color"], linewidth=1.0, alpha=0.6,
        )
        ax.text(
            centre[0], centre[1], centre[2] + 14,
            f"{style['label']}\n({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}) mm",
            color=style["color"], fontsize=8, ha="center",
        )
    _set_3d_limits(ax, bounds, np.asarray(centres))
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D camera layout and sampled calibration volume")
    ax.view_init(elev=22, azim=-54)
    ax.grid(color=GRID, linewidth=0.6)


def _projection_limits(
    first: int, second: int, bounds: np.ndarray, camera_centres: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    first_values = np.r_[bounds[first], camera_centres[:, first]]
    second_values = np.r_[bounds[second], camera_centres[:, second]]
    first_pad = max(15.0, np.ptp(first_values) * 0.07)
    second_pad = max(15.0, np.ptp(second_values) * 0.07)
    return (
        (float(first_values.min() - first_pad), float(first_values.max() + first_pad)),
        (float(second_values.min() - second_pad), float(second_values.max() + second_pad)),
    )


def _draw_projection(
    ax: plt.Axes,
    first: int,
    second: int,
    axis_names: tuple[str, str],
    training: np.ndarray,
    residuals: pd.DataFrame,
    cameras: Sequence[FlatPortCamera],
    gaps: Sequence[dict[str, float]],
    bounds: np.ndarray,
) -> None:
    centres = np.asarray([camera.camera_center for camera in cameras], dtype=float)
    ax.add_patch(
        plt.Rectangle(
            (bounds[first, 0], bounds[second, 0]),
            bounds[first, 1] - bounds[first, 0],
            bounds[second, 1] - bounds[second, 0],
            facecolor="#F7F7F5", edgecolor=INK, linewidth=1.2, zorder=0,
        )
    )
    ax.scatter(
        training[:, first], training[:, second], s=8,
        color=TRAINING_COLOR, alpha=0.35, edgecolors="none", zorder=1,
    )
    xyz = ("X", "Y", "Z")
    if len(residuals):
        ax.scatter(
            residuals[xyz[first]], residuals[xyz[second]], s=17,
            facecolors="none", edgecolors=RESIDUAL_COLOR, linewidths=0.7,
            alpha=0.8, zorder=2,
        )
    for gap in gaps[:8]:
        ax.scatter(
            gap[f"{xyz[first]}_mm"], gap[f"{xyz[second]}_mm"],
            marker="x", s=36, color=INK, linewidths=1.2, zorder=3,
        )
    tank_centre = bounds.mean(axis=1)
    for camera, centre in zip(cameras, centres, strict=True):
        style = CAMERA_STYLE[camera.name]
        ax.plot(
            [centre[first], tank_centre[first]],
            [centre[second], tank_centre[second]],
            linestyle="--", color=style["color"], linewidth=1.0, alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            centre[first], centre[second], s=82, marker=style["marker"],
            color=style["color"], edgecolor="white", linewidth=0.9, zorder=4,
        )
        omitted = ({0, 1, 2} - {first, second}).pop()
        ax.annotate(
            f"{style['label']}\n{xyz[omitted]}={centre[omitted]:.0f} mm",
            (centre[first], centre[second]), xytext=(6, 6),
            textcoords="offset points", fontsize=7.5, color=style["color"],
        )
    xlim, ylim = _projection_limits(first, second, bounds, centres)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{axis_names[0]} (mm)")
    ax.set_ylabel(f"{axis_names[1]} (mm)")
    ax.set_title(f"{axis_names[0]}{axis_names[1]} projection")
    ax.grid(color=GRID, linewidth=0.6)


def _draw_residual_cdf(ax: plt.Axes, residuals: pd.DataFrame) -> None:
    for column, label, color, style in (
        ("base_residual_mm", "Base beads model", "#737373", "--"),
        ("refined_residual_mm", "12-ball refined model", RESIDUAL_COLOR, "-"),
    ):
        values = np.sort(residuals[column].dropna().to_numpy(float))
        cumulative = np.arange(1, len(values) + 1) / len(values) * 100.0
        ax.plot(values, cumulative, color=color, linestyle=style, linewidth=2.2, label=label)
    ax.set(xlabel="Held-out Kabsch residual (mm)", ylabel="markers <= residual (%)", xlim=(0, None), ylim=(0, 100))
    ax.set_title(f"Held-out residual CDF (n={len(residuals)})")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-coords-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--base-model-folder", type=Path, required=True)
    parser.add_argument("--residuals-csv", type=Path, required=True)
    parser.add_argument("--volume-summary-json", type=Path, required=True)
    parser.add_argument("--camera-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tank-bounds", type=float, nargs=6,
        default=(0, 192, 0, 192, 0, 192),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bounds = _bounds_array(args.tank_bounds)
    training_coords = pd.read_csv(args.training_coords_csv)
    training_coords["pose_id"] = training_coords["pose_id"].astype(str)
    geometry = _geometry_lookup(args.geometry_csv)
    reconstructed = reconstruct_rigid_poses(
        training_coords,
        geometry,
        args.base_model_folder,
        pose_ids=set(training_coords["pose_id"]),
    )
    training = reconstructed[["X", "Y", "Z"]].drop_duplicates().to_numpy(float)
    residuals = pd.read_csv(args.residuals_csv)
    report = json.loads(args.volume_summary_json.read_text(encoding="utf-8"))
    gaps = report["priority_gaps"]
    cameras = _camera_models(args.camera_model_dir)

    fig = plt.figure(figsize=(18, 10.5), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=(1.3, 1.3, 1.0, 1.0))
    ax3d = fig.add_subplot(grid[:, :2], projection="3d")
    _draw_3d_context(ax3d, training, cameras, gaps, bounds)
    _draw_projection(fig.add_subplot(grid[0, 2]), 0, 1, ("X", "Y"), training, residuals, cameras, gaps, bounds)
    _draw_projection(fig.add_subplot(grid[0, 3]), 0, 2, ("X", "Z"), training, residuals, cameras, gaps, bounds)
    _draw_projection(fig.add_subplot(grid[1, 2]), 1, 2, ("Y", "Z"), training, residuals, cameras, gaps, bounds)
    _draw_residual_cdf(fig.add_subplot(grid[1, 3]), residuals)

    legend_handles = [
        Line2D([0], [0], marker=CAMERA_STYLE[name]["marker"], color="none", markerfacecolor=CAMERA_STYLE[name]["color"], markeredgecolor="white", markersize=9, label=CAMERA_STYLE[name]["label"])
        for name in ("cam0", "cam1", "cam2")
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TRAINING_COLOR, alpha=0.65, markersize=6, label=f"12-ball training samples (n={len(training)})"),
            Line2D([0], [0], marker="x", color=INK, linestyle="none", markersize=7, label="Priority refilming target"),
        ]
    )
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    spans = bounds[:, 1] - bounds[:, 0]
    fig.suptitle(
        "Camera geometry, sampled 12-ball volume, and held-out residuals\n"
        f"Analysis volume {spans[0]:g} x {spans[1]:g} x {spans[2]:g} mm; "
        "camera centres from the refractive bundle; water height/absolute Z provisional",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
