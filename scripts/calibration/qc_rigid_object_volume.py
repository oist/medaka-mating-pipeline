"""Visualize rigid-object calibration residuals and tank-volume coverage.

This QC is designed for moved rigid calibration objects whose absolute pose is
unknown.  Per-marker residuals are therefore shape residuals after fitting the
known object geometry to each reconstructed pose with Kabsch alignment.  They
measure local metric/shape fidelity, not absolute world-position error.

Two figures are written:

* ``rigid_object_residuals_in_tank.png`` places held-out residuals in the tank;
* ``rigid_object_training_coverage.png`` compares the old calibration samples,
  current rigid-object training samples, and all accepted rigid-object poses.

The coverage metrics are geometric diagnostics rather than accuracy bounds.
The convex hull approximates the interpolation envelope, while nearest-sample
distance exposes holes that an axis-aligned min/max span can hide.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.stats import qmc

from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration


VIEW_COLS = ["cam0_x", "cam0_y", "cam1_x", "cam1_y", "cam2_x", "cam2_y"]
XYZ_COLS = ["X", "Y", "Z"]
DEFAULT_RADII_MM = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
SET_COLORS = {
    "base_beads": "#7A7A7A",
    "current_12ball_training": "#E68613",
    "all_accepted_12ball": "#2A78B8",
}


def _load_pose_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _geometry_lookup(geometry_csv: Path) -> dict[int, np.ndarray]:
    geometry = pd.read_csv(geometry_csv)
    required = {"marker_id", "object_x", "object_y", "object_z"}
    missing = sorted(required.difference(geometry.columns))
    if missing:
        raise ValueError(f"{geometry_csv} is missing required columns: {missing}")
    return {
        int(row.marker_id): np.array([row.object_x, row.object_y, row.object_z], dtype=float)
        for row in geometry.itertuples(index=False)
    }


def reconstruct_rigid_poses(
    image_coords: pd.DataFrame,
    geometry: dict[int, np.ndarray],
    model_folder: Path,
    *,
    pose_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct poses and return Kabsch reference positions and residuals."""
    coords = image_coords.copy()
    coords["pose_id"] = coords["pose_id"].astype(str)
    if pose_ids is not None:
        coords = coords[coords["pose_id"].isin(pose_ids)]
    missing = [column for column in ["pose_id", "marker_id", *VIEW_COLS] if column not in coords]
    if missing:
        raise ValueError(f"image-coordinate table is missing required columns: {missing}")

    calibration = Calibration(model_folder)
    rows: list[dict[str, object]] = []
    for pose_id, group in coords.groupby("pose_id", sort=True):
        group = group.dropna(subset=VIEW_COLS)
        if len(group) < 4:
            continue
        marker_ids = group["marker_id"].astype(int).to_numpy()
        unknown = sorted(set(marker_ids).difference(geometry))
        if unknown:
            raise ValueError(f"pose {pose_id} contains marker IDs absent from geometry: {unknown}")

        features = group[VIEW_COLS].to_numpy(dtype=float)
        predicted = np.asarray(calibration._model_im2xyz.predict(features), dtype=float)
        object_xyz = np.asarray([geometry[int(marker_id)] for marker_id in marker_ids])
        rotation, translation = kabsch_rigid_fit(object_xyz, predicted)
        reference = apply_rigid(object_xyz, rotation, translation)
        residual = predicted - reference
        for marker_id, ref_xyz, pred_xyz, vector in zip(
            marker_ids, reference, predicted, residual, strict=True
        ):
            rows.append(
                {
                    "pose_id": str(pose_id),
                    "marker_id": int(marker_id),
                    "X": float(ref_xyz[0]),
                    "Y": float(ref_xyz[1]),
                    "Z": float(ref_xyz[2]),
                    "pred_X": float(pred_xyz[0]),
                    "pred_Y": float(pred_xyz[1]),
                    "pred_Z": float(pred_xyz[2]),
                    "residual_X_mm": float(vector[0]),
                    "residual_Y_mm": float(vector[1]),
                    "residual_Z_mm": float(vector[2]),
                    "residual_mm": float(np.linalg.norm(vector)),
                }
            )
    if not rows:
        raise ValueError("no reconstructable poses were found")
    return pd.DataFrame(rows)


def merge_model_residuals(base: pd.DataFrame, refined: pd.DataFrame) -> pd.DataFrame:
    """Combine base/refined held-out residuals using refined-fit tank positions."""
    keys = ["pose_id", "marker_id"]
    base_columns = keys + [
        "residual_X_mm",
        "residual_Y_mm",
        "residual_Z_mm",
        "residual_mm",
    ]
    merged = refined.merge(base[base_columns], on=keys, suffixes=("_refined", "_base"), validate="one_to_one")
    merged = merged.rename(
        columns={
            "residual_X_mm_refined": "refined_residual_X_mm",
            "residual_Y_mm_refined": "refined_residual_Y_mm",
            "residual_Z_mm_refined": "refined_residual_Z_mm",
            "residual_mm_refined": "refined_residual_mm",
            "residual_X_mm_base": "base_residual_X_mm",
            "residual_Y_mm_base": "base_residual_Y_mm",
            "residual_Z_mm_base": "base_residual_Z_mm",
            "residual_mm_base": "base_residual_mm",
        }
    )
    merged["improvement_mm"] = merged["base_residual_mm"] - merged["refined_residual_mm"]
    return merged


def _unique_points(frame: pd.DataFrame, columns: Sequence[str] = XYZ_COLS) -> np.ndarray:
    points = frame[list(columns)].dropna().drop_duplicates().to_numpy(dtype=float)
    if len(points) < 4:
        raise ValueError("at least four unique 3D sample points are required")
    return points


def _tank_qmc_points(bounds: np.ndarray, power: int = 18) -> np.ndarray:
    unit_points = qmc.Sobol(d=3, scramble=False).random_base2(power)
    return qmc.scale(unit_points, bounds[:, 0], bounds[:, 1])


def summarize_sampling_set(
    points: np.ndarray,
    tank_points: np.ndarray,
    bounds: np.ndarray,
    radii_mm: Iterable[float] = DEFAULT_RADII_MM,
) -> dict[str, object]:
    """Summarize point span, hull occupancy, and nearest-sample coverage."""
    points = np.asarray(points, dtype=float)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    tank_span = bounds[:, 1] - bounds[:, 0]
    span = maximum - minimum

    hull = ConvexHull(points)
    triangulation = Delaunay(points[hull.vertices])
    inside_hull = triangulation.find_simplex(tank_points) >= 0
    nearest = cKDTree(points).query(tank_points, workers=-1)[0]
    radii = [float(radius) for radius in radii_mm]
    return {
        "n_unique_points": int(len(points)),
        "minimum_mm": dict(zip(XYZ_COLS, minimum.tolist(), strict=True)),
        "maximum_mm": dict(zip(XYZ_COLS, maximum.tolist(), strict=True)),
        "axis_span_mm": dict(zip(XYZ_COLS, span.tolist(), strict=True)),
        "axis_span_fraction": dict(zip(XYZ_COLS, (span / tank_span).tolist(), strict=True)),
        "convex_hull_fraction_of_tank": float(np.mean(inside_hull)),
        "raw_convex_hull_volume_mm3": float(hull.volume),
        "nearest_sample_distance_mm": {
            "median": float(np.median(nearest)),
            "p90": float(np.percentile(nearest, 90)),
            "p95": float(np.percentile(nearest, 95)),
            "maximum": float(np.max(nearest)),
        },
        "tank_fraction_within_radius": {
            f"{radius:g}_mm": float(np.mean(nearest <= radius)) for radius in radii
        },
    }


def summarize_residual(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n_points": int(len(values)),
        "mean_mm": float(np.mean(values)),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "maximum_mm": float(np.max(values)),
    }


def find_priority_gaps(
    sample_points: np.ndarray,
    bounds: np.ndarray,
    *,
    grid_step_mm: float = 5.0,
    minimum_separation_mm: float = 45.0,
    count: int = 8,
) -> list[dict[str, float]]:
    """Find separated tank locations farthest from a calibration sample."""
    axes = [
        np.arange(low + grid_step_mm / 2, high, grid_step_mm)
        for low, high in bounds
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distances = cKDTree(sample_points).query(grid, workers=-1)[0]
    selected: list[tuple[np.ndarray, float]] = []
    for index in np.argsort(distances)[::-1]:
        point = grid[index]
        if all(np.linalg.norm(point - prior) >= minimum_separation_mm for prior, _ in selected):
            selected.append((point, float(distances[index])))
        if len(selected) == count:
            break
    return [
        {
            "X_mm": float(point[0]),
            "Y_mm": float(point[1]),
            "Z_mm": float(point[2]),
            "nearest_training_sample_mm": distance,
        }
        for point, distance in selected
    ]


def _draw_tank_wireframe(ax: plt.Axes, bounds: np.ndarray, *, color: str = "#303030") -> None:
    x0, x1 = bounds[0]
    y0, y1 = bounds[1]
    z0, z1 = bounds[2]
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ]
    )
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        ax.plot(*corners[[start, end]].T, color=color, linewidth=0.9, alpha=0.85)


def _format_3d_tank(ax: plt.Axes, bounds: np.ndarray) -> None:
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_box_aspect(bounds[:, 1] - bounds[:, 0])
    ax.view_init(elev=23, azim=-55)
    ax.invert_yaxis()


def _projection(
    ax: plt.Axes,
    residuals: pd.DataFrame,
    horizontal: str,
    vertical: str,
    norm: colors.Normalize,
    bounds: np.ndarray,
    arrow_multiplier: float,
) -> None:
    coordinate_index = {"X": 0, "Y": 1, "Z": 2}
    horizontal_resid = f"refined_residual_{horizontal}_mm"
    vertical_resid = f"refined_residual_{vertical}_mm"
    ax.scatter(
        residuals[horizontal],
        residuals[vertical],
        c=residuals["refined_residual_mm"],
        cmap="viridis",
        norm=norm,
        s=30,
        edgecolors="black",
        linewidths=0.25,
        zorder=3,
    )
    ax.quiver(
        residuals[horizontal],
        residuals[vertical],
        residuals[horizontal_resid] * arrow_multiplier,
        residuals[vertical_resid] * arrow_multiplier,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#222222",
        alpha=0.62,
        width=0.0025,
        headwidth=3.5,
        zorder=2,
    )
    ax.set_xlim(*bounds[coordinate_index[horizontal]])
    ax.set_ylim(*bounds[coordinate_index[vertical]])
    if vertical == "Y":
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{horizontal} (mm)")
    ax.set_ylabel(f"{vertical} (mm)")
    ax.set_title(f"{horizontal}{vertical} projection; arrows x{arrow_multiplier:g}")
    ax.grid(alpha=0.18)


def plot_residual_dashboard(
    residuals: pd.DataFrame,
    bounds: np.ndarray,
    output_path: Path,
    *,
    arrow_multiplier: float = 20.0,
) -> None:
    refined_values = residuals["refined_residual_mm"].to_numpy(dtype=float)
    upper = max(1.0, math.ceil(float(np.max(refined_values)) * 10) / 10)
    norm = colors.Normalize(vmin=0.0, vmax=upper)

    fig = plt.figure(figsize=(17.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1, 1))
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    scatter = ax_3d.scatter(
        residuals["X"],
        residuals["Y"],
        residuals["Z"],
        c=refined_values,
        cmap="viridis",
        norm=norm,
        s=38,
        edgecolors="black",
        linewidths=0.25,
        depthshade=False,
    )
    vectors = residuals[[
        "refined_residual_X_mm",
        "refined_residual_Y_mm",
        "refined_residual_Z_mm",
    ]].to_numpy(dtype=float) * arrow_multiplier
    ax_3d.quiver(
        residuals["X"], residuals["Y"], residuals["Z"],
        vectors[:, 0], vectors[:, 1], vectors[:, 2],
        color="#202020", alpha=0.50, linewidth=0.6, arrow_length_ratio=0.20,
    )
    _draw_tank_wireframe(ax_3d, bounds)
    _format_3d_tank(ax_3d, bounds)
    ax_3d.set_title(f"Held-out refined residuals; arrows x{arrow_multiplier:g}")

    _projection(fig.add_subplot(grid[0, 1]), residuals, "X", "Y", norm, bounds, arrow_multiplier)
    _projection(fig.add_subplot(grid[0, 2]), residuals, "X", "Z", norm, bounds, arrow_multiplier)
    _projection(fig.add_subplot(grid[1, 1]), residuals, "Y", "Z", norm, bounds, arrow_multiplier)

    ax_cdf = fig.add_subplot(grid[1, 2])
    for column, label, color in (
        ("base_residual_mm", "Base beads model", "#707070"),
        ("refined_residual_mm", "12-ball refined model", "#087F8C"),
    ):
        ordered = np.sort(residuals[column].to_numpy(dtype=float))
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        ax_cdf.plot(ordered, cumulative, label=label, color=color, linewidth=2.5)
    ax_cdf.set_xlabel("Kabsch shape residual (mm)")
    ax_cdf.set_ylabel("Held-out markers <= residual (%)")
    ax_cdf.set_xlim(left=0)
    ax_cdf.set_ylim(0, 100)
    ax_cdf.grid(alpha=0.25)
    ax_cdf.legend(loc="lower right")
    base_summary = summarize_residual(residuals["base_residual_mm"].to_numpy())
    refined_summary = summarize_residual(refined_values)
    ax_cdf.set_title(
        f"Held-out CDF (n={len(residuals)})\n"
        f"mean {base_summary['mean_mm']:.2f} -> {refined_summary['mean_mm']:.2f} mm; "
        f"P95 {base_summary['p95_mm']:.2f} -> {refined_summary['p95_mm']:.2f} mm"
    )

    fig.colorbar(scatter, ax=[ax_3d], shrink=0.55, pad=0.06, label="Refined shape residual (mm)")
    spans = bounds[:, 1] - bounds[:, 0]
    volume_label = " x ".join(f"{value:g}" for value in spans)
    fig.suptitle(
        f"12-ball calibration residuals inside the {volume_label} mm analysis volume\n"
        "Held-out poses only; Kabsch residual measures shape/scale fidelity, not absolute localization",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _coverage_curve(
    ax: plt.Axes,
    distances: dict[str, np.ndarray],
    radius_marker: float,
    *,
    n_training_poses: int | None = None,
    n_accepted_poses: int | None = None,
) -> None:
    radii = np.linspace(0, 65, 131)
    labels = {
        "base_beads": "Base beads",
        "current_12ball_training": f"12-ball: current {n_training_poses or '?'} train poses",
        "all_accepted_12ball": f"12-ball: all {n_accepted_poses or '?'} accepted poses",
    }
    styles = {
        "base_beads": "-",
        "current_12ball_training": "-",
        "all_accepted_12ball": "--",
    }
    for name, values in distances.items():
        fraction = [100 * np.mean(values <= radius) for radius in radii]
        ax.plot(
            radii,
            fraction,
            label=labels[name],
            color=SET_COLORS[name],
            linestyle=styles[name],
            linewidth=2.2,
        )
    ax.axvline(radius_marker, color="#222222", linestyle=":", linewidth=1.3)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Distance to nearest measured 3D point (mm)")
    ax.set_ylabel("Tank volume within distance (%)")
    ax.set_title("Sampling-density coverage")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)


def plot_coverage_dashboard(
    sample_sets: dict[str, np.ndarray],
    summaries: dict[str, dict[str, object]],
    tank_points: np.ndarray,
    priority_gaps: list[dict[str, float]],
    bounds: np.ndarray,
    output_path: Path,
    *,
    radius_marker_mm: float = 25.0,
    n_training_poses: int | None = None,
    n_accepted_poses: int | None = None,
) -> None:
    training = sample_sets["current_12ball_training"]
    accepted = sample_sets["all_accepted_12ball"]
    base = sample_sets["base_beads"]
    distances = {
        name: cKDTree(points).query(tank_points, workers=-1)[0]
        for name, points in sample_sets.items()
    }

    fig = plt.figure(figsize=(19, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(1.2, 1, 1, 1), height_ratios=(1.04, 1))
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_3d.scatter(*base.T, s=6, color=SET_COLORS["base_beads"], alpha=0.20, label="Base beads")
    ax_3d.scatter(
        *training.T,
        s=12,
        color=SET_COLORS["current_12ball_training"],
        alpha=0.72,
        label="Current 12-ball training",
        depthshade=False,
    )
    # Points not used in the current fit are shown as accepted/held-out footage.
    tree = cKDTree(training)
    heldout_mask = tree.query(accepted)[0] > 1e-5
    if np.any(heldout_mask):
        heldout = accepted[heldout_mask]
        ax_3d.scatter(
            *heldout.T,
            s=18,
            facecolors="none",
            edgecolors=SET_COLORS["all_accepted_12ball"],
            linewidths=0.6,
            alpha=0.8,
            label="Accepted, currently held out",
            depthshade=False,
        )
    hull = ConvexHull(training)
    hull_faces = Poly3DCollection(
        training[hull.simplices],
        facecolor=SET_COLORS["current_12ball_training"],
        edgecolor="none",
        alpha=0.055,
    )
    ax_3d.add_collection3d(hull_faces)
    _draw_tank_wireframe(ax_3d, bounds)
    _format_3d_tank(ax_3d, bounds)
    ax_3d.set_title("Measured positions and current interpolation hull")
    ax_3d.legend(loc="upper left", fontsize=8)

    ax_curve = fig.add_subplot(grid[0, 1])
    _coverage_curve(
        ax_curve,
        distances,
        radius_marker_mm,
        n_training_poses=n_training_poses,
        n_accepted_poses=n_accepted_poses,
    )

    ax_text = fig.add_subplot(grid[0, 2:])
    ax_text.axis("off")
    base_summary = summaries["base_beads"]
    training_summary = summaries["current_12ball_training"]
    accepted_summary = summaries["all_accepted_12ball"]

    def span_text(summary: dict[str, object]) -> str:
        spans = summary["axis_span_mm"]
        return "/".join(f"{float(spans[axis]):.0f}" for axis in XYZ_COLS)

    def radius_text(summary: dict[str, object]) -> float:
        key = f"{radius_marker_mm:g}_mm"
        return 100 * float(summary["tank_fraction_within_radius"][key])

    gap_lines = "\n".join(
        f"  ({gap['X_mm']:.0f}, {gap['Y_mm']:.0f}, {gap['Z_mm']:.0f}) mm  "
        f"nearest sample {gap['nearest_training_sample_mm']:.0f} mm"
        for gap in priority_gaps[:5]
    )
    text = (
        "WHAT 'COVERED' MEANS\n"
        "Axis span: reaches both sides on each axis, but can hide corner holes.\n"
        "Hull: fraction of tank inside the sample-point convex hull (interpolation proxy).\n"
        f"Near-sample: fraction within {radius_marker_mm:g} mm of a measured ball (density proxy).\n\n"
        "                               XYZ span     hull     near-sample\n"
        f"Base beads                    {span_text(base_summary):>11} mm   "
        f"{100*float(base_summary['convex_hull_fraction_of_tank']):5.1f}%       {radius_text(base_summary):5.1f}%\n"
        f"Current 12-ball ({n_training_poses if n_training_poses is not None else '?'} poses)"
        f"    {span_text(training_summary):>11} mm   "
        f"{100*float(training_summary['convex_hull_fraction_of_tank']):5.1f}%       {radius_text(training_summary):5.1f}%\n"
        f"All accepted ({n_accepted_poses if n_accepted_poses is not None else '?'} poses)"
        f"       {span_text(accepted_summary):>11} mm   "
        f"{100*float(accepted_summary['convex_hull_fraction_of_tank']):5.1f}%       {radius_text(accepted_summary):5.1f}%\n\n"
        "PRIORITY UNSAMPLED LOCATIONS (ball-coordinate targets)\n"
        f"{gap_lines}\n\n"
        "Low residual at a sampled point does not validate a distant empty region."
    )
    ax_text.text(
        0.01,
        0.98,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F6F4EF", "edgecolor": "#C7C0B2"},
    )

    x_values = np.linspace(bounds[0, 0], bounds[0, 1], 161)
    y_values = np.linspace(bounds[1, 0], bounds[1, 1], 161)
    xx, yy = np.meshgrid(x_values, y_values)
    slice_axes = []
    z_slices = np.linspace(bounds[2, 0] + 25, bounds[2, 1] - 25, 3)
    image = None
    training_tree = cKDTree(training)
    for column, z_value in enumerate(z_slices, start=1):
        ax = fig.add_subplot(grid[1, column])
        slice_axes.append(ax)
        query = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z_value)))
        distance = training_tree.query(query, workers=-1)[0].reshape(xx.shape)
        image = ax.pcolormesh(
            xx,
            yy,
            distance,
            shading="auto",
            cmap="magma",
            norm=colors.Normalize(0, 70),
        )
        ax.contour(xx, yy, distance, levels=[radius_marker_mm], colors="white", linewidths=1.0)
        near_slice = np.abs(training[:, 2] - z_value) <= 12.5
        ax.scatter(
            training[near_slice, 0],
            training[near_slice, 1],
            s=12,
            color="#39D0C8",
            edgecolors="#153A43",
            linewidths=0.25,
            label="training point within 12.5 mm",
        )
        for gap in priority_gaps:
            if abs(gap["Z_mm"] - z_value) <= 25:
                ax.plot(gap["X_mm"], gap["Y_mm"], marker="x", color="white", markersize=7, mew=1.5)
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Nearest training sample at Z={z_value:.0f} mm")
    if image is not None:
        fig.colorbar(image, ax=slice_axes, shrink=0.84, label="Nearest current training sample (mm)")

    spans = bounds[:, 1] - bounds[:, 0]
    volume_label = " x ".join(f"{value:g}" for value in spans)
    fig.suptitle(
        f"How much of the {volume_label} mm analysis volume is supported by calibration samples?\n"
        "White slice contour = within 25 mm of a current 12-ball training point; x = priority gap",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _bounds_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (6,):
        raise ValueError("tank bounds require XMIN XMAX YMIN YMAX ZMIN ZMAX")
    bounds = array.reshape(3, 2)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each tank maximum must be greater than its minimum")
    return bounds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize rigid-object residuals and geometric coverage in the tank"
    )
    parser.add_argument("--image-coords-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--base-model-folder", type=Path, required=True)
    parser.add_argument("--refined-model-folder", type=Path, required=True)
    parser.add_argument(
        "--base-mapping-csv",
        type=Path,
        required=True,
        help="base calibration point mapping containing X,Y,Z",
    )
    parser.add_argument(
        "--training-coords-csv",
        type=Path,
        default=None,
        help="subset of image coordinates used in the current rigid-object fit; default: all poses",
    )
    parser.add_argument(
        "--heldout-pose-list",
        type=Path,
        default=None,
        help="one held-out pose_id per line; default: poses absent from --training-coords-csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tank-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=(0, 200, 0, 200, 0, 200),
    )
    parser.add_argument("--coverage-radius-mm", type=float, default=25.0)
    parser.add_argument("--qmc-power", type=int, default=18)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bounds = _bounds_array(args.tank_bounds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_coords = pd.read_csv(args.image_coords_csv)
    image_coords["pose_id"] = image_coords["pose_id"].astype(str)
    geometry = _geometry_lookup(args.geometry_csv)
    all_pose_ids = set(image_coords["pose_id"])
    if args.training_coords_csv:
        training_coords = pd.read_csv(args.training_coords_csv)
        training_pose_ids = set(training_coords["pose_id"].astype(str))
    else:
        training_pose_ids = all_pose_ids
    heldout_pose_ids = _load_pose_ids(args.heldout_pose_list)
    if heldout_pose_ids is None:
        heldout_pose_ids = all_pose_ids.difference(training_pose_ids)
    if not heldout_pose_ids:
        raise ValueError("no held-out poses available for residual visualization")

    # The base model anchors unknown object poses to the established tank frame.
    all_base = reconstruct_rigid_poses(
        image_coords, geometry, args.base_model_folder, pose_ids=all_pose_ids
    )
    heldout_base = all_base[all_base["pose_id"].isin(heldout_pose_ids)].copy()
    heldout_refined = reconstruct_rigid_poses(
        image_coords, geometry, args.refined_model_folder, pose_ids=heldout_pose_ids
    )
    residuals = merge_model_residuals(heldout_base, heldout_refined)
    residual_csv = args.output_dir / "rigid_object_heldout_residuals.csv"
    residuals.to_csv(residual_csv, index=False)

    base_mapping = pd.read_csv(args.base_mapping_csv)
    sample_sets = {
        "base_beads": _unique_points(base_mapping),
        "current_12ball_training": _unique_points(
            all_base[all_base["pose_id"].isin(training_pose_ids)]
        ),
        "all_accepted_12ball": _unique_points(all_base),
    }
    tank_points = _tank_qmc_points(bounds, power=args.qmc_power)
    radii = sorted(set((*DEFAULT_RADII_MM, float(args.coverage_radius_mm))))
    summaries = {
        name: summarize_sampling_set(points, tank_points, bounds, radii)
        for name, points in sample_sets.items()
    }
    priority_gaps = find_priority_gaps(sample_sets["current_12ball_training"], bounds)

    residual_png = args.output_dir / "rigid_object_residuals_in_tank.png"
    coverage_png = args.output_dir / "rigid_object_training_coverage.png"
    plot_residual_dashboard(residuals, bounds, residual_png)
    plot_coverage_dashboard(
        sample_sets,
        summaries,
        tank_points,
        priority_gaps,
        bounds,
        coverage_png,
        radius_marker_mm=args.coverage_radius_mm,
        n_training_poses=len(training_pose_ids),
        n_accepted_poses=len(all_pose_ids),
    )

    summary = {
        "definitions": {
            "residual": (
                "Per-marker Euclidean residual after a per-pose Kabsch fit. This measures "
                "shape/scale fidelity, not absolute localization."
            ),
            "convex_hull_fraction_of_tank": (
                "Sobol-estimated tank fraction inside the sample-point convex hull; a geometric "
                "interpolation-envelope proxy, not an accuracy guarantee."
            ),
            "tank_fraction_within_radius": (
                "Sobol-estimated tank fraction whose nearest measured 3D sample is within the "
                "stated radius; a sampling-density proxy, not an accuracy guarantee."
            ),
        },
        "tank_bounds_mm": {
            axis: [float(bounds[index, 0]), float(bounds[index, 1])]
            for index, axis in enumerate(XYZ_COLS)
        },
        "n_poses": {
            "all_accepted": int(len(all_pose_ids)),
            "current_training": int(len(training_pose_ids)),
            "heldout": int(len(heldout_pose_ids)),
        },
        "heldout_residuals": {
            "base": summarize_residual(residuals["base_residual_mm"].to_numpy()),
            "refined": summarize_residual(residuals["refined_residual_mm"].to_numpy()),
        },
        "sampling_sets": summaries,
        "priority_gaps": priority_gaps,
        "outputs": {
            "residual_csv": str(residual_csv),
            "residual_figure": str(residual_png),
            "coverage_figure": str(coverage_png),
        },
    }
    summary_path = args.output_dir / "rigid_object_volume_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
