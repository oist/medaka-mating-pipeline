"""Plot native-frame 3D and projected 2D motion for F/M1/M2.

The plotted quantity is always the displacement between consecutive source
frames.  For a readable full-experiment overview, each horizontal sample is
the maximum native-frame displacement inside one source-second bin.  Exact
per-frame values for the largest events are written to CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


FISH_IDS = ("F", "M1", "M2")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
COLORS = {"x": "#2474B5", "y": "#E68613", "z": "#C14B82", "norm": "#252A31"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_csv", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--jump-3d-mm", type=float, default=8.0)
    parser.add_argument("--jump-2d-px", type=float, default=50.0)
    parser.add_argument("--top-events", type=int, default=500)
    return parser


def load_projected_motion(track_csv: Path, calibration: Calibration) -> pd.DataFrame:
    required = ["frame", "id_final", "X", "Y", "Z"]
    header = pd.read_csv(track_csv, nrows=0).columns
    missing = set(required) - set(header)
    if missing:
        raise ValueError("track is missing columns: " + ", ".join(sorted(missing)))
    optional = [name for name in ("geometry_slot", "id_source", "id_conf") if name in header]
    track = pd.read_csv(track_csv, usecols=required + optional, low_memory=False)
    track = track[track["id_final"].isin(FISH_IDS)].copy()
    parts: list[pd.DataFrame] = []

    for fish_id in FISH_IDS:
        fish = track[track["id_final"].eq(fish_id)].sort_values("frame", kind="stable").copy()
        if fish.duplicated("frame").any():
            raise ValueError(f"duplicate {fish_id} rows for a source frame")
        frames = fish["frame"].to_numpy(dtype=np.int64, copy=False)
        xyz = fish[["X", "Y", "Z"]].to_numpy(dtype=float, copy=False)
        finite = np.isfinite(xyz).all(axis=1)
        image_xy = np.full((len(fish), 3, 2), np.nan, dtype=float)
        if finite.any():
            image_xy[finite] = calibration.compute_imCoords_from_XYZ(xyz[finite])
        consecutive = np.r_[False, np.diff(frames) == 1]

        xyz_delta = np.vstack([np.full((1, 3), np.nan), np.diff(xyz, axis=0)])
        xyz_valid = consecutive & np.isfinite(xyz_delta).all(axis=1)
        xyz_delta[~xyz_valid] = np.nan
        for index, axis in enumerate("xyz"):
            fish[f"d{axis}"] = xyz_delta[:, index]
            fish[f"abs_d{axis}"] = np.abs(xyz_delta[:, index])
        fish["step_3d_mm"] = np.linalg.norm(xyz_delta, axis=1)

        for view_index, view in enumerate(VIEWS):
            delta = np.vstack(
                [np.full((1, 2), np.nan), np.diff(image_xy[:, view_index], axis=0)]
            )
            valid = consecutive & np.isfinite(delta).all(axis=1)
            delta[~valid] = np.nan
            fish[f"{view}_x_px"] = image_xy[:, view_index, 0]
            fish[f"{view}_y_px"] = image_xy[:, view_index, 1]
            fish[f"{view}_dx_px"] = delta[:, 0]
            fish[f"{view}_dy_px"] = delta[:, 1]
            fish[f"{view}_abs_dx_px"] = np.abs(delta[:, 0])
            fish[f"{view}_abs_dy_px"] = np.abs(delta[:, 1])
            fish[f"{view}_step_px"] = np.linalg.norm(delta, axis=1)
        parts.append(fish)

    return pd.concat(parts, ignore_index=True)


def _per_second_max(data: pd.DataFrame, columns: list[str], fps: float) -> pd.DataFrame:
    compact = data[["frame", "id_final", *columns]].copy()
    compact["second"] = np.floor(compact["frame"] / fps).astype(np.int64)
    return (
        compact.groupby(["id_final", "second"], observed=True)[columns]
        .max()
        .reset_index()
    )


def _plot_three_fish(
    maxima: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    output: Path,
    title: str,
    ylabel: str,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(17, 10), sharex=True, sharey=True)
    for axis, fish_id in zip(axes, FISH_IDS, strict=True):
        fish = maxima[maxima["id_final"].eq(fish_id)]
        minutes = fish["second"] / 60.0
        for column, label, color in zip(columns, labels, colors, strict=True):
            axis.plot(minutes, fish[column], label=label, color=color, linewidth=1.0)
        axis.axhline(threshold, color="#737B86", linewidth=0.9, linestyle=(0, (4, 3)))
        axis.set_yscale("symlog", linthresh=max(0.25, threshold / 20.0))
        axis.set_ylabel(f"{fish_id}\n{ylabel}", fontweight="bold")
        axis.grid(axis="y", color="#E1E5EA", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Experiment time (minutes)")
    handles, labels_found = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_found,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        ncol=len(columns),
        frameon=False,
    )
    fig.suptitle(title, x=0.07, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.07,
        0.962,
        "Consecutive source-frame displacement; maximum native-frame value in each 1-second bin. Log-like y scale preserves normal motion and spikes.",
        ha="left",
        fontsize=9.5,
        color="#525A65",
    )
    fig.subplots_adjust(top=0.90, left=0.08, right=0.98, bottom=0.07, hspace=0.13)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def write_outputs(
    motion: pd.DataFrame,
    *,
    output_dir: Path,
    fps: float,
    jump_3d_mm: float,
    jump_2d_px: float,
    top_events: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_columns = ["abs_dx", "abs_dy", "abs_dz", "step_3d_mm"]
    xyz_max = _per_second_max(motion, xyz_columns, fps)
    xyz_plot = output_dir / "speed_3d_native_frame.png"
    _plot_three_fish(
        xyz_max,
        columns=("abs_dx", "abs_dy", "abs_dz", "step_3d_mm"),
        labels=("|ΔX|", "|ΔY|", "|ΔZ|", "3D norm"),
        colors=(COLORS["x"], COLORS["y"], COLORS["z"], COLORS["norm"]),
        output=xyz_plot,
        title="K_1_2 round7 native-frame 3D displacement",
        ylabel="mm/frame",
        threshold=jump_3d_mm,
    )

    plot_paths = {"3d": str(xyz_plot)}
    for view in VIEWS:
        columns = [
            f"{view}_abs_dx_px",
            f"{view}_abs_dy_px",
            f"{view}_step_px",
        ]
        maxima = _per_second_max(motion, columns, fps)
        output = output_dir / f"speed_2d_{view}_native_frame.png"
        _plot_three_fish(
            maxima,
            columns=tuple(columns),
            labels=("|Δx|", "|Δy|", "2D norm"),
            colors=(COLORS["x"], COLORS["y"], COLORS["norm"]),
            output=output,
            title=f"K_1_2 round7 projected native-frame displacement: {view}",
            ylabel="px/frame",
            threshold=jump_2d_px,
        )
        plot_paths[view] = str(output)

    event_score_columns = ["step_3d_mm", *[f"{view}_step_px" for view in VIEWS]]
    candidate_indices: set[int] = set()
    for column in event_score_columns:
        candidate_indices.update(motion.nlargest(top_events, column).index.tolist())
    events = motion.loc[sorted(candidate_indices)].copy()
    events["jump_3d"] = events["step_3d_mm"].gt(jump_3d_mm)
    for view in VIEWS:
        events[f"jump_{view}"] = events[f"{view}_step_px"].gt(jump_2d_px)
    events = events.sort_values("step_3d_mm", ascending=False, kind="stable")
    event_csv = output_dir / "multiview_speed_top_events.csv"
    events.to_csv(event_csv, index=False)

    summary: dict[str, object] = {
        "source_fps": fps,
        "jump_3d_mm": jump_3d_mm,
        "jump_2d_px": jump_2d_px,
        "plots": plot_paths,
        "event_csv": str(event_csv),
        "fish": {},
    }
    for fish_id in FISH_IDS:
        fish = motion[motion["id_final"].eq(fish_id)]
        entry: dict[str, object] = {
            "max_3d_mm_per_frame": float(fish["step_3d_mm"].max()),
            "jumps_3d": int(fish["step_3d_mm"].gt(jump_3d_mm).sum()),
            "views": {},
        }
        for view in VIEWS:
            entry["views"][view] = {
                "max_abs_dx_px_per_frame": float(fish[f"{view}_dx_px"].abs().max()),
                "max_abs_dy_px_per_frame": float(fish[f"{view}_dy_px"].abs().max()),
                "max_2d_px_per_frame": float(fish[f"{view}_step_px"].max()),
                "jumps_2d": int(fish[f"{view}_step_px"].gt(jump_2d_px).sum()),
            }
        summary["fish"][fish_id] = entry
    summary_path = output_dir / "multiview_speed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    calibration = Calibration(str(args.calibration_model_dir))
    motion = load_projected_motion(args.track_csv, calibration)
    summary = write_outputs(
        motion,
        output_dir=args.output_dir,
        fps=args.source_fps,
        jump_3d_mm=args.jump_3d_mm,
        jump_2d_px=args.jump_2d_px,
        top_events=args.top_events,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
