"""Plot per-axis 3D motion and report the largest trajectory jumps.

The native-cadence plot uses consecutive source frames.  A second plot uses the
same stride as a subsampled QC video, which helps distinguish reconstruction
jumps from motion that only looks discontinuous because frames were skipped.
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


FISH_ORDER = ("F", "M1", "M2")
COLORS = {
    "abs_dX": "#2474B5",
    "abs_dY": "#E68613",
    "abs_dZ": "#C14B82",
    "step_3d": "#252A31",
}
LABELS = {
    "abs_dX": "|ΔX|",
    "abs_dY": "|ΔY|",
    "abs_dZ": "|ΔZ|",
    "step_3d": "3D",
}
LINESTYLES = {
    "abs_dX": "-",
    "abs_dY": "--",
    "abs_dZ": ":",
    "step_3d": "-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--id-column",
        default="id_final",
        help="Track identity column (default: id_final).",
    )
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        metavar="SOURCE=DISPLAY",
        help=(
            "Source identity and displayed label; repeat in desired panel order. "
            "When omitted, identities are inferred from --id-column."
        ),
    )
    parser.add_argument("--fps", type=float, default=166.0)
    parser.add_argument(
        "--render-stride",
        type=int,
        default=10,
        help="Source-frame stride used by the QC video.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=8.0,
        help="Native-frame 3D displacement threshold in mm/frame.",
    )
    parser.add_argument("--top-events", type=int, default=100)
    return parser.parse_args()


def parse_identity_specs(specs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--identity expects SOURCE=DISPLAY, got {spec!r}")
        source, display = (part.strip() for part in spec.split("=", 1))
        if not source or not display:
            raise ValueError(f"--identity expects SOURCE=DISPLAY, got {spec!r}")
        if source in mapping:
            raise ValueError(f"Duplicate --identity source: {source}")
        if display in mapping.values():
            raise ValueError(f"Duplicate --identity display label: {display}")
        mapping[source] = display
    return mapping


def load_tracks(
    path: Path,
    *,
    id_column: str = "id_final",
    identity_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not id_column:
        raise ValueError("--id-column must be non-empty")
    columns = [
        "frame",
        "X",
        "Y",
        "Z",
        id_column,
        "id_final_raw",
        "id_conf",
        "id_source",
        "geometry_slot",
    ]
    header = pd.read_csv(path, nrows=0).columns
    usecols = list(dict.fromkeys(column for column in columns if column in header))
    required = {"frame", "X", "Y", "Z", id_column}
    missing = required.difference(usecols)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = pd.read_csv(path, usecols=usecols)
    if id_column != "id_final":
        data = data.rename(columns={id_column: "id_final"})
    data["id_final"] = data["id_final"].astype(str)

    if identity_map:
        data = data[data["id_final"].isin(identity_map)].copy()
        data["id_final"] = data["id_final"].map(identity_map)
        fish_order = tuple(identity_map.values())
    else:
        identities = set(data["id_final"].dropna().astype(str))
        if set(FISH_ORDER).issubset(identities):
            fish_order = FISH_ORDER
        else:
            fish_order = tuple(sorted(identities))
        data = data[data["id_final"].isin(fish_order)].copy()

    if len(fish_order) != 3:
        raise ValueError(
            "Expected exactly three identities; resolved "
            f"{len(fish_order)}: {list(fish_order)}"
        )
    data = data.sort_values(["id_final", "frame"], kind="stable")

    counts = data.groupby(["id_final", "frame"], observed=True).size()
    if counts.max() > 1:
        raise ValueError("Expected at most one row per id_final and frame")
    return data, fish_order


def add_motion(data: pd.DataFrame, stride: int, prefix: str) -> pd.DataFrame:
    result = data.copy()
    grouped = result.groupby("id_final", observed=True, sort=False)
    frame_delta = grouped["frame"].diff(stride)
    valid_frame_delta = frame_delta.eq(stride)

    for axis in "XYZ":
        delta = grouped[axis].diff(stride).where(valid_frame_delta)
        result[f"{prefix}_d{axis}"] = delta
        result[f"{prefix}_abs_d{axis}"] = delta.abs()

    result[f"{prefix}_step_3d"] = np.sqrt(
        sum(result[f"{prefix}_d{axis}"] ** 2 for axis in "XYZ")
    )
    return result


def second_maxima(
    data: pd.DataFrame, *, fps: float, prefix: str
) -> pd.DataFrame:
    metric_columns = [
        f"{prefix}_abs_dX",
        f"{prefix}_abs_dY",
        f"{prefix}_abs_dZ",
        f"{prefix}_step_3d",
    ]
    compact = data[["frame", "id_final", *metric_columns]].copy()
    compact["second"] = np.floor(compact["frame"] / fps).astype(int)
    maxima = (
        compact.groupby(["id_final", "second"], observed=True)[metric_columns]
        .max()
        .reset_index()
    )
    return maxima.rename(
        columns={column: column.removeprefix(f"{prefix}_") for column in metric_columns}
    )


def plot_overview(
    maxima: pd.DataFrame,
    *,
    output: Path,
    title: str,
    subtitle: str,
    threshold: float | None = None,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    metric_columns = ("abs_dX", "abs_dY", "abs_dZ", "step_3d")
    finite_max = float(maxima[list(metric_columns)].max().max())
    ymax = max(0.25, finite_max * 1.06)

    for axis, fish in zip(axes, FISH_ORDER, strict=True):
        fish_data = maxima[maxima["id_final"] == fish]
        minutes = fish_data["second"] / 60.0
        for metric in metric_columns:
            axis.plot(
                minutes,
                fish_data[metric],
                color=COLORS[metric],
                linestyle=LINESTYLES[metric],
                linewidth=1.05 if metric != "step_3d" else 1.35,
                alpha=0.88,
                label=LABELS[metric],
            )
        if threshold is not None and threshold <= ymax:
            axis.axhline(
                threshold,
                color="#6B7280",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                label=f"{threshold:g} mm/frame threshold",
            )
        axis.set_ylim(0, ymax)
        axis.set_ylabel(f"{fish}\nmm/step", fontweight="bold")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

        maximum_row = fish_data.loc[fish_data["step_3d"].idxmax()]
        axis.scatter(
            [maximum_row["second"] / 60.0],
            [maximum_row["step_3d"]],
            color="#252A31",
            edgecolor="white",
            linewidth=0.8,
            s=38,
            zorder=5,
        )
        axis.annotate(
            f"max {maximum_row['step_3d']:.2f}",
            (maximum_row["second"] / 60.0, maximum_row["step_3d"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#252A31",
        )

    axes[-1].set_xlabel("Experiment time (minutes)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(unique),
        frameon=False,
    )
    fig.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.07, 0.952, subtitle, ha="left", va="top", fontsize=10, color="#525A65")
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.07, hspace=0.12)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def build_event_table(data: pd.DataFrame, top_events: int) -> pd.DataFrame:
    event_columns = [
        "frame",
        "id_final",
        "native_dX",
        "native_dY",
        "native_dZ",
        "native_step_3d",
        "X",
        "Y",
        "Z",
    ]
    optional = ["id_conf", "id_source", "id_final_raw"]
    events = data[event_columns + [column for column in optional if column in data]].copy()

    for column in optional:
        if column in data:
            events[f"previous_{column}"] = data.groupby(
                "id_final", observed=True, sort=False
            )[column].shift(1)

    # The nearest-previous-identity test is intentionally an outlier diagnostic,
    # not a million-row all-pairs calculation.
    events = (
        events.dropna(subset=["native_step_3d"])
        .sort_values("native_step_3d", ascending=False)
        .head(top_events)
        .copy()
    )

    previous_positions = (
        data.pivot(index="frame", columns="id_final", values=["X", "Y", "Z"])
        .shift(1)
    )
    nearest_previous: list[str | None] = []
    nearest_previous_distance: list[float] = []
    closer_to_other_previous: list[bool] = []

    for row in events.itertuples(index=False):
        if row.frame not in previous_positions.index or not np.isfinite(row.native_step_3d):
            nearest_previous.append(None)
            nearest_previous_distance.append(np.nan)
            closer_to_other_previous.append(False)
            continue

        distances: dict[str, float] = {}
        current = np.array([row.X, row.Y, row.Z], dtype=float)
        for fish in FISH_ORDER:
            try:
                previous = np.array(
                    [
                        previous_positions.loc[row.frame, (axis, fish)]
                        for axis in "XYZ"
                    ],
                    dtype=float,
                )
            except KeyError:
                continue
            if np.isfinite(current).all() and np.isfinite(previous).all():
                distances[fish] = float(np.linalg.norm(current - previous))

        if not distances:
            nearest_previous.append(None)
            nearest_previous_distance.append(np.nan)
            closer_to_other_previous.append(False)
            continue
        nearest = min(distances, key=distances.get)
        nearest_previous.append(nearest)
        nearest_previous_distance.append(distances[nearest])
        closer_to_other_previous.append(nearest != row.id_final)

    events["nearest_previous_identity"] = nearest_previous
    events["nearest_previous_distance_mm"] = nearest_previous_distance
    events["closer_to_other_previous_identity"] = closer_to_other_previous

    simultaneous = data.pivot(index="frame", columns="id_final", values="native_step_3d")
    simultaneous_counts = simultaneous.gt(2.0).sum(axis=1)
    events["fish_above_2mm_same_frame"] = events["frame"].map(simultaneous_counts)

    return events.reset_index(drop=True)


def plot_top_event_zooms(
    data: pd.DataFrame, *, fps: float, output: Path
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=False)
    signed_colors = {"native_dX": "#2474B5", "native_dY": "#E68613", "native_dZ": "#C14B82"}
    signed_styles = {"native_dX": "-", "native_dY": "--", "native_dZ": ":"}

    for axis, fish in zip(axes, FISH_ORDER, strict=True):
        fish_data = data[data["id_final"] == fish]
        peak_index = fish_data["native_step_3d"].idxmax()
        peak_frame = int(data.loc[peak_index, "frame"])
        radius = int(round(fps))
        window = fish_data[fish_data["frame"].between(peak_frame - radius, peak_frame + radius)]
        relative_ms = (window["frame"] - peak_frame) / fps * 1000.0

        for metric in ("native_dX", "native_dY", "native_dZ"):
            axis.plot(
                relative_ms,
                window[metric],
                color=signed_colors[metric],
                linestyle=signed_styles[metric],
                linewidth=1.1,
                label=metric.removeprefix("native_"),
            )
        axis.plot(
            relative_ms,
            window["native_step_3d"],
            color="#252A31",
            linewidth=1.35,
            label="3D speed",
        )
        axis.axvline(0, color="#6B7280", linewidth=0.9, linestyle=(0, (4, 3)))
        axis.axhline(0, color="#AEB5BF", linewidth=0.7)
        axis.set_ylabel(f"{fish}\nmm/frame", fontweight="bold")
        axis.set_title(
            f"Largest native step at source frame {peak_frame:,}: "
            f"{data.loc[peak_index, 'native_step_3d']:.3f} mm",
            loc="left",
            fontsize=10,
        )
        axis.grid(axis="y", color="#E1E5EA", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time relative to largest step (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.97, 0.978), frameon=False, ncol=4)
    fig.suptitle(
        "Round7 K_1_2 largest native-cadence motion events",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.952,
        "Signed axis displacement and total 3D displacement; ±1 source second around each fish's maximum.",
        ha="left",
        va="top",
        fontsize=10,
        color="#525A65",
    )
    fig.subplots_adjust(top=0.88, left=0.09, right=0.98, bottom=0.07, hspace=0.34)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def summarize(
    data: pd.DataFrame,
    *,
    fps: float,
    render_stride: int,
    jump_threshold: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "source_fps": fps,
        "render_stride": render_stride,
        "native_jump_threshold_mm_per_frame": jump_threshold,
        "fish": {},
    }
    for fish in FISH_ORDER:
        fish_data = data[data["id_final"] == fish]
        native = fish_data["native_step_3d"].dropna()
        rendered = fish_data["render_step_3d"].dropna()
        peak_index = fish_data["native_step_3d"].idxmax()
        rendered_peak_index = fish_data["render_step_3d"].idxmax()
        result["fish"][fish] = {
            "native_valid_steps": int(native.size),
            "native_p95_mm_per_frame": scalar(native.quantile(0.95)),
            "native_p99_mm_per_frame": scalar(native.quantile(0.99)),
            "native_p99_9_mm_per_frame": scalar(native.quantile(0.999)),
            "native_max_mm_per_frame": scalar(native.max()),
            "native_max_mm_per_s": scalar(native.max() * fps),
            "native_max_frame": int(data.loc[peak_index, "frame"]),
            "native_steps_over_2mm": int(native.gt(2.0).sum()),
            "native_steps_over_4mm": int(native.gt(4.0).sum()),
            "native_steps_over_threshold": int(native.gt(jump_threshold).sum()),
            "render_stride_max_mm_per_display_step": scalar(rendered.max()),
            "render_stride_max_frame": int(data.loc[rendered_peak_index, "frame"]),
        }

    jump_matrix = data.pivot(index="frame", columns="id_final", values="native_step_3d")
    jump_flags = jump_matrix.gt(jump_threshold)
    jump_frames = jump_flags[jump_flags.any(axis=1)]
    categories = jump_frames.apply(
        lambda row: "+".join(fish for fish in FISH_ORDER if bool(row.get(fish, False))),
        axis=1,
    )
    diagnostic: dict[str, object] = {
        "unique_frames_with_any_jump": int(jump_frames.shape[0]),
        "jumping_fish_rows": int(jump_flags.to_numpy().sum()),
        "frames_by_fish_combination": {
            str(key): int(value) for key, value in categories.value_counts().items()
        },
        "single_fish_jump_frames": int(jump_frames.sum(axis=1).eq(1).sum()),
        "paired_fish_jump_frames": int(jump_frames.sum(axis=1).eq(2).sum()),
        "three_fish_jump_frames": int(jump_frames.sum(axis=1).eq(3).sum()),
    }

    jump_rows = data[data["native_step_3d"].gt(jump_threshold)]
    if "id_conf" in jump_rows:
        diagnostic["jump_rows_by_id_conf"] = {
            str(key): int(value)
            for key, value in jump_rows["id_conf"].fillna("missing").value_counts().items()
        }
    if "id_source" in jump_rows:
        diagnostic["jump_rows_by_id_source"] = {
            str(key): int(value)
            for key, value in jump_rows["id_source"].fillna("missing").value_counts().items()
        }

    if "geometry_slot" in data:
        mapping = data.pivot(index="frame", columns="id_final", values="geometry_slot")
        previous_mapping = mapping.shift(1)
        valid_mapping = mapping.notna().all(axis=1) & previous_mapping.notna().all(axis=1)
        mapping_changed = mapping.ne(previous_mapping).any(axis=1) & valid_mapping
        mapping_change_frames = set(mapping.index[mapping_changed])
        native_jump_frames = set(jump_frames.index)
        diagnostic["geometry_slot_mapping_change_frames"] = int(mapping_changed.sum())
        diagnostic["jump_frames_with_geometry_slot_mapping_change"] = int(
            len(native_jump_frames & mapping_change_frames)
        )
        diagnostic["jump_frames_without_geometry_slot_mapping_change"] = int(
            len(native_jump_frames - mapping_change_frames)
        )

        geometry = data.sort_values(["geometry_slot", "frame"], kind="stable").copy()
        geometry_grouped = geometry.groupby("geometry_slot", observed=True, sort=False)
        geometry_frame_delta = geometry_grouped["frame"].diff()
        geometry_delta = [
            geometry_grouped[axis].diff().where(geometry_frame_delta.eq(1))
            for axis in "XYZ"
        ]
        geometry["geometry_step_3d"] = np.sqrt(sum(delta**2 for delta in geometry_delta))
        diagnostic["geometry_slot_speed"] = {}
        for slot, slot_data in geometry.groupby("geometry_slot", observed=True):
            steps = slot_data["geometry_step_3d"].dropna()
            diagnostic["geometry_slot_speed"][str(int(slot))] = {
                "max_mm_per_frame": scalar(steps.max()),
                "steps_over_threshold": int(steps.gt(jump_threshold).sum()),
                "p99_9_mm_per_frame": scalar(steps.quantile(0.999)),
            }

    result["jump_diagnostic"] = diagnostic
    return result


def main() -> None:
    global FISH_ORDER
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.render_stride < 1:
        raise ValueError("--render-stride must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_map = parse_identity_specs(args.identity)
    tracks, fish_order = load_tracks(
        args.track_csv,
        id_column=args.id_column,
        identity_map=identity_map,
    )
    FISH_ORDER = fish_order
    tracks = add_motion(tracks, stride=1, prefix="native")
    tracks = add_motion(tracks, stride=args.render_stride, prefix="render")

    native_maxima = second_maxima(tracks, fps=args.fps, prefix="native")
    render_maxima = second_maxima(tracks, fps=args.fps, prefix="render")

    native_plot = args.output_dir / "xyz_speed_native_per_second_max.png"
    render_plot = (
        args.output_dir
        / f"xyz_displacement_render_stride{args.render_stride}_per_second_max.png"
    )
    zoom_plot = args.output_dir / "xyz_speed_top_event_zooms.png"
    event_csv = args.output_dir / "xyz_speed_top_events.csv"
    summary_json = args.output_dir / "xyz_speed_summary.json"

    plot_overview(
        native_maxima,
        output=native_plot,
        title="Round7 K_1_2 native-cadence 3D motion",
        subtitle=(
            f"Maximum displacement within each 1-second bin; consecutive source frames at "
            f"{args.fps:g} fps. Components are absolute; 3D is the vector norm."
        ),
        threshold=args.jump_threshold,
    )
    plot_overview(
        render_maxima,
        output=render_plot,
        title=f"Round7 K_1_2 apparent motion in the stride-{args.render_stride} QC video",
        subtitle=(
            f"Maximum displacement within each 1-second source-time bin; adjacent displayed "
            f"positions are {args.render_stride} source frames apart."
        ),
    )
    plot_top_event_zooms(tracks, fps=args.fps, output=zoom_plot)

    events = build_event_table(tracks, args.top_events)
    events.to_csv(event_csv, index=False)
    summary = summarize(
        tracks,
        fps=args.fps,
        render_stride=args.render_stride,
        jump_threshold=args.jump_threshold,
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {native_plot}")
    print(f"Wrote {render_plot}")
    print(f"Wrote {zoom_plot}")
    print(f"Wrote {event_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
