from __future__ import annotations

"""Render reviewable figures for silhouette/SLEAP fusion experiments."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#263238"
GRID = "#D9DEE3"
BLUE = "#3B6FB6"
GOLD = "#D3A13B"
ORANGE = "#D87932"
OLIVE = "#708B43"
PINK = "#B64E67"
GREY = "#9AA3AA"
LIGHT_GREY = "#E8ECEF"
FISH_COLORS = {"ID_1": BLUE, "ID_2": ORANGE, "ID_3": OLIVE}
STATUS_COLORS = {
    "strong": BLUE,
    "supported": GOLD,
    "ambiguous": ORANGE,
    "no mask": GREY,
    "hard conflict": PINK,
}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GREY)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.tick_params(colors=INK, labelsize=9)
    axis.title.set_color(INK)


def _status_group(value: object) -> str:
    status = str(value)
    if status == "strong_match":
        return "strong"
    if status == "supported_match":
        return "supported"
    if status == "no_mask":
        return "no mask"
    if status.startswith("conflict"):
        return "hard conflict"
    return "ambiguous"


def _first_match(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {folder}")
    return matches[0]


def render_pilot_overview(pilot_dir: Path, output_path: Path) -> None:
    views = ("side1_cam", "side2_cam")
    fused = {
        view: pd.read_csv(pilot_dir / f"{view}.silhouette_fused_tracklets.csv")
        for view in views
    }
    links = {
        view: pd.read_csv(pilot_dir / f"{view}.silhouette_link_candidates.csv")
        for view in views
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        "Silhouette–SLEAP fusion pilot overview",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )

    axis = axes[0, 0]
    categories = ["strong", "supported", "ambiguous", "no mask", "hard conflict"]
    left = np.zeros(len(views))
    for category in categories:
        values = []
        for view in views:
            groups = fused[view]["silhouette_status"].map(_status_group)
            values.append(100.0 * float(groups.eq(category).mean()))
        axis.barh(
            ["Side 1", "Side 2"],
            values,
            left=left,
            color=STATUS_COLORS[category],
            edgecolor="white",
            linewidth=0.8,
            label=category,
        )
        left += np.asarray(values)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of SLEAP observations (%)")
    axis.set_title("Observation-level silhouette evidence", loc="left", fontweight="bold")
    axis.legend(ncol=3, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    _style_axis(axis)

    axis = axes[0, 1]
    conflict_labels = [
        "Mask merges skeletons",
        "Low overlap",
        "Split endpoints",
        "Temporal discontinuity",
        "Area discontinuity",
    ]
    conflict_codes = [
        "conflict_mask_merges_skeletons",
        "conflict_low_overlap",
        "conflict_split_endpoints",
        "mask_temporal_discontinuity",
        "mask_area_discontinuity",
    ]
    y = np.arange(len(conflict_codes))
    width = 0.36
    for offset, view, color, label in (
        (-width / 2, "side1_cam", PINK, "Side 1"),
        (width / 2, "side2_cam", BLUE, "Side 2"),
    ):
        reasons = fused[view]["silhouette_break_reason"].fillna("").value_counts()
        counts = [int(reasons.get(code, 0)) for code in conflict_codes]
        axis.barh(y + offset, counts, height=width, color=color, label=label)
        for yy, value in zip(y + offset, counts):
            axis.text(value + 8, yy, str(value), va="center", fontsize=8, color=INK)
    axis.set_yticks(y, conflict_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Observation rows")
    axis.set_title("Hard-break causes", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)

    axis = axes[1, 0]
    thresholds = np.unique(np.r_[np.arange(1, 31), np.arange(35, 105, 5), np.arange(125, 1201, 25)])
    for view, color, label, linestyle in (
        ("side1_cam", PINK, "Side 1", "-"),
        ("side2_cam", BLUE, "Side 2", "--"),
    ):
        accepted = fused[view][fused[view]["fused_tracklet_id"].ge(0)]
        lengths = accepted.groupby("fused_tracklet_id").size().to_numpy()
        survivor = np.asarray([(lengths >= threshold).sum() for threshold in thresholds])
        axis.plot(thresholds, survivor, color=color, linewidth=2.2, linestyle=linestyle, label=label)
    axis.axvline(30, color=INK, linewidth=1, linestyle=":")
    axis.axvline(100, color=INK, linewidth=1, linestyle=":")
    axis.text(30, axis.get_ylim()[1] * 0.88, "30 f", ha="center", fontsize=8, color=INK)
    axis.text(100, axis.get_ylim()[1] * 0.75, "100 f", ha="center", fontsize=8, color=INK)
    axis.set_xscale("log")
    axis.set_xlabel("Minimum fragment length (frames, log scale)")
    axis.set_ylabel("Fragments at least this long")
    axis.set_title("Concrete-fragment survival curve", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    _style_axis(axis)

    axis = axes[1, 1]
    for y_value, view, color, label in (
        (1, "side1_cam", PINK, "Side 1"),
        (0, "side2_cam", BLUE, "Side 2"),
    ):
        table = links[view]
        table = table[
            table["silhouette_link_status"].eq("candidate")
            & table["gap_frames"].between(0, 3)
            & table["mean_silhouette_weight"].ge(0.5)
        ].copy()
        midpoint = 0.5 * (table["left_frame_end"] + table["right_frame_start"])
        sizes = 18 + 42 * np.clip(table["mean_silhouette_weight"], 0, 1)
        axis.scatter(
            midpoint,
            np.full(len(table), y_value),
            s=sizes,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.45,
            alpha=0.82,
            label=f"{label} ({len(table)})",
        )
    axis.axvspan(11250, 11320, color=ORANGE, alpha=0.12)
    axis.text(11285, 1.32, "crossing QC window", ha="center", fontsize=8, color=INK)
    axis.set_yticks([0, 1], ["Side 2", "Side 1"])
    axis.set_ylim(-0.5, 1.55)
    axis.set_xlabel("Frame")
    axis.set_title("Trusted mask-supported fragment links", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _framewise_displacement(group: pd.DataFrame, xyz: list[str]) -> pd.DataFrame:
    group = group.groupby("frame", as_index=False)[xyz].median().sort_values("frame")
    delta = group[xyz].diff().to_numpy(float)
    dt = group["frame"].diff().to_numpy(float)
    group["displacement"] = np.linalg.norm(delta, axis=1) / np.where(dt > 0, dt, np.nan)
    return group


def _three_d_kinematics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fish_id, group in table.groupby("fish_id"):
        group = group.sort_values("frame").copy()
        points = group[["X", "Y", "Z"]].to_numpy(float)
        frames = group["frame"].to_numpy(int)
        velocity = np.full_like(points, np.nan)
        dt = np.diff(frames).astype(float)
        velocity[1:] = np.diff(points, axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        angle = np.full(len(group), np.nan)
        for index in range(2, len(group)):
            left, right = velocity[index - 1], velocity[index]
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            if denominator > 1e-12:
                angle[index] = np.degrees(
                    np.arccos(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
                )
        rows.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "fish_id": fish_id,
                    "speed_mm_per_frame": speed,
                    "turn_angle_deg": angle,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def render_crossing_kinematics(crossing_dir: Path, output_path: Path) -> None:
    identified = pd.read_csv(_first_match(crossing_dir, "*identified_2d.csv"), low_memory=False)
    xyz = pd.read_csv(_first_match(crossing_dir, "*_3d.csv"))
    conflicts = pd.read_csv(_first_match(crossing_dir, "*candidate_conflicts.csv"))
    kinematics = _three_d_kinematics(xyz)

    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True, constrained_layout=True)
    fig.suptitle("2D and 3D motion diagnostics around the crossing", fontsize=17, fontweight="bold", color=INK)

    for axis, view, label in zip(axes[:3], ("top_cam", "side1_cam", "side2_cam"), ("Top", "Side 1", "Side 2")):
        view_rows = identified[
            identified["view"].eq(view)
            & identified["final_fish_id"].astype(str).isin(FISH_COLORS)
        ]
        for fish_id, color in FISH_COLORS.items():
            group = view_rows[view_rows["final_fish_id"].astype(str).eq(fish_id)]
            if group.empty:
                continue
            motion = _framewise_displacement(group, ["x", "y"])
            axis.plot(motion.frame, motion.displacement, color=color, linewidth=1.7, marker="o", markersize=2.5, label=fish_id)
        axis.set_ylabel("px/frame")
        axis.set_title(f"{label} camera 2D displacement", loc="left", fontweight="bold")
        _style_axis(axis)

    axis = axes[3]
    for fish_id, color in FISH_COLORS.items():
        group = kinematics[kinematics.fish_id.astype(str).eq(fish_id)]
        axis.plot(group.frame, group.speed_mm_per_frame, color=color, linewidth=1.9, marker="o", markersize=2.6, label=fish_id)
        spikes = group[group.speed_mm_per_frame > 8]
        axis.scatter(spikes.frame, spikes.speed_mm_per_frame, s=48, facecolor="none", edgecolor=PINK, linewidth=1.5, zorder=5)
    axis.axhline(8, color=PINK, linestyle="--", linewidth=1.2, label="8 mm/frame gate")
    axis.set_ylabel("mm/frame")
    axis.set_title("3D speed", loc="left", fontweight="bold")
    _style_axis(axis)

    axis = axes[4]
    for fish_id, color in FISH_COLORS.items():
        group = kinematics[kinematics.fish_id.astype(str).eq(fish_id)]
        axis.plot(group.frame, group.turn_angle_deg, color=color, linewidth=1.7, marker="o", markersize=2.5, label=fish_id)
    axis.axhline(120, color=PINK, linestyle="--", linewidth=1.2, label="120° link gate")
    axis.set_ylabel("degrees")
    axis.set_xlabel("Frame")
    axis.set_title("3D displacement-vector angle", loc="left", fontweight="bold")
    _style_axis(axis)

    conflict_frames = sorted(conflicts["frame"].dropna().astype(int).unique()) if not conflicts.empty else []
    for axis in axes:
        axis.axvspan(11260, 11288, color=ORANGE, alpha=0.075, zorder=-5)
        for frame in conflict_frames:
            axis.axvline(frame, color=PINK, linewidth=0.8, linestyle=":", alpha=0.7)
        axis.set_xlim(11250, 11320)
    handles = [Line2D([0], [0], color=color, lw=2, marker="o", markersize=4, label=fish_id) for fish_id, color in FISH_COLORS.items()]
    handles += [
        Line2D([0], [0], color=PINK, lw=1, linestyle=":", label="identity conflict veto"),
        Line2D([0], [0], color=ORANGE, lw=8, alpha=0.25, label="reported swap interval"),
    ]
    axes[0].legend(handles=handles, ncol=5, frameon=False, fontsize=8, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_crossing_evidence(pilot_dir: Path, crossing_dir: Path, output_path: Path) -> None:
    conflicts = pd.read_csv(_first_match(crossing_dir, "*candidate_conflicts.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    fig.suptitle("Silhouette evidence and identity vetoes around the crossing", fontsize=17, fontweight="bold", color=INK)

    for axis, view, title in zip(axes[:2], ("side1_cam", "side2_cam"), ("Side 1", "Side 2")):
        table = pd.read_csv(pilot_dir / f"{view}.silhouette_fused_tracklets.csv", low_memory=False)
        table = table[table["frame"].between(11250, 11320)].copy()
        first = table.groupby("tracklet_id")["frame"].min().sort_values()
        lanes = {tracklet_id: lane for lane, tracklet_id in enumerate(first.index)}
        table["lane"] = table["tracklet_id"].map(lanes)
        table["status_group"] = table["silhouette_status"].map(_status_group)
        for status in ("no mask", "ambiguous", "supported", "strong"):
            group = table[table.status_group.eq(status)]
            axis.scatter(
                group.frame,
                group.lane,
                s=24,
                marker="o",
                facecolor=STATUS_COLORS[status],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
                label=status,
            )
        hard = table[table.status_group.eq("hard conflict") | ~table.silhouette_accepted.astype(bool)]
        axis.scatter(hard.frame, hard.lane, s=46, marker="x", color=PINK, linewidth=1.5, label="hard conflict")
        axis.set_yticks(list(lanes.values()), [str(value) for value in lanes])
        axis.set_ylabel("source tracklet")
        axis.set_title(f"{title}: mask support for SLEAP observations", loc="left", fontweight="bold")
        _style_axis(axis)

    axis = axes[2]
    if not conflicts.empty:
        for source_idx, group in conflicts.groupby("source_idx"):
            axis.scatter(group.frame, np.full(len(group), int(source_idx)), marker="X", s=80, color=PINK, edgecolor=INK, linewidth=0.5)
    axis.set_yticks([0, 1, 2], ["3D source a", "3D source b", "3D source c"])
    axis.set_xlabel("Frame")
    axis.set_title("Global identity conflicts vetoed before motion linking", loc="left", fontweight="bold")
    _style_axis(axis)

    for axis in axes:
        axis.axvspan(11260, 11288, color=ORANGE, alpha=0.075, zorder=-5)
        axis.set_xlim(11250, 11320)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=STATUS_COLORS[key], markeredgecolor="white", label=key)
        for key in ("strong", "supported", "ambiguous", "no mask")
    ]
    handles.append(Line2D([0], [0], marker="x", linestyle="", color=PINK, label="hard conflict / removed"))
    axes[0].legend(handles=handles, ncol=5, frameon=False, fontsize=8, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--crossing-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pilot_overview": args.output_dir / "silhouette_fusion_pilot_overview.png",
        "crossing_kinematics": args.output_dir / "crossing_2d_3d_kinematics.png",
        "crossing_evidence": args.output_dir / "crossing_silhouette_evidence.png",
    }
    render_pilot_overview(args.pilot_dir, outputs["pilot_overview"])
    render_crossing_kinematics(args.crossing_dir, outputs["crossing_kinematics"])
    render_crossing_evidence(args.pilot_dir, args.crossing_dir, outputs["crossing_evidence"])
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
