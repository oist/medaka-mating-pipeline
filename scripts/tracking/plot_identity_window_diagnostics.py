from __future__ import annotations

"""Plot per-view 2D displacement and 3D kinematics for an ID window.

The 2D series are reconstructed from the selected candidate slots in the
tracklet-first ``*_best.json`` and the corresponding coordinates in
``*_detection_dispositions.csv``.  This therefore measures the positions that
the QC overlay actually labels, rather than every raw detection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAMERA_TO_VIEW = {
    "cam0": "top_cam",
    "cam1": "side1_cam",
    "cam2": "side2_cam",
}

ID_STYLE = {
    "ID_1": {"color": "#3568A8", "marker": "o", "linestyle": "-"},
    "ID_2": {"color": "#D17A22", "marker": "s", "linestyle": "--"},
    "ID_3": {"color": "#7C8B3B", "marker": "^", "linestyle": ":"},
}


def selected_2d_positions(best_json: Path, dispositions_csv: Path) -> pd.DataFrame:
    """Return the selected 2D centroid for every frame/view/final ID."""
    payload = json.loads(best_json.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    dispositions = pd.read_csv(dispositions_csv)
    required = {"frame", "camera", "candidate_slot", "x", "y"}
    missing = required - set(dispositions.columns)
    if missing:
        raise ValueError(f"dispositions CSV missing columns: {sorted(missing)}")

    dispositions = dispositions.copy()
    dispositions["frame"] = dispositions["frame"].astype(int)
    dispositions["candidate_slot"] = dispositions["candidate_slot"].astype(int)
    lookup = {
        (int(row.frame), str(row.camera), int(row.candidate_slot)): row
        for row in dispositions.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for frame_text, info in frames.items():
        frame = int(frame_text)
        for camera, view in CAMERA_TO_VIEW.items():
            for slot, fish_id in enumerate(info.get(f"{camera}_to_id", [])):
                if not fish_id:
                    continue
                source = lookup.get((frame, camera, slot))
                if source is None:
                    continue
                rows.append(
                    {
                        "frame": frame,
                        "view": view,
                        "camera": camera,
                        "fish_id": str(fish_id),
                        "candidate_slot": slot,
                        "x": float(source.x),
                        "y": float(source.y),
                        "tracklet_node": getattr(source, "tracklet_node", np.nan),
                        "disposition": getattr(source, "disposition", ""),
                        "primary_reason": getattr(source, "primary_reason", ""),
                    }
                )

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values(["view", "fish_id", "frame"]).reset_index(drop=True)
    grouped = selected.groupby(["view", "fish_id"], sort=False)
    selected["previous_frame"] = grouped["frame"].shift()
    selected["frame_gap"] = selected["frame"] - selected["previous_frame"]
    selected["dx_px"] = grouped["x"].diff()
    selected["dy_px"] = grouped["y"].diff()
    selected["displacement_px"] = np.hypot(selected["dx_px"], selected["dy_px"])
    selected["displacement_px_per_frame"] = (
        selected["displacement_px"] / selected["frame_gap"]
    )
    selected["continuous_step"] = selected["frame_gap"] == 1
    return selected


def kinematics_3d(tracks_csv: Path, *, fps: float) -> pd.DataFrame:
    """Compute 3D speed and the angle between consecutive displacement vectors."""
    tracks = pd.read_csv(tracks_csv)
    required = {"frame", "fish_id", "X", "Y", "Z"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"3D CSV missing columns: {sorted(missing)}")

    tracks = tracks.sort_values(["fish_id", "frame"]).reset_index(drop=True)
    grouped = tracks.groupby("fish_id", sort=False)
    tracks["previous_frame"] = grouped["frame"].shift()
    tracks["frame_gap"] = tracks["frame"] - tracks["previous_frame"]
    for axis in ("X", "Y", "Z"):
        tracks[f"d{axis}_mm"] = grouped[axis].diff()

    vector_cols = ["dX_mm", "dY_mm", "dZ_mm"]
    vectors = tracks[vector_cols].to_numpy(dtype=float)
    distance = np.linalg.norm(vectors, axis=1)
    tracks["speed_mm_per_frame"] = distance / tracks["frame_gap"].to_numpy(dtype=float)
    tracks["speed_mm_per_second"] = tracks["speed_mm_per_frame"] * float(fps)

    previous_vectors = tracks.groupby("fish_id", sort=False)[vector_cols].shift().to_numpy(dtype=float)
    previous_norm = np.linalg.norm(previous_vectors, axis=1)
    current_norm = np.linalg.norm(vectors, axis=1)
    denominator = previous_norm * current_norm
    cosine = np.divide(
        np.einsum("ij,ij->i", previous_vectors, vectors),
        denominator,
        out=np.full(len(tracks), np.nan),
        where=denominator > 1e-12,
    )
    tracks["turn_angle_deg"] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    consecutive = tracks["frame_gap"].eq(1)
    previous_gap = tracks.groupby("fish_id", sort=False)["frame_gap"].shift().eq(1)
    tracks.loc[~consecutive, ["speed_mm_per_frame", "speed_mm_per_second"]] = np.nan
    tracks.loc[~(consecutive & previous_gap), "turn_angle_deg"] = np.nan
    if "is_interpolated" not in tracks:
        tracks["is_interpolated"] = False
    tracks["is_interpolated"] = tracks["is_interpolated"].fillna(False).astype(bool)
    return tracks


def _event_lines(axes, event_frames: list[int]) -> None:
    for ax in np.atleast_1d(axes):
        for frame in event_frames:
            ax.axvline(frame, color="#555555", lw=0.8, ls=(0, (3, 3)), alpha=0.55, zorder=0)


def plot_2d_displacement(
    data: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
    event_frames: list[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = list(CAMERA_TO_VIEW.values())
    fig, axes = plt.subplots(len(views), 1, figsize=(14, 10), sharex=True)
    for ax, view in zip(axes, views):
        view_data = data[data["view"].eq(view)]
        for fish_id in sorted(view_data["fish_id"].unique()):
            sub = view_data[view_data["fish_id"].eq(fish_id)].sort_values("frame")
            style = ID_STYLE.get(fish_id, {"color": "#666666", "marker": "o", "linestyle": "-"})
            continuous = sub["continuous_step"]
            continuous_values = sub["displacement_px_per_frame"].where(continuous)
            ax.plot(
                sub["frame"],
                continuous_values,
                label=fish_id,
                color=style["color"],
                marker=style["marker"],
                ls=style["linestyle"],
                lw=1.5,
                ms=4.0,
            )
            bridged = sub[~continuous & sub["displacement_px_per_frame"].notna()]
            if not bridged.empty:
                ax.scatter(
                    bridged["frame"],
                    bridged["displacement_px_per_frame"],
                    facecolors="white",
                    edgecolors=style["color"],
                    marker=style["marker"],
                    s=52,
                    linewidths=1.4,
                    zorder=4,
                )
        ax.set_ylabel("px / frame")
        ax.set_title(view.replace("_", " "), loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#D9DDE3", lw=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    _event_lines(axes, event_frames)
    axes[0].legend(ncol=3, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Source frame")
    axes[-1].set_xlim(frame_start - 0.5, frame_end + 0.5)
    axes[-1].set_xticks(np.arange(frame_start, frame_end + 1, 2))
    fig.suptitle("Selected 2D identity displacement by camera", x=0.08, y=0.985, ha="left", fontsize=16)
    fig.text(
        0.08,
        0.952,
        f"Euclidean centroid displacement divided by frame gap, frames {frame_start}–{frame_end}; "
        "hollow markers bridge missing observations",
        ha="left",
        color="#4C5663",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.925))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_3d_kinematics(
    data: pd.DataFrame,
    output_path: Path,
    *,
    frame_start: int,
    frame_end: int,
    event_frames: list[int],
    max_jump_mm: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for fish_id in sorted(data["fish_id"].unique()):
        sub = data[data["fish_id"].eq(fish_id)].sort_values("frame")
        style = ID_STYLE.get(fish_id, {"color": "#666666", "marker": "o", "linestyle": "-"})
        axes[0].plot(
            sub["frame"], sub["speed_mm_per_frame"], label=fish_id,
            color=style["color"], marker=style["marker"], ls=style["linestyle"], lw=1.5, ms=4,
        )
        axes[1].plot(
            sub["frame"], sub["turn_angle_deg"], label=fish_id,
            color=style["color"], marker=style["marker"], ls=style["linestyle"], lw=1.5, ms=4,
        )
        interpolated = sub[sub["is_interpolated"]]
        axes[0].scatter(
            interpolated["frame"], interpolated["speed_mm_per_frame"],
            facecolors="white", edgecolors=style["color"], s=55, linewidths=1.3, zorder=4,
        )
    if max_jump_mm is not None:
        axes[0].axhline(max_jump_mm, color="#30343B", lw=1.0, ls=(0, (5, 3)), label=f"gate: {max_jump_mm:g} mm/frame")
    axes[0].set_ylabel("Speed (mm / frame)")
    axes[0].set_title("3D speed", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Turn angle (degrees)")
    axes[1].set_title("Angle between consecutive 3D displacement vectors", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 185)
    axes[1].set_yticks([0, 45, 90, 135, 180])
    for ax in axes:
        ax.grid(axis="y", color="#D9DDE3", lw=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    _event_lines(axes, event_frames)
    axes[0].legend(ncol=4, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Source frame")
    axes[-1].set_xlim(frame_start - 0.5, frame_end + 0.5)
    axes[-1].set_xticks(np.arange(frame_start, frame_end + 1, 2))
    fig.suptitle("3D identity kinematics", x=0.08, y=0.985, ha="left", fontsize=16)
    fig.text(
        0.08,
        0.947,
        f"Frames {frame_start}–{frame_end}; hollow speed markers are spline-interpolated rows. "
        "Turn angle becomes unstable when speed is near zero.",
        ha="left",
        color="#4C5663",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.915))
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def summarize_window(two_d: pd.DataFrame, three_d: pd.DataFrame) -> dict[str, object]:
    max_2d: list[dict[str, object]] = []
    for (view, fish_id), sub in two_d.groupby(["view", "fish_id"], sort=True):
        valid = sub.dropna(subset=["displacement_px_per_frame"])
        if valid.empty:
            continue
        row = valid.loc[valid["displacement_px_per_frame"].idxmax()]
        max_2d.append(
            {
                "view": view,
                "fish_id": fish_id,
                "frame": int(row.frame),
                "displacement_px_per_frame": float(row.displacement_px_per_frame),
                "frame_gap": int(row.frame_gap),
                "tracklet_node": None if pd.isna(row.tracklet_node) else str(row.tracklet_node),
            }
        )

    max_3d: list[dict[str, object]] = []
    for fish_id, sub in three_d.groupby("fish_id", sort=True):
        speed = sub.dropna(subset=["speed_mm_per_frame"])
        angle = sub.dropna(subset=["turn_angle_deg"])
        max_3d.append(
            {
                "fish_id": fish_id,
                "max_speed_frame": int(speed.loc[speed["speed_mm_per_frame"].idxmax(), "frame"]),
                "max_speed_mm_per_frame": float(speed["speed_mm_per_frame"].max()),
                "max_turn_frame": int(angle.loc[angle["turn_angle_deg"].idxmax(), "frame"]),
                "max_turn_angle_deg": float(angle["turn_angle_deg"].max()),
            }
        )
    return {"max_2d_by_view_and_id": max_2d, "max_3d_by_id": max_3d}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-json", required=True, type=Path)
    parser.add_argument("--dispositions-csv", required=True, type=Path)
    parser.add_argument("--tracks-3d-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-start", required=True, type=int)
    parser.add_argument("--frame-end", required=True, type=int)
    parser.add_argument("--fps", type=float, default=166.0)
    parser.add_argument("--max-jump-mm", type=float, default=None)
    parser.add_argument("--event-frame", type=int, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.frame_end < args.frame_start:
        raise ValueError("--frame-end must be >= --frame-start")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = selected_2d_positions(args.best_json, args.dispositions_csv)
    motion = kinematics_3d(args.tracks_3d_csv, fps=args.fps)
    two_d_window = selected[selected["frame"].between(args.frame_start, args.frame_end)].copy()
    three_d_window = motion[motion["frame"].between(args.frame_start, args.frame_end)].copy()

    two_d_window.to_csv(args.output_dir / "identity_2d_displacement.csv", index=False)
    three_d_window.to_csv(args.output_dir / "identity_3d_kinematics.csv", index=False)
    plot_2d_displacement(
        two_d_window,
        args.output_dir / "identity_2d_displacement.png",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        event_frames=args.event_frame,
    )
    plot_3d_kinematics(
        three_d_window,
        args.output_dir / "identity_3d_kinematics.png",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        event_frames=args.event_frame,
        max_jump_mm=args.max_jump_mm,
    )
    summary = {
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "fps": args.fps,
        "event_frames": args.event_frame,
        **summarize_window(two_d_window, three_d_window),
    }
    (args.output_dir / "identity_window_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
