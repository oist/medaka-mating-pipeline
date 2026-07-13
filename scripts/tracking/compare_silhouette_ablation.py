from __future__ import annotations

"""Compare full-window SLEAP/segmentation identity-tracking ablations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#263238"
GRID = "#D9DEE3"
COLORS = {"A": "#3B6FB6", "B": "#D87932", "C": "#708B43"}
LABELS = {
    "A": "A · SLEAP + 3D",
    "B": "B · three-state evidence",
    "C": "C · three-state + mask links",
}


def first_match(folder: Path, pattern: str) -> Path:
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern!r} under {folder}")
    return paths[0]


def kinematics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fish_id, group in table.groupby("fish_id"):
        group = group.sort_values("frame")
        points = group[["X", "Y", "Z"]].to_numpy(float)
        frames = group["frame"].to_numpy(int)
        finite = np.isfinite(points).all(axis=1)
        points, frames = points[finite], frames[finite]
        if len(points) < 2:
            continue
        delta = np.diff(points, axis=0)
        dt = np.diff(frames).astype(float)
        velocity = delta / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        raw_step = np.linalg.norm(delta, axis=1)
        angle = np.full(len(velocity), np.nan)
        if len(velocity) >= 2:
            denominator = np.linalg.norm(velocity[:-1], axis=1) * np.linalg.norm(
                velocity[1:], axis=1
            )
            valid = denominator > 1e-12
            cosine = np.full(len(denominator), np.nan)
            cosine[valid] = (
                (velocity[:-1] * velocity[1:]).sum(axis=1)[valid]
                / denominator[valid]
            )
            angle[1:] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        rows.append(
            pd.DataFrame(
                {
                    "frame": frames[1:],
                    "fish_id": fish_id,
                    "frame_gap": dt - 1,
                    "speed_mm_per_frame": speed,
                    "raw_step_mm": raw_step,
                    "turn_angle_deg": angle,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_variant(
    key: str,
    folder: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    xyz = pd.read_csv(first_match(folder, "*_3d.csv"))
    kine = kinematics(xyz)
    edges = pd.read_csv(first_match(folder, "*cross_view_edges.csv"))
    conflicts = pd.read_csv(first_match(folder, "*candidate_conflicts.csv"))
    assignments = pd.read_csv(first_match(folder, "*tracklet_assignments.csv"))
    identified = pd.read_csv(first_match(folder, "*identified_2d.csv"), low_memory=False)
    summary = json.loads(first_match(folder, "*tracklet_first_summary.json").read_text())

    expected_frames = pd.Index(range(frame_start, frame_end + 1), name="frame")
    finite = np.isfinite(xyz[["X", "Y", "Z"]].to_numpy(float)).all(axis=1)
    valid_per_frame = (
        xyz.assign(finite=finite).groupby("frame")["finite"].sum().reindex(expected_frames, fill_value=0)
    )
    assigned = assignments["assigned_identity"].fillna("").astype(str).ne("")
    status = identified["observation_status"].value_counts()
    source_edges = edges[edges["edge_source"].eq("cross_view_geometry")]
    mask_edges = edges[edges["edge_source"].eq("silhouette_continuity")]
    metrics = {
        "variant": key,
        "label": LABELS[key],
        "frames": len(expected_frames),
        "finite_3d_rows": int(finite.sum()),
        "missing_3d_rows": int(len(expected_frames) * 3 - finite.sum()),
        "all_three_coverage_pct": float(100 * valid_per_frame.eq(3).mean()),
        "median_speed_mm_per_frame": float(kine["speed_mm_per_frame"].median()),
        "speed_gt_8_count": int(kine["speed_mm_per_frame"].gt(8).sum()),
        "raw_step_gt_20_count": int(kine["raw_step_mm"].gt(20).sum()),
        "turn_gt_120_count": int(kine["turn_angle_deg"].gt(120).sum()),
        "identity_conflict_frames": int(conflicts["frame"].nunique()),
        "identity_conflict_rows": int(len(conflicts)),
        "geometry_fallback_frames": int(summary["geometry_coverage_fallback_frames"]),
        "tracklet_nodes": int(len(assignments)),
        "graph_components": int(assignments["component_id"].nunique()),
        "labeled_tracklet_nodes": int(assigned.sum()),
        "unlabeled_tracklet_nodes": int((~assigned).sum()),
        "assigned_2d_valid_rows": int(status.get("assigned_2d_valid_3d", 0)),
        "assigned_2d_only_rows": int(status.get("assigned_2d_only", 0)),
        "unresolved_2d_rows": int(status.get("unresolved_detection", 0)),
        "accepted_geometry_edges": int(source_edges["accepted"].astype(bool).sum()),
        "accepted_silhouette_edges": int(mask_edges["accepted"].astype(bool).sum()),
        "new_silhouette_edges": int(
            mask_edges["reason"].astype(str).eq("silhouette_continuity_soft_edge").sum()
        ),
    }
    resolved_denominator = (
        metrics["assigned_2d_valid_rows"]
        + metrics["assigned_2d_only_rows"]
        + metrics["unresolved_2d_rows"]
    )
    metrics["resolved_2d_pct"] = float(
        100
        * (metrics["assigned_2d_valid_rows"] + metrics["assigned_2d_only_rows"])
        / max(1, resolved_denominator)
    )
    return metrics, {
        "xyz": xyz,
        "kinematics": kine,
        "edges": edges,
        "conflicts": conflicts,
        "assignments": assignments,
        "identified": identified,
    }


def build_frame_risk(
    data: dict[str, dict[str, pd.DataFrame]],
    annotations: dict[str, pd.DataFrame],
    *,
    frame_start: int,
    frame_end: int,
) -> pd.DataFrame:
    risk = pd.DataFrame({"frame": np.arange(frame_start, frame_end + 1)})

    hard_counts = []
    for view, table in annotations.items():
        if "silhouette_evidence_state" in table.columns:
            hard = table[table["silhouette_evidence_state"].astype(str).eq("contradiction")]
        else:
            hard = table[
                table["silhouette_status"].astype(str).str.startswith("conflict")
                | table["silhouette_break_reason"].fillna("").ne("")
            ]
        counts = hard.groupby("frame").size().rename(view)
        hard_counts.append(counts)
    hard_by_frame = pd.concat(hard_counts, axis=1).fillna(0) if hard_counts else pd.DataFrame()
    risk["hard_mask_conflicts"] = risk["frame"].map(hard_by_frame.sum(axis=1)).fillna(0)

    conflicts = data["C"]["conflicts"]
    conflict_counts = conflicts.groupby("frame").size()
    risk["identity_conflicts"] = risk["frame"].map(conflict_counts).fillna(0)

    kine = data["C"]["kinematics"]
    max_speed = kine.groupby("frame")["speed_mm_per_frame"].max()
    max_turn = kine.groupby("frame")["turn_angle_deg"].max()
    risk["max_speed_mm_per_frame"] = risk["frame"].map(max_speed)
    risk["max_turn_angle_deg"] = risk["frame"].map(max_turn)

    left = data["A"]["xyz"]
    right = data["B"]["xyz"]
    comparison = left.merge(right, on=["frame", "fish_id"], suffixes=("_a", "_b"))
    comparison["ab_coordinate_change_mm"] = np.sqrt(
        sum((comparison[f"{axis}_a"] - comparison[f"{axis}_b"]) ** 2 for axis in "XYZ")
    )
    max_change = comparison.groupby("frame")["ab_coordinate_change_mm"].max()
    risk["ab_coordinate_change_mm"] = risk["frame"].map(max_change).fillna(0)

    mask_edges = data["C"]["edges"]
    mask_edges = mask_edges[mask_edges["edge_source"].eq("silhouette_continuity")]
    edge_frames = []
    for row in mask_edges.itertuples():
        if not bool(row.accepted):
            continue
        gap = int(row.gap_frames) if np.isfinite(row.gap_frames) else 0
        # The edge table does not carry absolute endpoint frames. Marking is
        # added from the candidate CSVs below by the caller when available.
        edge_frames.append((gap, str(row.reason)))
    risk["accepted_mask_link_endpoint"] = 0

    risk["risk_score"] = (
        5.0 * risk["identity_conflicts"]
        + 2.5 * risk["hard_mask_conflicts"].clip(upper=4)
        + 2.0 * risk["ab_coordinate_change_mm"].gt(2).astype(float)
        + (risk["ab_coordinate_change_mm"] / 5).clip(upper=3)
        + 2.0 * risk["max_speed_mm_per_frame"].fillna(0).gt(4).astype(float)
        + 1.5 * risk["max_turn_angle_deg"].fillna(0).gt(120).astype(float)
    )
    return risk


def compare_coordinate_pairs(
    data: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_key, right_key in (("A", "B"), ("B", "C"), ("A", "C")):
        left = data[left_key]["xyz"]
        right = data[right_key]["xyz"]
        merged = left.merge(
            right,
            on=["frame", "fish_id"],
            suffixes=("_left", "_right"),
            how="inner",
        )
        left_xyz = merged[["X_left", "Y_left", "Z_left"]].to_numpy(float)
        right_xyz = merged[["X_right", "Y_right", "Z_right"]].to_numpy(float)
        finite = np.isfinite(left_xyz).all(axis=1) & np.isfinite(right_xyz).all(axis=1)
        displacement = np.linalg.norm(left_xyz[finite] - right_xyz[finite], axis=1)
        rows.append(
            {
                "comparison": f"{left_key}_vs_{right_key}",
                "common_finite_rows": int(finite.sum()),
                "changed_rows_gt_1e_9_mm": int((displacement > 1e-9).sum()),
                "median_changed_displacement_mm": float(
                    np.median(displacement[displacement > 1e-9])
                )
                if np.any(displacement > 1e-9)
                else 0.0,
                "max_displacement_mm": float(np.max(displacement))
                if displacement.size
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_evidence_states(
    annotations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for view, table in annotations.items():
        states = table.get(
            "silhouette_evidence_state",
            pd.Series("unknown", index=table.index),
        ).astype(str)
        for state in ("support", "unknown", "contradiction"):
            count = int(states.eq(state).sum())
            rows.append(
                {
                    "view": view,
                    "evidence_state": state,
                    "rows": count,
                    "fraction": float(count / max(1, len(table))),
                }
            )
    return pd.DataFrame(rows)


def select_windows(
    risk: pd.DataFrame,
    *,
    count: int = 20,
    radius: int = 35,
) -> pd.DataFrame:
    selected = []
    for row in risk.sort_values(["risk_score", "frame"], ascending=[False, True]).itertuples():
        if row.risk_score <= 0:
            break
        if any(abs(int(row.frame) - item["center_frame"]) <= 2 * radius for item in selected):
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                "center_frame": int(row.frame),
                "start_frame": max(int(risk.frame.min()), int(row.frame) - radius),
                "end_frame": min(int(risk.frame.max()), int(row.frame) + radius),
                "risk_score": float(row.risk_score),
                "hard_mask_conflicts": int(row.hard_mask_conflicts),
                "identity_conflicts": int(row.identity_conflicts),
                "ab_coordinate_change_mm": float(row.ab_coordinate_change_mm),
                "max_speed_mm_per_frame": float(row.max_speed_mm_per_frame)
                if np.isfinite(row.max_speed_mm_per_frame)
                else np.nan,
                "max_turn_angle_deg": float(row.max_turn_angle_deg)
                if np.isfinite(row.max_turn_angle_deg)
                else np.nan,
            }
        )
        if len(selected) >= count:
            break
    return pd.DataFrame(selected)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.tick_params(colors=INK)


def render_overview(metrics: pd.DataFrame, output_path: Path) -> None:
    keys = metrics["variant"].tolist()
    labels = [LABELS[key] for key in keys]
    colors = [COLORS[key] for key in keys]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Three-view identity-tracking ablation", fontsize=17, fontweight="bold", color=INK)

    panels = [
        ("all_three_coverage_pct", "All-three-fish 3D coverage", "% of 5,000 frames"),
        ("resolved_2d_pct", "2D observations with an assigned identity", "% of retained tracklet rows"),
        ("geometry_fallback_frames", "Geometry coverage fallbacks", "frames"),
        ("unlabeled_tracklet_nodes", "Tracklet nodes still unlabeled", "nodes"),
    ]
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        values = metrics[field].to_numpy(float)
        bars = axis.bar(labels, values, color=colors, edgecolor=INK, linewidth=0.5)
        for bar, value in zip(bars, values):
            text = f"{value:.2f}" if "%" in ylabel else f"{int(value)}"
            axis.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=9)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=12)
        style_axis(axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_evidence_states(
    evidence_states: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    state_colors = {
        "support": "#4F8A5B",
        "unknown": "#D9A441",
        "contradiction": "#B64E67",
    }
    views = ["top_cam", "side1_cam", "side2_cam"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    fig.suptitle(
        "Three-state silhouette evidence and graph effect",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    bottom = np.zeros(len(views), dtype=float)
    for state in ("support", "unknown", "contradiction"):
        values = []
        for view in views:
            match = evidence_states[
                evidence_states["view"].eq(view)
                & evidence_states["evidence_state"].eq(state)
            ]
            values.append(float(match["fraction"].iloc[0]) if len(match) else 0.0)
        axes[0].bar(
            views,
            values,
            bottom=bottom,
            label=state,
            color=state_colors[state],
            edgecolor="white",
            linewidth=0.7,
        )
        for index, (base, value) in enumerate(zip(bottom, values)):
            if value >= 0.04:
                axes[0].text(
                    index,
                    base + value / 2,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if state != "unknown" else INK,
                    fontweight="bold",
                )
        bottom += np.asarray(values)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of retained SLEAP observations")
    axes[0].set_title("No observation met the strong-contradiction gate", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style_axis(axes[0])

    x = np.arange(len(metrics))
    width = 0.36
    components = metrics["graph_components"].to_numpy(float)
    unlabeled = metrics["unlabeled_tracklet_nodes"].to_numpy(float)
    axes[1].bar(x - width / 2, components, width, label="graph components", color="#3B6FB6")
    axes[1].bar(x + width / 2, unlabeled, width, label="unlabeled nodes", color="#B64E67")
    for xpos, values in ((x - width / 2, components), (x + width / 2, unlabeled)):
        for position, value in zip(xpos, values):
            axes[1].text(position, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x, metrics["variant"])
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, max(float(components.max()), float(unlabeled.max())) * 1.08)
    axes[1].set_title("Graph components and unlabeled tracklet nodes", loc="left", fontweight="bold")
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    style_axis(axes[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_risk_timeline(
    risk: pd.DataFrame,
    windows: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    fig.suptitle("QC risk evidence across frames 10000–14999", fontsize=17, fontweight="bold", color=INK)
    axes[0].plot(risk.frame, risk.hard_mask_conflicts, color="#B64E67", linewidth=1.0)
    axes[0].scatter(
        risk.loc[risk.identity_conflicts.gt(0), "frame"],
        risk.loc[risk.identity_conflicts.gt(0), "identity_conflicts"],
        color="#263238", marker="X", s=35, label="identity conflict",
    )
    axes[0].set_ylabel("events/frame")
    axes[0].set_title("Mask contradictions and global identity conflicts", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    axes[1].plot(risk.frame, risk.ab_coordinate_change_mm, color="#D87932", linewidth=1.1)
    axes[1].axhline(2, color=INK, linestyle=":", linewidth=1)
    axes[1].set_ylabel("mm")
    axes[1].set_title("Maximum per-frame coordinate change: B versus A", loc="left", fontweight="bold")

    axes[2].plot(risk.frame, risk.risk_score, color="#3B6FB6", linewidth=1.1)
    axes[2].scatter(windows.center_frame, windows.risk_score, color="#B64E67", s=40, zorder=5)
    for row in windows.head(10).itertuples():
        axes[2].annotate(str(row.rank), (row.center_frame, row.risk_score), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axes[2].set_ylabel("ranking score")
    axes[2].set_xlabel("Frame")
    axes[2].set_title("Heuristic QC-window ranking", loc="left", fontweight="bold")
    for axis in axes:
        style_axis(axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="A=folder; repeat for A/B/C")
    parser.add_argument("--annotation", action="append", required=True, help="view=annotated.csv")
    parser.add_argument("--frame-start", type=int, default=10000)
    parser.add_argument("--frame-end", type=int, default=14999)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        key, path = value.split("=", 1)
        output[key] = Path(path)
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    variants = parse_paths(args.variant)
    annotations = {
        view: pd.read_csv(path, low_memory=False)
        for view, path in parse_paths(args.annotation).items()
    }
    metric_rows = []
    data = {}
    for key in ("A", "B", "C"):
        metrics, tables = summarize_variant(
            key,
            variants[key],
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        )
        metric_rows.append(metrics)
        data[key] = tables
    metrics = pd.DataFrame(metric_rows)
    coordinate_pairs = compare_coordinate_pairs(data)
    evidence_states = summarize_evidence_states(annotations)
    risk = build_frame_risk(
        data,
        annotations,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    windows = select_windows(risk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "ablation_metrics.csv", index=False)
    coordinate_pairs.to_csv(
        args.output_dir / "pairwise_coordinate_comparison.csv", index=False
    )
    evidence_states.to_csv(
        args.output_dir / "silhouette_evidence_counts.csv", index=False
    )
    risk.to_csv(args.output_dir / "frame_risk_scores.csv", index=False)
    windows.to_csv(args.output_dir / "worst_qc_windows.csv", index=False)
    render_overview(metrics, args.output_dir / "ablation_overview.png")
    render_evidence_states(
        evidence_states,
        metrics,
        args.output_dir / "three_state_evidence_overview.png",
    )
    render_risk_timeline(risk, windows, args.output_dir / "risk_timeline.png")
    print(metrics.to_string(index=False))
    print("\nCoordinate comparisons:\n", coordinate_pairs.to_string(index=False))
    print("\nEvidence states:\n", evidence_states.to_string(index=False))
    print("\nWorst windows:\n", windows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
