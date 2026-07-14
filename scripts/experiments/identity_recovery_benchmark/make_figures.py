from __future__ import annotations

"""Create neuroscience-readable figures from a completed common benchmark."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "idtracker_ai_v6": "idtracker.ai v6",
    "native_udmt": "Native UDMT",
    "multiview_constrained_contrastive": "Multi-view contrastive",
}
METHOD_COLORS = {
    "idtracker_ai_v6": "#2979B9",
    "native_udmt": "#D95F02",
    "multiview_constrained_contrastive": "#1B9E77",
}
TRACK_COLORS = ("#00C853", "#2979FF", "#E53935", "#8E24AA")


def _display_id(value) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def study_design_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.25, 1.5, 2.2, 2.0, "Historical recording\n3 synchronized cameras", "#E8F1FA"),
        (3.0, 0.35, 2.7, 1.1, "idtracker.ai v6\ntop-view appearance", "#E8F1FA"),
        (3.0, 1.95, 2.7, 1.1, "Native UDMT\ntop-view propagation", "#FCECDF"),
        (3.0, 3.55, 2.7, 1.1, "Our method\nmulti-view appearance\n+ one-to-one constraints", "#E4F5EC"),
        (6.45, 1.5, 2.4, 2.0, "Same 7 close approaches\nsame frames for every method", "#F5F5F5"),
        (9.65, 1.5, 2.9, 2.0, "Blinded human audit\nDid each label stay on the\nsame biological fish?", "#FFF4CC"),
    ]
    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color, edgecolor="#334155", linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
    for y in (0.9, 2.5, 4.1):
        ax.add_patch(FancyArrowPatch((2.45, 2.5), (3.0, y), arrowstyle="-|>", mutation_scale=14, color="#64748B"))
        ax.add_patch(FancyArrowPatch((5.7, y), (6.45, 2.5), arrowstyle="-|>", mutation_scale=14, color="#64748B"))
    ax.add_patch(FancyArrowPatch((8.85, 2.5), (9.65, 2.5), arrowstyle="-|>", mutation_scale=14, color="#64748B"))
    ax.set_title("Can biological identity be recovered from the recording itself?", loc="left", fontsize=17, weight="bold")
    ax.text(0.25, 0.08, "Primary outcome: independent visual continuity, not agreement with provisional geometry labels.", fontsize=10.5, color="#475569")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def localization_figure(summary: dict, output: Path) -> None:
    methods = summary["methods"]
    gates = sorted(
        {float(gate) for report in methods.values() for gate in report["localization_identity_blind"]}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for method, report in methods.items():
        scores = report["localization_identity_blind"]
        recall = [scores[str(int(g)) if g.is_integer() else str(g)]["position_recall"] for g in gates]
        coverage = [scores[str(int(g)) if g.is_integer() else str(g)]["all_fish_frame_coverage"] for g in gates]
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method)
        axes[0].plot(gates, recall, marker="o", linewidth=2.5, label=label, color=color)
        axes[1].plot(gates, coverage, marker="o", linewidth=2.5, label=label, color=color)
    axes[0].set_title("Are emitted points actually on fish?")
    axes[0].set_ylabel("Fraction of expected positions")
    axes[0].set_xlabel("Matching tolerance (pixels)")
    axes[1].set_title("Are all three fish found together?")
    axes[1].set_xlabel("Matching tolerance (pixels)")
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Localization is scored without using fish identity labels", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "A method that maintains three labels while drifting off the animals receives no credit.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _read_frame(video: Path, absolute_frame: int, video_frame_zero: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(absolute_frame - video_frame_zero))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {absolute_frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def example_panel(
    *,
    video: Path,
    video_frame_zero: int,
    events: pd.DataFrame,
    method_tracks: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    event = events.sort_values("min_distance_px").iloc[0]
    frames = [int(event.start_frame) - 40, int(event.center_frame), int(event.end_frame) + 40]
    images = [_read_frame(video, frame, video_frame_zero) for frame in frames]
    methods = list(method_tracks)
    fig, axes = plt.subplots(len(methods), 3, figsize=(12, 4 * len(methods)), squeeze=False)
    for row_index, method in enumerate(methods):
        table = method_tracks[method]
        ids = sorted({_display_id(value) for value in table["track_id"].unique()})
        colors = {track_id: TRACK_COLORS[i % len(TRACK_COLORS)] for i, track_id in enumerate(ids)}
        labels = {track_id: f"T{i + 1}" for i, track_id in enumerate(ids)}
        for column_index, (frame, image) in enumerate(zip(frames, images)):
            ax = axes[row_index, column_index]
            ax.imshow(image)
            rows = table[table["frame"] == frame]
            for _, point in rows.iterrows():
                track_id = _display_id(point.track_id)
                ax.scatter([point.x], [point.y], s=240, facecolors="none", edgecolors=colors[track_id], linewidths=3)
                ax.text(point.x + 18, point.y - 18, labels[track_id], color="white", fontsize=11, weight="bold", bbox={"facecolor": colors[track_id], "alpha": 0.85, "pad": 2, "edgecolor": "none"})
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(("Before" if column_index == 0 else "Closest approach" if column_index == 1 else "After") + f"\nframe {frame}", fontsize=12)
        axes[row_index, 0].text(-0.08, 0.5, METHOD_LABELS.get(method, method), transform=axes[row_index, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12, weight="bold")
    fig.suptitle(f"Example identity challenge: {event.event_id} ({event.min_distance_px:.1f} px minimum separation)", fontsize=16, weight="bold")
    fig.text(0.5, 0.01, "Colored T-labels are anonymous algorithm outputs. The audit asks whether each stays on the same fish.", ha="center", color="#475569")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def idtracker_learning_figure(curve_path: Path, summary_path: Path, output: Path) -> None:
    curve = pd.read_csv(curve_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["batch"], curve["silhouette"], color="#2979B9", linewidth=1.7, alpha=0.9)
    best_rows = curve[curve["new_best"].astype(str).str.lower().isin({"true", "1"})]
    ax.scatter(best_rows["batch"], best_rows["silhouette"], color="#2979B9", s=20, zorder=3)
    target = float(summary["target_silhouette"])
    ax.axhline(target, color="#D95F02", linestyle="--", linewidth=2, label=f"idtracker.ai target ({target:.2f})")
    recovered = float(summary["best_silhouette"])
    recovery_x = float(curve["batch"].max()) + 250
    ax.scatter([recovery_x], [recovered], marker="*", s=220, color="#1B9E77", zorder=5, label=f"saved-checkpoint fallback ({recovered:.3f})")
    ax.annotate("final saved model", (recovery_x, recovered), xytext=(-100, 24), textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#475569"}, fontsize=10)
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel("Contrastive training batches")
    ax.set_ylabel("Silhouette separation")
    ax.set_title("idtracker.ai learned appearance structure, but did not reach its target", fontsize=14, weight="bold")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.5, 0.01, "This internal training score is not biological identity accuracy; crossing videos provide the independent test.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-frame-zero", type=int, default=0)
    parser.add_argument("--idtracker-curve", type=Path)
    parser.add_argument("--idtracker-summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(args.benchmark_dir / "benchmark_summary.json")
    study_design_figure(args.output_dir / "FIGURE_1_STUDY_DESIGN.png")
    localization_figure(summary, args.output_dir / "FIGURE_2_LOCALIZATION.png")
    if args.video:
        events = pd.read_csv(args.benchmark_dir / "crossing_events.csv")
        tracks = {}
        for method in summary["methods"]:
            path = args.benchmark_dir / f"{method}.tracks.csv"
            if path.exists():
                tracks[method] = pd.read_csv(path)
        example_panel(
            video=args.video,
            video_frame_zero=args.video_frame_zero,
            events=events,
            method_tracks=tracks,
            output=args.output_dir / "FIGURE_3_EXAMPLE_CROSSING.png",
        )
    if args.idtracker_curve and args.idtracker_summary:
        idtracker_learning_figure(
            args.idtracker_curve,
            args.idtracker_summary,
            args.output_dir / "FIGURE_4_IDTRACKER_TRAINING.png",
        )
    print(f"figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
