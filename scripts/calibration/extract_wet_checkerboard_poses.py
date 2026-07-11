"""Extract synchronized, pose-diverse checkerboard observations in water.

The wet checkerboard is used to constrain refractive camera geometry after dry
intrinsics have been fixed.  Discovery is performed at reduced resolution,
while saved coordinates are subpixel-refined at the native resolution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAMERAS = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


@dataclass
class WetPose:
    frame: int
    corners: dict[str, np.ndarray]
    sharpness: dict[str, float]
    descriptor: np.ndarray

    @property
    def n_views(self) -> int:
        return len(self.corners)


def detect_board(
    gray: np.ndarray, pattern: tuple[int, int], scale: float
) -> np.ndarray | None:
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ok or corners is None:
        return None
    corners = corners.astype(np.float32) / float(scale)
    cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4),
    )
    return corners.reshape(-1, 2)


def view_descriptor(
    corners: np.ndarray, image_size: tuple[int, int], pattern_cols: int
) -> np.ndarray:
    width, height = image_size
    center = corners.mean(axis=0) / np.asarray([width, height], dtype=float)
    rect = cv2.minAreaRect(corners.astype(np.float32))
    area = max(rect[1][0] * rect[1][1] / float(width * height), 1e-7)
    axis = corners[pattern_cols - 1] - corners[0]
    angle = np.arctan2(axis[1], axis[0])
    return np.array([center[0], center[1], np.log(area), np.sin(angle), np.cos(angle)])


def pose_descriptor(
    corners_by_view: dict[str, np.ndarray], image_size: tuple[int, int], pattern_cols: int
) -> np.ndarray:
    parts: list[float] = []
    for camera in CAMERAS:
        corners = corners_by_view.get(camera)
        if corners is None:
            parts.extend([0.0] * 5)
        else:
            parts.extend(view_descriptor(corners, image_size, pattern_cols).tolist())
    parts.extend(float(camera in corners_by_view) for camera in CAMERAS)
    return np.asarray(parts, dtype=float)


def select_pose_diverse(
    candidates: Sequence[WetPose], maximum: int, minimum_frame_gap: int
) -> list[WetPose]:
    if not candidates:
        return []
    x = np.stack([pose.descriptor for pose in candidates])
    med = np.median(x, axis=0)
    scale = np.median(np.abs(x - med), axis=0)
    scale[scale < 1e-4] = 1.0
    z = (x - med) / scale
    quality = np.asarray(
        [pose.n_views + 0.02 * np.log1p(sum(pose.sharpness.values())) for pose in candidates]
    )
    chosen = [int(np.argmax(quality))]
    nearest = np.linalg.norm(z - z[chosen[0]], axis=1)
    while len(chosen) < min(maximum, len(candidates)):
        score = nearest + 0.35 * (np.asarray([p.n_views for p in candidates]) - 2.0)
        for index, pose in enumerate(candidates):
            if index in chosen or any(
                abs(pose.frame - candidates[old].frame) < minimum_frame_gap for old in chosen
            ):
                score[index] = -np.inf
        candidate = int(np.argmax(score))
        if not np.isfinite(score[candidate]):
            break
        chosen.append(candidate)
        nearest = np.minimum(nearest, np.linalg.norm(z - z[candidate], axis=1))
    return sorted((candidates[index] for index in chosen), key=lambda pose: pose.frame)


def _plot_qc(
    candidates: Sequence[WetPose],
    selected: Sequence[WetPose],
    image_size: tuple[int, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), layout="constrained")
    selected_frames = {pose.frame for pose in selected}
    for axis, camera in zip(axes[:3], CAMERAS, strict=True):
        visible = [(pose, pose.corners[camera]) for pose in candidates if camera in pose.corners]
        axis.scatter(
            [corners[:, 0].mean() for _, corners in visible],
            [corners[:, 1].mean() for _, corners in visible],
            s=9,
            color="#2563A6",
            alpha=0.18,
            label=f"candidate (n={len(visible)})",
        )
        kept = [(pose, corners) for pose, corners in visible if pose.frame in selected_frames]
        axis.scatter(
            [corners[:, 0].mean() for _, corners in kept],
            [corners[:, 1].mean() for _, corners in kept],
            s=28,
            facecolors="none",
            edgecolors="#1E4F7A",
            linewidths=1.0,
            label=f"selected (n={len(kept)})",
        )
        axis.set(
            xlim=(0, image_size[0]),
            ylim=(image_size[1], 0),
            xlabel="image x (px)",
            ylabel="image y (px)",
            title=camera,
        )
        axis.grid(color="#D9D9D9", linewidth=0.6)
        axis.legend(frameon=False, fontsize=7)
    labels = ["2 views", "3 views"]
    candidate_counts = [sum(p.n_views == n for p in candidates) for n in (2, 3)]
    selected_counts = [sum(p.n_views == n for p in selected) for n in (2, 3)]
    x = np.arange(2)
    axes[3].bar(x - 0.18, candidate_counts, 0.36, color="#BBD2E8", edgecolor="#1E4F7A", label="candidate")
    axes[3].bar(x + 0.18, selected_counts, 0.36, color="#2563A6", edgecolor="#1E4F7A", label="selected")
    axes[3].set_xticks(x, labels)
    axes[3].set(ylabel="synchronized poses", title="Visibility")
    axes[3].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("Wet 10 mm checkerboard: synchronized pose coverage", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern-cols", type=int, default=7)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=10.0)
    parser.add_argument("--start-frame", type=int, default=14000)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--discovery-scale", type=float, default=0.5)
    parser.add_argument("--min-views", type=int, choices=(2, 3), default=2)
    parser.add_argument("--max-selected-poses", type=int, default=80)
    parser.add_argument("--minimum-frame-gap", type=int, default=60)
    parser.add_argument("--pose-prefix", default="wet_cb")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    caps: dict[str, cv2.VideoCapture] = {}
    frame_counts: dict[str, int] = {}
    image_size: tuple[int, int] | None = None
    for camera, filename in CAMERAS.items():
        camera_dir = args.video_root / camera
        cap = cv2.VideoCapture(str(camera_dir / filename))
        if not cap.isOpened():
            raise FileNotFoundError(camera_dir / filename)
        caps[camera] = cap
        frame_counts[camera] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if image_size is not None and image_size != size:
            raise ValueError(f"Image-size mismatch: {image_size} vs {size}")
        image_size = size
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    assert image_size is not None
    end_frame = min(frame_counts.values()) if args.end_frame is None else min(args.end_frame, min(frame_counts.values()))
    pattern = (args.pattern_cols, args.pattern_rows)
    candidates: list[WetPose] = []
    scan_rows: list[dict[str, object]] = []
    for frame in range(args.start_frame, end_frame):
        retrieved: dict[str, np.ndarray] = {}
        failed = False
        for camera, cap in caps.items():
            if not cap.grab():
                failed = True
                break
        if failed:
            break
        if (frame - args.start_frame) % args.stride:
            continue
        for camera, cap in caps.items():
            ok, bgr = cap.retrieve()
            if ok:
                retrieved[camera] = bgr
        corners_by_view: dict[str, np.ndarray] = {}
        sharpness: dict[str, float] = {}
        for camera, bgr in retrieved.items():
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            corners = detect_board(gray, pattern, args.discovery_scale)
            if corners is not None:
                corners_by_view[camera] = corners
                sharpness[camera] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scan_rows.append(
            {
                "frame": frame,
                "n_views": len(corners_by_view),
                **{f"visible_{camera}": camera in corners_by_view for camera in CAMERAS},
            }
        )
        if len(corners_by_view) >= args.min_views:
            candidates.append(
                WetPose(
                    frame=frame,
                    corners=corners_by_view,
                    sharpness=sharpness,
                    descriptor=pose_descriptor(corners_by_view, image_size, args.pattern_cols),
                )
            )
        if frame % 3000 < args.stride:
            print(f"frame {frame}/{end_frame}: synchronized candidates={len(candidates)}", flush=True)
    for cap in caps.values():
        cap.release()
    selected = select_pose_diverse(candidates, args.max_selected_poses, args.minimum_frame_gap)
    selected_frames = {pose.frame for pose in selected}
    corner_rows: list[dict[str, object]] = []
    for pose in candidates:
        if pose.frame not in selected_frames:
            continue
        for camera, corners in pose.corners.items():
            for corner_id, xy in enumerate(corners):
                corner_rows.append(
                    {
                        "pose_id": f"{args.pose_prefix}_f{pose.frame}",
                        "frame": pose.frame,
                        "camera": camera,
                        "corner_id": corner_id,
                        "board_x_mm": float((corner_id % args.pattern_cols) * args.square_mm),
                        "board_y_mm": float((corner_id // args.pattern_cols) * args.square_mm),
                        "board_z_mm": 0.0,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                        "n_views": pose.n_views,
                    }
                )
    corners_path = args.output_dir / "wet_checkerboard_selected_corners.csv"
    scan_path = args.output_dir / "wet_checkerboard_scan.csv"
    pd.DataFrame(corner_rows).to_csv(corners_path, index=False)
    pd.DataFrame(scan_rows).to_csv(scan_path, index=False)
    qc_path = args.output_dir / "wet_checkerboard_pose_coverage.png"
    _plot_qc(candidates, selected, image_size, qc_path)
    n_all_three = sum(pose.n_views == 3 for pose in selected)
    n_two = sum(pose.n_views == 2 for pose in selected)
    gate_pass = n_all_three >= 15 and n_two + n_all_three >= 30
    report = {
        "video_root": str(args.video_root),
        "image_size": list(image_size),
        "pattern": {"inner_corners": list(pattern), "square_mm": args.square_mm},
        "scan": {
            "start_frame": args.start_frame,
            "end_frame": end_frame,
            "stride": args.stride,
            "candidate_poses": len(candidates),
        },
        "selected": {
            "poses": len(selected),
            "two_view_poses": n_two,
            "three_view_poses": n_all_three,
            "minimum_frame_gap": args.minimum_frame_gap,
        },
        "synchronization": {
            "source": "shared external frame trigger",
            "join_key": "frame index",
            "camera_internal_timestamps_used_cross_camera": False,
        },
        "physics_fit_gate": {
            "required_three_view_pose_clusters": 15,
            "required_at_least_two_view_pose_clusters": 30,
            "pass": gate_pass,
        },
        "outputs": {"corners_csv": str(corners_path), "scan_csv": str(scan_path), "qc_figure": str(qc_path)},
    }
    report_path = args.output_dir / "wet_checkerboard_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Selected {len(selected)} poses ({n_all_three} three-view, {n_two} two-view); "
        f"fit gate={'PASS' if gate_pass else 'FAIL'}"
    )
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
