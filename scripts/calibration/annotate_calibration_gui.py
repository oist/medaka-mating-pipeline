"""
Interactive calibration annotation GUI with draft-point preload and bbox-based
center refinement.

This tool is designed for the colored bead calibration rigs used in this repo.
It can:

- load template points for one way
- preload draft `*_image_coords.txt` files
- annotate with exact click placement
- refine a point from a dragged bounding box
- save the standard `index x y X Y Z label` text format
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

plt.rcParams["keymap.save"] = []

try:
    from .lib.calibration_core import run_calibration_pipeline
    from .lib.visualize_coords_from_template import get_point_color, parse_template
    from .remap_bead_labels import remap_single_camera_annotation
    from .qc_reprojection_2d import load_mapping_csv as load_reprojection_mapping_csv
    from .qc_reprojection_2d import plot_reprojection_2d_views, reprojection_errors_per_camera
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.calibration_core import run_calibration_pipeline  # type: ignore
    from lib.visualize_coords_from_template import get_point_color, parse_template  # type: ignore
    from remap_bead_labels import remap_single_camera_annotation  # type: ignore
    from qc_reprojection_2d import load_mapping_csv as load_reprojection_mapping_csv  # type: ignore
    from qc_reprojection_2d import plot_reprojection_2d_views, reprojection_errors_per_camera  # type: ignore


COLOR_SPECS = {
    "pink": [((145, 60, 80), (179, 255, 255))],
    "purple": [((120, 50, 40), (150, 255, 255))],
    "yellow": [((18, 60, 80), (40, 255, 255))],
    "orange": [((5, 60, 70), (22, 255, 255))],
    "blue": [((90, 60, 50), (130, 255, 255))],
}
DISPLAY_COLORS = {
    "pink": "magenta",
    "purple": "purple",
    "yellow": "yellow",
    "orange": "orange",
    "blue": "deepskyblue",
    "black": "black",
    "silver": "silver",
    "corner": "lime",
    "pillar_1": "crimson",
    "pillar_2": "darkorange",
    "pillar_3": "deepskyblue",
    "pillar_4": "mediumorchid",
}
CAM_PROJECTION = {
    "cam0": ("X", "Y"),
    "cam1": ("X", "Z"),
    "cam2": ("Y", "Z"),
}


@dataclass
class PointSpec:
    index: int
    label: str
    xyz: Optional[np.ndarray]
    kind: str
    color: str


def _color_display_name(color: str) -> str:
    return color.replace("_", " ").title()


def _load_review_targets(review_csv: Path | None) -> Dict[str, dict]:
    """Load `{label: metadata}` from a worst-beads review CSV."""
    if review_csv is None or not review_csv.exists():
        return {}
    df = pd.read_csv(review_csv)
    if "label" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        meta = {}
        if "err_mm" in row and pd.notna(row["err_mm"]):
            meta["err_mm"] = float(row["err_mm"])
        out[label] = meta
    return out


def _find_template_file(root: Path, way_name: str) -> Path:
    for candidate in [
        root / f"cali_coords_template_{way_name}.txt",
        root / f"Calibration_Coords_Template_{way_name.upper()}.txt",
    ]:
        if candidate.exists():
            return candidate
    matches = sorted(root.glob(f"*template*{way_name}*.txt"))
    if not matches:
        raise FileNotFoundError(f"Template file not found for {way_name} in {root}")
    return matches[0]


def _find_way_images(root: Path, way_name: str) -> Dict[str, Path]:
    way_dir_candidates = sorted(root.glob(f"*{way_name}*_pic"))
    if not way_dir_candidates:
        raise FileNotFoundError(f"Could not find raw image folder for {way_name} under {root}")
    way_dir = way_dir_candidates[0]
    valid_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    out: Dict[str, Path] = {}
    for path in sorted(way_dir.iterdir()):
        if path.suffix.lower() not in valid_suffixes:
            continue
        name = path.name.lower()
        if "cam0" in name or "top" in name:
            out["cam0"] = path
        elif "cam1" in name or "side1" in name:
            out["cam1"] = path
        elif "cam2" in name or "side2" in name:
            out["cam2"] = path
    missing = [cam for cam in ("cam0", "cam1", "cam2") if cam not in out]
    if missing:
        raise FileNotFoundError(f"Missing image(s) for {missing} in {way_dir}")
    return out


def _infer_cam_name(image_path: Path) -> str:
    name = image_path.name.lower()
    if "cam0" in name or "top" in name:
        return "cam0"
    if "cam1" in name or "side1" in name:
        return "cam1"
    if "cam2" in name or "side2" in name:
        return "cam2"
    raise ValueError(f"Could not infer camera name from {image_path}")


def _load_point_specs(template_path: Path, way_name: str) -> List[PointSpec]:
    sections = parse_template(template_path)
    beads_key = f"BEADS_{way_name.upper()}"
    if beads_key not in sections:
        raise ValueError(f"{template_path} does not contain section {beads_key}")

    specs: List[PointSpec] = []
    if "CORNERS" in sections:
        coords, labels = sections["CORNERS"]
        for i, (coord, label) in enumerate(zip(coords, labels)):
            specs.append(
                PointSpec(
                    index=i,
                    label=f"CORNER: {label}",
                    xyz=np.asarray(coord, dtype=float),
                    kind="corner",
                    color="corner",
                )
            )

    bead_coords, bead_labels = sections[beads_key]
    for i, coord in enumerate(bead_coords):
        bead_name = bead_labels[i] if i < len(bead_labels) and bead_labels[i] else f"bead_{i + 1:03d}"
        specs.append(
            PointSpec(
                index=4 + i,
                label=f"BEAD: {bead_name}",
                xyz=np.asarray(coord, dtype=float),
                kind="bead",
                color=get_point_color(4 + i),
            )
        )
    return specs


def _append_bottom_corners(specs: List[PointSpec]) -> List[PointSpec]:
    top_corners = [spec for spec in specs if spec.kind == "corner"][:4]
    if len(top_corners) != 4:
        return list(specs)

    next_index = max(spec.index for spec in specs) + 1
    extended = list(specs)
    for i, spec in enumerate(top_corners):
        bottom_xyz = np.asarray([spec.xyz[0], spec.xyz[1], 4.0], dtype=float)
        extended.append(
            PointSpec(
                index=next_index + i,
                label=f"CORNER_BOTTOM: corner_{i + 1}_bottom",
                xyz=bottom_xyz,
                kind="corner",
                color="corner",
            )
        )
    return extended


def _annotation_sort_key(spec: PointSpec) -> tuple[int, int]:
    """
    Order GUI annotation tasks to prioritize colored beads first, then the top
    corners, then any derived bottom-corner points.
    """
    if spec.kind == "bead":
        return (0, spec.index)
    if spec.label.startswith("CORNER_BOTTOM:"):
        return (2, spec.index)
    return (1, spec.index)


def _build_auxiliary_specs(aux_preset: str) -> List[PointSpec]:
    if aux_preset != "pillar_lines":
        raise ValueError(f"Unsupported auxiliary preset: {aux_preset}")

    specs: List[PointSpec] = []
    for pillar_idx in range(1, 5):
        for line_idx in range(1, 8):
            specs.append(
                PointSpec(
                    index=len(specs),
                    label=f"AUX: pillar_{pillar_idx}_line_{line_idx:02d}_inner",
                    xyz=None,
                    kind="aux",
                    color=f"pillar_{pillar_idx}",
                )
            )
    return specs


def _load_annotation_points(path: Path) -> Dict[int, np.ndarray]:
    points: Dict[int, np.ndarray] = {}
    if not path.exists():
        return points
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            points[idx] = np.array([float(parts[1]), float(parts[2])], dtype=float)
    return points


def _fit_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if len(src_pts) < 3 or len(dst_pts) < 3:
        raise ValueError("At least three point pairs are required to fit an affine transform")
    A = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=float)])
    tx, *_ = np.linalg.lstsq(A, dst_pts[:, 0], rcond=None)
    ty, *_ = np.linalg.lstsq(A, dst_pts[:, 1], rcond=None)
    return np.vstack([tx, ty])


def _apply_affine(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    A = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
    return np.column_stack([A @ affine[0], A @ affine[1]])


def select_target_index_from_guides(
    click_point: np.ndarray,
    guide_points: np.ndarray,
    occupied_mask: np.ndarray,
    *,
    current_index: int,
    max_distance: float = 80.0,
) -> int:
    """
    Pick the nearest reasonable unoccupied guide point for auto assignment.
    """
    click_point = np.asarray(click_point, dtype=float)
    candidate_indices = [i for i in range(len(guide_points)) if not occupied_mask[i] or i == current_index]
    if not candidate_indices:
        return current_index
    distances = [
        (i, float(np.linalg.norm(guide_points[i] - click_point)))
        for i in candidate_indices
        if np.all(np.isfinite(guide_points[i]))
    ]
    if not distances:
        return current_index
    best_index, best_distance = min(distances, key=lambda item: item[1])
    return best_index if best_distance <= max_distance else current_index


def _mask_from_ranges(hsv: np.ndarray, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return mask


def _best_contour_center(mask: np.ndarray, roi_center: np.ndarray, *, min_area: float = 10.0, max_area: float = 5000.0):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = np.inf
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=float)
        score = float(np.linalg.norm(center - roi_center))
        if score < best_score:
            best = center
            best_score = score
    return best


def _refine_color_bead_center(roi_bgr: np.ndarray, expected_color: str) -> np.ndarray | None:
    if roi_bgr.size == 0:
        return None
    roi_center = np.array([roi_bgr.shape[1] / 2.0, roi_bgr.shape[0] / 2.0], dtype=float)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    if expected_color in COLOR_SPECS:
        mask = _mask_from_ranges(hsv, COLOR_SPECS[expected_color])
        center = _best_contour_center(mask, roi_center)
        if center is not None:
            return center

    if expected_color == "black":
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        return _best_contour_center(mask, roi_center)

    if expected_color == "silver":
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        low_sat = cv2.inRange(hsv[:, :, 1], 0, 60)
        bright = cv2.inRange(hsv[:, :, 2], 150, 255)
        return _best_contour_center(low_sat & bright, roi_center)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=8,
        param1=60,
        param2=8,
        minRadius=2,
        maxRadius=max(3, min(30, max(gray.shape[:2]) // 2)),
    )
    if circles is not None:
        circles = circles[0]
        best = min(circles, key=lambda c: float(np.linalg.norm(np.array(c[:2]) - roi_center)))
        return np.array([float(best[0]), float(best[1])], dtype=float)
    return None


def _refine_corner_center(roi_bgr: np.ndarray) -> np.ndarray | None:
    if roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=5)
    if pts is None:
        return None
    roi_center = np.array([roi_bgr.shape[1] / 2.0, roi_bgr.shape[0] / 2.0], dtype=float)
    pts = pts.reshape(-1, 2)
    best = min(pts, key=lambda pt: float(np.linalg.norm(pt - roi_center)))
    return np.asarray(best, dtype=float)


def _refine_red_line_corner(roi_bgr: np.ndarray) -> np.ndarray | None:
    if roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array((0, 60, 50), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array((170, 60, 50), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
    mask = mask1 | mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if int(np.count_nonzero(mask)) < 8:
        return _refine_corner_center(roi_bgr)

    corners = cv2.goodFeaturesToTrack(mask, maxCorners=12, qualityLevel=0.01, minDistance=4)
    if corners is None:
        return _refine_corner_center(roi_bgr)

    roi_center = np.array([roi_bgr.shape[1] / 2.0, roi_bgr.shape[0] / 2.0], dtype=float)
    pts = corners.reshape(-1, 2)
    best = min(pts, key=lambda pt: float(np.linalg.norm(pt - roi_center)))
    return np.asarray(best, dtype=float)


def refine_point_from_bbox(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    spec: PointSpec,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    x0i = max(0, int(np.floor(min(x0, x1))))
    y0i = max(0, int(np.floor(min(y0, y1))))
    x1i = min(image_bgr.shape[1], int(np.ceil(max(x0, x1))))
    y1i = min(image_bgr.shape[0], int(np.ceil(max(y0, y1))))
    if x1i <= x0i or y1i <= y0i:
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)

    roi = image_bgr[y0i:y1i, x0i:x1i]
    if spec.kind == "corner":
        refined = _refine_corner_center(roi)
    elif spec.kind == "aux":
        refined = _refine_red_line_corner(roi)
    else:
        refined = _refine_color_bead_center(roi, spec.color)
    if refined is None:
        refined = np.array([(x1i - x0i) / 2.0, (y1i - y0i) / 2.0], dtype=float)
    return np.array([x0i + refined[0], y0i + refined[1]], dtype=float)


class CalibrationAnnotatorGUI:
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        specs: List[PointSpec],
        mode: str,
        draft_path: Path | None = None,
        review_targets: Dict[str, dict] | None = None,
        auto_assign_enabled: bool = False,
        refine_mode: bool = False,
        refine_window: float = 120.0,
    ) -> None:
        self.image_path = image_path
        self.output_path = output_path
        self.mode = mode
        self.refine_mode = refine_mode
        self.refine_window = float(refine_window)

        self.image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.cam_name = _infer_cam_name(image_path)

        self.specs = list(specs)
        self.n_points = len(self.specs)
        self.points = np.full((self.n_points, 2), np.nan, dtype=float)
        self.sources = np.array(["empty"] * self.n_points, dtype=object)
        self.skipped = np.zeros(self.n_points, dtype=bool)
        self.auto_assign = mode == "calibration" and auto_assign_enabled and not refine_mode
        self.projected_guides = None
        self.review_targets = review_targets or {}
        if self.refine_mode:
            # Refine walks every point in spec order so the user can confirm or
            # drag-refine each clicked position, one at a time, with auto-zoom.
            self.review_indices = list(range(self.n_points))
        else:
            self.review_indices = [
                i for i, spec in enumerate(self.specs)
                if spec.label in self.review_targets
            ]
        self.review_mode = self.refine_mode or len(self.review_indices) > 0

        if draft_path is not None and draft_path.exists():
            for idx, point in _load_annotation_points(draft_path).items():
                loc = self._spec_pos_from_index(idx)
                if loc is not None:
                    self.points[loc] = point
                    self.sources[loc] = "draft"

        if self.refine_mode:
            self.current = self.review_indices[0]
        else:
            self.current = (
                self.review_indices[0] if self.review_mode else self._first_missing_index()
            )
        if self.review_mode and not self.refine_mode:
            self.auto_assign = False
        self.saved = False
        self.drag_start = None
        self.drag_current = None
        self.drag_button = None
        self.xlim = None
        self.ylim = None

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_display()
        if self.review_mode:
            self._focus_on_index(self.current)
            self.update_display()

    def _spec_pos_from_index(self, index: int) -> int | None:
        for i, spec in enumerate(self.specs):
            if spec.index == index:
                return i
        return None

    def _first_missing_index(self) -> int:
        for i in range(self.n_points):
            if np.isnan(self.points[i]).any():
                return i
        return 0

    def _advance(self) -> None:
        if self.current < self.n_points - 1:
            self.current += 1

    def _retreat(self) -> None:
        if self.current > 0:
            self.current -= 1

    def _advance_review(self) -> None:
        if not self.review_mode:
            self._advance()
            return
        pos = self.review_indices.index(self.current)
        if pos < len(self.review_indices) - 1:
            self.current = self.review_indices[pos + 1]

    def _retreat_review(self) -> None:
        if not self.review_mode:
            self._retreat()
            return
        pos = self.review_indices.index(self.current)
        if pos > 0:
            self.current = self.review_indices[pos - 1]

    def _set_point(self, index: int, point: np.ndarray, source: str) -> None:
        self.points[index] = np.asarray(point, dtype=float)
        self.sources[index] = source
        self.skipped[index] = False

    def _project_world_point(self, spec: PointSpec) -> np.ndarray:
        ax0, ax1 = CAM_PROJECTION[self.cam_name]
        return np.array(
            [
                float(spec.xyz[{"X": 0, "Y": 1, "Z": 2}[ax0]]),
                float(spec.xyz[{"X": 0, "Y": 1, "Z": 2}[ax1]]),
            ],
            dtype=float,
        )

    def _estimate_projected_guides(self) -> np.ndarray | None:
        if self.mode != "calibration":
            return None

        src_pts = []
        dst_pts = []
        for i, spec in enumerate(self.specs):
            if spec.xyz is None or np.isnan(self.points[i]).any():
                continue
            src_pts.append(self._project_world_point(spec))
            dst_pts.append(self.points[i])

        if len(src_pts) < 3:
            return None

        affine = _fit_affine(np.vstack(src_pts), np.vstack(dst_pts))
        world_pts = np.vstack([self._project_world_point(spec) for spec in self.specs])
        return _apply_affine(affine, world_pts)

    def _choose_target_index(self, point: np.ndarray) -> int:
        if not self.auto_assign or self.mode != "calibration":
            return self.current
        guides = self._estimate_projected_guides()
        if guides is None:
            return self.current
        occupied = ~np.isnan(self.points[:, 0])
        return select_target_index_from_guides(
            point,
            guides,
            occupied,
            current_index=self.current,
            max_distance=85.0,
        )

    def _clear_current(self) -> None:
        self.points[self.current] = np.array([np.nan, np.nan], dtype=float)
        self.sources[self.current] = "empty"
        self.skipped[self.current] = False

    def _focus_on_index(self, index: int, window: float | None = None) -> None:
        if window is None:
            window = self.refine_window if self.refine_mode else 140.0
        point = self.points[index]
        if np.isnan(point).any():
            if self.refine_mode:
                # Refine walkthrough: always show the wide view for any point
                # without a real clicked position, so the user can place a new
                # point freely without being biased by a projected-guide zoom.
                self.xlim = None
                self.ylim = None
                return
            if self.projected_guides is not None and np.all(np.isfinite(self.projected_guides[index])):
                point = self.projected_guides[index]
            else:
                return
        x, y = float(point[0]), float(point[1])
        self.xlim = (x - window, x + window)
        self.ylim = (y + window, y - window)

    def _current_group_hint(self, spec: PointSpec) -> str:
        if spec.kind != "bead":
            return ""
        same_color = [s for s in self.specs if s.kind == "bead" and s.color == spec.color]
        same_color = sorted(same_color, key=lambda s: s.index)
        if not same_color:
            return ""
        position = 1 + next(i for i, s in enumerate(same_color) if s.index == spec.index)
        return f"{_color_display_name(spec.color)} bead {position}/{len(same_color)}"

    def update_display(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.image_rgb)
        self.projected_guides = self._estimate_projected_guides()

        if self.xlim is not None and self.ylim is not None:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
        else:
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()

        if self.drag_start is not None and self.drag_current is not None and self.drag_button == 1:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            rect = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor="yellow",
                linewidth=2,
                linestyle="--",
            )
            self.ax.add_patch(rect)

        if self.projected_guides is not None and self.mode == "calibration":
            for i, spec in enumerate(self.specs):
                if not np.isnan(self.points[i]).any():
                    continue
                gx, gy = self.projected_guides[i]
                color = DISPLAY_COLORS.get(spec.color, "cyan")
                self.ax.scatter([gx], [gy], s=32, marker="+", color=color, alpha=0.45, linewidths=1.2)

        if self.review_mode and self.review_targets and not self.refine_mode:
            for i in self.review_indices:
                spec = self.specs[i]
                point = self.points[i]
                if np.isnan(point).any():
                    if self.projected_guides is not None and np.all(np.isfinite(self.projected_guides[i])):
                        point = self.projected_guides[i]
                    else:
                        continue
                x, y = point
                err = self.review_targets.get(spec.label, {}).get("err_mm")
                self.ax.add_patch(
                    Circle(
                        (x, y),
                        radius=24 if i == self.current else 18,
                        fill=False,
                        color="red",
                        linewidth=3 if i == self.current else 1.8,
                        alpha=0.9 if i == self.current else 0.7,
                    )
                )
                if err is not None:
                    self.ax.text(x + 10, y + 14, f"{err:.1f} mm", color="red", fontsize=8, weight="bold")

        for i, spec in enumerate(self.specs):
            point = self.points[i]
            if np.isnan(point).any():
                continue
            x, y = point
            color = DISPLAY_COLORS.get(spec.color, "cyan")
            source = self.sources[i]
            marker = "o" if spec.kind == "bead" else "x"
            face = "none" if source == "draft" else color
            edge = color
            size = 130 if i == self.current else 90 if spec.kind == "corner" else 70
            self.ax.scatter([x], [y], s=size, marker=marker, facecolors=face, edgecolors=edge, linewidths=2, alpha=0.9)
            self.ax.text(x + 6, y - 6, str(spec.index), color=edge, fontsize=9, weight="bold")
            if i == self.current:
                self.ax.add_patch(Circle((x, y), radius=18, fill=False, color="cyan", linewidth=2.5))

        annotated = int(np.sum(~np.isnan(self.points[:, 0])))
        current_spec = self.specs[self.current]
        source = self.sources[self.current]
        if self.refine_mode:
            mode_label = "refine"
        elif self.mode == "auxiliary":
            mode_label = "auxiliary"
        else:
            mode_label = "calibration"
        color_hint = f" | color: {current_spec.color}" if current_spec.kind != "corner" else ""
        auto_hint = "on" if self.auto_assign else "off"
        group_hint = self._current_group_hint(current_spec)
        review_hint = ""
        if self.review_mode:
            review_pos = 1 + self.review_indices.index(self.current)
            err = self.review_targets.get(current_spec.label, {}).get("err_mm")
            err_text = f" | err={err:.1f} mm" if err is not None else ""
            tag = "refine" if self.refine_mode else "review"
            review_hint = f" | {tag} {review_pos}/{len(self.review_indices)}{err_text}"
        if self.refine_mode:
            instructions = (
                "Drag box: refine center | Click: replace position | Right-click: clear | "
                "Left/Right: prev/next point | m: next missing | f: re-focus | r: wide view | "
                "+/-: zoom window | s: save | q: quit"
            )
        else:
            instructions = (
                "Click: exact point | Drag box: refine center | Right-click: clear current | "
                "Left/Right: navigate | n: skip | m: next missing/review | a: toggle auto-assign | "
                "f: focus current | s: save | q: quit | r: reset zoom"
            )
        title = (
            f"{mode_label} | {self.current + 1}/{self.n_points} | annotated {annotated}/{self.n_points} | "
            f"current: {current_spec.label}{color_hint} | source: {source} | auto-assign: {auto_hint}{review_hint}\n"
            + instructions
        )
        self.ax.set_title(title, fontsize=11)
        badge_color = DISPLAY_COLORS.get(current_spec.color, "white")
        badge_text = group_hint or current_spec.label
        self.ax.text(
            0.015,
            0.98,
            badge_text,
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color="black" if badge_color in {"yellow", "lime", "silver"} else "white",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": badge_color,
                "edgecolor": "black",
                "alpha": 0.85,
            },
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def on_button_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.drag_start = (event.xdata, event.ydata)
        self.drag_current = (event.xdata, event.ydata)
        self.drag_button = event.button

    def on_motion(self, event) -> None:
        if self.drag_start is None or self.drag_button != 1:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.drag_current = (event.xdata, event.ydata)
        self.update_display()

    def on_button_release(self, event) -> None:
        if self.drag_start is None or self.drag_button is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.drag_start = None
            self.drag_current = None
            self.drag_button = None
            self.update_display()
            return

        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata
        button = self.drag_button
        self.drag_start = None
        self.drag_current = None
        self.drag_button = None

        if button == 3:
            self._clear_current()
            self.update_display()
            return

        if button != 1:
            self.update_display()
            return

        if abs(x1 - x0) < 4 and abs(y1 - y0) < 4:
            point = np.array([x1, y1], dtype=float)
            target_index = self._choose_target_index(point)
            self.current = target_index
            self._set_point(self.current, point, "manual")
        else:
            bbox = (x0, y0, x1, y1)
            center_guess = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)
            target_index = self._choose_target_index(center_guess)
            self.current = target_index
            point = refine_point_from_bbox(self.image_bgr, bbox, self.specs[self.current])
            self._set_point(self.current, point, "manual_refined")

        if self.review_mode:
            self._advance_review()
            self._focus_on_index(self.current)
        else:
            self.current = self._first_missing_index() if np.any(np.isnan(self.points[:, 0])) else self.current
        self.update_display()

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xlim = self.ax.get_xlim() if self.xlim is None else self.xlim
        ylim = self.ax.get_ylim() if self.ylim is None else self.ylim
        zoom_factor = 1.15 if event.button == "up" else 1 / 1.15

        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor
        self.xlim = (event.xdata - x_range / 2, event.xdata + x_range / 2)
        self.ylim = (event.ydata - y_range / 2, event.ydata + y_range / 2)
        self.update_display()

    def on_key(self, event) -> None:
        if event.key == "left":
            self._retreat_review()
            if self.review_mode:
                self._focus_on_index(self.current)
        elif event.key == "right":
            self._advance_review()
            if self.review_mode:
                self._focus_on_index(self.current)
        elif event.key == "m":
            if self.refine_mode:
                self._jump_to_next_missing_in_refine()
                self._focus_on_index(self.current)
            elif self.review_mode:
                self._advance_review()
                self._focus_on_index(self.current)
            else:
                self.current = self._first_missing_index()
        elif event.key == "n":
            self.skipped[self.current] = True
            self._advance_review()
            if self.review_mode:
                self._focus_on_index(self.current)
        elif event.key == "a":
            if self.mode == "calibration" and not self.refine_mode:
                self.auto_assign = not self.auto_assign
        elif event.key == "d":
            self._clear_current()
        elif event.key == "f":
            self._focus_on_index(self.current)
        elif event.key == "r":
            self.xlim = None
            self.ylim = None
        elif event.key in ("+", "="):
            # Shrink refine window (zoom in)
            self.refine_window = max(20.0, self.refine_window * 0.8)
            if self.refine_mode:
                self._focus_on_index(self.current)
        elif event.key in ("-", "_"):
            # Grow refine window (zoom out)
            self.refine_window = min(600.0, self.refine_window * 1.25)
            if self.refine_mode:
                self._focus_on_index(self.current)
        elif event.key == "s":
            self.save_points()
            self.saved = True
            plt.close(self.fig)
            return
        elif event.key == "q":
            self.saved = False
            plt.close(self.fig)
            return
        self.update_display()

    def _jump_to_next_missing_in_refine(self) -> None:
        """Jump to the next point in `review_indices` that still has no position."""
        order = self.review_indices
        if not order:
            return
        try:
            start = order.index(self.current) + 1
        except ValueError:
            start = 0
        n = len(order)
        for offset in range(n):
            idx = order[(start + offset) % n]
            if np.isnan(self.points[idx]).any():
                self.current = idx
                return
        # Everything already annotated - stay put
        return

    def save_points(self) -> None:
        valid_indices = [i for i in range(self.n_points) if not np.isnan(self.points[i]).any()]
        if not valid_indices:
            print("[WARN] No points to save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            if self.mode == "auxiliary":
                f.write("# Auxiliary annotation file\n")
                f.write("# Format: index x y label\n")
                f.write("# x y: 2D image coordinates (pixels)\n")
                f.write("# label: auxiliary correspondence label\n")
                f.write("#\n")
                for i in valid_indices:
                    spec = self.specs[i]
                    x, y = self.points[i]
                    f.write(f"{spec.index} {int(round(x))} {int(round(y))} {spec.label}\n")
            else:
                f.write("# Manual annotation file\n")
                f.write("# Format: index x y X Y Z label\n")
                f.write("# index: point index in template (0-based)\n")
                f.write("# x y: 2D image coordinates (pixels)\n")
                f.write("# X Y Z: 3D world coordinates (mm) from template\n")
                f.write("# label: point label\n")
                f.write("#\n")
                for i in valid_indices:
                    spec = self.specs[i]
                    x, y = self.points[i]
                    X, Y, Z = spec.xyz
                    f.write(f"{spec.index} {int(round(x))} {int(round(y))} {X:.6f} {Y:.6f} {Z:.6f} {spec.label}\n")

        marking_dir = self.output_path.parent / "manual_marking"
        marking_dir.mkdir(parents=True, exist_ok=True)
        preview_path = marking_dir / f"{self.image_path.stem}_manual_marking.png"
        self.fig.savefig(preview_path, dpi=200, bbox_inches="tight", format="png")
        print(f"Saved {len(valid_indices)} points to: {self.output_path}")
        print(f"Saved annotated image to: {preview_path}")


def annotate_way(
    folder: str,
    way_name: str,
    output_root: str,
    *,
    mode: str = "calibration",
    aux_preset: str = "pillar_lines",
    include_bottom_corners: bool = False,
    auto_remap_preview: bool = True,
    auto_assign_enabled: bool = False,
    use_remapped_for_post_pipeline: bool = False,
    draft_root: str | None = None,
    review_csv: str | None = None,
) -> None:
    raw_root = Path(folder)
    output_root_path = Path(output_root)
    image_paths = _find_way_images(raw_root, way_name)
    if mode == "auxiliary":
        specs = _build_auxiliary_specs(aux_preset)
    else:
        template_path = _find_template_file(raw_root, way_name)
        specs = _load_point_specs(template_path, way_name)
        if include_bottom_corners:
            specs = _append_bottom_corners(specs)
        specs = sorted(specs, key=_annotation_sort_key)
    review_targets = _load_review_targets(Path(review_csv)) if review_csv else {}

    for cam_name in ("cam0", "cam1", "cam2"):
        image_path = image_paths[cam_name]
        if mode == "auxiliary":
            output_path = output_root_path / f"cali_{way_name}_{cam_name}_aux_{aux_preset}.txt"
        else:
            output_path = output_root_path / f"cali_{way_name}_{cam_name}_image_coords.txt"
        draft_path = None
        if draft_root is not None:
            if mode == "auxiliary":
                candidate = Path(draft_root) / f"cali_{way_name}_{cam_name}_aux_{aux_preset}.txt"
            else:
                candidate = Path(draft_root) / f"cali_{way_name}_{cam_name}_image_coords.txt"
            if candidate.exists():
                draft_path = candidate
        if review_targets and draft_path is None and output_path.exists():
            draft_path = output_path
        print(f"\nAnnotating {cam_name}: {image_path.name}")
        if draft_path is not None:
            print(f"Using draft preload: {draft_path}")
        gui = CalibrationAnnotatorGUI(
            image_path=image_path,
            output_path=output_path,
            specs=specs,
            mode=mode,
            draft_path=draft_path,
            review_targets=review_targets,
            auto_assign_enabled=auto_assign_enabled,
        )
        plt.show()
        if not gui.saved:
            print("Annotation cancelled.")
            return
        if mode == "calibration" and auto_remap_preview:
            remap_dir = output_root_path / "remap_preview"
            remap_output_path = remap_dir / f"cali_{way_name}_{cam_name}_image_coords.txt"
            remap_preview_path = remap_dir / f"cali_{way_name}_{cam_name}_remapped_preview.png"
            remap_info = remap_single_camera_annotation(
                folder=str(raw_root),
                way_name=way_name,
                cam_name=cam_name,
                input_path=str(output_path),
                output_path=str(remap_output_path),
                preview_path=str(remap_preview_path),
            )
            print(
                f"Auto-remap preview: {remap_info['changed_indices']} label change(s) suggested for {cam_name}. "
                f"Preview saved to: {remap_info['preview_path']}"
            )

    if mode == "calibration" and auto_remap_preview:
        if use_remapped_for_post_pipeline:
            print("\nPost-annotation calibration will use remapped labels.")
        else:
            print("\nPost-annotation calibration will use your manual labels. Remap files are preview suggestions only.")
        summary = run_post_annotation_pipeline(
            folder=folder,
            way_name=way_name,
            manual_output_root=str(output_root_path),
            use_remapped=use_remapped_for_post_pipeline,
        )
        point_summary = summary.get("point_mapping_summary") or {}
        reproj_summary = summary.get("reprojection_summary") or {}
        print("\nPost-annotation pipeline completed.")
        if point_summary:
            print(
                "3D fit summary: "
                f"mean={point_summary.get('mean_error_mm', float('nan')):.2f} mm, "
                f"rmse={point_summary.get('rmse_mm', float('nan')):.2f} mm, "
                f"max={point_summary.get('max_error_mm', float('nan')):.2f} mm"
            )
        if reproj_summary:
            for cam_name in ("cam0", "cam1", "cam2"):
                if cam_name in reproj_summary:
                    stats = reproj_summary[cam_name]
                    print(
                        f"  {cam_name} reprojection: "
                        f"mean={stats['mean']:.2f} px, median={stats['median']:.2f} px, max={stats['max']:.2f} px"
                    )
        print(f"Full summary saved to: {summary['summary_path']}")


def refine_way(
    folder: str,
    way_name: str,
    annotation_dir: str,
    *,
    include_bottom_corners: bool = True,
    refine_window: float = 120.0,
) -> None:
    """
    Walk through an existing per-camera annotation file, auto-zoomed to each
    already-clicked point so the user can confirm / drag-box refine each
    position, and wide-zoomed for any point that has no position yet.

    The specs come from the way's template (so bottom-corner slots are
    available for fresh placement if they were not clicked before), and the
    existing `cali_{way}_{cam}_image_coords.txt` file inside `annotation_dir`
    is preloaded as the draft.
    """
    raw_root = Path(folder)
    annotation_root = Path(annotation_dir)
    image_paths = _find_way_images(raw_root, way_name)
    template_path = _find_template_file(raw_root, way_name)
    specs = _load_point_specs(template_path, way_name)
    if include_bottom_corners:
        specs = _append_bottom_corners(specs)
    specs = sorted(specs, key=_annotation_sort_key)

    for cam_name in ("cam0", "cam1", "cam2"):
        image_path = image_paths[cam_name]
        output_path = annotation_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        draft_path = output_path if output_path.exists() else None

        print(f"\nRefining {cam_name}: {image_path.name}")
        if draft_path is not None:
            print(f"Preload from: {draft_path}")
        else:
            print("No existing annotation found - starting fresh.")

        gui = CalibrationAnnotatorGUI(
            image_path=image_path,
            output_path=output_path,
            specs=specs,
            mode="calibration",
            draft_path=draft_path,
            refine_mode=True,
            refine_window=refine_window,
        )
        plt.show()
        if not gui.saved:
            print("Refinement cancelled - no changes written.")
            return


def run_post_annotation_pipeline(
    *,
    folder: str,
    way_name: str,
    manual_output_root: str,
    use_remapped: bool = False,
) -> Dict[str, object]:
    """
    Run remap -> calibration training -> reprojection QC after all three camera
    annotation files are saved.
    """
    manual_root = Path(manual_output_root)
    remap_root = manual_root / "remapped"
    raw_root = Path(folder)

    remap_summary: Dict[str, object] = {"cameras": {}}
    for cam_name in ("cam0", "cam1", "cam2"):
        input_path = manual_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        output_path = remap_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        preview_path = remap_root / f"cali_{way_name}_{cam_name}_remapped_preview.png"
        remap_summary["cameras"][cam_name] = remap_single_camera_annotation(
            folder=str(raw_root),
            way_name=way_name,
            cam_name=cam_name,
            input_path=str(input_path),
            output_path=str(output_path),
            preview_path=str(preview_path),
        )

    calibration_input_root = remap_root if use_remapped else manual_root
    trained_model_root = manual_root / "post_pipeline"
    model_output = trained_model_root / "models"
    qc_output = trained_model_root / "qc"
    calib_result = run_calibration_pipeline(
        calibration_folder=str(raw_root),
        image_coords_folder=str(calibration_input_root),
        filename_keyword=way_name,
        alpha_range=[1e-3],
        poly_deg_range=[1, 2],
        cv_folds=3,
        train_fraction=0.8,
        num_permutations=20,
        output_folder=str(model_output),
        beads_only=False,
        model_types=["polynomial"],
        results_root=str(qc_output),
    )

    mapping_csv = Path(calib_result["mapping_csv"])
    reproj_df = load_reprojection_mapping_csv(str(mapping_csv))
    models, errors, reproj_uv = reprojection_errors_per_camera(reproj_df)
    views_path = qc_output / f"{mapping_csv.stem}_reprojection_2d_views.png"
    plot_reprojection_2d_views(reproj_df, reproj_uv, models, str(views_path))

    reprojection_summary = {
        cam: {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "max": float(np.max(vals)),
        }
        for cam, vals in errors.items()
    }

    point_mapping_summary_path = qc_output / f"{mapping_csv.stem}_summary.json"
    point_mapping_summary = None
    if point_mapping_summary_path.exists():
        with point_mapping_summary_path.open("r", encoding="utf-8") as f:
            point_mapping_summary = json.load(f)

    summary = {
        "remap_summary": remap_summary,
        "calibration_input": str(calibration_input_root),
        "used_remapped_for_training": bool(use_remapped),
        "model_output": str(model_output),
        "qc_output": str(qc_output),
        "mapping_csv": str(mapping_csv),
        "point_mapping_summary": point_mapping_summary,
        "reprojection_summary": reprojection_summary,
        "reprojection_views_path": str(views_path),
    }
    summary_path = trained_model_root / "post_pipeline_summary.json"
    trained_model_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration annotation GUI with draft-point preload and bbox center refinement")
    parser.add_argument("--folder", required=True, help="Calibration raw root folder containing template + way*_pic directories")
    parser.add_argument("--way_name", required=True, help="Way name, for example way1")
    parser.add_argument("--output", required=True, help="Output folder for final *_image_coords.txt files")
    parser.add_argument("--draft_root", default=None, help="Optional folder containing draft preload *_image_coords.txt files")
    parser.add_argument(
        "--mode",
        choices=["calibration", "auxiliary"],
        default="calibration",
        help="Annotate standard world-coordinate calibration points or auxiliary 2D-only correspondences",
    )
    parser.add_argument(
        "--aux_preset",
        choices=["pillar_lines"],
        default="pillar_lines",
        help="Auxiliary-point preset to use in auxiliary mode",
    )
    parser.add_argument(
        "--include_bottom_corners",
        action="store_true",
        help="Add four bottom pillar-corner world-coordinate points at z=4, symmetric to the top corners at z=194",
    )
    parser.add_argument(
        "--no_auto_remap_preview",
        action="store_true",
        help="Do not generate per-camera remap-preview files automatically after each save in calibration mode",
    )
    parser.add_argument(
        "--auto_assign",
        action="store_true",
        help="Enable automatic snapping of a click to the nearest predicted missing bead ID. Off by default.",
    )
    parser.add_argument(
        "--use_remapped_for_post_pipeline",
        action="store_true",
        help="Train/QC the automatic post-annotation calibration on remapped labels instead of manual labels.",
    )
    parser.add_argument(
        "--review_csv",
        default=None,
        help="Optional worst-beads review CSV. When set, the GUI highlights those beads and navigates through them first.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help=(
            "Refine mode: load the existing *_image_coords.txt files from --output "
            "and step through every point with auto-zoom. Drag a box around a point "
            "to recompute its center; click to replace; +/- to change zoom window."
        ),
    )
    parser.add_argument(
        "--refine_window",
        type=float,
        default=120.0,
        help="Half-size of the refine-mode zoom window in pixels (default: 120).",
    )
    args = parser.parse_args()

    if args.refine:
        if args.mode != "calibration":
            raise SystemExit("--refine is only supported for calibration mode")
        refine_way(
            args.folder,
            args.way_name.lower(),
            args.output,
            include_bottom_corners=args.include_bottom_corners,
            refine_window=args.refine_window,
        )
        return

    annotate_way(
        args.folder,
        args.way_name.lower(),
        args.output,
        mode=args.mode,
        aux_preset=args.aux_preset,
        include_bottom_corners=args.include_bottom_corners,
        auto_remap_preview=not args.no_auto_remap_preview,
        auto_assign_enabled=args.auto_assign,
        use_remapped_for_post_pipeline=args.use_remapped_for_post_pipeline,
        draft_root=args.draft_root,
        review_csv=args.review_csv,
    )


if __name__ == "__main__":
    main()
