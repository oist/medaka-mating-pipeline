"""
Relabel calibration bead annotations to consistent bead IDs using image color
and template geometry.

This is designed for the colored-bead calibration rigs where manual clicking
finds the right bead centers but the bead IDs can become inconsistent across
camera views.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .lib.visualize_coords_from_template import get_point_color, parse_template
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.visualize_coords_from_template import get_point_color, parse_template  # type: ignore


CAM_PROJECTION = {
    "cam0": ("X", "Y"),
    "cam1": ("X", "Z"),
    "cam2": ("Y", "Z"),
}
DISPLAY_BGR = {
    "pink": (255, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "black": (20, 20, 20),
    "orange": (0, 165, 255),
    "purple": (150, 0, 150),
    "silver": (200, 200, 200),
}
COLOR_SPECS = {
    "pink": [((145, 60, 80), (179, 255, 255))],
    "purple": [((120, 50, 40), (150, 255, 255))],
    "yellow": [((18, 60, 80), (40, 255, 255))],
    "orange": [((5, 60, 70), (22, 255, 255))],
    "blue": [((90, 60, 50), (130, 255, 255))],
    "black": [((0, 0, 0), (179, 140, 90))],
    "silver": [((0, 0, 140), (179, 80, 255))],
}
BEAD_COLORS = ["pink", "yellow", "blue", "black", "orange", "purple", "silver"]
COLOR_HUE_CENTERS = {
    "pink": 160.0,
    "purple": 138.0,
    "blue": 112.0,
    "yellow": 30.0,
    "orange": 16.0,
}
INNER_SAMPLE_RADIUS = 8


@dataclass
class TemplateBead:
    index: int
    label: str
    color: str
    xyz: np.ndarray


@dataclass
class AnnotationRow:
    index: int
    x: float
    y: float
    X: float
    Y: float
    Z: float
    label: str
    inferred_color: str | None = None
    template_color: str | None = None


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
    return out


def _load_template_beads(template_path: Path, way_name: str) -> List[TemplateBead]:
    sections = parse_template(template_path)
    beads_key = f"BEADS_{way_name.upper()}"
    if beads_key not in sections:
        raise ValueError(f"{template_path} does not contain section {beads_key}")
    coords, labels = sections[beads_key]
    beads: List[TemplateBead] = []
    for i, coord in enumerate(coords):
        label = labels[i] if i < len(labels) and labels[i] else f"bead_{i + 1:03d}"
        beads.append(
            TemplateBead(
                index=4 + i,
                label=f"BEAD: {label}",
                color=get_point_color(4 + i),
                xyz=np.asarray(coord, dtype=float),
            )
        )
    return beads


def _load_annotation_rows(path: Path) -> List[AnnotationRow]:
    rows: List[AnnotationRow] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append(
                AnnotationRow(
                    index=int(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    X=float(parts[3]),
                    Y=float(parts[4]),
                    Z=float(parts[5]),
                    label=" ".join(parts[6:]),
                )
            )
    return rows


def _project_template_beads(beads: List[TemplateBead], cam_name: str) -> Dict[int, np.ndarray]:
    ax0, ax1 = CAM_PROJECTION[cam_name]
    axis_to_idx = {"X": 0, "Y": 1, "Z": 2}
    return {
        bead.index: np.array([bead.xyz[axis_to_idx[ax0]], bead.xyz[axis_to_idx[ax1]]], dtype=float)
        for bead in beads
    }


def _mask_from_ranges(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return mask


def infer_point_color(image_bgr: np.ndarray, point: np.ndarray, radius: int = 14) -> str:
    x = int(round(point[0]))
    y = int(round(point[1]))
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image_bgr.shape[1], x + radius + 1)
    y1 = min(image_bgr.shape[0], y + radius + 1)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return "black"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy, xx = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]
    center = np.array([(roi.shape[1] - 1) / 2.0, (roi.shape[0] - 1) / 2.0], dtype=float)
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    core_mask = dist <= min(float(radius), float(INNER_SAMPLE_RADIUS))
    if not np.any(core_mask):
        core_mask = np.ones(roi.shape[:2], dtype=bool)

    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)
    hue = hsv[:, :, 0].astype(float)

    core_sat = sat[core_mask]
    core_val = val[core_mask]
    core_hue = hue[core_mask]

    if float(np.median(core_sat)) < 55:
        return "black" if float(np.median(core_val)) < 100 else "silver"

    chromatic = (core_sat > 55) & (core_val > 50)
    if not np.any(chromatic):
        return "black" if float(np.median(core_val)) < 100 else "silver"

    hues = core_hue[chromatic]

    def circular_hue_distance(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    best_color = min(
        COLOR_HUE_CENTERS.items(),
        key=lambda item: float(np.mean([circular_hue_distance(h, item[1]) for h in hues])),
    )[0]

    if best_color == "purple" and float(np.median(core_sat)) > 120 and float(np.median(core_val)) > 120:
        pink_dist = float(np.mean([circular_hue_distance(h, COLOR_HUE_CENTERS["pink"]) for h in hues]))
        purple_dist = float(np.mean([circular_hue_distance(h, COLOR_HUE_CENTERS["purple"]) for h in hues]))
        if pink_dist <= purple_dist + 6.0:
            return "pink"
    return best_color


def _fit_affine(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    A = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=float)])
    tx, *_ = np.linalg.lstsq(A, dst_pts[:, 0], rcond=None)
    ty, *_ = np.linalg.lstsq(A, dst_pts[:, 1], rcond=None)
    return np.vstack([tx, ty])


def _apply_affine(affine: np.ndarray, pts: np.ndarray) -> np.ndarray:
    A = np.hstack([pts, np.ones((len(pts), 1), dtype=float)])
    return np.column_stack([A @ affine[0], A @ affine[1]])


def _choose_group_order(
    template_points: np.ndarray,
    observed_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort a same-color group along its dominant projected axis and choose the
    direction that best matches the observed points.
    """
    centered = template_points - np.mean(template_points, axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    tpl_scalar = template_points @ axis
    obs_scalar = observed_points @ axis

    tpl_order = np.argsort(tpl_scalar)
    obs_order_fwd = np.argsort(obs_scalar)
    obs_order_rev = obs_order_fwd[::-1]

    fwd_cost = float(np.sum(np.linalg.norm(template_points[tpl_order] - observed_points[obs_order_fwd], axis=1)))
    rev_cost = float(np.sum(np.linalg.norm(template_points[tpl_order] - observed_points[obs_order_rev], axis=1)))
    return tpl_order, (obs_order_fwd if fwd_cost <= rev_cost else obs_order_rev)


def _assign_group_by_order(
    beads: List[TemplateBead],
    rows: List[AnnotationRow],
    projected_img: Dict[int, np.ndarray],
    *,
    cam_name: str,
) -> List[AnnotationRow]:
    if len(beads) != len(rows):
        cost = np.zeros((len(beads), len(rows)), dtype=float)
        for i, bead in enumerate(beads):
            bead_point = projected_img[bead.index]
            for j, row in enumerate(rows):
                cost[i, j] = float(np.linalg.norm(bead_point - np.array([row.x, row.y], dtype=float)))
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(beads[int(i)], rows[int(j)]) for i, j in zip(row_ind, col_ind)]
    else:
        if cam_name == "cam2":
            bead_order = np.argsort([bead.index for bead in beads])
            obs_order = np.argsort([row.y for row in rows])
            pairs = [(beads[int(i)], rows[int(j)]) for i, j in zip(bead_order, obs_order)]
        else:
            template_points = np.vstack([projected_img[bead.index] for bead in beads])
            observed_points = np.vstack([[row.x, row.y] for row in rows])
            if np.max(np.ptp(template_points, axis=0)) < 1e-6:
                centered = observed_points - np.mean(observed_points, axis=0, keepdims=True)
                _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
                axis = vh[0]
                obs_scalar = observed_points @ axis
                obs_order = np.argsort(obs_scalar)
                bead_order = np.argsort([bead.index for bead in beads])
                pairs = [(beads[int(i)], rows[int(j)]) for i, j in zip(bead_order, obs_order)]
            else:
                tpl_order, obs_order = _choose_group_order(template_points, observed_points)
                pairs = [(beads[int(i)], rows[int(j)]) for i, j in zip(tpl_order, obs_order)]

    remapped_rows = []
    for bead, row in pairs:
        remapped_rows.append(
            AnnotationRow(
                index=bead.index,
                x=row.x,
                y=row.y,
                X=float(bead.xyz[0]),
                Y=float(bead.xyz[1]),
                Z=float(bead.xyz[2]),
                label=bead.label,
                inferred_color=row.inferred_color,
                template_color=bead.color,
            )
        )
    return remapped_rows


def remap_bead_rows_for_camera(
    image_bgr: np.ndarray,
    cam_name: str,
    template_beads: List[TemplateBead],
    rows: List[AnnotationRow],
) -> List[AnnotationRow]:
    fixed_rows = [row for row in rows if not row.label.startswith("BEAD:")]
    bead_rows = [row for row in rows if row.label.startswith("BEAD:")]
    if not bead_rows:
        fixed_rows.sort(key=lambda row: row.index)
        return fixed_rows

    projected = _project_template_beads(template_beads, cam_name)

    for row in bead_rows:
        row.inferred_color = infer_point_color(image_bgr, np.array([row.x, row.y], dtype=float))

    template_by_color = defaultdict(list)
    observed_by_color = defaultdict(list)
    for bead in template_beads:
        template_by_color[bead.color].append(bead)
    for row in bead_rows:
        observed_by_color[row.inferred_color].append(row)

    src_pts = []
    dst_pts = []
    for color in BEAD_COLORS:
        if not template_by_color[color] or not observed_by_color[color]:
            continue
        tpl_centroid = np.mean(np.vstack([projected[bead.index] for bead in template_by_color[color]]), axis=0)
        obs_centroid = np.mean(np.vstack([[row.x, row.y] for row in observed_by_color[color]]), axis=0)
        src_pts.append(tpl_centroid)
        dst_pts.append(obs_centroid)

    if len(src_pts) < 3:
        raise RuntimeError(f"Not enough color groups to estimate affine transform for {cam_name}")

    affine = _fit_affine(np.vstack(src_pts), np.vstack(dst_pts))
    projected_img = {
        bead.index: _apply_affine(affine, projected[bead.index].reshape(1, 2))[0]
        for bead in template_beads
    }

    remapped: List[AnnotationRow] = []
    for color in BEAD_COLORS:
        beads = sorted(template_by_color[color], key=lambda bead: bead.index)
        obs = observed_by_color[color]
        if not beads or not obs:
            continue
        remapped.extend(_assign_group_by_order(beads, obs, projected_img, cam_name=cam_name))

    used_xy = {(int(round(row.x)), int(round(row.y))) for row in remapped}
    remaining_rows = [
        row for row in bead_rows
        if (int(round(row.x)), int(round(row.y))) not in used_xy
    ]
    remaining_beads = [
        bead for bead in template_beads
        if bead.index not in {row.index for row in remapped}
    ]

    if remaining_beads and remaining_rows:
        cost = np.zeros((len(remaining_beads), len(remaining_rows)), dtype=float)
        for i, bead in enumerate(remaining_beads):
            bead_point = projected_img[bead.index]
            for j, row in enumerate(remaining_rows):
                obs_point = np.array([row.x, row.y], dtype=float)
                dist = float(np.linalg.norm(bead_point - obs_point))
                color_penalty = 0.0 if bead.color == row.inferred_color else 120.0
                if bead.color in {"black", "silver"} or row.inferred_color in {"black", "silver"}:
                    color_penalty *= 0.5
                cost[i, j] = dist + color_penalty
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            bead = remaining_beads[int(i)]
            row = remaining_rows[int(j)]
            remapped.append(
                AnnotationRow(
                    index=bead.index,
                    x=row.x,
                    y=row.y,
                    X=float(bead.xyz[0]),
                    Y=float(bead.xyz[1]),
                    Z=float(bead.xyz[2]),
                    label=bead.label,
                    inferred_color=row.inferred_color,
                    template_color=bead.color,
                )
            )

    remapped.extend(fixed_rows)
    remapped.sort(key=lambda row: row.index)
    return remapped


def _save_rows(path: Path, rows: List[AnnotationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Remapped annotation file\n")
        f.write("# Format: index x y X Y Z label\n")
        f.write("#\n")
        for row in rows:
            f.write(
                f"{row.index} {int(round(row.x))} {int(round(row.y))} "
                f"{row.X:.6f} {row.Y:.6f} {row.Z:.6f} {row.label}\n"
            )


def _save_preview(path: Path, image_bgr: np.ndarray, rows: List[AnnotationRow]) -> None:
    preview = image_bgr.copy()
    for row in rows:
        color_name = row.template_color or row.inferred_color or "black"
        color = DISPLAY_BGR.get(color_name, (255, 255, 255))
        x = int(round(row.x))
        y = int(round(row.y))
        cv2.circle(preview, (x, y), 10, color, 2)
        cv2.putText(preview, str(row.index), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if row.template_color is not None and row.inferred_color is not None and row.template_color != row.inferred_color:
            cv2.putText(
                preview,
                f"{row.inferred_color}?",
                (x + 6, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def remap_single_camera_annotation(
    *,
    folder: str,
    way_name: str,
    cam_name: str,
    input_path: str,
    output_path: str,
    preview_path: str,
) -> Dict[str, object]:
    raw_root = Path(folder)
    template_path = _find_template_file(raw_root, way_name)
    template_beads = _load_template_beads(template_path, way_name)
    image_paths = _find_way_images(raw_root, way_name)

    rows = _load_annotation_rows(Path(input_path))
    image_bgr = cv2.imread(str(image_paths[cam_name]), cv2.IMREAD_COLOR)
    remapped = remap_bead_rows_for_camera(image_bgr, cam_name, template_beads, rows)
    _save_rows(Path(output_path), remapped)
    _save_preview(Path(preview_path), image_bgr, remapped)

    old_by_xy = {(int(round(row.x)), int(round(row.y))): row for row in rows}
    changed = 0
    for row in remapped:
        old = old_by_xy.get((int(round(row.x)), int(round(row.y))))
        if old is not None and old.index != row.index:
            changed += 1

    return {
        "input_path": input_path,
        "output_path": output_path,
        "preview_path": preview_path,
        "n_points": len(remapped),
        "changed_indices": changed,
        "inferred_colors": dict(Counter(row.inferred_color for row in remapped)),
    }


def run_remap(folder: str, way_name: str, input_dir: str, output_dir: str) -> Dict:
    raw_root = Path(folder)
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    template_path = _find_template_file(raw_root, way_name)
    template_beads = _load_template_beads(template_path, way_name)
    image_paths = _find_way_images(raw_root, way_name)

    summary = {
        "folder": str(raw_root),
        "way_name": way_name,
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "cameras": {},
    }

    for cam_name in ("cam0", "cam1", "cam2"):
        input_path = input_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        output_path = output_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        preview_path = output_root / f"cali_{way_name}_{cam_name}_remapped_preview.png"
        summary["cameras"][cam_name] = remap_single_camera_annotation(
            folder=str(raw_root),
            way_name=way_name,
            cam_name=cam_name,
            input_path=str(input_path),
            output_path=str(output_path),
            preview_path=str(preview_path),
        )

    summary_path = output_root / f"cali_{way_name}_remap_summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap calibration bead annotations to consistent bead IDs using image color and template geometry")
    parser.add_argument("--folder", required=True, help="Calibration raw root folder containing templates and way*_pic directories")
    parser.add_argument("--way_name", required=True, help="Way name, for example way1")
    parser.add_argument("--input_dir", required=True, help="Input folder containing manually annotated *_image_coords.txt files")
    parser.add_argument("--output_dir", required=True, help="Output folder for remapped annotation files and previews")
    args = parser.parse_args()

    result = run_remap(args.folder, args.way_name.lower(), args.input_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
