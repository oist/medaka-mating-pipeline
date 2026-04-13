"""
Pre-annotate calibration bead locations from raw images using bead colors.

This script detects bead candidates in the raw calibration images, aligns them
to the 3D template using a color-guided affine fit, then exports draft
`*_image_coords.txt` files compatible with the calibration pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .lib.visualize_coords_from_template import get_point_color, parse_template
except ImportError:
    import sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.visualize_coords_from_template import get_point_color, parse_template  # type: ignore


COLOR_SPECS = {
    "pink": [((145, 60, 80), (179, 255, 255))],
    "purple": [((120, 50, 40), (150, 255, 255))],
    "yellow": [((18, 60, 80), (40, 255, 255))],
    "orange": [((5, 60, 70), (22, 255, 255))],
    "blue": [((90, 60, 50), (130, 255, 255))],
}
COLORFUL_ORDER = ["pink", "yellow", "blue", "orange", "purple"]
CAM_PROJECTION = {
    "cam0": ("X", "Y"),
    "cam1": ("X", "Z"),
    "cam2": ("Y", "Z"),
}


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

    out = {}
    valid_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
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


def _template_points_for_way(template_path: Path, way_name: str) -> List[dict]:
    sections = parse_template(template_path)
    beads_key = f"BEADS_{way_name.upper()}"
    if beads_key not in sections:
        raise ValueError(f"{template_path} does not contain section {beads_key}")

    points: List[dict] = []
    if "CORNERS" in sections:
        coords, labels = sections["CORNERS"]
        for i, (coord, label) in enumerate(zip(coords, labels)):
            points.append(
                {
                    "index": i,
                    "label": f"CORNER: {label}",
                    "kind": "corner",
                    "color": "corner",
                    "X": float(coord[0]),
                    "Y": float(coord[1]),
                    "Z": float(coord[2]),
                }
            )

    bead_coords, bead_labels = sections[beads_key]
    for i, coord in enumerate(bead_coords):
        bead_name = bead_labels[i] if i < len(bead_labels) and bead_labels[i] else f"bead_{i + 1:03d}"
        points.append(
            {
                "index": 4 + i,
                "label": f"BEAD: {bead_name}",
                "kind": "bead",
                "color": get_point_color(4 + i),
                "X": float(coord[0]),
                "Y": float(coord[1]),
                "Z": float(coord[2]),
            }
        )
    return points


def _project_template_points(points: List[dict], cam_name: str) -> Dict[int, np.ndarray]:
    ax0, ax1 = CAM_PROJECTION[cam_name]
    return {
        point["index"]: np.array([point[ax0], point[ax1]], dtype=float)
        for point in points
    }


def _mask_from_ranges(hsv: np.ndarray, ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return mask


def _extract_centers_from_mask(mask: np.ndarray, min_area: float = 18.0, max_area: float = 1500.0) -> List[np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers: List[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) > 5 * max(1, min(w, h)):
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        centers.append(np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=float))
    return centers


def _detect_color_candidates(image_bgr: np.ndarray) -> Dict[str, List[np.ndarray]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return {
        color_name: _extract_centers_from_mask(_mask_from_ranges(hsv, ranges))
        for color_name, ranges in COLOR_SPECS.items()
    }


def _detect_union_chromatic_candidates(image_bgr: np.ndarray) -> List[np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in COLOR_SPECS.values():
        mask |= _mask_from_ranges(hsv, ranges)
    centers = _extract_centers_from_mask(mask)
    deduped: List[np.ndarray] = []
    for center in centers:
        if any(np.linalg.norm(center - existing) < 18.0 for existing in deduped):
            continue
        deduped.append(center)
    return deduped


def _detect_generic_circle_candidates(image_bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=60,
        param2=10,
        minRadius=4,
        maxRadius=22,
    )
    if circles is None:
        return []
    return [np.array([float(x), float(y)], dtype=float) for x, y, _r in circles[0]]


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


def _match_points(
    template_indices: List[int],
    projected_template: Dict[int, np.ndarray],
    detections: List[np.ndarray],
    *,
    max_distance: float,
) -> Tuple[Dict[int, np.ndarray], List[np.ndarray]]:
    if not template_indices or not detections:
        return {}, list(detections)

    tpl = np.vstack([projected_template[idx] for idx in template_indices])
    det = np.vstack(detections)
    cost = np.linalg.norm(tpl[:, None, :] - det[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched: Dict[int, np.ndarray] = {}
    used = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance:
            idx = template_indices[int(r)]
            matched[idx] = det[int(c)]
            used.add(int(c))

    remaining = [detections[i] for i in range(len(detections)) if i not in used]
    return matched, remaining


def _filter_points_in_bbox(points: List[np.ndarray], bbox: tuple[float, float, float, float], margin: float = 60.0) -> List[np.ndarray]:
    x0, y0, x1, y1 = bbox
    return [
        point
        for point in points
        if (x0 - margin) <= point[0] <= (x1 + margin) and (y0 - margin) <= point[1] <= (y1 + margin)
    ]


def _resolve_duplicate_matches(
    matched_points: Dict[int, np.ndarray],
    projected_img: Dict[int, np.ndarray],
    *,
    min_distance: float = 12.0,
) -> Dict[int, np.ndarray]:
    kept: Dict[int, np.ndarray] = {}
    consumed = set()
    items = list(matched_points.items())

    for i, (idx, point) in enumerate(items):
        if idx in consumed:
            continue
        cluster = [(idx, point)]
        consumed.add(idx)
        for j in range(i + 1, len(items)):
            other_idx, other_point = items[j]
            if other_idx in consumed:
                continue
            if np.linalg.norm(point - other_point) < min_distance:
                cluster.append((other_idx, other_point))
                consumed.add(other_idx)

        if len(cluster) == 1:
            kept[idx] = point
            continue

        best_idx, best_point = min(
            cluster,
            key=lambda pair: float(np.linalg.norm(projected_img[pair[0]] - pair[1])),
        )
        kept[best_idx] = best_point

    return kept


def _save_annotation_file(
    output_path: Path,
    points: List[dict],
    matched_points: Dict[int, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Pre-annotation file\n")
        f.write("# Format: index x y X Y Z label\n")
        f.write("# Generated by preannotate_colored_beads.py\n")
        f.write("#\n")
        for point in points:
            idx = point["index"]
            if idx not in matched_points:
                continue
            x, y = matched_points[idx]
            f.write(
                f"{idx} {int(round(x))} {int(round(y))} "
                f"{point['X']:.6f} {point['Y']:.6f} {point['Z']:.6f} {point['label']}\n"
            )


def _save_preview(
    image_bgr: np.ndarray,
    output_path: Path,
    points: List[dict],
    matched_points: Dict[int, np.ndarray],
    unmatched_colored: Dict[str, List[np.ndarray]],
) -> None:
    preview = image_bgr.copy()
    color_map = {
        "pink": (255, 0, 255),
        "purple": (180, 0, 180),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "blue": (255, 0, 0),
        "black": (30, 30, 30),
        "silver": (200, 200, 200),
        "corner": (0, 255, 0),
    }

    for point in points:
        idx = point["index"]
        if idx not in matched_points:
            continue
        x, y = matched_points[idx]
        color = color_map.get(point["color"], (0, 255, 0))
        cv2.circle(preview, (int(round(x)), int(round(y))), 10, color, 2)
        cv2.putText(
            preview,
            str(idx),
            (int(round(x)) + 6, int(round(y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    for color_name, detections in unmatched_colored.items():
        for det in detections:
            cv2.circle(preview, (int(round(det[0])), int(round(det[1]))), 6, color_map.get(color_name, (255, 255, 255)), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def _preannotate_one_camera(
    image_path: Path,
    cam_name: str,
    points: List[dict],
) -> Tuple[Dict[int, np.ndarray], Dict[str, int], Dict[str, List[np.ndarray]]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    projected = _project_template_points(points, cam_name)
    colorful_points = [p for p in points if p["kind"] == "bead" and p["color"] in COLORFUL_ORDER]

    color_candidates = _detect_color_candidates(image_bgr)
    centroid_pairs_src = []
    centroid_pairs_dst = []
    for color_name in COLORFUL_ORDER:
        tpl_idx = [p["index"] for p in colorful_points if p["color"] == color_name]
        det = color_candidates.get(color_name, [])
        if len(tpl_idx) == 0 or len(det) == 0:
            continue
        tpl_centroid = np.mean(np.vstack([projected[idx] for idx in tpl_idx]), axis=0)
        det_centroid = np.mean(np.vstack(det), axis=0)
        centroid_pairs_src.append(tpl_centroid)
        centroid_pairs_dst.append(det_centroid)

    if len(centroid_pairs_src) < 3:
        raise RuntimeError(
            f"Only {len(centroid_pairs_src)} colorful centroid groups were detected in {image_path.name}; "
            "need at least 3 to estimate the initial affine transform."
        )

    affine = _fit_affine(np.vstack(centroid_pairs_src), np.vstack(centroid_pairs_dst))
    projected_img = {
        idx: _apply_affine(affine, np.array([coord]))[0]
        for idx, coord in projected.items()
    }

    matched_points: Dict[int, np.ndarray] = {}
    unmatched_colored: Dict[str, List[np.ndarray]] = {}

    colorful_match_pairs_src = []
    colorful_match_pairs_dst = []
    for color_name in COLORFUL_ORDER:
        tpl_idx = [p["index"] for p in colorful_points if p["color"] == color_name]
        matched, remaining = _match_points(
            tpl_idx,
            projected_img,
            color_candidates.get(color_name, []),
            max_distance=80.0,
        )
        matched_points.update(matched)
        unmatched_colored[color_name] = remaining
        for idx, det in matched.items():
            colorful_match_pairs_src.append(projected[idx])
            colorful_match_pairs_dst.append(det)

    if len(colorful_match_pairs_src) >= 3:
        affine = _fit_affine(np.vstack(colorful_match_pairs_src), np.vstack(colorful_match_pairs_dst))
        projected_img = {
            idx: _apply_affine(affine, np.array([coord]))[0]
            for idx, coord in projected.items()
        }

    all_projected = np.vstack(list(projected_img.values()))
    bbox = (
        float(np.min(all_projected[:, 0])),
        float(np.min(all_projected[:, 1])),
        float(np.max(all_projected[:, 0])),
        float(np.max(all_projected[:, 1])),
    )

    chromatic_candidates = _detect_union_chromatic_candidates(image_bgr)
    chromatic_candidates = _filter_points_in_bbox(chromatic_candidates, bbox, margin=80.0)

    colorful_all_idx = [p["index"] for p in colorful_points]
    pooled_matches, _unused = _match_points(
        colorful_all_idx,
        projected_img,
        chromatic_candidates,
        max_distance=55.0,
    )
    for idx, det in pooled_matches.items():
        matched_points[idx] = det

    generic_candidates = _detect_generic_circle_candidates(image_bgr)
    generic_remaining = _filter_points_in_bbox(list(generic_candidates), bbox, margin=60.0)

    # Remove already-used generic candidates if they overlap colorful matches.
    pruned_remaining = []
    for cand in generic_remaining:
        if any(np.linalg.norm(cand - matched) < 10.0 for matched in matched_points.values()):
            continue
        pruned_remaining.append(cand)
    generic_remaining = pruned_remaining

    for class_name in ("corner", "black", "silver"):
        tpl_idx = [p["index"] for p in points if p["color"] == class_name or p["kind"] == class_name]
        if class_name == "corner":
            tpl_idx = [p["index"] for p in points if p["kind"] == "corner"]
        tpl_idx = [idx for idx in tpl_idx if idx not in matched_points]
        matched, generic_remaining = _match_points(
            tpl_idx,
            projected_img,
            generic_remaining,
            max_distance=35.0,
        )
        matched_points.update(matched)

    matched_points = _resolve_duplicate_matches(matched_points, projected_img)

    summary = {
        "matched_total": len(matched_points),
        "matched_beads": sum(1 for p in points if p["index"] in matched_points and p["kind"] == "bead"),
        "matched_corners": sum(1 for p in points if p["index"] in matched_points and p["kind"] == "corner"),
    }
    return matched_points, summary, unmatched_colored


def run_preannotation(raw_root: str, way_name: str, output_root: str) -> Dict:
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    template_path = _find_template_file(raw_root_path, way_name)
    image_paths = _find_way_images(raw_root_path, way_name)
    points = _template_points_for_way(template_path, way_name)

    run_summary = {
        "raw_root": str(raw_root_path),
        "way_name": way_name,
        "template_path": str(template_path),
        "outputs": {},
    }

    for cam_name, image_path in image_paths.items():
        matched_points, summary, unmatched_colored = _preannotate_one_camera(image_path, cam_name, points)
        txt_path = output_root_path / f"cali_{way_name}_{cam_name}_image_coords.txt"
        preview_path = output_root_path / f"cali_{way_name}_{cam_name}_preannotated_preview.png"
        _save_annotation_file(txt_path, points, matched_points)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        _save_preview(image_bgr, preview_path, points, matched_points, unmatched_colored)
        run_summary["outputs"][cam_name] = {
            "image_path": str(image_path),
            "annotation_path": str(txt_path),
            "preview_path": str(preview_path),
            **summary,
        }

    summary_path = output_root_path / f"cali_{way_name}_preannotation_summary.json"
    output_root_path.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    run_summary["summary_path"] = str(summary_path)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-annotate calibration beads from raw images using bead colors")
    parser.add_argument("--folder", required=True, help="Calibration raw root folder containing templates and way*_pic directories")
    parser.add_argument("--way_name", required=True, help="Way name, for example way1")
    parser.add_argument("--output", required=True, help="Output folder for draft *_image_coords.txt files and previews")
    args = parser.parse_args()

    result = run_preannotation(args.folder, args.way_name.lower(), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
