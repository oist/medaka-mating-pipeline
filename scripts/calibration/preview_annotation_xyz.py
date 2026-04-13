"""
Render calibration annotation previews with assigned world coordinates.

The output overlays each annotated 2D point on its raw camera image with the
annotation index and `(X, Y, Z)` from the corresponding `*_image_coords.txt`
file. This is intended as a quick visual sanity check for bead/world-coordinate
assignment mistakes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import cv2

try:
    from .lib.visualize_coords_from_template import get_point_color, parse_template
except ImportError:
    import sys

    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    from lib.visualize_coords_from_template import get_point_color, parse_template  # type: ignore


BEAD_COLOR_BGR = {
    "pink": (255, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "black": (30, 30, 30),
    "orange": (0, 165, 255),
    "purple": (160, 0, 160),
    "silver": (210, 210, 210),
}


def _find_way_images(raw_root: Path, way_name: str) -> Dict[str, Path]:
    way_dirs = sorted(raw_root.glob(f"*{way_name}*_pic"))
    if not way_dirs:
        raise FileNotFoundError(f"Could not find image folder for {way_name} under {raw_root}")
    way_dir = way_dirs[0]
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    images: Dict[str, Path] = {}
    for path in sorted(way_dir.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        name = path.name.lower()
        if "cam0" in name or "top" in name:
            images["cam0"] = path
        elif "cam1" in name or "side1" in name:
            images["cam1"] = path
        elif "cam2" in name or "side2" in name:
            images["cam2"] = path
    missing = [cam for cam in ("cam0", "cam1", "cam2") if cam not in images]
    if missing:
        raise FileNotFoundError(f"Missing image(s) for {missing} in {way_dir}")
    return images


def _find_template_file(raw_root: Path, way_name: str) -> Path:
    for candidate in [
        raw_root / f"cali_coords_template_{way_name}.txt",
        raw_root / f"Calibration_Coords_Template_{way_name.upper()}.txt",
    ]:
        if candidate.exists():
            return candidate
    matches = sorted(raw_root.glob(f"*template*{way_name}*.txt"))
    if not matches:
        raise FileNotFoundError(f"Template file not found for {way_name} in {raw_root}")
    return matches[0]


def _template_xyz_by_index(raw_root: Path, way_name: str) -> Dict[int, tuple[float, float, float]]:
    sections = parse_template(_find_template_file(raw_root, way_name))
    out: Dict[int, tuple[float, float, float]] = {}
    if "CORNERS" in sections:
        coords, _labels = sections["CORNERS"]
        for i, coord in enumerate(coords):
            out[i] = (float(coord[0]), float(coord[1]), float(coord[2]))
    beads_key = f"BEADS_{way_name.upper()}"
    if beads_key in sections:
        coords, _labels = sections[beads_key]
        for i, coord in enumerate(coords):
            out[4 + i] = (float(coord[0]), float(coord[1]), float(coord[2]))
    return out


def _load_annotation_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "X": float(parts[3]),
                    "Y": float(parts[4]),
                    "Z": float(parts[5]),
                    "label": " ".join(parts[6:]),
                }
            )
    return rows


def _row_color(row: dict) -> tuple[int, int, int]:
    label = row["label"]
    if label.startswith("CORNER_BOTTOM:"):
        return (0, 200, 255)
    if label.startswith("CORNER:"):
        return (0, 255, 0)
    if label.startswith("BEAD:"):
        return BEAD_COLOR_BGR.get(get_point_color(row["index"]), (255, 255, 255))
    return (255, 255, 255)


def _draw_text_with_outline(image, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.42) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_preview(
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
    *,
    xyz_by_index: Dict[int, tuple[float, float, float]] | None = None,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rows = _load_annotation_rows(annotation_path)

    for row in rows:
        if xyz_by_index is not None and row["index"] in xyz_by_index:
            row["X"], row["Y"], row["Z"] = xyz_by_index[row["index"]]
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        color = _row_color(row)
        cv2.circle(image, (x, y), 10, color, 2)
        text = f"{row['index']}:({row['X']:.1f},{row['Y']:.1f},{row['Z']:.1f})"
        _draw_text_with_outline(image, text, (x + 8, y - 8), color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return {
        "image_path": str(image_path),
        "annotation_path": str(annotation_path),
        "output_path": str(output_path),
        "n_points": len(rows),
    }


def render_way(raw_root: str, annotation_dir: str, way_name: str, output_dir: str) -> Dict[str, dict]:
    raw_root_path = Path(raw_root)
    annotation_root = Path(annotation_dir)
    output_root = Path(output_dir)
    images = _find_way_images(raw_root_path, way_name)
    xyz_by_index = _template_xyz_by_index(raw_root_path, way_name)
    outputs: Dict[str, dict] = {}
    for cam_name in ("cam0", "cam1", "cam2"):
        annotation_path = annotation_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        output_path = output_root / f"cali_{way_name}_{cam_name}_xyz_preview.png"
        outputs[cam_name] = render_preview(
            images[cam_name],
            annotation_path,
            output_path,
            xyz_by_index=xyz_by_index,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render calibration previews labeled with assigned XYZ coordinates")
    parser.add_argument("--folder", required=True, help="Calibration raw root folder containing way*_pic directories")
    parser.add_argument("--way_name", required=True, help="Way name, for example way4")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing cali_{way}_cam*_image_coords.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory for labeled preview PNGs")
    args = parser.parse_args()

    outputs = render_way(args.folder, args.annotation_dir, args.way_name.lower(), args.output_dir)
    for cam_name, info in outputs.items():
        print(f"{cam_name}: wrote {info['output_path']} ({info['n_points']} point(s))")


if __name__ == "__main__":
    main()
