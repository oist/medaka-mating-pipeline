"""
Normalize calibration annotation XYZ columns from the corrected template.

This fixes stale or typo-prone world-coordinate values in `*_image_coords.txt`
files while preserving the manually clicked 2D image coordinates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .lib.visualize_coords_from_template import parse_template
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.visualize_coords_from_template import parse_template  # type: ignore


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


def _xyz_by_index(raw_root: Path, way_name: str) -> Dict[int, np.ndarray]:
    sections = parse_template(_find_template_file(raw_root, way_name))
    xyz: Dict[int, np.ndarray] = {}

    if "CORNERS" in sections:
        corner_coords, _labels = sections["CORNERS"]
        for i, coord in enumerate(corner_coords):
            xyz[i] = np.asarray(coord, dtype=float)

    beads_key = f"BEADS_{way_name.upper()}"
    if beads_key in sections:
        bead_coords, _labels = sections[beads_key]
        for i, coord in enumerate(bead_coords):
            xyz[4 + i] = np.asarray(coord, dtype=float)

    if "CORNERS" in sections:
        start_idx = max(xyz.keys(), default=52) + 1
        corner_coords, _labels = sections["CORNERS"]
        for i, coord in enumerate(corner_coords):
            bottom = np.asarray([coord[0], coord[1], 4.0], dtype=float)
            xyz[start_idx + i] = bottom
    return xyz


def normalize_annotation_file(path: Path, xyz_lookup: Dict[int, np.ndarray]) -> int:
    lines_out = []
    changed = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines_out.append(line)
                continue

            parts = stripped.split()
            if len(parts) < 6:
                lines_out.append(line)
                continue
            idx = int(parts[0])
            if idx not in xyz_lookup:
                lines_out.append(line)
                continue

            old_xyz = tuple(float(v) for v in parts[3:6])
            new_xyz = tuple(float(v) for v in xyz_lookup[idx])
            if not np.allclose(old_xyz, new_xyz):
                changed += 1
            parts[3:6] = [f"{v:.6f}" for v in new_xyz]
            lines_out.append(" ".join(parts))

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines_out) + "\n")
    return changed


def normalize_way(raw_root: str, annotation_dir: str, way_name: str) -> Dict[str, int]:
    raw_root_path = Path(raw_root)
    annotation_root = Path(annotation_dir)
    xyz_lookup = _xyz_by_index(raw_root_path, way_name)
    result = {}
    for cam_name in ("cam0", "cam1", "cam2"):
        path = annotation_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        result[cam_name] = normalize_annotation_file(path, xyz_lookup)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize annotation XYZ columns from corrected template coordinates")
    parser.add_argument("--folder", required=True, help="Calibration raw root containing template files")
    parser.add_argument("--way_name", required=True, help="Way name, for example way1")
    parser.add_argument("--annotation_dir", required=True, help="Folder containing cali_{way}_cam*_image_coords.txt files")
    args = parser.parse_args()

    result = normalize_way(args.folder, args.annotation_dir, args.way_name.lower())
    for cam_name, n_changed in result.items():
        print(f"{cam_name}: normalized {n_changed} row(s)")


if __name__ == "__main__":
    main()
