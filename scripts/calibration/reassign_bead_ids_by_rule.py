"""
Reassign bead IDs by camera/way ordering rules while preserving clicked pixels.

Manual annotation order is ignored. Within each color group, clicked 2D
positions are sorted according to view-specific rules and assigned to the
expected bead IDs/XYZ coordinates from the corrected template mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from .lib.visualize_coords_from_template import get_point_color, parse_template
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.visualize_coords_from_template import get_point_color, parse_template  # type: ignore


COLOR_GROUPS = {
    "pink": list(range(4, 11)),
    "yellow": list(range(11, 18)),
    "blue": list(range(18, 25)),
    "black": list(range(25, 32)),
    "orange": list(range(32, 39)),
    "purple": list(range(39, 46)),
    "silver": list(range(46, 53)),
}
WAY1_CAM0_ORDER = {
    "pink": [4, 5, 6, 7, 8, 9, 10],
    "yellow": [17, 11, 12, 13, 14, 15, 16],
    "blue": [23, 24, 18, 19, 20, 21, 22],
    "black": [29, 30, 31, 25, 26, 27, 28],
    "orange": [35, 36, 37, 38, 32, 33, 34],
    "purple": [41, 42, 43, 44, 45, 39, 40],
    "silver": [47, 48, 49, 50, 51, 52, 46],
}

# Canonical quadrant -> rank within the 4-corner group, per camera.
# Derived from way1 annotation where the 4 clicked corners with indices 0..3
# land in specific image quadrants of each camera. Reusing this rank map lets
# us reassign the 4 corners of any way onto the expected cube-frame indices
# purely from pixel layout, independent of the user's clicking order.
#
# Quadrants are computed relative to the centroid of the 4 clicked pixels:
#   TL = x<=cx and y<=cy
#   TR = x> cx and y<=cy
#   BR = x> cx and y> cy
#   BL = x<=cx and y> cy
CAM_CORNER_QUADRANT_RANK = {
    "cam0": {"TL": 0, "TR": 1, "BR": 2, "BL": 3},
    "cam1": {"TR": 0, "BL": 1, "TL": 2, "BR": 3},
    "cam2": {"BL": 0, "TR": 1, "TL": 2, "BR": 3},
}

# Canonical corner XYZ by rank (independent of the per-way template, which
# historically had the X axis inverted relative to the bead-fit coordinate
# system and is being replaced here). Frame A is the way1/way3 bead frame;
# frame B (way2/way4) is Frame A rotated 180 degrees about (95, 95).
#
# The rank corresponds to the `CAM_CORNER_QUADRANT_RANK` slot -- e.g. rank 0
# is the physical corner visible at cam0-TL / cam1-TR / cam2-BL, which lives
# at room XYZ (182.5, 7.5, 194) for the top face.
#
# Physical geometry: the calibration object has a 175x175 mm inner square
# with corners at x,y in {7.5, 182.5}; the base is 4 mm thick and the inner
# height is 190 mm, so the top face is at z=194 and the bottom at z=4.
CORNER_TOP_Z = 194.0
CORNER_BOTTOM_Z = 4.0

CANONICAL_TOP_CORNERS_FRAME_A: Dict[int, tuple] = {
    0: (182.5,   7.5, CORNER_TOP_Z),
    1: (  7.5,   7.5, CORNER_TOP_Z),
    2: (  7.5, 182.5, CORNER_TOP_Z),
    3: (182.5, 182.5, CORNER_TOP_Z),
}
CANONICAL_BOTTOM_CORNERS_FRAME_A: Dict[int, tuple] = {
    rank: (xyz[0], xyz[1], CORNER_BOTTOM_Z)
    for rank, xyz in CANONICAL_TOP_CORNERS_FRAME_A.items()
}


def _canonical_corner_xyz(rank: int, way_name: str, is_bottom: bool) -> tuple:
    """
    Return the canonical corner XYZ for a given rank and way.

    Ways 1 and 3 share Frame A; ways 2 and 4 share Frame B which is Frame A
    rotated 180 degrees about (95, 95). Top (Z=194) and bottom (Z=4)
    corners share the same (X, Y) pattern.
    """
    base = CANONICAL_BOTTOM_CORNERS_FRAME_A if is_bottom else CANONICAL_TOP_CORNERS_FRAME_A
    x, y, z = base[rank]
    way_num = int(way_name.replace("way", ""))
    if way_num % 2 == 0:
        x = 190.0 - x
        y = 190.0 - y
    return (x, y, z)


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
    out: Dict[int, np.ndarray] = {}

    if "CORNERS" in sections:
        coords, _labels = sections["CORNERS"]
        for i, coord in enumerate(coords):
            out[i] = np.asarray(coord, dtype=float)

    beads_key = f"BEADS_{way_name.upper()}"
    coords, _labels = sections[beads_key]
    for i, coord in enumerate(coords):
        out[4 + i] = np.asarray(coord, dtype=float)

    # Synthesize bottom-corner slots at z=CORNER_BOTTOM_Z using the top corner
    # (X, Y). The rule reassign overwrites these with canonical values anyway,
    # but keeping them present lets `reassign_rows` discover the bottom index
    # range and lets downstream scripts see the bottom-corner IDs.
    if "CORNERS" in sections:
        start = max(out) + 1
        coords, _labels = sections["CORNERS"]
        for i, coord in enumerate(coords):
            out[start + i] = np.asarray(
                [coord[0], coord[1], CORNER_BOTTOM_Z], dtype=float
            )
    return out


def _load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
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


def _bead_label(index: int) -> str:
    return f"BEAD: beads_way1_{index - 3:03d}"


def _color_for_index(index: int) -> str:
    return get_point_color(index)


def _order_for_camera(way_name: str, cam_name: str, color: str) -> List[int]:
    # The bead object is rotated 90 degrees between ways, but the physical
    # bead IDs do not shift. What changes is how the cam0 image is laid out:
    # the way1 top-to-bottom sequence becomes right-to-left in way2,
    # bottom-to-top in way3, and left-to-right in way4. The target ID list
    # itself is identical to way1.
    if cam_name == "cam0":
        return WAY1_CAM0_ORDER[color]
    return COLOR_GROUPS[color]


def _cam0_sort_key(way_name: str):
    # Returns a key function that walks the cam0 image in the direction
    # corresponding to way1 top-to-bottom, after the per-way 90 degree
    # rotation. Secondary key is the perpendicular axis, also rotated.
    way_num = int(way_name.replace("way", ""))
    if way_num == 1:
        # top -> bottom, then left -> right
        return lambda row: (row["y"], row["x"])
    if way_num == 2:
        # right -> left, then top -> bottom
        return lambda row: (-row["x"], row["y"])
    if way_num == 3:
        # bottom -> top, then right -> left
        return lambda row: (-row["y"], -row["x"])
    if way_num == 4:
        # left -> right, then bottom -> top
        return lambda row: (row["x"], -row["y"])
    raise ValueError(f"Unsupported way: {way_name}")


def _sort_rows_for_color(rows: List[dict], cam_name: str, way_name: str) -> List[dict]:
    if cam_name == "cam0":
        return sorted(rows, key=_cam0_sort_key(way_name))
    return sorted(rows, key=lambda row: (row["y"], row["x"]))


def _quadrant_of(row: dict, cx: float, cy: float) -> str:
    x, y = row["x"], row["y"]
    if x <= cx and y <= cy:
        return "TL"
    if x > cx and y <= cy:
        return "TR"
    if x > cx and y > cy:
        return "BR"
    return "BL"


def _corner_label(offset: int, is_bottom: bool) -> str:
    if is_bottom:
        return f"CORNER_BOTTOM: corner_{offset + 1}_bottom"
    if offset == 0:
        return "CORNER: corner_1(black dots)"
    return f"CORNER: corner_{offset + 1}"


def _reassign_corner_group(
    rows: List[dict],
    cam_name: str,
    way_name: str,
    base_index: int,
    is_bottom: bool,
) -> List[dict]:
    """
    Reassign a 4-corner group by pixel quadrant.

    The 4 clicked pixels are classified into TL/TR/BR/BL relative to their
    centroid, then sorted by the per-camera canonical rank so the output order
    is independent of the user's clicking order. Each sorted slot is written
    as `base_index + slot` using the canonical per-way XYZ (not the raw
    template value, which is inconsistent with the bead-fit coordinate system).
    """
    if len(rows) != 4:
        return list(rows)
    if cam_name not in CAM_CORNER_QUADRANT_RANK:
        return list(rows)

    cx = sum(r["x"] for r in rows) / 4.0
    cy = sum(r["y"] for r in rows) / 4.0
    rank_map = CAM_CORNER_QUADRANT_RANK[cam_name]

    quadrants = [_quadrant_of(r, cx, cy) for r in rows]
    if len(set(quadrants)) != 4:
        # Degenerate layout (two corners in the same quadrant); leave as-is.
        return list(rows)

    ranked = sorted(zip(rows, quadrants), key=lambda pair: rank_map[pair[1]])

    out: List[dict] = []
    for offset, (row, _quad) in enumerate(ranked):
        target_index = base_index + offset
        xyz = _canonical_corner_xyz(offset, way_name, is_bottom)
        out.append(
            {
                **row,
                "index": int(target_index),
                "X": float(xyz[0]),
                "Y": float(xyz[1]),
                "Z": float(xyz[2]),
                "label": _corner_label(offset, is_bottom),
            }
        )
    return out


def reassign_rows(rows: List[dict], xyz_by_index: Dict[int, np.ndarray], way_name: str, cam_name: str) -> List[dict]:
    beads = [row for row in rows if row["label"].startswith("BEAD:")]
    top_corners = [row for row in rows if row["label"].startswith("CORNER:")]
    bot_corners = [row for row in rows if row["label"].startswith("CORNER_BOTTOM:")]
    other = [
        row
        for row in rows
        if not row["label"].startswith(("BEAD:", "CORNER:", "CORNER_BOTTOM:"))
    ]

    by_color = {color: [] for color in COLOR_GROUPS}
    for row in beads:
        by_color[_color_for_index(row["index"])].append(row)

    reassigned: List[dict] = []
    for color in COLOR_GROUPS:
        sorted_rows = _sort_rows_for_color(by_color[color], cam_name, way_name)
        target_ids = _order_for_camera(way_name, cam_name, color)
        for row, target_id in zip(sorted_rows, target_ids):
            xyz = xyz_by_index[target_id]
            reassigned.append(
                {
                    **row,
                    "index": target_id,
                    "X": float(xyz[0]),
                    "Y": float(xyz[1]),
                    "Z": float(xyz[2]),
                    "label": _bead_label(target_id),
                }
            )

    reassigned.extend(
        _reassign_corner_group(
            top_corners,
            cam_name,
            way_name,
            base_index=0,
            is_bottom=False,
        )
    )

    bottom_template_indices = sorted(
        i
        for i, xyz in xyz_by_index.items()
        if i >= 4 and float(xyz[2]) == CORNER_BOTTOM_Z
    )
    if len(bottom_template_indices) >= 4:
        bottom_base = bottom_template_indices[0]
        reassigned.extend(
            _reassign_corner_group(
                bot_corners,
                cam_name,
                way_name,
                base_index=bottom_base,
                is_bottom=True,
            )
        )
    else:
        reassigned.extend(bot_corners)

    for row in other:
        if row["index"] in xyz_by_index:
            xyz = xyz_by_index[row["index"]]
            row = {**row, "X": float(xyz[0]), "Y": float(xyz[1]), "Z": float(xyz[2])}
        reassigned.append(row)

    reassigned.sort(key=lambda row: row["index"])
    return reassigned


def _save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Rule-reassigned annotation file\n")
        f.write("# Format: index x y X Y Z label\n")
        f.write("#\n")
        for row in rows:
            f.write(
                f"{row['index']} {int(round(row['x']))} {int(round(row['y']))} "
                f"{row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f} {row['label']}\n"
            )


def reassign_way(raw_root: str, annotation_dir: str, output_dir: str, way_name: str) -> Dict[str, dict]:
    raw_root_path = Path(raw_root)
    annotation_root = Path(annotation_dir)
    output_root = Path(output_dir)
    xyz = _xyz_by_index(raw_root_path, way_name)
    result = {}
    for cam_name in ("cam0", "cam1", "cam2"):
        src = annotation_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        dst = output_root / f"cali_{way_name}_{cam_name}_image_coords.txt"
        rows = _load_rows(src)
        reassigned = reassign_rows(rows, xyz, way_name, cam_name)
        _save_rows(dst, reassigned)
        changed = sum(1 for old, new in zip(sorted(rows, key=lambda row: row["index"]), reassigned) if old["index"] != new["index"])
        result[cam_name] = {"input": str(src), "output": str(dst), "n_points": len(reassigned), "changed_indices": changed}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassign bead IDs by camera/way ordering rules")
    parser.add_argument("--folder", required=True, help="Calibration raw root folder containing templates")
    parser.add_argument("--way_name", required=True, help="Way name, for example way1")
    parser.add_argument("--annotation_dir", required=True, help="Input annotation folder")
    parser.add_argument("--output_dir", required=True, help="Output annotation folder")
    args = parser.parse_args()

    result = reassign_way(args.folder, args.annotation_dir, args.output_dir, args.way_name.lower())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
