"""
Template parsing helpers used by the calibration annotation and QC scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

N_CORNERS = 4
BEADS_PER_COLUMN = 7
BEAD_COLUMN_COLOR_ORDER = ["pink", "yellow", "blue", "black", "orange", "purple", "silver"]
SECTION_NAME_RE = re.compile(r"^[A-Z0-9_\-\[\]\s:]+$")
BEAD_AXIS_VALUES = np.array([27.5, 50.0, 72.5, 95.0, 117.5, 140.0, 162.5], dtype=float)


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _normalize_section_name(line: str) -> str:
    line = line.strip()
    bracket_match = re.search(r"\[([^\]]+)\]", line)
    if bracket_match:
        line = bracket_match.group(1)
    else:
        line = line.strip("[]")
    line = line.strip().replace("-", "_").replace(" ", "_")
    line = line.rstrip(":")
    return line.upper()


def _parse_point_tokens(tokens: List[str]) -> tuple[np.ndarray, str] | None:
    if len(tokens) < 3:
        return None

    if len(tokens) >= 4 and _is_number(tokens[1]) and _is_number(tokens[2]) and _is_number(tokens[3]):
        coords = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=float)
        label = " ".join(tokens[4:]).strip() or str(tokens[0])
        return coords, label

    if _is_number(tokens[0]) and _is_number(tokens[1]) and _is_number(tokens[2]):
        coords = np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])], dtype=float)
        label = " ".join(tokens[3:]).strip()
        return coords, label

    if _is_number(tokens[-3]) and _is_number(tokens[-2]) and _is_number(tokens[-1]):
        coords = np.array([float(tokens[-3]), float(tokens[-2]), float(tokens[-1])], dtype=float)
        label = " ".join(tokens[:-3]).strip()
        return coords, label

    numeric = [_is_number(tok) for tok in tokens]
    if sum(numeric) >= 3:
        idx = [i for i, is_num in enumerate(numeric) if is_num][:3]
        coords = np.array([float(tokens[i]) for i in idx], dtype=float)
        label_tokens = [tok for i, tok in enumerate(tokens) if i not in idx]
        label = " ".join(label_tokens).strip()
        return coords, label

    return None


def get_point_color(i: int) -> str:
    """Return the standard calibration color for one point index."""
    if i < N_CORNERS:
        return "green"
    bead_idx = i - N_CORNERS
    col_idx = bead_idx // BEADS_PER_COLUMN
    if col_idx >= len(BEAD_COLUMN_COLOR_ORDER):
        return BEAD_COLUMN_COLOR_ORDER[-1]
    return BEAD_COLUMN_COLOR_ORDER[col_idx]


def _correct_way1_bead_coords() -> np.ndarray:
    coords = []
    for color_idx, x in enumerate(BEAD_AXIS_VALUES):
        for bead_idx in range(BEADS_PER_COLUMN):
            y = BEAD_AXIS_VALUES[(bead_idx + color_idx) % BEADS_PER_COLUMN]
            z = BEAD_AXIS_VALUES[BEADS_PER_COLUMN - 1 - bead_idx]
            coords.append([x, y, z])
    return np.asarray(coords, dtype=float)


def _rotate_xy(coords: np.ndarray, turns_clockwise: int) -> np.ndarray:
    out = np.asarray(coords, dtype=float).copy()
    turns_clockwise = turns_clockwise % 4
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    if turns_clockwise == 1:
        out[:, 0] = 190.0 - y
        out[:, 1] = x
    elif turns_clockwise == 2:
        out[:, 0] = 190.0 - x
        out[:, 1] = 190.0 - y
    elif turns_clockwise == 3:
        out[:, 0] = y
        out[:, 1] = 190.0 - x
    return out


def corrected_bead_coords_for_way(way_name: str) -> np.ndarray:
    """
    Return the corrected physical bead coordinates for way1-way4.

    The raw template files encode the bead order used during manual marking,
    but the physical bead ID to XYZ mapping is corrected here so calibration
    code consistently uses the intended tank/world coordinates.
    """
    match = re.search(r"WAY(\d+)", way_name.upper())
    way_num = int(match.group(1)) if match else 1
    return _rotate_xy(_correct_way1_bead_coords(), way_num - 1)


# Canonical 4-corner XYZs in the bead-fit coordinate system.
# Frame A is the way1 / way3 bead frame; Frame B (way2 / way4) is Frame A
# rotated 180 degrees about the cube center (95, 95). The ordering is the
# canonical CW order (rank 0..3) that the reassign rule writes after
# pixel-quadrant sorting -- rank 0 lives at cam0 top-left / cam1 top-right /
# cam2 bottom-left.
#
# Physical geometry: the calibration object has a 175x175 mm inner square
# with corners at x,y in {7.5, 182.5} and outer dimensions 190x190 mm. The
# base is 4 mm thick and the top is at z = 194 (4 mm base + 190 mm inner
# height). Top corners live at z=194 and bottom corners at z=4.
CORNER_TOP_Z = 194.0
CORNER_BOTTOM_Z = 4.0

CANONICAL_TOP_CORNERS_FRAME_A = np.array(
    [
        [182.5,   7.5, CORNER_TOP_Z],
        [  7.5,   7.5, CORNER_TOP_Z],
        [  7.5, 182.5, CORNER_TOP_Z],
        [182.5, 182.5, CORNER_TOP_Z],
    ],
    dtype=float,
)

CANONICAL_BOTTOM_CORNERS_FRAME_A = np.array(
    [
        [182.5,   7.5, CORNER_BOTTOM_Z],
        [  7.5,   7.5, CORNER_BOTTOM_Z],
        [  7.5, 182.5, CORNER_BOTTOM_Z],
        [182.5, 182.5, CORNER_BOTTOM_Z],
    ],
    dtype=float,
)


def _rotate_corners_180(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[:, 0] = 190.0 - out[:, 0]
    out[:, 1] = 190.0 - out[:, 1]
    return out


def corrected_corner_coords_for_way(way_name: str) -> np.ndarray:
    """Return canonical top-corner XYZs (4 rows) for the given way."""
    match = re.search(r"WAY(\d+)", way_name.upper())
    way_num = int(match.group(1)) if match else 1
    coords = CANONICAL_TOP_CORNERS_FRAME_A.copy()
    if way_num % 2 == 0:
        coords = _rotate_corners_180(coords)
    return coords


def corrected_bottom_corner_coords_for_way(way_name: str) -> np.ndarray:
    """Return canonical bottom-corner XYZs (4 rows) for the given way."""
    match = re.search(r"WAY(\d+)", way_name.upper())
    way_num = int(match.group(1)) if match else 1
    coords = CANONICAL_BOTTOM_CORNERS_FRAME_A.copy()
    if way_num % 2 == 0:
        coords = _rotate_corners_180(coords)
    return coords


def _apply_corrected_bead_coords(
    sections: Dict[str, Tuple[np.ndarray, List[str]]]
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    corrected = dict(sections)

    # Infer the way name from whichever BEADS_WAY* section is present, so we
    # can also patch the CORNERS section to the bead-fit canonical values.
    way_name = None
    for section in sections:
        if section.startswith("BEADS_WAY"):
            way_name = section.replace("BEADS_", "").lower()
            break

    for section, (coords, labels) in sections.items():
        if not section.startswith("BEADS_WAY"):
            continue
        if coords.shape[0] != BEADS_PER_COLUMN * len(BEAD_COLUMN_COLOR_ORDER):
            continue
        corrected[section] = (corrected_bead_coords_for_way(section), labels)

    if way_name is not None and "CORNERS" in sections:
        coords, labels = sections["CORNERS"]
        if coords.shape[0] == 4:
            corrected["CORNERS"] = (corrected_corner_coords_for_way(way_name), labels)

    return corrected


def parse_template(path: str | Path) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Parse a calibration template file into named sections.

    Supported line styles are intentionally flexible:
    - section headers such as `CORNERS`, `[BEADS_WAY1]`
    - data lines like `X Y Z label`
    - data lines like `index X Y Z label`
    - data lines like `label X Y Z`
    """
    path = Path(path)
    sections: Dict[str, List[tuple[np.ndarray, str]]] = {}
    current_section = "POINTS"

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            candidate_header = _normalize_section_name(line)
            tokens = line.replace(",", " ").split()
            parsed = _parse_point_tokens(tokens)

            if parsed is None and SECTION_NAME_RE.match(candidate_header):
                current_section = candidate_header
                sections.setdefault(current_section, [])
                continue

            if parsed is None:
                continue

            coords, label = parsed
            sections.setdefault(current_section, [])
            if not label:
                label = f"{current_section.lower()}_{len(sections[current_section]) + 1:03d}"
            sections[current_section].append((coords, label))

    parsed_sections = {
        section: (
            np.vstack([coords for coords, _ in items]) if items else np.empty((0, 3), dtype=float),
            [label for _, label in items],
        )
        for section, items in sections.items()
    }
    return _apply_corrected_bead_coords(parsed_sections)


def get_colors_from_template(path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load all template points in the same order used by the annotation tool and
    return `(coords, colors)`.
    """
    sections = parse_template(path)
    coords_list: List[np.ndarray] = []

    if "CORNERS" in sections:
        coords_list.extend(list(sections["CORNERS"][0]))

    for key in sorted(sections.keys()):
        if key.startswith("BEADS_WAY"):
            coords_list.extend(list(sections[key][0]))

    if not coords_list and "POINTS" in sections:
        coords_list.extend(list(sections["POINTS"][0]))

    coords = np.vstack(coords_list) if coords_list else np.empty((0, 3), dtype=float)
    colors = [get_point_color(i) for i in range(len(coords))]
    return coords, colors


__all__ = ["parse_template", "get_colors_from_template", "get_point_color", "corrected_bead_coords_for_way"]
