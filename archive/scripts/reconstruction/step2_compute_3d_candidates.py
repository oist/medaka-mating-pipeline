"""
Step 2: compute 3D candidates using the calibration model.

For each frame, enumerate all cam0/cam1/cam2 combinations, compute 3D
candidates with `imCoords_to_XYZ`, and score them with registration error.

Registration cost is used in Step 3 to choose the best correspondence.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .calibration import Calibration


def _triplets_with_filled_view(
    cal: Calibration,
    cam0: list,
    cam1: list,
    cam2: list,
    missing_view: str,
) -> List[tuple]:
    """
    If one view is completely missing, predict it from the other two and return
    `(i0, i1, i2, p0, p1, p2)` tuples.
    """
    out: List[tuple] = []
    if missing_view == "cam2":
        for i0 in range(len(cam0)):
            for i1 in range(len(cam1)):
                p0 = np.array(cam0[i0], dtype=float)
                p1 = np.array(cam1[i1], dtype=float)
                p2 = cal.compute_YZ_imcoords_from_XZ_XY(p1, p0)
                out.append((i0, i1, 0, p0, p1, p2))
    elif missing_view == "cam1":
        for i0 in range(len(cam0)):
            for i2 in range(len(cam2)):
                p0 = np.array(cam0[i0], dtype=float)
                p2 = np.array(cam2[i2], dtype=float)
                p1 = cal.compute_XZ_imcoords_from_XY_YZ(p0, p2)
                out.append((i0, 0, i2, p0, p1, p2))
    else:
        for i1 in range(len(cam1)):
            for i2 in range(len(cam2)):
                p1 = np.array(cam1[i1], dtype=float)
                p2 = np.array(cam2[i2], dtype=float)
                p0 = cal.compute_XY_imcoords_from_XZ_YZ(p1, p2)
                out.append((0, i1, i2, p0, p1, p2))
    return out


def compute_candidates_for_frame(
    cal: Calibration,
    points: Dict[str, List[Tuple[float, float]]],
    xyz_bounds: Tuple[float, float] | None = None,
    fill_missing_view: bool = True,
) -> List[Dict]:
    """
    Compute all 3D candidates and registration costs for one frame.

    If one view is missing and `fill_missing_view=True`, infer the missing
    2D points from the other two views first.
    """
    cam0 = list(points.get("cam0", []))
    cam1 = list(points.get("cam1", []))
    cam2 = list(points.get("cam2", []))
    n0, n1, n2 = len(cam0), len(cam1), len(cam2)

    triplets: List[tuple] = []
    if n0 > 0 and n1 > 0 and n2 > 0:
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    triplets.append((i0, i1, i2, np.array(cam0[i0]), np.array(cam1[i1]), np.array(cam2[i2])))
    elif fill_missing_view:
        if n0 == 0 and n1 > 0 and n2 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam0")
        elif n1 == 0 and n0 > 0 and n2 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam1")
        elif n2 == 0 and n0 > 0 and n1 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam2")

    out: List[Dict] = []
    for i0, i1, i2, p0, p1, p2 in triplets:
        imCoords = np.array(
            [
                [float(p0[0]), float(p0[1])],
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
            ],
            dtype=float,
        )
        xyz = cal.compute_XYZ_from_imageCoord_triplet(imCoords)
        invalid = np.any(np.isnan(xyz))
        if invalid:
            cost = np.inf
        else:
            cost = cal.compute_triplet_registration_error(imCoords)
        out.append({
            "cam0_idx": int(i0),
            "cam1_idx": int(i1),
            "cam2_idx": int(i2),
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "cost": float(cost) if not np.isnan(cost) else None,
        })
    return out


def compute_candidates(
    cal: Calibration,
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    xyz_bounds: Tuple[float, float] | None = None,
    fill_missing_view: bool = True,
) -> Dict[int, List[Dict]]:
    """Compute 3D candidates for all frames."""
    items = sorted(points_by_frame.items())
    return {
        frame: compute_candidates_for_frame(cal, pts, xyz_bounds, fill_missing_view=fill_missing_view)
        for frame, pts in tqdm(items, desc="Step 2 - 3D candidates", unit="frame")
    }


def run(
    model_folder: str,
    points_path: str,
    output_path: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict:
    """Run Step 2."""
    cal = Calibration(model_folder)

    with open(points_path, encoding="utf-8") as f:
        raw = json.load(f)
    points_by_frame = {int(k): v for k, v in raw.items()}
    print(f"Processing {len(points_by_frame)} frame(s)")

    candidates = compute_candidates(cal, points_by_frame, xyz_bounds)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in candidates.items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
    return candidates


def main():
    ap = argparse.ArgumentParser(description="Step 2: compute 3D candidates and registration cost")
    ap.add_argument("--model_folder", required=True, help="Calibration model directory")
    ap.add_argument("--points", required=True, help="JSON output from Step 1")
    ap.add_argument("--output", default=None, help="Output JSON path")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Valid 3D range [LOW, HIGH], for example --xyz_bounds 0 200",
    )
    args = ap.parse_args()
    bounds = tuple(args.xyz_bounds) if args.xyz_bounds is not None else None
    run(args.model_folder, args.points, args.output, xyz_bounds=bounds)


if __name__ == "__main__":
    main()
