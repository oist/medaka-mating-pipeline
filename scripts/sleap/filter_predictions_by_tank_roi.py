from __future__ import annotations

"""
Filter SLEAP-NN ``.slp`` predictions to instances whose centroid lies inside
the tank when projected from a 3D bounding cube via the calibration model.

Use case
--------
Inference at ``-n 6`` (= 2× n_fish) emits twice as many candidates per view as
real fish. The reflections off the water surface (top view) and tank glass
(side views) project *outside* the tank cube in their respective camera image,
because they correspond to 3D points outside the tank volume. Projecting the
known tank cube corners to each view via the calibration model gives a 2D
polygon hull; rejecting instances whose centroid falls outside it removes the
reflections, then keeping the top-N by score yields a clean per-view input for
3D reconstruction.

Inputs
------
- A directory containing ``top_cam.slp`` / ``side1_cam.slp`` / ``side2_cam.slp``
  (output of ``sleap-nn predict`` or the merge step).
- A calibration model directory loadable by ``Calibration``.
- An XYZ bounding cube (defaults to ``(0, 200)`` for all axes).

Outputs
-------
- ``<view>.tank_filtered.slp`` next to each input file (or under
  ``--output-dir``), containing only instances inside the projected tank ROI,
  truncated to ``--keep-per-frame`` (default 3) by descending score.
- ``filter_stats.json`` summarising kept-vs-dropped counts per view.

Usage
-----
    python -m scripts.sleap.filter_predictions_by_tank_roi \
        --slp-dir   "Z:/StephensU/medaka/sleap_output/20251225_K_1_2_FCMC/round2_trt_n6" \
        --paths-config configs/paths.local.toml \
        --keep-per-frame 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_TO_CAM_INDEX = {"top_cam": 0, "side1_cam": 1, "side2_cam": 2}


def _instance_centroid(inst) -> tuple[float, float] | None:
    """Mean of the visible (finite) keypoints for one SLEAP instance."""
    pts = inst.numpy()  # (n_nodes, 2)
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    c = pts[mask].mean(axis=0)
    return float(c[0]), float(c[1])


def _instance_score(inst) -> float:
    s = getattr(inst, "score", None)
    if s is None:
        return float("nan")
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def project_tank_cube(
    cal: Calibration,
    *,
    xyz_low: float,
    xyz_high: float,
    grid_steps: int = 5,
) -> dict[str, np.ndarray]:
    """Project a sampled tank cube to each view; return per-view 2D point sets.

    `grid_steps`-per-axis sampling (default 5 -> 125 points) gives a denser hull
    than the 8 corners alone, which matters when the calibration model is
    nonlinear (the cube faces curve in image space).
    """
    grid = np.linspace(xyz_low, xyz_high, grid_steps)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing="ij")
    xyz = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)  # (n, 3)
    proj = cal.compute_imCoords_from_XYZ(xyz)  # (n, 3, 2)
    return {view: proj[:, VIEW_TO_CAM_INDEX[view], :] for view in VIEWS}


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull. Returns hull vertices CCW."""
    pts = np.asarray(points, dtype=float)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) < 3:
        raise ValueError(f"Need >=3 finite points to build hull, got {len(pts)}")
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    pts = np.unique(pts, axis=0)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting test (returns True for points exactly on the boundary)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def filter_one_view(
    slp_path: Path,
    *,
    polygon: np.ndarray,
    out_path: Path,
    keep_per_frame: int,
) -> dict:
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))
    total_inst = 0
    kept_inst = 0
    dropped_outside = 0
    dropped_no_centroid = 0
    truncated = 0

    new_lfs = []
    for lf in labels.labeled_frames:
        scored = []
        for inst in lf.instances:
            total_inst += 1
            c = _instance_centroid(inst)
            if c is None:
                dropped_no_centroid += 1
                continue
            if not _point_in_polygon(c, polygon):
                dropped_outside += 1
                continue
            scored.append((_instance_score(inst), inst))

        # Top-N by descending score, NaN scores last.
        scored.sort(
            key=lambda r: (-r[0] if np.isfinite(r[0]) else float("inf"), 0)
        )
        if len(scored) > keep_per_frame:
            truncated += len(scored) - keep_per_frame
        keep = scored[:keep_per_frame]
        kept_inst += len(keep)
        if not keep:
            continue
        kept_instances = [inst for _s, inst in keep]
        new_lf = sio.LabeledFrame(
            video=lf.video,
            frame_idx=lf.frame_idx,
            instances=kept_instances,
        )
        new_lfs.append(new_lf)

    used_videos = list({id(lf.video): lf.video for lf in new_lfs}.values())
    out = sio.Labels(
        labeled_frames=new_lfs,
        videos=used_videos,
        skeletons=labels.skeletons,
        tracks=labels.tracks,
        provenance=labels.provenance,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(out, str(out_path))

    return {
        "total_instances": total_inst,
        "kept_instances": kept_inst,
        "dropped_outside_polygon": dropped_outside,
        "dropped_no_centroid": dropped_no_centroid,
        "truncated_to_keep_per_frame": truncated,
        "labeled_frames_in": len(labels.labeled_frames),
        "labeled_frames_out": len(new_lfs),
        "output": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slp-dir", type=Path, required=True,
                    help="Dir containing top_cam.slp / side1_cam.slp / side2_cam.slp.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write *.tank_filtered.slp (default: alongside inputs).")
    ap.add_argument("--paths-config", default=None)
    ap.add_argument("--calibration-model-dir", default=None,
                    help="Override calibration_model_dir from paths.local.toml.")
    ap.add_argument("--xyz-low", type=float, default=0.0)
    ap.add_argument("--xyz-high", type=float, default=200.0)
    ap.add_argument("--grid-steps", type=int, default=5,
                    help="Per-axis grid samples (default 5 -> 125 cube points).")
    ap.add_argument("--keep-per-frame", type=int, default=3,
                    help="Top-N detections per frame (by score) to keep after the ROI filter.")
    ap.add_argument("--suffix", default=".tank_filtered.slp",
                    help="Suffix for output filenames.")
    args = ap.parse_args()

    local = load_local_paths(args.paths_config)
    model_dir_str = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if not model_dir_str:
        raise SystemExit(
            "calibration_model_dir not provided (set in paths.local.toml or "
            "pass --calibration-model-dir)."
        )
    cal = Calibration(model_dir_str)
    if not cal.has_registration_model:
        raise SystemExit(
            f"Calibration at {model_dir_str} lacks XYZ_to_imCoords.joblib; "
            "need it to project the tank cube to each view."
        )

    print(f"Calibration: {cal.model_dir}")
    print(f"Tank cube: XYZ in [{args.xyz_low}, {args.xyz_high}], "
          f"grid_steps={args.grid_steps}")

    proj = project_tank_cube(
        cal, xyz_low=args.xyz_low, xyz_high=args.xyz_high, grid_steps=args.grid_steps
    )
    polygons = {view: _convex_hull(proj[view]) for view in VIEWS}
    for view, poly in polygons.items():
        print(
            f"  {view}: {len(poly)} hull vertices, "
            f"x[{poly[:, 0].min():.0f}, {poly[:, 0].max():.0f}], "
            f"y[{poly[:, 1].min():.0f}, {poly[:, 1].max():.0f}]"
        )

    out_dir = args.output_dir or args.slp_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"polygons": {v: polygons[v].tolist() for v in VIEWS}, "views": {}}
    for view in VIEWS:
        slp_in = args.slp_dir / f"{view}.slp"
        if not slp_in.exists():
            print(f"  {view}: SKIP (no input at {slp_in})")
            continue
        slp_out = out_dir / f"{view}{args.suffix}"
        print(f"\nFiltering {view}: {slp_in.name} -> {slp_out.name}")
        stats = filter_one_view(
            slp_in,
            polygon=polygons[view],
            out_path=slp_out,
            keep_per_frame=args.keep_per_frame,
        )
        summary["views"][view] = stats
        print(
            f"  total={stats['total_instances']}, "
            f"kept={stats['kept_instances']}, "
            f"dropped_outside={stats['dropped_outside_polygon']}, "
            f"truncated={stats['truncated_to_keep_per_frame']}"
        )

    stats_path = out_dir / "filter_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {stats_path}")


if __name__ == "__main__":
    main()
