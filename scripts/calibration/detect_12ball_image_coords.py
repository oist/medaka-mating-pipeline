"""Detect the 12-ball rigid calibration object across synced videos and export
``rigid_object_image_coords.csv`` for ``refine_calibration_rigid_object.py``.

The object is a 50 mm wireframe cube carrying colored foam balls at 8 corners +
4 edge midpoints (see ``rigid_object_geometry.csv``). Three hardware-synced
cameras (top=cam0/XY, side1=cam1/XZ, side2=cam2/YZ) film it being moved through
the tank. For each selected synchronized frame ("pose") we:

1. Detect saturated color blobs in each view and classify their hue.
2. Anchor the pose on 4 ultra-distinctive unique colors (light_blue, light_green,
   purple, orange) whose hues never collide with each other or the dark-tank
   green clutter, giving an unambiguous per-view triplet each.
3. Reconstruct those anchors to 3D via the current calibration and Kabsch-fit to
   the known cube geometry to recover the object pose (R, t).
4. Reproject every remaining marker's known object coordinate into each view and
   snap it to the nearest detection. Using object coordinates (not color) for the
   ambiguous markers resolves the duplicate colors (yellow/red/pink) and the
   deliberate yellow<->brown swap purely geometrically.
5. Keep a marker only if it reconstructs and its 3D lands close to the rigid cube.

Outputs the wide CSV plus per-pose debug overlays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration

VIEWS = ["top_cam", "side1_cam", "side2_cam"]  # cam0, cam1, cam2
VIDEO_NAMES = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
CAM_DIRS = {"top_cam": "top_cam", "side1_cam": "side1_cam", "side2_cam": "side2_cam"}

# 4 anchor markers with unique, well-separated, clutter-safe hues.
ANCHOR_MARKERS = [2, 4, 5, 6]  # light_blue, light_green, purple, orange
ANCHOR_COLORS = {2: "light_blue", 4: "light_green", 5: "purple", 6: "orange"}

DRAW_BGR = {
    "red": (0, 0, 255), "orange": (0, 140, 255), "yellow": (0, 230, 230),
    "light_green": (0, 230, 120), "green": (0, 130, 0), "light_blue": (230, 200, 0),
    "purple": (200, 0, 160), "pink": (200, 0, 255), "brown": (40, 60, 110), None: (128, 128, 128),
}


def classify_color(h: float, s: float, v: float) -> str | None:
    """Classify OpenCV HSV (h in 0..179) into one of the 9 ball colors.

    Anchor colors (light_blue, light_green, purple, orange) are isolated hue
    bands. The warm red/orange/yellow/brown cluster is left coarse on purpose:
    those markers are placed geometrically, not by color.
    """
    if h >= 168 or h <= 11:
        return "brown" if v < 95 else "red"
    if 11 < h <= 20:
        return "orange"
    if 20 < h <= 41:
        return "yellow"
    if 41 < h <= 60:
        return "light_green"
    if 60 < h <= 92:
        return "green"
    if 92 < h <= 121:
        return "light_blue"
    if 121 < h <= 148:
        return "purple"
    if 148 < h < 168:
        return "pink"
    return None


def detect_blobs(bgr: np.ndarray, s_min: int = 70, v_min: int = 40,
                 min_area: int = 140, max_area: int = 4000, max_aspect: float = 2.6) -> list[dict]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((S >= s_min) & (V >= v_min)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(w, h) > max_aspect * max(1, min(w, h)):
            continue
        comp = lbl == i
        hue_ang = H[comp].astype(float) * (2 * np.pi / 180.0)
        mh = (np.arctan2(np.sin(hue_ang).mean(), np.cos(hue_ang).mean()) % (2 * np.pi)) * 180.0 / (2 * np.pi)
        s_mean, v_mean = float(S[comp].mean()), float(V[comp].mean())
        out.append({
            "x": float(cent[i][0]), "y": float(cent[i][1]), "area": area,
            "hue": float(mh), "sat": s_mean, "val": v_mean,
            "color": classify_color(mh, s_mean, v_mean),
        })
    return out


def _unique_anchor_detection(blobs: list[dict], color: str) -> dict | None:
    cands = [b for b in blobs if b["color"] == color]
    if not cands:
        return None
    # highest saturation x size wins (real ball beats faint clutter of same hue)
    return max(cands, key=lambda b: b["sat"] * b["area"] ** 0.5)


def _nearest_blob(blobs: list[dict], xy: np.ndarray, radius: float) -> dict | None:
    best, bd = None, radius
    for b in blobs:
        d = float(np.hypot(b["x"] - xy[0], b["y"] - xy[1]))
        if d < bd:
            bd, best = d, b
    return best


def assign_pose(blobs_by_cam: dict[str, list[dict]], calib: Calibration,
                geom: pd.DataFrame, snap_radius_px: float = 28.0,
                anchor_cube_resid_mm: float = 12.0) -> dict:
    """Return {marker_id: {'cam0':(x,y),'cam1':..,'cam2':..}} plus diagnostics.

    Pose bootstrap leans on the clean side views: for each unique-color anchor we
    read cam1/cam2 by color, synthesize the expected top (cam0) pixel via the
    calibration's pairwise helper model, and snap it to the nearest top blob. This
    avoids trusting anchor picks in the clutter-heavy top view. Final per-marker
    coordinates are always REAL detected pixels in all three views (never synth),
    so the exported observations do not bias the subsequent refinement.
    """
    obj = {int(r.marker_id): np.array([r.object_x, r.object_y, r.object_z], float)
           for r in geom.itertuples(index=False)}

    # 1) anchor triplets bootstrapped from the clean side views
    anchor_obj, anchor_3d, used = [], [], []
    for m in ANCHOR_MARKERS:
        s1 = _unique_anchor_detection(blobs_by_cam["side1_cam"], ANCHOR_COLORS[m])
        s2 = _unique_anchor_detection(blobs_by_cam["side2_cam"], ANCHOR_COLORS[m])
        if s1 is None or s2 is None:
            continue
        cam1 = np.array([s1["x"], s1["y"]], float)
        cam2 = np.array([s2["x"], s2["y"]], float)
        try:
            cam0 = calib.compute_XY_imcoords_from_XZ_YZ(cam1, cam2)
        except RuntimeError:
            continue
        top = _nearest_blob(blobs_by_cam["top_cam"], cam0, snap_radius_px * 1.5)
        if top is not None:
            cam0 = np.array([top["x"], top["y"]], float)
        triplet = np.array([cam0, cam1, cam2], float)
        anchor_3d.append(calib.compute_XYZ_from_imageCoord_triplet(triplet))
        anchor_obj.append(obj[m])
        used.append(m)
    if len(used) < 4:
        return {"markers": {}, "n_anchors": len(used), "reason": "insufficient_anchors"}

    anchor_3d = np.array(anchor_3d)
    anchor_obj = np.array(anchor_obj)

    R, t = kabsch_rigid_fit(anchor_obj, anchor_3d)
    resid = np.linalg.norm(apply_rigid(anchor_obj, R, t) - anchor_3d, axis=1)
    if resid.max() > anchor_cube_resid_mm:
        keep = resid <= anchor_cube_resid_mm
        if keep.sum() < 4:
            return {"markers": {}, "n_anchors": len(used), "reason": "anchor_fit_bad",
                    "anchor_resid_mm": float(resid.max())}
        R, t = kabsch_rigid_fit(anchor_obj[keep], anchor_3d[keep])

    # 2) reproject every marker's object coord into each view, then do a per-view
    #    one-to-one (Hungarian) assignment marker<->blob so no detection is ever
    #    shared by two markers. Purely geometric (no color) -> keeps the
    #    yellow<->brown swap and duplicate colors resolved automatically.
    marker_ids = list(obj.keys())
    reproj = {}
    for m in marker_ids:
        world = apply_rigid(obj[m].reshape(1, 3), R, t)
        reproj[m] = np.asarray(calib.compute_imCoords_from_XYZ(world)).reshape(3, 2)

    per_view_assign: dict[str, dict[int, tuple]] = {v: {} for v in VIEWS}
    for ci, v in enumerate(VIEWS):
        blobs = blobs_by_cam[v]
        if not blobs:
            continue
        cost = np.array([[float(np.hypot(b["x"] - reproj[m][ci, 0], b["y"] - reproj[m][ci, 1]))
                          for b in blobs] for m in marker_ids])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= snap_radius_px:
                per_view_assign[v][marker_ids[r]] = (blobs[c]["x"], blobs[c]["y"])

    markers: dict[int, dict] = {}
    for m in marker_ids:
        pick = {}
        if all(m in per_view_assign[v] for v in VIEWS):
            for ci, v in enumerate(VIEWS):
                pick[f"cam{ci}"] = per_view_assign[v][m]
            markers[m] = pick

    # 3) recompute 3D cube residual on the finalized markers
    fit_ids = [m for m in markers]
    if len(fit_ids) >= 4:
        p3d = np.array([calib.compute_XYZ_from_imageCoord_triplet(
            np.array([markers[m]["cam0"], markers[m]["cam1"], markers[m]["cam2"]], float)) for m in fit_ids])
        pobj = np.array([obj[m] for m in fit_ids])
        R2, t2 = kabsch_rigid_fit(pobj, p3d)
        cube_resid = np.linalg.norm(apply_rigid(pobj, R2, t2) - p3d, axis=1)
        # drop gross outliers, then refit on the clean set and report its residual
        keep = cube_resid <= max(8.0, np.median(cube_resid) * 4)
        markers = {m: markers[m] for m, k in zip(fit_ids, keep) if k}
        if keep.sum() >= 4 and keep.sum() < len(fit_ids):
            R3, t3 = kabsch_rigid_fit(pobj[keep], p3d[keep])
            cube_resid = np.linalg.norm(apply_rigid(pobj[keep], R3, t3) - p3d[keep], axis=1)
            max_resid = float(cube_resid.max())
        else:
            max_resid = float(cube_resid[keep].max()) if keep.any() else None
    else:
        max_resid = None
    return {"markers": markers, "n_anchors": len(used), "n_markers": len(markers),
            "cube_resid_mm": max_resid, "reason": "ok"}


def draw_overlay(bgr: np.ndarray, cam_key: str, blobs: list[dict],
                 markers: dict[int, dict]) -> np.ndarray:
    vis = bgr.copy()
    for b in blobs:
        cv2.circle(vis, (int(b["x"]), int(b["y"])), 12, DRAW_BGR.get(b["color"], (128, 128, 128)), 1)
    for m, pick in markers.items():
        x, y = pick[cam_key]
        cv2.circle(vis, (int(x), int(y)), 15, (255, 255, 255), 2)
        cv2.putText(vis, str(m), (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Read frame `idx`, verifying the decoder landed on exactly that frame.

    All three cameras must return the SAME physical frame; a per-view seek slip
    would silently corrupt the moving-object triplet. If the backend lands off
    the requested index we return None so the pose is skipped rather than mixed.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    if landed != idx:
        return None
    return frame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect 12-ball calibration object -> image-coords CSV")
    p.add_argument("--calib-video-root", required=True, type=Path)
    p.add_argument("--geometry-csv", required=True, type=Path)
    p.add_argument("--base-model-folder", required=True, type=Path)
    p.add_argument("--output-csv", required=True, type=Path)
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="explicit frame indices to process")
    p.add_argument("--scan-start", type=int, default=None)
    p.add_argument("--scan-end", type=int, default=None)
    p.add_argument("--scan-stride", type=int, default=200)
    p.add_argument("--min-markers", type=int, default=12,
                   help="keep a pose only if >= this many markers assigned")
    p.add_argument("--max-cube-resid-mm", type=float, default=4.0,
                   help="keep a pose only if its rigid-cube fit residual is below this")
    p.add_argument("--overlay-dir", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--snap-radius-px", type=float, default=28.0)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    geom = pd.read_csv(args.geometry_csv)
    calib = Calibration(args.base_model_folder)
    caps = {v: cv2.VideoCapture(str(args.calib_video_root / CAM_DIRS[v] / VIDEO_NAMES[v])) for v in VIEWS}
    if args.frames:
        candidates = list(args.frames)
    elif args.scan_start is not None and args.scan_end is not None:
        candidates = list(range(args.scan_start, args.scan_end, args.scan_stride))
    else:
        raise SystemExit("provide --frames or --scan-start/--scan-end")

    rows, summary = [], []
    kept = 0
    for idx in candidates:
        imgs = {v: read_frame(caps[v], idx) for v in VIEWS}
        if any(im is None for im in imgs.values()):
            continue
        blobs = {v: detect_blobs(imgs[v]) for v in VIEWS}
        res = assign_pose(blobs, calib, geom, snap_radius_px=args.snap_radius_px)
        nm = res.get("n_markers", 0)
        cr = res.get("cube_resid_mm")
        passed = (res["reason"] == "ok" and nm >= args.min_markers
                  and cr is not None and cr <= args.max_cube_resid_mm)
        summary.append({"pose_id": f"f{idx}", "frame": idx, "n_anchors": res["n_anchors"],
                        "n_markers": nm, "cube_resid_mm": cr, "reason": res["reason"], "kept": passed})
        print(f"idx={idx}: anchors={res['n_anchors']} markers={nm} "
              f"cube_resid={cr if cr is None else round(cr,3)} kept={passed}")
        if not passed:
            continue
        kept += 1
        for m, pick in res["markers"].items():
            rows.append({"pose_id": f"f{idx}", "marker_id": m,
                         "cam0_x": pick["cam0"][0], "cam0_y": pick["cam0"][1],
                         "cam1_x": pick["cam1"][0], "cam1_y": pick["cam1"][1],
                         "cam2_x": pick["cam2"][0], "cam2_y": pick["cam2"][1]})
        if args.overlay_dir:
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            cam_keys = {"top_cam": "cam0", "side1_cam": "cam1", "side2_cam": "cam2"}
            for v in VIEWS:
                vis = draw_overlay(imgs[v], cam_keys[v], blobs[v], res["markers"])
                cv2.imwrite(str(args.overlay_dir / f"pose{idx}_{v}.jpg"), vis)
    for cap in caps.values():
        cap.release()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    if args.summary_csv:
        pd.DataFrame(summary).to_csv(args.summary_csv, index=False)
    print(f"kept {kept}/{len(candidates)} poses; wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
