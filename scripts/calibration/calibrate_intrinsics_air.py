"""Stage 0 of the physics/water calibration: per-camera air intrinsics.

Detects checkerboard corners in the in-air recordings (25 mm board = 10x7 inner
corners, 10 mm board = 7x6), sharpness-gates and de-duplicates for pose/coverage
diversity, then runs cv2.calibrateCamera per camera. Air = no refraction, so this
recovers the TRUE lens intrinsics (K + distortion) that the refractive water fit
holds fixed. Writes intrinsics_<cam>.json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np

CAMF = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
# (folder, inner_grid (cols,rows), square_mm)
AIR_SETS = [("calib_checkerboard_25mm_air", (10, 7), 25.0),
            ("calib_checkerboard_10mm_air", (7, 6), 10.0)]


def objp_for(grid, sq):
    c, r = grid
    o = np.zeros((c * r, 3), np.float32)
    o[:, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2)
    return o * sq


def collect_views(video, grid, sq, stride, sharp_min, max_views):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    obj, img, centers, meta = [], [], [], []
    for idx in range(0, N, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(g, cv2.CV_64F).var() < sharp_min:
            continue
        found, corners = cv2.findChessboardCornersSB(g, grid, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        obj.append(objp_for(grid, sq)); img.append(corners)
        centers.append(corners.reshape(-1, 2).mean(0)); meta.append(idx)
    cap.release()
    return obj, img, centers, meta


def diversify(obj, img, centers, size, max_views):
    """Keep a spatially spread subset: bin board-center in a 5x5 image grid, cap per bin."""
    if len(obj) <= max_views:
        return obj, img
    W, H = size
    bins = {}
    for i, c in enumerate(centers):
        key = (int(c[0] / W * 5), int(c[1] / H * 5))
        bins.setdefault(key, []).append(i)
    per_bin = max(1, max_views // max(1, len(bins)))
    keep = []
    for ids in bins.values():
        step = max(1, len(ids) // per_bin)
        keep.extend(ids[::step][:per_bin])
    keep = keep[:max_views]
    return [obj[i] for i in keep], [img[i] for i in keep]


def calibrate_camera(root, cam, stride, sharp_min, max_views):
    obj_all, img_all, cen_all = [], [], []
    size = None
    per_set = {}
    for folder, grid, sq in AIR_SETS:
        vid = str(Path(root) / folder / cam / CAMF[cam])
        cap = cv2.VideoCapture(vid); size = (int(cap.get(3)), int(cap.get(4))); cap.release()
        o, im, cen, meta = collect_views(vid, grid, sq, stride, sharp_min, max_views)
        per_set[folder] = len(o)
        obj_all += o; img_all += im; cen_all += cen
    if len(obj_all) < 8:
        return {"cam": cam, "error": "too few board views", "n_views": len(obj_all), "per_set": per_set}
    obj_all, img_all = diversify(obj_all, img_all, cen_all, size, max_views)
    flags = cv2.CALIB_RATIONAL_MODEL  # k1..k6 for a bit more distortion freedom; drop if unstable
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_all, img_all, size, None, None, flags=0)
    # per-view reprojection error
    errs = []
    for o, im, rv, tv in zip(obj_all, img_all, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((im - proj) ** 2, axis=2)))))
    return {"cam": cam, "image_size": size, "n_views": len(obj_all), "per_set": per_set,
            "rms_reproj_px": float(rms), "per_view_rms_px": {"mean": float(np.mean(errs)),
            "p95": float(np.percentile(errs, 95)), "max": float(np.max(errs))},
            "K": K.tolist(), "dist": dist.ravel().tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-camera air intrinsics from checkerboard recordings")
    p.add_argument("--calib-air-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--cameras", nargs="+", default=list(CAMF))
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--sharp-min", type=float, default=80.0)
    p.add_argument("--max-views", type=int, default=60)
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for cam in a.cameras:
        rep = calibrate_camera(a.calib_air_root, cam, a.stride, a.sharp_min, a.max_views)
        (a.output_dir / f"intrinsics_{cam}.json").write_text(json.dumps(rep, indent=2))
        if "error" in rep:
            print(f"{cam}: ERROR {rep['error']} (n={rep['n_views']}, per_set={rep['per_set']})")
        else:
            print(f"{cam}: {rep['n_views']} views {rep['per_set']}  fx={rep['fx']:.1f} fy={rep['fy']:.1f} "
                  f"cx={rep['cx']:.1f} cy={rep['cy']:.1f}  rms={rep['rms_reproj_px']:.3f}px "
                  f"(p95 {rep['per_view_rms_px']['p95']:.3f})  dist={[round(x,4) for x in rep['dist'][:5]]}")


if __name__ == "__main__":
    main()
