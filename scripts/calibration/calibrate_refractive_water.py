"""Stage 1: per-camera refractive calibration from the in-water checkerboard.

Camera fixed at the origin looking +Z (per-camera frame; the common world frame is
recovered later in Stage 2 from the 12-ball cube). Intrinsics (K + distortion) from
Stage 0 are FIXED. We fit the flat refractive interface (normal n̂, axial distance d0;
top camera = 1 air/water interface, side cameras = 2 air/glass/water with 4 mm glass)
plus a 6-DoF pose per detected board view, minimizing the robust point-to-ray distance
between each known planar corner (7x6 grid, 10 mm) and the pixel's back-projected
water-side ray. Back-projection uses cv2.undistortPoints so distortion is handled
exactly and the camera convention is the standard x_cam = R X + t (looks +Z).

Writes refractive_<cam>.json (interface params + fit residuals).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

N_AIR, N_GLASS, N_WATER = 1.0, 1.49, 1.3323
BOARD_GRID = (7, 6)
BOARD_SQ = 10.0
CAMF = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
# interfaces per camera: (has_glass, glass_thickness_mm)
CAM_INTERFACES = {"top_cam": (False, 0.0), "side1_cam": (True, 4.0), "side2_cam": (True, 4.0)}


def rodrigues(r):
    R, _ = cv2.Rodrigues(np.asarray(r, float)); return R


def refract(d, n, eta):
    if np.dot(n, d) > 0:
        n = -n
    cosi = -np.dot(n, d)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0:
        return None
    out = eta * d + (eta * cosi - np.sqrt(k)) * n
    return out / np.linalg.norm(out)


def board_objp():
    c, r = BOARD_GRID
    o = np.zeros((c * r, 3), np.float32)
    o[:, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2)
    return o * BOARD_SQ


def interface_planes(n_hat, d0, has_glass, tg):
    """Planes (point, normal, eta) along +Z-ish from camera at origin into water."""
    n_hat = n_hat / np.linalg.norm(n_hat)
    p0 = n_hat * d0
    if has_glass:
        return [(p0, n_hat, N_AIR / N_GLASS), (p0 + n_hat * tg, n_hat, N_GLASS / N_WATER)]
    return [(p0, n_hat, N_AIR / N_WATER)]


def backproject(norm_xy, planes):
    """norm_xy: undistorted normalized coord (x,y). Camera at origin looking +Z."""
    d = np.array([norm_xy[0], norm_xy[1], 1.0]); d /= np.linalg.norm(d)
    o = np.zeros(3)
    for (p0, nrm, eta) in planes:
        denom = np.dot(d, nrm)
        if abs(denom) < 1e-9:
            return None, None
        t = np.dot(p0 - o, nrm) / denom
        if t <= 0:
            return None, None
        o = o + t * d
        d = refract(d, nrm, eta)
        if d is None:
            return None, None
    return o, d


def detect_board_views(video, K, dist, phase_start_frac, stride, sharp_min, max_views):
    cap = cv2.VideoCapture(video)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(N * phase_start_frac)
    views = []
    objp = board_objp()
    for idx in range(start, N, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(g, cv2.CV_64F).var() < sharp_min:
            continue
        found, corners = cv2.findChessboardCornersSB(g, BOARD_GRID, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        undist = cv2.undistortPoints(corners, K, dist).reshape(-1, 2)  # normalized
        ok2, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)          # non-refractive pose init
        if not ok2:
            continue
        views.append({"idx": idx, "norm": undist, "rvec": rvec.ravel(), "tvec": tvec.ravel()})
        if len(views) >= max_views:
            break
    cap.release()
    return views


def residuals_pose(p, norm, planes):
    R = rodrigues(p[:3]); t = p[3:]
    objp = board_objp()
    out = []
    for k in range(len(objp)):
        X = R @ objp[k] + t
        o, d = backproject(norm[k], planes)
        if d is None:
            out += [1e3, 1e3, 1e3]; continue
        r = (X - o) - np.dot(X - o, d) * d
        out += list(r)
    return out


def fit_camera(video, K, dist, has_glass, tg, d0_init, stride, sharp_min, max_views):
    K = np.array(K); dist = np.array(dist)
    views = detect_board_views(video, K, dist, 0.45, stride, sharp_min, max_views)
    if len(views) < 6:
        return {"error": "too few board views", "n_views": len(views)}
    # interface params: n_hat (2 ang: az,el) + d0 ; board poses: 6 each
    def n_from_ang(az, el):
        return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])

    def unpack(v):
        az, el, d0 = v[:3]
        poses = v[3:].reshape(-1, 6)
        return n_from_ang(az, el), d0, poses

    p0 = [0.0, 0.0, d0_init] + list(np.concatenate([np.concatenate([w["rvec"], w["tvec"]]) for w in views]))

    def resid(v):
        n_hat, d0, poses = unpack(v)
        planes = interface_planes(n_hat, d0, has_glass, tg)
        out = []
        for w, pose in zip(views, poses):
            out += residuals_pose(pose, w["norm"], planes)
        return out

    # Jacobian sparsity: interface params (cols 0-2) touch every residual; each
    # board pose (cols 3+6i..8+6i) touches only its own view's residuals.
    m_per = len(board_objp()) * 3
    M = len(views) * m_per
    Np = len(p0)
    js = lil_matrix((M, Np), dtype=int)
    js[:, 0:3] = 1
    for i in range(len(views)):
        js[i * m_per:(i + 1) * m_per, 3 + 6 * i:9 + 6 * i] = 1

    r = least_squares(resid, p0, method="trf", loss="huber", f_scale=1.0,
                      jac_sparsity=js, max_nfev=200)
    n_hat, d0, poses = unpack(r.x)
    rr = np.array(resid(r.x)).reshape(-1, 3)
    rd = np.linalg.norm(rr, axis=1)
    return {"n_views": len(views), "interface_normal": n_hat.tolist(), "d0_mm": float(d0),
            "has_glass": has_glass, "glass_mm": tg, "n_water": N_WATER,
            "point_to_ray_mm": {"mean": float(rd.mean()), "rms": float(np.sqrt((rd ** 2).mean())),
                                "p95": float(np.percentile(rd, 95)), "max": float(rd.max())}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage 1: per-camera refractive fit from in-water checkerboard")
    p.add_argument("--water-video-root", required=True, type=Path,
                   help="folder with top_cam/side1_cam/side2_cam of the in-water recording")
    p.add_argument("--intrinsics-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--cameras", nargs="+", default=list(CAMF))
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--sharp-min", type=float, default=80.0)
    p.add_argument("--max-views", type=int, default=50)
    p.add_argument("--d0-init", type=float, default=120.0)
    a = p.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for cam in a.cameras:
        intr = json.loads((a.intrinsics_dir / f"intrinsics_{cam}.json").read_text())
        if "error" in intr:
            print(f"{cam}: SKIP (bad intrinsics)"); continue
        has_glass, tg = CAM_INTERFACES[cam]
        rep = fit_camera(str(a.water_video_root / cam / CAMF[cam]), intr["K"], intr["dist"],
                         has_glass, tg, a.d0_init, a.stride, a.sharp_min, a.max_views)
        rep["cam"] = cam
        (a.output_dir / f"refractive_{cam}.json").write_text(json.dumps(rep, indent=2))
        if "error" in rep:
            print(f"{cam}: ERROR {rep['error']} n={rep.get('n_views')}")
        else:
            print(f"{cam}: {rep['n_views']} views  d0={rep['d0_mm']:.1f}mm  n_hat={[round(x,3) for x in rep['interface_normal']]}  "
                  f"point-to-ray mean {rep['point_to_ray_mm']['mean']:.3f} rms {rep['point_to_ray_mm']['rms']:.3f} mm")


if __name__ == "__main__":
    main()
