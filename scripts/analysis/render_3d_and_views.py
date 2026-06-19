"""Composite render: three camera views with SLEAP predictions colored by 3D
identity (top), and an animated 3D plot of the three fish with N-second trailing
tails (bottom). Same identity colors throughout.

Usage::

    python -m scripts.analysis.render_3d_and_views \
        --csv-3d <...>_3d_identified.csv --slp-dir <...> --model-folder <...> \
        --video-top ... --video-side1 ... --video-side2 ... \
        --start-frame 133468 --end-frame 134291 --tail-s 5 --output out.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_predictions_identity_video import (
    VIEWS, VIEW_CAM_IDX, VIEW_SLP_NAME, ID_COLORS, TARGET_H,
    load_view_predictions, reproject_identities, draw_real,
)

# Manual orthographic 3D projection drawn with cv2 -- ~microseconds vs matplotlib's
# ~40 ms/frame, which was the render bottleneck.
_CUBE_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
               (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _project(pts_norm, azim, elev):
    """Project normalized [-1,1] xyz -> (u right, v up) via rotate-Z(azim), rotate-X(elev)."""
    a, e = np.radians(azim), np.radians(elev)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    Q = pts_norm @ Rz.T @ Rx.T
    return Q[:, 0], Q[:, 2]


def _to_px(u, v, w, h, pad=0.12):
    s = (1 - 2 * pad) * min(w, h) / 3.4
    return (w / 2 + u * s).astype(np.int32), (h / 2 - v * s).astype(np.int32)


def _draw_3d_panel(w, h, posarr, ids, id_to_cidx, fidx, tail, bounds, azim, elev, axes=None):
    canvas = np.full((h, w, 3), 18, np.uint8)
    lo, hi = bounds
    rng = (hi - lo) or 1.0

    def norm(P):
        return (np.asarray(P, float) - lo) / rng * 2.0 - 1.0

    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    cpx, cpy = _to_px(*_project(norm(corners), azim, elev), w, h)
    for i, j in _CUBE_EDGES:
        cv2.line(canvas, (int(cpx[i]), int(cpy[i])), (int(cpx[j]), int(cpy[j])),
                 (70, 70, 70), 1, cv2.LINE_AA)
    bg = np.array([18.0, 18.0, 18.0])
    for fid in ids:
        seg = posarr[fid][max(0, fidx - tail):fidx + 1]
        m = np.isfinite(seg).all(axis=1)
        if m.sum() < 1:
            continue
        s = seg[m]
        col = np.array(ID_COLORS[id_to_cidx[fid] % len(ID_COLORS)], float)
        colb = tuple(int(v) for v in col)
        # faded tail: subsample to ~150 pts, blend each segment toward the background
        # by recency (older = fainter).
        stride = max(1, len(s) // 150)
        sd = np.vstack([s[::stride], s[-1]]) if len(s) > 1 else s
        px, py = _to_px(*_project(norm(sd), azim, elev), w, h)
        for k in range(len(px) - 1):
            wgt = 0.12 + 0.88 * (k + 1) / max(len(px) - 1, 1)
            c = tuple(int(v) for v in (bg + (col - bg) * wgt))
            cv2.line(canvas, (int(px[k]), int(py[k])), (int(px[k + 1]), int(py[k + 1])),
                     c, 2, cv2.LINE_AA)
        # fish drawn as an arrow along its 3D body axis, pointing TAIL -> MOUTH (head)
        ax3 = axes.get(fid) if axes else None
        drawn = False
        if ax3 is not None:
            m3, t3 = np.asarray(ax3[0], float), np.asarray(ax3[1], float)
            d = m3 - t3                      # tail -> mouth
            nd = float(np.linalg.norm(d))
            if nd > 1e-6:
                mid = (m3 + t3) / 2.0
                dirn = d / nd
                base = mid - dirn * (0.065 * rng)     # tail end
                tip = mid + dirn * (0.065 * rng)      # mouth end (arrowhead)
                bp = _to_px(*_project(norm(base[None, :]), azim, elev), w, h)
                tp = _to_px(*_project(norm(tip[None, :]), azim, elev), w, h)
                cv2.arrowedLine(canvas, (int(bp[0][0]), int(bp[1][0])),
                                (int(tp[0][0]), int(tp[1][0])), colb, 3, cv2.LINE_AA, tipLength=0.5)
                drawn = True
        if not drawn:
            cv2.circle(canvas, (int(px[-1]), int(py[-1])), 6, colb, -1, cv2.LINE_AA)
    return canvas


def _body_axes(fi, preds, rp, ids, cal, assign_max_dist):
    """Reconstruct each fish's 3D (mouth, tail) by matching its instance per view
    (nearest SLEAP instance to the reprojected centroid) and triangulating both nodes.
    Returns {fid: (mouth3d, tail3d)} (value None if it can't be formed from all 3 views)."""
    node = {fid: {} for fid in ids}
    for v in VIEWS:
        cam = VIEW_CAM_IDX[v]
        insts = preds[v].get(fi, [])
        cents = []
        for pts in insts:
            fin = pts[np.isfinite(pts).all(axis=1)]
            cents.append(fin.mean(axis=0) if len(fin) else None)
        for fid, rx, ry in rp.get(v, []):
            best, bd = None, assign_max_dist
            for pts, c in zip(insts, cents):
                if c is None:
                    continue
                dd = float(np.hypot(c[0] - rx, c[1] - ry))
                if dd < bd:
                    bd, best = dd, pts
            if best is not None and len(best) >= 2:
                m = best[0] if np.isfinite(best[0]).all() else None
                t = best[1] if np.isfinite(best[1]).all() else None
                node[fid][cam] = (m, t)
    out = {}
    for fid in ids:
        nd = node[fid]
        if len(nd) == 3 and all(nd[c][0] is not None and nd[c][1] is not None for c in (0, 1, 2)):
            try:
                m3 = cal.compute_XYZ_from_imageCoord_triplet(np.array([nd[c][0] for c in (0, 1, 2)]))
                t3 = cal.compute_XYZ_from_imageCoord_triplet(np.array([nd[c][1] for c in (0, 1, 2)]))
                out[fid] = (m3, t3)
            except Exception:
                out[fid] = None
        else:
            out[fid] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--video-top", type=Path, required=True)
    ap.add_argument("--video-side1", type=Path, required=True)
    ap.add_argument("--video-side2", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tail-s", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=166.0)
    ap.add_argument("--target-h", type=int, default=480)   # was 360 -- a bit higher res
    ap.add_argument("--assign-max-dist", type=float, default=60.0)
    ap.add_argument("--xyz-bounds", type=float, nargs=2, default=[-10.0, 220.0])
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--step", type=int, default=1,
                    help="render every Nth frame (output fps = fps/step); use >1 for full-length")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary (for GPU NVENC encoding)")
    ap.add_argument("--encoder", default="h264_nvenc",
                    help="ffmpeg video encoder (h264_nvenc=GPU, or libx264=CPU)")
    args = ap.parse_args()

    tail = int(args.tail_s * args.fps)
    cal = Calibration(args.model_folder)
    df = pd.read_csv(args.csv_3d, usecols=lambda c: c in
                     ("frame", "final_fish_id", "fish_id", "X", "Y", "Z"))
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id"
    per_frame, id_to_cidx = reproject_identities(df, cal, id_col)
    ids = sorted(id_to_cidx)

    # per-fish XYZ arrays over [F0, end] (F0 reaches back one tail for full trails)
    F0 = max(0, args.start_frame - tail)
    piv = df.pivot_table(index="frame", columns=id_col, values=["X", "Y", "Z"])
    piv = piv.reindex(range(F0, args.end_frame + 1))
    posarr = {fid: np.stack([piv.get(("X", fid)), piv.get(("Y", fid)),
                             piv.get(("Z", fid))], axis=1) for fid in ids}

    preds = {v: load_view_predictions(args.slp_dir / VIEW_SLP_NAME[v]) for v in VIEWS}
    videos = {"top": args.video_top, "side1": args.video_side1, "side2": args.video_side2}
    caps = {v: cv2.VideoCapture(str(videos[v])) for v in VIEWS}
    for v, c in caps.items():
        if not c.isOpened():
            raise SystemExit(f"cannot open {videos[v]}")
    w0 = int(caps["top"].get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(caps["top"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    th = args.target_h - (args.target_h % 2)
    nw = int(w0 * th / h0) // 2 * 2            # even dims (h264/yuv420p requires even)
    row_w = nw * 3
    h3d = int(row_w * 0.42) // 2 * 2
    bar_h = 30
    out_w, out_h = row_w, th + h3d + bar_h

    step = max(1, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # encode via ffmpeg (GPU NVENC by default); decode below with OpenCV
    import subprocess
    proc = subprocess.Popen(
        [args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", f"{args.fps/step:.4f}", "-i", "-",
         "-an", "-c:v", args.encoder, "-preset", "p4", "-pix_fmt", "yuv420p", str(args.output)],
        stdin=subprocess.PIPE)
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    # decode SEQUENTIALLY (cheap) and render only every step-th frame -- far faster
    # than seeking per output frame.
    for v in VIEWS:
        if args.start_frame > 0:
            caps[v].set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    cur = {v: np.zeros((h0, w0, 3), np.uint8) for v in VIEWS}
    for fi in tqdm(range(args.start_frame, args.end_frame + 1), desc="render", ncols=80):
        render_this = (fi - args.start_frame) % step == 0
        for v in VIEWS:
            if render_this:
                ret, fr = caps[v].read()
                if ret and fr is not None:
                    cur[v] = fr
            else:
                caps[v].grab()          # decode-only (cheaper than read) for skipped frames
        if not render_this:
            continue
        rp = per_frame.get(fi, {v: [] for v in VIEWS})
        panels = []
        for v in VIEWS:
            frame = draw_real(cur[v].copy(), preds[v].get(fi, []), rp.get(v, []),
                              id_to_cidx, args.assign_max_dist)
            cv2.putText(frame, v, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panels.append(cv2.resize(frame, (nw, th)))
        top = np.hstack(panels)
        axes = _body_axes(fi, preds, rp, ids, cal, args.assign_max_dist)
        panel3d = _draw_3d_panel(row_w, h3d, posarr, ids, id_to_cidx,
                                 fi - F0, tail, tuple(args.xyz_bounds), args.azim, args.elev, axes)
        bar = np.zeros((bar_h, out_w, 3), np.uint8)
        cv2.putText(bar, f"t={fi/args.fps:7.2f}s  frame {fi}   (top: SLEAP preds colored by 3D id; "
                    f"bottom: 3D tracks, {args.tail_s:.0f}s tails)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x = out_w - 90 * len(ids) - 10
        for fid in ids:
            cv2.circle(bar, (x, 15), 7, ID_COLORS[id_to_cidx[fid] % len(ID_COLORS)], -1)
            cv2.putText(bar, f"ID {fid}", (x + 12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x += 90
        proc.stdin.write(np.ascontiguousarray(np.vstack([top, panel3d, bar])).tobytes())

    proc.stdin.close(); proc.wait()
    for c in caps.values():
        c.release()
    n = (args.end_frame - args.start_frame) // step + 1
    print(f"Wrote: {args.output}  ({n} frames @ {args.fps/step:.1f}fps)")


if __name__ == "__main__":
    main()
