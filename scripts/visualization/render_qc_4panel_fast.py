from __future__ import annotations

"""Fast 2x2 QC video: 3 camera views + a 3D trajectory panel, driven entirely by the
resolved dual-line 3-fish track so identity is CONSISTENT across all four panels.

Full-length, no-skip QC on this PC: threaded CPU decode + cv2-only drawing + a single GPU
**NVENC** encode pipe (no matplotlib, no libx264); `--workers K` splits the timeline into K
chunks rendered in parallel (each its own NVENC session) then `ffmpeg -c copy` concat.

Identity = the interp track's ``id_final`` in {F,M1,M2} (one source of truth):
  * Each fish's resolved 3D position is REPROJECTED into every view (calibration) and matched
    to the nearest SLEAP keypoint -> draw that fish's pose coloured by id_final. (Using the
    reprojected resolved track, NOT best.json instance indices which swap sort-order at
    crossings, is what keeps the 2D colour glued to the right fish and consistent with 3D.)
  * 3D panel: each fish's position + a 5 s motion tail + a **body-axis arrow** = the triangulated
    tail->mouth direction (mouth/tail 3D recovered from the matched per-view keypoints), i.e. where
    the fish is FACING, not where it is moving.
  * Confidence: a confident anchor (id_conf=high) draws SOLID; a gap-filled / occluded fish draws
    a FADED hollow marker (in 2D, at the reprojected position when no SLEAP keypoint is nearby) ->
    both panels show the same fish set without the "extra fish" confusion.

F=red, M1=blue, M2=orange. Layout: top_cam (NW) / side1_cam (NE) / side2_cam (SW) / 3D (SE).
"""

import argparse
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.visualization.render_4panel_video import _hex_to_bgr, label_panel_top  # noqa: E402
from scripts.visualization.render_3d_skeleton_video import load_slp_node_xy  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")           # = cam0, cam1, cam2 (calibration triplet order)
VIEW_CAM = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
ID_NAMES = ("F", "M1", "M2")
ID_HEX = {"F": "#E03030", "M1": "#1f77b4", "M2": "#ff9500"}  # female red, M1 blue, M2 orange
ID_BGR = {k: _hex_to_bgr(v) for k, v in ID_HEX.items()}
SRC_W = 1200
DEFAULT_CALIB = "calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models"


def _fade(color, alpha):
    return tuple(int(255 - (255 - c) * alpha) for c in color)


# ----------------------------------------------------------------- 3D projector
class Projector3D:
    """Orthographic az/el projection of 3D points onto a square panel (cv2, white bg)."""

    def __init__(self, panel: int, lo: float, hi: float, elev: float, azim: float):
        self.panel = panel
        el, az = np.radians(elev), np.radians(azim)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # target->camera
        right = np.array([-np.sin(az), np.cos(az), 0.0])
        up = np.cross(d, right)   # right-handed: world +Z projects UP (cross(right,d) flips Z)
        self.right, self.up = right, up / (np.linalg.norm(up) + 1e-9)
        self.center = np.array([(lo + hi) / 2] * 3)
        self.scale = panel * 0.92 / ((hi - lo) * 1.45)
        self.cx = self.cy = panel / 2.0
        c = [lo, hi]
        corners = np.array([[x, y, z] for x in c for y in c for z in c])
        self.edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                      if np.sum(np.abs(corners[i] - corners[j]) > 0) == 1]
        self.corners = corners

    def project(self, pts: np.ndarray) -> np.ndarray:
        rel = np.asarray(pts, float) - self.center
        return np.stack([self.cx + (rel @ self.right) * self.scale,
                         self.cy - (rel @ self.up) * self.scale], axis=-1)

    def base_canvas(self) -> np.ndarray:
        import cv2
        img = np.full((self.panel, self.panel, 3), 255, np.uint8)
        ce = self.project(self.corners)
        for i, j in self.edges:
            p, q = ce[i], ce[j]
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (210, 210, 210), 1, cv2.LINE_AA)
        return img


def draw_3d_panel(proj, base, pos, conf, body, f, tail_n, arrow_mm=16.0):
    """pos[id]=(N,3); conf[id]=(N,) bool; body[id]=(N,3) unit tail->mouth dir (NaN if unknown).
    Tail polyline + current marker + a body-axis arrow (facing direction)."""
    import cv2
    img = base.copy()
    for name in ID_NAMES:
        traj = pos[name]
        seg = traj[max(0, f - tail_n):f + 1]
        finite = np.isfinite(seg[:, 0])
        if finite.sum() == 0 or not finite[-1]:
            continue
        col = ID_BGR[name]
        pix = proj.project(seg)
        for k in range(1, len(pix)):
            if finite[k] and finite[k - 1]:
                a = 0.20 + 0.80 * (k / max(len(pix) - 1, 1))
                cv2.line(img, (int(pix[k - 1, 0]), int(pix[k - 1, 1])),
                         (int(pix[k, 0]), int(pix[k, 1])), _fade(col, a), 2, cv2.LINE_AA)
        cur = pix[-1]
        cp = (int(cur[0]), int(cur[1]))
        high = bool(conf[name][f])
        bd = body[name][f]
        if np.isfinite(bd).all():
            tip = proj.project(traj[f] + bd * arrow_mm)
            tp = (int(tip[0]), int(tip[1]))
            cv2.arrowedLine(img, cp, tp, (0, 0, 0), 5, cv2.LINE_AA, 0, 0.4)
            cv2.arrowedLine(img, cp, tp, col if high else _fade(col, 0.55), 3, cv2.LINE_AA, 0, 0.4)
        if high:
            cv2.circle(img, cp, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, cp, 5, col, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, cp, 6, _fade(col, 0.5), 2, cv2.LINE_AA)
        cv2.putText(img, name, (cp[0] + 9, cp[1] - 9), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (cp[0] + 9, cp[1] - 9), cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 1, cv2.LINE_AA)
    return img


def draw_2d_view(img, view, f, reproj, vis2d, scale):
    """Per fish in this view: if a SLEAP keypoint was matched (vis2d finite) draw its mouth->tail
    arrow (solid); else a faded hollow marker at the reprojected resolved position."""
    import cv2
    for name in ID_NAMES:
        rp = reproj[name][view][f]
        if not np.isfinite(rp).all():
            continue
        col = ID_BGR[name]
        kp = vis2d[name][view][f]   # (n_nodes, 2) source px, NaN if unmatched
        if np.isfinite(kp).any():
            ipts = [(int(round(p[0] * scale)), int(round(p[1] * scale))) for p in kp if np.isfinite(p).all()]
            head = ipts[0]
            if len(ipts) >= 2:
                cv2.arrowedLine(img, ipts[-1], head, (0, 0, 0), 5, cv2.LINE_AA, 0, 0.35)
                cv2.arrowedLine(img, ipts[-1], head, col, 3, cv2.LINE_AA, 0, 0.35)
            else:
                cv2.circle(img, head, 9, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, head, 7, col, -1, cv2.LINE_AA)
        else:
            head = (int(round(rp[0] * scale)), int(round(rp[1] * scale)))
            cv2.circle(img, head, 7, _fade(col, 0.45), 2, cv2.LINE_AA)
        cv2.putText(img, name, (head[0] + 10, head[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (head[0] + 10, head[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- threaded decode
class Decoder(threading.Thread):
    def __init__(self, path: Path, panel: int, start: int, end: int, qsize: int = 96):
        super().__init__(daemon=True)
        self.path, self.panel, self.f0, self.f1 = path, panel, start, end  # not self.start (Thread.start!)
        self.q: "queue.Queue" = queue.Queue(maxsize=qsize)
        self.err: str | None = None

    def run(self) -> None:
        import cv2
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            self.err = f"cannot open {self.path}"; self.q.put(None); return
        if self.f0 > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.f0)  # frame-accurate on these HEVC files (verified)
        f = self.f0
        while f <= self.f1:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if img.shape[0] != self.panel:
                img = cv2.resize(img, (self.panel, self.panel), interpolation=cv2.INTER_AREA)
            self.q.put(img)
            f += 1
        cap.release()
        self.q.put(None)


# --------------------------------------------------------------------- loaders
def load_track(csv_path: Path, n_frames: int):
    import pandas as pd
    cols = ["frame", "id_final", "X", "Y", "Z"]
    df = pd.read_csv(csv_path)
    cf = "id_conf" if "id_conf" in df.columns else None
    df = df[cols + ([cf] if cf else [])]
    pos = {k: np.full((n_frames, 3), np.nan) for k in ID_NAMES}
    conf = {k: np.zeros(n_frames, bool) for k in ID_NAMES}
    for name in ID_NAMES:
        sub = df[df.id_final == name]
        fr = sub.frame.to_numpy().astype(int)
        good = (fr >= 0) & (fr < n_frames)
        pos[name][fr[good]] = sub[["X", "Y", "Z"]].to_numpy()[good]
        conf[name][fr[good]] = (sub[cf].to_numpy()[good] == "high") if cf else True
    return pos, conf


def reproject_track(pos, cal, start, end):
    n = len(pos["F"])
    reproj = {name: {v: np.full((n, 2), np.nan) for v in VIEWS} for name in ID_NAMES}
    sl = slice(start, end + 1)
    for name in ID_NAMES:
        xyz = pos[name][sl]
        fin = np.isfinite(xyz).all(1)
        if fin.sum() == 0:
            continue
        for v in VIEWS:
            buf = np.full((xyz.shape[0], 2), np.nan)
            buf[fin] = np.asarray(cal.project_world_to_view(xyz[fin], v), float)
            reproj[name][v][sl] = buf
    return reproj


def precompute_overlay(reproj, slp, cal, start, end, n, gate):
    """Match each resolved fish to its nearest per-view SLEAP instance, store the matched
    mouth+tail keypoints (vis2d), and triangulate the tail->mouth 3D body axis (body)."""
    vis2d = {name: {v: np.full((n, 2, 2), np.nan) for v in VIEWS} for name in ID_NAMES}
    for v in VIEWS:
        sv = slp[v]
        rpv = {name: reproj[name][v] for name in ID_NAMES}
        for f in range(start, end + 1):
            arr = sv.get(f)
            if arr is None or len(arr) == 0:
                continue
            mouth = arr[:, 0]
            present = [(name, rpv[name][f]) for name in ID_NAMES if np.isfinite(rpv[name][f]).all()]
            pairs = []
            for fi, (_n, p) in enumerate(present):
                for ki in range(len(mouth)):
                    if np.isfinite(mouth[ki]).all():
                        pairs.append((float(np.hypot(p[0] - mouth[ki, 0], p[1] - mouth[ki, 1])), fi, ki))
            pairs.sort()
            assigned, used = {}, set()
            for dist, fi, ki in pairs:
                if fi in assigned or ki in used or dist > gate:
                    continue
                assigned[fi] = ki; used.add(ki)
            for fi, (name, _p) in enumerate(present):
                if fi in assigned:
                    vis2d[name][v][f] = arr[assigned[fi]]   # (mouth, tail) source px

    body = {name: np.full((n, 3), np.nan) for name in ID_NAMES}
    model = cal._model_im2xyz  # batched 6D(cam0,cam1,cam2 xy)->XYZ; same model project_world_to_view inverts
    for name in ID_NAMES:
        M = np.stack([vis2d[name][v][:, 0, :] for v in VIEWS], axis=1)   # (n,3,2) mouth
        T = np.stack([vis2d[name][v][:, 1, :] for v in VIEWS], axis=1)   # (n,3,2) tail
        ok = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(T).all(axis=(1, 2))
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        m3d = model.predict(M[idx].reshape(len(idx), 6))
        t3d = model.predict(T[idx].reshape(len(idx), 6))
        d = m3d - t3d
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        body[name][idx] = np.where(nrm > 1e-6, d / nrm, np.nan)
    return vis2d, body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--track-csv", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--panel", type=int, default=600)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--tail-seconds", type=float, default=5.0)
    ap.add_argument("--source-fps", type=float, default=166.0)
    ap.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    ap.add_argument("--gate-px", type=float, default=80.0)
    ap.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 200.0))
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or r"C:\ffmpeg\ffmpeg-8.0.1-full_build-shared\bin\ffmpeg.exe")
    ap.add_argument("--nvenc-cq", type=int, default=24)
    ap.add_argument("--codec", default="h264_nvenc")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    if args.workers > 1:
        return _orchestrate(args)
    import cv2

    n_total = int(cv2.VideoCapture(str(args.video_dir / "top_cam" / VIEW_CAM["top_cam"])).get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end_frame if args.end_frame is not None else n_total - 1
    start = max(0, args.start_frame)
    panel, scale = args.panel, args.panel / SRC_W
    gate = args.gate_px
    tail_n = int(round(args.tail_seconds * args.source_fps))
    print(f"{args.experiment}  frames {start}..{end}  panel {panel}  fps {args.fps}  codec {args.codec}", flush=True)

    print("loading track + reprojecting ...", flush=True)
    pos, conf = load_track(args.track_csv, n_total)
    from scripts.reconstruction_3d.calibration import Calibration
    cal = Calibration(str(args.calib))
    reproj = reproject_track(pos, cal, start, end)
    slp = {}
    for v in VIEWS:
        print(f"loading {v}.slp ...", flush=True)
        slp[v], _ = load_slp_node_xy(args.slp_dir / f"{v}.slp", node_names=tuple(args.node_names))
    print("matching keypoints + triangulating body axis ...", flush=True)
    vis2d, body = precompute_overlay(reproj, slp, cal, start, end, n_total, gate)

    proj = Projector3D(panel, args.xyz_lim[0], args.xyz_lim[1], args.elev, args.azim)
    base3d = proj.base_canvas()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_wh = panel * 2
    cmd = [args.ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_wh}x{out_wh}",
           "-r", str(args.fps), "-i", "-", "-an"]
    if args.codec.endswith("nvenc"):
        cmd += ["-c:v", args.codec, "-preset", "p5", "-rc", "vbr", "-cq", str(args.nvenc_cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", args.codec, "-crf", "20", "-preset", "veryfast"]
    cmd += ["-pix_fmt", "yuv420p", str(args.out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    decoders = {v: Decoder(args.video_dir / v / VIEW_CAM[v], panel, start, end) for v in VIEWS}
    for d in decoders.values():
        d.start()

    canvas = np.empty((out_wh, out_wh, 3), np.uint8)
    t0 = time.time(); next_report = t0 + 5.0; n_done = 0
    try:
        for f in range(start, end + 1):
            frames = {}
            for v in VIEWS:
                item = decoders[v].q.get()
                if item is None:
                    raise StopIteration
                frames[v] = item
            for v in VIEWS:
                draw_2d_view(frames[v], v, f, reproj, vis2d, scale)
                label_panel_top(frames[v], v, font_scale=0.8)
            p3d = draw_3d_panel(proj, base3d, pos, conf, body, f, tail_n)
            label_panel_top(p3d, f"3D  frame {f}", font_scale=0.8)
            canvas[:panel, :panel] = frames["top_cam"]
            canvas[:panel, panel:] = frames["side1_cam"]
            canvas[panel:, :panel] = frames["side2_cam"]
            canvas[panel:, panel:] = p3d
            enc.stdin.write(canvas.tobytes())
            n_done += 1
            now = time.time()
            if now >= next_report:
                rate = n_done / (now - t0)
                print(f"  {n_done}/{end-start+1}  {rate:.1f} fps  ETA {(end-f)/max(rate,1e-9)/60:.1f} min", flush=True)
                next_report = now + 15.0
    except StopIteration:
        print("  (a decoder ran dry; stopping)", flush=True)
    finally:
        for v, d in decoders.items():
            if d.err:
                print(f"  decoder {v} error: {d.err}", flush=True)
        enc.stdin.close(); enc.wait()
    dt = time.time() - t0
    print(f"done {n_done} frames in {dt/60:.1f} min ({n_done/max(dt,1e-9):.1f} fps) -> {args.out}")
    return 0


def _orchestrate(args) -> int:
    import cv2
    n_total = int(cv2.VideoCapture(str(args.video_dir / "top_cam" / VIEW_CAM["top_cam"])).get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, args.start_frame)
    end = args.end_frame if args.end_frame is not None else n_total - 1
    K = max(1, args.workers)
    per = math.ceil((end - start + 1) / K)
    ranges, s = [], start
    while s <= end:
        ranges.append((s, min(s + per - 1, end))); s += per
    tmpdir = args.out.parent / (args.out.stem + "_parts")
    tmpdir.mkdir(parents=True, exist_ok=True)
    print(f"orchestrating {len(ranges)} parallel chunks (~{per} frames each) -> {args.out}", flush=True)
    procs, parts = [], []
    for i, (cs, ce) in enumerate(ranges):
        part = tmpdir / f"part_{i:03d}_{cs}_{ce}.mp4"
        parts.append(part)
        cmd = [sys.executable, "-m", "scripts.visualization.render_qc_4panel_fast", args.experiment,
               "--slp-dir", str(args.slp_dir), "--track-csv", str(args.track_csv),
               "--video-dir", str(args.video_dir), "--out", str(part), "--calib", str(args.calib),
               "--panel", str(args.panel), "--fps", str(args.fps), "--start-frame", str(cs), "--end-frame", str(ce),
               "--tail-seconds", str(args.tail_seconds), "--source-fps", str(args.source_fps),
               "--gate-px", str(args.gate_px), "--xyz-lim", str(args.xyz_lim[0]), str(args.xyz_lim[1]),
               "--elev", str(args.elev), "--azim", str(args.azim), "--ffmpeg", args.ffmpeg,
               "--nvenc-cq", str(args.nvenc_cq), "--codec", args.codec, "--workers", "1",
               "--node-names", *args.node_names]
        log = open(tmpdir / f"part_{i:03d}.log", "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, part))
        print(f"  chunk {i}: frames {cs}..{ce} -> {part.name}", flush=True)
    rc = 0
    for p, log, part in procs:
        r = p.wait(); log.close()
        ok = r == 0 and part.exists() and part.stat().st_size > 0
        print(f"  {part.name}: {'ok' if ok else f'FAIL rc={r} (see {part.stem}.log)'}", flush=True)
        rc |= 0 if ok else 1
    if rc:
        print("some chunks failed; not concatenating.", flush=True); return rc
    listf = tmpdir / "concat.txt"
    listf.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    cc = [args.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listf), "-c", "copy", str(args.out)]
    print(f"concatenating {len(parts)} parts -> {args.out}", flush=True)
    r = subprocess.run(cc, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print("  -c copy concat failed; re-encoding ...", flush=True)
        r = subprocess.run([args.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                            "-c:v", args.codec, "-pix_fmt", "yuv420p", str(args.out)])
    print(f"done -> {args.out}" if r.returncode == 0 else "CONCAT FAILED", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
