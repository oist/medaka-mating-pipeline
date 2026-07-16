from __future__ import annotations

"""Fast 2x2 QC video: 3 camera views + a 3D trajectory panel.

Full-length, no-skip QC on this PC: threaded CPU decode + cv2-only drawing + a single GPU
**NVENC** encode pipe (no matplotlib, no libx264); `--workers K` splits the timeline into K
chunks rendered in parallel (each its own NVENC session) then `ffmpeg -c copy` concat.

Identity defaults to the interp track's ``id_final`` in {F,M1,M2}. Anonymous
geometry tracks can instead use, for example, ``--id-column fish_id --identity
ID_1=G1 --identity ID_2=G2 --identity ID_3=G3``:
  * Each fish's resolved 3D position is REPROJECTED into every view (calibration) and matched
    to the nearest SLEAP keypoint -> draw that fish's pose coloured by id_final. (Using the
    reprojected resolved track, NOT best.json instance indices which swap sort-order at
    crossings, is what keeps the 2D colour glued to the right fish and consistent with 3D.)
  * 3D panel: each fish's position + a 5 s motion tail + a **body-axis arrow** = the triangulated
    tail->mouth direction (mouth/tail 3D recovered from the matched per-view keypoints), i.e. where
    the fish is FACING, not where it is moving.
  * Confidence: ``id_conf=high`` draws SOLID. Geometry-only CSVs use
    ``is_interpolated`` instead, so gap-filled positions are faded while measured
    positions remain solid.

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

VIEWS = ("top_cam", "side1_cam", "side2_cam")           # = cam0, cam1, cam2 (calibration triplet order)
VIEW_CAM = {"top_cam": "cam0_top.mp4", "side1_cam": "cam1_side.mp4", "side2_cam": "cam2_side.mp4"}
DEFAULT_IDENTITIES = {"F": "F", "M1": "M1", "M2": "M2"}
IDENTITY_PALETTE = ("#E03030", "#1f77b4", "#ff9500", "#2ca02c", "#9467bd", "#8c564b")
ID_NAMES = tuple(DEFAULT_IDENTITIES.values())
ID_HEX = {name: IDENTITY_PALETTE[index] for index, name in enumerate(ID_NAMES)}
ID_BGR = {k: _hex_to_bgr(v) for k, v in ID_HEX.items()}
SRC_W = 1200
DEFAULT_CALIB = "calibration_results/cali_20251217_manual/merged_rule_reassigned_beads_only/models"


def load_indexed_slp_node_xy(
    slp_path: Path,
    *,
    node_names: tuple[str, ...],
    start_frame: int,
    end_frame: int,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Load only the requested raw SLEAP frames through the HDF5 frame index.

    Short held-out renders previously loaded the complete 348k-frame SLP once
    per clip. The review GUI's indexed reader keeps the raw predicted
    instances while making runtime proportional to the requested episode.
    """
    from scripts.analysis.geometry_swap_review import IndexedSlpPredictions

    source = IndexedSlpPredictions(
        slp_path,
        node_names=node_names,
        cache_size=max(1, end_frame - start_frame + 1),
    )
    try:
        frames = {
            frame: instances
            for frame in range(int(start_frame), int(end_frame) + 1)
            if (instances := source.get(frame)) is not None
        }
    finally:
        source.close()
    return frames, list(node_names)


def _fade(color, alpha):
    return tuple(int(255 - (255 - c) * alpha) for c in color)


def parse_identity_specs(specs: list[str] | None) -> dict[str, str]:
    """Parse ordered SOURCE=DISPLAY identity specifications."""
    if not specs:
        return dict(DEFAULT_IDENTITIES)
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--identity expects SOURCE=DISPLAY, got {spec!r}")
        source, display = (part.strip() for part in spec.split("=", 1))
        if not source or not display:
            raise ValueError(f"--identity expects SOURCE=DISPLAY, got {spec!r}")
        if source in mapping:
            raise ValueError(f"duplicate source identity {source!r}")
        mapping[source] = display
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("--identity display labels must be unique")
    return mapping


def configure_identities(identity_map: dict[str, str]) -> None:
    global ID_NAMES, ID_HEX, ID_BGR
    ID_NAMES = tuple(identity_map.values())
    ID_HEX = {
        name: IDENTITY_PALETTE[index % len(IDENTITY_PALETTE)]
        for index, name in enumerate(ID_NAMES)
    }
    ID_BGR = {key: _hex_to_bgr(value) for key, value in ID_HEX.items()}


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
            # gap-filled (predicted) frame: slightly-transparent FILLED marker (tail drawn solid above)
            cv2.circle(img, cp, 5, _fade(col, 0.5), -1, cv2.LINE_AA)
        cv2.putText(img, name, (cp[0] + 9, cp[1] - 9), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (cp[0] + 9, cp[1] - 9), cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 1, cv2.LINE_AA)
    return img


# raw-instance palette by within-frame instance index (0=primary ... ). BGR.
RAW_PALETTE = [(150, 150, 150), (0, 0, 255), (255, 0, 255), (200, 200, 0)]  # gray, red, magenta, cyan


def draw_raw_sleap(img, arr, scale, by_index=False):
    """Overlay EVERY raw SLEAP instance in this view (no ID filtering) as a tail->mouth arrow
    + node dots. Diagnostic: shows false-positive detections the resolved track does NOT use.
    by_index=False -> all gray. by_index=True -> 1st instance gray, 2nd red, 3rd+ magenta/cyan
    (instances are SLEAP-score-sorted, so the extra/spurious detection is the red/magenta one)."""
    import cv2
    if arr is None or len(arr) == 0:
        return
    for i, inst in enumerate(arr):                    # inst = (n_nodes, 2) src px; node0=mouth, node1=tail
        col = RAW_PALETTE[min(i, len(RAW_PALETTE) - 1)] if by_index else RAW_PALETTE[0]
        th = 1 if (not by_index or i == 0) else 2     # extra detections drawn thicker so they pop
        pts = [(int(round(p[0] * scale)), int(round(p[1] * scale))) for p in inst if np.isfinite(p).all()]
        if len(pts) >= 2:
            cv2.arrowedLine(img, pts[-1], pts[0], col, th, cv2.LINE_AA, 0, 0.3)
        for p in pts:
            cv2.circle(img, p, 3 if (by_index and i > 0) else 2, col, -1, cv2.LINE_AA)


def visible_node_centroids(instances: np.ndarray) -> np.ndarray:
    """Return one centroid per instance using only nodes with finite x and y."""
    array = np.asarray(instances, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError("instances must have shape (instances, nodes, 2)")
    visible = np.isfinite(array).all(axis=2)
    counts = visible.sum(axis=1)
    summed = np.where(visible[..., None], array, 0.0).sum(axis=1)
    centers = np.full((len(array), 2), np.nan, dtype=float)
    valid = counts > 0
    centers[valid] = summed[valid] / counts[valid, None]
    return centers


def match_projected_identities_to_instances(
    projected_by_identity: dict[str, np.ndarray],
    instances: np.ndarray,
    gate_px: float,
) -> dict[str, int]:
    """Greedily match projected geometry centroids to SLEAP instance centroids."""
    centers = visible_node_centroids(instances)
    pairs: list[tuple[float, str, int]] = []
    for identity, projected in projected_by_identity.items():
        point = np.asarray(projected, dtype=float)
        if not np.isfinite(point).all():
            continue
        for instance_index, center in enumerate(centers):
            if np.isfinite(center).all():
                distance = float(np.linalg.norm(point - center))
                pairs.append((distance, identity, instance_index))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    assigned: dict[str, int] = {}
    used_instances: set[int] = set()
    for distance, identity, instance_index in pairs:
        if distance > gate_px:
            break
        if identity in assigned or instance_index in used_instances:
            continue
        assigned[identity] = instance_index
        used_instances.add(instance_index)
    return assigned


def draw_2d_view(
    img,
    view,
    f,
    reproj,
    vis2d,
    scale,
    arrow_t=3,
    center_only=False,
):
    """Draw a colored identity only at a matched raw SLEAP mouth.

    The reprojected 3D centroid is used only to select the raw instance. It is
    deliberately never drawn as the 2D identity center.
    """
    import cv2
    for name in ID_NAMES:
        rp = reproj[name][view][f]
        if not np.isfinite(rp).all():
            continue
        col = ID_BGR[name]
        kp = vis2d[name][view][f]   # (n_nodes, 2) source px, NaN if unmatched
        if len(kp) == 0 or not np.isfinite(kp[0]).all():
            continue
        head = tuple(np.round(kp[0] * scale).astype(int))
        if center_only:
            cv2.circle(img, head, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, head, 4, col, -1, cv2.LINE_AA)
        elif len(kp) >= 2 and np.isfinite(kp[1]).all():
            tail = tuple(np.round(kp[1] * scale).astype(int))
            cv2.arrowedLine(img, tail, head, (0, 0, 0), arrow_t + 2, cv2.LINE_AA, 0, 0.35)
            cv2.arrowedLine(img, tail, head, col, arrow_t, cv2.LINE_AA, 0, 0.35)
        else:
            cv2.circle(img, head, 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, head, 7, col, -1, cv2.LINE_AA)
        cv2.putText(img, name, (head[0] + 10, head[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (head[0] + 10, head[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- threaded decode
class Decoder(threading.Thread):
    def __init__(self, path: Path, panel: int, start: int, end: int, qsize: int = 96, step: int = 1):
        super().__init__(daemon=True)
        self.path, self.panel, self.f0, self.f1 = path, panel, start, end  # not self.start (Thread.start!)
        self.step = max(1, int(step))
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
            # skip step-1 frames without decoding (grab is cheap) to land on the next strided frame
            for _ in range(self.step - 1):
                if not cap.grab():
                    break
            f += self.step
        cap.release()
        self.q.put(None)


# --------------------------------------------------------------------- loaders
def _truthy(values):
    import pandas as pd
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool).to_numpy()
    return (
        values.fillna(False)
        .astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "y"})
        .to_numpy()
    )


def load_track(
    csv_path: Path,
    n_frames: int,
    *,
    id_column: str = "id_final",
    identity_map: dict[str, str] | None = None,
):
    import pandas as pd
    identity_map = dict(identity_map or DEFAULT_IDENTITIES)
    header = pd.read_csv(csv_path, nrows=0).columns
    required = {"frame", id_column, "X", "Y", "Z"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"track CSV is missing columns: {sorted(missing)}")
    confidence_columns = [
        column for column in ("id_conf", "is_interpolated") if column in header
    ]
    df = pd.read_csv(
        csv_path,
        usecols=["frame", id_column, "X", "Y", "Z", *confidence_columns],
        low_memory=False,
    )
    df["_source_identity"] = df[id_column].astype(str)
    df["_display_identity"] = df["_source_identity"].map(identity_map)
    pos = {k: np.full((n_frames, 3), np.nan) for k in ID_NAMES}
    conf = {k: np.zeros(n_frames, bool) for k in ID_NAMES}
    for name in ID_NAMES:
        sub = df[df["_display_identity"] == name]
        fr = sub.frame.to_numpy().astype(int)
        good = (fr >= 0) & (fr < n_frames)
        pos[name][fr[good]] = sub[["X", "Y", "Z"]].to_numpy()[good]
        if "id_conf" in sub:
            confident = sub["id_conf"].astype(str).str.lower().eq("high").to_numpy()
        elif "is_interpolated" in sub:
            confident = ~_truthy(sub["is_interpolated"])
        else:
            confident = np.ones(len(sub), dtype=bool)
        conf[name][fr[good]] = confident[good]
    return pos, conf


def reproject_track(pos, cal, start, end):
    n = len(next(iter(pos.values())))
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
            projected = {
                name: rpv[name][f]
                for name in ID_NAMES
                if np.isfinite(rpv[name][f]).all()
            }
            assigned = match_projected_identities_to_instances(projected, arr, gate)
            for name, instance_index in assigned.items():
                vis2d[name][v][f] = arr[instance_index]   # (mouth, tail) source px

    body = {name: np.full((n, 3), np.nan) for name in ID_NAMES}
    for name in ID_NAMES:
        M = np.stack([vis2d[name][v][:, 0, :] for v in VIEWS], axis=1)   # (n,3,2) mouth
        T = np.stack([vis2d[name][v][:, 1, :] for v in VIEWS], axis=1)   # (n,3,2) tail
        ok = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(T).all(axis=(1, 2))
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        # Keep renderer geometry on the same public calibration backend as the
        # position track. This is a real three-view triangulation for both the
        # refractive and legacy polynomial adapters; never reach into joblib
        # model internals or silently switch geometry here.
        m3d = np.asarray(cal.triangulate(M[idx]), dtype=float)
        t3d = np.asarray(cal.triangulate(T[idx]), dtype=float)
        d = m3d - t3d
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        body[name][idx] = np.where(nrm > 1e-6, d / nrm, np.nan)
    return vis2d, body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--track-csv", type=Path, required=True)
    ap.add_argument("--id-column", default="id_final",
                    help="Track CSV column containing source identities.")
    ap.add_argument("--identity", action="append", default=[], metavar="SOURCE=DISPLAY",
                    help="Source ID and displayed label; repeat in desired color order.")
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib", default=DEFAULT_CALIB)
    ap.add_argument("--panel", type=int, default=600)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--frame-step", type=int, default=1,
                    help="render every Nth source frame (10 = ~stride-10 watchable QC; 1 = no-skip)")
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
    ap.add_argument("--draw-raw-sleap", action="store_true",
                    help="draw ALL raw SLEAP instances in gray beneath mouth-centered colored IDs")
    ap.add_argument("--raw-by-index", action="store_true",
                    help="color raw SLEAP instances by within-frame index: 1st gray, 2nd red, 3rd+ magenta")
    ap.add_argument("--no-track-overlay", action="store_true",
                    help="skip the colored resolved-track arrows in the 2D views (raw SLEAP only)")
    ap.add_argument("--arrow-thickness", type=int, default=3, help="2D colored mouth->tail arrow line thickness")
    args = ap.parse_args(argv)

    try:
        identity_map = parse_identity_specs(args.identity)
    except ValueError as error:
        ap.error(str(error))
    configure_identities(identity_map)

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
    print(f"identity column {args.id_column}: {identity_map}", flush=True)

    print("loading track + reprojecting ...", flush=True)
    pos, conf = load_track(
        args.track_csv,
        n_total,
        id_column=args.id_column,
        identity_map=identity_map,
    )
    from scripts.reconstruction_3d.calibration import Calibration
    cal = Calibration(str(args.calib))
    print(
        f"calibration backend {cal.backend_name} score_unit={cal.score_unit} "
        f"models={cal.provenance.get('model_files')}",
        flush=True,
    )
    reproj = reproject_track(pos, cal, start, end)
    slp = {}
    for v in VIEWS:
        print(f"loading indexed {v}.slp frames {start}..{end} ...", flush=True)
        slp[v], _ = load_indexed_slp_node_xy(
            args.slp_dir / f"{v}.slp",
            node_names=tuple(args.node_names),
            start_frame=start,
            end_frame=end,
        )
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

    step = max(1, int(getattr(args, "frame_step", 1)))
    decoders = {v: Decoder(args.video_dir / v / VIEW_CAM[v], panel, start, end, step=step) for v in VIEWS}
    for d in decoders.values():
        d.start()

    n_planned = len(range(start, end + 1, step))
    canvas = np.empty((out_wh, out_wh, 3), np.uint8)
    t0 = time.time(); next_report = t0 + 5.0; n_done = 0
    try:
        for f in range(start, end + 1, step):
            frames = {}
            for v in VIEWS:
                item = decoders[v].q.get()
                if item is None:
                    raise StopIteration
                frames[v] = item
            for v in VIEWS:
                if args.draw_raw_sleap:
                    draw_raw_sleap(frames[v], slp[v].get(f), scale, by_index=args.raw_by_index)
                if not args.no_track_overlay:
                    draw_2d_view(
                        frames[v],
                        v,
                        f,
                        reproj,
                        vis2d,
                        scale,
                        arrow_t=args.arrow_thickness,
                        center_only=args.draw_raw_sleap,
                    )
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
                print(f"  {n_done}/{n_planned}  {rate:.1f} fps  ETA {(n_planned-n_done)/max(rate,1e-9)/60:.1f} min", flush=True)
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
    step = max(1, int(getattr(args, "frame_step", 1)))
    # snap per-chunk size to a whole number of strided frames so chunk boundaries
    # land on the global stride grid (start + m*step) -> seamless concat, no dup/gap
    per = math.ceil(math.ceil((end - start + 1) / K) / step) * step
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
               "--id-column", str(args.id_column),
               "--video-dir", str(args.video_dir), "--out", str(part), "--calib", str(args.calib),
               "--panel", str(args.panel), "--fps", str(args.fps), "--frame-step", str(step),
               "--start-frame", str(cs), "--end-frame", str(ce),
               "--tail-seconds", str(args.tail_seconds), "--source-fps", str(args.source_fps),
               "--gate-px", str(args.gate_px), "--xyz-lim", str(args.xyz_lim[0]), str(args.xyz_lim[1]),
               "--elev", str(args.elev), "--azim", str(args.azim), "--ffmpeg", args.ffmpeg,
               "--nvenc-cq", str(args.nvenc_cq), "--codec", args.codec, "--workers", "1",
               "--arrow-thickness", str(args.arrow_thickness),
               *(["--draw-raw-sleap"] if args.draw_raw_sleap else []),
               *(["--raw-by-index"] if args.raw_by_index else []),
               *(["--no-track-overlay"] if args.no_track_overlay else []),
               *[item for spec in args.identity for item in ("--identity", spec)],
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
