"""Review GUI: watch every frame, fix identities and head/tail orientation.

Open one review folder (``<review_root>/<exp>/<round_tag>``; the proxies are
read from ``../proxy``)::

    python -m scripts.review.review_gui <review_root>/<exp>/<round_tag>

Only numpy / pandas / opencv / PyQt5 are needed, so this runs on a laptop.
Every key press is written immediately to ``human_events.csv``; what was
shown (and at which speed) goes to ``coverage.csv``. Press F1 for the key map.
Identity edits are names (once per fish) and swaps (the tracker exchanged two
fish from a frame on); see :mod:`scripts.review.ledger`. Export the result
with :mod:`scripts.review.export_reviewed`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.review.behaviors import (
    CATEGORIES,
    LANES,
    bout_rows,
    default_search,
    empty_bouts,
    find_boris_csv,
    load_bouts,
)
from scripts.review.geometry import (
    GEOMETRY_KEYS,
    Detections,
    Geometry,
    confirm_value,
    edited_ranges,
    effective_intervals,
    event_range,
    geometry_state_key,
    interval_ready,
    keyframes,
    load_geometry_cache,
    load_solutions,
    recompute,
    recompute_many,
    save_geometry_cache,
    solve,
)
from scripts.review.ledger import (
    ALL_FISH,
    FLAG_KINDS,
    PAIR_SEP,
    Ledger,
    ReviewLock,
    analysis_start,
    assign_label,
    covered_fraction,
    flip_ranges,
    owner_series,
    snap_frame,
    static_labels,
    swap_list,
    watched_intervals,
)

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_TITLES = ("top", "side 1", "side 2")
LABELS = ("F", "M1", "M2")
LABEL_COLORS = {"F": (70, 70, 240), "M1": (240, 160, 50), "M2": (0, 190, 255), "-": (140, 140, 140)}  # BGR
TRACK_COLORS = [(210, 210, 210), (170, 230, 170), (230, 180, 230), (180, 230, 230), (200, 200, 150)]
SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
AUTO_SLOW_SPEED = 0.5
SELECT_RADIUS_SRC_PX = 70.0
TRAIL_SECONDS = 0.5
COVERAGE_SPLIT_SECONDS = 15.0
ZOOM_HALF_SECONDS = 30.0          # half width of the zoomed timeline strip
OFF_PROJECTION_SRC_PX = 40.0      # detection this far from the projected 3D position gets a connecting line
CROP_HALF_SECONDS = 1.0           # view zoom frames the fish positions within +-1 s
CROP_MARGIN = 0.09                # of the image size, around the fish
CROP_MIN_FRACTION = 0.3           # strongest zoom: 30% of the image
CROP_SMOOTHING = 0.25             # per rendered frame, towards the new crop
MARK_SCALE_DEFAULT = 1.5
MARK_SCALE_RANGE = (0.6, 4.0)
SETTINGS = ("medaka-mating-pipeline", "review_gui")
TRACE_SPEED_MAX = 300.0           # mm/s, top of the speed plot (faster is clipped)
TRACE_DIST_MAX = 60.0             # mm, top of the nearest-fish distance plot
CLOSE_MM = 15.0                   # dashed line in the distance plot
SPEED_HALF_WINDOW = 5             # frames each side for the speed trace (~30 ms at 166 fps)
ERROR_LOG = "review_gui_errors.log"   # in the ledger folder
# Behaviour categories: the first three categorical slots of the dataviz reference palette, dark-surface
# steps (validated all-pairs there), plus a neutral grey for "other".
BEHAVIOR_COLORS = {"courtship": "#3987e5", "aggressive": "#d95926", "mating": "#199e70", "other": "#8a8985"}
MAX_SOLVE_SECONDS = 120.0         # longer intervals are split by the reviewer (solver memory ~ frames x 216 states)
SWAP_HINT_SECONDS = 60.0          # a swap placed by a jump reports the closest approach this far back
SWAP_SNAP_SECONDS = 2.0           # a swap entered while / right after playing snaps this far back at most
INTERP_COLOR = (150, 150, 150)

KEY_HELP = [
    ("Space", "play / pause (1x)"),
    ("L / J / K", "play forward / play backward (press again = faster) / pause"),
    ("Up / Down", "faster / slower (every speed counts as watched)"),
    ("Right / Left", "next / previous shown frame (~30 fps step)"),
    (". / ,", "next / previous source frame (exact frame stepping)"),
    ("Shift+Right / Shift+Left", "jump 1 s"),
    ("Click / Tab / Esc", "select fish (in any view) / cycle selection / clear selection, cancel a pending edit"),
    ("1 / 2 / 3", "select the fish now named F / M1 / M2 (before naming: track 1 / 2 / 3)"),
    ("S", "analysis starts at this frame (hands / net before it are not analysed)"),
    ("4 / 5 / 6", "the selected fish is F / M1 / M2. First time: names it for the whole recording. Once the "
                  "name is taken, the tracker swapped two fish: their tracks are exchanged from here on, snapped "
                  "to their closest approach in the last 2 s (after playing) or since the previously shown "
                  "frame (after stepping). Example: 1 then 5 = the fish shown as F is really M1"),
    ("Shift+4 / 5 / 6", "rename only (whole recording, no swap) - fixes a wrong first naming"),
    ("0", "selected track is not a fish (whole recording)"),
    ("H", "flip head/tail of the selected fish over its whole orientation segment (all views)"),
    ("Shift+H", "flip head/tail from this frame to the end of the segment"),
    ("B", "body wrong (tail short, orientation off, centre shifted): press at the first and at the last "
          "wrong frame; mouth, tail and centre are re-estimated there"),
    ("Q", "large view only: head/tail of the selected fish reversed - press at the first and last wrong frame"),
    ("R", "large view only: leave this view out of the selected fish's 3D - first and last wrong frame"),
    ("T", "large view only: the selected fish's detection belongs to another fish - T, then 1/2/3 for the "
          "other fish, then T again at the last wrong frame"),
    ("X", "toggle 'bad 3D' from here on (position wrong)"),
    ("U", "toggle 'uncertain' from here on (no selection = all fish)"),
    ("I / Shift+I", "next / previous close interval (fish stuck together); purple band on the timeline"),
    ("Home / End", "first / last frame of the current interval"),
    ("[ / ]", "move the interval start / end to this frame (outside an interval: start a new one)"),
    ("Backspace", "delete the current interval"),
    ("C", "keyframe: identities and head/tail are right at this frame (confirm both interval ends)"),
    ("Y", "solve the current interval: re-assign detections between its keyframes and recompute 3D"),
    ("Ctrl+Right / Ctrl+Left", "next / previous manual (BORIS) behaviour bout"),
    ("E", "show / hide the edit list (double-click = go there, Delete = remove)"),
    ("D", "show / hide the 3D window (drag rotate, wheel zoom, right-click top/front/side, double-click reset)"),
    ("N / P", "next / previous machine risk window"),
    ("W", "jump to the next frame not yet watched"),
    ("A", "toggle auto-slow (0.5x) inside risk windows"),
    ("O", "toggle overlays"),
    ("G", "go to frame"),
    ("Ctrl+Z", "undo last change of this session"),
    ("Double-click / V", "make that view the large one / cycle the large view"),
    ("Z", "zoom every view onto the fish (on / off)"),
    ("+ / -", "larger / smaller marks and text"),
    ("F1", "this help"),
    ("", ""),
    ("Cross", "3D position projected into the view (always drawn; grey = interpolated, no detection)"),
    ("Thin line", "from the detection to the cross: they are far apart in this view"),
    ("Dashed skeleton, ~", "fish overlaps another fish in this view: its nodes may sit on the other fish"),
    ("Dashed ring, no 2D", "not detected in this view this frame; the cross is the 3D position"),
    ("interp", "not detected in any view: the 3D position is interpolated (grey cross)"),
    ("Thin dashes, squares", "body nodes this view did not detect: projection of their 3D position"),
    ("3D window", "filled dot = triangulated node, hollow circle = fitted to one view, square = body model"),
    ("Plots", "speed (mm/s) and distance to the nearest fish (mm, dashed = 15 mm) per named fish"),
    ("Behaviour row", "manual BORIS bouts (lanes: towards F / between males / one fish); blue courtship, orange "
                      "aggressive, green mating, grey other; also a thin strip under the whole-video timeline"),
]


def bgr_hex(bgr: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])


def to_display(pts: np.ndarray, tf: tuple[float, float, float]) -> np.ndarray:
    """Source pixels -> displayed pixels (``tf`` = scale, x offset, y offset); NaN stays NaN."""
    a, bx, by = tf
    return np.asarray(pts, float) * a + np.array([bx, by])


def pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def dashed_line(img: np.ndarray, p: np.ndarray, q: np.ndarray, color, width: int, dash: float) -> None:
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(1, int(np.linalg.norm(q - p) // max(dash, 1.0)))
    for i in range(0, n, 2):
        cv2.line(img, pt(p + (q - p) * i / n), pt(p + (q - p) * min(i + 1, n) / n), color, width, cv2.LINE_AA)


def identity_changes(before: np.ndarray, after: np.ndarray, members: list[int], margin_mm: float = 2.0) -> np.ndarray:
    """``(F,)`` True where the group's fish exchanged identities between two versions of their 3D
    centroids ``(F, K, 3)``: some other assignment of the members' new positions to their old ones is
    clearly better (by ``margin_mm`` per fish) than keeping each fish, and keeping them is > 3 mm off."""
    import itertools

    m = len(members)
    b, a = before[:, members], after[:, members]
    own = np.nansum(np.linalg.norm(a - b, axis=2), axis=1)
    best_other = np.full(len(b), np.inf)
    for perm in itertools.permutations(range(m)):
        if list(perm) == list(range(m)):
            continue
        cost = np.nansum(np.linalg.norm(a - b[:, list(perm)], axis=2), axis=1)
        best_other = np.minimum(best_other, cost)
    return (own > 3.0 * m) & (best_other + margin_mm * m < own)


def kind_key(kind: str) -> str:
    return {"view_flip": "Q", "view_drop": "R", "view_swap": "T"}.get(kind, "?")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where ``mask`` is true."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def crop_target(centres: np.ndarray, width: int, height: int) -> np.ndarray:
    """Square crop ``(cx, cy, side)`` in proxy pixels framing ``centres (M, 2)``; whole image if none."""
    full = min(width, height)
    c = centres[np.isfinite(centres).all(axis=1)]
    if not len(c):
        return np.array([width / 2, height / 2, full], float)
    lo, hi = c.min(axis=0), c.max(axis=0)
    side = float(np.clip((hi - lo).max() + 2 * CROP_MARGIN * full, CROP_MIN_FRACTION * full, full))
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, side], float)


def crop_rect(state: np.ndarray, width: int, height: int, aspect: float = 1.0) -> tuple[int, int, int, int]:
    """``(x0, y0, w, h)`` of a crop state, widened or heightened towards the display ``aspect`` (w/h)
    so the view fills its slot, and kept inside the image."""
    side = min(state[2], width, height)
    cw = int(round(min(width, max(side, side * aspect))))
    ch = int(round(min(height, max(side, side / aspect))))
    x0 = int(np.clip(round(state[0] - cw / 2), 0, width - cw))
    y0 = int(np.clip(round(state[1] - ch / 2), 0, height - ch))
    return x0, y0, cw, ch


# ---------------------------------------------------------------------------
# Data


class ReviewPackage:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.proxy_dir = self.folder.parent / "proxy"
        self.proxy = json.loads((self.proxy_dir / "proxy_manifest.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.folder / "review_manifest.json").read_text(encoding="utf-8"))
        with np.load(self.folder / "review_data.npz") as z:
            self.d = {k: z[k] for k in z.files}
        self.risks = pd.read_csv(self.folder / "machine_risk.csv")
        mi = self.folder / "machine_intervals.csv"
        self.machine_intervals = pd.read_csv(mi) if mi.exists() else None
        self.base: dict | None = None        # untouched copies of the arrays geometry edits rewrite
        top = self.proxy["views"]["top_cam"]
        self.fps = float(top["source_fps"])
        self.step = int(top["play_step"])
        self.n_source = min(v["source_frames"] for v in self.proxy["views"].values())
        self.scale = [self.proxy["views"][v]["proxy_width"] / self.proxy["views"][v]["source_width"] for v in VIEWS]
        self.frames = self.d["frames"]
        self.first, self.last = int(self.frames[0]), int(self.frames[-1])
        self.fish_ids = [str(f) for f in self.d["fish_ids"]]
        self.edges = [tuple(int(x) for x in e) for e in self.d["edges"]]
        self.head, self.tail = int(self.d["head_node"]), int(self.d["tail_node"])
        self.kp2d, self.proj2d, self.xyz = self.d["kp2d"], self.d["proj2d"], self.d["xyz"]
        no_flags = np.zeros(self.kp2d.shape[:3], dtype=bool)   # packages built before data_version 2
        self.overlap = self.d.get("view_overlap", no_flags)
        self.present = self.d.get("view_present", np.isfinite(self.kp2d).all(axis=-1).any(axis=-1))
        self.interp = self.d.get("is_interpolated", np.zeros(self.kp2d.shape[:2], dtype=bool))
        self.segments = self.d.get("orientation_segment")      # (F, K); data_version 3+
        # Every node in 3D (data_version 4+: completed, with its source), and where a view lacks a 2D node,
        # the projection of the 3D node.
        self.nodes = self.d.get("nodes3d_filled", self.d.get("nodes3d"))          # (F, K, N, 3) or None
        self.node_source = self.d.get("node_source")                             # (F, K, N) or None
        self.missing_proj = self.d.get("missing_node_proj2d")                    # (F, K, V, N, 2) or None
        n_fish = len(self.fish_ids)
        with np.errstate(invalid="ignore"):
            near = np.full(self.xyz.shape[:2], np.inf, dtype=np.float32)
            for i in range(n_fish):
                for j in range(i + 1, n_fish):
                    d = np.nan_to_num(np.linalg.norm(self.xyz[:, i] - self.xyz[:, j], axis=1), nan=np.inf)
                    near[:, i], near[:, j] = np.minimum(near[:, i], d), np.minimum(near[:, j], d)
        self.nearest = np.where(np.isfinite(near), near, np.nan)   # (F, K) mm to the nearest other track
        self.risk_start = self.risks["frame_start"].to_numpy(int) if len(self.risks) else np.zeros(0, int)
        self.risk_end = self.risks["frame_end"].to_numpy(int) if len(self.risks) else np.zeros(0, int)

    def ensure_base(self) -> None:
        """Keep untouched copies of the arrays geometry edits rewrite (before the first rewrite)."""
        if self.base is None:
            from scripts.review.geometry import GEOMETRY_KEYS

            self.base = {key: np.array(self.d[key], copy=True) for key in GEOMETRY_KEYS if key in self.d}

    def update_nearest(self, lo: int, hi: int) -> None:
        """Recompute the nearest-fish distance for frames lo..hi after their 3D changed."""
        a, b = lo - self.first, hi - self.first + 1
        x = self.xyz[a:b].astype(float)
        near = np.full(x.shape[:2], np.inf)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                d = np.nan_to_num(np.linalg.norm(x[:, i] - x[:, j], axis=1), nan=np.inf)
                near[:, i], near[:, j] = np.minimum(near[:, i], d), np.minimum(near[:, j], d)
        self.nearest[a:b] = np.where(np.isfinite(near), near, np.nan)

    def t(self, frame: int) -> int | None:
        return frame - self.first if self.first <= frame <= self.last else None

    def risk_at(self, frame: int) -> int | None:
        hit = np.flatnonzero((self.risk_start <= frame) & (frame <= self.risk_end))
        return int(hit[0]) if len(hit) else None

    def fish_point(self, t: int, k: int, v: int) -> np.ndarray:
        """Representative source-pixel point of fish ``k`` in view ``v`` (detection centroid, else projection)."""
        pts = self.kp2d[t, k, v]
        ok = np.isfinite(pts).all(axis=1)
        return pts[ok].mean(axis=0) if ok.any() else self.proj2d[t, k, v]

    def centres(self, idx: np.ndarray, v: int) -> np.ndarray:
        """``(len(idx), K, 2)`` detection centroids in view ``v`` (projected 3D where not detected)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames without any visible node
            c = np.nanmean(self.kp2d[idx, :, v], axis=2)
        return np.where(np.isfinite(c), c, self.proj2d[idx, :, v])


def proxy_fps(manifest: dict, view: str, play: bool) -> float:
    """True frame rate of a view's full (source rate) or playback proxy (source rate / play step)."""
    meta = manifest["views"][view]
    return float(meta["source_fps"]) / (int(meta["play_step"]) if play else 1)


class ProxyVideo:
    """Frame-indexed reader with a small cache so backward stepping stays cheap.

    ``fps``: the file's true frame rate. OpenCV seeks by turning a frame index into a time with the
    container's rounded frame rate; for the playback proxy (166 / 6 = 27.666... fps, reported as 27.667144)
    that lands one frame early from about frame 30000 on. With ``fps`` every seek is checked against the
    timestamp of the frame it reached and corrected.
    """

    def __init__(self, path: Path, cache: int = 36, fps: float | None = None):
        self.path, self.cache_size, self.fps = path, cache, fps
        self.cap: cv2.VideoCapture | None = None
        self.pos = 0  # index of the next frame cap.read() returns
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _open(self):
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open {self.path}")
        self.pos = 0

    def _remember(self, k: int, img: np.ndarray):
        self.cache[k] = img
        self.cache.move_to_end(k)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _seek(self, k: int) -> None:
        """Position so that the next frame read is ``k`` or an earlier one (``self.pos`` = its true index)."""
        if not self.fps:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            self.pos = k
            return
        back = 2
        while True:
            target = max(0, k - back)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            if not self.cap.grab():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self.pos = target
                return
            got = int(round(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 * self.fps))
            if got < k or target == 0:
                self.pos = got + 1          # the frame grabbed was ``got``
                if got >= k:                # start of the file and still too far: begin again at 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.pos = 0
                return
            back *= 4

    def get(self, k: int) -> np.ndarray | None:
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        if self.cap is None:
            self._open()
        if k < self.pos:            # backwards: decode the block ending at k so the next steps back are cached
            self._seek(max(0, k - self.cache_size + 4))
        elif k - self.pos > 60:     # far ahead: seek instead of decoding everything in between
            self._seek(k)
        img = None
        while self.pos <= k:
            if self.pos < k and k - self.pos >= self.cache_size:
                ok = self.cap.grab()
                self.pos += 1
                if not ok:
                    return None
                continue
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._remember(self.pos, frame)
            img = frame
            self.pos += 1
        return img

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Qt


def run_app(folder: Path, *, ledger_dir: Path | None, backup_dir: Path | None, smoke_test: bool,
            boris: Path | None = None) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    class ViewLabel(QtWidgets.QLabel):
        clicked = QtCore.pyqtSignal(int, float, float)
        double_clicked = QtCore.pyqtSignal(int)
        resized = QtCore.pyqtSignal()

        def __init__(self, index: int):
            super().__init__()
            self.index = index
            self.setMinimumSize(200, 200)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet("background:#111;")
            self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            self._map = (1.0, 0.0, 0.0)

        def show_image(self, rgb: np.ndarray):
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.size(), QtCore.Qt.KeepAspectRatio,
                                                       QtCore.Qt.SmoothTransformation)
            s = pix.width() / w
            self._map = (s, (self.width() - pix.width()) / 2, (self.height() - pix.height()) / 2)
            self.setPixmap(pix)

        def mousePressEvent(self, event):
            s, ox, oy = self._map
            if s > 0:
                self.clicked.emit(self.index, (event.x() - ox) / s, (event.y() - oy) / s)

        def mouseDoubleClickEvent(self, event):
            self.double_clicked.emit(self.index)

        def resizeEvent(self, event):
            # The image is rendered at the label's size; a stale one would also misplace clicks.
            super().resizeEvent(event)
            self.resized.emit()

    class Timeline(QtWidgets.QWidget):
        seek = QtCore.pyqtSignal(int)

        def __init__(self, win, zoom: bool):
            super().__init__()
            self.win, self.zoom = win, zoom
            self.setMinimumHeight(34 if zoom else 36)
            self.setMouseTracking(False)

        def span(self) -> tuple[int, int]:
            if not self.zoom:
                return 0, self.win.pkg.n_source - 1
            half = int(ZOOM_HALF_SECONDS * self.win.pkg.fps)
            a = max(0, self.win.frame - half)
            return a, min(self.win.pkg.n_source - 1, a + 2 * half)

        def x_of(self, f: float, a: int, b: int) -> float:
            return (f - a) / max(1, b - a) * self.width()

        def paintEvent(self, _event):
            p = QtGui.QPainter(self)
            w, h = self.width(), self.height()
            a, b = self.span()
            p.fillRect(0, 0, w, h, QtGui.QColor("#1b1b1b"))
            pkg = self.win.pkg
            x0, x1 = self.x_of(pkg.first, a, b), self.x_of(pkg.last, a, b)
            p.fillRect(QtCore.QRectF(x0, 0, max(1.0, x1 - x0), h), QtGui.QColor("#2c2c34"))
            row = h / 3
            for s, e in self.win.watched:
                if e >= a and s <= b:
                    xa, xb = self.x_of(s, a, b), self.x_of(e + 1, a, b)
                    p.fillRect(QtCore.QRectF(xa, 0, max(1.0, xb - xa), row), QtGui.QColor("#3fa34d"))
            for s, e in zip(pkg.risk_start, pkg.risk_end):
                if e >= a and s <= b:
                    xa, xb = self.x_of(s, a, b), self.x_of(e + 1, a, b)
                    p.fillRect(QtCore.QRectF(xa, row, max(1.0, xb - xa), row), QtGui.QColor("#d9822b"))
            # Close co-moving intervals: purple band (brighter when solved), the current one outlined.
            cur = self.win.current_interval()
            for r in self.win.intervals.itertuples(index=False):
                s, e = int(r.frame_start), int(r.frame_end)
                if e >= a and s <= b:
                    xa, xb = self.x_of(s, a, b), self.x_of(e + 1, a, b)
                    solved = str(r.interval_id) in self.win.solutions
                    color = QtGui.QColor("#c77dff" if solved else "#7b3fb0")
                    p.fillRect(QtCore.QRectF(xa, row * 1.55, max(2.0, xb - xa), row * 0.45), color)
                    if cur is not None and str(r.interval_id) == str(cur["interval_id"]):
                        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
                        p.drawRect(QtCore.QRectF(xa, row * 1.55, max(2.0, xb - xa), row * 0.45))
            start = self.win.analysis_start
            if start is not None and start > a:
                xs = self.x_of(min(start, b), a, b)
                p.fillRect(QtCore.QRectF(0, 0, xs, h), QtGui.QColor(0, 0, 0, 170))
                p.setPen(QtGui.QPen(QtGui.QColor("#4fc3f7"), 2))
                p.drawLine(QtCore.QPointF(xs, 0), QtCore.QPointF(xs, h))
            colors = {"label": "#f2d13d", "swap": "#f2d13d", "flip": "#b56bff", "uncertain": "#9aa0a6",
                      "bad3d": "#e05555", "nodes_bad": "#e0a055", "analysis_start": "#4fc3f7",
                      "view_flip": "#66d9a8", "view_drop": "#66d9a8", "view_swap": "#66d9a8", "confirm": "#ffffff",
                      "interval": "#c77dff", "interval_del": "#c77dff"}
            for f, kind in self.win.event_marks:
                if a <= f <= b:
                    x = self.x_of(f, a, b)
                    p.setPen(QtGui.QPen(QtGui.QColor(colors.get(kind, "#fff")), 2))
                    p.drawLine(QtCore.QPointF(x, 2 * row), QtCore.QPointF(x, h))
            if not self.zoom and len(self.win.bouts):
                # Manual BORIS behaviour bouts: a thin coloured strip along the bottom of the overview.
                for r in self.win.bouts.itertuples(index=False):
                    if r.end >= a and r.start <= b:
                        xa, xb = self.x_of(r.start, a, b), self.x_of(r.end + 1, a, b)
                        p.fillRect(QtCore.QRectF(xa, h - 4, max(1.5, xb - xa), 4),
                                   QtGui.QColor(BEHAVIOR_COLORS[r.category]))
            x = self.x_of(self.win.frame, a, b)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            if self.zoom:
                p.setPen(QtGui.QColor("#aaaaaa"))
                p.drawText(4, h - 4, f"±{ZOOM_HALF_SECONDS:.0f} s")
            p.end()

        def mousePressEvent(self, event):
            self._seek_to(event.x())

        def mouseMoveEvent(self, event):
            if event.buttons() & QtCore.Qt.LeftButton:
                self._seek_to(event.x())

        def _seek_to(self, x: float):
            a, b = self.span()
            self.seek.emit(int(a + max(0.0, min(1.0, x / max(1, self.width()))) * (b - a)))

    class BehaviorRow(Timeline):
        """Manual BORIS behaviour bouts over the zoomed timeline's span: one lane each for behaviour
        towards the female, between the males and of one fish alone; colour = category, text = who and
        what (so colour is never the only cue). Bouts overlapping within a lane (a quick circle during a
        follow) get their own sub-row; each lane is as tall as its largest overlap in the recording."""

        def __init__(self, win):
            super().__init__(win, True)
            self.rows, self.sub = bout_rows(win.bouts)
            self.fit_to_scale()

        def font_px(self) -> int:
            return max(10, round(7.5 * min(self.win.mark_scale, 2.4)))   # follows +/- like the other text

        def row_h(self) -> int:
            return self.font_px() + 6

        def fit_to_scale(self):
            self.setFixedHeight(sum(self.rows) * self.row_h() + 2 * len(LANES) + 2)

        def paintEvent(self, _event):
            p = QtGui.QPainter(self)
            w, h = self.width(), self.height()
            a, b = self.span()
            p.fillRect(0, 0, w, h, QtGui.QColor("#1b1b1b"))
            font = p.font()
            font.setPixelSize(self.font_px())
            p.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            row_h = self.row_h()
            lane_top = np.cumsum([0] + [n * row_h + 2 for n in self.rows]) + 1
            lane_name_w = max(metrics.horizontalAdvance(n) for n in LANES) + 8
            for i in range(len(LANES)):
                p.fillRect(QtCore.QRectF(0, lane_top[i], w, self.rows[i] * row_h), QtGui.QColor("#222222"))
            bouts = self.win.bouts
            if len(bouts):
                visible = bouts[(bouts["end"] >= a) & (bouts["start"] <= b)]
                for i, r in zip(visible.index, visible.itertuples(index=False)):
                    xa, xb = self.x_of(r.start, a, b), self.x_of(r.end + 1, a, b)
                    xb = max(xb, xa + 3)
                    rect = QtCore.QRectF(xa, lane_top[r.lane] + self.sub[i] * row_h, xb - xa, row_h - 1)
                    p.fillRect(rect, QtGui.QColor(BEHAVIOR_COLORS[r.category]))
                    # Text on the visible part of the bar (a long bout may start left of the window).
                    left, right = max(xa, lane_name_w), min(xb, w)
                    text = metrics.elidedText(f"{r.who} {r.behavior}", QtCore.Qt.ElideRight, int(right - left) - 4)
                    if text and text != "…":
                        p.setPen(QtGui.QColor("#f0efec"))
                        p.drawText(QtCore.QRectF(left + 3, rect.top(), right - left - 4, rect.height()),
                                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)
            # Lane names (left, on their own background) and the colour legend (right), in muted text.
            for i, name in enumerate(LANES):
                cell = QtCore.QRectF(0, lane_top[i], lane_name_w, row_h)
                p.fillRect(cell, QtGui.QColor("#1b1b1b"))
                p.setPen(QtGui.QColor("#9a9993"))
                p.drawText(cell.adjusted(3, 0, 0, 0), QtCore.Qt.AlignVCenter, name)
            if not len(bouts):
                p.setPen(QtGui.QColor("#9a9993"))
                p.drawText(QtCore.QRectF(lane_name_w + 8, 0, w, h), QtCore.Qt.AlignVCenter,
                           "no BORIS behaviour file found for this experiment (--boris <file or folder>)")
            else:
                box = 0.8 * self.font_px()
                gap = 0.5 * self.font_px()
                legend_w = sum(metrics.horizontalAdvance(c) + box + 2 * gap for c in CATEGORIES) + 6
                p.fillRect(QtCore.QRectF(w - legend_w, lane_top[0], legend_w, row_h), QtGui.QColor("#1b1b1b"))
                x = w - 4
                for cat in reversed(CATEGORIES):
                    tw = metrics.horizontalAdvance(cat)
                    x -= tw
                    p.setPen(QtGui.QColor("#9a9993"))
                    p.drawText(QtCore.QRectF(x, lane_top[0], tw + 2, row_h), QtCore.Qt.AlignVCenter, cat)
                    x -= box + 0.4 * gap
                    p.fillRect(QtCore.QRectF(x, lane_top[0] + (row_h - box) / 2, box, box),
                               QtGui.QColor(BEHAVIOR_COLORS[cat]))
                    x -= 1.6 * gap
            xf = self.x_of(self.win.frame, a, b)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
            p.drawLine(QtCore.QPointF(xf, 0), QtCore.QPointF(xf, h))
            p.end()

    class TracePlot(Timeline):
        """Speed and distance to the nearest fish per named fish, over the zoomed timeline's span.

        Drawn with OpenCV into an image (QPainter paths were ~65 ms per frame).
        """

        def __init__(self, win):
            super().__init__(win, True)
            self.setMinimumHeight(90)

        def paintEvent(self, _event):
            w, h = max(1, self.width()), max(2, self.height())
            win, pkg = self.win, self.win.pkg
            img = np.full((h, w, 3), 22, np.uint8)
            a, b = self.span()
            ta, tb = max(a, pkg.first) - pkg.first, min(b, pkg.last) - pkg.first
            half = h // 2
            cv2.line(img, (0, half), (w, half), (51, 51, 51), 1)
            y_close = half + int(round((half - 1) * (1 - CLOSE_MM / TRACE_DIST_MAX)))
            for x in range(0, w, 8):
                cv2.line(img, (x, y_close), (x + 3, y_close), (102, 102, 102), 1)
            if tb > ta:
                cols = max(2, min(w, tb - ta + 1))
                edges = np.unique(np.linspace(ta, tb + 1, cols + 1).astype(int).clip(ta, tb + 1))
                starts = edges[:-1]
                xs = ((pkg.first + starts - a) / max(1, b - a) * w).round().astype(np.int32)
                for values, top, y0, use_max in ((win.speed_trace, TRACE_SPEED_MAX, 0, True),
                                                 (win.near_trace, TRACE_DIST_MAX, half, False)):
                    for c in range(values.shape[1]):
                        seg = values[ta:tb + 1, c]
                        filled = np.nan_to_num(seg, nan=-np.inf if use_max else np.inf)
                        pooled = (np.maximum.reduceat(filled, starts - ta) if use_max
                                  else np.minimum.reduceat(filled, starts - ta))
                        ok = np.isfinite(pooled)
                        if not ok.any():
                            continue
                        ys = (y0 + (half - 1) * (1 - np.clip(pooled, 0, top) / top)).round().astype(np.int32)
                        pts = np.stack([xs, ys], axis=1)
                        lines = [pts[s:e + 1] for s, e in _runs(ok)]
                        color = win.fish_color(c)
                        cv2.polylines(img, [ln.reshape(-1, 1, 2) for ln in lines if len(ln) > 1], False, color, 1,
                                      cv2.LINE_AA)
            for y0, text in ((0, f"speed (top {TRACE_SPEED_MAX:.0f} mm/s)"),
                             (half, f"nearest fish (top {TRACE_DIST_MAX:.0f} mm, dashed {CLOSE_MM:.0f} mm)")):
                cv2.putText(img, text, (w - 330, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
                            cv2.LINE_AA)
            x = int(round(self.x_of(win.frame, a, b)))
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            p = QtGui.QPainter(self)
            p.drawImage(0, 0, qimg)
            p.end()

    class EventList(QtWidgets.QListWidget):
        jump = QtCore.pyqtSignal(int)
        remove = QtCore.pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.itemDoubleClicked.connect(lambda item: self.jump.emit(int(item.data(QtCore.Qt.UserRole + 1))))

        def keyPressEvent(self, event):
            item = self.currentItem()
            if item is not None and event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                self.remove.emit(str(item.data(QtCore.Qt.UserRole)))
                return
            super().keyPressEvent(event)

    class View3D(QtWidgets.QWidget):
        """Rotatable 3D view of every fish's nodes at the current frame.

        Drag = rotate, wheel = zoom (zoomed in, it follows the fish), right-click = next preset view
        (oblique / top / front / side), double-click = reset. Filled dots are triangulated nodes,
        hollow circles were fitted to one view, hollow squares come from the body model alone.
        """

        # (name, azimuth, elevation); the camera presets match how those cameras show the tank
        # (top image: right = -X, down = +Y; side 1: right = +X, up = +Z; side 2: right = +Y, up = +Z).
        PRESETS = (("oblique (from side 1, above)", 20.0, 40.0), ("as top camera", 180.0, 89.0),
                   ("as side 1", 0.0, 0.0), ("as side 2", -90.0, 0.0))

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.setMinimumSize(380, 320)
            self.preset = 0
            _, self.az, self.el = self.PRESETS[0]
            self.zoom = 1.0
            self._drag = None
            xyz = win.pkg.xyz.reshape(-1, 3)
            ok = np.isfinite(xyz).all(axis=1)
            if ok.any():
                lo, hi = np.percentile(xyz[ok], [0.5, 99.5], axis=0)
            else:
                lo, hi = np.zeros(3), np.full(3, 200.0)
            pad = 0.05 * float((hi - lo).max())
            self.lo, self.hi = lo - pad, hi + pad
            self.centre = (self.lo + self.hi) / 2

        # -- interaction ---------------------------------------------------
        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.RightButton:
                self.preset = (self.preset + 1) % len(self.PRESETS)
                _, self.az, self.el = self.PRESETS[self.preset]
                self.update()
                return
            self._drag = (event.x(), event.y(), self.az, self.el)

        def mouseMoveEvent(self, event):
            if self._drag is not None:
                x0, y0, az, el = self._drag
                self.az = az - (event.x() - x0) * 0.4
                self.el = float(np.clip(el + (event.y() - y0) * 0.4, -89.0, 89.0))
                self.update()

        def mouseReleaseEvent(self, _event):
            self._drag = None

        def wheelEvent(self, event):
            self.zoom = float(np.clip(self.zoom * 1.15 ** (event.angleDelta().y() / 120), 1.0, 15.0))
            self.update()

        def mouseDoubleClickEvent(self, _event):
            self.preset, self.zoom = 0, 1.0
            _, self.az, self.el = self.PRESETS[0]
            self.update()

        # -- drawing ---------------------------------------------------------
        def project(self, pts: np.ndarray, centre: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
            """Orthographic view: azimuth about the vertical Z axis, then elevation."""
            a, e = np.radians(self.az), np.radians(self.el)
            d = np.asarray(pts, float) - centre
            x1 = d[..., 0] * np.cos(a) - d[..., 1] * np.sin(a)
            y1 = d[..., 0] * np.sin(a) + d[..., 1] * np.cos(a)
            up = d[..., 2] * np.cos(e) + y1 * np.sin(e)
            return np.stack([w / 2 + scale * x1, h / 2 - scale * up], axis=-1)

        def paintEvent(self, _event):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 18, np.uint8)
            win, pkg = self.win, self.win.pkg
            t = pkg.t(win.frame)
            scale = self.zoom * min(w, h) / (float((self.hi - self.lo).max()) * 1.3)
            centre = self.centre
            if t is not None and self.zoom > 1.0 and np.isfinite(pkg.xyz[t]).any():
                follow = min(1.0, self.zoom - 1.0)
                centre = (1 - follow) * self.centre + follow * np.nanmean(pkg.xyz[t], axis=0)

            def proj(p):
                return self.project(p, centre, scale, w, h)

            lo, hi = self.lo, self.hi
            # Tank box, a 20 mm floor grid and axis arrows.
            for i in np.arange(np.ceil(lo[0] / 20) * 20, hi[0], 20):
                q = proj(np.array([[i, lo[1], lo[2]], [i, hi[1], lo[2]]]))
                cv2.line(img, pt(q[0]), pt(q[1]), (38, 38, 38), 1, cv2.LINE_AA)
            for j in np.arange(np.ceil(lo[1] / 20) * 20, hi[1], 20):
                q = proj(np.array([[lo[0], j, lo[2]], [hi[0], j, lo[2]]]))
                cv2.line(img, pt(q[0]), pt(q[1]), (38, 38, 38), 1, cv2.LINE_AA)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            q = proj(corners)
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") == 1:
                        cv2.line(img, pt(q[i]), pt(q[j]), (85, 85, 85), 1, cv2.LINE_AA)
            arrow = 0.12 * float((hi - lo).max())
            for axis, name in enumerate("XYZ"):
                end = lo.copy()
                end[axis] += arrow
                qa = proj(np.array([lo, end]))
                cv2.arrowedLine(img, pt(qa[0]), pt(qa[1]), (150, 150, 150), 1, cv2.LINE_AA, tipLength=0.2)
                cv2.putText(img, name, pt(qa[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)

            if t is not None:
                n_trail = int(2 * TRAIL_SECONDS * pkg.fps)
                idx = np.arange(max(0, t - n_trail), t + 1, 2)
                m = win.mark_scale
                for k in range(len(pkg.fish_ids)):
                    color, name = win.fish_style(k)
                    trail = pkg.xyz[idx, k].astype(float)
                    ok_tr = np.isfinite(trail).all(axis=1)
                    q_tr = proj(trail)
                    for s_, e_ in _runs(ok_tr):
                        if e_ > s_:
                            cv2.polylines(img, [q_tr[s_:e_ + 1].round().astype(np.int32).reshape(-1, 1, 2)], False,
                                          tuple(int(c * 0.6) for c in color), 1, cv2.LINE_AA)
                    c3 = pkg.xyz[t, k].astype(float)
                    if np.isfinite(c3).all():          # drop line to the floor for depth
                        qc = proj(np.array([c3, [c3[0], c3[1], lo[2]]]))
                        cv2.line(img, pt(qc[0]), pt(qc[1]), (70, 70, 70), 1, cv2.LINE_AA)
                        cv2.circle(img, pt(qc[1]), 2, (90, 90, 90), -1, cv2.LINE_AA)
                    if pkg.nodes is None:
                        if np.isfinite(c3).all():
                            cv2.circle(img, pt(proj(c3)), round(4 * m), color, -1, cv2.LINE_AA)
                        continue
                    nodes = pkg.nodes[t, k].astype(float).copy()
                    src = (pkg.node_source[t, k].copy() if pkg.node_source is not None
                           else np.zeros(len(nodes), np.int8))
                    if win.flipped(k):
                        nodes[[pkg.head, pkg.tail]] = nodes[[pkg.tail, pkg.head]]
                        src[[pkg.head, pkg.tail]] = src[[pkg.tail, pkg.head]]
                    ok_n = np.isfinite(nodes).all(axis=1)
                    if not ok_n.any():
                        continue
                    qn = proj(nodes)
                    lw = max(1, round(1.5 * m))
                    for a, b in pkg.edges:
                        if ok_n[a] and ok_n[b]:
                            if src[a] == 0 and src[b] == 0:
                                cv2.line(img, pt(qn[a]), pt(qn[b]), color, lw, cv2.LINE_AA)
                            else:
                                dashed_line(img, qn[a], qn[b], color, lw, 3 * m)
                    for n in np.flatnonzero(ok_n):
                        r = round((4.5 if n == pkg.head else 3.0) * m)
                        c0 = pt(qn[n])
                        if src[n] == 0:
                            cv2.circle(img, c0, r, color, -1, cv2.LINE_AA)
                        elif src[n] == 1:
                            cv2.circle(img, c0, r, color, 1, cv2.LINE_AA)
                        else:
                            cv2.rectangle(img, (c0[0] - r, c0[1] - r), (c0[0] + r, c0[1] + r), color, 1, cv2.LINE_AA)
                    anchor = qn[pkg.head] if ok_n[pkg.head] else qn[ok_n][0]
                    if win.selected == k:
                        cv2.circle(img, pt(qn[ok_n].mean(axis=0)), round(14 * m), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, name, (int(anchor[0]) + 6, int(anchor[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42 * m, color, max(1, round(m)), cv2.LINE_AA)
            label = f"{self.PRESETS[self.preset][0] if self._drag is None else 'free'}  zoom x{self.zoom:.1f}"
            cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(img, "drag rotate | wheel zoom | right-click views | double-click reset", (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, qimg)
            painter.end()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.pkg = ReviewPackage(folder)
            self.ledger_folder = Path(ledger_dir) if ledger_dir else self.pkg.folder
            self.ledger_folder.mkdir(parents=True, exist_ok=True)
            self.lock = ReviewLock(self.ledger_folder)
            self.ledger = Ledger(self.ledger_folder, backup_dir=backup_dir)
            self.lock.acquire()
            pkg = self.pkg
            self.play = [ProxyVideo(pkg.proxy_dir / pkg.proxy["views"][v]["play_file"], fps=proxy_fps(pkg.proxy, v, True))
                         for v in VIEWS]
            self.full = [ProxyVideo(pkg.proxy_dir / pkg.proxy["views"][v]["full_file"], fps=proxy_fps(pkg.proxy, v, False))
                         for v in VIEWS]
            self.selected: int | None = None
            self.overlays = True
            self.auto_slow = True
            self.direction = 0          # 0 paused, +1 forward, -1 backward
            self.speed = 1.0
            self.shown: int | None = None       # frame currently on screen
            self.prev_shown: int | None = None  # frame shown before it (swap snapping window)
            self.last_nav = "seek"              # how the current frame was reached: play / step / seek
            self.pending: dict | None = None    # per-view edit waiting for its end frame (Q / R / T)
            self._geometry: Geometry | None = None
            self._geometry_error = ""
            self._state_cache: dict = {}
            self.analysis_start: int | None = None
            self._rebuild_states()
            self.watched = watched_intervals(self.ledger.coverage)
            self.run: dict | None = None  # open coverage run

            # Display preferences survive restarts (not in the smoke test).
            self.settings = None if smoke_test else QtCore.QSettings(*SETTINGS)
            get = (lambda key, default, kind: kind(self.settings.value(key, default))) if self.settings else \
                (lambda key, default, kind: default)
            self.mark_scale = float(np.clip(get("mark_scale", MARK_SCALE_DEFAULT, float), *MARK_SCALE_RANGE))
            self.crop_on = get("crop_on", "1", str) == "1"
            order = [int(x) for x in get("view_order", "0,1,2", str).split(",") if x.strip().isdigit()]
            self.order = order if sorted(order) == [0, 1, 2] else [0, 1, 2]   # order[0] = the large view
            self.view_tf: list[tuple[float, float, float]] = [(1.0, 0.0, 0.0)] * 3
            self._crop: dict[int, np.ndarray] = {}
            self._crop_frame: dict[int, int] = {}

            start = max((e for _, e in self.watched), default=pkg.first)
            start = max(start, self.analysis_start or 0)
            self.frame = int(min(max(start, 0), pkg.n_source - 1) // pkg.step * pkg.step)

            self.setWindowTitle(f"Review - {pkg.manifest['experiment']} / {pkg.manifest['round_tag']}")
            self.views = [ViewLabel(i) for i in range(3)]
            self._render_soon = QtCore.QTimer(self)    # coalesces label resizes into one render
            self._render_soon.setSingleShot(True)
            self._render_soon.timeout.connect(self.render)
            self._selected_before_click: int | None = None
            for v in self.views:
                v.clicked.connect(self.on_click)
                v.double_clicked.connect(self.make_large)
                v.resized.connect(lambda: self._render_soon.start(0))
            self.info = QtWidgets.QLabel()
            self.info.setTextFormat(QtCore.Qt.RichText)
            self.info.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            self._style_info()
            # Manual BORIS behaviour annotations (behaviour row; frame numbers = top-camera source frames).
            self.boris_path = find_boris_csv(pkg.manifest["experiment"], default_search([boris] if boris else []))
            self.bouts = load_bouts(self.boris_path) if self.boris_path else empty_bouts()
            self.overview, self.zoom, self.traces = Timeline(self, False), Timeline(self, True), TracePlot(self)
            self.behavior_row = BehaviorRow(self)
            for tl in (self.overview, self.zoom, self.traces, self.behavior_row):
                tl.seek.connect(lambda f: self.seek(f // self.pkg.step * self.pkg.step))
            self.event_list = EventList()
            self.event_list.jump.connect(self.seek_from_list)
            self.event_list.remove.connect(self.remove_event_group)
            self.event_dock = QtWidgets.QDockWidget("Edits (double-click: go, Delete: remove)", self)
            self.event_dock.setWidget(self.event_list)
            self.event_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.event_dock)
            self.event_dock.setVisible(get("events_visible", "0", str) == "1")
            self._fill_event_list()
            self.view3d = View3D(self)
            self.view3d_dock = QtWidgets.QDockWidget("3D (D: show / hide)", self)
            self.view3d_dock.setObjectName("view3d")
            self.view3d_dock.setWidget(self.view3d)
            self.view3d_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                                         QtWidgets.QDockWidget.DockWidgetFloatable |
                                         QtWidgets.QDockWidget.DockWidgetMovable)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.view3d_dock)
            self.splitDockWidget(self.view3d_dock, self.event_dock, QtCore.Qt.Vertical)
            self.resizeDocks([self.view3d_dock], [520], QtCore.Qt.Horizontal)
            self.view3d_dock.setVisible(get("view3d_visible", "1", str) == "1")

            # Square videos: the top view spans the full height, the two side
            # views stack beside it at half that size (2:1 widths).
            right = QtWidgets.QVBoxLayout()
            right.setSpacing(2)
            right.addWidget(self.views[1], 1)
            right.addWidget(self.views[2], 1)
            videos = QtWidgets.QHBoxLayout()
            videos.setSpacing(2)
            videos.addWidget(self.views[0], 2)
            videos.addLayout(right, 1)
            body = QtWidgets.QVBoxLayout()
            body.setContentsMargins(4, 4, 4, 4)
            body.setSpacing(3)
            body.addLayout(videos, 1)
            body.addWidget(self.info)
            body.addWidget(self.overview)
            body.addWidget(self.zoom)
            body.addWidget(self.behavior_row)
            body.addWidget(self.traces)
            central = QtWidgets.QWidget()
            central.setLayout(body)
            self.setCentralWidget(central)
            self.statusBar().showMessage("F1 = keys. Start: S at the first usable frame, then click each fish "
                                         "and press 1 (F) / 2 (M1) / 3 (M2) once.")

            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.on_tick)
            self._bind_keys()
            self.render()

        # -- keys ----------------------------------------------------------
        def _bind_keys(self):
            step = self.pkg.step
            sec = int(round(self.pkg.fps))
            keys = {
                "Space": self.toggle_play, "L": lambda: self.shuttle(+1), "J": lambda: self.shuttle(-1),
                "K": self.pause, "Up": lambda: self.change_speed(+1), "Down": lambda: self.change_speed(-1),
                "Right": lambda: self.step_by(step), "Left": lambda: self.step_by(-step),
                ".": lambda: self.step_by(1), ",": lambda: self.step_by(-1),
                "Shift+Right": lambda: self.jump(sec), "Shift+Left": lambda: self.jump(-sec),
                "Tab": self.cycle_selection, "Esc": self.clear_selection,
                "1": lambda: self.select_label("F"), "2": lambda: self.select_label("M1"),
                "3": lambda: self.select_label("M2"),
                "4": lambda: self.set_label("F"), "5": lambda: self.set_label("M1"),
                "6": lambda: self.set_label("M2"), "0": lambda: self.set_label("-", rename=True),
                "Shift+4": lambda: self.set_label("F", rename=True), "$": lambda: self.set_label("F", rename=True),
                "Shift+5": lambda: self.set_label("M1", rename=True), "%": lambda: self.set_label("M1", rename=True),
                "Shift+6": lambda: self.set_label("M2", rename=True), "^": lambda: self.set_label("M2", rename=True),
                "&": lambda: self.set_label("M2", rename=True),     # Shift+6 on a Japanese layout
                "H": lambda: self.flip_segment(False), "Shift+H": lambda: self.flip_segment(True),
                "B": lambda: self.toggle("nodes_bad"), "E": self.toggle_event_list,
                "Q": lambda: self.view_edit("view_flip"), "R": lambda: self.view_edit("view_drop"),
                "T": lambda: self.view_edit("view_swap"),
                "[": lambda: self.set_interval_end(start=True), "]": lambda: self.set_interval_end(start=False),
                "I": lambda: self.goto_interval(+1), "Shift+I": lambda: self.goto_interval(-1),
                "Home": lambda: self.goto_interval_end(start=True), "End": lambda: self.goto_interval_end(start=False),
                "Backspace": self.delete_interval, "C": self.confirm_keyframe, "Y": self.solve_current_interval,
                "Ctrl+Right": lambda: self.goto_bout(+1), "Ctrl+Left": lambda: self.goto_bout(-1),
                "D": lambda: self.view3d_dock.setVisible(not self.view3d_dock.isVisible()),
                "U": lambda: self.toggle("uncertain", allow_all=True),
                "X": lambda: self.toggle("bad3d"), "N": lambda: self.goto_risk(+1), "P": lambda: self.goto_risk(-1),
                "W": self.goto_unwatched, "A": self.toggle_auto_slow, "O": self.toggle_overlays,
                "G": self.goto_dialog, "Ctrl+Z": self.undo, "F1": self.show_help, "?": self.show_help,
                "S": self.mark_analysis_start, "Z": self.toggle_crop, "V": self.cycle_large,
                "+": lambda: self.change_mark_scale(+1), "=": lambda: self.change_mark_scale(+1),
                "-": lambda: self.change_mark_scale(-1),
            }
            for key, fn in keys.items():
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.ApplicationShortcut)
                sc.activated.connect(fn)
                # Windows reports e.g. both "Shift+1" and "!" for one press; Qt then only emits this.
                sc.activatedAmbiguously.connect(fn)

        # -- state cache -----------------------------------------------------
        def _rebuild_states(self):
            cache = {}
            ev = self.ledger.events
            pkg = self.pkg
            self.analysis_start = analysis_start(ev)
            for kind in FLAG_KINDS:
                for fid in pkg.fish_ids:
                    e = ev[(ev["kind"] == kind) & (ev["fish"] == fid)]
                    if e.empty:
                        continue
                    order = np.lexsort((np.arange(len(e)), e["frame"].astype(int).to_numpy()))
                    cache[(kind, fid)] = (e["frame"].astype(int).to_numpy()[order], e["value"].to_numpy()[order])
            self._state_cache = cache
            self.labels = static_labels(ev, pkg.fish_ids)                 # corrected fish -> name
            self.swaps = swap_list(ev, pkg.fish_ids)
            self.owner = owner_series(len(pkg.fish_ids), self.swaps, pkg.frames)   # (F, K) fish on each track
            self.flips = {fid: flip_ranges(ev, fid, pkg.last) for fid in pkg.fish_ids}
            # Traces per corrected fish: speed (mm/s over +-SPEED_HALF_WINDOW frames, which averages out the
            # tail beat) and distance to the nearest other fish.
            rows = np.arange(len(pkg.frames))[:, None]
            source = np.argsort(self.owner, axis=1)
            xyz = pkg.xyz[rows, source]
            speed = np.full(xyz.shape[:2], np.nan, dtype=np.float32)
            h = SPEED_HALF_WINDOW
            speed[h:-h] = np.linalg.norm(xyz[2 * h:] - xyz[:-2 * h], axis=2) / (2 * h) * pkg.fps
            self.speed_trace, self.near_trace = speed, pkg.nearest[rows, source]
            self.event_marks = [(int(f), k) for f, k in zip(ev["frame"], ev["kind"])] if len(ev) else []
            # Close co-moving intervals (machine list + reviewer edits), keyframes and cached solutions.
            self.intervals = effective_intervals(pkg.machine_intervals, ev)
            self.keyframes = keyframes(ev)
            self.solutions = load_solutions(self.ledger_folder, self.intervals, ev, pkg.fish_ids, pkg.fps)
            if hasattr(self, "event_list"):
                self._fill_event_list()

        def current_interval(self, frame: int | None = None) -> dict | None:
            """The (shortest) interval containing ``frame``."""
            frame = self.frame if frame is None else frame
            iv = self.intervals
            hit = iv[(iv["frame_start"] <= frame) & (iv["frame_end"] >= frame)]
            if hit.empty:
                return None
            return hit.loc[(hit["frame_end"] - hit["frame_start"]).idxmin()].to_dict()

        def state(self, kind: str, fid: str, frame: int | None = None) -> str:
            frame = self.frame if frame is None else frame
            item = self._state_cache.get((kind, fid))
            if item is None:
                return "0"
            frames, values = item
            i = np.searchsorted(frames, frame, side="right") - 1
            return str(values[i]) if i >= 0 else "0"

        def fish_of_track(self, k: int, t: int | None = None) -> int:
            """Corrected fish whose data is on track ``k`` at data index ``t`` (default: now)."""
            t = self.pkg.t(self.frame) if t is None else t
            return int(self.owner[t, k]) if t is not None else k

        def track_of_fish(self, c: int, t: int) -> int:
            return int(np.flatnonzero(self.owner[t] == c)[0])

        def fish_color(self, c: int) -> tuple[int, int, int]:
            """Colour of corrected fish ``c`` (by its name)."""
            label = self.labels[self.pkg.fish_ids[c]]
            return LABEL_COLORS.get(label, TRACK_COLORS[c % len(TRACK_COLORS)])

        def flipped(self, k: int, frame: int | None = None) -> bool:
            frame = self.frame if frame is None else frame
            return sum(a <= frame <= b for a, b in self.flips[self.pkg.fish_ids[k]]) % 2 == 1

        # -- playback ----------------------------------------------------------
        def effective_speed(self) -> float:
            if self.auto_slow and self.direction != 0 and self.pkg.risk_at(self.frame) is not None:
                return min(self.speed, AUTO_SLOW_SPEED)
            return self.speed

        def start_play(self, direction: int):
            self.flush_run()
            self.direction = direction
            self._anchor()
            self.open_run("play" if direction > 0 else "reverse")
            self.timer.start(10)
            self.render()

        def _anchor(self):
            self.anchor = (time.perf_counter(), self.frame, self.effective_speed())

        def pause(self):
            if self.direction:
                self.timer.stop()
                self.direction = 0
                self.flush_run()
                self.render()

        def toggle_play(self):
            if self.direction:
                self.pause()
            else:
                self.speed = 1.0
                self.start_play(+1)

        def shuttle(self, direction: int):
            if self.direction == direction:
                self.change_speed(+1)
            else:
                self.pause()
                self.speed = 1.0
                self.start_play(direction)

        def change_speed(self, delta: int):
            i = min(range(len(SPEEDS)), key=lambda j: abs(SPEEDS[j] - self.speed))
            self.speed = SPEEDS[max(0, min(len(SPEEDS) - 1, i + delta))]
            if self.direction:
                self.flush_run()
                self._anchor()
                self.open_run("play" if self.direction > 0 else "reverse")
            self.render()

        def on_tick(self):
            t0, f0, speed = self.anchor
            target = f0 + self.direction * (time.perf_counter() - t0) * self.pkg.fps * speed
            target = int(target // self.pkg.step * self.pkg.step) if self.direction > 0 else \
                int(-(-target // self.pkg.step) * self.pkg.step)
            if target < 0 or target > self.pkg.n_source - 1:
                self.pause()
                return
            if target == self.frame:
                return
            self.frame = target
            self.last_nav = "play"
            if self.run is not None:
                self.run["end"] = self.frame
                if time.time() - self.run["wall_start"] > COVERAGE_SPLIT_SECONDS:
                    self.flush_run()
                    self.open_run("play" if self.direction > 0 else "reverse")
            if abs(self.effective_speed() - speed) > 1e-9:
                self.flush_run()
                self._anchor()
                self.open_run("play" if self.direction > 0 else "reverse")
            self.render()

        # -- coverage ----------------------------------------------------------
        def open_run(self, mode: str):
            self.run = {"start": self.frame, "end": self.frame, "speed": self.effective_speed() if mode != "step" else 0.0,
                        "mode": mode, "wall_start": time.time()}

        def flush_run(self):
            r, self.run = self.run, None
            if r is None or (r["mode"] != "step" and r["end"] == r["start"]):
                return
            self.ledger.add_coverage(r["start"], r["end"], r["speed"], r["mode"], r["wall_start"], time.time())
            self.watched = watched_intervals(self.ledger.coverage)

        # -- navigation ----------------------------------------------------------
        def seek(self, frame: int):
            self.pause()
            self.flush_run()
            self.frame = int(max(0, min(self.pkg.n_source - 1, frame)))
            self.shown = self.prev_shown = None
            self.last_nav = "seek"
            self.render()

        def step_by(self, n: int):
            self.pause()
            new = int(max(0, min(self.pkg.n_source - 1, self.frame + n)))
            if new == self.frame:
                return
            r = self.run
            if r is not None and r["mode"] == "step" and min(r["start"], r["end"]) - self.pkg.step <= new \
                    <= max(r["start"], r["end"]) + self.pkg.step:
                r["start"], r["end"] = min(r["start"], r["end"], new), max(r["start"], r["end"], new)
            else:
                self.flush_run()
                self.open_run("step")
                self.run["start"], self.run["end"] = min(self.frame, new), max(self.frame, new)
            self.frame = new
            self.last_nav = "step"
            self.render()

        def jump(self, n: int):
            self.seek((self.frame + n) // self.pkg.step * self.pkg.step)

        def goto_risk(self, direction: int):
            starts = self.pkg.risk_start
            if not len(starts):
                self.statusBar().showMessage("No machine risk windows.")
                return
            if direction > 0:
                i = np.flatnonzero(starts > self.frame)
            else:
                i = np.flatnonzero(starts < self.frame - self.pkg.step)[::-1]
            if not len(i):
                self.statusBar().showMessage("No more risk windows in that direction.")
                return
            r = self.pkg.risks.iloc[int(i[0])]
            self.seek(int(r["frame_start"]) // self.pkg.step * self.pkg.step)
            self.statusBar().showMessage(f"{r['risk_id']}: {r['types']} - {r['details']}")

        def goto_unwatched(self):
            f = max(self.frame, self.analysis_start or 0)
            for s, e in self.watched:
                if s <= f <= e:
                    f = e + 1
            if f >= self.pkg.n_source:
                self.statusBar().showMessage("Everything after this point has been watched.")
                return
            self.seek(f // self.pkg.step * self.pkg.step)

        def goto_dialog(self):
            self.pause()
            f, ok = QtWidgets.QInputDialog.getInt(self, "Go to frame", "Source frame:", self.frame, 0,
                                                  self.pkg.n_source - 1)
            if ok:
                self.seek(f)

        # -- selection -------------------------------------------------------
        def on_click(self, slot: int, x: float, y: float):
            self._selected_before_click = self.selected
            t = self.pkg.t(self.frame)
            if t is None:
                return
            view = self.order[slot]
            a, bx, by = self.view_tf[slot]
            sx, sy = (x - bx) / a, (y - by) / a      # clicked point in source pixels
            best, best_d = None, SELECT_RADIUS_SRC_PX
            for k in range(len(self.pkg.fish_ids)):
                p = self.pkg.fish_point(t, k, view)
                if np.isfinite(p).all():
                    d = float(np.hypot(p[0] - sx, p[1] - sy))
                    if d < best_d:
                        best, best_d = k, d
            self.selected = best
            self.render()

        # -- display ---------------------------------------------------------
        def make_large(self, slot: int):
            # Qt delivers a plain click first; a double-click must not change the selection.
            self.selected = self._selected_before_click
            if slot != 0:
                self.order[0], self.order[slot] = self.order[slot], self.order[0]
                self.render()

        def cycle_large(self):
            self.order = self.order[1:] + self.order[:1]
            self.render()

        def toggle_crop(self):
            self.crop_on = not self.crop_on
            self._crop.clear()
            self.statusBar().showMessage(f"Zoom onto the fish {'on' if self.crop_on else 'off'}.")
            self.render()

        def change_mark_scale(self, direction: int):
            self.mark_scale = float(np.clip(self.mark_scale * 1.15 ** direction, *MARK_SCALE_RANGE))
            self._style_info()
            self.behavior_row.fit_to_scale()
            self.render()

        def _style_info(self):
            size = round(12 * min(self.mark_scale, 2.0))
            self.info.setStyleSheet(f"font-size:{size}px; padding:2px 4px;")
            self.info.setFixedHeight(round(5.6 * size))

        def crop_box(self, v: int, t: int, width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
            """Crop around where the fish are within +-1 s; eased so it does not jump while playing."""
            pkg = self.pkg
            half = int(CROP_HALF_SECONDS * pkg.fps)
            idx = np.arange(max(0, t - half), min(len(pkg.frames), t + half + 1), 3)
            target = crop_target(pkg.centres(idx, v).reshape(-1, 2) * pkg.scale[v], width, height)
            prev = self._crop.get(v)
            if prev is None or abs(self.frame - self._crop_frame.get(v, self.frame)) > pkg.fps:
                state = target
            else:
                state = prev + CROP_SMOOTHING * (target - prev)
            self._crop[v], self._crop_frame[v] = state, self.frame
            return crop_rect(state, width, height, aspect)

        def mark_analysis_start(self):
            f = self.frame
            self.ledger.add("analysis_start", {ALL_FISH: "1"}, f, f)
            self._rebuild_states()
            secs = f / self.pkg.fps
            self.statusBar().showMessage(f"Analysis starts at frame {f} ({int(secs // 60):02d}:{secs % 60:05.2f}); "
                                         "earlier frames are not analysed. Ctrl+Z undoes.")
            self.render()

        def cycle_selection(self):
            n = len(self.pkg.fish_ids)
            self.selected = 0 if self.selected is None else (self.selected + 1) % n
            self.render()

        def clear_selection(self):
            if self.pending is not None:
                self.pending = None
                self.statusBar().showMessage("Per-view edit cancelled.")
            self.selected = None
            self.render()

        # -- edits -----------------------------------------------------------
        def _risk_id(self) -> str:
            i = self.pkg.risk_at(self.frame)
            return str(self.pkg.risks.iloc[i]["risk_id"]) if i is not None else ""

        def set_label(self, label: str, rename: bool = False):
            """Name the selected fish; if another fish already has the name, the tracker swapped them:
            exchange the two tracks from here on (``rename`` only renames, for the whole recording)."""
            pkg = self.pkg
            t = pkg.t(self.frame)
            if self.selected is None or t is None:
                self.statusBar().showMessage("Select a fish first (click it, or Tab)." if t is not None
                                             else "No tracking data at this frame.")
                return
            k = self.selected
            c = self.fish_of_track(k, t)
            cid = pkg.fish_ids[c]
            holder = next((f for f, v in self.labels.items() if v == label and f != cid), None)
            if rename or holder is None or label == "-":
                changes = assign_label(self.labels, cid, label)
                if not changes:
                    self.statusBar().showMessage(f"This fish is already {label}.")
                    return
                self.ledger.add("label", changes, self.frame, self.frame, self._risk_id())
                self._rebuild_states()
                text = ", ".join(f"{f} = {v or 'unnamed'}" for f, v in changes.items())
                self.statusBar().showMessage(f"Named for the whole recording: {text}")
                self.render()
                return
            self.pause()
            r = self.track_of_fish(pkg.fish_ids.index(holder), t)
            d = np.linalg.norm(pkg.xyz[:, k] - pkg.xyz[:, r], axis=1)
            frame, hint = self.frame, ""
            # Snap window: the frames since the previously shown one (stepping), or the last few seconds
            # (while playing the key comes after the reaction time). Swaps touching either track do not
            # commute with this one, so the window never reaches back past the latest of them.
            if self.last_nav == "play":
                lo = self.frame - int(SWAP_SNAP_SECONDS * pkg.fps)
            elif self.last_nav == "step" and self.prev_shown is not None and self.prev_shown < frame:
                lo = self.prev_shown
            else:
                lo = None
            touching = [f for f, a, b in self.swaps if f <= frame and {a, b} & {k, r}]
            if lo is not None:
                frame = snap_frame(pkg.frames, d, max([lo] + touching), frame)
            else:
                # Reached by a jump: point at the closest approach before here, where the swap likely happened.
                lo = max([pkg.first, self.frame - int(SWAP_HINT_SECONDS * pkg.fps)] + touching) - pkg.first
                seg = d[lo:t + 1]
                if np.isfinite(seg).any():
                    i = int(np.nanargmin(seg))
                    hint = (f" Closest approach of the two in the last {SWAP_HINT_SECONDS:.0f} s: frame "
                            f"{pkg.frames[lo + i]} ({seg[i]:.0f} mm) - if the swap happened there, Delete this "
                            f"one in the edit list (E) and press {label} there.")
            pair = PAIR_SEP.join(sorted((pkg.fish_ids[k], pkg.fish_ids[r])))
            later = [f for f, a, b in self.swaps if f > frame and {a, b} == {k, r}]
            group = self.ledger.add("swap", {pair: "1"}, frame, self.frame, self._risk_id())
            self._rebuild_states()
            if frame != self.frame and self.labels[pkg.fish_ids[self.fish_of_track(k, t)]] != label:
                # Safety net: placed earlier, the swap did not give the clicked fish the name here.
                self.ledger.remove_group(group, log=False)
                frame = self.frame
                self.ledger.add("swap", {pair: "1"}, frame, self.frame, self._risk_id())
                self._rebuild_states()
            snapped = f" (snapped from {self.frame})" if frame != self.frame else ""
            note = (f" Note: these two tracks are swapped again at frame {later[0]}; delete that one (E) if this "
                    "replaces it.") if later else ""
            self.statusBar().showMessage(f"From frame {frame}{snapped}: tracks {pair.replace(PAIR_SEP, ' and ')} "
                                         f"exchanged - the clicked fish is {label}.{note}{hint}")
            self.render()

        def flip_segment(self, from_here: bool):
            """Flip head/tail of the selected track over its orientation segment (or from here to its end)."""
            pkg = self.pkg
            t = pkg.t(self.frame)
            if self.selected is None or t is None:
                self.statusBar().showMessage("Select a fish first (click it, or Tab).")
                return
            k = self.selected
            if pkg.segments is None:
                a, b = self.frame, pkg.last                     # old package: from here on
            else:
                col = pkg.segments[:, k]
                seg = col[t]
                if seg < 0:
                    self.statusBar().showMessage("No head/tail position for this fish at this frame.")
                    return
                where = np.flatnonzero(col == seg)
                # The run of this id around the current frame (recomputed ranges get their own ids).
                breaks = np.flatnonzero(np.diff(where) > 30)
                starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(where) - 1]
                run = int(np.searchsorted(where[starts], t, side="right") - 1)
                where = where[starts[run]:ends[run] + 1]
                a, b = int(pkg.frames[where[0]]), int(pkg.frames[where[-1]])
                if from_here:
                    a = self.frame
            self.ledger.add("flip", {pkg.fish_ids[k]: "1"}, a, self.frame, self._risk_id(), frame_end=b)
            self._rebuild_states()
            self.statusBar().showMessage(f"Head/tail flipped on frames {a}-{b} ({(b - a + 1) / pkg.fps:.1f} s). "
                                         "Ctrl+Z undoes.")
            self.render()

        # -- keyboard selection ------------------------------------------------
        def select_label(self, label: str):
            """1/2/3: select the fish now named F/M1/M2 (before naming: track 1/2/3). While a view swap
            (T) waits for its second fish, picks that fish instead."""
            pkg = self.pkg
            t = pkg.t(self.frame)
            if t is None:
                return
            holder = next((f for f, v in self.labels.items() if v == label), None)
            k = (self.track_of_fish(pkg.fish_ids.index(holder), t) if holder is not None
                 else LABELS.index(label) if LABELS.index(label) < len(pkg.fish_ids) else None)
            if k is None:
                return
            if self.pending is not None and self.pending.get("await_other"):
                if k == self.pending["track"]:
                    self.statusBar().showMessage("That is the same fish - press 1/2/3 for the OTHER fish.")
                    return
                self.pending["other"] = k
                self.pending["await_other"] = False
                self.view_edit("view_swap")
                return
            self.selected = k
            self.render()

        # -- geometry (needs calibration + detections.npz: desktop) -------------
        def geometry(self) -> Geometry | None:
            """Load calibration and detections on first use; None (with a message) if unavailable."""
            if self._geometry is not None or self._geometry_error:
                return self._geometry
            pkg = self.pkg
            try:
                from scripts.reconstruction_3d.calibration import Calibration

                self.statusBar().showMessage("Loading calibration and detections ...")
                QtWidgets.QApplication.processEvents()
                cal = Calibration(str(pkg.manifest["sources"]["calibration_dir"]), backend="auto")
                dets = Detections(pkg.folder)
            except Exception as exc:  # noqa: BLE001 - laptop without the calibration stack, or old package
                self._geometry_error = (f"3D cannot be recomputed here ({exc!r}); the edit is saved and applied "
                                        "on the desktop by the export.")
                self.statusBar().showMessage(self._geometry_error)
                return None
            pkg.ensure_base()
            self._geometry = Geometry.from_manifest({**pkg.d, **pkg.base}, dets, cal, pkg.manifest)
            return self._geometry

        def refresh_geometry(self, lo: int, hi: int):
            """Restore frames lo..hi to the review data, then recompute every edited range touching them."""
            pkg = self.pkg
            geo = self.geometry()
            if geo is None:
                return
            first = pkg.first
            lo, hi = max(lo, pkg.first), min(hi, pkg.last)
            if lo > hi:
                return
            for key, arr in pkg.base.items():
                pkg.d[key][lo - first:hi - first + 1] = arr[lo - first:hi - first + 1]
            todo = [(a, b) for a, b in edited_ranges(self.ledger.events, self.solutions, pkg.fish_ids, pkg.frames)
                    if a <= hi and b >= lo]
            t0 = time.time()
            for a, b in todo:
                part = recompute(geo, self.ledger.events, self.solutions, a, b)
                for key, value in part.items():
                    if key in pkg.d:
                        pkg.d[key][a - first:b - first + 1] = value
            span = (min([lo] + [a for a, _ in todo]), max([hi] + [b for _, b in todo]))
            pkg.update_nearest(*span)
            self._rebuild_states()
            if todo:
                self.statusBar().showMessage(f"3D recomputed for frames {span[0]}-{span[1]} "
                                             f"({time.time() - t0:.1f} s).")

        def apply_all_geometry(self):
            """At start-up: recompute every range the reviewer has edited (if any), in shared windows -
            or load them from the cache written when the window was last closed, if nothing changed."""
            pkg = self.pkg
            ranges = edited_ranges(self.ledger.events, self.solutions, pkg.fish_ids, pkg.frames)
            if not ranges:
                return
            key = geometry_state_key(self.ledger.events, self.solutions, pkg.manifest)
            cached = load_geometry_cache(self.ledger_folder, key)
            if cached is not None:
                rows, arrays = cached
                pkg.ensure_base()
                for name, values in arrays.items():
                    if name in pkg.d:
                        pkg.d[name][rows] = values
                for a, b in ranges:
                    pkg.update_nearest(a, b)
                self._rebuild_states()
                self.statusBar().showMessage(f"Earlier edits applied ({len(ranges)} range(s), from the cache).")
                return
            geo = self.geometry()
            if geo is None:
                return
            t0 = time.time()
            self.statusBar().showMessage(f"Applying earlier edits: recomputing {len(ranges)} range(s) ...")
            QtWidgets.QApplication.processEvents()
            first = pkg.first
            for i, (a, b, part) in enumerate(recompute_many(geo, self.ledger.events, self.solutions, ranges)):
                for key, value in part.items():
                    if key in pkg.d:
                        pkg.d[key][a - first:b - first + 1] = value
                pkg.update_nearest(a, b)
                if i % 10 == 0:
                    QtWidgets.QApplication.processEvents()
            self._rebuild_states()
            self.save_geometry_cache()
            self.statusBar().showMessage(f"Earlier edits applied: {len(ranges)} range(s) recomputed in "
                                         f"{time.time() - t0:.0f} s.")

        def save_geometry_cache(self):
            """Store the recomputed rows so the next start-up loads them instead of recomputing."""
            pkg = self.pkg
            if pkg.base is None:
                return
            ranges = edited_ranges(self.ledger.events, self.solutions, pkg.fish_ids, pkg.frames)
            try:
                save_geometry_cache(self.ledger_folder, geometry_state_key(self.ledger.events, self.solutions,
                                                                           pkg.manifest), pkg.d, ranges, pkg.first)
            except OSError:
                pass

        def _geometry_range_of(self, rows: pd.DataFrame) -> tuple[int, int] | None:
            """Frames affected by removed/added geometry events."""
            spans = []
            for r in rows.itertuples(index=False):
                if r.kind in ("view_flip", "view_drop", "view_swap"):
                    spans.append(event_range(r._asdict(), self.pkg.last))
                elif r.kind in ("nodes_bad", "interval", "interval_del", "confirm"):
                    iv = self.current_interval(int(r.frame))
                    a = int(r.frame)
                    b = int(r.frame_end) if str(r.frame_end).strip() not in ("", "nan") else a
                    if iv is not None:
                        a, b = min(a, int(iv["frame_start"])), max(b, int(iv["frame_end"]))
                    if r.kind == "nodes_bad":           # from this fish's previous B to its next one
                        e = self.ledger.events
                        same = e[(e["kind"] == "nodes_bad") & (e["fish"] == r.fish)]
                        f_same = same["frame"].astype(int)
                        earlier, later = f_same[f_same < int(r.frame)], f_same[f_same > int(r.frame)]
                        a = int(earlier.max()) if len(earlier) else a
                        b = int(later.min()) if len(later) else self.pkg.last
                    spans.append((a, b))
            if not spans:
                return None
            return min(a for a, _ in spans), max(b for _, b in spans)

        # -- per-view edits (Q / R / T: press at the first and at the last wrong frame) ----
        def view_edit(self, kind: str):
            pkg = self.pkg
            if pkg.t(self.frame) is None:
                self.statusBar().showMessage("No tracking data at this frame.")
                return
            if self.selected is None:
                self.statusBar().showMessage("Select a fish first (1/2/3, click, or Tab).")
                return
            view = self.order[0]
            k = self.selected
            what = {"view_flip": "head/tail reversed", "view_drop": "left out of 3D",
                    "view_swap": "detections exchanged"}[kind]
            p = self.pending
            if p is not None and p.get("await_other"):
                self.statusBar().showMessage("Press 1/2/3 for the other fish (Esc cancels).")
                return
            if kind == "view_swap" and (p is None or p.get("kind") != kind or p.get("track") != k
                                        or p.get("view") != view):
                self.pending = {"kind": kind, "track": k, "view": view, "await_other": True}
                self.statusBar().showMessage(f"{VIEW_TITLES[view]}: exchange {pkg.fish_ids[k]}'s detections with "
                                             "which fish? Press 1/2/3 (Esc cancels).")
                return
            if p is None or p.get("kind") != kind or p.get("track") != k or p.get("view") != view:
                self.pending = {"kind": kind, "track": k, "view": view, "start": self.frame}
                self.statusBar().showMessage(f"{VIEW_TITLES[view]}, {pkg.fish_ids[k]}: {what} from frame {self.frame}. "
                                             f"Go to the last wrong frame and press {kind_key(kind)} again "
                                             "(Esc cancels).")
                self.render()
                return
            if "start" not in p:                         # view swap: second fish just chosen
                p["start"] = self.frame
                other = pkg.fish_ids[p["other"]]
                self.statusBar().showMessage(f"{VIEW_TITLES[view]}: {pkg.fish_ids[k]} and {other} exchanged from "
                                             f"frame {self.frame}. Go to the last wrong frame and press T again.")
                self.render()
                return
            a, b = sorted((p["start"], self.frame))
            if kind == "view_swap":
                fish = PAIR_SEP.join(sorted((pkg.fish_ids[k], pkg.fish_ids[p["other"]])))
            else:
                fish = pkg.fish_ids[k]
            self.pending = None
            self.ledger.add(kind, {fish: VIEWS[view]}, a, self.frame, self._risk_id(), frame_end=b)
            self._rebuild_states()
            self.statusBar().showMessage(f"{VIEW_TITLES[view]}, {fish}: {what} on frames {a}-{b}.")
            self.refresh_geometry(a, b)
            self.render()

        # -- intervals ---------------------------------------------------------
        def _new_interval_id(self) -> str:
            used = set(self.intervals["interval_id"]) | set(self.ledger.events["value"])
            i = 1
            while f"U{i:03d}" in used:
                i += 1
            return f"U{i:03d}"

        def _interval_members(self) -> str:
            pkg = self.pkg
            t = pkg.t(self.frame)
            if t is None or self.selected is None:
                return PAIR_SEP.join(pkg.fish_ids)
            k = self.selected
            d = np.linalg.norm(pkg.xyz[t] - pkg.xyz[t, k], axis=1)
            members = [j for j in range(len(pkg.fish_ids)) if j == k or np.nan_to_num(d[j], nan=1e9) < 25.0]
            if len(members) < 2:
                members = list(range(len(pkg.fish_ids)))
            return PAIR_SEP.join(pkg.fish_ids[j] for j in members)

        def set_interval_end(self, start: bool):
            """[ / ]: move the start / end of the current interval here (or begin a new one)."""
            pkg = self.pkg
            if pkg.t(self.frame) is None:
                self.statusBar().showMessage("No tracking data at this frame.")
                return
            iv = self.current_interval()
            sec = int(round(pkg.fps))
            if iv is None:
                a, b = (self.frame, min(pkg.last, self.frame + sec)) if start else (max(pkg.first, self.frame - sec),
                                                                                    self.frame)
                iid, fish, old = self._new_interval_id(), self._interval_members(), (a, b)
            else:
                iid, fish = str(iv["interval_id"]), str(iv["fish"])
                old = (int(iv["frame_start"]), int(iv["frame_end"]))
                a, b = (self.frame, old[1]) if start else (old[0], self.frame)
                if a >= b:
                    self.statusBar().showMessage("The start must be before the end.")
                    return
            self.ledger.add("interval", {fish: iid}, a, self.frame, self._risk_id(), frame_end=b)
            self._rebuild_states()
            self.statusBar().showMessage(f"Interval {iid} ({fish.replace(PAIR_SEP, ', ')}): frames {a}-{b}. "
                                         "Confirm both ends with C, then Y solves it.")
            self.refresh_geometry(min(a, old[0]), max(b, old[1]))
            self.render()

        def goto_interval(self, direction: int):
            iv = self.intervals
            lo = self.analysis_start or 0
            starts = iv["frame_start"].to_numpy()
            ok = iv["frame_end"].to_numpy() >= lo
            if direction > 0:
                idx = np.flatnonzero(ok & (starts > self.frame))
            else:
                cur = self.current_interval()
                ref = int(cur["frame_start"]) if cur is not None else self.frame
                idx = np.flatnonzero(ok & (starts < ref))[::-1]
            if not len(idx):
                self.statusBar().showMessage("No more intervals in that direction.")
                return
            r = iv.iloc[int(idx[0])]
            self.seek(max(int(r["frame_start"]), lo))
            self.statusBar().showMessage(f"Interval {r['interval_id']} ({r['fish'].replace(PAIR_SEP, ', ')}): "
                                         f"{r['frame_start']}-{r['frame_end']}. Home/End = first/last frame.")

        def goto_bout(self, direction: int):
            """Ctrl+Right / Ctrl+Left: start of the next / previous BORIS behaviour bout."""
            b = self.bouts
            if not len(b):
                self.statusBar().showMessage("No BORIS behaviour file for this experiment.")
                return
            starts = b["start"].to_numpy()
            idx = np.flatnonzero(starts > self.frame) if direction > 0 else np.flatnonzero(starts < self.frame)[::-1]
            if not len(idx):
                self.statusBar().showMessage("No more behaviour bouts in that direction.")
                return
            r = b.iloc[int(idx[0])]
            self.seek(int(r["start"]))
            self.statusBar().showMessage(f"Behaviour: {r['who']} {r['behavior']} ({r['category']}), frames "
                                         f"{r['start']}-{r['end']} ({(r['end'] - r['start'] + 1) / self.pkg.fps:.1f} s).")

        def goto_interval_end(self, start: bool):
            iv = self.current_interval()
            if iv is None:
                self.statusBar().showMessage("Not inside an interval (I / Shift+I jump to one).")
                return
            self.seek(int(iv["frame_start"] if start else iv["frame_end"]))

        def delete_interval(self):
            iv = self.current_interval()
            if iv is None:
                return
            answer = QtWidgets.QMessageBox.question(self, "Delete interval",
                                                    f"Delete interval {iv['interval_id']} "
                                                    f"({iv['frame_start']}-{iv['frame_end']})?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
            self.ledger.add("interval_del", {str(iv["fish"]): str(iv["interval_id"])}, int(iv["frame_start"]),
                            self.frame, frame_end=int(iv["frame_end"]))
            self._rebuild_states()
            self.refresh_geometry(int(iv["frame_start"]), int(iv["frame_end"]))
            self.render()

        def confirm_keyframe(self):
            """C: identities and head/tail are right at this frame (keyframe for the interval solver)."""
            pkg = self.pkg
            t = pkg.t(self.frame)
            if t is None or pkg.nodes is None:
                return
            heads = {}
            for k, fid in enumerate(pkg.fish_ids):
                n = pkg.nodes[t, k].astype(float)
                h, tl = (n[pkg.tail], n[pkg.head]) if self.flipped(k) else (n[pkg.head], n[pkg.tail])
                heads[fid] = (h, tl, self.flipped(k))
            iv = self.current_interval()
            had = iv is not None and str(iv["interval_id"]) in self.solutions
            self.ledger.add("confirm", {ALL_FISH: confirm_value(heads)}, self.frame, self.frame, self._risk_id())
            self._rebuild_states()
            if had and str(iv["interval_id"]) not in self.solutions:      # the old solution no longer applies
                self.refresh_geometry(int(iv["frame_start"]), int(iv["frame_end"]))
            msg = f"Keyframe at {self.frame}: identities and head/tail confirmed."
            if iv is not None:
                ready = interval_ready(self.keyframes, int(iv["frame_start"]), int(iv["frame_end"]), pkg.fps)
                msg += " Both ends confirmed - Y solves the interval." if ready else \
                    " Confirm the other end too (Home / End)."
            self.statusBar().showMessage(msg)
            self.render()

        def solve_current_interval(self):
            pkg = self.pkg
            iv = self.current_interval()
            if iv is None:
                self.statusBar().showMessage("Not inside an interval (I / Shift+I jump to one, [ ] make one).")
                return
            a, b = int(iv["frame_start"]), int(iv["frame_end"])
            if (b - a + 1) > MAX_SOLVE_SECONDS * pkg.fps:
                self.statusBar().showMessage(f"Interval longer than {MAX_SOLVE_SECONDS:.0f} s: split it (move its end "
                                             "with ] and start a new one with [) and solve the parts.")
                return
            if not interval_ready(self.keyframes, a, b, pkg.fps):
                self.statusBar().showMessage("Confirm identities and head/tail at the first and the last frame "
                                             "of the interval first: Home, check / fix, C; End, check / fix, C.")
                return
            geo = self.geometry()
            if geo is None:
                return
            self.pause()
            self.statusBar().showMessage(f"Solving interval {iv['interval_id']} ({(b - a + 1) / pkg.fps:.1f} s) ...")
            QtWidgets.QApplication.processEvents()
            t0 = time.time()
            before = self.fish_positions(a, b)
            solve(geo, self.ledger.events, iv, self.ledger_folder)
            self._rebuild_states()
            self.refresh_geometry(a, b)
            after = self.fish_positions(a, b)
            moved = np.nan_to_num(np.linalg.norm(after - before, axis=2)) > 3.0           # (F, K)
            swapped = identity_changes(before, after, self.interval_fish(iv))
            first_diff = f" (first at frame {a + int(np.flatnonzero(swapped)[0])})" if swapped.any() else ""
            self.statusBar().showMessage(
                f"Interval {iv['interval_id']} solved in {time.time() - t0:.1f} s: 3D moved > 3 mm on "
                f"{int(moved.any(axis=1).sum())} of {b - a + 1} frames; identity differs from before on "
                f"{int(swapped.sum())} frames{first_diff}. Check those; where it is wrong, fix the identity, "
                "press C, then Y again.")
            self.render()

        def interval_fish(self, iv: dict) -> list[int]:
            """Corrected fish of an interval's member tracks at its first frame."""
            t = self.pkg.t(int(iv["frame_start"]))
            return [self.fish_of_track(self.pkg.fish_ids.index(f), t) for f in str(iv["fish"]).split(PAIR_SEP)
                    if f in self.pkg.fish_ids]

        def fish_positions(self, a: int, b: int) -> np.ndarray:
            """3D centroid of every corrected fish on frames a..b ``(F, K, 3)``."""
            t = np.arange(a - self.pkg.first, b - self.pkg.first + 1)
            return self.pkg.xyz[t[:, None], np.argsort(self.owner[t], axis=1)].astype(float)

        # -- edit list -------------------------------------------------------
        def _fill_event_list(self):
            ev = self.ledger.events
            lst = self.event_list
            lst.clear()
            if ev.empty:
                return
            ev = ev.assign(_f=ev["frame"].astype(int), _i=np.arange(len(ev)))
            parts = {g: ", ".join(f"{f}={v}" if kind in ("label",) + FLAG_KINDS else f
                                  for f, v, kind in zip(rows["fish"], rows["value"], rows["kind"]))
                     for g, rows in ev.groupby("group", sort=False)}
            for _, row in ev.sort_values(["_f", "_i"], kind="stable").drop_duplicates("group").iterrows():
                what = parts[row["group"]]
                end = f"-{row['frame_end']}" if str(row["frame_end"]).strip() else ""
                item = QtWidgets.QListWidgetItem(f"{int(row['frame']):>7}{end}  {row['kind']:<14} {what}")
                item.setData(QtCore.Qt.UserRole, row["group"])
                item.setData(QtCore.Qt.UserRole + 1, int(row["frame"]))
                lst.addItem(item)

        def toggle_event_list(self):
            self.event_dock.setVisible(not self.event_dock.isVisible())

        def seek_from_list(self, frame: int):
            self.seek(frame)
            self.setFocus()

        def remove_event_group(self, group: str):
            rows = self.ledger.events[self.ledger.events["group"] == group]
            if rows.empty:
                return
            r = rows.iloc[0]
            answer = QtWidgets.QMessageBox.question(
                self, "Remove edit", f"Remove {r['kind']} at frame {r['frame']} ({', '.join(rows['fish'])})?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
            span = self._geometry_range_of(rows)
            self.ledger.remove_group(group)
            self._rebuild_states()
            if span is not None:
                self.refresh_geometry(*span)
            self.statusBar().showMessage(f"Removed {r['kind']} at frame {r['frame']} "
                                         "(kept in human_events_removed.csv).")
            self.render()

        def toggle(self, kind: str, allow_all: bool = False):
            if self.selected is None and not allow_all:
                self.statusBar().showMessage("Select a fish first (1/2/3, click, or Tab).")
                return
            targets = [self.pkg.fish_ids[self.selected]] if self.selected is not None else self.pkg.fish_ids
            new = "0" if all(self.state(kind, f) == "1" for f in targets) else "1"
            opened = None
            if kind == "nodes_bad" and new == "0":
                e = self.ledger.events
                ons = e[(e["kind"] == kind) & (e["fish"] == targets[0]) & (e["value"] == "1")]
                ons = ons[ons["frame"].astype(int) <= self.frame]
                opened = int(ons["frame"].astype(int).max()) if len(ons) else None
            # B is pressed at the first and at the LAST wrong frame: the run ends after the current frame.
            at = self.frame + 1 if kind == "nodes_bad" and new == "0" else self.frame
            self.ledger.add(kind, {f: new for f in targets}, at, self.frame, self._risk_id())
            self._rebuild_states()
            if kind == "nodes_bad":
                msg = (f"Body of {targets[0]} wrong from frame {self.frame}: press B again at the last wrong frame."
                       if new == "1" else f"Body of {targets[0]} re-estimated on frames {opened}-{self.frame}.")
                self.statusBar().showMessage(msg)
                if opened is not None:
                    self.refresh_geometry(opened, self.frame + 1)
            else:
                self.statusBar().showMessage(f"From frame {self.frame}: {kind} = {new} for {', '.join(targets)}")
            self.render()

        def undo(self):
            removed = self.ledger.undo()
            self._rebuild_states()
            if removed.empty:
                self.statusBar().showMessage("Nothing to undo in this session.")
            else:
                r = removed.iloc[0]
                span = self._geometry_range_of(removed)
                if span is not None:
                    self.refresh_geometry(*span)
                self.statusBar().showMessage(f"Undid {r['kind']} at frame {r['frame']} ({len(removed)} row(s)).")
            self.render()

        def toggle_auto_slow(self):
            self.auto_slow = not self.auto_slow
            if self.direction:
                self.flush_run()
                self._anchor()
                self.open_run("play" if self.direction > 0 else "reverse")
            self.render()

        def toggle_overlays(self):
            self.overlays = not self.overlays
            self.render()

        def show_help(self):
            rows = "".join(f"<tr><td><b>{k}</b></td><td>&nbsp;{v}</td></tr>" for k, v in KEY_HELP)
            QtWidgets.QMessageBox.information(self, "Keys", f"<table>{rows}</table>")

        # -- drawing -----------------------------------------------------------
        def fish_style(self, k: int) -> tuple[tuple[int, int, int], str]:
            """Colour and on-screen name of track ``k`` at the current frame (by the fish it carries)."""
            fid = self.pkg.fish_ids[k]
            c = self.fish_of_track(k)
            label = self.labels[self.pkg.fish_ids[c]]
            name = label if label else f"#{self.pkg.fish_ids[c].split('_')[-1]}"
            if self.state("uncertain", fid) == "1":
                name += "?"
            if self.state("bad3d", fid) == "1":
                name += " x3D"
            if self.state("nodes_bad", fid) == "1":
                name += " xN"
            return self.fish_color(c), name

        def draw_view(self, img: np.ndarray, v: int, t: int, tf: tuple[float, float, float]):
            """Overlays for view ``v`` on the displayed image; ``tf`` maps source pixels to it.

            The cross (projected 3D position) is always drawn; the skeleton only
            where the fish was detected in this view, dashed where it overlaps
            another fish, and a dashed ring around the cross where it was not
            detected.
            """
            pkg, m = self.pkg, self.mark_scale
            lw = max(1, round(1.6 * m))
            r_head, r_node, r_sel, r_cross = round(4.5 * m), max(2, round(2.5 * m)), round(20 * m), round(9 * m)
            fs, ft = 0.45 * m, max(1, round(1.1 * m))
            for k in range(len(pkg.fish_ids)):
                color, name = self.fish_style(k)
                pts = pkg.kp2d[t, k, v].astype(float).copy()
                if self.flipped(k):
                    pts[[pkg.head, pkg.tail]] = pts[[pkg.tail, pkg.head]]
                ok = np.isfinite(pts).all(axis=1)
                if v == 0:
                    self.draw_trail(img, k, t, color, tf)
                proj = pkg.proj2d[t, k, v].astype(float)
                pq = to_display(proj, tf) if np.isfinite(proj).all() else None
                interp = bool(pkg.interp[t, k])
                if not ok.any() and pq is None:
                    continue
                if pkg.missing_proj is not None:
                    # Nodes this view did not detect: projection of their 3D position (thin dashes, squares).
                    mp = pkg.missing_proj[t, k, v].astype(float).copy()
                    if self.flipped(k):
                        mp[[pkg.head, pkg.tail]] = mp[[pkg.tail, pkg.head]]
                    disp = np.where(ok[:, None], pts, mp)
                    okd = np.isfinite(disp).all(axis=1)
                    qd = to_display(disp, tf)
                    for a, b in pkg.edges:
                        if okd[a] and okd[b] and not (ok[a] and ok[b]):
                            dashed_line(img, qd[a], qd[b], color, 1, 3 * m)
                    for n in np.flatnonzero(okd & ~ok):
                        r = (r_head if n == pkg.head else r_node) + 1
                        c0 = pt(qd[n])
                        cv2.rectangle(img, (c0[0] - r, c0[1] - r), (c0[0] + r, c0[1] + r), color, 1, cv2.LINE_AA)
                if pq is not None:
                    cross_color = INTERP_COLOR if interp else color
                    if ok.any() and np.linalg.norm(pts[ok].mean(axis=0) - proj) > OFF_PROJECTION_SRC_PX:
                        cv2.line(img, pt(to_display(pts[ok].mean(axis=0), tf)), pt(pq), color, 1, cv2.LINE_AA)
                    cv2.drawMarker(img, pt(pq), (0, 0, 0), cv2.MARKER_CROSS, r_cross + 2, lw + 2, cv2.LINE_AA)
                    cv2.drawMarker(img, pt(pq), cross_color, cv2.MARKER_CROSS, r_cross, lw, cv2.LINE_AA)
                if ok.any():
                    q = to_display(pts, tf)          # rows outside ``ok`` are NaN and never drawn
                    overlapped = bool(pkg.overlap[t, k, v])
                    for a, b in pkg.edges:
                        if ok[a] and ok[b]:
                            if overlapped:
                                dashed_line(img, q[a], q[b], color, lw, 4 * m)
                            else:
                                cv2.line(img, pt(q[a]), pt(q[b]), color, lw, cv2.LINE_AA)
                    fill = max(1, lw) if overlapped else -1   # hollow nodes: may sit on the other fish
                    for n in np.flatnonzero(ok):
                        if n == pkg.head:
                            cv2.circle(img, pt(q[n]), r_head + 1, (0, 0, 0), -1 if fill < 0 else fill + 2, cv2.LINE_AA)
                            cv2.circle(img, pt(q[n]), r_head, color, fill, cv2.LINE_AA)
                        else:
                            cv2.circle(img, pt(q[n]), r_node, color, fill, cv2.LINE_AA)
                    anchor = q[pkg.head] if ok[pkg.head] else q[ok][0]
                    center = q[ok].mean(axis=0)
                    if overlapped:
                        name += " ~"
                else:
                    center = anchor = pq
                    ring = r_cross + round(5 * m)
                    n_dash = 12
                    for i in range(0, n_dash, 2):
                        cv2.ellipse(img, pt(pq), (ring, ring), 0, i * 360 / n_dash, (i + 1) * 360 / n_dash,
                                    color, max(1, lw // 2), cv2.LINE_AA)
                    name += " interp" if interp else " no 2D"
                if self.selected == k:
                    cv2.circle(img, pt(center), r_sel, (255, 255, 255), lw, cv2.LINE_AA)
                org = (int(anchor[0]) + r_head + 3, int(anchor[1]) - r_head - 3)
                cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft + 2, cv2.LINE_AA)
                cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft, cv2.LINE_AA)

        def draw_trail(self, img: np.ndarray, k: int, t: int, color, tf: tuple[float, float, float]):
            n = int(TRAIL_SECONDS * self.pkg.fps)
            idx = np.arange(max(0, t - n), t + 1, 3)
            if len(idx) < 2:
                return
            q = to_display(self.pkg.centres(idx, 0)[:, k], tf)
            ok = np.isfinite(q).all(axis=1)
            for a, b in zip(range(len(idx) - 1), range(1, len(idx))):
                if ok[a] and ok[b]:
                    cv2.line(img, pt(q[a]), pt(q[b]), color, 1, cv2.LINE_AA)

        def render(self):
            pkg = self.pkg
            f = self.frame
            t = pkg.t(f)
            for slot, label in enumerate(self.views):
                v = self.order[slot]
                img = (self.play[v].get(f // pkg.step) if f % pkg.step == 0 else self.full[v].get(f))
                if img is None:
                    img = np.zeros((pkg.proxy["views"][VIEWS[v]]["proxy_height"],
                                    pkg.proxy["views"][VIEWS[v]]["proxy_width"], 3), np.uint8)
                h, w = img.shape[:2]
                lw_, lh_ = max(50, label.width()), max(50, label.height())
                cropped = self.crop_on and t is not None
                x0, y0, cw, ch = self.crop_box(v, t, w, h, lw_ / lh_) if cropped else (0, 0, w, h)
                # Render at the on-screen size so marks and text keep their size in screen pixels.
                scale = min(lw_ / cw, lh_ / ch)
                out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (max(1, round(cw * scale)), max(1, round(ch * scale))),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                tf = (pkg.scale[v] * scale, -x0 * scale, -y0 * scale)
                self.view_tf[slot] = tf
                if self.overlays and t is not None:
                    self.draw_view(out, v, t, tf)
                zoom = scale / min(lw_ / w, lh_ / h)
                title = VIEW_TITLES[v] + (f"  x{zoom:.1f}" if zoom > 1.05 else "")
                fs = 0.5 * self.mark_scale
                org = (8, round(28 * fs) + 6)
                cv2.putText(out, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (230, 230, 230), 1, cv2.LINE_AA)
                label.show_image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if f != self.shown:
                self.prev_shown, self.shown = self.shown, f
            self.update_info(t)
            self.overview.update()
            self.zoom.update()
            self.behavior_row.update()
            self.traces.update()
            if self.view3d_dock.isVisible():
                self.view3d.update()

        def update_info(self, t: int | None):
            pkg, f = self.pkg, self.frame
            secs = f / pkg.fps
            eff = self.effective_speed()
            state = {0: "paused", 1: "playing", -1: "playing backwards"}[self.direction]
            speed_txt = f"{eff:g}x" + (" (auto-slow)" if eff != self.speed else "")
            start = self.analysis_start
            start_txt = ("analysis start not marked (S)" if start is None else
                         f"analysis from frame {start} ({int(start / pkg.fps // 60):02d}:{start / pkg.fps % 60:05.2f})")
            col1 = [f"<b>frame {f}</b> / {pkg.n_source - 1} &nbsp; {int(secs // 60):02d}:{secs % 60:05.2f}",
                    f"{state}, {speed_txt}, auto-slow {'on' if self.auto_slow else 'off'}",
                    start_txt]
            col2 = []
            for k, fid in enumerate(pkg.fish_ids):
                color, name = self.fish_style(k)
                sel = " &larr;" if self.selected == k else ""
                extra = []
                if t is not None:
                    c = pkg.fish_ids[self.fish_of_track(k, t)]
                    if c != fid:
                        extra.append(f"fish {c} (swapped)")
                    n_views = int(pkg.present[t, k].sum())
                    if pkg.interp[t, k]:
                        extra.append("<span style='color:#999'>interpolated</span>")
                    elif n_views < 3:
                        extra.append(f"<span style='color:#d9822b'>{n_views} views</span>")
                    if self.flipped(k):
                        extra.append("head/tail flipped")
                    if pkg.node_source is not None:
                        names = [str(x) for x in pkg.d["node_names"]]
                        kinds = {1: "1 view", 2: "body model"}
                        filled = [f"{names[n]} {kinds[s]}" for n, s in enumerate(pkg.node_source[t, k]) if s in kinds]
                        if filled:
                            extra.append("<span style='color:#b0b0ff'>" + ", ".join(filled) + "</span>")
                info = f" <small>{', '.join(extra)}</small>" if extra else ""
                col2.append(f"<span style='color:{bgr_hex(color)}'>&#9632;</span> track {fid}: <b>{name}</b>"
                            f"{info}{sel}")
            ri = pkg.risk_at(f)
            col3 = []
            if start is not None and f < start:
                col3.append("<span style='color:#e05555'><b>before the analysis start - not analysed</b></span>")
            if ri is not None:
                r = pkg.risks.iloc[ri]
                col3.append(f"<span style='color:#d9822b'><b>{r['risk_id']}</b> {r['types']}</span>")
                col3.append(f"<small>{str(r['details'])[:120]}</small>")
            if len(self.bouts):
                now = self.bouts[(self.bouts["start"] <= f) & (self.bouts["end"] >= f)]
                for r in now.itertuples(index=False):
                    col3.append(f"<span style='color:{BEHAVIOR_COLORS[r.category]}'>&#9632;</span> behaviour "
                                f"<b>{r.who} {r.behavior}</b> ({r.category}, {(r.end - r.start + 1) / pkg.fps:.1f} s)")
            iv = self.current_interval()
            if iv is not None:
                a, b = int(iv["frame_start"]), int(iv["frame_end"])
                kf = [k for k in self.keyframes if a <= k <= b]
                state = ("solved" if str(iv["interval_id"]) in self.solutions else
                         "ready - Y solves" if interval_ready(self.keyframes, a, b, pkg.fps) else
                         "confirm both ends (Home/End, C)")
                col3.append(f"<span style='color:#c77dff'><b>interval {iv['interval_id']}</b> "
                            f"{str(iv['fish']).replace(PAIR_SEP, ', ')} {a}-{b} ({(b - a + 1) / pkg.fps:.1f} s), "
                            f"{len(kf)} keyframe(s), {state}</span>")
            if self.pending is not None:
                p = self.pending
                col3.append(f"<span style='color:#66d9a8'><b>{kind_key(p['kind'])}</b> {p['kind']} "
                            f"{pkg.fish_ids[p['track']]} in {VIEW_TITLES[p['view']]}"
                            + (f" from {p['start']}" if "start" in p else " - choose the other fish (1/2/3)")
                            + "</span>")
            if t is None:
                col3.append("<i>no tracking data for this frame</i>")
            lo = start or 0
            frac = covered_fraction(self.watched, lo, pkg.n_source - 1)
            frac_data = covered_fraction(self.watched, max(pkg.first, lo), pkg.last)
            scope = "analysed video" if start is not None else "video"
            col4 = [f"watched {100 * frac:.1f}% of {scope}", f"({100 * frac_data:.1f}% of tracked range)",
                    f"edits {len(self.ledger.events)} ({self.ledger.events_added} this session) &nbsp; F1 = keys"]
            cells = "".join(f"<td style='padding-right:28px; vertical-align:top'>{'<br>'.join(c)}</td>"
                            for c in (col1, col2, col3, col4))
            self.info.setText(f"<table><tr>{cells}</tr></table>")

        def closeEvent(self, event):
            self.pause()
            self.flush_run()
            self.save_geometry_cache()
            if self.settings is not None:
                self.settings.setValue("mark_scale", self.mark_scale)
                self.settings.setValue("crop_on", "1" if self.crop_on else "0")
                self.settings.setValue("view_order", ",".join(map(str, self.order)))
                self.settings.setValue("events_visible", "1" if self.event_dock.isVisible() else "0")
                self.settings.setValue("view3d_visible", "1" if self.view3d_dock.isVisible() else "0")
            self.ledger.close()
            self.lock.release()
            for p in self.play + self.full:
                p.close()
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    lock = ReviewLock(Path(ledger_dir) if ledger_dir else Path(folder))
    holder = lock.holder()
    if holder and not smoke_test:
        answer = QtWidgets.QMessageBox.question(
            None, "Review folder in use",
            f"This folder was opened on {holder.get('host')} since {holder.get('since')} and not closed.\n"
            "Opening it on two computers at once can lose edits. Open anyway?")
        if answer != QtWidgets.QMessageBox.Yes:
            return 1
    win = MainWindow()
    win.resize(1500, 900)
    if smoke_test:
        win.show()
        return _smoke(app, win)
    install_error_log(win.ledger_folder / ERROR_LOG, win)
    win.showMaximized()
    # Docks get their size once the window is laid out: give the 3D window a useful width.
    QtCore.QTimer.singleShot(0, lambda: win.resizeDocks([win.view3d_dock], [max(520, win.width() // 4)],
                                                        QtCore.Qt.Horizontal))
    # Ranges edited in earlier sessions (per-view fixes, B, solved intervals) are recomputed once shown.
    QtCore.QTimer.singleShot(300, lambda: (win.apply_all_geometry(), win.render()))
    return app.exec_()


def install_error_log(path: Path, win=None) -> None:
    """Keep the GUI alive on Python errors and record them.

    PyQt aborts the whole program on an exception raised inside a Qt callback
    (key, paint, timer) unless ``sys.excepthook`` is replaced. Errors are
    appended to ``path`` with a traceback and shown in the status bar; native
    crashes are recorded there by :mod:`faulthandler`.
    """
    import faulthandler
    import traceback

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        frame = getattr(win, "frame", None)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} frame {frame}\n{text}\n")
        except OSError:
            pass
        if sys.__stderr__ is not None:
            print(text, file=sys.__stderr__, flush=True)
        if win is not None:
            try:
                win.statusBar().showMessage(f"ERROR at frame {frame} - logged to {path.name}; your edits are saved, "
                                            f"you can keep working: {value!r}")
            except Exception:  # noqa: BLE001 - never raise from the hook
                pass

    sys.excepthook = hook
    try:
        install_error_log.fault_file = open(path, "a", encoding="utf-8")   # kept open for native crashes
        faulthandler.enable(install_error_log.fault_file)
    except OSError:
        pass


def _smoke(app, win) -> int:
    """Exercise the main code paths without a human, then close."""
    from PyQt5 import QtCore

    pkg = win.pkg
    win.seek(pkg.first // pkg.step * pkg.step)
    app.processEvents()
    for k, label in enumerate(("F", "M1", "M2")[:len(pkg.fish_ids)]):
        win.selected = k
        win.set_label(label)                  # first naming: whole recording, no swap
    if sum(1 for e in win.ledger.events["kind"] if e == "swap"):
        raise AssertionError("first naming wrote a swap")
    win.step_by(pkg.step)
    win.step_by(1)
    win.step_by(-1)
    win.selected = 0
    win.flip_segment(False)
    win.toggle("nodes_bad")
    if len(pkg.fish_ids) >= 2:
        win.set_label("M1")                   # name taken: tracks 0 and 1 swapped from here
        t = pkg.t(win.frame)
        if win.labels[pkg.fish_ids[win.fish_of_track(0, t)]] != "M1":
            raise AssertionError("swap did not give the clicked track M1")
        win.undo()
    # Keyboard selection, intervals, keyframes, per-view edits (without calibration they are saved only).
    win.select_label("F")
    if win.selected is None or win.labels[pkg.fish_ids[win.fish_of_track(win.selected)]] != "F":
        raise AssertionError("1 did not select the fish named F")
    win.set_interval_end(start=True)
    win.set_interval_end(start=False)
    win.goto_interval(-1)
    win.goto_interval_end(start=True)
    win.confirm_keyframe()
    win.goto_interval_end(start=False)
    win.confirm_keyframe()
    win.solve_current_interval()
    win.view_edit("view_flip")
    win.step_by(pkg.step)
    win.view_edit("view_flip")
    win.view_edit("view_swap")
    win.select_label("M1")
    win.view_edit("view_swap")
    win.clear_selection()
    if len(win.bouts):                       # behaviour row: navigation and painting
        win.seek(pkg.first)
        win.goto_bout(+1)
        win.behavior_row.repaint()
        win.overview.repaint()
        win.goto_bout(-1)
    win.view3d_dock.setVisible(True)
    for preset in range(len(win.view3d.PRESETS)):
        win.view3d.preset, win.view3d.zoom = preset, 1.0 + preset
        _, win.view3d.az, win.view3d.el = win.view3d.PRESETS[preset]
        win.view3d.repaint()
    win.toggle_event_list()
    win._fill_event_list()
    if win.event_list.count() == 0:
        raise AssertionError("edit list is empty")
    group = win.ledger.events.iloc[-1]["group"]
    win.ledger.remove_group(group)            # what Delete in the edit list does after confirming
    win._rebuild_states()
    win.toggle_event_list()
    # Display: large view, zoom, mark size; a click on fish 0 in the large view selects it.
    win.make_large(1)
    win.cycle_large()
    win.change_mark_scale(+1)
    win.toggle_crop()
    win.toggle_crop()
    app.processEvents()
    t = pkg.t(win.frame)
    view = win.order[0]
    p = to_display(pkg.fish_point(t, 0, view), win.view_tf[0])
    win.selected = None
    win.on_click(0, float(p[0]), float(p[1]))
    if win.selected != 0:
        raise AssertionError(f"click on fish 0 selected {win.selected}")
    win.mark_analysis_start()
    win.goto_unwatched()
    win.goto_risk(+1)
    win.toggle_play()
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(1500, loop.quit)
    loop.exec_()
    win.pause()
    win.close()
    print(f"smoke test ok: frame {win.frame}, events {len(win.ledger.events)}, coverage rows {len(win.ledger.coverage)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", type=Path, help="<review_root>/<exp>/<round_tag> (asks if omitted)")
    p.add_argument("--ledger-dir", type=Path, default=None, help="Write the ledger elsewhere (default: the folder).")
    p.add_argument("--backup-dir", type=Path, default=None, help="Also copy the ledger here after every save.")
    p.add_argument("--boris", type=Path, default=None,
                   help="BORIS behaviour CSV, or a folder holding <experiment>.csv (default: the paths config's "
                        "boris_events_dir and the repository's archive/boris).")
    p.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    folder = args.folder
    if folder is None:
        from PyQt5 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        chosen = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose <experiment>/<round_tag> review folder")
        if not chosen:
            return 1
        folder = Path(chosen)
        del app
    if not (folder / "review_data.npz").exists() or not (folder.parent / "proxy" / "proxy_manifest.json").exists():
        raise SystemExit(f"{folder} is not a review folder (needs review_data.npz and ../proxy/proxy_manifest.json)")
    return run_app(folder, ledger_dir=args.ledger_dir, backup_dir=args.backup_dir, smoke_test=args.smoke_test,
                   boris=args.boris)


if __name__ == "__main__":
    sys.exit(main())
