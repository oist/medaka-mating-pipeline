"""Judge GUI: grade old and new 3D reconstructions frame by frame, fish by fish (grades and machine hints:
:mod:`scripts.review.recon_judge`).

::

    python -m scripts.review.recon_judge_gui <reference folder> --recon-dir <reference>/recon_reviewed_old_m3dfull

Watch the whole recording (it plays straight through; the machine hints in the list and on the timeline
are only suggestions where to look). What is graded is the **3D reconstruction**: the large 3D view (old
dashed, new solid, one colour per fish name, green where both put a part on the same spot, grey floor
shadows, trails, the old-new distance of every node in mm). The camera views show the same 3D nodes
projected into the video - an aid to see whether the 3D is where the fish really is - and, in magenta,
every raw SLEAP instance of that frame, unfiltered (``slp_raw.npz``, built by :mod:`scripts.review.slp_raw`:
two when SLEAP found two, six when it found six; SLEAP's skeleton edges as lines, a marker shape per body
part - triangle mouth, square pectoral, x tail; B hides them). The reviewed result is hidden by default (V shows it as a thin white skeleton). Every panel border
can be dragged.

**Grading.** Each version is graded on its own - correct / small error / clear error / unsure - for its
3D IDs and for its 3D body parts; which version is better follows (old correct + new small error = old
better, by a little). A grade goes to the selected fish (1 / 2 / 3, Ctrl adds, 4 = all; a click on a fish in
the 3D or a camera view, or on its button, selects or deselects it)
on the selected frames: drag on the timeline, or press [ where it starts and grade when it ends - the
range runs to the current frame, and after grading it starts again there, so you can grade while you
watch. Without a range a grade covers the current frame. A later grade paints over an earlier one. The
timeline shows every fish's grades (per fish: ID row and body row, each old over new).

Keys:
  Space play/pause   Left/Right 1 frame   Shift+Left/Right 1 s   PageUp/PageDown 10 s   Up/Down speed
  N / P next / previous machine hint   Enter next frame not fully graded (selected fish)   G go to frame
  Fish:   1 F  2 M1  3 M2 only that fish   Ctrl+1/2/3 or a click on the fish: select / deselect   4 all
  Frames: drag on the timeline   [ start a range here (it runs to the current frame)   ] end it here
          \\ or Esc no range (current frame)   L loop the range
  3D IDs        old: Q correct  W small error  E clear error     new: R correct  T small error  Y clear error
  3D body parts old: A correct  S small error  D clear error     new: F correct  G small error  H clear error
  U IDs unsure   J body unsure   O everything correct (both versions, IDs and body)
  Backspace erase the selection's grades   C note on the range
  Timeline: wheel zoom, right-drag or Shift+wheel move, = / - zoom, Ctrl+wheel or Ctrl+= / Ctrl+- text size
            (drag the border above the timeline for taller rows)
  3D view: left-drag rotate, right- / middle- / Shift+left-drag move, wheel zoom (towards the mouse),
           double-click whole tank, 7 top 8 front 9 side 0 oblique, K trails, X follow the selected fish
  M both versions / only old / only new   B raw SLEAP points   V reviewed result   I own named tracks / matched per frame   Z zoom (camera views)
  F1 this help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.review.compare_gui import ReconCompareData, draw_pair, label_color, project, render_view
from scripts.review.recon_compare import NODE_INTERPOLATED
from scripts.review.recon_judge import ASPECTS, GRADE_CODES, GRADES, JUDGED, JudgeState
from scripts.review.review_gui import VIEWS, crop_target, pt
from scripts.review.slp_raw import RawSlp

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TICK_HZ = 30.0
KEYS = {("id", "old"): "QWE", ("id", "new"): "RTY", ("body", "old"): "ASD", ("body", "new"): "FGH"}
UNSURE_KEYS = {"id": "U", "body": "J"}
GRADE_RGB = {"ok": (150, 200, 150), "minor": (255, 193, 7), "major": (229, 57, 53), "unsure": (150, 150, 150)}
ASPECT_TITLES = {"id": "3D IDs", "body": "3D body parts"}
GRADED_RGB = (165, 205, 165)    # overview: frames graded for every fish (light, low saturation)
PARTLY_RGB = (95, 120, 95)       # overview: frames with some grades (not every fish / version / aspect)
HINT_RGB = {"identity": (235, 70, 70), "body": (225, 150, 40)}
REF_COLOR = (255, 255, 255)
RAW_COLOR = (230, 60, 230)      # BGR magenta: raw SLEAP instances
RAW_MARKERS = {"mouth": cv2.MARKER_TRIANGLE_UP, "pectoral": cv2.MARKER_SQUARE, "tail": cv2.MARKER_TILTED_CROSS}
RAW_OTHER_MARKERS = (cv2.MARKER_DIAMOND, cv2.MARKER_STAR, cv2.MARKER_CROSS, cv2.MARKER_TRIANGLE_DOWN)
RAW_MARKER_PX, RAW_LINE_PX = 13, 2
SELECT_COLOR = (255, 255, 255)
PICK_PX = 25.0                  # a click this close to a fish's node selects it
SORTS = {"time": ("start", True), "old vs new (mm)": ("old_new_mm", False), "duration": ("dur", False),
         "largest jump": ("jump", False)}
MIN_MM = (0, 5, 10, 20, 50)
ERROR_LOG = "recon_judge_gui_errors.log"


def draw_reference(img: np.ndarray, q: np.ndarray, edges, mark: float) -> None:
    """The reference skeletons ``(L, N, 2)`` (display pixels), thin white."""
    for c in range(len(q)):
        ok = np.isfinite(q[c]).all(axis=1)
        for i, j in edges:
            if ok[i] and ok[j]:
                cv2.line(img, pt(q[c, i]), pt(q[c, j]), REF_COLOR, 1, cv2.LINE_AA)
        for n in np.flatnonzero(ok):
            cv2.circle(img, pt(q[c, n]), round(2 * mark), REF_COLOR, 1, cv2.LINE_AA)


def raw_marker(names: list[str], n: int) -> int:
    """The OpenCV marker of node ``n`` (triangle mouth, square pectoral, x tail, others in turn)."""
    if names[n] in RAW_MARKERS:
        return RAW_MARKERS[names[n]]
    others = [k for k in names if k not in RAW_MARKERS]
    return RAW_OTHER_MARKERS[others.index(names[n]) % len(RAW_OTHER_MARKERS)]


def draw_raw(img: np.ndarray, q: np.ndarray, names: list[str], edges) -> int:
    """Raw SLEAP instances ``(S, N, 2)`` (display pixels): SLEAP's skeleton edges as magenta lines, a marker
    per body part (black outline for contrast). Returns how many instances were drawn."""
    drawn = 0
    for d in q:
        ok = np.isfinite(d).all(axis=1)
        if not ok.any():
            continue
        drawn += 1
        for i, j in edges:
            if ok[i] and ok[j]:
                cv2.line(img, pt(d[i]), pt(d[j]), RAW_COLOR, RAW_LINE_PX, cv2.LINE_AA)
        for n in np.flatnonzero(ok):
            m = raw_marker(names, n)
            cv2.drawMarker(img, pt(d[n]), (0, 0, 0), m, RAW_MARKER_PX, RAW_LINE_PX + 2, cv2.LINE_AA)
            cv2.drawMarker(img, pt(d[n]), RAW_COLOR, m, RAW_MARKER_PX, RAW_LINE_PX, cv2.LINE_AA)
    return drawn


def draw_raw_legend(img: np.ndarray, count: int, names: list[str], y: int) -> None:
    """``SLEAP raw: count`` and each body part's marker, from ``(8, y)``."""
    def text(s, x):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RAW_COLOR, 1, cv2.LINE_AA)
        return x + cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]

    x = text(f"SLEAP raw: {count}", 8) + 12
    for n, name in enumerate(names):
        c = (x + 7, y - 5)
        cv2.drawMarker(img, c, (0, 0, 0), raw_marker(names, n), 11, 4, cv2.LINE_AA)
        cv2.drawMarker(img, c, RAW_COLOR, raw_marker(names, n), 11, 2, cv2.LINE_AA)
        x = text(name, x + 17) + 10


def draw_selected(img: np.ndarray, qs: list[np.ndarray], fish: set[int], pos: int, radius: float) -> None:
    """A white ring around the position node of every selected fish, in each version's ``(L, N, 2)``."""
    for q in qs:
        for k in fish:
            if k < len(q) and np.isfinite(q[k, pos]).all():
                cv2.circle(img, pt(q[k, pos]), round(radius), SELECT_COLOR, 2, cv2.LINE_AA)


def nearest_fish(qs: list[np.ndarray], x: float, y: float, max_px: float = PICK_PX) -> int | None:
    """The fish whose node (in any version's display pixels ``(L, N, 2)``) is nearest to ``(x, y)``."""
    best, who = max_px, None
    for q in qs:
        with np.errstate(invalid="ignore"):
            d = np.linalg.norm(np.asarray(q, float) - [x, y], axis=-1)        # (L, N)
        d = np.where(np.isfinite(d), d, np.inf)
        if d.size and d.min() < best:
            best, who = float(d.min()), int(np.unravel_index(np.argmin(d), d.shape)[0])
    return who


def sample_runs(codes: np.ndarray, width: int) -> list[tuple[float, float, int]]:
    """Runs ``(x0, x1, code)`` in ``0..1`` of ``codes`` drawn ``width`` pixels wide. With more frames than
    pixels each pixel shows the largest code of its frames, so a short run (even one frame) never vanishes."""
    n = len(codes)
    if n == 0:
        return []
    if n > width:
        starts = (np.arange(width) * n / width).astype(int)
        codes, n = np.maximum.reduceat(np.asarray(codes), starts), width
    edges = np.flatnonzero(np.diff(np.r_[-1, codes, -1]) != 0)
    return [(s0 / n, s1 / n, int(codes[s0])) for s0, s1 in zip(edges[:-1], edges[1:]) if codes[s0]]


def run_app(folder: Path, recon_dir: Path, *, old: str | None = None, new: str | None = None,
            smoke_test: bool = False, screenshot: tuple[Path, int] | None = None) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    from scripts.review.review_gui import install_error_log

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    cd = ReconCompareData(folder, recon_dir)
    js = JudgeState(folder, recon_dir, old=old, new=new, fps=cd.fps, recon=cd.state)
    pair = (js.old, js.new)
    n_fish = len(cd.labels)
    raw_slp = RawSlp.open(folder)                    # None until ``python -m scripts.review.slp_raw`` has run

    class ViewLabel(QtWidgets.QLabel):
        clicked = QtCore.pyqtSignal(int, int, int, bool)

        def __init__(self, index: int):
            super().__init__()
            self.index = index
            self.setMinimumSize(160, 160)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet("background:#111;")
            self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        def show_image(self, bgr: np.ndarray):
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            h, w = rgb.shape[:2]
            self.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)))

        def mousePressEvent(self, e):
            pm = self.pixmap()
            ox, oy = ((self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2) if pm else (0, 0)
            self.clicked.emit(self.index, int(e.x() - ox), int(e.y() - oy),
                              bool(e.modifiers() & QtCore.Qt.ControlModifier))

    class View3D(QtWidgets.QWidget):
        """The 3D reconstruction - what is graded. Old dashed, new solid, green where they agree; grey floor
        shadows for depth. A left click selects a fish (Ctrl adds), left-drag rotates, right- or middle-drag
        (or Shift+left-drag) moves, the wheel zooms towards the mouse, double-click shows the whole tank;
        7/8/9/0 top / front / side / oblique; K trails; X keeps the selected fish in the middle."""

        VIEWS = {"7": (0.0, 90.0), "8": (0.0, 0.0), "9": (90.0, 0.0), "0": (30.0, 35.0)}

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.setMinimumSize(200, 200)
            self.az, self.el = self.VIEWS["0"]
            self.zoom, self.follow, self.trail = 1.0, False, True
            self.pan = np.zeros(2)                       # screen pixels
            self._drag, self._centre, self._q = None, None, []
            p = cd.bounds_nodes()
            ok = np.isfinite(p).all(axis=1)
            lo, hi = (np.percentile(p[ok], [0.5, 99.5], axis=0) if ok.any() else (np.zeros(3), np.full(3, 200.0)))
            self.lo, self.hi = lo - 5, hi + 5

        def set_view(self, key: str):
            self.az, self.el = self.VIEWS[key]
            self.update()

        def mousePressEvent(self, e):
            moving = (e.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton)
                      or e.modifiers() & QtCore.Qt.ShiftModifier)
            self._drag = ("pan" if moving else "rotate", e.x(), e.y(), self.az, self.el, self.pan.copy(), False)

        def mouseMoveEvent(self, e):
            if self._drag:
                mode, x0, y0, az, el, pan, _ = self._drag
                if abs(e.x() - x0) + abs(e.y() - y0) > 3:
                    self._drag = self._drag[:6] + (True,)
                if mode == "pan":
                    self.pan = pan + [e.x() - x0, e.y() - y0]
                else:
                    self.az, self.el = az - (e.x() - x0) * 0.4, float(np.clip(el + (e.y() - y0) * 0.4, -89, 90))
                self.update()

        def mouseReleaseEvent(self, e):
            if self._drag and self._drag[0] == "rotate" and not self._drag[6]:      # a click, not a drag
                k = nearest_fish(self._q, e.x(), e.y())
                if k is not None:
                    self.win.pick_fish(k, True)              # a click selects or deselects that fish
            self._drag = None

        def contextMenuEvent(self, e):
            e.accept()                                   # right-drag moves; no menu

        def wheelEvent(self, e):
            f = float(np.clip(self.zoom * 1.15 ** (e.angleDelta().y() / 120), 0.5, 40.0)) / self.zoom
            c = np.array([e.x() - self.width() / 2, e.y() - self.height() / 2])
            self.pan = c - (c - self.pan) * f            # the point under the mouse stays put
            self.zoom *= f
            self.update()

        def mouseDoubleClickEvent(self, _e):
            self.az, self.el = self.VIEWS["0"]
            self.zoom, self.pan, self._centre = 1.0, np.zeros(2), None
            self.update()

        def paintEvent(self, _e):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 18, np.uint8)
            win = self.win
            t = cd.t(win.frame)
            scale = self.zoom * min(w, h) / (float((self.hi - self.lo).max()) * 1.3)
            target = (self.lo + self.hi) / 2
            nodes = None
            if t is not None:
                nodes = [cd.series(v, t, t + 1, win.aligned)[0] for v in pair]
                if self.follow:
                    ks = sorted(win.sel_fish) or list(range(n_fish))
                    pts = np.concatenate([n[ks, cd.pos_node] for n in nodes])
                    if np.isfinite(pts).all(axis=1).any():
                        target = np.nanmean(pts, axis=0)
            self._centre = target if self._centre is None else self._centre + 0.3 * (target - self._centre)
            centre = self._centre
            ox, oy = w / 2 + self.pan[0], h / 2 + self.pan[1]
            a_, e_ = np.radians(self.az), np.radians(self.el)

            def proj(p):
                d = np.asarray(p, float) - centre
                x1 = d[..., 0] * np.cos(a_) - d[..., 1] * np.sin(a_)
                y1 = d[..., 0] * np.sin(a_) + d[..., 1] * np.cos(a_)
                return np.stack([ox + scale * x1, oy - scale * (d[..., 2] * np.cos(e_) + y1 * np.sin(e_))], -1)

            lo, hi = self.lo, self.hi
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            q = proj(corners)
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") == 1:
                        cv2.line(img, pt(q[i]), pt(q[j]), (70, 70, 70), 1, cv2.LINE_AA)
            axes = proj(np.array([lo, lo + [20, 0, 0], lo + [0, 20, 0], lo + [0, 0, 20]]))
            for k, name in enumerate("XYZ"):
                cv2.line(img, pt(axes[0]), pt(axes[k + 1]), (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(img, name, pt(axes[k + 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            self._q = []
            if nodes is not None:
                if self.trail:
                    half = round(0.3 * cd.fps)
                    lo_t, hi_t = max(0, t - half), min(len(cd.frames), t + 1)
                    for i_v, (v, dashed) in enumerate(zip(pair, (True, False))):
                        if win.show_ver == 2 - i_v:                  # this version hidden (M)
                            continue
                        tr =cd.series(v, lo_t, hi_t, win.aligned)[:, :, cd.pos_node]       # (T, L, 3)
                        for c, name in enumerate(cd.labels):
                            qt = proj(tr[:, c])
                            ok = np.isfinite(qt).all(axis=1)
                            col = tuple(int(x * 0.6) for x in label_color(name, c))
                            for i in range(1, len(qt)):
                                if ok[i] and ok[i - 1] and (not dashed or i % 4 < 2):
                                    cv2.line(img, pt(qt[i - 1]), pt(qt[i]), col, 1, cv2.LINE_AA)
                drawn = win.shown(*nodes)
                for n in drawn:                                  # floor shadows
                    s = n.copy()
                    s[..., 2] = lo[2]
                    qs = proj(s)
                    for c in range(len(qs)):
                        ok = np.isfinite(qs[c]).all(axis=1)
                        for i, j in cd.edges:
                            if ok[i] and ok[j]:
                                cv2.line(img, pt(qs[c, i]), pt(qs[c, j]), (60, 60, 60), 2, cv2.LINE_AA)
                if win.show_ref:
                    draw_reference(img, proj(cd.series(js.ref, t, t + 1, False)[0]), cd.edges, 1.4)
                self._q = [proj(drawn[0]), proj(drawn[1])]
                draw_pair(img, cd.labels, cd.edges, cd.head, cd.pos_node, self._q[0], self._q[1], 1.4, cross=False)
                draw_selected(img, self._q, win.sel_fish, cd.pos_node, 16)
                y = 18
                for k in range(n_fish):
                    with np.errstate(invalid="ignore"):
                        d = np.linalg.norm(nodes[0][k] - nodes[1][k], axis=-1)
                    txt = f"{cd.labels[k]} {js.old}-{js.new}: " + "  ".join(
                        f"{nm} {'-' if np.isnan(x) else f'{x:.1f}'}" for nm, x in zip(cd.node_names, d)) + " mm"
                    filled = [v for v in pair if (cd.sources(v, t, win.aligned)[k] == NODE_INTERPOLATED).any()]
                    if filled:
                        txt += f"  [{', '.join(filled)} interp.]"
                    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color(cd.labels[k], k), 1,
                                cv2.LINE_AA)
                    y += 18
            bar = 10 * scale
            cv2.line(img, (10, h - 26), (10 + int(bar), h - 26), (200, 200, 200), 2)
            cv2.putText(img, "10 mm", (14 + int(bar), h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
                        cv2.LINE_AA)
            cv2.putText(img, "3D (graded) - click a fish to select, left-drag rotate, right-drag move, wheel zoom, "
                        f"double-click whole tank, 7/8/9/0 views, K trail, X follow {'on' if self.follow else 'off'}",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))
            painter.end()

    class Timeline(QtWidgets.QWidget):
        """Top: the whole recording (ID hints red, body hints orange, frames graded for every fish light
        green, frames with some grades dark green, the shown span in white).
        Below: the shown span - time ruler, machine hints, and per fish an ID row and a body row, each with
        old (upper half) over new (lower half). Click seeks, drag selects frames, wheel zooms, right-drag or
        Shift+wheel moves, Ctrl+wheel (or Ctrl+= / Ctrl+-) changes the text size; the rows get taller when
        the timeline does (drag the border above it)."""

        HINT_H = 12

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.text_px = 14                            # label size; Ctrl+wheel changes it
            self.span = round(10 * cd.fps)               # frames shown
            self.set_text_px(self.text_px)
            self.v0 = cd.first
            self._press = self._pan = None
            self._graded_key, self._graded = None, None

        # geometry
        def set_text_px(self, px: int):
            self.text_px = int(np.clip(px, 10, 28))
            f = QtGui.QFont(self.font())
            f.setPixelSize(self.text_px)
            f.setBold(True)
            fm = QtGui.QFontMetrics(f)
            self.MARGIN = max(fm.horizontalAdvance(f"> {n} body") for n in cd.labels) + 12
            self.OVERVIEW_H = 3 * (self.text_px + 4)
            self.RULER_H = self.text_px + 8
            self.HINT_H = self.text_px + 2
            self.setMinimumHeight(self.OVERVIEW_H + self.RULER_H + self.HINT_H + 2 * n_fish * (self.text_px + 6) + 10)
            self.update()

        def plot_w(self) -> int:
            return max(1, self.width() - self.MARGIN - 4)

        def x_of(self, f: float) -> float:
            return self.MARGIN + (f - self.v0) / max(1, self.span) * self.plot_w()

        def frame_at(self, x: float) -> int:
            return int(round(self.v0 + (x - self.MARGIN) / self.plot_w() * self.span))

        def follow(self, frame: int):
            """Keep ``frame`` in the shown span (page when it leaves it)."""
            if not self.v0 <= frame <= self.v0 + self.span:
                self.v0 = frame - self.span // 5
            self.v0 = int(np.clip(self.v0, cd.first, max(cd.first, cd.last - self.span)))

        def zoom(self, factor: float, around: int | None = None):
            around = self.win.frame if around is None else around
            frac = (around - self.v0) / max(1, self.span)
            self.span = int(np.clip(self.span * factor, 20, cd.last - cd.first + 1))
            self.v0 = int(np.clip(around - frac * self.span, cd.first, max(cd.first, cd.last - self.span)))
            self.update()

        def graded_any(self) -> tuple[np.ndarray, np.ndarray]:
            """Per frame: every fish fully graded; anything graded."""
            n = len(js.segments)
            if self._graded_key != n:
                self._graded = (js.graded().all(axis=1), (js.grades > 0).any(axis=(0, 1, 3)))
                self._graded_key = n
            return self._graded

        def paintEvent(self, _e):
            p = QtGui.QPainter(self)
            p.fillRect(self.rect(), QtGui.QColor(22, 22, 22))
            m, w, h, pw = self.MARGIN, self.width(), self.height(), self.plot_w()
            font = p.font()
            font.setPixelSize(self.text_px)
            p.setFont(font)
            # Overview of the whole recording.
            oh = self.OVERVIEW_H
            total = max(1, cd.last - cd.first)
            ox = lambda f: m + (f - cd.first) / total * pw            # noqa: E731
            third = oh / 3
            for i, label in enumerate(("ID hints", "body hints", "graded")):
                p.fillRect(QtCore.QRectF(m, i * third, pw, third - 2), QtGui.QColor(38, 38, 38))
                p.setPen(QtGui.QColor(170, 170, 170))
                p.drawText(QtCore.QRectF(2, i * third, m - 4, third), QtCore.Qt.AlignVCenter, label)
            for r in js.events.itertuples(index=False):
                row = 0 if r.kind == "identity" else 1
                p.fillRect(QtCore.QRectF(ox(r.start), row * third, max(1.0, ox(r.end) - ox(r.start)), third - 2),
                           QtGui.QColor(*HINT_RGB[r.kind]))
            full, some = self.graded_any()
            level = np.where(full, 2, np.where(some, 1, 0)).astype(np.int8)
            for x0, x1, c in sample_runs(level, pw):
                p.fillRect(QtCore.QRectF(m + x0 * pw, 2 * third, max(2.0, (x1 - x0) * pw), third - 2),
                           QtGui.QColor(*(GRADED_RGB if c == 2 else PARTLY_RGB)))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(QtCore.QRectF(ox(self.v0), 0, max(2.0, ox(self.v0 + self.span) - ox(self.v0)), oh - 2))
            # Ruler.
            y = oh + 2
            v0, v1 = self.v0, self.v0 + self.span
            label = lambda t: f"{int(t // 60):02d}:{t % 60:05.2f} ({int(round(t * cd.fps))})"   # noqa: E731
            need = p.fontMetrics().horizontalAdvance(label(v1 / cd.fps)) + 16      # a label never overlaps the next
            step_s = next((s for s in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200)
                           if s * cd.fps / self.span * pw >= need), 1200)
            t = np.ceil(v0 / cd.fps / step_s) * step_s
            p.setPen(QtGui.QColor(150, 150, 150))
            while t * cd.fps <= v1:
                x = self.x_of(t * cd.fps)
                p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + 5))
                p.drawText(QtCore.QPointF(x + 2, y + self.text_px + 4), label(t))
                t += step_s
            p.drawText(QtCore.QRectF(2, y, m - 4, self.RULER_H), QtCore.Qt.AlignVCenter, "time (frame)")
            y += self.RULER_H
            # Machine hints in the span.
            p.setPen(QtGui.QColor(170, 170, 170))
            p.drawText(QtCore.QRect(2, y, m - 4, self.HINT_H), QtCore.Qt.AlignVCenter, "hints")
            ev = js.events[(js.events["end"] >= v0) & (js.events["start"] <= v1)]
            for r in ev.itertuples(index=False):
                col = QtGui.QColor(*HINT_RGB[r.kind])
                x0, x1 = self.x_of(max(r.start, v0)), self.x_of(min(r.end, v1) + 1)
                p.fillRect(QtCore.QRectF(x0, y + 1, max(1.0, x1 - x0), self.HINT_H - 2), col)
            y += self.HINT_H + 2
            # Grades: per fish an ID row and a body row; old upper half, new lower half.
            rows = 2 * n_fish
            row_h = max(self.text_px + 6.0, (h - y - 2) / rows)
            row_font = QtGui.QFont(font)
            a, b = max(cd.first, v0), min(cd.last, v1)
            sl = slice(a - cd.first, b - cd.first + 1)
            x_a, x_b = self.x_of(a), self.x_of(b + 1)
            for k, name in enumerate(cd.labels):
                sel = k in self.win.sel_fish
                bb, gg, rr = label_color(name, k)
                for j, asp in enumerate(ASPECTS):
                    yy = y + (2 * k + j) * row_h
                    p.fillRect(QtCore.QRectF(m, yy, pw, row_h - 2), QtGui.QColor(44, 44, 44))
                    for vi in range(len(JUDGED)):
                        codes = js.grades[ASPECTS.index(asp), vi, sl, k]
                        half = (row_h - 2) / 2
                        for x0, x1, c in sample_runs(codes, max(1, int(x_b - x_a))):
                            p.fillRect(QtCore.QRectF(x_a + x0 * (x_b - x_a), yy + vi * half,
                                                     max(1.0, (x1 - x0) * (x_b - x_a)), half - (1 if vi == 0 else 0)),
                                       QtGui.QColor(*GRADE_RGB[GRADE_CODES[c - 1]]))
                    p.setPen(QtGui.QColor(255, 255, 255) if sel else QtGui.QColor(rr, gg, bb))
                    row_font.setBold(sel)
                    p.setFont(row_font)
                    text = f"{'> ' if sel else ''}{name} {'ID' if asp == 'id' else 'body'}"
                    p.drawText(QtCore.QRectF(2, yy, m - 4, row_h - 2), QtCore.Qt.AlignVCenter | QtCore.Qt.TextSingleLine,
                               p.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, m - 6))
            p.setFont(font)
            # Selection and cursor.
            ra, rb = self.win.sel_range()
            pen = QtGui.QPen(QtGui.QColor(0, 230, 255), 2)
            if self.win.range_open:
                pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawRect(QtCore.QRectF(self.x_of(ra), oh + 1, max(2.0, self.x_of(rb + 1) - self.x_of(ra)), h - oh - 3))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            x = self.x_of(self.win.frame)
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            fx = ox(self.win.frame)
            p.drawLine(QtCore.QPointF(fx, 0), QtCore.QPointF(fx, oh))

        def mousePressEvent(self, e):
            if e.y() < self.OVERVIEW_H:                     # the whole recording: jump there
                f = int(cd.first + (e.x() - self.MARGIN) / self.plot_w() * (cd.last - cd.first))
                self.v0 = f - self.span // 2
                self.follow(f)
                self.win.seek(f)
                return
            if e.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
                self._pan = (e.x(), self.v0)
            else:
                self._press = e.x()

        def mouseMoveEvent(self, e):
            if self._pan is not None:
                x0, v0 = self._pan
                self.v0 = int(np.clip(v0 - (e.x() - x0) / self.plot_w() * self.span, cd.first,
                                      max(cd.first, cd.last - self.span)))
                self.update()
            elif self._press is not None and abs(e.x() - self._press) > 3:
                self.win.set_range(self.frame_at(self._press), self.frame_at(e.x()), seek=False)

        def mouseReleaseEvent(self, e):
            if self._pan is not None:
                self._pan = None
                return
            if self._press is None:
                return
            if abs(e.x() - self._press) <= 3:
                self.win.seek(self.frame_at(e.x()))
            else:
                self.win.set_range(self.frame_at(self._press), self.frame_at(e.x()))
            self._press = None

        def contextMenuEvent(self, e):
            e.accept()

        def wheelEvent(self, e):
            if e.modifiers() & QtCore.Qt.ControlModifier:
                self.set_text_px(self.text_px + (1 if e.angleDelta().y() > 0 else -1))
            elif e.modifiers() & QtCore.Qt.ShiftModifier:
                d = -e.angleDelta().y() / 120 * self.span * 0.15
                self.v0 = int(np.clip(self.v0 + d, cd.first, max(cd.first, cd.last - self.span)))
                self.update()
            else:
                self.zoom(1.25 ** (-e.angleDelta().y() / 120), self.frame_at(e.x()))

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Grade reconstructions - {cd.experiment}: {js.old} (dashed) vs {js.new} (solid)")
            start = cd.analysis_start if cd.analysis_start is not None else cd.first
            self.frame = int(np.clip(start, cd.first, cd.last))
            self.speed_i = SPEEDS.index(1.0)
            self.playing, self.loop, self.show_ref, self.zoom_on = False, False, False, True
            self.show_raw = raw_slp is not None          # raw SLEAP instances in the camera views (B)
            self.show_ver = 0                            # 0 both versions, 1 only old, 2 only new (M)
            self.aligned_override = False                # own named tracks by default (I: matched per frame)
            self.current: str | None = None             # the hint event last opened, if any
            self.sel_fish: set[int] = set(range(n_fish))
            self.range: tuple[int, int] | None = None   # closed range
            self.range_start: int | None = None         # open range: from here to the current frame
            self.crops = [None, None, None]
            self._view_q: list[list[np.ndarray]] = [[], [], []]
            self._acc = 0.0
            self.full = js.table()
            self.rows = self.full

            # Every border between panels is a splitter handle: drag it to resize.
            Split = QtWidgets.QSplitter
            outer = Split(QtCore.Qt.Horizontal)
            self.setCentralWidget(outer)
            left = Split(QtCore.Qt.Vertical)
            outer.addWidget(left)
            views = Split(QtCore.Qt.Horizontal)
            self.view3d = View3D(self)
            views.addWidget(self.view3d)
            cams = Split(QtCore.Qt.Vertical)
            self.views = [ViewLabel(i) for i in range(len(VIEWS))]
            for lab in self.views:
                lab.clicked.connect(self.on_view_click)
            cams.addWidget(self.views[0])
            sides = Split(QtCore.Qt.Horizontal)
            sides.addWidget(self.views[1])
            sides.addWidget(self.views[2])
            cams.addWidget(sides)
            views.addWidget(cams)
            views.setSizes([600, 500])
            left.addWidget(views)
            bottom = QtWidgets.QWidget()
            blay = QtWidgets.QVBoxLayout(bottom)
            blay.setContentsMargins(0, 0, 0, 0)
            self.info = QtWidgets.QLabel()
            self.info.setTextFormat(QtCore.Qt.RichText)
            self.info.setWordWrap(True)
            blay.addWidget(self.info)
            self.timeline = Timeline(self)
            blay.addWidget(self.timeline, 1)
            left.addWidget(bottom)
            left.setSizes([580, 340])
            for s in (outer, left, views, cams, sides):
                s.setHandleWidth(6)
                s.setChildrenCollapsible(False)

            panel = QtWidgets.QWidget()
            right = QtWidgets.QVBoxLayout(panel)
            outer.addWidget(panel)
            outer.setSizes([1200, 600])

            hints = QtWidgets.QGroupBox("Machine hints (only suggestions where to look; N / P)")
            hl = QtWidgets.QVBoxLayout(hints)
            filters = QtWidgets.QHBoxLayout()
            self.kind_box = self._combo(["all", "identity", "body"])
            self.sort_box = self._combo(list(SORTS))
            self.mm_box = self._combo([f"old vs new >= {m} mm" if m else "any size" for m in MIN_MM])
            self.todo = QtWidgets.QCheckBox("not fully graded")
            self.todo.setFocusPolicy(QtCore.Qt.NoFocus)
            self.todo.toggled.connect(self.apply_filter)
            for wdg in (self.kind_box, self.sort_box, self.mm_box, self.todo):
                filters.addWidget(wdg)
            hl.addLayout(filters)
            self.table = QtWidgets.QTableWidget()
            self.table.setFocusPolicy(QtCore.Qt.NoFocus)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.verticalHeader().setVisible(False)
            self.table.cellClicked.connect(lambda r, _c: self.goto_row(r))
            hl.addWidget(self.table, 1)
            self.detail = QtWidgets.QLabel()
            self.detail.setWordWrap(True)
            hl.addWidget(self.detail)
            right.addWidget(hints, 1)
            sel_box = QtWidgets.QGroupBox("Selection (what the next grade applies to)")
            sel_lay = QtWidgets.QVBoxLayout(sel_box)
            fish_row = QtWidgets.QHBoxLayout()
            self.fish_buttons = []
            for k, name in enumerate(cd.labels):
                b = QtWidgets.QPushButton(f"{k + 1}  {name}")
                b.setCheckable(True)
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                b.clicked.connect(lambda _c, i=k: self.pick_fish(i, True))
                fish_row.addWidget(b)
                self.fish_buttons.append(b)
            b = QtWidgets.QPushButton("4  all")
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.clicked.connect(self.select_all_fish)
            fish_row.addWidget(b)
            sel_lay.addLayout(fish_row)
            self.sel_label = QtWidgets.QLabel()
            self.sel_label.setWordWrap(True)
            self.sel_label.setTextFormat(QtCore.Qt.RichText)
            sel_lay.addWidget(self.sel_label)
            right.addWidget(sel_box)

            grade_box = QtWidgets.QGroupBox("Grade each version (3D)")
            grid = QtWidgets.QGridLayout(grade_box)
            for c, code in enumerate(("ok", "minor", "major")):
                head = QtWidgets.QLabel(f"<b>{GRADES[code]}</b>")
                r_, g_, b_ = GRADE_RGB[code]
                head.setStyleSheet(f"border-bottom: 3px solid rgb({r_},{g_},{b_});")
                grid.addWidget(head, 0, c + 1)
            for i, ((asp, ver), keys) in enumerate(KEYS.items()):
                grid.addWidget(QtWidgets.QLabel(f"{ASPECT_TITLES[asp]} - {js.old if ver == 'old' else js.new}"), i + 1, 0)
                for c, (key, code) in enumerate(zip(keys, ("ok", "minor", "major"))):
                    btn = QtWidgets.QPushButton(key)
                    btn.setFocusPolicy(QtCore.Qt.NoFocus)
                    r_, g_, b_ = GRADE_RGB[code]
                    btn.setStyleSheet(f"QPushButton {{ border-left: 8px solid rgb({r_},{g_},{b_}); padding: 3px; }}")
                    btn.clicked.connect(lambda _c, a=asp, v=ver, g=code: self.grade([(a, v)], g))
                    grid.addWidget(btn, i + 1, c + 1)
            row = len(KEYS) + 1
            for j, (asp, key) in enumerate(UNSURE_KEYS.items()):
                btn = QtWidgets.QPushButton(f"{key}  {ASPECT_TITLES[asp]} unsure")
                btn.setFocusPolicy(QtCore.Qt.NoFocus)
                btn.clicked.connect(lambda _c, a=asp: self.grade([(a, "old"), (a, "new")], "unsure"))
                grid.addWidget(btn, row, j * 2, 1, 2)
            btn = QtWidgets.QPushButton("O  everything correct")
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(self.all_correct)
            grid.addWidget(btn, row + 1, 0, 1, 2)
            btn = QtWidgets.QPushButton("Backspace  erase")
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.clicked.connect(self.erase)
            grid.addWidget(btn, row + 1, 2, 1, 2)
            right.addWidget(grade_box)
            self.now_label = QtWidgets.QLabel()
            self.now_label.setTextFormat(QtCore.Qt.RichText)
            self.now_label.setWordWrap(True)
            right.addWidget(self.now_label)
            self.note_label = QtWidgets.QLabel()
            self.note_label.setWordWrap(True)
            right.addWidget(self.note_label)

            self.counts = QtWidgets.QLabel()
            self.counts.setWordWrap(True)
            right.addWidget(self.counts)
            self.setStyleSheet("QPushButton:checked { background:#2f7d3a; color:white; font-weight:bold; }"
                               "QSplitter::handle { background:#555; }")

            self._actions = self.key_actions()
            for key, fn in self._actions.items():
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.WindowShortcut)
                sc.activated.connect(fn)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(int(1000 / TICK_HZ))
            self.apply_filter()
            self.timeline.follow(self.frame)
            self.render()

        def _combo(self, items):
            box = QtWidgets.QComboBox()
            box.addItems(items)
            box.setFocusPolicy(QtCore.Qt.NoFocus)
            box.currentIndexChanged.connect(self.apply_filter)
            return box

        # -- machine hints -----------------------------------------------------
        def apply_filter(self, *_):
            if not hasattr(self, "table"):
                return
            t = self.full.assign(dur=lambda d: d["end"] - d["start"] + 1,
                                 jump=lambda d: d[["old_jump_mm", "new_jump_mm"]].max(axis=1))
            kind = self.kind_box.currentText()
            if kind != "all":
                t = t[t["kind"] == kind]
            mm = MIN_MM[self.mm_box.currentIndex()]
            if mm:
                t = t[t["old_new_mm"].fillna(np.inf) >= mm]          # one-sided data counts as large
            if self.todo.isChecked():
                t = t[~t["judged"] | (t["event"] == self.current)]
            col, asc = SORTS[self.sort_box.currentText()]
            t = t.sort_values(col, ascending=asc, na_position="first" if not asc else "last", kind="stable")
            self.rows = t.reset_index(drop=True)
            self.fill_table()

        def fill_table(self):
            cols = [("kind", "kind"), ("fish", "fish"), ("start", "start"), ("dur", "s"),
                    ("old_new_mm", "old-new mm"), ("old_ref_mm", "old-rev"), ("new_ref_mm", "new-rev"),
                    ("jump", "jump"), ("id %", "ID graded %"), ("body %", "body graded %")]
            self.table.setUpdatesEnabled(False)
            self.table.setRowCount(len(self.rows))
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels([c[1] for c in cols])
            # Light backgrounds with black text, readable in any theme.
            red, done, ink = QtGui.QColor(250, 218, 218), QtGui.QColor(214, 238, 214), QtGui.QColor(20, 20, 20)
            for i, r in enumerate(self.rows.to_dict("records")):
                for j, (c, _) in enumerate(cols):
                    v = r[c]
                    if c == "dur":
                        v = f"{v / cd.fps:.2f}"
                    elif isinstance(v, float):
                        v = "" if np.isnan(v) else f"{v:.1f}"
                    item = QtWidgets.QTableWidgetItem(str(v))
                    if r["judged"] or r["kind"] == "identity":
                        item.setBackground(done if r["judged"] else red)
                        item.setForeground(ink)
                    self.table.setItem(i, j, item)
            self.table.resizeColumnsToContents()
            self.table.setUpdatesEnabled(True)
            self.select_current()
            self.update_counts()

        def select_current(self):
            idx = np.flatnonzero(self.rows["event"].to_numpy() == self.current) if len(self.rows) else []
            if len(idx):
                self.table.selectRow(int(idx[0]))
                self.table.scrollToItem(self.table.item(int(idx[0]), 0))
            else:
                self.table.clearSelection()

        def event_row(self):
            if self.current is None:
                return None
            sel = self.full[self.full["event"] == self.current]
            return sel.iloc[0] if len(sel) else None

        def goto_row(self, i: int):
            """Open hint ``i``: go there, select its fish and frames (only a convenience)."""
            if not 0 <= i < len(self.rows):
                return
            r = self.rows.iloc[i]
            self.current = str(r["event"])
            self.sel_fish = {cd.labels.index(f) for f in js.event_fish(self.current)} or set(range(n_fish))
            self.range, self.range_start = (int(r["start"]), int(r["end"])), None
            self.crops = [None, None, None]
            self.select_current()
            self.set_playing(False)
            # Zoom so the hint fills about a third of the timeline (at least 1 s shown).
            self.timeline.span = int(max(cd.fps, 3 * (r["end"] - r["start"] + 1)))
            self.timeline.v0 = int(r["start"]) - self.timeline.span // 3
            self.seek(int(r["start"]) - round(0.3 * cd.fps))

        def step_hint(self, d: int):
            st = self.rows.sort_values("start")["start"].to_numpy()
            if not len(st):
                return
            near = np.flatnonzero(st > self.frame + 0.3 * cd.fps) if d > 0 else np.flatnonzero(st < self.frame - 1)
            if not len(near):
                return
            f = st[near[0] if d > 0 else near[-1]]
            self.goto_row(int(np.flatnonzero(self.rows["start"].to_numpy() == f)[0]))

        def next_ungraded(self):
            ks = sorted(self.sel_fish) or list(range(n_fish))
            g = js.graded()[:, ks].all(axis=1)
            t = self.frame - cd.first + 1
            todo = np.flatnonzero(~g[t:])
            if not len(todo):
                self.statusBar().showMessage("every frame after this one is graded for the selected fish")
                return
            self.set_playing(False)
            self.seek(cd.first + t + int(todo[0]))

        # -- selection ---------------------------------------------------------
        def pick_fish(self, k: int, add: bool):
            if add:
                self.sel_fish ^= {k}
            else:
                self.sel_fish = {k}
            self.render()

        def select_all_fish(self):
            self.sel_fish = set(range(n_fish))
            self.render()

        @property
        def range_open(self) -> bool:
            return self.range_start is not None

        def sel_range(self) -> tuple[int, int]:
            if self.range_start is not None:
                return min(self.range_start, self.frame), max(self.range_start, self.frame)
            if self.range is not None:
                return self.range
            return self.frame, self.frame

        def set_range(self, a: int, b: int, seek: bool = True):
            a, b = js.clip(a, b)
            self.range, self.range_start = (a, b), None
            if seek:
                self.set_playing(False)
                self.seek(a)
            else:
                self.timeline.update()
                self.update_panel()

        def start_range(self):
            self.range, self.range_start = None, self.frame
            self.render()

        def end_range(self):
            """Close the range at the current frame (from [, or from the start of the selected range)."""
            a = self.range_start if self.range_start is not None else (self.range[0] if self.range else self.frame)
            self.range, self.range_start = js.clip(a, self.frame), None
            self.render()

        def clear_range(self):
            self.range = self.range_start = None
            self.render()

        def on_view_click(self, v: int, x: int, y: int, add: bool):
            k = nearest_fish(self._view_q[v], x, y)
            if k is not None:
                self.pick_fish(k, True)                      # a click selects or deselects that fish

        # -- grades ------------------------------------------------------------
        def grade(self, targets: list[tuple[str, str]], code: str):
            if not self.sel_fish:
                self.statusBar().showMessage("no fish selected: 1 / 2 / 3, 4 or click a fish")
                return
            fish = [cd.labels[k] for k in sorted(self.sel_fish)]
            a, b = self.sel_range()
            for asp, ver in targets:
                js.paint(fish, a, b, asp, ver, code, event=self.current or "")
            what = ", ".join(f"{ASPECT_TITLES[asp]} {js.old if ver == 'old' else js.new}" for asp, ver in targets)
            self.statusBar().showMessage(f"{'+'.join(fish)} frames {a}-{b}: {what} = "
                                         f"{GRADES[code] if code else 'erased'}")
            if self.range_start is not None:                 # grading while watching: go on from here
                self.range_start = self.frame
            self.refresh_grades()

        def all_correct(self):
            self.grade([(a, v) for a in ASPECTS for v in JUDGED], "ok")

        def erase(self):
            self.grade([(a, v) for a in ASPECTS for v in JUDGED], "")

        def note(self):
            self.set_playing(False)
            a, b = self.sel_range()
            old = js.notes(a, b)
            text, ok = QtWidgets.QInputDialog.getText(self, "Note", f"note for frames {a}-{b}:",
                                                      text=old[-1][2] if old else "")
            if ok:
                js.set_note(a, b, text, event=self.current or "")
                self.update_panel()

        def refresh_grades(self):
            self.full = js.table()
            self.apply_filter()
            self.render()

        def update_panel(self):
            for k, b in enumerate(self.fish_buttons):
                b.setChecked(k in self.sel_fish)
            a, b = self.sel_range()
            fish = ", ".join(cd.labels[k] for k in sorted(self.sel_fish)) or "no fish"
            if self.range_start is not None:
                how = "from [ to the current frame (grading moves the start here)"
            elif self.range is not None:
                how = "selected range (\\ / Esc = current frame only)"
            else:
                how = "current frame only - drag on the timeline or press [ for a range"
            self.sel_label.setText(f"<b>{fish}</b> &nbsp; frames {a}-{b} ({(b - a + 1) / cd.fps:.2f} s) - {how}")
            now = js.at(self.frame)
            lines = []
            for k, name in enumerate(cd.labels):
                cells = []
                for asp in ASPECTS:
                    go, gn = now[(asp, "old")][k], now[(asp, "new")][k]
                    span = lambda c: (f"<span style='color:rgb{GRADE_RGB[c]}'>{GRADES[c]}</span>"   # noqa: E731
                                      if c else "<span style='color:#777'>-</span>")
                    cmp_ = js.compare(go, gn)
                    cells.append(f"{'ID' if asp == 'id' else 'body'}: {js.old} {span(go)}, {js.new} {span(gn)}"
                                 + (f" <i>({cmp_})</i>" if cmp_ else ""))
                lines.append(f"<b>{name}</b> &nbsp; " + " &nbsp;|&nbsp; ".join(cells))
            self.now_label.setText(f"<b>at frame {self.frame}</b><br>" + "<br>".join(lines))
            notes = js.notes(a, b)
            self.note_label.setText("note (C): " + ("; ".join(f"{x}-{y}: {t}" for x, y, t in notes) or "-"))
            ev = self.event_row()
            self.detail.setText("" if ev is None else
                                f"<b>{ev['kind']} hint</b>: {ev['fish']}, frames {ev['start']}-{ev['end']}<br>{ev['detail']}")

        def update_counts(self):
            g = js.graded()
            t0 = max(0, (cd.analysis_start or cd.first) - cd.first)
            done = g[t0:].all(axis=1).mean() if len(g) > t0 else 0
            per = ", ".join(f"{n} {100 * g[t0:, k].mean():.1f}%" for k, n in enumerate(cd.labels))
            hints = self.full
            self.counts.setText(f"graded (both versions, IDs and body) after the analysis start: all fish "
                                f"{100 * done:.1f}% ({per}); hints fully graded "
                                f"{int(hints['judged'].sum())}/{len(hints)} - saved in "
                                f"{js.recon_dir.name}/recon_judgements.csv")

        # -- playback ----------------------------------------------------------
        @property
        def aligned(self) -> bool:
            return bool(self.aligned_override)

        def set_playing(self, on: bool):
            self.playing = on
            self._acc = 0.0
            self.update_info()

        def seek(self, frame: int):
            self.frame = int(np.clip(frame, cd.first, cd.last))
            self.timeline.follow(self.frame)
            self.render()

        def step(self, n: int):
            self.set_playing(False)
            self.seek(self.frame + n)

        def set_speed(self, i: int):
            self.speed_i = int(np.clip(i, 0, len(SPEEDS) - 1))
            self.update_info()

        def tick(self):
            if not self.playing:
                return
            self._acc += SPEEDS[self.speed_i] * cd.fps / TICK_HZ
            n = int(self._acc)
            if n >= cd.step:                     # fast enough for the playback proxy: whole steps
                n -= n % cd.step
            if n <= 0:
                return
            self._acc -= n
            if self.loop and self.range is not None:
                a, b = self.range
                if self.frame <= b < self.frame + n:
                    self.seek(a)
                    return
            if self.frame + n > cd.last:
                self.set_playing(False)
            self.seek(self.frame + n)

        # -- drawing -----------------------------------------------------------
        def render(self):
            t = cd.t(self.frame)
            ks = sorted(self.sel_fish) or list(range(n_fish))
            proj = ref = None
            if t is not None:
                a = cd.series(pair[0], t, t + 1, self.aligned)[0]
                b = cd.series(pair[1], t, t + 1, self.aligned)[0]
                r = cd.series(js.ref, t, t + 1, False)[0]
                allp = project(cd, np.concatenate([a, b, r]))
                n = len(a)
                proj, ref = (allp[:n], allp[n:2 * n]), allp[2 * n:]
            drawn = self.shown(*proj) if proj is not None else None
            for v, lab in enumerate(self.views):
                crop = None
                if self.zoom_on and proj is not None:
                    pts = np.concatenate([proj[0][ks][:, :, v], proj[1][ks][:, :, v]]).reshape(-1, 2) * cd.scale[v]
                    meta = cd.proxy["views"][VIEWS[v]]
                    target = crop_target(pts, meta["proxy_width"], meta["proxy_height"])
                    self.crops[v] = target if self.crops[v] is None else self.crops[v] + 0.35 * (target - self.crops[v])
                    crop = self.crops[v]
                img, tf = render_view(cd, v, self.frame, pair, self.aligned, (max(50, lab.width()),
                                      max(50, lab.height())), crop, projections=drawn)
                self._view_q[v] = []
                raw = raw_slp.points(self.frame) if self.show_raw else None
                if raw is not None:
                    n_raw = draw_raw(img, raw[v] * tf[0] + np.array(tf[1:]), raw_slp.node_names, raw_slp.edges)
                    draw_raw_legend(img, n_raw, raw_slp.node_names, 46)
                if proj is not None:
                    off = np.array(tf[1:])
                    self._view_q[v] = [drawn[0][:, :, v] * tf[0] + off, drawn[1][:, :, v] * tf[0] + off]
                    draw_selected(img, self._view_q[v], self.sel_fish, cd.pos_node, 15)
                    if self.show_ref:
                        draw_reference(img, ref[:, :, v] * tf[0] + off, cd.edges, 1.2)
                lab.show_image(img)
            self.view3d.update()
            self.timeline.update()
            self.update_info()

        def update_info(self):
            secs = self.frame / cd.fps
            legend = "  ".join(f"<span style='color:rgb{tuple(reversed(label_color(n, c)))}'>{n}</span>"
                               for c, n in enumerate(cd.labels))
            grades = " ".join(f"<span style='color:rgb{GRADE_RGB[c]}'>&#9632;</span>{GRADES[c]}" for c in GRADE_CODES)
            self.info.setText(
                f"<b>frame {self.frame}</b> {int(secs // 60):02d}:{secs % 60:05.2f} &nbsp; "
                f"{'playing' if self.playing else 'paused'} <b>{SPEEDS[self.speed_i]:g}x</b> &nbsp; "
                f"loop {'on' if self.loop else 'off'} (L) &nbsp; "
                f"{'matched per frame' if self.aligned else 'own named tracks'} (I) &nbsp; "
                f"reviewed {'shown' if self.show_ref else 'hidden'} (V) &nbsp; "
                f"SLEAP raw {'shown' if self.show_raw else ('hidden' if raw_slp is not None else 'n/a - run scripts.review.slp_raw')} (B) &nbsp;|&nbsp; "
                f"{('both versions', f'only {js.old}', f'only {js.new}')[self.show_ver]} (M) &nbsp; "
                f"{js.old} dashed, {js.new} solid, green = same spot &nbsp; {legend} &nbsp;|&nbsp; {grades}")
            self.update_panel()

        # -- keys --------------------------------------------------------------
        def toggle(self, attr: str):
            setattr(self, attr, not getattr(self, attr))
            if attr == "zoom_on":
                self.crops = [None, None, None]
            self.render()

        def toggle3d(self, attr: str):
            setattr(self.view3d, attr, not getattr(self.view3d, attr))
            self.view3d.update()

        def cycle_versions(self):
            self.show_ver = (self.show_ver + 1) % 3
            self.render()

        def shown(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Old / new as drawn: the version hidden by M becomes NaN (nothing drawn, nothing to click)."""
            hide = np.full_like(a, np.nan)
            return (hide if self.show_ver == 2 else a), (hide if self.show_ver == 1 else b)

        def toggle_aligned(self):
            self.aligned_override = not self.aligned_override
            self.render()

        def goto_dialog(self):
            self.set_playing(False)
            f, ok = QtWidgets.QInputDialog.getInt(self, "Go to frame", f"frame ({cd.first}-{cd.last}):",
                                                  self.frame, cd.first, cd.last)
            if ok:
                self.seek(int(f))

        def key_actions(self) -> dict:
            fps = int(cd.fps)
            keys = {
                "Space": lambda: self.set_playing(not self.playing),
                "Right": lambda: self.step(1), "Left": lambda: self.step(-1),
                "Shift+Right": lambda: self.step(fps), "Shift+Left": lambda: self.step(-fps),
                "PgDown": lambda: self.step(10 * fps), "PgUp": lambda: self.step(-10 * fps),
                "Up": lambda: self.set_speed(self.speed_i + 1), "Down": lambda: self.set_speed(self.speed_i - 1),
                "N": lambda: self.step_hint(1), "P": lambda: self.step_hint(-1),
                "Return": self.next_ungraded, "Enter": self.next_ungraded,
                "L": lambda: self.toggle("loop"), "V": lambda: self.toggle("show_ref"),
                "B": lambda: self.toggle("show_raw") if raw_slp is not None else None,
                "M": self.cycle_versions,
                "Z": lambda: self.toggle("zoom_on"), "I": self.toggle_aligned, "G": self.goto_dialog,
                "C": self.note, "Backspace": self.erase, "4": self.select_all_fish, "O": self.all_correct,
                "[": self.start_range, "]": self.end_range, "\\": self.clear_range, "Esc": self.clear_range,
                "=": lambda: self.timeline.zoom(0.8), "-": lambda: self.timeline.zoom(1.25),
                "Ctrl+=": lambda: self.timeline.set_text_px(self.timeline.text_px + 1),
                "Ctrl+-": lambda: self.timeline.set_text_px(self.timeline.text_px - 1),
                "K": lambda: self.toggle3d("trail"), "X": lambda: self.toggle3d("follow"),
                "F1": lambda: QtWidgets.QMessageBox.information(self, "Keys", __doc__.split("Keys:")[1].rstrip()),
            }
            for k in range(min(3, n_fish)):
                keys[str(k + 1)] = lambda i=k: self.pick_fish(i, False)
                keys[f"Ctrl+{k + 1}"] = lambda i=k: self.pick_fish(i, True)
            for key in View3D.VIEWS:
                keys[key] = lambda kk=key: self.view3d.set_view(kk)
            for (asp, ver), letters in KEYS.items():
                for key, code in zip(letters, ("ok", "minor", "major")):
                    keys[key] = lambda a=asp, v=ver, c=code: self.grade([(a, v)], c)
            for asp, key in UNSURE_KEYS.items():
                keys[key] = lambda a=asp: self.grade([(a, "old"), (a, "new")], "unsure")
            return keys

        def press(self, key: str):
            self._actions[key]()

        def closeEvent(self, e):
            cd.close()
            super().closeEvent(e)

    win = Main()
    install_error_log(js.recon_dir / ERROR_LOG, win)
    if screenshot is not None:                       # render one frame to a PNG and quit (checking the drawing)
        win.resize(1700, 1000)
        win.show()
        win.seek(int(screenshot[1]))
        app.processEvents()
        win.grab().save(str(screenshot[0]))
        win.close()
        return 0
    if smoke_test:
        win.resize(1400, 900)
        win.show()
        app.processEvents()
        for key in ("Right", "Up", "O", "1", "Ctrl+2", "[", "Shift+Right", "W", "Right", "Y", "]", "F", "Esc", "4",
                    "N", "U", "J", "S", "H", "\\", "P", "Return", "V", "B", "I", "Z", "L", "=", "-", "Ctrl+=", "Ctrl+-", "PgDown", "PgUp",
                    "7", "8", "9", "0", "K", "M", "X", "M", "X", "M", "Down", "Space", "Space"):
            win.press(key)
            app.processEvents()
        win.set_playing(True)
        for _ in range(5):
            win.tick()
        win.press("Backspace")
        win.timeline.zoom(100.0)
        win.kind_box.setCurrentIndex(2)
        win.sort_box.setCurrentIndex(1)
        win.mm_box.setCurrentIndex(1)
        win.todo.setChecked(True)
        win.goto_row(0)
        win.view3d.repaint()
        win.timeline.repaint()
        win.close()
        return 0
    win.showMaximized()
    return app.exec_()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="The reference review package folder.")
    p.add_argument("--recon-dir", type=Path, required=True, help="A recon_compare folder with three versions.")
    p.add_argument("--old", default=None, help="Default: the second version.")
    p.add_argument("--new", default=None, help="Default: the third version.")
    p.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--screenshot", nargs=2, metavar=("PNG", "FRAME"), default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    shot = (Path(args.screenshot[0]), int(args.screenshot[1])) if args.screenshot else None
    return run_app(args.folder, args.recon_dir, old=args.old, new=args.new, smoke_test=args.smoke_test,
                   screenshot=shot)


if __name__ == "__main__":
    raise SystemExit(main())
