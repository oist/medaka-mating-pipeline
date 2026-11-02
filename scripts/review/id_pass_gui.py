"""One pass over a recording to fix identities, without pausing: ``python -m scripts.review.id_pass_gui``.

The video plays straight through, 2x in safe frames and 1x in danger runs (``id_segments``). A frame the reviewer watched without pressing anything counts as right. When a name
sits on the wrong fish, select it (1/2/3 = the fish shown as F/M1/M2, or click) and press its true name
(4/5/6): the names change from there on and stay until the next change. The change goes back to the
latest danger run by itself, and that run becomes unusable. X ... X marks any other range as unusable;
Delete removes the X or B range under the frame, whichever session wrote it.

Everything is written to ``<package>/id_pass/``, never to the package's own ledger: ``human_events.csv``,
``coverage.csv``, ``session_log.csv`` (see ``ledger.py``) and, on closing, ``id_segments.csv`` and
``id_pass_summary.json`` with the clock (playback and window-open time), and two figures that are not
mixed (``id_figure``): ``id_human.png`` (what the reviewer watched and marked) and ``id_machine.png``
(the machine's danger runs, with their agreement with the marks, also in ``id_agreement.json``). Both open
by themselves once the whole recording has been watched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.review import id_figure, id_segments as S
from scripts.review.behaviors import default_search, empty_bouts, find_boris_csv, load_bouts
from scripts.review.ledger import (ALL_FISH, EVENT_COLUMNS, PAIR_SEP, REMOVED_FILE, Ledger, ReviewLock, analysis_start,
                                   assign_label, owner_series, read_csv, static_labels, swap_list, watched_intervals)
from scripts.review.review_gui import (BEHAVIOR_COLORS, LABEL_COLORS, TRACK_COLORS, VIEW_TITLES, VIEWS, ProxyVideo,
                                       ReviewPackage, crop_rect, crop_target, dashed_line, install_error_log,
                                       proxy_fps, pt, to_display)

SPEEDS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
FAST_SPEED, SLOW_SPEED = 2.0, 1.0
FLUSH_SECONDS = 15.0       # coverage goes to the (network) disk this often while playing, not at every speed change
CHANGE_WINDOW_S = 5.0      # how far back in video time an identity change may land
SEEK_SECONDS = 2.0
HOLD_FRAMES = 10           # Left/Right held down: frames per key repeat (about 2x at 30 repeats/s)
HOLD_GAP_S = 0.08          # a press this soon after the last step is a key repeat, no finger is that fast
ZOOM_SECONDS = 30.0        # half width of the zoomed timeline at start (wheel: 5 to 240 s)
ZOOM_RANGE = (5.0, 240.0)
TIMELINE_SHARE = 0.36      # of the window height at the first start; the divider is draggable
VIEW3D_SHARE = 0.52        # of the window width at the first start; the divider is draggable
VIEW3D_START = (20.0, 40.0)   # azimuth, elevation
TRAIL_S = 0.5
LANE_NAMES = ("status", "danger", "to F", "M-M", "solo", "unusable", "pose")
LABEL_WIDTH = 68
POSE_COLOR = "#c77dff"
SETTINGS = ("medaka-mating-pipeline", "id_pass_gui")
CROP_HALF_S, CROP_SMOOTHING = 1.0, 0.25
LEDGER_FOLDER = S.LEDGER_FOLDER
REVIEW_DONE_PCT = 99.0     # watched this much: the figure opens on closing
STATUS_COLORS = {S.NOT_ANALYSED: (17, 17, 17), S.UNWATCHED: (44, 44, 52), S.WATCHED_SAFE: (63, 163, 77),
                 S.WATCHED_DANGER: (150, 170, 60), S.UNUSABLE: (200, 60, 60)}     # RGB
ZONE_COLORS = ((30, 30, 30), (217, 130, 43), (224, 60, 60))                       # RGB: safe, danger, high risk
TIME_FILE = "id_pass_clock.json"
HELP = ("Space play/pause | Up/Down fast speed | Shift+Up/Down slow speed | Left/Right one frame, held down 10 frames at a time | , . one frame | Shift+Left/Right -/+2 s | "
        "double-click camera: large | 1 2 3 or click select, again or Esc deselect | 4 5 6 F/M1/M2 (first time names, then changes from here on) | Shift+4 5 6 rename | "
        "X start/end unusable range | B start/end wrong pose, identity fine (selected fish, else all) | "
        "Delete remove the X/B range under the frame (latest first) | S analysis start | "
        "Ctrl+Z undo (a half-marked X/B first, then this session's marks and deletions) | W / Shift+W next / previous unwatched stretch | O open another experiment (this one is saved first) | Z zoom | G go to | P save result | F1 help | "
        "wheel on the lower timeline: span | drag the grey dividers: panel sizes | "
        "status row: grey not watched, green safe, olive danger with no ID change, red unusable, "
        "bright yellow line an ID change (4/5/6)")


def mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def nudge_frames(seconds_since_last_step: float) -> int:
    """Frames one Left/Right press moves: one, or ``HOLD_FRAMES`` when it is a key repeat."""
    return HOLD_FRAMES if seconds_since_last_step < HOLD_GAP_S else 1


def skeleton_points(detected: np.ndarray, projected: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` nodes to draw, with a view's undetected ones projected from 3D, and which were detected."""
    seen = np.isfinite(detected).all(axis=1)
    return (detected if projected is None else np.where(seen[:, None], detected, projected)), seen


def status_strip(status: np.ndarray, zone: np.ndarray, a: int, b: int, width: int) -> np.ndarray:
    """``(2, width, 3)`` RGB: status and danger zone of frames ``a..b``, the worse one per pixel column."""
    edges = np.linspace(a, b + 1, width + 1).astype(int).clip(a, b + 1)
    edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)
    idx = np.minimum(edges[:-1], len(status) - 1)
    stop = np.minimum(edges[1:], len(status))
    keep = idx < stop
    out = np.zeros((2, width, 3), np.uint8)
    if not keep.any():
        return out
    code = status[idx]
    z = np.maximum.reduceat(np.r_[zone, 0], np.minimum(np.c_[idx, stop].ravel(), len(zone)))[::2]
    bad = np.maximum.reduceat(np.r_[status == S.UNUSABLE, False], np.minimum(np.c_[idx, stop].ravel(), len(zone)))[::2]
    code = np.where(bad, S.UNUSABLE, code)
    for value, color in STATUS_COLORS.items():
        out[0][code == value] = color
    for value, color in enumerate(ZONE_COLORS):
        out[1][z == value] = color
    return out


def pick_folder(paths_config: Path | None, current: Path | None = None) -> Path | None:
    """Ask which package to open: every ``*_m3dfull`` package whose experiment has proxies.

    ``current`` (the package already open) is the one offered first.
    """
    from PyQt5 import QtWidgets

    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import default_paths_config

    paths = load_local_paths(paths_config or default_paths_config())
    root = paths.data_root / "review_packages"
    found = sorted(p.parent for p in root.glob("*/*_m3dfull/review_manifest.json")
                   if (p.parent.parent / "proxy" / "proxy_manifest.json").is_file())
    if not found:
        raise SystemExit(f"no *_m3dfull package with proxies under {root}")
    names = [f"{p.parent.name}/{p.name}" for p in found]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    here = next((i for i, p in enumerate(found) if current is not None and p.resolve() == Path(current).resolve()), 0)
    choice, ok = QtWidgets.QInputDialog.getItem(None, "Identity pass", "Experiment:", names, here, False)
    del app
    return found[names.index(choice)] if ok else None


def run_app(folder: Path, *, ledger_dir: Path | None = None, boris: Path | None = None,
            params: S.SegmentParams = S.SegmentParams(), scenario=None, paths_config: Path | None = None) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    class ViewLabel(QtWidgets.QLabel):
        clicked = QtCore.pyqtSignal(int, float, float)
        double_clicked = QtCore.pyqtSignal(int)

        def __init__(self, index: int):
            super().__init__()
            self.index, self._map = index, (1.0, 0.0, 0.0)
            self.setMinimumSize(160, 110)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet("background:#111;")
            self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        def show_image(self, rgb: np.ndarray):
            h, w = rgb.shape[:2]
            image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self._map = (1.0, (self.width() - w) / 2, (self.height() - h) / 2)
            self.setPixmap(QtGui.QPixmap.fromImage(image))

        def mousePressEvent(self, event):
            s, ox, oy = self._map
            self.clicked.emit(self.index, (event.x() - ox) / s, (event.y() - oy) / s)

        def mouseDoubleClickEvent(self, _event):
            self.double_clicked.emit(self.index)

    class Timeline(QtWidgets.QWidget):
        """Status, danger zone and the three BORIS lanes, for the whole recording or around the frame.

        Rows share the widget's height, so dragging the divider above the timelines enlarges them.
        """

        seek = QtCore.pyqtSignal(int)

        def __init__(self, win, zoom: bool):
            super().__init__()
            self.win, self.zoom = win, zoom
            self.setMinimumHeight(160 if zoom else 110)

        def span(self) -> tuple[int, int]:
            first, last = self.win.pkg.first, self.win.pkg.last
            if not self.zoom:
                return first, last
            half = int(self.win.zoom_seconds * self.win.pkg.fps)
            a = max(first, self.win.frame - half)
            return a, min(last, a + 2 * half)

        def x_of(self, f: float, a: int, b: int) -> float:
            return (f - a) / max(1, b - a) * self.width()

        def paintEvent(self, _event):
            win, w, h = self.win, max(1, self.width()), self.height()
            a, b = self.span()
            first = win.pkg.first
            strip = status_strip(win.status, win.zone, a - first, b - first, w)
            tops = [round(h * i / len(LANE_NAMES)) for i in range(len(LANE_NAMES) + 1)]
            row = tops[2] // 2
            rgb = np.ascontiguousarray(np.repeat(strip, row, axis=0))
            p = QtGui.QPainter(self)
            p.fillRect(0, 0, w, h, QtGui.QColor("#1b1b1b"))
            p.drawImage(0, 0, QtGui.QImage(rgb.data, w, 2 * row, 3 * w, QtGui.QImage.Format_RGB888))
            if win.run is not None:
                xa, xb = self.x_of(win.run["start"], a, b), self.x_of(win.frame, a, b)
                p.fillRect(QtCore.QRectF(xa, 0, max(1.0, xb - xa), row), QtGui.QColor(63, 163, 77))
            bar_h = tops[3] - tops[2] - 2
            font = p.font()
            font.setPixelSize(max(12, min(16, bar_h - 4)))
            p.setFont(font)
            for lane in range(5):
                p.fillRect(QtCore.QRectF(0, tops[2 + lane], w, bar_h), QtGui.QColor("#242424"))
            for r in win.bouts.itertuples(index=False):
                if r.end < a or r.start > b:
                    continue
                xa, xb = self.x_of(r.start, a, b), max(self.x_of(r.end + 1, a, b), self.x_of(r.start, a, b) + 2)
                rect = QtCore.QRectF(xa, tops[2 + r.lane], xb - xa, bar_h)
                p.fillRect(rect, QtGui.QColor(BEHAVIOR_COLORS[r.category]))
                if self.zoom and xb - xa > 40:
                    left = max(rect.left(), LABEL_WIDTH + 2.0)
                    p.setPen(QtGui.QColor("#f0efec"))
                    p.drawText(QtCore.QRectF(left + 3, rect.top(), rect.right() - left - 3, bar_h),
                               QtCore.Qt.AlignVCenter, f"{r.who} {r.behavior}")
            # Own row, like the pose marks: a range shows from its first X and however short it is.
            unusable = list(win.unusable)
            if win.open_danger is not None:
                unusable.append(tuple(sorted((win.open_danger, win.frame))))
            for lo, hi in unusable:
                if hi >= a and lo <= b:
                    xa = self.x_of(lo, a, b)
                    rect = QtCore.QRectF(xa, tops[5], max(self.x_of(hi + 1, a, b) - xa, 3), bar_h)
                    p.fillRect(rect, QtGui.QColor(*STATUS_COLORS[S.UNUSABLE]))
            pose = list(win.pose)
            if win.open_pose is not None:
                pose.append((*sorted((win.open_pose[0], win.frame)), win.open_pose[1]))
            for lo, hi, fish in pose:
                if hi < a or lo > b:
                    continue
                xa, xb = self.x_of(lo, a, b), max(self.x_of(hi + 1, a, b), self.x_of(lo, a, b) + 3)
                rect = QtCore.QRectF(xa, tops[6], xb - xa, bar_h)
                p.fillRect(rect, QtGui.QColor(POSE_COLOR))
                if self.zoom and xb - xa > 40:
                    left = max(rect.left(), LABEL_WIDTH + 2.0)
                    p.setPen(QtGui.QColor("#111111"))
                    p.drawText(QtCore.QRectF(left + 3, rect.top(), rect.right() - left - 3, bar_h),
                               QtCore.Qt.AlignVCenter, win.fish_names(fish, lo))
            p.setPen(QtGui.QPen(QtGui.QColor("#f2d13d"), 3))
            for f, _a, _b in win.swaps:
                if a <= f <= b:
                    x = self.x_of(f, a, b)
                    p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, 2 * row))
            for i, name in enumerate(LANE_NAMES):
                cell = QtCore.QRectF(0, i * row if i < 2 else tops[i], LABEL_WIDTH, row if i < 2 else bar_h)
                p.fillRect(cell, QtGui.QColor(0, 0, 0, 170))
                p.setPen(QtGui.QColor("#d0d0d0"))
                p.drawText(cell.adjusted(4, 0, 0, 0), QtCore.Qt.AlignVCenter, name)
            if self.zoom:
                p.setPen(QtGui.QColor("#d0d0d0"))
                p.drawText(QtCore.QRectF(0, 0, w - 6, row), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
                           f"+-{win.zoom_seconds:.0f} s (wheel)")
            x = self.x_of(win.frame, a, b)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            p.end()

        def wheelEvent(self, event):
            if self.zoom:
                factor = 1.25 ** (-event.angleDelta().y() / 120)
                self.win.zoom_seconds = float(np.clip(self.win.zoom_seconds * factor, *ZOOM_RANGE))
                self.update()

        def mousePressEvent(self, event):
            self._go(event.x())

        def mouseMoveEvent(self, event):
            if event.buttons() & QtCore.Qt.LeftButton:
                self._go(event.x())

        def _go(self, x: float):
            a, b = self.span()
            self.seek.emit(int(a + max(0.0, min(1.0, x / max(1, self.width()))) * (b - a)))

    class View3D(QtWidgets.QWidget):
        """The main view: the fish's nodes in the tank, with floor shadows and 0.5 s trails.

        Drag rotates, the wheel zooms towards the fish, a click selects the nearest fish, a double click
        resets.
        """

        def __init__(self, win):
            super().__init__()
            self.win, self._drag, self.zoom, self.points = win, None, 1.0, []
            self.az, self.el = VIEW3D_START
            self.setMinimumSize(300, 240)
            xyz = win.pkg.xyz.reshape(-1, 3)
            ok = np.isfinite(xyz).all(axis=1)
            self.lo, self.hi = (np.percentile(xyz[ok], [0.5, 99.5], axis=0) if ok.any()
                                else (np.zeros(3), np.full(3, 200.0)))
            self.centre = (self.lo + self.hi) / 2

        def mousePressEvent(self, event):
            self._drag = (event.x(), event.y(), self.az, self.el, False)

        def mouseMoveEvent(self, event):
            if self._drag:
                x0, y0, az, el, moved = self._drag
                moved = moved or abs(event.x() - x0) + abs(event.y() - y0) > 4
                self._drag = (x0, y0, az, el, moved)
                self.az = az - (event.x() - x0) * 0.4
                self.el = float(np.clip(el + (event.y() - y0) * 0.4, -89, 89))
                self.update()

        def mouseReleaseEvent(self, event):
            moved = self._drag is not None and self._drag[4]
            self._drag = None
            if not moved and self.points:
                d = [np.hypot(q[0] - event.x(), q[1] - event.y()) for _k, q in self.points]
                self.win.pick(self.points[int(np.argmin(d))][0])

        def mouseDoubleClickEvent(self, _event):
            self.az, self.el = VIEW3D_START
            self.zoom = 1.0
            self.update()

        def wheelEvent(self, event):
            self.zoom = float(np.clip(self.zoom * 1.15 ** (event.angleDelta().y() / 120), 1.0, 15.0))
            self.update()

        def proj(self, pts: np.ndarray, centre: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
            a, e = np.radians(self.az), np.radians(self.el)
            d = np.asarray(pts, float) - centre
            x1 = d[..., 0] * np.cos(a) - d[..., 1] * np.sin(a)
            y1 = d[..., 0] * np.sin(a) + d[..., 1] * np.cos(a)
            return np.stack([w / 2 + scale * x1, h / 2 - scale * (d[..., 2] * np.cos(e) + y1 * np.sin(e))], axis=-1)

        def paintEvent(self, _event):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 18, np.uint8)
            win, pkg = self.win, self.win.pkg
            lo, hi = self.lo, self.hi
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            fit = np.ptp(self.proj(corners, self.centre, 1.0, 0, 0), axis=0)
            scale = 0.94 * self.zoom * min(w / max(1.0, fit[0]), h / max(1.0, fit[1]))
            t = pkg.t(win.frame)
            centre = self.centre
            if t is not None and self.zoom > 1.0 and np.isfinite(pkg.xyz[t]).any():
                follow = min(1.0, self.zoom - 1.0)
                centre = (1 - follow) * self.centre + follow * np.nanmean(pkg.xyz[t], axis=0)

            def proj(q):
                return self.proj(q, centre, scale, w, h)

            for x in np.arange(np.ceil(lo[0] / 20) * 20, hi[0], 20):
                q = proj(np.array([[x, lo[1], lo[2]], [x, hi[1], lo[2]]]))
                cv2.line(img, pt(q[0]), pt(q[1]), (42, 42, 42), 1, cv2.LINE_AA)
            for y in np.arange(np.ceil(lo[1] / 20) * 20, hi[1], 20):
                q = proj(np.array([[lo[0], y, lo[2]], [hi[0], y, lo[2]]]))
                cv2.line(img, pt(q[0]), pt(q[1]), (42, 42, 42), 1, cv2.LINE_AA)
            q = proj(corners)
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") == 1:
                        cv2.line(img, pt(q[i]), pt(q[j]), (95, 95, 95), 1, cv2.LINE_AA)
            self.points = []
            if t is not None and pkg.nodes is not None:
                trail = np.arange(max(0, t - int(TRAIL_S * pkg.fps)), t + 1, 3)
                for k in range(len(pkg.fish_ids)):
                    color, name = win.style(k, t)
                    dim = tuple(int(c * 0.55) for c in color)
                    path = pkg.xyz[trail, k].astype(float)
                    path_ok = np.isfinite(path).all(axis=1)
                    if path_ok.sum() > 1:
                        cv2.polylines(img, [proj(path[path_ok]).round().astype(np.int32).reshape(-1, 1, 2)], False, dim,
                                      1, cv2.LINE_AA)
                    nodes = pkg.nodes[t, k].astype(float)
                    ok = np.isfinite(nodes).all(axis=1)
                    if not ok.any():
                        continue
                    centre_3d = nodes[ok].mean(axis=0)
                    drop = proj(np.array([centre_3d, [centre_3d[0], centre_3d[1], lo[2]]]))
                    cv2.line(img, pt(drop[0]), pt(drop[1]), (75, 75, 75), 1, cv2.LINE_AA)
                    cv2.circle(img, pt(drop[1]), 3, (100, 100, 100), -1, cv2.LINE_AA)
                    qn = proj(nodes)
                    for a, b in pkg.edges:
                        if ok[a] and ok[b]:
                            cv2.line(img, pt(qn[a]), pt(qn[b]), color, 3, cv2.LINE_AA)
                    for n in np.flatnonzero(ok):
                        cv2.circle(img, pt(qn[n]), 7 if n == pkg.head else 4, color, -1, cv2.LINE_AA)
                    anchor = qn[pkg.head] if ok[pkg.head] else qn[ok][0]
                    self.points.append((k, qn[ok].mean(axis=0)))
                    if win.selected == k:
                        cv2.circle(img, pt(qn[ok].mean(axis=0)), 34, (255, 255, 255), 2, cv2.LINE_AA)
                    org = (int(anchor[0]) + 10, int(anchor[1]) - 10)
                    cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(img, "drag rotate | wheel zoom | click select | double-click reset", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            p = QtGui.QPainter(self)
            p.drawImage(0, 0, QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))
            p.end()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.pkg = pkg = ReviewPackage(folder)
            self.ledger_folder = Path(ledger_dir) if ledger_dir else pkg.folder / LEDGER_FOLDER
            self.ledger_folder.mkdir(parents=True, exist_ok=True)
            self.ledger = Ledger(self.ledger_folder)
            self.lock = ReviewLock(self.ledger_folder)
            self.lock.acquire()
            self.setWindowTitle(f"Identity pass - {pkg.folder.parent.name}/{pkg.folder.name}")
            self.map = S.danger_map(pkg.xyz.astype(float), pkg.present, pkg.interp, pkg.fps, params)
            self.zone = self.map["danger"].astype(np.int8) + self.map["high_frames"].astype(np.int8)
            self.frame, self.direction, self.anchor, self.run = pkg.first, 0, None, None
            self.fast, self.slow = FAST_SPEED, SLOW_SPEED
            self.selected, self.open_danger, self.last_nav = None, None, "seek"
            self.open_pose = None        # (first frame, fish) of a wrong-pose range being marked
            self.next_folder, self._closed = None, False
            self.history = []            # this session's "add" and ("delete", events before, rows), for Ctrl+Z
            self.last_nudge = 0.0
            self.crop_on, self._crop, self._crop_frame = True, {}, {}
            self.zoom_seconds = ZOOM_SECONDS
            self.bouts = empty_bouts()
            found = boris if boris and boris.is_file() else find_boris_csv(
                pkg.folder.parent.name, default_search([boris] if boris else None))
            if found is not None:
                self.bouts = load_bouts(found)
            self.clock_file = self.ledger_folder / TIME_FILE
            self.saved_clock = (json.loads(self.clock_file.read_text(encoding="utf-8"))
                                if self.clock_file.is_file() else {"play_seconds": 0.0, "open_seconds": 0.0})
            self.opened, self.play_seconds, self.play_since = time.time(), 0.0, None
            self.play = [ProxyVideo(pkg.proxy_dir / pkg.proxy["views"][v]["play_file"],
                                    fps=proxy_fps(pkg.proxy, v, True)) for v in VIEWS]
            self.full = [ProxyVideo(pkg.proxy_dir / pkg.proxy["views"][v]["full_file"],
                                    fps=proxy_fps(pkg.proxy, v, False)) for v in VIEWS]
            self.view_tf = [None] * 3
            self.order = [0, 1, 2]       # order[0] = the view in the large slot

            self.info = QtWidgets.QLabel()
            self.info.setTextFormat(QtCore.Qt.RichText)
            self.info.setStyleSheet("background:#1b1b1b;color:#e8e8e8;padding:4px;font-size:15px;")
            self.views = [ViewLabel(i) for i in range(3)]
            for label in self.views:
                label.clicked.connect(self.on_click)
                label.double_clicked.connect(self.make_large)
            self.view3d = View3D(self)
            cameras = QtWidgets.QWidget()
            # As in the review GUI: one large camera, the other two stacked beside it at half its size.
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(self.views[1], 1)
            column.addWidget(self.views[2], 1)
            row = QtWidgets.QHBoxLayout(cameras)
            row.setSpacing(2)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self.views[0], 2)
            row.addLayout(column, 1)
            self.overview, self.zoom = Timeline(self, False), Timeline(self, True)
            self.overview.seek.connect(self.seek)
            self.zoom.seek.connect(self.seek)
            help_line = QtWidgets.QLabel(HELP)
            help_line.setWordWrap(True)
            help_line.setStyleSheet("color:#9a9993;padding:2px;")
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(2)
            box.setContentsMargins(2, 2, 2, 2)
            box.addWidget(self.info)
            self.view_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.view_splitter.addWidget(cameras)
            self.view_splitter.addWidget(self.view3d)
            self.view_splitter.setStretchFactor(1, 1)
            self.view_splitter.setHandleWidth(8)
            self.view_splitter.setStyleSheet("QSplitter::handle{background:#555;}")
            lines = QtWidgets.QWidget()
            lines_box = QtWidgets.QVBoxLayout(lines)
            lines_box.setSpacing(4)
            lines_box.setContentsMargins(0, 0, 0, 0)
            lines_box.addWidget(self.overview, 2)
            lines_box.addWidget(self.zoom, 3)
            self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.splitter.addWidget(self.view_splitter)
            self.splitter.addWidget(lines)
            self.splitter.setStretchFactor(0, 1)
            self.splitter.setHandleWidth(8)
            self.splitter.setStyleSheet("QSplitter::handle{background:#555;}")
            self.settings = QtCore.QSettings(*SETTINGS)
            box.addWidget(self.splitter, 1)
            box.addWidget(help_line)
            centre = QtWidgets.QWidget()
            centre.setLayout(box)
            centre.setStyleSheet("background:#111;")
            self.setCentralWidget(centre)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.tick_save = time.time()
            self._bind_keys()
            self.rebuild()
            self.frame = self.start or pkg.first

        def _bind_keys(self):
            keys = {"Space": self.toggle_play, "Up": lambda: self.change_speed("fast", +1),
                    "Down": lambda: self.change_speed("fast", -1), "Shift+Up": lambda: self.change_speed("slow", +1),
                    "Shift+Down": lambda: self.change_speed("slow", -1), "Left": lambda: self.nudge(-1),
                    "Right": lambda: self.nudge(+1), "Shift+Left": lambda: self.jump(-SEEK_SECONDS),
                    "Shift+Right": lambda: self.jump(SEEK_SECONDS), ",": lambda: self.step(-1),
                    ".": lambda: self.step(+1), "Esc": lambda: self.pick(None),
                    "X": self.toggle_danger, "B": self.toggle_pose, "S": self.mark_start, "Ctrl+Z": self.undo,
                    "Delete": self.delete_range, "Z": self.toggle_crop,
                    "W": lambda: self.goto_unwatched(+1), "Shift+W": lambda: self.goto_unwatched(-1),
                    "O": self.switch_experiment,
                    "G": self.goto_dialog, "P": self.save_and_show, "F1": self.show_help}
            for i, name in enumerate(("F", "M1", "M2")):
                keys[str(i + 1)] = lambda n=name, k=i: self.select(n, k)
                keys[str(i + 4)] = lambda n=name: self.set_name(n)
                keys[f"Shift+{i + 4}"] = lambda n=name: self.set_name(n, rename=True)
                keys["$%^"[i]] = lambda n=name: self.set_name(n, rename=True)
            keys["&"] = lambda: self.set_name("M2", rename=True)      # Shift+6 on a Japanese layout
            for key, slot in keys.items():
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.ApplicationShortcut)
                sc.activated.connect(slot)
                # Windows reports e.g. both "Shift+1" and "!" for one press; Qt then only emits this.
                sc.activatedAmbiguously.connect(slot)

        # -- state -------------------------------------------------------------
        def rebuild(self):
            pkg, events = self.pkg, self.ledger.events
            self.labels = static_labels(events, pkg.fish_ids)
            self.swaps = swap_list(events, pkg.fish_ids)
            self.owner = owner_series(len(pkg.fish_ids), self.swaps, pkg.frames)
            self.start = analysis_start(events)
            self.pose = S.pose_ranges(events)
            self.update_status()

        def update_status(self):
            pkg = self.pkg
            self.status = S.frame_status(pkg.first, len(pkg.frames), self.start, self.map["danger"], self.map["runs"],
                                         watched_intervals(self.ledger.coverage), [f for f, _a, _b in self.swaps],
                                         S.manual_ranges(self.ledger.events), pkg.fps)
            self.unusable = S.manual_ranges(self.ledger.events)
            self.segments = S.usable_segments(self.status, pkg.first, pkg.fps, self.names_at)
            self.stats = S.summary(self.status, self.segments, pkg.fps)

        def names_at(self, t: int) -> str:
            return " ".join(f"{self.labels.get(self.pkg.fish_ids[self.owner[t, k]]) or '?'}={fid}"
                            for k, fid in enumerate(self.pkg.fish_ids))

        def name_of(self, k: int, t: int) -> str:
            return self.labels.get(self.pkg.fish_ids[self.owner[t, k]], "")

        def fish_names(self, fish: list[str], frame: int) -> str:
            """The names of these tracks at ``frame``."""
            t = self.pkg.t(frame)
            names = [self.name_of(self.pkg.fish_ids.index(f), t) if t is not None else "" for f in fish]
            return " ".join(n or f for n, f in zip(names, fish))

        def style(self, k: int, t: int) -> tuple[tuple[int, int, int], str]:
            name = self.name_of(k, t)
            return (LABEL_COLORS[name], name) if name in LABEL_COLORS else (TRACK_COLORS[k], self.pkg.fish_ids[k])

        def speed_here(self) -> float:
            t = self.pkg.t(self.frame)
            return min(self.fast, self.slow) if t is not None and self.map["slow"][t] else self.fast

        def clock(self) -> tuple[float, float]:
            """Playback and window-open seconds, over all sessions."""
            running = time.time() - self.play_since if self.play_since else 0.0
            return (self.saved_clock["play_seconds"] + self.play_seconds + running,
                    self.saved_clock["open_seconds"] + time.time() - self.opened)

        def left_seconds(self) -> float:
            todo = self.status == S.UNWATCHED
            t = self.pkg.t(self.frame)
            if self.run is not None and t is not None:
                todo = todo.copy()
                todo[:t] = False
            slow = int((todo & self.map["slow"]).sum())
            return S.play_seconds(int(todo.sum()) - slow, slow, self.pkg.fps, self.fast, self.slow)

        # -- playback ----------------------------------------------------------
        def toggle_play(self):
            if self.direction:
                self.pause()
            else:
                self.direction = 1
                self.play_since = time.time()
                self.run = {"start": self.frame, "wall": time.time()}
                self._anchor()
                self.timer.start(10)
                self.render()

        def _anchor(self):
            self.anchor = (time.perf_counter(), self.frame, self.speed_here())

        def pause(self):
            if self.direction:
                self.timer.stop()
                self.direction = 0
                self.play_seconds += time.time() - self.play_since
                self.play_since = None
                self.flush()
                self.render()

        def flush(self):
            r, self.run = self.run, None
            if r is not None and self.frame > r["start"]:
                self.ledger.add_coverage(r["start"], self.frame, self.fast, "play", r["wall"], time.time())
                self.update_status()
            if self.direction:
                self.run = {"start": self.frame, "wall": time.time()}
            self.tick_save = time.time()
            self.save_clock()

        def save_clock(self):
            play, opened = self.clock()
            tmp = self.clock_file.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps({"play_seconds": round(play, 1), "open_seconds": round(opened, 1)}),
                               encoding="utf-8")
                tmp.replace(self.clock_file)
            except OSError:
                pass

        def on_tick(self):
            t0, f0, speed = self.anchor
            pkg = self.pkg
            target = int((f0 + (time.perf_counter() - t0) * pkg.fps * speed) // pkg.step * pkg.step)
            if target > min(pkg.last, pkg.n_source - 1):
                self.pause()
                return
            if target <= self.frame:
                return
            self.frame = target
            self.last_nav = "play"
            if abs(self.speed_here() - speed) > 1e-9:
                self._anchor()
            if time.time() - self.tick_save > FLUSH_SECONDS:
                self.flush()
            self.render()

        def change_speed(self, which: str, delta: int):
            current = getattr(self, which)
            i = min(range(len(SPEEDS)), key=lambda j: abs(SPEEDS[j] - current))
            setattr(self, which, SPEEDS[max(0, min(len(SPEEDS) - 1, i + delta))])
            if self.direction:
                self._anchor()
            self.render()

        def seek(self, frame: int):
            playing = bool(self.direction)
            self.pause()
            self.frame = int(max(self.pkg.first, min(self.pkg.last, frame)))
            self.last_nav = "seek"
            if playing:
                self.toggle_play()
            else:
                self.render()

        def jump(self, seconds: float):
            self.seek(self.frame + int(seconds * self.pkg.fps))

        def step(self, n: int):
            self.pause()
            self.frame = int(max(self.pkg.first, min(self.pkg.last, self.frame + n)))
            self.last_nav = "step"
            self.render()

        def nudge(self, sign: int):
            self.step(sign * nudge_frames(time.time() - self.last_nudge))
            self.last_nudge = time.time()     # after the render: repeats queued behind a slow one still count

        def goto_dialog(self):
            value, ok = QtWidgets.QInputDialog.getInt(self, "Go to frame", "Frame:", self.frame, self.pkg.first,
                                                      self.pkg.last)
            if ok:
                self.seek(value)

        def switch_experiment(self):
            """O: pick another experiment; this one is saved and closed, the next opens in a new window."""
            self.pause()
            chosen = pick_folder(paths_config, current=self.pkg.folder)
            if chosen is not None and Path(chosen).resolve() != self.pkg.folder.resolve():
                self.open_experiment(chosen)

        def open_experiment(self, target: Path):
            """Close this experiment as on quitting (a half-marked range asks first); ``run_app`` opens ``target``."""
            self.next_folder = Path(target)
            if not self.close():                  # the reviewer cancelled the question: stay here
                self.next_folder = None

        def goto_unwatched(self, sign: int):
            """W / Shift+W: the start of the next / previous stretch not watched yet (wraps around)."""
            self.pause()                                   # what was played so far counts as watched first
            runs = S.runs_of(self.status == S.UNWATCHED)
            if not len(runs):
                self.statusBar().showMessage("Every analysed frame has been watched.")
                return
            t = self.frame - self.pkg.first
            if sign > 0:
                ahead = np.flatnonzero(runs[:, 0] > t)
                wrapped, i = not len(ahead), int(ahead[0]) if len(ahead) else 0
            else:
                ahead = np.flatnonzero(runs[:, 0] < t)
                wrapped, i = not len(ahead), int(ahead[-1]) if len(ahead) else len(runs) - 1
            a, b = runs[i]
            self.seek(self.pkg.first + int(a))
            self.statusBar().showMessage(
                f"Unwatched stretch {i + 1} of {len(runs)}: frames {self.pkg.first + a}-{self.pkg.first + b - 1} "
                f"({(b - a) / self.pkg.fps:.1f} s)" + (" - wrapped around." if wrapped else "."))

        # -- reviewer actions ----------------------------------------------------
        def select(self, name: str, index: int):
            t = self.pkg.t(self.frame)
            hit = [k for k in range(len(self.pkg.fish_ids)) if t is not None and self.name_of(k, t) == name]
            self.pick(hit[0] if hit else index)

        def pick(self, k: int | None):
            """Select fish ``k``; the fish already selected, or ``None``, clears the selection."""
            self.selected = None if k == self.selected else k
            self.render()

        def on_click(self, slot: int, x: float, y: float):
            t, tf = self.pkg.t(self.frame), self.view_tf[slot]
            if t is None or tf is None:
                return
            pkg = self.pkg
            v = self.order[slot]
            q = np.array([to_display(pkg.fish_point(t, k, v), tf) for k in range(len(pkg.fish_ids))], float)
            d = np.linalg.norm(q - [x, y], axis=1)
            self.pick(int(np.nanargmin(d)) if np.isfinite(d).any() else None)

        def make_large(self, slot: int):
            """Double click: this camera takes the large slot, as in the review GUI."""
            if slot:
                self.order[0], self.order[slot] = self.order[slot], self.order[0]
                self._crop.clear()
                self.render()

        def set_name(self, name: str, rename: bool = False):
            """Give the selected fish ``name``: the first time for the whole recording, later from here on."""
            pkg = self.pkg
            t = pkg.t(self.frame)
            if self.selected is None or t is None:
                self.statusBar().showMessage("Select a fish first (1/2/3 or click).")
                return
            k = self.selected
            cid = pkg.fish_ids[self.owner[t, k]]
            holder = next((f for f, v in self.labels.items() if v == name and f != cid), None)
            if rename or holder is None:
                changes = assign_label(self.labels, cid, name)
                if changes:
                    self.add_event("label", changes, self.frame, self.frame)
                self.selected = None
                self.rebuild()
                self.statusBar().showMessage(f"Named for the whole recording: {changes}")
                self.render()
                return
            r = int(np.flatnonzero(self.owner[t] == pkg.fish_ids.index(holder))[0])
            distance = np.linalg.norm(pkg.xyz[:, k].astype(float) - pkg.xyz[:, r].astype(float), axis=1)
            touching = [f for f, a, b in self.swaps if f <= self.frame and {a, b} & {k, r}]
            frame = self.frame
            if self.last_nav != "step":
                frame = S.place_change(self.frame, pkg.first, distance, self.map["runs"],
                                       max([pkg.first - 1] + touching), int(CHANGE_WINDOW_S * pkg.fps))
            pair = PAIR_SEP.join(sorted((pkg.fish_ids[k], pkg.fish_ids[r])))
            self.add_event("swap", {pair: "1"}, frame, self.frame)
            self.selected = None
            self.rebuild()
            self.statusBar().showMessage(f"{name} from frame {frame} on (pressed at {self.frame}).")
            self.render()

        def toggle_danger(self):
            if self.open_danger is None:
                self.open_danger = self.frame
                self.statusBar().showMessage(f"Unusable range from {self.frame}; press X again at its end.")
            else:
                a, b = sorted((self.open_danger, self.frame))
                self.open_danger = None
                self.add_event("danger", {ALL_FISH: "1"}, a, self.frame, frame_end=b)
                self.update_status()
                self.statusBar().showMessage(f"Frames {a}-{b} are unusable.")
            self.render()

        def toggle_pose(self):
            """A wrong pose with the right identity: kept apart from identity, the frames stay usable."""
            fish = [self.pkg.fish_ids[self.selected]] if self.selected is not None else list(self.pkg.fish_ids)
            if self.open_pose is None:
                self.open_pose = (self.frame, fish)
                self.statusBar().showMessage(f"Wrong pose of {', '.join(fish)} from {self.frame}; press B again at its end.")
            else:
                (start, fish), self.open_pose = self.open_pose, None
                a, b = sorted((start, self.frame))
                self.add_event("pose_bad", {f: "1" for f in fish}, a, self.frame, frame_end=b)
                self.rebuild()
                self.statusBar().showMessage(f"Pose of {', '.join(fish)} marked wrong on frames {a}-{b}; identity unchanged.")
            self.render()

        def mark_start(self):
            self.add_event("analysis_start", {ALL_FISH: "1"}, self.frame, self.frame)
            self.rebuild()
            self.statusBar().showMessage(f"Analysis starts at frame {self.frame}.")
            self.render()

        def add_event(self, *args, **kwargs) -> str:
            group = self.ledger.add(*args, **kwargs)
            if group:
                self.history.append("add")
            return group

        def range_under_frame(self) -> pd.DataFrame:
            """Rows of the most recently written X or B range holding the current frame (empty if none)."""
            e = self.ledger.events
            marks = e[e["kind"].isin(("danger", "pose_bad"))]
            if marks.empty:
                return marks
            lo, hi = marks["frame"].astype(int), marks["frame_end"].astype(int)
            hit = marks[(lo <= self.frame) & (self.frame <= hi)]
            return hit if hit.empty else marks[marks["group"] == hit.iloc[-1]["group"]]

        def delete_range(self):
            """Remove the X or B range under the frame, from any session; logged to the removed-events file."""
            rows = self.range_under_frame()
            if rows.empty:
                self.statusBar().showMessage(f"No unusable (X) or wrong-pose (B) range holds frame {self.frame}.")
                return
            before = self.ledger.events.copy()
            self.ledger.remove_group(rows.iloc[0]["group"])
            self.history.append(("delete", before, rows))
            self.rebuild()
            word = "Unusable" if rows.iloc[0]["kind"] == "danger" else f"Wrong-pose ({', '.join(rows['fish'])})"
            self.statusBar().showMessage(f"{word} range {rows.iloc[0]['frame']}-{rows.iloc[0]['frame_end']} "
                                         "deleted. Ctrl+Z brings it back.")
            self.render()

        def restore_deleted(self, before: pd.DataFrame, rows: pd.DataFrame):
            """Undo a Delete: the ledger as it was, and the rows out of the removed-events file again."""
            ledger = self.ledger
            ledger.events = before
            ledger.events_added += int((rows["session"] == ledger.session).sum())
            ledger.save_events()
            log = read_csv(ledger.folder / REMOVED_FILE, EVENT_COLUMNS)
            ledger._save(REMOVED_FILE, log[~log["event_id"].isin(rows["event_id"])])

        def undo(self):
            if self.open_danger is not None or self.open_pose is not None:
                what = "unusable" if self.open_danger is not None else "wrong-pose"
                self.open_danger = self.open_pose = None
                self.statusBar().showMessage(f"Half-marked {what} range cancelled.")
                self.render()
                return
            if self.history and self.history[-1] != "add":
                _kind, before, rows = self.history.pop()
                self.restore_deleted(before, rows)
                self.rebuild()
                self.statusBar().showMessage(f"Deletion undone: range {rows.iloc[0]['frame']}-"
                                             f"{rows.iloc[0]['frame_end']} is back.")
                self.render()
                return
            removed = self.ledger.undo()
            if len(removed) and self.history:
                self.history.pop()
            self.rebuild()
            self.statusBar().showMessage(f"Undone: {', '.join(removed['kind'].unique())}" if len(removed)
                                         else "Nothing to undo in this session (Delete removes an older X/B range).")
            self.render()

        def toggle_crop(self):
            self.crop_on = not self.crop_on
            self._crop.clear()
            self.render()

        def show_help(self):
            QtWidgets.QMessageBox.information(self, "Keys", HELP.replace(" | ", "\n"))

        def write_result(self):
            self.flush()
            self.segments.to_csv(self.ledger_folder / "id_segments.csv", index=False)
            play, opened = self.clock()
            out = {**self.stats, "experiment": self.pkg.folder.parent.name, "package": self.pkg.folder.name,
                   "analysis_start": self.start, "identity_changes": len(self.swaps),
                   "unusable_marks": len(S.manual_ranges(self.ledger.events)),
                   "pose_marks": int((self.ledger.events["kind"] == "pose_bad").sum()), "play_seconds": round(play, 1),
                   "open_seconds": round(opened, 1), "fast_speed": self.fast, "slow_speed": self.slow,
                   "params": asdict(params)}
            (self.ledger_folder / "id_pass_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
            title = f"{self.pkg.folder.parent.name}  /  {self.pkg.folder.name}"
            figures = id_figure.write_figures(self.ledger_folder, self.map, self.ledger.events, self.ledger.coverage,
                                              self.pkg.first, len(self.pkg.frames), self.pkg.fps, title, params,
                                              self.pkg.fish_ids)
            self.statusBar().showMessage(f"Saved {self.stats['segments']} segments to {self.ledger_folder}.")
            return figures

        def save_and_show(self):
            figures = self.write_result()
            if scenario is None:
                for figure in figures:
                    os.startfile(figure)      # noqa: S606 - Windows, the reviewer's own image viewer

        # -- drawing -------------------------------------------------------------
        def crop_box(self, v: int, t: int, width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
            pkg = self.pkg
            half = int(CROP_HALF_S * pkg.fps)
            idx = np.arange(max(0, t - half), min(len(pkg.frames), t + half + 1), 6)
            target = crop_target(pkg.centres(idx, v).reshape(-1, 2) * pkg.scale[v], width, height)
            prev = self._crop.get(v)
            near = prev is not None and abs(self.frame - self._crop_frame.get(v, self.frame)) <= pkg.fps
            state = prev + CROP_SMOOTHING * (target - prev) if near else target
            self._crop[v], self._crop_frame[v] = state, self.frame
            return crop_rect(state, width, height, aspect)

        def draw_fish(self, img: np.ndarray, v: int, t: int, tf):
            pkg = self.pkg
            for k in range(len(pkg.fish_ids)):
                color, name = self.style(k, t)
                missing = None if pkg.missing_proj is None else pkg.missing_proj[t, k, v].astype(float)
                pts, ok = skeleton_points(pkg.kp2d[t, k, v].astype(float), missing)
                shown = np.isfinite(pts).all(axis=1)
                proj = pkg.proj2d[t, k, v].astype(float)
                anchor = None
                if shown.any():
                    q = to_display(pts, tf)
                    for a, b in pkg.edges:
                        if not (shown[a] and shown[b]):
                            continue
                        if not (ok[a] and ok[b]):
                            dashed_line(img, q[a], q[b], color, 1, 4)       # a node this view did not detect
                        elif pkg.overlap[t, k, v]:
                            dashed_line(img, q[a], q[b], color, 2, 6)
                        else:
                            cv2.line(img, pt(q[a]), pt(q[b]), color, 2, cv2.LINE_AA)
                    for n in np.flatnonzero(shown & ~ok):
                        c = pt(q[n])
                        cv2.rectangle(img, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), color, 1, cv2.LINE_AA)
                    anchor = q[pkg.head] if shown[pkg.head] else q[shown][0]
                    cv2.circle(img, pt(anchor), 5, color, -1 if ok[pkg.head] else 1, cv2.LINE_AA)
                elif np.isfinite(proj).all():
                    anchor = to_display(proj, tf)
                    cv2.drawMarker(img, pt(anchor), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
                if anchor is None:
                    continue
                if self.selected == k:
                    cv2.circle(img, pt(anchor), 26, (255, 255, 255), 2, cv2.LINE_AA)
                org = (int(anchor[0]) + 9, int(anchor[1]) - 9)
                cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        def render(self):
            pkg, f = self.pkg, self.frame
            t = pkg.t(f)
            for slot, label in enumerate(self.views):
                v = self.order[slot]
                img = self.play[v].get(f // pkg.step) if f % pkg.step == 0 else self.full[v].get(f)
                if img is None:
                    meta = pkg.proxy["views"][VIEWS[v]]
                    img = np.zeros((meta["proxy_height"], meta["proxy_width"], 3), np.uint8)
                h, w = img.shape[:2]
                lw, lh = max(50, label.width()), max(50, label.height())
                x0, y0, cw, ch = (self.crop_box(v, t, w, h, lw / lh) if self.crop_on and t is not None
                                  else (0, 0, w, h))
                scale = min(lw / cw, lh / ch)
                out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (max(1, round(cw * scale)), max(1, round(ch * scale))),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                tf = (pkg.scale[v] * scale, -x0 * scale, -y0 * scale)
                self.view_tf[slot] = tf
                if t is not None:
                    self.draw_fish(out, v, t, tf)
                cv2.putText(out, VIEW_TITLES[v], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
                label.show_image(np.ascontiguousarray(out[:, :, ::-1]))
            self.update_info(t)
            self.overview.update()
            self.zoom.update()
            self.view3d.update()

        def update_info(self, t: int | None):
            zone = int(self.zone[t]) if t is not None else 0
            word, color = (("SAFE", "#7fd18b"), ("DANGER", "#e0a050"), ("HIGH RISK", "#ff7070"))[zone]
            play, opened = self.clock()
            s = self.stats
            state = "playing" if self.direction else "paused"
            self.info.setText(
                f"frame {self.frame} ({mmss(self.frame / self.pkg.fps)}) &nbsp; <b style='color:{color}'>{word}</b> "
                f"&nbsp; {state} x{self.speed_here():g} (fast x{self.fast:g}, slow x{self.slow:g}) &nbsp;|&nbsp; "
                f"<b>play {mmss(play)}</b>, open {mmss(opened)}, left about {mmss(self.left_seconds())} &nbsp;|&nbsp; "
                f"watched {s['watched_pct']:.1f}%, usable {s['usable_pct']:.1f}% "
                f"(safe {s['usable_safe_pct']:.1f} + danger {s['usable_danger_pct']:.1f}), {s['segments']} segments, "
                f"longest {s['longest_segment_s']:.0f} s &nbsp;|&nbsp; {len(self.swaps)} changes"
                + ("" if self.start is not None else " &nbsp; <b style='color:#ff7070'>press S at the first usable frame</b>")
                + ("" if all(self.labels.values()) else " &nbsp; <b style='color:#ff7070'>name the fish: 1/2/3 then 4/5/6</b>")
                + ("" if self.open_danger is None else
                   f" &nbsp; <b style='color:#ff7070'>MARKING UNUSABLE from {self.open_danger}: X at its end</b>")
                + ("" if self.open_pose is None else
                   f" &nbsp; <b style='color:{POSE_COLOR}'>MARKING POSE from {self.open_pose[0]}: B at its end</b>"))

        def size_timelines(self):
            """The divider where it was left, else the default share of the height."""
            state = self.settings.value("splitter")
            if state is None or not self.splitter.restoreState(state):
                total = sum(self.splitter.sizes()) or self.height()
                self.splitter.setSizes([round(total * (1 - TIMELINE_SHARE)), round(total * TIMELINE_SHARE)])
            state = self.settings.value("views_3d_right")
            if state is None or not self.view_splitter.restoreState(state):
                total = sum(self.view_splitter.sizes()) or self.width()
                self.view_splitter.setSizes([round(total * (1 - VIEW3D_SHARE)), round(total * VIEW3D_SHARE)])

        def ask_open_ranges(self) -> str:
            """``keep`` (end the half-marked ranges here), ``discard`` or ``cancel`` (do not close)."""
            if scenario is not None or (self.open_danger is None and self.open_pose is None):
                return "keep"
            starts = [f"unusable from {self.open_danger}"] if self.open_danger is not None else []
            starts += [f"wrong pose from {self.open_pose[0]}"] if self.open_pose is not None else []
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Half-marked range")
            box.setText(f"Still open: {', '.join(starts)}.\nEnd it at frame {self.frame}, or discard it?")
            keep = box.addButton(f"End at {self.frame}", QtWidgets.QMessageBox.AcceptRole)
            discard = box.addButton("Discard", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec_()
            return "keep" if box.clickedButton() is keep else "discard" if box.clickedButton() is discard else "cancel"

        def closeEvent(self, event):
            if self._closed:                    # closed already, e.g. to open another experiment
                event.accept()
                return
            self.pause()
            choice = self.ask_open_ranges()
            if choice == "cancel":
                event.ignore()
                return
            if scenario is None:                # a scripted run must not move the reviewer's layout
                self.settings.setValue("splitter", self.splitter.saveState())
                self.settings.setValue("views_3d_right", self.view_splitter.saveState())
            if choice == "discard":
                self.open_danger = self.open_pose = None
            if self.open_danger is not None:
                self.toggle_danger()
            if self.open_pose is not None:
                self.toggle_pose()
            if self.stats["watched_pct"] >= REVIEW_DONE_PCT:
                self.save_and_show()
            else:
                self.write_result()
            self.ledger.close()
            self.lock.release()
            for video in self.play + self.full:
                video.close()
            self._closed = True
            event.accept()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    while True:
        win = MainWindow()
        win.resize(1500, 950)
        if scenario is not None:
            win.show()
            win.size_timelines()
            app.processEvents()
            result = scenario(app, win)
            win.close()
            return result
        install_error_log(win.ledger_folder / "id_pass_gui_errors.log", win)
        win.showMaximized()
        QtCore.QTimer.singleShot(0, lambda w=win: (w.size_timelines(), w.render()))
        code = app.exec_()
        if win.next_folder is None:
            return code
        # The next experiment has its own ledger and its own BORIS file.
        folder, ledger_dir = win.next_folder, None
        boris = None if boris is not None and Path(boris).is_file() else boris


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", type=Path, help="<review_root>/<exp>/<tag> (asks if omitted)")
    p.add_argument("--paths-config", type=Path, default=None)
    p.add_argument("--boris", type=Path, default=None, help="BORIS CSV or a folder holding <experiment>.csv")
    p.add_argument("--safe-mm", type=float, default=S.SegmentParams.safe_mm)
    p.add_argument("--high-risk-mm", type=float, default=S.SegmentParams.high_risk_mm)
    p.add_argument("--min-safe-s", type=float, default=S.SegmentParams.min_safe_s)
    args = p.parse_args(argv)
    folder = args.folder or pick_folder(args.paths_config)
    if folder is None:
        return 1
    params = S.SegmentParams(safe_mm=args.safe_mm, high_risk_mm=args.high_risk_mm, min_safe_s=args.min_safe_s)
    return run_app(folder, boris=args.boris, params=params, paths_config=args.paths_config)


if __name__ == "__main__":
    raise SystemExit(main())
