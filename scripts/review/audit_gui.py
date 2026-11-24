"""Audit GUI: go through the audit items one by one (task list and statistics: :mod:`scripts.review.audit`).

::

    python -m scripts.review.audit_gui <reference folder> --recon-dir <reference>/recon_reviewed_old_m3dfull

The task list is built on first start (fixed random seed). Each item opens on its own frames and loops
them; answer with the number keys - the next unanswered item opens by itself.

* **id / geometry** (old vs new, blind): the two versions are called A and B; which is old and which new
  is random per item and hidden. Each is judged on its own: first only A is shown and judged, then only B.
  Judge the 3D (left); the camera views are an aid. Tab shows the other one / both for a look; A / B
  switch which one you are judging.
* **SLEAP**: raw SLEAP instances (magenta: triangle mouth, square pectoral, x tail). Judge the view marked
  with a yellow frame; a yellow ring marks the instance the item is about.
* **lost**: the same, but the yellow ring marks where the 3D puts a fish that has no detection in that view
  (fish close together); the automatic group of the item is not shown.

Keys:
  1 right   2 / 3 wrong (see the buttons)   0 unsure   Backspace or Ctrl+Z undo the last answer   C note
  A / B judge version A / B (old vs new items)
  Enter / N next unanswered item   PageDown / PageUp next / previous item   Home back to the key frame
  Space play/pause (loops the item)   Left/Right 1 frame   Shift+Left/Right 10 frames   Up/Down speed
  Tab or T: A+B / only A / only B   R raw SLEAP points (old vs new items; SLEAP items always show them)
  Z zoom (camera views)   F1 this help
  3D view: left-drag rotate, right-drag move, wheel zoom, double-click reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.review.audit import ANSWERS, QUESTIONS, SLEAP_CATEGORIES, TASKS_FILE, Audit, answer_kind, audit_dir, build
from scripts.review.compare_gui import ReconCompareData, draw_pair, project, render_view
from scripts.review.recon_judge_gui import draw_raw, draw_raw_legend
from scripts.review.review_gui import VIEW_TITLES, VIEWS, crop_target, pt
from scripts.review.slp_raw import RawSlp

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TICK_HZ = 30.0
PLAY_CACHE = 90                 # playback-proxy frames per view: a 3 s clip at 1x loops from memory
SHOW_MODES = ("A dashed + B solid", "only A", "only B")
SHOW_COLORS = ((230, 230, 230), (255, 230, 120), (120, 230, 255))   # BGR
JUDGE_COLOR = (0, 215, 255)          # BGR yellow: the view / instance an item is about
DONE_RGB = (205, 230, 205)
TASK_TITLES = {"id": "ID event (old vs new, blind)", "geometry": "3D body (old vs new, blind)",
               "sleap": "SLEAP: suspicious frame", "clean": "SLEAP: random unflagged frame",
               "lost": "SLEAP: fish without a detection (close encounter)"}
LOST_RING_PX = 55                    # source pixels (about half a body length)
ERROR_LOG = "audit_gui_errors.log"


def mask_version(q: np.ndarray, keep: bool) -> np.ndarray:
    return q if keep else np.full_like(q, np.nan)


def run_app(folder: Path, recon_dir: Path, *, smoke_test: bool = False,
            screenshot: tuple[Path, int] | None = None, check=None) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    from scripts.review.review_gui import install_error_log

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    if not (audit_dir(recon_dir) / TASKS_FILE).exists():
        build(folder, recon_dir)
    au = Audit(recon_dir)
    cd = ReconCompareData(folder, recon_dir)
    cd.play_cache = PLAY_CACHE
    raw = RawSlp.open(folder)
    tasks = au.tasks
    n_fish = len(cd.labels)

    class ViewLabel(QtWidgets.QLabel):
        def __init__(self):
            super().__init__()
            self.setMinimumSize(160, 160)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setStyleSheet("background:#111;")
            self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        def show_image(self, bgr: np.ndarray):
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            h, w = rgb.shape[:2]
            self.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)))

    class View3D(QtWidgets.QWidget):
        """A (dashed) and B (solid) in 3D: left-drag rotate, right-drag move, wheel zoom, double-click reset."""

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.setMinimumSize(200, 200)
            self.reset()
            p = cd.bounds_nodes()
            ok = np.isfinite(p).all(axis=1)
            lo, hi = (np.percentile(p[ok], [0.5, 99.5], axis=0) if ok.any() else (np.zeros(3), np.full(3, 200.0)))
            self.lo, self.hi = lo - 5, hi + 5
            self._drag = None

        def reset(self):
            self.az, self.el, self.zoom, self.pan = 30.0, 35.0, 1.0, np.zeros(2)
            self.update()

        def mousePressEvent(self, e):
            self._drag = (e.button() != QtCore.Qt.LeftButton, e.x(), e.y(), self.az, self.el, self.pan.copy())

        def mouseMoveEvent(self, e):
            if self._drag:
                moving, x0, y0, az, el, pan = self._drag
                if moving:
                    self.pan = pan + [e.x() - x0, e.y() - y0]
                else:
                    self.az, self.el = az - (e.x() - x0) * 0.4, float(np.clip(el + (e.y() - y0) * 0.4, -89, 90))
                self.update()

        def mouseReleaseEvent(self, _e):
            self._drag = None

        def contextMenuEvent(self, e):
            e.accept()

        def wheelEvent(self, e):
            f = float(np.clip(self.zoom * 1.15 ** (e.angleDelta().y() / 120), 0.5, 40.0)) / self.zoom
            c = np.array([e.x() - self.width() / 2, e.y() - self.height() / 2])
            self.pan = c - (c - self.pan) * f
            self.zoom *= f
            self.update()

        def mouseDoubleClickEvent(self, _e):
            self.reset()

        def paintEvent(self, _e):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 18, np.uint8)
            win = self.win
            nodes = win.nodes_ab()
            if nodes is None:
                cv2.putText(img, "SLEAP item: judge the raw points in the camera view with the yellow frame",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
            else:
                a, b = nodes
                scale = self.zoom * min(w, h) / (float((self.hi - self.lo).max()) * 1.3)
                pts = np.concatenate([a[:, cd.pos_node], b[:, cd.pos_node]])
                ok = np.isfinite(pts).all(axis=1)
                centre = pts[ok].mean(axis=0) if ok.any() else (self.lo + self.hi) / 2
                ox, oy = w / 2 + self.pan[0], h / 2 + self.pan[1]
                a_, e_ = np.radians(self.az), np.radians(self.el)

                def proj(p):
                    d = np.asarray(p, float) - centre
                    x1 = d[..., 0] * np.cos(a_) - d[..., 1] * np.sin(a_)
                    y1 = d[..., 0] * np.sin(a_) + d[..., 1] * np.cos(a_)
                    return np.stack([ox + scale * x1, oy - scale * (d[..., 2] * np.cos(e_) + y1 * np.sin(e_))], -1)

                lo, hi = self.lo, self.hi
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                qc = proj(corners)
                for i in range(8):
                    for j in range(i + 1, 8):
                        if bin(i ^ j).count("1") == 1:
                            cv2.line(img, pt(qc[i]), pt(qc[j]), (70, 70, 70), 1, cv2.LINE_AA)
                floor = lambda q: np.concatenate([q[..., :2], np.full(q.shape[:-1] + (1,), lo[2])], -1)  # noqa: E731
                for q in (a, b):
                    qs = proj(floor(q))
                    for c in range(len(q)):
                        okn = np.isfinite(qs[c]).all(axis=1)
                        for i, j in cd.edges:
                            if okn[i] and okn[j]:
                                cv2.line(img, pt(qs[c, i]), pt(qs[c, j]), (60, 60, 60), 2, cv2.LINE_AA)
                qa, qb = proj(a), proj(b)
                draw_pair(img, cd.labels, cd.edges, cd.head, cd.pos_node, qa, qb, 1.4, cross=False)
                k = win.item_fish()
                if k is not None:
                    for q in (qa, qb):
                        if np.isfinite(q[k, cd.pos_node]).all():
                            cv2.circle(img, pt(q[k, cd.pos_node]), 18, JUDGE_COLOR, 2, cv2.LINE_AA)
                hint = "green = both the same" if win.show_mode == 0 else "Tab: next"
                cv2.putText(img, f"3D: {win.show_text()} ({hint})", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            SHOW_COLORS[win.show_mode], 2, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))
            painter.end()

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Audit - {cd.experiment} (old vs new blind; SLEAP)")
            self.i = 0                                 # row in ``tasks``
            self.frame = int(tasks.iloc[0]["frame"]) if len(tasks) else cd.first
            self.playing, self.speed_i, self.show_mode, self.show_raw, self.zoom_on = False, SPEEDS.index(1.0), 0, False, True
            self.part = ""                              # version being judged: "A" / "B" (old vs new items)
            self.history: list[tuple[int, str]] = []    # answers given in this session, for undo
            self.crops = [None, None, None]
            self._acc = 0.0

            Split = QtWidgets.QSplitter
            outer = Split(QtCore.Qt.Horizontal)
            self.setCentralWidget(outer)
            left = Split(QtCore.Qt.Vertical)
            outer.addWidget(left)
            views = Split(QtCore.Qt.Horizontal)
            self.view3d = View3D(self)
            views.addWidget(self.view3d)
            cams = Split(QtCore.Qt.Vertical)
            self.views = [ViewLabel() for _ in VIEWS]
            cams.addWidget(self.views[0])
            sides = Split(QtCore.Qt.Horizontal)
            sides.addWidget(self.views[1])
            sides.addWidget(self.views[2])
            cams.addWidget(sides)
            views.addWidget(cams)
            views.setSizes([550, 550])
            left.addWidget(views)
            bottom = QtWidgets.QWidget()
            blay = QtWidgets.QVBoxLayout(bottom)
            blay.setContentsMargins(4, 4, 4, 4)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.valueChanged.connect(self.on_slider)
            blay.addWidget(self.slider)
            self.status = QtWidgets.QLabel()
            self.status.setTextFormat(QtCore.Qt.RichText)
            self.status.setWordWrap(True)
            blay.addWidget(self.status)
            left.addWidget(bottom)
            left.setSizes([800, 90])

            right = QtWidgets.QWidget()
            rlay = QtWidgets.QVBoxLayout(right)
            font = right.font()
            font.setPointSize(11)
            right.setFont(font)
            self.header = QtWidgets.QLabel()
            self.header.setTextFormat(QtCore.Qt.RichText)
            self.header.setWordWrap(True)
            rlay.addWidget(self.header)
            self.question = QtWidgets.QLabel()
            self.question.setWordWrap(True)
            self.question.setStyleSheet("background:#fff8d0; padding:8px; border:1px solid #d8c060; font-size:13pt;")
            rlay.addWidget(self.question)
            self.buttons_box = QtWidgets.QWidget()
            self.buttons = QtWidgets.QGridLayout(self.buttons_box)
            rlay.addWidget(self.buttons_box)
            row = QtWidgets.QHBoxLayout()
            for text, fn in (("Backspace  undo last answer", self.undo), ("C  note", self.note),
                             ("Enter  next unanswered", self.next_unanswered)):
                b = QtWidgets.QPushButton(text)
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                b.clicked.connect(fn)
                row.addWidget(b)
            rlay.addLayout(row)
            self.current = QtWidgets.QLabel()
            self.current.setWordWrap(True)
            rlay.addWidget(self.current)
            self.progress = QtWidgets.QLabel()
            self.progress.setWordWrap(True)
            rlay.addWidget(self.progress)
            frow = QtWidgets.QHBoxLayout()
            frow.addWidget(QtWidgets.QLabel("show"))
            self.filter = QtWidgets.QComboBox()
            self.filter.addItems(["all", "id", "geometry", "sleap", "clean", "lost"])
            self.filter.setFocusPolicy(QtCore.Qt.NoFocus)
            self.filter.currentIndexChanged.connect(self.fill_table)
            frow.addWidget(self.filter, 1)
            rlay.addLayout(frow)
            self.table = QtWidgets.QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(["item", "task", "group", "answer"])
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setFocusPolicy(QtCore.Qt.NoFocus)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.cellClicked.connect(lambda r, _c: self.open_item(int(self.table.item(r, 0).text())))
            rlay.addWidget(self.table, 1)
            outer.addWidget(right)
            outer.setSizes([1150, 450])
            for s in (outer, left, views, cams, sides):
                s.setHandleWidth(6)
                s.setChildrenCollapsible(False)

            self._actions = self.key_actions()
            for key, fn in self._actions.items():
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.WindowShortcut)
                sc.activated.connect(fn)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(int(1000 / TICK_HZ))
            self.fill_table()
            first = next((int(i) for i in tasks["item"] if not au.done(int(i))), int(tasks.iloc[0]["item"]))
            self.open_item(first)

        # -- item ----------------------------------------------------------------
        @property
        def row(self) -> dict:
            if getattr(self, "_row_i", None) != self.i:     # cached: read many times per frame
                self._row, self._row_i = tasks.iloc[self.i].to_dict(), self.i
            return self._row

        def kind(self) -> str:
            return answer_kind(self.row["task"])

        def is_recon(self) -> bool:
            return self.row["task"] in ("id", "geometry")

        def open_item(self, item: int, part: str | None = None):
            idx = np.flatnonzero(tasks["item"].to_numpy() == item)
            if not len(idx):
                return
            self.i = int(idx[0])
            r = self.row
            self.crops = [None, None, None]
            self.slider.blockSignals(True)
            self.slider.setRange(int(r["start"]), int(r["end"]))
            self.slider.blockSignals(False)
            self.view3d.setVisible(self.is_recon())        # SLEAP items: the camera views get the room
            self.build_buttons()
            self.select_row(item)
            parts = au.parts(item)
            if part is None:
                part = next((q for q in parts if au.code(item, q) == ""), parts[0])
            self.set_part(part)

        def set_part(self, part: str):
            """Judge version ``part`` (A / B; "" for SLEAP items): show it alone and start the clip again."""
            if part not in au.parts(int(self.row["item"])):
                return
            r = self.row
            self.part = part
            self.show_mode = {"A": 1, "B": 2}.get(part, 0)
            # An ID event plays from its start (judge the whole clip); everything else opens paused on its key
            # frame, the frame that is judged (the frames around it are context).
            self.frame = int(r["start"]) if r["task"] == "id" else int(r["frame"])
            self.playing = r["task"] == "id"
            self.update_panel()
            self.render()

        def build_buttons(self):
            while self.buttons.count():
                w = self.buttons.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
            for n, (code, text) in enumerate(ANSWERS[self.kind()].items()):
                b = QtWidgets.QPushButton(f"{code}   {text}")
                b.setFocusPolicy(QtCore.Qt.NoFocus)
                b.setMinimumHeight(34)
                b.clicked.connect(lambda _=False, c=code: self.answer(c))
                self.buttons.addWidget(b, n // 2, n % 2)

        def item_fish(self) -> int | None:
            name = str(self.row["fish"])
            return cd.labels.index(name) if name in cd.labels else None

        def show_text(self) -> str:
            return SHOW_MODES[self.show_mode]

        def nodes_ab(self):
            """3D nodes ``(L, N, 3)`` of A and B at the current frame (hidden one NaN), None for SLEAP items."""
            if not self.is_recon():
                return None
            t = cd.t(self.frame)
            if t is None:
                return None
            aligned = self.row["task"] == "geometry"
            a = cd.series(str(self.row["a_is"]), t, t + 1, aligned)[0]
            b = cd.series(str(self.row["b_is"]), t, t + 1, aligned)[0]
            if self.show_mode == 1:                      # one version alone is drawn solid
                return np.full_like(a, np.nan), a
            if self.show_mode == 2:
                return np.full_like(b, np.nan), b
            return a, b

        # -- answers -----------------------------------------------------------
        def answer(self, code: str):
            if code not in ANSWERS[self.kind()]:
                return
            item = int(self.row["item"])
            au.answer(item, code, part=self.part)
            self.history.append((item, self.part))
            self.refresh_row(item)
            rest = [q for q in au.parts(item) if au.code(item, q) == ""]
            if rest:
                self.set_part(rest[0])                   # A answered: now B
            else:
                self.next_unanswered()

        def undo(self):
            """Take back the last answer of this session and open it again (otherwise clear the current one)."""
            while self.history:
                item, part = self.history.pop()
                if au.code(item, part) != "":
                    au.answer(item, "", "", part=part)
                    self.refresh_row(item)
                    self.open_item(item, part)
                    return
            self.clear_answer()

        def clear_answer(self):
            item = int(self.row["item"])
            au.answer(item, "", "", part=self.part)
            self.refresh_row(item)
            self.update_panel()

        def note(self):
            item = int(self.row["item"])
            old_code, old_note = au.answers.get((item, self.part), ("", ""))
            what = f"item {item}" + (f", version {self.part}" if self.part else "")
            text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Note", f"note on {what}:", old_note)
            if ok:
                au.answer(item, old_code, text, part=self.part)
                self.update_panel()

        def visible_items(self) -> list[int]:
            f = self.filter.currentText()
            g = tasks if f == "all" else tasks[tasks["task"] == f]
            return [int(i) for i in g["item"]]

        def next_unanswered(self):
            items = self.visible_items()
            cur = int(self.row["item"])
            order = [i for i in items if i > cur] + [i for i in items if i <= cur]
            nxt = next((i for i in order if not au.done(i)), None)
            if nxt is None:
                self.update_panel()
                QtWidgets.QMessageBox.information(self, "Audit", "Every item shown here is answered.\n"
                                                  "Results: python -m scripts.review.audit stats ...")
                return
            self.open_item(nxt)

        def step_item(self, d: int):
            items = self.visible_items()
            cur = int(self.row["item"])
            if cur in items:
                j = items.index(cur) + d
            else:
                j = 0
            if 0 <= j < len(items):
                self.open_item(items[j])

        # -- table / panel -------------------------------------------------------
        def fill_table(self):
            items = self.visible_items()
            self.table.setRowCount(len(items))
            sub = tasks.set_index("item")
            for r, item in enumerate(items):
                row = sub.loc[item]
                for c, text in enumerate([str(item), row["task"], row["category"] if row["task"] == "sleap" else "",
                                          au.summary_text(item)]):
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                self.paint_row(r, item)
            self.table.resizeColumnsToContents()

        def paint_row(self, r: int, item: int):
            done = au.done(item)
            for c in range(4):
                cell = self.table.item(r, c)
                if cell is not None:
                    cell.setBackground(QtGui.QColor(*DONE_RGB) if done else QtGui.QColor(255, 255, 255))
                    cell.setForeground(QtGui.QColor(0, 0, 0))

        def refresh_row(self, item: int):
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).text() == str(item):
                    self.table.item(r, 3).setText(au.summary_text(item))
                    self.paint_row(r, item)
                    break

        def select_row(self, item: int):
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).text() == str(item):
                    self.table.selectRow(r)
                    self.table.scrollToItem(self.table.item(r, 0), QtWidgets.QAbstractItemView.PositionAtCenter)
                    return
            self.table.clearSelection()

        def update_panel(self):
            r = self.row
            task = r["task"]
            parts = [f"<b>item {r['item']} of {len(tasks)}</b> &nbsp; {TASK_TITLES[task]}",
                     f"frames {r['start']}-{r['end']}, key frame {r['frame']}"]
            if task == "sleap":
                parts.insert(1, f"group <b>{r['category']}</b>: {SLEAP_CATEGORIES[r['category']]}; "
                                f"view <b>{VIEW_TITLES[int(r['view'])]}</b>")
            if task == "lost":
                parts.insert(1, f"view <b>{VIEW_TITLES[int(r['view'])]}</b> (yellow ring = expected fish position)")
            if task == "geometry" and r["fish"]:
                parts.insert(1, f"mainly <b>{r['fish']}</b> (yellow ring)")
            self.header.setText("<br>".join(parts))
            q = QUESTIONS[r["category"] if task == "sleap" else task]
            if self.part:
                q = f"Now judging {self.part}.  " + q.replace("*", self.part)
            elif int(r["view"]) >= 0:
                q = q.replace("*", VIEW_TITLES[int(r["view"])])
            self.question.setText(q)
            item = int(r["item"])
            note = au.answers.get((item, self.part), ("", ""))[1]
            self.current.setText(f"answer: <b>{au.summary_text(item) or '-'}</b>"
                                 + (f" &nbsp; note: {note}" if note else ""))
            self.current.setTextFormat(QtCore.Qt.RichText)
            prog = au.progress()
            self.progress.setText("done: " + ", ".join(f"{t} {a}/{n}" for t, (a, n) in prog.items())
                                  + f" &nbsp; (saved in {au.dir.name}/audit_answers.csv)")
            self.progress.setTextFormat(QtCore.Qt.RichText)

        # -- playback ----------------------------------------------------------
        def tick(self):
            if not self.playing:
                return
            self._acc += SPEEDS[self.speed_i] * cd.fps / TICK_HZ
            n = int(self._acc)
            step = cd.step
            if n >= step:                            # fast enough for the playback proxy: whole steps on its frames
                n -= n % step
            if n < 1:
                return
            self._acc -= n
            r = self.row
            f = self.frame + n
            if n >= step:
                f -= f % step
            if f > int(r["end"]):                    # loop: back to the clip's first playback-proxy frame
                start = int(r["start"])
                f = start + (-start) % step if n >= step else start
            self.seek(f)

        def seek(self, f: int):
            r = self.row
            self.frame = int(np.clip(f, int(r["start"]), int(r["end"])))
            self.render()

        def on_slider(self, v: int):
            if v != self.frame:
                self.frame = int(v)
                self.render()

        def step(self, d: int):
            self.playing = False
            self.seek(self.frame + d)

        def set_speed(self, i: int):
            self.speed_i = int(np.clip(i, 0, len(SPEEDS) - 1))
            self.render()

        # -- drawing -------------------------------------------------------------
        def render(self):
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame)
            self.slider.blockSignals(False)
            r = self.row
            recon = self.is_recon()
            t = cd.t(self.frame)
            empty = np.full((n_fish, len(cd.node_names), len(VIEWS), 2), np.nan)
            pa = pb = empty
            nodes = self.nodes_ab()
            if nodes is not None:
                pa, pb = project(cd, nodes[0]), project(cd, nodes[1])
            rawp = raw.points(self.frame) if raw is not None else None
            judged_view = int(r["view"])
            inst = int(r["instance"])
            lost = np.array([float(r["x"]), float(r["y"])]) if r["task"] == "lost" else None
            for v, lab in enumerate(self.views):
                size = (max(50, lab.width()), max(50, lab.height()))
                crop = None
                if self.zoom_on:
                    if recon:
                        k = self.item_fish()
                        ks = [k] if k is not None else list(range(n_fish))
                        pts = np.concatenate([pa[ks][:, :, v], pb[ks][:, :, v]]).reshape(-1, 2) * cd.scale[v]
                    elif lost is not None and v == judged_view:
                        pts = (lost + np.array([[-2, -2], [2, 2]]) * LOST_RING_PX) * cd.scale[v]
                    elif rawp is not None:
                        sel = rawp[v][[inst]] if (v == judged_view and inst >= 0) else rawp[v]
                        pts = sel.reshape(-1, 2) * cd.scale[v]
                    else:
                        pts = np.empty((0, 2))
                    meta = cd.proxy["views"][VIEWS[v]]
                    target = crop_target(pts, meta["proxy_width"], meta["proxy_height"])
                    self.crops[v] = target if self.crops[v] is None else self.crops[v] + 0.35 * (target - self.crops[v])
                    crop = self.crops[v]
                img, tf = render_view(cd, v, self.frame, ("", ""), False, size, crop, projections=(pa, pb))
                off = np.array(tf[1:])
                if rawp is not None and (self.show_raw or not recon):
                    n_raw = draw_raw(img, rawp[v] * tf[0] + off, raw.node_names, raw.edges)
                    draw_raw_legend(img, n_raw, raw.node_names, 46)
                    if v == judged_view and inst >= 0:
                        q = rawp[v, inst] * tf[0] + off
                        ok = np.isfinite(q).all(axis=1)
                        if ok.any():
                            c = q[ok].mean(axis=0)
                            rad = max(22.0, float(np.linalg.norm(q[ok] - c, axis=1).max()) + 14)
                            cv2.circle(img, pt(c), round(rad), JUDGE_COLOR, 2, cv2.LINE_AA)
                if lost is not None and v == judged_view:
                    cv2.circle(img, pt(lost * tf[0] + off), max(22, round(LOST_RING_PX * tf[0])), JUDGE_COLOR, 2,
                               cv2.LINE_AA)
                if recon:
                    k = self.item_fish()
                    if k is not None:
                        for q in (pa, pb):
                            p = q[k, cd.pos_node, v] * tf[0] + off
                            if np.isfinite(p).all():
                                cv2.circle(img, pt(p), 16, JUDGE_COLOR, 2, cv2.LINE_AA)
                    txt = f"{SHOW_MODES[self.show_mode]} (Tab)"
                    cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                SHOW_COLORS[self.show_mode], 2, cv2.LINE_AA)
                if r["task"] in ("geometry", "sleap", "clean", "lost"):
                    key = self.frame == int(r["frame"])
                    txt = "KEY FRAME - judge this frame" if key else "context - Home: back to the key frame"
                    col = JUDGE_COLOR if key else (150, 150, 150)
                    cv2.putText(img, txt, (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, txt, (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                if v == judged_view:
                    cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2), JUDGE_COLOR, 4)
                    cv2.putText(img, "judge this view", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, "judge this view", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                JUDGE_COLOR, 2, cv2.LINE_AA)
                lab.show_image(img)
            self.view3d.update()
            secs = self.frame / cd.fps
            self.status.setText(
                f"<b>frame {self.frame}</b> {int(secs // 60):02d}:{secs % 60:05.2f} &nbsp; "
                f"{'playing' if self.playing else 'paused'} {SPEEDS[self.speed_i]:g}x (loops the item) &nbsp; " +
                (f"raw SLEAP {'shown' if self.show_raw else 'hidden'} (R) &nbsp; " if recon else "") +
                f"{'' if t is not None else 'outside the data'} &nbsp; F1 keys")

        # -- keys --------------------------------------------------------------
        def toggle(self, attr: str):
            setattr(self, attr, not getattr(self, attr))
            if attr == "zoom_on":
                self.crops = [None, None, None]
            self.render()

        def cycle_show(self):
            self.show_mode = (self.show_mode + 1) % len(SHOW_MODES)
            self.render()

        def key_actions(self) -> dict:
            keys = {
                "Space": lambda: (setattr(self, "playing", not self.playing), self.render()),
                "Right": lambda: self.step(1), "Left": lambda: self.step(-1),
                "Shift+Right": lambda: self.step(10), "Shift+Left": lambda: self.step(-10),
                "Up": lambda: self.set_speed(self.speed_i + 1), "Down": lambda: self.set_speed(self.speed_i - 1),
                "Return": self.next_unanswered, "Enter": self.next_unanswered, "N": self.next_unanswered,
                "PgDown": lambda: self.step_item(1), "PgUp": lambda: self.step_item(-1),
                "Home": lambda: self.seek(int(self.row["frame"])),
                "Backspace": self.undo, "Ctrl+Z": self.undo, "C": self.note, "Tab": self.cycle_show, "T": self.cycle_show,
                "R": lambda: self.toggle("show_raw"), "Z": lambda: self.toggle("zoom_on"),
                "A": lambda: self.set_part("A"), "B": lambda: self.set_part("B"),
                "F1": lambda: QtWidgets.QMessageBox.information(self, "Keys", __doc__.split("Keys:")[1].rstrip()),
            }
            for code in "0123456":
                keys[code] = lambda c=code: self.answer(c)
            return keys

        def press(self, key: str):
            self._actions[key]()

        def closeEvent(self, e):
            cd.close()
            super().closeEvent(e)

    win = Main()
    install_error_log(au.dir / ERROR_LOG, win)
    if screenshot is not None:
        win.resize(1700, 1000)
        win.show()
        win.open_item(int(screenshot[1]))
        win.playing = False
        app.processEvents()
        win.render()
        app.processEvents()
        win.grab().save(str(screenshot[0]))
        win.close()
        return 0
    if smoke_test:
        win.resize(1400, 900)
        win.show()
        app.processEvents()
        if check is not None:                        # tests: drive the window first
            check(win, app)
        for task in ("id", "geometry", "sleap", "clean", "lost"):
            g = tasks[tasks["task"] == task]
            if not len(g):
                continue
            win.open_item(int(g.iloc[0]["item"]))
            for key in ("Right", "Left", "Shift+Right", "Up", "Down", "Tab", "Tab", "Tab", "R", "R", "Z", "Z",
                        "B", "A", "Home", "Space", "Space", "PgDown", "PgUp"):
                win.press(key)
                app.processEvents()
            for _ in range(3):
                win.playing = True
                win.tick()
            win.view3d.repaint()
        win.close()
        return 0
    win.resize(1700, 1000)
    win.show()
    return app.exec_()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", type=Path, help="review package folder (reference)")
    p.add_argument("--recon-dir", type=Path, required=True, help="recon comparison folder (recon_versions.npz)")
    p.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--screenshot", nargs=2, metavar=("PNG", "ITEM"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    shot = (Path(args.screenshot[0]), int(args.screenshot[1])) if args.screenshot else None
    return run_app(args.folder, args.recon_dir, smoke_test=args.smoke_test, screenshot=shot)


if __name__ == "__main__":
    raise SystemExit(main())
