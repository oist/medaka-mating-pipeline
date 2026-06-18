"""Interactive GUI to review fish-ID quality at each crossing.

Steps through the N tightest fish-fish crossings of one experiment. For each, you
see the three views (zoomed onto the fish) with the SLEAP mouth/tail predictions
colored by 3D identity, scrub a slider / play the window, and record your call:

* which ID is the FEMALE (colored radios) — your visual ground truth; if it
  changes from the previous crossing, a swap happened in between;
* whether a swap occurred in this crossing (No / Yes / Unsure);
* severity (none / minor / moderate / severe — "how bad it happens");
* free-text notes.

Judgments auto-save to a CSV (resumable). Keyboard: Left/Right = prev/next
crossing, Space = play/pause, [ / ] = step frame.

Run per experiment::

    .venv\\Scripts\\python.exe -m scripts.analysis.crossing_review_gui \\
        --csv-3d   <...>/round5_260611_3d_identified.csv \\
        --slp-dir  <...>/sleap_output/round5_260611/<exp> \\
        --model-folder calibration_results/.../models \\
        --video-top ... --video-side1 ... --video-side2 ... \\
        --out-csv  <...>/round5_260611_crossing_review.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_predictions_identity_video import (
    load_view_predictions, draw_real, VIEWS, VIEW_CAM_IDX, ID_COLORS,
)
from scripts.analysis.render_crossing_montage import rank_crossings, reproj_for_frames

VIEW_SLP = {"top": "top_cam.slp", "side1": "side1_cam.slp", "side2": "side2_cam.slp"}
SEVERITY = ["0 - none", "1 - minor", "2 - moderate", "3 - severe"]
SWAP = ["No", "Yes", "Unsure"]
COLS = ["event", "center_frame", "min_mm", "female_id", "swap", "severity", "notes"]
UNCLEAR_ID = -999  # QButtonGroup id for the "unclear" radio (not -1, which == "none")


def bgr_qcolor(bgr):
    return QtGui.QColor(bgr[2], bgr[1], bgr[0])


def _parse_female(val):
    """CSV value -> QButtonGroup id (fish id int, UNCLEAR_ID, or None)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        v = val.strip()
        if v == "":
            return None
        if v.lower() == "unclear":
            return UNCLEAR_ID
        try:
            return int(float(v))
        except ValueError:
            return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


class ReviewWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fps = 166.0
        self.win = int(args.window_s * self.fps)
        self.zoom = args.zoom_crop
        self._loading_dialog("Loading 3D + predictions (~1-2 min)...")

        self.df = pd.read_csv(args.csv_3d, usecols=lambda c: c in
                              ("frame", "final_fish_id", "fish_id", "X", "Y", "Z"))
        self.id_col = "final_fish_id" if "final_fish_id" in self.df.columns else "fish_id"
        self.events, self.ids = rank_crossings(self.df, self.id_col, args.close_mm,
                                               args.n_events, min_sep=self.win)
        if not self.events:
            raise SystemExit("no crossings below threshold")
        self.id_to_cidx = {fid: i for i, fid in enumerate(self.ids)}
        self.cal = Calibration(args.model_folder)

        win_frames = set()
        for c, _ in self.events:
            win_frames.update(range(max(0, c - self.win), c + self.win + 1))
        self.reproj = reproj_for_frames(self.df, self.cal, self.id_col, win_frames)
        self.preds = {v: load_view_predictions(args.slp_dir / VIEW_SLP[v]) for v in VIEWS}
        self.caps = {v: cv2.VideoCapture(str(getattr(args, f"video_{v}"))) for v in VIEWS}
        self.w = int(self.caps["top"].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.caps["top"].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._read_pos = {v: -10 for v in VIEWS}

        self.records = self._load_existing()
        self.cur = 0
        self._build_ui()
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance_play)
        self._loading.close()
        self.show_event(0)

    # ---- data ----
    def _load_existing(self):
        recs = {}
        if self.args.out_csv.exists():
            for r in pd.read_csv(self.args.out_csv).to_dict("records"):
                recs[int(r["event"])] = r
        return recs

    def _save(self):
        rows = []
        for i, (center, mind) in enumerate(self.events):
            r = self.records.get(i, {})
            rows.append({"event": i, "center_frame": center, "min_mm": round(mind, 2),
                         "female_id": r.get("female_id", ""), "swap": r.get("swap", ""),
                         "severity": r.get("severity", ""), "notes": r.get("notes", "")})
        pd.DataFrame(rows, columns=COLS).to_csv(self.args.out_csv, index=False)

    # ---- ui ----
    def _loading_dialog(self, msg):
        self._loading = QtWidgets.QProgressDialog(msg, None, 0, 0, self)
        self._loading.setWindowTitle("Loading"); self._loading.setMinimumDuration(0)
        self._loading.setCancelButton(None); self._loading.show()
        QtWidgets.QApplication.processEvents()

    def _build_ui(self):
        self.setWindowTitle(f"Crossing review — {self.args.csv_3d.parent.name}")
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # left: image + transport
        left = QtWidgets.QVBoxLayout(); root.addLayout(left, 3)
        self.img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.img.setMinimumSize(1100, 420); self.img.setStyleSheet("background:#111;")
        left.addWidget(self.img, 1)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        left.addWidget(self.slider)
        trow = QtWidgets.QHBoxLayout(); left.addLayout(trow)
        self.play_btn = QtWidgets.QPushButton("Play"); self.play_btn.clicked.connect(self._toggle_play)
        self.frame_lbl = QtWidgets.QLabel("")
        trow.addWidget(self.play_btn); trow.addWidget(self.frame_lbl, 1)

        # right: navigation + judgment
        right = QtWidgets.QVBoxLayout(); root.addLayout(right, 1)
        self.title = QtWidgets.QLabel(); self.title.setStyleSheet("font-weight:bold;font-size:15px;")
        right.addWidget(self.title)
        self.info = QtWidgets.QLabel(); right.addWidget(self.info)

        nav = QtWidgets.QHBoxLayout(); right.addLayout(nav)
        b_prev = QtWidgets.QPushButton("◀ Prev"); b_prev.clicked.connect(lambda: self.go(self.cur - 1))
        b_next = QtWidgets.QPushButton("Next ▶"); b_next.clicked.connect(lambda: self.go(self.cur + 1))
        nav.addWidget(b_prev); nav.addWidget(b_next)
        self.jump = QtWidgets.QComboBox()
        self.jump.addItems([f"{i+1}/{len(self.events)}  (t={c/self.fps:.0f}s, {m:.1f}mm)"
                            for i, (c, m) in enumerate(self.events)])
        self.jump.currentIndexChanged.connect(self.go)
        right.addWidget(self.jump)

        right.addWidget(self._hline())
        right.addWidget(QtWidgets.QLabel("Female ID (which color is the female):"))
        self.fem_group = QtWidgets.QButtonGroup(self)
        for fid in self.ids:
            rb = QtWidgets.QRadioButton(f"ID {fid}")
            sw = bgr_qcolor(ID_COLORS[self.id_to_cidx[fid] % len(ID_COLORS)])
            rb.setStyleSheet(f"QRadioButton{{padding:3px;}} QRadioButton::indicator{{width:16px;height:16px;}}")
            pix = QtGui.QPixmap(16, 16); pix.fill(sw)
            rb.setIcon(QtGui.QIcon(pix))
            self.fem_group.addButton(rb, fid); right.addWidget(rb)
        rb_un = QtWidgets.QRadioButton("unclear"); self.fem_group.addButton(rb_un, UNCLEAR_ID); right.addWidget(rb_un)
        self.fem_group.buttonClicked.connect(self._on_change)

        right.addWidget(self._hline())
        right.addWidget(QtWidgets.QLabel("Swap in this crossing?"))
        self.swap_group = QtWidgets.QButtonGroup(self)
        srow = QtWidgets.QHBoxLayout(); right.addLayout(srow)
        for k, s in enumerate(SWAP):
            rb = QtWidgets.QRadioButton(s); self.swap_group.addButton(rb, k); srow.addWidget(rb)
        self.swap_group.buttonClicked.connect(self._on_change)

        right.addWidget(QtWidgets.QLabel("Severity (how bad):"))
        self.sev = QtWidgets.QComboBox(); self.sev.addItems([""] + SEVERITY)
        self.sev.currentIndexChanged.connect(self._on_change); right.addWidget(self.sev)

        right.addWidget(QtWidgets.QLabel("Notes:"))
        self.notes = QtWidgets.QPlainTextEdit(); self.notes.setMaximumHeight(120)
        self.notes.textChanged.connect(self._on_change); right.addWidget(self.notes)

        self.progress = QtWidgets.QLabel(); right.addWidget(self.progress)
        b_save = QtWidgets.QPushButton("Save CSV now")
        b_save.clicked.connect(lambda: (self._collect(), self._save()))
        right.addWidget(b_save); right.addStretch(1)

        for key, fn in [("Right", lambda: self.go(self.cur + 1)),
                        ("Left", lambda: self.go(self.cur - 1)),
                        ("Space", self._toggle_play),
                        ("]", lambda: self.slider.setValue(self.slider.value() + 1)),
                        ("[", lambda: self.slider.setValue(self.slider.value() - 1))]:
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

    def _hline(self):
        ln = QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.HLine); return ln

    # ---- navigation / state ----
    def go(self, i):
        if 0 <= i < len(self.events) and i != self.cur:
            self._collect()
            self._save()
            self.show_event(i)

    def show_event(self, i):
        self._stop_play()
        self.cur = i
        center, mind = self.events[i]
        self.title.setText(f"Crossing {i+1} / {len(self.events)}")
        self.info.setText(f"t = {center/self.fps:.1f} s   min-dist = {mind:.1f} mm   "
                          f"center frame {center}")
        lo, hi = max(0, center - self.win), center + self.win
        self.slider.blockSignals(True)
        self.slider.setRange(lo, hi); self.slider.setValue(center)
        self.slider.blockSignals(False)
        self.jump.blockSignals(True); self.jump.setCurrentIndex(i); self.jump.blockSignals(False)
        self._load_widgets(self.records.get(i, {}))
        self._render(center)
        self._update_progress()

    def _load_widgets(self, r):
        # female radios
        want = _parse_female(r.get("female_id", ""))
        fbtn = self.fem_group.button(want) if want is not None else None
        for b in self.fem_group.buttons():
            b.setChecked(b is fbtn)
        # swap radios
        sval = r.get("swap", "")
        sidx = SWAP.index(sval) if isinstance(sval, str) and sval in SWAP else None
        sbtn = self.swap_group.button(sidx) if sidx is not None else None
        for b in self.swap_group.buttons():
            b.setChecked(b is sbtn)
        self.sev.blockSignals(True)
        sv = r.get("severity", "")
        self.sev.setCurrentIndex(SEVERITY.index(sv) + 1 if sv in SEVERITY else 0)
        self.sev.blockSignals(False)
        self.notes.blockSignals(True)
        self.notes.setPlainText("" if pd.isna(r.get("notes", "")) else str(r.get("notes", "")))
        self.notes.blockSignals(False)

    def _collect(self):
        """Persist the current widgets into self.records (in memory only)."""
        fem_btn = self.fem_group.checkedButton()
        if fem_btn is None:
            female = ""
        else:
            cid = self.fem_group.id(fem_btn)
            female = "unclear" if cid == UNCLEAR_ID else cid
        swap_btn = self.swap_group.checkedId()
        center, mind = self.events[self.cur]
        self.records[self.cur] = {
            "event": self.cur, "center_frame": center, "min_mm": round(mind, 2),
            "female_id": female,
            "swap": SWAP[swap_btn] if swap_btn >= 0 else "",
            "severity": SEVERITY[self.sev.currentIndex() - 1] if self.sev.currentIndex() > 0 else "",
            "notes": self.notes.toPlainText().strip(),
        }

    def _on_change(self, *a):
        self._collect(); self._update_progress()

    def _update_progress(self):
        done = sum(1 for r in self.records.values()
                   if r.get("swap") or r.get("female_id") not in ("", None))
        self.progress.setText(f"reviewed: {done} / {len(self.events)}   (auto-saved)")

    # ---- rendering ----
    def _read(self, v, f):
        cap = self.caps[v]
        if f != self._read_pos[v] + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read(); self._read_pos[v] = f
        if not ok or fr is None:
            fr = np.zeros((self.h, self.w, 3), np.uint8)
        return fr

    def _render(self, f):
        rp = self.reproj.get(f, {v: [] for v in VIEWS})
        panels = []
        for v in VIEWS:
            fr = self._read(v, f)
            vrp = rp.get(v, [])
            fr = draw_real(fr, self.preds[v].get(f, []), vrp, self.id_to_cidx, self.args.assign_max_dist)
            if self.zoom > 0:
                z = self.zoom
                if vrp:
                    cx = int(np.mean([p[1] for p in vrp])); cy = int(np.mean([p[2] for p in vrp]))
                else:
                    cx, cy = self.w // 2, self.h // 2
                x0 = min(max(cx - z // 2, 0), max(0, self.w - z))
                y0 = min(max(cy - z // 2, 0), max(0, self.h - z))
                fr = fr[y0:y0 + min(z, self.h), x0:x0 + min(z, self.w)]
            cv2.putText(fr, v, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(cv2.resize(fr, (380, 380)))
        comp = cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)
        h, w, _ = comp.shape
        qimg = QtGui.QImage(comp.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.img.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.img.width(), self.img.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.frame_lbl.setText(f"frame {f}   t={f/self.fps:.2f}s")

    def _on_slider(self, val):
        self._render(val)

    def _toggle_play(self):
        if self._play_timer.isActive():
            self._stop_play()
        else:
            self.play_btn.setText("Pause"); self._play_timer.start(40)

    def _stop_play(self):
        self._play_timer.stop(); self.play_btn.setText("Play")

    def _advance_play(self):
        nv = self.slider.value() + 1
        if nv > self.slider.maximum():
            nv = self.slider.minimum()
        self.slider.setValue(nv)

    def closeEvent(self, e):
        self._collect(); self._save(); super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--video-top", type=Path, required=True)
    ap.add_argument("--video-side1", type=Path, required=True)
    ap.add_argument("--video-side2", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--n-events", type=int, default=40)
    ap.add_argument("--window-s", type=float, default=1.5)
    ap.add_argument("--close-mm", type=float, default=18.0)
    ap.add_argument("--zoom-crop", type=int, default=500)
    ap.add_argument("--assign-max-dist", type=float, default=60.0)
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = ReviewWindow(args)
    w.resize(1500, 720); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
