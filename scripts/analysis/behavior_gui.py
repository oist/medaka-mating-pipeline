"""Browse one trial or one group: timeline, features, aligned curves, competition.

The tables are the ones :mod:`scripts.analysis.behavior_report` writes, and the export
buttons write the same PNG and PDF files. Click the timeline to see that moment on the
top-camera playback proxy.

::

    .venv\\Scripts\\python.exe -m scripts.analysis.behavior_gui
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from scripts.analysis.behavior_dataset import Trial, ensure_trial, group_status, pool, resolve_roots
from scripts.analysis.behavior_report import (aligned_figure, competition_figure, features_figure, save_figure,
                                              timeline_figure)
from scripts.review.behaviors import default_search
from scripts.review.review_gui import ProxyVideo


class Canvas(FigureCanvasQTAgg):
    def __init__(self, fig: Figure | None = None):
        super().__init__(fig or Figure(layout="constrained"))


def _image(frame: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class Main(QtWidgets.QMainWindow):
    def __init__(self, review: Path, out: Path, search: list[Path]):
        super().__init__()
        self.review, self.out, self.search = review, out, search
        self.days: list[Trial] = []
        self.current: Trial | None = None
        self.video: ProxyVideo | None = None
        self.video_path: Path | None = None
        self.setWindowTitle("Behaviour features")
        self.resize(1280, 860)

        self.choice = QtWidgets.QComboBox()
        self.choice.activated.connect(lambda _i: self.load_current())
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(lambda: self.load_current(rebuild=True))
        export_btn = QtWidgets.QPushButton("Export all")
        export_btn.clicked.connect(self.export_all)
        this_btn = QtWidgets.QPushButton("Export this view")
        this_btn.clicked.connect(self.export_current)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Trial or group"))
        bar.addWidget(self.choice, 1)
        for button in (reload_btn, this_btn, export_btn):
            bar.addWidget(button)

        self.tabs = QtWidgets.QTabWidget()
        self.canvases = {name: Canvas() for name in ("Timeline", "Features", "Aligned", "Competition")}
        self.still = QtWidgets.QLabel("click the timeline")
        self.still.setAlignment(QtCore.Qt.AlignCenter)
        self.still.setMinimumHeight(220)
        timeline = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(timeline)
        self.day = QtWidgets.QComboBox()
        self.day.activated.connect(lambda _i: self.draw_timeline())
        layout.addWidget(self.day)
        layout.addWidget(self.canvases["Timeline"], 1)
        layout.addWidget(self.still)
        self.tabs.addTab(timeline, "Timeline")
        for name in ("Features", "Aligned", "Competition"):
            self.tabs.addTab(self.canvases[name], name)
        self.canvases["Timeline"].mpl_connect("button_press_event", self.on_timeline_click)

        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.addLayout(bar)
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage(str(out))
        self.fill_choices()

    def fill_choices(self) -> None:
        self.choice.clear()
        ready = []
        for folder in sorted(p for p in self.review.iterdir() if p.is_dir()):
            if len(list(folder.glob("*/id_pass/human_events.csv"))) == 1:
                ready.append(folder.name)
                self.choice.addItem(folder.name, folder.name)
        # 20260112_P_1_1_FCMC -> group P_1, the token between the date and the trial number.
        seen: list[str] = []
        for name in ready:
            parts = name.split("_")
            if len(parts) >= 4 and parts[0].isdigit():
                group = f"{parts[1]}_{parts[2]}"
                if group not in seen:
                    seen.append(group)
                    self.choice.addItem(f"{group} (pooled)", f"group:{group}")
        self.statusBar().showMessage("choose a trial or a pooled group")

    def load_current(self, rebuild: bool = False) -> None:
        key = self.choice.currentData()
        if not key:
            return
        self.statusBar().showMessage(f"loading {key}")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            if str(key).startswith("group:"):
                group = str(key).split(":", 1)[1]
                names, skipped = group_status(self.review, group)
                self.days = [ensure_trial(self.review, self.out, name, self.search, rebuild=rebuild) for name in names]
                self.current = pool(self.days, group) if self.days else None
                note = "; ".join(skipped)
            else:
                self.days = [ensure_trial(self.review, self.out, str(key), self.search, rebuild=rebuild)]
                self.current = self.days[0]
                note = ""
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if self.current is None:
            self.statusBar().showMessage("nothing reviewed in this group")
            return
        self.day.blockSignals(True)
        self.day.clear()
        for trial in self.days:
            self.day.addItem(trial.experiment)
        self.day.blockSignals(False)
        self.draw_all()
        usable = self.current.provenance.get("usable_pct")
        text = f"{self.current.experiment}: {len(self.current.bouts)} bouts"
        if usable is not None:
            text += f", usable {usable}%"
        if self.current.review_open or self.current.provenance.get("review_open"):
            text += " — review still open, these numbers are partial"
        if note:
            text += " | " + note
        self.statusBar().showMessage(text)

    def draw_all(self) -> None:
        self.draw_timeline()
        self._swap("Features", features_figure(self.current))
        self._swap("Aligned", aligned_figure(self.current))
        self._swap("Competition", competition_figure(self.current))

    def draw_timeline(self) -> None:
        if not self.days:
            return
        self._swap("Timeline", timeline_figure(self.days[self.day.currentIndex()]))

    def _swap(self, name: str, fig: Figure) -> None:
        old = self.canvases[name]
        canvas = Canvas(fig)
        if name == "Timeline":
            canvas.mpl_connect("button_press_event", self.on_timeline_click)
        index = self.tabs.indexOf(old) if name != "Timeline" else None
        self.canvases[name] = canvas
        if name == "Timeline":
            layout = old.parentWidget().layout()
            layout.replaceWidget(old, canvas)
        else:
            self.tabs.removeTab(self.tabs.indexOf(old))
            self.tabs.insertTab(index, canvas, name)
        old.setParent(None)

    def selected_day(self) -> Trial | None:
        if not self.days:
            return None
        return self.days[self.day.currentIndex()]

    def on_timeline_click(self, event) -> None:
        trial = self.selected_day()
        if trial is None or event.xdata is None:
            return
        frame = int(round(event.xdata * 60 * trial.fps))
        path = Path(trial.provenance["proxy_dir"]) / trial.provenance["play_file"]
        step = int(trial.provenance["play_step"])
        if self.video is None or self.video_path != path:
            self.video = ProxyVideo(path, fps=trial.fps / step)
            self.video_path = path
        image = self.video.get(max(0, frame // step))
        if image is None:
            self.still.setText(f"frame {frame}: proxy frame not read")
            return
        shown = _image(image).scaledToHeight(240, QtCore.Qt.SmoothTransformation)
        self.still.setPixmap(QtGui.QPixmap.fromImage(shown))
        self.statusBar().showMessage(f"{trial.experiment} frame {frame} ({event.xdata:.2f} min)")

    def export_current(self) -> None:
        name = self.tabs.tabText(self.tabs.currentIndex()).lower()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export this view")
        if not folder or self.current is None:
            return
        fig = {"timeline": timeline_figure, "features": features_figure, "aligned": aligned_figure,
               "competition": competition_figure}[name]
        target = self.selected_day() if name == "timeline" else self.current
        save_figure(fig(target), Path(folder) / f"{target.experiment}_{name}.png")
        self.statusBar().showMessage(f"wrote {name} png and pdf to {folder}")

    def export_all(self) -> None:
        from scripts.analysis.behavior_report import write_figures

        if self.current is None:
            return
        for trial in self.days:
            write_figures(trial, self.out / trial.experiment, timeline=True)
        if self.current.provenance.get("days"):
            write_figures(self.current, self.out / self.current.experiment, timeline=False)
        self.statusBar().showMessage(f"wrote png and pdf to {self.out}")


def main() -> None:
    _config, review, out = resolve_roots()
    app = QtWidgets.QApplication(sys.argv)
    window = Main(review, out, default_search(None))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
