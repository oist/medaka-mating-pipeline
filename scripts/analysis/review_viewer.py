"""Unified multi-method review viewer for medaka identity QC / ground-truth.

A manifest-driven PyQt5 viewer to confirm fish identity by eye across the
different methods we have tried (centroid bind, dedup, mouth-anchored,
idtracker-bound, UDMT-resolved, ...). It reuses the reprojection / multi-view /
event machinery from ``render_crossing_montage`` and the review-form pattern
from ``crossing_review_gui`` but generalizes them:

* load MANY method result CSVs for one experiment and overlay each method's
  reprojected identities on the three camera views;
* compare via a toggle overlay (checkbox which methods to draw) OR a side-by-side
  split (one column per method for the selected view);
* step through an EVENT list from any source: a UDMT crossings CSV, a UDMT
  resolver-decisions CSV, or ``rank_crossings`` on a reference method;
* per-event colors are aligned across methods (matched to the reference method's
  3D positions at the event center) so the SAME physical fish keeps the SAME
  color when methods agree, and a disagreement shows as a color mismatch;
* record human ground truth (female id, swap yes/no/unsure, which methods are
  correct, whether a resolver suspect is right, severity, notes) to a resumable
  CSV.

Run::

    python -m scripts.analysis.review_viewer --manifest <trial.toml>

See ``configs/review/example.toml`` for the manifest schema. Everything in the
manifest can be overridden on the CLI; nothing here writes to the production 3D
path -- the only output is the ground-truth CSV.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration
from scripts.analysis.render_crossing_montage import (
    ID_COLORS, VIEWS, VIEW_CAM_IDX, rank_crossings, reproj_for_frames,
)
from scripts.analysis.render_predictions_identity_video import load_view_predictions

GRAY = (160, 160, 160)
PRED_COLOR = (210, 210, 210)   # SLEAP skeleton overlay (neutral)
MASK_COLOR = (255, 80, 255)    # segmentation contour overlay (magenta)
# Marker SHAPE encodes the method; marker COLOR encodes the (aligned) fish id.
MARKER_SHAPES = ["circle", "square", "triangle", "diamond", "plus", "x"]
# Canonical fish label per aligned color group — the same physical fish gets the
# same letter across methods (within an event), so GT is annotated method-agnostically.
CANONICAL_LABELS = "ABCDEF"
VIEW_SLP = {"top": "top_cam.slp", "side1": "side1_cam.slp", "side2": "side2_cam.slp"}
SWAP = ["No", "Yes", "Unsure"]
SEVERITY = ["none", "minor", "moderate", "severe"]
YESNO = ["Yes", "No", "Unsure"]
UNCLEAR_ID = -999
GT_COLS = [
    "event", "center_frame", "min_mm", "source", "track_a", "track_b",
    "suspect_fish", "female_id", "female_raw_by_method", "swap", "correct_methods",
    "suspect_correct", "severity", "notes",
]


@dataclass
class MethodSpec:
    label: str
    csv_3d: Path
    id_col: str = "final_fish_id"
    slp_dir: Path | None = None          # optional: <view>.slp for SLEAP-prediction overlay
    masks_dir: Path | None = None        # optional: segmentation masks for contour overlay
    mask_pattern: str = "{frame:07d}.png"  # path inside masks_dir; may use {view} and {frame}


@dataclass
class Manifest:
    experiment: str
    videos: dict[str, Path]                 # {"top":..,"side1":..,"side2":..}
    model_folder: Path
    methods: list[MethodSpec]
    gt_out: Path
    event_source: str = "rank_crossings"    # rank_crossings | crossings | resolver
    event_path: Path | None = None
    reference_method: str = ""
    fps: float = 166.0
    window_s: float = 1.5
    close_mm: float = 18.0
    n_events: int = 40
    zoom_crop: int = 0
    max_events: int | None = None
    review_only: bool = False               # for crossings/resolver sources


# --------------------------------------------------------------------------- #
# Manifest / events (pure-Python, unit-testable without Qt or video).
# --------------------------------------------------------------------------- #
def load_manifest(path: Path, overrides: dict | None = None) -> Manifest:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    videos = {k: Path(v) for k, v in raw["videos"].items()}
    top_slp = raw.get("slp_dir")
    methods = []
    for m in raw["methods"]:
        slp = m.get("slp_dir", top_slp)
        methods.append(MethodSpec(
            label=m["label"], csv_3d=Path(m["csv_3d"]),
            id_col=m.get("id_col", "final_fish_id"),
            slp_dir=Path(slp) if slp else None,
            masks_dir=Path(m["masks_dir"]) if m.get("masks_dir") else None,
            mask_pattern=m.get("mask_pattern", "{frame:07d}.png"),
        ))
    ev = raw.get("events", {})
    ref = raw.get("reference_method") or (methods[0].label if methods else "")
    gt_out_val = raw.get("gt_out") or ev.get("gt_out")
    if not gt_out_val:
        raise SystemExit("manifest must set `gt_out` (top-level, before any [table])")
    return Manifest(
        experiment=raw.get("experiment", path.stem),
        videos=videos,
        model_folder=Path(raw["model_folder"]),
        methods=methods,
        gt_out=Path(gt_out_val),
        event_source=ev.get("source", "rank_crossings"),
        event_path=Path(ev["path"]) if ev.get("path") else None,
        reference_method=ev.get("reference_method", ref),
        fps=float(raw.get("fps", 166.0)),
        window_s=float(raw.get("window_s", 1.5)),
        close_mm=float(raw.get("close_mm", 18.0)),
        n_events=int(raw.get("n_events", 40)),
        zoom_crop=int(raw.get("zoom_crop", 0)),
        max_events=raw.get("max_events"),
        review_only=bool(ev.get("review_only", False)),
    )


def _clean(v) -> str:
    """Stringify an id/label, mapping NaN/None to '' (so blank cells stay blank)."""
    return "" if pd.isna(v) else str(v)


def build_events(cfg: Manifest, method_dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Return a list of event dicts: {center, min_mm, track_a, track_b,
    suspect, source}. Window bounds are derived from cfg.window_s at view time."""
    src = cfg.event_source
    if src == "rank_crossings":
        ref = method_dfs[cfg.reference_method]
        id_col = next(m.id_col for m in cfg.methods if m.label == cfg.reference_method)
        win = int(cfg.window_s * cfg.fps)
        events, _ = rank_crossings(ref, id_col, cfg.close_mm, cfg.n_events, min_sep=win)
        out = [{"center": int(c), "min_mm": float(m), "track_a": "", "track_b": "",
                "suspect": "", "source": src} for c, m in events]
    elif src in ("crossings", "resolver"):
        df = pd.read_csv(cfg.event_path)
        if cfg.review_only and "review_candidate" in df.columns:
            df = df[df["review_candidate"].fillna(False).astype(bool)]
        center_col = "min_frame" if "min_frame" in df.columns else "center_frame"
        out = []
        for r in df.to_dict("records"):   # robust to any header; NaN-safe
            mm = r.get("min_distance", r.get("min_mm", float("nan")))
            out.append({
                "center": int(r[center_col]),
                "min_mm": float(mm) if pd.notna(mm) else float("nan"),
                "track_a": _clean(r.get("track_id_a", "")),
                "track_b": _clean(r.get("track_id_b", "")),
                "suspect": _clean(r.get("suspect_fish", "")),
                "source": src,
            })
    else:
        raise SystemExit(f"unknown event source: {src}")
    out.sort(key=lambda e: e["center"])
    if cfg.max_events:
        out = out[: int(cfg.max_events)]
    return out


def align_colors(method_dfs: dict[str, pd.DataFrame], id_cols: dict[str, str],
                 reference: str, center: int) -> dict[str, dict]:
    """Per-event color map: match each method's fish to the reference method's
    fish by nearest 3D position at the center frame, so agreeing methods share a
    color. Returns {method_label: {fish_id: color_index}}."""
    ref_df = method_dfs[reference]
    ref_rows = ref_df[ref_df["frame"] == center].dropna(subset=["X", "Y", "Z"])
    ref_ids = sorted(ref_rows[id_cols[reference]].dropna().unique(),
                     key=lambda v: str(v))
    ref_color = {fid: i for i, fid in enumerate(ref_ids)}
    ref_pos = {r[id_cols[reference]]: np.array([r["X"], r["Y"], r["Z"]], float)
               for _, r in ref_rows.iterrows()}
    out: dict[str, dict] = {}
    for label, df in method_dfs.items():
        idc = id_cols[label]
        if label == reference:
            out[label] = dict(ref_color)
            continue
        rows = df[df["frame"] == center].dropna(subset=["X", "Y", "Z"])
        cands = [(r[idc], np.array([r["X"], r["Y"], r["Z"]], float))
                 for _, r in rows.iterrows()]
        # greedy global-nearest ONE-TO-ONE assignment to reference fish, so two
        # method fish can never collide on the same reference color.
        pairs = sorted(
            (float(np.linalg.norm(rp - p)), str(fid), fid, rid)
            for fid, p in cands for rid, rp in ref_pos.items()
        )
        cmap: dict = {}
        used_ref: set = set()
        for _dist, _fkey, fid, rid in pairs:
            if fid in cmap or rid in used_ref:
                continue
            cmap[fid] = ref_color[rid]
            used_ref.add(rid)
        # unmatched method fish (more fish than reference) -> fresh colors
        nxt = len(ref_color)
        for fid in sorted(df[idc].dropna().unique(), key=lambda v: str(v)):
            if fid not in cmap:
                cmap[fid] = nxt
                nxt += 1
        out[label] = cmap
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers (cv2 only; unit-testable without Qt).
# --------------------------------------------------------------------------- #
def draw_marker(frame, cx, cy, color, shape, r=10, t=2):
    """Draw one marker of the given SHAPE (encodes method) and COLOR (fish id)."""
    cx, cy = int(round(cx)), int(round(cy))
    if shape == "square":
        cv2.rectangle(frame, (cx - r, cy - r), (cx + r, cy + r), color, t, cv2.LINE_AA)
    elif shape == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.polylines(frame, [pts], True, color, t, cv2.LINE_AA)
    elif shape == "diamond":
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
        cv2.polylines(frame, [pts], True, color, t, cv2.LINE_AA)
    elif shape == "plus":
        cv2.line(frame, (cx - r, cy), (cx + r, cy), color, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), color, t, cv2.LINE_AA)
    elif shape == "x":
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), color, t, cv2.LINE_AA)
        cv2.line(frame, (cx - r, cy + r), (cx + r, cy - r), color, t, cv2.LINE_AA)
    else:  # circle (default)
        cv2.circle(frame, (cx, cy), r, color, t, cv2.LINE_AA)
    return frame


def draw_predictions(frame, instances, color=PRED_COLOR):
    """Overlay raw SLEAP instances (mouth dot, tail ring, connecting line)."""
    for pts in instances:
        pts = np.asarray(pts, dtype=float)
        if pts.size < 2:
            continue
        mouth = pts[0] if (len(pts) > 0 and np.all(np.isfinite(pts[0]))) else None
        tail = pts[1] if (len(pts) > 1 and np.all(np.isfinite(pts[1]))) else None
        if mouth is not None and tail is not None:
            cv2.line(frame, tuple(mouth.astype(int)), tuple(tail.astype(int)), color, 1, cv2.LINE_AA)
        if mouth is not None:
            cv2.circle(frame, tuple(mouth.astype(int)), 4, color, -1, cv2.LINE_AA)
        if tail is not None:
            cv2.circle(frame, tuple(tail.astype(int)), 4, color, 1, cv2.LINE_AA)
    return frame


def draw_mask_contours(frame, mask, color=MASK_COLOR):
    """Overlay the outline of a binary segmentation mask."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)
    return frame


def mask_path_for(masks_dir: Path, pattern: str, view: str, frame: int) -> Path:
    """Resolve a per-(view,frame) mask path from a format pattern, e.g.
    '{frame:07d}.png' or '{view}_cam/{frame:07d}.png'."""
    return Path(masks_dir) / pattern.format(view=view, frame=int(frame))


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def _build_gui_classes():
    """Import PyQt5 lazily so the data layer is importable without a display."""
    from PyQt5 import QtCore, QtGui, QtWidgets

    class TrialData:
        def __init__(self, cfg: Manifest):
            self.cfg = cfg
            self.id_cols = {m.label: m.id_col for m in cfg.methods}
            self._spec = {m.label: m for m in cfg.methods}
            self._preds_cache: dict = {}
            self.method_dfs = {
                m.label: pd.read_csv(m.csv_3d, usecols=lambda c: c in
                                     ("frame", m.id_col, "X", "Y", "Z"))
                for m in cfg.methods
            }
            self.events = build_events(cfg, self.method_dfs)
            if not self.events:
                raise SystemExit("no events to review")
            self.cal = Calibration(cfg.model_folder)
            self.win = int(cfg.window_s * cfg.fps)
            # precompute reproj per method over the union of event windows
            frames: set[int] = set()
            for e in self.events:
                frames.update(range(max(0, e["center"] - self.win), e["center"] + self.win + 1))
            self.reproj = {
                m.label: reproj_for_frames(self.method_dfs[m.label], self.cal,
                                           m.id_col, frames, VIEWS)
                for m in cfg.methods
            }
            self.caps = {v: cv2.VideoCapture(str(cfg.videos[v])) for v in VIEWS}
            for v, c in self.caps.items():
                if not c.isOpened():
                    raise SystemExit(f"cannot open {cfg.videos[v]}")
            self.w = int(self.caps["top"].get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.caps["top"].get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._read_pos = {v: -10 for v in VIEWS}

        def read(self, v: str, f: int) -> np.ndarray:
            cap = self.caps[v]
            if f != self._read_pos[v] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            self._read_pos[v] = f
            if not ok or fr is None:
                fr = np.zeros((self.h, self.w, 3), np.uint8)
            return fr

        def preds(self, label: str, view: str) -> dict:
            """SLEAP predictions {frame: [instances]} for a method/view (lazy, cached)."""
            spec = self._spec[label]
            if spec.slp_dir is None:
                return {}
            key = (label, view)
            if key not in self._preds_cache:
                p = Path(spec.slp_dir) / VIEW_SLP[view]
                try:
                    self._preds_cache[key] = load_view_predictions(p) if p.exists() else {}
                except Exception:
                    self._preds_cache[key] = {}
            return self._preds_cache[key]

        def mask(self, label: str, view: str, frame: int):
            """Grayscale segmentation mask for a method/view/frame, or None."""
            spec = self._spec[label]
            if spec.masks_dir is None:
                return None
            path = mask_path_for(spec.masks_dir, spec.mask_pattern, view, frame)
            if not path.exists():
                return None
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    class ReviewWindow(QtWidgets.QMainWindow):
        def __init__(self, data: "TrialData"):
            super().__init__()
            self.d = data
            self.cfg = data.cfg
            self.labels = [m.label for m in self.cfg.methods]
            self.cur = 0
            self.color_map: dict[str, dict] = {}
            self.records: dict[int, dict] = {}
            self.split_view = "top"
            self.setWindowTitle(f"Review — {self.cfg.experiment}")
            self.resize(1500, 850)
            self._play = QtCore.QTimer(self); self._play.timeout.connect(self._advance)
            self._load_existing()
            self._build()
            self.show_event(0)

        # ---- persistence ----
        def _load_existing(self):
            if self.cfg.gt_out.exists():
                for r in pd.read_csv(self.cfg.gt_out).to_dict("records"):
                    self.records[int(r["event"])] = r

        def _save(self):
            rows = [self.records[i] for i in sorted(self.records)]
            self.cfg.gt_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=GT_COLS).to_csv(self.cfg.gt_out, index=False)

        # ---- layout ----
        def _build(self):
            central = QtWidgets.QWidget(); self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            left = QtWidgets.QVBoxLayout(); root.addLayout(left, 4)
            self.img = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            self.img.setMinimumSize(900, 600); left.addWidget(self.img, 1)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.valueChanged.connect(self._render); left.addWidget(self.slider)
            trow = QtWidgets.QHBoxLayout(); left.addLayout(trow)
            self.play_btn = QtWidgets.QPushButton("Play"); self.play_btn.clicked.connect(self._toggle_play)
            trow.addWidget(self.play_btn)
            self.frame_lbl = QtWidgets.QLabel(""); trow.addWidget(self.frame_lbl); trow.addStretch(1)

            right = QtWidgets.QVBoxLayout(); root.addLayout(right, 1)
            self.title = QtWidgets.QLabel(); self.title.setStyleSheet("font-weight:bold;font-size:15px;")
            right.addWidget(self.title)
            self.info = QtWidgets.QLabel(); self.info.setWordWrap(True); right.addWidget(self.info)

            nav = QtWidgets.QHBoxLayout(); right.addLayout(nav)
            bp = QtWidgets.QPushButton("◀ Prev"); bp.clicked.connect(lambda: self.go(self.cur - 1))
            bn = QtWidgets.QPushButton("Next ▶"); bn.clicked.connect(lambda: self.go(self.cur + 1))
            nav.addWidget(bp); nav.addWidget(bn)
            self.jump = QtWidgets.QComboBox()
            self.jump.addItems([f"{i+1}: f{e['center']}" for i, e in enumerate(self.d.events)])
            self.jump.currentIndexChanged.connect(self.go); right.addWidget(self.jump)

            right.addWidget(self._hline())
            # method overlay toggles
            right.addWidget(QtWidgets.QLabel("Methods to overlay (color = fish):"))
            self.method_boxes = {}
            for lab in self.labels:
                cb = QtWidgets.QCheckBox(lab); cb.setChecked(True)
                cb.stateChanged.connect(lambda *_: self._render(self.slider.value()))
                self.method_boxes[lab] = cb; right.addWidget(cb)
            srow = QtWidgets.QHBoxLayout(); right.addLayout(srow)
            self.split_cb = QtWidgets.QCheckBox("Split (columns per method)")
            self.split_cb.stateChanged.connect(lambda *_: self._render(self.slider.value()))
            srow.addWidget(self.split_cb)
            self.view_combo = QtWidgets.QComboBox(); self.view_combo.addItems(list(VIEWS))
            self.view_combo.currentTextChanged.connect(self._on_view); srow.addWidget(self.view_combo)
            orow = QtWidgets.QHBoxLayout(); right.addLayout(orow)
            self.sleap_cb = QtWidgets.QCheckBox("SLEAP preds")
            self.sleap_cb.stateChanged.connect(lambda *_: self._render(self.slider.value()))
            self.mask_cb = QtWidgets.QCheckBox("Seg masks")
            self.mask_cb.stateChanged.connect(lambda *_: self._render(self.slider.value()))
            orow.addWidget(self.sleap_cb); orow.addWidget(self.mask_cb); orow.addStretch(1)

            right.addWidget(self._hline())
            # GT form
            right.addWidget(QtWidgets.QLabel("Female (canonical fish — color matches markers):"))
            self.fem_group = QtWidgets.QButtonGroup(self)
            self.fem_group.buttonClicked.connect(self._on_change)  # connect once
            self.fem_box = QtWidgets.QHBoxLayout(); right.addLayout(self.fem_box)
            self._fem_buttons = []  # rebuilt per event

            right.addWidget(QtWidgets.QLabel("Swap in this crossing?"))
            self.swap_group = QtWidgets.QButtonGroup(self)
            wr = QtWidgets.QHBoxLayout(); right.addLayout(wr)
            for k, s in enumerate(SWAP):
                rb = QtWidgets.QRadioButton(s); self.swap_group.addButton(rb, k); wr.addWidget(rb)
            self.swap_group.buttonClicked.connect(self._on_change)

            right.addWidget(QtWidgets.QLabel("Which methods are CORRECT here:"))
            self.correct_boxes = {}
            for lab in self.labels:
                cb = QtWidgets.QCheckBox(lab); cb.stateChanged.connect(self._on_change)
                self.correct_boxes[lab] = cb; right.addWidget(cb)

            self.suspect_lbl = QtWidgets.QLabel("Resolver suspect correct?")
            right.addWidget(self.suspect_lbl)
            self.suspect_group = QtWidgets.QButtonGroup(self)
            sr = QtWidgets.QHBoxLayout(); right.addLayout(sr)
            for k, s in enumerate(YESNO):
                rb = QtWidgets.QRadioButton(s); self.suspect_group.addButton(rb, k); sr.addWidget(rb)
            self.suspect_group.buttonClicked.connect(self._on_change)

            right.addWidget(QtWidgets.QLabel("Severity:"))
            self.sev = QtWidgets.QComboBox(); self.sev.addItems([""] + SEVERITY)
            self.sev.currentIndexChanged.connect(self._on_change); right.addWidget(self.sev)
            right.addWidget(QtWidgets.QLabel("Notes:"))
            self.notes = QtWidgets.QPlainTextEdit(); self.notes.setMaximumHeight(90); right.addWidget(self.notes)
            self.progress = QtWidgets.QLabel(); right.addWidget(self.progress)
            bs = QtWidgets.QPushButton("Save CSV now"); bs.clicked.connect(lambda: (self._collect(), self._save()))
            right.addWidget(bs); right.addStretch(1)

            for key, fn in [("Right", lambda: self.go(self.cur + 1)),
                            ("Left", lambda: self.go(self.cur - 1)),
                            ("Space", self._toggle_play),
                            ("]", lambda: self.slider.setValue(self.slider.value() + 1)),
                            ("[", lambda: self.slider.setValue(self.slider.value() - 1))]:
                QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        def _hline(self):
            ln = QtWidgets.QFrame(); ln.setFrameShape(QtWidgets.QFrame.HLine); return ln

        def _on_view(self, v):
            self.split_view = v; self._render(self.slider.value())

        # ---- navigation ----
        def go(self, i):
            if 0 <= i < len(self.d.events) and i != self.cur:
                self._collect(); self._save(); self.show_event(i)

        def show_event(self, i):
            self._stop_play(); self.cur = i
            e = self.d.events[i]
            center = e["center"]
            self.color_map = align_colors(self.d.method_dfs, self.d.id_cols,
                                          self.cfg.reference_method, center)
            self.title.setText(f"Event {i+1} / {len(self.d.events)}  ({e['source']})")
            extra = f"  pair {e['track_a']}/{e['track_b']}" if e["track_a"] else ""
            susp = f"  suspect={e['suspect']}" if e["suspect"] else ""
            mm_s = f"{e['min_mm']:.1f}mm" if np.isfinite(e["min_mm"]) else "n/a"
            self.info.setText(f"center f{center}  t={center/self.cfg.fps:.1f}s  "
                              f"min-dist={mm_s}{extra}{susp}")
            lo, hi = max(0, center - self.d.win), center + self.d.win
            # open at the window START so you can play INTO the crossing (not at its peak)
            self.slider.blockSignals(True); self.slider.setRange(lo, hi); self.slider.setValue(lo)
            self.slider.blockSignals(False)
            self.jump.blockSignals(True); self.jump.setCurrentIndex(i); self.jump.blockSignals(False)
            self._rebuild_female_buttons()
            self.suspect_lbl.setVisible(bool(e["suspect"]))
            for b in self.suspect_group.buttons():
                b.setVisible(bool(e["suspect"]))
            self._load_widgets(self.records.get(i, {}))
            self._render(lo); self._update_progress()

        def _rebuild_female_buttons(self):
            for b in self._fem_buttons:
                self.fem_group.removeButton(b); b.deleteLater()
            self._fem_buttons = []
            ref_cmap = self.color_map.get(self.cfg.reference_method, {})
            for ci in sorted(set(ref_cmap.values())):
                letter = CANONICAL_LABELS[ci % len(CANONICAL_LABELS)]
                rb = QtWidgets.QRadioButton(f"Fish {letter}")
                b, g, r = ID_COLORS[ci % len(ID_COLORS)]   # BGR -> RGB for Qt
                rb.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight:bold;")
                self.fem_group.addButton(rb, 1000 + ci)
                rb.setProperty("fid", letter); self.fem_box.addWidget(rb); self._fem_buttons.append(rb)
            rb = QtWidgets.QRadioButton("unclear"); self.fem_group.addButton(rb, UNCLEAR_ID)
            rb.setProperty("fid", "unclear"); self.fem_box.addWidget(rb); self._fem_buttons.append(rb)

        # ---- GT widgets ----
        def _load_widgets(self, r):
            want = str(r.get("female_id", ""))
            for b in self._fem_buttons:
                b.setChecked(b.property("fid") == want)
            sval = r.get("swap", "")
            for b in self.swap_group.buttons():
                b.setChecked(b.text() == sval)
            correct = str(r.get("correct_methods", "") or "").split(";")
            for lab, cb in self.correct_boxes.items():
                cb.blockSignals(True); cb.setChecked(lab in correct); cb.blockSignals(False)
            scval = str(r.get("suspect_correct", "") or "")
            for b in self.suspect_group.buttons():
                b.setChecked(b.text() == scval)
            self.sev.blockSignals(True)
            sv = str(r.get("severity", "") or "")
            self.sev.setCurrentIndex(SEVERITY.index(sv) + 1 if sv in SEVERITY else 0)
            self.sev.blockSignals(False)
            self.notes.blockSignals(True)
            nt = r.get("notes", "")
            self.notes.setPlainText("" if (nt is None or pd.isna(nt)) else str(nt)); self.notes.blockSignals(False)

        def _collect(self):
            e = self.d.events[self.cur]
            fem_btn = self.fem_group.checkedButton()
            female = fem_btn.property("fid") if fem_btn else ""
            # map the chosen canonical female back to each method's raw id (traceability)
            raw_by = ""
            if female and female != "unclear" and female in CANONICAL_LABELS:
                ci = CANONICAL_LABELS.index(female)
                parts = []
                for lab in self.labels:
                    match = [str(k) for k, v in self.color_map.get(lab, {}).items() if v == ci]
                    if match:
                        parts.append(f"{lab}={match[0]}")
                raw_by = ";".join(parts)
            swap_btn = self.swap_group.checkedButton()
            sc_btn = self.suspect_group.checkedButton()
            correct = ";".join(lab for lab, cb in self.correct_boxes.items() if cb.isChecked())
            self.records[self.cur] = {
                "event": self.cur, "center_frame": e["center"], "min_mm": round(e["min_mm"], 2),
                "source": e["source"], "track_a": e["track_a"], "track_b": e["track_b"],
                "suspect_fish": e["suspect"],
                "female_id": female,
                "female_raw_by_method": raw_by,
                "swap": swap_btn.text() if swap_btn else "",
                "correct_methods": correct,
                "suspect_correct": sc_btn.text() if (sc_btn and e["suspect"]) else "",
                "severity": SEVERITY[self.sev.currentIndex() - 1] if self.sev.currentIndex() > 0 else "",
                "notes": self.notes.toPlainText().strip(),
            }

        def _on_change(self, *a):
            self._collect(); self._update_progress()

        def _update_progress(self):
            done = sum(1 for r in self.records.values() if r.get("swap") or r.get("female_id"))
            self.progress.setText(f"reviewed: {done} / {len(self.d.events)}   (auto-saved)")

        # ---- rendering ----
        def _draw_method(self, frame, pts, cmap, method_idx, label):
            # SHAPE = method; COLOR = aligned fish; text = canonical letter + this
            # method's raw id (e.g. "A:15"), so the canonical id reads the same across
            # methods while the raw id stays visible for traceability.
            shape = MARKER_SHAPES[method_idx % len(MARKER_SHAPES)]
            dy = -6 - 13 * method_idx   # stagger so overlaid methods don't collide
            for fid, x, y in pts:
                ci = cmap.get(fid)
                col = ID_COLORS[ci % len(ID_COLORS)] if ci is not None else GRAY
                canon = CANONICAL_LABELS[ci % len(CANONICAL_LABELS)] if ci is not None else "?"
                draw_marker(frame, x, y, col, shape, r=10, t=2)
                cv2.putText(frame, f"{canon}:{fid}", (int(x) + 12, int(y) + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            return frame

        def _draw_legend(self, comp, sel):
            """Shape -> method legend (top-left). Color still = fish id."""
            y = 18
            for lab in sel:
                shape = MARKER_SHAPES[self.labels.index(lab) % len(MARKER_SHAPES)]
                draw_marker(comp, 16, y, (255, 255, 255), shape, r=7, t=1)
                cv2.putText(comp, lab, (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                y += 22
            return comp

        def _zoom(self, fr, pts):
            z = self.cfg.zoom_crop
            if z <= 0 or not pts:
                return fr
            cx = int(np.mean([p[1] for p in pts])); cy = int(np.mean([p[2] for p in pts]))
            x0 = min(max(cx - z // 2, 0), max(0, self.d.w - z))
            y0 = min(max(cy - z // 2, 0), max(0, self.d.h - z))
            return fr[y0:y0 + min(z, self.d.h), x0:x0 + min(z, self.d.w)]

        def _selected_methods(self):
            return [lab for lab in self.labels if self.method_boxes[lab].isChecked()]

        def _render(self, f):
            sel = self._selected_methods()
            if self.split_cb.isChecked() and sel:
                comp = self._render_split(f, sel)
            else:
                comp = self._render_overlay(f, sel)
            self._draw_legend(comp, sel)
            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            h, w, _ = comp.shape
            qimg = QtGui.QImage(comp.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.img.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                self.img.width(), self.img.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.frame_lbl.setText(f"frame {f}   t={f/self.cfg.fps:.2f}s   "
                                   f"[{'split:'+self.split_view if self.split_cb.isChecked() else 'overlay'}]")

        def _overlay_context(self, fr, lab, v, f, show_sleap, show_mask):
            if show_mask:
                m = self.d.mask(lab, v, f)
                if m is not None and m.shape[:2] == fr.shape[:2]:
                    draw_mask_contours(fr, m)
            if show_sleap:
                insts = self.d.preds(lab, v).get(f)
                if insts:
                    draw_predictions(fr, insts)

        def _render_overlay(self, f, sel):
            show_sleap, show_mask = self.sleap_cb.isChecked(), self.mask_cb.isChecked()
            panels = []
            for v in VIEWS:
                fr = self.d.read(v, f).copy()
                for lab in sel:
                    self._overlay_context(fr, lab, v, f, show_sleap, show_mask)
                allpts = []
                for lab in sel:
                    pts = self.d.reproj[lab].get(f, {}).get(v, [])
                    self._draw_method(fr, pts, self.color_map.get(lab, {}), self.labels.index(lab), lab)
                    allpts += pts
                fr = self._zoom(fr, allpts)
                cv2.putText(fr, v, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                panels.append(cv2.resize(fr, (440, 440)))
            return np.hstack(panels)

        def _render_split(self, f, sel):
            v = self.split_view
            show_sleap, show_mask = self.sleap_cb.isChecked(), self.mask_cb.isChecked()
            panels = []
            for lab in sel:
                fr = self.d.read(v, f).copy()
                self._overlay_context(fr, lab, v, f, show_sleap, show_mask)
                pts = self.d.reproj[lab].get(f, {}).get(v, [])
                self._draw_method(fr, pts, self.color_map.get(lab, {}), self.labels.index(lab), lab)
                fr = self._zoom(fr, pts)
                cv2.putText(fr, f"{lab} [{v}]", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                panels.append(cv2.resize(fr, (440, 440)))
            return np.hstack(panels)

        def _toggle_play(self):
            if self._play.isActive():
                self._stop_play()
            else:
                self.play_btn.setText("Pause"); self._play.start(40)

        def _stop_play(self):
            self._play.stop(); self.play_btn.setText("Play")

        def _advance(self):
            nv = self.slider.value() + 1
            self.slider.setValue(self.slider.minimum() if nv > self.slider.maximum() else nv)

        def closeEvent(self, e):
            self._collect(); self._save(); super().closeEvent(e)

    return TrialData, ReviewWindow


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gt-out", type=Path, help="override manifest gt_out")
    ap.add_argument("--max-events", type=int, help="override manifest max_events")
    args = ap.parse_args(argv)
    cfg = load_manifest(args.manifest, {"gt_out": args.gt_out, "max_events": args.max_events})

    from PyQt5 import QtWidgets
    TrialData, ReviewWindow = _build_gui_classes()
    data = TrialData(cfg)
    app = QtWidgets.QApplication(sys.argv)
    win = ReviewWindow(data)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
