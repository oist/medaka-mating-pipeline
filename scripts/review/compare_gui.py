"""Compare GUI: two versions of the 3D tracks on top of each other. Two modes, chosen at start-up
(``--mode``; without it a dialog lists what the folder has):

* **review** - what each correction step changed (read-only). Needs ``versions.npz`` from
  :mod:`scripts.review.export_reviewed`. Versions (:mod:`scripts.review.versions`): ``initial`` (plain
  triangulation), ``machine`` (the review data, what the reviewer watched), ``manual`` (after the
  reviewer's ledger) and ``final`` (after the machine post-pass). Presets: machine step
  (initial -> machine), review (machine -> manual), post-pass (manual -> final), all (initial -> final).
  "Same identity" (I) gives every version the reviewed identity, so only geometry differences remain;
  off, the earlier versions keep the tracker's identity and the review's swaps show as differences.
  The change list holds every build fix, reviewer edit and post-pass run.
* **recon** - two 3D reconstructions of the same recording (:mod:`scripts.review.recon_compare`;
  ``<folder>/recon_vs_<other>/recon_versions.npz``). Needs no human review. Name the reference's fish
  once (1/2/3 or click, then 4/5/6 = F/M1/M2); the other version takes the names of its tracks at that
  frame and then follows its own tracks, so a swap in either one shows. "Same identity" (I) matches the
  versions frame by frame instead (geometry only). The change list holds every disagreement (swap,
  position, head/tail, one-sided data); judge each with A / B / W / O (A right, B right, both wrong,
  both OK; Backspace clears). Names and verdicts are saved in the ``recon_vs_*`` folder only.

::

    python -m scripts.review.compare_gui <review_root>/<exp>/<round_tag>
    python -m scripts.review.compare_gui <folder> --mode recon --recon-dir <folder>/recon_vs_<other>

Shown in the three camera views (the 3D nodes projected with the calibration; A dashed, B solid), a
rotatable 3D view and traces under them. Filter the change list, double-click or N / P to go there.
E exports the span shown in the traces (or any range) as an mp4 into ``compare_clips/``; the same is
available without the GUI::

    python -m scripts.review.compare_gui <folder> --clip 105000 105400 --a machine --b final --speed 0.25

Keys: Space play/pause, Left/Right one frame, Shift+Left/Right 1 s, Up/Down playback speed,
N/P next/previous change, 1-4 presets (review) or 1/2/3 select a fish and 4/5/6 name it F/M1/M2 (recon),
7/8/9 version pairs (recon, three versions), A/B/W/O verdict and Backspace clear (recon), I same identity,
Z zoom onto the fish, E export clip, F1 help. All keys work whatever widget has the focus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.review.ledger import EVENT_COLUMNS, PAIR_SEP, read_csv, static_labels
from scripts.review.postprocess import REPORT_COLUMNS
from scripts.review.recon_compare import GROUPS as RECON_GROUPS
from scripts.review.recon_compare import RECON_FILE, VERDICTS, ReconState
from scripts.review.review_gui import (
    LABEL_COLORS,
    SELECT_RADIUS_SRC_PX,
    TRACK_COLORS,
    VIEW_TITLES,
    VIEWS,
    ProxyVideo,
    _runs,
    crop_rect,
    crop_target,
    dashed_line,
    proxy_fps,
    pt,
)
from scripts.review.versions import VERSION_TITLES, VERSIONS, VERSIONS_FILE

PRESETS = (("machine step", "initial", "machine"), ("review", "machine", "manual"),
           ("post-pass", "manual", "final"), ("all", "initial", "final"))
GROUPS = ("build", "review", "post")
GROUP_COLORS = {"build": (90, 180, 90), "review": (230, 150, 60), "post": (60, 150, 240)}   # BGR
RECON_GROUP_COLORS = {"identity": (70, 70, 235), "geometry": (40, 190, 250), "coverage": (170, 170, 170)}
REVIEW_COLUMNS = ["start", "end", "fish", "group", "kind", "mm", "mm/s"]
RECON_COLUMNS = REVIEW_COLUMNS + ["verdict"]
AGREE_COLOR = (60, 215, 60)    # BGR green: A and B put this node on the same spot
AGREE_PX = 4.0                 # display pixels (x mark size) that look like one line
NAME_KEYS = ("F", "M1", "M2")                       # recon mode: keys 4 / 5 / 6
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TRACE_QUANTITIES = ("speed (mm/s)", "|A - B| (mm)", "acceleration (mm/frame^2)", "body length (mm)",
                    "X (mm)", "Y (mm)", "Z (mm)")
SPEED_HALF = 2                 # frames each side for speed
BUILD_DIFF_MM = 1.0            # initial vs machine difference that counts as a build fix
FAST_MM_S = 200.0              # "fast" filter of the change list
CLIP_FPS = 30.0
ERROR_LOG = "compare_gui_errors.log"


# ---------------------------------------------------------------------------
# Data (no Qt)


def _proxy_manifest(folder: Path) -> dict:
    return json.loads((Path(folder).parent / "proxy" / "proxy_manifest.json").read_text(encoding="utf-8"))


class _ViewData:
    """Proxies, calibration (for drawing into the views) and frame bookkeeping of both modes."""

    def _setup(self, folder: Path, frames: np.ndarray, calibration: bool):
        self.folder = Path(folder)
        self.frames = frames
        self.first, self.last = int(self.frames[0]), int(self.frames[-1])
        self.proxy_dir = self.folder.parent / "proxy"
        self.proxy = _proxy_manifest(self.folder)
        top = self.proxy["views"]["top_cam"]
        self.fps = float(top["source_fps"])
        self.step = int(top["play_step"])
        self.scale = [self.proxy["views"][v]["proxy_width"] / self.proxy["views"][v]["source_width"] for v in VIEWS]
        manifest_path = self.folder / "review_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.experiment = str(self.manifest.get("experiment", self.folder.parent.name))
        self.cal = None
        if calibration:
            try:
                from scripts.reconstruction_3d.calibration import Calibration

                self.cal = Calibration(str(self.manifest["sources"]["calibration_dir"]), backend="auto")
            except Exception:  # noqa: BLE001 - views then show no 3D overlay
                self.cal = None
        self._play: dict[int, ProxyVideo] = {}
        self.play_cache = 36                     # playback-proxy frames kept per view (a looped clip replays from it)
        self._full: dict[int, ProxyVideo] = {}

    def t(self, frame: int) -> int | None:
        return frame - self.first if self.first <= frame <= self.last else None

    def speed(self, nodes: np.ndarray) -> np.ndarray:
        """``(T, L)`` mm/s of the position node from ``(T, L, N, 3)`` (central difference)."""
        p = nodes[:, :, self.pos_node]
        out = np.full(p.shape[:2], np.nan)
        h = SPEED_HALF
        if len(p) > 2 * h:
            out[h:-h] = np.linalg.norm(p[2 * h:] - p[:-2 * h], axis=-1) / (2 * h) * self.fps
        return out

    # -- video -----------------------------------------------------------------
    def image(self, v: int, frame: int) -> np.ndarray:
        if frame % self.step == 0:
            if v not in self._play:
                self._play[v] = ProxyVideo(self.proxy_dir / self.proxy["views"][VIEWS[v]]["play_file"],
                                           cache=self.play_cache, fps=proxy_fps(self.proxy, VIEWS[v], True))
            img = self._play[v].get(frame // self.step)
        else:
            if v not in self._full:
                self._full[v] = ProxyVideo(self.proxy_dir / self.proxy["views"][VIEWS[v]]["full_file"],
                                           fps=proxy_fps(self.proxy, VIEWS[v], False))
            img = self._full[v].get(frame)
        if img is None:
            meta = self.proxy["views"][VIEWS[v]]
            img = np.zeros((meta["proxy_height"], meta["proxy_width"], 3), np.uint8)
        return img

    def close(self):
        for vid in list(self._play.values()) + list(self._full.values()):
            vid.close()


class CompareData(_ViewData):
    """Review mode: ``versions.npz``, the proxies, the calibration and the change list."""

    mode = "review"
    versions = VERSIONS
    titles = VERSION_TITLES
    presets = PRESETS
    groups = GROUPS
    group_colors = GROUP_COLORS
    reviewed = ("manual", "final")            # versions that already carry the reviewed identity
    default_pair = ("machine", "final")
    table_columns = REVIEW_COLUMNS

    def __init__(self, folder: Path, versions_dir: Path | None = None, calibration: bool = True):
        vdir = Path(versions_dir) if versions_dir else Path(folder)
        with np.load(vdir / VERSIONS_FILE) as z:
            self.nodes = z["nodes"]                                  # (V, F, L, N, 3)
            self.source = z["source"]                                # (V, F, L, N)
            self.track = z["track"]                                  # (V, F, L)
            self.present = [str(v) for v in z["present"]]
            self.labels = [str(x) for x in z["labels"]]
            frames = z["frames"]
            self.node_names = [str(x) for x in z["node_names"]]
            self.edges = [tuple(int(a) for a in e) for e in z["edges"]]
            self.head, self.tail = int(z["head_node"]), int(z["tail_node"])
            self.pos_node = int(z["position_node"])
            start = int(z["analysis_start"])
            self.fish_ids = [str(x) for x in z["fish_ids"]]
        self.analysis_start = None if start < 0 else start
        self._setup(folder, frames, calibration)
        self.clip_dir = self.folder / "compare_clips"
        report = vdir / "postprocess_report.csv"
        self.report = pd.read_csv(report) if report.exists() else pd.DataFrame(columns=REPORT_COLUMNS)
        self.events = read_csv(self.folder / "human_events.csv", EVENT_COLUMNS)
        self.track_label = static_labels(self.events, self.fish_ids) if len(self.events) else {}
        self.changes = self.build_changes()

    # -- versions ------------------------------------------------------------
    def vindex(self, name: str) -> int:
        return VERSIONS.index(name)

    def bounds_nodes(self) -> np.ndarray:
        """Positions ``(M, 3)`` that set the 3D view's box."""
        return self.nodes[self.vindex("final"), :, :, self.pos_node].reshape(-1, 3)

    def series(self, version: str, lo: int, hi: int, aligned: bool) -> np.ndarray:
        """Nodes ``(T, L, N, 3)`` of rows ``lo:hi``. ``aligned``: data taken per name from the track the
        reviewed (``manual``) version gives that name, so every version has the reviewed identity."""
        v = self.vindex(version)
        out = self.nodes[v, lo:hi].astype(float)
        if not aligned or version in ("manual", "final"):
            return out
        own, want = self.track[v, lo:hi], self.track[self.vindex("manual"), lo:hi]       # (T, L)
        res = np.full_like(out, np.nan)
        for c in range(len(self.labels)):
            match = own == want[:, [c]]                                                  # (T, L)
            ok = match.any(axis=1)
            idx = match.argmax(axis=1)
            res[ok, c] = out[np.flatnonzero(ok), idx[ok]]
        return res

    def sources(self, version: str, t: int, aligned: bool = False) -> np.ndarray:
        return self.source[self.vindex(version), t]

    # -- change list -----------------------------------------------------------
    def build_changes(self) -> pd.DataFrame:
        rows = []
        lab_of = lambda k: self.labels[k] if 0 <= k < len(self.labels) else "?"   # noqa: E731
        if "initial" in self.present:
            a, b = self.nodes[self.vindex("initial")], self.nodes[self.vindex("machine")]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # names without data in a version
                diff = np.nanmax(np.linalg.norm(a.astype(np.float32) - b, axis=-1), axis=-1)   # (F, L)
            despiked = (self.source[self.vindex("machine")] == 3).any(axis=-1)
            for c in range(len(self.labels)):
                for s, e in _runs(np.nan_to_num(diff[:, c]) > BUILD_DIFF_MM):
                    kind = "despike" if despiked[s:e + 1, c].any() else "two-view"
                    rows.append((self.first + s, self.first + e, lab_of(c), "build", kind,
                                 float(np.nanmax(diff[s:e + 1, c])), ""))
        names = {k: v for k, v in self.track_label.items()}
        for r in self.events.itertuples(index=False):
            if r.kind in ("label", "analysis_start", "confirm"):
                continue
            fish = [names.get(f, f) for f in str(r.fish).split(PAIR_SEP)] if r.fish != "*" else ["*"]
            try:
                f0 = int(r.frame)
            except ValueError:
                continue
            end = str(r.frame_end).strip()
            f1 = int(end) if end not in ("", "nan") else f0
            rows.append((f0, max(f0, f1), "/".join(fish), "review", str(r.kind), np.nan, str(r.value)))
        for r in self.report.itertuples(index=False):
            kind = f"{r.kind} {r.node}" + ("" if r.repaired else " (left)")
            rows.append((int(r.frame_start), int(r.frame_end), str(r.label or r.fish_id), "post", kind,
                         float(r.size_mm), ""))
        df = pd.DataFrame(rows, columns=["start", "end", "fish", "group", "kind", "size_mm", "note"])
        if len(df):
            # Speed of the fish there (final version), for the "fast" filter.
            fin = self.nodes[self.vindex("final")]
            spd = np.full(len(df), np.nan)
            for i, r in enumerate(df.itertuples(index=False)):
                if r.fish in self.labels:
                    c = self.labels.index(r.fish)
                    lo = max(0, r.start - self.first - SPEED_HALF)
                    hi = min(len(self.frames), r.end - self.first + SPEED_HALF + 1)
                    seg = self.speed(fin[lo:hi, [c]].astype(float))
                    if np.isfinite(seg).any():
                        spd[i] = np.nanmax(seg)
            df["speed_mm_s"] = spd
        else:
            df["speed_mm_s"] = []
        return df.sort_values(["start", "group"], kind="stable").reset_index(drop=True)


class ReconCompareData(_ViewData):
    """Reconstruction mode: :class:`scripts.review.recon_compare.ReconState` plus proxies and calibration."""

    mode = "recon"
    presets = ()
    groups = RECON_GROUPS
    group_colors = RECON_GROUP_COLORS
    table_columns = RECON_COLUMNS

    def __init__(self, folder: Path, recon_dir: Path, calibration: bool = True, params=None):
        fps = float(_proxy_manifest(folder)["views"]["top_cam"]["source_fps"])
        self.state = s = ReconState(folder, recon_dir, params, fps=fps)
        self._setup(folder, s.frames, calibration)
        self.recon_dir = Path(recon_dir)
        self.clip_dir = self.recon_dir / "compare_clips"
        self.versions = tuple(s.versions)
        self.present = list(s.versions)
        self.titles = {v: f"{v}: {tag} ({kind})" for v, tag, kind in zip(s.versions, s.tags, s.kinds)}
        self.reviewed = tuple(s.versions)        # sources follow the shown mapping in every version
        self.default_pair = (s.versions[0], s.versions[1])
        ref, others = s.versions[0], s.versions[1:]
        pairs = [(ref, o) for o in others] + [(a, b) for i, a in enumerate(others) for b in others[i + 1:]]
        self.presets = tuple((f"{a} vs {b}", a, b) for a, b in pairs) if len(s.versions) > 2 else ()
        self.node_names, self.edges = s.node_names, s.edges
        self.head, self.tail, self.pos_node = s.head, s.tail, s.pos_node
        self.fish_ids = s.ref_ids
        self.analysis_start = s.analysis_start

    @property
    def labels(self) -> list[str]:
        return self.state.labels

    @property
    def changes(self) -> pd.DataFrame:
        return self.state.changes

    def vindex(self, name: str) -> int:
        return self.state.vindex(name)

    def series(self, version: str, lo: int, hi: int, aligned: bool) -> np.ndarray:
        return self.state.series(version, lo, hi, aligned)

    def sources(self, version: str, t: int, aligned: bool = False) -> np.ndarray:
        return self.state.sources(version, t, aligned)

    def bounds_nodes(self) -> np.ndarray:
        return self.state.nodes[:, :, :, self.pos_node].reshape(-1, 3)

    def set_name(self, k: int, label: str, frame: int) -> str:
        return self.state.set_name(k, label, frame)

    def set_verdict(self, row: int, verdict: str):
        self.state.set_verdict(row, verdict)

    def verdict_counts(self) -> dict[str, int]:
        return self.state.verdict_counts()

    def naming_text(self) -> str:
        s = self.state
        if s.reviewed_reference:
            return (f"reference '{s.versions[0]}' = the reviewed result (names fixed); other versions took its "
                    f"names at frame {s.link_frame}")
        names = ", ".join(f"{f}={s.names[f]}" for f in s.ref_ids if s.names.get(f)) or "nobody named yet"
        return f"names ({s.versions[0]}): {names}; other versions linked at frame {s.link_frame}"


def available_modes(folder: Path, versions_dir: Path | None = None) -> list[tuple[str, Path, str]]:
    """``(mode, path, title)`` of everything ``folder`` can be opened with: review mode when
    ``versions.npz`` exists, one reconstruction mode per ``recon_vs_*`` folder."""
    folder = Path(folder)
    out = []
    vdir = Path(versions_dir) if versions_dir else folder
    if (vdir / VERSIONS_FILE).exists():
        out.append(("review", vdir, "Review: machine vs manual corrections (versions.npz)"))
    if folder.is_dir():
        for d in sorted(p for p in folder.iterdir() if (p / RECON_FILE).exists()):
            out.append(("recon", d, f"Reconstruction comparison: {d.name}"))
    return out


def label_color(name: str, c: int) -> tuple[int, int, int]:
    return LABEL_COLORS.get(name, TRACK_COLORS[c % len(TRACK_COLORS)])


def project(cd: CompareData, nodes: np.ndarray) -> np.ndarray:
    """``(L, N, 3)`` -> source pixels ``(L, N, V, 2)`` (NaN without calibration)."""
    flat = nodes.reshape(-1, 3)
    out = np.full((len(flat), len(VIEWS), 2), np.nan)
    ok = np.isfinite(flat).all(axis=1)
    if cd.cal is not None and ok.any():
        out[ok] = cd.cal.project_world(flat[ok])
    return out.reshape(nodes.shape[:2] + (len(VIEWS), 2))


def coincide(qa: np.ndarray, qb: np.ndarray, mark: float = 1.0) -> np.ndarray:
    """``(L, N)`` True where A's and B's node fall on the same display pixel area (< ``AGREE_PX * mark``)."""
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(np.asarray(qa, float) - np.asarray(qb, float), axis=-1)
    return np.nan_to_num(d, nan=np.inf) <= AGREE_PX * mark


def draw_pair(img: np.ndarray, labels: list[str], edges, head: int, pos: int, qa: np.ndarray, qb: np.ndarray,
              mark: float, *, cross: bool = True, names: bool = True) -> None:
    """Skeletons of A (dashed) and B (solid) from display pixels ``(L, N, 2)``. Where both put a node on the
    same spot the part is drawn once in ``AGREE_COLOR`` - green = both versions have it there; a lone
    dashed / solid part is only in A / only in B (or there they differ). Names keep the fish colour."""
    lw = max(1, round(1.5 * mark))
    agree = coincide(qa, qb, mark)

    def cross_at(p, color):
        s = round(5 * mark)
        cv2.line(img, (int(p[0]) - s, int(p[1])), (int(p[0]) + s, int(p[1])), color, 1, cv2.LINE_AA)
        cv2.line(img, (int(p[0]), int(p[1]) - s), (int(p[0]), int(p[1]) + s), color, 1, cv2.LINE_AA)

    for c, name in enumerate(labels):
        color = label_color(name, c)
        same = agree[c]
        for q, dashed in ((qa[c], True), (qb[c], False)):
            ok = np.isfinite(q).all(axis=1)
            for i, j in edges:
                if ok[i] and ok[j] and not (same[i] and same[j]):
                    if dashed:
                        dashed_line(img, q[i], q[j], color, lw, 4 * mark)
                    else:
                        cv2.line(img, pt(q[i]), pt(q[j]), color, lw, cv2.LINE_AA)
            for n in np.flatnonzero(ok & ~same):
                cv2.circle(img, pt(q[n]), round((4.0 if n == head else 2.5) * mark), color, 1 if dashed else -1,
                           cv2.LINE_AA)
            if cross and ok[pos] and not same[pos]:
                cross_at(q[pos], color)
        q = qb[c]
        for i, j in edges:
            if same[i] and same[j]:
                cv2.line(img, pt(q[i]), pt(q[j]), AGREE_COLOR, lw + 1, cv2.LINE_AA)
        for n in np.flatnonzero(same):
            cv2.circle(img, pt(q[n]), round((4.0 if n == head else 2.5) * mark), AGREE_COLOR, -1, cv2.LINE_AA)
        if cross and same[pos]:
            cross_at(q[pos], AGREE_COLOR)
        if names:
            src = qb[c] if np.isfinite(qb[c]).all(axis=1).any() else qa[c]
            ok = np.isfinite(src).all(axis=1)
            if ok.any():
                anchor = src[head] if ok[head] else src[ok][0]
                cv2.putText(img, name, (int(anchor[0]) + 6, int(anchor[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45 * mark, color, max(1, round(mark)), cv2.LINE_AA)


def render_view(cd: CompareData, v: int, frame: int, pair: tuple[str, str], aligned: bool, size: tuple[int, int],
                crop: np.ndarray | None = None, mark: float = 1.2, projections=None,
                selected: int | None = None) -> tuple[np.ndarray, tuple]:
    """One camera view at ``size (w, h)`` with version A (dashed) and B (solid) drawn on it; ``selected``
    rings that fish. Returns the image and its source-pixel -> display transform."""
    img = cd.image(v, frame)
    h, w = img.shape[:2]
    lw_, lh_ = size
    x0, y0, cw, ch = crop_rect(crop, w, h, lw_ / lh_) if crop is not None else (0, 0, w, h)
    s = min(lw_ / cw, lh_ / ch)
    out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (max(1, round(cw * s)), max(1, round(ch * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    tf = (cd.scale[v] * s, -x0 * s, -y0 * s)
    t = cd.t(frame)
    if t is not None:
        pa, pb = projections if projections is not None else frame_projections(cd, t, pair, aligned)
        off = np.array(tf[1:])
        draw_pair(out, cd.labels, cd.edges, cd.head, cd.pos_node, pa[:, :, v] * tf[0] + off,
                  pb[:, :, v] * tf[0] + off, mark)
        if selected is not None and 0 <= selected < len(cd.labels):
            q = pa[selected, cd.pos_node, v] * tf[0] + np.array(tf[1:])
            if np.isfinite(q).all():
                cv2.circle(out, pt(q), round(14 * mark), (255, 255, 255), 2, cv2.LINE_AA)
    title = VIEW_TITLES[v]
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out, tf


def frame_projections(cd: CompareData, t: int, pair: tuple[str, str], aligned: bool) -> tuple[np.ndarray, np.ndarray]:
    a = cd.series(pair[0], t, t + 1, aligned)[0]
    b = cd.series(pair[1], t, t + 1, aligned)[0]
    both = project(cd, np.concatenate([a, b]))
    return both[:len(a)], both[len(a):]


def crop_for(cd: CompareData, t: int, pair: tuple[str, str], aligned: bool, v: int) -> np.ndarray:
    """Crop state (proxy pixels) framing every fish of both versions in view ``v``."""
    pa, pb = frame_projections(cd, t, pair, aligned)
    pts = np.concatenate([pa[:, :, v], pb[:, :, v]]).reshape(-1, 2) * cd.scale[v]
    meta = cd.proxy["views"][VIEWS[v]]
    return crop_target(pts, meta["proxy_width"], meta["proxy_height"])


def trace_strip(cd: CompareData, lo: int, hi: int, pair: tuple[str, str], aligned: bool, width: int,
                height: int) -> np.ndarray:
    """Speed of every fish over source frames ``lo..hi``: A dashed, B solid (BGR image)."""
    img = np.full((height, width, 3), 22, np.uint8)
    t0, t1 = max(0, lo - cd.first), min(len(cd.frames), hi - cd.first + 1)
    if t1 - t0 < 2:
        return img
    speeds = [cd.speed(cd.series(name, t0, t1, aligned)) for name in pair]
    top = max(50.0, float(np.nanpercentile(np.concatenate([s.ravel() for s in speeds]), 99.5))
              if any(np.isfinite(s).any() for s in speeds) else 50.0)
    xs = np.linspace(4, width - 4, t1 - t0)
    for k, (sp, dashed) in enumerate(zip(speeds, (True, False))):
        for c, name in enumerate(cd.labels):
            y = height - 6 - (height - 20) * np.clip(sp[:, c] / top, 0, 1)
            pts = np.stack([xs, y], axis=1)
            for s, e in _runs(np.isfinite(y)):
                if e <= s:
                    continue
                if dashed:
                    for i in range(s, e, 6):
                        cv2.line(img, pt(pts[i]), pt(pts[min(i + 3, e)]), label_color(name, c), 1, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts[s:e + 1].round().astype(np.int32).reshape(-1, 1, 2)], False,
                                  label_color(name, c), 1, cv2.LINE_AA)
    cv2.putText(img, f"speed 0-{top:.0f} mm/s   A {pair[0]} dashed / B {pair[1]} solid", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def export_clip(cd: CompareData, start: int, end: int, pair: tuple[str, str], *, aligned: bool = True,
                speed: float = 0.25, out: Path | None = None, view_size: int = 420, zoom: bool = True,
                progress=None) -> Path:
    """Write the three views (A dashed, B solid) plus a speed strip for source frames ``start..end``
    as an mp4 at 30 fps, ``speed`` x real time (0.25 = four times slower)."""
    start, end = max(start, cd.first), min(end, cd.last)
    if end < start:
        raise ValueError("empty frame range")
    if out is None:
        out = cd.clip_dir / f"{cd.experiment}_{pair[0]}_vs_{pair[1]}_{start}-{end}.mp4"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(speed * cd.fps / CLIP_FPS)))
    header_h, strip_h = 34, 110
    width, height = 3 * view_size, header_h + view_size + strip_h
    strip = trace_strip(cd, start, end, pair, aligned, width, strip_h)
    tmp = out.with_name(out.stem + ".partial.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (width, height))
    crops: list[np.ndarray | None] = [None, None, None]
    frames = list(range(start, end + 1, step))
    try:
        for i, f in enumerate(frames):
            canvas = np.zeros((height, width, 3), np.uint8)
            t = cd.t(f)
            proj = frame_projections(cd, t, pair, aligned) if t is not None else None
            for v in range(len(VIEWS)):
                crop = None
                if zoom and t is not None:
                    target = crop_for(cd, t, pair, aligned, v)
                    crops[v] = target if crops[v] is None else crops[v] + 0.25 * (target - crops[v])
                    crop = crops[v]
                img, _ = render_view(cd, v, f, pair, aligned, (view_size, view_size), crop, projections=proj)
                hh, ww = img.shape[:2]
                y0, x0 = header_h + (view_size - hh) // 2, v * view_size + (view_size - ww) // 2
                canvas[y0:y0 + hh, x0:x0 + ww] = img
            s = strip.copy()
            x = int(round(4 + (width - 8) * (f - start) / max(1, end - start)))
            cv2.line(s, (x, 0), (x, strip_h), (255, 255, 255), 1)
            canvas[header_h + view_size:] = s
            secs = f / cd.fps
            text = (f"{cd.experiment}  frame {f}  {int(secs // 60):02d}:{secs % 60:05.2f}   "
                    f"A {cd.titles[pair[0]]} (dashed)  vs  B {cd.titles[pair[1]]} (solid)   "
                    f"{speed:g}x{'  same identity' if aligned else ''}")
            cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            writer.write(canvas)
            if progress is not None and not progress(i + 1, len(frames)):
                break
    finally:
        writer.release()
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Qt


def open_data(folder: Path, *, mode: str | None = None, versions_dir: Path | None = None,
              recon_dir: Path | None = None, choose=None, calibration: bool = True):
    """The data for ``mode`` (``review`` / ``recon``). Without a mode (or a recon mode without a folder)
    ``choose(titles) -> index | None`` picks among :func:`available_modes`; ``None`` there = cancel."""
    if recon_dir is not None:
        mode = "recon"
    if mode == "review":
        if not (Path(versions_dir) if versions_dir else Path(folder)).joinpath(VERSIONS_FILE).exists():
            raise SystemExit(f"no {VERSIONS_FILE} in {versions_dir or folder}: run scripts.review.export_reviewed")
        return CompareData(folder, versions_dir, calibration=calibration)
    if mode == "recon" and recon_dir is not None:
        return ReconCompareData(folder, recon_dir, calibration=calibration)
    modes = [m for m in available_modes(folder, versions_dir) if mode is None or m[0] == mode]
    if not modes:
        raise SystemExit(f"nothing to compare in {folder}: no {VERSIONS_FILE} (scripts.review.export_reviewed) "
                         f"and no recon_vs_*/{RECON_FILE} (scripts.review.recon_compare)")
    index = 0 if len(modes) == 1 or choose is None else choose([m[2] for m in modes])
    if index is None:
        return None
    kind, path, _ = modes[index]
    if kind == "review":
        return CompareData(folder, path, calibration=calibration)
    return ReconCompareData(folder, path, calibration=calibration)


def run_app(folder: Path, *, versions_dir: Path | None = None, smoke_test: bool = False, mode: str | None = None,
            recon_dir: Path | None = None) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    from scripts.review.review_gui import install_error_log

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

    def choose(titles):
        if smoke_test:
            return 0
        item, ok = QtWidgets.QInputDialog.getItem(None, "Compare", "What to compare:", titles, 0, False)
        return titles.index(item) if ok else None

    cd = open_data(folder, mode=mode, versions_dir=versions_dir, recon_dir=recon_dir, choose=choose)
    if cd is None:
        return 0
    recon = cd.mode == "recon"

    class ViewLabel(QtWidgets.QLabel):
        double_clicked = QtCore.pyqtSignal(int)
        clicked = QtCore.pyqtSignal(int, int, int)

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

        def image_offset(self) -> tuple[float, float]:
            pm = self.pixmap()
            if pm is None:
                return 0.0, 0.0
            return (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2

        def mousePressEvent(self, event):
            self.clicked.emit(self.index, event.x(), event.y())

        def mouseDoubleClickEvent(self, _event):
            self.double_clicked.emit(self.index)

    class View3D(QtWidgets.QWidget):
        """Both versions in 3D: drag rotates, wheel zooms, double-click resets."""

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.setMinimumSize(220, 220)
            self.az, self.el, self.zoom = 20.0, 40.0, 1.0
            self._drag = None
            p = cd.bounds_nodes()
            ok = np.isfinite(p).all(axis=1)
            lo, hi = (np.percentile(p[ok], [0.5, 99.5], axis=0) if ok.any() else (np.zeros(3), np.full(3, 200.0)))
            self.lo, self.hi = lo - 5, hi + 5

        def mousePressEvent(self, e):
            self._drag = (e.x(), e.y(), self.az, self.el)

        def mouseMoveEvent(self, e):
            if self._drag:
                x0, y0, az, el = self._drag
                self.az, self.el = az - (e.x() - x0) * 0.4, float(np.clip(el + (e.y() - y0) * 0.4, -89, 89))
                self.update()

        def mouseReleaseEvent(self, _e):
            self._drag = None

        def wheelEvent(self, e):
            self.zoom = float(np.clip(self.zoom * 1.15 ** (e.angleDelta().y() / 120), 1.0, 20.0))
            self.update()

        def mouseDoubleClickEvent(self, _e):
            self.az, self.el, self.zoom = 20.0, 40.0, 1.0
            self.update()

        def paintEvent(self, _e):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 18, np.uint8)
            win = self.win
            t = cd.t(win.frame)
            scale = self.zoom * min(w, h) / (float((self.hi - self.lo).max()) * 1.3)
            centre = (self.lo + self.hi) / 2
            if t is not None and self.zoom > 1:
                b = cd.series(win.pair[1], t, t + 1, win.aligned)[0, :, cd.pos_node]
                if np.isfinite(b).any():
                    centre = centre + min(1.0, self.zoom - 1) * (np.nanmean(b, axis=0) - centre)
            a_, e_ = np.radians(self.az), np.radians(self.el)

            def proj(p):
                d = np.asarray(p, float) - centre
                x1 = d[..., 0] * np.cos(a_) - d[..., 1] * np.sin(a_)
                y1 = d[..., 0] * np.sin(a_) + d[..., 1] * np.cos(a_)
                return np.stack([w / 2 + scale * x1, h / 2 - scale * (d[..., 2] * np.cos(e_) + y1 * np.sin(e_))], -1)

            lo, hi = self.lo, self.hi
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            q = proj(corners)
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") == 1:
                        cv2.line(img, pt(q[i]), pt(q[j]), (80, 80, 80), 1, cv2.LINE_AA)
            if t is not None:
                qa, qb = (proj(cd.series(v, t, t + 1, win.aligned)[0]) for v in win.pair)
                draw_pair(img, cd.labels, cd.edges, cd.head, cd.pos_node, qa, qb, 1.0, cross=False)
            cv2.putText(img, "3D: drag rotate, wheel zoom | dashed A, solid B, green = both", (6, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))
            painter.end()

    class Traces(QtWidgets.QWidget):
        """The chosen quantity over the span around the current frame; click = go there, wheel = span."""

        def __init__(self, win):
            super().__init__()
            self.win = win
            self.half = 2.0           # seconds each side
            self.setMinimumHeight(170)

        def span(self) -> tuple[int, int]:
            h = int(self.half * cd.fps)
            return self.win.frame - h, self.win.frame + h

        def wheelEvent(self, e):
            self.half = float(np.clip(self.half * 1.25 ** (-e.angleDelta().y() / 120), 0.1, 120.0))
            self.update()

        def mousePressEvent(self, e):
            a, b = self.span()
            self.win.seek(int(round(a + (b - a) * e.x() / max(1, self.width()))))

        def values(self, lo: int, hi: int) -> tuple[list[np.ndarray], str]:
            win = self.win
            q = win.quantity.currentText()
            node = win.node_box.currentIndex()
            t0, t1 = max(0, lo - cd.first), min(len(cd.frames), hi - cd.first + 1)
            if t1 - t0 < 3:
                return [], q
            a, b = (cd.series(v, t0, t1, win.aligned) for v in win.pair)
            if q.startswith("|A - B|"):
                with np.errstate(invalid="ignore"):
                    return [np.linalg.norm(a[:, :, node] - b[:, :, node], axis=-1)], q
            out = []
            for x in (a, b):
                if q.startswith("speed"):
                    out.append(cd.speed(x))
                elif q.startswith("acceleration"):
                    p = x[:, :, node]
                    acc = np.full(p.shape[:2], np.nan)
                    acc[1:-1] = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
                    out.append(acc)
                elif q.startswith("body length"):
                    out.append(np.linalg.norm(x[:, :, cd.head] - x[:, :, cd.tail], axis=-1))
                else:
                    out.append(x[:, :, node, "XYZ".index(q[0])])
            return out, q

        def paintEvent(self, _e):
            w, h = max(1, self.width()), max(1, self.height())
            img = np.full((h, w, 3), 24, np.uint8)
            lo, hi = self.span()
            win = self.win
            # Changes in the span: a coloured band per group at the top.
            ch = win.filtered
            if len(ch):
                m = (ch["end"] >= lo) & (ch["start"] <= hi)
                for r in ch[m].itertuples(index=False):
                    x0 = int((r.start - lo) / max(1, hi - lo) * w)
                    x1 = max(x0 + 2, int((r.end + 1 - lo) / max(1, hi - lo) * w))
                    y = 2 + 5 * cd.groups.index(r.group)
                    cv2.rectangle(img, (x0, y), (x1, y + 4), cd.group_colors[r.group], -1)
            vals, q = self.values(lo, hi)
            if vals:
                allv = np.concatenate([v.ravel() for v in vals])
                allv = allv[np.isfinite(allv)]
                if len(allv):
                    vmin, vmax = np.percentile(allv, [0.5, 99.5])
                    if q.startswith(("speed", "|A", "accel", "body")):
                        vmin = 0.0
                    if vmax - vmin < 1e-6:
                        vmax = vmin + 1
                    pad = 0.08 * (vmax - vmin)
                    vmin, vmax = vmin - (0 if vmin == 0 else pad), vmax + pad
                    t0 = max(0, lo - cd.first)
                    x_of = lambda t: (cd.first + t0 + t - lo) / max(1, hi - lo) * w     # noqa: E731
                    y_of = lambda v: h - 16 - (h - 36) * (v - vmin) / (vmax - vmin)       # noqa: E731
                    for k, val in enumerate(vals):
                        dashed = len(vals) == 2 and k == 0
                        for c, name in enumerate(cd.labels):
                            ys = y_of(val[:, c])
                            xs = x_of(np.arange(len(ys)))
                            pts = np.stack([xs, ys], 1)
                            for s, e in _runs(np.isfinite(ys)):
                                if e <= s:
                                    continue
                                color = label_color(name, c)
                                if dashed:
                                    for i in range(s, e, 4):
                                        cv2.line(img, pt(pts[i]), pt(pts[min(i + 2, e)]), color, 1, cv2.LINE_AA)
                                else:
                                    cv2.polylines(img, [pts[s:e + 1].round().astype(np.int32).reshape(-1, 1, 2)],
                                                  False, color, 1, cv2.LINE_AA)
                    if q.startswith("acceleration"):
                        y3 = int(y_of(3.0))
                        dashed_line(img, np.array([0, y3]), np.array([w, y3]), (120, 120, 120), 1, 6)
                    cv2.putText(img, f"{vmax:.1f}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
                    cv2.putText(img, f"{vmin:.1f}", (4, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
            xc = int((win.frame - lo) / max(1, hi - lo) * w)
            cv2.line(img, (xc, 0), (xc, h), (255, 255, 255), 1)
            legend = (f"{q}  node {cd.node_names[win.node_box.currentIndex()]}  |  A {win.pair[0]} dashed, "
                      f"B {win.pair[1]} solid  |  span {2 * self.half:g} s (wheel)  |  bands top to bottom: "
                      + ", ".join(cd.groups))
            cv2.putText(img, legend, (40, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            painter = QtGui.QPainter(self)
            painter.drawImage(0, 0, QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888))
            painter.end()

    class Overview(QtWidgets.QWidget):
        def __init__(self, win):
            super().__init__()
            self.win = win
            self.setFixedHeight(34)

        def x_of(self, f):
            return (f - cd.first) / max(1, cd.last - cd.first) * self.width()

        def mousePressEvent(self, e):
            self.win.seek(int(cd.first + e.x() / max(1, self.width()) * (cd.last - cd.first)))

        def mouseMoveEvent(self, e):
            self.mousePressEvent(e)

        def paintEvent(self, _e):
            p = QtGui.QPainter(self)
            w, h = self.width(), self.height()
            p.fillRect(0, 0, w, h, QtGui.QColor(28, 28, 28))
            if cd.analysis_start is not None:
                p.fillRect(0, 0, int(self.x_of(cd.analysis_start)), h, QtGui.QColor(60, 30, 30))
            ch = self.win.filtered
            for gi, g in enumerate(cd.groups):
                b, gg, r = cd.group_colors[g]
                p.setPen(QtGui.QColor(r, gg, b))
                for s in ch.loc[ch["group"] == g, "start"].to_numpy():
                    x = int(self.x_of(s))
                    p.drawLine(x, 4 + 9 * gi, x, 11 + 9 * gi)
            p.setPen(QtGui.QColor(255, 255, 255))
            x = int(self.x_of(self.win.frame))
            p.drawLine(x, 0, x, h)
            p.end()

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            what = "reconstructions" if recon else "versions"
            self.setWindowTitle(f"Compare {what} - {cd.experiment}")
            self.frame = cd.analysis_start if cd.analysis_start is not None else cd.first
            if recon:
                self.frame = max(self.frame, cd.state.link_frame)
            self.pair = cd.default_pair
            self.aligned = not recon             # recon: start with each version's own identity
            self.zoom_on = True
            self.crops: list[np.ndarray | None] = [None, None, None]
            self.speed_i = SPEEDS.index(0.25)
            self.playing = False
            self._acc = 0.0
            self._proj_cache: dict = {}
            self._tf: dict[int, tuple] = {}
            self.selected: int | None = None
            self.filtered = cd.changes

            bar = QtWidgets.QWidget()
            hb = QtWidgets.QHBoxLayout(bar)
            hb.setContentsMargins(4, 2, 4, 2)
            self.a_box, self.b_box = QtWidgets.QComboBox(), QtWidgets.QComboBox()
            for box in (self.a_box, self.b_box):
                for v in cd.versions:
                    box.addItem(cd.titles[v], v)
                    if v not in cd.present:
                        box.model().item(box.count() - 1).setEnabled(False)
                box.currentIndexChanged.connect(self.on_pair)
            hb.addWidget(QtWidgets.QLabel("A (dashed)"))
            hb.addWidget(self.a_box)
            hb.addWidget(QtWidgets.QLabel("B (solid)"))
            hb.addWidget(self.b_box)
            for i, (name, a, b) in enumerate(cd.presets):
                btn = QtWidgets.QPushButton(f"{i + (7 if recon else 1)} {name}")
                btn.setEnabled(a in cd.present and b in cd.present)
                btn.clicked.connect(lambda _=False, a=a, b=b: self.set_pair(a, b))
                hb.addWidget(btn)
            self.same_id = QtWidgets.QCheckBox("I same identity")
            self.same_id.setChecked(self.aligned)
            self.same_id.toggled.connect(self.on_aligned)
            hb.addWidget(self.same_id)
            self.quantity = QtWidgets.QComboBox()
            self.quantity.addItems(TRACE_QUANTITIES)
            self.quantity.currentIndexChanged.connect(lambda _: self.traces.update())
            self.node_box = QtWidgets.QComboBox()
            self.node_box.addItems(cd.node_names)
            self.node_box.setCurrentIndex(cd.pos_node)
            self.node_box.currentIndexChanged.connect(lambda _: self.traces.update())
            hb.addWidget(self.quantity)
            hb.addWidget(self.node_box)
            exp_btn = QtWidgets.QPushButton("E export clip")
            exp_btn.clicked.connect(self.export_dialog)
            hb.addWidget(exp_btn)
            hb.addStretch(1)

            # Playback row: play / pause, frame steps, speed, go to frame.
            prow = QtWidgets.QWidget()
            ph = QtWidgets.QHBoxLayout(prow)
            ph.setContentsMargins(4, 0, 4, 0)
            self.play_btn = QtWidgets.QPushButton("> Play (Space)")
            self.play_btn.clicked.connect(self.toggle_play)
            ph.addWidget(self.play_btn)
            for text, d in (("<< 1 s", -int(cd.fps)), ("< frame", -1), ("frame >", 1), ("1 s >>", int(cd.fps))):
                btn = QtWidgets.QPushButton(text)
                btn.clicked.connect(lambda _=False, d=d: self.step(d))
                ph.addWidget(btn)
            ph.addWidget(QtWidgets.QLabel("speed"))
            self.speed_box = QtWidgets.QComboBox()
            self.speed_box.addItems([f"{s:g}x" for s in SPEEDS])
            self.speed_box.setCurrentIndex(self.speed_i)
            self.speed_box.currentIndexChanged.connect(self.set_speed)
            ph.addWidget(self.speed_box)
            goto_btn = QtWidgets.QPushButton("G go to frame")
            goto_btn.clicked.connect(self.goto_dialog)
            ph.addWidget(goto_btn)
            prev_btn, next_btn = QtWidgets.QPushButton("P previous"), QtWidgets.QPushButton("N next")
            prev_btn.clicked.connect(lambda: self.next_change(-1))
            next_btn.clicked.connect(lambda: self.next_change(1))
            ph.addWidget(prev_btn)
            ph.addWidget(next_btn)
            ph.addStretch(1)
            for wdg in bar.findChildren(QtWidgets.QWidget) + prow.findChildren(QtWidgets.QWidget):
                wdg.setFocusPolicy(QtCore.Qt.NoFocus)            # keys stay with the window

            self.labels_v = [ViewLabel(i) for i in range(len(VIEWS))]
            self.view3d = View3D(self)
            views = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            for lab in self.labels_v:
                views.addWidget(lab)
            views.addWidget(self.view3d)
            self.traces = Traces(self)
            self.overview = Overview(self)
            self.info = QtWidgets.QLabel()
            self.info.setTextFormat(QtCore.Qt.RichText)
            central = QtWidgets.QWidget()
            vb = QtWidgets.QVBoxLayout(central)
            vb.setContentsMargins(2, 2, 2, 2)
            vb.addWidget(bar)
            vb.addWidget(prow)
            split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            split.addWidget(views)
            split.addWidget(self.traces)
            split.setSizes([620, 200])
            vb.addWidget(split, 1)
            vb.addWidget(self.overview)
            vb.addWidget(self.info)
            self.setCentralWidget(central)

            # Change list dock.
            dock_w = QtWidgets.QWidget()
            dl = QtWidgets.QVBoxLayout(dock_w)
            frow = QtWidgets.QHBoxLayout()
            self.group_box = QtWidgets.QComboBox()
            self.group_box.addItems(["all groups"] + list(cd.groups))
            self.fish_box = QtWidgets.QComboBox()
            self.fish_box.addItems(["all fish"] + cd.labels)
            self.kind_box = QtWidgets.QComboBox()
            self.kind_box.addItems(["all kinds"] + sorted(cd.changes["kind"].unique().tolist()))
            self.other_box = QtWidgets.QComboBox()       # recon: which version against the reference
            others = sorted(cd.changes["other"].unique().tolist()) if "other" in cd.changes else []
            self.other_box.addItems(["all versions"] + others)
            self.other_box.setVisible(recon and len(others) > 1)
            self.fast_box = QtWidgets.QCheckBox(f"fast > {FAST_MM_S:.0f} mm/s")
            self.after_start = QtWidgets.QCheckBox("after analysis start")
            self.after_start.setChecked(cd.analysis_start is not None)
            self.after_start.setEnabled(cd.analysis_start is not None)
            self.min_size = QtWidgets.QDoubleSpinBox()
            self.min_size.setRange(0, 100)
            self.min_size.setSuffix(" mm min")
            for wdg in (self.group_box, self.fish_box, self.kind_box, self.other_box):
                wdg.currentIndexChanged.connect(self.apply_filter)
                frow.addWidget(wdg)
            self.fast_box.toggled.connect(self.apply_filter)
            self.after_start.toggled.connect(self.apply_filter)
            self.min_size.valueChanged.connect(self.apply_filter)
            dl.addLayout(frow)
            r2 = QtWidgets.QHBoxLayout()
            r2.addWidget(self.fast_box)
            r2.addWidget(self.after_start)
            r2.addWidget(self.min_size)
            dl.addLayout(r2)
            self.table = QtWidgets.QTableWidget(0, len(cd.table_columns))
            self.table.setHorizontalHeaderLabels(cd.table_columns)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.verticalHeader().setVisible(False)
            self.table.cellDoubleClicked.connect(lambda row, _c: self.goto_row(row))
            dl.addWidget(self.table, 1)
            for wdg in (self.group_box, self.fish_box, self.kind_box, self.other_box, self.fast_box,
                        self.after_start, self.table):
                wdg.setFocusPolicy(QtCore.Qt.NoFocus)          # keys stay with the window
            # An input box with the focus swallows every shortcut: only a click focuses it, and Enter or
            # leaving it hands the keys back to the window.
            self.min_size.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.min_size.editingFinished.connect(self.setFocus)
            self.count_label = QtWidgets.QLabel()
            dl.addWidget(self.count_label)
            dock = QtWidgets.QDockWidget("Disagreements" if recon else "Changes", self)
            dock.setWidget(dock_w)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(int(1000 / CLIP_FPS))
            for lab in self.labels_v:
                lab.double_clicked.connect(lambda _i: self.toggle_zoom())
                lab.clicked.connect(self.on_view_click)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self._actions = self.key_actions()
            for key, fn in self._actions.items():
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
                sc.setContext(QtCore.Qt.ApplicationShortcut)
                sc.activated.connect(fn)
                sc.activatedAmbiguously.connect(fn)
            self.set_pair(*self.pair)
            self.apply_filter()
            if recon:
                self.statusBar().showMessage(
                    f"{cd.naming_text()}  |  name: 1/2/3 or click a fish, then 4/5/6 = F/M1/M2  |  "
                    "verdict on the selected disagreement: A / B right, W both wrong, O both OK")

        # -- state -----------------------------------------------------------
        def set_pair(self, a: str, b: str):
            for box, v in ((self.a_box, a), (self.b_box, b)):
                box.blockSignals(True)
                box.setCurrentIndex(list(cd.versions).index(v))
                box.blockSignals(False)
            self.on_pair()

        def on_pair(self, *_):
            self.pair = (self.a_box.currentData(), self.b_box.currentData())
            self._proj_cache.clear()
            self.render()

        def on_aligned(self, on: bool):
            self.aligned = bool(on)
            self._proj_cache.clear()
            self.render()

        def apply_filter(self, *_):
            ch = cd.changes
            if self.group_box.currentIndex() > 0:
                ch = ch[ch["group"] == self.group_box.currentText()]
            if self.fish_box.currentIndex() > 0:
                ch = ch[ch["fish"].str.contains(self.fish_box.currentText(), regex=False)]
            if self.kind_box.currentIndex() > 0:
                ch = ch[ch["kind"] == self.kind_box.currentText()]
            if self.other_box.currentIndex() > 0:
                ch = ch[ch["other"] == self.other_box.currentText()]
            if self.after_start.isChecked() and cd.analysis_start is not None:
                ch = ch[ch["end"] >= cd.analysis_start]
            if self.fast_box.isChecked():
                ch = ch[ch["speed_mm_s"] > FAST_MM_S]
            if self.min_size.value() > 0:
                ch = ch[ch["size_mm"].fillna(np.inf) >= self.min_size.value()]
            self.filtered = ch.reset_index(drop=True)
            shown = self.filtered.head(5000)
            self.table.setUpdatesEnabled(False)
            self.table.setRowCount(len(shown))
            for i, r in enumerate(shown.itertuples(index=False)):
                vals = [r.start, r.end, r.fish, r.group, r.kind,
                        "" if not np.isfinite(r.size_mm) else f"{r.size_mm:.1f}",
                        "" if not np.isfinite(r.speed_mm_s) else f"{r.speed_mm_s:.0f}"]
                if recon:
                    vals.append(VERDICTS.get(r.verdict, ""))
                for j, val in enumerate(vals):
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(val)))
            self.table.setUpdatesEnabled(True)
            more = " (first 5000 listed)" if len(self.filtered) > 5000 else ""
            text = f"{len(self.filtered)} of {len(cd.changes)} {'disagreements' if recon else 'changes'}{more}"
            if recon:
                counts = cd.verdict_counts()
                text += "  |  judged: " + (", ".join(f"{VERDICTS[k]} {counts[k]}" for k in VERDICTS if k in counts)
                                           or "none")
            self.count_label.setText(text)
            self.overview.update()
            self.traces.update()

        def goto_row(self, row: int):
            if 0 <= row < len(self.filtered):
                r = self.filtered.iloc[row]
                if recon and r["other"] not in self.pair:
                    self.set_pair(cd.versions[0], r["other"])     # show the version this entry is about
                self.seek(int(r["start"]) - int(0.1 * cd.fps))
                self.table.selectRow(row)

        def next_change(self, direction: int):
            st = self.filtered["start"].to_numpy()
            if not len(st):
                return
            rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
            cur = rows[0].row() if rows else None
            if cur is not None and cur < len(st) and abs(self.frame - (st[cur] - int(0.1 * cd.fps))) <= 1:
                row = cur + direction                      # step through the list, entries sharing a start too
                if 0 <= row < len(st):
                    self.goto_row(row)
                return
            if direction > 0:
                idx = np.flatnonzero(st > self.frame + int(0.1 * cd.fps))
                row = int(idx[0]) if len(idx) else None
            else:
                idx = np.flatnonzero(st < self.frame + int(0.1 * cd.fps) - 1)
                row = int(idx[-1]) if len(idx) else None
            if row is not None:
                self.goto_row(row)

        def seek(self, frame: int):
            self.frame = int(np.clip(frame, cd.first, cd.last))
            self.render()

        def goto_dialog(self):
            f, ok = QtWidgets.QInputDialog.getInt(self, "Go to frame", f"frame ({cd.first}-{cd.last}):",
                                                  self.frame, cd.first, cd.last)
            if ok:
                self.seek(int(f))
            self.setFocus()

        def showEvent(self, e):
            super().showEvent(e)
            self.setFocus()                      # the window, not an input box, gets the keys at start

        def toggle_zoom(self):
            self.zoom_on = not self.zoom_on
            self.crops = [None, None, None]
            self.render()

        # -- playback ----------------------------------------------------------
        def tick(self):
            if not self.playing:
                return
            self._acc += SPEEDS[self.speed_i] * cd.fps / CLIP_FPS
            n = int(self._acc)
            if n >= cd.step:                        # fast enough for the playback proxy: whole steps
                n -= n % cd.step
            if n <= 0:
                return
            self._acc -= n
            if self.frame + n > cd.last:
                self.playing = False
                self.play_btn.setText("> Play (Space)")
            self.seek(self.frame + n)

        # -- drawing -----------------------------------------------------------
        def render(self):
            t = cd.t(self.frame)
            proj = None
            if t is not None:
                key = (t, self.pair, self.aligned)
                proj = self._proj_cache.get(key)
                if proj is None:
                    proj = frame_projections(cd, t, self.pair, self.aligned)
                    if len(self._proj_cache) > 64:
                        self._proj_cache.clear()
                    self._proj_cache[key] = proj
            for v, lab in enumerate(self.labels_v):
                crop = None
                if self.zoom_on and t is not None:
                    pts = np.concatenate([proj[0][:, :, v], proj[1][:, :, v]]).reshape(-1, 2) * cd.scale[v]
                    meta = cd.proxy["views"][VIEWS[v]]
                    target = crop_target(pts, meta["proxy_width"], meta["proxy_height"])
                    self.crops[v] = target if self.crops[v] is None else self.crops[v] + 0.35 * (target - self.crops[v])
                    crop = self.crops[v]
                img, tf = render_view(cd, v, self.frame, self.pair, self.aligned,
                                      (max(50, lab.width()), max(50, lab.height())), crop, projections=proj,
                                      selected=self.selected)
                self._tf[v] = tf
                lab.show_image(img)
            self.view3d.update()
            self.traces.update()
            self.overview.update()
            self.update_info(t)

        def update_info(self, t):
            secs = self.frame / cd.fps
            parts = [f"<b>frame {self.frame}</b> {int(secs // 60):02d}:{secs % 60:05.2f}",
                     f"{'playing' if self.playing else 'paused'} {SPEEDS[self.speed_i]:g}x"]
            if t is not None:
                a = cd.series(self.pair[0], t, t + 1, self.aligned)[0]
                b = cd.series(self.pair[1], t, t + 1, self.aligned)[0]
                for c, name in enumerate(cd.labels):
                    with np.errstate(invalid="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        d = np.nanmax(np.linalg.norm(a[c] - b[c], axis=-1)) if np.isfinite(a[c]).any() else np.nan
                    src_a, src_b = (cd.sources(v, t, self.aligned)[c] if not self.aligned or v in cd.reviewed else None
                                    for v in self.pair)
                    src = ""
                    if src_b is not None:
                        src = f" src B {''.join(str(int(s)) if s >= 0 else '-' for s in src_b)}"
                    if src_a is not None:
                        src += f" A {''.join(str(int(s)) if s >= 0 else '-' for s in src_a)}"
                    sel = "&#9654;" if c == self.selected else ""
                    parts.append(f"{sel}{name}: |A-B| {d:.1f} mm{src}" if np.isfinite(d) else f"{sel}{name}: -")
            parts.append(f"dashed = A {self.pair[0]}, solid = B {self.pair[1]}, "
                         "<span style='color:#3cd73c'><b>green = both on the same spot (green = both on the same spotlt; 4 px)</b></span>")
            if recon:
                parts.append("source codes per node: 0 tri, 1 one view, 2 body model, -1 missing")
            else:
                parts.append("source codes per node: 0 tri, 1 one view, 2 body model, 3 despiked, 4 B, 5 bridged")
            self.info.setText(" &nbsp;|&nbsp; ".join(parts))

        # -- reconstruction mode: names and verdicts -------------------------------
        def on_view_click(self, v: int, x: int, y: int):
            """Select the fish nearest to the click (the A version's position node)."""
            t = cd.t(self.frame)
            if t is None or v not in self._tf:
                return
            ox, oy = self.labels_v[v].image_offset()
            a, bx, by = self._tf[v]
            src = np.array([(x - ox - bx) / a, (y - oy - by) / a])
            pa, _ = frame_projections(cd, t, self.pair, self.aligned)
            d = np.linalg.norm(pa[:, cd.pos_node, v] - src, axis=-1)
            d = np.where(np.isfinite(d), d, np.inf)
            c = int(np.argmin(d)) if len(d) else -1
            self.setFocus()
            self.select(c if 0 <= c and d[c] <= SELECT_RADIUS_SRC_PX else None)

        def select(self, c: int | None):
            self.selected = c
            self.render()

        def name_selected(self, label: str):
            if self.selected is None:
                self.statusBar().showMessage("select a fish first (1/2/3 or click)")
                return
            msg = cd.set_name(self.selected, label, self.frame)
            self.apply_filter()
            self.fish_box.blockSignals(True)
            self.fish_box.clear()
            self.fish_box.addItems(["all fish"] + cd.labels)
            self.fish_box.blockSignals(False)
            self._proj_cache.clear()
            self.render()
            self.statusBar().showMessage(msg)

        def current_row(self) -> int | None:
            """The selected disagreement, else the first one around the current frame."""
            rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
            if rows and rows[0].row() < len(self.filtered):
                return rows[0].row()
            inside = np.flatnonzero((self.filtered["start"].to_numpy() <= self.frame)
                                    & (self.filtered["end"].to_numpy() >= self.frame))
            return int(inside[0]) if len(inside) else None

        def judge(self, verdict: str):
            row = self.current_row()
            if row is None:
                self.statusBar().showMessage("no disagreement selected or at this frame")
                return
            r = self.filtered.iloc[row]
            match = cd.changes.index[(cd.changes["start"] == r["start"]) & (cd.changes["end"] == r["end"])
                                     & (cd.changes["ref_fish"] == r["ref_fish"]) & (cd.changes["kind"] == r["kind"])
                                     & (cd.changes["other"] == r["other"])]
            if not len(match):
                return
            cd.set_verdict(int(match[0]), verdict)
            self.apply_filter()
            what = VERDICTS.get(verdict, "cleared")
            self.statusBar().showMessage(f"{r['kind']} {r['fish']} {r['start']}-{r['end']}: {what}")
            if verdict and row + 1 < len(self.filtered):
                self.goto_row(row + 1)
            else:
                self.table.selectRow(row)

        # -- export ------------------------------------------------------------
        def export_dialog(self):
            lo, hi = self.traces.span()
            lo, hi = max(lo, cd.first), min(hi, cd.last)
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Export clip", "start frame, end frame, speed (x real time):", text=f"{lo}, {hi}, 0.25")
            if not ok:
                return
            try:
                a, b, s = (x.strip() for x in text.split(","))
                start, end, speed = int(a), int(b), float(s)
            except ValueError:
                self.statusBar().showMessage("Give: start, end, speed - e.g. 105000, 105400, 0.25")
                return
            dlg = QtWidgets.QProgressDialog("Writing clip ...", "Stop", 0, 100, self)
            dlg.setWindowModality(QtCore.Qt.WindowModal)

            def progress(i, n):
                dlg.setMaximum(n)
                dlg.setValue(i)
                app.processEvents()
                return not dlg.wasCanceled()

            out = export_clip(cd, start, end, self.pair, aligned=self.aligned, speed=speed, zoom=self.zoom_on,
                              progress=progress)
            dlg.close()
            self.statusBar().showMessage(f"Clip written: {out}")

        # -- keys ----------------------------------------------------------------
        # -- keys ----------------------------------------------------------------
        def toggle_play(self):
            self.playing = not self.playing
            self._acc = 0.0
            self.play_btn.setText("|| Pause (Space)" if self.playing else "> Play (Space)")
            self.update_info(cd.t(self.frame))

        def step(self, d: int):
            self.playing = False
            self.play_btn.setText("> Play (Space)")
            self.seek(self.frame + d)

        def set_speed(self, i: int):
            self.speed_i = int(np.clip(i, 0, len(SPEEDS) - 1))
            self.speed_box.blockSignals(True)
            self.speed_box.setCurrentIndex(self.speed_i)
            self.speed_box.blockSignals(False)
            self.update_info(cd.t(self.frame))

        def preset(self, i: int):
            if i < len(cd.presets):
                _, a, b = cd.presets[i]
                if a in cd.present and b in cd.present:
                    self.set_pair(a, b)

        def key_actions(self) -> dict:
            """Key sequence -> action. Registered as application-wide shortcuts, so they work whatever
            widget (change list, combo box) has the keyboard focus."""
            fps = int(cd.fps)
            keys = {
                "Space": self.toggle_play, "Right": lambda: self.step(1), "Left": lambda: self.step(-1),
                "Shift+Right": lambda: self.step(fps), "Shift+Left": lambda: self.step(-fps),
                "Up": lambda: self.set_speed(self.speed_i + 1), "Down": lambda: self.set_speed(self.speed_i - 1),
                "N": lambda: self.next_change(1), "P": lambda: self.next_change(-1),
                "I": self.same_id.toggle, "Z": self.toggle_zoom, "E": self.export_dialog, "G": self.goto_dialog,
                "F1": lambda: QtWidgets.QMessageBox.information(self, "Keys", __doc__.split("Keys:")[1].strip()),
            }
            if recon:
                for i in range(3):
                    keys[str(i + 1)] = lambda c=i: self.select(None if self.selected == c or c >= len(cd.labels)
                                                               else c)
                    keys[str(i + 4)] = lambda lab=NAME_KEYS[i]: self.name_selected(lab)
                    keys[str(i + 7)] = lambda j=i: self.preset(j)
                for key, verdict in (("A", "a_right"), ("B", "b_right"), ("W", "both_wrong"), ("O", "both_ok"),
                                     ("Backspace", "")):
                    keys[key] = lambda v=verdict: self.judge(v)
            else:
                for i in range(4):
                    keys[str(i + 1)] = lambda j=i: self.preset(j)
            return keys

        def press(self, key: str):
            """Run the action of ``key`` (as the shortcut would)."""
            self._actions[key]()

        def closeEvent(self, e):
            cd.close()
            super().closeEvent(e)

    win = Main()
    install_error_log((cd.recon_dir if recon else cd.folder) / ERROR_LOG, win)
    if smoke_test:
        from PyQt5 import QtTest

        win.resize(1400, 900)
        win.show()
        win.activateWindow()
        app.processEvents()
        # Keys pressed right after opening go to whatever has the focus; an input box there would swallow
        # every shortcut (arrows, Space, letters, digits).
        target = app.focusWidget() or win
        if isinstance(target, (QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit)):
            raise RuntimeError(f"keyboard focus starts in a {type(target).__name__}")
        before = win.frame
        QtTest.QTest.keyClick(target, QtCore.Qt.Key_Right)
        app.processEvents()
        if win.frame != before + 1:
            raise RuntimeError(f"Right after opening moved {before} -> {win.frame}")
        keys = (("Right", "1", "5", "N", "A", "Backspace", "7", "I", "Z", "Space", "Space") if recon
                else ("Right", "N", "2", "I", "Z", "Space", "Space"))
        for key in keys:
            win.press(key)
            app.processEvents()
        # A real key press with the change list focused must still reach the window's shortcuts.
        from PyQt5 import QtTest

        before = win.frame
        win.table.setFocus()
        QtTest.QTest.keyClick(win.table, QtCore.Qt.Key_Right)
        app.processEvents()
        if win.frame != before + 1:
            raise RuntimeError(f"Right with the list focused moved {before} -> {win.frame}")
        win.tick()
        win.on_view_click(0, win.labels_v[0].width() // 2, win.labels_v[0].height() // 2)
        win.quantity.setCurrentIndex(1)
        win.group_box.setCurrentIndex(2)
        win.traces.repaint()
        win.view3d.repaint()
        win.close()
        return 0
    win.showMaximized()
    return app.exec_()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, help="<review_root>/<exp>/<round_tag>")
    p.add_argument("--mode", choices=("review", "recon"), default=None,
                   help="review: correction steps (versions.npz); recon: reconstructions (recon_vs_*). "
                        "Default: ask when the folder has both.")
    p.add_argument("--versions-dir", type=Path, default=None, help="Where versions.npz is (default: the folder).")
    p.add_argument("--recon-dir", type=Path, default=None,
                   help="A recon_vs_* folder (scripts.review.recon_compare); implies --mode recon.")
    p.add_argument("--clip", nargs=2, type=int, metavar=("START", "END"), help="Export a clip without the GUI.")
    p.add_argument("--a", default=None, help="Clip: version A (default: machine / the reference).")
    p.add_argument("--b", default=None, help="Clip: version B (default: final / the other reconstruction).")
    p.add_argument("--speed", type=float, default=0.25, help="Clip speed, x real time.")
    p.add_argument("--tracker-identity", action="store_true",
                   help="Clip: each version keeps its own identity (default: the same identity).")
    p.add_argument("--no-zoom", action="store_true", help="Clip: whole views instead of zooming onto the fish.")
    p.add_argument("--out", type=Path, default=None, help="Clip file (default: compare_clips/ of the mode).")
    p.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.clip:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cd = open_data(args.folder, mode=args.mode, versions_dir=args.versions_dir, recon_dir=args.recon_dir)
        pair = (args.a or cd.default_pair[0], args.b or cd.default_pair[1])
        unknown = [v for v in pair if v not in cd.present]
        if unknown:
            raise SystemExit(f"unknown version(s) {unknown}; this folder has {cd.present}")
        t0 = time.time()
        out = export_clip(cd, args.clip[0], args.clip[1], pair, aligned=not args.tracker_identity,
                          speed=args.speed, out=args.out, zoom=not args.no_zoom)
        cd.close()
        print(f"{out} ({time.time() - t0:.0f} s)")
        return 0
    return run_app(args.folder, versions_dir=args.versions_dir, smoke_test=args.smoke_test, mode=args.mode,
                   recon_dir=args.recon_dir)


if __name__ == "__main__":
    sys.exit(main())
