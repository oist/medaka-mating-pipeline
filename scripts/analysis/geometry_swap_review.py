"""Source-frame GUI for reviewing hidden swaps in anonymous 3D geometry tracks.

The GUI never edits the geometry CSV.  It records a small, resumable event
ledger with exact pair swaps, ambiguous intervals, explicit track breaks,
mapping anchors, per-pose node/link defects, review-progress checkpoints, and
reviewed-no-swap markers. Run with ``--build-candidates-only`` to create the
ranked event queue without opening Qt.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pandas as pd

from scripts.tracking.geometry_swap_annotations import (
    EVENT_COLUMNS,
    false_positive_slots_at_frame,
    format_mapping,
    identity_state_at_frame,
    infer_slot_map,
    mapping_at_frame,
    node_qc_events_at_frame,
    normalize_events,
    review_progress_checkpoint,
)


VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEO_NAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
HUMAN_COLORS = {
    "H1": (48, 48, 224),
    "H2": (180, 119, 31),
    "H3": (0, 149, 255),
}
GRAY = (145, 145, 145)
RAW_SLEAP_GRAY = (165, 165, 165)
REVIEW_SCOPES = (*VIEWS, "3d")
SCOPE_LABELS = {
    "top_cam": "top camera",
    "side1_cam": "side 1 camera",
    "side2_cam": "side 2 camera",
    "3d": "3D track",
}
NODE_QC_LABELS = {
    "bad_mouth": "bad mouth",
    "bad_tail": "bad tail",
    "wrong_link": "wrong mouth-tail link",
}


class IndexedSlpPredictions:
    """Read raw predicted instances for one source frame without loading a full SLP.

    SLEAP inference files keep a compact ``frames`` table that points to the
    corresponding rows in ``instances`` and ``pred_points``.  Keeping just that
    index in memory makes full-length review files quick to open, while a small
    LRU cache keeps frame stepping and short playback responsive.
    """

    def __init__(
        self,
        path: Path,
        *,
        node_names: tuple[str, ...] = ("mouth", "tail"),
        cache_size: int = 2048,
    ):
        import h5py

        self.path = Path(path)
        self.node_names = tuple(node_names)
        self.cache_size = max(1, int(cache_size))
        self._file = h5py.File(self.path, "r")
        required = {"frames", "instances", "pred_points", "metadata"}
        missing = required - set(self._file)
        if missing:
            self.close()
            raise ValueError(f"SLP {self.path} is missing datasets: {sorted(missing)}")

        metadata_raw = self._file["metadata"].attrs.get("json")
        if metadata_raw is None:
            self.close()
            raise ValueError(f"SLP {self.path} has no skeleton metadata")
        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")
        metadata = json.loads(str(metadata_raw))
        global_nodes = [str(node["name"]) for node in metadata.get("nodes", [])]
        self._node_indices_by_skeleton: dict[int, tuple[int, ...]] = {}
        for skeleton_index, skeleton in enumerate(metadata.get("skeletons", [])):
            skeleton_names = [
                global_nodes[int(node["id"])] for node in skeleton.get("nodes", [])
            ]
            missing_nodes = [name for name in self.node_names if name not in skeleton_names]
            if not missing_nodes:
                self._node_indices_by_skeleton[skeleton_index] = tuple(
                    skeleton_names.index(name) for name in self.node_names
                )
        if not self._node_indices_by_skeleton:
            self.close()
            raise ValueError(
                f"SLP {self.path} skeleton does not contain requested nodes {self.node_names}"
            )

        frames = self._file["frames"][:]
        order = np.argsort(frames["frame_idx"], kind="stable")
        self._frame_indices = frames["frame_idx"][order].astype(np.int64, copy=False)
        self._instance_starts = frames["instance_id_start"][order].astype(
            np.int64, copy=False
        )
        self._instance_ends = frames["instance_id_end"][order].astype(
            np.int64, copy=False
        )
        self._instances = self._file["instances"]
        self._pred_points = self._file["pred_points"]
        self._cache: OrderedDict[int, np.ndarray | None] = OrderedDict()

    def get(self, frame: int, default=None):
        frame = int(frame)
        if frame in self._cache:
            value = self._cache.pop(frame)
            self._cache[frame] = value
            return default if value is None else value

        index = int(np.searchsorted(self._frame_indices, frame))
        if index >= len(self._frame_indices) or self._frame_indices[index] != frame:
            result = None
        else:
            rows = self._instances[
                int(self._instance_starts[index]) : int(self._instance_ends[index])
            ]
            rows = rows[rows["instance_type"] == 1]
            supported = np.array(
                [int(row["skeleton"]) in self._node_indices_by_skeleton for row in rows],
                dtype=bool,
            )
            rows = rows[supported]
            if len(rows) == 0:
                result = None
            else:
                point_start = int(rows["point_id_start"].min())
                point_end = int(rows["point_id_end"].max())
                points = self._pred_points[point_start:point_end]
                instances: list[np.ndarray] = []
                for row in rows:
                    start = int(row["point_id_start"]) - point_start
                    end = int(row["point_id_end"]) - point_start
                    row_points = points[start:end]
                    selected = self._node_indices_by_skeleton[int(row["skeleton"])]
                    xy = np.full((len(selected), 2), np.nan, dtype=float)
                    for output_index, node_index in enumerate(selected):
                        if node_index < len(row_points) and bool(row_points[node_index]["visible"]):
                            xy[output_index] = (
                                float(row_points[node_index]["x"]),
                                float(row_points[node_index]["y"]),
                            )
                    instances.append(xy)
                result = np.stack(instances)

        self._cache[frame] = result
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return default if result is None else result

    def close(self) -> None:
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()
            self._file = None


def _truthy(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna(False).astype(str).str.lower().isin({"1", "true", "yes", "y"})


def parse_slot_specs(specs: list[str]) -> dict[str, str] | None:
    if not specs:
        return None
    mapping: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--slot-map expects SOURCE=SLOT, got {spec!r}")
        source, slot = (part.strip() for part in spec.split("=", 1))
        if not source or not slot:
            raise ValueError(f"--slot-map expects SOURCE=SLOT, got {spec!r}")
        mapping[source] = slot
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("slot display names must be unique")
    return mapping


def load_extra_candidate_table(path: Path) -> pd.DataFrame:
    """Load candidate frames from CSV or a motion-identity QC JSON report."""
    if path.suffix.lower() != ".json":
        return pd.read_csv(path, low_memory=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames: set[int] = set()
    after = payload.get("speed_qc", {}).get("after", {})
    for fish in after.get("per_id", {}).values():
        frames.update(int(frame) for frame in fish.get("jump_frames", []))
    if not frames:
        def collect(value, key: str = ""):
            if isinstance(value, dict):
                for child_key, child in value.items():
                    collect(child, child_key)
            elif isinstance(value, list) and key in {"jump_frames", "change_frames"}:
                for frame in value:
                    if isinstance(frame, (int, float)) and np.isfinite(frame):
                        frames.add(int(frame))
        collect(payload)
    if not frames:
        raise ValueError(f"JSON candidate report {path} contains no jump frames")
    return pd.DataFrame({"frame": sorted(frames)})


def load_geometry_table(
    path: Path,
    *,
    id_column: str = "fish_id",
    slot_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    header = pd.read_csv(path, nrows=0).columns
    required = {"frame", id_column, "X", "Y", "Z"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"geometry CSV is missing columns: {sorted(missing)}")
    optional = ["is_interpolated"] if "is_interpolated" in header else []
    table = pd.read_csv(
        path,
        usecols=["frame", id_column, "X", "Y", "Z", *optional],
        low_memory=False,
    )
    table["frame"] = pd.to_numeric(table["frame"], errors="raise").astype(np.int64)
    table["geometry_source_id"] = table[id_column].astype(str)
    resolved = dict(slot_map or infer_slot_map(table["geometry_source_id"]))
    table["geometry_slot"] = table["geometry_source_id"].map(resolved)
    if table["geometry_slot"].isna().any():
        unknown = sorted(table.loc[table["geometry_slot"].isna(), "geometry_source_id"].unique())
        raise ValueError(f"slot map is missing source IDs: {unknown}")
    if "is_interpolated" not in table:
        table["is_interpolated"] = False
    else:
        table["is_interpolated"] = _truthy(table["is_interpolated"])
    table.sort_values(["frame", "geometry_slot"], kind="stable", inplace=True)
    return table.reset_index(drop=True), resolved


def _cluster_true_frames(frames: np.ndarray) -> list[np.ndarray]:
    if len(frames) == 0:
        return []
    split = np.where(np.diff(frames) > 1)[0] + 1
    return [part for part in np.split(frames, split) if len(part)]


def build_review_candidates(
    geometry: pd.DataFrame,
    *,
    close_mm: float = 18.0,
    jump_mm: float = 8.0,
    merge_frames: int = 166,
    min_interpolation_frames: int = 30,
    extra_candidate_tables: list[tuple[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Build a compact, source-frame review queue from geometry risk signals."""
    required = {"frame", "geometry_slot", "X", "Y", "Z", "is_interpolated"}
    missing = required - set(geometry)
    if missing:
        raise ValueError(f"geometry candidate table is missing columns: {sorted(missing)}")
    slots = sorted(geometry["geometry_slot"].astype(str).unique())
    pivot = geometry.pivot(index="frame", columns="geometry_slot", values=["X", "Y", "Z"])
    rows: list[dict[str, object]] = []

    for left_index, left in enumerate(slots):
        for right in slots[left_index + 1 :]:
            if any((axis, left) not in pivot or (axis, right) not in pivot for axis in "XYZ"):
                continue
            a = np.stack([pivot[(axis, left)].to_numpy(float) for axis in "XYZ"], axis=1)
            b = np.stack([pivot[(axis, right)].to_numpy(float) for axis in "XYZ"], axis=1)
            distance = np.linalg.norm(a - b, axis=1)
            close_frames = pivot.index.to_numpy(dtype=np.int64)[
                np.isfinite(distance) & (distance <= close_mm)
            ]
            frame_to_distance = dict(zip(pivot.index.to_numpy(dtype=np.int64), distance, strict=True))
            for run in _cluster_true_frames(close_frames):
                center = int(min(run, key=lambda frame: frame_to_distance[int(frame)]))
                rows.append(
                    {
                        "frame": center,
                        "source": "close_approach",
                        "slots": f"{left}<->{right}",
                        "score": float(frame_to_distance[center]),
                        "details": f"minimum distance {frame_to_distance[center]:.3f} mm",
                    }
                )

    for slot, group in geometry.groupby("geometry_slot", sort=True, observed=True):
        group = group.sort_values("frame", kind="stable")
        frames = group["frame"].to_numpy(dtype=np.int64)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        delta = np.vstack([np.full((1, 3), np.nan), np.diff(xyz, axis=0)])
        step = np.linalg.norm(delta, axis=1)
        valid = np.r_[False, np.diff(frames) == 1] & np.isfinite(step)
        for index in np.where(valid & (step > jump_mm))[0]:
            rows.append(
                {
                    "frame": int(frames[index]),
                    "source": "geometry_jump",
                    "slots": str(slot),
                    "score": float(step[index]),
                    "details": f"native step {step[index]:.3f} mm",
                }
            )
        interpolation = group["is_interpolated"].astype(bool).to_numpy()
        true_frames = frames[interpolation]
        for run in _cluster_true_frames(true_frames):
            duration = int(run[-1] - run[0] + 1)
            if duration < min_interpolation_frames:
                continue
            rows.append(
                {
                    "frame": int(run[0]),
                    "source": "interpolation_boundary",
                    "slots": str(slot),
                    "score": float(duration),
                    "details": f"interpolation starts; gap length {duration} frames",
                }
            )
            resume = int(run[-1] + 1)
            if resume <= int(frames[-1]):
                rows.append(
                    {
                        "frame": resume,
                        "source": "interpolation_boundary",
                        "slots": str(slot),
                        "score": float(duration),
                        "details": f"measured data resumes after {duration}-frame gap",
                    }
                )

    for source_name, extra in extra_candidate_tables or []:
        frame_column = next(
            (name for name in ("frame", "center_frame", "center", "jump_frame") if name in extra),
            None,
        )
        if frame_column is None:
            raise ValueError(f"extra candidate table {source_name!r} has no frame column")
        for frame in pd.to_numeric(extra[frame_column], errors="coerce").dropna().astype(int).unique():
            rows.append(
                {
                    "frame": int(frame),
                    "source": f"external:{source_name}",
                    "slots": "",
                    "score": np.nan,
                    "details": "imported candidate",
                }
            )

    columns = ["candidate_id", "frame", "source", "slots", "score", "details"]
    if not rows:
        return pd.DataFrame(columns=columns)
    raw = pd.DataFrame(rows).sort_values("frame", kind="stable").reset_index(drop=True)
    clusters: list[list[int]] = []
    for index, frame in enumerate(raw["frame"].astype(int)):
        if not clusters or frame - int(raw.loc[clusters[-1][-1], "frame"]) > merge_frames:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    merged = []
    for indices in clusters:
        group = raw.loc[indices]
        close = group[group["source"].eq("close_approach") & group["score"].notna()]
        representative = (
            close.loc[close["score"].idxmin()] if not close.empty else group.iloc[len(group) // 2]
        )
        merged.append(
            {
                "frame": int(representative["frame"]),
                "source": ";".join(sorted(set(group["source"].astype(str)))),
                "slots": ";".join(sorted({value for value in group["slots"].astype(str) if value})),
                "score": representative["score"],
                "details": " | ".join(dict.fromkeys(group["details"].astype(str))),
            }
        )
    result = pd.DataFrame(merged).sort_values("frame", kind="stable").reset_index(drop=True)
    result.insert(0, "candidate_id", [f"C{index + 1:06d}" for index in range(len(result))])
    return result[columns]


def atomic_write_events(events: pd.DataFrame, output: Path, slots: list[str]) -> None:
    normalized = normalize_events(events, slots)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    normalized.to_csv(temporary, index=False)
    temporary.replace(output)


def append_audit(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def draw_raw_sleap_gray(image: np.ndarray, instances: np.ndarray | None, scale: float) -> None:
    """Draw every raw mouth/tail prediction in neutral gray."""
    if instances is None:
        return
    array = np.asarray(instances, dtype=float)
    for instance in array:
        visible = np.isfinite(instance).all(axis=1)
        if len(instance) >= 2 and visible[0] and visible[1]:
            mouth = tuple(np.round(instance[0] * scale).astype(int))
            tail = tuple(np.round(instance[1] * scale).astype(int))
            cv2.arrowedLine(
                image, tail, mouth, RAW_SLEAP_GRAY, 1, cv2.LINE_AA, 0, 0.28
            )
        if len(instance) >= 1 and visible[0]:
            mouth = tuple(np.round(instance[0] * scale).astype(int))
            cv2.circle(image, mouth, 2, RAW_SLEAP_GRAY, -1, cv2.LINE_AA)
        if len(instance) >= 2 and visible[1]:
            tail = tuple(np.round(instance[1] * scale).astype(int))
            cv2.circle(image, tail, 2, RAW_SLEAP_GRAY, 1, cv2.LINE_AA)


@dataclass
class ReviewData:
    geometry: pd.DataFrame
    slot_map: dict[str, str]
    positions: dict[str, np.ndarray]
    interpolated: dict[str, np.ndarray]
    n_frames: int
    candidates: pd.DataFrame
    events: pd.DataFrame


def prepare_review_data(
    geometry: pd.DataFrame,
    slot_map: dict[str, str],
    candidates: pd.DataFrame,
    events: pd.DataFrame,
) -> ReviewData:
    slots = sorted(slot_map.values())
    n_frames = int(geometry["frame"].max()) + 1
    positions = {slot: np.full((n_frames, 3), np.nan) for slot in slots}
    interpolated = {slot: np.zeros(n_frames, dtype=bool) for slot in slots}
    for slot in slots:
        rows = geometry[geometry["geometry_slot"].eq(slot)]
        frames = rows["frame"].to_numpy(dtype=np.int64)
        positions[slot][frames] = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
        interpolated[slot][frames] = rows["is_interpolated"].astype(bool).to_numpy()
    return ReviewData(
        geometry=geometry,
        slot_map=slot_map,
        positions=positions,
        interpolated=interpolated,
        n_frames=n_frames,
        candidates=candidates,
        events=normalize_events(events, slots),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_gui(
    args: argparse.Namespace,
    data: ReviewData,
    slp_predictions: Mapping[str, object],
) -> int:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise SystemExit("PyQt5 is required; install with `uv sync --extra gui`.") from error

    from scripts.reconstruction_3d.calibration import Calibration
    from scripts.visualization.render_qc_4panel_fast import (
        Projector3D,
        match_projected_identities_to_instances,
    )

    calibration = Calibration(str(args.calibration_model_dir))
    slots = sorted(data.positions)
    caps = {
        view: cv2.VideoCapture(str(args.video_dir / view / VIDEO_NAMES[view]))
        for view in VIEWS
    }
    unopened = [view for view, cap in caps.items() if not cap.isOpened()]
    if unopened:
        raise SystemExit(f"could not open videos for: {', '.join(unopened)}")
    panel = int(args.panel)
    projector = Projector3D(panel, args.xyz_lim[0], args.xyz_lim[1], args.elev, args.azim)
    base_3d = projector.base_canvas()

    class ReviewWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Geometry swap review — {args.experiment}")
            self.resize(1580, 960)
            self.events = data.events.copy()
            self.history: list[str] = []
            self.pending_ambiguous_start: tuple[str, int] | None = None
            self.pending_fp_start: tuple[str, str, int] | None = None
            self.pending_node_qc_start: tuple[str, str, str, int] | None = None
            self.current_candidate = 0
            self.last_video_frame = {view: -100 for view in VIEWS}
            self.zoom_factor = 1.0
            self.base_pixmap = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._advance_playback)
            self._build()
            self._refresh_event_list()
            if len(data.candidates):
                self._go_candidate(0)
            else:
                self._set_frame(0)

        def _build(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            left = QtWidgets.QVBoxLayout()
            root.addLayout(left, 4)
            self.image = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            self.image.setStyleSheet("background:#202124")
            self.image.resize(panel * 2, panel * 2)
            self.image_scroll = QtWidgets.QScrollArea()
            self.image_scroll.setAlignment(QtCore.Qt.AlignCenter)
            self.image_scroll.setWidgetResizable(False)
            self.image_scroll.setWidget(self.image)
            self.image_scroll.viewport().installEventFilter(self)
            self.image.installEventFilter(self)
            left.addWidget(self.image_scroll, 1)
            zoom_row = QtWidgets.QHBoxLayout()
            left.addLayout(zoom_row)
            zoom_out = QtWidgets.QPushButton("Zoom −")
            zoom_fit = QtWidgets.QPushButton("Fit")
            zoom_in = QtWidgets.QPushButton("Zoom +")
            zoom_out.clicked.connect(lambda: self._zoom(1 / 1.25))
            zoom_fit.clicked.connect(self._fit_zoom)
            zoom_in.clicked.connect(lambda: self._zoom(1.25))
            zoom_row.addWidget(zoom_out)
            zoom_row.addWidget(zoom_fit)
            zoom_row.addWidget(zoom_in)
            self.zoom_label = QtWidgets.QLabel("100%  (Ctrl+wheel)")
            zoom_row.addWidget(self.zoom_label)
            self.camera_overlays = QtWidgets.QCheckBox("Camera overlays [U]")
            self.camera_overlays.setChecked(True)
            self.camera_overlays.toggled.connect(self._overlay_visibility_changed)
            zoom_row.addWidget(self.camera_overlays)
            zoom_row.addStretch(1)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, data.n_frames - 1)
            self.slider.valueChanged.connect(self._render)
            left.addWidget(self.slider)
            timeline = QtWidgets.QHBoxLayout()
            left.addLayout(timeline)
            self.play = QtWidgets.QPushButton("Play")
            self.play.clicked.connect(self._toggle_play)
            timeline.addWidget(self.play)
            timeline.addWidget(QtWidgets.QLabel("Playback step:"))
            self.play_step = QtWidgets.QComboBox()
            self.play_step.addItems(["1", "2", "3", "5", "10"])
            self.play_step.setCurrentText("3")
            timeline.addWidget(self.play_step)
            self.frame_spin = QtWidgets.QSpinBox()
            self.frame_spin.setRange(0, data.n_frames - 1)
            self.frame_spin.setKeyboardTracking(False)
            self.frame_spin.valueChanged.connect(self._set_frame)
            timeline.addWidget(QtWidgets.QLabel("Source frame:"))
            timeline.addWidget(self.frame_spin)
            self.frame_label = QtWidgets.QLabel()
            timeline.addWidget(self.frame_label)
            timeline.addStretch(1)

            right = QtWidgets.QVBoxLayout()
            root.addLayout(right, 1)
            title = QtWidgets.QLabel("Review candidates")
            title.setStyleSheet("font-weight:bold;font-size:15px")
            right.addWidget(title)
            nav = QtWidgets.QHBoxLayout()
            right.addLayout(nav)
            prev_button = QtWidgets.QPushButton("Prev")
            next_button = QtWidgets.QPushButton("Next")
            prev_button.clicked.connect(lambda: self._go_candidate(self.current_candidate - 1))
            next_button.clicked.connect(lambda: self._go_candidate(self.current_candidate + 1))
            nav.addWidget(prev_button)
            nav.addWidget(next_button)
            self.candidate_combo = QtWidgets.QComboBox()
            self.candidate_combo.currentIndexChanged.connect(self._go_candidate)
            right.addWidget(self.candidate_combo)
            self.candidate_info = QtWidgets.QLabel()
            self.candidate_info.setWordWrap(True)
            right.addWidget(self.candidate_info)
            self.mapping_label = QtWidgets.QLabel()
            self.mapping_label.setWordWrap(True)
            self.mapping_label.setStyleSheet("font-weight:bold")
            right.addWidget(self.mapping_label)

            right.addWidget(self._line())
            scope_title = QtWidgets.QLabel("Annotation target (independent mappings):")
            scope_title.setStyleSheet("font-weight:bold")
            right.addWidget(scope_title)
            self.scope_combo = QtWidgets.QComboBox()
            for scope in REVIEW_SCOPES:
                self.scope_combo.addItem(SCOPE_LABELS[scope], scope)
            self.scope_combo.setCurrentIndex(self.scope_combo.findData("3d"))
            self.scope_combo.currentIndexChanged.connect(self._scope_changed)
            right.addWidget(self.scope_combo)
            scope_hint = QtWidgets.QLabel(
                "Click a video/3D panel to select it. Q/W/E changes only the selected panel."
            )
            scope_hint.setWordWrap(True)
            right.addWidget(scope_hint)

            right.addWidget(QtWidgets.QLabel("Exact swap in selected panel:"))
            pairs = [("G1", "G2", "Q"), ("G1", "G3", "W"), ("G2", "G3", "E")]
            for left_slot, right_slot, key in pairs:
                if left_slot not in slots or right_slot not in slots:
                    continue
                button = QtWidgets.QPushButton(f"{left_slot} ↔ {right_slot}   [{key}]")
                button.clicked.connect(
                    lambda _checked=False, a=left_slot, b=right_slot: self._add_swap(a, b)
                )
                right.addWidget(button)

            ambiguity_row = QtWidgets.QHBoxLayout()
            right.addLayout(ambiguity_row)
            start_button = QtWidgets.QPushButton("Ambiguous start [I]")
            end_button = QtWidgets.QPushButton("Ambiguous end [O]")
            start_button.clicked.connect(self._start_ambiguous)
            end_button.clicked.connect(self._end_ambiguous)
            ambiguity_row.addWidget(start_button)
            ambiguity_row.addWidget(end_button)
            fp_row = QtWidgets.QHBoxLayout()
            right.addLayout(fp_row)
            fp_row.addWidget(QtWidgets.QLabel("Selected geometry slot (FP/node QC):"))
            self.fp_slot_combo = QtWidgets.QComboBox()
            self.fp_slot_combo.addItems(slots)
            fp_row.addWidget(self.fp_slot_combo)
            fp_button = QtWidgets.QPushButton("Start/end selected G false positive [F]")
            fp_button.clicked.connect(self._toggle_false_positive)
            right.addWidget(fp_button)
            node_row = QtWidgets.QHBoxLayout()
            right.addLayout(node_row)
            node_row.addWidget(QtWidgets.QLabel("Node QC:"))
            self.node_qc_type_combo = QtWidgets.QComboBox()
            for event_type, label in NODE_QC_LABELS.items():
                self.node_qc_type_combo.addItem(label, event_type)
            node_row.addWidget(self.node_qc_type_combo)
            node_button = QtWidgets.QPushButton("Start/end selected G node QC [K]")
            node_button.clicked.connect(self._toggle_node_qc)
            right.addWidget(node_button)
            break_button = QtWidgets.QPushButton("Break selected-scope tracklets here [B]")
            break_button.clicked.connect(lambda: self._add_simple_event("track_break"))
            right.addWidget(break_button)
            anchor_button = QtWidgets.QPushButton("Assign all G → H identities [A]")
            anchor_button.clicked.connect(self._add_anchor)
            right.addWidget(anchor_button)
            no_swap_button = QtWidgets.QPushButton("Reviewed: no swap [R]")
            no_swap_button.clicked.connect(lambda: self._add_simple_event("reviewed_no_swap"))
            right.addWidget(no_swap_button)
            progress_button = QtWidgets.QPushButton(
                "Mark selected camera reviewed through this frame [C]"
            )
            progress_button.clicked.connect(self._mark_review_progress)
            right.addWidget(progress_button)
            right.addWidget(QtWidgets.QLabel("Notes for the next annotation:"))
            self.notes = QtWidgets.QPlainTextEdit()
            self.notes.setMaximumHeight(65)
            right.addWidget(self.notes)

            right.addWidget(self._line())
            right.addWidget(QtWidgets.QLabel("Saved annotations (double-click to jump):"))
            self.event_list = QtWidgets.QListWidget()
            self.event_list.itemDoubleClicked.connect(self._jump_to_event)
            right.addWidget(self.event_list, 1)
            edit_row = QtWidgets.QHBoxLayout()
            right.addLayout(edit_row)
            undo_button = QtWidgets.QPushButton("Undo last [Ctrl+Z]")
            delete_button = QtWidgets.QPushButton("Delete selected")
            undo_button.clicked.connect(self._undo)
            delete_button.clicked.connect(self._delete_selected)
            edit_row.addWidget(undo_button)
            edit_row.addWidget(delete_button)
            self.status = QtWidgets.QLabel("Autosave enabled")
            self.status.setWordWrap(True)
            right.addWidget(self.status)

            shortcuts = {
                "Q": lambda: self._add_swap("G1", "G2"),
                "W": lambda: self._add_swap("G1", "G3"),
                "E": lambda: self._add_swap("G2", "G3"),
                "I": self._start_ambiguous,
                "O": self._end_ambiguous,
                "F": self._toggle_false_positive,
                "K": self._toggle_node_qc,
                "B": lambda: self._add_simple_event("track_break"),
                "A": self._add_anchor,
                "R": lambda: self._add_simple_event("reviewed_no_swap"),
                "C": self._mark_review_progress,
                "Space": self._toggle_play,
                "Left": lambda: self._step(-1),
                "Right": lambda: self._step(1),
                "Shift+Left": lambda: self._step(-10),
                "Shift+Right": lambda: self._step(10),
                "Ctrl+Left": lambda: self._step(-100),
                "Ctrl+Right": lambda: self._step(100),
                "Ctrl+Z": self._undo,
                "Ctrl++": lambda: self._zoom(1.25),
                "Ctrl+=": lambda: self._zoom(1.25),
                "Ctrl+-": lambda: self._zoom(1 / 1.25),
                "Ctrl+0": self._fit_zoom,
                "U": self._toggle_camera_overlays,
                "P": lambda: self._go_candidate(self.current_candidate - 1),
                "N": lambda: self._go_candidate(self.current_candidate + 1),
            }
            for key, function in shortcuts.items():
                QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=function)
            self._refresh_candidate_combo()

        @property
        def active_scope(self) -> str:
            return str(self.scope_combo.currentData())

        def _set_scope(self, scope: str):
            index = self.scope_combo.findData(scope)
            if index >= 0:
                self.scope_combo.setCurrentIndex(index)

        def _scope_changed(self, _index: int):
            scope = self.active_scope
            self.status.setText(f"Annotation target: {SCOPE_LABELS[scope]}")
            self._render(self.slider.value())

        def _toggle_camera_overlays(self):
            self.camera_overlays.setChecked(not self.camera_overlays.isChecked())

        def _overlay_visibility_changed(self, visible: bool):
            self.status.setText(
                "Camera overlays shown" if visible else "Raw camera preview (annotations unchanged)"
            )
            self._render(self.slider.value())

        def _line(self):
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            return line

        def eventFilter(self, watched, event):
            if watched is self.image and event.type() == QtCore.QEvent.MouseButtonPress:
                half_width = max(1, self.image.width() // 2)
                half_height = max(1, self.image.height() // 2)
                if event.pos().y() < half_height:
                    scope = "top_cam" if event.pos().x() < half_width else "side1_cam"
                else:
                    scope = "side2_cam" if event.pos().x() < half_width else "3d"
                self._set_scope(scope)
                event.accept()
                return True
            if (
                watched is self.image_scroll.viewport()
                and event.type() == QtCore.QEvent.Wheel
                and event.modifiers() & QtCore.Qt.ControlModifier
            ):
                self._zoom(1.25 if event.angleDelta().y() > 0 else 1 / 1.25)
                event.accept()
                return True
            return super().eventFilter(watched, event)

        def _zoom(self, multiplier: float):
            self._set_zoom(self.zoom_factor * multiplier)

        def _fit_zoom(self):
            if self.base_pixmap is None:
                return
            viewport = self.image_scroll.viewport().size()
            factor = min(
                viewport.width() / max(1, self.base_pixmap.width()),
                viewport.height() / max(1, self.base_pixmap.height()),
            )
            self._set_zoom(factor)

        def _set_zoom(self, factor: float):
            factor = max(0.25, min(6.0, float(factor)))
            horizontal = self.image_scroll.horizontalScrollBar()
            vertical = self.image_scroll.verticalScrollBar()
            old_width = max(1, horizontal.maximum() + horizontal.pageStep())
            old_height = max(1, vertical.maximum() + vertical.pageStep())
            center_x = (horizontal.value() + horizontal.pageStep() / 2) / old_width
            center_y = (vertical.value() + vertical.pageStep() / 2) / old_height
            self.zoom_factor = factor
            self._apply_zoom()
            new_width = max(1, horizontal.maximum() + horizontal.pageStep())
            new_height = max(1, vertical.maximum() + vertical.pageStep())
            horizontal.setValue(int(center_x * new_width - horizontal.pageStep() / 2))
            vertical.setValue(int(center_y * new_height - vertical.pageStep() / 2))

        def _apply_zoom(self):
            if self.base_pixmap is None:
                return
            width = max(1, int(round(self.base_pixmap.width() * self.zoom_factor)))
            height = max(1, int(round(self.base_pixmap.height() * self.zoom_factor)))
            scaled = self.base_pixmap.scaled(
                width,
                height,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.image.setPixmap(scaled)
            self.image.resize(scaled.size())
            self.zoom_label.setText(f"{self.zoom_factor * 100:.0f}%  (Ctrl+wheel)")

        def _set_frame(self, frame: int):
            frame = max(0, min(data.n_frames - 1, int(frame)))
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame)
            self.frame_spin.blockSignals(False)
            self._render(frame)

        def _step(self, amount: int):
            self._set_frame(self.slider.value() + amount)

        def _toggle_play(self):
            if self.timer.isActive():
                self.timer.stop()
                self.play.setText("Play")
            else:
                self.timer.start(33)
                self.play.setText("Pause")

        def _advance_playback(self):
            next_frame = self.slider.value() + int(self.play_step.currentText())
            if next_frame >= data.n_frames:
                self._toggle_play()
                return
            self._set_frame(next_frame)

        def _go_candidate(self, index: int):
            if not 0 <= index < len(data.candidates):
                return
            self.current_candidate = int(index)
            self.candidate_combo.blockSignals(True)
            self.candidate_combo.setCurrentIndex(index)
            self.candidate_combo.blockSignals(False)
            row = data.candidates.iloc[index]
            self.candidate_info.setText(
                f"{row['source']}  {row['slots']}\n{row['details']}"
            )
            self._set_frame(int(row["frame"]))

        def _new_event(self, event_type: str, start: int, end: int, **values) -> dict:
            numeric_ids = [
                int(value[1:]) for value in self.events.get("event_id", pd.Series(dtype=str)).astype(str)
                if value.startswith("E") and value[1:].isdigit()
            ]
            event_id = f"E{max(numeric_ids, default=0) + 1:06d}"
            candidate_source = ""
            candidate_id = ""
            if len(data.candidates):
                candidate = data.candidates.iloc[self.current_candidate]
                if abs(int(start) - int(candidate["frame"])) <= args.candidate_merge_frames:
                    candidate_source = str(candidate["source"])
                    candidate_id = str(candidate["candidate_id"])
            event = {column: "" for column in EVENT_COLUMNS}
            event.update(
                event_id=event_id,
                event_type=event_type,
                view_scope=self.active_scope,
                frame_start=int(start),
                frame_end=int(end),
                confidence="reviewed",
                notes=self.notes.toPlainText().strip(),
                candidate_source=candidate_source,
                candidate_id=candidate_id,
                created_at=_now(),
            )
            event.update(values)
            return event

        def _commit_event(self, event: dict):
            self.events = pd.concat([self.events, pd.DataFrame([event])], ignore_index=True)
            self.events = normalize_events(self.events, slots)
            self.history.append(str(event["event_id"]))
            atomic_write_events(self.events, args.events_csv, slots)
            append_audit(args.audit_jsonl, {"action": "add", "timestamp": _now(), "event": event})
            self.notes.clear()
            self.status.setText(f"Saved {event['event_id']}: {event['event_type']}")
            self._refresh_event_list()
            self._refresh_candidate_combo()
            self._render(self.slider.value())

        def _add_swap(self, left_slot: str, right_slot: str):
            if left_slot not in slots or right_slot not in slots:
                return
            frame = self.slider.value()
            scope = self.active_scope
            _mapping, active, resolved = identity_state_at_frame(
                self.events, slots, frame, view_scope=scope
            )
            if active or not resolved:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Mapping unresolved",
                    f"Re-anchor the {SCOPE_LABELS[scope]} mapping before recording an exact pair swap.",
                )
                return
            self._commit_event(
                self._new_event(
                    "pair_swap", frame, frame, slot_a=left_slot, slot_b=right_slot
                )
            )

        def _start_ambiguous(self):
            self.pending_ambiguous_start = (self.active_scope, self.slider.value())
            scope, frame = self.pending_ambiguous_start
            self.status.setText(
                f"{SCOPE_LABELS[scope]} ambiguity starts at f{frame}; move to its end and press O"
            )

        def _end_ambiguous(self):
            if self.pending_ambiguous_start is None:
                self.status.setText("Set the ambiguous start first with I")
                return
            scope, first_frame = self.pending_ambiguous_start
            start, end = sorted((first_frame, self.slider.value()))
            self._commit_event(
                self._new_event("ambiguous", start, end, view_scope=scope)
            )
            self.pending_ambiguous_start = None

        def _toggle_false_positive(self):
            if self.pending_fp_start is None:
                scope = self.active_scope
                if scope == "3d":
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Select a camera",
                        "A raw SLEAP false positive must be assigned to top, side 1, or side 2.",
                    )
                    return
                slot = str(self.fp_slot_combo.currentText())
                frame = self.slider.value()
                self.pending_fp_start = (scope, slot, frame)
                self.status.setText(
                    f"{SCOPE_LABELS[scope]} {slot} false positive starts at f{frame}; "
                    "move to its last frame and press F again"
                )
                return
            scope, slot, first_frame = self.pending_fp_start
            start, end = sorted((first_frame, self.slider.value()))
            self._commit_event(
                self._new_event(
                    "false_positive",
                    start,
                    end,
                    view_scope=scope,
                    slot_a=slot,
                )
            )
            self.pending_fp_start = None

        def _toggle_node_qc(self):
            if self.pending_node_qc_start is None:
                scope = self.active_scope
                if scope == "3d":
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Select a camera",
                        "Mouth/tail QC must target top, side 1, or side 2.",
                    )
                    return
                slot = str(self.fp_slot_combo.currentText())
                event_type = str(self.node_qc_type_combo.currentData())
                frame = self.slider.value()
                self.pending_node_qc_start = (scope, slot, event_type, frame)
                self.status.setText(
                    f"{SCOPE_LABELS[scope]} {slot} {NODE_QC_LABELS[event_type]} starts "
                    f"at f{frame}; move to its last frame and press K again"
                )
                return
            scope, slot, event_type, first_frame = self.pending_node_qc_start
            start, end = sorted((first_frame, self.slider.value()))
            self._commit_event(
                self._new_event(
                    event_type,
                    start,
                    end,
                    view_scope=scope,
                    slot_a=slot,
                )
            )
            self.pending_node_qc_start = None

        def _mark_review_progress(self):
            scope = self.active_scope
            if scope == "3d":
                QtWidgets.QMessageBox.warning(
                    self,
                    "Select a camera",
                    "Review-progress checkpoints are independent for the three cameras.",
                )
                return
            frame = self.slider.value()
            self._commit_event(
                self._new_event(
                    "review_progress",
                    frame,
                    frame,
                    view_scope=scope,
                    notes=self.notes.toPlainText().strip() or "reviewed through frame",
                )
            )

        def _add_simple_event(self, event_type: str):
            frame = self.slider.value()
            self._commit_event(self._new_event(event_type, frame, frame))

        def _add_anchor(self):
            scope = self.active_scope
            mapping, _active = mapping_at_frame(
                self.events, slots, self.slider.value(), view_scope=scope
            )
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"Assign identities — {SCOPE_LABELS[scope]}")
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(
                QtWidgets.QLabel(
                    "Assign one human identity to each geometry slot. "
                    "Each H identity must be used exactly once."
                )
            )
            form = QtWidgets.QFormLayout()
            layout.addLayout(form)
            identities = sorted(mapping.values())
            selectors = {}
            for slot in slots:
                selector = QtWidgets.QComboBox()
                selector.addItems(identities)
                selector.setCurrentText(mapping[slot])
                selectors[slot] = selector
                form.addRow(f"{slot} →", selector)
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            while True:
                if dialog.exec_() != QtWidgets.QDialog.Accepted:
                    return
                assigned = {slot: selector.currentText() for slot, selector in selectors.items()}
                if len(set(assigned.values())) == len(assigned):
                    break
                QtWidgets.QMessageBox.warning(
                    dialog,
                    "Duplicate identity",
                    "Each H identity must be assigned to exactly one G slot.",
                )
            frame = self.slider.value()
            event = self._new_event(
                "mapping_anchor", frame, frame, mapping_after=format_mapping(assigned)
            )
            try:
                normalize_events(pd.concat([self.events, pd.DataFrame([event])]), slots)
            except ValueError as error:
                QtWidgets.QMessageBox.warning(self, "Invalid mapping", str(error))
                return
            self._commit_event(event)

        def _undo(self):
            if not self.history:
                self.status.setText("Nothing from this session to undo")
                return
            event_id = self.history.pop()
            self._delete_event(event_id, action="undo")

        def _delete_selected(self):
            item = self.event_list.currentItem()
            if item is None:
                return
            self._delete_event(str(item.data(QtCore.Qt.UserRole)), action="delete")

        def _delete_event(self, event_id: str, *, action: str):
            removed = self.events[self.events["event_id"].astype(str).eq(event_id)]
            if removed.empty:
                return
            self.events = self.events[~self.events["event_id"].astype(str).eq(event_id)].copy()
            atomic_write_events(self.events, args.events_csv, slots)
            append_audit(
                args.audit_jsonl,
                {"action": action, "timestamp": _now(), "event": removed.iloc[0].to_dict()},
            )
            self.status.setText(f"Removed {event_id}")
            self._refresh_event_list()
            self._refresh_candidate_combo()
            self._render(self.slider.value())

        def _refresh_candidate_combo(self):
            current = self.current_candidate
            reviewed_by_candidate: dict[str, set[str]] = {}
            for row in self.events.itertuples(index=False):
                candidate_id = str(row.candidate_id)
                if not candidate_id:
                    continue
                scope = str(row.view_scope)
                scopes = set(REVIEW_SCOPES) if scope == "all" else {scope}
                reviewed_by_candidate.setdefault(candidate_id, set()).update(scopes)
            self.candidate_combo.blockSignals(True)
            self.candidate_combo.clear()
            for row in data.candidates.itertuples(index=False):
                reviewed_scopes = reviewed_by_candidate.get(str(row.candidate_id), set())
                count = len(reviewed_scopes & set(REVIEW_SCOPES))
                mark = "✓" if count == len(REVIEW_SCOPES) else f"{count}/{len(REVIEW_SCOPES)}"
                self.candidate_combo.addItem(
                    f"{mark} {row.candidate_id}  f{row.frame}  {row.source}"
                )
            if len(data.candidates):
                self.candidate_combo.setCurrentIndex(
                    max(0, min(current, len(data.candidates) - 1))
                )
            self.candidate_combo.blockSignals(False)
            reviewed_total = sum(
                len(value & set(REVIEW_SCOPES)) for value in reviewed_by_candidate.values()
            )
            self.status.setText(
                f"Scoped reviews: {reviewed_total} / {len(data.candidates) * len(REVIEW_SCOPES)}  (autosaved)"
            )

        def _refresh_event_list(self):
            self.event_list.clear()
            for row in self.events.itertuples(index=False):
                if row.event_type == "pair_swap":
                    suffix = f" {row.slot_a}<->{row.slot_b}"
                elif row.event_type == "false_positive":
                    suffix = f" {row.slot_a}=FP"
                elif row.event_type in NODE_QC_LABELS:
                    suffix = f" {row.slot_a}={NODE_QC_LABELS[row.event_type]}"
                elif row.event_type == "review_progress":
                    suffix = " reviewed-through"
                else:
                    suffix = ""
                item = QtWidgets.QListWidgetItem(
                    f"{row.event_id}  [{row.view_scope}]  {row.event_type}  "
                    f"f{row.frame_start}..{row.frame_end}{suffix}"
                )
                item.setData(QtCore.Qt.UserRole, str(row.event_id))
                item.setData(QtCore.Qt.UserRole + 1, int(row.frame_start))
                self.event_list.addItem(item)

        def _jump_to_event(self, item):
            self._set_frame(int(item.data(QtCore.Qt.UserRole + 1)))

        def _read_video(self, view: str, frame: int) -> np.ndarray:
            cap = caps[view]
            if frame != self.last_video_frame[view] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            self.last_video_frame[view] = frame
            if not ok or image is None:
                image = np.zeros((1200, 1200, 3), dtype=np.uint8)
            return cv2.resize(image, (panel, panel), interpolation=cv2.INTER_AREA)

        def _draw_views(self, frame: int, states: dict[str, tuple]):
            images = {view: self._read_video(view, frame) for view in VIEWS}
            if not self.camera_overlays.isChecked():
                return images
            scale = panel / 1200.0
            for view in VIEWS:
                mapping, active, resolved = states[view]
                ambiguous = bool(active) or not resolved
                false_positives = false_positive_slots_at_frame(
                    self.events, slots, frame, view_scope=view
                )
                node_qc = node_qc_events_at_frame(
                    self.events, slots, frame, view_scope=view
                )
                instances = slp_predictions[view].get(frame)
                draw_raw_sleap_gray(images[view], instances, scale)
                if instances is None or len(instances) == 0:
                    continue
                instance_array = np.asarray(instances, dtype=float)
                projected: dict[str, np.ndarray] = {}
                for slot in slots:
                    xyz = data.positions[slot][frame]
                    if np.isfinite(xyz).all():
                        projected[slot] = np.asarray(
                            calibration.project_world_to_view(xyz.reshape(1, 3), view)
                        )[0]
                assigned = match_projected_identities_to_instances(
                    projected, instance_array, args.mouth_match_gate_px
                )
                for slot, instance_index in assigned.items():
                    mouth = instance_array[instance_index, 0]
                    tail = (
                        instance_array[instance_index, 1]
                        if instance_array.shape[1] > 1
                        else np.full(2, np.nan)
                    )
                    point = (
                        tuple(np.round(mouth * scale).astype(int))
                        if np.isfinite(mouth).all()
                        else None
                    )
                    tail_point = (
                        tuple(np.round(tail * scale).astype(int))
                        if np.isfinite(tail).all()
                        else None
                    )
                    if slot in false_positives:
                        if point is None:
                            continue
                        fp_color = (255, 0, 255)
                        cv2.line(
                            images[view],
                            (point[0] - 7, point[1] - 7),
                            (point[0] + 7, point[1] + 7),
                            fp_color,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            images[view],
                            (point[0] - 7, point[1] + 7),
                            (point[0] + 7, point[1] - 7),
                            fp_color,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            images[view],
                            f"{slot}/FP",
                            (point[0] + 10, point[1] - 8),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.4,
                            fp_color,
                            1,
                            cv2.LINE_AA,
                        )
                        continue
                    faults = node_qc.get(slot, {})
                    fault_color = (255, 0, 255)

                    def draw_fault_cross(target):
                        if target is None:
                            return
                        cv2.line(
                            images[view],
                            (target[0] - 6, target[1] - 6),
                            (target[0] + 6, target[1] + 6),
                            fault_color,
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            images[view],
                            (target[0] - 6, target[1] + 6),
                            (target[0] + 6, target[1] - 6),
                            fault_color,
                            1,
                            cv2.LINE_AA,
                        )

                    if "bad_mouth" in faults:
                        draw_fault_cross(point)
                    if "bad_tail" in faults:
                        draw_fault_cross(tail_point)
                    if "wrong_link" in faults:
                        if point is not None and tail_point is not None:
                            cv2.line(
                                images[view], point, tail_point, fault_color, 1, cv2.LINE_AA
                            )
                        draw_fault_cross(point)
                        draw_fault_cross(tail_point)
                    if faults:
                        label_point = point or tail_point
                        if label_point is not None:
                            abbreviations = {
                                "bad_mouth": "BM",
                                "bad_tail": "BT",
                                "wrong_link": "LINK",
                            }
                            fault_label = "/".join(
                                abbreviations[name]
                                for name in NODE_QC_LABELS
                                if name in faults
                            )
                            cv2.putText(
                                images[view],
                                f"{slot}/{fault_label}",
                                (label_point[0] + 7, label_point[1] + 12),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.36,
                                fault_color,
                                1,
                                cv2.LINE_AA,
                            )
                    if point is None or "bad_mouth" in faults:
                        continue
                    identity = mapping[slot]
                    color = GRAY if ambiguous else HUMAN_COLORS.get(identity, GRAY)
                    label = f"{slot}/{'?' if ambiguous else identity}"
                    if data.interpolated[slot][frame]:
                        cv2.circle(images[view], point, 5, color, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(images[view], point, 5, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(images[view], point, 3, color, -1, cv2.LINE_AA)
                    cv2.putText(images[view], label, (point[0] + 6, point[1] - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(images[view], label, (point[0] + 6, point[1] - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)
            for view, image in images.items():
                cv2.putText(image, view, (8, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
            return images

        def _draw_3d(self, frame: int, state: tuple):
            mapping, active, resolved = state
            ambiguous = bool(active) or not resolved
            image = base_3d.copy()
            tail = max(0, frame - int(args.tail_seconds * args.source_fps))
            for slot in slots:
                identity = mapping[slot]
                color = GRAY if ambiguous else HUMAN_COLORS.get(identity, GRAY)
                trajectory = data.positions[slot][tail : frame + 1]
                pixels = projector.project(trajectory)
                finite = np.isfinite(trajectory).all(axis=1)
                for index in range(1, len(pixels)):
                    if finite[index - 1] and finite[index]:
                        cv2.line(image, tuple(pixels[index - 1].astype(int)),
                                 tuple(pixels[index].astype(int)), color, 1, cv2.LINE_AA)
                current = data.positions[slot][frame]
                if not np.isfinite(current).all():
                    continue
                point = tuple(projector.project(current.reshape(1, 3))[0].astype(int))
                cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
                cv2.putText(image, f"{slot}/{'?' if ambiguous else identity}",
                            (point[0] + 6, point[1] - 5), cv2.FONT_HERSHEY_DUPLEX,
                            0.4, color, 1, cv2.LINE_AA)
            cv2.putText(image, f"3D  frame {frame}", (8, 20), cv2.FONT_HERSHEY_DUPLEX,
                        0.55, (30, 30, 30), 1, cv2.LINE_AA)
            return image

        def _render(self, frame: int):
            states = {
                scope: identity_state_at_frame(
                    self.events, slots, int(frame), view_scope=scope
                )
                for scope in REVIEW_SCOPES
            }
            images = self._draw_views(int(frame), states)
            panel_3d = self._draw_3d(int(frame), states["3d"])
            composite = np.empty((panel * 2, panel * 2, 3), dtype=np.uint8)
            composite[:panel, :panel] = images["top_cam"]
            composite[:panel, panel:] = images["side1_cam"]
            composite[panel:, :panel] = images["side2_cam"]
            composite[panel:, panel:] = panel_3d
            selected = REVIEW_SCOPES.index(self.active_scope)
            left = 0 if selected in (0, 2) else panel
            top = 0 if selected in (0, 1) else panel
            cv2.rectangle(
                composite,
                (left + 2, top + 2),
                (left + panel - 3, top + panel - 3),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            qimage = QtGui.QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888
            ).copy()
            self.base_pixmap = QtGui.QPixmap.fromImage(qimage)
            self._apply_zoom()
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(int(frame))
            self.frame_spin.blockSignals(False)
            self.frame_label.setText(f"t={frame / args.source_fps:.3f}s")
            mapping_lines = []
            for scope in REVIEW_SCOPES:
                mapping, active, resolved = states[scope]
                prefix = "▶" if scope == self.active_scope else " "
                if active:
                    status = " AMBIGUOUS"
                elif not resolved:
                    status = " UNRESOLVED"
                else:
                    status = ""
                if scope != "3d":
                    fp_slots = false_positive_slots_at_frame(
                        self.events, slots, int(frame), view_scope=scope
                    )
                    if fp_slots:
                        status += " FP:" + ",".join(sorted(fp_slots))
                    node_qc = node_qc_events_at_frame(
                        self.events, slots, int(frame), view_scope=scope
                    )
                    if node_qc:
                        fault_slots = [
                            f"{slot}=" + "/".join(sorted(faults))
                            for slot, faults in sorted(node_qc.items())
                        ]
                        status += " NODE:" + ",".join(fault_slots)
                    checkpoint = review_progress_checkpoint(
                        self.events, slots, view_scope=scope
                    )
                    if checkpoint is not None:
                        status += f" REVIEWED≤{checkpoint['frame']}"
                mapping_lines.append(
                    f"{prefix} {SCOPE_LABELS[scope]}: {format_mapping(mapping)}{status}"
                )
            self.mapping_label.setText("\n".join(mapping_lines))

        def closeEvent(self, event):
            self.timer.stop()
            for cap in caps.values():
                cap.release()
            event.accept()

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ReviewWindow()
    window.show()
    if args.smoke_test:
        def finish_smoke_test():
            window._set_scope("top_cam")
            window._set_scope("side2_cam")
            window._set_scope("3d")
            window._toggle_camera_overlays()
            window._toggle_camera_overlays()
            window._zoom(1.25)
            window._zoom(1 / 1.25)
            window._fit_zoom()
            window.close()
        QtCore.QTimer.singleShot(750, finish_smoke_test)
    return int(application.exec_())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--slp-dir", type=Path,
                        help="Directory containing top_cam/side1_cam/side2_cam.slp.")
    parser.add_argument("--calibration-model-dir", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--extra-candidates", action="append", type=Path, default=[])
    parser.add_argument("--id-column", default="fish_id")
    parser.add_argument("--slot-map", action="append", default=[], metavar="SOURCE=SLOT")
    parser.add_argument("--close-mm", type=float, default=18.0)
    parser.add_argument("--jump-mm", type=float, default=8.0)
    parser.add_argument("--candidate-merge-frames", type=int, default=166,
                        help="Deduplicate risk signals within this many source frames.")
    parser.add_argument("--min-interpolation-frames", type=int, default=30)
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    parser.add_argument("--mouth-match-gate-px", type=float, default=80.0,
                        help="Gate for centroid-to-instance matching before drawing the ID at its mouth.")
    parser.add_argument("--panel", type=int, default=520)
    parser.add_argument("--tail-seconds", type=float, default=3.0)
    parser.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 200.0))
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--build-candidates-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Open, render once, and close automatically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        slot_map = parse_slot_specs(args.slot_map)
    except ValueError as error:
        parser.error(str(error))
    if args.audit_jsonl is None:
        args.audit_jsonl = args.events_csv.with_suffix(".audit.jsonl")
    if args.candidate_csv is None:
        args.candidate_csv = args.events_csv.with_name("geometry_review_candidates.csv")

    geometry, resolved_slot_map = load_geometry_table(
        args.geometry_csv, id_column=args.id_column, slot_map=slot_map
    )
    existing_events = (
        pd.read_csv(args.events_csv, keep_default_na=False)
        if args.events_csv.exists()
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    atomic_write_events(
        existing_events,
        args.events_csv,
        sorted(resolved_slot_map.values()),
    )
    extras = []
    for path in args.extra_candidates:
        extras.append((path.stem, load_extra_candidate_table(path)))
    if args.candidate_csv.exists() and not args.build_candidates_only:
        candidates = pd.read_csv(args.candidate_csv, keep_default_na=False)
    else:
        candidates = build_review_candidates(
            geometry,
            close_mm=args.close_mm,
            jump_mm=args.jump_mm,
            merge_frames=args.candidate_merge_frames,
            min_interpolation_frames=args.min_interpolation_frames,
            extra_candidate_tables=extras,
        )
        args.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.candidate_csv, index=False)
    print(f"review candidates: {len(candidates):,} -> {args.candidate_csv}")
    if args.build_candidates_only:
        return 0
    if args.slp_dir is None:
        parser.error("--slp-dir is required for raw SLEAP and mouth-centered ID overlays")
    slp_predictions: dict[str, IndexedSlpPredictions] = {}
    try:
        for view in VIEWS:
            path = args.slp_dir / f"{view}.slp"
            print(f"indexing raw SLEAP predictions: {path}", flush=True)
            slp_predictions[view] = IndexedSlpPredictions(
                path, node_names=tuple(args.node_names)
            )
    except Exception:
        for predictions in slp_predictions.values():
            predictions.close()
        raise
    events = pd.read_csv(args.events_csv, keep_default_na=False)
    data = prepare_review_data(geometry, resolved_slot_map, candidates, events)
    try:
        return run_gui(args, data, slp_predictions)
    finally:
        for predictions in slp_predictions.values():
            predictions.close()


if __name__ == "__main__":
    raise SystemExit(main())
