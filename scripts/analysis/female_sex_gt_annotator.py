"""PyQt5 GUI for conservative per-frame female-fish ground truth.

The two side views are synchronized on exact source-frame numbers.  Human
clicks are canonical; raw SLEAP and optional geometry IDs are saved only as
frame-local provenance and never choose or propagate the female label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from scripts.analysis.build_female_sex_gt_candidates import CandidateConfig
from scripts.analysis.build_female_sex_gt_candidates import \
    run_from_paths as run_candidate_selector
from scripts.analysis.female_sex_gt import (AnnotationLedger,
                                            candidate_from_annotation,
                                            candidate_tracklet_context,
                                            empty_selection,
                                            geometry_consistency,
                                            latest_annotated_frame,
                                            load_and_merge_candidate_csvs,
                                            make_annotation_row,
                                            manual_jump_candidate,
                                            match_geometry_to_instances,
                                            normalize_candidate_table,
                                            normalize_path, queue_status,
                                            selection_from_click)
from scripts.analysis.geometry_swap_review import (VIDEO_NAMES,
                                                   IndexedSlpPredictions)
from scripts.common.local_paths import (DEFAULT_PATHS_CONFIG, REPO_ROOT,
                                        load_local_paths)

SIDE_VIEWS = ("side1_cam", "side2_cam")
VIEW_PREFIX = {"side1_cam": "side1", "side2_cam": "side2"}
CAMERA_INDEX = {"side1_cam": 1, "side2_cam": 2}
RAW_SLEAP_GRAY = (165, 165, 165)
SELECTED_COLOR = (0, 220, 255)


def _three_fish_focus_points(
    instances: np.ndarray | None,
    instance_geometry_ids: Mapping[int, Any] | None = None,
    projected_geometry: Mapping[Any, Sequence[float]] | None = None,
) -> np.ndarray:
    """Return points that bound exactly three fish, excluding extra detections."""
    array = np.asarray(instances, dtype=float) if instances is not None else None
    valid_array = (
        array is not None
        and array.ndim == 3
        and array.shape[-1] == 2
    )

    projected: dict[str, np.ndarray] = {}
    for geometry_id, point in (projected_geometry or {}).items():
        value = np.asarray(point, dtype=float)
        if value.shape == (2,) and np.isfinite(value).all():
            projected[str(geometry_id)] = value

    matched = {
        int(index): str(geometry_id)
        for index, geometry_id in (instance_geometry_ids or {}).items()
    }
    matched_ids = set(matched.values())
    geometry_supplied = (
        instance_geometry_ids is not None or projected_geometry is not None
    )

    points: list[np.ndarray] = []
    expected_ids: set[str] = set()
    if len(projected) == 3:
        expected_ids = set(projected)
        points.extend(projected.values())
    elif len(matched_ids) == 3:
        expected_ids = matched_ids

    if expected_ids and valid_array:
        for index, geometry_id in matched.items():
            if geometry_id not in expected_ids or index < 0 or index >= len(array):
                continue
            pose = array[index]
            points.extend(pose[np.all(np.isfinite(pose), axis=1)])
    if expected_ids:
        return np.asarray(points, dtype=float).reshape(-1, 2)
    if geometry_supplied or not valid_array:
        return np.empty((0, 2), dtype=float)

    visible_poses = []
    for pose in array:
        finite = pose[np.all(np.isfinite(pose), axis=1)]
        if len(finite):
            visible_poses.append(finite)
    if len(visible_poses) != 3:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(visible_poses, axis=0)


def _focus_transform(
    *,
    image_size: tuple[int, int],
    viewport_size: tuple[int, int],
    points: np.ndarray,
) -> tuple[float, tuple[float, float]] | None:
    """Compute a padded source-pixel crop center and zoom for fish points."""
    image_width, image_height = (float(value) for value in image_size)
    viewport_width, viewport_height = (float(value) for value in viewport_size)
    if min(image_width, image_height, viewport_width, viewport_height) <= 0:
        return None
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    finite = np.all(np.isfinite(values), axis=1)
    inside = (
        finite
        & (values[:, 0] >= 0)
        & (values[:, 0] <= image_width)
        & (values[:, 1] >= 0)
        & (values[:, 1] <= image_height)
    )
    values = values[inside]
    if not len(values):
        return None

    low = values.min(axis=0)
    high = values.max(axis=0)
    center = 0.5 * (low + high)
    span = np.maximum(high - low, np.minimum((96.0, 96.0), (image_width, image_height)))
    padding = max(36.0, 0.18 * float(span.max()))
    crop_width = min(image_width, float(span[0] + 2.0 * padding))
    crop_height = min(image_height, float(span[1] + 2.0 * padding))
    center[0] = np.clip(center[0], crop_width / 2.0, image_width - crop_width / 2.0)
    center[1] = np.clip(center[1], crop_height / 2.0, image_height - crop_height / 2.0)
    zoom = 0.96 * min(viewport_width / crop_width, viewport_height / crop_height)
    return float(np.clip(zoom, 0.05, 6.0)), (float(center[0]), float(center[1]))


def _anchored_scroll_value(
    *,
    source_coordinate: float,
    source_extent: int,
    scaled_extent: int,
    viewport_coordinate: float,
    viewport_extent: int,
) -> int:
    """Scroll so a source coordinate remains under a viewport coordinate."""
    if source_extent <= 0 or scaled_extent <= 0 or viewport_extent <= 0:
        return 0
    scaled_coordinate = float(source_coordinate) * scaled_extent / source_extent
    centered_offset = max(0.0, (viewport_extent - scaled_extent) / 2.0)
    return max(0, int(round(centered_offset + scaled_coordinate - viewport_coordinate)))


class OpenCvVideo:
    """Small exact-frame, read-only OpenCV source with an LRU cache."""

    def __init__(self, path: str | Path, *, cache_size: int = 32):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"could not open source video: {self.path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.close()
            raise ValueError(f"video has invalid dimensions/frame count: {self.path}")
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if frame < 0 or frame >= self.frame_count:
            raise IndexError(f"frame {frame} outside 0..{self.frame_count - 1}")
        if frame in self.cache:
            image = self.cache.pop(frame)
            self.cache[frame] = image
            return image.copy()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.capture.read()
        if not ok or image is None:
            raise IOError(f"failed to decode exact source frame {frame} from {self.path}")
        self.cache[frame] = image
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return image.copy()

    def close(self) -> None:
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()
            self.capture = None


class SyntheticVideo:
    """In-memory source used by the offscreen end-to-end smoke test."""

    def __init__(self, view: str, frame_count: int = 12):
        self.view = view
        self.path = Path(f"synthetic/{view}.mp4")
        self.frame_count = int(frame_count)
        self.width = 320
        self.height = 180
        self.reported_fps = 166.0

    def get(self, frame: int) -> np.ndarray:
        if frame < 0 or frame >= self.frame_count:
            raise IndexError(frame)
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        color = (70, 120, 220) if self.view == "side1_cam" else (180, 100, 60)
        cv2.circle(image, (70 + frame * 4, 70), 18, color, -1, cv2.LINE_AA)
        cv2.circle(image, (220 - frame * 3, 112), 16, (80, 180, 90), -1, cv2.LINE_AA)
        cv2.circle(image, (150, 145 - frame), 15, (180, 90, 170), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{self.view}  source frame {frame}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return image

    def close(self) -> None:
        return None


class SyntheticSlp:
    def __init__(self, view: str):
        self.path = Path(f"synthetic/{view}.slp")
        self.node_names = ("mouth", "tail")
        self.max_source_frame = 11

    def get(self, frame: int, default=None):
        offset = frame * 4
        if self.path.stem == "side1_cam":
            return np.array(
                [
                    [[80 + offset, 70], [55 + offset, 70]],
                    [[230 - frame * 3, 112], [205 - frame * 3, 112]],
                    [[160, 145 - frame], [135, 145 - frame]],
                ],
                dtype=float,
            )
        return np.array(
            [
                [[82 + offset, 72], [57 + offset, 72]],
                [[228 - frame * 3, 110], [203 - frame * 3, 110]],
                [[158, 143 - frame], [133, 143 - frame]],
            ],
            dtype=float,
        )

    def close(self) -> None:
        return None


def _geometry_id_column(table: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in table.columns:
            raise ValueError(f"geometry CSV has no requested ID column {explicit!r}")
        return explicit
    for column in ("final_fish_id", "geometry_slot", "fish_id", "geometry_id"):
        if column in table.columns:
            return column
    raise ValueError(
        "geometry CSV needs an ID column: final_fish_id, geometry_slot, fish_id, "
        "or geometry_id"
    )


class GeometryAssociator:
    """Read-only geometry reprojection and per-view one-to-one pose matching."""

    def __init__(
        self,
        path: str | Path,
        calibration_model_dir: str | Path,
        *,
        backend: str,
        id_column: str | None = None,
        max_frame: int | None = None,
    ):
        from scripts.reconstruction_3d.calibration import Calibration

        self.path = Path(path)
        header = pd.read_csv(self.path, nrows=0).columns
        required = {"frame", "X", "Y", "Z"}
        missing = required - set(header)
        if missing:
            raise ValueError(f"geometry CSV is missing columns: {sorted(missing)}")
        table = pd.read_csv(self.path, low_memory=False)
        self.id_column = _geometry_id_column(table, id_column)
        numeric_frame = pd.to_numeric(table["frame"], errors="raise")
        if not np.all(np.isfinite(numeric_frame)) or not np.all(
            numeric_frame == np.floor(numeric_frame)
        ):
            raise ValueError("geometry CSV frame values must be finite integers")
        table["frame"] = numeric_frame.astype(np.int64)
        if (table["frame"] < 0).any():
            raise ValueError("geometry CSV contains negative source frames")
        if max_frame is not None and not table.empty and int(table["frame"].max()) > max_frame:
            raise ValueError(
                f"geometry CSV extends to frame {int(table['frame'].max())}, "
                f"beyond video frame {max_frame}"
            )
        table = table[table[self.id_column].notna()].copy()
        table[self.id_column] = table[self.id_column].astype(str)
        if table.duplicated(["frame", self.id_column]).any():
            raise ValueError(
                f"geometry CSV has duplicate ({'frame'}, {self.id_column}) rows"
            )
        self.table = table
        self.by_frame = {
            int(frame): group[[self.id_column, "X", "Y", "Z"]].copy()
            for frame, group in table.groupby("frame", sort=False)
        }
        self.calibration = Calibration(calibration_model_dir, backend=backend)
        self.backend = self.calibration.backend_name
        self.model_files = self.calibration.provenance.get("model_files", {})
        self._projected_cache: OrderedDict[int, dict[str, dict[str, np.ndarray]]] = (
            OrderedDict()
        )

    def projected(self, frame: int) -> dict[str, dict[str, np.ndarray]]:
        frame = int(frame)
        if frame in self._projected_cache:
            result = self._projected_cache.pop(frame)
            self._projected_cache[frame] = result
            return result
        result = {view: {} for view in SIDE_VIEWS}
        rows = self.by_frame.get(frame)
        if rows is not None and not rows.empty:
            xyz = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
            valid = np.all(np.isfinite(xyz), axis=1)
            if np.any(valid):
                projected = self.calibration.project_world(xyz[valid])
                ids = rows.loc[valid, self.id_column].astype(str).tolist()
                for row_index, geometry_id in enumerate(ids):
                    for view in SIDE_VIEWS:
                        result[view][geometry_id] = np.asarray(
                            projected[row_index, CAMERA_INDEX[view]], dtype=float
                        )
        self._projected_cache[frame] = result
        while len(self._projected_cache) > 256:
            self._projected_cache.popitem(last=False)
        return result

    def associate(
        self,
        frame: int,
        instances_by_view: Mapping[str, np.ndarray | None],
        gate_px: float,
    ) -> tuple[dict[str, dict[int, str]], dict[str, dict[str, np.ndarray]]]:
        projected = self.projected(frame)
        matches = {
            view: match_geometry_to_instances(
                projected[view], instances_by_view.get(view), gate_px
            )
            for view in SIDE_VIEWS
        }
        return matches, projected


def _draw_raw_sleap(image: np.ndarray, instances: np.ndarray | None) -> None:
    if instances is None:
        return
    for pose in np.asarray(instances, dtype=float):
        if len(pose) >= 2 and np.isfinite(pose[:2]).all():
            mouth = tuple(np.round(pose[0]).astype(int))
            tail = tuple(np.round(pose[1]).astype(int))
            cv2.arrowedLine(
                image, tail, mouth, RAW_SLEAP_GRAY, 2, cv2.LINE_AA, 0, 0.24
            )
        for point in pose:
            if np.isfinite(point).all():
                cv2.circle(
                    image,
                    tuple(np.round(point).astype(int)),
                    3,
                    RAW_SLEAP_GRAY,
                    -1,
                    cv2.LINE_AA,
                )


def _draw_selection(image: np.ndarray, selection: Mapping[str, Any]) -> None:
    if selection.get("selection_status", "not_selected") == "not_selected":
        return
    click = (int(round(float(selection["click_x"]))), int(round(float(selection["click_y"]))))
    cv2.drawMarker(
        image, click, SELECTED_COLOR, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA
    )
    mouth = np.array([selection.get("mouth_x", ""), selection.get("mouth_y", "")])
    tail = np.array([selection.get("tail_x", ""), selection.get("tail_y", "")])
    try:
        mouth = mouth.astype(float)
        tail = tail.astype(float)
    except (TypeError, ValueError):
        mouth = tail = np.array([np.nan, np.nan])
    if np.isfinite(mouth).all() and np.isfinite(tail).all():
        cv2.line(
            image,
            tuple(np.round(tail).astype(int)),
            tuple(np.round(mouth).astype(int)),
            SELECTED_COLOR,
            4,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "F selection",
        (click[0] + 10, max(18, click[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        SELECTED_COLOR,
        2,
        cv2.LINE_AA,
    )


def _draw_geometry_labels(
    image: np.ndarray, projected: Mapping[str, np.ndarray]
) -> None:
    for geometry_id, point in projected.items():
        point = np.asarray(point, dtype=float)
        if not np.isfinite(point).all():
            continue
        xy = tuple(np.round(point).astype(int))
        cv2.circle(image, xy, 7, (200, 180, 60), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(geometry_id),
            (xy[0] + 8, xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 180, 60),
            1,
            cv2.LINE_AA,
        )


def _selection_from_saved(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    selection = empty_selection()
    for name in selection:
        value = row.get(f"{prefix}_{name}", selection[name])
        selection[name] = value
    index = selection.get("instance_index", "")
    if index != "":
        selection["instance_index"] = int(float(index))
    return selection


def _load_qt():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise SystemExit("PyQt5 is required; install with `uv sync --extra gui`.") from error
    return QtCore, QtGui, QtWidgets


def create_annotation_window(
    args: argparse.Namespace,
    *,
    videos: Mapping[str, Any],
    slp_predictions: Mapping[str, Any],
    candidates: pd.DataFrame,
    ledger: AnnotationLedger,
    geometry: GeometryAssociator | None,
):
    QtCore, QtGui, QtWidgets = _load_qt()

    class ImagePanel(QtWidgets.QWidget):
        clicked = QtCore.pyqtSignal(str, float, float)

        def __init__(self, view: str):
            super().__init__()
            self.view = view
            self.base_pixmap = QtGui.QPixmap()
            self.zoom = 1.0
            self.focus_points = np.empty((0, 2), dtype=float)
            self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            self.label.setStyleSheet("background:#15171a")
            self.label.setCursor(QtCore.Qt.CrossCursor)
            self.label.installEventFilter(self)
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidget(self.label)
            self.scroll.setWidgetResizable(False)
            self.scroll.setAlignment(QtCore.Qt.AlignCenter)
            self.scroll.setMinimumSize(480, 360)
            title = QtWidgets.QLabel(view)
            title.setStyleSheet("font-weight:bold;font-size:15px")
            fish_button = QtWidgets.QPushButton("Fish")
            full_button = QtWidgets.QPushButton("Full")
            reset_button = QtWidgets.QPushButton("100%")
            minus_button = QtWidgets.QPushButton("−")
            plus_button = QtWidgets.QPushButton("+")
            fish_button.setToolTip("Fit the current three-fish region")
            full_button.setToolTip("Fit the complete source frame")
            fish_button.clicked.connect(self.fit_fish)
            full_button.clicked.connect(self.fit)
            reset_button.clicked.connect(lambda: self.set_zoom(1.0))
            minus_button.clicked.connect(lambda: self.set_zoom(self.zoom / 1.25))
            plus_button.clicked.connect(lambda: self.set_zoom(self.zoom * 1.25))
            self.zoom_label = QtWidgets.QLabel("100%")
            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(title)
            controls.addStretch(1)
            controls.addWidget(QtWidgets.QLabel("auto fish fit; Ctrl+wheel at pointer"))
            controls.addWidget(minus_button)
            controls.addWidget(fish_button)
            controls.addWidget(full_button)
            controls.addWidget(reset_button)
            controls.addWidget(plus_button)
            controls.addWidget(self.zoom_label)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.addLayout(controls)
            layout.addWidget(self.scroll, 1)

        def set_active(self, active: bool) -> None:
            border = "#ffcf33" if active else "#383c42"
            self.setStyleSheet(f"ImagePanel {{border:3px solid {border};}}")

        def set_image(
            self,
            bgr: np.ndarray,
            *,
            focus_points: np.ndarray | None = None,
            auto_fit: bool = False,
        ) -> None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            image = QtGui.QImage(
                rgb.data,
                width,
                height,
                int(rgb.strides[0]),
                QtGui.QImage.Format_RGB888,
            ).copy()
            self.base_pixmap = QtGui.QPixmap.fromImage(image)
            self.focus_points = (
                np.asarray(focus_points, dtype=float).reshape(-1, 2)
                if focus_points is not None
                else np.empty((0, 2), dtype=float)
            )
            if auto_fit:
                if not self.fit_fish():
                    self.fit()
                return
            self._apply_zoom()

        def _source_from_label_position(self, position) -> tuple[float, float]:
            pixmap = self.label.pixmap()
            if self.base_pixmap.isNull() or pixmap is None or pixmap.isNull():
                return 0.0, 0.0
            return (
                float(position.x()) * self.base_pixmap.width() / pixmap.width(),
                float(position.y()) * self.base_pixmap.height() / pixmap.height(),
            )

        def _viewport_center_source(self) -> tuple[float, float]:
            viewport = self.scroll.viewport()
            center = viewport.rect().center()
            return self._source_from_label_position(
                self.label.mapFrom(viewport, center)
            )

        def _anchor_source(
            self,
            source_anchor: tuple[float, float],
            viewport_anchor,
        ) -> None:
            pixmap = self.label.pixmap()
            if self.base_pixmap.isNull() or pixmap is None or pixmap.isNull():
                return
            viewport = self.scroll.viewport()
            horizontal = self.scroll.horizontalScrollBar()
            vertical = self.scroll.verticalScrollBar()
            horizontal.setValue(
                _anchored_scroll_value(
                    source_coordinate=source_anchor[0],
                    source_extent=self.base_pixmap.width(),
                    scaled_extent=pixmap.width(),
                    viewport_coordinate=float(viewport_anchor.x()),
                    viewport_extent=viewport.width(),
                )
            )
            vertical.setValue(
                _anchored_scroll_value(
                    source_coordinate=source_anchor[1],
                    source_extent=self.base_pixmap.height(),
                    scaled_extent=pixmap.height(),
                    viewport_coordinate=float(viewport_anchor.y()),
                    viewport_extent=viewport.height(),
                )
            )

        def set_zoom(
            self,
            zoom: float,
            *,
            source_anchor: tuple[float, float] | None = None,
            viewport_anchor=None,
        ) -> None:
            if self.base_pixmap.isNull():
                return
            viewport = self.scroll.viewport()
            if viewport_anchor is None:
                viewport_anchor = viewport.rect().center()
            if source_anchor is None:
                source_anchor = self._viewport_center_source()
            self.zoom = float(np.clip(zoom, 0.05, 12.0))
            self._apply_zoom()
            self._anchor_source(source_anchor, viewport_anchor)

        def _apply_zoom(self) -> None:
            if self.base_pixmap.isNull():
                return
            size = QtCore.QSize(
                max(1, int(round(self.base_pixmap.width() * self.zoom))),
                max(1, int(round(self.base_pixmap.height() * self.zoom))),
            )
            self.label.setPixmap(
                self.base_pixmap.scaled(
                    size,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
            self.label.resize(self.label.pixmap().size())
            self.zoom_label.setText(f"{self.zoom * 100:.0f}%")

        def fit(self) -> None:
            if self.base_pixmap.isNull():
                return
            viewport = self.scroll.viewport().size()
            self.set_zoom(
                min(
                    viewport.width() / self.base_pixmap.width(),
                    viewport.height() / self.base_pixmap.height(),
                )
                * 0.98,
                source_anchor=(
                    self.base_pixmap.width() / 2.0,
                    self.base_pixmap.height() / 2.0,
                ),
            )

        def fit_fish(self) -> bool:
            if self.base_pixmap.isNull():
                return False
            viewport = self.scroll.viewport().size()
            transform = _focus_transform(
                image_size=(self.base_pixmap.width(), self.base_pixmap.height()),
                viewport_size=(viewport.width(), viewport.height()),
                points=self.focus_points,
            )
            if transform is None:
                return False
            zoom, center = transform
            self.set_zoom(zoom, source_anchor=center)
            return True

        def eventFilter(self, watched, event):
            if watched is self.label and event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton and not self.base_pixmap.isNull():
                    x, y = self._source_from_label_position(event.pos())
                    self.clicked.emit(
                        self.view,
                        x,
                        y,
                    )
                    return True
            if watched is self.label and event.type() == QtCore.QEvent.Wheel:
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    source_anchor = self._source_from_label_position(event.pos())
                    viewport_anchor = self.label.mapTo(
                        self.scroll.viewport(), event.pos()
                    )
                    self.set_zoom(
                        self.zoom * (1.25 if event.angleDelta().y() > 0 else 0.8),
                        source_anchor=source_anchor,
                        viewport_anchor=viewport_anchor,
                    )
                    return True
            return super().eventFilter(watched, event)

    class AnnotationWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Female sex GT — {args.experiment}")
            self.resize(1740, 940)
            self.videos = dict(videos)
            self.slps = dict(slp_predictions)
            self.candidates = candidates.copy()
            self.ledger = ledger
            self.geometry = geometry
            self.n_frames = min(source.frame_count for source in self.videos.values())
            self.max_frame = self.n_frames - 1
            self.current_frame = 0
            self.current_candidate: dict[str, Any] = manual_jump_candidate(
                args.experiment, 0
            )
            self.current_candidate_position = -1
            self.filtered_candidate_indices: list[int] = []
            self.active_view = "side1_cam"
            self.selections = {view: empty_selection() for view in SIDE_VIEWS}
            self.current_status = ""
            self.dirty = False
            self.frame_ok = False
            self.instances_by_view: dict[str, np.ndarray | None] = {}
            self.geometry_matches = {view: {} for view in SIDE_VIEWS}
            self.projected_geometry = {view: {} for view in SIDE_VIEWS}
            self.last_rendered_frame: int | None = None
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(max(15, int(round(1000 / args.source_fps))))
            self.timer.timeout.connect(lambda: self._step_source(1, playback=True))
            self._build()
            self._install_shortcuts()
            self._refresh_saved_rows()
            self._refresh_queue()
            self._resume()

        def _build(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)

            main = QtWidgets.QVBoxLayout()
            root.addLayout(main, 5)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            main.addWidget(splitter, 1)
            self.panels = {view: ImagePanel(view) for view in SIDE_VIEWS}
            for panel in self.panels.values():
                panel.clicked.connect(self._select_view)
                splitter.addWidget(panel)
            splitter.setSizes([700, 700])
            self._update_active_panel()

            overlay_row = QtWidgets.QHBoxLayout()
            main.addLayout(overlay_row)
            self.raw_toggle = QtWidgets.QCheckBox("Raw frame: hide every overlay [H]")
            self.sleap_toggle = QtWidgets.QCheckBox("Neutral raw SLEAP poses")
            self.sleap_toggle.setChecked(True)
            self.geometry_toggle = QtWidgets.QCheckBox("Geometry IDs (off by default)")
            self.geometry_toggle.setEnabled(self.geometry is not None)
            for toggle in (self.raw_toggle, self.sleap_toggle, self.geometry_toggle):
                toggle.toggled.connect(self._render)
                overlay_row.addWidget(toggle)
            overlay_row.addStretch(1)

            nav_row = QtWidgets.QHBoxLayout()
            main.addLayout(nav_row)
            self.play_button = QtWidgets.QPushButton("Play [Space]")
            self.play_button.clicked.connect(self._toggle_play)
            nav_row.addWidget(self.play_button)
            for label, amount in (("−100", -100), ("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10), ("+100", 100)):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=amount: self._step_source(value))
                nav_row.addWidget(button)
            nav_row.addWidget(QtWidgets.QLabel("Exact source frame:"))
            self.frame_spin = QtWidgets.QSpinBox()
            self.frame_spin.setRange(0, self.max_frame)
            self.frame_spin.setKeyboardTracking(False)
            self.frame_spin.editingFinished.connect(self._manual_spin_jump)
            nav_row.addWidget(self.frame_spin)
            self.frame_label = QtWidgets.QLabel()
            nav_row.addWidget(self.frame_label)
            nav_row.addStretch(1)

            self.context_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.context_slider.valueChanged.connect(self._slider_navigate)
            main.addWidget(self.context_slider)
            self.context_label = QtWidgets.QLabel()
            main.addWidget(self.context_label)

            sidebar = QtWidgets.QVBoxLayout()
            root.addLayout(sidebar, 2)
            title = QtWidgets.QLabel("Candidate queue")
            title.setStyleSheet("font-size:16px;font-weight:bold")
            sidebar.addWidget(title)
            queue_nav = QtWidgets.QHBoxLayout()
            sidebar.addLayout(queue_nav)
            previous = QtWidgets.QPushButton("Previous [P]")
            following = QtWidgets.QPushButton("Next [N]")
            previous.clicked.connect(lambda: self._go_candidate_position(self.current_candidate_position - 1))
            following.clicked.connect(lambda: self._go_candidate_position(self.current_candidate_position + 1))
            queue_nav.addWidget(previous)
            queue_nav.addWidget(following)
            self.filter_combo = QtWidgets.QComboBox()
            self.filter_combo.addItems(
                ["all", "unreviewed", "confirmed", "ambiguous", "not_visible", "bad_frame"]
            )
            self.filter_combo.currentTextChanged.connect(self._refresh_queue)
            sidebar.addWidget(self.filter_combo)
            self.candidate_combo = QtWidgets.QComboBox()
            self.candidate_combo.activated.connect(self._go_candidate_position)
            sidebar.addWidget(self.candidate_combo)
            self.candidate_info = QtWidgets.QLabel()
            self.candidate_info.setWordWrap(True)
            sidebar.addWidget(self.candidate_info)
            self.progress_label = QtWidgets.QLabel()
            sidebar.addWidget(self.progress_label)

            sidebar.addWidget(self._line())
            sidebar.addWidget(QtWidgets.QLabel("Saved annotation rows (double-click/jump):"))
            self.saved_combo = QtWidgets.QComboBox()
            self.saved_combo.activated.connect(self._jump_saved_row)
            sidebar.addWidget(self.saved_combo)

            sidebar.addWidget(self._line())
            annotator_label = QtWidgets.QLabel(f"Annotator: {args.annotator}")
            annotator_label.setStyleSheet("font-weight:bold")
            sidebar.addWidget(annotator_label)
            self.selection_label = QtWidgets.QLabel()
            self.selection_label.setWordWrap(True)
            sidebar.addWidget(self.selection_label)
            clear_button = QtWidgets.QPushButton("Clear active-view selection [U]")
            clear_button.clicked.connect(self._clear_active_selection)
            sidebar.addWidget(clear_button)
            self.warning_label = QtWidgets.QLabel()
            self.warning_label.setWordWrap(True)
            self.warning_label.setStyleSheet(
                "background:#5b1515;color:#fff;padding:8px;font-weight:bold"
            )
            self.warning_label.hide()
            sidebar.addWidget(self.warning_label)

            sidebar.addWidget(QtWidgets.QLabel("Decision (saved immediately):"))
            decision_grid = QtWidgets.QGridLayout()
            sidebar.addLayout(decision_grid)
            for index, (status, label) in enumerate(
                (
                    ("confirmed", "1  Confirmed F"),
                    ("ambiguous", "2  Ambiguous"),
                    ("not_visible", "3  Not visible"),
                    ("bad_frame", "4  Bad frame"),
                )
            ):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, value=status: self._save_decision(value)
                )
                decision_grid.addWidget(button, index // 2, index % 2)
            self.status_label = QtWidgets.QLabel("Unreviewed")
            self.status_label.setStyleSheet("font-size:15px;font-weight:bold")
            sidebar.addWidget(self.status_label)
            sidebar.addWidget(QtWidgets.QLabel("Notes:"))
            self.notes = QtWidgets.QPlainTextEdit()
            self.notes.setMaximumHeight(130)
            self.notes.textChanged.connect(self._mark_dirty)
            sidebar.addWidget(self.notes)
            save_button = QtWidgets.QPushButton("Save now [Ctrl+S]")
            save_button.clicked.connect(self._save_now)
            undo_button = QtWidgets.QPushButton("Undo last session edit [Ctrl+Z]")
            undo_button.clicked.connect(self._undo)
            sidebar.addWidget(save_button)
            sidebar.addWidget(undo_button)
            sidebar.addStretch(1)

        @staticmethod
        def _line():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            return line

        def _install_shortcuts(self) -> None:
            shortcuts = {
                "Left": lambda: self._step_source(-1),
                "Right": lambda: self._step_source(1),
                "Shift+Left": lambda: self._step_source(-10),
                "Shift+Right": lambda: self._step_source(10),
                "Ctrl+Left": lambda: self._step_source(-100),
                "Ctrl+Right": lambda: self._step_source(100),
                "P": lambda: self._go_candidate_position(self.current_candidate_position - 1),
                "N": lambda: self._go_candidate_position(self.current_candidate_position + 1),
                "Space": self._toggle_play,
                "J": self._focus_frame_entry,
                "H": lambda: self.raw_toggle.toggle(),
                "U": self._clear_active_selection,
                "1": lambda: self._save_decision("confirmed"),
                "2": lambda: self._save_decision("ambiguous"),
                "3": lambda: self._save_decision("not_visible"),
                "4": lambda: self._save_decision("bad_frame"),
                "Ctrl+Z": self._undo,
                "Ctrl+S": self._save_now,
            }
            self._shortcuts = []
            for sequence, callback in shortcuts.items():
                shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
                shortcut.setContext(QtCore.Qt.WindowShortcut)
                if sequence in {"Ctrl+Z", "Ctrl+S"}:
                    shortcut.activated.connect(callback)
                else:
                    shortcut.activated.connect(
                        lambda checked=False, fn=callback: self._run_shortcut(fn)
                    )
                self._shortcuts.append(shortcut)

        def _run_shortcut(self, callback) -> None:
            focus = QtWidgets.QApplication.focusWidget()
            if isinstance(
                focus,
                (
                    QtWidgets.QLineEdit,
                    QtWidgets.QTextEdit,
                    QtWidgets.QPlainTextEdit,
                    QtWidgets.QAbstractSpinBox,
                ),
            ):
                return
            callback()

        def _focus_frame_entry(self) -> None:
            self.frame_spin.setFocus()
            self.frame_spin.selectAll()

        def _mark_dirty(self) -> None:
            self.dirty = True

        def _resume(self) -> None:
            latest = latest_annotated_frame(
                self.ledger.rows, experiment=args.experiment, annotator=args.annotator
            )
            if latest is not None:
                saved = self.ledger.get(args.experiment, latest, args.annotator)
                candidate_frame = saved.get("candidate_frame", "") if saved else ""
                candidate_match = None
                if candidate_frame != "":
                    matches = self.candidates.index[
                        self.candidates["frame"].astype(int).eq(int(float(candidate_frame)))
                    ].tolist()
                    candidate_match = matches[0] if matches else None
                if candidate_match is not None:
                    positions = [
                        pos
                        for pos, actual_index in enumerate(self.filtered_candidate_indices)
                        if actual_index == candidate_match
                    ]
                    if positions:
                        self.current_candidate_position = positions[0]
                        self.candidate_combo.setCurrentIndex(positions[0])
                    self._navigate_to(
                        latest,
                        candidate=candidate_from_annotation(saved),
                        save_before=False,
                    )
                else:
                    self._navigate_to(
                        latest,
                        candidate=candidate_from_annotation(saved),
                        save_before=False,
                    )
                return
            if self.filtered_candidate_indices:
                self._go_candidate_position(0)
            else:
                self._navigate_to(
                    0,
                    candidate=manual_jump_candidate(args.experiment, 0),
                    save_before=False,
                )

        def _status_for_candidate_index(self, index: int) -> str:
            return queue_status(
                self.candidates.loc[index].to_dict(),
                self.ledger.rows,
                experiment=args.experiment,
                annotator=args.annotator,
            )

        def _refresh_queue(self, *_args) -> None:
            wanted = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "all"
            indices = []
            statuses = {}
            for index in self.candidates.index:
                status = self._status_for_candidate_index(int(index))
                statuses[int(index)] = status
                if wanted == "all" or wanted == status:
                    indices.append(int(index))
            previous_index = (
                self.filtered_candidate_indices[self.current_candidate_position]
                if self.filtered_candidate_indices
                and 0 <= self.current_candidate_position < len(self.filtered_candidate_indices)
                else None
            )
            self.filtered_candidate_indices = indices
            self.candidate_combo.blockSignals(True)
            self.candidate_combo.clear()
            for index in indices:
                row = self.candidates.loc[index]
                self.candidate_combo.addItem(
                    f"{int(row['frame']):,}  [{statuses[index]}]  {row['candidate_sources']}"
                )
            if previous_index in indices:
                self.current_candidate_position = indices.index(previous_index)
                self.candidate_combo.setCurrentIndex(self.current_candidate_position)
            elif indices:
                self.current_candidate_position = -1
                self.candidate_combo.setCurrentIndex(0)
            else:
                self.current_candidate_position = -1
            self.candidate_combo.blockSignals(False)
            counts = {status: 0 for status in ("unreviewed", "confirmed", "ambiguous", "not_visible", "bad_frame")}
            for status in statuses.values():
                counts[status] = counts.get(status, 0) + 1
            reviewed = len(self.candidates) - counts["unreviewed"]
            self.progress_label.setText(
                f"Reviewed {reviewed:,}/{len(self.candidates):,} · "
                + " · ".join(f"{key} {value}" for key, value in counts.items())
            )

        def _refresh_saved_rows(self) -> None:
            rows = self.ledger.rows
            if not rows.empty:
                rows = rows[
                    rows["experiment"].eq(args.experiment)
                    & rows["annotator"].eq(args.annotator)
                ].copy()
                if not rows.empty:
                    rows["__frame"] = pd.to_numeric(rows["frame"]).astype(int)
                    rows = rows.sort_values("__frame")
            self._saved_frame_values = []
            self.saved_combo.blockSignals(True)
            self.saved_combo.clear()
            for row in rows.to_dict("records") if not rows.empty else []:
                frame = int(row["frame"])
                self._saved_frame_values.append(frame)
                self.saved_combo.addItem(f"{frame:,}  [{row['gt_status']}]")
            self.saved_combo.blockSignals(False)

        def _jump_saved_row(self, position: int) -> None:
            if 0 <= position < len(self._saved_frame_values):
                frame = self._saved_frame_values[position]
                saved = self.ledger.get(args.experiment, frame, args.annotator)
                self._navigate_to(
                    frame,
                    candidate=(
                        candidate_from_annotation(saved)
                        if saved is not None
                        else manual_jump_candidate(args.experiment, frame)
                    ),
                )

        def _go_candidate_position(self, position: int) -> None:
            if not self.filtered_candidate_indices:
                return
            position = max(0, min(int(position), len(self.filtered_candidate_indices) - 1))
            self.current_candidate_position = position
            self.candidate_combo.blockSignals(True)
            self.candidate_combo.setCurrentIndex(position)
            self.candidate_combo.blockSignals(False)
            self._go_candidate_index(self.filtered_candidate_indices[position])

        def _go_candidate_index(self, index: int, target_frame: int | None = None) -> None:
            candidate = self.candidates.loc[int(index)].to_dict()
            positions = [
                pos for pos, actual_index in enumerate(self.filtered_candidate_indices)
                if actual_index == int(index)
            ]
            if positions:
                self.current_candidate_position = positions[0]
                self.candidate_combo.setCurrentIndex(positions[0])
            self._navigate_to(
                int(candidate["frame"]) if target_frame is None else int(target_frame),
                candidate=candidate,
            )

        def _manual_spin_jump(self) -> None:
            frame = int(self.frame_spin.value())
            if frame != self.current_frame:
                self._navigate_to(
                    frame, candidate=manual_jump_candidate(args.experiment, frame)
                )

        def _slider_navigate(self, frame: int) -> None:
            if int(frame) != self.current_frame:
                self._navigate_to(int(frame))

        def _step_source(self, amount: int, *, playback: bool = False) -> None:
            target = self.current_frame + int(amount)
            if playback and target > self.context_slider.maximum():
                self._toggle_play()
                return
            self._navigate_to(max(0, min(target, self.max_frame)))

        def _navigate_to(
            self,
            frame: int,
            *,
            candidate: Mapping[str, Any] | None = None,
            save_before: bool = True,
        ) -> None:
            frame = int(frame)
            if frame < 0 or frame > self.max_frame:
                self._error(f"Source frame {frame} is outside 0..{self.max_frame}")
                return
            if save_before and not self._save_if_reviewed():
                return
            if candidate is not None:
                self.current_candidate = dict(candidate)
                center = int(candidate["frame"])
                low = max(0, center - int(args.context_frames))
                high = min(self.max_frame, center + int(args.context_frames))
                self.context_slider.blockSignals(True)
                self.context_slider.setRange(low, high)
                self.context_slider.blockSignals(False)
                self.context_label.setText(
                    f"Context around queue/manual origin {center:,}: {low:,}..{high:,}"
                )
            elif not (self.context_slider.minimum() <= frame <= self.context_slider.maximum()):
                low = max(0, frame - int(args.context_frames))
                high = min(self.max_frame, frame + int(args.context_frames))
                self.context_slider.setRange(low, high)
                self.context_label.setText(f"Extended context: {low:,}..{high:,}")
            self.current_frame = frame
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(frame)
            self.frame_spin.blockSignals(False)
            self.context_slider.blockSignals(True)
            self.context_slider.setValue(frame)
            self.context_slider.blockSignals(False)
            self._load_annotation()
            self._render()

        def _load_annotation(self) -> None:
            row = self.ledger.get(args.experiment, self.current_frame, args.annotator)
            self.notes.blockSignals(True)
            if row is None:
                self.selections = {view: empty_selection() for view in SIDE_VIEWS}
                self.current_status = ""
                self.notes.setPlainText("")
            else:
                self.selections = {
                    view: _selection_from_saved(row, VIEW_PREFIX[view])
                    for view in SIDE_VIEWS
                }
                self.current_status = str(row["gt_status"])
                self.notes.setPlainText(str(row.get("notes", "")))
            self.notes.blockSignals(False)
            self.dirty = False
            self._update_annotation_labels()

        def _render(self, *_args) -> None:
            self.frame_ok = False
            try:
                auto_fit = self.last_rendered_frame != self.current_frame
                images = {view: self.videos[view].get(self.current_frame) for view in SIDE_VIEWS}
                self.instances_by_view = {
                    view: self.slps[view].get(self.current_frame, default=None)
                    for view in SIDE_VIEWS
                }
                if self.geometry is not None:
                    self.geometry_matches, self.projected_geometry = self.geometry.associate(
                        self.current_frame, self.instances_by_view, args.match_gate_px
                    )
                else:
                    self.geometry_matches = {view: {} for view in SIDE_VIEWS}
                    self.projected_geometry = {view: {} for view in SIDE_VIEWS}
                raw_only = self.raw_toggle.isChecked()
                for view in SIDE_VIEWS:
                    image = images[view]
                    if not raw_only:
                        if self.sleap_toggle.isChecked():
                            _draw_raw_sleap(image, self.instances_by_view[view])
                        if self.geometry_toggle.isChecked() and self.geometry is not None:
                            _draw_geometry_labels(image, self.projected_geometry[view])
                        _draw_selection(image, self.selections[view])
                    if self.geometry is not None:
                        focus_points = _three_fish_focus_points(
                            self.instances_by_view[view],
                            self.geometry_matches[view],
                            self.projected_geometry[view],
                        )
                    else:
                        focus_points = _three_fish_focus_points(
                            self.instances_by_view[view]
                        )
                    self.panels[view].set_image(
                        image,
                        focus_points=focus_points,
                        auto_fit=auto_fit,
                    )
                self.last_rendered_frame = self.current_frame
                self.frame_ok = True
                self.frame_label.setText(
                    f"frame {self.current_frame:,} / {self.max_frame:,}  ·  "
                    f"{self.current_frame / args.source_fps:.6f} s"
                )
                self.candidate_info.setText(
                    f"Origin frame: {int(self.current_candidate['frame']):,}\n"
                    f"IDs: {self.current_candidate.get('candidate_ids', '')}\n"
                    f"Sources: {self.current_candidate.get('candidate_sources', '')}"
                )
                self._update_annotation_labels()
            except Exception as error:
                self.frame_ok = False
                self._error(str(error))

        def _select_view(self, view: str, x: float, y: float) -> None:
            self.active_view = view
            self._update_active_panel()
            self.selections[view] = selection_from_click(
                experiment=args.experiment,
                view=view,
                frame=self.current_frame,
                click_xy=(x, y),
                instances=self.instances_by_view.get(view),
                gate_px=args.match_gate_px,
                instance_geometry_ids=self.geometry_matches.get(view),
                node_names=getattr(self.slps[view], "node_names", ("mouth", "tail")),
            )
            tracklet_map, _source_mode = candidate_tracklet_context(
                self.current_candidate
            )
            geometry_id = str(self.selections[view].get("geometry_id", "") or "")
            if geometry_id in tracklet_map:
                self.selections[view]["tracklet_id"] = tracklet_map[geometry_id]
            self.dirty = True
            self._update_annotation_labels()
            self._render()

        def _update_active_panel(self) -> None:
            for view, panel in self.panels.items():
                panel.set_active(view == self.active_view)

        def _clear_active_selection(self) -> None:
            self.selections[self.active_view] = empty_selection()
            self.dirty = True
            self._update_annotation_labels()
            self._render()

        def _update_annotation_labels(self) -> None:
            descriptions = []
            for view in SIDE_VIEWS:
                selection = self.selections[view]
                status = selection.get("selection_status", "not_selected")
                geometry_id = selection.get("geometry_id", "")
                tracklet_id = selection.get("tracklet_id", "")
                suffix = f", geometry {geometry_id}" if geometry_id else ""
                if tracklet_id:
                    suffix += f", segment {tracklet_id}"
                descriptions.append(f"{view}: {status}{suffix}")
            self.selection_label.setText("\n".join(descriptions))
            consistency, _consensus = geometry_consistency(
                self.selections["side1_cam"].get("geometry_id", ""),
                self.selections["side2_cam"].get("geometry_id", ""),
            )
            if consistency == "disagree":
                self.warning_label.setText(
                    "TRACKER DISAGREEMENT: the two human selections map to different "
                    "geometry IDs. This is retained as valuable swap evidence."
                )
                self.warning_label.show()
            else:
                self.warning_label.hide()
            self.status_label.setText(
                f"Saved decision: {self.current_status}"
                if self.current_status
                else "Unreviewed"
            )

        def _build_row(self, status: str) -> dict[str, Any]:
            if not self.frame_ok:
                raise ValueError("cannot save because the synchronized source frame did not load")
            calibration_backend = self.geometry.backend if self.geometry else ""
            model_files = self.geometry.model_files if self.geometry else {}
            return make_annotation_row(
                experiment=args.experiment,
                frame=self.current_frame,
                source_fps=args.source_fps,
                annotator=args.annotator,
                gt_status=status,
                side1_selection=self.selections["side1_cam"],
                side2_selection=self.selections["side2_cam"],
                candidate=self.current_candidate,
                video_paths={view: self.videos[view].path for view in SIDE_VIEWS},
                slp_paths={view: self.slps[view].path for view in SIDE_VIEWS},
                geometry_csv_path=self.geometry.path if self.geometry else None,
                calibration_backend=calibration_backend,
                calibration_model_files=model_files,
                notes=self.notes.toPlainText(),
            )

        def _save_decision(self, status: str, *, quiet: bool = False) -> bool:
            try:
                saved = self.ledger.upsert(self._build_row(status))
            except Exception as error:
                if not quiet:
                    self._error(str(error))
                return False
            self.current_status = str(saved["gt_status"])
            self.dirty = False
            self._update_annotation_labels()
            self._refresh_saved_rows()
            self._refresh_queue()
            return True

        def _save_if_reviewed(self) -> bool:
            if self.dirty and self.current_status:
                return self._save_decision(self.current_status)
            return True

        def _save_now(self) -> None:
            if not self.current_status:
                self._error("Choose one of the four explicit decisions before saving.")
                return
            self._save_decision(self.current_status)

        def _undo(self) -> None:
            if not self.ledger.can_undo:
                return
            previous = self.ledger.undo()
            frame = (
                int(previous["frame"])
                if previous is not None
                else int(self.ledger.last_undo_key[1])
            )
            self._refresh_saved_rows()
            self._refresh_queue()
            self._navigate_to(
                frame,
                candidate=(
                    candidate_from_annotation(previous)
                    if previous is not None
                    else manual_jump_candidate(args.experiment, frame)
                ),
                save_before=False,
            )

        def _toggle_play(self) -> None:
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText("Play [Space]")
            else:
                self.timer.start()
                self.play_button.setText("Pause [Space]")

        def _error(self, message: str) -> None:
            QtWidgets.QMessageBox.critical(self, "Female sex GT", message)

        def closeEvent(self, event) -> None:
            if not self._save_if_reviewed():
                event.ignore()
                return
            self.timer.stop()
            event.accept()

    return AnnotationWindow()


def run_gui(
    args: argparse.Namespace,
    *,
    videos: Mapping[str, Any],
    slp_predictions: Mapping[str, Any],
    candidates: pd.DataFrame,
    ledger: AnnotationLedger,
    geometry: GeometryAssociator | None,
    smoke_actions: bool = False,
) -> int:
    _QtCore, _QtGui, QtWidgets = _load_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = create_annotation_window(
        args,
        videos=videos,
        slp_predictions=slp_predictions,
        candidates=candidates,
        ledger=ledger,
        geometry=geometry,
    )
    window.show()
    if smoke_actions:
        QtCore, _QtGui, _QtWidgets = _load_qt()

        def exercise() -> None:
            try:
                window._go_candidate_position(0)
                panel = window.panels["side1_cam"]
                if len(panel.focus_points) < 6 or not panel.fit_fish():
                    raise RuntimeError("synthetic three-fish auto-fit was not available")
                viewport_anchor = panel.scroll.viewport().rect().center()
                source_anchor = (160.0, 100.0)
                panel.set_zoom(
                    panel.zoom * 1.25,
                    source_anchor=source_anchor,
                    viewport_anchor=viewport_anchor,
                )
                QtWidgets.QApplication.processEvents()
                pixmap = panel.label.pixmap()
                label_anchor = QtCore.QPoint(
                    round(source_anchor[0] * pixmap.width() / panel.base_pixmap.width()),
                    round(source_anchor[1] * pixmap.height() / panel.base_pixmap.height()),
                )
                actual_anchor = panel.label.mapTo(panel.scroll.viewport(), label_anchor)
                if (
                    abs(actual_anchor.x() - viewport_anchor.x()) > 2
                    or abs(actual_anchor.y() - viewport_anchor.y()) > 2
                ):
                    raise RuntimeError("pointer-centered zoom did not preserve its anchor")
                window._select_view("side1_cam", 78.0, 70.0)
                if not window._save_decision("confirmed"):
                    raise RuntimeError("synthetic decision did not save")
                window._step_source(1)
                window._go_candidate_position(1)
                window._render()
                if not ledger.csv_path.exists() or len(ledger.rows) != 1:
                    raise RuntimeError("synthetic smoke ledger was not written")
                window.close()
                app.quit()
            except Exception as error:
                print(f"female-sex GT smoke test failed: {error}", file=sys.stderr)
                window.close()
                app.exit(2)

        QtCore.QTimer.singleShot(80, exercise)
    return int(app.exec_())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--candidate-csv", type=Path, action="append", default=[])
    parser.add_argument(
        "--build-candidates",
        action="store_true",
        help="build/rebuild the clean three-view queue before opening the GUI",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="open without a candidate queue and annotate exact manual frames",
    )
    parser.add_argument("--candidate-output-csv", type=Path)
    parser.add_argument("--candidate-summary-json", type=Path)
    parser.add_argument("--candidate-clean-tracklets-csv", type=Path)
    parser.add_argument("--reconstruction-provenance-json", type=Path)
    parser.add_argument("--candidate-target-count", type=int, default=300)
    parser.add_argument("--candidate-min-frame-spacing", type=int, default=166)
    parser.add_argument(
        "--candidate-min-view-separation-px", type=float, default=40.0
    )
    parser.add_argument("--candidate-border-margin-px", type=float, default=20.0)
    parser.add_argument("--candidate-max-local-jump-mm", type=float, default=8.0)
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--slp-dir", type=Path)
    parser.add_argument("--geometry-csv", type=Path)
    parser.add_argument("--geometry-id-column")
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument(
        "--calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default="auto",
    )
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--annotator", default="")
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--context-frames", type=int, default=166)
    parser.add_argument("--match-gate-px", type=float, default=80.0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _resolved_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    local = load_local_paths(args.paths_config)
    if not args.annotator.strip():
        if args.smoke_test:
            args.annotator = "smoke-test"
        else:
            parser.error("--annotator NAME is required")
    if args.source_fps <= 0:
        parser.error("--source-fps must be positive")
    if args.context_frames < 0:
        parser.error("--context-frames must be non-negative")
    if args.match_gate_px < 0:
        parser.error("--match-gate-px must be non-negative")
    if args.annotations_csv is None and not args.smoke_test:
        args.annotations_csv = (
            REPO_ROOT
            / "annotations"
            / "fish_sex_gt"
            / args.experiment
            / "female_sex_gt.csv"
        )
    if args.candidate_output_csv is None and not args.smoke_test:
        args.candidate_output_csv = (
            REPO_ROOT
            / "annotations"
            / "fish_sex_gt"
            / args.experiment
            / "clean_three_view_candidates.csv"
        )
    if args.candidate_summary_json is None and args.candidate_output_csv is not None:
        args.candidate_summary_json = args.candidate_output_csv.with_name(
            args.candidate_output_csv.stem + "_summary.json"
        )
    if args.audit_jsonl is None and args.annotations_csv is not None:
        args.audit_jsonl = args.annotations_csv.with_suffix(".audit.jsonl")
    if args.video_dir is None and local is not None and local.medaka_data_dir is not None:
        args.video_dir = local.medaka_data_dir / args.experiment
    if args.slp_dir is None and local is not None and local.sleap_output_dir is not None:
        args.slp_dir = local.sleap_output_dir / args.experiment
    if (
        args.calibration_model_dir is None
        and local is not None
        and local.calibration_model_dir is not None
    ):
        args.calibration_model_dir = local.calibration_model_dir
    return args


def _ensure_outputs_are_not_sources(args: argparse.Namespace, source_paths: Sequence[Path]) -> None:
    outputs = [Path(args.annotations_csv), Path(args.audit_jsonl)]
    if normalize_path(outputs[0]) == normalize_path(outputs[1]):
        raise ValueError("--annotations-csv and --audit-jsonl must be different files")
    protected = {normalize_path(path) for path in source_paths}
    for output in outputs:
        if normalize_path(output) in protected:
            raise ValueError(f"refusing to overwrite a source/candidate file: {output}")


def _require_experiment_in_path(path: Path, experiment: str, label: str) -> None:
    """Block detectable cross-experiment path mistakes before any annotation save."""
    parts = {part.casefold() for part in path.resolve(strict=False).parts}
    if experiment.casefold() not in parts:
        raise ValueError(
            f"{label} does not contain the requested experiment directory "
            f"{experiment!r}: {path}"
        )


def _append_candidate_path(args: argparse.Namespace, path: Path) -> None:
    normalized = normalize_path(path)
    if all(normalize_path(existing) != normalized for existing in args.candidate_csv):
        args.candidate_csv.append(path)


def _prepare_candidate_queue(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Build the selector queue explicitly, or reuse its default output."""
    if args.manual_only:
        if args.build_candidates or args.candidate_csv:
            parser.error(
                "--manual-only cannot be combined with --build-candidates or "
                "--candidate-csv"
            )
        return
    if args.build_candidates:
        if args.geometry_csv is None:
            parser.error("--geometry-csv is required with --build-candidates")
        if args.calibration_model_dir is None:
            parser.error(
                "--calibration-model-dir is required with --build-candidates"
            )
        if args.candidate_output_csv is None or args.candidate_summary_json is None:
            parser.error("candidate output paths could not be resolved")
        config = CandidateConfig(
            target_count=args.candidate_target_count,
            min_frame_spacing=args.candidate_min_frame_spacing,
            match_gate_px=args.match_gate_px,
            min_view_separation_px=args.candidate_min_view_separation_px,
            border_margin_px=args.candidate_border_margin_px,
            max_local_jump_mm=args.candidate_max_local_jump_mm,
            seed=args.candidate_seed,
        )
        try:
            config.validate()
        except ValueError as error:
            parser.error(str(error))
        print(
            "Building clean three-view candidate queue before annotation; "
            "this scans the full synchronized recording.",
            flush=True,
        )
        summary = run_candidate_selector(
            experiment=args.experiment,
            clean_tracklets_csv=args.candidate_clean_tracklets_csv,
            geometry_csv=Path(args.geometry_csv),
            reconstruction_provenance_json=args.reconstruction_provenance_json,
            geometry_id_column=args.geometry_id_column,
            slp_dir=Path(args.slp_dir),
            video_dir=Path(args.video_dir),
            calibration_model_dir=Path(args.calibration_model_dir),
            calibration_backend=args.calibration_backend,
            output_csv=Path(args.candidate_output_csv),
            summary_json=Path(args.candidate_summary_json),
            config=config,
        )
        print(
            f"Candidate queue ready: {summary['selected_candidate_count']} selected "
            f"from {summary['eligible_frame_count']} eligible frames.",
            flush=True,
        )
        _append_candidate_path(args, Path(args.candidate_output_csv))
        return

    if args.candidate_csv:
        return
    default_candidate = Path(args.candidate_output_csv)
    if default_candidate.is_file():
        print(f"Reusing candidate queue: {default_candidate}", flush=True)
        _append_candidate_path(args, default_candidate)
        return
    parser.error(
        "no candidate queue was supplied or found; pass --build-candidates for "
        "the first run, or pass --candidate-csv PATH"
    )


def run_smoke_test(args: argparse.Namespace) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    temporary_context = None
    if args.annotations_csv is None:
        temporary_context = tempfile.TemporaryDirectory(prefix="female-sex-gt-smoke-")
        root = Path(temporary_context.name)
        args.annotations_csv = root / "female_sex_gt.csv"
        args.audit_jsonl = root / "female_sex_gt.audit.jsonl"
    elif args.audit_jsonl is None:
        args.audit_jsonl = Path(args.annotations_csv).with_suffix(".audit.jsonl")
    args.annotator = args.annotator.strip() or "smoke-test"
    videos = {view: SyntheticVideo(view) for view in SIDE_VIEWS}
    slps = {view: SyntheticSlp(view) for view in SIDE_VIEWS}
    candidates = normalize_candidate_table(
        pd.DataFrame(
            {
                "center_frame": [2, 7],
                "candidate_id": ["smoke-a", "smoke-b"],
                "source": ["synthetic", "synthetic"],
                "fixture_note": ["first", "second"],
            }
        ),
        experiment=args.experiment,
        max_frame=11,
        source_name="<memory>",
    )
    ledger = AnnotationLedger(args.annotations_csv, args.audit_jsonl)
    try:
        result = run_gui(
            args,
            videos=videos,
            slp_predictions=slps,
            candidates=candidates,
            ledger=ledger,
            geometry=None,
            smoke_actions=True,
        )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke_test:
        args = _resolved_args(args, parser)
        return run_smoke_test(args)
    args = _resolved_args(args, parser)
    if args.video_dir is None:
        parser.error("--video-dir is required when it cannot be resolved from paths config")
    if args.slp_dir is None:
        parser.error("--slp-dir is required when it cannot be resolved from paths config")
    if args.geometry_csv is not None and args.calibration_model_dir is None:
        parser.error("--calibration-model-dir is required with --geometry-csv")

    _require_experiment_in_path(Path(args.video_dir), args.experiment, "--video-dir")
    _require_experiment_in_path(Path(args.slp_dir), args.experiment, "--slp-dir")
    if args.geometry_csv is not None:
        _require_experiment_in_path(
            Path(args.geometry_csv), args.experiment, "--geometry-csv"
        )
    _prepare_candidate_queue(args, parser)

    video_paths = {
        view: Path(args.video_dir) / view / VIDEO_NAMES[view] for view in SIDE_VIEWS
    }
    slp_paths = {view: Path(args.slp_dir) / f"{view}.slp" for view in SIDE_VIEWS}
    missing = [str(path) for path in (*video_paths.values(), *slp_paths.values()) if not path.is_file()]
    if missing:
        raise SystemExit("missing required read-only inputs:\n  " + "\n  ".join(missing))
    source_paths = [*video_paths.values(), *slp_paths.values(), *args.candidate_csv]
    if args.geometry_csv is not None:
        source_paths.append(args.geometry_csv)
    _ensure_outputs_are_not_sources(args, source_paths)

    videos: dict[str, OpenCvVideo] = {}
    slps: dict[str, IndexedSlpPredictions] = {}
    try:
        videos = {view: OpenCvVideo(path) for view, path in video_paths.items()}
        counts = {view: source.frame_count for view, source in videos.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"side video source-frame counts disagree: {counts}")
        max_frame = next(iter(counts.values())) - 1
        for view, path in slp_paths.items():
            source = IndexedSlpPredictions(path, node_names=("mouth", "tail"))
            indices = np.asarray(source._frame_indices, dtype=np.int64)
            if len(indices) and (int(indices.min()) < 0 or int(indices.max()) > max_frame):
                source.close()
                raise ValueError(
                    f"{view} SLP source-frame extent {int(indices.min())}..{int(indices.max())} "
                    f"does not fit video 0..{max_frame}"
                )
            source.max_source_frame = int(indices.max()) if len(indices) else -1
            slps[view] = source
        candidates = load_and_merge_candidate_csvs(
            args.candidate_csv,
            experiment=args.experiment,
            max_frame=max_frame,
        )
        geometry = (
            GeometryAssociator(
                args.geometry_csv,
                args.calibration_model_dir,
                backend=args.calibration_backend,
                id_column=args.geometry_id_column,
                max_frame=max_frame,
            )
            if args.geometry_csv is not None
            else None
        )
        ledger = AnnotationLedger(args.annotations_csv, args.audit_jsonl)
        return run_gui(
            args,
            videos=videos,
            slp_predictions=slps,
            candidates=candidates,
            ledger=ledger,
            geometry=geometry,
        )
    finally:
        for source in videos.values():
            source.close()
        for source in slps.values():
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
