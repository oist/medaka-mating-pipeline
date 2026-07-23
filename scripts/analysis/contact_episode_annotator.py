"""Synchronized three-view contact-episode detector and annotation GUI.

The upper half shows the same exact source frame from all three raw videos. The
lower half shows aligned 3D IFD and switchable raw-SLEAP 2D displacement/IFD
traces. Human Anchor A/B clicks are canonical; all SLEAP and geometry links are
provenance.
Run with ``--preflight`` to validate inputs and create detector artifacts without
opening an interactive window.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.analysis.contact_episode_gt import (
    CANDIDATE_COLUMNS,
    CONTACT_VALIDITIES,
    EVIDENCE_QUALITIES,
    IDENTITY_TARGETS,
    PHASES,
    TRACKER_ASSESSMENTS,
    VIEWS,
    ContactAnnotationLedger,
    DetectorConfig,
    TrackingIFD,
    deserialize_clicks,
    detect_contact_episodes,
    empty_clicks,
    make_annotation_row,
    make_click_evidence,
    manual_episode,
    pair_colors_for,
    parse_tracking_csv_specs,
    prepare_tracking_ifd,
    queue_decision_state,
    stable_json,
    suggest_anchor_frames,
    suggest_anchor_mapping,
    utc_now,
    validate_false_negative_detections,
    validate_false_positive_detections,
    write_candidate_artifacts,
)
from scripts.analysis.female_sex_gt import hit_test_sleap_pose
from scripts.analysis.female_sex_gt_annotator import OpenCvVideo
from scripts.analysis.geometry_swap_review import IndexedSlpPredictions, VIDEO_NAMES
from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, REPO_ROOT, load_local_paths


CAMERA_INDEX = {"top_cam": 0, "side1_cam": 1, "side2_cam": 2}
RAW_SLEAP_COLOR = (165, 165, 165)
FISH_ID_COLORS = ("#4e79a7", "#59a14f", "#b07aa1")
ACTIVE_CLICK_COLORS = {
    "female": (0, 215, 255),
    "anchor_male": (80, 220, 80),
}
CONTACT_VALIDITY_LABELS = {
    "confirmed_contact_crossing": "Confirmed contact/crossing",
    "near_miss": "Near miss",
    "false_contact_candidate": "False contact candidate",
    "uncertain": "Uncertain",
}
TRACKER_ASSESSMENT_LABELS = {
    "correct_continuity": "Correct continuity",
    "permanent_id_swap": "Permanent ID swap",
    "temporary_swap_flicker": "Temporary swap/flicker",
    "motion_noise_no_identity_swap": "Motion noise but no identity swap",
    "observations_unusable": "Observations unusable",
    "uncertain": "Uncertain",
}
SUGGESTION_LABELS = {
    "same_ids_no_swap": "same IDs / no swap",
    "female_involved_swap": "female-involved swap",
    "male_to_male_swap": "male-to-male swap",
    "incomplete_association": "incomplete association",
}


class SceneCachedVideo:
    """Read-only exact-frame source with bounded background scene prefetch.

    The foreground source retains its own small exact-frame LRU.  Prefetch uses
    an independent OpenCV capture, so mouse scrubbing never races the UI decoder.
    Half the byte budget is reserved for frames at/after the scene center and
    half for frames before it.  The backward partition evicts its oldest (most
    distant) frames as it approaches the center.
    """

    def __init__(self, source: Any, *, max_cache_bytes: int):
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        self.source = source
        self.path = Path(source.path)
        self.frame_count = int(source.frame_count)
        self.width = int(source.width)
        self.height = int(source.height)
        self.reported_fps = float(source.reported_fps)
        self.max_cache_bytes = int(max_cache_bytes)
        self._partition_budget = max(1, self.max_cache_bytes // 2)
        self._forward: OrderedDict[int, np.ndarray] = OrderedDict()
        self._backward: OrderedDict[int, np.ndarray] = OrderedDict()
        self._forward_bytes = 0
        self._backward_bytes = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"scene-cache-{self.path.stem}"
        )
        self._future: Future | None = None
        self._generation = 0
        self._closed = False
        self.scene_start = 0
        self.scene_end = -1
        self.scene_center = -1
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, frame: int) -> np.ndarray:
        frame = int(frame)
        with self._lock:
            for cache in (self._forward, self._backward):
                if frame in cache:
                    image = cache.pop(frame)
                    cache[frame] = image
                    self.cache_hits += 1
                    return image.copy()
            self.cache_misses += 1
        return self.source.get(frame)

    def contains_scene(self, frame: int) -> bool:
        with self._lock:
            return self.scene_start <= int(frame) <= self.scene_end

    def has_cached(self, frame: int) -> bool:
        with self._lock:
            return int(frame) in self._forward or int(frame) in self._backward

    def prime(self, start_frame: int, end_frame: int, center_frame: int) -> None:
        start = max(0, min(int(start_frame), self.frame_count - 1))
        end = max(start, min(int(end_frame), self.frame_count - 1))
        center = max(start, min(int(center_frame), end))
        with self._lock:
            if self._closed:
                return
            if (start, end, center) == (
                self.scene_start,
                self.scene_end,
                self.scene_center,
            ):
                return
            self._generation += 1
            generation = self._generation
            self.scene_start, self.scene_end, self.scene_center = start, end, center
            self._forward.clear()
            self._backward.clear()
            self._forward_bytes = self._backward_bytes = 0
            self._future = self._executor.submit(
                self._prefetch, generation, start, end, center
            )

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return not self._closed and generation == self._generation

    def _store(
        self, generation: int, direction: str, frame: int, image: np.ndarray
    ) -> bool:
        size = int(image.nbytes)
        if size > self._partition_budget:
            return False
        with self._lock:
            if not self._is_current(generation):
                return False
            cache = self._forward if direction == "forward" else self._backward
            byte_name = "_forward_bytes" if direction == "forward" else "_backward_bytes"
            used = int(getattr(self, byte_name))
            if direction == "forward" and used + size > self._partition_budget:
                return False
            cache[int(frame)] = image.copy()
            used += size
            if direction == "backward":
                while cache and used > self._partition_budget:
                    _old_frame, old_image = cache.popitem(last=False)
                    used -= int(old_image.nbytes)
            setattr(self, byte_name, used)
            return True

    def _prefetch(
        self, generation: int, start: int, end: int, center: int
    ) -> None:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            return
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, center)
            for frame in range(center, end + 1):
                if not self._is_current(generation):
                    return
                ok, image = capture.read()
                if not ok or image is None:
                    break
                if not self._store(generation, "forward", frame, image):
                    break

            if start < center and self._is_current(generation):
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                for frame in range(start, center):
                    if not self._is_current(generation):
                        return
                    ok, image = capture.read()
                    if not ok or image is None:
                        break
                    self._store(generation, "backward", frame, image)
        finally:
            capture.release()

    def cache_info(self) -> dict[str, int | bool]:
        with self._lock:
            cached_keys = [*self._forward.keys(), *self._backward.keys()]
            return {
                "cached_frames": len(self._forward) + len(self._backward),
                "forward_frames": len(self._forward),
                "backward_frames": len(self._backward),
                "cached_min_frame": min(cached_keys) if cached_keys else -1,
                "cached_max_frame": max(cached_keys) if cached_keys else -1,
                "cached_bytes": self._forward_bytes + self._backward_bytes,
                "scene_start": self.scene_start,
                "scene_end": self.scene_end,
                "scene_center": self.scene_center,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "warming": bool(self._future and not self._future.done()),
            }

    def wait_until_ready(self, timeout: float | None = None) -> None:
        future = self._future
        if future is not None:
            future.result(timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.source.close()


def instance_anchor_points(
    instances: np.ndarray | None, anchor_mode: str = "mouth"
) -> np.ndarray:
    if instances is None:
        return np.empty((0, 2), dtype=float)
    values = np.asarray(instances, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("instances must have shape (instances, nodes, 2)")
    if anchor_mode == "mouth":
        return values[:, 0, :].copy()
    if anchor_mode != "visible-node-centroid":
        raise ValueError("anchor_mode must be mouth or visible-node-centroid")
    visible = np.all(np.isfinite(values), axis=2)
    counts = visible.sum(axis=1)
    anchors = np.full((len(values), 2), np.nan, dtype=float)
    valid = counts > 0
    anchors[valid] = np.where(visible[..., None], values, 0.0).sum(axis=1)[valid]
    anchors[valid] /= counts[valid, None]
    return anchors


def associate_projected_ids(
    projected_by_id: Mapping[str, Sequence[float]],
    instances: np.ndarray | None,
    *,
    anchor_mode: str,
    gate_px: float,
    excluded_instance_indices: set[int] | None = None,
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """One-to-one projected-ID to raw-instance association at one frame/view."""
    if gate_px < 0:
        raise ValueError("gate_px must be non-negative")
    anchors = instance_anchor_points(instances, anchor_mode)
    for instance_index in excluded_instance_indices or set():
        if 0 <= int(instance_index) < len(anchors):
            anchors[int(instance_index)] = np.nan
    ids: list[str] = []
    projected: list[np.ndarray] = []
    for fish_id, value in projected_by_id.items():
        point = np.asarray(value, dtype=float)
        if point.shape == (2,) and np.isfinite(point).all():
            ids.append(str(fish_id))
            projected.append(point)
    if not ids or not len(anchors):
        return {}, {}, anchors
    points = np.stack(projected)
    costs = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2)
    safe = np.where(np.isfinite(costs), costs, 1e12)
    rows, columns = linear_sum_assignment(safe)
    by_id: dict[str, int] = {}
    by_instance: dict[int, str] = {}
    for row, column in zip(rows, columns):
        if costs[row, column] <= float(gate_px):
            by_id[ids[int(row)]] = int(column)
            by_instance[int(column)] = ids[int(row)]
    return by_id, by_instance, anchors


@dataclass
class ViewIFDWindow:
    view: str
    start_frame: int
    end_frame: int
    frames: np.ndarray
    distances: np.ndarray
    matched_positions: np.ndarray
    displacements: np.ndarray
    source_instance_indices: np.ndarray
    match_status: np.ndarray


def compute_2d_ifd_window(
    store: TrackingIFD,
    slp_source: Any,
    calibration: Any,
    *,
    view: str,
    start_frame: int,
    end_frame: int,
    anchor_mode: str = "mouth",
    gate_px: float = 80.0,
    excluded_instances_by_frame: Mapping[int, set[int]] | None = None,
) -> ViewIFDWindow:
    """Calculate matched raw-SLEAP 2D IFD and per-ID XY displacement."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    window = store.slice_positions(start_frame, end_frame)
    frames = store.frames[window].copy()
    xyz = store.xyz[window].astype(float, copy=False)
    distances = np.full((len(frames), 3), np.nan, dtype=np.float32)
    matched_positions = np.full((len(frames), 3, 2), np.nan, dtype=np.float32)
    displacements = np.full((len(frames), 3), np.nan, dtype=np.float32)
    source_indices = np.full((len(frames), 3), -1, dtype=np.int16)
    statuses = np.full((len(frames), 3), "missing_3d", dtype="U20")
    if not len(frames):
        return ViewIFDWindow(
            view=view,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            frames=frames,
            distances=distances,
            matched_positions=matched_positions,
            displacements=displacements,
            source_instance_indices=source_indices,
            match_status=statuses,
        )

    flat_xyz = xyz.reshape(-1, 3)
    projected_flat = np.full((len(flat_xyz), 3, 2), np.nan, dtype=float)
    valid_xyz = np.all(np.isfinite(flat_xyz), axis=1)
    if np.any(valid_xyz):
        projected_flat[valid_xyz] = calibration.project_world(flat_xyz[valid_xyz])
    projected = projected_flat.reshape(len(frames), 3, 3, 2)
    camera = CAMERA_INDEX[view]
    for local_index, frame in enumerate(frames):
        instances = slp_source.get(int(frame), None)
        projected_by_id = {
            fish_id: projected[local_index, fish_index, camera]
            for fish_index, fish_id in enumerate(store.fish_ids)
        }
        by_id, _by_instance, anchors = associate_projected_ids(
            projected_by_id,
            instances,
            anchor_mode=anchor_mode,
            gate_px=gate_px,
            excluded_instance_indices=(excluded_instances_by_frame or {}).get(
                int(frame), set()
            ),
        )
        for fish_index, fish_id in enumerate(store.fish_ids):
            if not np.isfinite(projected[local_index, fish_index, camera]).all():
                continue
            instance_index = by_id.get(fish_id)
            if instance_index is None:
                statuses[local_index, fish_index] = "unmatched"
                continue
            source_indices[local_index, fish_index] = instance_index
            matched_positions[local_index, fish_index] = anchors[instance_index]
            statuses[local_index, fish_index] = "matched"
        for pair_index, (first, second) in enumerate(((0, 1), (0, 2), (1, 2))):
            if np.isfinite(matched_positions[local_index, [first, second]]).all():
                distances[local_index, pair_index] = np.linalg.norm(
                    matched_positions[local_index, first]
                    - matched_positions[local_index, second]
                )
    if len(frames) > 1:
        deltas = matched_positions[1:] - matched_positions[:-1]
        steps = np.linalg.norm(deltas, axis=2).astype(np.float32, copy=False)
        consecutive = np.diff(frames) == 1
        valid = np.all(np.isfinite(deltas), axis=2) & consecutive[:, None]
        steps[~valid] = np.nan
        displacements[1:] = steps
    return ViewIFDWindow(
        view=view,
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        frames=frames,
        distances=distances,
        matched_positions=matched_positions,
        displacements=displacements,
        source_instance_indices=source_indices,
        match_status=statuses,
    )


class Lazy2DIFDCache:
    def __init__(self, max_windows: int = 12):
        self.max_windows = max(1, int(max_windows))
        self._cache: OrderedDict[tuple[Any, ...], ViewIFDWindow] = OrderedDict()

    def get(
        self,
        store: TrackingIFD,
        slp_source: Any,
        calibration: Any,
        *,
        view: str,
        start_frame: int,
        end_frame: int,
        anchor_mode: str,
        gate_px: float,
        excluded_instances_by_frame: Mapping[int, set[int]] | None = None,
    ) -> ViewIFDWindow:
        exclusion_key = tuple(
            (int(frame), tuple(sorted(int(index) for index in indices)))
            for frame, indices in sorted((excluded_instances_by_frame or {}).items())
        )
        key = (
            store.method,
            view,
            int(start_frame),
            int(end_frame),
            anchor_mode,
            float(gate_px),
            exclusion_key,
        )
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = compute_2d_ifd_window(
            store,
            slp_source,
            calibration,
            view=view,
            start_frame=start_frame,
            end_frame=end_frame,
            anchor_mode=anchor_mode,
            gate_px=gate_px,
            excluded_instances_by_frame=excluded_instances_by_frame,
        )
        self._cache[key] = value
        while len(self._cache) > self.max_windows:
            self._cache.popitem(last=False)
        return value


def _manual_candidate_rows(path: str | Path) -> pd.DataFrame:
    """Recover explicitly entered episodes before regenerating detector rows."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return pd.DataFrame()
    try:
        table = pd.read_csv(source, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()
    required = set(CANDIDATE_COLUMNS)
    if not required.issubset(table.columns):
        return pd.DataFrame()
    manual_mask = table["manual"].astype(str).str.casefold().isin({"true", "1", "yes", "y"})
    return table.loc[manual_mask].copy()


def _merge_saved_ledger_candidates(
    candidates: pd.DataFrame, ledger: ContactAnnotationLedger
) -> pd.DataFrame:
    """Keep saved rows editable when later thresholds change their episode IDs."""
    result = candidates.copy()
    known = set(result["episode_id"].astype(str)) if not result.empty else set()
    recovered = []
    for annotation in ledger.rows.to_dict("records"):
        try:
            candidate = json.loads(str(annotation["candidate_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        episode_id = str(candidate.get("episode_id", ""))
        if not episode_id or episode_id in known:
            continue
        if not set(result.columns).issubset(candidate):
            continue
        recovered.append({column: candidate[column] for column in result.columns})
        known.add(episode_id)
    if recovered:
        result = pd.concat([result, pd.DataFrame(recovered, columns=result.columns)], ignore_index=True)
    return result


def _draw_sleap(
    image: np.ndarray,
    instances: np.ndarray | None,
    excluded_instance_indices: set[int] | None = None,
) -> None:
    if instances is None:
        return
    excluded = {int(value) for value in (excluded_instance_indices or set())}
    for instance_index, pose in enumerate(np.asarray(instances, dtype=float)):
        color = (50, 50, 235) if instance_index in excluded else RAW_SLEAP_COLOR
        if len(pose) >= 2 and np.isfinite(pose[:2]).all():
            mouth = tuple(np.round(pose[0]).astype(int))
            tail = tuple(np.round(pose[1]).astype(int))
            cv2.arrowedLine(
                image,
                tail,
                mouth,
                color,
                2 if instance_index in excluded else 1,
                cv2.LINE_AA,
                0,
                0.25,
            )
        for point in pose:
            if np.isfinite(point).all():
                cv2.circle(
                    image,
                    tuple(np.round(point).astype(int)),
                    4 if instance_index in excluded else 3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
        if instance_index in excluded:
            visible = pose[np.all(np.isfinite(pose), axis=1)]
            if len(visible):
                center = tuple(np.round(visible.mean(axis=0)).astype(int))
                cv2.drawMarker(
                    image,
                    center,
                    color,
                    cv2.MARKER_TILTED_CROSS,
                    20,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "FP",
                    (center[0] + 8, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )


def _draw_false_negative_markers(
    image: np.ndarray,
    markers: Sequence[Mapping[str, Any]],
    *,
    view: str,
    frame: int,
) -> None:
    """Draw FN evidence distinctly from SLEAP FPs and identity crosses."""
    color = (255, 40, 210)  # magenta in BGR
    for marker in markers:
        if marker["view"] != view or int(marker["frame"]) != int(frame):
            continue
        center = (
            int(round(float(marker["click_x"]))),
            int(round(float(marker["click_y"]))),
        )
        cv2.circle(image, center, 13, color, 2, cv2.LINE_AA)
        cv2.drawMarker(image, center, color, cv2.MARKER_DIAMOND, 18, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "FN",
            (center[0] + 10, max(16, center[1] - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def toggle_false_negative_marker(
    markers: Sequence[Mapping[str, Any]],
    *,
    view: str,
    frame: int,
    click_x: float,
    click_y: float,
    radius_px: float,
    marked_at_utc: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Toggle the nearest same-frame FN marker; return ``(markers, added)``."""
    normalized = validate_false_negative_detections(list(markers))
    candidates = [
        (math.hypot(item["click_x"] - click_x, item["click_y"] - click_y), index)
        for index, item in enumerate(normalized)
        if item["view"] == view and int(item["frame"]) == int(frame)
    ]
    if candidates:
        distance, index = min(candidates)
        if distance <= float(radius_px):
            del normalized[index]
            return normalized, False
    normalized.append(
        {
            "view": view,
            "frame": int(frame),
            "click_x": float(click_x),
            "click_y": float(click_y),
            "marked_at_utc": str(marked_at_utc),
        }
    )
    return validate_false_negative_detections(normalized), True


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _load_gui_dependencies():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError(
            "The GUI extra is required. Run: uv sync --extra gui"
        ) from error
    return QtCore, QtGui, QtWidgets, FigureCanvasQTAgg, NavigationToolbar2QT, Figure


def create_annotation_window(
    args: argparse.Namespace,
    *,
    videos: Mapping[str, Any],
    slp_predictions: Mapping[str, Any],
    tracking_stores: Mapping[str, TrackingIFD],
    calibration: Any,
    candidates: pd.DataFrame,
    ledger: ContactAnnotationLedger,
    source_provenance: Mapping[str, Any],
):
    QtCore, QtGui, QtWidgets, FigureCanvas, NavigationToolbar, Figure = _load_gui_dependencies()

    class VideoPanel(QtWidgets.QWidget):
        clicked = QtCore.pyqtSignal(str, float, float)
        context_requested = QtCore.pyqtSignal(str, float, float, object)
        frame_step_requested = QtCore.pyqtSignal(int)

        def __init__(self, view: str):
            super().__init__()
            self.view = view
            self.source_width = 1
            self.source_height = 1
            self.crop = (0, 0, 1, 1)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(2, 2, 2, 2)
            title = QtWidgets.QLabel(view)
            title.setAlignment(QtCore.Qt.AlignCenter)
            title.setStyleSheet("font-weight:bold")
            title.setToolTip(
                "Mouse wheel steps frames; right-click a fish to assign Anchor A/B female or chosen male"
            )
            layout.addWidget(title)
            self.label = QtWidgets.QLabel()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setMinimumSize(240, 150)
            self.label.setStyleSheet("background:#111")
            self.label.installEventFilter(self)
            layout.addWidget(self.label, 1)

        def set_image(self, image: np.ndarray, crop: tuple[int, int, int, int] | None = None) -> None:
            self.source_height, self.source_width = image.shape[:2]
            if crop is None:
                self.crop = (0, 0, self.source_width, self.source_height)
            else:
                x0, y0, x1, y1 = crop
                x0 = int(np.clip(x0, 0, self.source_width - 1))
                y0 = int(np.clip(y0, 0, self.source_height - 1))
                x1 = int(np.clip(x1, x0 + 1, self.source_width))
                y1 = int(np.clip(y1, y0 + 1, self.source_height))
                self.crop = (x0, y0, x1, y1)
            x0, y0, x1, y1 = self.crop
            shown = np.ascontiguousarray(image[y0:y1, x0:x1])
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            qimage = QtGui.QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.label.setPixmap(
                pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

        def resizeEvent(self, event):
            super().resizeEvent(event)
            # A frame refresh follows ordinary window resizing; avoid retaining a
            # second full-resolution pixmap per panel.

        def _source_position(self, position) -> tuple[float, float] | None:
            pixmap = self.label.pixmap()
            if pixmap is None or pixmap.isNull():
                return None
            left = (self.label.width() - pixmap.width()) / 2.0
            top = (self.label.height() - pixmap.height()) / 2.0
            px = float(position.x()) - left
            py = float(position.y()) - top
            if px < 0 or py < 0 or px >= pixmap.width() or py >= pixmap.height():
                return None
            x0, y0, x1, y1 = self.crop
            return (
                x0 + px * (x1 - x0) / pixmap.width(),
                y0 + py * (y1 - y0) / pixmap.height(),
            )

        def eventFilter(self, watched, event):
            if watched is self.label and event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
                    point = self._source_position(event.pos())
                    if point is not None:
                        if event.button() == QtCore.Qt.LeftButton:
                            self.clicked.emit(self.view, point[0], point[1])
                        else:
                            self.context_requested.emit(
                                self.view,
                                point[0],
                                point[1],
                                self.label.mapToGlobal(event.pos()),
                            )
                        return True
            if watched is self.label and event.type() == QtCore.QEvent.Wheel:
                step = 10 if event.modifiers() & QtCore.Qt.ShiftModifier else 1
                self.frame_step_requested.emit(step if event.angleDelta().y() > 0 else -step)
                return True
            return super().eventFilter(watched, event)

    class ContactWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Contact episode GT — {args.experiment}")
            self.resize(1780, 1080)
            self.videos = dict(videos)
            self.slps = dict(slp_predictions)
            self.stores = dict(tracking_stores)
            self.calibration = calibration
            self.ledger = ledger
            self.source_provenance = dict(source_provenance)
            self.candidates = candidates.copy().reset_index(drop=True)
            self.current_frame = 0
            self.current_episode: dict[str, Any] | None = None
            self.current_queue_position = -1
            self.filtered_indices: list[int] = []
            self.anchor_a_frame: int | None = None
            self.anchor_b_frame: int | None = None
            self.clicks = empty_clicks()
            self.false_positive_detections: list[dict[str, Any]] = []
            self.false_negative_detections: list[dict[str, Any]] = []
            self.preserved_issue_flags = ""
            self.dirty = False
            self.instances_by_view: dict[str, np.ndarray | None] = {}
            self.instance_geometry: dict[str, dict[str, dict[int, str]]] = {}
            self.projected_by_method: dict[str, dict[str, dict[str, np.ndarray]]] = {}
            self.association_frame: int | None = None
            self.ifd2d_cache = Lazy2DIFDCache()
            self.cursor_lines = []
            self.dragging_plot = False
            self.active_method = str(args.detector_method)
            self.max_frame = min(
                min(source.frame_count for source in self.videos.values()) - 1,
                min(store.frame_max for store in self.stores.values()),
            )
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(max(15, int(round(1000.0 / args.source_fps))))
            self.timer.timeout.connect(lambda: self._step(1, playback=True))
            self.cache_status_timer = QtCore.QTimer(self)
            self.cache_status_timer.setInterval(400)
            self.cache_status_timer.timeout.connect(self._update_cache_status)
            self.cache_status_timer.start()
            self._build()
            self._install_shortcuts()
            self._refresh_queue()
            self._resume()

        def _build(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)
            split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            root.addWidget(split)

            top = QtWidgets.QWidget()
            top_layout = QtWidgets.QVBoxLayout(top)
            panels = QtWidgets.QHBoxLayout()
            self.panels = {view: VideoPanel(view) for view in VIEWS}
            for view in VIEWS:
                self.panels[view].clicked.connect(self._record_click)
                self.panels[view].context_requested.connect(self._show_identity_menu)
                self.panels[view].frame_step_requested.connect(self._step)
                panels.addWidget(self.panels[view], 1)
            top_layout.addLayout(panels, 1)

            overlay = QtWidgets.QHBoxLayout()
            top_layout.addLayout(overlay)
            self.sleap_toggle = QtWidgets.QCheckBox("SLEAP poses")
            self.ids_toggle = QtWidgets.QCheckBox("Current 3D IDs")
            self.clicks_toggle = QtWidgets.QCheckBox("Click markers")
            self.tails_toggle = QtWidgets.QCheckBox("Short trajectory tails")
            for toggle in (self.sleap_toggle, self.ids_toggle, self.clicks_toggle, self.tails_toggle):
                toggle.toggled.connect(self._render_videos)
                overlay.addWidget(toggle)
            self.method_combo = QtWidgets.QComboBox()
            self.method_combo.addItems(list(self.stores))
            self.method_combo.setCurrentText(self.active_method)
            self.method_combo.currentTextChanged.connect(self._change_active_method)
            overlay.addWidget(QtWidgets.QLabel("Overlay:"))
            overlay.addWidget(self.method_combo)
            self.view_mode = QtWidgets.QComboBox()
            self.view_mode.addItem("Full frame", "full")
            self.view_mode.addItem("Fish focused", "fish")
            self.view_mode.currentIndexChanged.connect(self._render_videos)
            overlay.addWidget(self.view_mode)
            self.cache_label = QtWidgets.QLabel("Scene cache: idle")
            self.cache_label.setToolTip(
                "A bounded raw-frame cache warms around the current scene for smooth bidirectional scrubbing"
            )
            overlay.addWidget(self.cache_label)
            overlay.addStretch(1)

            navigation = QtWidgets.QHBoxLayout()
            top_layout.addLayout(navigation)
            self.play_button = QtWidgets.QPushButton("Play [Space]")
            self.play_button.clicked.connect(self._toggle_play)
            navigation.addWidget(self.play_button)
            for label, delta in (("−100", -100), ("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10), ("+100", 100)):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _checked=False, value=delta: self._step(value))
                navigation.addWidget(button)
            navigation.addWidget(QtWidgets.QLabel("Exact source frame:"))
            self.frame_spin = QtWidgets.QSpinBox()
            self.frame_spin.setRange(0, self.max_frame)
            self.frame_spin.setKeyboardTracking(False)
            self.frame_spin.editingFinished.connect(lambda: self._navigate_to(self.frame_spin.value()))
            navigation.addWidget(self.frame_spin)
            self.frame_label = QtWidgets.QLabel()
            navigation.addWidget(self.frame_label)
            navigation.addStretch(1)
            split.addWidget(top)

            bottom = QtWidgets.QWidget()
            bottom_layout = QtWidgets.QVBoxLayout(bottom)
            collapse = QtWidgets.QHBoxLayout()
            bottom_layout.addLayout(collapse)
            self.plot_checks = []
            for label in ("3D IFD", "top 2D", "side1 2D", "side2 2D"):
                checkbox = QtWidgets.QCheckBox(label)
                checkbox.setChecked(True)
                checkbox.toggled.connect(self._layout_axes)
                collapse.addWidget(checkbox)
                self.plot_checks.append(checkbox)
            collapse.addWidget(QtWidgets.QLabel("2D metric:"))
            self.metric_2d_combo = QtWidgets.QComboBox()
            self.metric_2d_combo.addItem("XY displacement / frame", "displacement")
            self.metric_2d_combo.addItem("Pairwise IFD", "ifd")
            self.metric_2d_combo.setToolTip(
                "Displacement is per current ID and exposes instance/identity jumps; "
                "IFD shows pairwise separation."
            )
            self.metric_2d_combo.currentIndexChanged.connect(
                lambda _index: self._refresh_plots()
            )
            collapse.addWidget(self.metric_2d_combo)
            collapse.addStretch(1)
            self.figure = Figure(figsize=(12, 6), tight_layout=False)
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            bottom_layout.addWidget(self.toolbar)
            bottom_layout.addWidget(self.canvas, 1)
            self.axes = []
            first_axis = self.figure.add_subplot(4, 1, 1)
            self.axes.append(first_axis)
            for index in range(2, 5):
                self.axes.append(self.figure.add_subplot(4, 1, index, sharex=first_axis))
            self.canvas.mpl_connect("button_press_event", self._plot_press)
            self.canvas.mpl_connect("motion_notify_event", self._plot_motion)
            self.canvas.mpl_connect("button_release_event", self._plot_release)
            split.addWidget(bottom)
            split.setSizes([560, 480])

            dock = QtWidgets.QDockWidget("Episode annotation", self)
            dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            dock_scroll = QtWidgets.QScrollArea()
            dock_scroll.setWidgetResizable(True)
            dock.setWidget(dock_scroll)
            sidebar_widget = QtWidgets.QWidget()
            dock_scroll.setWidget(sidebar_widget)
            sidebar = QtWidgets.QVBoxLayout(sidebar_widget)

            queue_nav = QtWidgets.QHBoxLayout()
            previous = QtWidgets.QPushButton("Previous episode [P]")
            following = QtWidgets.QPushButton("Next episode [N]")
            previous.clicked.connect(lambda: self._go_queue_position(self.current_queue_position - 1))
            following.clicked.connect(lambda: self._go_queue_position(self.current_queue_position + 1))
            queue_nav.addWidget(previous)
            queue_nav.addWidget(following)
            sidebar.addLayout(queue_nav)
            self.filter_combo = QtWidgets.QComboBox()
            self.filter_combo.addItem("All", "all")
            self.filter_combo.addItem("Unreviewed", "unreviewed")
            for value in CONTACT_VALIDITIES:
                self.filter_combo.addItem("Contact: " + CONTACT_VALIDITY_LABELS[value], "contact:" + value)
            for value in TRACKER_ASSESSMENTS:
                self.filter_combo.addItem("Tracker: " + TRACKER_ASSESSMENT_LABELS[value], "tracker:" + value)
            self.filter_combo.currentIndexChanged.connect(self._refresh_queue)
            sidebar.addWidget(self.filter_combo)
            self.sort_combo = QtWidgets.QComboBox()
            self.sort_combo.addItem("Sort: source frame", "frame")
            self.sort_combo.addItem("Sort: minimum IFD", "minimum")
            self.sort_combo.addItem("Sort: duration", "duration")
            self.sort_combo.addItem("Sort: missing burden", "missing")
            self.sort_combo.currentIndexChanged.connect(self._refresh_queue)
            sidebar.addWidget(self.sort_combo)
            self.episode_combo = QtWidgets.QComboBox()
            self.episode_combo.activated.connect(self._go_queue_position)
            sidebar.addWidget(self.episode_combo)
            self.progress_label = QtWidgets.QLabel()
            sidebar.addWidget(self.progress_label)
            self.episode_info = QtWidgets.QLabel()
            self.episode_info.setWordWrap(True)
            sidebar.addWidget(self.episode_info)

            thresholds = QtWidgets.QGridLayout()
            sidebar.addLayout(thresholds)
            thresholds.addWidget(QtWidgets.QLabel("Enter contact < mm"), 0, 0)
            self.enter_spin = QtWidgets.QDoubleSpinBox()
            self.enter_spin.setRange(0.1, 1000.0)
            self.enter_spin.setDecimals(1)
            self.enter_spin.setValue(args.enter_mm)
            thresholds.addWidget(self.enter_spin, 0, 1)
            thresholds.addWidget(QtWidgets.QLabel("Exit contact > mm"), 1, 0)
            self.exit_spin = QtWidgets.QDoubleSpinBox()
            self.exit_spin.setRange(0.1, 1000.0)
            self.exit_spin.setDecimals(1)
            self.exit_spin.setValue(args.exit_mm)
            thresholds.addWidget(self.exit_spin, 1, 1)
            update_detector = QtWidgets.QPushButton("Update detector candidates")
            update_detector.clicked.connect(self._update_detector)
            sidebar.addWidget(update_detector)

            manual_row = QtWidgets.QHBoxLayout()
            self.manual_range = QtWidgets.QLineEdit()
            self.manual_range.setPlaceholderText("missed frame or start-end")
            manual_add = QtWidgets.QPushButton("Add manual episode")
            manual_add.clicked.connect(self._add_manual_episode)
            manual_row.addWidget(self.manual_range)
            manual_row.addWidget(manual_add)
            sidebar.addLayout(manual_row)
            sidebar.addWidget(self._line())

            identity_goal = QtWidgets.QLabel(
                "Essential evidence: identify the female and one chosen physical male "
                "at both automatically suggested anchors around the detector cue. "
                "The anchors are editable and are not assumed contact ground truth.\n\n"
                "Fast path: right-click a fish in any preview and choose its Anchor A/B identity. "
                "After one click, the GUI cues the same identity at the other anchor."
            )
            identity_goal.setWordWrap(True)
            identity_goal.setStyleSheet(
                "background:#173a4d;color:white;padding:8px;font-weight:bold"
            )
            sidebar.addWidget(identity_goal)

            frame_buttons = QtWidgets.QHBoxLayout()
            anchor_a_button = QtWidgets.QPushButton("Set Anchor A = current")
            anchor_b_button = QtWidgets.QPushButton("Set Anchor B = current")
            anchor_a_button.clicked.connect(lambda: self._set_phase_frame("anchor_a"))
            anchor_b_button.clicked.connect(lambda: self._set_phase_frame("anchor_b"))
            frame_buttons.addWidget(anchor_a_button)
            frame_buttons.addWidget(anchor_b_button)
            sidebar.addLayout(frame_buttons)
            self.phase_frame_label = QtWidgets.QLabel("Anchor A: unset | Anchor B: unset")
            sidebar.addWidget(self.phase_frame_label)
            self.anchor_summary = QtWidgets.QLabel()
            self.anchor_summary.setWordWrap(True)
            self.anchor_summary.setStyleSheet("background:#242424;color:white;padding:6px")
            sidebar.addWidget(self.anchor_summary)
            selection_row = QtWidgets.QHBoxLayout()
            self.phase_combo = QtWidgets.QComboBox()
            self.phase_combo.addItem("Anchor A", "anchor_a")
            self.phase_combo.addItem("Anchor B", "anchor_b")
            self.target_combo = QtWidgets.QComboBox()
            self.target_combo.addItem("Female", "female")
            self.target_combo.addItem("Anchor male", "anchor_male")
            self.phase_combo.currentIndexChanged.connect(self._update_click_status)
            self.target_combo.currentIndexChanged.connect(self._update_click_status)
            selection_row.addWidget(QtWidgets.QLabel("Active phase:"))
            selection_row.addWidget(self.phase_combo)
            selection_row.addWidget(QtWidgets.QLabel("Target:"))
            selection_row.addWidget(self.target_combo)
            sidebar.addLayout(selection_row)
            self.click_status = QtWidgets.QLabel()
            self.click_status.setWordWrap(True)
            sidebar.addWidget(self.click_status)
            clear_click = QtWidgets.QPushButton("Clear active phase/target clicks")
            clear_click.clicked.connect(self._clear_active_clicks)
            sidebar.addWidget(clear_click)
            self.auto_cue_checkbox = QtWidgets.QCheckBox(
                "Automatically cue the same identity at the other anchor"
            )
            self.auto_cue_checkbox.setChecked(True)
            sidebar.addWidget(self.auto_cue_checkbox)

            sidebar.addWidget(self._line())
            sidebar.addWidget(QtWidgets.QLabel("Contact validity:"))
            self.contact_group = QtWidgets.QButtonGroup(self)
            contact_grid = QtWidgets.QGridLayout()
            sidebar.addLayout(contact_grid)
            for index, value in enumerate(CONTACT_VALIDITIES):
                button = QtWidgets.QRadioButton(CONTACT_VALIDITY_LABELS[value])
                button.setProperty("decision_value", value)
                button.toggled.connect(self._mark_dirty)
                self.contact_group.addButton(button)
                contact_grid.addWidget(button, index // 2, index % 2)
                if value == "uncertain":
                    button.setChecked(True)

            sidebar.addWidget(QtWidgets.QLabel("Tracker assessment:"))
            self.tracker_group = QtWidgets.QButtonGroup(self)
            tracker_grid = QtWidgets.QGridLayout()
            sidebar.addLayout(tracker_grid)
            for index, value in enumerate(TRACKER_ASSESSMENTS):
                button = QtWidgets.QRadioButton(TRACKER_ASSESSMENT_LABELS[value])
                button.setProperty("decision_value", value)
                button.toggled.connect(self._mark_dirty)
                self.tracker_group.addButton(button)
                tracker_grid.addWidget(button, index // 2, index % 2)
                if value == "uncertain":
                    button.setChecked(True)

            self.assessed_method = QtWidgets.QComboBox()
            self.assessed_method.addItems(list(self.stores))
            self.assessed_method.setCurrentText(self.active_method)
            self.assessed_method.currentTextChanged.connect(self._update_suggestion)
            sidebar.addWidget(QtWidgets.QLabel("Assessed tracking label:"))
            sidebar.addWidget(self.assessed_method)
            self.suggestion_label = QtWidgets.QLabel()
            self.suggestion_label.setWordWrap(True)
            self.suggestion_label.setStyleSheet("font-weight:bold;background:#333;color:white;padding:6px")
            sidebar.addWidget(self.suggestion_label)

            sidebar.addWidget(QtWidgets.QLabel("Anchor usability:"))
            self.evidence_quality = QtWidgets.QComboBox()
            for value, label in (
                ("unreviewed", "Unreviewed"),
                ("usable", "Usable identity anchors"),
                ("human_only", "Human identity only; ignore associations"),
                ("unusable", "Cannot identify the same physical fish"),
            ):
                self.evidence_quality.addItem(label, value)
            self.evidence_quality.currentIndexChanged.connect(self._mark_dirty)
            sidebar.addWidget(self.evidence_quality)
            self.evidence_quality.setCurrentIndex(
                self.evidence_quality.findData("usable")
            )
            sidebar.addWidget(
                QtWidgets.QLabel("Optional identity/crossing note (keep brief):")
            )
            self.notes = QtWidgets.QPlainTextEdit()
            self.notes.setMaximumHeight(65)
            self.notes.setPlaceholderText(
                "Only note ambiguity that affects Anchor A/B physical identity."
            )
            self.notes.textChanged.connect(self._mark_dirty)
            sidebar.addWidget(self.notes)
            save = QtWidgets.QPushButton("Save annotation [Ctrl+S]")
            save.clicked.connect(self._save)
            undo = QtWidgets.QPushButton("Undo last session edit [Ctrl+Z]")
            undo.clicked.connect(self._undo)
            sidebar.addWidget(save)
            sidebar.addWidget(undo)
            self.status_label = QtWidgets.QLabel("Ready")
            self.status_label.setWordWrap(True)
            sidebar.addWidget(self.status_label)
            sidebar.addStretch(1)

        @staticmethod
        def _line():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            return line

        def _install_shortcuts(self) -> None:
            shortcuts = {
                "Left": lambda: self._step(-1),
                "Right": lambda: self._step(1),
                "Shift+Left": lambda: self._step(-10),
                "Shift+Right": lambda: self._step(10),
                "Ctrl+Left": lambda: self._step(-100),
                "Ctrl+Right": lambda: self._step(100),
                "Space": self._toggle_play,
                "P": lambda: self._go_queue_position(self.current_queue_position - 1),
                "N": lambda: self._go_queue_position(self.current_queue_position + 1),
                "Ctrl+S": self._save,
                "Ctrl+Z": self._undo,
            }
            self._shortcuts = []
            for sequence, callback in shortcuts.items():
                shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

        def _mark_dirty(self, *_args) -> None:
            self.dirty = True

        def _decision(self, group) -> str:
            button = group.checkedButton()
            return str(button.property("decision_value")) if button else "uncertain"

        def _set_decision(self, group, value: str) -> None:
            for button in group.buttons():
                if button.property("decision_value") == value:
                    button.setChecked(True)
                    return

        def _refresh_queue(self, *_args, preferred_episode_id: str | None = None) -> None:
            states = queue_decision_state(
                self.ledger.rows, experiment=args.experiment, annotator=args.annotator
            )
            indices = list(range(len(self.candidates)))
            filter_value = str(self.filter_combo.currentData() or "all")
            if filter_value == "unreviewed":
                indices = [i for i in indices if str(self.candidates.iloc[i]["episode_id"]) not in states]
            elif filter_value.startswith("contact:"):
                wanted = filter_value.split(":", 1)[1]
                indices = [i for i in indices if states.get(str(self.candidates.iloc[i]["episode_id"]), {}).get("contact_validity") == wanted]
            elif filter_value.startswith("tracker:"):
                wanted = filter_value.split(":", 1)[1]
                indices = [i for i in indices if states.get(str(self.candidates.iloc[i]["episode_id"]), {}).get("tracker_assessment") == wanted]
            sort_value = str(self.sort_combo.currentData() or "frame")
            column, ascending = {
                "frame": ("contact_start_frame", True),
                "minimum": ("minimum_ifd_mm", True),
                "duration": ("duration_frames", False),
                "missing": ("missing_direct_fraction", False),
            }[sort_value]
            indices.sort(
                key=lambda i: (
                    math.inf if self.candidates.iloc[i][column] == "" or pd.isna(self.candidates.iloc[i][column]) else float(self.candidates.iloc[i][column])
                ),
                reverse=not ascending,
            )
            self.filtered_indices = indices
            current_id = preferred_episode_id or (
                str(self.current_episode["episode_id"]) if self.current_episode else None
            )
            self.episode_combo.blockSignals(True)
            self.episode_combo.clear()
            selected_position = -1
            for position, index in enumerate(indices):
                row = self.candidates.iloc[index]
                episode_id = str(row["episode_id"])
                reviewed = "✓" if episode_id in states else "·"
                self.episode_combo.addItem(
                    f"{reviewed} {int(row['contact_start_frame']):,}–{int(row['contact_end_frame']):,} | min {row['minimum_ifd_mm']}",
                    index,
                )
                if episode_id == current_id:
                    selected_position = position
            self.episode_combo.blockSignals(False)
            reviewed_count = sum(str(row["episode_id"]) in states for row in self.candidates.to_dict("records"))
            self.progress_label.setText(
                f"Reviewed {reviewed_count}/{len(self.candidates)} | showing {len(indices)}"
            )
            if selected_position >= 0:
                self.current_queue_position = selected_position
                self.episode_combo.setCurrentIndex(selected_position)
            elif indices and self.current_episode is None:
                self._go_queue_position(0)

        def _resume(self) -> None:
            latest = self.ledger.latest(args.experiment, args.annotator)
            if latest is not None:
                episode_id = str(latest["episode_id"])
                matches = self.candidates.index[self.candidates["episode_id"].astype(str).eq(episode_id)].tolist()
                if matches:
                    positions = [p for p, actual in enumerate(self.filtered_indices) if actual == matches[0]]
                    if positions:
                        self._go_queue_position(positions[0])
                        return
            if self.filtered_indices:
                self._go_queue_position(0)

        def _go_queue_position(self, position: int) -> None:
            if not self.filtered_indices:
                return
            position = int(np.clip(position, 0, len(self.filtered_indices) - 1))
            self.current_queue_position = position
            self.episode_combo.setCurrentIndex(position)
            index = self.filtered_indices[position]
            self.current_episode = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in self.candidates.iloc[index].to_dict().items()
            }
            self._load_saved_or_reset()
            self._navigate_to(int(self.anchor_a_frame), refresh_plots=True)
            self._update_episode_info()

        def _update_episode_info(self) -> None:
            if not self.current_episode:
                self.episode_info.setText("No active episode")
                return
            row = self.current_episode
            self.episode_info.setText(
                f"Contact {int(row['contact_start_frame']):,}–{int(row['contact_end_frame']):,}; "
                f"review {int(row['review_start_frame']):,}–{int(row['review_end_frame']):,}; "
                f"min {row['minimum_ifd_mm']} mm at {row['minimum_ifd_frame']} ({row['minimum_pair']}); "
                f"duration {row['duration_frames']} frames; missing burden {row['missing_direct_fraction']}"
            )

        def _load_saved_or_reset(self) -> None:
            assert self.current_episode is not None
            saved = self.ledger.get(
                args.experiment, str(self.current_episode["episode_id"]), args.annotator
            )
            self.clicks = empty_clicks()
            self.false_positive_detections = []
            self.false_negative_detections = []
            self.anchor_a_frame, self.anchor_b_frame = suggest_anchor_frames(
                self.stores[args.detector_method], self.current_episode
            )
            self.preserved_issue_flags = ""
            self._set_decision(self.contact_group, "uncertain")
            self._set_decision(self.tracker_group, "uncertain")
            self.evidence_quality.setCurrentIndex(
                self.evidence_quality.findData("usable")
            )
            self.notes.blockSignals(True)
            self.notes.clear()
            self.notes.blockSignals(False)
            if saved is not None:
                self.anchor_a_frame = int(saved["anchor_a_frame"])
                self.anchor_b_frame = int(saved["anchor_b_frame"])
                self.clicks = deserialize_clicks(str(saved["clicks_json"]))
                self.false_positive_detections = validate_false_positive_detections(
                    saved["false_positive_detections_json"]
                )
                self.false_negative_detections = validate_false_negative_detections(
                    saved["false_negative_detections_json"]
                )
                self._set_decision(self.contact_group, str(saved["contact_validity"]))
                self._set_decision(self.tracker_group, str(saved["tracker_assessment"]))
                self.assessed_method.setCurrentText(str(saved["assessed_tracking_label"]))
                quality_index = self.evidence_quality.findData(str(saved["evidence_quality"]))
                self.evidence_quality.setCurrentIndex(max(0, quality_index))
                # Older ledgers may contain fine-grained model-specific flags.
                # Preserve them without asking annotators to reproduce them.
                self.preserved_issue_flags = str(saved["issue_flags"])
                self.notes.setPlainText(str(saved["notes"]))
            self.phase_combo.setCurrentIndex(
                self.phase_combo.findData("anchor_a")
            )
            self.target_combo.setCurrentIndex(
                self.target_combo.findData("female")
            )
            self.association_frame = None
            self.dirty = False
            self._update_phase_label()
            self._update_click_status()
            self._update_anchor_summary()
            self._update_suggestion()

        def _navigate_to(self, frame: int, *, refresh_plots: bool = False) -> None:
            target = int(np.clip(frame, 0, self.max_frame))
            self.current_frame = target
            self._ensure_scene_cache(target)
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(target)
            self.frame_spin.blockSignals(False)
            self.frame_label.setText(f"{target / args.source_fps:.3f} s")
            self._render_videos()
            if refresh_plots:
                self._refresh_plots()
            else:
                self._update_plot_cursor()

        def _ensure_scene_cache(self, frame: int, *, force: bool = False) -> None:
            if self.current_episode:
                review_start = int(self.current_episode["review_start_frame"])
                review_end = int(self.current_episode["review_end_frame"])
            else:
                review_start, review_end = 0, self.max_frame
            radius = max(1, int(args.scene_cache_radius))
            start = max(review_start, int(frame) - radius)
            end = min(review_end, int(frame) + radius)
            recenter_distance = max(8, radius // 2)
            for source in self.videos.values():
                if not isinstance(source, SceneCachedVideo):
                    continue
                info = source.cache_info()
                cached_direction = max(
                    int(info["forward_frames"]), int(info["backward_frames"])
                )
                cached_recenter_distance = max(8, int(cached_direction * 0.6))
                if (
                    force
                    or not source.contains_scene(frame)
                    or abs(int(frame) - int(info["scene_center"])) >= recenter_distance
                    or (
                        not bool(info["warming"])
                        and not source.has_cached(frame)
                    )
                    or (
                        not bool(info["warming"])
                        and int(info["cached_frames"]) > 0
                        and abs(int(frame) - int(info["scene_center"]))
                        >= cached_recenter_distance
                    )
                ):
                    source.prime(start, end, int(frame))

        def _update_cache_status(self) -> None:
            cached_sources = [
                source
                for source in self.videos.values()
                if isinstance(source, SceneCachedVideo)
            ]
            if not cached_sources:
                self.cache_label.setText("Scene cache: synthetic/direct")
                return
            infos = [source.cache_info() for source in cached_sources]
            cached_frames = sum(int(info["cached_frames"]) for info in infos)
            cached_megabytes = sum(int(info["cached_bytes"]) for info in infos) / (1024**2)
            hits = sum(int(info["hits"]) for info in infos)
            misses = sum(int(info["misses"]) for info in infos)
            state = "warming" if any(bool(info["warming"]) for info in infos) else "ready"
            self.cache_label.setText(
                f"Scene cache: {state}, {cached_frames} frames / {cached_megabytes:.0f} MB, "
                f"hits {hits}, misses {misses}"
            )

        def plot_frame_clicked(self, frame: int) -> None:
            self._navigate_to(frame)

        def _step(self, delta: int, playback: bool = False) -> None:
            target = self.current_frame + int(delta)
            if playback and self.current_episode and target > int(self.current_episode["review_end_frame"]):
                self._toggle_play()
                return
            self._navigate_to(target)

        def _toggle_play(self) -> None:
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText("Play [Space]")
            else:
                self.timer.start()
                self.play_button.setText("Pause [Space]")

        def _frame_for_phase(self, phase: str) -> int | None:
            if phase == "anchor_a":
                return self.anchor_a_frame
            if phase == "anchor_b":
                return self.anchor_b_frame
            raise ValueError(f"unknown anchor phase {phase!r}")

        def _excluded_instances(self, view: str, frame: int) -> set[int]:
            return {
                int(item["instance_index"])
                for item in self.false_positive_detections
                if item["view"] == view and int(item["frame"]) == int(frame)
            }

        def _excluded_by_frame(self, view: str) -> dict[int, set[int]]:
            result: dict[int, set[int]] = {}
            for item in self.false_positive_detections:
                if item["view"] == view:
                    result.setdefault(int(item["frame"]), set()).add(
                        int(item["instance_index"])
                    )
            return result

        def _project_methods(self) -> None:
            self.projected_by_method = {}
            self.instance_geometry = {}
            for method, store in self.stores.items():
                projected_views = {view: {} for view in VIEWS}
                try:
                    xyz_by_id = store.xyz_at(self.current_frame)
                except IndexError:
                    xyz_by_id = {}
                ids = []
                xyz = []
                for fish_id, point in xyz_by_id.items():
                    if np.isfinite(point).all():
                        ids.append(fish_id)
                        xyz.append(point)
                if ids:
                    pixels = self.calibration.project_world(np.asarray(xyz, dtype=float))
                    for fish_index, fish_id in enumerate(ids):
                        for view in VIEWS:
                            projected_views[view][fish_id] = np.asarray(
                                pixels[fish_index, CAMERA_INDEX[view]], dtype=float
                            )
                self.projected_by_method[method] = projected_views
                self.instance_geometry[method] = {}
                for view in VIEWS:
                    _by_id, by_instance, _anchors = associate_projected_ids(
                        projected_views[view],
                        self.instances_by_view.get(view),
                        anchor_mode=args.anchor_mode,
                        gate_px=args.match_gate_px,
                        excluded_instance_indices=self._excluded_instances(
                            view, self.current_frame
                        ),
                    )
                    self.instance_geometry[method][view] = by_instance

        def _ensure_frame_associations(self) -> None:
            if self.association_frame == self.current_frame:
                return
            self.instances_by_view = {
                view: self.slps[view].get(self.current_frame, None) for view in VIEWS
            }
            self._project_methods()
            self.association_frame = self.current_frame

        def _draw_tails(self, image: np.ndarray, view: str) -> None:
            store = self.stores[self.active_method]
            start = max(store.frame_min, self.current_frame - args.tail_frames)
            section = store.xyz[store.slice_positions(start, self.current_frame)].astype(float)
            if not len(section):
                return
            flat = section.reshape(-1, 3)
            pixels = np.full((len(flat), 3, 2), np.nan, dtype=float)
            valid = np.all(np.isfinite(flat), axis=1)
            if np.any(valid):
                pixels[valid] = self.calibration.project_world(flat[valid])
            pixels = pixels.reshape(len(section), 3, 3, 2)
            colors = pair_colors_for(store.fish_ids)
            fish_colors = [_hex_to_bgr(color) for color in colors.values()]
            for fish_index in range(3):
                points = pixels[:, fish_index, CAMERA_INDEX[view]]
                for first, second in zip(points[:-1], points[1:]):
                    if np.isfinite(first).all() and np.isfinite(second).all():
                        cv2.line(
                            image,
                            tuple(np.round(first).astype(int)),
                            tuple(np.round(second).astype(int)),
                            fish_colors[fish_index],
                            1,
                            cv2.LINE_AA,
                        )

        def _fish_crop(self, image: np.ndarray, view: str) -> tuple[int, int, int, int] | None:
            points: list[np.ndarray] = []
            for point in self.projected_by_method.get(self.active_method, {}).get(view, {}).values():
                value = np.asarray(point, dtype=float)
                if np.isfinite(value).all():
                    points.append(value)
            instances = self.instances_by_view.get(view)
            if instances is not None:
                values = np.asarray(instances, dtype=float)
                for instance_index in self.instance_geometry.get(self.active_method, {}).get(view, {}):
                    if 0 <= instance_index < len(values):
                        points.extend(
                            point
                            for point in values[instance_index]
                            if np.isfinite(point).all()
                        )
            if not points:
                anchors = instance_anchor_points(instances, args.anchor_mode)
                points = [point for point in anchors if np.isfinite(point).all()]
            if not points:
                return None
            values = np.stack(points)
            low = values.min(axis=0)
            high = values.max(axis=0)
            span = np.maximum(high - low, 80.0)
            pad = np.maximum(45.0, span * 0.25)
            height, width = image.shape[:2]
            return (
                int(max(0, low[0] - pad[0])),
                int(max(0, low[1] - pad[1])),
                int(min(width, high[0] + pad[0])),
                int(min(height, high[1] + pad[1])),
            )

        def _render_videos(self, *_args) -> None:
            try:
                images = {view: self.videos[view].get(self.current_frame) for view in VIEWS}
                needs_associations = (
                    self.sleap_toggle.isChecked()
                    or self.ids_toggle.isChecked()
                    or self.view_mode.currentData() == "fish"
                )
                if needs_associations:
                    self._ensure_frame_associations()
                for view, image in images.items():
                    if self.sleap_toggle.isChecked():
                        _draw_sleap(
                            image,
                            self.instances_by_view[view],
                            self._excluded_instances(view, self.current_frame),
                        )
                    if self.tails_toggle.isChecked():
                        self._draw_tails(image, view)
                    if self.ids_toggle.isChecked():
                        fish_ids = self.stores[self.active_method].fish_ids
                        fish_colors = {
                            fish_id: _hex_to_bgr(FISH_ID_COLORS[index % len(FISH_ID_COLORS)])
                            for index, fish_id in enumerate(fish_ids)
                        }
                        for fish_index, (fish_id, point) in enumerate(
                            self.projected_by_method[self.active_method][view].items()
                        ):
                            if np.isfinite(point).all():
                                xy = tuple(np.round(point).astype(int))
                                color = fish_colors[fish_id]
                                cv2.circle(image, xy, 7, color, 2, cv2.LINE_AA)
                                cv2.putText(image, fish_id, (xy[0] + 8, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    if self.clicks_toggle.isChecked():
                        for phase in PHASES:
                            phase_frame = self._frame_for_phase(phase)
                            if phase_frame != self.current_frame:
                                continue
                            for target in IDENTITY_TARGETS:
                                evidence = self.clicks[phase][target][view]
                                if evidence:
                                    xy = (int(round(evidence["click_x"])), int(round(evidence["click_y"])))
                                    color = ACTIVE_CLICK_COLORS[target]
                                    cv2.drawMarker(image, xy, color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                                    phase_label = "A" if phase == "anchor_a" else "B"
                                    cv2.putText(image, f"{phase_label} {'F' if target == 'female' else 'M'}", (xy[0] + 8, max(16, xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    _draw_false_negative_markers(
                        image,
                        self.false_negative_detections,
                        view=view,
                        frame=self.current_frame,
                    )
                    crop = self._fish_crop(image, view) if self.view_mode.currentData() == "fish" else None
                    self.panels[view].set_image(image, crop)
            except Exception as error:  # frame decode errors should be visible, not crash silently
                self.status_label.setText(f"Frame render failed: {error}")

        def _change_active_method(self, method: str) -> None:
            if method not in self.stores:
                return
            self.active_method = method
            self._render_videos()
            self._refresh_plots()

        def _record_click(self, view: str, x: float, y: float) -> None:
            phase = str(self.phase_combo.currentData())
            target = str(self.target_combo.currentData())
            expected_frame = self._frame_for_phase(phase)
            phase_label = "Anchor A" if phase == "anchor_a" else "Anchor B"
            if expected_frame is None:
                self.status_label.setText(f"Set {phase_label} before clicking.")
                return
            if self.current_frame != expected_frame:
                self.status_label.setText(
                    f"{phase_label} is frame {expected_frame:,}; navigate there before adding its click."
                )
                return
            already_selected = any(self.clicks[phase][target][candidate_view] for candidate_view in VIEWS)
            self._ensure_frame_associations()
            instances = self.instances_by_view.get(view)
            instance_index, _distance = hit_test_sleap_pose(
                instances, (x, y), args.click_gate_px
            )
            if instance_index in self._excluded_instances(view, self.current_frame):
                # A click remains valid human evidence, but an explicitly false
                # SLEAP prediction must never supply an identity association.
                instance_index = None
            anchor = None
            if instance_index is not None:
                anchors = instance_anchor_points(instances, args.anchor_mode)
                if instance_index < len(anchors) and np.isfinite(anchors[instance_index]).all():
                    anchor = anchors[instance_index]
            geometry = {}
            if instance_index is not None:
                for method in self.stores:
                    value = self.instance_geometry.get(method, {}).get(view, {}).get(instance_index)
                    if value is not None:
                        geometry[method] = value
            node_names = tuple(getattr(self.slps[view], "node_names", ("mouth", "tail")))
            self.clicks[phase][target][view] = make_click_evidence(
                experiment=args.experiment,
                view=view,
                frame=self.current_frame,
                click_x=x,
                click_y=y,
                instance_index=instance_index,
                anchor_xy=anchor,
                node_names=node_names,
                anchor_mode=args.anchor_mode,
                geometry_by_method=geometry,
            )
            self.clicks_toggle.setChecked(True)
            self.dirty = True
            self._update_click_status()
            self._update_anchor_summary()
            self._update_suggestion()
            self._render_videos()
            if not already_selected:
                self._cue_paired_identity(phase, target)

        def _has_identity_click(self, phase: str, target: str) -> bool:
            return any(self.clicks[phase][target][view] for view in VIEWS)

        def _cue_identity(self, phase: str, target: str, message: str) -> None:
            phase_index = self.phase_combo.findData(phase)
            target_index = self.target_combo.findData(target)
            self.phase_combo.setCurrentIndex(phase_index)
            self.target_combo.setCurrentIndex(target_index)
            frame = self._frame_for_phase(phase)
            if frame is not None and self.current_frame != frame:
                self._navigate_to(frame)
            self.status_label.setText(message)

        def _cue_paired_identity(self, phase: str, target: str) -> None:
            if not self.auto_cue_checkbox.isChecked():
                return
            other_phase = "anchor_b" if phase == "anchor_a" else "anchor_a"
            target_label = "female" if target == "female" else "same chosen male"
            other_label = "B" if other_phase == "anchor_b" else "A"
            if not self._has_identity_click(other_phase, target):
                self._cue_identity(
                    other_phase,
                    target,
                    f"Now select the {target_label} at Anchor {other_label}.",
                )
                return

            next_target = "anchor_male" if target == "female" else "female"
            next_label = "chosen male" if next_target == "anchor_male" else "female"
            for next_phase, anchor_label in (("anchor_a", "A"), ("anchor_b", "B")):
                if not self._has_identity_click(next_phase, next_target):
                    self._cue_identity(
                        next_phase,
                        next_target,
                        f"{target_label.capitalize()} pair complete. Select the {next_label} at Anchor {anchor_label}.",
                    )
                    return
            self.status_label.setText(
                "Female and the same chosen male are selected at both anchors."
            )

        def _show_identity_menu(
            self, view: str, x: float, y: float, global_position
        ) -> None:
            menu = QtWidgets.QMenu(self)
            menu.setTitle(f"Identity at source frame {self.current_frame:,}")
            for phase, phase_label in (("anchor_a", "Anchor A"), ("anchor_b", "Anchor B")):
                for target, target_label in (
                    ("female", "Female"),
                    ("anchor_male", "Chosen anchor male"),
                ):
                    action = menu.addAction(
                        f"Set current frame as {phase_label} and mark {target_label}"
                    )
                    action.triggered.connect(
                        lambda _checked=False, selected_phase=phase, selected_target=target: self._assign_preview_identity(
                            selected_phase, selected_target, view, x, y
                        )
                    )
                if phase == "anchor_a":
                    menu.addSeparator()
            self._ensure_frame_associations()
            nearest_index, _distance = hit_test_sleap_pose(
                self.instances_by_view.get(view), (x, y), args.click_gate_px
            )
            if nearest_index is not None:
                menu.addSeparator()
                is_excluded = nearest_index in self._excluded_instances(
                    view, self.current_frame
                )
                fp_action = menu.addAction(
                    ("Unmark" if is_excluded else "Mark")
                    + f" SLEAP prediction {nearest_index} as false positive"
                )
                fp_action.triggered.connect(
                    lambda _checked=False: self._toggle_false_positive(view, x, y)
                )
            menu.addSeparator()
            nearby_fn = any(
                item["view"] == view
                and int(item["frame"]) == int(self.current_frame)
                and math.hypot(item["click_x"] - x, item["click_y"] - y)
                <= args.click_gate_px
                for item in self.false_negative_detections
            )
            fn_action = menu.addAction(
                "Remove missing SLEAP fish (FN) marker"
                if nearby_fn
                else "Mark missing SLEAP fish (FN) here"
            )
            fn_action.triggered.connect(
                lambda _checked=False: self._toggle_false_negative(view, x, y)
            )
            menu.exec_(global_position)

        def _toggle_false_negative(self, view: str, x: float, y: float) -> None:
            self.false_negative_detections, added = toggle_false_negative_marker(
                self.false_negative_detections,
                view=view,
                frame=self.current_frame,
                click_x=x,
                click_y=y,
                radius_px=args.click_gate_px,
                marked_at_utc=utc_now(),
            )
            self.dirty = True
            self._render_videos()
            action = "Marked" if added else "Removed"
            self.status_label.setText(
                f"{action} false-negative evidence at {view} frame "
                f"{self.current_frame:,}. This is mining evidence, not an ID assignment."
            )

        def _toggle_false_positive(self, view: str, x: float, y: float) -> None:
            self._ensure_frame_associations()
            instance_index, _distance = hit_test_sleap_pose(
                self.instances_by_view.get(view), (x, y), args.click_gate_px
            )
            if instance_index is None:
                self.status_label.setText(
                    "No SLEAP prediction is close enough to mark as a false positive."
                )
                return
            detection_key = (
                f"{args.experiment}|{view}|{self.current_frame}|{instance_index}"
            )
            existing = next(
                (
                    item
                    for item in self.false_positive_detections
                    if item["detection_key"] == detection_key
                ),
                None,
            )
            if existing is not None:
                self.false_positive_detections.remove(existing)
                action = "restored to ID assignment"
            else:
                self.false_positive_detections.append(
                    {
                        "view": view,
                        "frame": int(self.current_frame),
                        "instance_index": int(instance_index),
                        "detection_key": detection_key,
                        "marked_at_utc": utc_now(),
                    }
                )
                self.false_positive_detections = validate_false_positive_detections(
                    self.false_positive_detections
                )
                action = "excluded from ID assignment and 2D scoring"
                for phase in PHASES:
                    for target in IDENTITY_TARGETS:
                        evidence = self.clicks[phase][target][view]
                        if evidence.get("detection_key") != detection_key:
                            continue
                        evidence.update(
                            {
                                "selection_status": "free_click",
                                "detection_key": "",
                                "instance_index": None,
                                "anchor_x": None,
                                "anchor_y": None,
                                "geometry_by_method": {},
                            }
                        )
            self.association_frame = None
            self.sleap_toggle.setChecked(True)
            self.dirty = True
            self._update_click_status()
            self._update_anchor_summary()
            self._update_suggestion()
            self._refresh_plots()
            self._render_videos()
            self.status_label.setText(
                f"SLEAP prediction {instance_index} at {view} frame {self.current_frame:,}: {action}."
            )

        def _assign_preview_identity(
            self, phase: str, target: str, view: str, x: float, y: float
        ) -> None:
            """Right-click fast path: set phase frame and store one raw identity click."""
            phase_index = self.phase_combo.findData(phase)
            target_index = self.target_combo.findData(target)
            if phase_index < 0 or target_index < 0:
                raise ValueError("unknown identity phase/target")
            expected = self._frame_for_phase(phase)
            if expected != self.current_frame:
                self._set_phase_frame(phase)
            else:
                self.phase_combo.setCurrentIndex(phase_index)
            self.target_combo.setCurrentIndex(target_index)
            self._record_click(view, x, y)

        def _set_phase_frame(self, phase: str) -> None:
            old = self._frame_for_phase(phase)
            if old != self.current_frame:
                self.clicks[phase] = {
                    target: {view: {} for view in VIEWS} for target in IDENTITY_TARGETS
                }
            if phase == "anchor_a":
                self.anchor_a_frame = self.current_frame
            else:
                self.anchor_b_frame = self.current_frame
            index = self.phase_combo.findData(phase)
            self.phase_combo.setCurrentIndex(index)
            self.dirty = True
            self._update_phase_label()
            self._update_click_status()
            self._update_anchor_summary()
            self._render_videos()

        def _update_phase_label(self) -> None:
            anchor_a = "unset" if self.anchor_a_frame is None else f"{self.anchor_a_frame:,}"
            anchor_b = "unset" if self.anchor_b_frame is None else f"{self.anchor_b_frame:,}"
            self.phase_frame_label.setText(
                f"Anchor A: {anchor_a} | Anchor B: {anchor_b}"
            )

        def _update_click_status(self, *_args) -> None:
            phase = str(self.phase_combo.currentData())
            target = str(self.target_combo.currentData())
            values = self.clicks[phase][target]
            labels = []
            for view in VIEWS:
                evidence = values[view]
                labels.append(
                    f"{view}: {evidence.get('selection_status', 'none') if evidence else 'none'}"
                )
            self.click_status.setText(" | ".join(labels))

        def _update_anchor_summary(self) -> None:
            summaries = []
            for phase in PHASES:
                frame = self._frame_for_phase(phase)
                parts = []
                for target, short_label in (("female", "F"), ("anchor_male", "M")):
                    selected_views = [
                        view for view in VIEWS if self.clicks[phase][target][view]
                    ]
                    mark = "✓" if selected_views else "missing"
                    parts.append(
                        f"{short_label}: {mark}"
                        + (f" ({len(selected_views)} view)" if selected_views else "")
                    )
                frame_label = "unset" if frame is None else f"frame {frame:,}"
                anchor_label = "A" if phase == "anchor_a" else "B"
                summaries.append(f"Anchor {anchor_label} {frame_label} — " + ", ".join(parts))
            self.anchor_summary.setText("\n".join(summaries))

        def _clear_active_clicks(self) -> None:
            phase = str(self.phase_combo.currentData())
            target = str(self.target_combo.currentData())
            self.clicks[phase][target] = {view: {} for view in VIEWS}
            self.dirty = True
            self._update_click_status()
            self._update_anchor_summary()
            self._update_suggestion()
            self._render_videos()

        def _update_suggestion(self, *_args) -> None:
            method = str(self.assessed_method.currentText())
            suggestion = suggest_anchor_mapping(self.clicks, method)
            self.suggestion_label.setText(
                f"Automatic mapping suggestion ({method}): {SUGGESTION_LABELS[suggestion]}. "
                "This never changes the human tracker assessment."
            )

        def _update_detector(self) -> None:
            if self.exit_spin.value() <= self.enter_spin.value():
                self.status_label.setText("Exit threshold must be greater than enter threshold.")
                return
            config = DetectorConfig(
                enter_mm=self.enter_spin.value(),
                exit_mm=self.exit_spin.value(),
                merge_gap_frames=args.merge_gap_frames,
                context_frames=args.context_frames,
            )
            try:
                config.validate()
            except ValueError as error:
                self.status_label.setText(str(error))
                return
            store = self.stores[args.detector_method]
            candidates = detect_contact_episodes(
                store, config, experiment=args.experiment, tracking_method=args.detector_method
            )
            manual_rows = self.candidates.loc[
                self.candidates["manual"].fillna(False).astype(str).str.casefold().isin(
                    {"true", "1", "yes", "y"}
                )
            ].copy()
            if not manual_rows.empty:
                candidates = pd.concat([candidates, manual_rows], ignore_index=True)
                candidates = candidates.drop_duplicates("episode_id", keep="last").reset_index(drop=True)
            write_candidate_artifacts(
                candidates,
                args.candidates_csv,
                args.detector_summary_json,
                experiment=args.experiment,
                tracking_method=args.detector_method,
                config=config,
                source_provenance=store.source_provenance,
            )
            preferred = str(self.current_episode["episode_id"]) if self.current_episode else None
            self.candidates = candidates
            self.current_episode = None
            self._refresh_queue(preferred_episode_id=preferred)
            if self.filtered_indices:
                closest = min(
                    range(len(self.filtered_indices)),
                    key=lambda p: abs(int(self.candidates.iloc[self.filtered_indices[p]]["contact_start_frame"]) - self.current_frame),
                )
                self._go_queue_position(closest)
            self.status_label.setText(
                f"Detector updated: {len(candidates)} candidates; source tracking CSV unchanged."
            )

        def _add_manual_episode(self) -> None:
            value = self.manual_range.text().strip()
            try:
                if "-" in value:
                    raw_start, raw_end = value.split("-", 1)
                    start, end = int(raw_start.strip()), int(raw_end.strip())
                else:
                    start = end = int(value)
            except ValueError:
                self.status_label.setText("Manual range must be a frame or start-end.")
                return
            row = manual_episode(
                experiment=args.experiment,
                tracking_method=args.detector_method,
                start_frame=start,
                end_frame=end,
                frame_min=0,
                frame_max=self.max_frame,
                context_frames=args.context_frames,
                source_provenance=self.stores[args.detector_method].source_provenance,
            )
            if not self.candidates["episode_id"].astype(str).eq(str(row["episode_id"])).any():
                self.candidates = pd.concat(
                    [self.candidates, pd.DataFrame([row], columns=self.candidates.columns)], ignore_index=True
                )
            config = DetectorConfig(
                enter_mm=self.enter_spin.value(),
                exit_mm=self.exit_spin.value(),
                merge_gap_frames=args.merge_gap_frames,
                context_frames=args.context_frames,
            )
            try:
                config.validate()
            except ValueError as error:
                self.status_label.setText(str(error))
                return
            write_candidate_artifacts(
                self.candidates,
                args.candidates_csv,
                args.detector_summary_json,
                experiment=args.experiment,
                tracking_method=args.detector_method,
                config=config,
                source_provenance=self.stores[args.detector_method].source_provenance,
            )
            self.manual_range.clear()
            self.current_episode = None
            self._refresh_queue(preferred_episode_id=str(row["episode_id"]))
            positions = [
                p for p, index in enumerate(self.filtered_indices)
                if str(self.candidates.iloc[index]["episode_id"]) == str(row["episode_id"])
            ]
            if positions:
                self._go_queue_position(positions[0])

        def _layout_axes(self, *_args) -> None:
            visible = [index for index, check in enumerate(self.plot_checks) if check.isChecked()]
            if not visible:
                self.plot_checks[0].setChecked(True)
                return
            height = 0.88 / len(visible)
            for index, axis in enumerate(self.axes):
                if index in visible:
                    order = visible.index(index)
                    bottom = 0.08 + (len(visible) - order - 1) * height
                    axis.set_position([0.07, bottom, 0.9, height * 0.88])
                    axis.set_visible(True)
                else:
                    axis.set_visible(False)
            self.canvas.draw_idle()

        def _refresh_plots(self) -> None:
            if not self.current_episode:
                return
            start = int(self.current_episode["review_start_frame"])
            end = int(self.current_episode["review_end_frame"])
            store = self.stores[self.active_method]
            section = store.slice_positions(start, end)
            frames = store.frames[section]
            colors = pair_colors_for(store.fish_ids)
            pair_keys = store.pair_keys
            fish_colors = {
                fish_id: FISH_ID_COLORS[index % len(FISH_ID_COLORS)]
                for index, fish_id in enumerate(store.fish_ids)
            }
            metric_2d = str(self.metric_2d_combo.currentData() or "displacement")
            for axis in self.axes:
                axis.clear()
            for pair_index, key in enumerate(pair_keys):
                color = colors[key]
                self.axes[0].plot(
                    frames,
                    store.direct_distances[section, pair_index],
                    color=color,
                    linewidth=1.1,
                    marker=".",
                    markersize=2.0,
                    label=key,
                )
                self.axes[0].plot(
                    frames,
                    store.interpolated_distances[section, pair_index],
                    color=color,
                    linewidth=0.9,
                    linestyle=":",
                    marker=".",
                    markersize=1.8,
                    alpha=0.45,
                )
            metric_label = (
                "XY step (px/frame)" if metric_2d == "displacement" else "2D IFD (px)"
            )
            labels = ["3D IFD (mm)"] + [
                f"{view.replace('_cam', '')} {metric_label}" for view in VIEWS
            ]
            for axis_index, view in enumerate(VIEWS, start=1):
                self.status_label.setText(
                    f"Calculating lazy {view} 2D metrics for {start:,}–{end:,}…"
                )
                QtWidgets.QApplication.processEvents()
                values = self.ifd2d_cache.get(
                    store,
                    self.slps[view],
                    self.calibration,
                    view=view,
                    start_frame=start,
                    end_frame=end,
                    anchor_mode=args.anchor_mode,
                    gate_px=args.match_gate_px,
                    excluded_instances_by_frame=self._excluded_by_frame(view),
                )
                if metric_2d == "displacement":
                    series = values.displacements
                    trace_keys = store.fish_ids
                    trace_colors = fish_colors
                else:
                    series = values.distances
                    trace_keys = pair_keys
                    trace_colors = colors
                for trace_index, key in enumerate(trace_keys):
                    self.axes[axis_index].plot(
                        values.frames,
                        series[:, trace_index],
                        color=trace_colors[key],
                        linewidth=1.0,
                        marker=".",
                        markersize=2.0,
                        label=key,
                    )
            active_id = str(self.current_episode["episode_id"])
            for axis in self.axes:
                for row in self.candidates.to_dict("records"):
                    candidate_start = int(row["contact_start_frame"])
                    candidate_end = int(row["contact_end_frame"])
                    if candidate_end < start or candidate_start > end:
                        continue
                    is_active = str(row["episode_id"]) == active_id
                    axis.axvspan(
                        candidate_start,
                        candidate_end,
                        color="#e15759",
                        alpha=0.24 if is_active else 0.07,
                        zorder=-5,
                    )
                axis.set_ylabel(labels[self.axes.index(axis)], fontsize=8)
                axis.grid(True, alpha=0.18)
                axis.set_xlim(start, end)
            self.axes[0].legend(loc="upper right", ncol=3, fontsize=7)
            for axis in self.axes[1:]:
                axis.legend(loc="upper right", ncol=3, fontsize=7)
            self.axes[-1].set_xlabel("Exact source frame")
            self.cursor_lines = [
                axis.axvline(self.current_frame, color="#222222", linewidth=0.9, alpha=0.85)
                for axis in self.axes
            ]
            self._layout_axes()
            self.canvas.draw_idle()
            self.status_label.setText("Ready")

        def _update_plot_cursor(self) -> None:
            for line in self.cursor_lines:
                line.set_xdata([self.current_frame, self.current_frame])
            if self.cursor_lines:
                self.canvas.draw_idle()

        def _plot_press(self, event) -> None:
            if event.button == 1 and event.xdata is not None and event.inaxes in self.axes:
                self.dragging_plot = True
                self.plot_frame_clicked(int(round(event.xdata)))

        def _plot_motion(self, event) -> None:
            if self.dragging_plot and event.xdata is not None and event.inaxes in self.axes:
                self.plot_frame_clicked(int(round(event.xdata)))

        def _plot_release(self, _event) -> None:
            self.dragging_plot = False

        def _save(self) -> None:
            if not self.current_episode:
                self.status_label.setText("No active episode to save.")
                return
            if self.anchor_a_frame is None or self.anchor_b_frame is None:
                self.status_label.setText("Set both Anchor A and Anchor B before saving.")
                return
            tracker_assessment = self._decision(self.tracker_group)
            evidence_quality = str(self.evidence_quality.currentData())
            if tracker_assessment != "observations_unusable" and evidence_quality != "unusable":
                missing_targets = []
                for phase in PHASES:
                    for target in IDENTITY_TARGETS:
                        if not any(self.clicks[phase][target][view] for view in VIEWS):
                            anchor_label = "A" if phase == "anchor_a" else "B"
                            missing_targets.append(f"Anchor {anchor_label} {target}")
                if missing_targets:
                    self.status_label.setText(
                        "Raw click evidence is missing for: " + ", ".join(missing_targets)
                    )
                    return
            try:
                row = make_annotation_row(
                    experiment=args.experiment,
                    episode=self.current_episode,
                    annotator=args.annotator,
                    anchor_a_frame=self.anchor_a_frame,
                    anchor_b_frame=self.anchor_b_frame,
                    clicks=self.clicks,
                    false_positive_detections=self.false_positive_detections,
                    false_negative_detections=self.false_negative_detections,
                    contact_validity=self._decision(self.contact_group),
                    tracker_assessment=tracker_assessment,
                    assessed_tracking_label=str(self.assessed_method.currentText()),
                    evidence_quality=evidence_quality,
                    issue_flags=self.preserved_issue_flags,
                    notes=self.notes.toPlainText(),
                    source_provenance=self.source_provenance,
                )
                action = self.ledger.upsert(row)
            except ValueError as error:
                self.status_label.setText(str(error))
                return
            self.dirty = False
            episode_id = str(self.current_episode["episode_id"])
            self._refresh_queue(preferred_episode_id=episode_id)
            self.status_label.setText(f"Annotation {action} saved atomically.")

        def _undo(self) -> None:
            if not self.ledger.can_undo:
                self.status_label.setText("Nothing from this session to undo.")
                return
            self.ledger.undo()
            self._load_saved_or_reset()
            self._refresh_queue()
            self.status_label.setText("Last session edit undone; audit record appended.")

    return ContactWindow()


def run_gui(
    args: argparse.Namespace,
    *,
    videos: Mapping[str, Any],
    slp_predictions: Mapping[str, Any],
    tracking_stores: Mapping[str, TrackingIFD],
    calibration: Any,
    candidates: pd.DataFrame,
    ledger: ContactAnnotationLedger,
    source_provenance: Mapping[str, Any],
    smoke_actions: bool = False,
) -> int:
    QtCore, _QtGui, QtWidgets, _Canvas, _Toolbar, _Figure = _load_gui_dependencies()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = create_annotation_window(
        args,
        videos=videos,
        slp_predictions=slp_predictions,
        tracking_stores=tracking_stores,
        calibration=calibration,
        candidates=candidates,
        ledger=ledger,
        source_provenance=source_provenance,
    )
    window.show()
    if smoke_actions:
        def exercise():
            try:
                episode = window.current_episode
                assert episode is not None
                assert window.metric_2d_combo.currentData() == "displacement"
                window.metric_2d_combo.setCurrentIndex(
                    window.metric_2d_combo.findData("ifd")
                )
                window.metric_2d_combo.setCurrentIndex(
                    window.metric_2d_combo.findData("displacement")
                )
                anchor_a, anchor_b = suggest_anchor_frames(
                    window.stores[args.detector_method], episode
                )
                window.plot_frame_clicked(anchor_a)
                window._ensure_frame_associations()
                false_positive_xy = window.projected_by_method[args.detector_method]["top_cam"]["ID_3"]
                window._toggle_false_positive(
                    "top_cam", float(false_positive_xy[0]), float(false_positive_xy[1])
                )
                window._toggle_false_negative("side2_cam", 120.0, 140.0)
                female_xy = window.projected_by_method[args.detector_method]["top_cam"]["ID_1"]
                window._assign_preview_identity(
                    "anchor_a", "female", "top_cam", float(female_xy[0]), float(female_xy[1])
                )
                assert window.current_frame == anchor_b
                female_xy = window.projected_by_method[args.detector_method]["top_cam"]["ID_1"]
                window._assign_preview_identity(
                    "anchor_b", "female", "top_cam", float(female_xy[0]), float(female_xy[1])
                )
                assert window.current_frame == anchor_a
                male_xy = window.projected_by_method[args.detector_method]["top_cam"]["ID_2"]
                window._assign_preview_identity(
                    "anchor_a", "anchor_male", "top_cam", float(male_xy[0]), float(male_xy[1])
                )
                assert window.current_frame == anchor_b
                male_xy = window.projected_by_method[args.detector_method]["top_cam"]["ID_2"]
                window._assign_preview_identity(
                    "anchor_b", "anchor_male", "top_cam", float(male_xy[0]), float(male_xy[1])
                )
                window._set_decision(window.contact_group, "confirmed_contact_crossing")
                window._set_decision(window.tracker_group, "correct_continuity")
                window.evidence_quality.setCurrentIndex(window.evidence_quality.findData("usable"))
                window._save()
                if len(ledger.rows) != 1:
                    raise AssertionError("smoke annotation was not saved")
                window.close()
                app.exit(0)
            except Exception:
                import traceback
                traceback.print_exc()
                window.close()
                app.exit(2)

        QtCore.QTimer.singleShot(100, exercise)
    return int(app.exec_())


class SyntheticVideo:
    def __init__(self, view: str, frame_count: int = 20):
        self.view = view
        self.path = Path(f"synthetic/{view}.mp4")
        self.frame_count = frame_count
        self.width = 320
        self.height = 220
        self.reported_fps = 166.0

    def get(self, frame: int) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        cv2.putText(image, f"{self.view} frame {frame}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        return image

    def close(self) -> None:
        pass


class SyntheticCalibration:
    backend_name = "synthetic"
    provenance = {"backend": "synthetic", "model_files": {}}

    def project_world(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float).reshape(-1, 3)
        result = np.full((len(points), 3, 2), np.nan, dtype=float)
        result[:, 0] = points[:, [0, 1]]
        result[:, 1] = points[:, [0, 2]]
        result[:, 2] = points[:, [1, 2]]
        return result


class SyntheticSlp:
    def __init__(self, view: str, store: TrackingIFD, calibration: SyntheticCalibration):
        self.view = view
        self.path = Path(f"synthetic/{view}.slp")
        self.node_names = ("mouth", "tail")
        self.store = store
        self.calibration = calibration

    def get(self, frame: int, default=None):
        try:
            xyz = np.stack(list(self.store.xyz_at(frame).values()))
        except IndexError:
            return default
        projected = self.calibration.project_world(xyz)[:, CAMERA_INDEX[self.view]]
        poses = np.stack([np.stack([point, point + np.array([-12.0, 0.0])]) for point in projected])
        return poses

    def close(self) -> None:
        pass


def _synthetic_store(path: Path) -> TrackingIFD:
    rows = []
    for frame in range(20):
        positions = {
            "ID_1": (80.0 + frame, 80.0, 100.0),
            "ID_2": (100.0 - frame, 80.0, 100.0),
            "ID_3": (220.0, 150.0, 150.0),
        }
        for fish_id, xyz in positions.items():
            rows.append({"frame": frame, "fish_id": fish_id, "X": xyz[0], "Y": xyz[1], "Z": xyz[2], "is_interpolated": False})
    pd.DataFrame(rows).to_csv(path, index=False)
    return prepare_tracking_ifd(path, method="synthetic")[0]


def run_smoke_test(args: argparse.Namespace) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    temporary = tempfile.TemporaryDirectory(prefix="contact-episode-smoke-")
    root = Path(temporary.name)
    if args.annotations_csv is None:
        args.annotations_csv = root / "annotations.csv"
    if args.audit_jsonl is None:
        args.audit_jsonl = Path(args.annotations_csv).with_suffix(".audit.jsonl")
    args.candidates_csv = root / "candidates.csv"
    args.detector_summary_json = root / "summary.json"
    args.experiment = args.experiment or "synthetic_experiment"
    args.annotator = args.annotator or "qt-smoke"
    args.detector_method = "synthetic"
    store = _synthetic_store(root / "tracks.csv")
    config = DetectorConfig(args.enter_mm, args.exit_mm, args.merge_gap_frames, args.context_frames)
    candidates = detect_contact_episodes(store, config, experiment=args.experiment)
    write_candidate_artifacts(candidates, args.candidates_csv, args.detector_summary_json, experiment=args.experiment, tracking_method="synthetic", config=config, source_provenance=store.source_provenance)
    calibration = SyntheticCalibration()
    videos = {view: SyntheticVideo(view) for view in VIEWS}
    slps = {view: SyntheticSlp(view, store, calibration) for view in VIEWS}
    provenance = {"tracking_csvs": {"synthetic": store.source_provenance}, "synthetic": True}
    ledger = ContactAnnotationLedger(args.annotations_csv, args.audit_jsonl)
    try:
        result = run_gui(
            args,
            videos=videos,
            slp_predictions=slps,
            tracking_stores={"synthetic": store},
            calibration=calibration,
            candidates=candidates,
            ledger=ledger,
            source_provenance=provenance,
            smoke_actions=True,
        )
        reopened = ContactAnnotationLedger(args.annotations_csv, args.audit_jsonl)
        if result == 0 and len(reopened.rows) != 1:
            raise AssertionError("reopened smoke ledger did not contain one row")
        return result
    finally:
        temporary.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--tracking-csv", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--detector-method")
    parser.add_argument("--tracking-id-column")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--slp-dir", type=Path)
    parser.add_argument("--sleap-subdir", default="")
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--calibration-backend", choices=("auto", "refractive", "polynomial"), default="auto")
    parser.add_argument("--anchor-mode", choices=("mouth", "visible-node-centroid"), default="mouth")
    parser.add_argument("--match-gate-px", type=float, default=80.0)
    parser.add_argument("--click-gate-px", type=float, default=35.0)
    parser.add_argument("--tail-frames", type=int, default=30)
    parser.add_argument(
        "--scene-cache-mb",
        type=int,
        default=768,
        help="total RAM budget across the three background raw-frame scene caches",
    )
    parser.add_argument(
        "--scene-cache-radius",
        type=int,
        help="frames prefetched before/after the current cursor (default: context frames)",
    )
    parser.add_argument("--enter-mm", type=float, default=12.0)
    parser.add_argument("--exit-mm", type=float, default=16.0)
    parser.add_argument("--merge-gap-frames", type=int, default=5)
    parser.add_argument("--context-frames", type=int, default=166)
    parser.add_argument("--source-fps", type=float, default=166.0)
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--candidates-csv", type=Path)
    parser.add_argument("--detector-summary-json", type=Path)
    parser.add_argument("--ifd-cache-dir", type=Path)
    parser.add_argument("--annotator", default="")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _resolved_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.smoke_test:
        args.experiment = args.experiment or "synthetic_experiment"
        args.annotator = args.annotator or "qt-smoke"
    elif not args.experiment:
        parser.error("experiment is required")
    if not args.annotator.strip():
        parser.error("--annotator NAME is required")
    try:
        DetectorConfig(args.enter_mm, args.exit_mm, args.merge_gap_frames, args.context_frames).validate()
    except ValueError as error:
        parser.error(str(error))
    if args.scene_cache_radius is None:
        args.scene_cache_radius = args.context_frames
    if (
        args.source_fps <= 0
        or args.match_gate_px < 0
        or args.click_gate_px < 0
        or args.tail_frames < 0
        or args.scene_cache_mb <= 0
        or args.scene_cache_radius <= 0
    ):
        parser.error("FPS/gates/tail settings are outside their valid ranges")
    if args.smoke_test:
        return args
    try:
        tracking_paths = parse_tracking_csv_specs(args.tracking_csv)
    except ValueError as error:
        parser.error(str(error))
    args.tracking_paths = tracking_paths
    args.detector_method = args.detector_method or next(iter(tracking_paths))
    if args.detector_method not in tracking_paths:
        parser.error("--detector-method must name one of the --tracking-csv inputs")
    local = load_local_paths(args.paths_config)
    if args.video_dir is None and local and local.medaka_data_dir:
        args.video_dir = local.medaka_data_dir / args.experiment
    if args.slp_dir is None and local and local.sleap_output_dir:
        args.slp_dir = local.sleap_output_dir / args.experiment
    if args.slp_dir is not None and args.sleap_subdir:
        args.slp_dir = args.slp_dir / args.sleap_subdir
    if args.calibration_model_dir is None and local and local.calibration_model_dir:
        args.calibration_model_dir = local.calibration_model_dir
    annotation_root = REPO_ROOT / "annotations" / "contact_episode_gt" / args.experiment
    args.annotations_csv = args.annotations_csv or annotation_root / "contact_episode_gt.csv"
    args.audit_jsonl = args.audit_jsonl or args.annotations_csv.with_suffix(".audit.jsonl")
    args.candidates_csv = args.candidates_csv or annotation_root / f"contact_candidates_{args.detector_method}.csv"
    args.detector_summary_json = args.detector_summary_json or annotation_root / f"contact_candidates_{args.detector_method}_summary.json"
    args.ifd_cache_dir = args.ifd_cache_dir or annotation_root / "ifd_cache"
    for label in ("video_dir", "slp_dir", "calibration_model_dir"):
        if getattr(args, label) is None:
            parser.error(f"--{label.replace('_', '-')} is required when it cannot be resolved from paths config")
    outputs = [args.annotations_csv, args.audit_jsonl, args.candidates_csv, args.detector_summary_json]
    normalized_outputs = [os.path.normcase(os.path.abspath(str(path))) for path in outputs]
    if len(set(normalized_outputs)) != len(normalized_outputs):
        parser.error("annotation, audit, candidate, and summary outputs must be distinct")
    source_paths = {os.path.normcase(os.path.abspath(str(path))) for path in tracking_paths.values()}
    collision = [str(path) for path in outputs if os.path.normcase(os.path.abspath(str(path))) in source_paths]
    if collision:
        parser.error("refusing to overwrite tracking source: " + ", ".join(collision))
    return args


def _load_real_session(args: argparse.Namespace):
    from scripts.reconstruction_3d.calibration import Calibration

    for method, path in args.tracking_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing tracking CSV for {method}: {path}")
    video_paths = {view: Path(args.video_dir) / view / VIDEO_NAMES[view] for view in VIEWS}
    slp_paths = {view: Path(args.slp_dir) / f"{view}.slp" for view in VIEWS}
    missing = [str(path) for path in (*video_paths.values(), *slp_paths.values(), Path(args.calibration_model_dir)) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required read-only inputs:\n  " + "\n  ".join(missing))

    stores: dict[str, TrackingIFD] = {}
    for method, path in args.tracking_paths.items():
        cache_path = Path(args.ifd_cache_dir) / f"{method}.ifd.npz"
        print(f"Loading compact 3D IFD data for {method}: {path}", flush=True)
        store, cache_hit = prepare_tracking_ifd(
            path,
            method=method,
            cache_path=cache_path,
            id_column=args.tracking_id_column,
        )
        stores[method] = store
        print(
            f"  {'cache hit' if cache_hit else 'cache built'}: {cache_path} ({store.frame_max + 1:,} frames)",
            flush=True,
        )
    config = DetectorConfig(args.enter_mm, args.exit_mm, args.merge_gap_frames, args.context_frames)
    detector_store = stores[args.detector_method]
    candidates = detect_contact_episodes(
        detector_store,
        config,
        experiment=args.experiment,
        tracking_method=args.detector_method,
    )
    manual_rows = _manual_candidate_rows(args.candidates_csv)
    if not manual_rows.empty:
        manual_rows = manual_rows.loc[
            manual_rows.get("experiment", "").astype(str).eq(args.experiment)
            & manual_rows.get("tracking_method", "").astype(str).eq(args.detector_method)
        ]
        if not manual_rows.empty:
            candidates = pd.concat(
                [candidates, manual_rows[candidates.columns]], ignore_index=True
            ).drop_duplicates("episode_id", keep="last").reset_index(drop=True)
    summary = write_candidate_artifacts(
        candidates,
        args.candidates_csv,
        args.detector_summary_json,
        experiment=args.experiment,
        tracking_method=args.detector_method,
        config=config,
        source_provenance=detector_store.source_provenance,
    )
    print(f"Wrote {len(candidates):,} candidates: {args.candidates_csv}", flush=True)
    print(f"Wrote detector summary: {args.detector_summary_json}", flush=True)

    videos: dict[str, Any] = {}
    slps: dict[str, IndexedSlpPredictions] = {}
    try:
        videos = {view: OpenCvVideo(path) for view, path in video_paths.items()}
        counts = {view: source.frame_count for view, source in videos.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f"video frame counts disagree: {counts}")
        max_frame = min(counts.values()) - 1
        for method, store in stores.items():
            if store.frame_max > max_frame:
                raise ValueError(f"{method} extends to {store.frame_max}, beyond video frame {max_frame}")
        for view, path in slp_paths.items():
            source = IndexedSlpPredictions(path, node_names=("mouth", "tail"), cache_size=1024)
            indices = np.asarray(source._frame_indices, dtype=np.int64)
            if len(indices) and (int(indices.min()) < 0 or int(indices.max()) > max_frame):
                source.close()
                raise ValueError(f"{view} SLP extent does not fit video source frames")
            slps[view] = source
        calibration = Calibration(args.calibration_model_dir, backend=args.calibration_backend)
        provenance = {
            "tracking_csvs": {method: store.source_provenance for method, store in stores.items()},
            "video_paths": {view: str(path.resolve(strict=False)) for view, path in video_paths.items()},
            "slp_paths": {view: str(path.resolve(strict=False)) for view, path in slp_paths.items()},
            "calibration": calibration.provenance,
            "anchor_mode": args.anchor_mode,
            "detector_summary": summary,
        }
        return stores, candidates, videos, slps, calibration, provenance
    except Exception:
        for source in videos.values():
            source.close()
        for source in slps.values():
            source.close()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _resolved_args(parser.parse_args(argv), parser)
    if args.smoke_test:
        return run_smoke_test(args)
    stores: dict[str, TrackingIFD] = {}
    videos: dict[str, OpenCvVideo] = {}
    slps: dict[str, IndexedSlpPredictions] = {}
    try:
        stores, candidates, videos, slps, calibration, provenance = _load_real_session(args)
        ledger = ContactAnnotationLedger(args.annotations_csv, args.audit_jsonl)
        candidates = _merge_saved_ledger_candidates(candidates, ledger)
        print(
            f"Preflight OK: {len(candidates):,} episodes, {len(stores)} tracking method(s), "
            f"calibration={calibration.backend_name}.",
            flush=True,
        )
        per_view_cache_bytes = max(
            1, int(args.scene_cache_mb * 1024 * 1024 // len(VIEWS))
        )
        if args.preflight:
            if not candidates.empty:
                center = int(candidates.iloc[0]["minimum_ifd_frame"])
            else:
                center = min(2, videos["top_cam"].frame_count - 1)
            start = max(0, center - 2)
            end = min(videos["top_cam"].frame_count - 1, center + 2)
            probe = SceneCachedVideo(
                videos["top_cam"], max_cache_bytes=per_view_cache_bytes
            )
            videos["top_cam"] = probe
            probe.prime(start, end, center)
            probe.wait_until_ready(timeout=30)
            info = probe.cache_info()
            if start < center and int(info["backward_frames"]) < 1:
                raise RuntimeError("scene-cache preflight did not cache backward frames")
            if end > center and int(info["forward_frames"]) < 2:
                raise RuntimeError("scene-cache preflight did not cache forward frames")
            probe.get(max(start, center - 1))
            probe.get(min(end, center + 1))
            print(
                "Scene-cache preflight OK: "
                f"forward={info['forward_frames']}, backward={info['backward_frames']}, "
                f"RAM={int(info['cached_bytes']) / (1024**2):.1f} MB.",
                flush=True,
            )
            return 0
        videos = {
            view: SceneCachedVideo(source, max_cache_bytes=per_view_cache_bytes)
            for view, source in videos.items()
        }
        return run_gui(
            args,
            videos=videos,
            slp_predictions=slps,
            tracking_stores=stores,
            calibration=calibration,
            candidates=candidates,
            ledger=ledger,
            source_provenance=provenance,
        )
    finally:
        for source in videos.values():
            source.close()
        for source in slps.values():
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
