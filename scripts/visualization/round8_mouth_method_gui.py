"""Interactive three-view comparison of Round8 mouth-tracking methods.

The viewer is intentionally read-only. It overlays:

* synchronized raw frames from ``top_cam``, ``side1_cam``, and ``side2_cam``;
* the authenticated, mouth-required Round8 candidate cache (up to six raw
  SLEAP candidates, retaining the source-instance index);
* one switchable tracking result: A0, A1, A2, or A4;
* exact selected candidate slots where the method records them; and
* confirmed female ground-truth clicks on their exact annotated frames.

It also derives a review queue from changes in the independently scored female
assignment sequence. For each change, the display window is centered on the
closest 3D approach of the before/after method identities. A4 was never run
full-length, so the GUI preserves its bounded-window abstentions instead of
inventing tracks outside the evaluated windows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, REPO_ROOT, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


EXPERIMENT_DEFAULT = "20251225_K_1_2_FCMC"
ROUND_TAG_DEFAULT = "round8_260729_snn031_m6"
STUDY_DIR_DEFAULT = (
    REPO_ROOT / "report" / "20260731" / "round8_mouth_tracking_ablation"
)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_INDEX = {view: index for index, view in enumerate(VIEWS)}
VIDEO_NAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
METHODS = ("A0", "A1", "A2", "A4")
METHOD_DESCRIPTIONS = {
    "A0": "visible-node-centroid reference",
    "A1": "strict-mouth, position-only",
    "A2": "strict-mouth with robust velocity",
    "A4": "bounded reciprocal tracklet paths (abstaining)",
}
METHOD_ANCHORS = {"A0": "visible centroid", "A1": "mouth", "A2": "mouth", "A4": "mouth"}

# BGR colors. The same base identity keeps one color while its segment suffix
# remains visible in the label.
IDENTITY_COLORS = (
    (48, 48, 230),
    (220, 130, 35),
    (30, 180, 245),
    (160, 70, 200),
    (70, 190, 90),
    (210, 200, 50),
)
RAW_COLOR = (155, 155, 155)
FEMALE_COLOR = (230, 70, 230)
HYPOTHESIS_SLOT_RE = re.compile(r"_s(-?\d+)_(-?\d+)_(-?\d+)$")


def _clean_id(value: Any) -> str:
    """Normalize CSV IDs without turning integer-looking floats into ``"8.0"``."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _segment_text(value: Any) -> str:
    cleaned = _clean_id(value)
    return cleaned if cleaned else "-1"


def method_identity_key(base_id: Any, segment: Any, *, include_segment: bool) -> str:
    base = _clean_id(base_id)
    if not base:
        return ""
    if not include_segment:
        return base
    return f"{base}#identity_segment={_segment_text(segment)}"


def split_method_identity_key(key: str) -> tuple[str, str]:
    marker = "#identity_segment="
    if marker not in str(key):
        return str(key), ""
    return tuple(str(key).split(marker, 1))  # type: ignore[return-value]


def parse_hypothesis_slots(value: Any) -> tuple[int, int, int] | None:
    match = HYPOTHESIS_SLOT_RE.search("" if value is None else str(value))
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _identity_sort_key(value: str) -> tuple[str, int, str]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if match:
        return text[: match.start()], int(match.group(1)), text
    return text, -1, text


@dataclass(frozen=True)
class MethodSource:
    label: str
    paths: tuple[Path, ...]
    include_segment: bool
    anchor: str


@dataclass
class FemaleChangeEvent:
    source_method: str
    previous_frame: int
    next_frame: int
    from_id: str
    to_id: str
    kind: str = "female_assignment_change"
    fixed_center: int | None = None
    center: int | None = None
    minimum_distance_mm: float | None = None
    note: str = ""

    @property
    def label(self) -> str:
        if self.kind == "female_assignment_change":
            center = "pending" if self.center is None else f"f{self.center}"
            return (
                f"{self.source_method} F assignment {self.from_id} → {self.to_id} "
                f"({center}; GT f{self.previous_frame}→f{self.next_frame})"
            )
        center = self.fixed_center if self.fixed_center is not None else self.center
        return f"{self.source_method} {self.note} (f{center})"


@dataclass(frozen=True)
class CandidatePose:
    slot: int
    source_instance: int
    mouth: np.ndarray
    tail: np.ndarray
    mouth_score: float
    tail_score: float

    def anchor(self, policy: str) -> np.ndarray:
        if policy == "mouth":
            return np.asarray(self.mouth, dtype=float)
        visible = [
            point
            for point in (np.asarray(self.mouth, float), np.asarray(self.tail, float))
            if np.all(np.isfinite(point))
        ]
        if not visible:
            return np.full(2, np.nan, dtype=float)
        return np.mean(np.stack(visible), axis=0)


class CandidateCache:
    """Small random-access reader for the authenticated Round8 candidate HDF5s."""

    REQUIRED = {"mouth_xy", "tail_xy", "source_instance"}

    def __init__(self, root: Path):
        import h5py

        self.root = Path(root)
        self._files: dict[str, Any] = {}
        self.frame_count: int | None = None
        for view in VIEWS:
            path = self.root / f"{view}.candidates.h5"
            handle = h5py.File(path, "r")
            missing = self.REQUIRED - set(handle)
            if missing:
                handle.close()
                self.close()
                raise ValueError(f"candidate cache {path} is missing {sorted(missing)}")
            count = int(handle.attrs["frame_count"])
            if self.frame_count is None:
                self.frame_count = count
            elif count != self.frame_count:
                handle.close()
                self.close()
                raise ValueError("candidate-cache frame counts disagree across views")
            self._files[view] = handle
        if self.frame_count is None:
            raise ValueError("no candidate caches opened")

    def get(self, view: str, frame: int) -> list[CandidatePose]:
        handle = self._files[view]
        frame = int(frame)
        if frame < 0 or frame >= int(handle.attrs["frame_count"]):
            return []
        mouths = np.asarray(handle["mouth_xy"][frame], dtype=float)
        tails = np.asarray(handle["tail_xy"][frame], dtype=float)
        sources = np.asarray(handle["source_instance"][frame], dtype=int)
        mouth_scores = (
            np.asarray(handle["mouth_score"][frame], dtype=float)
            if "mouth_score" in handle
            else np.full(len(mouths), np.nan)
        )
        tail_scores = (
            np.asarray(handle["tail_score"][frame], dtype=float)
            if "tail_score" in handle
            else np.full(len(mouths), np.nan)
        )
        result = []
        for slot, (mouth, tail) in enumerate(zip(mouths, tails)):
            if not (np.all(np.isfinite(mouth)) or np.all(np.isfinite(tail))):
                continue
            result.append(
                CandidatePose(
                    slot=int(slot),
                    source_instance=int(sources[slot]),
                    mouth=np.asarray(mouth, dtype=float),
                    tail=np.asarray(tail, dtype=float),
                    mouth_score=float(mouth_scores[slot]),
                    tail_score=float(tail_scores[slot]),
                )
            )
        return result

    def close(self) -> None:
        for handle in getattr(self, "_files", {}).values():
            handle.close()
        self._files = {}


class OpenCvVideo:
    """Exact source-frame reader with a small LRU cache."""

    def __init__(self, path: Path, *, cache_size: int = 48):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"could not open video: {self.path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            self.close()
            raise ValueError(f"invalid video metadata: {self.path}")
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame: int) -> np.ndarray:
        frame = int(frame)
        if frame < 0 or frame >= self.frame_count:
            raise IndexError(f"frame {frame} outside 0..{self.frame_count - 1}")
        if frame in self._cache:
            image = self._cache.pop(frame)
            self._cache[frame] = image
            return image.copy()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.capture.read()
        if not ok or image is None:
            raise IOError(f"failed to decode frame {frame} from {self.path}")
        self._cache[frame] = image
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return image.copy()

    def close(self) -> None:
        capture = getattr(self, "capture", None)
        if capture is not None:
            capture.release()
            self.capture = None


class TrackTable:
    """Frame-indexed, method-normalized track table."""

    def __init__(self, label: str, table: pd.DataFrame, *, include_segment: bool):
        self.label = label
        self.include_segment = bool(include_segment)
        required = {"frame", "X", "Y", "Z", "_base_id"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{label} table is missing columns {sorted(missing)}")
        table = table.copy()
        table["frame"] = pd.to_numeric(table["frame"], errors="raise").astype(np.int64)
        table["_base_id"] = table["_base_id"].map(_clean_id)
        if "_segment" not in table:
            table["_segment"] = "-1"
        table["_segment"] = table["_segment"].map(_segment_text)
        table["_method_key"] = [
            method_identity_key(base, segment, include_segment=self.include_segment)
            for base, segment in zip(table["_base_id"], table["_segment"])
        ]
        table = table[table["_base_id"].ne("")].sort_values(
            ["frame", "_base_id", "_segment"], kind="stable"
        )
        table.reset_index(drop=True, inplace=True)
        self.table = table
        self._frames = table["frame"].to_numpy(dtype=np.int64)
        bases = sorted(table["_base_id"].unique(), key=_identity_sort_key)
        self.color_index = {base: index for index, base in enumerate(bases)}

    @classmethod
    def from_csvs(cls, source: MethodSource) -> "TrackTable":
        tables = []
        for path in source.paths:
            header = list(pd.read_csv(path, nrows=0).columns)
            wanted = {
                "frame",
                "fish_id",
                "final_fish_id",
                "X",
                "Y",
                "Z",
                "identity_status",
                "identity_segment",
                "source_index",
                "ray_residual_mm2",
                "hypothesis_id",
                "is_interpolated",
                "slot_cam0",
                "slot_cam1",
                "slot_cam2",
                "bind_source",
                "bind_score",
            }
            usecols = [column for column in header if column in wanted]
            table = pd.read_csv(path, usecols=usecols, low_memory=False)
            table["_source_path"] = str(path)
            if source.label == "A0":
                if "final_fish_id" not in table:
                    raise ValueError(f"A0 CSV has no final_fish_id: {path}")
                normalized = table["final_fish_id"].map(_clean_id).astype(object)
                if "fish_id" in table:
                    fallback = "geometry:" + table["fish_id"].map(_clean_id)
                    normalized = normalized.mask(normalized.eq(""), fallback)
                table["_base_id"] = normalized
                table["_segment"] = "-1"
                table["_status"] = (
                    table["bind_source"] if "bind_source" in table else "A0"
                )
            else:
                if "fish_id" not in table:
                    raise ValueError(f"{source.label} CSV has no fish_id: {path}")
                table["_base_id"] = table["fish_id"]
                table["_segment"] = (
                    table["identity_segment"] if "identity_segment" in table else "-1"
                )
                table["_status"] = (
                    table["identity_status"]
                    if "identity_status" in table
                    else source.label
                )
            tables.append(table)
        if not tables:
            raise ValueError(f"{source.label} has no input CSVs")
        combined = pd.concat(tables, ignore_index=True)
        if len(tables) > 1:
            subset = ["frame", "_base_id", "_segment"]
            duplicated = combined.duplicated(subset, keep=False)
            if duplicated.any():
                groups = combined.loc[duplicated].groupby(subset, dropna=False)
                for key, group in groups:
                    xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
                    finite = np.all(np.isfinite(xyz), axis=1)
                    if finite.sum() > 1 and not np.allclose(
                        xyz[finite], xyz[finite][0], equal_nan=True
                    ):
                        raise ValueError(
                            f"{source.label} bounded CSVs conflict at {key}"
                        )
                combined = combined.drop_duplicates(subset, keep="first")
        return cls(source.label, combined, include_segment=source.include_segment)

    def rows_at(self, frame: int) -> pd.DataFrame:
        left = int(np.searchsorted(self._frames, int(frame), side="left"))
        right = int(np.searchsorted(self._frames, int(frame), side="right"))
        return self.table.iloc[left:right]

    def rows_between(self, start: int, end: int) -> pd.DataFrame:
        left = int(np.searchsorted(self._frames, int(start), side="left"))
        right = int(np.searchsorted(self._frames, int(end), side="right"))
        return self.table.iloc[left:right]

    def color(self, base_id: str) -> tuple[int, int, int]:
        index = self.color_index.get(str(base_id), len(self.color_index))
        return IDENTITY_COLORS[index % len(IDENTITY_COLORS)]

    def closest_approach(
        self, first_key: str, second_key: str, start: int, end: int
    ) -> tuple[int, float] | None:
        window = self.rows_between(start, end)
        finite = np.all(
            np.isfinite(window[["X", "Y", "Z"]].to_numpy(dtype=float)), axis=1
        )
        window = window.loc[finite]
        first = window.loc[
            window["_method_key"].eq(first_key), ["frame", "X", "Y", "Z"]
        ]
        second = window.loc[
            window["_method_key"].eq(second_key), ["frame", "X", "Y", "Z"]
        ]
        if first.empty or second.empty:
            return None
        paired = first.merge(second, on="frame", suffixes=("_a", "_b"))
        if paired.empty:
            return None
        delta = (
            paired[["X_a", "Y_a", "Z_a"]].to_numpy(dtype=float)
            - paired[["X_b", "Y_b", "Z_b"]].to_numpy(dtype=float)
        )
        distance = np.linalg.norm(delta, axis=1)
        index = int(np.argmin(distance))
        return int(paired.iloc[index]["frame"]), float(distance[index])


def build_female_change_events(assignments: pd.DataFrame) -> list[FemaleChangeEvent]:
    required = {"method", "frame", "consensus_method_id"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"female assignment CSV is missing {sorted(missing)}")
    events: list[FemaleChangeEvent] = []
    for method in ("A0", "A1", "A2"):
        group = assignments.loc[
            assignments["method"].astype(str).eq(method)
            & assignments["consensus_method_id"].notna()
        ].copy()
        group["frame"] = pd.to_numeric(group["frame"], errors="raise").astype(np.int64)
        group["_id"] = group["consensus_method_id"].map(_clean_id)
        group.sort_values("frame", inplace=True)
        previous: Mapping[str, Any] | None = None
        for row in group.to_dict("records"):
            if previous is not None and row["_id"] != previous["_id"]:
                events.append(
                    FemaleChangeEvent(
                        source_method=method,
                        previous_frame=int(previous["frame"]),
                        next_frame=int(row["frame"]),
                        from_id=str(previous["_id"]),
                        to_id=str(row["_id"]),
                    )
                )
            previous = row
    events.sort(key=lambda event: (event.next_frame, event.source_method))
    return events


def _peek_frame_range(path: Path) -> tuple[int, int] | None:
    try:
        frames = pd.read_csv(path, usecols=["frame"])["frame"]
    except (OSError, ValueError):
        return None
    if frames.empty:
        return None
    return int(frames.min()), int(frames.max())


def add_descriptive_events(
    events: list[FemaleChangeEvent], a4_paths: Sequence[Path]
) -> list[FemaleChangeEvent]:
    result = list(events)
    result.append(
        FemaleChangeEvent(
            source_method="A0",
            previous_frame=34547,
            next_frame=34555,
            from_id="8/11 close pair",
            to_id="reported crossover",
            kind="critical_descriptive",
            fixed_center=34548,
            center=34548,
            note="reported 34547→34548 crossover",
        )
    )
    for path in a4_paths:
        bounds = _peek_frame_range(path)
        if bounds is None:
            continue
        start, end = bounds
        relative = path.parent.parent.name
        result.append(
            FemaleChangeEvent(
                source_method="A4",
                previous_frame=start,
                next_frame=end,
                from_id="",
                to_id="",
                kind="a4_window",
                fixed_center=(start + end) // 2,
                center=(start + end) // 2,
                note=f"evaluated window {relative} [{start}..{end}]",
            )
        )
    result.sort(
        key=lambda event: (
            event.fixed_center if event.fixed_center is not None else event.next_frame,
            event.source_method,
            event.kind,
        )
    )
    return result


def nearest_candidate(
    candidates: Sequence[CandidatePose],
    point: np.ndarray,
    *,
    anchor_policy: str,
    max_distance_px: float,
) -> tuple[CandidatePose | None, float]:
    point = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(point)):
        return None, float("inf")
    choices = []
    for candidate in candidates:
        anchor = candidate.anchor(anchor_policy)
        if np.all(np.isfinite(anchor)):
            choices.append((float(np.linalg.norm(anchor - point)), candidate.slot, candidate))
    if not choices:
        return None, float("inf")
    distance, _slot, candidate = min(choices, key=lambda item: (item[0], item[1]))
    if distance > float(max_distance_px):
        return None, distance
    return candidate, distance


def row_slot(row: Mapping[str, Any], view: str) -> int | None:
    column = f"slot_cam{VIEW_INDEX[view]}"
    value = row.get(column)
    if value is not None and not pd.isna(value):
        return int(float(value))
    parsed = parse_hypothesis_slots(row.get("hypothesis_id"))
    if parsed is None:
        return None
    return int(parsed[VIEW_INDEX[view]])


def _draw_candidate(
    image: np.ndarray,
    candidate: CandidatePose,
    color: tuple[int, int, int],
    *,
    show_mouth: bool = True,
    show_tail_link: bool = True,
    thickness: int = 1,
    mouth_radius: int = 3,
    tail_radius: int = 4,
    label: str = "",
) -> None:
    """Draw one two-node pose without obscuring the underlying fish.

    Mouth is a filled dot, tail is an open ring, and the optional connecting
    line is deliberately not arrowed.  Keeping those glyphs distinct makes it
    possible to inspect swaps while all raw candidates are visible.
    """
    mouth = np.asarray(candidate.mouth, dtype=float)
    tail = np.asarray(candidate.tail, dtype=float)
    mouth_finite = bool(np.all(np.isfinite(mouth)))
    tail_finite = bool(np.all(np.isfinite(tail)))
    if show_tail_link and mouth_finite and tail_finite:
        cv2.line(
            image,
            tuple(np.round(tail).astype(int)),
            tuple(np.round(mouth).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    if show_mouth and mouth_finite:
        mouth_xy = tuple(np.round(mouth).astype(int))
        cv2.circle(image, mouth_xy, mouth_radius, color, -1, cv2.LINE_AA)
        text_at = (mouth_xy[0] + 7, mouth_xy[1] + 16)
    elif tail_finite:
        text_at = tuple(np.round(tail).astype(int) + np.array([7, 16]))
    else:
        return
    if show_tail_link and tail_finite:
        cv2.circle(
            image,
            tuple(np.round(tail).astype(int)),
            tail_radius,
            color,
            thickness,
            cv2.LINE_AA,
        )
    if label:
        cv2.putText(
            image,
            label,
            text_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_raw_candidates(
    image: np.ndarray,
    candidates: Sequence[CandidatePose],
    *,
    show_mouth: bool = True,
    show_tail_link: bool = True,
    show_labels: bool = False,
) -> None:
    for candidate in candidates:
        _draw_candidate(
            image,
            candidate,
            RAW_COLOR,
            show_mouth=show_mouth,
            show_tail_link=show_tail_link,
            thickness=1,
            mouth_radius=2,
            tail_radius=3,
            label=(
                f"c{candidate.slot}/raw{candidate.source_instance}"
                if show_labels
                else ""
            ),
        )


def draw_method_anchor(
    image: np.ndarray,
    point: np.ndarray,
    color: tuple[int, int, int],
    *,
    anchor_policy: str,
    marker_size: int,
    label: str = "",
) -> None:
    """Draw centroid as a cross and mouth-based methods as an open circle."""
    point = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(point)):
        return
    xy = tuple(np.round(point).astype(int))
    marker_size = max(2, int(marker_size))
    if anchor_policy == "visible centroid":
        cv2.drawMarker(
            image,
            xy,
            color,
            cv2.MARKER_CROSS,
            2 * marker_size + 1,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(image, xy, marker_size, color, 1, cv2.LINE_AA)
        cv2.circle(image, xy, 1, color, -1, cv2.LINE_AA)
    if label:
        cv2.putText(
            image,
            label,
            (xy[0] + marker_size + 4, max(16, xy[1] - marker_size - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_female_ground_truth(
    image: np.ndarray,
    row: Mapping[str, Any],
    view: str,
    *,
    show_label: bool = False,
) -> None:
    prefix = "side1" if view == "side1_cam" else "side2"
    if view == "top_cam":
        return
    status = str(row.get(f"{prefix}_selection_status", ""))
    if status in {"", "not_selected", "nan"}:
        return
    point = np.array(
        [row.get(f"{prefix}_mouth_x", np.nan), row.get(f"{prefix}_mouth_y", np.nan)],
        dtype=float,
    )
    if not np.all(np.isfinite(point)):
        point = np.array(
            [row.get(f"{prefix}_click_x", np.nan), row.get(f"{prefix}_click_y", np.nan)],
            dtype=float,
        )
    if not np.all(np.isfinite(point)):
        return
    xy = tuple(np.round(point).astype(int))
    cv2.drawMarker(image, xy, FEMALE_COLOR, cv2.MARKER_STAR, 16, 1, cv2.LINE_AA)
    if show_label:
        cv2.putText(
            image,
            "F ground truth",
            (xy[0] + 10, max(18, xy[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            FEMALE_COLOR,
            1,
            cv2.LINE_AA,
        )


class ComparisonData:
    def __init__(
        self,
        *,
        experiment: str,
        method_sources: Mapping[str, MethodSource],
        videos: Mapping[str, Any],
        candidate_cache: Any,
        calibration: Any,
        female_assignments: pd.DataFrame,
        female_ledger: pd.DataFrame,
        events: Sequence[FemaleChangeEvent],
        max_match_px: float,
        initial_tables: Mapping[str, TrackTable] | None = None,
    ):
        self.experiment = experiment
        self.method_sources = dict(method_sources)
        self.videos = dict(videos)
        self.candidate_cache = candidate_cache
        self.calibration = calibration
        self.female_assignments = female_assignments.copy()
        self.female_ledger = female_ledger.copy()
        self.events = list(events)
        self.max_match_px = float(max_match_px)
        self._tables = dict(initial_tables or {})
        counts = {view: int(video.frame_count) for view, video in self.videos.items()}
        if set(counts) != set(VIEWS) or len(set(counts.values())) != 1:
            raise ValueError(f"video frame counts disagree: {counts}")
        self.frame_count = next(iter(counts.values()))
        if int(self.candidate_cache.frame_count) != self.frame_count:
            raise ValueError(
                "candidate cache and video frame counts disagree: "
                f"{self.candidate_cache.frame_count} versus {self.frame_count}"
            )
        self.female_assignments["frame"] = pd.to_numeric(
            self.female_assignments["frame"], errors="raise"
        ).astype(np.int64)
        self.female_ledger["frame"] = pd.to_numeric(
            self.female_ledger["frame"], errors="coerce"
        )
        self.female_ledger = self.female_ledger[
            self.female_ledger["frame"].notna()
        ].copy()
        self.female_ledger["frame"] = self.female_ledger["frame"].astype(np.int64)
        self._female_by_frame = {
            int(frame): group.iloc[-1].to_dict()
            for frame, group in self.female_ledger.groupby("frame", sort=False)
        }

    def method(self, label: str) -> TrackTable:
        if label not in self._tables:
            self._tables[label] = TrackTable.from_csvs(self.method_sources[label])
        return self._tables[label]

    def resolve_event(self, event: FemaleChangeEvent) -> FemaleChangeEvent:
        if event.center is not None:
            return event
        table = self.method(event.source_method)
        closest = table.closest_approach(
            event.from_id,
            event.to_id,
            event.previous_frame,
            event.next_frame,
        )
        if closest is None:
            event.center = (event.previous_frame + event.next_frame) // 2
            event.minimum_distance_mm = None
        else:
            event.center, event.minimum_distance_mm = closest
        return event

    def projected_rows(
        self, method: str, frame: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        rows = self.method(method).rows_at(frame)
        xyz = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
        valid = np.all(np.isfinite(xyz), axis=1)
        projected = np.full((len(rows), 3, 2), np.nan, dtype=float)
        if np.any(valid):
            projected[valid] = np.asarray(
                self.calibration.project_world(xyz[valid]), dtype=float
            )
        return rows, projected

    def projected_trails(
        self, method: str, start: int, end: int
    ) -> dict[str, dict[str, np.ndarray]]:
        rows = self.method(method).rows_between(start, end)
        xyz = rows[["X", "Y", "Z"]].to_numpy(dtype=float)
        valid = np.all(np.isfinite(xyz), axis=1)
        rows = rows.loc[valid]
        xyz = xyz[valid]
        result = {view: {} for view in VIEWS}
        if rows.empty:
            return result
        projected = np.asarray(self.calibration.project_world(xyz), dtype=float)
        for base_id in rows["_base_id"].unique():
            mask = rows["_base_id"].to_numpy() == base_id
            for view in VIEWS:
                result[view][str(base_id)] = projected[mask, VIEW_INDEX[view]]
        return result

    def female_row(self, frame: int) -> Mapping[str, Any] | None:
        return self._female_by_frame.get(int(frame))

    def female_bracket(self, method: str, frame: int) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        if method == "A4":
            return None, None
        group = self.female_assignments.loc[
            self.female_assignments["method"].astype(str).eq(method)
            & self.female_assignments["consensus_method_id"].notna()
        ].sort_values("frame")
        if group.empty:
            return None, None
        frames = group["frame"].to_numpy(dtype=np.int64)
        index = int(np.searchsorted(frames, int(frame), side="right"))
        previous = group.iloc[index - 1].to_dict() if index > 0 else None
        following = group.iloc[index].to_dict() if index < len(group) else None
        return previous, following

    def close(self) -> None:
        for video in self.videos.values():
            video.close()
        self.candidate_cache.close()


def _load_female_ledger(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    if "gt_status" in table:
        table = table.loc[table["gt_status"].astype(str).eq("confirmed")]
    if "sex_label" in table:
        table = table.loc[table["sex_label"].astype(str).eq("F")]
    return table.copy()


def _default_a4_paths(study_dir: Path) -> tuple[Path, ...]:
    roots = (
        study_dir / "critical_descriptive_postfreeze",
        study_dir / "validation_postfreeze",
        study_dir / "heldout_postfreeze",
    )
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(root.glob("**/A4/a4_direct.csv"))
    return tuple(sorted({path.resolve() for path in paths if path.is_file()}))


def resolve_method_sources(args: argparse.Namespace) -> dict[str, MethodSource]:
    study = Path(args.study_dir)
    local = load_local_paths(args.paths_config)
    if args.a0_csv is not None:
        a0 = Path(args.a0_csv)
    else:
        prelaunch = study / "full_length_postfreeze" / "prelaunch_auth.json"
        a0 = None
        if prelaunch.is_file():
            payload = json.loads(prelaunch.read_text(encoding="utf-8"))
            record = (
                payload.get("a0_reference", {})
                .get("artifacts", {})
                .get("identified_csv", {})
            )
            if record.get("path"):
                a0 = Path(record["path"])
        if a0 is None and local is not None and local.reconstruction_output_dir:
            a0 = (
                local.reconstruction_output_dir
                / args.experiment
                / args.round_tag
                / f"{args.round_tag}_3d_identified.csv"
            )
        if a0 is None:
            raise ValueError("could not resolve A0; pass --a0-csv")
    a1 = Path(args.a1_csv) if args.a1_csv else study / "full_length_postfreeze" / "A1" / "a1_direct.csv"
    a2 = Path(args.a2_csv) if args.a2_csv else study / "full_length_postfreeze" / "A2" / "a2_direct.csv"
    a4 = tuple(Path(path) for path in args.a4_csv) if args.a4_csv else _default_a4_paths(study)
    return {
        "A0": MethodSource("A0", (a0,), False, "visible centroid"),
        "A1": MethodSource("A1", (a1,), True, "mouth"),
        "A2": MethodSource("A2", (a2,), True, "mouth"),
        "A4": MethodSource("A4", a4, True, "mouth"),
    }


def _require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {label}:\n  " + "\n  ".join(missing))


def prepare_real_data(args: argparse.Namespace) -> ComparisonData:
    study = Path(args.study_dir)
    local = load_local_paths(args.paths_config)
    method_sources = resolve_method_sources(args)
    for method in METHODS:
        if not method_sources[method].paths:
            raise FileNotFoundError(f"{method} has no resolved CSVs")
        _require_files(method_sources[method].paths, f"{method} CSV")

    if args.video_dir is not None:
        video_dir = Path(args.video_dir)
    elif local is not None and local.medaka_data_dir is not None:
        video_dir = local.medaka_data_dir / args.experiment
    else:
        raise ValueError("could not resolve videos; pass --video-dir")
    video_paths = {
        view: video_dir / view / VIDEO_NAMES[view] for view in VIEWS
    }
    _require_files(video_paths.values(), "video")

    candidate_root = (
        Path(args.candidate_cache_dir)
        if args.candidate_cache_dir
        else study / "candidate_cache_strict_mouth_unfiltered_nodescores"
    )
    _require_files(
        [candidate_root / f"{view}.candidates.h5" for view in VIEWS],
        "candidate cache",
    )
    calibration_dir = (
        Path(args.calibration_model_dir)
        if args.calibration_model_dir
        else (local.calibration_model_dir if local is not None else None)
    )
    if calibration_dir is None:
        raise ValueError("could not resolve calibration; pass --calibration-model-dir")
    assignments_path = (
        Path(args.female_assignments)
        if args.female_assignments
        else study / "quality_review" / "full_length_independent_female_assignments.csv"
    )
    ledger_path = (
        Path(args.female_ledger)
        if args.female_ledger
        else REPO_ROOT
        / "annotations"
        / "fish_sex_gt"
        / args.experiment
        / "female_sex_gt.csv"
    )
    _require_files([assignments_path, ledger_path], "female evidence")

    videos: dict[str, OpenCvVideo] = {}
    cache: CandidateCache | None = None
    try:
        videos = {view: OpenCvVideo(path) for view, path in video_paths.items()}
        cache = CandidateCache(candidate_root)
        calibration = Calibration(calibration_dir, backend="auto")
        assignments = pd.read_csv(assignments_path, low_memory=False)
        ledger = _load_female_ledger(ledger_path)
        events = add_descriptive_events(
            build_female_change_events(assignments), method_sources["A4"].paths
        )
        return ComparisonData(
            experiment=args.experiment,
            method_sources=method_sources,
            videos=videos,
            candidate_cache=cache,
            calibration=calibration,
            female_assignments=assignments,
            female_ledger=ledger,
            events=events,
            max_match_px=args.max_match_px,
        )
    except Exception:
        for video in videos.values():
            video.close()
        if cache is not None:
            cache.close()
        raise


def _build_gui_classes():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise SystemExit("PyQt5 is required; run `uv sync --extra gui`.") from error

    class ImageView(QtWidgets.QGraphicsView):
        def __init__(self, title: str):
            super().__init__()
            self.title = title
            self._scene = QtWidgets.QGraphicsScene(self)
            self.setScene(self._scene)
            self._pixmap = self._scene.addPixmap(QtGui.QPixmap())
            self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(18, 18, 18)))
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
            self._has_image = False

        def set_bgr(self, image: np.ndarray) -> None:
            rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            qimage = QtGui.QImage(
                rgb.data, width, height, int(rgb.strides[0]), QtGui.QImage.Format_RGB888
            ).copy()
            self._pixmap.setPixmap(QtGui.QPixmap.fromImage(qimage))
            self._scene.setSceneRect(0, 0, width, height)
            if not self._has_image:
                self.fit_image()
                self._has_image = True

        def fit_image(self) -> None:
            if not self._pixmap.pixmap().isNull():
                self.resetTransform()
                self.fitInView(self._pixmap, QtCore.Qt.KeepAspectRatio)

        def wheelEvent(self, event) -> None:
            factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
            self.scale(factor, factor)

    class ComparisonWindow(QtWidgets.QMainWindow):
        def __init__(self, data: ComparisonData, args: argparse.Namespace):
            super().__init__()
            self.data = data
            self.args = args
            self.current_method = str(args.method)
            self.current_frame = int(args.frame) if args.frame is not None else 0
            self.current_event_index = -1
            self.context_frames = int(args.context_frames)
            self._rendering = False
            self.setWindowTitle(
                f"Round8 method comparison — {data.experiment}"
            )
            self.resize(1840, 980)
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._advance_play)
            self._build_ui()
            if args.frame is not None:
                self._set_window_around(int(args.frame), render=False)
                self._set_frame(int(args.frame))
            elif self.data.events:
                first = next(
                    (
                        index
                        for index, event in enumerate(self.data.events)
                        if event.kind == "female_assignment_change"
                    ),
                    0,
                )
                self.event_combo.setCurrentIndex(first)
                self._select_event(first)
            else:
                self._set_window_around(0, render=False)
                self._set_frame(0)

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            left = QtWidgets.QVBoxLayout()
            root.addLayout(left, 5)
            panels = QtWidgets.QHBoxLayout()
            left.addLayout(panels, 1)
            self.views: dict[str, ImageView] = {}
            for view in VIEWS:
                box = QtWidgets.QGroupBox(view)
                layout = QtWidgets.QVBoxLayout(box)
                image_view = ImageView(view)
                layout.addWidget(image_view)
                panels.addWidget(box, 1)
                self.views[view] = image_view

            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.valueChanged.connect(self._set_frame)
            left.addWidget(self.slider)
            transport = QtWidgets.QHBoxLayout()
            left.addLayout(transport)
            self.play_button = QtWidgets.QPushButton("Play [Space]")
            self.play_button.clicked.connect(self._toggle_play)
            transport.addWidget(self.play_button)
            for label, delta in (("-10", -10), ("-1", -1), ("+1", 1), ("+10", 10)):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(
                    lambda _checked=False, step=delta: self._step(step)
                )
                transport.addWidget(button)
            transport.addWidget(QtWidgets.QLabel("Playback step"))
            self.play_step = QtWidgets.QSpinBox()
            self.play_step.setRange(1, 30)
            self.play_step.setValue(1)
            transport.addWidget(self.play_step)
            self.frame_label = QtWidgets.QLabel("")
            transport.addWidget(self.frame_label, 1)
            fit_button = QtWidgets.QPushButton("Fit all views")
            fit_button.clicked.connect(self._fit_all)
            transport.addWidget(fit_button)

            side = QtWidgets.QVBoxLayout()
            root.addLayout(side, 2)
            heading = QtWidgets.QLabel("Tracking method")
            heading.setStyleSheet("font-weight:bold;font-size:15px")
            side.addWidget(heading)
            self.method_combo = QtWidgets.QComboBox()
            for method in METHODS:
                self.method_combo.addItem(
                    f"{method} — {METHOD_DESCRIPTIONS[method]}", method
                )
            self.method_combo.setCurrentIndex(METHODS.index(self.current_method))
            self.method_combo.currentIndexChanged.connect(self._change_method)
            side.addWidget(self.method_combo)
            self.method_status = QtWidgets.QLabel("")
            self.method_status.setWordWrap(True)
            side.addWidget(self.method_status)

            side.addWidget(self._line())
            side.addWidget(QtWidgets.QLabel("Review event"))
            self.event_combo = QtWidgets.QComboBox()
            self.event_combo.addItems([event.label for event in self.data.events])
            self.event_combo.currentIndexChanged.connect(self._select_event)
            side.addWidget(self.event_combo)
            nav = QtWidgets.QHBoxLayout()
            previous = QtWidgets.QPushButton("◀ Event")
            previous.clicked.connect(lambda: self._step_event(-1))
            following = QtWidgets.QPushButton("Event ▶")
            following.clicked.connect(lambda: self._step_event(1))
            nav.addWidget(previous)
            nav.addWidget(following)
            side.addLayout(nav)
            self.event_info = QtWidgets.QLabel("")
            self.event_info.setWordWrap(True)
            side.addWidget(self.event_info)

            jump = QtWidgets.QHBoxLayout()
            jump.addWidget(QtWidgets.QLabel("Exact frame"))
            self.frame_spin = QtWidgets.QSpinBox()
            self.frame_spin.setRange(0, self.data.frame_count - 1)
            jump.addWidget(self.frame_spin, 1)
            go = QtWidgets.QPushButton("Go")
            go.clicked.connect(self._jump_exact)
            jump.addWidget(go)
            side.addLayout(jump)

            side.addWidget(self._line())
            overlay_heading = QtWidgets.QLabel("Overlay detail (sparse defaults)")
            overlay_heading.setStyleSheet("font-weight:bold")
            side.addWidget(overlay_heading)
            self.raw_mouth_box = QtWidgets.QCheckBox("Raw prediction mouths ·")
            self.raw_mouth_box.setChecked(True)
            self.raw_mouth_box.stateChanged.connect(self._render)
            side.addWidget(self.raw_mouth_box)
            self.raw_pose_box = QtWidgets.QCheckBox("Raw prediction tails ○ + links")
            self.raw_pose_box.setChecked(True)
            self.raw_pose_box.stateChanged.connect(self._render)
            side.addWidget(self.raw_pose_box)
            self.selected_box = QtWidgets.QCheckBox(
                "Selected/nearest raw pose (thin color)"
            )
            self.selected_box.setChecked(True)
            self.selected_box.stateChanged.connect(self._render)
            side.addWidget(self.selected_box)
            self.projection_box = QtWidgets.QCheckBox(
                "Method anchor: centroid + / mouth ○"
            )
            self.projection_box.setChecked(True)
            self.projection_box.stateChanged.connect(self._render)
            side.addWidget(self.projection_box)
            self.candidate_labels_box = QtWidgets.QCheckBox(
                "Raw candidate indices (c#/raw#)"
            )
            self.candidate_labels_box.setChecked(False)
            self.candidate_labels_box.stateChanged.connect(self._render)
            side.addWidget(self.candidate_labels_box)
            self.identity_labels_box = QtWidgets.QCheckBox("Identity / GT text labels")
            self.identity_labels_box.setChecked(False)
            self.identity_labels_box.stateChanged.connect(self._render)
            side.addWidget(self.identity_labels_box)
            self.trail_box = QtWidgets.QCheckBox("Show recent identity trails")
            self.trail_box.setChecked(False)
            self.trail_box.stateChanged.connect(self._render)
            side.addWidget(self.trail_box)
            detail_row = QtWidgets.QHBoxLayout()
            detail_row.addWidget(QtWidgets.QLabel("Marker radius"))
            self.marker_size = QtWidgets.QSpinBox()
            self.marker_size.setRange(2, 12)
            self.marker_size.setValue(4)
            self.marker_size.valueChanged.connect(self._render)
            detail_row.addWidget(self.marker_size)
            detail_row.addWidget(QtWidgets.QLabel("Trail frames"))
            self.trail_frames = QtWidgets.QSpinBox()
            self.trail_frames.setRange(1, 300)
            self.trail_frames.setValue(30)
            self.trail_frames.valueChanged.connect(self._render)
            detail_row.addWidget(self.trail_frames)
            side.addLayout(detail_row)
            self.female_box = QtWidgets.QCheckBox(
                "Confirmed F marker on exact GT frames"
            )
            self.female_box.setChecked(True)
            self.female_box.stateChanged.connect(self._render)
            side.addWidget(self.female_box)
            overlay_legend = QtWidgets.QLabel(
                "Raw: mouth ·  tail ○  |  A0 centroid +  |  A1/A2/A4 mouth ○  |  F ☆"
            )
            overlay_legend.setWordWrap(True)
            overlay_legend.setStyleSheet("color:#888")
            side.addWidget(overlay_legend)

            side.addWidget(self._line())
            self.female_status = QtWidgets.QLabel("")
            self.female_status.setWordWrap(True)
            side.addWidget(self.female_status)
            self.row_table = QtWidgets.QTableWidget(0, 5)
            self.row_table.setHorizontalHeaderLabels(
                ["ID", "segment", "status", "slots", "residual"]
            )
            self.row_table.horizontalHeader().setStretchLastSection(True)
            self.row_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.row_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            side.addWidget(self.row_table, 1)

            shortcuts = {
                "Space": self._toggle_play,
                "Left": lambda: self._step(-1),
                "Right": lambda: self._step(1),
                "Shift+Left": lambda: self._step(-10),
                "Shift+Right": lambda: self._step(10),
                "0": lambda: self._select_method("A0"),
                "1": lambda: self._select_method("A1"),
                "2": lambda: self._select_method("A2"),
                "4": lambda: self._select_method("A4"),
                "F": self._fit_all,
            }
            for key, callback in shortcuts.items():
                QtWidgets.QShortcut(
                    QtGui.QKeySequence(key), self, activated=callback
                )

        @staticmethod
        def _line():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            return line

        def _select_method(self, method: str) -> None:
            self.method_combo.setCurrentIndex(METHODS.index(method))

        def _change_method(self, index: int) -> None:
            self.current_method = str(self.method_combo.itemData(index))
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.data.method(self.current_method)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self._render()

        def _step_event(self, delta: int) -> None:
            if not self.data.events:
                return
            target = min(
                max(self.event_combo.currentIndex() + int(delta), 0),
                len(self.data.events) - 1,
            )
            self.event_combo.setCurrentIndex(target)

        def _select_event(self, index: int) -> None:
            if index < 0 or index >= len(self.data.events):
                return
            event = self.data.events[index]
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                event = self.data.resolve_event(event)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.current_event_index = index
            self.event_combo.setItemText(index, event.label)
            if event.source_method in METHODS:
                self._select_method(event.source_method)
            center = int(event.center if event.center is not None else event.next_frame)
            self._set_window_around(center, render=False)
            distance = (
                "not available"
                if event.minimum_distance_mm is None
                else f"{event.minimum_distance_mm:.3f} mm"
            )
            if event.kind == "female_assignment_change":
                self.event_info.setText(
                    f"Female evidence changed from {event.from_id} at "
                    f"f{event.previous_frame} to {event.to_id} at f{event.next_frame}. "
                    f"Window center: closest approach f{center}; distance {distance}."
                )
            else:
                self.event_info.setText(event.note)
            self._set_frame(center)

        def _set_window_around(self, center: int, *, render: bool) -> None:
            low = max(0, int(center) - self.context_frames)
            high = min(self.data.frame_count - 1, int(center) + self.context_frames)
            self.slider.blockSignals(True)
            self.slider.setRange(low, high)
            self.slider.setValue(min(max(int(center), low), high))
            self.slider.blockSignals(False)
            if render:
                self._set_frame(int(center))

        def _jump_exact(self) -> None:
            frame = int(self.frame_spin.value())
            if frame < self.slider.minimum() or frame > self.slider.maximum():
                self._set_window_around(frame, render=False)
            self._set_frame(frame)

        def _set_frame(self, frame: int) -> None:
            self.current_frame = min(max(int(frame), 0), self.data.frame_count - 1)
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(self.current_frame)
            self.frame_spin.blockSignals(False)
            self._render()

        def _step(self, delta: int) -> None:
            target = min(
                max(self.current_frame + int(delta), self.slider.minimum()),
                self.slider.maximum(),
            )
            self._set_frame(target)

        def _toggle_play(self) -> None:
            if self._timer.isActive():
                self._timer.stop()
                self.play_button.setText("Play [Space]")
            else:
                self._timer.start(40)
                self.play_button.setText("Pause [Space]")

        def _advance_play(self) -> None:
            target = self.current_frame + int(self.play_step.value())
            if target > self.slider.maximum():
                self._timer.stop()
                self.play_button.setText("Play [Space]")
                return
            self._set_frame(target)

        def _fit_all(self) -> None:
            for view in self.views.values():
                view.fit_image()

        def _render(self, *_args) -> None:
            if self._rendering:
                return
            self._rendering = True
            try:
                method = self.current_method
                frame = self.current_frame
                table = self.data.method(method)
                rows, projected = self.data.projected_rows(method, frame)
                trails = (
                    self.data.projected_trails(
                        method,
                        max(0, frame - int(self.trail_frames.value())),
                        frame,
                    )
                    if self.trail_box.isChecked()
                    else {view: {} for view in VIEWS}
                )
                female = self.data.female_row(frame)
                for view in VIEWS:
                    image = self.data.videos[view].get(frame)
                    candidates = self.data.candidate_cache.get(view, frame)
                    for base_id, points in trails[view].items():
                        finite = np.all(np.isfinite(points), axis=1)
                        points = np.round(points[finite]).astype(np.int32)
                        if len(points) >= 2:
                            cv2.polylines(
                                image,
                                [points.reshape(-1, 1, 2)],
                                False,
                                table.color(base_id),
                                1,
                                cv2.LINE_AA,
                            )
                    if self.raw_mouth_box.isChecked() or self.raw_pose_box.isChecked():
                        draw_raw_candidates(
                            image,
                            candidates,
                            show_mouth=self.raw_mouth_box.isChecked(),
                            show_tail_link=self.raw_pose_box.isChecked(),
                            show_labels=self.candidate_labels_box.isChecked(),
                        )
                    for row_index, (_index, row) in enumerate(rows.iterrows()):
                        point = projected[row_index, VIEW_INDEX[view]]
                        if not np.all(np.isfinite(point)):
                            continue
                        base_id = str(row["_base_id"])
                        segment = str(row["_segment"])
                        color = table.color(base_id)
                        selected: CandidatePose | None = None
                        match_distance = float("nan")
                        exact_slot = row_slot(row, view)
                        if exact_slot is not None and exact_slot >= 0:
                            selected = next(
                                (
                                    candidate
                                    for candidate in candidates
                                    if candidate.slot == exact_slot
                                ),
                                None,
                            )
                        elif exact_slot is None:
                            selected, match_distance = nearest_candidate(
                                candidates,
                                point,
                                anchor_policy=METHOD_ANCHORS[method],
                                max_distance_px=self.data.max_match_px,
                            )
                        if self.selected_box.isChecked() and selected is not None:
                            selected_label = ""
                            if self.candidate_labels_box.isChecked():
                                selected_label = (
                                    f"c{selected.slot}/raw{selected.source_instance}"
                                )
                                if np.isfinite(match_distance):
                                    selected_label += f" Δ{match_distance:.1f}px"
                            _draw_candidate(
                                image,
                                selected,
                                color,
                                show_mouth=True,
                                show_tail_link=True,
                                thickness=1,
                                mouth_radius=int(self.marker_size.value()),
                                tail_radius=int(self.marker_size.value()) + 1,
                                label=selected_label,
                            )
                            anchor = selected.anchor(METHOD_ANCHORS[method])
                            if (
                                self.projection_box.isChecked()
                                and np.all(np.isfinite(anchor))
                                and float(np.linalg.norm(anchor - point)) > 2.0
                            ):
                                cv2.line(
                                    image,
                                    tuple(np.round(anchor).astype(int)),
                                    tuple(np.round(point).astype(int)),
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )
                        label = base_id
                        if table.include_segment:
                            label += f" · seg {segment}"
                        if self.projection_box.isChecked():
                            draw_method_anchor(
                                image,
                                point,
                                color,
                                anchor_policy=METHOD_ANCHORS[method],
                                marker_size=int(self.marker_size.value()),
                                label=(
                                    label if self.identity_labels_box.isChecked() else ""
                                ),
                            )
                    if female is not None and self.female_box.isChecked():
                        draw_female_ground_truth(
                            image,
                            female,
                            view,
                            show_label=self.identity_labels_box.isChecked(),
                        )
                    self.views[view].set_bgr(image)
                self._update_status(rows)
                fps = float(getattr(self.data.videos["top_cam"], "fps", 166.0))
                self.frame_label.setText(
                    f"frame {frame} / {self.data.frame_count - 1}   "
                    f"t={frame / fps:.3f}s   method={method}"
                )
            except Exception as error:
                self.method_status.setText(f"Render failed: {error}")
                raise
            finally:
                self._rendering = False

        def _update_status(self, rows: pd.DataFrame) -> None:
            method = self.current_method
            finite_count = int(
                np.all(
                    np.isfinite(rows[["X", "Y", "Z"]].to_numpy(dtype=float)), axis=1
                ).sum()
            )
            if rows.empty:
                message = "No method rows at this frame."
            elif finite_count == 0:
                message = "All method rows are unresolved/unassigned at this frame."
            else:
                message = f"{finite_count}/{len(rows)} finite method tracks."
            if method == "A4":
                message += (
                    " A4 is a bounded diagnostic; blanks outside its evaluated "
                    "windows and explicit abstentions are expected."
                )
            self.method_status.setText(message)
            previous, following = self.data.female_bracket(
                method, self.current_frame
            )
            if method == "A4":
                female_text = (
                    "No full-length female assignment sequence exists for A4. "
                    "Use exact magenta F frames and the bounded A4 windows."
                )
            else:
                def describe(row):
                    if row is None:
                        return "none"
                    return (
                        f"f{int(row['frame'])}: "
                        f"{_clean_id(row['consensus_method_id'])} "
                        f"({row.get('mapping_status', '')})"
                    )

                female_text = (
                    f"Female-assignment bracket for {method}: "
                    f"previous {describe(previous)}; next {describe(following)}. "
                    "This is sparse human evidence, not interpolation."
                )
            self.female_status.setText(female_text)
            self.row_table.setRowCount(len(rows))
            for output_row, (_index, row) in enumerate(rows.iterrows()):
                slots = "/".join(
                    "-" if row_slot(row, view) is None else str(row_slot(row, view))
                    for view in VIEWS
                )
                residual = row.get("ray_residual_mm2", row.get("bind_score", ""))
                residual_text = (
                    ""
                    if residual is None or pd.isna(residual)
                    else f"{float(residual):.4g}"
                )
                values = (
                    str(row["_base_id"]),
                    str(row["_segment"]) if self.data.method(method).include_segment else "",
                    str(row.get("_status", "")),
                    slots,
                    residual_text,
                )
                for column, value in enumerate(values):
                    self.row_table.setItem(
                        output_row, column, QtWidgets.QTableWidgetItem(value)
                    )
            self.row_table.resizeColumnsToContents()

        def closeEvent(self, event) -> None:
            self._timer.stop()
            self.data.close()
            event.accept()

    return QtCore, QtGui, QtWidgets, ComparisonWindow


class _SyntheticVideo:
    def __init__(self, view: str):
        self.view = view
        self.frame_count = 24
        self.width = 480
        self.height = 360
        self.fps = 24.0

    def get(self, frame: int) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        for index, color in enumerate(IDENTITY_COLORS[:3]):
            x = 80 + index * 140 + (frame if index == 0 else -frame)
            y = 110 + index * 70
            cv2.ellipse(image, (x, y), (36, 13), 0, 0, 360, color, -1)
        return image

    def close(self) -> None:
        return None


class _SyntheticCache:
    frame_count = 24

    def get(self, view: str, frame: int) -> list[CandidatePose]:
        del view
        return [
            CandidatePose(
                slot=index,
                source_instance=index + 2,
                mouth=np.array([80 + index * 140 + (frame if index == 0 else -frame), 110 + index * 70]),
                tail=np.array([55 + index * 140 + (frame if index == 0 else -frame), 110 + index * 70]),
                mouth_score=0.99,
                tail_score=0.98,
            )
            for index in range(3)
        ]

    def close(self) -> None:
        return None


class _SyntheticCalibration:
    def project_world(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        xy = xyz[:, :2]
        return np.stack((xy, xy, xy), axis=1)


def _synthetic_data() -> ComparisonData:
    rows_by_method: dict[str, TrackTable] = {}
    sources = {}
    for method in METHODS:
        rows = []
        for frame in range(24):
            for identity in range(1, 4):
                rows.append(
                    {
                        "frame": frame,
                        "X": 80 + (identity - 1) * 140 + (frame if identity == 1 else -frame),
                        "Y": 110 + (identity - 1) * 70,
                        "Z": 10.0,
                        "_base_id": (
                            str(7 + identity)
                            if method == "A0"
                            else (f"PATH_{identity}" if method == "A4" else f"ID_{identity}")
                        ),
                        "_segment": 0,
                        "_status": "synthetic",
                        "hypothesis_id": f"f{frame:06d}_m7_s0_1_2",
                        "ray_residual_mm2": 0.1,
                    }
                )
        rows_by_method[method] = TrackTable(
            method, pd.DataFrame(rows), include_segment=method != "A0"
        )
        sources[method] = MethodSource(
            method, (Path(f"synthetic/{method}.csv"),), method != "A0", METHOD_ANCHORS[method]
        )
    assignments = pd.DataFrame(
        [
            {"method": method, "frame": 2, "consensus_method_id": ("8" if method == "A0" else "ID_1#identity_segment=0"), "mapping_status": "agree"}
            for method in ("A0", "A1", "A2")
        ]
        + [
            {"method": method, "frame": 20, "consensus_method_id": ("9" if method == "A0" else "ID_2#identity_segment=0"), "mapping_status": "agree"}
            for method in ("A0", "A1", "A2")
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "frame": 2,
                "side1_selection_status": "sleap_linked",
                "side1_mouth_x": 82.0,
                "side1_mouth_y": 110.0,
                "side2_selection_status": "sleap_linked",
                "side2_mouth_x": 82.0,
                "side2_mouth_y": 110.0,
            }
        ]
    )
    events = add_descriptive_events(build_female_change_events(assignments), ())
    return ComparisonData(
        experiment="synthetic",
        method_sources=sources,
        videos={view: _SyntheticVideo(view) for view in VIEWS},
        candidate_cache=_SyntheticCache(),
        calibration=_SyntheticCalibration(),
        female_assignments=assignments,
        female_ledger=ledger,
        events=events,
        max_match_px=80.0,
        initial_tables=rows_by_method,
    )


def run_smoke_test(args: argparse.Namespace) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtCore, _QtGui, QtWidgets, ComparisonWindow = _build_gui_classes()
    data = _synthetic_data()
    args.method = "A0"
    args.frame = 2
    args.context_frames = 8
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = ComparisonWindow(data, args)
    window.show()
    result = {"code": 0}

    def exercise() -> None:
        try:
            window._select_method("A1")
            window._set_frame(5)
            window._select_method("A4")
            window._set_frame(7)
            if any(view._pixmap.pixmap().isNull() for view in window.views.values()):
                raise RuntimeError("one or more view panels did not render")
            window.close()
            app.quit()
        except Exception as error:
            print(f"Round8 method GUI smoke test failed: {error}", file=sys.stderr)
            result["code"] = 2
            window.close()
            app.exit(2)

    QtCore.QTimer.singleShot(80, exercise)
    app.exec_()
    return int(result["code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default=EXPERIMENT_DEFAULT)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--study-dir", type=Path, default=STUDY_DIR_DEFAULT)
    parser.add_argument("--round-tag", default=ROUND_TAG_DEFAULT)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--a0-csv", type=Path)
    parser.add_argument("--a1-csv", type=Path)
    parser.add_argument("--a2-csv", type=Path)
    parser.add_argument("--a4-csv", type=Path, action="append", default=[])
    parser.add_argument("--candidate-cache-dir", type=Path)
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--female-assignments", type=Path)
    parser.add_argument("--female-ledger", type=Path)
    parser.add_argument("--method", choices=METHODS, default="A0")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--context-frames", type=int, default=500)
    parser.add_argument("--max-match-px", type=float, default=80.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context_frames < 1:
        parser.error("--context-frames must be positive")
    if args.max_match_px < 0:
        parser.error("--max-match-px must be non-negative")
    if args.smoke_test:
        return run_smoke_test(args)
    data = prepare_real_data(args)
    if args.frame is not None and not 0 <= args.frame < data.frame_count:
        data.close()
        parser.error(f"--frame must be in 0..{data.frame_count - 1}")
    if args.check_only:
        try:
            resolved_events = []
            for event in data.events:
                if event.kind != "female_assignment_change":
                    continue
                data.resolve_event(event)
                resolved_events.append(event)
            check_frame = (
                int(args.frame)
                if args.frame is not None
                else min(34548, data.frame_count - 1)
            )
            decoded_shapes = {
                view: list(data.videos[view].get(check_frame).shape)
                for view in VIEWS
            }
            candidate_counts = {
                view: len(data.candidate_cache.get(view, check_frame))
                for view in VIEWS
            }
            projected_counts = {}
            for method in METHODS:
                rows, projected = data.projected_rows(method, check_frame)
                projected_counts[method] = {
                    "rows": len(rows),
                    "finite_xyz": int(
                        np.all(
                            np.isfinite(
                                rows[["X", "Y", "Z"]].to_numpy(dtype=float)
                            ),
                            axis=1,
                        ).sum()
                    ),
                    "finite_three_view_projections": int(
                        np.all(np.isfinite(projected), axis=(1, 2)).sum()
                    ),
                }
            summary = {
                "experiment": data.experiment,
                "frame_count": data.frame_count,
                "methods": {
                    method: {
                        "paths": [str(path) for path in data.method_sources[method].paths],
                        "rows": len(data.method(method).table),
                    }
                    for method in METHODS
                },
                "female_change_events": sum(
                    event.kind == "female_assignment_change" for event in data.events
                ),
                "female_change_events_centered_on_closest_approach": sum(
                    event.minimum_distance_mm is not None for event in resolved_events
                ),
                "all_review_events": len(data.events),
                "render_check": {
                    "frame": check_frame,
                    "decoded_video_shapes": decoded_shapes,
                    "raw_candidate_counts": candidate_counts,
                    "method_projection_counts": projected_counts,
                },
            }
            print(json.dumps(summary, indent=2))
            return 0
        finally:
            data.close()
    _QtCore, _QtGui, QtWidgets, ComparisonWindow = _build_gui_classes()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = ComparisonWindow(data, args)
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
