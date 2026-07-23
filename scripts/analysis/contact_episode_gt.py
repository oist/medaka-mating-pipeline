"""Core contracts for contact-episode detection and annotation.

This module deliberately has no Qt, OpenCV, SLEAP, or calibration dependency.
It owns compact three-fish tracking data, direct-observation pairwise IFDs, the
hysteretic contact detector, structured raw-click evidence, mapping suggestions,
and the authoritative CSV/JSONL annotation ledger used by the GUI.

Human evidence is always a click in an original video frame.  SLEAP instance
and geometry-ID links are optional, frame-local provenance and are never used to
invent or propagate a click.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.2"
PREVIOUS_SCHEMA_VERSION = "1.1"
LEGACY_SCHEMA_VERSION = "1.0"
CANDIDATE_SCHEMA_VERSION = "1.0"
CACHE_VERSION = 1
SOFTWARE_NAME = "scripts.analysis.contact_episode_annotator"

VIEWS = ("top_cam", "side1_cam", "side2_cam")
PHASES = ("anchor_a", "anchor_b")
IDENTITY_TARGETS = ("female", "anchor_male")
PAIR_ORDER = (("ID_1", "ID_2"), ("ID_1", "ID_3"), ("ID_2", "ID_3"))
PAIR_KEYS = ("ID_1-ID_2", "ID_1-ID_3", "ID_2-ID_3")
PAIR_COLORS = {
    "ID_1-ID_2": "#f28e2b",  # orange
    "ID_1-ID_3": "#9467bd",  # purple
    "ID_2-ID_3": "#17becf",  # cyan
}

CONTACT_VALIDITIES = (
    "confirmed_contact_crossing",
    "near_miss",
    "false_contact_candidate",
    "uncertain",
)
TRACKER_ASSESSMENTS = (
    "correct_continuity",
    "permanent_id_swap",
    "temporary_swap_flicker",
    "motion_noise_no_identity_swap",
    "observations_unusable",
    "uncertain",
)
EVIDENCE_QUALITIES = ("unreviewed", "usable", "human_only", "unusable")
ISSUE_FLAGS = (
    "close_contact",
    "sleap_false_negative",
    "shared_tail_or_node",
    "duplicate_or_merged_instance",
    "wrong_3d_reconstruction",
    "occluded_view",
    "other_sleap_error",
)
MAPPING_SUGGESTIONS = (
    "same_ids_no_swap",
    "female_involved_swap",
    "male_to_male_swap",
    "incomplete_association",
)

CANDIDATE_COLUMNS = [
    "candidate_schema_version",
    "episode_id",
    "experiment",
    "tracking_method",
    "contact_start_frame",
    "contact_end_frame",
    "review_start_frame",
    "review_end_frame",
    "minimum_ifd_mm",
    "minimum_ifd_frame",
    "minimum_pair",
    "duration_frames",
    "missing_direct_frames",
    "missing_direct_fraction",
    "merged_raw_interval_count",
    "detector_parameters_json",
    "source_provenance_json",
    "manual",
]

ANNOTATION_COLUMNS = [
    "schema_version",
    "annotation_id",
    "experiment",
    "episode_id",
    "annotator",
    "tracking_method",
    "contact_start_frame",
    "contact_end_frame",
    "review_start_frame",
    "review_end_frame",
    "detector_parameters_json",
    "source_provenance_json",
    "candidate_json",
    "anchor_a_frame",
    "anchor_b_frame",
    "clicks_json",
    "false_positive_detections_json",
    "false_negative_detections_json",
    "contact_validity",
    "tracker_assessment",
    "assessed_tracking_label",
    "automatic_mapping_suggestion",
    "evidence_quality",
    "issue_flags",
    "notes",
    "created_at_utc",
    "updated_at_utc",
]

LEGACY_ANNOTATION_COLUMNS = [
    "pre_frame" if column == "anchor_a_frame" else
    "post_frame" if column == "anchor_b_frame" else column
    for column in ANNOTATION_COLUMNS
    if column not in {"false_positive_detections_json", "false_negative_detections_json"}
]

V1_1_ANNOTATION_COLUMNS = [
    column for column in ANNOTATION_COLUMNS
    if column != "false_negative_detections_json"
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and math.isnan(float(value))
    ) or (isinstance(value, str) and not value.strip())


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return normalize_path(value)
    if _missing(value):
        return ""
    return value


def _json_value(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return normalize_path(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_json(value: Any) -> str:
    return json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def normalize_path(path: str | Path | None) -> str:
    if path is None or not str(path).strip():
        return ""
    return os.path.normpath(str(Path(path).expanduser().resolve(strict=False)))


def file_provenance(path: str | Path, *, sha256: bool = False) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    result: dict[str, Any] = {
        "path": normalize_path(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def stable_pair_order(fish_ids: Sequence[Any] | None = None) -> tuple[tuple[str, str], ...]:
    """Return the canonical three pair order used by data, plots, and legends."""
    if fish_ids is None:
        return PAIR_ORDER
    ids = tuple(str(value) for value in fish_ids)
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("exactly three distinct fish IDs are required")
    return ((ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2]))


def pair_key(first: Any, second: Any) -> str:
    return f"{first}-{second}"


def pair_colors_for(fish_ids: Sequence[Any]) -> dict[str, str]:
    pairs = stable_pair_order(fish_ids)
    return {
        pair_key(*pair): color
        for pair, color in zip(pairs, PAIR_COLORS.values(), strict=True)
    }


def _strict_bool(values: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    normalized = values.fillna(False).astype(str).str.strip().str.casefold()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", ""}
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(f"{label} contains non-boolean values: {unexpected[:5]}")
    return normalized.isin({"true", "1", "yes", "y"}).to_numpy(dtype=bool)


def _natural_id_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


@dataclass
class TrackingIFD:
    """Dense, compact three-fish coordinates and pairwise distances."""

    method: str
    source_path: Path
    source_provenance: dict[str, Any]
    fish_ids: tuple[str, str, str]
    frames: np.ndarray
    xyz: np.ndarray
    interpolated: np.ndarray
    direct_distances: np.ndarray
    interpolated_distances: np.ndarray
    minimum_direct: np.ndarray
    minimum_pair_index: np.ndarray
    missing_fish_count: np.ndarray

    def __post_init__(self) -> None:
        n_frames = len(self.frames)
        if self.xyz.shape != (n_frames, 3, 3):
            raise ValueError("xyz must have shape (frames, 3 fish, 3 coordinates)")
        if self.direct_distances.shape != (n_frames, 3):
            raise ValueError("direct_distances must have shape (frames, 3 pairs)")

    @property
    def frame_min(self) -> int:
        return int(self.frames[0]) if len(self.frames) else 0

    @property
    def frame_max(self) -> int:
        return int(self.frames[-1]) if len(self.frames) else -1

    @property
    def pair_keys(self) -> tuple[str, str, str]:
        return tuple(pair_key(*pair) for pair in stable_pair_order(self.fish_ids))

    def position(self, frame: int) -> int:
        frame = int(frame)
        position = frame - self.frame_min
        if position < 0 or position >= len(self.frames) or int(self.frames[position]) != frame:
            raise IndexError(f"frame {frame} is outside {self.frame_min}..{self.frame_max}")
        return position

    def slice_positions(self, start: int, end: int) -> slice:
        low = max(self.frame_min, int(start))
        high = min(self.frame_max, int(end))
        if high < low:
            return slice(0, 0)
        return slice(low - self.frame_min, high - self.frame_min + 1)

    def xyz_at(self, frame: int) -> dict[str, np.ndarray]:
        position = self.position(frame)
        return {
            fish_id: self.xyz[position, index].astype(float, copy=True)
            for index, fish_id in enumerate(self.fish_ids)
        }


def pairwise_ifd(
    xyz: np.ndarray,
    *,
    direct_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate three pair distances without filling or joining missing values.

    Returns ``direct``, ``interpolated``, ``minimum_direct``, and the index of
    the pair producing that minimum.  A pair is direct only when both fish have
    finite XYZ and both are marked direct.  Finite pairs involving an
    interpolated row are retained separately for secondary plotting only.
    """
    coordinates = np.asarray(xyz, dtype=float)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
        squeeze = True
    else:
        squeeze = False
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
        raise ValueError("xyz must have shape (frames, 3 fish, 3 coordinates)")
    finite_fish = np.all(np.isfinite(coordinates), axis=2)
    if direct_mask is None:
        direct_fish = finite_fish
    else:
        direct_fish = np.asarray(direct_mask, dtype=bool)
        if direct_fish.shape != finite_fish.shape:
            raise ValueError("direct_mask must have shape (frames, 3 fish)")
        direct_fish &= finite_fish

    all_distances = np.full((len(coordinates), 3), np.nan, dtype=float)
    direct = np.full_like(all_distances, np.nan)
    interpolated = np.full_like(all_distances, np.nan)
    for pair_index, (first, second) in enumerate(((0, 1), (0, 2), (1, 2))):
        valid = finite_fish[:, first] & finite_fish[:, second]
        values = np.linalg.norm(
            coordinates[:, first] - coordinates[:, second], axis=1
        )
        all_distances[valid, pair_index] = values[valid]
        direct_pair = valid & direct_fish[:, first] & direct_fish[:, second]
        direct[direct_pair, pair_index] = values[direct_pair]
        interpolated[valid & ~direct_pair, pair_index] = values[valid & ~direct_pair]

    finite_direct = np.isfinite(direct)
    minimum = np.full(len(coordinates), np.nan, dtype=float)
    minimum_pair = np.full(len(coordinates), -1, dtype=np.int8)
    any_direct = finite_direct.any(axis=1)
    if np.any(any_direct):
        safe = np.where(finite_direct[any_direct], direct[any_direct], np.inf)
        indices = np.argmin(safe, axis=1)
        minimum[any_direct] = safe[np.arange(len(safe)), indices]
        minimum_pair[any_direct] = indices.astype(np.int8)
    if squeeze:
        return direct[0], interpolated[0], minimum[0], minimum_pair[0]
    return direct, interpolated, minimum, minimum_pair


# Descriptive public alias used by downstream analysis code.
compute_pairwise_ifd = pairwise_ifd


def minimum_ifd(pair_distances: np.ndarray) -> np.ndarray:
    """Return the per-frame finite pair minimum while preserving all-missing NaN."""
    values = np.asarray(pair_distances, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 2:
        raise ValueError("pair_distances must have shape (frames, pairs)")
    finite = np.isfinite(values)
    result = np.full(len(values), np.nan, dtype=float)
    valid = finite.any(axis=1)
    if np.any(valid):
        result[valid] = np.min(np.where(finite[valid], values[valid], np.inf), axis=1)
    return result[0] if squeeze else result


def load_tracking_csv(
    path: str | Path,
    *,
    method: str = "tracking",
    id_column: str | None = None,
    max_frame: int | None = None,
) -> TrackingIFD:
    """Load one controlled long-form tracking CSV into compact dense arrays."""
    source = Path(path)
    header = list(pd.read_csv(source, nrows=0).columns)
    required = {"frame", "X", "Y", "Z"}
    missing = required - set(header)
    if missing:
        raise ValueError(f"tracking CSV is missing columns: {sorted(missing)}")
    if id_column is None:
        id_column = next(
            (
                name
                for name in ("final_fish_id", "fish_id", "geometry_id", "geometry_slot")
                if name in header
            ),
            None,
        )
    if id_column is None:
        raise ValueError("tracking CSV needs a fish/geometry ID column")
    interpolation_column = next(
        (name for name in ("is_interpolated", "interpolated", "is_filled") if name in header),
        None,
    )
    usecols = ["frame", id_column, "X", "Y", "Z"]
    if interpolation_column:
        usecols.append(interpolation_column)
    table = pd.read_csv(source, usecols=usecols, low_memory=False)
    frames_numeric = pd.to_numeric(table["frame"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(frames_numeric).all() or not np.equal(frames_numeric, np.floor(frames_numeric)).all():
        raise ValueError("tracking frame values must be finite integers")
    frames = frames_numeric.astype(np.int64)
    if np.any(frames < 0):
        raise ValueError("tracking CSV contains negative source frames")
    table["frame"] = frames
    table = table.loc[table[id_column].notna()].copy()
    table[id_column] = table[id_column].astype(str)
    fish_ids = tuple(sorted(table[id_column].unique(), key=_natural_id_key))
    if len(fish_ids) != 3:
        raise ValueError(
            f"tracking CSV must contain exactly three fish IDs, found {list(fish_ids)}"
        )
    if table.duplicated(["frame", id_column]).any():
        duplicate = table.loc[table.duplicated(["frame", id_column], keep=False), ["frame", id_column]].iloc[0]
        raise ValueError(
            f"tracking CSV has duplicate frame/ID row: {int(duplicate['frame'])}, {duplicate[id_column]}"
        )
    observed_max = int(table["frame"].max()) if len(table) else -1
    if max_frame is not None:
        if observed_max > int(max_frame):
            raise ValueError(
                f"tracking CSV extends to frame {observed_max}, beyond source frame {int(max_frame)}"
            )
        frame_max = int(max_frame)
    else:
        frame_max = observed_max
    frame_axis = np.arange(frame_max + 1, dtype=np.int64)
    xyz = np.full((len(frame_axis), 3, 3), np.nan, dtype=np.float32)
    interpolated = np.zeros((len(frame_axis), 3), dtype=bool)
    id_to_index = {fish_id: index for index, fish_id in enumerate(fish_ids)}
    id_indices = table[id_column].map(id_to_index).to_numpy(dtype=np.int64)
    row_frames = table["frame"].to_numpy(dtype=np.int64)
    coordinates = table[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    xyz[row_frames, id_indices] = coordinates
    if interpolation_column:
        interpolated[row_frames, id_indices] = _strict_bool(
            table[interpolation_column], interpolation_column
        )
    direct_fish = np.all(np.isfinite(xyz), axis=2) & ~interpolated
    direct, secondary, minimum, minimum_pair = pairwise_ifd(
        xyz, direct_mask=direct_fish
    )
    return TrackingIFD(
        method=str(method),
        source_path=source,
        source_provenance=file_provenance(source),
        fish_ids=fish_ids,  # type: ignore[arg-type]
        frames=frame_axis,
        xyz=xyz,
        interpolated=interpolated,
        direct_distances=direct.astype(np.float32),
        interpolated_distances=secondary.astype(np.float32),
        minimum_direct=minimum.astype(np.float32),
        minimum_pair_index=minimum_pair,
        missing_fish_count=(3 - direct_fish.sum(axis=1)).astype(np.int8),
    )


def _cache_metadata(store: TrackingIFD) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "method": store.method,
        "source_provenance": store.source_provenance,
        "fish_ids": list(store.fish_ids),
    }


def save_ifd_cache(store: TrackingIFD, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(stable_json(_cache_metadata(store))),
            frames=store.frames,
            xyz=store.xyz,
            interpolated=store.interpolated,
            direct_distances=store.direct_distances,
            interpolated_distances=store.interpolated_distances,
            minimum_direct=store.minimum_direct,
            minimum_pair_index=store.minimum_pair_index,
            missing_fish_count=store.missing_fish_count,
        )
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_ifd_cache(
    path: str | Path, source_path: str | Path, *, method: str
) -> TrackingIFD | None:
    cache = Path(path)
    source = Path(source_path)
    if not cache.is_file() or not source.is_file():
        return None
    current_provenance = file_provenance(source)
    try:
        with np.load(cache, allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata_json"].item()))
            if int(metadata.get("cache_version", -1)) != CACHE_VERSION:
                return None
            if metadata.get("source_provenance") != current_provenance:
                return None
            fish_ids = tuple(str(value) for value in metadata["fish_ids"])
            return TrackingIFD(
                method=str(method),
                source_path=source,
                source_provenance=current_provenance,
                fish_ids=fish_ids,  # type: ignore[arg-type]
                frames=values["frames"].copy(),
                xyz=values["xyz"].copy(),
                interpolated=values["interpolated"].copy(),
                direct_distances=values["direct_distances"].copy(),
                interpolated_distances=values["interpolated_distances"].copy(),
                minimum_direct=values["minimum_direct"].copy(),
                minimum_pair_index=values["minimum_pair_index"].copy(),
                missing_fish_count=values["missing_fish_count"].copy(),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def prepare_tracking_ifd(
    path: str | Path,
    *,
    method: str,
    cache_path: str | Path | None = None,
    id_column: str | None = None,
    max_frame: int | None = None,
) -> tuple[TrackingIFD, bool]:
    """Load a valid compact cache or build and atomically write a new one."""
    if cache_path is not None:
        cached = load_ifd_cache(cache_path, path, method=method)
        if cached is not None:
            if max_frame is None or cached.frame_max == int(max_frame):
                return cached, True
    store = load_tracking_csv(
        path, method=method, id_column=id_column, max_frame=max_frame
    )
    if cache_path is not None:
        save_ifd_cache(store, cache_path)
    return store, False


@dataclass(frozen=True)
class DetectorConfig:
    enter_mm: float = 12.0
    exit_mm: float = 16.0
    merge_gap_frames: int = 5
    context_frames: int = 166

    def validate(self) -> None:
        if not np.isfinite(self.enter_mm) or self.enter_mm <= 0:
            raise ValueError("enter_mm must be positive and finite")
        if not np.isfinite(self.exit_mm) or self.exit_mm <= self.enter_mm:
            raise ValueError("exit_mm must be finite and greater than enter_mm")
        if self.merge_gap_frames < 0:
            raise ValueError("merge_gap_frames must be non-negative")
        if self.context_frames < 0:
            raise ValueError("context_frames must be non-negative")


def clip_frame_range(start: int, end: int, frame_min: int, frame_max: int) -> tuple[int, int]:
    if frame_max < frame_min:
        raise ValueError("frame_max must be at least frame_min")
    low = max(int(frame_min), min(int(start), int(end)))
    high = min(int(frame_max), max(int(start), int(end)))
    if high < low:
        boundary = int(np.clip(start, frame_min, frame_max))
        return boundary, boundary
    return low, high


def expand_frame_range(
    start: int, end: int, context_frames: int, frame_min: int, frame_max: int
) -> tuple[int, int]:
    if context_frames < 0:
        raise ValueError("context_frames must be non-negative")
    return clip_frame_range(
        int(start) - int(context_frames),
        int(end) + int(context_frames),
        int(frame_min),
        int(frame_max),
    )


def suggest_anchor_frames(
    store: TrackingIFD,
    episode: Mapping[str, Any],
    *,
    separation_mm: float | None = None,
) -> tuple[int, int]:
    """Suggest clean temporal anchors around, but not defined by, a detector cue.

    A preferred anchor has all three fish directly observed and minimum IFD
    greater than the detector exit threshold. The nearest preferred frame on
    each side is used. If none exists, the nearest all-three-direct frame is
    used, followed by the review boundary as a final navigation fallback.
    """
    contact_start = int(episode["contact_start_frame"])
    contact_end = int(episode["contact_end_frame"])
    review_start = max(store.frame_min, int(episode["review_start_frame"]))
    review_end = min(store.frame_max, int(episode["review_end_frame"]))
    if review_end <= review_start:
        raise ValueError("review range needs at least two frames for Anchor A/B")
    if separation_mm is None:
        try:
            parameters = json.loads(str(episode.get("detector_parameters_json", "{}")))
            separation_mm = float(parameters.get("exit_mm", DetectorConfig().exit_mm))
        except (TypeError, ValueError, json.JSONDecodeError):
            separation_mm = DetectorConfig().exit_mm
    if not np.isfinite(separation_mm) or float(separation_mm) <= 0:
        raise ValueError("separation_mm must be positive and finite")

    direct_all = store.missing_fish_count == 0
    separated = (
        direct_all
        & np.isfinite(store.minimum_direct)
        & (store.minimum_direct > float(separation_mm))
    )

    def candidates_between(low: int, high: int, mask: np.ndarray) -> np.ndarray:
        if high < low:
            return np.empty(0, dtype=np.int64)
        section = store.slice_positions(low, high)
        return store.frames[section][mask[section]]

    before_high = min(review_end, contact_start - 1)
    after_low = max(review_start, contact_end + 1)
    before = candidates_between(review_start, before_high, separated)
    after = candidates_between(after_low, review_end, separated)
    if not len(before):
        before = candidates_between(review_start, before_high, direct_all)
    if not len(after):
        after = candidates_between(after_low, review_end, direct_all)
    anchor_a = int(before[-1]) if len(before) else review_start
    anchor_b = int(after[0]) if len(after) else review_end
    if anchor_a >= anchor_b:
        anchor_a, anchor_b = review_start, review_end
    return anchor_a, anchor_b


def _deterministic_episode_id(
    experiment: str, method: str, start: int, end: int, *, manual: bool = False
) -> str:
    kind = "manual" if manual else "detected"
    key = f"{experiment}\x1f{method}\x1f{int(start)}\x1f{int(end)}\x1f{kind}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medaka:contact-episode:{key}"))


def detect_contact_episodes(
    data: TrackingIFD | Sequence[float] | np.ndarray,
    config: DetectorConfig | None = None,
    *,
    frames: Sequence[int] | np.ndarray | None = None,
    pair_distances: np.ndarray | None = None,
    pair_keys: Sequence[str] | None = None,
    experiment: str = "",
    tracking_method: str = "",
    source_provenance: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Detect contact with hysteresis, explicit missing breaks, then gap merge."""
    config = config or DetectorConfig()
    config.validate()
    if isinstance(data, TrackingIFD):
        minimum = np.asarray(data.minimum_direct, dtype=float)
        frame_axis = np.asarray(data.frames, dtype=np.int64)
        distances = np.asarray(data.direct_distances, dtype=float)
        keys = data.pair_keys
        tracking_method = tracking_method or data.method
        source_provenance = source_provenance or data.source_provenance
    else:
        minimum = np.asarray(data, dtype=float).reshape(-1)
        frame_axis = (
            np.arange(len(minimum), dtype=np.int64)
            if frames is None
            else np.asarray(frames, dtype=np.int64).reshape(-1)
        )
        if len(frame_axis) != len(minimum):
            raise ValueError("frames and minimum IFD must have equal length")
        distances = (
            np.asarray(pair_distances, dtype=float)
            if pair_distances is not None
            else minimum[:, None]
        )
        if distances.shape[0] != len(minimum):
            raise ValueError("pair_distances must have one row per frame")
        keys = tuple(pair_keys or PAIR_KEYS[: distances.shape[1]])
    if not len(frame_axis):
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if np.any(np.diff(frame_axis) != 1):
        raise ValueError("detector frames must be sorted and contiguous")

    raw: list[tuple[int, int]] = []
    active = False
    start = last_non_exit = -1
    for position, value in enumerate(minimum):
        finite = bool(np.isfinite(value))
        if not active:
            if finite and value < config.enter_mm:
                active = True
                start = position
                last_non_exit = position
            continue
        if not finite:
            raw.append((start, last_non_exit))
            active = False
        elif value > config.exit_mm:
            raw.append((start, last_non_exit))
            active = False
        else:
            last_non_exit = position
    if active:
        raw.append((start, last_non_exit))

    merged: list[tuple[int, int, int]] = []
    for start_position, end_position in raw:
        if merged and start_position - merged[-1][1] - 1 <= config.merge_gap_frames:
            previous_start, _previous_end, count = merged[-1]
            merged[-1] = (previous_start, end_position, count + 1)
        else:
            merged.append((start_position, end_position, 1))

    detector_json = stable_json(asdict(config))
    provenance_json = stable_json(source_provenance or {})
    rows: list[dict[str, Any]] = []
    for start_position, end_position, raw_count in merged:
        contact_start = int(frame_axis[start_position])
        contact_end = int(frame_axis[end_position])
        window = distances[start_position : end_position + 1]
        finite = np.isfinite(window)
        if finite.any():
            safe = np.where(finite, window, np.inf)
            flat_index = int(np.argmin(safe))
            local_frame, pair_index = np.unravel_index(flat_index, safe.shape)
            minimum_value = float(safe[local_frame, pair_index])
            minimum_frame = int(frame_axis[start_position + local_frame])
            minimum_pair = keys[pair_index] if pair_index < len(keys) else str(pair_index)
        else:  # defensive; entry requires a finite value
            minimum_value, minimum_frame, minimum_pair = math.nan, contact_start, ""
        missing_count = int(np.count_nonzero(~np.isfinite(minimum[start_position : end_position + 1])))
        duration = contact_end - contact_start + 1
        review_start, review_end = expand_frame_range(
            contact_start,
            contact_end,
            config.context_frames,
            int(frame_axis[0]),
            int(frame_axis[-1]),
        )
        rows.append(
            {
                "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                "episode_id": _deterministic_episode_id(
                    experiment, tracking_method, contact_start, contact_end
                ),
                "experiment": str(experiment),
                "tracking_method": str(tracking_method),
                "contact_start_frame": contact_start,
                "contact_end_frame": contact_end,
                "review_start_frame": review_start,
                "review_end_frame": review_end,
                "minimum_ifd_mm": minimum_value,
                "minimum_ifd_frame": minimum_frame,
                "minimum_pair": minimum_pair,
                "duration_frames": duration,
                "missing_direct_frames": missing_count,
                "missing_direct_fraction": missing_count / duration,
                "merged_raw_interval_count": raw_count,
                "detector_parameters_json": detector_json,
                "source_provenance_json": provenance_json,
                "manual": False,
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def manual_episode(
    *,
    experiment: str,
    tracking_method: str,
    start_frame: int,
    end_frame: int,
    frame_min: int,
    frame_max: int,
    context_frames: int = 166,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start, end = clip_frame_range(start_frame, end_frame, frame_min, frame_max)
    review_start, review_end = expand_frame_range(
        start, end, context_frames, frame_min, frame_max
    )
    config = DetectorConfig(context_frames=context_frames)
    return {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "episode_id": _deterministic_episode_id(
            experiment, tracking_method, start, end, manual=True
        ),
        "experiment": experiment,
        "tracking_method": tracking_method,
        "contact_start_frame": start,
        "contact_end_frame": end,
        "review_start_frame": review_start,
        "review_end_frame": review_end,
        "minimum_ifd_mm": "",
        "minimum_ifd_frame": "",
        "minimum_pair": "",
        "duration_frames": end - start + 1,
        "missing_direct_frames": "",
        "missing_direct_fraction": "",
        "merged_raw_interval_count": 1,
        "detector_parameters_json": stable_json(asdict(config)),
        "source_provenance_json": stable_json(source_provenance or {}),
        "manual": True,
    }


def _atomic_write_text(text: str, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_candidate_artifacts(
    candidates: pd.DataFrame,
    csv_path: str | Path,
    summary_json_path: str | Path,
    *,
    experiment: str,
    tracking_method: str,
    config: DetectorConfig,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    missing = set(CANDIDATE_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    ordered = candidates[CANDIDATE_COLUMNS].copy()
    _atomic_write_text(ordered.to_csv(index=False, lineterminator="\n"), csv_path)
    summary = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "software": SOFTWARE_NAME,
        "generated_at_utc": utc_now(),
        "experiment": experiment,
        "tracking_method": tracking_method,
        "detector_parameters": asdict(config),
        "source_provenance": _json_value(source_provenance),
        "candidate_count": int(len(ordered)),
        "detected_candidate_count": int(
            np.count_nonzero(
                ~ordered["manual"].fillna(False).astype(str).str.casefold().isin(
                    {"true", "1", "yes", "y"}
                )
            )
        ),
        "manual_candidate_count": int(
            np.count_nonzero(
                ordered["manual"].fillna(False).astype(str).str.casefold().isin(
                    {"true", "1", "yes", "y"}
                )
            )
        ),
        "columns": CANDIDATE_COLUMNS,
        "direct_observations_only": True,
        "missing_observation_policy": "break_then_optional_gap_merge",
    }
    _atomic_write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", summary_json_path)
    return summary


def empty_clicks() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    return {
        phase: {target: {view: {} for view in VIEWS} for target in IDENTITY_TARGETS}
        for phase in PHASES
    }


def validate_false_positive_detections(value: Any) -> list[dict[str, Any]]:
    """Validate frame/view-local raw SLEAP instances excluded from association."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("false_positive_detections_json is not valid JSON") from error
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError("false-positive detections must be a JSON list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, Mapping):
            raise ValueError("each false-positive detection must be an object")
        view = str(item.get("view", ""))
        if view not in VIEWS:
            raise ValueError(f"false-positive view must be one of {VIEWS}")
        try:
            frame = int(item.get("frame", -1))
            instance_index = int(item.get("instance_index", -1))
        except (TypeError, ValueError) as error:
            raise ValueError("false-positive frame/instance_index must be integers") from error
        if frame < 0 or instance_index < 0:
            raise ValueError("false-positive frame/instance_index must be non-negative")
        detection_key = str(item.get("detection_key", ""))
        if not detection_key:
            raise ValueError("false-positive detection_key is required")
        if detection_key in seen:
            continue
        seen.add(detection_key)
        normalized.append(
            {
                "view": view,
                "frame": frame,
                "instance_index": instance_index,
                "detection_key": detection_key,
                "marked_at_utc": str(item.get("marked_at_utc", "")),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (item["frame"], VIEWS.index(item["view"]), item["instance_index"]),
    )


def serialize_false_positive_detections(value: Any) -> str:
    return stable_json(validate_false_positive_detections(value))


def validate_false_negative_detections(value: Any) -> list[dict[str, Any]]:
    """Validate frame/view-local clicks marking a human-visible missing fish.

    FN marks are candidate-mining evidence only: unlike identity clicks they
    intentionally have no SLEAP instance index or geometry association.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("false_negative_detections_json is not valid JSON") from error
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ValueError("false-negative detections must be a JSON list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float, float]] = set()
    for item in parsed:
        if not isinstance(item, Mapping):
            raise ValueError("each false-negative detection must be an object")
        view = str(item.get("view", ""))
        if view not in VIEWS:
            raise ValueError(f"false-negative view must be one of {VIEWS}")
        try:
            frame = int(item.get("frame", -1))
            click_x = float(item.get("click_x"))
            click_y = float(item.get("click_y"))
        except (TypeError, ValueError) as error:
            raise ValueError("false-negative frame/click coordinates must be numeric") from error
        if frame < 0 or not np.isfinite([click_x, click_y]).all():
            raise ValueError(
                "false-negative frame must be non-negative and click coordinates finite"
            )
        key = (view, frame, round(click_x, 6), round(click_y, 6))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "view": view,
                "frame": frame,
                "click_x": click_x,
                "click_y": click_y,
                "marked_at_utc": str(item.get("marked_at_utc", "")),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["frame"],
            VIEWS.index(item["view"]),
            item["click_x"],
            item["click_y"],
        ),
    )


def serialize_false_negative_detections(value: Any) -> str:
    return stable_json(validate_false_negative_detections(value))


def make_click_evidence(
    *,
    experiment: str,
    view: str,
    frame: int,
    click_x: float,
    click_y: float,
    instance_index: int | None = None,
    anchor_xy: Sequence[float] | None = None,
    node_names: Sequence[str] = ("mouth", "tail"),
    anchor_mode: str = "mouth",
    geometry_by_method: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    coordinates = np.asarray([click_x, click_y], dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("raw click coordinates must be finite")
    if int(frame) < 0:
        raise ValueError("click frame must be non-negative")
    linked = instance_index is not None
    evidence: dict[str, Any] = {
        "selection_status": "sleap_linked" if linked else "free_click",
        "view": view,
        "frame": int(frame),
        "click_x": float(click_x),
        "click_y": float(click_y),
        "detection_key": (
            f"{experiment}|{view}|{int(frame)}|{int(instance_index)}" if linked else ""
        ),
        "instance_index": int(instance_index) if linked else None,
        "anchor_mode": str(anchor_mode),
        "anchor_x": None,
        "anchor_y": None,
        "node_names": [str(name) for name in node_names],
        "geometry_by_method": {
            str(method): str(value)
            for method, value in (geometry_by_method or {}).items()
            if not _missing(value)
        },
    }
    if anchor_xy is not None:
        anchor = np.asarray(anchor_xy, dtype=float)
        if anchor.shape == (2,) and np.isfinite(anchor).all():
            evidence["anchor_x"] = float(anchor[0])
            evidence["anchor_y"] = float(anchor[1])
    return evidence


def validate_clicks(clicks: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(clicks, Mapping):
        raise ValueError("clicks must be a mapping")
    normalized = empty_clicks()
    extra_phases = set(clicks) - set(PHASES)
    if extra_phases:
        raise ValueError(f"unsupported click phases: {sorted(extra_phases)}")
    for phase in PHASES:
        phase_value = clicks.get(phase, {})
        if not isinstance(phase_value, Mapping):
            raise ValueError(f"click phase {phase!r} must be a mapping")
        extra_targets = set(phase_value) - set(IDENTITY_TARGETS)
        if extra_targets:
            raise ValueError(f"unsupported identity targets: {sorted(extra_targets)}")
        for target in IDENTITY_TARGETS:
            target_value = phase_value.get(target, {})
            if not isinstance(target_value, Mapping):
                raise ValueError(f"click target {target!r} must be a mapping")
            extra_views = set(target_value) - set(VIEWS)
            if extra_views:
                raise ValueError(f"unsupported click views: {sorted(extra_views)}")
            for view in VIEWS:
                evidence = target_value.get(view, {})
                if not evidence:
                    continue
                if not isinstance(evidence, Mapping):
                    raise ValueError("click evidence must be a mapping")
                status = str(evidence.get("selection_status", ""))
                if status not in {"free_click", "sleap_linked"}:
                    raise ValueError("selection_status must be free_click or sleap_linked")
                if str(evidence.get("view", "")) != view:
                    raise ValueError("click evidence view does not match its view key")
                frame = int(evidence.get("frame", -1))
                coordinates = np.asarray(
                    [evidence.get("click_x"), evidence.get("click_y")], dtype=float
                )
                if frame < 0 or not np.isfinite(coordinates).all():
                    raise ValueError("click evidence needs a non-negative frame and finite raw coordinates")
                instance = evidence.get("instance_index")
                detection_key = str(evidence.get("detection_key", ""))
                if status == "sleap_linked" and (instance is None or not detection_key):
                    raise ValueError("SLEAP-linked evidence needs instance_index and detection_key")
                geometry = evidence.get("geometry_by_method", {})
                if not isinstance(geometry, Mapping):
                    raise ValueError("geometry_by_method must be a mapping")
                normalized[phase][target][view] = _json_value(dict(evidence))
    return normalized


def serialize_clicks(clicks: Mapping[str, Any]) -> str:
    return stable_json(validate_clicks(clicks))


def deserialize_clicks(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return empty_clicks()
    parsed = json.loads(value) if isinstance(value, str) else value
    if isinstance(parsed, Mapping) and ("pre" in parsed or "post" in parsed):
        parsed = {
            "anchor_a": parsed.get("pre", parsed.get("anchor_a", {})),
            "anchor_b": parsed.get("post", parsed.get("anchor_b", {})),
        }
    return validate_clicks(parsed)


def _consensus_geometry(
    clicks: Mapping[str, Any], phase: str, target: str, tracking_label: str
) -> str | None:
    values: set[str] = set()
    for view in VIEWS:
        evidence = clicks.get(phase, {}).get(target, {}).get(view, {})
        value = evidence.get("geometry_by_method", {}).get(tracking_label, "") if evidence else ""
        if not _missing(value):
            values.add(str(value))
    return next(iter(values)) if len(values) == 1 else None


def suggest_tracking_assessment(
    anchor_a_associations: Mapping[str, Any], anchor_b_associations: Mapping[str, Any]
) -> str:
    """Classify current-ID continuity between the two physical anchors."""
    anchor_a_f = anchor_a_associations.get("female")
    anchor_a_m = anchor_a_associations.get("anchor_male")
    anchor_b_f = anchor_b_associations.get("female")
    anchor_b_m = anchor_b_associations.get("anchor_male")
    if any(
        _missing(value)
        for value in (anchor_a_f, anchor_a_m, anchor_b_f, anchor_b_m)
    ):
        return "incomplete_association"
    values = tuple(
        str(value)
        for value in (anchor_a_f, anchor_a_m, anchor_b_f, anchor_b_m)
    )
    anchor_a_f, anchor_a_m, anchor_b_f, anchor_b_m = values
    if anchor_a_f == anchor_a_m or anchor_b_f == anchor_b_m:
        return "incomplete_association"
    if anchor_b_f == anchor_a_f and anchor_b_m == anchor_a_m:
        return "same_ids_no_swap"
    if anchor_b_f != anchor_a_f:
        return "female_involved_swap"
    if anchor_b_m != anchor_a_m:
        return "male_to_male_swap"
    return "incomplete_association"


def suggest_anchor_mapping(clicks: Mapping[str, Any], tracking_label: str) -> str:
    normalized = validate_clicks(clicks)
    anchor_a = {
        target: _consensus_geometry(normalized, "anchor_a", target, tracking_label)
        for target in IDENTITY_TARGETS
    }
    anchor_b = {
        target: _consensus_geometry(normalized, "anchor_b", target, tracking_label)
        for target in IDENTITY_TARGETS
    }
    return suggest_tracking_assessment(anchor_a, anchor_b)


# Backward-compatible function name; values are now neutral temporal anchors.
suggest_pre_post_mapping = suggest_anchor_mapping


# Short compatibility name used in notebooks and tests.
suggest_mapping = suggest_tracking_assessment


def deterministic_annotation_id(experiment: str, episode_id: str, annotator: str) -> str:
    key = f"{experiment}\x1f{episode_id}\x1f{annotator.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"medaka:contact-episode-gt:{key}"))


def normalize_issue_flags(value: str | Iterable[str] | None) -> str:
    if value is None:
        return ""
    raw = value.split(";") if isinstance(value, str) else list(value)
    selected = {str(item).strip() for item in raw if str(item).strip()}
    unexpected = selected - set(ISSUE_FLAGS)
    if unexpected:
        raise ValueError(f"unsupported issue flags: {sorted(unexpected)}")
    return ";".join(flag for flag in ISSUE_FLAGS if flag in selected)


def make_annotation_row(
    *,
    experiment: str,
    episode: Mapping[str, Any],
    annotator: str,
    anchor_a_frame: int | None = None,
    anchor_b_frame: int | None = None,
    pre_frame: int | None = None,
    post_frame: int | None = None,
    clicks: Mapping[str, Any],
    false_positive_detections: Any = None,
    false_negative_detections: Any = None,
    contact_validity: str,
    tracker_assessment: str,
    assessed_tracking_label: str,
    evidence_quality: str = "unreviewed",
    issue_flags: str | Iterable[str] | None = None,
    notes: str = "",
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not annotator.strip():
        raise ValueError("annotator is required")
    episode_id = str(episode.get("episode_id", ""))
    if not episode_id:
        raise ValueError("episode_id is required")
    if anchor_a_frame is None:
        anchor_a_frame = pre_frame
    if anchor_b_frame is None:
        anchor_b_frame = post_frame
    if anchor_a_frame is None or anchor_b_frame is None:
        raise ValueError("anchor_a_frame and anchor_b_frame are required")
    now = utc_now()
    source_json = stable_json(
        source_provenance
        if source_provenance is not None
        else json.loads(str(episode.get("source_provenance_json", "{}")) or "{}")
    )
    row = {column: "" for column in ANNOTATION_COLUMNS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "annotation_id": deterministic_annotation_id(experiment, episode_id, annotator),
            "experiment": experiment,
            "episode_id": episode_id,
            "annotator": annotator.strip(),
            "tracking_method": str(episode.get("tracking_method", "")),
            "contact_start_frame": int(episode["contact_start_frame"]),
            "contact_end_frame": int(episode["contact_end_frame"]),
            "review_start_frame": int(episode["review_start_frame"]),
            "review_end_frame": int(episode["review_end_frame"]),
            "detector_parameters_json": str(episode.get("detector_parameters_json", "{}")),
            "source_provenance_json": source_json,
            "candidate_json": stable_json(dict(episode)),
            "anchor_a_frame": int(anchor_a_frame),
            "anchor_b_frame": int(anchor_b_frame),
            "clicks_json": serialize_clicks(clicks),
            "false_positive_detections_json": serialize_false_positive_detections(
                false_positive_detections
            ),
            "false_negative_detections_json": serialize_false_negative_detections(
                false_negative_detections
            ),
            "contact_validity": contact_validity,
            "tracker_assessment": tracker_assessment,
            "assessed_tracking_label": assessed_tracking_label,
            "automatic_mapping_suggestion": suggest_anchor_mapping(
                clicks, assessed_tracking_label
            ),
            "evidence_quality": evidence_quality,
            "issue_flags": normalize_issue_flags(issue_flags),
            "notes": notes.strip(),
            "created_at_utc": now,
            "updated_at_utc": now,
        }
    )
    return validate_annotation_row(row)


def _validate_json_object(value: Any, label: str) -> str:
    if not isinstance(value, str):
        value = stable_json(value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return stable_json(parsed)


def validate_annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(ANNOTATION_COLUMNS) - set(row)
    extra = set(row) - set(ANNOTATION_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"annotation row schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    normalized = {column: _plain(row[column]) for column in ANNOTATION_COLUMNS}
    if str(normalized["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    experiment = str(normalized["experiment"])
    episode_id = str(normalized["episode_id"])
    annotator = str(normalized["annotator"]).strip()
    if not experiment or not episode_id or not annotator:
        raise ValueError("experiment, episode_id, and annotator are required")
    expected = deterministic_annotation_id(experiment, episode_id, annotator)
    if str(normalized["annotation_id"]) != expected:
        raise ValueError("annotation_id does not match experiment/episode/annotator")
    frames = {}
    for column in (
        "contact_start_frame",
        "contact_end_frame",
        "review_start_frame",
        "review_end_frame",
        "anchor_a_frame",
        "anchor_b_frame",
    ):
        try:
            numeric = float(normalized[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be a non-negative integer") from error
        if not np.isfinite(numeric) or numeric < 0 or numeric != math.floor(numeric):
            raise ValueError(f"{column} must be a non-negative integer")
        frames[column] = int(numeric)
        normalized[column] = int(numeric)
    if not (
        frames["review_start_frame"] <= frames["contact_start_frame"]
        <= frames["contact_end_frame"] <= frames["review_end_frame"]
    ):
        raise ValueError("contact range must be ordered inside the review range")
    if not (
        frames["review_start_frame"] <= frames["anchor_a_frame"]
        < frames["anchor_b_frame"] <= frames["review_end_frame"]
    ):
        raise ValueError("Anchor A/B frames must be ordered inside the review range")
    if str(normalized["contact_validity"]) not in CONTACT_VALIDITIES:
        raise ValueError(f"unsupported contact_validity {normalized['contact_validity']!r}")
    if str(normalized["tracker_assessment"]) not in TRACKER_ASSESSMENTS:
        raise ValueError(f"unsupported tracker_assessment {normalized['tracker_assessment']!r}")
    if str(normalized["evidence_quality"]) not in EVIDENCE_QUALITIES:
        raise ValueError(f"unsupported evidence_quality {normalized['evidence_quality']!r}")
    if not str(normalized["assessed_tracking_label"]).strip():
        raise ValueError("assessed_tracking_label is required")
    normalized["detector_parameters_json"] = _validate_json_object(
        normalized["detector_parameters_json"], "detector_parameters_json"
    )
    normalized["source_provenance_json"] = _validate_json_object(
        normalized["source_provenance_json"], "source_provenance_json"
    )
    normalized["candidate_json"] = _validate_json_object(
        normalized["candidate_json"], "candidate_json"
    )
    clicks = deserialize_clicks(str(normalized["clicks_json"]))
    normalized["clicks_json"] = serialize_clicks(clicks)
    false_positives = validate_false_positive_detections(
        normalized["false_positive_detections_json"]
    )
    normalized["false_positive_detections_json"] = stable_json(false_positives)
    false_negatives = validate_false_negative_detections(
        normalized["false_negative_detections_json"]
    )
    normalized["false_negative_detections_json"] = stable_json(false_negatives)
    excluded_keys = {item["detection_key"] for item in false_positives}
    for phase in PHASES:
        expected_frame = frames[
            "anchor_a_frame" if phase == "anchor_a" else "anchor_b_frame"
        ]
        for target in IDENTITY_TARGETS:
            for view in VIEWS:
                evidence = clicks[phase][target][view]
                if not evidence:
                    continue
                if int(evidence["frame"]) != expected_frame:
                    raise ValueError(
                        f"{phase}/{target}/{view} click frame does not match its anchor"
                    )
                if str(evidence.get("detection_key", "")) in excluded_keys:
                    raise ValueError(
                        "false-positive SLEAP predictions cannot supply ID-linked click evidence"
                    )
    expected_suggestion = suggest_anchor_mapping(
        clicks, str(normalized["assessed_tracking_label"])
    )
    if str(normalized["automatic_mapping_suggestion"]) != expected_suggestion:
        raise ValueError("automatic_mapping_suggestion does not match saved click associations")
    normalized["issue_flags"] = normalize_issue_flags(str(normalized["issue_flags"]))
    for timestamp in ("created_at_utc", "updated_at_utc"):
        if not str(normalized[timestamp]).strip():
            raise ValueError(f"{timestamp} is required")
    normalized["notes"] = str(normalized["notes"])
    return normalized


def _empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def _upgrade_legacy_annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if str(row.get("schema_version", "")) != LEGACY_SCHEMA_VERSION:
        raise ValueError("legacy contact annotation row has an unsupported schema version")
    upgraded = {column: "" for column in ANNOTATION_COLUMNS}
    for column in ANNOTATION_COLUMNS:
        if column in row:
            upgraded[column] = row[column]
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["anchor_a_frame"] = row.get("pre_frame", "")
    upgraded["anchor_b_frame"] = row.get("post_frame", "")
    clicks = deserialize_clicks(row.get("clicks_json", ""))
    upgraded["clicks_json"] = serialize_clicks(clicks)
    upgraded["false_positive_detections_json"] = "[]"
    upgraded["false_negative_detections_json"] = "[]"
    upgraded["automatic_mapping_suggestion"] = suggest_anchor_mapping(
        clicks, str(upgraded["assessed_tracking_label"])
    )
    return validate_annotation_row(upgraded)


def _upgrade_v1_1_annotation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate v1.1 in memory; the source CSV is untouched until a save."""
    if str(row.get("schema_version", "")) != PREVIOUS_SCHEMA_VERSION:
        raise ValueError("contact annotation row is not schema v1.1")
    upgraded = {column: row.get(column, "") for column in ANNOTATION_COLUMNS}
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["false_negative_detections_json"] = "[]"
    return validate_annotation_row(upgraded)


def read_annotation_ledger(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return _empty_annotations()
    table = pd.read_csv(source, keep_default_na=False, dtype=str)
    columns = set(table.columns)
    if columns == set(LEGACY_ANNOTATION_COLUMNS):
        rows = [
            _upgrade_legacy_annotation_row(row)
            for row in table.to_dict("records")
        ]
    elif columns == set(V1_1_ANNOTATION_COLUMNS):
        rows = [
            _upgrade_v1_1_annotation_row(row)
            for row in table.to_dict("records")
        ]
    elif columns == set(ANNOTATION_COLUMNS):
        rows = [validate_annotation_row(row) for row in table.to_dict("records")]
    else:
        missing = set(ANNOTATION_COLUMNS) - columns
        extra = columns - set(ANNOTATION_COLUMNS)
        raise ValueError(
            f"annotation ledger schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    if result.duplicated(["experiment", "episode_id", "annotator"]).any():
        raise ValueError("annotation ledger contains duplicate canonical keys")
    return sort_annotations(result)


def sort_annotations(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _empty_annotations()
    result = table.copy()
    result["contact_start_frame"] = pd.to_numeric(result["contact_start_frame"]).astype(np.int64)
    return result.sort_values(
        ["experiment", "contact_start_frame", "episode_id", "annotator"],
        kind="stable",
    ).reset_index(drop=True)[ANNOTATION_COLUMNS]


def atomic_write_annotation_ledger(table: pd.DataFrame, path: str | Path) -> None:
    rows = [validate_annotation_row(row) for row in table.to_dict("records")]
    normalized = sort_annotations(pd.DataFrame(rows, columns=ANNOTATION_COLUMNS))
    if normalized.duplicated(["experiment", "episode_id", "annotator"]).any():
        raise ValueError("refusing to write duplicate canonical annotation keys")
    _atomic_write_text(normalized.to_csv(index=False, lineterminator="\n"), path)


def append_audit_record(path: str | Path, record: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(stable_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _annotation_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["experiment"]),
        str(row["episode_id"]),
        str(row["annotator"]).strip(),
    )


class ContactAnnotationLedger:
    """Atomic canonical upserts plus append-only audit and session undo."""

    def __init__(self, csv_path: str | Path, audit_jsonl_path: str | Path):
        self.csv_path = Path(csv_path)
        self.audit_jsonl_path = Path(audit_jsonl_path)
        if normalize_path(self.csv_path) == normalize_path(self.audit_jsonl_path):
            raise ValueError("canonical CSV and audit JSONL must be different files")
        self.rows = read_annotation_ledger(self.csv_path)
        self._session_history: list[tuple[tuple[str, str, str], dict[str, Any] | None, dict[str, Any]]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._session_history)

    def get(self, experiment: str, episode_id: str, annotator: str) -> dict[str, Any] | None:
        if self.rows.empty:
            return None
        mask = (
            self.rows["experiment"].astype(str).eq(str(experiment))
            & self.rows["episode_id"].astype(str).eq(str(episode_id))
            & self.rows["annotator"].astype(str).eq(str(annotator).strip())
        )
        if not mask.any():
            return None
        return {column: _plain(self.rows.loc[mask].iloc[0][column]) for column in ANNOTATION_COLUMNS}

    def upsert(self, row: Mapping[str, Any]) -> str:
        normalized = validate_annotation_row(row)
        key = _annotation_key(normalized)
        old = self.get(*key)
        if old is not None:
            normalized["created_at_utc"] = str(old["created_at_utc"])
            normalized["updated_at_utc"] = utc_now()
            normalized = validate_annotation_row(normalized)
            mask = (
                self.rows["experiment"].astype(str).eq(key[0])
                & self.rows["episode_id"].astype(str).eq(key[1])
                & self.rows["annotator"].astype(str).eq(key[2])
            )
            self.rows = self.rows.loc[~mask].copy()
            action = "edit"
        else:
            action = "add"
        self.rows = pd.concat(
            [self.rows, pd.DataFrame([normalized], columns=ANNOTATION_COLUMNS)],
            ignore_index=True,
        )
        atomic_write_annotation_ledger(self.rows, self.csv_path)
        self.rows = read_annotation_ledger(self.csv_path)
        append_audit_record(
            self.audit_jsonl_path,
            {
                "action": action,
                "timestamp_utc": utc_now(),
                "schema_version": SCHEMA_VERSION,
                "key": list(key),
                "old_row": old,
                "new_row": normalized,
            },
        )
        self._session_history.append((key, old, normalized))
        return action

    def undo(self) -> dict[str, Any] | None:
        if not self._session_history:
            return None
        key, old, new = self._session_history.pop()
        mask = (
            self.rows["experiment"].astype(str).eq(key[0])
            & self.rows["episode_id"].astype(str).eq(key[1])
            & self.rows["annotator"].astype(str).eq(key[2])
        )
        self.rows = self.rows.loc[~mask].copy()
        if old is not None:
            self.rows = pd.concat(
                [self.rows, pd.DataFrame([old], columns=ANNOTATION_COLUMNS)],
                ignore_index=True,
            )
        atomic_write_annotation_ledger(self.rows, self.csv_path)
        self.rows = read_annotation_ledger(self.csv_path)
        append_audit_record(
            self.audit_jsonl_path,
            {
                "action": "undo",
                "timestamp_utc": utc_now(),
                "schema_version": SCHEMA_VERSION,
                "key": list(key),
                "old_row": new,
                "new_row": old,
            },
        )
        return old

    def latest(self, experiment: str, annotator: str) -> dict[str, Any] | None:
        if self.rows.empty:
            return None
        selected = self.rows.loc[
            self.rows["experiment"].astype(str).eq(str(experiment))
            & self.rows["annotator"].astype(str).eq(str(annotator).strip())
        ]
        if selected.empty:
            return None
        index = selected["updated_at_utc"].astype(str).idxmax()
        return {column: _plain(self.rows.loc[index, column]) for column in ANNOTATION_COLUMNS}


# Familiar short name for callers of the existing female ledger.
AnnotationLedger = ContactAnnotationLedger


def queue_decision_state(
    ledger: pd.DataFrame, *, experiment: str, annotator: str
) -> dict[str, dict[str, str]]:
    if ledger.empty:
        return {}
    selected = ledger.loc[
        ledger["experiment"].astype(str).eq(str(experiment))
        & ledger["annotator"].astype(str).eq(str(annotator).strip())
    ]
    return {
        str(row["episode_id"]): {
            "contact_validity": str(row["contact_validity"]),
            "tracker_assessment": str(row["tracker_assessment"]),
        }
        for row in selected.to_dict("records")
    }


def parse_tracking_csv_specs(specs: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError("--tracking-csv must be NAME=PATH")
        name, raw_path = str(spec).split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid tracking method name {name!r}")
        if not raw_path:
            raise ValueError(f"tracking path for {name!r} is empty")
        if name in result:
            raise ValueError(f"duplicate tracking method name {name!r}")
        result[name] = Path(raw_path)
    if not result:
        raise ValueError("at least one --tracking-csv NAME=PATH is required")
    return result
