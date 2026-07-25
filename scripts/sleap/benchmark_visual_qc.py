"""Manual visual QC for the common Round 7 / Round 8 SLEAP benchmark.

The interactive window is intentionally a thin PyQt5 layer over headless,
tested data and rendering helpers. Source videos, manifests, prediction
artifacts, and SLP packages are read-only. Review decisions are written to a
separate CSV plus JSON sidecar.

The preferred cached prediction format is a long CSV/Parquet/JSON table with
one row per GT/prediction match. It must contain ``frame_key`` and either
``model`` or ``round``.
Coordinates may be encoded as ``gt_points`` / ``pred_points`` JSON arrays with
shape ``(nodes, 2)`` or as explicit ``gt_mouth_x``, ``gt_mouth_y``, ...,
``pred_tail_y`` columns. Native ``.slp`` files are also accepted.

Run ``python -m scripts.sleap.benchmark_visual_qc --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"
REVIEW_LABELS = (
    "model_error",
    "annotation_error",
    "matching_error",
    "ambiguous",
    "exclude",
    "looks_correct",
)
REVIEW_COLUMNS = (
    "schema_version",
    "frame_key",
    "experiment",
    "view",
    "frame_idx",
    "review_label",
    "notes",
    "annotator",
    "updated_at_utc",
)
FILTERS = (
    "all",
    "unreviewed",
    "reviewed",
    "misses",
    "extras",
    "merges",
    "swaps",
    "shared_tails",
    "partial",
    "needs_manual_review",
)
SORTS = ("manifest", "frame", "mouth_error_desc", "tail_error_desc")
MODEL_COLORS_RGB = {
    "round7": (231, 88, 176),
    "round8": (45, 204, 222),
}
GT_COLOR_RGB = (95, 222, 95)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(float(value)))
    return isinstance(value, str) and not value.strip()


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value)


def _number(value: Any, default: float = float("nan")) -> float:
    if _missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    return default if not np.isfinite(number) else int(number)


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and value != 0)
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if _missing(value):
        return ""
    return value


def _json_value(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ReviewDecision:
    frame_key: str
    experiment: str
    view: str
    frame_idx: int
    review_label: str
    notes: str
    annotator: str
    updated_at_utc: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.frame_key:
            raise ValueError("frame_key is required")
        if self.review_label and self.review_label not in REVIEW_LABELS:
            raise ValueError(f"unknown review label: {self.review_label}")


class ReviewStore:
    """In-memory review ledger with atomic CSV and JSON persistence."""

    def __init__(
        self,
        csv_path: Path,
        *,
        json_path: Path | None = None,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path or self.csv_path.with_suffix(".json"))
        self.source_provenance = dict(source_provenance or {})
        self.decisions: dict[str, ReviewDecision] = {}
        self.load()

    def load(self) -> None:
        self.decisions.clear()
        if not self.csv_path.exists():
            return
        table = pd.read_csv(self.csv_path, keep_default_na=False)
        missing = {"frame_key", "review_label", "notes"} - set(table.columns)
        if missing:
            raise ValueError(
                f"review CSV {self.csv_path} is missing columns: {sorted(missing)}"
            )
        for row in table.to_dict("records"):
            decision = ReviewDecision(
                schema_version=_text(row.get("schema_version")) or SCHEMA_VERSION,
                frame_key=_text(row["frame_key"]),
                experiment=_text(row.get("experiment")),
                view=_text(row.get("view")),
                frame_idx=_integer(row.get("frame_idx")),
                review_label=_text(row.get("review_label")),
                notes=_text(row.get("notes")),
                annotator=_text(row.get("annotator")),
                updated_at_utc=_text(row.get("updated_at_utc")) or utc_now(),
            )
            self.decisions[decision.frame_key] = decision

    def get(self, frame_key: str) -> ReviewDecision | None:
        return self.decisions.get(str(frame_key))

    def set(self, decision: ReviewDecision, *, save: bool = True) -> None:
        self.decisions[decision.frame_key] = decision
        if save:
            self.save()

    def remove(self, frame_key: str, *, save: bool = True) -> None:
        self.decisions.pop(str(frame_key), None)
        if save:
            self.save()

    def reviewed_count(self, frame_keys: Iterable[str] | None = None) -> int:
        keys = (
            set(self.decisions)
            if frame_keys is None
            else {str(value) for value in frame_keys}
        )
        return sum(
            bool(self.decisions[key].review_label)
            for key in keys
            if key in self.decisions
        )

    def save(self) -> None:
        rows = [
            {column: _plain(asdict(decision).get(column, "")) for column in REVIEW_COLUMNS}
            for decision in sorted(
                self.decisions.values(), key=lambda item: item.frame_key
            )
        ]
        table = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
        _atomic_text(self.csv_path, table.to_csv(index=False, lineterminator="\n"))
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "review_csv": str(self.csv_path.resolve(strict=False)),
            "source_provenance": self.source_provenance,
            "counts": {
                "saved_rows": len(rows),
                "reviewed": sum(bool(row["review_label"]) for row in rows),
                "with_notes": sum(bool(row["notes"]) for row in rows),
            },
            "decisions": rows,
        }
        _atomic_text(self.json_path, stable_json(sidecar, indent=2) + "\n")


@dataclass
class PoseInstance:
    mouth: tuple[float, float] | None = None
    tail: tuple[float, float] | None = None
    instance_id: str = ""


@dataclass
class FrameEvidence:
    gt: list[PoseInstance]
    predictions: dict[str, list[PoseInstance]]
    model_metrics: dict[str, dict[str, Any]]


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path))).casefold()


def _view_family(value: Any) -> str:
    normalized = _text(value).casefold().replace("-", "_")
    if "side" in normalized:
        return "side"
    if "top" in normalized or "cam0" in normalized:
        return "top"
    return normalized


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in row and not _missing(row[name]):
            return row[name]
    return default


def load_manifest(path: Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".parquet":
        table = pd.read_parquet(source)
    elif suffix in {".json", ".jsonl"}:
        table = _json_table(source)
    else:
        table = pd.read_csv(source, keep_default_na=False)
    aliases = {
        "frame_key": ("frame_key", "stable_frame_key", "key"),
        "frame_idx": ("frame_idx", "frame_index", "frame"),
        "video_path": ("video_path", "video_source_path", "source_path"),
    }
    for canonical, candidates in aliases.items():
        if canonical not in table.columns:
            found = next((name for name in candidates if name in table.columns), None)
            if found:
                table[canonical] = table[found]
    required = {"frame_key", "experiment", "view", "frame_idx"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"manifest {source} is missing columns: {sorted(missing)}")
    table = table.copy()
    table["frame_key"] = table["frame_key"].astype(str)
    if table["frame_key"].duplicated().any():
        duplicates = table.loc[table["frame_key"].duplicated(), "frame_key"].tolist()
        raise ValueError(f"manifest frame_key values are not unique: {duplicates[:5]}")
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(int)
    table["_manifest_order"] = np.arange(len(table), dtype=int)
    return table


def _json_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".jsonl":
        return pd.read_json(path, lines=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("frames", "frame_results", "matches", "rows", "predictions"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ValueError(
                f"{path} must contain a list or one of frames/frame_results/"
                "matches/rows/predictions"
            )
    else:
        raise ValueError(f"unsupported JSON prediction structure in {path}")
    return pd.DataFrame(rows)


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _point(value: Any) -> tuple[float, float] | None:
    parsed = _parse_jsonish(value)
    if not isinstance(parsed, (list, tuple, np.ndarray)) or len(parsed) < 2:
        return None
    x, y = _number(parsed[0]), _number(parsed[1])
    return (x, y) if np.isfinite(x) and np.isfinite(y) else None


def _instance_from_points(value: Any, instance_id: str = "") -> PoseInstance | None:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, Mapping):
        mouth = _point(parsed.get("mouth"))
        tail = _point(parsed.get("tail"))
    elif isinstance(parsed, (list, tuple, np.ndarray)):
        mouth = _point(parsed[0]) if len(parsed) > 0 else None
        tail = _point(parsed[1]) if len(parsed) > 1 else None
    else:
        return None
    if mouth is None and tail is None:
        return None
    return PoseInstance(mouth=mouth, tail=tail, instance_id=instance_id)


def _instances_from_value(value: Any) -> list[PoseInstance]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, Mapping) and "instances" in parsed:
        parsed = parsed["instances"]
    if not isinstance(parsed, (list, tuple)):
        return []
    if parsed and _instance_from_points(parsed) is not None:
        # A single nodes x 2 array.
        first = parsed[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            scalar_pair = all(
                isinstance(item, (int, float, np.number)) or item is None
                for item in first[:2]
            )
            if scalar_pair:
                instance = _instance_from_points(parsed)
                return [instance] if instance else []
    output = []
    for index, item in enumerate(parsed):
        instance = _instance_from_points(item, str(index))
        if instance is not None:
            output.append(instance)
    return output


def _instance_from_columns(
    row: Mapping[str, Any], prefix: str, instance_id: str
) -> PoseInstance | None:
    points_value = _first(
        row,
        (f"{prefix}_points", f"{prefix}_keypoints", f"{prefix}_pose"),
        None,
    )
    if points_value is not None:
        result = _instance_from_points(points_value, instance_id)
        if result is not None:
            return result

    def coordinate(node: str) -> tuple[float, float] | None:
        x = _number(
            _first(
                row,
                (f"{prefix}_{node}_x", f"{prefix}_{node}_x_px"),
                float("nan"),
            )
        )
        y = _number(
            _first(
                row,
                (f"{prefix}_{node}_y", f"{prefix}_{node}_y_px"),
                float("nan"),
            )
        )
        return (x, y) if np.isfinite(x) and np.isfinite(y) else None

    mouth, tail = coordinate("mouth"), coordinate("tail")
    if mouth is None and tail is None:
        return None
    return PoseInstance(mouth=mouth, tail=tail, instance_id=instance_id)


def _merge_metric(metrics: dict[str, Any], row: Mapping[str, Any]) -> None:
    flag_aliases = {
        "is_miss": ("is_miss", "miss", "missed"),
        "is_extra": ("is_extra", "extra"),
        "is_merge": ("is_merge", "merge", "merged"),
        "is_swap": ("is_swap", "swap", "mouth_tail_swap"),
        "is_shared_tail": (
            "is_shared_tail",
            "shared_tail",
            "duplicated_tail",
        ),
        "is_partial": ("is_partial", "partial", "partial_label"),
    }
    for canonical, aliases in flag_aliases.items():
        if any(_truthy(row.get(name)) for name in aliases if name in row):
            metrics[canonical] = True
    status = _text(row.get("status")).casefold()
    for token, canonical in (
        ("miss", "is_miss"),
        ("extra", "is_extra"),
        ("merge", "is_merge"),
        ("swap", "is_swap"),
        ("shared_tail", "is_shared_tail"),
        ("partial", "is_partial"),
    ):
        if token in status:
            metrics[canonical] = True
    error_aliases = {
        "mouth_error_px": ("mouth_error_px", "mouth_error"),
        "tail_error_px": ("tail_error_px", "tail_error"),
        "match_distance_px": ("match_distance_px", "match_distance"),
    }
    for canonical, aliases in error_aliases.items():
        value = _number(_first(row, aliases, float("nan")))
        if np.isfinite(value):
            previous = _number(metrics.get(canonical))
            metrics[canonical] = (
                value if not np.isfinite(previous) else max(previous, value)
            )
    match_basis = _text(row.get("match_basis"))
    if match_basis:
        bases = set(metrics.get("match_bases", []))
        bases.add(match_basis)
        metrics["match_bases"] = sorted(bases)


def load_tabular_predictions(
    path: Path,
    *,
    default_model: str = "",
) -> dict[str, FrameEvidence]:
    source = Path(path)
    if source.suffix.casefold() == ".parquet":
        table = pd.read_parquet(source)
    elif source.suffix.casefold() in {".json", ".jsonl"}:
        table = _json_table(source)
    else:
        table = pd.read_csv(source, keep_default_na=False)
    if "frame_key" not in table.columns and "stable_frame_key" in table.columns:
        table["frame_key"] = table["stable_frame_key"]
    if "frame_key" not in table.columns:
        raise ValueError(f"prediction artifact {source} lacks frame_key")

    working: dict[str, dict[str, Any]] = {}
    for row in table.to_dict("records"):
        frame_key = _text(row.get("frame_key"))
        if not frame_key:
            continue
        model = _text(_first(row, ("model", "round"))) or default_model
        if not model:
            raise ValueError(
                f"prediction artifact {source} needs a model column or MODEL=PATH CLI"
            )
        slot = working.setdefault(
            frame_key,
            {"gt": OrderedDict(), "pred": {}, "metrics": {}},
        )
        model_key = model.casefold()
        model_predictions = slot["pred"].setdefault(model_key, OrderedDict())
        model_metrics = slot["metrics"].setdefault(model_key, {})
        _merge_metric(model_metrics, row)

        for field in ("gt_instances", "gt_instances_json"):
            if field in row and not _missing(row[field]):
                for index, instance in enumerate(_instances_from_value(row[field])):
                    slot["gt"].setdefault(str(index), instance)
        for field in (
            "pred_instances",
            "pred_instances_json",
            "prediction_instances",
        ):
            if field in row and not _missing(row[field]):
                for index, instance in enumerate(_instances_from_value(row[field])):
                    model_predictions.setdefault(str(index), instance)

        gt_index = _text(row.get("gt_index")) or _text(row.get("gt_instance"))
        pred_index = _text(row.get("pred_index")) or _text(row.get("pred_instance"))
        gt = _instance_from_columns(row, "gt", gt_index)
        pred = _instance_from_columns(row, "pred", pred_index)
        if gt is not None:
            slot["gt"].setdefault(gt_index or f"row{len(slot['gt'])}", gt)
        if pred is not None:
            model_predictions.setdefault(
                pred_index or f"row{len(model_predictions)}", pred
            )

    return {
        frame_key: FrameEvidence(
            gt=list(value["gt"].values()),
            predictions={
                model: list(instances.values())
                for model, instances in value["pred"].items()
            },
            model_metrics=value["metrics"],
        )
        for frame_key, value in working.items()
    }


def _video_source_path(video: Any) -> str:
    source = getattr(video, "source_video", None)
    if source is not None and getattr(source, "filename", None):
        return str(source.filename)
    backend = getattr(video, "backend", None)
    source_name = (
        getattr(backend, "source_filename", None) if backend is not None else None
    )
    if source_name:
        return str(source_name)
    return str(getattr(video, "filename", "")) or ""


def _slp_instances(labeled_frame: Any) -> list[PoseInstance]:
    output: list[PoseInstance] = []
    for index, instance in enumerate(labeled_frame.instances):
        points = np.asarray(instance.numpy(), dtype=float)
        names: list[str] = []
        skeleton = getattr(instance, "skeleton", None)
        for node in getattr(skeleton, "nodes", []) if skeleton is not None else []:
            names.append(str(getattr(node, "name", node)).casefold())

        def named(name: str, fallback: int) -> tuple[float, float] | None:
            position = names.index(name) if name in names else fallback
            if position >= len(points) or not np.all(np.isfinite(points[position, :2])):
                return None
            return float(points[position, 0]), float(points[position, 1])

        mouth, tail = named("mouth", 0), named("tail", 1)
        if mouth is not None or tail is not None:
            output.append(PoseInstance(mouth, tail, str(index)))
    return output


def load_slp_predictions(
    path: Path,
    manifest: pd.DataFrame,
    *,
    model: str,
) -> dict[str, FrameEvidence]:
    """Map a native SLP to manifest frames by source video and exact frame."""
    try:
        import sleap_io as sio
    except ImportError as error:
        raise RuntimeError("sleap-io is required to read native .slp predictions") from error
    labels = sio.load_file(path)
    by_exact: dict[tuple[str, int], str] = {}
    by_identity: dict[tuple[str, str, int], str] = {}
    for row in manifest.to_dict("records"):
        frame_key = str(row["frame_key"])
        frame = int(row["frame_idx"])
        video_path = _text(row.get("video_path"))
        if video_path:
            by_exact[(_path_key(video_path), frame)] = frame_key
        by_identity[
            (
                _text(row.get("experiment")).casefold(),
                _text(row.get("view")).casefold(),
                frame,
            )
        ] = frame_key

    output: dict[str, FrameEvidence] = {}
    labeled_frames = list(labels.labeled_frames)
    for labeled_frame in labeled_frames:
        video_path = _video_source_path(labeled_frame.video)
        frame = int(labeled_frame.frame_idx)
        frame_key = by_exact.get((_path_key(video_path), frame))
        if frame_key is None:
            normalized = video_path.replace("\\", "/").casefold()
            identity = next(
                (
                    key
                    for (experiment, view, index), key in by_identity.items()
                    if index == frame
                    and experiment in normalized
                    and (
                        view in normalized
                        or view.replace("_cam", "") in normalized
                    )
                ),
                None,
            )
            frame_key = identity
        if frame_key is None:
            continue
        output[frame_key] = FrameEvidence(
            gt=[],
            predictions={model.casefold(): _slp_instances(labeled_frame)},
            model_metrics={model.casefold(): {}},
        )
    # sleap-nn preserves input order, but after SLP serialization video-object
    # or embedded-package paths can differ. The evaluator verifies cache length
    # against its ordered manifest subset, so use that order only when the
    # detected view family gives an exact length match.
    families = {
        _view_family(_video_source_path(labeled_frame.video))
        for labeled_frame in labeled_frames
    }
    families.discard("")
    if len(families) != 1:
        path_family = _view_family(path.name)
        families = {path_family} if path_family in {"top", "side"} else families
    if len(families) == 1:
        family = next(iter(families))
        candidates = [
            row
            for row in manifest.to_dict("records")
            if _view_family(row.get("view")) == family
        ]
        if len(candidates) == len(labeled_frames) and len(output) != len(labeled_frames):
            output = {
                str(row["frame_key"]): FrameEvidence(
                    gt=[],
                    predictions={
                        model.casefold(): _slp_instances(labeled_frame)
                    },
                    model_metrics={model.casefold(): {}},
                )
                for row, labeled_frame in zip(
                    candidates, labeled_frames, strict=True
                )
            }
    return output


def merge_evidence(
    destination: dict[str, FrameEvidence],
    incoming: Mapping[str, FrameEvidence],
) -> None:
    for frame_key, evidence in incoming.items():
        target = destination.setdefault(frame_key, FrameEvidence([], {}, {}))
        if evidence.gt:
            target.gt = evidence.gt
        for model, instances in evidence.predictions.items():
            target.predictions[model.casefold()] = instances
        for model, metrics in evidence.model_metrics.items():
            current = target.model_metrics.setdefault(model.casefold(), {})
            current.update(metrics)


def load_prediction_spec(
    spec: str,
    *,
    manifest: pd.DataFrame,
) -> tuple[str, Path, dict[str, FrameEvidence]]:
    if "=" in spec:
        model, raw_path = spec.split("=", 1)
    else:
        model, raw_path = "", spec
    path = Path(raw_path)
    inferred = model.strip().casefold()
    if not inferred:
        name = path.stem.casefold()
        inferred = "round8" if "round8" in name else "round7" if "round7" in name else ""
    if path.suffix.casefold() == ".slp":
        if not inferred:
            raise ValueError("native SLP prediction specs require MODEL=PATH")
        evidence = load_slp_predictions(path, manifest, model=inferred)
    else:
        evidence = load_tabular_predictions(path, default_model=inferred)
        if not inferred:
            models = {
                model
                for item in evidence.values()
                for model in item.predictions
            }
            inferred = next(iter(models)) if len(models) == 1 else "mixed"
    return inferred, path, evidence


def load_ground_truth_spec(
    path: Path,
    *,
    manifest: pd.DataFrame,
) -> dict[str, FrameEvidence]:
    """Load GT poses from a native SLP or a table containing GT columns."""
    source = Path(path)
    if source.suffix.casefold() != ".slp":
        loaded = load_tabular_predictions(source, default_model="__ground_truth__")
        for frame_evidence in loaded.values():
            if not frame_evidence.gt:
                frame_evidence.gt = frame_evidence.predictions.get(
                    "__ground_truth__", []
                )
            frame_evidence.predictions.pop("__ground_truth__", None)
            frame_evidence.model_metrics.pop("__ground_truth__", None)
        return loaded
    loaded = load_slp_predictions(source, manifest, model="__ground_truth__")
    for frame_evidence in loaded.values():
        frame_evidence.gt = frame_evidence.predictions.pop("__ground_truth__", [])
        frame_evidence.model_metrics.pop("__ground_truth__", None)
    return loaded


def row_summary(
    row: Mapping[str, Any],
    evidence: FrameEvidence | None,
    decision: ReviewDecision | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(row)
    summary["reviewed"] = bool(decision and decision.review_label)
    summary["review_label"] = decision.review_label if decision else ""
    metrics = evidence.model_metrics if evidence else {}
    for flag in (
        "is_miss",
        "is_extra",
        "is_merge",
        "is_swap",
        "is_shared_tail",
        "is_partial",
    ):
        summary[flag] = _truthy(summary.get(flag)) or any(
            _truthy(model_metrics.get(flag)) for model_metrics in metrics.values()
        )
    for error in ("mouth_error_px", "tail_error_px"):
        values = [
            _number(summary.get(error)),
            *[_number(model_metrics.get(error)) for model_metrics in metrics.values()],
        ]
        finite = [value for value in values if np.isfinite(value)]
        summary[error] = max(finite) if finite else float("nan")
    difficulty = _text(summary.get("difficulty")).casefold()
    visibility = _text(summary.get("node_visibility")).casefold()
    summary["is_partial"] = (
        summary["is_partial"]
        or "partial" in difficulty
        or "partial" in visibility
        or "occlud" in visibility
    )
    return summary


def filter_sort_rows(
    manifest: pd.DataFrame,
    evidence: Mapping[str, FrameEvidence],
    decisions: Mapping[str, ReviewDecision],
    *,
    filter_name: str = "all",
    sort_name: str = "manifest",
) -> list[int]:
    if filter_name not in FILTERS:
        raise ValueError(f"unknown filter: {filter_name}")
    if sort_name not in SORTS:
        raise ValueError(f"unknown sort: {sort_name}")
    summaries = []
    for index, row in manifest.iterrows():
        frame_key = str(row["frame_key"])
        summaries.append(
            (
                int(index),
                row_summary(row, evidence.get(frame_key), decisions.get(frame_key)),
            )
        )
    predicates = {
        "all": lambda value: True,
        "unreviewed": lambda value: not value["reviewed"],
        "reviewed": lambda value: value["reviewed"],
        "misses": lambda value: value["is_miss"],
        "extras": lambda value: value["is_extra"],
        "merges": lambda value: value["is_merge"],
        "swaps": lambda value: value["is_swap"],
        "shared_tails": lambda value: value["is_shared_tail"],
        "partial": lambda value: value["is_partial"],
        "needs_manual_review": lambda value: _truthy(
            value.get("needs_manual_review")
        ),
    }
    selected = [item for item in summaries if predicates[filter_name](item[1])]
    if sort_name == "frame":
        selected.sort(
            key=lambda item: (
                _text(item[1].get("experiment")),
                _text(item[1].get("view")),
                _integer(item[1].get("frame_idx")),
            )
        )
    elif sort_name in {"mouth_error_desc", "tail_error_desc"}:
        column = sort_name.removesuffix("_desc") + "_px"
        selected.sort(
            key=lambda item: (
                -_number(item[1].get(column), default=-math.inf),
                _integer(item[1].get("_manifest_order")),
            )
        )
    else:
        selected.sort(key=lambda item: _integer(item[1].get("_manifest_order")))
    return [index for index, _summary in selected]


def model_color(model: str) -> tuple[int, int, int]:
    normalized = model.casefold()
    if normalized in MODEL_COLORS_RGB:
        return MODEL_COLORS_RGB[normalized]
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return tuple(int(80 + value * 0.65) for value in digest[:3])


def draw_pose_overlay(
    image_rgb: np.ndarray,
    *,
    gt_instances: Sequence[PoseInstance],
    predictions: Sequence[PoseInstance],
    model: str,
) -> np.ndarray:
    """Draw a legible GT/model overlay without any GUI dependency."""
    canvas = np.ascontiguousarray(image_rgb.copy())

    def draw_instance(
        instance: PoseInstance,
        color: tuple[int, int, int],
        *,
        gt: bool,
        index: int,
    ) -> None:
        thickness = 3 if gt else 2
        mouth, tail = instance.mouth, instance.tail
        if mouth is not None and tail is not None:
            cv2.line(
                canvas,
                tuple(int(round(value)) for value in mouth),
                tuple(int(round(value)) for value in tail),
                color,
                thickness,
                cv2.LINE_AA,
            )
        if mouth is not None:
            center = tuple(int(round(value)) for value in mouth)
            cv2.circle(canvas, center, 7 if gt else 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 9 if gt else 8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{'G' if gt else 'P'}{index}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        if tail is not None:
            x, y = (int(round(value)) for value in tail)
            size = 8 if gt else 7
            triangle = np.array(
                [[x, y - size], [x - size, y + size], [x + size, y + size]],
                dtype=np.int32,
            )
            cv2.polylines(
                canvas, [triangle], True, color, thickness, cv2.LINE_AA
            )

    for index, instance in enumerate(gt_instances):
        draw_instance(instance, GT_COLOR_RGB, gt=True, index=index)
    for index, instance in enumerate(predictions):
        draw_instance(instance, model_color(model), gt=False, index=index)
    return canvas


def metadata_text(
    row: Mapping[str, Any],
    evidence: FrameEvidence | None,
) -> str:
    summary = row_summary(row, evidence, None)
    fields = [
        f"key={_text(row.get('frame_key'))}",
        f"experiment={_text(row.get('experiment'))}",
        f"view={_text(row.get('view'))}",
        f"frame={_integer(row.get('frame_idx')):,}",
        f"cohort={_text(row.get('cohort'))}",
        f"difficulty={_text(row.get('difficulty'))}",
        f"visibility={_text(row.get('node_visibility'))}",
    ]
    mouth = _number(summary.get("mouth_error_px"))
    tail = _number(summary.get("tail_error_px"))
    if np.isfinite(mouth):
        fields.append(f"max mouth error={mouth:.2f}px")
    if np.isfinite(tail):
        fields.append(f"max tail error={tail:.2f}px")
    failures = [
        label
        for key, label in (
            ("is_miss", "miss"),
            ("is_extra", "extra"),
            ("is_merge", "merge"),
            ("is_swap", "swap"),
            ("is_shared_tail", "shared tail"),
            ("is_partial", "partial"),
        )
        if summary[key]
    ]
    if failures:
        fields.append("flags=" + ", ".join(failures))
    if evidence:
        bases = sorted(
            {
                basis
                for metrics in evidence.model_metrics.values()
                for basis in metrics.get("match_bases", [])
            }
        )
        if bases:
            fields.append("match basis=" + ", ".join(bases))
    return " | ".join(fields)


def model_metric_text(model: str, metrics: Mapping[str, Any]) -> str:
    fields = [model]
    mouth = _number(metrics.get("mouth_error_px"))
    tail = _number(metrics.get("tail_error_px"))
    if np.isfinite(mouth):
        fields.append(f"max mouth={mouth:.2f}px")
    if np.isfinite(tail):
        fields.append(f"max tail={tail:.2f}px")
    flags = [
        label
        for key, label in (
            ("is_miss", "miss"),
            ("is_extra", "extra"),
            ("is_merge", "merge"),
            ("is_swap", "swap"),
            ("is_shared_tail", "shared tail"),
            ("is_partial", "partial"),
        )
        if _truthy(metrics.get(key))
    ]
    if flags:
        fields.append(", ".join(flags))
    bases = metrics.get("match_bases", [])
    if bases:
        fields.append("basis=" + ",".join(str(value) for value in bases))
    return " | ".join(fields)


def export_frame_overlay(
    manifest: pd.DataFrame,
    evidence: Mapping[str, FrameEvidence],
    *,
    frame_key: str,
    output_path: Path,
    models: Sequence[str] = ("round7", "round8"),
    reader: "ExactFrameReader | None" = None,
) -> Path:
    """Render one comparison PNG headlessly using the interactive styling."""
    matches = manifest.index[
        manifest["frame_key"].astype(str).eq(str(frame_key))
    ].tolist()
    if not matches:
        raise ValueError(f"unknown frame_key: {frame_key}")
    row = manifest.loc[matches[0]]
    image = (reader or ExactFrameReader()).get(row)
    frame_evidence = evidence.get(str(frame_key), FrameEvidence([], {}, {}))
    available = [
        model.casefold()
        for model in models
        if model.casefold() in frame_evidence.predictions
        or model.casefold() in frame_evidence.model_metrics
    ]
    if not available:
        available = sorted(
            set(frame_evidence.predictions) | set(frame_evidence.model_metrics)
        )
    if not available:
        raise ValueError(f"no cached prediction evidence for frame_key: {frame_key}")

    panels: list[np.ndarray] = []
    for model in available:
        overlay = draw_pose_overlay(
            image,
            gt_instances=frame_evidence.gt,
            predictions=frame_evidence.predictions.get(model, []),
            model=model,
        )
        header = np.full((54, overlay.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(
            header,
            model_metric_text(
                model, frame_evidence.model_metrics.get(model, {})
            )[:140],
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            model_color(model),
            2,
            cv2.LINE_AA,
        )
        panels.append(np.vstack([header, overlay]))
    body = np.hstack(panels)
    title = np.full((72, body.shape[1], 3), 250, dtype=np.uint8)
    cv2.putText(
        title,
        metadata_text(row, frame_evidence)[:220],
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 15, 15),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(title, (22, 54), 6, GT_COLOR_RGB, -1, cv2.LINE_AA)
    cv2.putText(
        title,
        "GT (circle mouth / triangle tail / connecting skeleton)",
        (34, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    canvas = np.vstack([title, body])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"could not write PNG: {destination}")
    return destination


class ExactFrameReader:
    """Small read-only LRU for exact image or video frames."""

    def __init__(self, max_frames: int = 8) -> None:
        self.max_frames = int(max_frames)
        self.cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def get(self, row: Mapping[str, Any]) -> np.ndarray:
        frame_idx = _integer(row.get("frame_idx"))
        image_path = _text(_first(row, ("image_path", "frame_path")))
        video_path = _text(
            _first(row, ("video_path", "video_source_path", "source_path"))
        )
        source = image_path or video_path
        if not source:
            raise ValueError(
                f"frame {_text(row.get('frame_key'))} has no image_path or video_path"
            )
        key = (_path_key(source), frame_idx if not image_path else -1)
        if key in self.cache:
            image = self.cache.pop(key)
            self.cache[key] = image
            return image.copy()
        if image_path:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"could not read image: {image_path}")
        else:
            capture = cv2.VideoCapture(video_path)
            try:
                if not capture.isOpened():
                    raise FileNotFoundError(f"could not open video: {video_path}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(
                        f"could not decode exact frame {frame_idx} from {video_path}"
                    )
            finally:
                capture.release()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.cache[key] = rgb
        while len(self.cache) > self.max_frames:
            self.cache.popitem(last=False)
        return rgb.copy()


def _load_gui_dependencies():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError(
            "The GUI extra is required. Run: uv sync --extra gui"
        ) from error
    return (
        QtCore,
        QtGui,
        QtWidgets,
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
        Figure,
        Line2D,
    )


def create_qc_window(
    args: argparse.Namespace,
    *,
    manifest: pd.DataFrame,
    evidence: dict[str, FrameEvidence],
    store: ReviewStore,
):
    (
        QtCore,
        QtGui,
        QtWidgets,
        FigureCanvas,
        NavigationToolbar,
        Figure,
        Line2D,
    ) = _load_gui_dependencies()

    class QcWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("SLEAP common-benchmark visual QC")
            self.resize(1660, 980)
            self.manifest = manifest
            self.evidence = evidence
            self.store = store
            self.reader = ExactFrameReader()
            self.filtered_indices: list[int] = []
            self.position = -1
            self.current_index: int | None = None
            self.loading_decision = False
            self.current_image: np.ndarray | None = None
            self.models = sorted(
                {
                    model
                    for frame in evidence.values()
                    for model in frame.predictions
                }
            )
            if not self.models:
                self.models = ["round7", "round8"]
            self.autosave_timer = QtCore.QTimer(self)
            self.autosave_timer.setSingleShot(True)
            self.autosave_timer.setInterval(350)
            self.autosave_timer.timeout.connect(self._save_decision)
            self._build()
            self._shortcuts()
            self._refresh_queue()

        def _build(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QHBoxLayout(central)
            left = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left)
            root.addWidget(left, 1)
            self.figure = Figure(figsize=(12, 8), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)
            left_layout.addWidget(self.toolbar)
            left_layout.addWidget(self.canvas, 1)
            self.canvas.mpl_connect("scroll_event", self._scroll_zoom)

            controls = QtWidgets.QWidget()
            controls.setMaximumWidth(390)
            sidebar = QtWidgets.QVBoxLayout(controls)
            root.addWidget(controls)

            sidebar.addWidget(QtWidgets.QLabel("<b>Comparison</b>"))
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItem("Side by side", "side_by_side")
            self.mode_combo.addItem("Toggle one model", "toggle")
            self.mode_combo.currentIndexChanged.connect(self._render)
            sidebar.addWidget(self.mode_combo)
            self.model_combo = QtWidgets.QComboBox()
            for model in self.models:
                self.model_combo.addItem(model, model)
            self.model_combo.currentIndexChanged.connect(self._render)
            sidebar.addWidget(self.model_combo)

            sidebar.addWidget(QtWidgets.QLabel("<b>Queue</b>"))
            self.filter_combo = QtWidgets.QComboBox()
            for name, label in (
                ("all", "All"),
                ("unreviewed", "Unreviewed"),
                ("reviewed", "Reviewed"),
                ("misses", "Misses"),
                ("extras", "Extras"),
                ("merges", "Merges"),
                ("swaps", "Mouth-tail swaps"),
                ("shared_tails", "Shared/duplicated tails"),
                ("partial", "Partial labels"),
                ("needs_manual_review", "Manifest: needs review"),
            ):
                self.filter_combo.addItem(label, name)
            self.filter_combo.currentIndexChanged.connect(self._refresh_queue)
            sidebar.addWidget(self.filter_combo)
            self.sort_combo = QtWidgets.QComboBox()
            for name, label in (
                ("manifest", "Manifest order"),
                ("frame", "Experiment/view/frame"),
                ("mouth_error_desc", "Largest mouth error"),
                ("tail_error_desc", "Largest tail error"),
            ):
                self.sort_combo.addItem(label, name)
            self.sort_combo.currentIndexChanged.connect(self._refresh_queue)
            sidebar.addWidget(self.sort_combo)
            nav = QtWidgets.QHBoxLayout()
            previous = QtWidgets.QPushButton("Previous [P]")
            following = QtWidgets.QPushButton("Next [N]")
            previous.clicked.connect(lambda: self._go(self.position - 1))
            following.clicked.connect(lambda: self._go(self.position + 1))
            nav.addWidget(previous)
            nav.addWidget(following)
            sidebar.addLayout(nav)
            self.queue_combo = QtWidgets.QComboBox()
            self.queue_combo.activated.connect(self._go)
            sidebar.addWidget(self.queue_combo)
            jump = QtWidgets.QHBoxLayout()
            self.jump_edit = QtWidgets.QLineEdit()
            self.jump_edit.setPlaceholderText("exact frame_key")
            self.jump_edit.returnPressed.connect(self._jump)
            jump_button = QtWidgets.QPushButton("Jump")
            jump_button.clicked.connect(self._jump)
            jump.addWidget(self.jump_edit)
            jump.addWidget(jump_button)
            sidebar.addLayout(jump)
            self.count_label = QtWidgets.QLabel()
            sidebar.addWidget(self.count_label)
            self.info_label = QtWidgets.QLabel()
            self.info_label.setWordWrap(True)
            self.info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.info_label.setStyleSheet(
                "background:#222;color:white;padding:7px"
            )
            sidebar.addWidget(self.info_label)

            sidebar.addWidget(QtWidgets.QLabel("<b>Review decision</b>"))
            self.label_combo = QtWidgets.QComboBox()
            self.label_combo.addItem("Unreviewed", "")
            for label in REVIEW_LABELS:
                self.label_combo.addItem(label.replace("_", " "), label)
            self.label_combo.currentIndexChanged.connect(self._decision_changed)
            sidebar.addWidget(self.label_combo)
            self.notes = QtWidgets.QPlainTextEdit()
            self.notes.setPlaceholderText("Free-text review note")
            self.notes.setMaximumHeight(130)
            self.notes.textChanged.connect(self._decision_changed)
            sidebar.addWidget(self.notes)
            save_button = QtWidgets.QPushButton("Save now [Ctrl+S]")
            save_button.clicked.connect(self._save_decision)
            sidebar.addWidget(save_button)
            export_button = QtWidgets.QPushButton("Export current PNG [E]")
            export_button.clicked.connect(self._export)
            sidebar.addWidget(export_button)
            self.status_label = QtWidgets.QLabel("Ready")
            self.status_label.setWordWrap(True)
            sidebar.addWidget(self.status_label)
            help_text = QtWidgets.QLabel(
                "<b>Shortcuts</b><br>"
                "N/P or Right/Left: next/previous<br>"
                "1–6: set review label; 0: unreviewed<br>"
                "R: Round 7; T: Round 8; C: comparison mode<br>"
                "E: export PNG; Ctrl+S: save<br>"
                "Mouse wheel: zoom; toolbar: pan/home"
            )
            help_text.setWordWrap(True)
            sidebar.addWidget(help_text)
            sidebar.addStretch(1)

        def _shortcuts(self) -> None:
            bindings = {
                "N": lambda: self._go(self.position + 1),
                "Right": lambda: self._go(self.position + 1),
                "P": lambda: self._go(self.position - 1),
                "Left": lambda: self._go(self.position - 1),
                "Ctrl+S": self._save_decision,
                "E": self._export,
                "C": self._toggle_mode,
                "R": lambda: self._select_model("round7"),
                "T": lambda: self._select_model("round8"),
                "0": lambda: self._select_label(""),
            }
            for index, label in enumerate(REVIEW_LABELS, start=1):
                bindings[str(index)] = lambda value=label: self._select_label(value)
            self.shortcuts = []
            for sequence, callback in bindings.items():
                shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
                shortcut.activated.connect(callback)
                self.shortcuts.append(shortcut)

        def _decision_changed(self, *_args) -> None:
            if not self.loading_decision:
                self.autosave_timer.start()

        def _select_label(self, label: str) -> None:
            position = self.label_combo.findData(label)
            if position >= 0:
                self.label_combo.setCurrentIndex(position)

        def _select_model(self, model: str) -> None:
            position = self.model_combo.findData(model)
            if position >= 0:
                self.model_combo.setCurrentIndex(position)

        def _toggle_mode(self) -> None:
            self.mode_combo.setCurrentIndex(1 - self.mode_combo.currentIndex())

        def _refresh_queue(self, *_args) -> None:
            current_key = (
                str(self.manifest.iloc[self.current_index]["frame_key"])
                if self.current_index is not None
                else ""
            )
            self.filtered_indices = filter_sort_rows(
                self.manifest,
                self.evidence,
                self.store.decisions,
                filter_name=str(self.filter_combo.currentData() or "all"),
                sort_name=str(self.sort_combo.currentData() or "manifest"),
            )
            self.queue_combo.blockSignals(True)
            self.queue_combo.clear()
            selected = -1
            for queue_position, index in enumerate(self.filtered_indices):
                row = self.manifest.iloc[index]
                key = str(row["frame_key"])
                mark = "✓" if self.store.get(key) and self.store.get(key).review_label else "·"
                self.queue_combo.addItem(
                    f"{mark} {row['experiment']} | {row['view']} | {int(row['frame_idx']):,}",
                    index,
                )
                if key == current_key:
                    selected = queue_position
            self.queue_combo.blockSignals(False)
            self._update_counts()
            if selected >= 0:
                self.position = selected
                self.queue_combo.setCurrentIndex(selected)
            elif self.filtered_indices:
                self._go(0)
            else:
                self.position = -1
                self.current_index = None
                self.figure.clear()
                self.canvas.draw_idle()
                self.info_label.setText("No frames match the selected filter.")

        def _update_counts(self) -> None:
            all_keys = self.manifest["frame_key"].astype(str).tolist()
            reviewed = self.store.reviewed_count(all_keys)
            self.count_label.setText(
                f"Reviewed {reviewed}/{len(all_keys)} | "
                f"unreviewed {len(all_keys) - reviewed} | "
                f"showing {len(self.filtered_indices)}"
            )

        def _go(self, queue_position: int) -> None:
            if not self.filtered_indices:
                return
            if self.current_index is not None:
                self._save_decision()
            self.position = int(np.clip(queue_position, 0, len(self.filtered_indices) - 1))
            self.current_index = self.filtered_indices[self.position]
            self.queue_combo.setCurrentIndex(self.position)
            row = self.manifest.iloc[self.current_index]
            frame_key = str(row["frame_key"])
            self.loading_decision = True
            decision = self.store.get(frame_key)
            self.label_combo.setCurrentIndex(
                self.label_combo.findData(decision.review_label if decision else "")
            )
            self.notes.setPlainText(decision.notes if decision else "")
            self.loading_decision = False
            try:
                self.current_image = self.reader.get(row)
                self.status_label.setText("Loaded exact source frame.")
            except Exception as error:
                self.current_image = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    self.current_image,
                    str(error)[:140],
                    (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 80, 80),
                    2,
                    cv2.LINE_AA,
                )
                self.status_label.setText(f"Frame load failed: {error}")
            self._render()

        def _jump(self) -> None:
            wanted = self.jump_edit.text().strip()
            matches = self.manifest.index[
                self.manifest["frame_key"].astype(str).eq(wanted)
            ].tolist()
            if not matches:
                self.status_label.setText(f"Unknown frame_key: {wanted}")
                return
            target = matches[0]
            if target not in self.filtered_indices:
                self.filter_combo.setCurrentIndex(
                    self.filter_combo.findData("all")
                )
            if target in self.filtered_indices:
                self._go(self.filtered_indices.index(target))

        def _render(self, *_args) -> None:
            if self.current_index is None or self.current_image is None:
                return
            row = self.manifest.iloc[self.current_index]
            frame_key = str(row["frame_key"])
            frame_evidence = self.evidence.get(frame_key, FrameEvidence([], {}, {}))
            mode = str(self.mode_combo.currentData())
            if mode == "side_by_side":
                shown_models = [
                    name for name in ("round7", "round8") if name in self.models
                ]
                if not shown_models:
                    shown_models = self.models[:2]
            else:
                shown_models = [str(self.model_combo.currentData())]
            old_limits = []
            for axis in self.figure.axes:
                old_limits.append((axis.get_xlim(), axis.get_ylim()))
            self.figure.clear()
            axes = self.figure.subplots(1, len(shown_models), squeeze=False)[0]
            for axis, model in zip(axes, shown_models, strict=True):
                rendered = draw_pose_overlay(
                    self.current_image,
                    gt_instances=frame_evidence.gt,
                    predictions=frame_evidence.predictions.get(model, []),
                    model=model,
                )
                axis.imshow(rendered)
                axis.set_title(
                    model_metric_text(
                        model, frame_evidence.model_metrics.get(model, {})
                    ),
                    fontsize=9,
                )
                axis.set_axis_off()
            legend = [
                Line2D([0], [0], color=np.asarray(GT_COLOR_RGB) / 255, lw=3, label="GT"),
                *[
                    Line2D(
                        [0],
                        [0],
                        color=np.asarray(model_color(model)) / 255,
                        lw=3,
                        label=model,
                    )
                    for model in shown_models
                ],
                Line2D(
                    [0], [0], marker="o", color="white", markerfacecolor="gray",
                    lw=0, label="mouth (circle)"
                ),
                Line2D(
                    [0], [0], marker="^", color="white", markerfacecolor="none",
                    lw=0, label="tail (triangle)"
                ),
            ]
            self.figure.legend(handles=legend, loc="lower center", ncol=len(legend))
            self.figure.suptitle(metadata_text(row, frame_evidence), fontsize=9)
            if len(old_limits) == len(axes):
                for axis, limits in zip(axes, old_limits, strict=True):
                    axis.set_xlim(limits[0])
                    axis.set_ylim(limits[1])
            self.canvas.draw_idle()
            self.info_label.setText(metadata_text(row, frame_evidence))

        def _scroll_zoom(self, event) -> None:
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            scale = 0.8 if event.button == "up" else 1.25
            axis = event.inaxes
            xlim, ylim = axis.get_xlim(), axis.get_ylim()
            width, height = (xlim[1] - xlim[0]) * scale, (ylim[1] - ylim[0]) * scale
            rx = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
            ry = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])
            axis.set_xlim(event.xdata - width * rx, event.xdata + width * (1 - rx))
            axis.set_ylim(event.ydata - height * ry, event.ydata + height * (1 - ry))
            self.canvas.draw_idle()

        def _save_decision(self) -> None:
            if self.current_index is None:
                return
            self.autosave_timer.stop()
            row = self.manifest.iloc[self.current_index]
            decision = ReviewDecision(
                frame_key=str(row["frame_key"]),
                experiment=str(row["experiment"]),
                view=str(row["view"]),
                frame_idx=int(row["frame_idx"]),
                review_label=str(self.label_combo.currentData() or ""),
                notes=self.notes.toPlainText().strip(),
                annotator=str(args.annotator),
                updated_at_utc=utc_now(),
            )
            if not decision.review_label and not decision.notes:
                self.store.remove(decision.frame_key)
            else:
                self.store.set(decision)
            self._update_counts()
            self.status_label.setText(
                f"Autosaved review to {self.store.csv_path.name} and "
                f"{self.store.json_path.name}."
            )

        def _export(self) -> None:
            if self.current_index is None:
                return
            row = self.manifest.iloc[self.current_index]
            safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["frame_key"]))
            output = Path(args.export_dir) / f"{safe_key}_overlay.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(output, dpi=160, bbox_inches="tight")
            self.status_label.setText(f"Exported {output}")

        def closeEvent(self, event) -> None:
            self._save_decision()
            event.accept()

    return QcWindow()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Cached long-table or native .slp prediction artifact; repeatable.",
    )
    parser.add_argument(
        "--ground-truth",
        action="append",
        default=[],
        type=Path,
        help=(
            "Ground-truth .slp or cached table containing GT coordinates; "
            "repeatable for separate views/packages."
        ),
    )
    parser.add_argument(
        "--no-manifest-ground-truth",
        action="store_true",
        help=(
            "Do not auto-load unique source_slp paths from the manifest when "
            "cached match rows omit GT coordinates."
        ),
    )
    parser.add_argument(
        "--round7-predictions",
        action="append",
        default=[],
        type=Path,
        help="Convenience alias for --predictions round7=PATH; repeatable by view.",
    )
    parser.add_argument(
        "--round8-predictions",
        action="append",
        default=[],
        type=Path,
        help="Convenience alias for --predictions round8=PATH; repeatable by view.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("report/20260724/sleap_topdown_optimization/manual_visual_qc.csv"),
    )
    parser.add_argument("--review-json", type=Path)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("report/20260724/sleap_topdown_optimization/visual_qc_exports"),
    )
    parser.add_argument(
        "--export-frame-key",
        help="Render this exact frame_key to PNG without opening Qt, then exit.",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        help="Output PNG for --export-frame-key (default: --export-dir/<key>_overlay.png).",
    )
    parser.add_argument("--annotator", default=os.environ.get("USERNAME", ""))
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate inputs and print coverage without opening a window.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    evidence: dict[str, FrameEvidence] = {}
    specs = [
        *args.predictions,
        *[f"round7={path}" for path in args.round7_predictions],
        *[f"round8={path}" for path in args.round8_predictions],
    ]
    provenance: dict[str, Any] = {
        "manifest": {
            "path": str(args.manifest.resolve(strict=False)),
            "sha256": sha256_file(args.manifest),
        },
        "predictions": [],
        "ground_truth": [],
    }
    for spec in specs:
        model, path, loaded = load_prediction_spec(spec, manifest=manifest)
        merge_evidence(evidence, loaded)
        provenance["predictions"].append(
            {
                "model": model,
                "path": str(path.resolve(strict=False)),
                "sha256": sha256_file(path),
                "covered_frames": len(loaded),
            }
        )
    ground_truth_paths = list(args.ground_truth)
    has_cached_gt = any(item.gt for item in evidence.values())
    if (
        not ground_truth_paths
        and not has_cached_gt
        and not args.no_manifest_ground_truth
        and "source_slp" in manifest.columns
    ):
        ground_truth_paths = [
            Path(value)
            for value in dict.fromkeys(
                _text(value)
                for value in manifest["source_slp"]
                if not _missing(value)
            )
        ]
    for path in ground_truth_paths:
        loaded = load_ground_truth_spec(path, manifest=manifest)
        merge_evidence(evidence, loaded)
        provenance["ground_truth"].append(
            {
                "path": str(path.resolve(strict=False)),
                "sha256": sha256_file(path),
                "covered_frames": sum(bool(item.gt) for item in loaded.values()),
            }
        )
    store = ReviewStore(
        args.review_csv,
        json_path=args.review_json,
        source_provenance=provenance,
    )
    models = sorted(
        {
            model
            for frame_evidence in evidence.values()
            for model in frame_evidence.predictions
        }
    )
    coverage = {
        model: sum(
            bool(frame_evidence.predictions.get(model))
            for frame_evidence in evidence.values()
        )
        for model in models
    }
    print(f"Manifest frames: {len(manifest):,}")
    print(f"Cached evidence frames: {len(evidence):,}")
    print(f"Prediction coverage: {coverage}")
    print(
        f"Review resume: {store.reviewed_count(manifest['frame_key']):,}/"
        f"{len(manifest):,}"
    )
    print(f"Review outputs: {store.csv_path} | {store.json_path}")
    if args.preflight:
        if not store.csv_path.exists() or not store.json_path.exists():
            store.save()
            print("Materialized empty review CSV/JSON with input provenance.")
        return 0
    if args.export_frame_key:
        if not specs:
            raise SystemExit("Headless export requires cached --predictions artifacts")
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.export_frame_key)
        output = args.export_path or args.export_dir / f"{safe_key}_overlay.png"
        export_frame_overlay(
            manifest,
            evidence,
            frame_key=args.export_frame_key,
            output_path=output,
        )
        print(f"Exported headless overlay: {output}")
        return 0
    if not specs:
        raise SystemExit("At least one cached --predictions artifact is required")
    _, _, QtWidgets, _, _, _, _ = _load_gui_dependencies()
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = create_qc_window(
        args, manifest=manifest, evidence=evidence, store=store
    )
    window.show()
    return int(application.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
