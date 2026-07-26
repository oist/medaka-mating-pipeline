"""Merge and freeze a model-assisted HITL benchmark package.

The base package remains authoritative except where an exact
``(experiment, frame_idx, view)`` key is present in the replacement package.
Only the 40 source frames named by the selection manifest are retained.

Outputs are new, versioned artifacts. Inputs and existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_FILES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
SOURCE_FRAME_COUNT = 40
VIEW_FRAME_COUNT = SOURCE_FRAME_COUNT * len(VIEWS)
FISH_PER_VIEW = 3
SCHEMA_VERSION = "hitl-benchmark-final-v1"
EVALUATION_MANIFEST_VERSION = "hitl-benchmark-evaluation-v1"

INSTANCE_LEDGER_COLUMNS = (
    "selection_frame_key",
    "source_frame_order",
    "source_kind",
    "experiment",
    "frame_idx",
    "view",
    "fish_index",
    "instance_key",
    "mouth_x",
    "mouth_y",
    "mouth_finite",
    "mouth_visible",
    "tail_x",
    "tail_y",
    "tail_finite",
    "tail_visible",
    "fully_labeled",
    "partial_annotation",
)

EVALUATION_COLUMNS = (
    "manifest_version",
    "selection_seed",
    "selection_rule",
    "frame_key",
    "experiment",
    "view",
    "camera",
    "video_path",
    "frame_idx",
    "source_slp",
    "source_slp_available",
    "source_slp_size_bytes",
    "source_slp_mtime_ns",
    "source_video_index",
    "source_labeled_frame_index",
    "source_provenance",
    "cohort",
    "split_role",
    "provenance_class",
    "event_id",
    "event_start_frame",
    "event_end_frame",
    "difficulty",
    "source_category",
    "node_visibility",
    "instance_count",
    "all_three_mouths_finite",
    "strict_centroid_eligible",
    "needs_manual_review",
    "annotation_present",
    "round7_event_exposed",
    "round8_finetune_member",
    "selection_reason",
    "fully_labeled",
    "partial_annotation",
    "selection_frame_key",
    "annotation_source",
)


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable serialization."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def original_video_path(video: Any) -> str:
    """Return the deepest real-video path from an embedded SLP chain."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    current = video
    seen: set[int] = set()
    fallback = ""
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        backend = getattr(current, "backend", None)
        values = (
            getattr(backend, "source_filename", None) if backend else None,
            getattr(current, "filename", None),
        )
        for value in values:
            if not value:
                continue
            fallback = fallback or str(value)
            if str(value).lower().endswith(video_extensions):
                return str(value)
        current = getattr(current, "source_video", None)
    return fallback


def experiment_from_path(path: str, experiments: Iterable[str]) -> str | None:
    """Find a manifest experiment name in a normalized video path."""
    normalized = "/" + str(path).replace("\\", "/").strip("/") + "/"
    for experiment in experiments:
        if f"/{experiment}/" in normalized:
            return experiment
    return None


def view_from_path(path: str) -> str | None:
    """Return the canonical view encoded in a video path."""
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    for view, filename in VIEW_FILES.items():
        if f"/{view}/" in normalized or normalized.endswith(f"/{filename}/"):
            return view
    if "side1" in normalized:
        return "side1_cam"
    if "side2" in normalized:
        return "side2_cam"
    if "top_cam" in normalized or "cam0_top" in normalized:
        return "top_cam"
    return None


def load_selection_manifest(path: Path) -> list[dict[str, str]]:
    """Load and validate the exact 40-row source-frame selection."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    required = {"experiment", "frame_idx", "frame_key"}
    if not rows:
        raise ValueError("selection manifest is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            "selection manifest is missing columns: "
            + ", ".join(sorted(missing))
        )
    if len(rows) != SOURCE_FRAME_COUNT:
        raise ValueError(
            f"selection manifest must contain {SOURCE_FRAME_COUNT} rows; "
            f"found {len(rows)}"
        )
    source_keys = [
        (str(row["experiment"]), int(row["frame_idx"])) for row in rows
    ]
    if len(set(source_keys)) != SOURCE_FRAME_COUNT:
        raise ValueError(
            "selection manifest must contain 40 unique "
            "(experiment, frame_idx) rows"
        )
    frame_keys = [str(row["frame_key"]) for row in rows]
    if len(set(frame_keys)) != SOURCE_FRAME_COUNT:
        raise ValueError("selection manifest contains duplicate frame_key values")
    return rows


def _labels_provenance(labels: Any) -> dict[str, Any]:
    return deepcopy(dict(getattr(labels, "provenance", {}) or {}))


def _frame_key(
    labeled_frame: Any,
    experiments: tuple[str, ...],
) -> tuple[str, int, str]:
    source_path = original_video_path(labeled_frame.video)
    experiment = experiment_from_path(source_path, experiments)
    view = view_from_path(source_path)
    if experiment is None or view is None:
        raise ValueError(
            "could not identify experiment/view from labeled-frame video: "
            f"{source_path!r}"
        )
    return experiment, int(labeled_frame.frame_idx), view


def index_labeled_frames(
    labels: Any,
    experiments: tuple[str, ...],
    *,
    source_name: str,
) -> dict[tuple[str, int, str], Any]:
    """Index all labeled frames, rejecting duplicate semantic keys."""
    indexed: dict[tuple[str, int, str], Any] = {}
    for labeled_frame in labels.labeled_frames:
        key = _frame_key(labeled_frame, experiments)
        if key in indexed:
            raise ValueError(
                f"{source_name} contains duplicate labeled-frame key: {key}"
            )
        indexed[key] = labeled_frame
    return indexed


def _skeleton_signature(skeleton: Any) -> tuple[Any, ...]:
    names = tuple(str(name) for name in skeleton.node_names)
    edges = tuple(
        (names[int(source)], names[int(destination)])
        for source, destination in skeleton.edge_inds
    )
    return names, edges


def _validate_mouth_tail_skeleton(skeleton: Any) -> None:
    signature = _skeleton_signature(skeleton)
    expected = (("mouth", "tail"), (("mouth", "tail"),))
    if signature != expected:
        raise ValueError(
            "selected instances must use the mouth->tail skeleton; "
            f"found {signature}"
        )


def _point_state(instance: Any, node_name: str) -> dict[str, Any]:
    names = tuple(str(name) for name in instance.skeleton.node_names)
    node_index = names.index(node_name)
    xy = np.array([np.nan, np.nan], dtype=float)
    visible = False
    complete = False
    try:
        point = instance.points[node_index]
        fields = getattr(getattr(point, "dtype", None), "names", None) or ()
        if "xy" in fields:
            xy = np.asarray(point["xy"], dtype=float)
        if "visible" in fields:
            visible = bool(point["visible"])
        if "complete" in fields:
            complete = bool(point["complete"])
    except (IndexError, KeyError, TypeError):
        coordinates = np.asarray(instance.numpy(), dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] <= node_index:
            raise ValueError(f"invalid point array for node {node_name}")
        xy = np.asarray(coordinates[node_index, :2], dtype=float)
        visible = bool(np.isfinite(xy).all())
    finite = bool(xy.shape == (2,) and np.isfinite(xy).all())
    return {
        "xy": xy if finite else np.array([np.nan, np.nan], dtype=float),
        "finite": finite,
        "visible": bool(visible) if finite else False,
        "complete": bool(complete),
    }


def _rebuild_user_instance(
    instance: Any,
    canonical_skeleton: Any,
    *,
    sio: Any,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    if isinstance(instance, sio.PredictedInstance) or (
        type(instance).__name__ == "PredictedInstance"
    ):
        raise ValueError("PredictedInstance found in selected HITL labels")
    if not isinstance(instance, sio.Instance) and (
        type(instance).__name__ != "Instance"
    ):
        raise ValueError(
            f"unsupported instance type in selected HITL labels: "
            f"{type(instance).__name__}"
        )
    _validate_mouth_tail_skeleton(instance.skeleton)
    states = {
        node: _point_state(instance, node) for node in ("mouth", "tail")
    }
    if not any(state["finite"] for state in states.values()):
        raise ValueError(
            "each selected fish instance must contain at least one finite node"
        )
    points = np.array(
        [
            [
                states[node]["xy"][0],
                states[node]["xy"][1],
                states[node]["visible"],
                states[node]["complete"],
            ]
            for node in ("mouth", "tail")
        ],
        dtype=float,
    )
    rebuilt = sio.Instance.from_numpy(
        points,
        skeleton=canonical_skeleton,
        track=None,
        tracking_score=None,
        from_predicted=None,
    )
    return rebuilt, states


def _row_node_summary(
    states_by_instance: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    visibility_parts: list[str] = []
    fully_labeled = True
    for fish_index, states in enumerate(states_by_instance):
        node_parts: list[str] = []
        for node in ("mouth", "tail"):
            state = states[node]
            if not state["finite"]:
                status = "missing"
            elif state["visible"]:
                status = "visible"
            else:
                status = "hidden"
            node_parts.append(f"{node}:{status}")
            fully_labeled &= status == "visible"
        visibility_parts.append(
            f"fish{fish_index + 1}[{','.join(node_parts)}]"
        )
    # SLEAP's public ``Instance.numpy()`` masks hidden points as NaN. Match
    # benchmark_manifest/evaluation semantics here while retaining the hidden
    # point's finite coordinate in the locked SLP and instance ledger.
    all_three_mouths_finite = (
        len(states_by_instance) == FISH_PER_VIEW
        and all(
            states["mouth"]["finite"] and states["mouth"]["visible"]
            for states in states_by_instance
        )
    )
    return {
        "node_visibility": ";".join(visibility_parts),
        "fully_labeled": bool(fully_labeled),
        "partial_annotation": not bool(fully_labeled),
        "all_three_mouths_finite": bool(all_three_mouths_finite),
        "strict_centroid_eligible": bool(all_three_mouths_finite),
    }


def merge_hitl_labels(
    base_labels: Any,
    replacement_labels: Any,
    manifest_rows: list[dict[str, str]],
    *,
    output_provenance: Mapping[str, Any],
    sio: Any | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Merge selected frames and normalize skeletons, tracks, and instances."""
    if sio is None:
        import sleap_io as sio

    experiments = tuple(
        dict.fromkeys(str(row["experiment"]) for row in manifest_rows)
    )
    base_index = index_labeled_frames(
        base_labels, experiments, source_name="base HITL package"
    )
    replacement_index = index_labeled_frames(
        replacement_labels,
        experiments,
        source_name="replacement annotation package",
    )
    expected_keys = {
        (str(row["experiment"]), int(row["frame_idx"]), view)
        for row in manifest_rows
        for view in VIEWS
    }
    missing = sorted(
        key
        for key in expected_keys
        if key not in base_index and key not in replacement_index
    )
    if missing:
        raise ValueError(
            "selected HITL view frames are missing from both inputs: "
            f"{missing[:5]} ({len(missing)} total)"
        )

    selected_sources: list[tuple[Any, str, dict[str, str], str]] = []
    for row in manifest_rows:
        experiment = str(row["experiment"])
        frame_idx = int(row["frame_idx"])
        for view in VIEWS:
            key = (experiment, frame_idx, view)
            if key in replacement_index:
                selected_sources.append(
                    (replacement_index[key], "replacement", row, view)
                )
            else:
                selected_sources.append((base_index[key], "base", row, view))

    skeletons = {
        id(instance.skeleton): instance.skeleton
        for labeled_frame, _kind, _row, _view in selected_sources
        for instance in labeled_frame.instances
    }
    for skeleton in skeletons.values():
        _validate_mouth_tail_skeleton(skeleton)
    canonical_name = (
        str(getattr(next(iter(skeletons.values())), "name", "") or "")
        if skeletons
        else "medaka_mouth_tail"
    )
    canonical_skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
        name=canonical_name or "medaka_mouth_tail",
    )

    labeled_frames: list[Any] = []
    suggestions: list[Any] = []
    ledger_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    videos_by_id: dict[int, Any] = {}
    replacement_count = 0
    finite_mouths = 0
    finite_tails = 0
    visible_mouths = 0
    visible_tails = 0
    fully_labeled_instances = 0

    for position, (source_lf, source_kind, row, view) in enumerate(
        selected_sources
    ):
        instances = list(source_lf.instances)
        if len(instances) != FISH_PER_VIEW:
            raise ValueError(
                "final HITL package requires exactly "
                f"{FISH_PER_VIEW} instances per view; "
                f"{row['experiment']} frame {int(row['frame_idx'])} "
                f"{view} has {len(instances)}"
            )
        rebuilt_instances: list[Any] = []
        states_by_instance: list[dict[str, dict[str, Any]]] = []
        for fish_index, instance in enumerate(instances):
            rebuilt, states = _rebuild_user_instance(
                instance, canonical_skeleton, sio=sio
            )
            rebuilt_instances.append(rebuilt)
            states_by_instance.append(states)
            finite_mouths += int(states["mouth"]["finite"])
            finite_tails += int(states["tail"]["finite"])
            visible_mouths += int(
                states["mouth"]["finite"] and states["mouth"]["visible"]
            )
            visible_tails += int(
                states["tail"]["finite"] and states["tail"]["visible"]
            )
            instance_full = all(
                states[node]["finite"] and states[node]["visible"]
                for node in ("mouth", "tail")
            )
            fully_labeled_instances += int(instance_full)
            ledger_rows.append(
                {
                    "selection_frame_key": str(row["frame_key"]),
                    "source_frame_order": position // len(VIEWS),
                    "source_kind": source_kind,
                    "experiment": str(row["experiment"]),
                    "frame_idx": int(row["frame_idx"]),
                    "view": view,
                    "fish_index": fish_index,
                    "instance_key": (
                        f"{row['experiment']}|{int(row['frame_idx'])}|"
                        f"{view}|fish_index={fish_index}"
                    ),
                    "mouth_x": (
                        float(states["mouth"]["xy"][0])
                        if states["mouth"]["finite"]
                        else ""
                    ),
                    "mouth_y": (
                        float(states["mouth"]["xy"][1])
                        if states["mouth"]["finite"]
                        else ""
                    ),
                    "mouth_finite": states["mouth"]["finite"],
                    "mouth_visible": states["mouth"]["visible"],
                    "tail_x": (
                        float(states["tail"]["xy"][0])
                        if states["tail"]["finite"]
                        else ""
                    ),
                    "tail_y": (
                        float(states["tail"]["xy"][1])
                        if states["tail"]["finite"]
                        else ""
                    ),
                    "tail_finite": states["tail"]["finite"],
                    "tail_visible": states["tail"]["visible"],
                    "fully_labeled": instance_full,
                    "partial_annotation": not instance_full,
                }
            )

        rebuilt_lf = sio.LabeledFrame(
            video=source_lf.video,
            frame_idx=int(source_lf.frame_idx),
            instances=rebuilt_instances,
        )
        labeled_frames.append(rebuilt_lf)
        suggestions.append(
            sio.SuggestionFrame(
                video=source_lf.video,
                frame_idx=int(source_lf.frame_idx),
            )
        )
        videos_by_id[id(source_lf.video)] = source_lf.video
        replacement_count += int(source_kind == "replacement")
        frame_summaries.append(
            {
                "source_labeled_frame_index": position,
                "source_kind": source_kind,
                "selection": dict(row),
                "experiment": str(row["experiment"]),
                "frame_idx": int(row["frame_idx"]),
                "view": view,
                "video_path": original_video_path(source_lf.video),
                **_row_node_summary(states_by_instance),
            }
        )

    merged = sio.Labels(
        labeled_frames=labeled_frames,
        videos=list(videos_by_id.values()),
        skeletons=[canonical_skeleton],
        tracks=[],
        suggestions=suggestions,
        provenance=deepcopy(dict(output_provenance)),
    )
    expected_instances = VIEW_FRAME_COUNT * FISH_PER_VIEW
    counts = {
        "source_frames": SOURCE_FRAME_COUNT,
        "view_frames": VIEW_FRAME_COUNT,
        "fish_per_view": FISH_PER_VIEW,
        "instances": expected_instances,
        "replacement_view_frames": replacement_count,
        "base_view_frames": VIEW_FRAME_COUNT - replacement_count,
        "base_view_frames_dropped": len(set(base_index) - expected_keys),
        "replacement_view_frames_dropped": len(
            set(replacement_index) - expected_keys
        ),
        "finite_mouths": finite_mouths,
        "finite_tails": finite_tails,
        "visible_mouths": visible_mouths,
        "visible_tails": visible_tails,
        "fully_labeled_instances": fully_labeled_instances,
        "partial_instances": expected_instances - fully_labeled_instances,
        "fully_labeled_view_frames": sum(
            not summary["partial_annotation"] for summary in frame_summaries
        ),
        "partial_view_frames": sum(
            summary["partial_annotation"] for summary in frame_summaries
        ),
    }
    return merged, ledger_rows, {
        "counts": counts,
        "frame_summaries": frame_summaries,
    }


def build_evaluation_manifest_rows(
    labels: Any,
    frame_summaries: list[dict[str, Any]],
    *,
    locked_output_path: Path,
    package_size_bytes: int,
    package_mtime_ns: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build a compare_rounds/eval_topdown-compatible 120-row manifest."""
    if len(labels.labeled_frames) != VIEW_FRAME_COUNT:
        raise ValueError("evaluation manifest requires exactly 120 labeled frames")
    if len(frame_summaries) != VIEW_FRAME_COUNT:
        raise ValueError("evaluation frame summary count mismatch")

    provenance_json = json.dumps(
        _labels_provenance(labels),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    video_positions = {id(video): index for index, video in enumerate(labels.videos)}
    selection_columns: list[str] = []
    for summary in frame_summaries:
        for column in summary["selection"]:
            candidate = f"selection_{column}"
            if candidate not in selection_columns:
                selection_columns.append(candidate)
    columns = list(EVALUATION_COLUMNS)
    for column in selection_columns:
        if column not in columns:
            columns.append(column)

    rows: list[dict[str, Any]] = []
    for position, (labeled_frame, summary) in enumerate(
        zip(labels.labeled_frames, frame_summaries)
    ):
        selection = summary["selection"]
        source_category = str(selection.get("source_category", ""))
        source_difficulty = str(
            selection.get("difficulty", "")
            or selection.get("difficulty_hint", "")
        ).strip()
        if summary["partial_annotation"]:
            difficulty = "partial_or_occluded"
        elif source_difficulty and source_difficulty != "unknown_until_annotation":
            difficulty = source_difficulty
        elif source_category.casefold().startswith("close_"):
            difficulty = "close_contact_or_crossing"
        else:
            difficulty = "ordinary"
        selection_frame_key = str(selection["frame_key"])
        row: dict[str, Any] = {
            "manifest_version": EVALUATION_MANIFEST_VERSION,
            "selection_seed": selection.get("selection_seed", ""),
            "selection_rule": selection.get(
                "selection_rule", selection.get("selection_source", "")
            ),
            "frame_key": f"{selection_frame_key}|view={summary['view']}",
            "experiment": summary["experiment"],
            "view": summary["view"],
            "camera": summary["view"],
            "video_path": summary["video_path"],
            "frame_idx": summary["frame_idx"],
            "source_slp": str(Path(locked_output_path).resolve()),
            "source_slp_available": True,
            "source_slp_size_bytes": int(package_size_bytes),
            "source_slp_mtime_ns": int(package_mtime_ns),
            "source_video_index": int(
                video_positions[id(labeled_frame.video)]
            ),
            "source_labeled_frame_index": position,
            "source_provenance": provenance_json,
            "cohort": "hitl_benchmark_40",
            "split_role": "locked_test",
            "provenance_class": "model_assisted_human_review",
            "event_id": selection.get("event_id", ""),
            "event_start_frame": selection.get("event_start_frame", ""),
            "event_end_frame": selection.get("event_end_frame", ""),
            "difficulty": difficulty,
            "source_category": source_category,
            "node_visibility": summary["node_visibility"],
            "instance_count": FISH_PER_VIEW,
            "all_three_mouths_finite": summary[
                "all_three_mouths_finite"
            ],
            "strict_centroid_eligible": summary[
                "strict_centroid_eligible"
            ],
            "needs_manual_review": summary["partial_annotation"],
            "annotation_present": True,
            "round7_event_exposed": False,
            "round8_finetune_member": False,
            "selection_reason": selection.get("selection_source", ""),
            "fully_labeled": summary["fully_labeled"],
            "partial_annotation": summary["partial_annotation"],
            "selection_frame_key": selection_frame_key,
            "annotation_source": "human_reviewed_round8_model_assisted",
        }
        row.update(
            {f"selection_{key}": value for key, value in selection.items()}
        )
        rows.append(row)
    if len({row["source_labeled_frame_index"] for row in rows}) != len(rows):
        raise RuntimeError("evaluation manifest contains duplicate LF positions")
    if len({row["frame_key"] for row in rows}) != len(rows):
        raise RuntimeError("evaluation manifest contains duplicate frame_key values")
    return columns, rows


def _write_csv(
    path: Path,
    columns: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _new_temp_path(output_dir: Path, name: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _validate_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(
            "prefix must start with an alphanumeric character and contain "
            "only letters, numbers, dot, underscore, or hyphen"
        )
    return value


def finalize_hitl_benchmark(
    *,
    base_slp: Path,
    replacement_slp: Path,
    manifest_path: Path,
    output_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    """Merge, normalize, validate, and atomically publish locked artifacts."""
    base_slp = Path(base_slp)
    replacement_slp = Path(replacement_slp)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    prefix = _validate_prefix(prefix)
    for name, path in (
        ("base HITL SLP", base_slp),
        ("replacement annotated SLP", replacement_slp),
        ("selection manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if base_slp.resolve() == replacement_slp.resolve():
        raise ValueError("base and replacement SLP inputs must be distinct")

    output_dir.mkdir(parents=True, exist_ok=True)
    locked_path = output_dir / f"{prefix}_locked.pkg.slp"
    ledger_path = output_dir / f"{prefix}_instance_ledger.csv"
    evaluation_path = output_dir / f"{prefix}_evaluation_manifest.csv"
    provenance_path = output_dir / f"{prefix}_provenance.json"
    outputs = (locked_path, ledger_path, evaluation_path, provenance_path)
    input_paths = {
        base_slp.resolve(),
        replacement_slp.resolve(),
        manifest_path.resolve(),
    }
    if any(path.resolve() in input_paths for path in outputs):
        raise ValueError("output artifacts must be distinct from every input")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing HITL finalization artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    import sleap_io as sio

    manifest_rows = load_selection_manifest(manifest_path)
    input_hashes_before = {
        "base_slp": sha256_file(base_slp),
        "replacement_slp": sha256_file(replacement_slp),
        "manifest": sha256_file(manifest_path),
    }
    base_labels = sio.load_slp(str(base_slp), open_videos=False)
    replacement_labels = sio.load_slp(str(replacement_slp), open_videos=False)
    base_provenance = _labels_provenance(base_labels)
    replacement_provenance = _labels_provenance(replacement_labels)
    candidate_review = replacement_provenance.get(
        "prediction_candidate_review",
        base_provenance.get("prediction_candidate_review", {}),
    )
    package_provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked_model_assisted_annotations",
        "model_assisted_annotation": True,
        "contains_model_predictions": False,
        "contains_keypoint_coordinates": True,
        "inputs": {
            "base_slp": {
                "path": str(base_slp.resolve()),
                "sha256": input_hashes_before["base_slp"],
                "provenance_sha256": canonical_json_sha256(base_provenance),
            },
            "replacement_slp": {
                "path": str(replacement_slp.resolve()),
                "sha256": input_hashes_before["replacement_slp"],
                "provenance_sha256": canonical_json_sha256(
                    replacement_provenance
                ),
            },
            "selection_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": input_hashes_before["manifest"],
            },
        },
        "candidate_review_provenance": {
            "present": bool(candidate_review),
            "sha256": canonical_json_sha256(candidate_review),
            "candidate_artifact": deepcopy(
                candidate_review.get("candidate_artifact", {})
                if isinstance(candidate_review, dict)
                else {}
            ),
        },
        "normalization": {
            "replacement_precedence": "exact_experiment_frame_view_key",
            "manifest_only": True,
            "canonical_skeleton_count": 1,
            "instance_type": "Instance",
            "tracks_stripped": True,
            "predicted_instances_rejected": True,
            "partial_or_invisible_nodes_allowed": True,
            "minimum_finite_nodes_per_instance": 1,
        },
    }
    merged, ledger_rows, merge_report = merge_hitl_labels(
        base_labels,
        replacement_labels,
        manifest_rows,
        output_provenance=package_provenance,
        sio=sio,
    )
    package_provenance["counts"] = deepcopy(merge_report["counts"])
    merged.provenance = deepcopy(package_provenance)

    temporary_paths: list[Path] = []
    locked_tmp = _new_temp_path(output_dir, locked_path.name, ".tmp.slp")
    ledger_tmp = _new_temp_path(output_dir, ledger_path.name, ".tmp.csv")
    evaluation_tmp = _new_temp_path(
        output_dir, evaluation_path.name, ".tmp.csv"
    )
    provenance_tmp = _new_temp_path(
        output_dir, provenance_path.name, ".tmp.json"
    )
    temporary_paths.extend(
        (locked_tmp, ledger_tmp, evaluation_tmp, provenance_tmp)
    )
    try:
        sio.save_slp(
            merged,
            str(locked_tmp),
            embed=True,
            embed_inplace=False,
            verbose=False,
        )
        reloaded = sio.load_slp(str(locked_tmp), open_videos=False)
        if len(reloaded.labeled_frames) != VIEW_FRAME_COUNT:
            raise RuntimeError("locked SLP reload did not retain 120 view frames")
        if len(reloaded.skeletons) != 1:
            raise RuntimeError("locked SLP reload did not retain one skeleton")
        if getattr(reloaded, "tracks", []):
            raise RuntimeError("locked SLP reload unexpectedly contains tracks")
        reloaded_types = Counter(
            type(instance).__name__
            for labeled_frame in reloaded.labeled_frames
            for instance in labeled_frame.instances
        )
        if reloaded_types != {"Instance": VIEW_FRAME_COUNT * FISH_PER_VIEW}:
            raise RuntimeError(
                "locked SLP reload has unexpected instance types/counts: "
                f"{dict(reloaded_types)}"
            )

        _write_csv(ledger_tmp, INSTANCE_LEDGER_COLUMNS, ledger_rows)
        package_stat = locked_tmp.stat()
        evaluation_columns, evaluation_rows = build_evaluation_manifest_rows(
            merged,
            merge_report["frame_summaries"],
            locked_output_path=locked_path,
            package_size_bytes=package_stat.st_size,
            package_mtime_ns=package_stat.st_mtime_ns,
        )
        _write_csv(evaluation_tmp, evaluation_columns, evaluation_rows)

        input_hashes_after = {
            "base_slp": sha256_file(base_slp),
            "replacement_slp": sha256_file(replacement_slp),
            "manifest": sha256_file(manifest_path),
        }
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("an input changed during HITL finalization")
        frozen = {
            "schema_version": SCHEMA_VERSION,
            "status": "locked_model_assisted_annotations",
            "model_assisted_annotation": True,
            "contains_model_predictions": False,
            "contains_keypoint_coordinates": True,
            "inputs": package_provenance["inputs"],
            "candidate_review_provenance": package_provenance[
                "candidate_review_provenance"
            ],
            "counts": merge_report["counts"],
            "artifacts": {
                "locked_package": {
                    "path": str(locked_path.resolve()),
                    "sha256": sha256_file(locked_tmp),
                    "size_bytes": locked_tmp.stat().st_size,
                    "embedded": True,
                },
                "instance_ledger": {
                    "path": str(ledger_path.resolve()),
                    "sha256": sha256_file(ledger_tmp),
                    "rows": len(ledger_rows),
                    "columns": list(INSTANCE_LEDGER_COLUMNS),
                },
                "evaluation_manifest": {
                    "path": str(evaluation_path.resolve()),
                    "sha256": sha256_file(evaluation_tmp),
                    "rows": len(evaluation_rows),
                    "columns": evaluation_columns,
                },
            },
            "validation": {
                "exact_40_source_frames": True,
                "exact_120_view_frames": True,
                "exactly_three_instances_per_view": True,
                "single_canonical_mouth_tail_skeleton": True,
                "all_instances_user_type": True,
                "all_tracks_stripped": True,
                "at_least_one_finite_node_per_instance": True,
                "partial_nodes_allowed_and_flagged": True,
                "inputs_unchanged": True,
            },
        }
        provenance_tmp.write_text(
            json.dumps(frozen, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        for temporary, final in (
            (locked_tmp, locked_path),
            (ledger_tmp, ledger_path),
            (evaluation_tmp, evaluation_path),
            (provenance_tmp, provenance_path),
        ):
            if final.exists():
                raise FileExistsError(
                    f"refusing output created during finalization: {final}"
                )
            os.replace(temporary, final)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    return {
        "locked_package": str(locked_path),
        "instance_ledger": str(ledger_path),
        "evaluation_manifest": str(evaluation_path),
        "provenance": str(provenance_path),
        **merge_report["counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-slp", type=Path, required=True)
    parser.add_argument("--replacement-slp", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_hitl_benchmark(
        base_slp=args.base_slp,
        replacement_slp=args.replacement_slp,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
