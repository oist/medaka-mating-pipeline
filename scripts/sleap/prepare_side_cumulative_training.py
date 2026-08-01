"""Build leakage-safe cumulative side-view labels from Round 8 and HITL labels.

This is a side-only preparation step. It does not train a model and never
modifies its inputs. The reviewed package is split according to the
authoritative ``selection_210.csv`` ``split_role`` column. New training rows
are unioned over the existing Round 8 side training package. The new balanced
validation split is used alone for checkpoint selection; the old K_1_2-heavy
Round 8 validation split is preserved as a separate regression package.

The new validation rows are protected with a source-frame guard: every
existing Round 8 training source frame within ``+/- guard_frames`` is removed
from both side views. After the union, the complete train and validation sets
must be guard-disjoint. New human labels win exact same-split collisions.

Point cloning deliberately reads the structured ``points["xy"]`` and
``points["visible"]`` fields. ``Instance.numpy()`` masks invisible points as
NaN and would therefore destroy the finite coordinates retained for genuine
occlusions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
)


SCHEMA_VERSION = "side-cumulative-training-v1"
SIDE_VIEWS = ("side1_cam", "side2_cam")
EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
SPLIT_ROLES = ("train", "validation")
DEFAULT_GUARD_FRAMES = 166

LabelKey = tuple[str, str, int]
SourceKey = tuple[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def label_key(labeled_frame: Any) -> LabelKey:
    source = _video_source_path(labeled_frame.video)
    experiment = experiment_from_video(source)
    view = classify_view(source)
    if experiment is None or view is None:
        raise ValueError(
            "cannot recover experiment/view from labeled-frame video: "
            f"{source!r}"
        )
    return str(experiment), str(view), int(labeled_frame.frame_idx)


def source_key(key: LabelKey) -> SourceKey:
    return key[0], key[2]


def _node_names(skeleton: Any) -> list[str]:
    return [str(node.name) for node in skeleton.nodes]


def _edge_names(skeleton: Any) -> list[tuple[str, str]]:
    return [
        (str(edge.source.name), str(edge.destination.name))
        for edge in skeleton.edges
    ]


def _validate_canonical_skeleton(labels: Any, *, path: Path) -> None:
    if len(labels.skeletons) != 1:
        raise ValueError(f"{path} must contain exactly one skeleton")
    skeleton = labels.skeletons[0]
    names = _node_names(skeleton)
    edges = _edge_names(skeleton)
    if names != ["mouth", "tail"] or edges != [("mouth", "tail")]:
        raise ValueError(
            f"{path} must use canonical mouth->tail skeleton; "
            f"nodes={names}, edges={edges}"
        )


def _structured_points(instance: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return raw XY and visibility without masking invisible coordinates."""
    points = getattr(instance, "points", None)
    if points is None or len(points) != 2:
        raise ValueError(
            "instances must expose two structured mouth/tail points"
        )
    xy = np.asarray([points[index]["xy"] for index in range(2)], dtype=float)
    visible = np.asarray(
        [bool(points[index]["visible"]) for index in range(2)], dtype=bool
    )
    if xy.shape != (2, 2):
        raise ValueError(f"unexpected structured point shape: {xy.shape}")
    return xy, visible


def _instances_are_user_only(labeled_frame: Any) -> bool:
    return len(labeled_frame.user_instances) == len(labeled_frame.instances)


def _load_lookup(
    path: Path,
    *,
    require_three_instances: bool,
    require_raw_finite: bool,
) -> tuple[Any, dict[LabelKey, Any], dict[str, Any]]:
    import sleap_io as sio

    labels = sio.load_slp(str(path), open_videos=False)
    _validate_canonical_skeleton(labels, path=path)
    lookup: dict[LabelKey, Any] = {}
    instance_distribution: Counter[int] = Counter()
    raw_missing_nodes = 0
    hidden_finite_nodes = 0
    visible_finite_nodes = 0
    for labeled_frame in labels.labeled_frames:
        key = label_key(labeled_frame)
        if key[0] not in EXPERIMENTS:
            raise ValueError(f"{path} contains unsupported experiment: {key}")
        if key[1] not in SIDE_VIEWS:
            raise ValueError(f"{path} must contain side views only: {key}")
        if key in lookup:
            raise ValueError(f"{path} contains duplicate labeled frame: {key}")
        if not _instances_are_user_only(labeled_frame):
            raise ValueError(f"{path} contains predicted instances at {key}")
        instance_count = len(labeled_frame.instances)
        instance_distribution[instance_count] += 1
        if require_three_instances and instance_count != 3:
            raise ValueError(
                f"{path} must contain three instances per view; "
                f"{key} has {instance_count}"
            )
        for instance_index, instance in enumerate(labeled_frame.instances):
            xy, visible = _structured_points(instance)
            finite = np.isfinite(xy).all(axis=1)
            raw_missing_nodes += int((~finite).sum())
            hidden_finite_nodes += int((finite & ~visible).sum())
            visible_finite_nodes += int((finite & visible).sum())
            if require_raw_finite and not bool(finite.all()):
                missing = [
                    name
                    for name, is_finite in zip(("mouth", "tail"), finite)
                    if not bool(is_finite)
                ]
                raise ValueError(
                    f"{path} has non-finite raw coordinates at {key}, "
                    f"instance={instance_index}, nodes={missing}"
                )
        lookup[key] = labeled_frame
    return labels, lookup, {
        "labeled_frames": len(lookup),
        "source_frames": len({source_key(key) for key in lookup}),
        "instances": sum(
            len(labeled_frame.instances) for labeled_frame in lookup.values()
        ),
        "instance_distribution": {
            str(count): int(frames)
            for count, frames in sorted(instance_distribution.items())
        },
        "raw_missing_nodes": int(raw_missing_nodes),
        "hidden_finite_nodes": int(hidden_finite_nodes),
        "visible_finite_nodes": int(visible_finite_nodes),
    }


def load_selection_manifest(
    path: Path, *, split_column: str = "split_role"
) -> dict[SourceKey, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"selection manifest is empty: {path}")
        required = {"experiment", "frame", str(split_column)}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"selection manifest is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"selection manifest is empty: {path}")

    split_by_source: dict[SourceKey, str] = {}
    for row in rows:
        key = (str(row["experiment"]).strip(), int(row["frame"]))
        role = str(row[str(split_column)]).strip()
        if key[0] not in EXPERIMENTS:
            raise ValueError(
                f"selection manifest contains unsupported experiment: {key[0]}"
            )
        if role not in SPLIT_ROLES:
            raise ValueError(
                f"selection manifest contains unsupported split_role: {role!r}"
            )
        if key in split_by_source:
            raise ValueError(f"duplicate selection source frame: {key}")
        split_by_source[key] = role

    observed_experiments = {key[0] for key in split_by_source}
    if observed_experiments != set(EXPERIMENTS):
        raise ValueError(
            "selection manifest must contain exactly the three K_1_[1,2,3] "
            f"experiments; observed={sorted(observed_experiments)}"
        )
    return split_by_source


def parse_inherited_source_exclusion(value: str) -> SourceKey:
    """Parse one explicit ``EXPERIMENT:FRAME`` inherited-source exclusion."""
    text = str(value).strip()
    if ":" not in text:
        raise ValueError(
            "inherited source exclusions must use EXPERIMENT:FRAME, "
            f"got {value!r}"
        )
    experiment, frame_text = text.rsplit(":", 1)
    experiment = experiment.strip()
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"unsupported inherited exclusion experiment: {experiment!r}"
        )
    try:
        frame = int(frame_text)
    except ValueError as exc:
        raise ValueError(
            f"invalid inherited exclusion frame: {frame_text!r}"
        ) from exc
    return experiment, frame


def load_inherited_source_exclusions(
    *,
    manifest: Path,
    split_report: Path | None,
    explicit_sources: Sequence[SourceKey],
) -> tuple[set[SourceKey], dict[str, Any]]:
    """Resolve synchronized inherited exclusions with auditable provenance."""
    exclusions = {
        (str(experiment), int(frame))
        for experiment, frame in explicit_sources
    }
    for experiment, _frame in exclusions:
        if experiment not in EXPERIMENTS:
            raise ValueError(
                f"unsupported inherited exclusion experiment: {experiment!r}"
            )
    provenance: dict[str, Any] = {
        "split_report": None,
        "explicit_cli_sources": [
            {"experiment": experiment, "frame": int(frame)}
            for experiment, frame in sorted(exclusions)
        ],
    }
    if split_report is not None:
        if not split_report.is_file():
            raise FileNotFoundError(split_report)
        payload = json.loads(split_report.read_text(encoding="utf-8"))
        declared_manifest = (
            payload.get("outputs", {}).get("selection_210", {})
        )
        declared_manifest_sha256 = declared_manifest.get("sha256")
        actual_manifest_sha256 = sha256_file(manifest)
        if (
            declared_manifest_sha256 is not None
            and str(declared_manifest_sha256) != actual_manifest_sha256
        ):
            raise ValueError(
                "split report/selection manifest hash mismatch: "
                f"declared={declared_manifest_sha256}, "
                f"actual={actual_manifest_sha256}"
            )
        rows = (
            payload.get("inherited_training", {}).get(
                "malformed_source_groups_removed"
            )
        )
        if not isinstance(rows, list):
            raise ValueError(
                "split report lacks "
                "inherited_training.malformed_source_groups_removed"
            )
        report_sources: set[SourceKey] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(
                    "split report inherited exclusion rows must be objects"
                )
            experiment = str(row.get("experiment", "")).strip()
            if experiment not in EXPERIMENTS:
                raise ValueError(
                    "split report contains unsupported inherited exclusion "
                    f"experiment: {experiment!r}"
                )
            report_sources.add((experiment, int(row["frame"])))
        exclusions.update(report_sources)
        provenance["split_report"] = {
            **file_fingerprint(split_report),
            "schema_version": payload.get("schema_version"),
            "declared_selection_manifest_sha256": (
                declared_manifest_sha256
            ),
            "resolved_sources": [
                {"experiment": experiment, "frame": int(frame)}
                for experiment, frame in sorted(report_sources)
            ],
        }
    provenance["resolved_sources"] = [
        {"experiment": experiment, "frame": int(frame)}
        for experiment, frame in sorted(exclusions)
    ]
    return exclusions, provenance


def _expected_new_keys(
    split_by_source: Mapping[SourceKey, str],
) -> set[LabelKey]:
    return {
        (experiment, view, frame)
        for experiment, frame in split_by_source
        for view in SIDE_VIEWS
    }


def _guard_violations(
    train_sources: Iterable[SourceKey],
    validation_sources: Iterable[SourceKey],
    *,
    guard_frames: int,
) -> list[dict[str, Any]]:
    validation_by_experiment: dict[str, list[int]] = {}
    for experiment, frame in validation_sources:
        validation_by_experiment.setdefault(experiment, []).append(int(frame))
    violations: list[dict[str, Any]] = []
    for experiment, train_frame in sorted(set(train_sources)):
        for validation_frame in sorted(
            validation_by_experiment.get(experiment, [])
        ):
            distance = abs(int(train_frame) - int(validation_frame))
            if distance <= int(guard_frames):
                violations.append(
                    {
                        "experiment": experiment,
                        "train_frame": int(train_frame),
                        "validation_frame": int(validation_frame),
                        "distance_frames": int(distance),
                    }
                )
    return violations


def assert_train_val_guard_integrity(
    train_keys: Iterable[LabelKey],
    validation_keys: Iterable[LabelKey],
    *,
    guard_frames: int,
) -> None:
    train_set = set(train_keys)
    validation_set = set(validation_keys)
    key_leakage = sorted(train_set & validation_set)
    if key_leakage:
        raise ValueError(
            f"train/validation labeled-frame key leakage: {key_leakage[:12]}"
        )
    violations = _guard_violations(
        (source_key(key) for key in train_set),
        (source_key(key) for key in validation_set),
        guard_frames=guard_frames,
    )
    if violations:
        raise ValueError(
            "combined train/validation source-frame guard violation; "
            f"guard_frames={guard_frames}, preview={violations[:12]}"
        )


def _clone_instance(instance: Any, skeleton: Any) -> Any:
    import sleap_io as sio

    xy, visible = _structured_points(instance)
    points_data = np.full((2, 3), np.nan, dtype=float)
    points_data[:, :2] = xy
    points_data[:, 2] = visible.astype(float)
    return sio.Instance.from_numpy(points_data=points_data, skeleton=skeleton)


def _clone_labeled_frame(labeled_frame: Any, skeleton: Any) -> Any:
    import sleap_io as sio

    return sio.LabeledFrame(
        video=labeled_frame.video,
        frame_idx=int(labeled_frame.frame_idx),
        instances=[
            _clone_instance(instance, skeleton)
            for instance in labeled_frame.instances
        ],
    )


def _assert_instances_equal(expected: Any, observed: Any, *, key: LabelKey) -> None:
    if len(expected.instances) != len(observed.instances):
        raise RuntimeError(f"instance count changed while writing {key}")
    for instance_index, (first, second) in enumerate(
        zip(expected.instances, observed.instances)
    ):
        first_xy, first_visible = _structured_points(first)
        second_xy, second_visible = _structured_points(second)
        if not np.array_equal(first_xy, second_xy, equal_nan=True):
            raise RuntimeError(
                f"raw coordinates changed while writing {key}, "
                f"instance={instance_index}"
            )
        if not np.array_equal(first_visible, second_visible):
            raise RuntimeError(
                f"visibility changed while writing {key}, "
                f"instance={instance_index}"
            )


def _package_summary(lookup: Mapping[LabelKey, Any]) -> dict[str, Any]:
    instance_distribution: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    for labeled_frame in lookup.values():
        instance_distribution[len(labeled_frame.instances)] += 1
        for instance in labeled_frame.instances:
            xy, visible = _structured_points(instance)
            finite = np.isfinite(xy).all(axis=1)
            counts["instances"] += 1
            counts["raw_finite_mouths"] += int(finite[0])
            counts["raw_finite_tails"] += int(finite[1])
            counts["visible_finite_mouths"] += int(finite[0] and visible[0])
            counts["visible_finite_tails"] += int(finite[1] and visible[1])
            counts["hidden_finite_mouths"] += int(finite[0] and not visible[0])
            counts["hidden_finite_tails"] += int(finite[1] and not visible[1])
    return {
        "labeled_frames": len(lookup),
        "source_frames": len({source_key(key) for key in lookup}),
        "instances": int(counts["instances"]),
        "instance_distribution": {
            str(count): int(frames)
            for count, frames in sorted(instance_distribution.items())
        },
        "raw_finite_mouths": int(counts["raw_finite_mouths"]),
        "raw_finite_tails": int(counts["raw_finite_tails"]),
        "visible_finite_mouths": int(counts["visible_finite_mouths"]),
        "visible_finite_tails": int(counts["visible_finite_tails"]),
        "hidden_finite_mouths": int(counts["hidden_finite_mouths"]),
        "hidden_finite_tails": int(counts["hidden_finite_tails"]),
    }


def _write_thin_package(
    lookup: Mapping[LabelKey, Any],
    *,
    path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    import sleap_io as sio

    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"], edges=[("mouth", "tail")]
    )
    expected = {
        key: _clone_labeled_frame(labeled_frame, skeleton)
        for key, labeled_frame in sorted(lookup.items())
    }
    labeled_frames = list(expected.values())
    videos = list(
        {id(labeled_frame.video): labeled_frame.video for labeled_frame in labeled_frames}.values()
    )
    suggestions = [
        sio.SuggestionFrame(
            video=labeled_frame.video,
            frame_idx=int(labeled_frame.frame_idx),
        )
        for labeled_frame in labeled_frames
    ]
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=videos,
        labeled_frames=labeled_frames,
        suggestions=suggestions,
        provenance=dict(provenance),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(path), embed=False, verbose=False)

    reloaded, observed, _summary = _load_lookup(
        path,
        require_three_instances=False,
        require_raw_finite=False,
    )
    del reloaded
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(
            f"written package coverage mismatch; missing={missing[:12]}, "
            f"extra={extra[:12]}"
        )
    for key in expected:
        _assert_instances_equal(expected[key], observed[key], key=key)
    return {
        **file_fingerprint(path),
        **_package_summary(observed),
        "embed": False,
    }


def _write_validation_evaluation_manifest(
    *,
    validation_labels: Path,
    path: Path,
) -> dict[str, Any]:
    """Write one visibility-aware row per primary validation labeled frame."""
    import sleap_io as sio

    labels = sio.load_slp(str(validation_labels), open_videos=False)
    rows: list[dict[str, Any]] = []
    for source_index, labeled_frame in enumerate(labels.labeled_frames):
        key = label_key(labeled_frame)
        visible_mouths = 0
        hidden_mouths = 0
        for instance in labeled_frame.instances:
            xy, visible = _structured_points(instance)
            mouth_finite = bool(np.isfinite(xy[0]).all())
            visible_mouths += int(mouth_finite and visible[0])
            hidden_mouths += int(mouth_finite and not visible[0])
        instance_count = len(labeled_frame.instances)
        rows.append(
            {
                "manifest_version": SCHEMA_VERSION,
                "split_role": "validation",
                "source_slp": str(validation_labels.resolve()),
                "source_labels_path": str(validation_labels.resolve()),
                "source_labeled_frame_index": int(source_index),
                "experiment": key[0],
                "view": key[1],
                "frame_idx": int(key[2]),
                "instance_count": int(instance_count),
                "visible_mouth_instances": int(visible_mouths),
                "hidden_mouth_instances": int(hidden_mouths),
                "strict_centroid_eligible": bool(
                    instance_count > 0
                    and visible_mouths == instance_count
                ),
                "centered_instance_eligible_instances": int(
                    visible_mouths
                ),
            }
        )
    fieldnames = [
        "manifest_version",
        "split_role",
        "source_slp",
        "source_labels_path",
        "source_labeled_frame_index",
        "experiment",
        "view",
        "frame_idx",
        "instance_count",
        "visible_mouth_instances",
        "hidden_mouth_instances",
        "strict_centroid_eligible",
        "centered_instance_eligible_instances",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        **file_fingerprint(path),
        "rows": len(rows),
        "strict_centroid_eligible_rows": sum(
            bool(row["strict_centroid_eligible"]) for row in rows
        ),
        "strict_centroid_ineligible_rows": sum(
            not bool(row["strict_centroid_eligible"]) for row in rows
        ),
        "centered_instance_eligible_instances": sum(
            int(row["centered_instance_eligible_instances"])
            for row in rows
        ),
    }


def _validate_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory must be absent or empty: {output_dir}"
        )


def prepare_cumulative_side_labels(
    *,
    base_train: Path,
    base_validation: Path,
    new_labels: Path,
    manifest: Path,
    output_dir: Path,
    guard_frames: int = DEFAULT_GUARD_FRAMES,
    copy_new_labels_to_output: bool = False,
    split_column: str = "split_role",
    inherited_exclusions_report: Path | None = None,
    inherited_exclusion_sources: Sequence[SourceKey] = (),
) -> dict[str, Any]:
    """Validate, union, and write cumulative side train/validation packages."""
    if int(guard_frames) < 0:
        raise ValueError("guard_frames must be non-negative")
    inputs = (base_train, base_validation, new_labels, manifest)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_empty_output_dir(output_dir)
    input_fingerprints = {
        "base_train": file_fingerprint(base_train),
        "base_validation": file_fingerprint(base_validation),
        "new_labels": file_fingerprint(new_labels),
        "manifest": file_fingerprint(manifest),
    }
    new_labels_for_union = new_labels
    if copy_new_labels_to_output:
        source_copy = output_dir / "source_labels" / new_labels.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new_labels, source_copy)
        source_copy_fingerprint = file_fingerprint(source_copy)
        if (
            source_copy_fingerprint["sha256"]
            != input_fingerprints["new_labels"]["sha256"]
        ):
            raise RuntimeError(
                "copied reviewed-label package does not match its source"
            )
        input_fingerprints["new_labels_training_source_copy"] = (
            source_copy_fingerprint
        )
        new_labels_for_union = source_copy

    split_by_source = load_selection_manifest(
        manifest, split_column=str(split_column)
    )
    inherited_exclusions, inherited_exclusion_provenance = (
        load_inherited_source_exclusions(
            manifest=manifest,
            split_report=inherited_exclusions_report,
            explicit_sources=inherited_exclusion_sources,
        )
    )
    _base_train_labels, base_train_lookup, base_train_summary = _load_lookup(
        base_train,
        require_three_instances=False,
        require_raw_finite=False,
    )
    _base_val_labels, base_val_lookup, base_val_summary = _load_lookup(
        base_validation,
        require_three_instances=False,
        require_raw_finite=False,
    )
    _new_labels, new_lookup, new_summary = _load_lookup(
        new_labels_for_union,
        require_three_instances=True,
        require_raw_finite=True,
    )

    expected_new = _expected_new_keys(split_by_source)
    if set(new_lookup) != expected_new:
        missing = sorted(expected_new - set(new_lookup))
        extra = sorted(set(new_lookup) - expected_new)
        raise ValueError(
            "new reviewed-label coverage does not match the selection manifest; "
            f"missing={missing[:12]}, extra={extra[:12]}"
        )
    new_train = {
        key: labeled_frame
        for key, labeled_frame in new_lookup.items()
        if split_by_source[source_key(key)] == "train"
    }
    new_validation = {
        key: labeled_frame
        for key, labeled_frame in new_lookup.items()
        if split_by_source[source_key(key)] == "validation"
    }
    new_validation_sources = {source_key(key) for key in new_validation}

    excluded_inherited_lookup: dict[LabelKey, Any] = {}
    if inherited_exclusions:
        views_by_source = {
            excluded_source: {
                key[1]
                for key in base_train_lookup
                if source_key(key) == excluded_source
            }
            for excluded_source in inherited_exclusions
        }
        missing_sources = sorted(
            key for key, views in views_by_source.items() if not views
        )
        if missing_sources:
            raise ValueError(
                "requested inherited source exclusions are absent from the "
                f"base training package: {missing_sources}"
            )
        incomplete_sources = {
            key: sorted(set(SIDE_VIEWS) - views)
            for key, views in views_by_source.items()
            if views != set(SIDE_VIEWS)
        }
        if incomplete_sources:
            raise ValueError(
                "requested inherited exclusions must remove synchronized "
                f"side pairs: {incomplete_sources}"
            )
        excluded_inherited_lookup = {
            key: labeled_frame
            for key, labeled_frame in base_train_lookup.items()
            if source_key(key) in inherited_exclusions
        }
    inherited_after_explicit_exclusion = {
        key: labeled_frame
        for key, labeled_frame in base_train_lookup.items()
        if source_key(key) not in inherited_exclusions
    }

    removed_base_train: dict[LabelKey, Any] = {}
    retained_base_train: dict[LabelKey, Any] = {}
    for key, labeled_frame in inherited_after_explicit_exclusion.items():
        experiment, frame = source_key(key)
        blocked = any(
            experiment == val_experiment
            and abs(int(frame) - int(val_frame)) <= int(guard_frames)
            for val_experiment, val_frame in new_validation_sources
        )
        target = removed_base_train if blocked else retained_base_train
        target[key] = labeled_frame

    train_collisions = sorted(set(retained_base_train) & set(new_train))
    combined_validation_collisions = sorted(
        set(base_val_lookup) & set(new_validation)
    )
    cumulative_train = dict(retained_base_train)
    cumulative_train.update(new_train)
    combined_validation = dict(base_val_lookup)
    combined_validation.update(new_validation)

    # Primary checkpoint selection is balanced across all three experiments.
    assert_train_val_guard_integrity(
        cumulative_train,
        new_validation,
        guard_frames=int(guard_frames),
    )
    # The old Round 8 validation set remains useful only as a frozen regression
    # slice. Verify that newly added training rows do not leak into its guard.
    assert_train_val_guard_integrity(
        new_train,
        base_val_lookup,
        guard_frames=int(guard_frames),
    )

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "inputs": input_fingerprints,
        "guard_frames": int(guard_frames),
        "split_column": str(split_column),
        "inherited_source_exclusions": inherited_exclusion_provenance,
        "new_labels_win_exact_collisions": True,
        "point_clone_policy": (
            "structured_points_xy_and_visible; Instance.numpy_not_used"
        ),
        "split_source": f"selection_manifest.{split_column}",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "side" / "train_labels.pkg.slp"
    validation_path = output_dir / "side" / "val_labels.pkg.slp"
    regression_validation_path = (
        output_dir / "side" / "regression_val_round8.pkg.slp"
    )
    combined_validation_path = (
        output_dir / "side" / "val_labels_combined.pkg.slp"
    )
    train_output = _write_thin_package(
        cumulative_train,
        path=train_path,
        provenance={**provenance, "split_role": "train"},
    )
    validation_output = _write_thin_package(
        new_validation,
        path=validation_path,
        provenance={
            **provenance,
            "split_role": "validation",
            "validation_role": "primary_balanced_checkpoint_selection",
        },
    )
    regression_validation_output = _write_thin_package(
        base_val_lookup,
        path=regression_validation_path,
        provenance={
            **provenance,
            "split_role": "validation",
            "validation_role": "frozen_round8_regression_only",
        },
    )
    combined_validation_output = _write_thin_package(
        combined_validation,
        path=combined_validation_path,
        provenance={
            **provenance,
            "split_role": "validation",
            "validation_role": "audit_only_combined_not_checkpoint_selection",
        },
    )
    validation_evaluation_manifest = (
        _write_validation_evaluation_manifest(
            validation_labels=validation_path,
            path=output_dir / "side" / "val_evaluation_manifest.csv",
        )
    )

    removed_sources = sorted(
        {source_key(key) for key in removed_base_train}
    )
    report: dict[str, Any] = {
        **provenance,
        "manifest_counts": {
            "source_frames": len(split_by_source),
            "train_source_frames": sum(
                role == "train" for role in split_by_source.values()
            ),
            "validation_source_frames": sum(
                role == "validation" for role in split_by_source.values()
            ),
            "expected_side_view_frames": len(expected_new),
        },
        "input_label_summaries": {
            "base_train": base_train_summary,
            "base_validation": base_val_summary,
            "new_complete": new_summary,
        },
        "inherited_source_exclusion": {
            "requested_source_frames": len(inherited_exclusions),
            "removed_labeled_frames": len(excluded_inherited_lookup),
            "removed_summary": _package_summary(
                excluded_inherited_lookup
            ),
            "provenance": inherited_exclusion_provenance,
        },
        "guard_removal": {
            "base_train_source_frames_removed": len(removed_sources),
            "base_train_labeled_frames_removed": len(removed_base_train),
            "removed_source_frames": [
                {"experiment": experiment, "frame": int(frame)}
                for experiment, frame in removed_sources
            ],
        },
        "collisions": {
            "train_exact_labeled_frame_collisions_new_won": len(
                train_collisions
            ),
            "combined_validation_exact_labeled_frame_collisions_new_won": len(
                combined_validation_collisions
            ),
            "train_collision_keys": [list(key) for key in train_collisions],
            "combined_validation_collision_keys": [
                list(key) for key in combined_validation_collisions
            ],
        },
        "outputs": {
            "train": train_output,
            "primary_validation": validation_output,
            "primary_validation_evaluation_manifest": (
                validation_evaluation_manifest
            ),
            "round8_regression_validation": regression_validation_output,
            "combined_validation_audit_only": combined_validation_output,
        },
        "verification": {
            "primary_train_vs_primary_validation_key_leakage": [],
            "primary_train_vs_primary_validation_guard_violations": [],
            "new_train_vs_round8_regression_validation_key_leakage": [],
            "new_train_vs_round8_regression_validation_guard_violations": [],
            "canonical_skeleton": {
                "nodes": ["mouth", "tail"],
                "edges": [["mouth", "tail"]],
            },
            "primary_validation_policy": (
                "new balanced 60-view validation only"
            ),
            "round8_validation_policy": (
                "preserved separately for regression; excluded from "
                "checkpoint selection"
            ),
            "new_labels_three_instances_per_view": True,
            "new_labels_raw_coordinates_finite": True,
            "new_hidden_coordinates_and_visibility_preserved": True,
            "inputs_modified": False,
        },
    }
    report_path = output_dir / "prepare_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--new-labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split-column",
        default="split_role",
        help=(
            "Manifest column containing train/validation assignments. "
            "Defaults to split_role for backward compatibility."
        ),
    )
    parser.add_argument(
        "--inherited-exclusions-report",
        type=Path,
        help=(
            "Optional split-design JSON whose "
            "inherited_training.malformed_source_groups_removed rows are "
            "excluded synchronously from the inherited side train package."
        ),
    )
    parser.add_argument(
        "--exclude-inherited-source",
        action="append",
        type=parse_inherited_source_exclusion,
        default=[],
        metavar="EXPERIMENT:FRAME",
        help=(
            "Explicit synchronized inherited source-frame exclusion. "
            "Repeat for multiple source frames."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES
    )
    parser.add_argument(
        "--copy-new-labels-to-output",
        action="store_true",
        help=(
            "Copy the self-contained reviewed SLP under output-dir/source_labels "
            "before building thin packages, so training does not depend on the "
            "original annotation-workstation path."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_cumulative_side_labels(
        base_train=args.base_train,
        base_validation=args.base_validation,
        new_labels=args.new_labels,
        manifest=args.manifest,
        output_dir=args.output_dir,
        guard_frames=args.guard_frames,
        copy_new_labels_to_output=args.copy_new_labels_to_output,
        split_column=args.split_column,
        inherited_exclusions_report=args.inherited_exclusions_report,
        inherited_exclusion_sources=args.exclude_inherited_source,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
