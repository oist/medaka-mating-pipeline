"""Prepare and finalize a thin Round-8 side-mouth completion patch.

Preparation reads only the Round-8 side training package. Every synchronized
``(experiment, frame_idx)`` group with at least one missing mouth is selected,
and both original side-view labeled frames are copied into a thin, non-embedded
seed package. Existing instances, coordinates, point visibility, and instance
order are preserved.

Finalization reads a GUI-edited package, extracts only the original selected
side-view labeled frames, and validates that:

* every originally missing mouth is now finite;
* every originally finite mouth/tail coordinate is exactly unchanged;
* instance count and order are unchanged.

Neither mode mutates the Round-8 source package. Both modes require a new or
empty output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths
from scripts.sleap.multiview_annotation_gui import (
    classify_view,
    experiment_from_video,
)

VERSION = "mouth-anchor-completion-v1"
SIDE_VIEWS = ("side1_cam", "side2_cam")
DEFAULT_GUARD_FRAMES = 166
PREPARED_MANIFEST_NAME = "mouth_anchor_completion_manifest.csv"
PREPARED_PROVENANCE_NAME = "mouth_anchor_completion_provenance.json"
SEED_PACKAGE_NAME = "mouth_anchor_completion_seed.pkg.slp"
COMPLETED_MANIFEST_NAME = "mouth_anchor_completion_completed_manifest.csv"
COMPLETED_PROVENANCE_NAME = "mouth_anchor_completion_completed_provenance.json"
COMPLETED_PACKAGE_NAME = "mouth_anchor_completion_completed_patch.pkg.slp"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leaf_video_path(video: Any) -> str:
    """Return the deepest source filename from an embedded-video chain."""
    current = video
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        source = getattr(current, "source_video", None)
        if source is None:
            break
        current = source
    filename = str(getattr(current, "filename", "") or "")
    if filename:
        return filename
    backend = getattr(video, "backend", None)
    return str(getattr(backend, "source_filename", "") or "")


def label_key(labeled_frame: Any) -> tuple[str, str, int]:
    source = leaf_video_path(labeled_frame.video)
    experiment = experiment_from_video(source)
    view = classify_view(source)
    if experiment is None or view is None:
        raise ValueError(
            "cannot recover experiment/view from labeled-frame video: "
            f"{source!r}"
        )
    return experiment, view, int(labeled_frame.frame_idx)


def _node_names(skeleton: Any) -> list[str]:
    return [str(node.name) for node in skeleton.nodes]


def _validate_canonical_skeleton(labels: Any, *, source: Path) -> Any:
    if len(labels.skeletons) != 1:
        raise ValueError(f"{source} must contain exactly one skeleton")
    skeleton = labels.skeletons[0]
    names = _node_names(skeleton)
    if names != ["mouth", "tail"]:
        raise ValueError(
            f"{source} node order must be ['mouth', 'tail'], got {names}"
        )
    return skeleton


def _instance_array(instance: Any) -> np.ndarray:
    points = np.asarray(instance.numpy(), dtype=float)
    if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2:
        raise ValueError(
            "every source instance must have a canonical two-node point array; "
            f"got shape {points.shape}"
        )
    return points[:, :2]


def _point_visibility(instance: Any) -> tuple[bool, bool]:
    structured = getattr(instance, "points", None)
    if structured is None:
        points = _instance_array(instance)
        return tuple(  # type: ignore[return-value]
            bool(np.isfinite(point).all()) for point in points
        )
    return tuple(  # type: ignore[return-value]
        bool(structured[index]["visible"]) for index in range(2)
    )


def missing_mouth_indices(labeled_frame: Any) -> tuple[int, ...]:
    return tuple(
        index
        for index, instance in enumerate(labeled_frame.instances)
        if not np.isfinite(_instance_array(instance)[0]).all()
    )


@dataclass(frozen=True)
class SelectedFrame:
    source_index: int
    key: tuple[str, str, int]
    labeled_frame: Any
    missing_mouth_indices: tuple[int, ...]


def select_missing_mouth_groups(labels: Any) -> list[SelectedFrame]:
    """Select both side LFs for every group with any missing side mouth."""
    by_group: dict[
        tuple[str, int], dict[str, tuple[int, Any, tuple[int, ...]]]
    ] = {}
    for source_index, labeled_frame in enumerate(labels.labeled_frames):
        key = label_key(labeled_frame)
        experiment, view, frame_idx = key
        if view not in SIDE_VIEWS:
            raise ValueError(
                "Round-8 side training labels must contain side views only; "
                f"found {key}"
            )
        if len(labeled_frame.user_instances) != len(labeled_frame.instances):
            raise ValueError(
                "Round-8 side training labels must contain user instances only; "
                f"found predicted instances at {key}"
            )
        group = by_group.setdefault((experiment, frame_idx), {})
        if view in group:
            raise ValueError(f"duplicate source labeled frame: {key}")
        group[view] = (
            source_index,
            labeled_frame,
            missing_mouth_indices(labeled_frame),
        )

    selected: list[SelectedFrame] = []
    for (experiment, frame_idx), views in sorted(by_group.items()):
        if not any(item[2] for item in views.values()):
            continue
        missing_views = sorted(set(SIDE_VIEWS) - set(views))
        if missing_views:
            raise ValueError(
                "selected mouth-completion group does not contain both original "
                f"side LFs: {(experiment, frame_idx)} missing={missing_views}"
            )
        for view in SIDE_VIEWS:
            source_index, labeled_frame, missing = views[view]
            selected.append(
                SelectedFrame(
                    source_index=source_index,
                    key=(experiment, view, frame_idx),
                    labeled_frame=labeled_frame,
                    missing_mouth_indices=missing,
                )
            )
    return selected


def locked_manifest_overlaps(
    selected: Iterable[SelectedFrame],
    locked_manifest: Path,
    *,
    guard_frames: int = DEFAULT_GUARD_FRAMES,
) -> list[dict[str, Any]]:
    """Return exact or guarded source-frame overlaps with a locked manifest."""
    with locked_manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    columns = set(rows[0])
    if "experiment" not in columns:
        raise ValueError("locked manifest must contain an experiment column")
    frame_column = (
        "frame_idx"
        if "frame_idx" in columns
        else "frame"
        if "frame" in columns
        else None
    )
    if frame_column is None:
        raise ValueError("locked manifest must contain frame_idx or frame")

    locked_by_experiment: dict[str, set[int]] = {}
    for row in rows:
        experiment = str(row["experiment"]).strip()
        if not experiment:
            raise ValueError("locked manifest contains an empty experiment")
        locked_by_experiment.setdefault(experiment, set()).add(
            int(row[frame_column])
        )

    group_keys = sorted({(item.key[0], item.key[2]) for item in selected})
    overlaps: list[dict[str, Any]] = []
    for experiment, frame_idx in group_keys:
        for locked_frame in sorted(locked_by_experiment.get(experiment, set())):
            delta = abs(frame_idx - locked_frame)
            if delta <= int(guard_frames):
                overlaps.append(
                    {
                        "experiment": experiment,
                        "selected_frame_idx": frame_idx,
                        "locked_frame_idx": locked_frame,
                        "distance_frames": delta,
                        "overlap_kind": "exact" if delta == 0 else "guard_band",
                    }
                )
    return overlaps


def _validate_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory must be absent or empty: {output_dir}"
        )


def _clone_instance(instance: Any, skeleton: Any) -> Any:
    import sleap_io as sio

    coordinates = _instance_array(instance)
    visibility = _point_visibility(instance)
    points_data = np.full((2, 3), np.nan, dtype=float)
    points_data[:, 2] = 0.0
    for index in range(2):
        if np.isfinite(coordinates[index]).all():
            points_data[index, :2] = coordinates[index]
            points_data[index, 2] = float(visibility[index])
    return sio.Instance.from_numpy(points_data=points_data, skeleton=skeleton)


def _clone_labeled_frame(
    labeled_frame: Any,
    *,
    skeleton: Any,
    video: Any | None = None,
) -> Any:
    import sleap_io as sio

    return sio.LabeledFrame(
        video=labeled_frame.video if video is None else video,
        frame_idx=int(labeled_frame.frame_idx),
        instances=[
            _clone_instance(instance, skeleton)
            for instance in labeled_frame.instances
        ],
    )


def _assert_instance_equal(
    expected: Any,
    observed: Any,
    *,
    context: str,
) -> None:
    expected_points = _instance_array(expected)
    observed_points = _instance_array(observed)
    if not np.array_equal(expected_points, observed_points, equal_nan=True):
        raise RuntimeError(f"coordinate verification failed for {context}")
    if _point_visibility(expected) != _point_visibility(observed):
        raise RuntimeError(f"point-visibility verification failed for {context}")


def _assert_labeled_frame_equal(
    expected: Any,
    observed: Any,
    *,
    context: str,
) -> None:
    if len(expected.instances) != len(observed.instances):
        raise RuntimeError(
            f"instance-count verification failed for {context}: "
            f"{len(expected.instances)} != {len(observed.instances)}"
        )
    for index, (first, second) in enumerate(
        zip(expected.instances, observed.instances)
    ):
        _assert_instance_equal(
            first,
            second,
            context=f"{context} instance={index}",
        )


def _selected_lookup(
    selected: Sequence[SelectedFrame],
) -> dict[tuple[str, str, int], SelectedFrame]:
    lookup = {item.key: item for item in selected}
    if len(lookup) != len(selected):
        raise ValueError("selected labeled-frame keys are not unique")
    return lookup


def _labels_lookup(
    labels: Any,
    *,
    side_only: bool,
) -> tuple[dict[tuple[str, str, int], Any], int]:
    lookup: dict[tuple[str, str, int], Any] = {}
    ignored = 0
    for labeled_frame in labels.labeled_frames:
        try:
            key = label_key(labeled_frame)
        except ValueError:
            ignored += 1
            continue
        if side_only and key[1] not in SIDE_VIEWS:
            ignored += 1
            continue
        if key in lookup:
            raise ValueError(f"duplicate labeled frame in edited package: {key}")
        lookup[key] = labeled_frame
    return lookup, ignored


def _manifest_rows(
    selected: Sequence[SelectedFrame],
    *,
    source_path: Path,
    source_sha256: str,
    guard_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        experiment, view, frame_idx = item.key
        instances = list(item.labeled_frame.instances)
        rows.append(
            {
                "version": VERSION,
                "group_key": f"{experiment}:{frame_idx}",
                "experiment": experiment,
                "frame_idx": frame_idx,
                "view": view,
                "source_labeled_frame_index": item.source_index,
                "source_video": leaf_video_path(item.labeled_frame.video),
                "instance_count": len(instances),
                "missing_mouth_count": len(item.missing_mouth_indices),
                "missing_mouth_instance_indices": ";".join(
                    str(index) for index in item.missing_mouth_indices
                ),
                "finite_mouth_count": sum(
                    np.isfinite(_instance_array(instance)[0]).all()
                    for instance in instances
                ),
                "finite_tail_count": sum(
                    np.isfinite(_instance_array(instance)[1]).all()
                    for instance in instances
                ),
                "selection_reason": "group_has_any_missing_side_mouth",
                "split_role": "training_label_repair_only",
                "contains_locked_test": False,
                "locked_manifest_guard_frames": int(guard_frames),
                "source_labels": str(source_path.resolve()),
                "source_labels_sha256": source_sha256,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty manifest: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"completion manifest is empty: {path}")
    return rows


def _verify_manifest_against_source(
    rows: Sequence[Mapping[str, str]],
    selected: Sequence[SelectedFrame],
    *,
    source_sha256: str,
) -> None:
    selected_by_key = _selected_lookup(selected)
    row_by_key: dict[tuple[str, str, int], Mapping[str, str]] = {}
    declared_hashes: set[str] = set()
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["view"]),
            int(row["frame_idx"]),
        )
        if key in row_by_key:
            raise ValueError(f"duplicate completion manifest row: {key}")
        row_by_key[key] = row
        declared_hashes.add(str(row.get("source_labels_sha256", "")))
    if set(row_by_key) != set(selected_by_key):
        missing = sorted(set(selected_by_key) - set(row_by_key))
        extra = sorted(set(row_by_key) - set(selected_by_key))
        raise ValueError(
            f"completion manifest/source mismatch; missing={missing}, extra={extra}"
        )
    if declared_hashes != {source_sha256}:
        raise ValueError(
            "Round-8 source hash differs from the prepared manifest: "
            f"declared={sorted(declared_hashes)}, actual={source_sha256}"
        )
    for key, item in selected_by_key.items():
        row = row_by_key[key]
        if int(row["source_labeled_frame_index"]) != item.source_index:
            raise ValueError(f"source labeled-frame index changed for {key}")
        declared_missing = tuple(
            int(value)
            for value in str(row["missing_mouth_instance_indices"]).split(";")
            if value != ""
        )
        if declared_missing != item.missing_mouth_indices:
            raise ValueError(f"missing-mouth indices changed for {key}")
        if int(row["instance_count"]) != len(item.labeled_frame.instances):
            raise ValueError(f"source instance count changed for {key}")


def prepare_completion_labels(
    *,
    source_labels: Path,
    output_dir: Path,
    locked_manifest: Path | None = None,
    guard_frames: int = DEFAULT_GUARD_FRAMES,
) -> dict[str, Any]:
    """Build a thin seed containing both side LFs for every affected group."""
    import sleap_io as sio

    _validate_empty_output_dir(output_dir)
    if not source_labels.is_file():
        raise FileNotFoundError(f"Round-8 side train labels not found: {source_labels}")
    source_sha256_before = sha256_file(source_labels)
    labels = sio.load_slp(str(source_labels), open_videos=False)
    skeleton = _validate_canonical_skeleton(labels, source=source_labels)
    selected = select_missing_mouth_groups(labels)
    if not selected:
        raise ValueError("Round-8 side train labels contain no missing mouths")

    overlaps: list[dict[str, Any]] = []
    if locked_manifest is not None:
        if not locked_manifest.is_file():
            raise FileNotFoundError(f"locked manifest not found: {locked_manifest}")
        overlaps = locked_manifest_overlaps(
            selected,
            locked_manifest,
            guard_frames=guard_frames,
        )
        if overlaps:
            raise ValueError(
                "mouth-completion frames overlap the locked manifest at exact "
                f"or +/-{guard_frames} frames; count={len(overlaps)}, "
                f"preview={overlaps[:8]}"
            )

    rows = _manifest_rows(
        selected,
        source_path=source_labels,
        source_sha256=source_sha256_before,
        guard_frames=guard_frames,
    )
    group_count = len({(item.key[0], item.key[2]) for item in selected})
    missing_mouth_count = sum(
        len(item.missing_mouth_indices) for item in selected
    )
    provenance: dict[str, Any] = {
        "version": VERSION,
        "created_at_utc": utc_now(),
        "mode": "prepare",
        "selection_rule": (
            "Select every unique experiment/frame group in the Round-8 side "
            "training package with any non-finite mouth and include both "
            "original side1/side2 labeled frames."
        ),
        "source_labels": {
            "path": str(source_labels.resolve()),
            "sha256_before": source_sha256_before,
            "labeled_frames": len(labels.labeled_frames),
        },
        "locked_manifest": (
            {
                "path": str(locked_manifest.resolve()),
                "sha256": sha256_file(locked_manifest),
                "guard_frames": int(guard_frames),
                "exact_or_guarded_overlap_count": 0,
            }
            if locked_manifest is not None
            else {
                "provided": False,
                "guard_frames": int(guard_frames),
                "exact_or_guarded_overlap_count": None,
            }
        ),
        "selected_groups": group_count,
        "selected_side_labeled_frames": len(selected),
        "originally_missing_mouths": missing_mouth_count,
        "contains_original_coordinates": True,
        "contains_model_predictions": False,
        "contains_locked_test": False,
        "split_role": "training_label_repair_only",
        "embedded_frames": False,
        "source_labels_modified": False,
    }

    cloned_lfs = [
        _clone_labeled_frame(item.labeled_frame, skeleton=skeleton)
        for item in selected
    ]
    videos = list({id(lf.video): lf.video for lf in cloned_lfs}.values())
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        for lf in cloned_lfs
    ]
    seed_labels = sio.Labels(
        skeletons=[skeleton],
        videos=videos,
        labeled_frames=cloned_lfs,
        suggestions=suggestions,
        provenance=provenance,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / PREPARED_MANIFEST_NAME
    seed_path = output_dir / SEED_PACKAGE_NAME
    provenance_path = output_dir / PREPARED_PROVENANCE_NAME
    _write_csv(manifest_path, rows)
    sio.save_slp(seed_labels, str(seed_path), embed=False, verbose=False)

    reloaded = sio.load_slp(str(seed_path), open_videos=False)
    reloaded_lookup, ignored = _labels_lookup(reloaded, side_only=True)
    if ignored or len(reloaded_lookup) != len(selected):
        raise RuntimeError(
            "seed-package coverage verification failed: "
            f"side={len(reloaded_lookup)}, ignored={ignored}, "
            f"expected={len(selected)}"
        )
    for item in selected:
        observed = reloaded_lookup.get(item.key)
        if observed is None:
            raise RuntimeError(f"seed package is missing {item.key}")
        _assert_labeled_frame_equal(
            item.labeled_frame,
            observed,
            context=f"seed {item.key}",
        )

    source_sha256_after = sha256_file(source_labels)
    if source_sha256_after != source_sha256_before:
        raise RuntimeError("Round-8 source labels changed during preparation")
    provenance["source_labels"]["sha256_after"] = source_sha256_after
    provenance["manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "rows": len(rows),
    }
    provenance["seed_package"] = {
        "path": str(seed_path.resolve()),
        "sha256": sha256_file(seed_path),
        "labeled_frames": len(reloaded.labeled_frames),
        "instances": sum(len(lf.instances) for lf in reloaded.labeled_frames),
        "embed": False,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def _validate_completed_labeled_frame(
    original: Any,
    edited: Any,
    *,
    key: tuple[str, str, int],
    originally_missing_mouths: Sequence[int],
) -> int:
    if len(original.instances) != len(edited.instances):
        raise ValueError(
            f"instance count/order changed for {key}: "
            f"{len(original.instances)} != {len(edited.instances)}"
        )
    missing_set = set(int(index) for index in originally_missing_mouths)
    completed = 0
    for instance_index, (source_instance, edited_instance) in enumerate(
        zip(original.instances, edited.instances)
    ):
        source_points = _instance_array(source_instance)
        edited_points = _instance_array(edited_instance)
        source_visibility = _point_visibility(source_instance)
        edited_visibility = _point_visibility(edited_instance)
        for node_index, node_name in enumerate(("mouth", "tail")):
            if np.isfinite(source_points[node_index]).all():
                if not np.array_equal(
                    source_points[node_index],
                    edited_points[node_index],
                ):
                    raise ValueError(
                        f"originally finite {node_name} coordinate changed for "
                        f"{key} instance={instance_index}"
                    )
                if source_visibility[node_index] != edited_visibility[node_index]:
                    raise ValueError(
                        f"originally finite {node_name} visibility changed for "
                        f"{key} instance={instance_index}"
                    )
        if instance_index in missing_set:
            if not np.isfinite(edited_points[0]).all():
                raise ValueError(
                    f"originally missing mouth remains non-finite for "
                    f"{key} instance={instance_index}"
                )
            completed += 1
    if completed != len(missing_set):
        raise ValueError(
            f"missing-mouth completion count mismatch for {key}: "
            f"{completed} != {len(missing_set)}"
        )
    return completed


def finalize_completion_labels(
    *,
    source_labels: Path,
    prepared_manifest: Path,
    gui_output: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate GUI edits and write a thin completed side-only patch."""
    import sleap_io as sio

    _validate_empty_output_dir(output_dir)
    for path, description in (
        (source_labels, "Round-8 side train labels"),
        (prepared_manifest, "prepared completion manifest"),
        (gui_output, "GUI output"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    source_sha256_before = sha256_file(source_labels)
    source = sio.load_slp(str(source_labels), open_videos=False)
    source_skeleton = _validate_canonical_skeleton(source, source=source_labels)
    selected = select_missing_mouth_groups(source)
    manifest_rows = _load_manifest(prepared_manifest)
    _verify_manifest_against_source(
        manifest_rows,
        selected,
        source_sha256=source_sha256_before,
    )

    edited_labels = sio.load_slp(str(gui_output), open_videos=False)
    _validate_canonical_skeleton(edited_labels, source=gui_output)
    edited_lookup, ignored_gui_lfs = _labels_lookup(
        edited_labels,
        side_only=True,
    )
    selected_lookup = _selected_lookup(selected)
    missing_edited = sorted(set(selected_lookup) - set(edited_lookup))
    if missing_edited:
        raise ValueError(
            f"GUI output is missing original selected side LFs: {missing_edited}"
        )

    completed_lfs = []
    completed_by_key: dict[tuple[str, str, int], Any] = {}
    completed_rows: list[dict[str, Any]] = []
    total_completed = 0
    for key in sorted(selected_lookup):
        item = selected_lookup[key]
        edited = edited_lookup[key]
        completed = _validate_completed_labeled_frame(
            item.labeled_frame,
            edited,
            key=key,
            originally_missing_mouths=item.missing_mouth_indices,
        )
        total_completed += completed
        completed_lf = _clone_labeled_frame(
            edited,
            skeleton=source_skeleton,
            video=item.labeled_frame.video,
        )
        completed_lfs.append(completed_lf)
        completed_by_key[key] = completed_lf
        completed_rows.append(
            {
                "version": VERSION,
                "experiment": key[0],
                "view": key[1],
                "frame_idx": key[2],
                "source_labeled_frame_index": item.source_index,
                "instance_count": len(item.labeled_frame.instances),
                "originally_missing_mouth_count": len(
                    item.missing_mouth_indices
                ),
                "completed_mouth_count": completed,
                "originally_missing_mouth_instance_indices": ";".join(
                    str(index) for index in item.missing_mouth_indices
                ),
                "original_finite_coordinates_unchanged": True,
                "split_role": "training_label_repair_only",
                "contains_locked_test": False,
                "source_labels_sha256": source_sha256_before,
                "prepared_manifest_sha256": sha256_file(prepared_manifest),
                "gui_output_sha256": sha256_file(gui_output),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest_path = output_dir / COMPLETED_MANIFEST_NAME
    completed_package_path = output_dir / COMPLETED_PACKAGE_NAME
    completed_provenance_path = output_dir / COMPLETED_PROVENANCE_NAME
    _write_csv(completed_manifest_path, completed_rows)
    provenance: dict[str, Any] = {
        "version": VERSION,
        "created_at_utc": utc_now(),
        "mode": "finalize",
        "source_labels": {
            "path": str(source_labels.resolve()),
            "sha256_before": source_sha256_before,
        },
        "prepared_manifest": {
            "path": str(prepared_manifest.resolve()),
            "sha256": sha256_file(prepared_manifest),
        },
        "gui_output": {
            "path": str(gui_output.resolve()),
            "sha256": sha256_file(gui_output),
            "ignored_nonoriginal_or_non_side_labeled_frames": (
                ignored_gui_lfs
                + len(set(edited_lookup) - set(selected_lookup))
            ),
        },
        "selected_side_labeled_frames": len(completed_lfs),
        "completed_originally_missing_mouths": total_completed,
        "original_finite_coordinates_unchanged": True,
        "contains_only_original_selected_side_lfs": True,
        "contains_locked_test": False,
        "split_role": "training_label_repair_only",
        "embedded_frames": False,
        "source_labels_modified": False,
    }
    videos = list({id(lf.video): lf.video for lf in completed_lfs}.values())
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        for lf in completed_lfs
    ]
    completed_labels = sio.Labels(
        skeletons=[source_skeleton],
        videos=videos,
        labeled_frames=completed_lfs,
        suggestions=suggestions,
        provenance=provenance,
    )
    sio.save_slp(
        completed_labels,
        str(completed_package_path),
        embed=False,
        verbose=False,
    )

    reloaded = sio.load_slp(str(completed_package_path), open_videos=False)
    reloaded_lookup, ignored = _labels_lookup(reloaded, side_only=True)
    if ignored or set(reloaded_lookup) != set(completed_by_key):
        raise RuntimeError(
            "completed-package coverage verification failed: "
            f"observed={sorted(reloaded_lookup)}, "
            f"expected={sorted(completed_by_key)}, ignored={ignored}"
        )
    for key, expected in completed_by_key.items():
        _assert_labeled_frame_equal(
            expected,
            reloaded_lookup[key],
            context=f"completed {key}",
        )

    source_sha256_after = sha256_file(source_labels)
    if source_sha256_after != source_sha256_before:
        raise RuntimeError("Round-8 source labels changed during finalization")
    provenance["source_labels"]["sha256_after"] = source_sha256_after
    provenance["completed_manifest"] = {
        "path": str(completed_manifest_path.resolve()),
        "sha256": sha256_file(completed_manifest_path),
        "rows": len(completed_rows),
    }
    provenance["completed_patch_package"] = {
        "path": str(completed_package_path.resolve()),
        "sha256": sha256_file(completed_package_path),
        "labeled_frames": len(reloaded.labeled_frames),
        "instances": sum(len(lf.instances) for lf in reloaded.labeled_frames),
        "embed": False,
    }
    completed_provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def _resolve_round8_side_train_labels(
    explicit: Path | None,
    *,
    paths_config: Path,
) -> Path:
    if explicit is not None:
        return explicit
    local = load_local_paths(paths_config)
    if local is None or local.sleap_models_dir is None:
        raise ValueError(
            f"{paths_config} must define paths.sleap_models_dir, or pass "
            "--round8-side-train-labels"
        )
    return (
        local.sleap_models_dir
        / "retrain_20260722_round8"
        / "side"
        / "train_labels.pkg.slp"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Build a thin seed from Round-8 side train labels.",
    )
    prepare.add_argument(
        "--round8-side-train-labels",
        type=Path,
        default=None,
    )
    prepare.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--locked-manifest", type=Path, default=None)
    prepare.add_argument(
        "--guard-frames",
        type=int,
        default=DEFAULT_GUARD_FRAMES,
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="Validate one GUI output and write a side-only completion patch.",
    )
    finalize.add_argument(
        "--round8-side-train-labels",
        type=Path,
        default=None,
    )
    finalize.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
    )
    finalize.add_argument("--prepared-manifest", type=Path, required=True)
    finalize.add_argument("--gui-output", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_labels = _resolve_round8_side_train_labels(
        args.round8_side_train_labels,
        paths_config=args.paths_config,
    )
    if args.mode == "prepare":
        result = prepare_completion_labels(
            source_labels=source_labels,
            output_dir=args.output_dir,
            locked_manifest=args.locked_manifest,
            guard_frames=args.guard_frames,
        )
    else:
        result = finalize_completion_labels(
            source_labels=source_labels,
            prepared_manifest=args.prepared_manifest,
            gui_output=args.gui_output,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
