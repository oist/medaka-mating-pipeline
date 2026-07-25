from __future__ import annotations

"""Build the versioned Phase-1 SLEAP top-down benchmark manifest.

The builder is deliberately read-only with respect to SLEAP packages. It
records stable source-video/frame keys, separates provenance classes, selects
the in-sample stress cohort by a stable event-window hash, and emits an
unannotated future-lock queue without treating predictions as ground truth.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
)


MANIFEST_VERSION = "phase1-v1"
DEFAULT_SEED = 20260724
DEFAULT_EVENT_WINDOW_FRAMES = 166
VIEWS = ("top_cam", "side1_cam", "side2_cam")
KNOWN_MANUAL_REVIEW_FRAMES = {97219, 97340, 97630}

MANIFEST_COLUMNS = [
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
]


def stable_hash(*parts: object, seed: int = DEFAULT_SEED) -> str:
    """Return a stable SHA-256 digest for a semantic key."""
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_frame_key(
    experiment: str,
    view: str,
    video_path: str,
    frame_idx: int,
) -> str:
    """Return a stable key independent of package-local video identifiers."""
    digest = stable_hash(
        str(experiment),
        str(view),
        _normalise_path(video_path),
        int(frame_idx),
        seed=0,
    )
    return f"mf1_{digest[:20]}"


def _normalise_path(value: object) -> str:
    return str(value).replace("\\", "/").casefold()


def assert_no_event_split_leakage(
    manifest: pd.DataFrame,
    *,
    split_column: str = "split_role",
) -> None:
    """Raise when one experiment/event is assigned to multiple split roles."""
    required = {"experiment", "event_id", split_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks leakage-check columns: {sorted(missing)}")
    table = manifest.copy()
    table = table.loc[table["event_id"].astype(str).str.len().gt(0)]
    conflicts = (
        table.groupby(["experiment", "event_id"], dropna=False)[split_column]
        .nunique(dropna=False)
        .loc[lambda values: values > 1]
    )
    if not conflicts.empty:
        keys = ["|".join(map(str, key)) for key in conflicts.index[:10]]
        raise ValueError(f"event-level split leakage: {keys}")


def select_event_aware(
    rows: pd.DataFrame,
    *,
    limit: int | None,
    seed: int = DEFAULT_SEED,
    event_column: str = "event_id",
) -> pd.DataFrame:
    """Select at most one deterministic row per event, then cap by event hash."""
    if rows.empty:
        return rows.copy()
    if event_column not in rows:
        raise ValueError(f"missing event column: {event_column}")
    work = rows.copy()
    work["_row_hash"] = [
        stable_hash(
            getattr(row, event_column),
            getattr(row, "frame_key", ""),
            seed=seed,
        )
        for row in work.itertuples(index=False)
    ]
    one_per_event = (
        work.sort_values(["_row_hash", "frame_key"], kind="mergesort")
        .drop_duplicates(event_column, keep="first")
        .copy()
    )
    one_per_event["_event_hash"] = [
        stable_hash(value, seed=seed)
        for value in one_per_event[event_column].astype(str)
    ]
    one_per_event = one_per_event.sort_values(
        ["_event_hash", "_row_hash", "frame_key"], kind="mergesort"
    )
    if limit is not None and limit >= 0:
        one_per_event = one_per_event.head(int(limit))
    return one_per_event.drop(
        columns=["_row_hash", "_event_hash"], errors="ignore"
    ).sort_values("frame_key", kind="mergesort")


def _node_names(inst: Any, n_nodes: int) -> list[str]:
    names: list[str] = []
    points = getattr(inst, "points", None)
    dtype = getattr(points, "dtype", None)
    if dtype is not None and dtype.names and "name" in dtype.names:
        names = [str(value) for value in points["name"]]
    if len(names) != n_nodes:
        skeleton = getattr(inst, "skeleton", None)
        nodes = getattr(skeleton, "nodes", []) if skeleton is not None else []
        names = [str(getattr(node, "name", node)) for node in nodes]
    if len(names) != n_nodes:
        names = [f"node_{index}" for index in range(n_nodes)]
    return names


def _instance_statuses(inst: Any) -> tuple[np.ndarray, dict[str, str]]:
    points = np.asarray(inst.numpy(), dtype=float)[:, :2]
    names = _node_names(inst, len(points))
    visible = np.ones(len(points), dtype=bool)
    raw_points = getattr(inst, "points", None)
    dtype = getattr(raw_points, "dtype", None)
    if dtype is not None and dtype.names and "visible" in dtype.names:
        visible = np.asarray(raw_points["visible"], dtype=bool)
    statuses: dict[str, str] = {}
    for index, name in enumerate(names):
        if not np.isfinite(points[index]).all():
            statuses[name] = "missing"
        elif visible[index]:
            statuses[name] = "visible"
        else:
            statuses[name] = "hidden"
    return points, statuses


def summarize_instances(
    instances: Iterable[Any],
    *,
    source_category: str = "",
    close_contact_px: float = 25.0,
) -> dict[str, Any]:
    """Summarize visibility and scientific difficulty for one labeled frame."""
    arrays: list[np.ndarray] = []
    statuses_by_instance: list[dict[str, str]] = []
    for inst in instances:
        points, statuses = _instance_statuses(inst)
        arrays.append(points)
        statuses_by_instance.append(statuses)

    mouths: list[np.ndarray] = []
    all_nodes_complete = True
    visibility_parts: list[str] = []
    for index, (points, statuses) in enumerate(zip(arrays, statuses_by_instance)):
        ordered = ",".join(f"{name}:{state}" for name, state in statuses.items())
        visibility_parts.append(f"fish{index + 1}[{ordered}]")
        all_nodes_complete &= all(state == "visible" for state in statuses.values())
        mouth_index = next(
            (
                node_index
                for node_index, name in enumerate(statuses)
                if name.casefold() == "mouth"
            ),
            0 if len(points) else -1,
        )
        if mouth_index >= 0 and np.isfinite(points[mouth_index]).all():
            mouths.append(points[mouth_index])

    instance_count = len(arrays)
    all_three_mouths_finite = instance_count == 3 and len(mouths) == 3
    min_mouth_distance = float("inf")
    for first in range(len(mouths)):
        for second in range(first + 1, len(mouths)):
            min_mouth_distance = min(
                min_mouth_distance,
                float(np.linalg.norm(mouths[first] - mouths[second])),
            )
    category_is_close = str(source_category).casefold().startswith("close_")
    if not all_nodes_complete:
        difficulty = "partial_or_occluded"
    elif category_is_close or min_mouth_distance <= float(close_contact_px):
        difficulty = "close_contact_or_crossing"
    else:
        difficulty = "ordinary"
    return {
        "difficulty": difficulty,
        "node_visibility": ";".join(visibility_parts),
        "instance_count": instance_count,
        "all_three_mouths_finite": bool(all_three_mouths_finite),
        "strict_centroid_eligible": bool(all_three_mouths_finite),
    }


def _source_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {
            "source_slp_available": False,
            "source_slp_size_bytes": "",
            "source_slp_mtime_ns": "",
        }
    return {
        "source_slp_available": True,
        "source_slp_size_bytes": int(stat.st_size),
        "source_slp_mtime_ns": int(stat.st_mtime_ns),
    }


def _selection_lookup(selection: pd.DataFrame) -> dict[int, Mapping[str, Any]]:
    return {
        int(row["frame"]): row
        for row in selection.to_dict(orient="records")
    }


def load_package_rows(
    source_slp: Path,
    *,
    cohort: str,
    split_role: str,
    provenance_class: str,
    selection: pd.DataFrame | None = None,
    seed: int = DEFAULT_SEED,
    event_window_frames: int = DEFAULT_EVENT_WINDOW_FRAMES,
    round7_event_exposed: bool = False,
    round8_finetune_member: bool = False,
) -> pd.DataFrame:
    """Load SLEAP label metadata and GT summaries without modifying the package."""
    import sleap_io as sio

    labels = sio.load_slp(str(source_slp))
    provenance = json.dumps(
        getattr(labels, "provenance", {}) or {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    stat_fields = _source_stat(source_slp)
    selection_by_frame = (
        _selection_lookup(selection) if selection is not None else {}
    )
    video_index_by_id = {id(video): index for index, video in enumerate(labels.videos)}
    rows: list[dict[str, Any]] = []
    for label_position, lf in enumerate(labels.labeled_frames):
        video_path = _video_source_path(lf.video)
        experiment = experiment_from_video(video_path)
        view = classify_view(video_path)
        if experiment is None or view is None:
            raise ValueError(
                f"cannot recover experiment/view from {source_slp}: {video_path}"
            )
        frame_idx = int(lf.frame_idx)
        selected = dict(selection_by_frame.get(frame_idx, {}))
        source_category = str(selected.get("category", ""))
        event_id = str(selected.get("event_group_id", "")).strip()
        if not event_id:
            window = frame_idx // int(event_window_frames)
            event_id = (
                f"source_window_{window * int(event_window_frames)}_"
                f"{(window + 1) * int(event_window_frames) - 1}"
            )
        event_start = selected.get(
            "event_start_frame",
            (frame_idx // int(event_window_frames)) * int(event_window_frames),
        )
        event_end = selected.get(
            "event_end_frame",
            (frame_idx // int(event_window_frames) + 1)
            * int(event_window_frames)
            - 1,
        )
        summary = summarize_instances(
            lf.instances,
            source_category=source_category,
        )
        needs_review = (
            summary["difficulty"] == "partial_or_occluded"
            or frame_idx in KNOWN_MANUAL_REVIEW_FRAMES
            or source_category == "manual_fp_or_fn"
        )
        frame_key = stable_frame_key(experiment, view, video_path, frame_idx)
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "selection_seed": int(seed),
                "selection_rule": (
                    "authoritative_event_holdout"
                    if selection is not None
                    else "sha256(seed,event_id,frame_key);one_frame_per_event"
                ),
                "frame_key": frame_key,
                "experiment": experiment,
                "view": view,
                "camera": view,
                "video_path": video_path,
                "frame_idx": frame_idx,
                "source_slp": str(source_slp),
                **stat_fields,
                "source_video_index": int(video_index_by_id[id(lf.video)]),
                "source_labeled_frame_index": int(label_position),
                "source_provenance": provenance,
                "cohort": cohort,
                "split_role": split_role,
                "provenance_class": provenance_class,
                "event_id": event_id,
                "event_start_frame": int(event_start),
                "event_end_frame": int(event_end),
                **summary,
                "source_category": source_category,
                "needs_manual_review": bool(needs_review),
                "annotation_present": True,
                "round7_event_exposed": bool(round7_event_exposed),
                "round8_finetune_member": bool(round8_finetune_member),
                "selection_reason": str(selected.get("selection_reason", "")),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _frame_exposure_map(rows: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    exposure: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        exposure[(str(row.experiment), str(row.view))].append(int(row.frame_idx))
    return {key: sorted(set(frames)) for key, frames in exposure.items()}


def event_is_exposed(
    exposure_frames: Sequence[int],
    *,
    event_start: int,
    event_end: int,
    guard_frames: int = DEFAULT_EVENT_WINDOW_FRAMES,
) -> bool:
    """Return whether a training frame touches an event plus its guard band."""
    low = int(event_start) - int(guard_frames)
    high = int(event_end) + int(guard_frames)
    return any(low <= int(frame) <= high for frame in exposure_frames)


def build_future_annotation_queue(
    expansion: pd.DataFrame,
    *,
    video_paths_by_view: Mapping[str, str],
    round7_exposure: Mapping[tuple[str, str], Sequence[int]],
    round8_pilot_train_frames: Sequence[int],
    prospective_source_slp: Path,
    seed: int = DEFAULT_SEED,
    guard_frames: int = DEFAULT_EVENT_WINDOW_FRAMES,
) -> pd.DataFrame:
    """Expand uncorrected candidates by view and mark future-lock eligibility."""
    rows: list[dict[str, Any]] = []
    stat_fields = _source_stat(prospective_source_slp)
    for selected in expansion.to_dict(orient="records"):
        experiment = str(selected["experiment"])
        frame_idx = int(selected["frame"])
        event_start = int(selected["event_start_frame"])
        event_end = int(selected["event_end_frame"])
        source_category = str(selected.get("category", ""))
        if source_category.startswith("close_"):
            difficulty = "close_contact_or_crossing"
        else:
            difficulty = "unknown_until_annotation"
        for view in VIEWS:
            video_path = str(video_paths_by_view[view])
            round7_exposed = event_is_exposed(
                round7_exposure.get((experiment, view), []),
                event_start=event_start,
                event_end=event_end,
                guard_frames=guard_frames,
            )
            pilot_exposed = event_is_exposed(
                round8_pilot_train_frames,
                event_start=event_start,
                event_end=event_end,
                guard_frames=guard_frames,
            )
            eligible = not round7_exposed and not pilot_exposed
            frame_key = stable_frame_key(experiment, view, video_path, frame_idx)
            rows.append(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "selection_seed": int(seed),
                    "selection_rule": (
                        "future_lock_candidate;reserve_entire_event;"
                        f"guard={int(guard_frames)}"
                    ),
                    "frame_key": frame_key,
                    "experiment": experiment,
                    "view": view,
                    "camera": view,
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "source_slp": str(prospective_source_slp),
                    **stat_fields,
                    "source_video_index": "",
                    "source_labeled_frame_index": "",
                    "source_provenance": (
                        "uncorrected expansion candidate; annotations unavailable"
                    ),
                    "cohort": "future_locked_test_annotation_queue",
                    "split_role": "future_manual_annotation",
                    "provenance_class": "unknown_provenance",
                    "event_id": str(selected["event_group_id"]),
                    "event_start_frame": event_start,
                    "event_end_frame": event_end,
                    "difficulty": difficulty,
                    "source_category": source_category,
                    "node_visibility": "unknown_until_annotation",
                    "instance_count": "",
                    "all_three_mouths_finite": False,
                    "strict_centroid_eligible": False,
                    "needs_manual_review": True,
                    "annotation_present": False,
                    "round7_event_exposed": bool(round7_exposed),
                    "round8_finetune_member": bool(pilot_exposed),
                    "selection_reason": (
                        f"{selected.get('selection_reason', '')};"
                        f"future_lock_eligible={str(eligible).lower()}"
                    ),
                    "future_lock_eligible": bool(eligible),
                }
            )
    columns = [*MANIFEST_COLUMNS, "future_lock_eligible"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["event_id", "frame_idx", "view"], kind="mergesort"
    )


def _select_stress_rows(
    rows: pd.DataFrame,
    *,
    per_experiment_view_difficulty: int,
    seed: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    group_columns = ["experiment", "view", "difficulty"]
    for _key, group in rows.groupby(group_columns, sort=True, dropna=False):
        selected.append(
            select_event_aware(
                group,
                limit=int(per_experiment_view_difficulty),
                seed=seed,
            )
        )
    if not selected:
        return rows.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True).sort_values(
        ["experiment", "view", "difficulty", "frame_idx"], kind="mergesort"
    )


def build_phase1_manifests(
    *,
    round7_dir: Path,
    round8_dir: Path,
    selection_manifest: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    stress_per_experiment_view_difficulty: int = 5,
    guard_frames: int = DEFAULT_EVENT_WINDOW_FRAMES,
) -> dict[str, Any]:
    """Build the benchmark, future annotation queue, and provenance summary."""
    selection = pd.read_csv(selection_manifest, keep_default_na=False)
    pilot = selection.loc[selection["annotation_batch"].eq("pilot")].copy()
    pilot_holdout = pilot.loc[pilot["split"].eq("holdout")].copy()
    pilot_train_frames = (
        pilot.loc[pilot["split"].eq("train"), "frame"].astype(int).tolist()
    )
    expansion = selection.loc[selection["annotation_batch"].eq("expansion")].copy()

    round7_top = load_package_rows(
        round7_dir / "top" / "labels.pkg.slp",
        cohort="multi_experiment_in_sample_stress",
        split_role="stress_regression",
        provenance_class="in_sample_stress_regression",
        seed=seed,
        round7_event_exposed=True,
        round8_finetune_member=True,
    )
    round7_side = load_package_rows(
        round7_dir / "side" / "labels.pkg.slp",
        cohort="multi_experiment_in_sample_stress",
        split_role="stress_regression",
        provenance_class="in_sample_stress_regression",
        seed=seed,
        round7_event_exposed=True,
        round8_finetune_member=True,
    )
    all_round7 = pd.concat([round7_top, round7_side], ignore_index=True)
    stress = _select_stress_rows(
        all_round7,
        per_experiment_view_difficulty=stress_per_experiment_view_difficulty,
        seed=seed,
    )
    stress["selection_rule"] = (
        "sha256(seed,event_id,frame_key);one_frame_per_event;"
        f"max={int(stress_per_experiment_view_difficulty)}"
        "_per_experiment_view_difficulty"
    )

    pilot_top = load_package_rows(
        round8_dir / "top" / "val_labels.pkg.slp",
        cohort="round8_pilot_event_holdout",
        split_role="round8_finetune_holdout",
        provenance_class="held_out_round8_but_seen_round7",
        selection=pilot_holdout,
        seed=seed,
        round7_event_exposed=True,
        round8_finetune_member=False,
    )
    pilot_side = load_package_rows(
        round8_dir / "side" / "val_labels.pkg.slp",
        cohort="round8_pilot_event_holdout",
        split_role="round8_finetune_holdout",
        provenance_class="held_out_round8_but_seen_round7",
        selection=pilot_holdout,
        seed=seed,
        round7_event_exposed=True,
        round8_finetune_member=False,
    )
    pilot_rows = pd.concat([pilot_top, pilot_side], ignore_index=True)
    benchmark = pd.concat([pilot_rows, stress], ignore_index=True)

    # Split isolation is checked separately within each scientific cohort:
    # the same source event can intentionally occur in the stress regression
    # and in the fine-tune holdout, but is never represented as a common locked
    # test. That overlap is explicit in provenance_class.
    for _cohort, group in benchmark.groupby("cohort", sort=False):
        assert_no_event_split_leakage(group)

    exposure = _frame_exposure_map(all_round7)
    video_paths_by_view = {
        str(row.view): str(row.video_path)
        for row in pilot_rows.itertuples(index=False)
    }
    missing_views = set(VIEWS) - set(video_paths_by_view)
    if missing_views:
        raise ValueError(f"could not resolve raw video paths for {sorted(missing_views)}")
    prospective_expansion = (
        round8_dir.parent.parent
        / "runs"
        / "round8_active_learning"
        / str(selection.iloc[0]["experiment"])
        / "expansion_240_edits_full.pkg.slp"
    )
    # Prefer the sibling runs root that is discoverable from the corrected
    # pilot package provenance when available.
    pilot_provenance = json.loads(str(pilot_rows.iloc[0]["source_provenance"]))
    corrected_inputs = pilot_provenance.get("corrected_inputs", [])
    if corrected_inputs:
        prospective_expansion = Path(corrected_inputs[0]).with_name(
            "expansion_240_edits_full.pkg.slp"
        )
    future_queue = build_future_annotation_queue(
        expansion,
        video_paths_by_view=video_paths_by_view,
        round7_exposure=exposure,
        round8_pilot_train_frames=pilot_train_frames,
        prospective_source_slp=prospective_expansion,
        seed=seed,
        guard_frames=guard_frames,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_manifest_phase1_v1.csv"
    pilot_path = output_dir / "benchmark_manifest_pilot_event_holdout_v1.csv"
    stress_path = output_dir / "benchmark_manifest_in_sample_stress_v1.csv"
    queue_path = output_dir / "future_locked_test_annotation_queue_v1.csv"
    eligible_queue_path = (
        output_dir / "future_locked_test_annotation_queue_eligible_v1.csv"
    )
    provenance_path = output_dir / "benchmark_manifest_provenance_v1.json"
    benchmark.to_csv(benchmark_path, index=False, lineterminator="\n")
    pilot_rows.to_csv(pilot_path, index=False, lineterminator="\n")
    stress.to_csv(stress_path, index=False, lineterminator="\n")
    future_queue.to_csv(queue_path, index=False, lineterminator="\n")
    future_queue.loc[future_queue["future_lock_eligible"]].to_csv(
        eligible_queue_path,
        index=False,
        lineterminator="\n",
    )

    summary = {
        "manifest_version": MANIFEST_VERSION,
        "selection_seed": int(seed),
        "selection_manifest": str(selection_manifest.resolve()),
        "round7_dir": str(round7_dir),
        "round8_dir": str(round8_dir),
        "benchmark_manifest": str(benchmark_path.resolve()),
        "pilot_manifest": str(pilot_path.resolve()),
        "stress_manifest": str(stress_path.resolve()),
        "future_annotation_queue": str(queue_path.resolve()),
        "future_annotation_queue_eligible": str(eligible_queue_path.resolve()),
        "counts": {
            "benchmark_rows": int(len(benchmark)),
            "by_cohort": {
                str(key): int(value)
                for key, value in benchmark["cohort"].value_counts().items()
            },
            "by_provenance_class": {
                str(key): int(value)
                for key, value in benchmark["provenance_class"]
                .value_counts()
                .items()
            },
            "by_view": {
                str(key): int(value)
                for key, value in benchmark["view"].value_counts().items()
            },
            "by_difficulty": {
                str(key): int(value)
                for key, value in benchmark["difficulty"].value_counts().items()
            },
            "future_queue_rows": int(len(future_queue)),
            "future_lock_eligible_rows": int(
                future_queue["future_lock_eligible"].sum()
            ),
            "future_lock_eligible_events": int(
                future_queue.loc[future_queue["future_lock_eligible"], "event_id"]
                .nunique()
            ),
        },
        "leakage_policy": {
            "event_window_frames": int(DEFAULT_EVENT_WINDOW_FRAMES),
            "future_queue_guard_frames": int(guard_frames),
            "pilot_provenance_class": "held_out_round8_but_seen_round7",
            "reason": (
                "Round 7 contains labeled frames in the pilot sentinel event, "
                "and Round 8 was warm-started from Round 7."
            ),
        },
    }
    provenance_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round7-dir", type=Path, required=True)
    parser.add_argument("--round8-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--stress-per-experiment-view-difficulty",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--guard-frames",
        type=int,
        default=DEFAULT_EVENT_WINDOW_FRAMES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_phase1_manifests(
        round7_dir=args.round7_dir,
        round8_dir=args.round8_dir,
        selection_manifest=args.selection_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        stress_per_experiment_view_difficulty=(
            args.stress_per_experiment_view_difficulty
        ),
        guard_frames=args.guard_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
