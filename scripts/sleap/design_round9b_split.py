"""Design the deterministic leakage-safe Round 9b side-label split.

The input review manifest contains 210 annotated source frames. This script
selects 30 primary validation frames with:

* exactly 10 source frames per experiment;
* the fixed six-category quota requested for Round 9b;
* no frame within ``+/-166`` of sanitized inherited training labels;
* no frame within ``+/-166`` of any remaining newly reviewed training frame;
* no overlap with the locked HITL or future-lock manifests.

Selection is deterministic: within an explicit experiment/category quota cell,
historically unexposed rows are ordered by descending
``selection_category_score`` and then ascending source frame.

The inherited Round 8 side package is already cumulative over useful Round 7
labels. Three known malformed source groups are excluded synchronously from
both side views. This command writes only small CSV/JSON design artifacts. It
does not write SLPs, touch shared storage, or launch training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.sleap.prepare_side_cumulative_training import (
    EXPERIMENTS,
    SIDE_VIEWS,
    _load_lookup,
    _package_summary,
    _structured_points,
    file_fingerprint,
    source_key,
)


SCHEMA_VERSION = "round9b-side-split-v1"
DEFAULT_GUARD_FRAMES = 166
CATEGORIES = (
    "clean_diverse_control",
    "close_contact_or_crossing",
    "count_partial_or_low_confidence",
    "long_body_crop_margin_proxy",
    "round7_round8_tail_disagreement",
    "temporal_or_shared_tail",
)
CATEGORY_TARGETS = {
    "clean_diverse_control": 6,
    "close_contact_or_crossing": 6,
    "count_partial_or_low_confidence": 2,
    "long_body_crop_margin_proxy": 4,
    "round7_round8_tail_disagreement": 6,
    "temporal_or_shared_tail": 6,
}
# K_1_2 has exactly ten historically unexposed reviewed frames. Its row is
# therefore forced by availability. The remaining targets are divided across
# K_1_1/K_1_3 while retaining every category in both experiments.
EXPERIMENT_CATEGORY_QUOTAS = {
    "20251225_K_1_1_FCMC": {
        "clean_diverse_control": 1,
        "close_contact_or_crossing": 2,
        "count_partial_or_low_confidence": 1,
        "long_body_crop_margin_proxy": 1,
        "round7_round8_tail_disagreement": 3,
        "temporal_or_shared_tail": 2,
    },
    "20251225_K_1_2_FCMC": {
        "clean_diverse_control": 4,
        "close_contact_or_crossing": 2,
        "count_partial_or_low_confidence": 0,
        "long_body_crop_margin_proxy": 1,
        "round7_round8_tail_disagreement": 1,
        "temporal_or_shared_tail": 2,
    },
    "20251225_K_1_3_FCMC": {
        "clean_diverse_control": 1,
        "close_contact_or_crossing": 2,
        "count_partial_or_low_confidence": 1,
        "long_body_crop_margin_proxy": 2,
        "round7_round8_tail_disagreement": 2,
        "temporal_or_shared_tail": 2,
    },
}
MALFORMED_INHERITED_SOURCE_FRAMES = (
    ("20251225_K_1_2_FCMC", 135885),
    ("20251225_K_1_2_FCMC", 281988),
    ("20251225_K_1_3_FCMC", 92579),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_row_id(experiment: str, frame: int) -> str:
    payload = f"{SCHEMA_VERSION}|{experiment}|{int(frame)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _validate_quota_matrix() -> None:
    if set(EXPERIMENT_CATEGORY_QUOTAS) != set(EXPERIMENTS):
        raise RuntimeError("experiment/category quota matrix is incomplete")
    for experiment, quotas in EXPERIMENT_CATEGORY_QUOTAS.items():
        if set(quotas) != set(CATEGORIES):
            raise RuntimeError(
                f"category quota row is incomplete for {experiment}"
            )
        if sum(quotas.values()) != 10:
            raise RuntimeError(
                f"category quota row does not sum to 10 for {experiment}"
            )
    observed = {
        category: sum(
            EXPERIMENT_CATEGORY_QUOTAS[experiment][category]
            for experiment in EXPERIMENTS
        )
        for category in CATEGORIES
    }
    if observed != CATEGORY_TARGETS:
        raise RuntimeError(
            f"category quota columns do not match targets: {observed}"
        )


def _load_selection_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames, raw_rows = _read_csv(path)
    required = {
        "experiment",
        "frame",
        "split_role",
        "selection_category",
        "selection_category_score",
    }
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"selection manifest is missing columns: {sorted(missing)}"
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
        row = dict(raw)
        experiment = str(row["experiment"]).strip()
        frame = int(row["frame"])
        category = str(row["selection_category"]).strip()
        if experiment not in EXPERIMENTS:
            raise ValueError(f"unsupported experiment: {experiment}")
        if category not in CATEGORIES:
            raise ValueError(f"unsupported selection category: {category}")
        key = (experiment, frame)
        if key in seen:
            raise ValueError(f"duplicate selected source frame: {key}")
        seen.add(key)
        row["experiment"] = experiment
        row["frame"] = frame
        row["selection_category_score"] = float(
            row["selection_category_score"]
        )
        rows.append(row)
    if len(rows) != 210:
        raise ValueError(f"selection manifest must contain 210 rows, got {len(rows)}")
    experiment_counts = Counter(row["experiment"] for row in rows)
    if experiment_counts != Counter(
        {
            "20251225_K_1_1_FCMC": 80,
            "20251225_K_1_2_FCMC": 50,
            "20251225_K_1_3_FCMC": 80,
        }
    ):
        raise ValueError(
            f"unexpected selection counts by experiment: {experiment_counts}"
        )
    return fieldnames, rows


def _nearest_inherited(
    experiment: str,
    frame: int,
    inherited_by_experiment: Mapping[str, Sequence[int]],
) -> tuple[int | None, int | None]:
    candidates = inherited_by_experiment.get(experiment, ())
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda candidate: abs(frame - candidate))
    return int(nearest), int(abs(frame - nearest))


def _locked_intervals(
    path: Path,
    *,
    guard_frames: int,
    eligibility_column: str | None = None,
) -> tuple[list[tuple[str, int, int]], int]:
    fieldnames, rows = _read_csv(path)
    required = {"experiment"}
    if not required.issubset(fieldnames):
        raise ValueError(f"locked manifest is missing experiment: {path}")
    frame_column = (
        "frame_idx"
        if "frame_idx" in fieldnames
        else "frame"
        if "frame" in fieldnames
        else None
    )
    if frame_column is None:
        raise ValueError(f"locked manifest has no frame column: {path}")
    if eligibility_column is not None:
        if eligibility_column not in fieldnames:
            raise ValueError(
                f"locked manifest lacks {eligibility_column}: {path}"
            )
        rows = [row for row in rows if _as_bool(row[eligibility_column])]
    intervals = []
    for row in rows:
        frame = int(row[frame_column])
        start_value = row.get("event_start_frame", "")
        end_value = row.get("event_end_frame", "")
        start = frame if str(start_value).strip() == "" else int(start_value)
        end = frame if str(end_value).strip() == "" else int(end_value)
        intervals.append(
            (
                str(row["experiment"]),
                min(start, end) - int(guard_frames),
                max(start, end) + int(guard_frames),
            )
        )
    return intervals, len(rows)


def _interval_violations(
    rows: Iterable[Mapping[str, Any]],
    intervals: Sequence[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    violations = []
    for row in rows:
        experiment = str(row["experiment"])
        frame = int(row["frame"])
        for locked_experiment, start, end in intervals:
            if experiment == locked_experiment and start <= frame <= end:
                violations.append(
                    {
                        "experiment": experiment,
                        "frame": frame,
                        "blocked_start": int(start),
                        "blocked_end": int(end),
                    }
                )
    return violations


def _source_guard_violations(
    validation_rows: Iterable[Mapping[str, Any]],
    training_sources: Iterable[tuple[str, int]],
    *,
    guard_frames: int,
) -> list[dict[str, Any]]:
    training_by_experiment: dict[str, list[int]] = defaultdict(list)
    for experiment, frame in set(training_sources):
        training_by_experiment[experiment].append(int(frame))
    violations = []
    for row in validation_rows:
        experiment = str(row["experiment"])
        validation_frame = int(row["frame"])
        for training_frame in training_by_experiment.get(experiment, []):
            distance = abs(validation_frame - training_frame)
            if distance <= int(guard_frames):
                violations.append(
                    {
                        "experiment": experiment,
                        "validation_frame": validation_frame,
                        "training_frame": int(training_frame),
                        "distance_frames": int(distance),
                    }
                )
    return violations


def _visibility_summary(lookup: Mapping[tuple[str, str, int], Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for labeled_frame in lookup.values():
        counts["labeled_frames"] += 1
        counts["instances"] += len(labeled_frame.instances)
        all_mouths = True
        any_mouth = False
        for instance in labeled_frame.instances:
            xy, visible = _structured_points(instance)
            finite = np.isfinite(xy).all(axis=1)
            mouth = bool(finite[0] and visible[0])
            tail = bool(finite[1] and visible[1])
            counts["visible_finite_mouths"] += int(mouth)
            counts["hidden_finite_mouths"] += int(finite[0] and not visible[0])
            counts["raw_missing_mouths"] += int(not finite[0])
            counts["visible_finite_tails"] += int(tail)
            counts["hidden_finite_tails"] += int(finite[1] and not visible[1])
            counts["raw_missing_tails"] += int(not finite[1])
            all_mouths = all_mouths and mouth
            any_mouth = any_mouth or mouth
        counts["frames_all_mouths_visible"] += int(all_mouths)
        counts["frames_any_mouth_visible"] += int(any_mouth)
    return {key: int(value) for key, value in counts.items()}


def _validate_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"output directory must be absent or empty: {output_dir}"
        )


def design_round9b_split(
    *,
    selection_manifest: Path,
    inherited_train_labels: Path,
    reviewed_labels: Path,
    locked_manifest: Path,
    future_locked_manifest: Path,
    output_dir: Path,
    guard_frames: int = DEFAULT_GUARD_FRAMES,
) -> dict[str, Any]:
    """Select and write the deterministic Round 9b CSV/JSON design artifacts."""
    _validate_quota_matrix()
    if int(guard_frames) < 0:
        raise ValueError("guard_frames must be non-negative")
    inputs = (
        selection_manifest,
        inherited_train_labels,
        reviewed_labels,
        locked_manifest,
        future_locked_manifest,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_empty_output_dir(output_dir)

    original_fields, rows = _load_selection_rows(selection_manifest)
    _inherited_labels, inherited_lookup, inherited_summary = _load_lookup(
        inherited_train_labels,
        require_three_instances=False,
        require_raw_finite=False,
    )
    _reviewed_labels, reviewed_lookup, reviewed_summary = _load_lookup(
        reviewed_labels,
        require_three_instances=True,
        require_raw_finite=True,
    )
    expected_reviewed_keys = {
        (str(row["experiment"]), view, int(row["frame"]))
        for row in rows
        for view in SIDE_VIEWS
    }
    if set(reviewed_lookup) != expected_reviewed_keys:
        missing = sorted(expected_reviewed_keys - set(reviewed_lookup))
        extra = sorted(set(reviewed_lookup) - expected_reviewed_keys)
        raise ValueError(
            "reviewed-label coverage differs from selection manifest; "
            f"missing={missing[:12]}, extra={extra[:12]}"
        )

    malformed_expected = set(MALFORMED_INHERITED_SOURCE_FRAMES)
    malformed_observed = {
        source_key(key)
        for key, labeled_frame in inherited_lookup.items()
        if len(labeled_frame.instances) != 3
    }
    if malformed_observed != malformed_expected:
        raise ValueError(
            "inherited non-three-instance source groups changed; "
            f"expected={sorted(malformed_expected)}, "
            f"observed={sorted(malformed_observed)}"
        )
    for malformed_key in malformed_expected:
        present_views = {
            key[1]
            for key in inherited_lookup
            if source_key(key) == malformed_key
        }
        if present_views != set(SIDE_VIEWS):
            raise ValueError(
                f"malformed source group lacks synchronized sides: "
                f"{malformed_key}, views={sorted(present_views)}"
            )
    malformed_lookup = {
        key: labeled_frame
        for key, labeled_frame in inherited_lookup.items()
        if source_key(key) in malformed_expected
    }
    inherited_sanitized = {
        key: labeled_frame
        for key, labeled_frame in inherited_lookup.items()
        if source_key(key) not in malformed_expected
    }
    inherited_sources = {source_key(key) for key in inherited_sanitized}
    inherited_by_experiment: dict[str, list[int]] = defaultdict(list)
    for experiment, frame in inherited_sources:
        inherited_by_experiment[experiment].append(int(frame))

    original_exposed = 0
    eligible_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        nearest_frame, nearest_distance = _nearest_inherited(
            str(row["experiment"]),
            int(row["frame"]),
            inherited_by_experiment,
        )
        row["inherited_nearest_frame"] = nearest_frame
        row["inherited_nearest_distance_frames"] = nearest_distance
        exposed = (
            nearest_distance is not None
            and nearest_distance <= int(guard_frames)
        )
        row["historically_exposed_within_guard"] = bool(exposed)
        original_exposed += int(exposed)
        if not exposed:
            eligible_by_cell[
                (str(row["experiment"]), str(row["selection_category"]))
            ].append(row)

    selected_keys: set[tuple[str, int]] = set()
    category_rank_by_key: dict[tuple[str, int], int] = {}
    for experiment in EXPERIMENTS:
        for category in CATEGORIES:
            wanted = EXPERIMENT_CATEGORY_QUOTAS[experiment][category]
            candidates = sorted(
                eligible_by_cell[(experiment, category)],
                key=lambda row: (
                    -float(row["selection_category_score"]),
                    int(row["frame"]),
                ),
            )
            if len(candidates) < wanted:
                raise ValueError(
                    f"insufficient eligible rows for {experiment}/{category}: "
                    f"wanted={wanted}, available={len(candidates)}"
                )
            for rank, row in enumerate(candidates[:wanted], start=1):
                key = (experiment, int(row["frame"]))
                selected_keys.add(key)
                category_rank_by_key[key] = rank
    if len(selected_keys) != 30:
        raise RuntimeError(
            f"Round 9b validation selection has {len(selected_keys)} rows"
        )

    output_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["experiment"]), int(row["frame"]))
        out = dict(row)
        out["original_split_role"] = str(row["split_role"])
        out["round9b_split_role"] = (
            "validation" if key in selected_keys else "train"
        )
        out["round9b_row_id"] = _stable_row_id(*key)
        out["round9b_guard_frames"] = int(guard_frames)
        out["round9b_category_quota"] = (
            EXPERIMENT_CATEGORY_QUOTAS[key[0]][
                str(row["selection_category"])
            ]
        )
        out["round9b_category_rank"] = category_rank_by_key.get(key, "")
        out["round9b_selection_reason"] = (
            "highest_score_historically_unexposed_within_experiment_category_quota"
            if key in selected_keys
            else "remaining_reviewed_annotation_for_training"
        )
        output_rows.append(out)
        (validation_rows if key in selected_keys else training_rows).append(out)

    new_training_sources = {
        (str(row["experiment"]), int(row["frame"])) for row in training_rows
    }
    inherited_violations = _source_guard_violations(
        validation_rows,
        inherited_sources,
        guard_frames=int(guard_frames),
    )
    new_violations = _source_guard_violations(
        validation_rows,
        new_training_sources,
        guard_frames=int(guard_frames),
    )
    if inherited_violations or new_violations:
        raise RuntimeError(
            "Round 9b train/validation guard verification failed; "
            f"inherited={inherited_violations[:8]}, new={new_violations[:8]}"
        )

    locked_intervals, locked_rows = _locked_intervals(
        locked_manifest, guard_frames=int(guard_frames)
    )
    future_intervals, future_rows = _locked_intervals(
        future_locked_manifest,
        guard_frames=int(guard_frames),
        eligibility_column="future_lock_eligible",
    )
    locked_violations = _interval_violations(output_rows, locked_intervals)
    future_violations = _interval_violations(output_rows, future_intervals)
    if locked_violations or future_violations:
        raise RuntimeError(
            "Round 9b rows enter a locked guard; "
            f"locked={locked_violations[:8]}, future={future_violations[:8]}"
        )

    reviewed_train_lookup = {
        key: labeled_frame
        for key, labeled_frame in reviewed_lookup.items()
        if source_key(key) in new_training_sources
    }
    reviewed_val_lookup = {
        key: labeled_frame
        for key, labeled_frame in reviewed_lookup.items()
        if source_key(key) in selected_keys
    }
    cumulative_train_lookup = dict(inherited_sanitized)
    collisions = sorted(
        set(cumulative_train_lookup) & set(reviewed_train_lookup)
    )
    cumulative_train_lookup.update(reviewed_train_lookup)

    validation_by_experiment = Counter(
        str(row["experiment"]) for row in validation_rows
    )
    validation_by_category = Counter(
        str(row["selection_category"]) for row in validation_rows
    )
    if validation_by_experiment != Counter(
        {experiment: 10 for experiment in EXPERIMENTS}
    ):
        raise RuntimeError(
            f"validation experiment quotas failed: {validation_by_experiment}"
        )
    if validation_by_category != Counter(CATEGORY_TARGETS):
        raise RuntimeError(
            f"validation category quotas failed: {validation_by_category}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    added_fields = [
        "original_split_role",
        "round9b_split_role",
        "round9b_row_id",
        "round9b_guard_frames",
        "round9b_category_quota",
        "round9b_category_rank",
        "round9b_selection_reason",
        "inherited_nearest_frame",
        "inherited_nearest_distance_frames",
        "historically_exposed_within_guard",
    ]
    output_fields = list(original_fields)
    for field in added_fields:
        if field not in output_fields:
            output_fields.append(field)
    selection_output = output_dir / "round9b_selection_210.csv"
    validation_output = output_dir / "round9b_validation_30.csv"
    training_output = output_dir / "round9b_train_180.csv"
    _write_csv(selection_output, output_rows, output_fields)
    _write_csv(validation_output, validation_rows, output_fields)
    _write_csv(training_output, training_rows, output_fields)

    original_validation_exposed = sum(
        bool(row["historically_exposed_within_guard"])
        and str(row["split_role"]) == "validation"
        for row in rows
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "policy": {
            "guard_frames": int(guard_frames),
            "validation_source_frames": 30,
            "validation_source_frames_per_experiment": 10,
            "category_targets": CATEGORY_TARGETS,
            "experiment_category_quotas": EXPERIMENT_CATEGORY_QUOTAS,
            "selection_order": (
                "selection_category_score descending, frame ascending"
            ),
            "inherited_policy": (
                "Round8 side train is cumulative over useful Round7 labels"
            ),
            "new_labels_win_exact_collisions": True,
        },
        "inputs": {
            "selection_manifest": file_fingerprint(selection_manifest),
            "inherited_train_labels": file_fingerprint(
                inherited_train_labels
            ),
            "reviewed_labels": file_fingerprint(reviewed_labels),
            "locked_manifest": file_fingerprint(locked_manifest),
            "future_locked_manifest": file_fingerprint(
                future_locked_manifest
            ),
        },
        "inherited_training": {
            "input": inherited_summary,
            "malformed_source_groups_removed": [
                {"experiment": experiment, "frame": int(frame)}
                for experiment, frame in sorted(malformed_expected)
            ],
            "malformed_removed": _package_summary(malformed_lookup),
            "sanitized": _package_summary(inherited_sanitized),
        },
        "reviewed_labels": {
            "input": reviewed_summary,
            "round9b_train": _visibility_summary(reviewed_train_lookup),
            "round9b_validation": _visibility_summary(reviewed_val_lookup),
        },
        "split_counts": {
            "train_source_frames": len(training_rows),
            "train_side_labeled_frames": len(reviewed_train_lookup),
            "validation_source_frames": len(validation_rows),
            "validation_side_labeled_frames": len(reviewed_val_lookup),
            "validation_by_experiment": dict(
                sorted(validation_by_experiment.items())
            ),
            "validation_by_category": {
                category: int(validation_by_category[category])
                for category in CATEGORIES
            },
            "original_validation_historically_exposed": int(
                original_validation_exposed
            ),
            "round9b_validation_historically_exposed": 0,
            "all_210_historically_exposed": int(original_exposed),
        },
        "cumulative_training_counts": _visibility_summary(
            cumulative_train_lookup
        ),
        "collisions": {
            "exact_labeled_frame_collisions_new_won": len(collisions),
            "keys": [list(key) for key in collisions],
        },
        "verification": {
            "validation_vs_inherited_train_guard_violations": [],
            "validation_vs_new_train_guard_violations": [],
            "locked_hitl_rows": int(locked_rows),
            "locked_hitl_guard_violations": [],
            "future_locked_eligible_rows": int(future_rows),
            "future_locked_guard_violations": [],
            "locked_artifacts_modified": False,
            "shared_storage_artifacts_written": False,
            "training_launched": False,
        },
        "outputs": {},
    }
    report["outputs"] = {
        "selection_210": file_fingerprint(selection_output),
        "validation_30": file_fingerprint(validation_output),
        "train_180": file_fingerprint(training_output),
    }
    report_path = output_dir / "round9b_split_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--inherited-train-labels", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--future-locked-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = design_round9b_split(
        selection_manifest=args.selection_manifest,
        inherited_train_labels=args.inherited_train_labels,
        reviewed_labels=args.reviewed_labels,
        locked_manifest=args.locked_manifest,
        future_locked_manifest=args.future_locked_manifest,
        output_dir=args.output_dir,
        guard_frames=args.guard_frames,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
