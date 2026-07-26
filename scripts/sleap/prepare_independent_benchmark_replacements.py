"""Build an immutable HITL-v2 replacement package for invalid benchmark rows.

The v1 manifest remains the selection record of origin. This tool replaces
explicitly invalid K_1_2 rows with the next stable-hash-ranked, future-lock
eligible events while reserving every original K_1_2 event interval. The SLP
output is deliberately blank and contains only the replacement source frames
in all three views; it is a candidate-generation input, not a locked test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths
from scripts.sleap.prepare_independent_benchmark_labels import (
    DEFAULT_SEED,
    build_blank_package,
    expanded_interval,
    intervals_overlap,
    reserve_new_output_dir,
    sha256_file,
    stable_hex,
)


VERSION = "shared-benchmark40-hitl-v2"
TARGET_EXPERIMENT = "20251225_K_1_2_FCMC"
EXPECTED_SOURCE_FRAMES = 40
DEFAULT_GUARD_FRAMES = 166
DEFAULT_INVALID_FRAMES = (
    f"{TARGET_EXPERIMENT}:1207",
    f"{TARGET_EXPERIMENT}:4626",
)
DEFAULT_INVALID_REASON = "invalid_hitl_annotation"
DEFAULT_V1_MANIFEST = Path(
    "report/20260725/sleap_independent_benchmark_annotation/"
    "independent_annotation_40_manifest.csv"
)
DEFAULT_FUTURE_QUEUE = Path(
    "report/20260724/sleap_topdown_optimization/"
    "future_locked_test_annotation_queue_eligible_v1.csv"
)

V2_MANIFEST_FILENAME = "shared_benchmark_40_hitl_v2_manifest.csv"
V2_BLANK_PACKAGE_FILENAME = (
    "shared_benchmark_40_hitl_v2_replacements_blank.pkg.slp"
)
V2_PROVENANCE_FILENAME = "shared_benchmark_40_hitl_v2_provenance.json"

MANIFEST_REQUIRED_COLUMNS = {
    "experiment",
    "frame_idx",
    "event_id",
    "event_start_frame",
    "event_end_frame",
    "frame_key",
}
QUEUE_REQUIRED_COLUMNS = {
    "experiment",
    "view",
    "frame_idx",
    "event_id",
    "event_start_frame",
    "event_end_frame",
    "future_lock_eligible",
}
LINEAGE_COLUMNS = (
    "hitl_manifest_version",
    "replacement_status",
    "superseded_frame_key",
    "superseded_frame_idx",
    "superseded_event_id",
    "replacement_rank",
    "replacement_reason",
)


def parse_invalid_frames(specs: Iterable[str]) -> list[tuple[str, int]]:
    """Parse repeatable ``EXPERIMENT:FRAME`` selectors without ambiguity."""
    parsed: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for spec in specs:
        experiment, separator, frame_text = str(spec).rpartition(":")
        if not separator or not experiment or not frame_text:
            raise ValueError(
                f"invalid frame selector {spec!r}; expected EXPERIMENT:FRAME"
            )
        try:
            frame_idx = int(frame_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid frame selector {spec!r}; FRAME must be an integer"
            ) from exc
        if frame_idx < 0:
            raise ValueError(
                f"invalid frame selector {spec!r}; FRAME must be non-negative"
            )
        key = (experiment, frame_idx)
        if key in seen:
            raise ValueError(f"duplicate invalid frame selector: {spec}")
        parsed.append(key)
        seen.add(key)
    if not parsed:
        raise ValueError("at least one invalid frame selector is required")
    return parsed


def _truthy(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _canonical_future_event_rows(
    queue: pd.DataFrame,
    *,
    experiment: str,
    seed: int,
) -> list[dict[str, object]]:
    """Return one representative per event using the v1 selection rule."""
    missing = QUEUE_REQUIRED_COLUMNS.difference(queue.columns)
    if missing:
        raise ValueError(
            "future queue is missing columns: " + ", ".join(sorted(missing))
        )
    eligible = queue.loc[
        _truthy(queue["future_lock_eligible"])
        & queue["experiment"].astype(str).eq(experiment)
        & queue["view"].astype(str).eq("top_cam")
    ].copy()
    if eligible.empty:
        raise ValueError(
            f"future queue has no eligible top_cam rows for {experiment}"
        )

    event_rows: list[dict[str, object]] = []
    for event_id, group in eligible.groupby("event_id", sort=True):
        starts = {int(value) for value in group["event_start_frame"]}
        ends = {int(value) for value in group["event_end_frame"]}
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                f"event {event_id!r} has inconsistent interval metadata"
            )
        start = starts.pop()
        end = ends.pop()
        if end < start:
            raise ValueError(f"event {event_id!r} ends before it starts")
        midpoint = (start + end) / 2.0
        candidates = sorted(
            group.to_dict("records"),
            key=lambda row: (
                not (start <= int(row["frame_idx"]) <= end),
                abs(int(row["frame_idx"]) - midpoint),
                stable_hex(seed, event_id, row["frame_idx"]),
            ),
        )
        source = candidates[0]
        event_rows.append(
            {
                "experiment": experiment,
                "frame_idx": int(source["frame_idx"]),
                "event_id": str(event_id),
                "event_start_frame": start,
                "event_end_frame": end,
                "difficulty_hint": str(
                    source.get("difficulty", "unknown_until_annotation")
                ),
                "source_category": str(source.get("source_category", "")),
                "selection_source": "future_lock_eligible_queue",
                "stable_rank_hash": stable_hex(seed, experiment, event_id),
            }
        )
    return sorted(event_rows, key=lambda row: str(row["stable_rank_hash"]))


def _validate_manifest(
    manifest: pd.DataFrame,
    *,
    guard_frames: int,
) -> pd.DataFrame:
    missing = MANIFEST_REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(
            "v1 manifest is missing columns: " + ", ".join(sorted(missing))
        )
    if len(manifest) != EXPECTED_SOURCE_FRAMES:
        raise ValueError(
            f"v1 manifest must contain {EXPECTED_SOURCE_FRAMES} rows; "
            f"found {len(manifest)}"
        )
    validated = manifest.copy()
    validated["experiment"] = validated["experiment"].astype(str)
    validated["frame_idx"] = validated["frame_idx"].astype(int)
    validated["event_start_frame"] = validated["event_start_frame"].astype(int)
    validated["event_end_frame"] = validated["event_end_frame"].astype(int)
    validated["event_id"] = validated["event_id"].astype(str)
    if validated["frame_key"].astype(str).duplicated().any():
        raise ValueError("v1 manifest contains duplicate frame_key values")
    if validated.duplicated(["experiment", "frame_idx"]).any():
        raise ValueError("v1 manifest contains duplicate experiment/frame rows")
    if "guard_frames" in validated:
        guards = {int(value) for value in validated["guard_frames"]}
        if guards != {guard_frames}:
            raise ValueError(
                "requested guard does not match the v1 manifest: "
                f"requested {guard_frames}, manifest has {sorted(guards)}"
            )
    return validated


def _quota_counts(manifest: pd.DataFrame) -> dict[str, int]:
    counts = manifest.groupby("experiment", sort=True).size()
    return {str(experiment): int(count) for experiment, count in counts.items()}


def _assert_event_disjoint(
    manifest: pd.DataFrame,
    *,
    experiment: str,
    guard_frames: int,
) -> None:
    rows = manifest.loc[manifest["experiment"].astype(str).eq(experiment)]
    records = rows.to_dict("records")
    intervals = [expanded_interval(row, guard_frames) for row in records]
    for first_index, first in enumerate(intervals):
        for second_index in range(first_index + 1, len(intervals)):
            if intervals_overlap(first, intervals[second_index]):
                raise RuntimeError(
                    "replacement manifest violates the guarded event-disjoint "
                    f"constraint: {records[first_index]['event_id']} overlaps "
                    f"{records[second_index]['event_id']}"
                )


def build_replacement_plan(
    v1_manifest: pd.DataFrame,
    future_queue: pd.DataFrame,
    *,
    invalid_frames: Iterable[tuple[str, int]],
    seed: int,
    guard_frames: int,
    reason: str = DEFAULT_INVALID_REASON,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    """Build the full v2 manifest and replacement-only package manifest."""
    if guard_frames < 0:
        raise ValueError("guard_frames must be non-negative")
    if not reason.strip():
        raise ValueError("replacement reason must not be blank")
    manifest = _validate_manifest(v1_manifest, guard_frames=guard_frames)
    invalid = list(invalid_frames)
    if not invalid:
        raise ValueError("at least one invalid frame is required")
    if len(set(invalid)) != len(invalid):
        raise ValueError("invalid frame selectors must be unique")
    wrong_experiments = sorted(
        {experiment for experiment, _frame in invalid}
        - {TARGET_EXPERIMENT}
    )
    if wrong_experiments:
        raise ValueError(
            "replacement planning is restricted to "
            f"{TARGET_EXPERIMENT}; got {', '.join(wrong_experiments)}"
        )

    invalid_indices: list[int] = []
    for experiment, frame_idx in invalid:
        matches = manifest.index[
            manifest["experiment"].eq(experiment)
            & manifest["frame_idx"].eq(frame_idx)
        ].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"invalid frame {experiment}:{frame_idx} occurs "
                f"{len(matches)} times in the v1 manifest"
            )
        invalid_indices.append(int(matches[0]))

    original_target = manifest.loc[
        manifest["experiment"].eq(TARGET_EXPERIMENT)
    ]
    reserved_event_ids = set(original_target["event_id"].astype(str))
    reserved_intervals = [
        expanded_interval(row, guard_frames)
        for row in original_target.to_dict("records")
    ]
    ranked_events = _canonical_future_event_rows(
        future_queue,
        experiment=TARGET_EXPERIMENT,
        seed=seed,
    )

    selected: list[dict[str, object]] = []
    selected_intervals: list[tuple[int, int]] = []
    for queue_rank, row in enumerate(ranked_events, start=1):
        interval = expanded_interval(row, guard_frames)
        if str(row["event_id"]) in reserved_event_ids:
            continue
        if any(
            intervals_overlap(interval, reserved)
            for reserved in reserved_intervals
        ):
            continue
        if any(
            intervals_overlap(interval, other)
            for other in selected_intervals
        ):
            continue
        replacement = dict(row)
        replacement["eligible_queue_stable_rank"] = queue_rank
        replacement["replacement_rank"] = len(selected) + 1
        selected.append(replacement)
        selected_intervals.append(interval)
        if len(selected) == len(invalid_indices):
            break
    if len(selected) != len(invalid_indices):
        raise RuntimeError(
            f"only {len(selected)} guarded, non-overlapping replacement events "
            f"were available; requested {len(invalid_indices)}"
        )

    v2 = manifest.copy()
    if "nearest_exposed_frame_distance" in v2.columns:
        v2["nearest_exposed_frame_distance"] = v2[
            "nearest_exposed_frame_distance"
        ].astype(object)
    v2["hitl_manifest_version"] = VERSION
    v2["replacement_status"] = "retained_v1"
    for column in LINEAGE_COLUMNS[2:]:
        if column in {"superseded_frame_idx", "replacement_rank"}:
            v2[column] = pd.Series(pd.NA, index=v2.index, dtype="Int64")
        else:
            v2[column] = ""

    lineage: list[dict[str, object]] = []
    for source_index, replacement in zip(invalid_indices, selected):
        superseded = manifest.loc[source_index].to_dict()
        replacement_frame_key = "li2_" + stable_hex(
            VERSION,
            replacement["experiment"],
            replacement["frame_idx"],
        )[:20]
        updated = superseded.copy()
        for column in (
            "experiment",
            "frame_idx",
            "event_id",
            "event_start_frame",
            "event_end_frame",
            "difficulty_hint",
            "source_category",
            "selection_source",
        ):
            if column in v2.columns:
                updated[column] = replacement[column]
        if "nearest_exposed_frame_distance" in v2.columns:
            updated["nearest_exposed_frame_distance"] = ""
        if "guard_frames" in v2.columns:
            updated["guard_frames"] = guard_frames
        if "annotation_source" in v2.columns:
            updated["annotation_source"] = "blank_hitl_replacement_candidate_v2"
        updated.update(
            {
                "frame_key": replacement_frame_key,
                "hitl_manifest_version": VERSION,
                "replacement_status": "replacement_candidate",
                "superseded_frame_key": str(superseded["frame_key"]),
                "superseded_frame_idx": int(superseded["frame_idx"]),
                "superseded_event_id": str(superseded["event_id"]),
                "replacement_rank": int(replacement["replacement_rank"]),
                "replacement_reason": reason,
            }
        )
        for column, value in updated.items():
            if column in v2.columns:
                v2.at[source_index, column] = value
        lineage.append(
            {
                "superseded": {
                    "frame_key": str(superseded["frame_key"]),
                    "experiment": str(superseded["experiment"]),
                    "frame_idx": int(superseded["frame_idx"]),
                    "event_id": str(superseded["event_id"]),
                    "event_start_frame": int(
                        superseded["event_start_frame"]
                    ),
                    "event_end_frame": int(superseded["event_end_frame"]),
                },
                "replacement": {
                    "frame_key": replacement_frame_key,
                    "experiment": str(replacement["experiment"]),
                    "frame_idx": int(replacement["frame_idx"]),
                    "event_id": str(replacement["event_id"]),
                    "event_start_frame": int(
                        replacement["event_start_frame"]
                    ),
                    "event_end_frame": int(replacement["event_end_frame"]),
                    "stable_rank_hash": str(replacement["stable_rank_hash"]),
                },
                "replacement_rank": int(replacement["replacement_rank"]),
                "eligible_queue_stable_rank": int(
                    replacement["eligible_queue_stable_rank"]
                ),
                "reason": reason,
            }
        )

    v2 = v2.sort_values(
        ["experiment", "frame_idx"], kind="stable"
    ).reset_index(drop=True)
    if v2["frame_key"].astype(str).duplicated().any():
        raise RuntimeError("v2 manifest contains duplicate frame keys")
    if v2.duplicated(["experiment", "frame_idx"]).any():
        raise RuntimeError("v2 manifest contains duplicate source frames")
    quotas_before = _quota_counts(manifest)
    quotas_after = _quota_counts(v2)
    if quotas_after != quotas_before:
        raise RuntimeError(
            f"experiment quotas changed: {quotas_before} -> {quotas_after}"
        )
    _assert_event_disjoint(
        v2,
        experiment=TARGET_EXPERIMENT,
        guard_frames=guard_frames,
    )
    replacements = v2.loc[
        v2["replacement_status"].eq("replacement_candidate")
    ].copy()
    replacements["replacement_rank"] = replacements[
        "replacement_rank"
    ].astype(int)
    replacements = replacements.sort_values(
        "replacement_rank", kind="stable"
    ).reset_index(drop=True)
    validation: dict[str, object] = {
        "source_rows": len(manifest),
        "v2_rows": len(v2),
        "replacement_rows": len(replacements),
        "quotas_before": quotas_before,
        "quotas_after": quotas_after,
        "quotas_preserved": True,
        "target_event_intervals_disjoint_with_guard": True,
        "all_v1_target_event_intervals_reserved": True,
        "guard_frames": guard_frames,
    }
    return v2, replacements, lineage, validation


def write_replacement_bundle(
    *,
    v1_manifest_path: Path,
    future_queue_path: Path,
    invalid_frames: Iterable[tuple[str, int]],
    seed: int,
    guard_frames: int,
    output_dir: Path,
    raw_data_dir: Path,
    reason: str = DEFAULT_INVALID_REASON,
) -> dict[str, object]:
    """Write the immutable v2 manifest, replacement-only blank SLP and record."""
    if not v1_manifest_path.is_file():
        raise FileNotFoundError(f"v1 manifest not found: {v1_manifest_path}")
    if not future_queue_path.is_file():
        raise FileNotFoundError(f"future queue not found: {future_queue_path}")
    source_manifest_hash = sha256_file(v1_manifest_path)
    future_queue_hash = sha256_file(future_queue_path)
    v1_manifest = pd.read_csv(v1_manifest_path, keep_default_na=False)
    future_queue = pd.read_csv(future_queue_path, keep_default_na=False)
    v2, replacements, lineage, validation = build_replacement_plan(
        v1_manifest,
        future_queue,
        invalid_frames=invalid_frames,
        seed=seed,
        guard_frames=guard_frames,
        reason=reason,
    )

    reserve_new_output_dir(output_dir)
    manifest_path = output_dir / V2_MANIFEST_FILENAME
    blank_package_path = output_dir / V2_BLANK_PACKAGE_FILENAME
    provenance_path = output_dir / V2_PROVENANCE_FILENAME
    v2.to_csv(manifest_path, index=False, lineterminator="\n")

    provenance: dict[str, object] = {
        "schema_version": VERSION,
        "status": "replacement_candidates_pending_hitl_annotation",
        "selection_seed": seed,
        "guard_frames": guard_frames,
        "selection_rule": (
            "Use the v1 midpoint representative and stable SHA-256 event "
            "ranking. Reserve every v1 K_1_2 event interval expanded by the "
            "guard, then greedily take the next future-lock-eligible events "
            "whose expanded intervals do not overlap each other."
        ),
        "source_v1_manifest": {
            "path": str(v1_manifest_path.resolve()),
            "sha256": source_manifest_hash,
            "rows": len(v1_manifest),
            "immutable": True,
        },
        "future_queue": {
            "path": str(future_queue_path.resolve()),
            "sha256": future_queue_hash,
        },
        "v2_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "rows": len(v2),
        },
        "replacements": lineage,
        "validation": validation,
        "blank_package_scope": (
            "replacement source frames only; all three synchronized views"
        ),
        "contains_model_predictions": False,
        "contains_keypoint_coordinates": False,
    }
    package_summary = build_blank_package(
        replacements,
        raw_data_dir=raw_data_dir,
        output_path=blank_package_path,
        provenance=provenance,
    )
    provenance["blank_package"] = {
        "path": str(blank_package_path.resolve()),
        "sha256": sha256_file(blank_package_path),
        **package_summary,
    }
    if sha256_file(v1_manifest_path) != source_manifest_hash:
        raise RuntimeError("source v1 manifest changed while building v2")
    provenance["validation"]["source_v1_manifest_unchanged"] = True
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "blank_package": str(blank_package_path),
        "provenance": str(provenance_path),
        "replacement_frames": [
            int(value) for value in replacements["frame_idx"]
        ],
        **package_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--v1-manifest",
        "--manifest",
        dest="v1_manifest",
        type=Path,
        default=DEFAULT_V1_MANIFEST,
    )
    parser.add_argument(
        "--future-queue",
        type=Path,
        default=DEFAULT_FUTURE_QUEUE,
    )
    parser.add_argument(
        "--invalid-frame",
        action="append",
        metavar="EXPERIMENT:FRAME",
        help=(
            "Invalid v1 row to replace. Repeat for multiple rows. If omitted, "
            "the known K_1_2 frames 1207 and 4626 are used."
        ),
    )
    parser.add_argument(
        "--invalid-reason",
        default=DEFAULT_INVALID_REASON,
        help="Reason recorded for every superseded row.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--guard-frames",
        type=int,
        default=DEFAULT_GUARD_FRAMES,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--medaka-data-dir",
        type=Path,
        help="Override paths.medaka_data_dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = args.invalid_frame or list(DEFAULT_INVALID_FRAMES)
    try:
        invalid_frames = parse_invalid_frames(specs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    local = load_local_paths(args.paths_config)
    raw_data_dir = args.medaka_data_dir or (
        local.medaka_data_dir if local is not None else None
    )
    if raw_data_dir is None:
        raise SystemExit(
            "medaka_data_dir is required; pass --medaka-data-dir or define it "
            f"in {args.paths_config}"
        )
    try:
        result = write_replacement_bundle(
            v1_manifest_path=args.v1_manifest,
            future_queue_path=args.future_queue,
            invalid_frames=invalid_frames,
            seed=args.seed,
            guard_frames=args.guard_frames,
            output_dir=args.output_dir,
            raw_data_dir=raw_data_dir,
            reason=args.invalid_reason,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
