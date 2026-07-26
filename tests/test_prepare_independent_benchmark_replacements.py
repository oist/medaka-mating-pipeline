from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.sleap.prepare_independent_benchmark_labels import (
    expanded_interval,
    intervals_overlap,
    sha256_file,
)
from scripts.sleap.prepare_independent_benchmark_replacements import (
    DEFAULT_FUTURE_QUEUE,
    DEFAULT_INVALID_FRAMES,
    DEFAULT_SEED,
    DEFAULT_V1_MANIFEST,
    TARGET_EXPERIMENT,
    build_replacement_plan,
    parse_invalid_frames,
    write_replacement_bundle,
)


def test_real_v1_replacements_are_deterministic_and_preserve_quotas() -> None:
    manifest = pd.read_csv(DEFAULT_V1_MANIFEST, keep_default_na=False)
    queue = pd.read_csv(DEFAULT_FUTURE_QUEUE, keep_default_na=False)
    kwargs = {
        "invalid_frames": parse_invalid_frames(DEFAULT_INVALID_FRAMES),
        "seed": DEFAULT_SEED,
        "guard_frames": 166,
    }
    first = build_replacement_plan(manifest, queue, **kwargs)
    second = build_replacement_plan(manifest, queue, **kwargs)
    first_v2, first_replacements, first_lineage, validation = first
    second_v2, second_replacements, second_lineage, _ = second

    pd.testing.assert_frame_equal(first_v2, second_v2)
    pd.testing.assert_frame_equal(first_replacements, second_replacements)
    assert first_lineage == second_lineage
    assert set(first_replacements["frame_idx"].astype(int)) == {19323, 27474}
    assert first_replacements["replacement_rank"].astype(int).tolist() == [1, 2]
    assert validation["quotas_before"] == validation["quotas_after"]
    assert len(first_v2) == 40
    assert set(first_v2.loc[
        first_v2["experiment"].eq(TARGET_EXPERIMENT), "frame_idx"
    ].astype(int)).isdisjoint({1207, 4626})


def _synthetic_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quotas = {"K1": 14, TARGET_EXPERIMENT: 13, "K3": 13}
    for experiment, count in quotas.items():
        for index in range(count):
            frame = (index + 1) * 1000
            rows.append(
                {
                    "experiment": experiment,
                    "frame_idx": frame,
                    "event_id": f"{experiment}_event_{index}",
                    "event_start_frame": frame,
                    "event_end_frame": frame,
                    "guard_frames": 10,
                    "frame_key": f"{experiment}_{frame}",
                    "difficulty_hint": "test",
                    "source_category": "test",
                    "selection_source": "test",
                    "nearest_exposed_frame_distance": 100,
                    "split_role": "locked_test_candidate_pending_annotation",
                    "annotation_source": "blank_manual_independent",
                }
            )
    return pd.DataFrame(rows)


def test_replacement_planner_skips_reserved_and_candidate_collisions() -> None:
    manifest = _synthetic_manifest()
    queue_rows = [
        # Always invalid: overlaps the reserved v1 event at frame 1000.
        ("reserved_collision", 1005, 1005, 1005),
        ("candidate_a", 20000, 20000, 20000),
        # Its expanded interval overlaps candidate_a's expanded interval.
        ("candidate_a_collision", 20015, 20015, 20015),
        ("candidate_b", 30000, 30000, 30000),
        ("candidate_c", 40000, 40000, 40000),
    ]
    queue = pd.DataFrame(
        [
            {
                "experiment": TARGET_EXPERIMENT,
                "view": "top_cam",
                "frame_idx": frame,
                "event_id": event_id,
                "event_start_frame": start,
                "event_end_frame": end,
                "difficulty": "test",
                "source_category": "test",
                "future_lock_eligible": True,
            }
            for event_id, frame, start, end in queue_rows
        ]
    )
    _, replacements, _, _ = build_replacement_plan(
        manifest,
        queue,
        invalid_frames=[
            (TARGET_EXPERIMENT, 1000),
            (TARGET_EXPERIMENT, 2000),
        ],
        seed=7,
        guard_frames=10,
    )

    replacement_records = replacements.to_dict("records")
    replacement_intervals = [
        expanded_interval(row, 10) for row in replacement_records
    ]
    assert len(replacement_intervals) == 2
    assert not intervals_overlap(
        replacement_intervals[0], replacement_intervals[1]
    )
    reserved_intervals = [
        expanded_interval(row, 10)
        for row in manifest.loc[
            manifest["experiment"].eq(TARGET_EXPERIMENT)
        ].to_dict("records")
    ]
    assert all(
        not intervals_overlap(replacement, reserved)
        for replacement in replacement_intervals
        for reserved in reserved_intervals
    )


def test_bundle_writer_refuses_existing_output_without_touching_v1(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "hitl-v2-existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    source_hash = sha256_file(DEFAULT_V1_MANIFEST)

    with pytest.raises(FileExistsError, match="refusing existing output"):
        write_replacement_bundle(
            v1_manifest_path=DEFAULT_V1_MANIFEST,
            future_queue_path=DEFAULT_FUTURE_QUEUE,
            invalid_frames=parse_invalid_frames(DEFAULT_INVALID_FRAMES),
            seed=DEFAULT_SEED,
            guard_frames=166,
            output_dir=output_dir,
            raw_data_dir=tmp_path / "unused-raw-data",
        )

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert sha256_file(DEFAULT_V1_MANIFEST) == source_hash
