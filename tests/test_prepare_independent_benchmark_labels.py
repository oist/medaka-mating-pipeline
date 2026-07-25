from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.sleap.multiview_annotation_gui import (
    FishKeypoints,
    clone_fish_keypoints,
    fish_have_annotations,
)
from scripts.sleap.prepare_independent_benchmark_labels import (
    balanced_quotas,
    expanded_interval,
    intervals_overlap,
    select_future_queue_events,
    select_uniform_unexposed_frames,
)


def test_balanced_40_frame_quotas() -> None:
    assert balanced_quotas(40, ["K1", "K2", "K3"]) == {
        "K1": 14,
        "K2": 13,
        "K3": 13,
    }


def test_uniform_selection_is_deterministic_unexposed_and_disjoint() -> None:
    kwargs = {
        "experiment": "K1",
        "count": 6,
        "exposed_frames": {1000, 3000, 7000},
        "total_frames": 12000,
        "guard_frames": 100,
        "seed": 42,
    }
    first = select_uniform_unexposed_frames(**kwargs)
    second = select_uniform_unexposed_frames(**kwargs)
    assert first == second
    assert len(first) == 6
    for row in first:
        frame = int(row["frame_idx"])
        assert min(abs(frame - exposed) for exposed in kwargs["exposed_frames"]) > 100
    intervals = [expanded_interval(row, 100) for row in first]
    assert not any(
        intervals_overlap(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )


def test_future_queue_selects_one_row_per_nonoverlapping_event(
    tmp_path: Path,
) -> None:
    rows = []
    for event, start, end, frames in [
        ("event_a", 1000, 1010, [1001, 1005]),
        ("event_b", 2000, 2010, [2002]),
        ("event_c", 2075, 2080, [2077]),
        ("event_d", 3000, 3010, [3005]),
    ]:
        for frame in frames:
            rows.append(
                {
                    "experiment": "K2",
                    "view": "top_cam",
                    "frame_idx": frame,
                    "event_id": event,
                    "event_start_frame": start,
                    "event_end_frame": end,
                    "difficulty": "close_contact_or_crossing",
                    "source_category": "test",
                    "future_lock_eligible": True,
                }
            )
    queue = tmp_path / "queue.csv"
    pd.DataFrame(rows).to_csv(queue, index=False)
    selected = select_future_queue_events(
        queue,
        experiment="K2",
        count=3,
        exposed_frames={500},
        guard_frames=50,
        seed=7,
    )
    assert len(selected) == 3
    assert len({row["event_id"] for row in selected}) == 3
    intervals = [expanded_interval(row, 50) for row in selected]
    assert not any(
        intervals_overlap(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )


def test_annotation_cache_copy_is_independent() -> None:
    source = [FishKeypoints()]
    source[0].mouth["top_cam"] = (10.0, 20.0)
    copied = clone_fish_keypoints(source)
    assert fish_have_annotations(copied)
    copied[0].mouth["top_cam"] = (30.0, 40.0)
    assert source[0].mouth["top_cam"] == (10.0, 20.0)
    assert copied[0].mouth["top_cam"] == (30.0, 40.0)
