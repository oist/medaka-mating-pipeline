from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.sleap.benchmark_manifest import (
    assert_no_event_split_leakage,
    event_is_exposed,
    select_event_aware,
    stable_frame_key,
    stable_hash,
    summarize_instances,
)


class FakeInstance:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def numpy(self):
        return self._points


def test_stable_frame_key_and_hash_are_reproducible() -> None:
    first = stable_frame_key("exp", "side1_cam", "Z:/raw/a.mp4", 42)
    second = stable_frame_key("exp", "side1_cam", r"z:\raw\a.mp4", 42)
    assert first == second
    assert stable_hash("event", seed=7) == stable_hash("event", seed=7)
    assert stable_hash("event", seed=7) != stable_hash("event", seed=8)


def test_event_aware_selection_is_deterministic_and_one_per_event() -> None:
    rows = pd.DataFrame(
        {
            "frame_key": [f"k{index}" for index in range(8)],
            "event_id": ["a", "a", "b", "b", "c", "d", "e", "f"],
        }
    )
    first = select_event_aware(rows, limit=4, seed=17)
    second = select_event_aware(rows, limit=4, seed=17)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert len(first) == 4
    assert first["event_id"].nunique() == 4


def test_event_level_leakage_is_rejected() -> None:
    safe = pd.DataFrame(
        {
            "experiment": ["exp", "exp", "exp"],
            "event_id": ["a", "a", "b"],
            "split_role": ["train", "train", "test"],
        }
    )
    assert_no_event_split_leakage(safe)
    leaked = safe.copy()
    leaked.loc[1, "split_role"] = "test"
    with pytest.raises(ValueError, match="event-level split leakage"):
        assert_no_event_split_leakage(leaked)


def test_event_exposure_uses_full_guarded_window() -> None:
    assert event_is_exposed([80], event_start=100, event_end=120, guard_frames=20)
    assert event_is_exposed([140], event_start=100, event_end=120, guard_frames=20)
    assert not event_is_exposed([79, 141], event_start=100, event_end=120, guard_frames=20)


def test_visibility_summary_distinguishes_partial_and_strict_centroid() -> None:
    complete = [
        FakeInstance([[0, 0], [10, 0]]),
        FakeInstance([[30, 0], [40, 0]]),
        FakeInstance([[60, 0], [70, 0]]),
    ]
    summary = summarize_instances(complete)
    assert summary["difficulty"] == "ordinary"
    assert summary["all_three_mouths_finite"] is True
    assert summary["strict_centroid_eligible"] is True

    partial = [*complete[:2], FakeInstance([[60, 0], [np.nan, np.nan]])]
    summary = summarize_instances(partial)
    assert summary["difficulty"] == "partial_or_occluded"
    assert summary["all_three_mouths_finite"] is True
    assert summary["strict_centroid_eligible"] is True


def test_close_category_takes_effect_for_complete_instances() -> None:
    instances = [
        FakeInstance([[0, 0], [10, 0]]),
        FakeInstance([[100, 0], [110, 0]]),
        FakeInstance([[200, 0], [210, 0]]),
    ]
    summary = summarize_instances(
        instances,
        source_category="close_multi_view_ambiguity",
    )
    assert summary["difficulty"] == "close_contact_or_crossing"
