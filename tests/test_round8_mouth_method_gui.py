from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.visualization.round8_mouth_method_gui import (
    CandidateCache,
    CandidatePose,
    MethodSource,
    TrackTable,
    build_female_change_events,
    draw_female_ground_truth,
    draw_raw_candidates,
    method_identity_key,
    nearest_candidate,
    parse_hypothesis_slots,
    row_slot,
)


def _track_rows() -> pd.DataFrame:
    rows = []
    for frame in range(6):
        rows.extend(
            [
                {
                    "frame": frame,
                    "X": float(frame),
                    "Y": 0.0,
                    "Z": 0.0,
                    "_base_id": "ID_1",
                    "_segment": 0,
                    "_status": "resolved",
                },
                {
                    "frame": frame,
                    "X": float(10 - frame),
                    "Y": 0.0,
                    "Z": 0.0,
                    "_base_id": "ID_2",
                    "_segment": 4,
                    "_status": "resolved",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_method_identity_and_hypothesis_slot_parsing() -> None:
    assert method_identity_key(8.0, -1, include_segment=False) == "8"
    assert (
        method_identity_key("ID_2", 4.0, include_segment=True)
        == "ID_2#identity_segment=4"
    )
    assert parse_hypothesis_slots("f034547_m5_s1_-1_2") == (1, -1, 2)
    assert parse_hypothesis_slots("") is None


def test_track_table_exact_frame_and_closest_approach() -> None:
    table = TrackTable("A1", _track_rows(), include_segment=True)
    at_three = table.rows_at(3)
    assert list(at_three["_base_id"]) == ["ID_1", "ID_2"]
    closest = table.closest_approach(
        "ID_1#identity_segment=0",
        "ID_2#identity_segment=4",
        0,
        5,
    )
    assert closest == (5, 0.0)


def test_track_table_merges_identical_bounded_a4_rows(tmp_path: Path) -> None:
    columns = {
        "frame": [10],
        "fish_id": ["PATH_1"],
        "identity_segment": [0],
        "X": [1.0],
        "Y": [2.0],
        "Z": [3.0],
        "identity_status": ["resolved_tracklet_path_segment"],
        "slot_cam0": [0],
        "slot_cam1": [1],
        "slot_cam2": [-1],
    }
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(columns).to_csv(first, index=False)
    pd.DataFrame(columns).to_csv(second, index=False)
    table = TrackTable.from_csvs(
        MethodSource("A4", (first, second), True, "mouth")
    )
    assert len(table.table) == 1
    assert table.table.iloc[0]["_method_key"] == "PATH_1#identity_segment=0"


def test_a0_integer_final_ids_load_as_normalized_strings(tmp_path: Path) -> None:
    path = tmp_path / "a0.csv"
    pd.DataFrame(
        {
            "frame": [0, 0],
            "fish_id": ["ID_1", "ID_2"],
            "final_fish_id": [11, 8],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Z": [5.0, 6.0],
            "bind_source": ["top_cam_2d", "geometry_only"],
        }
    ).to_csv(path, index=False)
    table = TrackTable.from_csvs(
        MethodSource("A0", (path,), False, "visible centroid")
    )
    assert set(table.table["_base_id"]) == {"8", "11"}
    assert set(table.table["_method_key"]) == {"8", "11"}


def test_female_change_events_preserve_segment_aware_ids() -> None:
    assignments = pd.DataFrame(
        [
            {
                "method": "A0",
                "frame": 10,
                "consensus_method_id": 11.0,
            },
            {
                "method": "A0",
                "frame": 20,
                "consensus_method_id": 8.0,
            },
            {
                "method": "A1",
                "frame": 10,
                "consensus_method_id": "ID_1#identity_segment=0",
            },
            {
                "method": "A1",
                "frame": 20,
                "consensus_method_id": "ID_2#identity_segment=4",
            },
        ]
    )
    events = build_female_change_events(assignments)
    by_method = {event.source_method: event for event in events}
    assert by_method["A0"].from_id == "11"
    assert by_method["A0"].to_id == "8"
    assert by_method["A1"].to_id == "ID_2#identity_segment=4"


def _write_candidate_cache(root: Path, view: str) -> None:
    path = root / f"{view}.candidates.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["frame_count"] = 2
        handle.create_dataset(
            "mouth_xy",
            data=np.array(
                [
                    [[10.0, 20.0], [np.nan, np.nan]],
                    [[12.0, 20.0], [30.0, 40.0]],
                ]
            ),
        )
        handle.create_dataset(
            "tail_xy",
            data=np.array(
                [
                    [[5.0, 20.0], [np.nan, np.nan]],
                    [[7.0, 20.0], [25.0, 40.0]],
                ]
            ),
        )
        handle.create_dataset("source_instance", data=np.array([[3, -1], [4, 7]]))
        handle.create_dataset(
            "mouth_score", data=np.array([[0.9, np.nan], [0.95, 0.8]])
        )
        handle.create_dataset(
            "tail_score", data=np.array([[0.8, np.nan], [0.85, 0.7]])
        )


def test_candidate_cache_random_access_and_source_indices(tmp_path: Path) -> None:
    for view in ("top_cam", "side1_cam", "side2_cam"):
        _write_candidate_cache(tmp_path, view)
    cache = CandidateCache(tmp_path)
    try:
        candidates = cache.get("top_cam", 1)
        assert [candidate.slot for candidate in candidates] == [0, 1]
        assert [candidate.source_instance for candidate in candidates] == [4, 7]
        assert np.allclose(candidates[1].mouth, [30.0, 40.0])
    finally:
        cache.close()


def test_nearest_candidate_uses_requested_anchor_policy() -> None:
    candidates = [
        CandidatePose(
            slot=0,
            source_instance=5,
            mouth=np.array([10.0, 10.0]),
            tail=np.array([0.0, 10.0]),
            mouth_score=1.0,
            tail_score=1.0,
        ),
        CandidatePose(
            slot=1,
            source_instance=6,
            mouth=np.array([30.0, 10.0]),
            tail=np.array([20.0, 10.0]),
            mouth_score=1.0,
            tail_score=1.0,
        ),
    ]
    mouth_match, _ = nearest_candidate(
        candidates,
        np.array([11.0, 10.0]),
        anchor_policy="mouth",
        max_distance_px=20.0,
    )
    centroid_match, _ = nearest_candidate(
        candidates,
        np.array([24.0, 10.0]),
        anchor_policy="visible centroid",
        max_distance_px=20.0,
    )
    assert mouth_match is not None and mouth_match.slot == 0
    assert centroid_match is not None and centroid_match.slot == 1


def test_row_slot_prefers_explicit_a4_columns_then_hypothesis() -> None:
    explicit = {"slot_cam0": 2.0, "hypothesis_id": "f1_m7_s0_1_2"}
    assert row_slot(explicit, "top_cam") == 2
    parsed = {"hypothesis_id": "f1_m7_s0_-1_2"}
    assert row_slot(parsed, "side1_cam") == -1
    assert row_slot({}, "side2_cam") is None


def test_overlay_helpers_draw_candidates_and_female_gt() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    candidate = CandidatePose(
        slot=0,
        source_instance=3,
        mouth=np.array([30.0, 30.0]),
        tail=np.array([10.0, 30.0]),
        mouth_score=1.0,
        tail_score=1.0,
    )
    draw_raw_candidates(image, [candidate])
    draw_female_ground_truth(
        image,
        {
            "side1_selection_status": "sleap_linked",
            "side1_mouth_x": 50.0,
            "side1_mouth_y": 50.0,
        },
        "side1_cam",
    )
    assert image.any()
