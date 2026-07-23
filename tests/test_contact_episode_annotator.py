from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.analysis.contact_episode_annotator import (
    SceneCachedVideo,
    SyntheticCalibration,
    SyntheticSlp,
    _draw_false_negative_markers,
    associate_projected_ids,
    compute_2d_ifd_window,
    instance_anchor_points,
    toggle_false_negative_marker,
)
from scripts.analysis.female_sex_gt_annotator import OpenCvVideo
from scripts.analysis.contact_episode_gt import deserialize_clicks, read_annotation_ledger


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_false_negative_marker_toggle_and_distinct_rendering() -> None:
    markers, added = toggle_false_negative_marker(
        [],
        view="side1_cam",
        frame=42,
        click_x=30.0,
        click_y=20.0,
        radius_px=10.0,
        marked_at_utc="2026-07-21T00:00:00Z",
    )
    assert added is True and len(markers) == 1
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    _draw_false_negative_markers(
        image, markers, view="side1_cam", frame=42
    )
    assert int(image[:, :, 0].max()) == 255  # distinctive magenta/blue channel
    assert int(image[:, :, 2].max()) > 0

    markers, added = toggle_false_negative_marker(
        markers,
        view="side1_cam",
        frame=42,
        click_x=34.0,
        click_y=23.0,
        radius_px=10.0,
        marked_at_utc="later",
    )
    assert added is False and markers == []


def test_scene_cache_prefetches_both_sides_and_serves_hits(tmp_path) -> None:
    video_path = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for frame in range(20):
        image = np.full((48, 64, 3), frame * 8, dtype=np.uint8)
        writer.write(image)
    writer.release()

    cached = SceneCachedVideo(
        OpenCvVideo(video_path, cache_size=2), max_cache_bytes=1024 * 1024
    )
    try:
        cached.prime(2, 16, 9)
        cached.wait_until_ready(timeout=5)
        info = cached.cache_info()
        assert info["warming"] is False
        assert info["cached_frames"] >= 12
        assert info["forward_frames"] > 0
        assert info["backward_frames"] > 0
        before_hits = int(info["hits"])
        cached.get(8)
        cached.get(10)
        assert cached.cache_info()["hits"] == before_hits + 2
    finally:
        cached.close()


def test_anchor_modes_and_one_to_one_association() -> None:
    instances = np.array(
        [
            [[10.0, 10.0], [6.0, 10.0]],
            [[30.0, 10.0], [26.0, 10.0]],
            [[50.0, 10.0], [np.nan, np.nan]],
        ]
    )
    np.testing.assert_allclose(instance_anchor_points(instances, "mouth"), instances[:, 0])
    centroids = instance_anchor_points(instances, "visible-node-centroid")
    np.testing.assert_allclose(centroids, [[8, 10], [28, 10], [50, 10]])
    by_id, by_instance, _anchors = associate_projected_ids(
        {"ID_1": (11, 10), "ID_2": (29, 10), "ID_3": (51, 10)},
        instances,
        anchor_mode="mouth",
        gate_px=5,
    )
    assert by_id == {"ID_1": 0, "ID_2": 1, "ID_3": 2}
    assert by_instance == {0: "ID_1", 1: "ID_2", 2: "ID_3"}
    excluded_by_id, excluded_by_instance, excluded_anchors = associate_projected_ids(
        {"ID_1": (11, 10), "ID_2": (29, 10), "ID_3": (51, 10)},
        instances,
        anchor_mode="mouth",
        gate_px=5,
        excluded_instance_indices={1},
    )
    assert excluded_by_id == {"ID_1": 0, "ID_3": 2}
    assert excluded_by_instance == {0: "ID_1", 2: "ID_3"}
    assert np.isnan(excluded_anchors[1]).all()


def test_lazy_2d_ifd_keeps_unmatched_pairs_nan(tmp_path) -> None:
    from scripts.analysis.contact_episode_annotator import _synthetic_store

    store = _synthetic_store(tmp_path / "tracks.csv")
    calibration = SyntheticCalibration()
    slp = SyntheticSlp("top_cam", store, calibration)
    window = compute_2d_ifd_window(
        store,
        slp,
        calibration,
        view="top_cam",
        start_frame=3,
        end_frame=6,
        anchor_mode="mouth",
        gate_px=3,
    )
    assert window.distances.shape == (4, 3)
    assert np.isfinite(window.distances).all()
    assert window.matched_positions.shape == (4, 3, 2)
    assert window.displacements.shape == (4, 3)
    assert np.isnan(window.displacements[0]).all()
    np.testing.assert_allclose(
        window.displacements[1:],
        np.array([[1.0, 1.0, 0.0]] * 3),
    )
    assert set(window.match_status.ravel()) == {"matched"}
    too_strict = compute_2d_ifd_window(
        store,
        slp,
        calibration,
        view="top_cam",
        start_frame=3,
        end_frame=3,
        anchor_mode="mouth",
        gate_px=0,
    )
    # Exact synthetic projections still match even at a zero-pixel gate.
    assert np.isfinite(too_strict.distances).all()
    excluded = compute_2d_ifd_window(
        store,
        slp,
        calibration,
        view="top_cam",
        start_frame=3,
        end_frame=3,
        anchor_mode="mouth",
        gate_px=3,
        excluded_instances_by_frame={3: {1}},
    )
    assert excluded.match_status[0].tolist() == ["matched", "unmatched", "matched"]
    assert excluded.source_instance_indices[0].tolist() == [0, -1, 2]
    assert np.isnan(excluded.distances[0, [0, 2]]).all()
    assert np.isfinite(excluded.distances[0, 1])
    excluded_sequence = compute_2d_ifd_window(
        store,
        slp,
        calibration,
        view="top_cam",
        start_frame=3,
        end_frame=5,
        anchor_mode="mouth",
        gate_px=3,
        excluded_instances_by_frame={4: {1}},
    )
    # ID_2 has no step into the excluded frame or immediately out of its gap.
    assert np.isnan(excluded_sequence.displacements[:, 1]).all()
    np.testing.assert_allclose(
        excluded_sequence.displacements[1:, [0, 2]],
        np.array([[1.0, 0.0], [1.0, 0.0]]),
    )


def test_offscreen_cli_smoke_navigates_plot_clicks_and_reopens(tmp_path) -> None:
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 GUI extra is not installed")
    csv_path = tmp_path / "contact_gt.csv"
    command = [
        sys.executable,
        "-m",
        "scripts.analysis.contact_episode_annotator",
        "synthetic_experiment",
        "--annotator",
        "qt-smoke",
        "--annotations-csv",
        str(csv_path),
        "--smoke-test",
    ]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    reopened = read_annotation_ledger(csv_path)
    assert len(reopened) == 1
    row = reopened.iloc[0]
    assert row["contact_validity"] == "confirmed_contact_crossing"
    assert row["tracker_assessment"] == "correct_continuity"
    assert row["automatic_mapping_suggestion"] == "same_ids_no_swap"
    assert row["anchor_a_frame"] < row["anchor_b_frame"]
    assert row["issue_flags"] == ""
    clicks = deserialize_clicks(row["clicks_json"])
    for phase in ("anchor_a", "anchor_b"):
        assert clicks[phase]["female"]["top_cam"]
        assert clicks[phase]["anchor_male"]["top_cam"]
    false_positives = json.loads(row["false_positive_detections_json"])
    assert len(false_positives) == 1
    assert false_positives[0]["instance_index"] == 2
    false_negatives = json.loads(row["false_negative_detections_json"])
    assert len(false_negatives) == 1
    assert false_negatives[0]["view"] == "side2_cam"
