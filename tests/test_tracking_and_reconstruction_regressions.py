import numpy as np
import pandas as pd

from scripts.reconstruction_3d.load_2d import get_points_per_frame
from scripts.reconstruction_3d.reconstruct_no_calibration import compute_candidates_no_calib
from scripts.tracking.data_loader import parse_sleap_data
from scripts.tracking.id_assignment import assign_ids_frame, assign_ids_temporal


def test_parse_sleap_data_aggregates_multi_node_rows():
    df = pd.DataFrame(
        [
            {"frame": 0, "track": 1, "node": "head", "x": 10.0, "y": 20.0, "score": 0.9},
            {"frame": 0, "track": 1, "node": "tail", "x": 14.0, "y": 24.0, "score": 0.7},
            {"frame": 0, "track": 2, "node": "head", "x": 100.0, "y": 200.0, "score": 0.5},
        ]
    )

    parsed = parse_sleap_data(df)

    assert parsed["frames"].tolist() == [0]
    assert np.allclose(parsed["positions"][0][1], (12.0, 22.0, 0.8))
    assert np.allclose(parsed["positions"][0][2], (100.0, 200.0, 0.5))


def test_assign_ids_frame_rejects_bad_matches_with_max_distance():
    assignment = assign_ids_frame(
        prev_positions_dict={"a": (0.0, 0.0), "b": (100.0, 0.0)},
        curr_positions_dict={
            1: (1000.0, 1000.0, 0.9),
            2: (102.0, 1.0, 0.8),
        },
        fish_ids=["a", "b"],
        max_distance=10.0,
    )

    assert assignment == {2: "b"}


def test_assign_ids_temporal_uses_each_views_own_frames():
    data = {
        "player1": {
            "frames": np.array([0, 2]),
            "positions": {
                0: {1: (0.0, 0.0, 1.0)},
                2: {1: (2.0, 0.0, 1.0)},
            },
        },
        "player2": {
            "frames": np.array([0, 1, 2]),
            "positions": {
                0: {1: (10.0, 0.0, 1.0)},
                1: {1: (11.0, 0.0, 1.0)},
                2: {1: (12.0, 0.0, 1.0)},
            },
        },
        "player3": {
            "frames": np.array([0, 2]),
            "positions": {
                0: {1: (20.0, 0.0, 1.0)},
                2: {1: (22.0, 0.0, 1.0)},
            },
        },
    }

    assignments = assign_ids_temporal(data, num_fish=1, max_distance=5.0)

    assert set(assignments["player2"]) == {0, 1, 2}
    assert assignments["player2"][1] == {1: "a"}


def test_get_points_per_frame_auto_uses_fish_id_and_aggregates_nodes():
    dfs = {
        "player1": pd.DataFrame(
            [
                {"frame": 0, "track": 1, "fish_id": "a", "x": 10.0, "y": 20.0, "score": 0.9},
                {"frame": 0, "track": 1, "fish_id": "a", "x": 14.0, "y": 24.0, "score": 0.7},
                {"frame": 0, "track": 2, "fish_id": "c", "x": 30.0, "y": 40.0, "score": 0.8},
            ]
        ),
        "player2": pd.DataFrame(
            [
                {"frame": 0, "track": 1, "fish_id": "a", "x": 50.0, "y": 60.0, "score": 0.9},
                {"frame": 0, "track": 2, "fish_id": "c", "x": 70.0, "y": 80.0, "score": 0.8},
            ]
        ),
        "player3": pd.DataFrame(
            [
                {"frame": 0, "track": 1, "fish_id": "a", "x": 90.0, "y": 100.0, "score": 0.9},
                {"frame": 0, "track": 2, "fish_id": "c", "x": 110.0, "y": 120.0, "score": 0.8},
            ]
        ),
    }

    points = get_points_per_frame(dfs, use_fish_id=None)

    assert np.allclose(points[0]["cam0"][0], (12.0, 22.0))
    assert np.isnan(points[0]["cam0"][1][0])
    assert np.allclose(points[0]["cam0"][2], (30.0, 40.0))


def test_no_calibration_candidates_can_keep_input_order_only():
    points_by_frame = {
        0: {
            "cam0": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            "cam1": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            "cam2": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        }
    }

    candidates = compute_candidates_no_calib(points_by_frame, allow_permutations=False)

    assert len(candidates[0]) == 1
    assert candidates[0][0]["perm_cam1"] == [0, 1, 2]
    assert candidates[0][0]["perm_cam2"] == [0, 1, 2]
