from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.sleap.select_side_centered_instance_frames import (
    EXPERIMENTS,
    SelectionConfig,
    _merge_intervals,
    _minimum_pair_distance,
    add_current_model_scores,
    aggregate_view_features,
    compare_round7_round8,
    select_final_set,
)


def test_merge_intervals_combines_overlap_and_preserves_reasons() -> None:
    result = _merge_intervals(
        [
            (10, 20, "locked"),
            (18, 30, "validation"),
            (40, 45, "training"),
        ]
    )
    assert result == [
        (10, 30, "locked;validation"),
        (40, 45, "training"),
    ]


def test_pair_distance_ignores_unused_global_instance_slots() -> None:
    table = pd.DataFrame(
        {
            "frame": [0, 0, 0, 0, 1, 1],
            "instance_index": [0, 1, 2, 3, 0, 1],
            "tail_x": [0.0, 10.0, 40.0, 100.0, 5.0, 25.0],
            "tail_y": [0.0] * 6,
        }
    )
    distance = _minimum_pair_distance(
        table, x_column="tail_x", y_column="tail_y"
    )
    assert distance.loc[0] == 10.0
    assert distance.loc[1] == 20.0


def test_aggregate_view_features_adds_tail_and_body_signals() -> None:
    frames = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "instance_count": [2, 2, 2],
            "plausible_count": [2, 2, 2],
            "partial_instance_count": [0, 0, 0],
            "minimum_mouth_distance_px": [20.0, 20.0, 20.0],
            "minimum_score": [0.8, 0.8, 0.8],
            "maximum_score": [0.9, 0.9, 0.9],
            "score_std": [0.05, 0.05, 0.05],
            "score_range": [0.1, 0.1, 0.1],
            "maximum_xy_residual_px": [1.0, 1.0, 1.0],
            "outlier_count": [0, 0, 0],
        }
    )
    rows = []
    for frame in range(3):
        for slot in range(2):
            rows.append(
                {
                    "frame": frame,
                    "instance_index": slot,
                    "mouth_x": float(slot * 20),
                    "mouth_y": 0.0,
                    "tail_x": float(slot * 20 + 10 + (20 if frame == 1 else 0)),
                    "tail_y": 0.0,
                    "mouth_score": 0.9,
                    "tail_score": 0.5 + 0.1 * slot,
                    "instance_score": 0.8,
                    "visible_node_count": 2,
                    "plausible": True,
                    "track_id": f"fish_{slot}",
                }
            )
    instances = pd.DataFrame(rows)
    summary, selected = aggregate_view_features(
        frames, instances, view="side1_cam"
    )
    middle = summary.set_index("frame").loc[1]
    assert middle["side1_cam_maximum_body_length_px"] == 30.0
    assert middle["side1_cam_maximum_mouth_tail_score_gap"] == 0.4
    assert middle["side1_cam_tail_midpoint_residual_px"] == 20.0
    assert set(selected["body_length_px"]) == {10.0, 30.0}


def test_compare_round7_round8_is_mouth_matched_and_tail_specific() -> None:
    old = pd.DataFrame(
        [
            {
                "experiment": EXPERIMENTS[0],
                "view": "side1_cam",
                "frame": 100,
                "mouth_x": 10.0,
                "mouth_y": 10.0,
                "tail_x": 20.0,
                "tail_y": 10.0,
            },
            {
                "experiment": EXPERIMENTS[0],
                "view": "side1_cam",
                "frame": 100,
                "mouth_x": 50.0,
                "mouth_y": 10.0,
                "tail_x": 60.0,
                "tail_y": 10.0,
            },
        ]
    )
    current_rows = []
    for view in ("side1_cam", "side2_cam"):
        current_rows.extend(
            [
                {
                    "experiment": EXPERIMENTS[0],
                    "view": view,
                    "frame": 100,
                    "mouth_x": 50.0,
                    "mouth_y": 10.0,
                    "tail_x": 70.0,
                    "tail_y": 10.0,
                    "tail_score": 0.7,
                },
                {
                    "experiment": EXPERIMENTS[0],
                    "view": view,
                    "frame": 100,
                    "mouth_x": 10.0,
                    "mouth_y": 10.0,
                    "tail_x": 20.0,
                    "tail_y": 10.0,
                    "tail_score": 0.8,
                },
            ]
        )
    # Supply the same Round-7 geometry for both views.
    old = pd.concat(
        [
            old,
            old.assign(view="side2_cam"),
        ],
        ignore_index=True,
    )
    comparison = compare_round7_round8(old, pd.DataFrame(current_rows))
    row = comparison.iloc[0]
    assert row["round8_view_count"] == 2
    assert row["round8_maximum_mouth_disagreement_px"] == 0.0
    assert row["round8_maximum_tail_disagreement_px"] == 10.0
    assert bool(row["round8_any_count_not_three"])


def test_current_model_score_uses_minimum_tail_confidence() -> None:
    pool = pd.DataFrame(
        [
            {"experiment": EXPERIMENTS[0], "frame": 100},
            {"experiment": EXPERIMENTS[0], "frame": 200},
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "experiment": EXPERIMENTS[0],
                "frame": 100,
                "round8_view_count": 2,
                "round8_any_count_not_three": False,
                "round8_minimum_tail_score": 0.2,
                "round8_maximum_mouth_disagreement_px": 1.0,
                "round8_maximum_tail_disagreement_px": 10.0,
            },
            {
                "experiment": EXPERIMENTS[0],
                "frame": 200,
                "round8_view_count": 2,
                "round8_any_count_not_three": False,
                "round8_minimum_tail_score": 0.9,
                "round8_maximum_mouth_disagreement_px": 1.0,
                "round8_maximum_tail_disagreement_px": 1.0,
            },
        ]
    )
    result = add_current_model_scores(pool, comparison).set_index("frame")
    assert (
        result.at[100, "round7_round8_tail_disagreement_score"]
        > result.at[200, "round7_round8_tail_disagreement_score"]
    )
    assert bool(result.at[100, "ci_oriented_tail_disagreement"])


def test_final_selection_has_requested_experiment_split_counts() -> None:
    rows = []
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        for index in range(210):
            rows.append(
                {
                    "experiment": experiment,
                    "frame": 50_000 + experiment_index * 10 + index * 1_000,
                    "time_bin": index % 30,
                    "round8_any_count_not_three": False,
                    "ci_oriented_tail_disagreement": True,
                    "round7_round8_tail_disagreement_score": 0.9,
                    "temporal_or_shared_tail_score": 0.9,
                    "shared_tail_prediction": True,
                    "round8_minimum_tail_distance_px": 2.0,
                    "close_contact_score": 0.9,
                    "crop_margin_proxy_score": 0.9,
                    "planned_crop_margin_risk": True,
                    "r7_count_or_partial": True,
                    "current_count_partial_low_confidence_score": 0.9,
                    "r7_good_three_both_views": True,
                    "round8_minimum_tail_score": 0.9,
                    "clean_control_score": 0.9,
                }
            )
    result = select_final_set(
        pd.DataFrame(rows),
        config=SelectionConfig(selection_spacing_frames=166),
    )
    assert len(result) == 210
    by_split = result["split_role"].value_counts().to_dict()
    assert by_split == {"train": 180, "validation": 30}
    by_experiment = (
        result.groupby(["experiment", "split_role"]).size().to_dict()
    )
    assert by_experiment[(EXPERIMENTS[0], "train")] == 70
    assert by_experiment[(EXPERIMENTS[1], "train")] == 40
    assert by_experiment[(EXPERIMENTS[2], "train")] == 70
    assert all(
        by_experiment[(experiment, "validation")] == 10
        for experiment in EXPERIMENTS
    )
