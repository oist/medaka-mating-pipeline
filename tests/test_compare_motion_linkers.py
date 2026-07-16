from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.tracking.compare_motion_linkers import (
    ComparisonConfig,
    build_anonymous_observations,
    compare_motion_linkers,
    write_outputs,
)


def _crossing_reference() -> pd.DataFrame:
    rows = []
    positions = {
        0: {"H1": 0.0, "H2": 10.0},
        1: {"H1": 4.0, "H2": 6.0},
        2: {"H1": 8.0, "H2": 2.0},
        3: {"H1": 12.0, "H2": -2.0},
    }
    for frame, by_identity in positions.items():
        for human, x in by_identity.items():
            rows.append(
                {
                    "frame": frame,
                    "human_identity": human,
                    "X": x,
                    "Y": 0.0,
                    "Z": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _config(*, calibration_end: int = 1) -> ComparisonConfig:
    return ComparisonConfig(
        calibration_start_frame=0,
        calibration_end_frame=calibration_end,
        num_individuals=2,
        max_jump_mm=None,
        max_jump_hard_cap_mm=None,
        max_gap_frames=100,
        velocity_prediction_weight=1.0,
        flow_weight=0.0,
        acceleration_weight=0.0,
        motion_uncertainty_growth_per_frame=0.0,
        max_motion_gap_frames=None,
    )


def _episodes(start: int, end: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"episode_id": "crossing", "start_frame": start, "end_frame": end}]
    )


def test_motion_aware_linker_preserves_identity_through_heldout_crossing():
    result = compare_motion_linkers(
        _crossing_reference(),
        config=_config(),
        episodes=_episodes(2, 3),
    )

    position = result["methods"]["position_only"]
    motion = result["methods"]["motion_aware"]
    assert position["identity"]["identity_accuracy_on_associated"] == 0.0
    assert motion["identity"]["identity_accuracy_on_associated"] == 1.0
    assert len(position["error_spans"]) == 2
    assert {span["n_frames"] for span in position["error_spans"]} == {2}
    assert motion["error_spans"] == []
    assert result["leakage_audit"][
        "effective_context_heldout_intersection_frames"
    ] == 0


def test_missing_reviewed_rows_are_reported_as_timeline_gaps():
    rows = []
    for frame in range(6):
        if frame not in {3, 4}:
            rows.append(
                {
                    "frame": frame,
                    "human_identity": "H1",
                    "X": float(frame),
                    "Y": 0.0,
                    "Z": 0.0,
                }
            )
        rows.append(
            {
                "frame": frame,
                "human_identity": "H2",
                "X": float(20 - frame),
                "Y": 0.0,
                "Z": 0.0,
            }
        )

    result = compare_motion_linkers(
        pd.DataFrame(rows),
        config=_config(),
        episodes=_episodes(2, 5),
    )
    summary = result["methods"]["motion_aware"]
    assert summary["coverage"]["expected_identity_rows"] == 8
    assert summary["coverage"]["associated_rows"] == 6
    assert summary["coverage"]["timeline_coverage"] == pytest.approx(0.75)
    assert summary["gaps"]["per_identity"]["H1"]["longest_gap_frames"] == 2
    assert summary["gaps"]["per_identity"]["H2"]["longest_gap_frames"] == 0


def test_test_episode_labels_cannot_change_context_fitted_mapping():
    reference = _crossing_reference()
    relabeled = reference.copy()
    test = relabeled["frame"].between(2, 3)
    relabeled.loc[test, "human_identity"] = relabeled.loc[
        test, "human_identity"
    ].map({"H1": "H2", "H2": "H1"})

    # The requested context deliberately overlaps the test episode. The
    # benchmark must remove those frames before fitting the label mapping.
    original_result = compare_motion_linkers(
        reference,
        config=_config(calibration_end=3),
        episodes=_episodes(2, 3),
    )
    relabeled_result = compare_motion_linkers(
        relabeled,
        config=_config(calibration_end=3),
        episodes=_episodes(2, 3),
    )

    for method in ("position_only", "motion_aware"):
        assert (
            original_result["methods"][method]["global_mapping"]
            == relabeled_result["methods"][method]["global_mapping"]
        )
    assert original_result["leakage_audit"][
        "requested_context_heldout_overlap_frames_excluded"
    ] == 2
    assert original_result["methods"]["motion_aware"]["identity"][
        "identity_accuracy_on_associated"
    ] == 1.0
    assert relabeled_result["methods"]["motion_aware"]["identity"][
        "identity_accuracy_on_associated"
    ] == 0.0


def test_anonymous_source_order_is_independent_of_row_and_identity_order():
    reference = _crossing_reference()
    renamed = reference.sample(frac=1.0, random_state=7).reset_index(drop=True)
    renamed["human_identity"] = renamed["human_identity"].map(
        {"H1": "female", "H2": "male"}
    )

    best_a, index_a, _ = build_anonymous_observations(
        reference, num_individuals=2
    )
    best_b, index_b, _ = build_anonymous_observations(
        renamed, num_individuals=2
    )
    assert best_a == best_b
    pd.testing.assert_frame_equal(
        index_a[["frame", "source_index", "source_slot", "X", "Y", "Z"]],
        index_b[["frame", "source_index", "source_slot", "X", "Y", "Z"]],
    )
    assert "human_identity" not in index_a.columns


def test_outputs_write_json_and_optional_predictions(tmp_path):
    result = compare_motion_linkers(
        _crossing_reference(),
        config=_config(),
        episodes=_episodes(2, 3),
    )
    report_path = tmp_path / "comparison.json"
    predictions_path = tmp_path / "predictions.csv"
    paths = write_outputs(
        result,
        output_json=report_path,
        predictions_csv=predictions_path,
    )

    assert paths == {
        "report_json": report_path,
        "predictions_csv": predictions_path,
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "motion_linker_comparison_v1"
    assert report["anonymous_candidate_policy"][
        "human_identity_passed_to_linker"
    ] is False
    predictions = pd.read_csv(predictions_path)
    assert set(predictions["method"]) == {"position_only", "motion_aware"}
    assert {"track_id", "mapped_human_identity", "human_identity"} <= set(
        predictions.columns
    )
