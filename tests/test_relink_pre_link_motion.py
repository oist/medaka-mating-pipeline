import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.reconstruction_3d.reconstruct_from_slp import (
    save_best_pre_link_json_to,
)
from scripts.reconstruction_3d.relink_pre_link_motion import (
    build_parser,
    run_pre_link_motion_relink,
)


def _frame(*xs: float) -> dict:
    labels = [chr(ord("a") + index) for index in range(len(xs))]
    return {
        "groups": {
            label: [float(x), 0.0, 0.0]
            for label, x in zip(labels, xs)
        },
        "errors": {label: 1.0 for label in labels},
        # Exercise the extra-candidate post-link mapping schema: selected
        # source labels can point at any of six original detection slots.
        "perm_cam0": [4, 1],
        "perm_cam1": [5, 2],
        "perm_cam2": [3, 0],
    }


def test_motion_relink_uses_saved_candidates_and_writes_drop_in_outputs(
    tmp_path: Path,
):
    pre_link_path = tmp_path / "historical_best_pre_link.json"
    save_best_pre_link_json_to(
        {
            0: _frame(0, 10),
            1: _frame(4, 6),
            2: _frame(8, 2),
        },
        pre_link_path,
        num_individuals=2,
    )

    result = run_pre_link_motion_relink(
        pre_link_path,
        output_tag="motion",
        num_individuals=2,
        max_jump_mm=8.0,
        max_jump_hard_cap_mm=80.0,
        max_gap_frames=100,
        motion_velocity_prediction_weight=1.0,
        motion_tangent_weight=2.0,
        motion_acceleration_weight=0.0,
        max_motion_gap_frames=5,
        motion_history_frames=5,
        fill_missing=False,
        shift_to_positive=False,
    )

    tracked_payload = json.loads(result["best_json_path"].read_text())
    final_frame = tracked_payload["frames"]["2"]
    assert final_frame["groups"]["ID_1"] == [8.0, 0.0, 0.0]
    assert final_frame["groups"]["ID_2"] == [2.0, 0.0, 0.0]
    assert len(final_frame["cam0_to_id"]) == 6
    assert final_frame["cam0_to_id"][4] == "ID_1"
    assert final_frame["cam0_to_id"][1] == "ID_2"
    assert final_frame["cam1_to_id"][5] == "ID_1"
    assert final_frame["cam2_to_id"][0] == "ID_2"

    csv = pd.read_csv(result["csv_path"])
    id1_final = csv[(csv.frame == 2) & (csv.fish_id == "ID_1")].iloc[0]
    assert id1_final.X == pytest.approx(8.0)
    assert list(csv.columns) == ["frame", "fish_id", "X", "Y", "Z"]

    summary = json.loads(result["summary_path"].read_text())
    assert summary["schema_version"] == "pre_link_motion_relink_v1"
    assert summary["num_frames"] == 3
    assert summary["linker_settings"]["velocity_prediction_weight"] == 1.0
    assert summary["linker_settings"]["flow_weight"] == 2.0
    assert summary["linker_settings"]["motion_history_frames"] == 5
    assert summary["source_hint_frames"] == 0


def test_motion_relink_defaults_are_opt_in_and_outputs_are_protected(
    tmp_path: Path,
):
    defaults = build_parser().parse_args(["input_best_pre_link.json"])
    assert defaults.motion_velocity_prediction_weight == 0.0
    assert defaults.motion_tangent_weight == 0.0
    assert defaults.motion_acceleration_weight == 0.0
    assert defaults.motion_uncertainty_growth_per_frame == 0.25
    assert defaults.max_motion_gap_frames is None
    assert defaults.motion_history_frames == 2
    assert defaults.start_frame is None
    assert defaults.end_frame is None
    assert defaults.no_fill_missing is False
    assert defaults.no_shift_to_positive is False

    pre_link_path = tmp_path / "baseline_best_pre_link.json"
    save_best_pre_link_json_to(
        {0: _frame(0, 10)},
        pre_link_path,
        num_individuals=2,
    )
    first = run_pre_link_motion_relink(
        pre_link_path,
        num_individuals=2,
        fill_missing=False,
    )
    assert first["best_json_path"].name == "baseline_motion_relink_best.json"
    assert first["csv_path"].name == "baseline_motion_relink_3d.csv"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_pre_link_motion_relink(
            pre_link_path,
            num_individuals=2,
            fill_missing=False,
        )


def test_motion_relink_preserves_source_hints(tmp_path: Path):
    pre_link_path = tmp_path / "hinted_best_pre_link.json"
    first = _frame(0, 10)
    second = _frame(1, 9)
    second["source_id_hints"] = {0: "ID_2"}
    save_best_pre_link_json_to(
        {0: first, 1: second},
        pre_link_path,
        num_individuals=2,
    )

    result = run_pre_link_motion_relink(
        pre_link_path,
        num_individuals=2,
        max_jump_mm=20.0,
        fill_missing=False,
        shift_to_positive=False,
    )

    payload = json.loads(result["best_json_path"].read_text())
    assert payload["frames"]["1"]["groups"]["ID_2"] == [1.0, 0.0, 0.0]
    summary = json.loads(result["summary_path"].read_text())
    assert summary["source_hint_frames"] == 1


def test_motion_relink_frame_filter_is_inclusive_and_recorded(tmp_path: Path):
    pre_link_path = tmp_path / "window_best_pre_link.json"
    save_best_pre_link_json_to(
        {
            0: _frame(0, 10),
            1: _frame(1, 9),
            2: _frame(2, 8),
            3: _frame(3, 7),
        },
        pre_link_path,
        num_individuals=2,
    )

    result = run_pre_link_motion_relink(
        pre_link_path,
        output_tag="window_f1-2",
        num_individuals=2,
        start_frame=1,
        end_frame=2,
        fill_missing=False,
        shift_to_positive=False,
    )

    csv = pd.read_csv(result["csv_path"])
    assert sorted(csv.frame.unique().tolist()) == [1, 2]
    tracked = json.loads(result["best_json_path"].read_text())
    assert list(tracked["frames"]) == ["1", "2"]
    summary = json.loads(result["summary_path"].read_text())
    assert summary["requested_frame_range"] == [1, 2]
    assert summary["frame_range"] == [1, 2]
    assert summary["num_frames"] == 2


def test_motion_relink_frame_filter_validates_range_and_nonempty(tmp_path: Path):
    pre_link_path = tmp_path / "filter_best_pre_link.json"
    save_best_pre_link_json_to(
        {0: _frame(0, 10), 1: _frame(1, 9)},
        pre_link_path,
        num_individuals=2,
    )

    with pytest.raises(ValueError, match="start_frame must be"):
        run_pre_link_motion_relink(
            pre_link_path,
            output_tag="invalid",
            num_individuals=2,
            start_frame=2,
            end_frame=1,
        )

    with pytest.raises(ValueError, match="No pre-link frames remain"):
        run_pre_link_motion_relink(
            pre_link_path,
            output_tag="empty",
            num_individuals=2,
            start_frame=10,
            end_frame=20,
        )
