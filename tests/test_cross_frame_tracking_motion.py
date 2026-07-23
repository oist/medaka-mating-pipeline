from pathlib import Path

import pytest

import scripts.reconstruction_3d.reconstruct_from_slp as reconstruction
import scripts.reconstruction_3d.triangulate_extra_candidates as extra_candidates
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
)
from scripts.reconstruction_3d.reconstruct_from_slp import build_parser


def _frame(*xs: float) -> dict:
    labels = [chr(ord("a") + i) for i in range(len(xs))]
    return {
        "groups": {
            label: [float(x), 0.0, 0.0]
            for label, x in zip(labels, xs)
        },
        "perm_cam0": list(range(len(xs))),
        "perm_cam1": list(range(len(xs))),
        "perm_cam2": list(range(len(xs))),
    }


def test_motion_costs_are_opt_in_and_default_schema_is_unchanged():
    best = {
        0: _frame(0, 10),
        1: _frame(4, 6),
        2: _frame(8, 2),
    }

    default = apply_cross_frame_tracking(best, num_individuals=2)
    explicit_zero = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        flow_weight=0,
        velocity_prediction_weight=0,
        acceleration_weight=0,
    )

    assert default == explicit_zero
    assert default[2]["id_mapping"] == [1, 0]
    assert "tracking_diagnostics" not in default[2]


def test_velocity_prediction_preserves_identity_through_a_crossing():
    best = {
        0: _frame(0, 10),
        1: _frame(4, 6),
        2: _frame(8, 2),
    }

    tracked = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        velocity_prediction_weight=1.0,
    )

    assert tracked[2]["id_mapping"] == [0, 1]
    assert tracked[2]["groups"]["ID_1"] == [8.0, 0.0, 0.0]
    assert tracked[2]["groups"]["ID_2"] == [2.0, 0.0, 0.0]
    diagnostic = tracked[2]["tracking_diagnostics"]["ID_1"]
    assert diagnostic["prediction_error_mm"] == pytest.approx(0.0)
    assert diagnostic["total_cost"] == pytest.approx(4.0)


def test_flow_weight_penalizes_direction_reversal():
    best = {
        0: _frame(0, 10),
        1: _frame(4, 6),
        2: _frame(8, 2),
    }

    tracked = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        flow_weight=2.0,
    )

    assert tracked[2]["id_mapping"] == [0, 1]
    assert tracked[2]["tracking_diagnostics"]["ID_1"][
        "tangent_discontinuity"
    ] == pytest.approx(0.0)


def test_robust_motion_history_does_not_let_one_tangent_spike_define_velocity():
    best = {
        0: _frame(0),
        1: _frame(1),
        2: _frame(2),
        3: _frame(1),  # one-frame reversed displacement / bad observation
        4: _frame(2),
    }

    last_interval = apply_cross_frame_tracking(
        best,
        num_individuals=1,
        flow_weight=1.0,
        motion_history_frames=2,
    )
    robust = apply_cross_frame_tracking(
        best,
        num_individuals=1,
        flow_weight=1.0,
        motion_history_frames=5,
    )

    assert last_interval[4]["tracking_diagnostics"]["ID_1"][
        "tangent_discontinuity"
    ] == pytest.approx(2.0)
    assert robust[4]["tracking_diagnostics"]["ID_1"][
        "tangent_discontinuity"
    ] == pytest.approx(0.0)
    assert robust[4]["tracking_diagnostics"]["ID_1"][
        "motion_history_observations"
    ] == 4


def test_acceleration_cost_uses_persistent_three_frame_state():
    best = {
        0: _frame(0, 10),
        1: _frame(2, 8),
        2: _frame(4, 6),
        3: _frame(6, 4),
    }

    position_only = apply_cross_frame_tracking(best, num_individuals=2)
    motion_aware = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        acceleration_weight=2.0,
    )

    assert position_only[3]["id_mapping"] == [1, 0]
    assert motion_aware[3]["id_mapping"] == [0, 1]
    assert motion_aware[3]["tracking_diagnostics"]["ID_1"][
        "acceleration_discontinuity_mm_per_frame2"
    ] == pytest.approx(0.0)


def test_motion_prior_can_expire_across_an_uncertain_gap():
    best = {
        0: _frame(0, 10),
        1: _frame(1, 9),
        10: _frame(10, 0),
    }

    predicted = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        velocity_prediction_weight=1.0,
        motion_uncertainty_growth_per_frame=0.0,
    )
    expired = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        velocity_prediction_weight=1.0,
        motion_uncertainty_growth_per_frame=0.0,
        max_motion_gap_frames=5,
    )

    assert predicted[10]["id_mapping"] == [0, 1]
    assert expired[10]["id_mapping"] == [1, 0]
    expired_diagnostic = expired[10]["tracking_diagnostics"]["ID_1"]
    assert expired_diagnostic["gap_frames"] == 9
    assert expired_diagnostic["motion_available"] is False


def test_source_hint_still_overrides_motion_assignment_for_the_whole_frame():
    best = {
        0: _frame(0, 10),
        1: _frame(4, 6),
        2: _frame(8, 2),
    }

    tracked = apply_cross_frame_tracking(
        best,
        num_individuals=2,
        velocity_prediction_weight=1.0,
        source_id_hints_by_frame={2: {0: "ID_2"}},
    )

    assert tracked[2]["id_mapping"] == [1, 0]
    assert tracked[2]["id_anchor"] == [False, True]
    assert tracked[2]["tracking_diagnostics"]["ID_2"]["anchored"] is True


def test_motion_parameters_reject_invalid_values():
    with pytest.raises(ValueError, match="flow_weight"):
        apply_cross_frame_tracking(
            {0: _frame(0, 10)},
            num_individuals=2,
            flow_weight=-1,
        )
    with pytest.raises(ValueError, match="max_motion_gap_frames"):
        apply_cross_frame_tracking(
            {0: _frame(0, 10)},
            num_individuals=2,
            max_motion_gap_frames=0,
        )


def test_reconstruction_cli_exposes_opt_in_motion_costs():
    defaults = build_parser().parse_args(["example_experiment"])
    enabled = build_parser().parse_args(
        [
            "example_experiment",
            "--motion-velocity-prediction-weight",
            "1.25",
            "--motion-tangent-weight",
            "2.5",
            "--motion-acceleration-weight",
            "0.5",
            "--motion-uncertainty-growth-per-frame",
            "0.1",
            "--max-motion-gap-frames",
            "20",
            "--motion-history-frames",
            "7",
        ]
    )

    assert defaults.motion_velocity_prediction_weight == 0.0
    assert defaults.motion_tangent_weight == 0.0
    assert defaults.motion_acceleration_weight == 0.0
    assert enabled.motion_velocity_prediction_weight == pytest.approx(1.25)
    assert enabled.motion_tangent_weight == pytest.approx(2.5)
    assert enabled.motion_acceleration_weight == pytest.approx(0.5)
    assert enabled.motion_uncertainty_growth_per_frame == pytest.approx(0.1)
    assert enabled.max_motion_gap_frames == 20
    assert defaults.motion_history_frames == 2
    assert enabled.motion_history_frames == 7


def test_six_candidate_reconstruction_forwards_motion_costs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    points_by_frame = {0: {"loaded": True}}
    selected = {0: _frame(0, 10, 20)}
    calls = {}

    class FakeCalibration:
        backend_name = "polynomial"
        score_unit = "px^2"
        provenance = {"model_files": ["fake.joblib"]}
        capabilities = {"hypothesis_scoring": True, "projection": True}

        def __init__(self, _model_folder: str, *, backend: str = "auto"):
            pass

    def fake_select(_cal, points, **kwargs):
        calls["selector"] = {"points": points, **kwargs}
        return selected

    def fake_link(best, **kwargs):
        calls["linker"] = {"best": best, **kwargs}
        return best

    monkeypatch.setattr(
        reconstruction,
        "load_points_by_frame",
        lambda *_args, **_kwargs: points_by_frame,
    )
    monkeypatch.setattr(reconstruction, "Calibration", FakeCalibration)
    monkeypatch.setattr(
        extra_candidates,
        "compute_best_assignments_per_frame",
        fake_select,
    )
    monkeypatch.setattr(reconstruction, "apply_cross_frame_tracking", fake_link)
    monkeypatch.setattr(
        reconstruction,
        "best_by_frame_to_dfs",
        lambda *_args, **_kwargs: {"ID_1": object()},
    )
    monkeypatch.setattr(
        reconstruction,
        "dfs_to_single_csv",
        lambda *_args, **_kwargs: str(tmp_path / "out.csv"),
    )

    result = reconstruction.run_reconstruction_from_slp(
        "example_experiment",
        slp_dir=tmp_path,
        model_folder=tmp_path,
        output_dir=tmp_path,
        num_individuals=3,
        num_candidates=6,
        max_jump_mm=8.0,
        max_jump_hard_cap_mm=80.0,
        max_gap_frames=100,
        motion_velocity_prediction_weight=1.25,
        motion_tangent_weight=2.5,
        motion_acceleration_weight=0.5,
        motion_uncertainty_growth_per_frame=0.1,
        max_motion_gap_frames=20,
        motion_history_frames=7,
        fill_missing=False,
        plot=False,
        save_best_json=False,
        save_best_pre_link_json=False,
    )

    assert calls["selector"]["points"] is points_by_frame
    assert calls["selector"]["num_individuals"] == 3
    assert calls["selector"]["num_candidates"] == 6
    assert result["calibration"]["temporal_linking"]["motion_history_frames"] == 7
    assert result["calibration"]["temporal_linking"][
        "motion_tangent_weight"
    ] == pytest.approx(2.5)
    assert calls["linker"] == {
        "best": selected,
        "num_individuals": 3,
        "max_jump_mm": 8.0,
        "max_jump_hard_cap_mm": 80.0,
        "max_gap_frames": 100,
        "velocity_prediction_weight": 1.25,
        "flow_weight": 2.5,
        "acceleration_weight": 0.5,
        "motion_uncertainty_growth_per_frame": 0.1,
        "max_motion_gap_frames": 20,
        "motion_history_frames": 7,
        "source_id_hints_by_frame": None,
        "source_hints_override_motion_gate": False,
    }
