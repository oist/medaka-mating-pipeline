from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sam2_id_experiment.src.id_model.motion_identity_optimizer import (
    LayeredIdentityConfig,
    LocalWindowIdentityConfig,
    MotionIdentityConfig,
    TransitionRelocationConfig,
    optimize_identity_track,
    repair_identity_track_layered,
    repair_identity_local_windows,
    relocate_identity_transitions,
)


def _track(
    positions: list[tuple[float, float, float]],
    labels: list[tuple[str, str, str]],
    *,
    confidence: str = "high",
) -> pd.DataFrame:
    rows = []
    for frame, (slot_x, frame_labels) in enumerate(zip(positions, labels)):
        for slot in range(3):
            rows.append(
                {
                    "frame": frame,
                    "geometry_slot": slot,
                    "X": slot_x[slot],
                    "Y": 0.0,
                    "Z": 0.0,
                    "id_final": frame_labels[slot],
                    "id_conf": confidence,
                    "id_source": "high",
                }
            )
    return pd.DataFrame(rows)


def _labels_by_frame(result) -> list[tuple[str, str, str]]:
    out = result.optimized_track.sort_values(["frame", "geometry_slot"])
    return [
        tuple(rows.id_final)
        for _, rows in out.groupby("frame", sort=True, observed=True)
    ]


def test_global_optimizer_rejects_far_one_frame_swap() -> None:
    positions = [(0.0, 100.0, 200.0)] * 5
    labels = [
        ("F", "M1", "M2"),
        ("F", "M1", "M2"),
        ("M1", "F", "M2"),
        ("F", "M1", "M2"),
        ("F", "M1", "M2"),
    ]
    track = _track(positions, labels)

    result = optimize_identity_track(track)

    assert _labels_by_frame(result) == [("F", "M1", "M2")] * 5
    assert result.diagnostics["changed_frames"] == 1
    assert result.diagnostics["optimized_permutation_changes"] == 0


def test_global_optimizer_can_switch_at_close_approach() -> None:
    positions = [
        (0.0, 10.0, 100.0),
        (3.0, 7.0, 100.0),
        (6.0, 4.0, 100.0),
        (9.0, 1.0, 100.0),
    ]
    labels = [
        ("F", "M1", "M2"),
        ("F", "M1", "M2"),
        ("M1", "F", "M2"),
        ("M1", "F", "M2"),
    ]
    track = _track(positions, labels)
    config = MotionIdentityConfig(
        appearance_weight=10.0,
        position_weight=0.1,
        velocity_weight=0.0,
        tangent_weight=0.0,
        acceleration_weight=0.0,
        switch_penalty=0.1,
        jump_penalty=0.0,
        hard_jump_mm=8.0,
    )

    result = optimize_identity_track(track, config=config)
    optimized = _labels_by_frame(result)

    assert optimized[:2] == [("F", "M1", "M2")] * 2
    assert optimized[2:] == [("M1", "F", "M2")] * 2
    assert result.diagnostics["optimized_permutation_changes"] == 1


def test_result_is_row_aligned_and_one_each_for_shuffled_input() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 3,
        [
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
        ],
    ).sample(frac=1.0, random_state=3)

    result = optimize_identity_track(track)

    assert result.optimized_track.index.equals(track.index)
    assert len(result.labels) == len(track)
    for _, rows in result.optimized_track.groupby("frame"):
        assert sorted(rows.id_final) == ["F", "M1", "M2"]


def test_missing_position_does_not_break_permutation_constraint() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 3,
        [("F", "M1", "M2")] * 3,
    )
    track.loc[(track.frame == 1) & (track.geometry_slot == 1), ["X", "Y", "Z"]] = np.nan

    result = optimize_identity_track(track)

    assert np.isfinite(result.diagnostics["normalized_path_cost"])
    assert all(
        sorted(rows.id_final) == ["F", "M1", "M2"]
        for _, rows in result.optimized_track.groupby("frame")
    )


def test_invalid_duplicate_frame_slot_is_rejected() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)],
        [("F", "M1", "M2")],
    )
    duplicate = pd.concat([track, track.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        optimize_identity_track(duplicate)


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="velocity_weight"):
        MotionIdentityConfig(velocity_weight=-1.0)
    with pytest.raises(ValueError, match="hard_jump_mm"):
        MotionIdentityConfig(hard_jump_mm=0.0)


def test_conservative_relocation_moves_boundary_to_continuous_crossing() -> None:
    positions = [
        (float(frame), float(10 - frame), 100.0)
        for frame in range(10)
    ]
    labels = [
        ("F", "M1", "M2") if frame < 8 else ("M1", "F", "M2")
        for frame in range(10)
    ]
    track = _track(positions, labels)

    result = relocate_identity_transitions(
        track,
        config=TransitionRelocationConfig(
            jump_threshold_mm=2.0,
            search_radius_frames=5,
        ),
    )
    optimized = _labels_by_frame(result)
    boundary = result.diagnostics["decisions"][0]["relocated_frame"]

    # Frames 5 and 6 are equally continuous around this symmetric crossing;
    # the deterministic tie-break chooses whichever is closer to the input
    # boundary.
    assert boundary in (5, 6)
    assert optimized[:boundary] == [("F", "M1", "M2")] * boundary
    assert optimized[boundary:] == [("M1", "F", "M2")] * (10 - boundary)
    assert result.diagnostics["relocated_boundaries"] == 1
    assert result.diagnostics["transition_jumps_over_threshold_after"] == 0


def test_conservative_relocation_leaves_already_continuous_boundary() -> None:
    track = _track(
        [(0.0, 2.0, 100.0), (1.0, 1.0, 100.0), (2.0, 0.0, 100.0)],
        [("F", "M1", "M2"), ("M1", "F", "M2"), ("M1", "F", "M2")],
    )

    result = relocate_identity_transitions(
        track,
        config=TransitionRelocationConfig(jump_threshold_mm=2.0),
    )

    assert result.labels.tolist() == track.id_final.tolist()
    assert result.diagnostics["relocated_boundaries"] == 0


def test_layered_repair_removes_bracketed_one_frame_teleport() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 5,
        [
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
        ],
    )

    result = repair_identity_track_layered(
        track,
        config=LayeredIdentityConfig(
            jump_threshold_mm=8.0,
            max_flicker_frames=2,
            search_radius_frames=2,
        ),
    )

    assert _labels_by_frame(result) == [("F", "M1", "M2")] * 5
    assert result.diagnostics["removed_flicker_spans"] == 1
    assert result.diagnostics["jump_qc"]["before"] == {
        "jump_rows": 4,
        "unique_jump_frames": 2,
    }
    assert result.diagnostics["jump_qc"]["after_relocation"] == {
        "jump_rows": 0,
        "unique_jump_frames": 0,
    }


def test_jump_diagnostics_ignore_missing_coordinates() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 3,
        [("F", "M1", "M2")] * 3,
    )
    track.loc[
        (track.frame == 1) & (track.geometry_slot == 1), ["X", "Y", "Z"]
    ] = np.nan

    result = relocate_identity_transitions(track)

    assert result.diagnostics["jump_qc_before"] == {
        "jump_rows": 0,
        "unique_jump_frames": 0,
    }


def test_local_window_repair_removes_bounded_impossible_excursion() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 7,
        [
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
        ],
    )

    result = repair_identity_local_windows(
        track,
        config=LocalWindowIdentityConfig(
            jump_threshold_mm=8.0,
            radius_frames=2,
        ),
    )

    assert _labels_by_frame(result) == [("F", "M1", "M2")] * 7
    assert result.diagnostics["accepted_clusters"] == 1
    assert result.diagnostics["output_jump_qc"] == {
        "jump_rows": 0,
        "unique_jump_frames": 0,
    }


def test_local_window_repair_rejects_unreachable_fixed_end_state() -> None:
    track = _track(
        [(0.0, 100.0, 200.0)] * 8,
        [("F", "M1", "M2")] * 4 + [("M1", "F", "M2")] * 4,
    )

    result = repair_identity_local_windows(
        track,
        config=LocalWindowIdentityConfig(
            jump_threshold_mm=8.0,
            radius_frames=2,
        ),
    )

    assert result.labels.tolist() == track.id_final.tolist()
    assert result.diagnostics["accepted_clusters"] == 0
    assert result.diagnostics["rejected_clusters"] == 1
