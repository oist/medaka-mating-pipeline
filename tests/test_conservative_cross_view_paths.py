import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import CandidateWindow
from scripts.reconstruction_3d.conservative_cross_view_paths import (
    CrossViewGroupConfig,
    ThreePathConfig,
    _robust_velocity,
    build_cross_view_groups,
    solve_three_identity_paths,
)


VIEWS = ("top_cam", "side1_cam", "side2_cam")


def _window(mouth: np.ndarray) -> CandidateWindow:
    score = np.where(np.all(np.isfinite(mouth), axis=2), 0.9, np.nan)
    return CandidateWindow(
        frame_start=0,
        frame_end=mouth.shape[0] - 1,
        mouth_xy=mouth,
        tail_xy=np.full_like(mouth, np.nan),
        instance_score=score.copy(),
        mouth_score=score.copy(),
        tail_score=np.full_like(score, np.nan),
        visibility=np.where(np.isfinite(score), 1, 0).astype(np.uint8),
        source_instance=np.where(np.isfinite(score), 0, -1).astype(np.int16),
    )


def _hypothesis(
    frame: int,
    *,
    group: int,
    residual: float = 0.2,
    view_mask: int = 0b111,
    slots: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    return {
        "frame": frame,
        "view_mask": view_mask,
        "view_count": view_mask.bit_count(),
        "X": float(group * 40 + frame),
        "Y": 0.0,
        "Z": 0.0,
        "ray_residual_mm2": residual,
        "hypothesis_id": f"h_{group}_{frame}_{view_mask}",
        "slot_cam0": slots[0],
        "slot_cam1": slots[1],
        "slot_cam2": slots[2],
        "tracklet_top_cam": group if slots[0] >= 0 else -1,
        "tracklet_side1_cam": group if slots[1] >= 0 else -1,
        "tracklet_side2_cam": group if slots[2] >= 0 else -1,
    }


def test_two_view_group_rejects_retained_tracklet_count_mismatch():
    mouth = np.full((3, 3, 2), np.nan)
    mouth[:, :, :] = np.asarray([[0, 0], [40, 0], [80, 0]])
    windows = {view: _window(mouth.copy()) for view in VIEWS}
    retained = {
        "top_cam": np.asarray([[0, 1, -1]] * 3),
        "side1_cam": np.asarray([[0, -1, -1]] * 3),
        "side2_cam": np.asarray([[0, -1, -1]] * 3),
    }
    hypotheses = pd.DataFrame(
        [
            _hypothesis(
                frame,
                group=0,
                view_mask=0b011,
                slots=(0, 0, -1),
            )
            for frame in range(3)
        ]
    )

    groups, _observations = build_cross_view_groups(
        hypotheses,
        windows,
        retained,
        config=CrossViewGroupConfig(minimum_support_frames=1),
    )

    assert len(groups) == 1
    assert not bool(groups.iloc[0].eligible)
    assert groups.iloc[0].rejection_reason == (
        "retained_tracklet_count_mismatch"
    )


def test_three_view_group_also_rejects_retained_count_mismatch():
    mouth = np.zeros((3, 3, 2), dtype=float)
    windows = {view: _window(mouth.copy()) for view in VIEWS}
    retained = {
        "top_cam": np.asarray([[0, 1, -1]] * 3),
        "side1_cam": np.asarray([[0, -1, -1]] * 3),
        "side2_cam": np.asarray([[0, -1, -1]] * 3),
    }
    hypotheses = pd.DataFrame(
        [_hypothesis(frame, group=0) for frame in range(3)]
    )

    groups, _observations = build_cross_view_groups(
        hypotheses,
        windows,
        retained,
        config=CrossViewGroupConfig(minimum_support_frames=1),
    )

    assert not bool(groups.iloc[0].eligible)
    assert groups.iloc[0].rejection_reason == (
        "retained_tracklet_count_mismatch"
    )


def test_robust_velocity_work_is_bounded_for_long_fragment():
    frames = np.arange(10000, dtype=int)
    xyz = np.column_stack(
        [frames * 0.5, frames * -0.25, frames * 0.1]
    )

    velocity = _robust_velocity(frames, xyz)

    assert np.allclose(velocity, [0.5, -0.25, 0.1])


def test_group_cost_uses_robust_residual_and_saturating_support_not_sum():
    frame_count = 100
    mouth = np.zeros((frame_count, 2, 2), dtype=float)
    windows = {view: _window(mouth.copy()) for view in VIEWS}
    retained = {
        view: np.tile(np.asarray([[0, 1]]), (frame_count, 1))
        for view in VIEWS
    }
    rows = [
        _hypothesis(frame, group=0, residual=0.5)
        for frame in range(10)
    ]
    rows.extend(
        _hypothesis(frame, group=1, residual=0.5, slots=(1, 1, 1))
        for frame in range(100)
    )

    groups, _observations = build_cross_view_groups(
        pd.DataFrame(rows),
        windows,
        retained,
        config=CrossViewGroupConfig(
            minimum_support_frames=1,
            support_tau_frames=10,
            support_bonus_weight=2,
        ),
    )

    assert set(groups.median_ray_residual_mm2) == {0.5}
    assert groups.support_bonus.max() <= 2.0
    short = groups.loc[groups.support_frames.eq(10)].iloc[0]
    long = groups.loc[groups.support_frames.eq(100)].iloc[0]
    assert 0 < long.support_bonus - short.support_bonus < 2.0
    assert groups.attrs.get("summed_residual_used", False) is False


def _group_record(
    group_id: str,
    *,
    frame_start: int,
    frame_end: int,
    x_start: float,
    x_end: float,
    resource: int,
    assignment_cost: float,
) -> dict:
    velocity = (x_end - x_start) / max(frame_end - frame_start, 1)
    return {
        "group_id": group_id,
        "view_mask": 0b111,
        "view_count": 3,
        "tracklet_top_cam": resource,
        "tracklet_side1_cam": resource,
        "tracklet_side2_cam": resource,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "start_X": x_start,
        "start_Y": 0.0,
        "start_Z": 0.0,
        "end_X": x_end,
        "end_Y": 0.0,
        "end_Z": 0.0,
        "velocity_X": velocity,
        "velocity_Y": 0.0,
        "velocity_Z": 0.0,
        "assignment_cost": assignment_cost,
        "eligible": True,
        "rejection_reason": "eligible",
    }


def _observations(group_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_id": group_id,
                "frame": index,
                "ray_residual_mm2": 0.1,
                "hypothesis_id": f"h_{group_id}",
            }
            for index, group_id in enumerate(group_ids)
        ]
    )


def test_solver_leaves_positive_group_unassigned_and_enforces_resource_owner():
    groups = pd.DataFrame(
        [
            _group_record(
                f"g{index}",
                frame_start=0,
                frame_end=2,
                x_start=index * 40,
                x_end=index * 40 + 2,
                resource=index,
                assignment_cost=-1,
            )
            for index in range(3)
        ]
        + [
            _group_record(
                "ambiguous_positive",
                frame_start=3,
                frame_end=4,
                x_start=3,
                x_end=4,
                resource=0,
                assignment_cost=5,
            )
        ]
    )

    result = solve_three_identity_paths(
        groups,
        _observations(groups.group_id.tolist()),
        config=ThreePathConfig(eligible_group_drop_cost=0.5),
    )

    assert result.summary["status"] == "completed"
    assert set(result.paths.group_id) == {"g0", "g1", "g2"}
    assert not bool(
        result.dispositions.set_index("group_id").loc[
            "ambiguous_positive", "selected"
        ]
    )
    owners = (
        result.paths.merge(groups, on="group_id")
        .groupby("tracklet_top_cam")
        .path_id.nunique()
    )
    assert (owners == 1).all()


def test_solver_records_explicit_all_views_break_and_segment_restart():
    first = [
        _group_record(
            f"first_{index}",
            frame_start=0,
            frame_end=2,
            x_start=index * 40,
            x_end=index * 40 + 2,
            resource=index,
            assignment_cost=-2,
        )
        for index in range(3)
    ]
    brief = [
        _group_record(
            f"brief_{index}",
            frame_start=4,
            frame_end=6,
            x_start=index * 40 + 4,
            x_end=index * 40 + 6,
            resource=index + 10,
            assignment_cost=-2,
        )
        for index in range(3)
    ]
    long = [
        _group_record(
            f"long_{index}",
            frame_start=200,
            frame_end=202,
            x_start=index * 40 + 200,
            x_end=index * 40 + 202,
            resource=index + 20,
            assignment_cost=-2,
        )
        for index in range(3)
    ]
    groups = pd.DataFrame([*first, *brief, *long])

    result = solve_three_identity_paths(
        groups,
        _observations(groups.group_id.tolist()),
        config=ThreePathConfig(
            beam_width=1024,
            eligible_group_drop_cost=1,
            all_views_break_penalty=0.1,
            segment_restart_penalty=0.2,
        ),
    )

    assert result.summary["status"] == "completed"
    assert result.summary["all_views_break_links"] == 6
    assert result.summary["segment_restart_links"] == 3
    assert result.links.loc[
        ~result.links.segment_restart, "continuity_claimed"
    ].all()
    assert not result.links.loc[
        result.links.segment_restart, "continuity_claimed"
    ].any()
    assert result.paths.groupby("path_id").path_segment.max().eq(1).all()
    assert "path_segment" in result.selected_observations
    assert (
        result.selected_observations.groupby("path_id").path_segment.max()
        == 1
    ).all()


def test_tracklet_support_can_reject_framewise_low_residual_one_frame_decoy():
    mouth = np.zeros((3, 2, 2), dtype=float)
    windows = {view: _window(mouth.copy()) for view in VIEWS}
    retained = {
        view: np.tile(np.asarray([[0, 1]]), (3, 1))
        for view in VIEWS
    }
    rows = [
        _hypothesis(frame, group=0, residual=0.5)
        for frame in range(3)
    ]
    rows.append(
        _hypothesis(0, group=1, residual=0.0, slots=(1, 1, 1))
    )

    groups, _observations = build_cross_view_groups(
        pd.DataFrame(rows),
        windows,
        retained,
        config=CrossViewGroupConfig(minimum_support_frames=2),
    )

    by_tracklet = groups.set_index("tracklet_top_cam")
    assert bool(by_tracklet.loc[0, "eligible"])
    assert not bool(by_tracklet.loc[1, "eligible"])
    assert by_tracklet.loc[1, "rejection_reason"] == "support_below_minimum"


def test_beam_uses_bounded_last_state_on_many_groups():
    groups = []
    for index in range(300):
        path = index % 3
        frame = (index // 3) * 3
        groups.append(
            _group_record(
                f"many_{index}",
                frame_start=frame,
                frame_end=frame + 1,
                x_start=path * 40 + frame,
                x_end=path * 40 + frame + 1,
                resource=index,
                assignment_cost=-1,
            )
        )
    table = pd.DataFrame(groups)

    result = solve_three_identity_paths(
        table,
        _observations(table.group_id.tolist()),
        config=ThreePathConfig(
            beam_width=64,
            eligible_group_drop_cost=0.5,
            all_views_break_penalty=0.0,
        ),
    )

    assert result.summary["status"] == "completed"
    assert result.summary["selected_groups"] > 250
