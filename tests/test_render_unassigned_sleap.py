import numpy as np

from scripts.visualization.render_4panel_video import (
    lookup_unassigned_2d_arrows,
    parse_frame_ranges,
    video_writer_kwargs,
)


def test_unassigned_lookup_returns_unmapped_finite_slots():
    detections = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
            [[90.0, 100.0], [110.0, 120.0]],
        ]
    )
    best_frame = {"cam0_to_id": ["ID_1", None]}
    slp_per_view = {"top_cam": {123: detections}}

    arrows = lookup_unassigned_2d_arrows(
        best_frame, slp_per_view, 123, "top_cam"
    )

    assert [slot for _mouth, _tail, slot in arrows] == [1, 2]
    np.testing.assert_array_equal(arrows[0][0], [50.0, 60.0])
    np.testing.assert_array_equal(arrows[1][1], [110.0, 120.0])


def test_unassigned_lookup_excludes_invalid_and_assigned_slots():
    detections = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [np.nan, 80.0]],
        ]
    )
    best_frame = {"cam1_to_id": ["ID_2"]}
    slp_per_view = {"side1_cam": {456: detections}}

    assert lookup_unassigned_2d_arrows(
        best_frame, slp_per_view, 456, "side1_cam"
    ) == []


def test_unassigned_lookup_handles_missing_frame():
    best_frame = {"cam2_to_id": []}
    slp_per_view = {"side2_cam": {}}

    assert lookup_unassigned_2d_arrows(
        best_frame, slp_per_view, 789, "side2_cam"
    ) == []


def test_nvenc_writer_uses_constant_quality_without_generic_qscale():
    kwargs = video_writer_kwargs("h264_nvenc", 30)

    assert kwargs["codec"] == "h264_nvenc"
    assert kwargs["quality"] is None
    assert kwargs["output_params"][-4:] == ["-cq", "19", "-b:v", "0"]


def test_libx264_writer_preserves_existing_quality():
    kwargs = video_writer_kwargs("libx264", 24)

    assert kwargs["codec"] == "libx264"
    assert kwargs["fps"] == 24
    assert kwargs["quality"] == 8


def test_parse_multiple_frame_ranges():
    assert parse_frame_ranges(["100:110", "200-205"]) == [(100, 110), (200, 205)]
