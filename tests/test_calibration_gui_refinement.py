import cv2
import numpy as np

from scripts.calibration.annotate_calibration_gui import (
    PointSpec,
    _append_bottom_corners,
    _annotation_sort_key,
    CalibrationAnnotatorGUI,
    refine_point_from_bbox,
    select_target_index_from_guides,
)


def test_refine_point_from_bbox_finds_colored_bead_center():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    center = (120, 80)
    cv2.circle(image, center, 12, (255, 0, 255), -1)

    spec = PointSpec(
        index=4,
        label="BEAD: bead_001",
        xyz=np.array([27.5, 27.5, 27.5], dtype=float),
        kind="bead",
        color="pink",
    )
    refined = refine_point_from_bbox(image, (90, 50, 150, 110), spec)

    assert np.allclose(refined, np.array(center, dtype=float), atol=3.0)


def test_refine_point_from_bbox_finds_red_auxiliary_corner():
    image = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.line(image, (40, 70), (110, 70), (0, 0, 255), 4)
    cv2.line(image, (110, 70), (110, 120), (0, 0, 255), 4)

    spec = PointSpec(
        index=0,
        label="AUX: pillar_1_line_01_inner",
        xyz=None,
        kind="aux",
        color="pillar_1",
    )
    refined = refine_point_from_bbox(image, (70, 40, 130, 130), spec)

    assert np.allclose(refined, np.array([110.0, 70.0]), atol=6.0)


def test_select_target_index_from_guides_prefers_nearest_missing_point():
    guides = np.array([[20.0, 20.0], [80.0, 80.0], [140.0, 140.0]], dtype=float)
    occupied = np.array([True, False, False], dtype=bool)

    picked = select_target_index_from_guides(
        np.array([76.0, 84.0], dtype=float),
        guides,
        occupied,
        current_index=0,
        max_distance=50.0,
    )

    assert picked == 1


def test_append_bottom_corners_adds_symmetric_bottom_points():
    specs = [
        PointSpec(index=0, label="CORNER: corner_1", xyz=np.array([0.0, 0.0, 190.0]), kind="corner", color="corner"),
        PointSpec(index=1, label="CORNER: corner_2", xyz=np.array([190.0, 0.0, 190.0]), kind="corner", color="corner"),
        PointSpec(index=2, label="CORNER: corner_3", xyz=np.array([190.0, 190.0, 190.0]), kind="corner", color="corner"),
        PointSpec(index=3, label="CORNER: corner_4", xyz=np.array([0.0, 190.0, 190.0]), kind="corner", color="corner"),
        PointSpec(index=4, label="BEAD: bead_001", xyz=np.array([27.5, 27.5, 27.5]), kind="bead", color="pink"),
    ]

    extended = _append_bottom_corners(specs)

    assert len(extended) == 9
    assert extended[-4].index == 5
    assert np.allclose(extended[-4].xyz, np.array([0.0, 0.0, 4.0]))
    assert np.allclose(extended[-1].xyz, np.array([0.0, 190.0, 4.0]))


def test_annotation_sort_key_prioritizes_beads_then_top_then_bottom_corners():
    specs = [
        PointSpec(index=0, label="CORNER: corner_1", xyz=np.array([0.0, 0.0, 190.0]), kind="corner", color="corner"),
        PointSpec(index=4, label="BEAD: bead_001", xyz=np.array([27.5, 27.5, 27.5]), kind="bead", color="pink"),
        PointSpec(index=53, label="CORNER_BOTTOM: corner_1_bottom", xyz=np.array([0.0, 0.0, 0.0]), kind="corner", color="corner"),
        PointSpec(index=5, label="BEAD: bead_002", xyz=np.array([27.5, 50.0, 50.0]), kind="bead", color="pink"),
    ]

    ordered = sorted(specs, key=_annotation_sort_key)

    assert [spec.label for spec in ordered] == [
        "BEAD: bead_001",
        "BEAD: bead_002",
        "CORNER: corner_1",
        "CORNER_BOTTOM: corner_1_bottom",
    ]


def test_auto_assignment_is_off_by_default_for_calibration_gui(tmp_path):
    image_path = tmp_path / "test_cam0.png"
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    output_path = tmp_path / "out.txt"
    specs = [
        PointSpec(index=4, label="BEAD: bead_001", xyz=np.array([0.0, 0.0, 0.0]), kind="bead", color="pink"),
        PointSpec(index=5, label="BEAD: bead_002", xyz=np.array([1.0, 1.0, 1.0]), kind="bead", color="pink"),
    ]

    gui = CalibrationAnnotatorGUI(
        image_path=image_path,
        output_path=output_path,
        specs=specs,
        mode="calibration",
    )

    try:
        assert not gui.auto_assign
    finally:
        import matplotlib.pyplot as plt
        plt.close(gui.fig)
