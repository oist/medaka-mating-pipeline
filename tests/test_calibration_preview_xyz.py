from scripts.calibration.preview_annotation_xyz import _row_color


def test_row_color_uses_bead_index_not_world_x_for_rotated_ways():
    row = {
        "index": 4,
        "label": "BEAD: beads_way2_001",
        "X": 162.5,
        "Y": 27.5,
        "Z": 162.5,
    }

    assert _row_color(row) == (255, 0, 255)
