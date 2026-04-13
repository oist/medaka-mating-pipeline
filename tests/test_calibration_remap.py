import numpy as np
import cv2

from scripts.calibration.remap_bead_labels import AnnotationRow, TemplateBead, remap_bead_rows_for_camera


def test_remap_bead_rows_for_camera_recovers_indices_from_color_and_geometry():
    image = np.full((280, 280, 3), 255, dtype=np.uint8)
    template_beads = [
        TemplateBead(index=4, label="BEAD: bead_001", color="pink", xyz=np.array([10.0, 10.0, 0.0])),
        TemplateBead(index=5, label="BEAD: bead_002", color="pink", xyz=np.array([10.0, 30.0, 0.0])),
        TemplateBead(index=6, label="BEAD: bead_003", color="yellow", xyz=np.array([30.0, 10.0, 0.0])),
        TemplateBead(index=7, label="BEAD: bead_004", color="yellow", xyz=np.array([30.0, 30.0, 0.0])),
        TemplateBead(index=8, label="BEAD: bead_005", color="blue", xyz=np.array([50.0, 10.0, 0.0])),
        TemplateBead(index=9, label="BEAD: bead_006", color="blue", xyz=np.array([50.0, 30.0, 0.0])),
    ]

    image_points = {
        4: (40, 40),
        5: (40, 140),
        6: (150, 60),
        7: (150, 160),
        8: (240, 80),
        9: (240, 180),
    }
    cv2.circle(image, image_points[4], 10, (255, 0, 255), -1)
    cv2.circle(image, image_points[5], 10, (255, 0, 255), -1)
    cv2.circle(image, image_points[6], 10, (0, 255, 255), -1)
    cv2.circle(image, image_points[7], 10, (0, 255, 255), -1)
    cv2.circle(image, image_points[8], 10, (255, 0, 0), -1)
    cv2.circle(image, image_points[9], 10, (255, 0, 0), -1)

    rows = [
        AnnotationRow(index=7, x=40.0, y=140.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
        AnnotationRow(index=6, x=150.0, y=60.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
        AnnotationRow(index=5, x=40.0, y=40.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
        AnnotationRow(index=4, x=150.0, y=160.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
        AnnotationRow(index=9, x=240.0, y=180.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
        AnnotationRow(index=8, x=240.0, y=80.0, X=0.0, Y=0.0, Z=0.0, label="bad"),
    ]

    remapped = remap_bead_rows_for_camera(image, "cam0", template_beads, rows)

    assert [row.index for row in remapped] == [4, 5, 6, 7, 8, 9]
    assert {
        (int(round(row.x)), int(round(row.y)))
        for row in remapped
    } == set(image_points.values())


def test_remap_cam2_orders_same_color_group_top_to_bottom_by_index():
    image = np.full((220, 120, 3), 255, dtype=np.uint8)
    template_beads = [
        TemplateBead(index=4, label="BEAD: bead_001", color="pink", xyz=np.array([10.0, 10.0, 0.0])),
        TemplateBead(index=5, label="BEAD: bead_002", color="pink", xyz=np.array([10.0, 30.0, 0.0])),
        TemplateBead(index=6, label="BEAD: bead_003", color="pink", xyz=np.array([10.0, 50.0, 0.0])),
        TemplateBead(index=11, label="BEAD: bead_008", color="yellow", xyz=np.array([30.0, 10.0, 0.0])),
        TemplateBead(index=12, label="BEAD: bead_009", color="yellow", xyz=np.array([30.0, 30.0, 0.0])),
        TemplateBead(index=13, label="BEAD: bead_010", color="yellow", xyz=np.array([30.0, 50.0, 0.0])),
        TemplateBead(index=18, label="BEAD: bead_015", color="blue", xyz=np.array([50.0, 10.0, 0.0])),
        TemplateBead(index=19, label="BEAD: bead_016", color="blue", xyz=np.array([50.0, 30.0, 0.0])),
        TemplateBead(index=20, label="BEAD: bead_017", color="blue", xyz=np.array([50.0, 50.0, 0.0])),
    ]
    for point in [(30, 170), (30, 30), (30, 100)]:
        cv2.circle(image, point, 8, (255, 0, 255), -1)
    for point in [(80, 100), (80, 170), (80, 30)]:
        cv2.circle(image, point, 8, (0, 255, 255), -1)
    for point in [(105, 170), (105, 30), (105, 100)]:
        cv2.circle(image, point, 8, (255, 0, 0), -1)

    rows = [
        AnnotationRow(index=6, x=30.0, y=170.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=4, x=30.0, y=30.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=5, x=30.0, y=100.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=12, x=80.0, y=100.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=13, x=80.0, y=170.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=11, x=80.0, y=30.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=20, x=105.0, y=170.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=18, x=105.0, y=30.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
        AnnotationRow(index=19, x=105.0, y=100.0, X=0.0, Y=0.0, Z=0.0, label="BEAD: old"),
    ]

    remapped = remap_bead_rows_for_camera(image, "cam2", template_beads, rows)
    by_index = {row.index: row for row in remapped}

    assert [by_index[i].y for i in [4, 5, 6]] == [30.0, 100.0, 170.0]
    assert [by_index[i].y for i in [11, 12, 13]] == [30.0, 100.0, 170.0]
    assert [by_index[i].y for i in [18, 19, 20]] == [30.0, 100.0, 170.0]
