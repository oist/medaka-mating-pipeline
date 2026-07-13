import numpy as np

from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    write_mask_instances_h5,
)
from scripts.visualization.render_4panel_video import (
    draw_segmentation_masks,
    load_segmentation_masks,
)


def test_segmentation_loader_filters_frames_and_preserves_metadata(tmp_path):
    path = tmp_path / "masks.h5"
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:12, 8:18] = True
    write_mask_instances_h5(
        path,
        [
            MaskInstance(9, 0, 0, mask, (8, 5, 18, 12), 0.8),
            MaskInstance(10, 4, 4, mask, (8, 5, 18, 12), 0.95),
            MaskInstance(11, 5, 5, mask, (8, 5, 18, 12), 0.9),
        ],
        image_shape=mask.shape,
    )
    loaded = load_segmentation_masks(
        [f"side1_cam={path}"], start_frame=10, end_frame=10
    )
    assert list(loaded["side1_cam"]) == [10]
    assert loaded["side1_cam"][10][0]["instance_id"] == 4
    assert loaded["side1_cam"][10][0]["score"] == 0.95


def test_segmentation_overlay_is_translucent_and_keeps_background_visible():
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:12, 8:18] = True
    output = draw_segmentation_masks(
        image.copy(),
        [{"mask": mask, "bbox": (8, 5, 18, 12), "instance_id": 2, "score": 0.9}],
        alpha=0.2,
        contour_thickness=1,
        draw_labels=False,
    )
    assert np.array_equal(output[0, 0], image[0, 0])
    assert not np.array_equal(output[8, 12], image[8, 12])
    assert np.all(output[8, 12] > 0)
