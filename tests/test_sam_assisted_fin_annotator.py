from argparse import Namespace
import importlib

import numpy as np


annotator_module = importlib.import_module(
    "sam2_id_experiment.scripts.47_sam_assisted_fin_annotator"
)


def _annotator() -> object:
    annotator = annotator_module.Annotator.__new__(annotator_module.Annotator)
    annotator.image = np.zeros((1200, 1200, 3), dtype=np.uint8)
    annotator.args = Namespace(display_scale=0.5, auto_zoom_padding=2.5, no_auto_zoom=False)
    annotator.zoom = 1.0
    annotator.view_center = (600.0, 600.0)
    annotator.frame_masks = []
    annotator.selected_mask_index = 0
    annotator.overlays_visible = True
    annotator.states = {
        "body": annotator_module.RegionState(),
        "fins": annotator_module.RegionState(),
    }
    annotator.status = ""
    return annotator


def test_zoom_anchor_keeps_image_point_under_cursor() -> None:
    annotator = _annotator()
    anchor = (450, 150)
    before = annotator._display_to_image_float(*anchor)

    annotator.set_zoom(2.0, anchor=anchor)

    after = annotator._display_to_image_float(*anchor)
    assert np.allclose(after, before, atol=1.0)
    assert annotator._view_bounds() == (450, 150, 1050, 750)


def test_zoom_mapping_and_reset() -> None:
    annotator = _annotator()
    annotator.set_zoom(2.0)

    assert annotator._display_to_image(300, 300) == (600, 600)
    assert annotator._display_to_image(0, 0) == (300, 300)

    annotator.set_zoom(1.0)
    assert annotator._view_bounds() == (0, 0, 1200, 1200)
    assert annotator._display_to_image(300, 300) == (600, 600)


def test_zoom_is_bounded() -> None:
    annotator = _annotator()
    annotator.set_zoom(100.0)
    assert annotator.zoom == 8.0
    annotator.set_zoom(0.01)
    assert annotator.zoom == 1.0


def test_focus_selected_instance_centers_and_frames_mask() -> None:
    annotator = _annotator()
    mask = np.zeros((1200, 1200), dtype=bool)
    mask[400:500, 700:900] = True
    annotator.frame_masks = [{"instance_id": 4, "mask": mask}]

    assert annotator.focus_selected_instance()

    assert annotator.zoom == 2.4
    assert np.allclose(annotator.view_center, (799.5, 449.5))
    x0, y0, x1, y1 = annotator._view_bounds()
    assert x0 < 700 < 900 < x1
    assert y0 < 400 < 500 < y1


def test_auto_focus_can_be_disabled() -> None:
    annotator = _annotator()
    annotator.args.no_auto_zoom = True
    mask = np.zeros((1200, 1200), dtype=bool)
    mask[400:500, 700:900] = True
    annotator.frame_masks = [{"instance_id": 4, "mask": mask}]

    assert not annotator.focus_selected_instance()
    assert annotator.zoom == 1.0
    assert annotator.view_center == (600.0, 600.0)


def test_raw_preview_toggle_preserves_annotation_state() -> None:
    annotator = _annotator()
    mask = np.zeros((1200, 1200), dtype=bool)
    mask[400:500, 700:900] = True
    annotator.states["body"].points = [[712.0, 432.0]]
    annotator.states["body"].labels = [1]
    annotator.states["body"].masks = [mask]
    annotator.states["body"].scores = [0.9]
    state_before = (
        list(annotator.states["body"].points),
        list(annotator.states["body"].labels),
        annotator.states["body"].masks[0],
        list(annotator.states["body"].scores),
    )

    annotator.toggle_overlays()
    assert not annotator.overlays_visible
    assert annotator.states["body"].points == state_before[0]
    assert annotator.states["body"].labels == state_before[1]
    assert annotator.states["body"].masks[0] is state_before[2]
    assert annotator.states["body"].scores == state_before[3]

    annotator.toggle_overlays()
    assert annotator.overlays_visible
