from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scripts.reconstruction_3d.calibration import Calibration
from scripts.sleap.make_annotator_bundle import (
    _build_helpers_module,
    _patch_annotator_imports,
    build_bundle,
)
from scripts.sleap.pyqt_multiview_annotator import (
    DEFAULT_KEYPOINT_SIZE,
    FishKeypoints,
    KEYPOINT_SIZE_RADII_PX,
    MultiviewWindow,
    VIEWS,
    ViewPane,
    copy_labels_provenance,
    first_empty_fish_slot,
    full_save_path,
    instance_fit_zoom,
    keypoint_radius,
    predict_missing_view,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFRACTIVE_DIR = REPO_ROOT / "calibration_results" / "fixed_refractive_192x192x186"


def test_keypoint_circle_sizes_keep_existing_radius_as_large() -> None:
    assert DEFAULT_KEYPOINT_SIZE == "Large"
    assert KEYPOINT_SIZE_RADII_PX == {
        "Small": 4.0,
        "Medium": 6.0,
        "Large": 8.0,
    }
    assert keypoint_radius("Small") < keypoint_radius("Medium")
    assert keypoint_radius("Medium") < keypoint_radius("Large")
    with pytest.raises(ValueError, match="Unknown keypoint size"):
        keypoint_radius("Extra large")


def test_instance_fit_zoom_uses_tightest_populated_view_constraint() -> None:
    zoom = instance_fit_zoom(
        points_by_view={
            "top_cam": [(100.0, 100.0), (300.0, 200.0)],
            "side1_cam": [(50.0, 100.0), (350.0, 200.0)],
            "side2_cam": [],
        },
        focus_by_view={
            "top_cam": (200.0, 150.0),
            "side1_cam": (200.0, 150.0),
            "side2_cam": (200.0, 150.0),
        },
        viewport_sizes={view: (400, 400) for view in VIEWS},
    )
    # A 400 px pane has 336 usable display pixels after margins. side1's
    # 300 px horizontal span is the limiting dimension.
    assert zoom == pytest.approx(336.0 / 300.0)


def test_instance_fit_zoom_returns_none_without_points() -> None:
    assert instance_fit_zoom(
        points_by_view={view: [] for view in VIEWS},
        focus_by_view={view: (0.0, 0.0) for view in VIEWS},
        viewport_sizes={view: (400, 400) for view in VIEWS},
    ) is None


def test_full_save_path_places_full_before_pkg_suffix() -> None:
    assert full_save_path(Path("edits.pkg.slp")) == Path("edits_full.pkg.slp")
    assert full_save_path(Path("edits.slp")) == Path("edits_full.slp")


def test_blank_seed_reuses_first_empty_fish_slot() -> None:
    fish = [FishKeypoints(fish_id=0)]
    assert first_empty_fish_slot(fish) == 0

    fish[0].mouth["top_cam"] = (10.0, 20.0)
    assert first_empty_fish_slot(fish) is None

    fish.append(FishKeypoints(fish_id=1))
    assert first_empty_fish_slot(fish) == 1


def test_saved_provenance_copy_is_deep_and_detached() -> None:
    class Labels:
        provenance = {"selection": {"seed": 20260724}}

    copied = copy_labels_provenance(Labels())
    copied["selection"]["seed"] = 0
    assert Labels.provenance["selection"]["seed"] == 20260724


def test_independent_mode_rejects_calibration_coordinate_fill() -> None:
    class Status:
        message = ""

        def showMessage(self, message, _timeout):
            self.message = message

    window = object.__new__(MultiviewWindow)
    window.allow_calibration_fill = False
    window.status = Status()
    window.fill_missing_views()
    assert "disabled" in window.status.message
    assert "manually" in window.status.message


def test_view_pane_zoom_keeps_scene_pixel_under_cursor(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pane = ViewPane(
        "top_cam",
        lambda *_args: None,
        lambda *_args: None,
        lambda *_args: None,
    )
    pane.view.resize(420, 420)
    pane.view.show()
    app.processEvents()
    pane.set_image(np.zeros((1200, 1200, 3), dtype=np.uint8))
    pane.set_focus((600.0, 600.0), 0.5)
    app.processEvents()

    cursor = QtCore.QPoint(315, 90)
    before = pane.view.mapToScene(cursor)
    _, zoom = pane.zoom_at_viewport_pos(cursor, 1.6)
    app.processEvents()
    after = pane.view.mapToScene(cursor)

    assert zoom == pytest.approx(0.8)
    assert after.x() == pytest.approx(before.x(), abs=2.0)
    assert after.y() == pytest.approx(before.y(), abs=2.0)
    pane.view.close()


def test_view_pane_pan_moves_image_with_drag(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pan_events = []
    pane = ViewPane(
        "top_cam",
        lambda *_args: None,
        lambda *args: pan_events.append(args),
        lambda *_args: None,
    )
    pane.view.resize(420, 420)
    pane.view.show()
    app.processEvents()
    pane.set_image(np.zeros((1200, 1200, 3), dtype=np.uint8))
    pane.set_focus((600.0, 600.0), 1.0)
    app.processEvents()

    marker = QtCore.QPointF(600.0, 600.0)
    before = pane.view.mapFromScene(marker)
    viewport = pane.view.viewport()

    def send_mouse(event_type, position, button, buttons) -> None:
        point = QtCore.QPointF(*position)
        event = QtGui.QMouseEvent(
            event_type,
            point,
            point,
            point,
            button,
            buttons,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(viewport, event)
        app.processEvents()

    send_mouse(
        QtCore.QEvent.MouseButtonPress,
        (210.0, 210.0),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
    )
    send_mouse(
        QtCore.QEvent.MouseMove,
        (260.0, 180.0),
        QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton,
    )
    send_mouse(
        QtCore.QEvent.MouseButtonRelease,
        (260.0, 180.0),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
    )
    app.processEvents()
    after = pane.view.mapFromScene(marker)

    assert pane.view.viewport().hasMouseTracking()
    assert len(pan_events) == 1
    assert pan_events[0][0] == "top_cam"
    assert after.x() - before.x() == pytest.approx(50.0, abs=2.0)
    assert after.y() - before.y() == pytest.approx(-30.0, abs=2.0)
    pane.view.close()


def test_refractive_two_view_prediction_recovers_missing_projection() -> None:
    calibration = Calibration(REFRACTIVE_DIR, backend="refractive")
    world = np.array([100.0, 100.0, 50.0])
    projected = calibration.project_world(world)

    for missing_index, missing_view in enumerate(VIEWS):
        points = {
            view: tuple(projected[index]) for index, view in enumerate(VIEWS)
        }
        points[missing_view] = None
        completed = predict_missing_view(calibration, points)
        assert completed[missing_view] is not None
        np.testing.assert_allclose(
            completed[missing_view], projected[missing_index], atol=1e-6
        )


def test_bundled_helpers_are_standalone_and_load_refractive_backend() -> None:
    source = _build_helpers_module()
    assert "from scripts.calibration.refractive_geometry" not in source
    module = types.ModuleType("_test_annotator_helpers")
    module.__file__ = str(REPO_ROOT / "_test_bundle" / "annotator_helpers.py")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        calibration = module.Calibration(REFRACTIVE_DIR, backend="auto")
    finally:
        sys.modules.pop(module.__name__, None)
    assert calibration.backend_name == "refractive"
    assert calibration.score_unit == "mm^2"
    assert calibration.capabilities["two_view_triangulation"] is True


def test_bundle_builder_copies_refractive_assets_and_patches_gui(tmp_path) -> None:
    seed = tmp_path / "source.pkg.slp"
    seed.write_bytes(b"test seed placeholder")
    out_dir = tmp_path / "bundle"
    build_bundle(
        seed=seed,
        out_dir=out_dir,
        calibration_dir=REFRACTIVE_DIR,
        annotator_path=REPO_ROOT / "scripts" / "sleap" / "pyqt_multiview_annotator.py",
    )

    for camera_index in range(3):
        assert (
            out_dir / "calibration" / f"cam{camera_index}_bundle_refractive.json"
        ).is_file()
    assert (out_dir / "calibration" / "calibration_manifest.json").is_file()
    helper_source = (out_dir / "annotator_helpers.py").read_text(encoding="utf-8")
    gui_source = (out_dir / "pyqt_multiview_annotator.py").read_text(encoding="utf-8")
    compile(helper_source, "annotator_helpers.py", "exec")
    compile(gui_source, "pyqt_multiview_annotator.py", "exec")
    assert "from annotator_helpers import Calibration" in gui_source
    assert "--calibration-backend" in gui_source
    assert "--calibration-backend auto" in (out_dir / "run.bat").read_text()


def test_patched_annotator_has_no_repository_imports() -> None:
    source = (
        REPO_ROOT / "scripts" / "sleap" / "pyqt_multiview_annotator.py"
    ).read_text(encoding="utf-8")
    patched = _patch_annotator_imports(source)
    assert "from scripts.common.local_paths" not in patched
    assert "from scripts.reconstruction_3d.calibration" not in patched
    compile(patched, "pyqt_multiview_annotator.py", "exec")
