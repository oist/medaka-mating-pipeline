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
    KEYPOINT_SIZE_RADII_PX,
    VIEWS,
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
