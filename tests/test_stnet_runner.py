from __future__ import annotations

from pathlib import Path

import pytest

from UDMT.stnet import runner


def _fake_clone(root: Path) -> Path:
    stnet = root / "udmt" / "gui" / "tabs" / "ST_Net"
    pool = stnet / "ltr" / "external" / "PreciseRoIPooling" / "pytorch" / "prroi_pool"
    pool.mkdir(parents=True)
    (pool / "functional.py").write_text("from ._prroi_pooling import prroi_pool2d\n", encoding="utf-8")
    (stnet / "ltr" / "models").mkdir(parents=True)
    (stnet / "ltr" / "models" / "x.py").write_text(
        "from udmt.gui.tabs.ST_Net.ltr.models.layers import y\n"
        "from udmt.gui import BASE_DIR\n",
        encoding="utf-8",
    )
    (stnet / "pytracking").mkdir(parents=True)
    (stnet / "pytracking" / "x.py").write_text(
        "from ...pytracking.libs import dcf\n"
        "from udmt.gui.tabs.ST_Net.pytracking.utils import TrackerParams\n",
        encoding="utf-8",
    )
    return stnet


def test_patch_clone_replaces_prroi_and_decouples_imports(tmp_path: Path):
    udmt = tmp_path / "UDMT"
    stnet = _fake_clone(udmt)
    pure = tmp_path / "pure.py"
    pure.write_text("def prroi_pool2d(*args):\n    return None\n", encoding="utf-8")

    report = runner.patch_clone(udmt, pure)

    functional = Path(report.functional_py).read_text(encoding="utf-8")
    assert "from .pure_prroi import prroi_pool2d" in functional
    assert Path(report.functional_backup).exists()
    assert report.remaining_gui_import_refs == 0
    assert report.remaining_base_dir_imports == 0
    assert "from ltr.models.layers import y" in (stnet / "ltr" / "models" / "x.py").read_text(encoding="utf-8")
    assert "from pytracking.libs import dcf" in (stnet / "pytracking" / "x.py").read_text(encoding="utf-8")


def test_read_start_positions_accepts_header_variants(tmp_path: Path):
    p = tmp_path / "start.csv"
    p.write_text("frame,centroid_x,centroid_y\n0,12.5,34.0\n", encoding="utf-8")
    assert runner.read_start_positions(p) == [(12.5, 34.0)]


def test_prepare_masks_requires_a_mask_source(tmp_path: Path):
    with pytest.raises(ValueError):
        runner.prepare_masks(tmp_path / "masks", n_frames=2)


def test_prepare_masks_can_repeat_initial_mask(tmp_path: Path):
    init = tmp_path / "init.png"
    init.write_bytes(b"not really a png but copyable")
    n = runner.prepare_masks(
        tmp_path / "masks",
        n_frames=3,
        init_mask=init,
        repeat_init_mask=True,
    )
    assert n == 3
    assert (tmp_path / "masks" / "0000000.png").read_bytes() == init.read_bytes()
    assert (tmp_path / "masks" / "0000002.png").exists()
