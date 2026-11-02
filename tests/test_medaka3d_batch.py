from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reconstruction_3d.medaka3d_batch import S14_TAG, S16_TAG, choose_source, same_detections


def _views(folder, names=("top_cam", "side1_cam", "side2_cam")):
    folder.mkdir(parents=True)
    for name in names:
        (folder / f"{name}.slp").touch()


def test_complete_round_is_used(tmp_path):
    _views(tmp_path / "exp" / S14_TAG)
    src = choose_source("exp", tmp_path, S14_TAG, ())
    assert (src.tag, src.s16) == (S14_TAG, False)


def test_incomplete_round_falls_back_to_s16_for_k1(tmp_path):
    _views(tmp_path / "exp" / S14_TAG, ("top_cam",))
    src = choose_source("exp", tmp_path, S14_TAG, ("exp",))
    assert (src.tag, src.s16, src.slp_dir) == (S16_TAG, True, tmp_path / "exp" / S16_TAG)


def test_no_detections_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        choose_source("exp", tmp_path, S14_TAG, ())


def test_same_detections_treats_nan_as_equal_and_sees_differences():
    a = SimpleNamespace(frame_idx=np.array([0, 1]), points=np.array([[np.nan, 1.0], [2.0, 3.0]]))
    b = SimpleNamespace(frame_idx=np.array([0, 1]), points=a.points.copy())
    assert same_detections(a, b)
    b.points[1, 1] = 4.0
    assert not same_detections(a, b)
    assert not same_detections(a, SimpleNamespace(frame_idx=np.array([0, 2]), points=a.points))
