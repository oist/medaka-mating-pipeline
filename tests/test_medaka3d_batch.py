from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reconstruction_3d.medaka3d_batch import (
    S14_TAG,
    S16_TAG,
    chain_readiness,
    choose_source,
    same_detections,
)


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


def test_chain_readiness_waits_then_runs_then_is_done(tmp_path):
    sleap = tmp_path / "sleap"
    recon = tmp_path / "recon"
    root = tmp_path / "data"
    kwargs = dict(sleap_output_dir=sleap, reconstruction_output_dir=recon, data_root=root,
                  round_tag=S14_TAG, s16_experiments=())
    assert chain_readiness("exp", **kwargs) == "waiting"

    _views(sleap / "exp" / S14_TAG)
    assert chain_readiness("exp", **kwargs) == "ready"

    tag = f"{S14_TAG}_m3dfull"
    (recon / "exp" / tag).mkdir(parents=True)
    (recon / "exp" / tag / f"{tag}_3d.csv").touch()
    (root / "review_packages" / "exp" / tag).mkdir(parents=True)
    (root / "review_packages" / "exp" / tag / "review_manifest.json").touch()
    assert chain_readiness("exp", **kwargs) == "ready"     # trajectory QC still missing
    (root / "review_packages" / "exp" / tag / "trajectory_qc_summary.json").touch()
    assert chain_readiness("exp", **kwargs) == "done"


def test_chain_readiness_treats_k1_s16_as_ready_without_s14(tmp_path):
    state = chain_readiness(
        "k1", sleap_output_dir=tmp_path / "sleap", reconstruction_output_dir=tmp_path / "recon",
        data_root=tmp_path / "data", round_tag=S14_TAG, s16_experiments=("k1",))
    assert state == "ready"


def test_same_detections_treats_nan_as_equal_and_sees_differences():
    a = SimpleNamespace(frame_idx=np.array([0, 1]), points=np.array([[np.nan, 1.0], [2.0, 3.0]]))
    b = SimpleNamespace(frame_idx=np.array([0, 1]), points=a.points.copy())
    assert same_detections(a, b)
    b.points[1, 1] = 4.0
    assert not same_detections(a, b)
    assert not same_detections(a, SimpleNamespace(frame_idx=np.array([0, 2]), points=a.points))
