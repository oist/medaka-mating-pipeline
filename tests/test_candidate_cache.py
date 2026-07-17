from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from scripts.reconstruction_3d.candidate_cache import (
    build_view_cache,
    cache_matches_source,
    load_candidate_window,
)


class _Instance:
    def __init__(self, points, score, scores=(0.9, 0.8)):
        self._points = np.asarray(points, dtype=np.float64)
        self.score = float(score)
        self.scores = np.asarray(scores, dtype=np.float64)

    def numpy(self):
        return self._points.copy()


def test_candidate_cache_strict_mouth_and_slot_order(tmp_path, monkeypatch):
    source = tmp_path / "top_cam.slp"
    source.write_bytes(b"fake")
    # Tail-only has the highest score but must be excluded.  The two strict
    # mouths are retained in descending instance-score order.
    tail_only = _Instance([[np.nan, np.nan], [5.0, 20.0]], 0.99)
    lower = _Instance([[10.0, 10.0], [10.0, 40.0]], 0.7)
    higher = _Instance([[30.0, 30.0], [30.0, 60.0]], 0.9)
    labels = SimpleNamespace(
        labeled_frames=[SimpleNamespace(frame_idx=0, instances=[lower, tail_only, higher])],
        skeletons=[SimpleNamespace(nodes=[SimpleNamespace(name="mouth"), SimpleNamespace(name="tail")])],
    )
    monkeypatch.setitem(sys.modules, "sleap_io", SimpleNamespace(load_slp=lambda _path: labels))
    cache = tmp_path / "top_cam.candidates.h5"
    metadata = build_view_cache(source, cache, view="top_cam", num_candidates=3)
    assert metadata["coordinate_dtype"] == "float64"
    assert cache_matches_source(cache, source)
    window = load_candidate_window(cache, 0, 0)
    np.testing.assert_array_equal(window.mouth_xy[0, 0], [30.0, 30.0])
    np.testing.assert_array_equal(window.mouth_xy[0, 1], [10.0, 10.0])
    assert np.isnan(window.mouth_xy[0, 2]).all()
    assert window.visibility[0].tolist() == [3, 3, 0]
    assert window.source_instance[0].tolist() == [2, 0, -1]


def test_candidate_cache_invalidates_when_source_changes(tmp_path, monkeypatch):
    source = tmp_path / "side1_cam.slp"
    source.write_bytes(b"a")
    labels = SimpleNamespace(
        labeled_frames=[],
        skeletons=[SimpleNamespace(nodes=[SimpleNamespace(name="mouth"), SimpleNamespace(name="tail")])],
    )
    monkeypatch.setitem(sys.modules, "sleap_io", SimpleNamespace(load_slp=lambda _path: labels))
    cache = tmp_path / "side1_cam.candidates.h5"
    build_view_cache(source, cache, view="side1_cam", frame_count=1)
    assert cache_matches_source(cache, source)
    source.write_bytes(b"changed")
    assert not cache_matches_source(cache, source)
