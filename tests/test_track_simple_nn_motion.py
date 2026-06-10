from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from scripts.tracking.track_simple_nn import (
    MotionAwareConfig,
    _ActiveTrack,
    _build_motion_cost_matrix,
    _instance_position,
    _record_detection,
    track,
    write_csv,
)


class _FakeNode:
    def __init__(self, name: str):
        self.name = name


class _FakeSkeleton:
    def __init__(self, node_names: list[str]):
        self.nodes = [_FakeNode(name) for name in node_names]


class _FakeInstance:
    def __init__(self, points: list[tuple[float, float]], score: float = 1.0):
        self._points = np.asarray(points, dtype=float)
        self.score = score

    def numpy(self):
        return self._points


class _FakeFrame:
    def __init__(self, frame_idx: int, instances: list[_FakeInstance]):
        self.frame_idx = frame_idx
        self.instances = instances


class _FakeLabels:
    def __init__(self, frames: list[_FakeFrame], node_names: list[str] | None = None):
        self.skeletons = [_FakeSkeleton(node_names or ["mouth", "tail"])]
        self.labeled_frames = frames


def _install_fake_sleap(monkeypatch, labels: _FakeLabels) -> None:
    fake = types.SimpleNamespace(load_slp=lambda _path: labels)
    monkeypatch.setitem(sys.modules, "sleap_io", fake)


def test_mouth_anchor_uses_mouth_point_instead_of_centroid():
    inst = _FakeInstance([(10.0, 20.0), (30.0, 40.0)])

    assert _instance_position(inst, node_names=["mouth", "tail"], anchor_node="mouth") == (10.0, 20.0)
    assert _instance_position(inst, node_names=["mouth", "tail"], anchor_node=None) == (20.0, 30.0)


def test_motion_cost_penalizes_reversal_more_than_forward_motion():
    tr = _ActiveTrack(track_id=0, last_frame=0, last_pos=(0.0, 0.0))
    _record_detection(tr, 0, {"x": 0.0, "y": 0.0, "score": 1.0})
    _record_detection(tr, 1, {"x": 10.0, "y": 0.0, "score": 1.0})
    active = {0: tr}
    detections = [
        {"x": 20.0, "y": 0.0, "score": 1.0, "close_contact": False},
        {"x": 0.0, "y": 0.0, "score": 1.0, "close_contact": False},
    ]

    _, cost, diagnostics = _build_motion_cost_matrix(
        active,
        detections,
        frame=2,
        cfg=MotionAwareConfig(enabled=True),
        max_distance=100.0,
    )

    assert cost[0, 0] < cost[0, 1]
    assert diagnostics[0][1]["heading_change_rad"] > 3.0


def test_motion_aware_tracker_bridges_ambiguous_close_contact(monkeypatch, tmp_path):
    labels = _FakeLabels(
        [
            _FakeFrame(0, [_FakeInstance([(0.0, 0.0), (0.0, 5.0)]), _FakeInstance([(100.0, 0.0), (100.0, 5.0)])]),
            _FakeFrame(1, [_FakeInstance([(10.0, 0.0), (10.0, 5.0)]), _FakeInstance([(90.0, 0.0), (90.0, 5.0)])]),
            # Fish are very close and both assignments are intentionally ambiguous.
            _FakeFrame(2, [_FakeInstance([(49.0, 0.0), (49.0, 5.0)]), _FakeInstance([(51.0, 0.0), (51.0, 5.0)])]),
            _FakeFrame(3, [_FakeInstance([(30.0, 0.0), (30.0, 5.0)]), _FakeInstance([(70.0, 0.0), (70.0, 5.0)])]),
        ]
    )
    _install_fake_sleap(monkeypatch, labels)

    rows, stats = track(
        tmp_path / "fake.slp",
        max_distance=100.0,
        max_tracks=2,
        max_gap=10,
        motion_aware=True,
        motion_config=MotionAwareConfig(
            enabled=True,
            close_contact_distance_px=10.0,
            close_contact_min_margin=100.0,
            max_speed_ratio=10.0,
            local_repair=False,
        ),
    )

    frame2_rows = [row for row in rows if row["frame"] == 2]
    assert frame2_rows == []
    assert stats["bridged_matches"] >= 1
    assert {row["node"] for row in rows} == {"mouth"}


def test_motion_aware_csv_keeps_bind_columns_and_adds_diagnostics(tmp_path):
    out = tmp_path / "tracks.csv"
    write_csv(
        [
            {
                "frame": 1,
                "track_id": 0,
                "x": 10.0,
                "y": 20.0,
                "score": 0.9,
                "node": "mouth",
                "match_cost": 1.5,
                "zigzag_flag": False,
            }
        ],
        out,
    )

    df = pd.read_csv(out)
    assert {"frame", "track_id", "x", "y", "score", "node"}.issubset(df.columns)
    assert {"match_cost", "zigzag_flag"}.issubset(df.columns)
