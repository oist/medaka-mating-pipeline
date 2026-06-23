"""Unit tests for the reprojection-based slot-point builder that feeds the
female-anchor seg-ID chain (script 33) on experiments without tracklet-bind."""

import numpy as np
import pandas as pd
import pytest

from sam2_id_experiment.src.id_model.reproject_slot_points import (
    SLOT_POINT_COLUMNS,
    build_slot_points,
    slot_from_fish_id,
)


class _StubProjector:
    """Deterministic fake calibration: each view shifts the world point by a
    fixed per-view offset so we can assert exactly which pixel each fish lands on,
    without loading the real joblib model."""

    OFFSET = {"side1_cam": 1000.0, "side2_cam": 2000.0, "top_cam": 0.0}

    def project_world_to_view(self, xyz, view):
        xyz = np.asarray(xyz, dtype=float)
        # map (X, Y, Z) -> (X + offset, Y) so x encodes both fish and view
        return np.column_stack([xyz[:, 0] + self.OFFSET[view], xyz[:, 1]])


def test_slot_from_fish_id():
    assert slot_from_fish_id("ID_1") == 0
    assert slot_from_fish_id("ID_2") == 1
    assert slot_from_fish_id("ID_3") == 2
    # plain ints already zero-based
    assert slot_from_fish_id(0) == 0
    assert slot_from_fish_id(2) == 2


def test_slot_from_fish_id_rejects_garbage():
    with pytest.raises(ValueError):
        slot_from_fish_id("female")


def _frame(frame, fish_id, x, y, z):
    return {"frame": frame, "fish_id": fish_id, "X": x, "Y": y, "Z": z}


def test_build_slot_points_maps_slots_and_reprojects():
    df = pd.DataFrame([
        _frame(0, "ID_1", 10.0, 50.0, 0.0),
        _frame(0, "ID_2", 20.0, 60.0, 5.0),
        _frame(0, "ID_3", 30.0, 70.0, 9.0),
        _frame(1, "ID_1", 11.0, 51.0, 1.0),
    ])
    out = build_slot_points(df, _StubProjector(), ["side1_cam", "side2_cam"])

    assert set(out) == {"side1_cam", "side2_cam"}
    s1 = out["side1_cam"]
    assert list(s1.columns) == SLOT_POINT_COLUMNS
    # global_id == tracklet_id == slot, with ID_1/2/3 -> 0/1/2
    row = s1[(s1.frame == 0) & (s1.global_id == 1)].iloc[0]
    assert row.tracklet_id == 1
    assert row.x == pytest.approx(20.0 + 1000.0)  # side1 offset applied
    assert row.y == pytest.approx(60.0)
    # side2 uses a different offset for the same fish
    row2 = out["side2_cam"]
    r2 = row2[(row2.frame == 0) & (row2.global_id == 1)].iloc[0]
    assert r2.x == pytest.approx(20.0 + 2000.0)


def test_build_slot_points_skips_non_finite_positions():
    df = pd.DataFrame([
        _frame(0, "ID_1", 10.0, 50.0, 0.0),
        _frame(0, "ID_2", np.nan, np.nan, np.nan),   # top occluded -> dropped
        _frame(0, "ID_3", 30.0, 70.0, 9.0),
    ])
    out = build_slot_points(df, _StubProjector(), ["side1_cam"])
    s1 = out["side1_cam"]
    assert sorted(s1.global_id.tolist()) == [0, 2]   # slot 1 (ID_2) dropped
    assert len(s1) == 2


def test_build_slot_points_missing_columns_raises():
    df = pd.DataFrame([{"frame": 0, "fish_id": "ID_1", "X": 1.0, "Y": 2.0}])  # no Z
    with pytest.raises(ValueError):
        build_slot_points(df, _StubProjector(), ["side1_cam"])


def test_build_slot_points_all_nan_returns_empty():
    df = pd.DataFrame([_frame(0, "ID_1", np.nan, np.nan, np.nan)])
    out = build_slot_points(df, _StubProjector(), ["side1_cam"])
    assert out["side1_cam"].empty
    assert list(out["side1_cam"].columns) == SLOT_POINT_COLUMNS
