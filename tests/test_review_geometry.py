"""Reviewer geometry edits recompute 3D from all detections (fake orthographic rig)."""

import numpy as np
import pandas as pd

from scripts.review.geometry import (
    Detections,
    Geometry,
    apply_geometry,
    confirm_value,
    effective_intervals,
    load_solutions,
    recompute,
    solve,
)
from scripts.review.ledger import EVENT_COLUMNS
from scripts.review.review_data import DET_NONE, DET_SCALE

from test_review_reconstruct import FakeCal

FISH = ["ID_1", "ID_2", "ID_3"]


def _events(rows):
    out = []
    for i, row in enumerate(rows):
        kind, fish, frame, value = row[:4]
        out.append({c: "" for c in EVENT_COLUMNS} | {
            "event_id": str(i), "group": str(i), "kind": kind, "fish": fish, "frame": str(frame),
            "frame_end": str(row[4]) if len(row) > 4 else "", "frame_marked": str(frame), "value": value})
    return pd.DataFrame(out, columns=EVENT_COLUMNS)


def _scene(tmp_path, n=120):
    """Three fish swimming apart; detections in slot order = fish order; tracker slots = identity."""
    cal = FakeCal()
    t = np.arange(n)
    axis = np.array([1.0, 0.3, 0.1]) / np.linalg.norm([1.0, 0.3, 0.1])
    pect = np.stack([np.stack([30 + 0.3 * t + 25 * k, 40 + 12 * k + 0 * t, 60 - 0.2 * t + 7 * k], axis=1)
                     for k in range(3)], axis=1)                                         # (F, K, 3)
    nodes = np.stack([pect + 9 * axis, pect, pect - 16 * axis], axis=2)                 # (F, K, N, 3)
    pts = cal.project_world(nodes)                                                       # (F, K, N, V, 2)
    det = np.moveaxis(pts, 3, 1)                                                         # (F, V, K, N, 2)
    np.savez_compressed(tmp_path / "detections.npz", frames=t, det=np.round(det * DET_SCALE).astype(np.int16),
                        slots=np.tile(np.arange(3, dtype=np.int8)[None, :, None], (n, 1, 3)),
                        scale=DET_SCALE, none=DET_NONE)
    xyz = (nodes[:, :, 0] + nodes[:, :, 2]) / 2
    data = {"frames": t, "fish_ids": np.array(FISH), "node_names": np.array(["mouth", "pectoral", "tail"]),
            "head_node": 0, "tail_node": 2, "xyz": xyz.astype(np.float32),
            "kp2d": np.moveaxis(pts, 3, 2).astype(np.float32), "proj2d": cal.project_world(xyz).astype(np.float32),
            "nodes3d_filled": nodes.astype(np.float32), "node_source": np.zeros((n, 3, 3), np.int8),
            "missing_node_proj2d": np.full((n, 3, 3, 3, 2), np.nan, np.float32)}
    geo = Geometry(data, Detections(tmp_path), cal, fps=166.0)
    return geo, nodes, xyz


def _break_tracker(tmp_path, frames, views, a=0, b=1):
    """The tracker gave tracks a and b each other's detections in ``views`` on ``frames``."""
    z = dict(np.load(tmp_path / "detections.npz"))
    for v in views:
        z["slots"][frames, a, v], z["slots"][frames, b, v] = b, a
    np.savez_compressed(tmp_path / "detections.npz", **z)


def test_view_swap_event_repairs_a_wrong_side_view_assignment(tmp_path):
    geo, nodes, xyz = _scene(tmp_path)
    _break_tracker(tmp_path, slice(50, 60), [1])
    geo.dets = Detections(tmp_path)
    broken = recompute(geo, _events([]), {}, 40, 70)
    err_broken = np.linalg.norm(broken["xyz"][10:20, 0] - xyz[50:60, 0], axis=1)
    fixed = recompute(geo, _events([("view_swap", "ID_1|ID_2", 50, "side1_cam", 59)]), {}, 40, 70)
    err_fixed = np.linalg.norm(fixed["xyz"][:, :2] - xyz[40:71, :2], axis=2)
    assert err_broken.min() > 3.0
    assert err_fixed.max() < 0.2
    head_err = np.linalg.norm(fixed["nodes3d_filled"][:, :2, 0] - nodes[40:71, :2, 0], axis=2)
    assert head_err.max() < 0.5


def test_flagged_body_is_reestimated_and_its_centre_follows(tmp_path):
    geo, nodes, xyz = _scene(tmp_path)
    ev = _events([("nodes_bad", "ID_3", 30, "1"), ("nodes_bad", "ID_3", 40, "0")])
    out = recompute(geo, ev, {}, 30, 39)
    assert (out["node_source"][:, 2, [0, 2]] == 4).all()
    assert np.linalg.norm(out["nodes3d_filled"][:, 2] - nodes[30:40, 2], axis=2).max() < 0.5
    assert np.linalg.norm(out["xyz"][:, 2] - xyz[30:40, 2], axis=1).max() < 0.5


def test_solving_an_interval_between_keyframes_undoes_a_tracker_swap(tmp_path):
    geo, nodes, xyz = _scene(tmp_path)
    _break_tracker(tmp_path, slice(70, 80), [0, 1, 2])          # tracks ID_1 and ID_2 swapped in every view
    geo.dets = Detections(tmp_path)
    heads = {f: (nodes[60, k, 0], nodes[60, k, 2], False) for k, f in enumerate(FISH)}
    heads_end = {f: (nodes[99, k, 0], nodes[99, k, 2], False) for k, f in enumerate(FISH)}
    rows = [("interval", "ID_1|ID_2", 60, "U001", 99), ("confirm", "*", 60, confirm_value(heads)),
            ("confirm", "*", 99, confirm_value(heads_end))]
    ev = _events(rows)
    intervals = effective_intervals(None, ev)
    res = solve(geo, ev, intervals.iloc[0].to_dict(), tmp_path)
    assert res["frames_changed"] == 10
    assert set(load_solutions(tmp_path, intervals, ev, FISH, 166.0)) == {"U001"}
    data, ranges = apply_geometry(geo, ev, tmp_path, intervals)
    assert ranges == [(60, 99)]
    err = np.linalg.norm(data["xyz"][60:100, :2] - xyz[60:100, :2], axis=2)
    assert err.max() < 0.2
    # Anything the solution depends on makes it stale (ignored until solved again): a moved keyframe,
    # an identity swap before the interval, a view swap inside it.
    for extra in ([("confirm", "*", 95, confirm_value(heads_end))], [("swap", "ID_1|ID_2", 30, "1")],
                  [("view_swap", "ID_1|ID_3", 80, "top_cam", 85)]):
        ev2 = _events(rows + extra)
        assert load_solutions(tmp_path, effective_intervals(None, ev2), ev2, FISH, 166.0) == {}, extra


def test_a_keyframe_on_a_solved_frame_keeps_what_the_reviewer_saw(tmp_path):
    """Confirming a frame the solver had fixed must pin the fixed identities, not the tracker's."""
    geo, nodes, xyz = _scene(tmp_path)
    _break_tracker(tmp_path, slice(70, 80), [0, 1, 2])
    geo.dets = Detections(tmp_path)
    ends = {t: {f: (nodes[t, k, 0], nodes[t, k, 2], False) for k, f in enumerate(FISH)} for t in (60, 75, 99)}
    rows = [("interval", "ID_1|ID_2", 60, "U001", 99), ("confirm", "*", 60, confirm_value(ends[60])),
            ("confirm", "*", 99, confirm_value(ends[99]))]
    ev = _events(rows)
    intervals = effective_intervals(None, ev)
    solve(geo, ev, intervals.iloc[0].to_dict(), tmp_path)
    # The reviewer sees the solved (correct) fish at 75 and confirms it, then solves again.
    ev = _events(rows + [("confirm", "*", 75, confirm_value(ends[75]))])
    solve(geo, ev, intervals.iloc[0].to_dict(), tmp_path)
    data, _ = apply_geometry(geo, ev, tmp_path, intervals)
    assert np.linalg.norm(data["xyz"][60:100, :2] - xyz[60:100, :2], axis=2).max() < 0.2


def test_an_edit_starting_before_the_first_data_frame_still_applies(tmp_path):
    geo, nodes, xyz = _scene(tmp_path)
    _break_tracker(tmp_path, slice(0, 10), [1])
    geo.dets = Detections(tmp_path)
    ev = _events([("view_swap", "ID_1|ID_2", -5, "side1_cam", 9)])
    data, ranges = apply_geometry(geo, ev, tmp_path, effective_intervals(None, ev))
    assert ranges == [(0, 9)]
    assert np.linalg.norm(data["xyz"][0:10, :2] - xyz[0:10, :2], axis=2).max() < 0.2


def test_an_open_body_flag_is_not_reestimated_until_closed(tmp_path):
    geo, nodes, xyz = _scene(tmp_path)
    ev = _events([("nodes_bad", "ID_3", 30, "1"), ("view_drop", "ID_3", 25, "top_cam", 45)])
    out = recompute(geo, ev, {}, 25, 45)
    assert not (out["node_source"] == 4).any()


def test_effective_intervals_apply_reviewer_moves_and_deletions():
    machine = pd.DataFrame({"interval_id": ["M001", "M002"], "frame_start": [10, 50], "frame_end": [20, 60],
                            "fish": ["ID_1|ID_2", "ID_2|ID_3"], "min_mm": [3.0, 4.0]})
    ev = _events([("interval", "ID_1|ID_2", 12, "M001", 25), ("interval_del", "ID_2|ID_3", 50, "M002", 60),
                  ("interval", "ID_1|ID_3", 80, "U001", 90)])
    out = effective_intervals(machine, ev)
    assert list(out["interval_id"]) == ["M001", "U001"]
    assert (out.iloc[0]["frame_start"], out.iloc[0]["frame_end"], out.iloc[0]["origin"]) == (12, 25, "machine")
