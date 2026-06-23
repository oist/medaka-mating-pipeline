"""Unit tests for the tracklet -> 3-fish identity benchmark (pure logic only)."""

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.tracklet_id_benchmark import methods, metrics
from scripts.experiments.tracklet_id_benchmark.windows import select_windows


# --------------------------- window selection --------------------------- #

def _events():
    return pd.DataFrame([
        {"pair": "ID_1|ID_2", "start": 1000, "end": 1010, "n_frames": 11, "min_mm": 5.0,
         "verdict": "clear", "reason": "velocity_consistent"},
        {"pair": "ID_1|ID_3", "start": 50000, "end": 50300, "n_frames": 301, "min_mm": 6.0,
         "verdict": "review", "reason": "ambiguous"},
        {"pair": "ID_2|ID_3", "start": 120000, "end": 120020, "n_frames": 21, "min_mm": 4.0,
         "verdict": "review", "reason": "suspect_swap"},
        {"pair": "ID_1|ID_2", "start": 200000, "end": 200015, "n_frames": 16, "min_mm": 3.5,
         "verdict": "review", "reason": "suspect_swap"},
    ])


def test_select_windows_covers_taxonomy():
    w = select_windows(_events(), total_frames=300000, n_suspect=4)
    kinds = {x["kind"] for x in w}
    assert {"early_fp", "easy_separated", "crossing_clear",
            "dwell_ambiguous", "suspect_swap"} <= kinds
    # suspect windows: one per suspect event (sorted by n_frames desc)
    assert sum(x["kind"] == "suspect_swap" for x in w) == 2


def test_select_windows_bounds_and_determinism():
    w1 = select_windows(_events(), total_frames=300000, pad_frames=400)
    w2 = select_windows(_events(), total_frames=300000, pad_frames=400)
    assert [x["name"] for x in w1] == [x["name"] for x in w2]
    for x in w1:
        assert 0 <= x["start"] <= x["end"] <= 299999
        assert x["end"] - x["start"] + 1 == x["n_frames"]


# --------------------------- AssignmentTable metrics --------------------------- #

def _assign(frames, slots, flag="", with_xyz=True):
    rows = []
    for f in frames:
        for s in slots:
            rows.append({"frame": f, "fish_id": s,
                         "X": float(s) + 0.01 * f if with_xyz else np.nan,
                         "Y": 0.0, "Z": 0.0, "conf": 1.0, "flag": flag, "method": "m"})
    return pd.DataFrame(rows)[metrics.ASSIGN_COLS]


def test_coverage_full_and_partial():
    df = _assign(range(0, 100), [0, 1, 2])
    assert metrics.coverage(df, 0, 99, 3) == 1.0
    df2 = _assign(range(0, 100), [0, 1])  # only 2 of 3 fish
    assert metrics.coverage(df2, 0, 99, 3) == 0.0


def test_unknown_frac_counts_flags_and_missing():
    df = _assign(range(0, 10), [0, 1, 2])
    assert metrics.unknown_frac(df, 0, 9, 3) == 0.0
    flagged = _assign(range(0, 10), [0, 1, 2], flag="hardcore")
    assert metrics.unknown_frac(flagged, 0, 9, 3) == 1.0
    # one fish missing entirely -> 1/3 unknown
    df2 = _assign(range(0, 10), [0, 1])
    assert metrics.unknown_frac(df2, 0, 9, 3) == pytest.approx(1 / 3, abs=1e-6)


def test_speed_spike_detects_jump():
    # smooth track: tiny steps -> no spikes
    df = _assign(range(0, 50), [0, 1, 2])
    assert metrics.speed_spike_rate(df, 0, 49, spike_mm=8.0)["mean_per_k"] == 0.0
    # inject a teleport for fish 0 at frame 25
    df.loc[(df.frame == 25) & (df.fish_id == 0), "X"] = 1000.0
    out = metrics.speed_spike_rate(df, 0, 49, spike_mm=8.0)
    assert out["per_fish"][0] > 0


def test_exactly_three():
    df = _assign(range(0, 10), [0, 1, 2])
    g = metrics.exactly_three(df, 0, 9, 3)
    assert g["exactly_three"] and g["n_distinct_ids"] == 3


def test_reconnection_rate_before_after():
    # fish present before and after an event -> reconnected
    df = pd.concat([_assign(range(0, 100), [0, 1, 2]),
                    _assign(range(120, 220), [0, 1, 2])], ignore_index=True)
    events = [{"pair": "ID_1|ID_2", "start": 100, "end": 119}]
    out = metrics.reconnection_rate(df, events, 0, 219, look=10)
    assert out["reconnection_rate"] == 1.0
    # drop fish 0 after the event -> it cannot reconnect
    df2 = df[~((df.fish_id == 0) & (df.frame >= 120))]
    out2 = metrics.reconnection_rate(df2, events, 0, 219, look=10)
    assert out2["reconnection_rate"] < 1.0


def test_align_labels_recovers_permutation():
    a = _assign(range(0, 50), [0, 1, 2])
    b = a.copy()
    # relabel b: 0->2, 1->0, 2->1 ; positions identical so Hungarian should invert it
    remap = {0: 2, 1: 0, 2: 1}
    b["fish_id"] = b["fish_id"].map(remap)
    out = metrics.align_labels(a, b, 0, 49)
    assert out["label_agreement"] == 1.0


# --------------------------- method adapters --------------------------- #

def test_canonical_slot():
    assert methods.canonical_slot("ID_1") == 0
    assert methods.canonical_slot("ID_3") == 2
    assert methods.canonical_slot(2) == 2


def test_geometry_table_commits_all_slots():
    d3 = pd.DataFrame({"frame": np.repeat(np.arange(10), 3),
                       "fish_id": ["ID_1", "ID_2", "ID_3"] * 10,
                       "X": 1.0, "Y": 2.0, "Z": 3.0, "bind_score": 0.5})
    d3["slot"] = d3["fish_id"].map(methods.canonical_slot)
    t = methods.geometry_3d_table(d3, 0, 9)
    assert set(t["fish_id"]) == {0, 1, 2}
    assert (t["flag"] == "").all()


def test_method_swap_calls():
    resolved = pd.DataFrame([
        {"pair": "ID_1|ID_2", "start": 100, "end": 110, "reason": "suspect_swap"},
        {"pair": "ID_1|ID_3", "start": 200, "end": 210, "reason": "velocity_consistent"},
        {"pair": "ID_2|ID_3", "start": 300, "end": 310, "reason": "ambiguous"},
    ])
    calls = methods.method_swap_calls("motion_resolver", resolved, 0, 1000)
    assert calls[105] is True       # suspect_swap -> swap
    assert calls[205] is False      # velocity_consistent -> no swap
    assert 305 not in calls         # ambiguous -> abstain
    geo = methods.method_swap_calls("geometry_3d", resolved, 0, 1000)
    assert all(v is False for v in geo.values())  # geometry never calls a swap


def test_gt_swap_accuracy():
    gt = pd.DataFrame({"center_frame": [105, 205], "swap": ["Yes", "No"]})
    calls = {105: True, 205: False}
    out = metrics.gt_swap_accuracy(calls, gt)
    assert out["accuracy"] == 1.0 and out["n_scored"] == 2
