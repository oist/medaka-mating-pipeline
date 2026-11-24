import numpy as np
import pandas as pd

from scripts.review.ledger import (
    EVENT_COLUMNS,
    Ledger,
    analysis_start,
    apply_events,
    assign_label,
    compare_with_machine,
    covered_fraction,
    flip_series,
    merge_intervals,
    owner_at,
    owner_series,
    snap_frame,
    state_at,
    state_series,
    static_labels,
    swap_list,
    watched_intervals,
)

FISH = ["ID_1", "ID_2", "ID_3"]


def _events(rows):
    """Rows of (kind, fish, frame, value) or (kind, fish, frame, value, frame_end)."""
    out = []
    for i, row in enumerate(rows):
        kind, fish, frame, value = row[:4]
        out.append({c: "" for c in EVENT_COLUMNS} | {
            "event_id": str(i), "group": str(i), "kind": kind, "fish": fish, "frame": str(frame),
            "frame_end": str(row[4]) if len(row) > 4 else "", "frame_marked": str(frame), "value": value})
    return pd.DataFrame(out, columns=EVENT_COLUMNS)


def _data(n_frames=6, first=0):
    frames = np.arange(first, first + n_frames)
    nodes = np.zeros((n_frames, 3, 3, 3), np.float32)
    nodes[:, :, 0] = 1.0   # mouth
    nodes[:, :, 2] = -1.0  # tail
    xyz = np.zeros((n_frames, 3, 3))
    xyz[:, :, 0] = [0.0, 10.0, 20.0]   # track k sits at x = 10 k
    return {"frames": frames, "fish_ids": np.array(FISH), "node_names": np.array(["mouth", "pectoral", "tail"]),
            "head_node": 0, "tail_node": 2, "nodes3d": nodes, "xyz": xyz,
            "is_interpolated": np.zeros((n_frames, 3), bool)}


def test_assign_label_swaps_with_current_holder():
    current = {"ID_1": "F", "ID_2": "M1", "ID_3": "M2"}
    assert assign_label(current, "ID_2", "F") == {"ID_2": "F", "ID_1": "M1"}
    assert assign_label(current, "ID_1", "F") == {}
    # First identification: nobody holds the label yet.
    assert assign_label({"ID_1": "", "ID_2": ""}, "ID_2", "M2") == {"ID_2": "M2"}
    # "Not a fish" never displaces another track.
    assert assign_label(current, "ID_3", "-") == {"ID_3": "-"}


def test_flag_state_is_last_event_at_or_before_frame():
    ev = _events([("bad3d", "ID_1", 20, "1"), ("bad3d", "ID_1", 30, "0"), ("bad3d", "ID_1", 30, "1")])
    assert state_at(ev, "bad3d", "ID_1", 5) == "0"
    assert state_at(ev, "bad3d", "ID_1", 25) == "1"
    assert state_at(ev, "bad3d", "ID_1", 40) == "1"   # same frame: later row wins
    frames = np.arange(0, 50)
    series = state_series(ev, "bad3d", "ID_1", frames)
    assert [state_at(ev, "bad3d", "ID_1", int(f)) for f in frames] == list(series)


def test_static_labels_keep_one_name_per_fish():
    ev = _events([("label", "ID_1", 900, "F"), ("label", "ID_2", 900, "M1"), ("label", "ID_3", 50000, "F")])
    # An old-style ledger naming two fish F: the most recently written keeps the name.
    assert static_labels(ev, FISH) == {"ID_1": "", "ID_2": "M1", "ID_3": "F"}


def test_swaps_compose_in_frame_order_whatever_the_entry_order():
    # Entered out of order: the swap at 50 is written before the one at 20.
    ev = _events([("swap", "ID_2|ID_3", 50, "1"), ("swap", "ID_1|ID_2", 20, "1")])
    swaps = swap_list(ev, FISH)
    assert swaps == [(20, 0, 1), (50, 1, 2)]
    owner = owner_series(3, swaps, np.arange(0, 60))
    assert list(owner[10]) == [0, 1, 2]
    assert list(owner[30]) == [1, 0, 2]          # track 0 carries fish 1, track 1 carries fish 0
    assert list(owner[55]) == [1, 2, 0]
    assert list(owner_at(3, swaps, 55)) == [1, 2, 0]
    # Swapping the same pair twice restores the original assignment.
    twice = owner_series(3, swap_list(_events([("swap", "ID_1|ID_2", 5, "1"), ("swap", "ID_1|ID_2", 8, "1")]), FISH),
                         np.arange(10))
    assert list(twice[6]) == [1, 0, 2] and list(twice[9]) == [0, 1, 2]


def test_flip_ranges_cancel_where_they_overlap():
    ev = _events([("flip", "ID_1", 2, "1", 5), ("flip", "ID_1", 4, "1", 7), ("flip", "ID_2", 8, "1")])
    frames = np.arange(10)
    assert list(np.flatnonzero(flip_series(ev, "ID_1", frames))) == [2, 3, 6, 7]
    assert list(np.flatnonzero(flip_series(ev, "ID_2", frames))) == [8, 9]   # no end: to the last frame


def test_snap_frame_picks_closest_approach_inside_window():
    frames = np.arange(100, 120)
    d = np.full(20, 30.0)
    d[7] = 2.0   # frame 107
    d[15] = 1.0  # frame 115, outside (100, 110]
    assert snap_frame(frames, d, 100, 110) == 107
    assert snap_frame(frames, np.full(20, np.nan), 100, 110) == 110
    assert snap_frame(frames, d, 110, 110) == 110


def test_coverage_counts_every_speed():
    cov = pd.DataFrame({"frame_start": ["0", "50", "200"], "frame_end": ["60", "100", "300"],
                        "speed": ["1", "0.5", "2"]})
    assert watched_intervals(cov) == [(0, 100), (200, 300)]
    assert watched_intervals(cov, max_speed=1.0) == [(0, 100)]
    assert merge_intervals([(5, 9), (0, 4), (20, 30)]) == [(0, 9), (20, 30)]
    assert covered_fraction([(0, 99)], 0, 199) == 0.5


def test_ledger_persists_undo_remove_and_coverage(tmp_path):
    led = Ledger(tmp_path)
    g1 = led.add("label", {"ID_1": "F"}, 10, 10)
    g2 = led.add("flip", {"ID_1": "1"}, 20, 22, frame_end=40)
    led.add("swap", {"ID_1|ID_2": "1"}, 40, 42)
    led.add_coverage(0, 100, 1.0, "play", 0.0, 1.0)
    removed = led.undo()
    assert list(removed["kind"]) == ["swap"]
    led.close()
    again = Ledger(tmp_path)
    assert list(again.events["group"]) == [g1, g2]
    assert again.events.iloc[1]["frame_end"] == "40"
    assert watched_intervals(again.coverage) == [(0, 100)]
    assert len(again.sessions) == 2 and again.sessions.iloc[0]["ended"] != ""
    # Removing an event of an earlier session logs it.
    again.remove_group(g2)
    assert list(again.events["group"]) == [g1]
    log = pd.read_csv(tmp_path / "human_events_removed.csv", dtype=str, keep_default_na=False)
    assert list(log["group"]) == [g2] and "removed" in log.iloc[0]["note"]


def test_apply_events_follows_swaps_names_and_flips():
    data = _data()
    ev = _events([("label", "ID_1", 0, "F"), ("label", "ID_2", 0, "M1"), ("label", "ID_3", 0, "M2"),
                  ("swap", "ID_1|ID_2", 3, "1"), ("flip", "ID_2", 1, "1", 1), ("nodes_bad", "ID_1", 4, "1")])
    cov = pd.DataFrame({"frame_start": ["0"], "frame_end": ["1"], "speed": ["1"]})
    out = apply_events(data, ev, cov)
    f = out[out["fish_id"] == "ID_1"]
    assert set(f["label"]) == {"F"}
    # Fish ID_1 is on track ID_1 until the swap at 3, then on track ID_2.
    assert list(f["source_track"]) == ["ID_1"] * 3 + ["ID_2"] * 3
    assert list(f["X"]) == [0.0] * 3 + [10.0] * 3
    # Track flags stay with the track: nodes_bad on track ID_1 from 4 belongs to fish ID_2 then.
    assert not f["nodes_bad"].any()
    m1 = out[out["fish_id"] == "ID_2"]
    assert list(m1["nodes_bad"]) == [False] * 4 + [True] * 2
    # The flip of track ID_2 at frame 1 belongs to fish ID_2 (before the swap).
    assert list(m1["mouth_X"]) == [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    assert list(f["watched"]) == [True, True, False, False, False, False]


def test_compare_with_machine_counts_swaps_flips_and_flags_not_names():
    ev = _events([("label", "ID_1", 0, "F"), ("label", "ID_2", 0, "M1"), ("label", "ID_1", 400, "M1"),
                  ("swap", "ID_1|ID_2", 500, "1"), ("flip", "ID_2", 900, "1", 950)])
    risks = pd.DataFrame({"frame_start": [480, 2000], "frame_end": [520, 2100]})
    cov = pd.DataFrame({"frame_start": ["0"], "frame_end": ["999"], "speed": ["1"]})
    s = compare_with_machine(ev, risks, cov, 0, 1999)
    assert s["corrections"] == 2 and s["corrections_by_kind"] == {"swap": 1, "flip": 1}
    assert s["corrections_inside_risk"] == 1 and s["corrections_outside_risk"] == 1
    assert s["risk_windows_with_correction"] == 1
    assert s["watched_fraction"] == 0.5


def test_analysis_start_is_latest_mark_and_limits_apply_and_compare():
    ev = _events([("label", "ID_1", 0, "F"), ("analysis_start", "*", 300, "1"), ("analysis_start", "*", 200, "1"),
                  ("swap", "ID_1|ID_2", 100, "1"), ("swap", "ID_1|ID_2", 600, "1")])
    assert analysis_start(ev) == 200          # most recently written mark wins, not the latest frame
    assert analysis_start(ev.iloc[:1]) is None
    data = _data(2, first=150)
    data["frames"] = np.array([150, 250])
    assert list(apply_events(data, ev)["analysed"]) == [False] * 3 + [True] * 3
    risks = pd.DataFrame({"frame_start": [90, 590], "frame_end": [110, 610]})
    cov = pd.DataFrame({"frame_start": ["200"], "frame_end": ["599"], "speed": ["1"]})
    s = compare_with_machine(ev, risks, cov, 0, 999)
    # The swap at 100 and the risk window before the start are left out; the start mark is not a correction.
    assert s["corrections"] == 1 and s["corrections_by_kind"] == {"swap": 1}
    assert s["risk_windows"] == 1 and s["corrections_inside_risk"] == 1
    assert s["frames"] == 800 and s["watched_fraction"] == 0.5


def test_export_reviewed_writes_table_and_readiness(tmp_path):
    import json

    from scripts.review.export_reviewed import export

    np.savez_compressed(tmp_path / "review_data.npz", **_data(10))
    (tmp_path / "review_manifest.json").write_text(json.dumps({"experiment": "exp", "round_tag": "tag",
                                                               "data_version": 3}))
    pd.DataFrame({"risk_id": ["R1"], "frame_start": [4], "frame_end": [6]}).to_csv(tmp_path / "machine_risk.csv",
                                                                                  index=False)
    led = Ledger(tmp_path)
    led.add("analysis_start", {"*": "1"}, 2, 2)
    for fish, name in zip(FISH, ("F", "M1", "M2")):
        led.add("label", {fish: name}, 2, 2)
    led.add("swap", {"ID_1|ID_3": "1"}, 5, 5)
    led.add_coverage(2, 9, 1.0, "play", 0.0, 1.0)
    led.close()
    summary = export(tmp_path)
    assert summary["ready_for_analysis"] and summary["labels"] == {"ID_1": "F", "ID_2": "M1", "ID_3": "M2"}
    assert summary["comparison_with_machine"]["corrections"] == 1
    table = pd.read_csv(tmp_path / "reviewed_3d.csv")
    f = table[table["label"] == "F"]
    # X/Y/Z is the pectoral node; the 3D centroid (where the tracks differ here) is centroid_X/Y/Z.
    assert list(f["centroid_X"]) == [0.0] * 5 + [20.0] * 5
    assert list(table["analysed"][:6]) == [False] * 6


def test_apply_events_uses_completed_nodes_and_their_source():
    data = _data(4)
    data["nodes3d"][2, 0, 0] = np.nan                       # mouth of track ID_1 not triangulated at frame 2
    filled = data["nodes3d"].copy()
    filled[2, 0, 0] = [0.5, 0.0, 0.0]
    source = np.zeros((4, 3, 3), np.int8)
    source[2, 0, 0] = 1
    data["nodes3d_filled"], data["node_source"] = filled, source
    ev = _events([("flip", "ID_1", 2, "1", 2)])
    out = apply_events(data, ev)
    f = out[out["fish_id"] == "ID_1"].reset_index(drop=True)
    assert np.isfinite(f[["mouth_X", "tail_X"]].to_numpy()).all()
    # The human flip at frame 2 moves the completed mouth (and its source) to the tail.
    assert f.loc[2, "tail_X"] == 0.5 and f.loc[2, "tail_source"] == 1 and f.loc[2, "mouth_source"] == 0
    assert list(f["mouth_source"]) == [0, 0, 0, 0]


def test_review_lock_ignores_a_crashed_process_on_this_computer(tmp_path):
    import json
    import socket
    import subprocess
    import sys

    from scripts.review.ledger import ReviewLock, pid_alive

    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead = int(done.stdout.strip())
    assert not pid_alive(dead)
    lock = ReviewLock(tmp_path)
    lock.path.write_text(json.dumps({"host": socket.gethostname(), "pid": dead, "since": "x"}))
    assert lock.holder() is None                       # crashed session: no warning
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.path.write_text(json.dumps({"host": socket.gethostname(), "pid": alive.pid, "since": "x"}))
        assert lock.holder() is not None               # another live window on this computer
    finally:
        alive.kill()
    lock.path.write_text(json.dumps({"host": "other-pc", "pid": 1, "since": "x"}))
    assert lock.holder()["host"] == "other-pc"


def test_atomic_write_retries_while_target_is_held(tmp_path, monkeypatch):
    import os

    from scripts.review import ledger as led_mod

    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(led_mod.os, "replace", flaky)
    monkeypatch.setattr(led_mod, "SAVE_RETRY_SECONDS", 0.0)
    led_mod.atomic_write_csv(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")
    assert calls["n"] == 3 and (tmp_path / "x.csv").exists()
