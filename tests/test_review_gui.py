"""Offscreen smoke test of the review GUI on a tiny synthetic package."""

import json
import os

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
# PyQt5 is imported inside the test, not at collection: loading Qt before torch
# makes torch's c10.dll fail to initialise on Windows for later-collected tests.

VIEWS = ("top_cam", "side1_cam", "side2_cam")


def _write_video(path, n, size=64):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (size, size))
    for i in range(n):
        img = np.full((size, size, 3), i % 255, np.uint8)
        writer.write(img)
    writer.release()


def _package(root, n_source=120, step=6, first=12, xyz=None):
    proxy = root / "exp" / "proxy"
    proxy.mkdir(parents=True)
    views = {}
    for v in VIEWS:
        _write_video(proxy / f"{v}.full.mp4", n_source)
        _write_video(proxy / f"{v}.play.mp4", -(-n_source // step))
        views[v] = {"source_fps": 166.0, "source_frames": n_source, "source_width": 128, "source_height": 128,
                    "proxy_width": 64, "proxy_height": 64, "play_step": step, "play_frames": -(-n_source // step),
                    "full_file": f"{v}.full.mp4", "play_file": f"{v}.play.mp4"}
    (proxy / "proxy_manifest.json").write_text(json.dumps({"views": views}))
    folder = root / "exp" / "tag"
    folder.mkdir()
    frames = np.arange(first, n_source)
    n, k = len(frames), 3
    kp = np.zeros((n, k, 3, 3, 2), np.float32)
    for i in range(k):
        kp[:, i, :, :, 0] = 20 + 40 * i + np.arange(3)[None, None, :] * 5
        kp[:, i, :, :, 1] = 60
    kp[5:8, 2] = np.nan  # a few frames where fish 3 is only known in 3D
    if xyz is None:
        xyz = np.zeros((n, k, 3), np.float32)
        xyz[:, :, 0] = [0, 10, 40]
    segments = np.zeros((n, k), np.int32)
    segments[n // 2:] = 1                 # a head/tail jump half way
    segments[n // 2 - 5:n // 2, 0] = -1   # track 0 has no head/tail just before it
    interp = np.zeros((n, k), bool)
    interp[5:8, 2] = True
    np.savez_compressed(
        folder / "review_data.npz", frames=frames, fish_ids=np.array(["ID_1", "ID_2", "ID_3"]),
        node_names=np.array(["mouth", "pectoral", "tail"]), edges=np.array([[0, 1], [1, 2]]), head_node=0,
        tail_node=2, kp2d=kp, proj2d=np.full((n, k, 3, 2), 50, np.float32), xyz=xyz,
        orientation_segment=segments, is_interpolated=interp,
        nodes3d_filled=np.zeros((n, k, 3, 3), np.float32) + np.arange(3, dtype=np.float32)[None, None, :, None],
        node_source=np.zeros((n, k, 3), np.int8), missing_node_proj2d=np.full((n, k, 3, 3, 2), 40, np.float32),
    )
    pd.DataFrame({"risk_id": ["R00001"], "frame_start": [30], "frame_end": [60], "frame_peak": [40],
                  "types": ["close"], "fish": ["ID_1|ID_2"], "score": [3.0], "details": ["test"]}
                 ).to_csv(folder / "machine_risk.csv", index=False)
    (folder / "review_manifest.json").write_text(json.dumps({"experiment": "exp", "round_tag": "tag"}))
    return folder


def test_review_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.ledger import Ledger
    from scripts.review.review_gui import run_app

    folder = _package(tmp_path)
    ledger_dir = tmp_path / "ledger"
    from test_review_behaviors import write_boris

    boris = write_boris(tmp_path / "exp.csv", [("m1tf", "follow", "courtship", "START", 20),
                                               ("m1tf", "follow", "courtship", "STOP", 70),
                                               ("m2tm1", "attack", "aggressive", "START", 40),
                                               ("m2tm1", "attack", "aggressive", "STOP", 45)])
    assert run_app(folder, ledger_dir=ledger_dir, backup_dir=tmp_path / "backup", smoke_test=True,
                   boris=boris) == 0
    led = Ledger(ledger_dir)
    # Naming and the segment flip survive; the swap was undone; the last edit was removed (and logged).
    assert sorted(led.events[led.events["kind"] == "label"]["value"]) == ["F", "M1", "M2"]
    flips = led.events[led.events["kind"] == "flip"]
    assert len(flips) == 1 and (flips.iloc[0]["frame"], flips.iloc[0]["frame_end"]) == ("12", "60")
    assert not (led.events["kind"] == "swap").any()
    removed = pd.read_csv(ledger_dir / "human_events_removed.csv", dtype=str)
    assert len(removed) == 1
    # An interval, keyframes at both of its ends and two completed per-view edits were written (the view
    # swap - T, 1/2/3, T - was the last edit, so it is the removed one).
    kinds = list(led.events["kind"]) + list(removed["kind"])
    assert kinds.count("interval") == 1 and kinds.count("confirm") == 2 and kinds.count("view_flip") == 1
    assert list(removed["kind"]) == ["view_swap"] and removed.iloc[0]["fish"].count("|") == 1
    assert (led.events["kind"] == "analysis_start").sum() == 1
    assert len(led.coverage) >= 1
    assert (tmp_path / "backup" / "human_events.csv").exists()
    assert not (ledger_dir / "review.lock").exists()
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_two_swaps_after_playback_compose_to_what_the_reviewer_chose(tmp_path, monkeypatch):
    """The tracker rotated all three fish; the reviewer fixes it with two keys after playing past it.

    Each swap snaps back to the closest approach of its pair, but never before an earlier swap
    touching the same tracks, so the identities at the marked frame are the chosen ones.
    """
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import scripts.review.review_gui as g

    first, n_source = 12, 120
    frames = np.arange(first, n_source)
    xyz = np.zeros((len(frames), 3, 3), np.float32)
    xyz[:, :, 0] = [0, 30, 60]
    xyz[frames == 112, 1, 0] = 1      # tracks 0 and 1 meet at 112
    xyz[frames == 108, 2, 0] = 31     # tracks 1 and 2 meet at 108 (earlier!)
    folder = _package(tmp_path, n_source=n_source, first=first, xyz=xyz)
    result = {}

    def scenario(app, win):
        win.seek(first)
        for k, label in enumerate(("F", "M1", "M2")):
            win.selected = k
            win.set_label(label)
        win.seek(115)
        win.last_nav = "play"             # as if the key came right after playing past the swap
        win.selected = 0
        win.set_label("M1")
        win.selected = 1
        win.set_label("M2")
        t = win.pkg.t(115)
        result["labels"] = [win.labels[win.pkg.fish_ids[win.fish_of_track(k, t)]] for k in range(3)]
        result["swaps"] = [(f, a, b) for f, a, b in win.swaps]
        win.close()
        return 0

    monkeypatch.setattr(g, "_smoke", scenario)
    assert g.run_app(folder, ledger_dir=tmp_path / "ledger", backup_dir=None, smoke_test=True) == 0
    assert result["labels"] == ["M1", "M2", "F"]
    assert result["swaps"][0] == (112, 0, 1) and result["swaps"][1][0] > 112


def test_error_log_keeps_running_and_records_the_traceback(tmp_path):
    import sys

    from scripts.review.review_gui import install_error_log

    old = sys.excepthook
    try:
        install_error_log(tmp_path / "errors.log")
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        text = (tmp_path / "errors.log").read_text(encoding="utf-8")
        assert "ValueError: boom" in text and "Traceback" in text
    finally:
        sys.excepthook = old
        import faulthandler

        faulthandler.disable()
        install_error_log.fault_file.close()


def test_identity_changes_counts_only_clear_group_swaps():
    from scripts.review.review_gui import identity_changes

    n = 6
    before = np.zeros((n, 3, 3))
    before[:, 1, 0] = 8.0                       # fish 1 is 8 mm from fish 0
    before[:, 2, 0] = 100.0
    after = before.copy()
    after[2, [0, 1]] = after[2, [1, 0]]         # fish 0 and 1 exchanged on frame 2
    after[4, 0, 1] += 3.5                       # fish 0 moved 3.5 mm, no exchange
    assert list(np.flatnonzero(identity_changes(before, after, [0, 1]))) == [2]
