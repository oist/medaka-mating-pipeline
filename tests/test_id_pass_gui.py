"""Offscreen run of the identity pass GUI on a tiny synthetic package."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.review import id_segments as S

pytest.importorskip("cv2")
PARAMS = S.SegmentParams(safe_mm=15, high_risk_mm=8, min_safe_s=0.02, slow_pad_s=0.0)


def _xyz(first=12, n_source=120):
    from_frame = np.arange(first, n_source)
    xyz = np.zeros((len(from_frame), 3, 3), np.float32)
    xyz[:, :, 0] = [0, 30, 60]
    xyz[(from_frame >= 60) & (from_frame < 70), 1, 0] = 6      # tracks 0 and 1 close: a high-risk danger run
    xyz[:, 1, 0] = np.convolve(np.pad(xyz[:, 1, 0], 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return xyz


def test_status_strip_shows_the_worst_of_a_pixel_column():
    from scripts.review.id_pass_gui import STATUS_COLORS, ZONE_COLORS, status_strip

    status = np.full(100, S.WATCHED_SAFE, np.int8)
    status[50] = S.UNUSABLE
    zone = np.zeros(100, np.int8)
    zone[60] = 2
    strip = status_strip(status, zone, 0, 99, 10)
    assert tuple(strip[0, 5]) == STATUS_COLORS[S.UNUSABLE] and tuple(strip[0, 4]) == STATUS_COLORS[S.WATCHED_SAFE]
    assert tuple(strip[1, 6]) == ZONE_COLORS[2] and tuple(strip[1, 5]) == ZONE_COLORS[0]


def test_left_right_move_one_frame_a_press_and_more_when_held():
    from scripts.review.id_pass_gui import HOLD_FRAMES, nudge_frames

    assert nudge_frames(0.4) == 1 and nudge_frames(0.1) == 1     # single presses, however quick
    assert nudge_frames(0.03) == HOLD_FRAMES == 10               # the keyboard's repeat


def test_a_node_a_view_did_not_detect_is_taken_from_its_projection():
    from scripts.review.id_pass_gui import skeleton_points

    nan = np.nan
    pts = np.array([[1, 2], [nan, nan], [5, 6]], float)
    projected = np.array([[nan, nan], [3, 4], [nan, nan]], float)
    shown, seen = skeleton_points(pts, projected)
    assert shown.tolist() == [[1, 2], [3, 4], [5, 6]] and seen.tolist() == [True, False, True]
    shown, seen = skeleton_points(pts, None)
    assert np.isnan(shown[1]).all() and seen.tolist() == [True, False, True]


def test_a_range_still_open_on_closing_is_kept(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from test_review_gui import _package

    import scripts.review.id_pass_gui as g

    folder = _package(tmp_path, xyz=_xyz())

    def scenario(app, win):
        win.seek(30)
        win.toggle_danger()
        win.toggle_pose()
        win.seek(36)
        win.render()
        win.overview.repaint()
        win.zoom.repaint()
        assert "MARKING UNUSABLE from 30" in win.info.text() and "MARKING POSE from 30" in win.info.text()
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=scenario) == 0
    events = pd.read_csv(folder / "id_pass" / "human_events.csv")
    marks = events[events["kind"] == "danger"]
    assert marks[["frame", "frame_end"]].values.tolist() == [[30, 36]]
    assert len(events[(events["kind"] == "pose_bad") & (events["frame_end"] == 36)]) == 3


def test_delete_removes_an_earlier_sessions_range_and_undo_brings_it_back(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from test_review_gui import _package

    import scripts.review.id_pass_gui as g
    from scripts.review.ledger import REMOVED_FILE

    folder = _package(tmp_path, xyz=_xyz())

    def first_session(app, win):
        for a, b in ((30, 36), (32, 40)):
            win.seek(a)
            win.toggle_danger()
            win.seek(b)
            win.toggle_danger()
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=first_session) == 0
    out = {}

    def second_session(app, win):
        win.seek(34)
        win.undo()                                    # nothing of this session to undo
        out["kept"] = len(win.unusable)
        win.toggle_danger()                           # X pressed once by mistake
        win.undo()
        out["open"] = win.open_danger
        win.delete_range()                            # the latest range holding frame 34
        out["after_delete"] = list(win.unusable)
        out["logged"] = pd.read_csv(folder / "id_pass" / REMOVED_FILE)["frame"].tolist()
        win.undo()
        out["after_undo"] = sorted(win.unusable)
        out["log_after_undo"] = len(pd.read_csv(folder / "id_pass" / REMOVED_FILE))
        win.delete_range()
        win.delete_range()
        win.seek(20)
        win.delete_range()                            # no range here: nothing happens
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=second_session) == 0
    assert out["kept"] == 2 and out["open"] is None
    assert out["after_delete"] == [(30, 36)] and out["logged"] == [32]
    assert out["after_undo"] == [(30, 36), (32, 40)] and out["log_after_undo"] == 0
    events = pd.read_csv(folder / "id_pass" / "human_events.csv")
    assert (events["kind"] == "danger").sum() == 0
    assert sorted(pd.read_csv(folder / "id_pass" / REMOVED_FILE)["frame"]) == [30, 32]


def test_w_jumps_to_the_next_unwatched_stretch_and_wraps(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from test_review_gui import _package

    import scripts.review.id_pass_gui as g

    folder = _package(tmp_path, xyz=_xyz())
    out = {}

    def scenario(app, win):
        win.seek(18)
        win.mark_start()
        win.ledger.add_coverage(30, 50, 1.0, "play", 0, 0)       # frames 30-50 watched
        win.rebuild()
        win.seek(20)
        win.goto_unwatched(+1)
        out["next"] = win.frame                                  # the stretch 18-29 starts before 20: on to 51
        win.seek(25)
        win.goto_unwatched(+1)
        out["after_gap"] = win.frame
        win.goto_unwatched(+1)
        out["wrapped"] = win.frame                               # nothing after 51: back to the first stretch
        win.goto_unwatched(-1)
        out["back"] = win.frame                                  # nothing before 18: wraps to the last
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=scenario) == 0
    assert out["next"] == 51 and out["after_gap"] == 51
    assert out["wrapped"] == 18 and out["back"] == 51


def test_opening_another_experiment_saves_and_closes_this_one(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from test_review_gui import _package

    import scripts.review.id_pass_gui as g

    folder = _package(tmp_path, xyz=_xyz())
    other = tmp_path / "other" / "tag"
    out = {}

    def scenario(app, win):
        win.seek(30)
        win.toggle_danger()
        win.seek(36)
        win.open_experiment(other)                     # as after picking it with O
        out["next"] = win.next_folder
        out["closed"] = win._closed
        out["lock"] = (folder / "id_pass" / "review.lock").exists()
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=scenario) == 0
    assert out["next"] == other and out["closed"] and not out["lock"]
    assert (folder / "id_pass" / "id_pass_summary.json").is_file()
    events = pd.read_csv(folder / "id_pass" / "human_events.csv")
    assert events[events["kind"] == "danger"][["frame", "frame_end"]].values.tolist() == [[30, 36]]


def test_one_pass_names_changes_marks_and_writes_the_result(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from test_review_gui import _package

    import scripts.review.id_pass_gui as g

    folder = _package(tmp_path, xyz=_xyz())
    out = {}

    def scenario(app, win):
        assert len(win.map["runs"]) >= 1 and win.map["high"].any()
        win.seek(18)
        win.mark_start()
        for key, name in zip((0, 1, 2), ("F", "M1", "M2")):
            win.select(name, key)
            win.set_name(name)
        assert win.labels == {"ID_1": "F", "ID_2": "M1", "ID_3": "M2"} and not win.swaps
        win.toggle_play()
        win.frame, win.last_nav = 84, "play"          # as if played past the close encounter
        win.flush()
        win.pause()
        win.select("F", 0)                            # the fish shown as F is really M1
        win.set_name("M1")
        out["swap_frame"] = win.swaps[0][0]
        t = win.pkg.t(win.frame)
        out["names"] = (win.name_of(0, t), win.name_of(1, t))
        out["before"] = (win.name_of(0, win.pkg.t(30)), win.name_of(1, win.pkg.t(30)))
        out["status_in_run"] = int(win.status[win.pkg.t(out["swap_frame"])])
        win.seek(30)
        win.toggle_danger()
        win.seek(36)
        win.toggle_danger()
        out["marked"] = int(win.status[win.pkg.t(33)])
        win.seek(40)
        win.pick(None)
        win.select("M2", 2)
        assert win.selected == 2
        win.select("M2", 2)                           # the same fish again clears the selection
        assert win.selected is None
        win.select("M2", 2)
        win.toggle_pose()
        win.overview.repaint()                        # the range being marked is drawn while it is open
        win.seek(44)
        win.toggle_pose()
        win.overview.repaint()
        win.zoom.repaint()
        out["pose_range"] = list(win.pose)
        pose = win.ledger.events[win.ledger.events["kind"] == "pose_bad"].iloc[0]
        out["pose"] = (pose["fish"], pose["frame"], pose["frame_end"], int(win.status[win.pkg.t(42)]))
        win.undo()
        win.render()
        win.overview.repaint()
        win.zoom.repaint()
        win.view3d.repaint()
        win.undo()
        out["unmarked"] = int(win.status[win.pkg.t(33)])
        return 0

    assert g.run_app(folder, params=PARAMS, scenario=scenario) == 0
    assert 60 <= out["swap_frame"] < 75
    assert out["before"] == ("F", "M1") and out["names"] == ("M1", "F")
    assert out["status_in_run"] == S.UNUSABLE and out["marked"] == S.UNUSABLE
    assert out["unmarked"] == S.WATCHED_SAFE
    assert out["pose"] == ("ID_3", "40", "44", S.WATCHED_SAFE)      # a wrong pose leaves the frame usable
    assert out["pose_range"] == [(40, 44, ["ID_3"])]
    folder_out = folder / "id_pass"
    summary = json.loads((folder_out / "id_pass_summary.json").read_text(encoding="utf-8"))
    assert summary["identity_changes"] == 1 and summary["analysis_start"] == 18 and summary["unusable_marks"] == 0
    seg = pd.read_csv(folder_out / "id_segments.csv")
    assert len(seg) == 2 and "F=ID_1" in seg["names"].iloc[0] and "M1=ID_1" in seg["names"].iloc[1]
    assert not (folder_out / "review.lock").exists()
