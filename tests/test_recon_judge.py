"""Judging old vs new reconstructions: event list, verdicts, GUI smoke test."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from test_review_recon_compare import FIRST, GAP, N_SOURCE, OFFSET, OLD_SWAP_AT, SWAP_AT, _reviewed  # noqa: E402


def _state(tmp_path, **params):
    from scripts.review.recon_compare import build
    from scripts.review.recon_judge import JudgeParams, JudgeState

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    return folder_a, out, JudgeState(folder_a, out, params=JudgeParams(**params))


def test_events_hold_both_switches_and_the_body_disagreements(tmp_path):
    _, out, js = _state(tmp_path, switch_merge_s=0.05)        # 8 frames: old's and new's switches stay apart
    ev = js.events
    ids = ev[ev["kind"] == "identity"].reset_index(drop=True)
    assert ids["fish"].tolist() == ["F+M1", "F+M2"]
    assert ids["detail"].str.startswith("old").tolist() == [True, False]
    pad = round(0.5 * js.fps)
    assert ids["start"].tolist() == [FIRST, FIRST]                # the pad is clipped at the first frame
    assert abs(ids.loc[0, "end"] - min(N_SOURCE - 1, OLD_SWAP_AT + pad)) <= 1
    body = ev[ev["kind"] == "body"].set_index("fish")
    assert body.loc["M1", "start"] == OFFSET[0] and body.loc["M1", "end"] == OFFSET[1]
    assert np.isclose(body.loc["M1", "old_new_mm"], 8.0, atol=0.01) and body.loc["M1", "old_ref_mm"] == 0
    assert body.loc["F", "start"] == GAP[0] and "only old has 3D" in body.loc["F", "detail"]
    assert (out / "recon_judge_events.csv").exists()


def test_switches_close_together_are_one_event(tmp_path):
    _, _, js = _state(tmp_path)                               # 1 s: the two switches are 20 frames apart
    ids = js.events[js.events["kind"] == "identity"]
    assert len(ids) == 1 and ids.iloc[0]["fish"] == "F+M1+M2"


def test_grades_are_per_version_fish_and_frames_anywhere_and_paint_over_each_other(tmp_path):
    from scripts.review.recon_judge import JudgeState

    folder_a, out, js = _state(tmp_path)
    # Not tied to a hint: frames 30-60, with no event open.
    js.paint(["F", "M1", "M2"], 30, 60, "id", "old", "ok")
    js.paint(["F", "M1", "M2"], 30, 60, "id", "new", "ok")
    js.paint(["F"], 30, 60, "body", "old", "major")              # F: both versions wrong ...
    js.paint(["F"], 30, 60, "body", "new", "major")
    js.paint(["F"], 45, 60, "body", "new", "minor")              # ... new only a little from 45
    js.paint(["M1"], 30, 60, "body", "old", "minor")             # M1: old a little wrong, new right
    js.paint(["M1"], 30, 60, "body", "new", "ok")
    js.set_note(40, 50, "F hidden\nbehind M1")
    with pytest.raises(ValueError):
        js.paint(["F"], 30, 60, "id", "old", "b_right")
    with pytest.raises(ValueError):
        js.paint(["X"], 30, 60, "id", "old", "ok")
    now = js.at(50)
    assert now[("body", "old")][:2] == ["major", "minor"] and now[("body", "new")][:2] == ["minor", "ok"]
    assert js.compare("ok", "minor") == "old better, a little" and js.compare("major", "ok") == "new better, clearly"
    assert js.compare("minor", "minor") == "same" and js.compare("ok", "unsure") == "unsure" and js.compare("", "ok") == ""
    again = JudgeState(folder_a, out)                            # everything is read back from the file
    assert again.at(44)[("body", "new")][0] == "major" and again.at(45)[("body", "new")][0] == "minor"
    assert again.notes(0, 100) == [(40, 50, "F hidden behind M1")]
    s = again.summary(30, 60).set_index(["aspect", "fish"])
    assert s.loc[("id", "F"), "same"] == 31 and s.loc[("body", "F"), "new better, a little"] == 16
    assert s.loc[("body", "F"), "same"] == 15 and s.loc[("body", "M1"), "new better, a little"] == 31
    assert s.loc[("body", "M2"), "frames graded"] == 0
    assert again.done(30, 60, ["F", "M1"], "body") == 1 and again.done(30, 60, ["M2"], "body") == 0
    again.paint(["F"], 30, 60, "body", "old", "")                # erase
    assert again.at(50)[("body", "old")][0] == "" and set(again.segments["judged_on"]) == {"3D"}


def test_hints_count_as_judged_when_their_fish_and_frames_are_graded(tmp_path):
    _, _, js = _state(tmp_path)
    body = js.events[js.events["kind"] == "body"].iloc[0]
    fish = js.event_fish(body["event"])
    for v in ("old", "new"):
        js.paint(fish, int(body["start"]), int(body["end"]), "body", v, "ok")
    t = js.table().set_index("event")
    assert t.loc[body["event"], "body %"] == 100 and t.loc[body["event"], "judged"]
    assert t.loc[body["event"], "id %"] == 0


def test_earlier_formats_are_moved_aside(tmp_path):
    from scripts.review.recon_judge import JUDGE_FILE, JudgeState

    folder_a, out, js = _state(tmp_path)
    (out / JUDGE_FILE).write_text("event,fish,aspect,start,end,verdict,note,judged_on,created\n")
    again = JudgeState(folder_a, out)
    assert len(again.segments) == 0 and (out / "recon_judgements.earlier_format_1.csv").exists()


def test_judge_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.recon_judge_gui import run_app

    folder_a, out, _ = _state(tmp_path)
    assert run_app(folder_a, out, smoke_test=True) == 0


def _write_slp(path, frame_points: dict):
    import sleap_io as sio

    sk = sio.Skeleton(["mouth", "pectoral", "tail"], edges=[("mouth", "pectoral"), ("pectoral", "tail")])
    video = sio.Video(filename=str(path.with_suffix(".mp4")))
    lfs = [sio.LabeledFrame(video=video, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.asarray(pts, float), skeleton=sk, score=0.9) for pts in insts])
        for f, insts in frame_points.items()]
    sio.save_slp(sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[sk]), str(path))


def test_raw_sleap_instances_are_all_kept_and_drawn_in_the_camera_views(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    pytest.importorskip("sleap_io")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.recon_judge_gui import RAW_COLOR, draw_raw, run_app
    from scripts.review.slp_raw import RawSlp, build

    folder_a, out, _ = _state(tmp_path)
    with np.load(folder_a / "review_data.npz") as z:
        frames = z["frames"]
    f0 = int(frames[0])
    fish = [[40, 40], [50, 45], [60, 50]]
    tiny = [[10, 10], [11, 10], [np.nan, np.nan]]                     # 1 px long, tail not found
    slp_dir = tmp_path / "slp"
    slp_dir.mkdir()
    for view in ("top_cam", "side1_cam", "side2_cam"):
        _write_slp(slp_dir / f"{view}.slp", {f0: [fish, fish], f0 + 1: [fish] * 5 + [tiny]})   # a duplicate, a tiny one
    build(folder_a, slp_dir, log=lambda *_: None)
    raw = RawSlp(folder_a)
    assert raw.node_names == ["mouth", "pectoral", "tail"] and raw.edges == [(0, 1), (1, 2)]
    p0, p1 = raw.points(f0), raw.points(f0 + 1)
    assert p0.shape == (3, 6, 3, 2) and np.isfinite(p0[:, :2]).all() and np.isnan(p0[:, 2:]).all()     # two = two
    assert np.isfinite(p1[:, :5]).all() and np.allclose(p1[0, 5, :2], [[10, 10], [11, 10]])            # six = six
    assert np.isnan(p1[0, 5, 2]).all() and raw.points(int(frames[-1]) + 1) is None
    img = np.zeros((100, 100, 3), np.uint8)
    assert draw_raw(img, p1[0], raw.node_names, raw.edges) == 6
    assert (img == RAW_COLOR).all(axis=-1).any()
    assert run_app(folder_a, out, smoke_test=True) == 0


def test_short_grades_stay_visible_when_the_timeline_is_zoomed_out():
    pytest.importorskip("PyQt5")
    from scripts.review.recon_judge_gui import sample_runs

    codes = np.zeros(10000, np.int8)
    codes[5003] = 2                                            # one graded frame among 10000
    runs = sample_runs(codes, 500)
    assert len(runs) == 1 and runs[0][2] == 2 and abs(runs[0][0] - 0.5) < 0.01
    assert sample_runs(np.array([0, 1, 1, 3], np.int8), 100) == [(0.25, 0.75, 1), (0.75, 1.0, 3)]
