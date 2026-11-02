"""Audit: SLEAP screening, blind old-vs-new items, answers, unblinded statistics, GUI smoke test."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from test_review_recon_compare import _reviewed  # noqa: E402


def _audit_package(tmp_path):
    from scripts.review.recon_compare import build
    from scripts.review.review_data import DET_NONE, DET_SCALE

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    d = dict(np.load(folder_a / "review_data.npz"))
    frames = d["frames"]
    n = len(frames)
    fish = np.array([[40, 40], [50, 45], [60, 50]], float)
    pts = np.stack([fish + 30 * s for s in range(4)])                 # 4 instances, 3 used per frame
    det = np.full((n, 3, 4, 3, 2), DET_NONE, np.int16)
    det[:, :, :3] = np.round(pts[:3] * DET_SCALE)
    det[40, 1, 2] = DET_NONE                                          # side 1: two instances
    det[50, 0, 3] = np.round(pts[3] * DET_SCALE)                      # top: four
    det[60, 2, 0, 2] = DET_NONE                                       # side 2: tail not found
    det[70, 0, 1, 2] = np.round(np.array([200, 200]) * DET_SCALE)     # top: far too long
    np.savez_compressed(folder_a / "slp_raw.npz", frames=frames, det=det, scale=DET_SCALE, none=DET_NONE,
                        node_names=np.array(["mouth", "pectoral", "tail"]), edges=np.array([[0, 1], [1, 2]]))
    d["auto_flipped"] = np.zeros((n, 3, 3), bool)
    d["auto_flipped"][80, 2, 1] = True                                # fish 2 head-tail against the others in side 1
    d["view_left_out"] = np.full((n, 3, 3), -1, np.int8)
    d["view_left_out"][90, 0, 1] = 2                                  # side 2 left out for fish 0's pectoral
    d["kp2d"] = np.broadcast_to(pts[:3][None, :, None], (n, 3, 3, 3, 2)).astype(np.float32).copy()
    np.savez_compressed(folder_a / "review_data.npz", **d)
    return folder_a, out


def test_audit_screens_sleap_and_lists_blind_old_new_items(tmp_path):
    from scripts.review.audit import Audit, AuditParams, build

    folder_a, out = _audit_package(tmp_path)
    p = AuditParams(id_persist_frames=5, geometry_pad_frames=2, id_pad_frames=2, sleap_pad_frames=2, n_clean=5)
    tasks = build(folder_a, out, params=p, log=lambda *_: None)
    s = tasks[tasks["task"] == "sleap"].set_index("category")
    assert s.loc["missing", "view"] == 1 and s.loc["extra", "view"] == 0 and s.loc["incomplete", "view"] == 2
    assert s.loc["too_long", "instance"] == 1 and s.loc["head_tail", "instance"] == 2
    assert s.loc["view_outlier", "view"] == 2 and s.loc["view_outlier", "instance"] == 0
    assert len(tasks[tasks["task"] == "clean"]) == 5
    rec = tasks[tasks["task"].isin(["id", "geometry"])]
    assert len(rec[rec["task"] == "id"]) >= 1 and len(rec[rec["task"] == "geometry"]) >= 1
    assert set(map(tuple, rec[["a_is", "b_is"]].to_numpy())) <= {("old", "new"), ("new", "old")}
    assert tasks["task"].iloc[0] == "id" and list(tasks["item"]) == list(range(1, len(tasks) + 1))

    au = Audit(out)
    first_id = rec[rec["task"] == "id"].iloc[0]
    it = int(first_id["item"])
    assert au.parts(it) == ("A", "B")
    au.answer(it, "1", "A keeps them", part="A")                       # each version on its own
    au.answer(it, "2", part="B")
    with pytest.raises(ValueError):
        au.answer(it, "1")                                              # an old-vs-new item needs a part
    ids = rec[rec["task"] == "id"]
    if len(ids) > 1:                                                    # "wrong briefly" ranks between
        it2 = int(ids.iloc[1]["item"])
        au.answer(it2, "3", part="A")
        au.answer(it2, "2", part="B")
        assert Audit(out).unblinded("id").set_index("item").loc[it2, "better"] == ids.iloc[1]["a_is"]
    sl = int(tasks[tasks["task"] == "sleap"].iloc[0]["item"])
    au.answer(sl, "2")                                                  # SLEAP wrong
    with pytest.raises(ValueError):
        au.answer(sl, "5")
    again = Audit(out)
    assert again.label(it, "A") == "right" and again.answers[(it, "A")][1] == "A keeps them"
    assert again.done(it) and again.summary_text(it) == "A: right, B: wrong"
    st = again.stats()
    row = st["id"].set_index("item").loc[it]
    assert row["better"] == first_id["a_is"] and row[first_id["b_is"]] == "wrong"
    assert st["sleap"]["real"].sum() == 1
    again.answer(sl, "", "")
    assert (sl, "") not in Audit(out).answers
    with pytest.raises(FileExistsError):
        build(folder_a, out, params=p, log=lambda *_: None)


def _with_close_encounter(folder_a):
    """Fish 1 on top of fish 0 in rows 60-69, detected only in the top view there: in side 1 SLEAP has a
    free instance where it is (B1), in side 2 only two instances (A)."""
    from scripts.review.review_data import DET_NONE

    d = dict(np.load(folder_a / "review_data.npz"))
    n = len(d["frames"])
    d["xyz"][60:70, 1] = d["xyz"][60:70, 0] + [2, 0, 0]
    d["view_present"] = np.ones((n, 3, 3), bool)
    d["view_present"][60:70, 1, 1:] = False
    d["proj2d"] = np.broadcast_to(d["kp2d"].mean(axis=3), (n, 3, 3, 2)).astype(np.float32).copy()
    np.savez_compressed(folder_a / "review_data.npz", **d)
    r = dict(np.load(folder_a / "slp_raw.npz"))
    r["det"][60:70, 2, 1] = DET_NONE
    np.savez_compressed(folder_a / "slp_raw.npz", **r)


def test_lost_views_are_sorted_appended_and_counted(tmp_path):
    from scripts.review.audit import Audit, AuditParams, add_lost, build

    folder_a, out = _audit_package(tmp_path)
    _with_close_encounter(folder_a)
    p = AuditParams(id_persist_frames=5, n_clean=3, lost_per_group=3, lost_min_gap_frames=1)
    tasks = build(folder_a, out, params=p, log=lambda *_: None)
    first = int(tasks["item"].iloc[0])
    Audit(out).answer(first, "1", part="A")
    new = add_lost(folder_a, out, params=p, log=lambda *_: None)
    assert sorted(new["category"]) == ["A"] * 3 + ["B1"] * 3
    assert set(new.loc[new["category"] == "B1", "view"]) == {1} and set(new.loc[new["category"] == "A", "view"]) == {2}
    assert list(new["item"]) == list(range(len(tasks) + 1, len(tasks) + 7))
    assert (new["fish"] == new["fish"].iloc[0]).all() and new["x"].notna().all()
    au = Audit(out)
    assert au.code(first, "A") == "1" and len(au.tasks) == len(tasks) + 6     # earlier answers kept
    assert au.summary["lost"]["population"] == {"A": 10, "B1": 10, "B2": 0, "C": 0}
    b1 = new.loc[new["category"] == "B1", "item"].astype(int)
    for i in b1:
        au.answer(int(i), "1")
    a_items = new.loc[new["category"] == "A", "item"].astype(int).tolist()
    au.answer(a_items[0], "2")
    au.answer(a_items[1], "3")
    st = Audit(out).stats()["lost"].set_index("group")
    assert st.loc["B1", "detected"] == 1.0 and st.loc["A", "checked"] == 2
    assert st.loc["all", "detected"] == pytest.approx(0.5) and st.loc["all", "not detected"] == pytest.approx(0.25)
    with pytest.raises(FileExistsError):
        add_lost(folder_a, out, params=p, log=lambda *_: None)
    assert len(list((out / "audit").glob("audit_tasks.before_lost_*.csv"))) == 1


def test_audit_gui_shows_lost_items(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.audit import AuditParams, add_lost, build
    from scripts.review.audit_gui import run_app

    folder_a, out = _audit_package(tmp_path)
    _with_close_encounter(folder_a)
    p = AuditParams(id_persist_frames=5, n_clean=3, lost_per_group=2, lost_min_gap_frames=1)
    build(folder_a, out, params=p, log=lambda *_: None)
    add_lost(folder_a, out, params=p, log=lambda *_: None)

    def check(win, app):
        item = int(win_tasks(win).query("task == 'lost'")["item"].iloc[0])
        win.open_item(item)
        app.processEvents()
        assert "expected fish position" in win.header.text() and "group" not in win.header.text()
        win.press("2")
        app.processEvents()

    def win_tasks(win):
        import scripts.review.audit as audit
        return audit.Audit(out).tasks

    assert run_app(folder_a, out, smoke_test=True, check=check) == 0


def test_wilson_interval():
    from scripts.review.audit import wilson

    lo, hi = wilson(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04
    lo, hi = wilson(15, 30)
    assert 0.3 < lo < 0.35 and 0.65 < hi < 0.7


def test_audit_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.audit import AuditParams, build
    from scripts.review.audit_gui import run_app

    folder_a, out = _audit_package(tmp_path)
    build(folder_a, out, params=AuditParams(id_persist_frames=5, n_clean=3), log=lambda *_: None)
    assert run_app(folder_a, out, smoke_test=True) == 0


def test_audit_gui_undo_goes_back_to_the_last_answer(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.audit import Audit, AuditParams, build
    from scripts.review.audit_gui import run_app

    folder_a, out = _audit_package(tmp_path)
    build(folder_a, out, params=AuditParams(id_persist_frames=5, n_clean=3), log=lambda *_: None)
    seen = {}

    def check(win, app):
        item = int(win.row["item"])
        win.press("1")                              # A answered -> B is open
        assert win.part == "B" and Audit(out).code(item, "A") == "1"
        win.press("Backspace")                      # undo: back to A, its answer cleared
        assert win.part == "A" and int(win.row["item"]) == item and Audit(out).code(item, "A") == ""
        seen["ok"] = True

    assert run_app(folder_a, out, smoke_test=True, check=check) == 0
    assert seen["ok"]
