import json

import numpy as np
import pytest

from scripts.review import id_figure as F
from scripts.review import id_segments as S

pytest.importorskip("matplotlib")


def test_machine_codes_and_shares():
    dmap = {"danger": np.array([0, 1, 1, 0], bool), "high_frames": np.array([0, 0, 1, 0], bool)}
    codes = F.machine_codes(dmap)
    assert codes.tolist() == [1, 2, 3, 1]
    assert [round(s[2]) for s in F.shares(codes, F.MACHINE)] == [50, 25, 25]


def test_shares_leave_out_frames_before_the_analysis_start():
    codes = np.array([S.NOT_ANALYSED, S.NOT_ANALYSED, F.H_WATCHED, F.H_UNUSABLE], np.int8)
    assert {label: s for label, _c, s in F.shares(codes, F.HUMAN)}["Marked unusable (X)"] == 50.0


def test_human_codes_use_only_the_reviewers_marks():
    codes = F.human_codes(first=10, n=100, start=20, watched=[(10, 59)], changes=[40], manual=[(50, 54)],
                          pose=[(30, 34, ["ID_1"])], fps=10.0)
    assert (codes[:10] == S.NOT_ANALYSED).all()                        # before the analysis start
    assert codes[15] == F.H_WATCHED and codes[22] == F.H_POSE and codes[42] == F.H_UNUSABLE
    assert (codes[25:35] == F.H_CHANGE).all()                          # 0.5 s = 5 frames either side of frame 40
    assert codes[60] == F.H_UNWATCHED and codes[95] == F.H_UNWATCHED


def test_agreement_counts_changes_and_marks_against_the_machine_danger():
    n = 2000
    danger = np.zeros(n, bool)
    high = np.zeros(n, bool)
    danger[100:200] = danger[500:520] = danger[800:830] = True
    high[150:160] = True
    runs = np.array([[100, 200], [500, 520], [800, 830]])
    dmap = {"danger": danger, "high_frames": high, "runs": runs, "high": np.array([True, False, False])}
    # a change in danger, one pressed 1.8 s after an encounter, one nowhere near; an X over the 800 run
    a = F.agreement(dmap, first=0, fps=100.0, start=None, changes=[120, 700, 1300], manual=[(790, 805)])
    assert a["id_changes"] == 3 and a["changes_in_danger_strict"] == 1
    assert a["changes_in_danger_lenient"] == 2                          # 700 is within 3 s of the 500-520 run
    assert a["changes_in_high_strict"] == 1 and a["changes_in_high_lenient"] == 1      # only 120 is near 150-160
    assert a["unusable_touching_danger"] == 1
    assert (a["danger_runs"], a["danger_runs_with_mark"], a["high_runs"], a["high_runs_with_mark"]) == (3, 2, 1, 1)
    # the analysis start leaves out what comes before it
    later = F.agreement(dmap, first=0, fps=100.0, start=600, changes=[120, 700, 1300], manual=[(790, 805)])
    assert later["id_changes"] == 2 and later["danger_runs"] == 1


def _package(folder, n=4000):
    folder.mkdir(parents=True)
    xyz = np.zeros((n, 3, 3), np.float32)
    xyz[:, 1, 0], xyz[:, 2, 0] = 40, 200
    xyz[1000:1100, 1, 0] = 12
    (folder / "review_manifest.json").write_text(json.dumps({"experiment": "exp", "fps": 100.0}))
    np.savez(folder / "review_data.npz", frames=np.arange(n), fish_ids=np.array(["ID_1", "ID_2", "ID_3"]), xyz=xyz,
             view_present=np.ones((n, 3, 3), bool), is_interpolated=np.zeros((n, 3), bool))


def test_unreviewed_folder_gets_the_machine_figure_only(tmp_path):
    folder = tmp_path / "exp" / "tag"
    _package(folder)
    out = F.build(folder)
    assert out == [folder / F.MACHINE_FIGURE] and out[0].stat().st_size > 1000
    assert not (folder / F.HUMAN_FIGURE).exists()


def test_identity_pass_wins_over_the_package_ledger_and_gives_two_figures(tmp_path):
    folder = tmp_path / "exp" / "tag"
    _package(folder)
    (folder / "coverage.csv").write_text("session,frame_start,frame_end,speed,mode,wall_start,wall_end\ns,0,3999,1,play,a,b\n")
    ledger = folder / S.LEDGER_FOLDER
    ledger.mkdir()
    (ledger / "coverage.csv").write_text("session,frame_start,frame_end,speed,mode,wall_start,wall_end\ns,0,2000,2,play,a,b\n")
    assert F.build(folder) == [ledger / F.HUMAN_FIGURE, ledger / F.MACHINE_FIGURE]
    agree = json.loads((ledger / F.AGREEMENT_FILE).read_text())
    assert agree["id_changes"] == 0 and agree["danger_runs"] == 1
