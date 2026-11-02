"""medaka-3d's full chain: step layout, provenance status, and the export to review-package inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.reconstruction_3d import medaka3d_chain as chain


def _fake_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(chain, "medaka3d_repo", lambda path=None: tmp_path)
    monkeypatch.setattr(chain, "m3d_paths", lambda m3d: SimpleNamespace(work_dir=tmp_path / "work"))


def test_chain_steps_follow_the_production_chain(monkeypatch, tmp_path):
    _fake_repo(monkeypatch, tmp_path)
    steps = chain.chain_steps("exp", lambdas=[0.767])
    assert [s[0] for s in steps] == ["c27", "c28", "c33", "c30", "c34"]
    by_tool = {tool: (argv, out) for tool, argv, out in steps}
    work = tmp_path / "work"
    assert "--lambdas" not in by_tool["c27"][0]                  # c27 has no lambda grid
    assert by_tool["c28"][0][-2:] == ["--lambdas", "0.767"]
    # Each step reads the directory the previous one writes.
    assert by_tool["c33"][0][by_tool["c33"][0].index("--arm-dir") + 1] == str(by_tool["c28"][1])
    assert by_tool["c30"][0][by_tool["c30"][0].index("--arm-dir") + 1] == str(by_tool["c33"][1])
    assert by_tool["c34"][0][by_tool["c34"][0].index("--arm-dir") + 1] == str(by_tool["c30"][1])
    assert by_tool["c34"][1] == work / "c34_microflip" / f"exp__{chain.CHAIN_TAG}"
    assert chain.chain_arm("exp", 0.767) == by_tool["c34"][1] / "C34-label__exp__lam0.767.npz"


def test_step_status_reads_the_provenance_sidecar(tmp_path):
    assert chain.step_status("c28", tmp_path) == "missing"
    (tmp_path / "c28.provenance.json").write_text(json.dumps({"status": "failed"}))
    assert chain.step_status("c28", tmp_path) == "failed"
    (tmp_path / "c28.provenance.json").write_text(json.dumps({"status": "ok"}))
    assert chain.step_status("c28", tmp_path) == "ok"


def test_tool_command_routes_calibration_tools_through_run(monkeypatch, tmp_path):
    _fake_repo(monkeypatch, tmp_path)
    cmd, env, cwd = chain.tool_command("c28", ["--experiment", "exp"])
    assert cmd[1:5] == ["-m", "scripts.reconstruction_3d.medaka3d_chain", "run", "c28"]
    assert cmd[-3:] == ["--", "--experiment", "exp"] and cwd == tmp_path
    cmd, _, _ = chain.tool_command("c27", ["--experiment", "exp"])
    assert cmd[1] == str(tmp_path / "tools" / chain.TOOLS["c27"])  # plain script: c27 uses a process pool
    assert str(tmp_path / "src") in env["PYTHONPATH"]


def test_export_names_slots_by_column_and_translates_detector_rows(monkeypatch, tmp_path):
    _fake_repo(monkeypatch, tmp_path)
    nan = np.nan
    track = np.array([[0, 1, 2], [0, 1, -1], [-1, -1, -1]], np.int32)
    xyz = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4], [nan, nan, nan]],
                    [[nan] * 3] * 3], np.float32)
    source = np.full((3, 3, 3), -1, np.int8)
    source[0, :, 0] = [2, 0, 1]          # cam0 detector rows of slots 0..2 at frame 0
    source[0, :, 1] = [0, 1, 2]
    source[0, :, 2] = [1, 2, -1]         # slot 2 seen by two views
    source[1, :2, :] = [[0, 0, 0], [1, 1, 1]]
    arm = tmp_path / "arm.npz"
    np.savez(arm, slot_track=track, xyz=xyz, source=source, n_views_used=np.array([2, 2, 0], np.int8))
    monkeypatch.setattr(chain, "chain_arm", lambda *a, **k: arm)
    # Detector row -> load_2d_slp slot. cam0 frame 0: row 2 was dropped as a false positive.
    rows = np.full((3, 8), -1, np.int8)
    rows[:2, :3] = [[1, 0, -1], [0, 1, 2]]
    rows_by_view = {"top_cam": rows, "side1_cam": np.tile(np.arange(8, dtype=np.int8), (3, 1)),
                    "side2_cam": np.tile(np.arange(8, dtype=np.int8), (3, 1))}
    monkeypatch.setattr(chain, "detector_row_slots", lambda slp, n: rows_by_view[slp.stem])

    out = chain.export_review_inputs("exp", tmp_path, tmp_path / "out", "tag")

    best = json.loads(out["best_json"].read_text())
    assert best["id_labels"] == ["ID_1", "ID_2", "ID_3"]
    f0 = best["frames"]["0"]
    # Slot 0 used cam0 row 2 (no load_2d_slp slot), slot 1 row 0 -> slot 1, slot 2 row 1 -> slot 0.
    assert f0["cam0_to_id"] == ["ID_3", "ID_2", None]
    assert f0["cam1_to_id"] == ["ID_1", "ID_2", "ID_3"]
    assert f0["cam2_to_id"] == [None, "ID_1", "ID_2"]
    assert best["frames"]["1"]["cam0_to_id"][:2] == ["ID_1", "ID_2"]
    assert out["provenance"]["detections_without_slot"]["top_cam"] == 1
    assert out["provenance"]["fish_frames_present"] == 5

    csv = pd.read_csv(out["csv_3d"])
    assert len(csv) == 9 and list(csv.columns) == ["frame", "fish_id", "X", "Y", "Z", "is_interpolated"]
    row = lambda f, fid: csv[(csv.frame == f) & (csv.fish_id == fid)].iloc[0]   # noqa: E731
    assert row(0, "ID_3").X == 2 and row(1, "ID_2").Y == 4
    assert np.isnan(row(1, "ID_3").X) and csv[csv.frame == 2].X.isna().all()
    assert not csv.is_interpolated.any()                       # nothing observed after the gap: left empty
    prov = json.loads((tmp_path / "out" / "tag_reconstruction_provenance.json").read_text())
    assert prov["triangulation"]["provenance"]["model_folder"] == str(chain.CALIBRATION_DIR)


def _track_csv(path, x):
    n = len(x)
    pd.DataFrame({"frame": np.arange(n), "fish_id": "ID_1", "X": x, "Y": x, "Z": x,
                  "is_interpolated": False}).to_csv(path, index=False)


def test_short_gaps_are_filled_and_flagged_long_ones_and_the_ends_stay_empty(tmp_path):
    nan = np.nan
    x = np.array([nan, 0, nan, nan, 3, nan, nan, nan, nan, nan, nan, 10, nan], float)
    path = tmp_path / "t_3d.csv"
    _track_csv(path, x)
    assert chain.fill_track_csv(path, max_gap_frames=4) == 2
    csv = pd.read_csv(path)
    assert csv.X.tolist()[1:5] == [0, 1, 2, 3]
    assert csv.is_interpolated.tolist() == [i in (2, 3) for i in range(13)]
    assert csv.X[5:11].isna().all() and np.isnan(csv.X[0]) and np.isnan(csv.X[12])
    assert chain.fill_track_csv(path, max_gap_frames=4) == 2     # filling again changes nothing
    assert pd.read_csv(path).equals(csv)


def test_lambda_switch_table_scores_each_arm_against_the_review(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("cv2")
    from test_review_recon_compare import FIRST, N_SOURCE, _nodes, _reviewed

    from scripts.review.recon_compare import ReconState, build

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    state = ReconState(folder_a, out)
    truth = _nodes(N_SOURCE - FIRST)[:, :, 1]               # pectoral, the position node
    arms = {}
    for lam, swap_at in ((0.5, None), (0.767, 100)):
        xyz = np.full((N_SOURCE, 3, 3), np.nan, np.float32)
        xyz[FIRST:] = truth
        if swap_at is not None:
            xyz[swap_at:, [0, 1]] = xyz[swap_at:, [1, 0]]
        arms[lam] = tmp_path / f"arm{lam}.npz"
        np.savez(arms[lam], xyz=xyz)
    monkeypatch.setattr(chain, "chain_arm", lambda exp, lam, **k: arms.get(lam, tmp_path / "none.npz"))

    table = chain.lambda_switch_table("exp", state, lambdas=[0.5, 0.767, 2.0]).set_index("lambda_mm")
    assert list(table.index) == [0.5, 0.767]                  # lambda 2 has no arm on disk
    assert table.loc[0.5, "switches"] == 0 and table.loc[0.5, "coverage %"] == 100
    assert table.loc[0.767, "switches"] == 1 and table.loc[0.767, "with F"] == 1 and table.loc[0.767, "with M2"] == 0


def test_run_passes_only_the_tool_arguments_on(monkeypatch, tmp_path):
    _fake_repo(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(chain, "run_tool", lambda tool, argv, m3d=None: seen.update(tool=tool, argv=argv, m3d=m3d))
    cmd, _, _ = chain.tool_command("c28", ["--experiment", "exp", "--lambdas", "0", "0.767"])
    chain.main(cmd[3:])                                        # exactly what run_chain launches
    assert seen["tool"] == "c28" and seen["m3d"] == tmp_path
    assert seen["argv"] == ["--experiment", "exp", "--lambdas", "0", "0.767"]   # no --m3d leaking into c28
