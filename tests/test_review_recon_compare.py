"""Reconstruction-comparison mode: version file, per-frame matching, disagreement list, naming, verdicts."""

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from test_review_gui import _package  # noqa: E402  (synthetic review package)

N_SOURCE, FIRST = 120, 12
SWAP_AT, OFFSET, GAP = 70, (40, 45), (90, 110)


def _nodes(n):
    """Three fish 40 mm apart, drifting in y; nodes 5 mm apart along x."""
    nodes = np.zeros((n, 3, 3, 3), np.float32)
    nodes[..., 0] = np.array([0.0, 40.0, 80.0])[None, :, None] + 5.0 * np.arange(3)[None, None, :]
    nodes[..., 1] = np.linspace(0, 30, n, dtype=np.float32)[:, None, None]
    nodes[..., 2] = 50.0
    return nodes


def _set_nodes(folder, nodes, fish_ids=("ID_1", "ID_2", "ID_3")):
    with np.load(folder / "review_data.npz") as z:
        d = {key: z[key] for key in z.files}
    d["nodes3d_filled"] = d["nodes3d"] = nodes
    d["node_source"] = np.where(np.isfinite(nodes).all(-1), 0, -1).astype(np.int8)
    d["fish_ids"] = np.array(fish_ids)
    np.savez_compressed(folder / "review_data.npz", **d)


def _two_reconstructions(tmp_path):
    """A = the reference; B = the same fish in another track order, with one identity swap, one
    position disagreement and one gap."""
    import shutil

    folder_a = _package(tmp_path, n_source=N_SOURCE, first=FIRST)
    n = N_SOURCE - FIRST
    truth = _nodes(n)
    _set_nodes(folder_a, truth)
    folder_b = folder_a.parent / "tag_medaka3d"
    folder_b.mkdir()
    for name in ("review_data.npz", "review_manifest.json", "machine_risk.csv"):
        shutil.copy(folder_a / name, folder_b / name)
    order = [2, 0, 1]                                   # B track j holds true fish order[j]
    b = truth[:, order].copy()
    s = SWAP_AT - FIRST
    b[s:, [0, 1]] = b[s:, [1, 0]]                       # B swaps its tracks 0 and 1 (fish 2 and 0)
    b[OFFSET[0] - FIRST:OFFSET[1] - FIRST + 1, 2] += np.array([8.0, 0, 0], np.float32)   # fish 1, 8 mm off
    b[GAP[0] - FIRST:GAP[1] - FIRST + 1, 0] = np.nan   # B loses one fish for a while
    _set_nodes(folder_b, b)
    return folder_a, folder_b


def _built(tmp_path):
    from scripts.review.recon_compare import build

    folder_a, folder_b = _two_reconstructions(tmp_path)
    out = build([folder_a, folder_b], ["machine", "machine"], names=["old", "new"])
    return folder_a, folder_b, out


def test_build_writes_both_versions_on_a_common_frame_axis(tmp_path):
    from scripts.review.recon_compare import RECON_FILE

    folder_a, folder_b, out = _built(tmp_path)
    assert out == folder_a / f"recon_vs_{folder_b.name}"
    z = np.load(out / RECON_FILE)
    assert list(z["versions"]) == ["old", "new"] and z["nodes"].shape == (2, N_SOURCE - FIRST, 3, 3, 3)
    assert int(z["frames"][0]) == FIRST and list(z["kinds"]) == ["machine", "machine"]


def test_alignment_shows_geometry_only_and_naming_shows_the_swap(tmp_path):
    from scripts.review.compare_gui import ReconCompareData

    folder_a, _, out = _built(tmp_path)
    cd = ReconCompareData(folder_a, out, calibration=False)
    assert cd.labels == ["ID_1", "ID_2", "ID_3"]          # nobody named yet: the reference's tracks
    t_before, t_after = 30 - FIRST, 80 - FIRST
    ref = cd.series("old", 0, len(cd.frames), aligned=False)
    al = cd.series("new", 0, len(cd.frames), aligned=True)
    named = cd.series("new", 0, len(cd.frames), aligned=False)
    assert np.allclose(al[t_after], ref[t_after])         # same identity: B's swap is invisible
    assert np.allclose(named[t_before], ref[t_before])    # B inherited the reference's tracks
    assert not np.allclose(named[t_after], ref[t_after])  # after B's swap the names disagree
    cd.close()


def test_changes_list_the_swap_the_offset_and_the_gap(tmp_path):
    from scripts.review.compare_gui import ReconCompareData

    folder_a, _, out = _built(tmp_path)
    cd = ReconCompareData(folder_a, out, calibration=False)
    ch = cd.changes
    swaps = ch[ch["kind"] == "swap"]
    assert sorted(swaps["fish"]) == ["ID_1", "ID_3"] and (abs(swaps["start"] - SWAP_AT) <= 1).all()
    pos = ch[(ch["kind"] == "position")]
    assert len(pos) == 1 and pos.iloc[0]["fish"] == "ID_2"
    assert pos.iloc[0]["start"] == OFFSET[0] and pos.iloc[0]["end"] == OFFSET[1]
    assert np.isclose(pos.iloc[0]["size_mm"], 8.0, atol=0.01)
    gap = ch[ch["kind"] == "only old"]
    assert len(gap) == 1 and gap.iloc[0]["start"] == GAP[0] and gap.iloc[0]["end"] == GAP[1]
    assert set(ch["group"]) == {"identity", "geometry", "coverage"}
    cd.close()


def test_naming_the_reference_names_the_other_version_and_persists(tmp_path):
    from scripts.review.compare_gui import ReconCompareData

    folder_a, _, out = _built(tmp_path)
    cd = ReconCompareData(folder_a, out, calibration=False)
    msg = cd.set_name(0, "F", 20)
    assert "F" in msg and cd.labels[0] == "F"
    cd.set_name(1, "M1", 20)
    cd.set_name(2, "F", 20)                     # F moves; the old F track gets ID_3's old name (none)
    assert cd.labels == ["ID_1", "M1", "F"]
    cd.close()
    again = ReconCompareData(folder_a, out, calibration=False)
    assert again.labels == ["ID_1", "M1", "F"]
    t = 30 - FIRST
    assert np.allclose(again.series("new", t, t + 1, False), again.series("old", t, t + 1, False))
    again.close()


def test_reviewed_names_are_taken_from_the_ledger(tmp_path):
    from scripts.review.compare_gui import ReconCompareData

    folder_a, _, out = _built(tmp_path)
    rows = [("analysis_start", "*", "15", "1"), ("label", "ID_1", "15", "M2"), ("label", "ID_2", "15", "F"),
            ("label", "ID_3", "15", "M1")]
    pd.DataFrame([dict(event_id=f"e{i}", group=f"g{i}", kind=k, fish=f, frame=fr, frame_end="", frame_marked=fr,
                       value=v, risk_id="", session="s", created="", note="") for i, (k, f, fr, v) in enumerate(rows)]
                 ).to_csv(folder_a / "human_events.csv", index=False)
    cd = ReconCompareData(folder_a, out, calibration=False)
    assert cd.labels == ["M2", "F", "M1"] and cd.analysis_start == 15
    cd.close()


def test_verdicts_are_saved_and_counted(tmp_path):
    from scripts.review.compare_gui import ReconCompareData
    from scripts.review.recon_compare import VERDICTS_FILE

    folder_a, _, out = _built(tmp_path)
    cd = ReconCompareData(folder_a, out, calibration=False)
    row = cd.changes.index[cd.changes["kind"] == "position"][0]
    cd.set_verdict(row, "b_right")
    assert (out / VERDICTS_FILE).exists()
    cd.close()
    again = ReconCompareData(folder_a, out, calibration=False)
    assert again.changes.loc[again.changes["kind"] == "position", "verdict"].tolist() == ["b_right"]
    assert again.verdict_counts()["b_right"] == 1
    again.set_verdict(row, "")                            # cleared
    assert again.verdict_counts().get("b_right", 0) == 0
    again.close()


def test_mode_discovery_lists_review_and_reconstruction(tmp_path):
    from scripts.review.compare_gui import available_modes

    folder_a, _, out = _built(tmp_path)
    modes = available_modes(folder_a)
    assert [m[0] for m in modes] == ["recon"] and modes[0][1] == out
    (folder_a / "versions.npz").write_bytes(b"")          # only its presence matters here
    assert [m[0] for m in available_modes(folder_a)] == ["review", "recon"]


def test_recon_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.compare_gui import run_app

    folder_a, _, out = _built(tmp_path)
    assert run_app(folder_a, mode="recon", recon_dir=out, smoke_test=True) == 0


OLD_SWAP_AT = 50


def _reviewed(tmp_path):
    """A has a finished review (``versions.npz``, final = the truth, labels F/M1/M2 = fish 0/1/2); its
    machine tracks swap fish 0 and 1 at OLD_SWAP_AT. B is as in ``_two_reconstructions``."""
    from scripts.review.versions import VERSIONS

    folder_a, folder_b = _two_reconstructions(tmp_path)
    n = N_SOURCE - FIRST
    truth = _nodes(n)
    old = truth.copy()
    s = OLD_SWAP_AT - FIRST
    old[s:, [0, 1]] = old[s:, [1, 0]]
    _set_nodes(folder_a, old)
    nodes = np.full((len(VERSIONS), n, 3, 3, 3), np.nan, np.float32)
    nodes[VERSIONS.index("final")] = nodes[VERSIONS.index("manual")] = truth
    np.savez_compressed(
        folder_a / "versions.npz", nodes=nodes, source=np.zeros(nodes.shape[:4], np.int8),
        track=np.zeros(nodes.shape[:3], np.int8), versions=np.array(VERSIONS),
        present=np.array(["machine", "manual", "final"]), labels=np.array(["F", "M1", "M2"]),
        frames=np.arange(FIRST, N_SOURCE), node_names=np.array(["mouth", "pectoral", "tail"]),
        edges=np.array([[0, 1], [1, 2]]), head_node=0, tail_node=2, position_node=1, analysis_start=15,
        fish_ids=np.array(["ID_1", "ID_2", "ID_3"]))
    return folder_a, folder_b


def test_reviewed_reference_is_the_answer_key(tmp_path):
    from scripts.review.compare_gui import ReconCompareData
    from scripts.review.recon_compare import build, error_table

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    assert out == folder_a / "recon_reviewed_old_new"
    cd = ReconCompareData(folder_a, out, calibration=False)
    assert cd.labels == ["F", "M1", "M2"] and cd.analysis_start == 15
    assert "fixed" in cd.set_name(0, "M2", 20) and cd.labels == ["F", "M1", "M2"]   # names cannot change
    ch = cd.changes
    old_swaps = ch[(ch["kind"] == "swap") & (ch["other"] == "old")]
    new_swaps = ch[(ch["kind"] == "swap") & (ch["other"] == "new")]
    assert sorted(old_swaps["fish"]) == ["F", "M1"] and (abs(old_swaps["start"] - OLD_SWAP_AT) <= 1).all()
    assert sorted(new_swaps["fish"]) == ["F", "M2"] and (abs(new_swaps["start"] - SWAP_AT) <= 1).all()
    assert [p[1:] for p in cd.presets] == [("reviewed", "old"), ("reviewed", "new"), ("old", "new")]
    table = error_table(cd.state).set_index("kind")
    n_after_old = (N_SOURCE - OLD_SWAP_AT) * 2          # two fish wrong from the swap to the end
    assert table.loc["swap", "old fish-frames"] == n_after_old
    assert table.loc["swap", "both"] == (N_SOURCE - SWAP_AT) * 1           # F is wrong in both after 70
    cd.close()


def test_switch_and_geometry_tables_against_the_answer_key(tmp_path):
    from scripts.review.recon_compare import ReconState, build, geometry_table, switch_events, switch_table

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    state = ReconState(folder_a, out)
    old, new = switch_events(state, "old"), switch_events(state, "new")
    assert old["fish"].tolist() == ["F+M1"] and abs(int(old["frame"][0]) - OLD_SWAP_AT) <= 1
    assert new["fish"].tolist() == ["F+M2"] and abs(int(new["frame"][0]) - SWAP_AT) <= 1   # the gap is no switch
    assert not old["correct_after"].any()
    table = switch_table(state, near_s=0.05).set_index("version")
    assert table.loc["old", "switches"] == 1 and table.loc["old", "into wrong"] == 1
    assert table.loc["old", "with F"] == 1 and table.loc["old", "with M2"] == 0
    assert table.loc["new", "only this (±0.05 s)"] == 1                  # 20 frames from old's switch
    assert switch_table(state).loc[0, "only this (±1 s)"] == 0           # within a second: a shared encounter
    geo = geometry_table(state).set_index("version")
    assert geo.loc["reviewed", "coverage %"] == 100 and geo.loc["new", "coverage %"] < 100
    assert geo.loc["old", "steps > 30 mm"] == 2 and geo.loc["new", "steps > 30 mm"] == 2   # the two swapped tracks
    assert geo.loc["reviewed", "steps > 10 mm"] == 0 and geo.loc["new", "steps > 10 mm"] == 2   # 8 mm offset: not a step


def test_reviewed_recon_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.compare_gui import run_app
    from scripts.review.recon_compare import build

    folder_a, folder_b = _reviewed(tmp_path)
    out = build([folder_a, folder_a, folder_b], ["final", "machine", "machine"], names=["reviewed", "old", "new"])
    assert run_app(folder_a, mode="recon", recon_dir=out, smoke_test=True) == 0


def test_overlapping_parts_are_drawn_once_in_the_agree_colour():
    from scripts.review.compare_gui import AGREE_COLOR, coincide, draw_pair

    a = np.array([[[20.0, 20.0], [40.0, 20.0], [60.0, 20.0]]])          # one fish, 3 nodes

    def has_green(qa, qb):
        img = np.zeros((80, 100, 3), np.uint8)
        draw_pair(img, ["F"], [(0, 1), (1, 2)], 0, 1, qa, qb, 1.0, names=False)
        return np.all(img == np.array(AGREE_COLOR, np.uint8), axis=-1).any(), img

    same, _ = has_green(a, a.copy())
    apart, _ = has_green(a, a + np.array([0.0, 30.0]))
    only_a, _ = has_green(a, np.full_like(a, np.nan))
    assert same and not apart and not only_a
    assert coincide(a, a + np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 9.0]]])).tolist() == [[True, True, False]]


def test_fill_gaps_interpolates_inner_gaps_of_one_version(tmp_path):
    from scripts.review.recon_compare import NODE_INTERPOLATED, RECON_FILE, fill_gaps

    _, _, out = _built(tmp_path)
    before = np.load(out / RECON_FILE)["nodes"].copy()
    gap = slice(GAP[0] - FIRST, GAP[1] - FIRST + 1)
    assert np.isnan(before[1, gap, 0]).all()                 # "new" track 0 has the gap
    filled = fill_gaps(out, "new", log=lambda *_: None)
    z = np.load(out / RECON_FILE)
    after, source = z["nodes"], z["source"]
    assert filled["track 0"] == 3 * (GAP[1] - GAP[0] + 1) and filled["track 1"] == 0
    assert np.isfinite(after[1, gap, 0]).all() and (source[1, gap, 0] == NODE_INTERPOLATED).all()
    lo, hi = before[1, GAP[0] - FIRST - 1, 0], before[1, GAP[1] - FIRST + 1, 0]
    mid = after[1, (GAP[0] + GAP[1]) // 2 - FIRST, 0]
    assert np.all((np.minimum(lo, hi) - 1e-4 <= mid) & (mid <= np.maximum(lo, hi) + 1e-4))   # between the ends
    assert np.array_equal(after[0], before[0], equal_nan=True)  # the other version is untouched
    assert (out / "recon_versions.before_fill.npz").exists()
