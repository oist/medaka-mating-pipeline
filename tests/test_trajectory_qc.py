import json

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.trajectory_qc import (
    TYPES,
    QCParams,
    find_anomalies,
    flag_mask,
    merge_intervals,
    run_qc,
    score_against_review,
)

FISH = ["ID_1", "ID_2", "ID_3"]


def _tracks(n=2000, seed=0):
    """Three fish swimming smoothly far apart; nodes 12 mm ahead / behind the pectoral."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    xyz = np.stack([np.c_[40 + 20 * np.sin(t / 200 + k), 50 + 50 * k + 5 * np.cos(t / 150), 90 + 0 * t]
                    for k in range(3)], axis=1).astype(float)
    xyz += rng.normal(0, 0.05, xyz.shape)
    nodes = np.stack([xyz + [12, 0, 0], xyz, xyz - [12, 0, 0]], axis=2)
    source = np.zeros((n, 3, 3), np.int8)
    present = np.ones((n, 3, 3), bool)
    interp = np.zeros((n, 3), bool)
    resid = np.full((n, 3, 3), 0.3)
    return xyz, nodes, source, present, interp, resid


def _find(xyz, nodes, source, present, interp, resid, c30=None, **kw):
    frames = np.arange(len(xyz))
    return find_anomalies(frames, FISH, xyz, nodes, source, 1, present, interp, resid, QCParams(**kw), c30)


def test_clean_tracks_have_no_findings():
    assert _find(*_tracks()).empty


def test_each_check_finds_its_planted_fault():
    xyz, nodes, source, present, interp, resid = _tracks()
    xyz[500, 0] += 20                       # one-frame spike: jump, spike, local outlier
    nodes[800:806, 1, 2] = nodes[800:806, 1, 1] - [40, 0, 0]   # tail far off: body length
    xyz[1200:1205, 2, 2] = 230              # above the volume (186 + 10)
    interp[1300:1330, 1] = True             # 30 interpolated frames
    present[1400:1430, 2, 1:] = False       # one view only for 30 frames
    resid[1500:1510, 0] = 50.0
    xyz[1700:1720, 1] = xyz[1700:1720, 0] + [3, 0, 0]          # two fish 3 mm apart
    rows = _find(xyz, nodes, source, present, interp, resid)
    found = {(r.type, r.fish) for r in rows.itertuples()}
    for item in [("jump", "ID_1"), ("spike", "ID_1"), ("local_outlier", "ID_1"), ("body_length", "ID_2"),
                 ("out_of_volume", "ID_3"), ("gap", "ID_2"), ("few_views", "ID_3"), ("residual", "ID_1"),
                 ("close", "ID_1|ID_2")]:
        assert item in found, item
    close = rows[rows["type"] == "close"].iloc[0]
    assert (close.start, close.end, close.category) == (1700, 1719, "identity")
    assert rows.loc[rows["type"] == "body_length", "category"].eq("geometry").all()


def test_filled_nodes_are_not_judged_by_body_length():
    xyz, nodes, source, present, interp, resid = _tracks()
    nodes[800:806, 1, 2] = nodes[800:806, 1, 1] - [40, 0, 0]
    source[800:806, 1, 2] = 2              # body model, not measured
    assert "body_length" not in set(_find(xyz, nodes, source, present, interp, resid)["type"])


def test_c30_joins_and_zone_refusals_become_identity_findings():
    c30 = ([{"frame": 100, "evidence": "median", "permutation": [1, 0, 2], "margin_mm": 40.0}],
           [{"zone": [300, 340], "reason": "zone unvouched"}, {"frame": 9999, "reason": "window incomplete"}])
    rows = _find(*_tracks(), c30=c30)
    assert rows[["type", "start", "end"]].values.tolist() == [["c30_join", 100, 100], ["c30_refusal", 300, 340]]
    assert set(rows["category"]) == {"identity"}


def test_flags_and_intervals():
    frames = np.arange(100, 200)
    rows = pd.DataFrame([
        {"type": "close", "category": "identity", "fish": "ID_1|ID_3", "start": 110, "end": 112, "peak": 111,
         "score": 3.0, "details": ""},
        {"type": "spike", "category": "geometry", "fish": "ID_2", "start": 115, "end": 115, "peak": 115,
         "score": 5.0, "details": ""},
        {"type": "c30_join", "category": "identity", "fish": "", "start": 180, "end": 180, "peak": 180,
         "score": 1.0, "details": ""},
    ])
    flags = flag_mask(rows, frames, FISH)
    close, spike, join = (1 << TYPES.index(t) for t in ("close", "spike", "c30_join"))
    assert flags[11].tolist() == [close, 0, close]
    assert flags[15].tolist() == [0, spike, 0]
    assert flags[80].tolist() == [join] * 3
    iv = merge_intervals(rows, 3, 100, 199)
    assert iv[["frame_start", "frame_end", "categories", "fish"]].values.tolist() == [
        [107, 118, "geometry;identity", "ID_1;ID_2;ID_3"], [177, 183, "identity", "*"]]


def test_score_against_review_counts_covered_swaps_and_ranges():
    iv = pd.DataFrame({"frame_start": [100, 400], "frame_end": [120, 410],
                       "categories": ["identity", "geometry"]})
    events = pd.DataFrame({"kind": ["swap", "swap", "swap", "danger", "pose_bad", "pose_bad"],
                           "frame": [50, 125, 300, 405, 405, 900], "frame_end": [None, None, None, 406, 408, 905]})
    s = score_against_review(iv, events, start=80, last=999, fps=10.0, tolerance_s=0.5)
    assert (s["swaps"], s["swaps_covered"], s["swaps_missed"]) == (2, 1, [300])   # frame 50 is before start
    assert (s["danger_ranges"], s["danger_covered"]) == (1, 1)
    assert (s["pose_bad_ranges"], s["pose_bad_covered"]) == (2, 1)


def test_run_qc_writes_its_outputs(tmp_path):
    xyz, nodes, source, present, interp, resid = _tracks(600)
    xyz[300, 1] += 25
    np.savez_compressed(tmp_path / "review_data.npz", frames=np.arange(600), fish_ids=np.array(FISH),
                        node_names=np.array(["mouth", "pectoral", "tail"]), xyz=xyz.astype(np.float32),
                        nodes3d_filled=nodes.astype(np.float32), node_source=source, view_present=present,
                        is_interpolated=interp, node_resid=resid.astype(np.float32))
    (tmp_path / "review_manifest.json").write_text(json.dumps({"fps": 166.0}))
    summary = run_qc(tmp_path, log=lambda *_: None)
    assert summary["by_type"]["jump"]["findings"] == 1 and summary["review"] is None
    with np.load(tmp_path / "trajectory_flags.npz") as z:
        assert z["flags"].shape == (600, 3) and list(z["type_names"]) == list(TYPES)
    assert len(pd.read_csv(tmp_path / "trajectory_anomaly_intervals.csv")) == 1
    assert not list(tmp_path.glob("*.partial*"))
