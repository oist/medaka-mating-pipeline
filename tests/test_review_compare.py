"""Versions file, compare data / rendering / clip export and an offscreen smoke test of the compare GUI."""

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from test_review_gui import _package  # noqa: E402  (synthetic review package)


def _ledger(folder):
    rows = [
        ("e1", "g1", "analysis_start", "*", "15", "", "15", "1"),
        ("e2", "g2", "label", "ID_1", "15", "", "15", "F"),
        ("e3", "g3", "label", "ID_2", "15", "", "15", "M1"),
        ("e4", "g4", "label", "ID_3", "15", "", "15", "M2"),
        ("e5", "g5", "swap", "ID_1|ID_2", "60", "", "60", "1"),
    ]
    ev = pd.DataFrame([dict(event_id=a, group=b, kind=c, fish=d, frame=e, frame_end=f, frame_marked=g, value=h,
                            risk_id="", session="s", created="", note="") for a, b, c, d, e, f, g, h in rows])
    ev.to_csv(folder / "human_events.csv", index=False)
    pd.DataFrame({"session": ["s"], "frame_start": [12], "frame_end": [119], "speed": [2.0], "mode": ["play"],
                  "wall_start": [0], "wall_end": [1]}).to_csv(folder / "coverage.csv", index=False)


def _exported(tmp_path):
    from scripts.review.export_reviewed import export

    n, k = 108, 3
    xyz = np.zeros((n, k, 3), np.float32)
    xyz[:, :, 0] = [0, 10, 40]
    folder = _package(tmp_path, xyz=xyz)
    # Spread the tracks apart and give track 0 a one-frame spike.
    with np.load(folder / "review_data.npz") as z:
        d = {key: z[key] for key in z.files}
    nodes = d["nodes3d_filled"].copy()
    nodes += np.array([0.0, 40.0, 80.0], np.float32)[None, :, None, None]
    nodes[:, :, :, 1] += np.linspace(0, 30, n, dtype=np.float32)[:, None, None]
    nodes[20, 0, 1, 2] += 6.0                    # frame 32, before the swap
    d["nodes3d_filled"] = d["nodes3d"] = nodes
    np.savez_compressed(folder / "review_data.npz", **d)
    _ledger(folder)
    summary = export(folder)
    return folder, summary


def test_export_writes_post_pass_and_versions(tmp_path):
    folder, summary = _exported(tmp_path)
    assert summary["checks"]["fully_watched"]            # every playback speed counts
    table = pd.read_csv(folder / "reviewed_3d.csv")
    assert np.allclose(table["X"], table["pectoral_X"]) and "centroid_X" in table
    assert (table["pectoral_source"] == 5).any()          # the spike was bridged
    z = np.load(folder / "versions.npz")
    assert list(z["labels"]) == ["F", "M1", "M2"]
    assert list(z["present"]) == ["machine", "manual", "final"]     # no calibration here: no initial
    # Before the swap F is track 0 in every version; after it, the reviewed F is track 1.
    t_before, t_after = 30 - 12, 80 - 12
    assert z["track"][1, t_after, 0] == 0 and z["track"][2, t_after, 0] == 1 and z["track"][2, t_before, 0] == 0


def test_compare_data_alignment_render_and_clip(tmp_path):
    from scripts.review.compare_gui import CompareData, export_clip, render_view

    folder, _ = _exported(tmp_path)
    cd = CompareData(folder, calibration=False)
    t = 80 - cd.first
    plain = cd.series("machine", t, t + 1, aligned=False)[0]
    aligned = cd.series("machine", t, t + 1, aligned=True)[0]
    manual = cd.series("manual", t, t + 1, aligned=True)[0]
    assert np.allclose(aligned, manual, equal_nan=True)           # same identity: only geometry differs
    assert not np.allclose(plain, manual, equal_nan=True)         # tracker identity: the swap shows
    assert set(cd.changes["group"]) >= {"review", "post"}
    img, _ = render_view(cd, 0, 40, ("machine", "final"), True, (120, 100))
    assert img.shape[0] <= 100 and img.shape[1] <= 120
    out = export_clip(cd, 30, 50, ("machine", "final"), speed=1.0, view_size=96, out=tmp_path / "c.mp4")
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()
    cd.close()


def test_compare_gui_smoke(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from scripts.review.compare_gui import run_app

    folder, _ = _exported(tmp_path)
    assert run_app(folder, smoke_test=True) == 0
