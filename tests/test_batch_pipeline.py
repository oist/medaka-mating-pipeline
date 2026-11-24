import json
import os

from scripts.reconstruction_3d import batch_reconstruct_3d
from scripts.sleap.batch_bottomup_track import lock_is_live, write_lock
from scripts.tracking.track_simple_nn import write_csv


def test_lock_written_by_this_process_is_live(tmp_path):
    lock = tmp_path / "top_cam.running"
    write_lock(lock)
    assert lock_is_live(lock)


def test_lock_with_reused_pid_is_stale(tmp_path):
    # Same PID as a running process, but a different start time: e.g. the
    # PID was handed out again after a reboot.
    lock = tmp_path / "top_cam.running"
    lock.write_text(json.dumps({"pid": os.getpid(), "started": 0.0}))
    assert not lock_is_live(lock)


def test_missing_or_unreadable_lock_is_not_live(tmp_path):
    lock = tmp_path / "top_cam.running"
    assert not lock_is_live(lock)
    lock.write_text("not json")
    assert not lock_is_live(lock)


def test_write_csv_leaves_no_partial_file(tmp_path):
    out = tmp_path / "top_cam.tracks.csv"
    write_csv([{"frame": 0}], out)
    assert out.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["top_cam.tracks.csv"]


def test_existing_csv_without_done_marker_is_redone(tmp_path, monkeypatch):
    # A final CSV from an interrupted run must not count as finished.
    class Paths:
        sleap_output_dir = tmp_path / "sleap"
        reconstruction_output_dir = tmp_path / "recon"

    exp, tag = "20260112_P_1_1_FCMC", "t"
    slp_dir = Paths.sleap_output_dir / exp / tag
    slp_dir.mkdir(parents=True)
    for v in batch_reconstruct_3d.VIEWS:
        (slp_dir / f"{v}.slp").touch()
    (slp_dir / "top_cam.tracks.csv").touch()
    out_dir = Paths.reconstruction_output_dir / exp / tag
    out_dir.mkdir(parents=True)
    (out_dir / f"{tag}_3d_identified.csv").write_text("half")

    calls = []
    monkeypatch.setattr(batch_reconstruct_3d, "run", lambda cmd, log, dry: calls.append(cmd) or True)
    assert batch_reconstruct_3d.process(exp, Paths, tmp_path / "cfg.toml", tag, False) == "done (new)"
    assert len(calls) == 1 and "scripts.reconstruction_3d.reconstruct_from_slp" in calls[0]
    assert (out_dir / batch_reconstruct_3d.DONE_MARKER).exists()
    assert batch_reconstruct_3d.process(exp, Paths, tmp_path / "cfg.toml", tag, False) == "done"
