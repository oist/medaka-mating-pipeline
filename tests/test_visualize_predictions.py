import csv

import numpy as np

from scripts.sleap.visualize_predictions import (
    TrackPoint,
    draw_predictions,
    draw_track_points,
    load_track_points_by_frame,
    render,
)


def test_load_track_points_by_frame_reads_identity_rows(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "track_id", "x", "y", "identity"])
        writer.writeheader()
        writer.writerow({"frame": 5, "track_id": 1, "x": "10.2", "y": "20.7", "identity": "ID_2"})

    points = load_track_points_by_frame(csv_path)

    assert points[5][0].track_id == 1
    assert points[5][0].identity == "ID_2"


def test_draw_predictions_nearest_matches_blob_centroid_track_point():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    pts = np.array([[30.0, 30.0]], dtype=float)
    out = draw_predictions(
        frame.copy(),
        [(pts, None, 0)],
        ["mouth"],
        frame_idx=0,
        track_points_by_frame={0: [TrackPoint(frame=0, track_id=2, x=35.0, y=34.0, identity="ID_3")]},
        track_match_distance=20.0,
    )

    assert out[30, 30].sum() > 0


def test_draw_track_points_draws_unknown_track():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    out = draw_track_points(frame, [TrackPoint(frame=0, track_id=-1, x=40.0, y=40.0, identity="unknown")])

    assert out[40, 40].sum() > 0


def test_render_track_only_requires_no_sleap_drawing(monkeypatch, tmp_path):
    import cv2

    video_path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (40, 30))
    writer.write(np.zeros((30, 40, 3), dtype=np.uint8))
    writer.release()

    slp_path = tmp_path / "dummy.slp"
    slp_path.write_text("placeholder", encoding="utf-8")
    track_csv = tmp_path / "tracks.csv"
    track_csv.write_text("frame,track_id,x,y,identity\n0,1,20,15,ID_2\n", encoding="utf-8")

    called = {"draw": False}

    def fake_index(_path):
        return {0: [(np.array([[20.0, 15.0]]), None, 0)]}, ["mouth"]

    def fail_draw(*_args, **_kwargs):
        called["draw"] = True
        raise AssertionError("draw_predictions should not be called in track-only mode")

    monkeypatch.setattr("scripts.sleap.visualize_predictions.index_predictions_by_frame", fake_index)
    monkeypatch.setattr("scripts.sleap.visualize_predictions.draw_predictions", fail_draw)

    render(
        video_path,
        slp_path,
        tmp_path / "out.mp4",
        frames_spec="0-0",
        track_csv=track_csv,
        draw_sleap_points=False,
    )

    assert called["draw"] is False
    assert (tmp_path / "out.mp4").exists()
