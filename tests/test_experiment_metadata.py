"""Tests for folder-name parsing and metadata.csv reading."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.common.experiment_metadata import (
    build_experiment_metadata,
    build_view_recording,
    parse_experiment_name,
    read_metadata_csv,
)


def test_parse_experiment_name_basic():
    n = parse_experiment_name("20251220_D_1_1_FCMC")
    assert n.folder_name == "20251220_D_1_1_FCMC"
    assert n.start_date == "2025-12-20"
    assert n.group == "D"
    assert n.trial == 1
    assert n.day == 1
    assert n.paradigm == "FCMC"


def test_parse_experiment_name_multi_digit_trial_day():
    n = parse_experiment_name("20260106_E_12_3_FCMC")
    assert n.group == "E"
    assert n.trial == 12
    assert n.day == 3
    assert n.paradigm == "FCMC"


def test_parse_experiment_name_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_experiment_name("not-an-experiment")
    with pytest.raises(ValueError):
        parse_experiment_name("20251320_D_1_1_FCMC")  # invalid month


def test_read_metadata_csv(tmp_path: Path):
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(
        '"cameraName","top_cam"\n'
        '"frameRate","166"\n'
        '"totalFrames","348600"\n'
        '"videoFilename","cam0_top.mp4"\n',
        encoding="utf-8",
    )
    meta = read_metadata_csv(csv_path)
    assert meta["cameraName"] == "top_cam"
    assert meta["frameRate"] == "166"
    assert meta["totalFrames"] == "348600"
    assert meta["videoFilename"] == "cam0_top.mp4"


def test_build_view_recording(tmp_path: Path):
    cam_dir = tmp_path / "top_cam"
    cam_dir.mkdir()
    (cam_dir / "metadata.csv").write_text(
        '"cameraName","top_cam"\n'
        '"frameRate","166"\n'
        '"totalFrames","348600"\n'
        '"frameWidth","1200"\n'
        '"frameHeight","1200"\n'
        '"videoFilename","cam0_top.mp4"\n'
        '"codec","h265"\n'
        '"cameraMake","basler"\n'
        '"cameraSerialNo","25188743"\n',
        encoding="utf-8",
    )
    (cam_dir / "cam0_top.mp4").write_bytes(b"")

    v = build_view_recording(cam_dir)
    assert v.camera_name == "top_cam"
    assert v.total_frames == 348600
    assert v.frame_rate == 166.0
    assert v.frame_width == 1200
    assert v.codec == "h265"
    assert v.camera_make == "basler"
    assert v.video_path is not None and v.video_path.endswith("cam0_top.mp4")


def test_build_experiment_metadata(tmp_path: Path):
    exp_dir = tmp_path / "20251220_D_1_1_FCMC"
    for view, video_name in (
        ("top_cam", "cam0_top.mp4"),
        ("side1_cam", "cam1_side.mp4"),
        ("side2_cam", "cam2_side.mp4"),
    ):
        cam_dir = exp_dir / view
        cam_dir.mkdir(parents=True)
        (cam_dir / "metadata.csv").write_text(
            f'"cameraName","{view}"\n'
            '"frameRate","166"\n'
            '"totalFrames","100"\n'
            f'"videoFilename","{video_name}"\n',
            encoding="utf-8",
        )
        (cam_dir / video_name).write_bytes(b"")

    meta = build_experiment_metadata(exp_dir)
    assert meta.folder_name == "20251220_D_1_1_FCMC"
    assert meta.experiment.group == "D"
    assert set(meta.views.keys()) == {"top_cam", "side1_cam", "side2_cam"}
    assert meta.views["top_cam"].total_frames == 100
