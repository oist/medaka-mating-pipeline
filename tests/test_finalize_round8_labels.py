from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.sleap.finalize_round8_labels import (
    blocked_training_ranges,
    finalize,
    frame_is_blocked,
    validate_corrected_labels,
)


EXPERIMENT = "synthetic_round8_experiment"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def _write_video(path: Path, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for frame in range(frames):
        writer.write(np.full((24, 32, 3), frame * 30 + 50, dtype=np.uint8))
    writer.release()


def _three_instances(sio, skeleton, *, partial_first: bool = False):
    instances = []
    for fish in range(3):
        points = np.array(
            [[5.0 + fish * 5, 8.0], [7.0 + fish * 5, 9.0]], dtype=float
        )
        if partial_first and fish == 0:
            points[1] = np.nan
        instances.append(sio.Instance.from_numpy(points, skeleton=skeleton))
    return instances


def _save_labels(path: Path, *, skeleton, videos, frames_by_view, sio) -> None:
    lfs = []
    for view, video in videos.items():
        for frame in frames_by_view.get(view, []):
            lfs.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=frame,
                    instances=_three_instances(
                        sio, skeleton, partial_first=(view == "side2_cam" and frame == 0)
                    ),
                )
            )
    labels = sio.Labels(
        skeletons=[skeleton], videos=list(videos.values()), labeled_frames=lfs
    )
    sio.save_slp(labels, str(path), embed=False, verbose=False)


def test_holdout_event_ranges_block_training_guard_band() -> None:
    manifest = pd.DataFrame(
        {
            "split": ["holdout", "holdout", "train"],
            "event_group_id": ["held", "held", "train"],
            "event_start_frame": [100, 100, 300],
            "event_end_frame": [120, 120, 300],
        }
    )
    ranges = blocked_training_ranges(manifest, guard_band_frames=10)
    assert ranges == [(90, 130, "held")]
    assert frame_is_blocked(90, ranges)
    assert frame_is_blocked(130, ranges)
    assert not frame_is_blocked(131, ranges)


def test_finalize_builds_explicit_leakage_safe_train_and_val_packages(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    skeleton = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])
    videos = {}
    for view in VIEWS:
        video_path = tmp_path / "raw" / EXPERIMENT / view / "camera.avi"
        _write_video(video_path)
        videos[view] = sio.Video.from_filename(str(video_path))

    corrected = tmp_path / "pilot_120_edits_full.pkg.slp"
    _save_labels(
        corrected,
        skeleton=skeleton,
        videos=videos,
        frames_by_view={view: [0, 1] for view in VIEWS},
        sio=sio,
    )
    models_dir = tmp_path / "models"
    top_base = models_dir / "retrain_20260704" / "top" / "labels.pkg.slp"
    side_base = models_dir / "retrain_20260704" / "side" / "labels.pkg.slp"
    top_base.parent.mkdir(parents=True)
    side_base.parent.mkdir(parents=True)
    _save_labels(
        top_base,
        skeleton=skeleton,
        videos={"top_cam": videos["top_cam"]},
        frames_by_view={"top_cam": [0, 1]},
        sio=sio,
    )
    _save_labels(
        side_base,
        skeleton=skeleton,
        videos={"side1_cam": videos["side1_cam"], "side2_cam": videos["side2_cam"]},
        frames_by_view={"side1_cam": [0, 1], "side2_cam": [0, 1]},
        sio=sio,
    )

    manifest = pd.DataFrame(
        {
            "experiment": [EXPERIMENT, EXPERIMENT],
            "frame": [0, 1],
            "split": ["train", "holdout"],
            "annotation_batch": ["pilot", "pilot"],
            "event_group_id": ["train_0", "held_1"],
            "event_start_frame": [0, 1],
            "event_end_frame": [0, 1],
        }
    )
    manifest_path = tmp_path / "selection_360.csv"
    manifest.to_csv(manifest_path, index=False)
    paths_config = tmp_path / "paths.toml"
    paths_config.write_text(
        "[paths]\n"
        f'runs_dir = "{str(tmp_path / "runs").replace(chr(92), "/")}"\n'
        f'sleap_models_dir = "{str(models_dir).replace(chr(92), "/")}"\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "finalized"
    report = finalize(
        argparse.Namespace(
            corrected=[corrected],
            manifest=manifest_path,
            annotation_batch="pilot",
            experiment=EXPERIMENT,
            paths_config=paths_config,
            round7_top_labels=top_base,
            round7_side_labels=side_base,
            output_dir=output_dir,
            instances_per_view=3,
            guard_band_frames=0,
        )
    )

    assert report["corrected_counts"] == {
        "train_view_images": 3,
        "holdout_view_images": 3,
    }
    assert report["merge"]["top"]["blocked_base_frames_removed"] == 1
    assert report["merge"]["top"]["corrected_collisions_won"] == 1
    assert report["merge"]["side"]["blocked_base_frames_removed"] == 2
    assert report["merge"]["side"]["corrected_collisions_won"] == 2
    assert report["random_frame_validation_disabled"] is True
    assert report["round7_inputs_modified"] is False

    top_train = sio.load_slp(str(output_dir / "top" / "train_labels.pkg.slp"))
    top_val = sio.load_slp(str(output_dir / "top" / "val_labels.pkg.slp"))
    side_train = sio.load_slp(str(output_dir / "side" / "train_labels.pkg.slp"))
    side_val = sio.load_slp(str(output_dir / "side" / "val_labels.pkg.slp"))
    combined = sio.load_slp(str(output_dir / "combined_labels.pkg.slp"))
    assert [lf.frame_idx for lf in top_train.labeled_frames] == [0]
    assert [lf.frame_idx for lf in top_val.labeled_frames] == [1]
    assert len(side_train.labeled_frames) == 2
    assert len(side_val.labeled_frames) == 2
    assert len(combined.labeled_frames) == 3

    for bucket in ("top", "side"):
        for kind in ("centroid", "centered_instance"):
            config = yaml.safe_load(
                (output_dir / bucket / "configs" / f"{kind}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            assert config["data_config"]["validation_fraction"] == 0.0
            assert config["data_config"]["train_labels_path"]
            assert config["data_config"]["val_labels_path"]
            assert (
                config["data_config"]["data_pipeline_fw"]
                == "torch_dataset_cache_img_disk"
            )
            assert config["data_config"]["delete_cache_imgs_after_training"] is False
            assert f"finalized_{bucket}_{kind}" in config["data_config"]["cache_img_path"]


def test_invalid_training_view_excludes_whole_synchronized_source_frame(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    skeleton = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])
    videos = {}
    for view in VIEWS:
        video_path = tmp_path / "raw" / EXPERIMENT / view / "camera.avi"
        _write_video(video_path, frames=1)
        videos[view] = sio.Video.from_filename(str(video_path))
    corrected = tmp_path / "corrected.slp"
    _save_labels(
        corrected,
        skeleton=skeleton,
        videos=videos,
        frames_by_view={view: [0] for view in VIEWS},
        sio=sio,
    )
    labels = sio.load_slp(str(corrected))
    top_lf = next(
        lf for lf in labels.labeled_frames if "top_cam" in str(lf.video.filename)
    )
    top_lf.instances.pop()
    sio.save_slp(labels, str(corrected), embed=False, verbose=False)
    manifest = pd.DataFrame(
        {
            "experiment": [EXPERIMENT],
            "frame": [0],
            "split": ["train"],
            "annotation_batch": ["pilot"],
        }
    )

    observed, _skeleton, report = validate_corrected_labels(
        [corrected],
        manifest,
        experiment=EXPERIMENT,
        annotation_batch="pilot",
        exclude_invalid_training_source_frames=True,
    )

    assert observed == {}
    assert report["excluded_invalid_training_source_frames"] == [0]
    assert len(report["excluded_invalid_view_images"]) == 1
