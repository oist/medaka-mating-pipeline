from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.sleap.prepare_side_cumulative_training import (
    EXPERIMENTS,
    SIDE_VIEWS,
    _structured_points,
    assert_train_val_guard_integrity,
    label_key,
    prepare_cumulative_side_labels,
    sha256_file,
)


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
    writer.release()


def _make_instance(
    sio,
    skeleton,
    *,
    offset: float,
    mouth_visible: bool = True,
):
    points = np.asarray(
        [
            [offset + 1.0, offset + 2.0, float(mouth_visible)],
            [offset + 3.0, offset + 4.0, 1.0],
        ],
        dtype=float,
    )
    return sio.Instance.from_numpy(points_data=points, skeleton=skeleton)


def _save_package(
    path: Path,
    *,
    sio,
    skeleton,
    videos,
    source_frames,
    offset_by_source=None,
    hidden_mouth_source=None,
) -> None:
    offset_by_source = offset_by_source or {}
    labeled_frames = []
    for experiment, frame in source_frames:
        for view in SIDE_VIEWS:
            offset = float(offset_by_source.get((experiment, frame), frame))
            instances = [
                _make_instance(
                    sio,
                    skeleton,
                    offset=offset + fish * 10.0,
                    mouth_visible=not (
                        hidden_mouth_source == (experiment, view, frame)
                        and fish == 0
                    ),
                )
                for fish in range(3)
            ]
            labeled_frames.append(
                sio.LabeledFrame(
                    video=videos[(experiment, view)],
                    frame_idx=int(frame),
                    instances=instances,
                )
            )
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=list(videos.values()),
        labeled_frames=labeled_frames,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(path), embed=False, verbose=False)


def _write_manifest(path: Path) -> None:
    rows = []
    for experiment in EXPERIMENTS:
        rows.extend(
            [
                {
                    "experiment": experiment,
                    "frame": 105,
                    "split_role": "validation",
                },
                {
                    "experiment": experiment,
                    "frame": 200,
                    "split_role": "train",
                },
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["experiment", "frame", "split_role"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_cumulative_union_guards_and_preserves_hidden_raw_xy(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"], edges=[("mouth", "tail")]
    )
    videos = {}
    for experiment in EXPERIMENTS:
        for view in SIDE_VIEWS:
            video_path = (
                tmp_path / "raw" / experiment / view / f"{view}.avi"
            )
            _write_video(video_path)
            videos[(experiment, view)] = sio.Video.from_filename(
                str(video_path)
            )

    base_train = tmp_path / "round8" / "side" / "train_labels.pkg.slp"
    base_validation = (
        tmp_path / "round8" / "side" / "val_labels.pkg.slp"
    )
    new_labels = tmp_path / "review" / "complete.pkg.slp"
    manifest = tmp_path / "selection_210.csv"
    base_train_sources = [
        (experiment, frame)
        for experiment in EXPERIMENTS
        for frame in (0, 100)
    ]
    # Exercise exact same-split collision precedence for K_1_1 frame 200.
    base_train_sources.append((EXPERIMENTS[0], 200))
    _save_package(
        base_train,
        sio=sio,
        skeleton=skeleton,
        videos=videos,
        source_frames=base_train_sources,
        offset_by_source={(EXPERIMENTS[0], 200): 10.0},
    )
    _save_package(
        base_validation,
        sio=sio,
        skeleton=skeleton,
        videos=videos,
        source_frames=[(experiment, 1000) for experiment in EXPERIMENTS],
    )
    _save_package(
        new_labels,
        sio=sio,
        skeleton=skeleton,
        videos=videos,
        source_frames=[
            (experiment, frame)
            for experiment in EXPERIMENTS
            for frame in (105, 200)
        ],
        offset_by_source={(EXPERIMENTS[0], 200): 900.0},
        hidden_mouth_source=(EXPERIMENTS[0], "side1_cam", 200),
    )
    _write_manifest(manifest)

    hashes_before = {
        path: sha256_file(path)
        for path in (base_train, base_validation, new_labels, manifest)
    }
    output_dir = tmp_path / "cumulative"
    report = prepare_cumulative_side_labels(
        base_train=base_train,
        base_validation=base_validation,
        new_labels=new_labels,
        manifest=manifest,
        output_dir=output_dir,
        guard_frames=10,
        copy_new_labels_to_output=True,
    )

    assert report["guard_removal"][
        "base_train_source_frames_removed"
    ] == 3
    assert report["guard_removal"][
        "base_train_labeled_frames_removed"
    ] == 6
    assert report["collisions"][
        "train_exact_labeled_frame_collisions_new_won"
    ] == 2
    assert report["outputs"]["train"]["labeled_frames"] == 12
    assert report["outputs"]["primary_validation"]["labeled_frames"] == 6
    assert report["outputs"]["round8_regression_validation"][
        "labeled_frames"
    ] == 6
    assert report["outputs"]["combined_validation_audit_only"][
        "labeled_frames"
    ] == 12
    assert report["outputs"]["primary_validation_evaluation_manifest"][
        "rows"
    ] == 6
    assert report["outputs"]["primary_validation_evaluation_manifest"][
        "strict_centroid_eligible_rows"
    ] == 6
    evaluation_manifest_path = (
        output_dir / "side" / "val_evaluation_manifest.csv"
    )
    with evaluation_manifest_path.open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        evaluation_rows = list(csv.DictReader(stream))
    assert "source_slp" in evaluation_rows[0]
    assert all(
        row["source_slp"] == row["source_labels_path"]
        for row in evaluation_rows
    )
    assert [
        int(row["source_labeled_frame_index"]) for row in evaluation_rows
    ] == list(range(6))
    assert report["outputs"]["train"]["hidden_finite_mouths"] == 1
    assert report["outputs"]["train"]["visible_finite_mouths"] == 35
    assert report["outputs"]["train"]["raw_finite_mouths"] == 36
    copied_source = report["inputs"]["new_labels_training_source_copy"]
    assert Path(copied_source["path"]).is_file()
    assert copied_source["sha256"] == hashes_before[new_labels]

    train = sio.load_slp(
        str(output_dir / "side" / "train_labels.pkg.slp"),
        open_videos=False,
    )
    train_lookup = {label_key(lf): lf for lf in train.labeled_frames}
    collision_key = (EXPERIMENTS[0], "side1_cam", 200)
    collision_xy, collision_visibility = _structured_points(
        train_lookup[collision_key].instances[0]
    )
    np.testing.assert_allclose(collision_xy[0], [901.0, 902.0])
    assert not collision_visibility[0]
    # The ordinary NumPy view masks the hidden mouth, demonstrating why the
    # structured point array must be used for cloning.
    assert np.isnan(train_lookup[collision_key].instances[0].numpy()[0]).all()

    observed_train_sources = {
        (key[0], key[2]) for key in train_lookup
    }
    assert all(
        (experiment, 100) not in observed_train_sources
        for experiment in EXPERIMENTS
    )
    assert all(
        sha256_file(path) == expected
        for path, expected in hashes_before.items()
    )

    round9b_manifest = tmp_path / "round9b_selection_210.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    for row in original_rows:
        row["round9b_split_role"] = (
            "train"
            if row["split_role"] == "validation"
            else "validation"
        )
    with round9b_manifest.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "experiment",
                "frame",
                "split_role",
                "round9b_split_role",
            ],
        )
        writer.writeheader()
        writer.writerows(original_rows)
    split_report = tmp_path / "round9b_split_report.json"
    split_report.write_text(
        json.dumps(
            {
                "schema_version": "round9b-side-split-v1",
                "outputs": {
                    "selection_210": {
                        "sha256": sha256_file(round9b_manifest)
                    }
                },
                "inherited_training": {
                    "malformed_source_groups_removed": [
                        {
                            "experiment": EXPERIMENTS[0],
                            "frame": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    round9b_output = tmp_path / "round9b_cumulative"
    round9b_report = prepare_cumulative_side_labels(
        base_train=base_train,
        base_validation=base_validation,
        new_labels=new_labels,
        manifest=round9b_manifest,
        output_dir=round9b_output,
        guard_frames=10,
        split_column="round9b_split_role",
        inherited_exclusions_report=split_report,
    )
    assert round9b_report["split_column"] == "round9b_split_role"
    assert round9b_report["inherited_source_exclusion"][
        "requested_source_frames"
    ] == 1
    assert round9b_report["inherited_source_exclusion"][
        "removed_labeled_frames"
    ] == 2
    assert round9b_report["inherited_source_exclusion"][
        "removed_summary"
    ]["instances"] == 6
    assert round9b_report["outputs"]["train"]["labeled_frames"] == 16
    assert round9b_report["outputs"][
        "primary_validation_evaluation_manifest"
    ]["strict_centroid_eligible_rows"] == 5
    assert round9b_report["outputs"][
        "primary_validation_evaluation_manifest"
    ]["strict_centroid_ineligible_rows"] == 1


def test_full_guard_integrity_rejects_nearby_source_frames() -> None:
    train = {
        ("20251225_K_1_1_FCMC", "side1_cam", 100),
        ("20251225_K_1_1_FCMC", "side2_cam", 100),
    }
    validation = {
        ("20251225_K_1_1_FCMC", "side1_cam", 105),
        ("20251225_K_1_1_FCMC", "side2_cam", 105),
    }

    with pytest.raises(ValueError, match="guard violation"):
        assert_train_val_guard_integrity(
            train, validation, guard_frames=10
        )


def test_full_guard_integrity_rejects_exact_key_leakage() -> None:
    key = ("20251225_K_1_1_FCMC", "side1_cam", 100)

    with pytest.raises(ValueError, match="key leakage"):
        assert_train_val_guard_integrity(
            [key], [key], guard_frames=0
        )
