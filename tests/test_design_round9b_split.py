from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.sleap.design_round9b_split import (
    CATEGORIES,
    CATEGORY_TARGETS,
    EXPERIMENT_CATEGORY_QUOTAS,
    MALFORMED_INHERITED_SOURCE_FRAMES,
    design_round9b_split,
)
from scripts.sleap.prepare_side_cumulative_training import (
    EXPERIMENTS,
    SIDE_VIEWS,
)


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    writer.write(np.full((24, 32, 3), 60, dtype=np.uint8))
    writer.release()


def _instances(sio, skeleton, count: int, offset: float):
    return [
        sio.Instance.from_numpy(
            points_data=np.asarray(
                [
                    [offset + index * 3, 5.0, 1.0],
                    [offset + index * 3 + 1, 6.0, 1.0],
                ],
                dtype=float,
            ),
            skeleton=skeleton,
        )
        for index in range(count)
    ]


def _save_labels(path: Path, sio, skeleton, videos, specs) -> None:
    labeled_frames = []
    for experiment, frame, counts in specs:
        for view, count in zip(SIDE_VIEWS, counts):
            labeled_frames.append(
                sio.LabeledFrame(
                    video=videos[(experiment, view)],
                    frame_idx=int(frame),
                    instances=_instances(
                        sio, skeleton, int(count), float(frame % 500)
                    ),
                )
            )
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=list(videos.values()),
        labeled_frames=labeled_frames,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(path), embed=False, verbose=False)


def _selection_rows():
    totals = {
        "20251225_K_1_1_FCMC": 80,
        "20251225_K_1_2_FCMC": 50,
        "20251225_K_1_3_FCMC": 80,
    }
    rows = []
    base_frame = {
        "20251225_K_1_1_FCMC": 400_000,
        "20251225_K_1_2_FCMC": 600_000,
        "20251225_K_1_3_FCMC": 800_000,
    }
    for experiment in EXPERIMENTS:
        counter = 0
        for category in CATEGORIES:
            wanted = EXPERIMENT_CATEGORY_QUOTAS[experiment][category]
            # One spare candidate in every non-empty cell makes ranking
            # behavior observable without affecting feasibility.
            for index in range(wanted + int(wanted > 0)):
                rows.append(
                    {
                        "experiment": experiment,
                        "frame": base_frame[experiment] + counter * 1000,
                        "split_role": (
                            "validation" if counter < 10 else "train"
                        ),
                        "selection_category": category,
                        "selection_category_score": 1.0 - index * 0.01,
                    }
                )
                counter += 1
        while counter < totals[experiment]:
            rows.append(
                {
                    "experiment": experiment,
                    "frame": base_frame[experiment] + counter * 1000,
                    "split_role": "train",
                    "selection_category": "clean_diverse_control",
                    "selection_category_score": 0.1,
                }
            )
            counter += 1
    return rows


def _write_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_round9b_split_is_deterministic_stratified_and_guarded(
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

    rows = _selection_rows()
    selection_manifest = tmp_path / "selection_210.csv"
    _write_csv(
        selection_manifest,
        rows,
        [
            "experiment",
            "frame",
            "split_role",
            "selection_category",
            "selection_category_score",
        ],
    )
    reviewed = tmp_path / "reviewed.pkg.slp"
    _save_labels(
        reviewed,
        sio,
        skeleton,
        videos,
        [
            (row["experiment"], int(row["frame"]), (3, 3))
            for row in rows
        ],
    )

    # The highest-scoring K_1_1 clean candidate is historically exposed and
    # must be replaced by the next deterministic candidate.
    exposed_frame = next(
        int(row["frame"])
        for row in rows
        if row["experiment"] == EXPERIMENTS[0]
        and row["selection_category"] == "clean_diverse_control"
    )
    inherited = tmp_path / "round8_train.pkg.slp"
    malformed_specs = []
    for experiment, frame in MALFORMED_INHERITED_SOURCE_FRAMES:
        side1_count = (
            2
            if (experiment, frame)
            == ("20251225_K_1_2_FCMC", 281988)
            else 4
        )
        malformed_specs.append(
            (experiment, int(frame), (side1_count, 3))
        )
    _save_labels(
        inherited,
        sio,
        skeleton,
        videos,
        [
            *malformed_specs,
            (EXPERIMENTS[0], exposed_frame + 1, (3, 3)),
        ],
    )

    locked = tmp_path / "locked.csv"
    _write_csv(
        locked,
        [
            {
                "experiment": EXPERIMENTS[0],
                "frame_idx": 100,
                "event_start_frame": 100,
                "event_end_frame": 100,
            }
        ],
        [
            "experiment",
            "frame_idx",
            "event_start_frame",
            "event_end_frame",
        ],
    )
    future_locked = tmp_path / "future_locked.csv"
    _write_csv(
        future_locked,
        [
            {
                "experiment": EXPERIMENTS[1],
                "frame_idx": 200,
                "future_lock_eligible": True,
            }
        ],
        ["experiment", "frame_idx", "future_lock_eligible"],
    )

    output_dir = tmp_path / "round9b"
    report = design_round9b_split(
        selection_manifest=selection_manifest,
        inherited_train_labels=inherited,
        reviewed_labels=reviewed,
        locked_manifest=locked,
        future_locked_manifest=future_locked,
        output_dir=output_dir,
        guard_frames=166,
    )

    assert report["split_counts"]["validation_source_frames"] == 30
    assert report["split_counts"]["validation_by_experiment"] == {
        experiment: 10 for experiment in EXPERIMENTS
    }
    assert report["split_counts"]["validation_by_category"] == (
        CATEGORY_TARGETS
    )
    assert report["inherited_training"]["malformed_removed"][
        "labeled_frames"
    ] == 6
    assert report["inherited_training"]["malformed_removed"][
        "instances"
    ] == 19
    assert report["cumulative_training_counts"]["labeled_frames"] == 362
    assert report["cumulative_training_counts"]["instances"] == 1086
    assert report["verification"][
        "validation_vs_inherited_train_guard_violations"
    ] == []
    assert report["verification"]["locked_hitl_guard_violations"] == []

    with (output_dir / "round9b_validation_30.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        validation_rows = list(csv.DictReader(stream))
    assert len(validation_rows) == 30
    assert not any(
        int(row["frame"]) == exposed_frame for row in validation_rows
    )
    assert Counter(row["selection_category"] for row in validation_rows) == (
        Counter(CATEGORY_TARGETS)
    )
