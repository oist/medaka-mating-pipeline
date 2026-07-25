from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.sleap.prepare_mouth_anchor_completion_labels import (
    COMPLETED_MANIFEST_NAME,
    COMPLETED_PACKAGE_NAME,
    COMPLETED_PROVENANCE_NAME,
    PREPARED_MANIFEST_NAME,
    PREPARED_PROVENANCE_NAME,
    SEED_PACKAGE_NAME,
    finalize_completion_labels,
    label_key,
    locked_manifest_overlaps,
    prepare_completion_labels,
    select_missing_mouth_groups,
    sha256_file,
)

EXPERIMENT = "synthetic_round8_experiment"
SIDE_VIEWS = ("side1_cam", "side2_cam")


def _write_video(path: Path, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for frame in range(frames):
        writer.write(
            np.full((24, 32, 3), 40 + frame * 20, dtype=np.uint8)
        )
    writer.release()


def _instances(
    sio,
    skeleton,
    *,
    missing_mouths: tuple[int, ...] = (),
    tail_offset: float = 0.0,
):
    instances = []
    for fish in range(3):
        points = np.array(
            [
                [5.0 + fish * 7, 8.0 + fish],
                [8.0 + fish * 7 + tail_offset, 10.0 + fish],
            ],
            dtype=float,
        )
        if fish in missing_mouths:
            points[0] = np.nan
        instances.append(
            sio.Instance.from_numpy(points_data=points, skeleton=skeleton)
        )
    return instances


def _make_source_package(
    tmp_path: Path,
    *,
    affected_frame: int = 1,
) -> tuple[Path, object, dict[str, object]]:
    sio = pytest.importorskip("sleap_io")
    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
        name="medaka_mouth_tail",
    )
    videos = {}
    for view, camera in (
        ("side1_cam", "cam1_side.avi"),
        ("side2_cam", "cam2_side.avi"),
    ):
        path = tmp_path / "raw" / EXPERIMENT / view / camera
        _write_video(path)
        videos[view] = sio.Video.from_filename(str(path))

    labeled_frames = []
    for frame_idx in (0, affected_frame):
        for view in SIDE_VIEWS:
            missing = ()
            if frame_idx == affected_frame and view == "side1_cam":
                missing = (1,)
            labeled_frames.append(
                sio.LabeledFrame(
                    video=videos[view],
                    frame_idx=frame_idx,
                    instances=_instances(
                        sio,
                        skeleton,
                        missing_mouths=missing,
                        tail_offset=0.25 if view == "side2_cam" else 0.0,
                    ),
                )
            )
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=list(videos.values()),
        labeled_frames=labeled_frames,
    )
    path = tmp_path / "round8" / "side" / "train_labels.pkg.slp"
    path.parent.mkdir(parents=True)
    sio.save_slp(labels, str(path), embed=False, verbose=False)
    return path, skeleton, videos


def _lookup(labels) -> dict[tuple[str, str, int], object]:
    return {label_key(labeled_frame): labeled_frame for labeled_frame in labels}


def _copy_gui_output_with_completion(
    *,
    seed_path: Path,
    output_path: Path,
    completed_mouth: tuple[float, float] | None = (22.0, 13.0),
    mutate_finite_tail: bool = False,
    add_top_lf: bool = False,
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    labels = sio.load_slp(str(seed_path), open_videos=False)
    side1 = next(
        lf
        for lf in labels.labeled_frames
        if label_key(lf)[1] == "side1_cam"
    )
    points = np.asarray(side1.instances[1].numpy(), dtype=float)
    if completed_mouth is not None:
        points[0] = completed_mouth
    if mutate_finite_tail:
        points[1, 0] += 1.0
    side1.instances[1] = sio.Instance.from_numpy(
        points_data=points,
        skeleton=labels.skeletons[0],
    )

    if add_top_lf:
        top_path = (
            tmp_path
            / "raw"
            / EXPERIMENT
            / "top_cam"
            / "cam0_top.avi"
        )
        _write_video(top_path)
        top_video = sio.Video.from_filename(str(top_path))
        labels.videos.append(top_video)
        labels.labeled_frames.append(
            sio.LabeledFrame(
                video=top_video,
                frame_idx=int(side1.frame_idx),
                instances=_instances(sio, labels.skeletons[0]),
            )
        )
    sio.save_slp(labels, str(output_path), embed=False, verbose=False)


def test_selects_both_original_side_lfs_for_every_affected_group(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    source_path, _skeleton, _videos = _make_source_package(tmp_path)
    labels = sio.load_slp(str(source_path), open_videos=False)

    selected = select_missing_mouth_groups(labels)

    assert [item.key for item in selected] == [
        (EXPERIMENT, "side1_cam", 1),
        (EXPERIMENT, "side2_cam", 1),
    ]
    assert selected[0].missing_mouth_indices == (1,)
    assert selected[1].missing_mouth_indices == ()
    assert [len(item.labeled_frame.instances) for item in selected] == [3, 3]


@pytest.mark.parametrize(
    ("locked_frame", "kind"),
    [(1000, "exact"), (834, "guard_band"), (1166, "guard_band")],
)
def test_locked_manifest_overlap_rejects_exact_and_inclusive_166_guard(
    tmp_path: Path,
    locked_frame: int,
    kind: str,
) -> None:
    sio = pytest.importorskip("sleap_io")
    source_path, _skeleton, _videos = _make_source_package(
        tmp_path,
        affected_frame=1000,
    )
    labels = sio.load_slp(str(source_path), open_videos=False)
    selected = select_missing_mouth_groups(labels)
    manifest = tmp_path / "locked.csv"
    manifest.write_text(
        f"experiment,frame_idx,view\n{EXPERIMENT},{locked_frame},top_cam\n",
        encoding="utf-8",
    )

    overlaps = locked_manifest_overlaps(
        selected,
        manifest,
        guard_frames=166,
    )

    assert len(overlaps) == 1
    assert overlaps[0]["overlap_kind"] == kind
    assert overlaps[0]["distance_frames"] == abs(1000 - locked_frame)


def test_prepare_writes_thin_exact_seed_manifest_and_provenance(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    source_path, _skeleton, _videos = _make_source_package(tmp_path)
    source_sha = sha256_file(source_path)
    locked = tmp_path / "locked.csv"
    locked.write_text(
        f"experiment,frame_idx\n{EXPERIMENT},1000\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "prepared"

    report = prepare_completion_labels(
        source_labels=source_path,
        output_dir=output_dir,
        locked_manifest=locked,
    )

    assert sha256_file(source_path) == source_sha
    assert report["selected_groups"] == 1
    assert report["selected_side_labeled_frames"] == 2
    assert report["originally_missing_mouths"] == 1
    assert report["embedded_frames"] is False
    assert report["source_labels_modified"] is False
    assert report["contains_locked_test"] is False
    assert report["split_role"] == "training_label_repair_only"
    assert (output_dir / SEED_PACKAGE_NAME).is_file()
    assert (output_dir / PREPARED_MANIFEST_NAME).is_file()
    assert (output_dir / PREPARED_PROVENANCE_NAME).is_file()

    source = sio.load_slp(str(source_path), open_videos=False)
    seed = sio.load_slp(
        str(output_dir / SEED_PACKAGE_NAME),
        open_videos=False,
    )
    source_lookup = _lookup(source)
    seed_lookup = _lookup(seed)
    assert set(seed_lookup) == {
        (EXPERIMENT, "side1_cam", 1),
        (EXPERIMENT, "side2_cam", 1),
    }
    for key, seeded_lf in seed_lookup.items():
        source_lf = source_lookup[key]
        assert len(seeded_lf.instances) == len(source_lf.instances)
        for seeded, original in zip(seeded_lf.instances, source_lf.instances):
            assert np.array_equal(
                seeded.numpy(),
                original.numpy(),
                equal_nan=True,
            )

    with (output_dir / PREPARED_MANIFEST_NAME).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["view"] for row in rows} == set(SIDE_VIEWS)
    assert {
        row["split_role"] for row in rows
    } == {"training_label_repair_only"}
    assert {row["contains_locked_test"] for row in rows} == {"False"}
    assert {
        row["missing_mouth_instance_indices"]
        for row in rows
    } == {"", "1"}
    provenance = json.loads(
        (output_dir / PREPARED_PROVENANCE_NAME).read_text(encoding="utf-8")
    )
    assert provenance["source_labels"]["sha256_before"] == source_sha
    assert provenance["source_labels"]["sha256_after"] == source_sha


def test_prepare_refuses_nonempty_output_and_locked_overlap(
    tmp_path: Path,
) -> None:
    source_path, _skeleton, _videos = _make_source_package(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="absent or empty"):
        prepare_completion_labels(
            source_labels=source_path,
            output_dir=nonempty,
        )
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "user data"

    locked = tmp_path / "locked.csv"
    locked.write_text(
        f"experiment,frame\n{EXPERIMENT},167\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"exact or \+/-166"):
        prepare_completion_labels(
            source_labels=source_path,
            output_dir=tmp_path / "blocked",
            locked_manifest=locked,
        )
    assert not (tmp_path / "blocked").exists()


def test_finalize_extracts_original_side_lfs_and_validates_completion(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    source_path, _skeleton, _videos = _make_source_package(tmp_path)
    source_sha = sha256_file(source_path)
    prepared_dir = tmp_path / "prepared"
    prepare_completion_labels(
        source_labels=source_path,
        output_dir=prepared_dir,
    )
    gui_output = tmp_path / "gui_output.pkg.slp"
    _copy_gui_output_with_completion(
        seed_path=prepared_dir / SEED_PACKAGE_NAME,
        output_path=gui_output,
        add_top_lf=True,
        tmp_path=tmp_path,
    )
    finalized_dir = tmp_path / "finalized"

    report = finalize_completion_labels(
        source_labels=source_path,
        prepared_manifest=prepared_dir / PREPARED_MANIFEST_NAME,
        gui_output=gui_output,
        output_dir=finalized_dir,
    )

    assert sha256_file(source_path) == source_sha
    assert report["selected_side_labeled_frames"] == 2
    assert report["completed_originally_missing_mouths"] == 1
    assert report["contains_only_original_selected_side_lfs"] is True
    assert report["contains_locked_test"] is False
    assert report["gui_output"][
        "ignored_nonoriginal_or_non_side_labeled_frames"
    ] == 1
    assert report["embedded_frames"] is False
    assert (finalized_dir / COMPLETED_PACKAGE_NAME).is_file()
    assert (finalized_dir / COMPLETED_MANIFEST_NAME).is_file()
    assert (finalized_dir / COMPLETED_PROVENANCE_NAME).is_file()

    completed = sio.load_slp(
        str(finalized_dir / COMPLETED_PACKAGE_NAME),
        open_videos=False,
    )
    completed_lookup = _lookup(completed)
    assert set(completed_lookup) == {
        (EXPERIMENT, "side1_cam", 1),
        (EXPERIMENT, "side2_cam", 1),
    }
    completed_points = completed_lookup[
        (EXPERIMENT, "side1_cam", 1)
    ].instances[1].numpy()
    assert np.array_equal(completed_points[0], [22.0, 13.0])


@pytest.mark.parametrize(
    ("completed_mouth", "mutate_finite_tail", "error"),
    [
        (None, False, "remains non-finite"),
        ((22.0, 13.0), True, "originally finite tail coordinate changed"),
    ],
)
def test_finalize_rejects_incomplete_or_modified_original_coordinates(
    tmp_path: Path,
    completed_mouth: tuple[float, float] | None,
    mutate_finite_tail: bool,
    error: str,
) -> None:
    source_path, _skeleton, _videos = _make_source_package(tmp_path)
    prepared_dir = tmp_path / "prepared"
    prepare_completion_labels(
        source_labels=source_path,
        output_dir=prepared_dir,
    )
    gui_output = tmp_path / "gui_output.pkg.slp"
    _copy_gui_output_with_completion(
        seed_path=prepared_dir / SEED_PACKAGE_NAME,
        output_path=gui_output,
        completed_mouth=completed_mouth,
        mutate_finite_tail=mutate_finite_tail,
        tmp_path=tmp_path,
    )
    rejected_dir = tmp_path / "rejected"

    with pytest.raises(ValueError, match=error):
        finalize_completion_labels(
            source_labels=source_path,
            prepared_manifest=prepared_dir / PREPARED_MANIFEST_NAME,
            gui_output=gui_output,
            output_dir=rejected_dir,
        )
    assert not rejected_dir.exists()
