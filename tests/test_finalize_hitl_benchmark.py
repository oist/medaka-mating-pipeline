from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio

from scripts.sleap.eval_topdown_vs_bottomup import manifest_positions
from scripts.sleap.finalize_hitl_benchmark import (
    VIEWS,
    build_evaluation_manifest_rows,
    finalize_hitl_benchmark,
    merge_hitl_labels,
)


EXPERIMENT = "20251225_K_1_1_FCMC"


def _manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "experiment": EXPERIMENT,
            "frame_idx": str(1000 + index),
            "frame_key": f"selection-{index:02d}",
            "event_id": f"event-{index:02d}",
            "event_start_frame": str(1000 + index),
            "event_end_frame": str(1000 + index),
            "difficulty_hint": "unknown_until_annotation",
            "source_category": "model_independent_uniform_time",
            "selection_source": "unit_test_selection",
            "split_role": "locked_test_candidate_pending_annotation",
        }
        for index in range(40)
    ]


def _videos() -> dict[str, sio.Video]:
    return {
        view: sio.Video(
            filename=(
                f"Z:/{EXPERIMENT}/{view}/"
                + {
                    "top_cam": "cam0_top.mp4",
                    "side1_cam": "cam1_side.mp4",
                    "side2_cam": "cam2_side.mp4",
                }[view]
            ),
            open_backend=False,
            backend_metadata={
                "shape": (5000, 64, 64, 3),
                "fps": 166.0,
                "grayscale": False,
            },
        )
        for view in VIEWS
    }


def _instance(
    skeleton: sio.Skeleton,
    *,
    offset: float,
    track: sio.Track | None = None,
    partial: bool = False,
    hidden_tail: bool = False,
) -> sio.Instance:
    mouth = [10.0 + offset, 11.0 + offset, 1.0, 0.0]
    tail = (
        [np.nan, np.nan, 0.0, 0.0]
        if partial
        else [20.0 + offset, 21.0 + offset, not hidden_tail, 0.0]
    )
    return sio.Instance.from_numpy(
        np.asarray([mouth, tail], dtype=float),
        skeleton=skeleton,
        track=track,
    )


def _labels(
    *,
    skeleton: sio.Skeleton,
    videos: dict[str, sio.Video],
    replacement_key: tuple[int, str] | None = None,
    replacement_instances: int = 3,
    partial: bool = False,
    predicted: bool = False,
    include_extra: bool = False,
) -> sio.Labels:
    frames: list[sio.LabeledFrame] = []
    tracks: list[sio.Track] = []
    rows = _manifest_rows()
    if replacement_key is not None:
        rows = [
            row
            for row in rows
            if (int(row["frame_idx"]), replacement_key[1])
            == replacement_key
        ]
    for row in rows:
        frame_idx = int(row["frame_idx"])
        views = VIEWS if replacement_key is None else (replacement_key[1],)
        for view in views:
            instances = []
            for fish_index in range(
                replacement_instances if replacement_key is not None else 3
            ):
                track = sio.Track(name=f"source-track-{frame_idx}-{view}-{fish_index}")
                tracks.append(track)
                if predicted:
                    points = np.asarray(
                        [
                            [30 + fish_index, 31 + fish_index],
                            [40 + fish_index, 41 + fish_index],
                        ],
                        dtype=float,
                    )
                    instance = sio.PredictedInstance.from_numpy(
                        points,
                        skeleton=skeleton,
                        score=0.9,
                        track=track,
                    )
                else:
                    instance = _instance(
                        skeleton,
                        offset=(
                            100.0
                            if replacement_key is not None
                            else float(fish_index)
                        ),
                        track=track,
                        partial=partial and fish_index == 0,
                        hidden_tail=partial and fish_index == 1,
                    )
                instances.append(instance)
            frames.append(
                sio.LabeledFrame(
                    video=videos[view],
                    frame_idx=frame_idx,
                    instances=instances,
                )
            )
    if include_extra:
        frames.append(
            sio.LabeledFrame(
                video=videos["top_cam"],
                frame_idx=9999,
                instances=[
                    _instance(skeleton, offset=float(index))
                    for index in range(3)
                ],
            )
        )
    return sio.Labels(
        labeled_frames=frames,
        videos=list(videos.values()),
        skeletons=[skeleton],
        tracks=tracks,
        provenance={
            "prediction_candidate_review": {
                "candidate_artifact": {"sha256": "a" * 64},
            }
        },
    )


def _merge(
    base: sio.Labels,
    replacement: sio.Labels,
):
    return merge_hitl_labels(
        base,
        replacement,
        _manifest_rows(),
        output_provenance={
            "model_assisted_annotation": True,
            "contains_model_predictions": False,
            "contains_keypoint_coordinates": True,
        },
        sio=sio,
    )


def test_merge_replaces_exact_key_and_drops_extra_base_frames() -> None:
    videos = _videos()
    base_skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
        name="medaka_mouth_tail",
    )
    replacement_skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
        name="medaka_mouth_tail",
    )
    base = _labels(
        skeleton=base_skeleton,
        videos=videos,
        include_extra=True,
    )
    replacement = _labels(
        skeleton=replacement_skeleton,
        videos=videos,
        replacement_key=(1005, "side1_cam"),
    )

    merged, ledger, report = _merge(base, replacement)

    assert len(merged.labeled_frames) == 120
    assert len(ledger) == 360
    assert report["counts"]["replacement_view_frames"] == 1
    assert report["counts"]["base_view_frames_dropped"] == 1
    replaced = merged.labeled_frames[5 * 3 + 1]
    assert replaced.frame_idx == 1005
    assert replaced.instances[0].numpy()[0, 0] == pytest.approx(110.0)
    assert ledger[5 * 9 + 3]["source_kind"] == "replacement"


def test_merge_canonicalizes_skeletons_and_strips_tracks() -> None:
    videos = _videos()
    base = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
            name="medaka_mouth_tail",
        ),
        videos=videos,
    )
    replacement = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
            name="medaka_mouth_tail",
        ),
        videos=videos,
        replacement_key=(1005, "side1_cam"),
    )

    merged, _ledger, _report = _merge(base, replacement)

    assert len(merged.skeletons) == 1
    assert merged.tracks == []
    assert {
        id(instance.skeleton)
        for frame in merged.labeled_frames
        for instance in frame.instances
    } == {id(merged.skeletons[0])}
    assert all(
        type(instance).__name__ == "Instance" and instance.track is None
        for frame in merged.labeled_frames
        for instance in frame.instances
    )


def test_merge_allows_partial_and_hidden_nodes_but_flags_them() -> None:
    videos = _videos()
    base = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
        ),
        videos=videos,
    )
    replacement = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
        ),
        videos=videos,
        replacement_key=(1005, "side1_cam"),
        partial=True,
    )

    merged, ledger, report = _merge(base, replacement)

    assert len(merged.labeled_frames) == 120
    assert report["counts"]["partial_view_frames"] == 1
    replacement_rows = [
        row for row in ledger if row["source_kind"] == "replacement"
    ]
    assert replacement_rows[0]["tail_finite"] is False
    assert replacement_rows[0]["partial_annotation"] is True
    assert replacement_rows[1]["tail_finite"] is True
    assert replacement_rows[1]["tail_visible"] is False


def test_merge_rejects_invalid_instance_count_and_predictions() -> None:
    videos = _videos()
    skeleton = sio.Skeleton(
        nodes=["mouth", "tail"],
        edges=[("mouth", "tail")],
    )
    base = _labels(skeleton=skeleton, videos=videos)
    short = _labels(
        skeleton=skeleton,
        videos=videos,
        replacement_key=(1005, "side1_cam"),
        replacement_instances=2,
    )
    with pytest.raises(ValueError, match="exactly 3 instances"):
        _merge(base, short)

    predicted = _labels(
        skeleton=skeleton,
        videos=videos,
        replacement_key=(1005, "side1_cam"),
        predicted=True,
    )
    with pytest.raises(ValueError, match="PredictedInstance"):
        _merge(base, predicted)


def test_evaluation_manifest_positions_match_locked_label_order(
    tmp_path: Path,
) -> None:
    videos = _videos()
    base = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
        ),
        videos=videos,
    )
    replacement = _labels(
        skeleton=sio.Skeleton(
            nodes=["mouth", "tail"],
            edges=[("mouth", "tail")],
        ),
        videos=videos,
        replacement_key=(1005, "side1_cam"),
        partial=True,
    )
    merged, _ledger, report = _merge(base, replacement)
    columns, rows = build_evaluation_manifest_rows(
        merged,
        report["frame_summaries"],
        locked_output_path=tmp_path / "locked.pkg.slp",
        package_size_bytes=123,
        package_mtime_ns=456,
    )

    csv_rows = [
        {key: str(value) if value is not None else "" for key, value in row.items()}
        for row in rows
    ]
    positions, resolved = manifest_positions(merged, csv_rows)
    assert positions == list(range(120))
    assert len(resolved) == 120
    assert len({row["frame_key"] for row in rows}) == 120
    partial = rows[5 * 3 + 1]
    assert partial["partial_annotation"] is True
    assert partial["difficulty"] == "partial_or_occluded"
    assert partial["needs_manual_review"] is True
    assert "selection_source_category" in columns
    assert partial["selection_source_category"] == (
        "model_independent_uniform_time"
    )


def test_finalizer_refuses_existing_outputs_before_loading_inputs(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.slp"
    replacement = tmp_path / "replacement.slp"
    manifest = tmp_path / "manifest.csv"
    base.write_bytes(b"base")
    replacement.write_bytes(b"replacement")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("experiment", "frame_idx", "frame_key"),
        )
        writer.writeheader()
        for index in range(40):
            writer.writerow(
                {
                    "experiment": EXPERIMENT,
                    "frame_idx": 1000 + index,
                    "frame_key": f"selection-{index:02d}",
                }
            )
    output_dir = tmp_path / "final"
    output_dir.mkdir()
    (output_dir / "cohort_locked.pkg.slp").write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_hitl_benchmark(
            base_slp=base,
            replacement_slp=replacement,
            manifest_path=manifest,
            output_dir=output_dir,
            prefix="cohort",
        )
    assert (output_dir / "cohort_locked.pkg.slp").read_bytes() == (
        b"do-not-overwrite"
    )
