from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sam2_id_experiment.src.config import load_config
from sam2_id_experiment.src.id_dataset.pair_sampler import sample_positive_negative_pairs
from sam2_id_experiment.src.sam2_adapter.mask_io import (
    MaskInstance,
    read_mask_instances_h5,
    write_mask_instances_h5,
)
from sam2_id_experiment.src.sam2_adapter.sam2_runner import run_sam2_video_segmentation
from sam2_id_experiment.src.sam2_adapter.sam2_runner import (
    _nearest_sampled_frame,
    _resize_bool_mask,
    _resize_frame_for_sam2,
    _sampled_frames,
)
from sam2_id_experiment.src.sam2_adapter.prompt_frame import resolve_prompt_frame_index
from sam2_id_experiment.src.sam2_adapter.prompt_seed import seed_prompts_from_mask_h5
from sam2_id_experiment.src.tracklets.filter_rules import classify_tracklets
from sam2_id_experiment.src.tracklets.mask_to_tracklet import masks_to_instance_dataframe
from sam2_id_experiment.src.tracklets.qc_metrics import compute_mask_qc_frames


pytest.importorskip("h5py")


def _rect_mask(x0: int, y0: int, x1: int, y1: int, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _mask_row(
    *,
    frame: int,
    tracklet_id: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    confidence: float = 0.95,
    review_label: str = "",
) -> dict:
    contour = [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]]
    return {
        "tracklet_id": tracklet_id,
        "sam2_track_id": tracklet_id,
        "instance_id": tracklet_id,
        "frame": frame,
        "centroid_x": (x0 + x1 - 1) / 2.0,
        "centroid_y": (y0 + y1 - 1) / 2.0,
        "bbox_x0": x0,
        "bbox_y0": y0,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "area": (x1 - x0) * (y1 - y0),
        "confidence": confidence,
        "contour_json": json.dumps(contour),
        "review_label": review_label,
    }


def test_config_resolves_experiment_view_video_path(tmp_path):
    paths_config = tmp_path / "paths.local.toml"
    paths_config.write_text(
        "[paths]\nmedaka_data_dir = \"Z:/StephensU/medaka/medaka_202512\"\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "sam2_test.yaml"
    cfg_path.write_text(
        "\n".join([
            'experiment_name: "20251225_K_1_2_FCMC"',
            'view: "top_cam"',
            f'paths_config: "{paths_config.as_posix()}"',
            'output_dir: "outputs/{experiment_name}/{view}"',
            "n_animals_expected: 3",
        ]),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.experiment_name == "20251225_K_1_2_FCMC"
    assert cfg.view == "top_cam"
    assert cfg.video_path.as_posix().endswith(
        "20251225_K_1_2_FCMC/top_cam/cam0_top.mp4"
    )
    assert cfg.output_dir.as_posix() == "outputs/20251225_K_1_2_FCMC/top_cam"


def test_prompt_frame_index_resolution():
    assert resolve_prompt_frame_index(start_frame=171300, frame=None, local_frame=None) == 171300
    assert resolve_prompt_frame_index(start_frame=171300, frame=None, local_frame=5) == 171305
    assert resolve_prompt_frame_index(start_frame=171300, frame=171400, local_frame=None) == 171400
    with pytest.raises(ValueError):
        resolve_prompt_frame_index(start_frame=171300, frame=171400, local_frame=5)


def test_mask_h5_roundtrip_three_frames_two_masks(tmp_path):
    instances = []
    for frame in range(3):
        for track_id in range(2):
            mask = _rect_mask(2 + frame + 10 * track_id, 4, 7 + frame + 10 * track_id, 9)
            instances.append(MaskInstance(
                frame=frame,
                instance_id=track_id,
                sam2_track_id=track_id,
                mask=mask,
                bbox=(2 + frame + 10 * track_id, 4, 7 + frame + 10 * track_id, 9),
                score=0.95,
            ))

    path = tmp_path / "sam2_masks.h5"
    write_mask_instances_h5(path, instances, image_shape=(32, 32))
    loaded = read_mask_instances_h5(path)

    assert len(loaded) == 6
    assert loaded[0].frame == 0
    assert loaded[-1].sam2_track_id == 1
    assert loaded[0].mask.dtype == bool
    assert int(loaded[0].mask.sum()) == 25


def test_seed_prompts_from_previous_masks_uses_nearest_frame(tmp_path):
    path = tmp_path / "masks.h5"
    write_mask_instances_h5(
        path,
        [
            MaskInstance(100, 0, 0, _rect_mask(2, 4, 8, 10), (2, 4, 8, 10), 1.0),
            MaskInstance(110, 0, 1, _rect_mask(20, 22, 28, 30), (20, 22, 28, 30), 1.0),
        ],
        image_shape=(32, 32),
    )

    prompts, selected_frame = seed_prompts_from_mask_h5(
        path,
        source_frame=108,
        target_local_frame=0,
        min_area=5,
        points_per_mask=3,
    )

    assert selected_frame == 110
    assert [prompt["obj_id"] for prompt in prompts] == [1]
    assert prompts[0]["frame"] == 0
    assert len(prompts[0]["points"]) >= 1
    assert prompts[0]["labels"] == [1] * len(prompts[0]["points"])


def test_masks_to_tracklet_dataframe_computes_geometry(tmp_path):
    instances = [
        MaskInstance(0, 0, 10, _rect_mask(1, 2, 5, 6), (1, 2, 5, 6), 0.91),
        MaskInstance(1, 0, 10, _rect_mask(2, 2, 6, 6), (2, 2, 6, 6), 0.92),
        MaskInstance(0, 1, 20, _rect_mask(20, 2, 25, 7), (20, 2, 25, 7), 0.93),
    ]
    path = tmp_path / "masks.h5"
    write_mask_instances_h5(path, instances)

    df = masks_to_instance_dataframe(path)

    assert set(df["tracklet_id"]) == {10, 20}
    first = df[(df["frame"] == 0) & (df["tracklet_id"] == 10)].iloc[0]
    assert first["area"] == 16
    assert first["centroid_x"] == pytest.approx(2.5)
    assert first["centroid_y"] == pytest.approx(3.5)


def test_qc_marks_crossing_for_review_and_clean_far_tracklets():
    rows = []
    for frame in range(12):
        rows.append({
            "tracklet_id": 1,
            "sam2_track_id": 1,
            "instance_id": 0,
            "frame": frame,
            "centroid_x": float(frame),
            "centroid_y": 0.0,
            "bbox_x0": frame,
            "bbox_y0": 0,
            "bbox_x1": frame + 5,
            "bbox_y1": 5,
            "area": 25,
            "confidence": 0.95,
            "contour_json": "[]",
        })
        rows.append({
            "tracklet_id": 2,
            "sam2_track_id": 2,
            "instance_id": 1,
            "frame": frame,
            "centroid_x": 100.0 - frame,
            "centroid_y": 0.0,
            "bbox_x0": 100 - frame,
            "bbox_y0": 0,
            "bbox_x1": 105 - frame,
            "bbox_y1": 5,
            "area": 25,
            "confidence": 0.96,
            "contour_json": "[]",
        })
    df = pd.DataFrame(rows)
    _per_frame, summary = classify_tracklets(df, {
        "min_tracklet_length_frames": 10,
        "min_mask_area": 5,
        "max_mask_area": 100,
        "max_area_cv": 0.3,
        "max_centroid_speed_px_per_frame": 10,
        "min_confidence": 0.8,
        "crossing_distance_px": 200,
        "reject_near_crossing": True,
    })

    assert set(summary["qc_status"]) == {"manual_review"}
    assert summary["qc_reason"].str.contains("near_crossing").all()


def test_mask_qc_stable_three_objects_has_no_flags():
    rows = []
    for frame in range(3):
        rows.extend([
            _mask_row(frame=frame, tracklet_id=0, x0=10 + frame, y0=10, x1=20 + frame, y1=20),
            _mask_row(frame=frame, tracklet_id=1, x0=40, y0=10 + frame, x1=50, y1=20 + frame),
            _mask_row(frame=frame, tracklet_id=2, x0=10, y0=40, x1=20, y1=50),
        ])
    qc = compute_mask_qc_frames(
        pd.DataFrame(rows),
        qc_config={"expected_object_count": 3, "crossing_distance_px": 10},
        image_shape=(64, 64),
    )

    assert not qc["auto_qc_flag"].any()
    assert qc["expected_object_count_match"].all()


def test_mask_qc_flags_edge_area_jump_overlap_and_missing_count():
    rows = [
        _mask_row(frame=0, tracklet_id=0, x0=0, y0=10, x1=10, y1=20),  # edge
        _mask_row(frame=0, tracklet_id=1, x0=30, y0=10, x1=40, y1=20),
        _mask_row(frame=0, tracklet_id=2, x0=10, y0=40, x1=20, y1=50),
        _mask_row(frame=1, tracklet_id=0, x0=10, y0=10, x1=30, y1=30),  # area jump
        _mask_row(frame=1, tracklet_id=1, x0=20, y0=20, x1=35, y1=35),  # overlap with tracklet 0
        # tracklet 2 missing in frame 1 -> object_count_mismatch for that frame
    ]
    qc = compute_mask_qc_frames(
        pd.DataFrame(rows),
        qc_config={
            "expected_object_count": 3,
            "edge_margin_px": 5,
            "max_area_ratio_change": 2.0,
            "max_overlap_fraction": 0.01,
        },
        image_shape=(64, 64),
    )

    reasons = ";".join(qc["auto_qc_reason"].fillna("").astype(str))
    assert "bbox_touches_image_edge" in reasons
    assert "area_ratio_change" in reasons
    assert "mask_overlap" in reasons
    assert "object_count_mismatch" in reasons


def test_qc_flags_make_tracklet_manual_review_not_clean_train():
    rows = []
    for frame in range(10):
        rows.append(_mask_row(frame=frame, tracklet_id=0, x0=0 if frame == 5 else 10, y0=10, x1=20, y1=20))
        rows.append(_mask_row(frame=frame, tracklet_id=1, x0=35, y0=10, x1=45, y1=20))
        rows.append(_mask_row(frame=frame, tracklet_id=2, x0=10, y0=35, x1=20, y1=45))

    per_frame, summary = classify_tracklets(
        pd.DataFrame(rows),
        {
            "min_tracklet_length_frames": 10,
            "min_mask_area": 5,
            "max_mask_area": 1000,
            "max_area_cv": 1.0,
            "max_centroid_speed_px_per_frame": 100,
            "min_confidence": 0.8,
            "expected_object_count": 3,
            "edge_margin_px": 5,
        },
        image_shape=(64, 64),
    )

    flagged = summary[summary["tracklet_id"] == 0].iloc[0]
    assert flagged["qc_status"] == "manual_review"
    assert "auto_bbox_touches_image_edge" in flagged["qc_reason"]
    assert per_frame[per_frame["tracklet_id"] == 0]["auto_qc_flag"].any()


def test_human_review_labels_override_conservatively():
    good_rows = []
    bad_rows = []
    for frame in range(10):
        good_rows.append(_mask_row(frame=frame, tracklet_id=0, x0=10, y0=10, x1=20, y1=20, review_label="good"))
        bad_rows.append(_mask_row(frame=frame, tracklet_id=1, x0=35, y0=10, x1=45, y1=20, review_label="merged"))
        good_rows.append(_mask_row(frame=frame, tracklet_id=2, x0=10, y0=35, x1=20, y1=45, review_label="good"))
    df = pd.DataFrame(good_rows + bad_rows)
    _per_frame, summary = classify_tracklets(
        df,
        {
            "min_tracklet_length_frames": 10,
            "min_mask_area": 5,
            "max_mask_area": 1000,
            "max_area_cv": 1.0,
            "max_centroid_speed_px_per_frame": 100,
            "min_confidence": 0.8,
            "max_missing_frames": 100,
            "expected_object_count": 3,
            "crossing_distance_px": 5,
        },
        image_shape=(64, 64),
    )

    statuses = dict(zip(summary["tracklet_id"], summary["qc_status"]))
    assert statuses[0] == "clean_train"
    assert statuses[1] == "reject"


def test_qc_segments_respect_sampled_frame_stride():
    rows = []
    for frame in [100, 110, 120, 140]:
        rows.append(_mask_row(frame=frame, tracklet_id=0, x0=10, y0=10, x1=20, y1=20))
        rows.append(_mask_row(frame=frame, tracklet_id=1, x0=35, y0=10, x1=45, y1=20))
        rows.append(_mask_row(frame=frame, tracklet_id=2, x0=10, y0=35, x1=20, y1=45))

    per_frame, _summary = classify_tracklets(
        pd.DataFrame(rows),
        {
            "min_tracklet_length_frames": 4,
            "min_mask_area": 5,
            "max_mask_area": 1000,
            "max_area_cv": 1.0,
            "max_centroid_speed_px_per_frame": 100,
            "min_confidence": 0.8,
            "max_missing_frames": 100,
            "expected_object_count": 3,
            "crossing_distance_px": 5,
        },
        image_shape=(64, 64),
    )

    track0 = per_frame[per_frame["tracklet_id"] == 0].sort_values("frame")
    assert track0.loc[track0["frame"].isin([100, 110, 120]), "qc_segment_id"].nunique() == 1
    assert track0.loc[track0["frame"] == 140, "qc_segment_id"].iloc[0] != track0.loc[
        track0["frame"] == 120, "qc_segment_id"
    ].iloc[0]


def test_pair_sampler_uses_same_tracklet_positive_and_overlapping_negative():
    meta = pd.DataFrame({
        "crop_id": [0, 1, 2, 3, 4, 5],
        "frame": [0, 1, 2, 1, 2, 3],
        "tracklet_id": [1, 1, 1, 2, 2, 2],
        "qc_status": ["clean_train"] * 6,
    })
    pairs = sample_positive_negative_pairs(meta, {
        "positive_pairs_per_tracklet": 2,
        "negative_pairs_per_tracklet_pair": 3,
        "random_seed": 1,
    })

    assert {0, 1}.issubset(set(pairs["label"]))
    pos = pairs[pairs["label"] == 1]
    neg = pairs[pairs["label"] == 0]
    assert pos["reason"].str.startswith("same_clean_").all()
    assert neg["reason"].str.startswith("coexisting_clean_tracklets").all()


def test_opencv_bgsub_runner_writes_mask_h5(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "moving_fish.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 48),
    )
    for frame in range(8):
        img = np.full((48, 64, 3), 180, dtype=np.uint8)
        cv2.ellipse(img, (12 + frame * 4, 24), (5, 3), 0, 0, 360, (40, 40, 40), -1)
        writer.write(img)
    writer.release()

    output_path = tmp_path / "masks.h5"
    result = run_sam2_video_segmentation(
        video_path=video_path,
        output_path=output_path,
        config={
            "runner": "opencv_bgsub",
            "start_frame": 0,
            "end_frame": 7,
            "n_animals_expected": 1,
            "opencv_bgsub": {
                "threshold": 20,
                "open_px": 1,
                "close_px": 1,
                "min_area": 5,
                "max_area": 200,
                "background_samples": 2,
                "max_track_distance_px": 100,
            },
        },
    )

    loaded = read_mask_instances_h5(output_path)
    assert result["runner"] == "opencv_bgsub"
    assert output_path.exists()
    assert len(loaded) >= 1
    assert min(inst.sam2_track_id for inst in loaded) == 0


def test_sam2_resize_helpers_preserve_expected_shapes():
    frame = np.zeros((1200, 1000, 3), dtype=np.uint8)
    resized = _resize_frame_for_sam2(frame, max_side_px=600)
    assert resized.shape[:2] == (600, 500)

    mask = np.zeros((600, 500), dtype=bool)
    mask[10:20, 30:40] = True
    up = _resize_bool_mask(mask, (1200, 1000))
    assert up.shape == (1200, 1000)
    assert up.dtype == bool
    assert up.sum() > mask.sum()


def test_sam2_sampled_frames_include_end_and_nearest_prompt_frame():
    frames = _sampled_frames(100, 135, 10)
    assert frames == [100, 110, 120, 130, 135]
    assert _nearest_sampled_frame(121, frames) == 120
    assert _nearest_sampled_frame(134, frames) == 135
