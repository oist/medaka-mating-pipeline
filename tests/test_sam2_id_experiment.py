from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reconstruction_3d.bind_topcam_id import _top_label_lookup, bind
from scripts.reconstruction_3d.load_2d_slp import _instance_point as _reconstruction_instance_point
from sam2_id_experiment.src.config import load_config
from sam2_id_experiment.src.detectron2_adapter.coco_merge import merge_coco_exports
from sam2_id_experiment.src.detectron2_adapter.coco_export import mask_to_coco_polygons
from sam2_id_experiment.src.evaluation.backend_compare import compare_mask_files, mask_file_summary, mask_iou
from sam2_id_experiment.src.id_model.assign import (
    cosine_distance_matrix,
    nearest_reference_assignment,
    one_to_one_min_distance_assignment,
)
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
from sam2_id_experiment.src.sleap_adapter.mask_gate import match_masks_to_sleap_points
from sam2_id_experiment.src.sleap_adapter.body_metrics import compute_body_metrics_for_tracks
from sam2_id_experiment.src.sleap_adapter.points import _instance_point as _sleap_instance_point
from sam2_id_experiment.src.sleap_adapter.top_track_export import export_segmentation_top_tracks
from sam2_id_experiment.src.tracklets.filter_rules import classify_tracklets
from sam2_id_experiment.src.tracklets.link_detections import link_mask_instances
from sam2_id_experiment.src.tracklets.mask_to_tracklet import masks_to_instance_dataframe, summarize_tracklets
from sam2_id_experiment.src.tracklets.qc_metrics import compute_mask_qc_frames
from sam2_id_experiment.src.tracklets.stitch_fragments import stitch_identity_fragments
from sam2_id_experiment.src.tracking.hybrid_sleap_segmentation import track_sleap_with_segmentation_identity


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


def _identity_tracklet_row(*, tracklet_id: int, frame: int, x: float, y: float, status: str = "clean_train") -> dict:
    return {
        "tracklet_id": tracklet_id,
        "sam2_track_id": tracklet_id,
        "instance_id": 0,
        "frame": frame,
        "centroid_x": x,
        "centroid_y": y,
        "bbox_x0": int(x),
        "bbox_y0": int(y),
        "bbox_x1": int(x) + 5,
        "bbox_y1": int(y) + 5,
        "area": 25,
        "confidence": 0.99,
        "qc_status": status,
        "qc_reason": "passed_clean_frame_rules",
    }


def _assignment_row(tracklet_id: int, identity: str) -> dict:
    return {
        "tracklet_id": tracklet_id,
        "assigned_identity": identity,
        "assignment_status": "assigned",
        "match_distance": 0.01,
        "match_margin": 0.5,
        "reference_tracklet_id": 0,
    }


def _sleap_point(frame: int, idx: int, x: float, y: float = 0.0, score: float = 0.9) -> dict:
    return {
        "frame": frame,
        "sleap_index": idx,
        "x": x,
        "y": y,
        "score": score,
        "node": "mouth",
    }


def _seg_anchor(frame: int, label: str, x: float, y: float = 0.0, tracklet_id: int = 0) -> dict:
    return {
        "frame": frame,
        "centroid_x": x,
        "centroid_y": y,
        "assigned_identity": label,
        "identity_trajectory_id": f"{label}_seg0",
        "tracklet_id": tracklet_id,
        "qc_status": "clean_train",
    }


class _IdentityProjectionCalibration:
    backend_name = "test_projection"
    provenance = {"source": "unit_test"}

    def require_capability(self, capability: str) -> None:
        assert capability == "projection"

    def project_world(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return np.stack(
            [xyz[:, :2], xyz[:, [0, 2]], xyz[:, [1, 2]]], axis=1
        )

    def project_world_to_view(self, xyz: np.ndarray, view: str) -> np.ndarray:
        assert view == "top_cam"
        return np.asarray(xyz, dtype=float)[:, :2]


class _FakeSleapInstance:
    score = 0.8

    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def numpy(self):
        return self._points


def test_sleap_point_loader_uses_mouth_anchor_when_requested():
    inst = _FakeSleapInstance([[10, 20], [30, 40]])

    point = _sleap_instance_point(inst, node_names=["mouth", "tail"], anchor_node="mouth")

    assert point == (10.0, 20.0, "mouth")


def test_sleap_point_loader_centroid_is_explicit_experimental_mode():
    inst = _FakeSleapInstance([[10, 20], [30, 40]])

    point = _sleap_instance_point(inst, node_names=["mouth", "tail"], anchor_node=None)

    assert point == (20.0, 30.0, "centroid")


def test_reconstruction_loader_uses_mouth_anchor_when_requested():
    inst = _FakeSleapInstance([[10, 20], [30, 40]])

    x, y, score = _reconstruction_instance_point(
        inst,
        node_names=["mouth", "tail"],
        anchor_node="mouth",
    )

    assert (x, y) == (10.0, 20.0)
    assert score == pytest.approx(0.8)


def test_reconstruction_loader_centroid_is_explicit_experimental_mode():
    inst = _FakeSleapInstance([[10, 20], [30, 40]])

    x, y, _score = _reconstruction_instance_point(
        inst,
        node_names=["mouth", "tail"],
        anchor_node=None,
    )

    assert (x, y) == (20.0, 30.0)


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


def test_seed_prompts_include_other_masks_as_negative_points(tmp_path):
    path = tmp_path / "masks.h5"
    write_mask_instances_h5(
        path,
        [
            MaskInstance(100, 0, 0, _rect_mask(2, 4, 8, 10), (2, 4, 8, 10), 1.0),
            MaskInstance(100, 1, 1, _rect_mask(20, 22, 28, 30), (20, 22, 28, 30), 1.0),
        ],
        image_shape=(32, 32),
    )

    prompts, selected_frame = seed_prompts_from_mask_h5(
        path,
        source_frame=100,
        target_local_frame=0,
        min_area=5,
        points_per_mask=2,
        negative_points_per_other_mask=1,
    )

    assert selected_frame == 100
    assert len(prompts) == 2
    assert all(prompt["labels"].count(0) == 1 for prompt in prompts)
    assert all(prompt["labels"].count(1) >= 1 for prompt in prompts)


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


def test_tracklet_summary_missing_frames_respects_stride():
    rows = [_mask_row(frame=frame, tracklet_id=0, x0=10, y0=10, x1=20, y1=20) for frame in [100, 110, 120, 140]]
    summary = summarize_tracklets(pd.DataFrame(rows))
    row = summary.iloc[0]
    assert row["frame_step"] == 10
    assert row["n_missing_frames"] == 1


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
    per_frame, summary = classify_tracklets(
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
    row_statuses = {
        tracklet_id: set(group["qc_status"])
        for tracklet_id, group in per_frame.groupby("tracklet_id")
    }
    assert row_statuses[1] == {"reject"}
    assert "human_review_merged" in set(per_frame[per_frame["tracklet_id"] == 1]["qc_reason"])


def test_human_review_good_is_frame_local_auto_qc_override():
    rows = []
    for frame in range(10):
        label = "merged" if frame == 5 else "good"
        rows.append(_mask_row(frame=frame, tracklet_id=0, x0=0 if frame == 4 else 10, y0=10, x1=20, y1=20, review_label=label))
        rows.append(_mask_row(frame=frame, tracklet_id=1, x0=35, y0=10, x1=45, y1=20, review_label="good"))
        rows.append(_mask_row(frame=frame, tracklet_id=2, x0=10, y0=35, x1=20, y1=45, review_label="good"))

    per_frame, _summary = classify_tracklets(
        pd.DataFrame(rows),
        {
            "min_tracklet_length_frames": 10,
            "min_mask_area": 5,
            "max_mask_area": 1000,
            "max_area_cv": 1.0,
            "max_centroid_speed_px_per_frame": 100,
            "min_confidence": 0.8,
            "max_missing_frames": 100,
            "expected_object_count": 3,
            "edge_margin_px": 5,
            "crossing_distance_px": 5,
        },
        image_shape=(64, 64),
    )

    reviewed = per_frame[per_frame["tracklet_id"] == 0].sort_values("frame")
    assert reviewed[reviewed["frame"] == 5]["qc_status"].iloc[0] == "reject"
    assert reviewed[reviewed["frame"] == 5]["qc_reason"].iloc[0] == "human_review_merged"
    assert set(reviewed[reviewed["frame"] != 5]["qc_status"]) == {"clean_train"}
    assert set(reviewed[reviewed["frame"] != 5]["qc_reason"]) == {"human_review_good"}


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


def test_cross_chunk_assignment_prefers_nearest_one_to_one_match():
    reference = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    target = np.asarray([
        [0.99, 0.05, 0.0],
        [0.0, 0.99, 0.01],
        [0.01, 0.0, 0.99],
    ])
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    distance = cosine_distance_matrix(target, reference)
    assignment = one_to_one_min_distance_assignment(distance)

    assert assignment == {0: 0, 1: 1, 2: 2}
    assert distance[0, 0] < distance[0, 1]
    assert distance[2, 2] < distance[2, 0]


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


def test_coco_polygon_export_from_mask_has_valid_coordinates():
    mask = _rect_mask(5, 7, 14, 18, shape=(32, 32))
    polygons = mask_to_coco_polygons(mask)

    assert polygons
    assert all(len(poly) >= 6 and len(poly) % 2 == 0 for poly in polygons)


def test_backend_mask_comparison_reports_iou(tmp_path):
    ref_path = tmp_path / "ref.h5"
    cand_path = tmp_path / "cand.h5"
    ref_mask = _rect_mask(5, 5, 15, 15)
    cand_mask = _rect_mask(7, 5, 17, 15)
    write_mask_instances_h5(
        ref_path,
        [MaskInstance(0, 0, 0, ref_mask, (5, 5, 15, 15), 1.0)],
        image_shape=(32, 32),
    )
    write_mask_instances_h5(
        cand_path,
        [MaskInstance(0, 0, 0, cand_mask, (7, 5, 17, 15), 1.0)],
        image_shape=(32, 32),
    )

    report = compare_mask_files(ref_path, cand_path)

    assert report["n_shared_frames"] == 1
    assert report["mean_best_iou"] == pytest.approx(mask_iou(ref_mask, cand_mask))


def test_mask_file_summary_reports_counts_and_overlap(tmp_path):
    path = tmp_path / "masks.h5"
    write_mask_instances_h5(
        path,
        [
            MaskInstance(0, 0, 0, _rect_mask(0, 0, 10, 10), (0, 0, 10, 10), 1.0),
            MaskInstance(0, 1, 1, _rect_mask(5, 0, 15, 10), (5, 0, 15, 10), 1.0),
            MaskInstance(1, 0, 0, _rect_mask(20, 20, 30, 30), (20, 20, 30, 30), 1.0),
        ],
        image_shape=(32, 32),
    )

    summary = mask_file_summary(path)

    assert summary["instance_count_distribution"] == {"1": 1, "2": 1}
    assert summary["frames_with_overlap_fraction_gt_0_05"] == 1
    assert summary["mean_max_overlap_fraction"] == pytest.approx(0.25)


def test_link_mask_instances_keeps_tracklets_when_detection_order_swaps():
    instances = [
        MaskInstance(0, 0, 0, _rect_mask(0, 0, 6, 6), (0, 0, 6, 6), 0.9),
        MaskInstance(0, 1, 1, _rect_mask(20, 20, 26, 26), (20, 20, 26, 26), 0.9),
        MaskInstance(10, 0, 0, _rect_mask(21, 20, 27, 26), (21, 20, 27, 26), 0.9),
        MaskInstance(10, 1, 1, _rect_mask(1, 0, 7, 6), (1, 0, 7, 6), 0.9),
    ]

    linked = link_mask_instances(instances, max_frame_gap=10, max_centroid_distance_px=20, min_iou=0.01)
    ids_by_frame_and_instance = {
        (inst.frame, inst.instance_id): inst.sam2_track_id
        for inst in linked
    }

    assert ids_by_frame_and_instance[(0, 0)] == ids_by_frame_and_instance[(10, 1)]
    assert ids_by_frame_and_instance[(0, 1)] == ids_by_frame_and_instance[(10, 0)]
    assert len({inst.sam2_track_id for inst in linked}) == 2


def test_sleap_gate_keeps_mask_containing_point():
    instances = [MaskInstance(100, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9)]
    points = pd.DataFrame([{"frame": 100, "sleap_index": 0, "x": 10, "y": 10, "score": 0.9}])

    table = match_masks_to_sleap_points(instances, points)

    assert table.loc[0, "coincidence_status"] == "keep"
    assert table.loc[0, "point_inside_mask"]
    assert table.loc[0, "coincidence_reason"] == "point_inside_mask"


def test_sleap_gate_keeps_nearby_point_outside_mask():
    instances = [MaskInstance(100, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9)]
    points = pd.DataFrame([{"frame": 100, "sleap_index": 0, "x": 17, "y": 10, "score": 0.9}])

    table = match_masks_to_sleap_points(instances, points, {"max_distance_to_mask_px": 3})

    assert table.loc[0, "coincidence_status"] == "keep"
    assert not table.loc[0, "point_inside_mask"]
    assert table.loc[0, "distance_to_mask_px"] == pytest.approx(3.0)
    assert table.loc[0, "coincidence_reason"] == "near_mask"


def test_sleap_gate_rejects_mask_without_nearby_point():
    instances = [MaskInstance(100, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9)]
    points = pd.DataFrame([{"frame": 100, "sleap_index": 0, "x": 30, "y": 30, "score": 0.9}])

    table = match_masks_to_sleap_points(instances, points, {"max_distance_to_mask_px": 3})

    assert table.loc[0, "coincidence_status"] == "reject"
    assert table.loc[0, "coincidence_reason"] == "no_sleap_match"


def test_sleap_gate_marks_two_masks_matching_one_point_manual_review():
    instances = [
        MaskInstance(100, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9),
        MaskInstance(100, 1, 1, _rect_mask(8, 5, 18, 15), (8, 5, 18, 15), 0.9),
    ]
    points = pd.DataFrame([{"frame": 100, "sleap_index": 0, "x": 10, "y": 10, "score": 0.9}])

    table = match_masks_to_sleap_points(instances, points)

    assert set(table["coincidence_status"]) == {"manual_review"}
    assert set(table["coincidence_reason"]) == {"multiple_masks_for_sleap_point"}


def test_sleap_gate_marks_one_mask_with_two_points_manual_review():
    instances = [MaskInstance(100, 0, 0, _rect_mask(5, 5, 20, 20), (5, 5, 20, 20), 0.9)]
    points = pd.DataFrame([
        {"frame": 100, "sleap_index": 0, "x": 10, "y": 10, "score": 0.9},
        {"frame": 100, "sleap_index": 1, "x": 15, "y": 15, "score": 0.9},
    ])

    table = match_masks_to_sleap_points(instances, points)

    assert table.loc[0, "coincidence_status"] == "manual_review"
    assert table.loc[0, "coincidence_reason"] == "multiple_sleap_points_for_mask"


def test_sleap_gate_ignores_low_score_point():
    instances = [MaskInstance(100, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9)]
    points = pd.DataFrame([{"frame": 100, "sleap_index": 0, "x": 10, "y": 10, "score": 0.1}])

    table = match_masks_to_sleap_points(instances, points, {"min_sleap_score": 0.5})

    assert table.loc[0, "coincidence_status"] == "reject"
    assert table.loc[0, "n_sleap_points_in_frame"] == 0


def test_sleap_gate_uses_absolute_frame_numbers_for_skipped_masks():
    instances = [MaskInstance(181300, 0, 0, _rect_mask(5, 5, 15, 15), (5, 5, 15, 15), 0.9)]
    points = pd.DataFrame([
        {"frame": 0, "sleap_index": 0, "x": 10, "y": 10, "score": 0.9},
        {"frame": 181300, "sleap_index": 1, "x": 10, "y": 10, "score": 0.9},
    ])

    table = match_masks_to_sleap_points(instances, points)

    assert table.loc[0, "coincidence_status"] == "keep"
    assert table.loc[0, "matched_sleap_index"] == 1


def test_nearest_reference_assignment_allows_many_fragments_per_identity():
    distance = np.asarray([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.9, 0.1],
    ])

    assert nearest_reference_assignment(distance) == {0: 0, 1: 0, 2: 1}


def test_stitch_identity_fragments_connects_same_id_motion_gap():
    tracklets = pd.DataFrame([
        _identity_tracklet_row(tracklet_id=1, frame=0, x=0, y=0),
        _identity_tracklet_row(tracklet_id=1, frame=10, x=10, y=0),
        _identity_tracklet_row(tracklet_id=2, frame=30, x=30, y=0),
        _identity_tracklet_row(tracklet_id=2, frame=40, x=40, y=0),
    ])
    assignments = pd.DataFrame([
        _assignment_row(1, "pseudo_ID_0"),
        _assignment_row(2, "pseudo_ID_0"),
    ])

    report = stitch_identity_fragments(
        tracklets_qc=tracklets,
        assignments=assignments,
        max_gap_frames=30,
        max_speed_px_per_frame=2.0,
    )

    assert report["n_identity_trajectories"] == 1
    assert len(report["stitch_events"]) == 1


def test_stitch_identity_fragments_splits_large_jump():
    tracklets = pd.DataFrame([
        _identity_tracklet_row(tracklet_id=1, frame=0, x=0, y=0),
        _identity_tracklet_row(tracklet_id=1, frame=10, x=10, y=0),
        _identity_tracklet_row(tracklet_id=2, frame=30, x=500, y=0),
        _identity_tracklet_row(tracklet_id=2, frame=40, x=510, y=0),
    ])
    assignments = pd.DataFrame([
        _assignment_row(1, "pseudo_ID_0"),
        _assignment_row(2, "pseudo_ID_0"),
    ])

    report = stitch_identity_fragments(
        tracklets_qc=tracklets,
        assignments=assignments,
        max_gap_frames=30,
        max_speed_px_per_frame=2.0,
    )

    assert report["n_identity_trajectories"] == 2
    assert report["stitch_events"] == []


def test_export_segmentation_top_tracks_writes_bind_compatible_csv(tmp_path):
    tracks = pd.DataFrame([
        {
            **_identity_tracklet_row(tracklet_id=8, frame=100, x=10, y=20),
            "identity_trajectory_id": "pseudo_ID_0_seg0",
            "assigned_identity": "pseudo_ID_0",
        },
        {
            **_identity_tracklet_row(tracklet_id=8, frame=110, x=11, y=21, status="manual_review"),
            "identity_trajectory_id": "pseudo_ID_0_seg0",
            "assigned_identity": "pseudo_ID_0",
        },
    ])
    output_csv = tmp_path / "top_tracks.csv"

    summary = export_segmentation_top_tracks(identity_tracks=tracks, output_csv=output_csv)
    out = pd.read_csv(output_csv)

    assert summary["n_tracks"] == 1
    assert {"frame", "track_id", "x", "y", "score", "node"}.issubset(out.columns)
    assert out["track_label"].tolist() == ["pseudo_ID_0", "pseudo_ID_0"]
    assert out["identity"].tolist() == ["pseudo_ID_0", "pseudo_ID_0"]
    assert out["bind_weight"].tolist() == [1.0, 0.1]


def test_hybrid_sleap_tracking_uses_segmentation_anchor_after_crossing():
    sleap = pd.DataFrame([
        _sleap_point(0, 0, 0),
        _sleap_point(0, 1, 100),
        _sleap_point(1, 0, 40),
        _sleap_point(1, 1, 60),
        _sleap_point(2, 0, 40),
        _sleap_point(2, 1, 60),
    ])
    seg = pd.DataFrame([
        _seg_anchor(0, "pseudo_ID_0", 0, tracklet_id=1),
        _seg_anchor(0, "pseudo_ID_1", 100, tracklet_id=2),
        _seg_anchor(2, "pseudo_ID_0", 60, tracklet_id=1),
        _seg_anchor(2, "pseudo_ID_1", 40, tracklet_id=2),
    ])

    tracks, _events, summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap,
        segmentation_tracks=seg,
        config={
            "num_individuals": 2,
            "segmentation_match_distance_px": 10,
            "max_sleap_distance_px_per_frame": 80,
            "segmentation_bonus": 100,
            "reject_large_segmentation_anchor_jumps": False,
        },
    )

    frame2 = tracks[tracks["frame"] == 2].set_index("track_label")
    assert frame2.loc["pseudo_ID_0", "x"] == pytest.approx(60)
    assert frame2.loc["pseudo_ID_1", "x"] == pytest.approx(40)
    assert summary["tracking_source_counts"]["segmentation_anchor"] == 4


def test_hybrid_sleap_tracking_recovers_after_sleap_false_negative():
    sleap = pd.DataFrame([
        _sleap_point(0, 0, 0),
        _sleap_point(0, 1, 100),
        _sleap_point(1, 0, 90),
        _sleap_point(2, 0, 20),
        _sleap_point(2, 1, 80),
    ])
    seg = pd.DataFrame([
        _seg_anchor(0, "pseudo_ID_0", 0, tracklet_id=1),
        _seg_anchor(0, "pseudo_ID_1", 100, tracklet_id=2),
        _seg_anchor(2, "pseudo_ID_0", 20, tracklet_id=1),
        _seg_anchor(2, "pseudo_ID_1", 80, tracklet_id=2),
    ])

    tracks, events, _summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap,
        segmentation_tracks=seg,
        config={
            "num_individuals": 2,
            "segmentation_match_distance_px": 10,
            "max_sleap_distance_px_per_frame": 20,
            "segmentation_bonus": 100,
            "reject_large_segmentation_anchor_jumps": False,
        },
    )

    frame2 = tracks[tracks["frame"] == 2].set_index("track_label")
    assert frame2.loc["pseudo_ID_0", "x"] == pytest.approx(20)
    assert frame2.loc["pseudo_ID_1", "x"] == pytest.approx(80)
    assert "missing_sleap_detection" in set(events["event_type"])


def test_hybrid_sleap_tracking_bootstraps_before_first_segmentation_anchor():
    sleap = pd.DataFrame([
        _sleap_point(0, 0, 0),
        _sleap_point(0, 1, 100),
        _sleap_point(1, 0, 10),
        _sleap_point(1, 1, 90),
    ])
    seg = pd.DataFrame([
        _seg_anchor(1, "pseudo_ID_0", 10, tracklet_id=1),
        _seg_anchor(1, "pseudo_ID_1", 90, tracklet_id=2),
    ])

    tracks, _events, summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap,
        segmentation_tracks=seg,
        config={
            "num_individuals": 2,
            "segmentation_match_distance_px": 10,
            "max_sleap_distance_px_per_frame": 20,
        },
    )

    frame0 = tracks[tracks["frame"] == 0]
    assert set(frame0["tracking_source"]) == {"sleap_bootstrap"}
    assert set(frame0["identity"]) == {"pseudo_ID_0", "pseudo_ID_1"}
    assert set(frame0["bind_weight"]) == {0.05}
    assert summary["frame_start"] == 0


def test_hybrid_sleap_tracking_resets_velocity_after_anchor_correction():
    sleap = pd.DataFrame([
        _sleap_point(0, 0, 0),
        _sleap_point(0, 1, 100),
        _sleap_point(1, 0, 1),
        _sleap_point(1, 1, 101),
        _sleap_point(2, 0, 2),
        _sleap_point(2, 1, 102),
        _sleap_point(3, 0, 3),
        _sleap_point(3, 1, 103),
    ])
    seg = pd.DataFrame([
        _seg_anchor(2, "pseudo_ID_0", 102, tracklet_id=1),
        _seg_anchor(2, "pseudo_ID_1", 2, tracklet_id=2),
    ])

    tracks, events, _summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap,
        segmentation_tracks=seg,
        config={
            "num_individuals": 2,
            "segmentation_match_distance_px": 10,
            "max_sleap_distance_px_per_frame": 20,
            "segmentation_bonus": 100,
            "reject_large_segmentation_anchor_jumps": False,
        },
    )

    frame2 = tracks[tracks["frame"] == 2].set_index("track_label")
    frame3 = tracks[tracks["frame"] == 3].set_index("track_label")
    assert frame2.loc["pseudo_ID_0", "x"] == pytest.approx(102)
    assert frame3.loc["pseudo_ID_0", "x"] == pytest.approx(103)
    assert not events[(events["frame"] == 3) & (events["event_type"] == "unassigned_sleap_detection")].any().any()


def test_hybrid_sleap_tracking_rejects_huge_segmentation_anchor_flip():
    sleap = pd.DataFrame([
        _sleap_point(0, 0, 0),
        _sleap_point(0, 1, 100),
        _sleap_point(1, 0, 1),
        _sleap_point(1, 1, 101),
        _sleap_point(2, 0, 2),
        _sleap_point(2, 1, 102),
    ])
    seg = pd.DataFrame([
        _seg_anchor(0, "pseudo_ID_0", 0, tracklet_id=1),
        _seg_anchor(0, "pseudo_ID_1", 100, tracklet_id=2),
        _seg_anchor(2, "pseudo_ID_0", 102, tracklet_id=1),
        _seg_anchor(2, "pseudo_ID_1", 2, tracklet_id=2),
    ])

    tracks, events, _summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap,
        segmentation_tracks=seg,
        config={
            "num_individuals": 2,
            "segmentation_match_distance_px": 10,
            "max_sleap_distance_px_per_frame": 20,
            "segmentation_bonus": 100,
            "reject_large_segmentation_anchor_jumps": True,
            "max_segmentation_anchor_jump_px_per_frame": 25,
            "max_segmentation_anchor_hard_jump_px": 50,
        },
    )

    frame2 = tracks[tracks["frame"] == 2].set_index("track_label")
    assert frame2.loc["pseudo_ID_0", "x"] == pytest.approx(2)
    assert frame2.loc["pseudo_ID_1", "x"] == pytest.approx(102)


def test_bind_topcam_id_preserves_segmentation_track_labels(tmp_path):
    geom_path = tmp_path / "geom.csv"
    top_path = tmp_path / "top.csv"
    out_path = tmp_path / "identified.csv"
    pd.DataFrame([
        {"frame": 0, "fish_id": "a", "X": 10.0, "Y": 20.0, "Z": 1.0},
        {"frame": 1, "fish_id": "a", "X": 11.0, "Y": 21.0, "Z": 1.0},
    ]).to_csv(geom_path, index=False)
    pd.DataFrame([
        {
            "frame": 0,
            "track_id": 0,
            "x": 10.0,
            "y": 20.0,
            "score": 0.9,
            "track_label": "pseudo_ID_0",
            "identity_trajectory_id": "pseudo_ID_0_seg0",
            "qc_status": "clean_train",
            "bind_weight": 1.0,
        },
        {
            "frame": 1,
            "track_id": 0,
            "x": 11.0,
            "y": 21.0,
            "score": 0.9,
            "track_label": "pseudo_ID_0",
            "identity_trajectory_id": "pseudo_ID_0_seg0",
            "qc_status": "clean_train",
            "bind_weight": 1.0,
        },
    ]).to_csv(top_path, index=False)

    out = bind(
        geom_path,
        top_path,
        calib=_IdentityProjectionCalibration(),
        out_csv=out_path,
        bind_distance=5.0,
        require_min_frames=1,
    )

    assert set(out["source_top_label"]) == {"pseudo_ID_0"}
    assert set(out["source_identity_trajectory_id"]) == {"pseudo_ID_0_seg0"}
    assert set(out["final_fish_id"]) == {"pseudo_ID_0"}


def test_top_label_lookup_normalizes_float_track_ids_and_skips_missing_ids():
    bound = pd.DataFrame([
        {"source_top_id": 0.0, "source_top_label": "pseudo_ID_0"},
        {"source_top_id": 0.0, "source_top_label": "pseudo_ID_0"},
        {"source_top_id": 1.0, "source_top_label": "pseudo_ID_1"},
        {"source_top_id": np.nan, "source_top_label": "missing"},
    ])

    assert _top_label_lookup(bound) == {0: "pseudo_ID_0", 1: "pseudo_ID_1"}


def test_body_metrics_attach_tail_without_changing_identity():
    tracks = pd.DataFrame([
        {
            "frame": 10,
            "track_id": 0,
            "x": 100.0,
            "y": 50.0,
            "identity": "pseudo_ID_0",
            "track_label": "pseudo_ID_0",
            "identity_trajectory_id": "pseudo_ID_0_seg0",
        },
    ])
    body_points = pd.DataFrame([
        {
            "frame": 10,
            "sleap_index": 2,
            "mouth_x": 101.0,
            "mouth_y": 51.0,
            "tail_x": 91.0,
            "tail_y": 51.0,
            "score": 0.9,
        },
    ])

    metrics = compute_body_metrics_for_tracks(
        tracks=tracks,
        body_points=body_points,
        max_mouth_match_distance_px=5.0,
    )

    assert len(metrics) == 1
    assert metrics.loc[0, "identity"] == "pseudo_ID_0"
    assert metrics.loc[0, "track_label"] == "pseudo_ID_0"
    assert metrics.loc[0, "body_qc_status"] == "good"
    assert metrics.loc[0, "tail_x"] == pytest.approx(91.0)
    assert metrics.loc[0, "body_vector_x"] == pytest.approx(9.0)
    assert metrics.loc[0, "body_length_px"] == pytest.approx(np.hypot(9.0, -1.0))


def test_body_metrics_missing_tail_keeps_mouth_track_row():
    tracks = pd.DataFrame([
        {
            "frame": 10,
            "track_id": 0,
            "x": 100.0,
            "y": 50.0,
            "identity": "pseudo_ID_0",
            "track_label": "pseudo_ID_0",
        },
    ])
    body_points = pd.DataFrame([
        {
            "frame": 10,
            "sleap_index": 2,
            "mouth_x": 100.0,
            "mouth_y": 50.0,
            "tail_x": np.nan,
            "tail_y": np.nan,
            "score": 0.9,
        },
    ])

    metrics = compute_body_metrics_for_tracks(
        tracks=tracks,
        body_points=body_points,
        max_mouth_match_distance_px=5.0,
    )

    assert len(metrics) == 1
    assert metrics.loc[0, "identity"] == "pseudo_ID_0"
    assert metrics.loc[0, "body_qc_status"] == "missing"
    assert metrics.loc[0, "body_qc_reason"] == "missing_tail"
    assert np.isnan(metrics.loc[0, "tail_x"])


def test_body_metrics_implausible_tail_flags_manual_review():
    tracks = pd.DataFrame([
        {"frame": 10, "track_id": 0, "x": 100.0, "y": 50.0, "identity": "pseudo_ID_0"},
    ])
    body_points = pd.DataFrame([
        {
            "frame": 10,
            "sleap_index": 2,
            "mouth_x": 100.0,
            "mouth_y": 50.0,
            "tail_x": 99.8,
            "tail_y": 50.0,
            "score": 0.9,
        },
    ])

    metrics = compute_body_metrics_for_tracks(
        tracks=tracks,
        body_points=body_points,
        max_mouth_match_distance_px=5.0,
        min_body_length_px=2.0,
    )

    assert metrics.loc[0, "body_qc_status"] == "manual_review"
    assert metrics.loc[0, "body_qc_reason"] == "implausible_body_length"


def test_merge_coco_exports_rewrites_ids_and_copies_images(tmp_path):
    for idx in range(2):
        coco_dir = tmp_path / f"coco{idx}"
        image_dir = coco_dir / "images"
        image_dir.mkdir(parents=True)
        (image_dir / "frame_000000.jpg").write_bytes(b"fake")
        payload = {
            "images": [{"id": 1, "file_name": "frame_000000.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 3, 3], "segmentation": [], "area": 9, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "medaka"}],
        }
        (coco_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = merge_coco_exports(coco_dirs=[tmp_path / "coco0", tmp_path / "coco1"], output_dir=tmp_path / "merged")
    merged = json.loads(Path(summary["annotation_path"]).read_text(encoding="utf-8"))

    assert summary["n_images"] == 2
    assert summary["n_annotations"] == 2
    assert {image["id"] for image in merged["images"]} == {1, 2}
    assert {ann["image_id"] for ann in merged["annotations"]} == {1, 2}
