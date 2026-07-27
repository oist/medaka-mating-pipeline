from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.analysis.contact_episode_gt import (
    serialize_false_negative_detections,
    serialize_false_positive_detections,
)
from scripts.sleap.mine_round8_hard_frames import (
    FeatureConfig,
    SelectionConfig,
    _extract_view_features,
    add_existing_label_overlap,
    add_geometry_ambiguity,
    assign_event_groups,
    build_new_only_annotation_seed,
    count_ambiguous_views,
    ingest_manual_evidence,
    isolated_residuals,
    merge_contact_events,
    select_with_quotas_and_locked_holdout,
    validate_annotation_seed,
    validate_event_split,
)


def test_isolated_residual_uses_exact_midpoint_and_detects_spike() -> None:
    x = np.arange(41, dtype=float)
    x[20] = 60.0
    observations = pd.DataFrame(
        {
            "frame": np.arange(41),
            "track_id": ["fish_0"] * 41,
            "x": x,
            "y": np.zeros(41),
        }
    )
    result = isolated_residuals(
        observations,
        local_window_frames=31,
        residual_floor_px=15.0,
        mad_multiplier=6.0,
    ).set_index("frame")
    assert result.at[20, "residual_px"] == pytest.approx(40.0)
    assert bool(result.at[20, "is_isolated_outlier"])
    assert result.at[19, "residual_px"] == pytest.approx(20.0)


def test_isolated_residual_keeps_nonconsecutive_neighbors_missing() -> None:
    observations = pd.DataFrame(
        {
            "frame": [0, 1, 3, 4, 5],
            "track_id": ["fish_0"] * 5,
            "x": [0.0, 100.0, 3.0, 4.0, 5.0],
            "y": [0.0] * 5,
        }
    )
    result = isolated_residuals(observations).set_index("frame")
    assert np.isnan(result.at[1, "residual_px"])
    assert np.isnan(result.at[3, "residual_px"])
    assert not bool(result.at[1, "is_isolated_outlier"])


def test_multiview_ambiguity_count_and_names() -> None:
    table = pd.DataFrame(
        {
            "top_cam_ambiguous": [True, False],
            "side1_cam_ambiguous": [True, True],
            "side2_cam_ambiguous": [False, True],
        }
    )
    result = count_ambiguous_views(table)
    assert result["ambiguous_view_count"].tolist() == [2, 2]
    assert result["ambiguous_views"].tolist() == [
        "top_cam;side1_cam",
        "side1_cam;side2_cam",
    ]


def test_geometry_competition_and_small_margin_are_vectorized_per_frame() -> None:
    frames = pd.DataFrame(
        {
            "frame": [0, 1],
            "plausible_count": [3, 3],
            "partial_instance_count": [0, 0],
            "minimum_mouth_distance_px": [100.0, 100.0],
            "minimum_score": [0.9, 0.9],
            "score_range": [0.0, 0.0],
        }
    )
    instances = pd.DataFrame(
        [
            {"frame": frame, "instance_index": index, "assoc_x": x, "assoc_y": 0.0, "plausible": True}
            for frame, values in ((0, [0.0, 100.0, 200.0]), (1, [0.0, 10.0, 100.0]))
            for index, x in enumerate(values)
        ]
    )
    projected = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [199.0, 0.0]],
            [[5.0, 0.0], [100.0, 0.0], [200.0, 0.0]],
        ]
    )
    result = add_geometry_ambiguity(
        frames,
        instances,
        projected_geometry=projected,
        geometry_frames=np.array([0, 1]),
        config=FeatureConfig(),
    )
    assert bool(result.loc[0, "geometry_competition"])
    assert not bool(result.loc[0, "small_association_margin"])
    assert bool(result.loc[1, "small_association_margin"])


def test_contact_event_grouping_merges_known_29811_and_29859_event() -> None:
    candidates = pd.DataFrame(
        {
            "contact_start_frame": [29_811, 29_859, 40_000],
            "contact_end_frame": [29_830, 29_880, 40_020],
            "episode_id": ["ep_a", "ep_b", "ep_c"],
        }
    )
    events = merge_contact_events(candidates, merge_gap_frames=166)
    assigned = assign_event_groups(
        [29_811, 29_859, 40_000], events, context_frames=166
    ).set_index("frame")
    assert assigned.at[29_811, "event_group_id"] == assigned.at[29_859, "event_group_id"]
    assert assigned.at[29_811, "contact_episode_ids"] == "ep_a;ep_b"
    assert assigned.at[40_000, "event_group_id"] != assigned.at[29_811, "event_group_id"]


def _selection_candidates() -> pd.DataFrame:
    categories = [
        "close_multi_view_ambiguity",
        "manual_fp_or_fn",
        "isolated_xy_outlier",
        "close_single_view_ambiguity",
    ]
    rows: list[dict[str, object]] = []
    # Two locked holdout rows, one from each of the first two categories.
    for frame, category in ((100, categories[0]), (101, categories[1])):
        rows.append(
            {
                "frame": frame,
                "category": category,
                "category_score": 1000.0 - frame,
                "event_group_id": "holdout_sentinel_100_101",
                "event_start_frame": 100,
                "event_end_frame": 101,
                "manual_fp_views": "",
                "manual_fn_views": "",
            }
        )
    by_category = {
        categories[0]: [0, 10, 20, 30, 40, 50],
        categories[1]: [60, 70],
        categories[2]: [80, 90],
        categories[3]: [200, 210],
        # Deliberately no clean-control rows: its quota must be redistributed.
    }
    for category, frames in by_category.items():
        for frame in frames:
            rows.append(
                {
                    "frame": frame,
                    "category": category,
                    "category_score": 1000.0 - frame,
                    "event_group_id": f"event_{frame}",
                    "event_start_frame": frame,
                    "event_end_frame": frame,
                    "manual_fp_views": "",
                    "manual_fn_views": "",
                }
            )
    return pd.DataFrame(rows)


def test_quota_deficit_redistribution_and_selection_are_deterministic() -> None:
    quotas = OrderedDict(
        [
            ("close_multi_view_ambiguity", 3),
            ("manual_fp_or_fn", 3),
            ("isolated_xy_outlier", 2),
            ("close_single_view_ambiguity", 2),
            ("close_clean_control", 2),
        ]
    )
    config = SelectionConfig(
        sentinel_start=100,
        sentinel_end=101,
        holdout_frames=2,
        train_frames=10,
        guard_band_frames=1,
        min_spacing_frames=1,
        time_bins=4,
    )
    first, first_report = select_with_quotas_and_locked_holdout(
        _selection_candidates(), quotas=quotas, config=config
    )
    second, second_report = select_with_quotas_and_locked_holdout(
        _selection_candidates(), quotas=quotas, config=config
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
    assert len(first) == 12
    assert first["split"].value_counts().to_dict() == {"train": 10, "holdout": 2}
    assert first_report["redistribution"]
    assert sum(item["count"] for item in first_report["redistribution"]) == 2


def test_event_split_has_zero_guard_band_leakage() -> None:
    valid = pd.DataFrame(
        {
            "frame": [-100, 300, 110],
            "split": ["train", "train", "holdout"],
            "event_group_id": ["train_a", "train_b", "holdout_event"],
            "event_start_frame": [-100, 300, 100],
            "event_end_frame": [-100, 300, 120],
        }
    )
    report = validate_event_split(valid, guard_band_frames=166)
    assert report["passed"] is True
    invalid = valid.copy()
    invalid.loc[invalid["frame"].eq(-100), "frame"] = 50
    with pytest.raises(ValueError, match="leakage"):
        validate_event_split(invalid, guard_band_frames=166)


def test_existing_label_overlap_is_deduplicated_by_view_and_frame() -> None:
    features = pd.DataFrame({"frame": [10, 11]})
    index = {("top_cam", 10), ("side1_cam", 10), ("top_cam", 10)}
    result = add_existing_label_overlap(features, index)
    assert result["existing_label_overlap"].tolist() == ["top_cam;side1_cam", ""]


def test_manual_fp_and_fn_evidence_is_ingested_without_identity_assignment() -> None:
    ledger = pd.DataFrame(
        [
            {
                "episode_id": "episode_1",
                "false_positive_detections_json": serialize_false_positive_detections(
                    [
                        {
                            "view": "top_cam",
                            "frame": 42,
                            "instance_index": 3,
                            "detection_key": "exp|top_cam|42|3",
                            "marked_at_utc": "2026-07-21T00:00:00Z",
                        }
                    ]
                ),
                "false_negative_detections_json": serialize_false_negative_detections(
                    [
                        {
                            "view": "side2_cam",
                            "frame": 42,
                            "click_x": 20,
                            "click_y": 21,
                            "marked_at_utc": "2026-07-21T00:01:00Z",
                        }
                    ]
                ),
            }
        ]
    )
    evidence = ingest_manual_evidence(ledger).iloc[0]
    assert evidence["frame"] == 42
    assert evidence["manual_fp_views"] == "top_cam"
    assert evidence["manual_fn_views"] == "side2_cam"
    assert "identity" not in " ".join(evidence.index).lower()


def _write_video(path: Path, frames: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for frame in range(frames):
        writer.write(np.full((24, 32, 3), 40 + frame * 20, dtype=np.uint8))
    writer.release()


def test_new_only_seed_preserves_partial_and_empty_predictions(tmp_path: Path) -> None:
    sio = pytest.importorskip("sleap_io")
    experiment = "synthetic_experiment"
    skeleton = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])
    slp_paths: dict[str, Path] = {}
    for view in ("top_cam", "side1_cam", "side2_cam"):
        video_path = tmp_path / experiment / view / "camera.avi"
        _write_video(video_path)
        video = sio.Video.from_filename(str(video_path))
        partial = sio.PredictedInstance.from_numpy(
            points_data=np.array([[10.0, 11.0], [np.nan, np.nan]]),
            point_scores=np.array([0.9, 0.0]),
            score=0.9,
            skeleton=skeleton,
        )
        labels = sio.Labels(
            skeletons=[skeleton],
            videos=[video],
            labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0, instances=[partial])],
        )
        slp_path = tmp_path / f"{view}.slp"
        sio.save_slp(labels, str(slp_path), embed=False, verbose=False)
        slp_paths[view] = slp_path

    features, instances, metadata = _extract_view_features(
        slp_paths["top_cam"], config=FeatureConfig()
    )
    assert metadata["extraction_backend"] == "validated_native_hdf5"
    assert len(features) == 2
    assert features.loc[features["frame"].eq(1), "instance_count"].item() == 0
    assert len(instances) == 1

    manifest = pd.DataFrame(
        {
            "frame": [0, 1],
            "category": ["manual_fp_or_fn", "manual_fp_or_fn"],
            "event_group_id": ["a", "b"],
            "split": ["train", "holdout"],
            "annotation_batch": ["pilot", "pilot"],
            "selection_reason": ["partial", "empty"],
        }
    )
    manifest_path = tmp_path / "pilot.csv"
    manifest.to_csv(manifest_path, index=False)
    output_path = tmp_path / "pilot_seed.pkg.slp"
    report = build_new_only_annotation_seed(
        manifest,
        experiment=experiment,
        slp_paths=slp_paths,
        output_path=output_path,
        selection_manifest_path=manifest_path,
    )
    validation = validate_annotation_seed(output_path, manifest, experiment=experiment)
    assert validation["view_image_count"] == 6
    assert validation["partial_instances"] == 3
    assert validation["empty_labeled_frames"] == 3
    assert report["conversion_counts"]["partial_instances"] == 3
    loaded = sio.load_slp(str(output_path))
    assert loaded.provenance["workflow"] == "round8_active_learning_new_only_seed"
    assert json.loads(json.dumps(loaded.provenance))["instructions"]


def test_fast_features_ignore_absent_global_slots_in_mouth_distance(
    tmp_path: Path,
) -> None:
    sio = pytest.importorskip("sleap_io")
    video_path = tmp_path / "experiment" / "side1_cam" / "camera.avi"
    _write_video(video_path)
    video = sio.Video.from_filename(str(video_path))
    skeleton = sio.Skeleton(nodes=["mouth", "tail"], edges=[("mouth", "tail")])

    def predicted(mouth_x: float):
        return sio.PredictedInstance.from_numpy(
            points_data=np.array(
                [[mouth_x, 5.0], [mouth_x + 2.0, 5.0]], dtype=float
            ),
            point_scores=np.array([0.9, 0.9]),
            score=0.9,
            skeleton=skeleton,
        )

    labels = sio.Labels(
        skeletons=[skeleton],
        videos=[video],
        labeled_frames=[
            sio.LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[predicted(float(index * 10)) for index in range(6)],
            ),
            sio.LabeledFrame(
                video=video,
                frame_idx=1,
                instances=[predicted(0.0), predicted(12.0), predicted(40.0)],
            ),
        ],
    )
    slp_path = tmp_path / "predictions.slp"
    sio.save_slp(labels, str(slp_path), embed=False, verbose=False)
    features, _instances, metadata = _extract_view_features(
        slp_path, config=FeatureConfig()
    )
    assert metadata["extraction_backend"] == "validated_native_hdf5"
    distances = features.set_index("frame")["minimum_mouth_distance_px"]
    assert distances.loc[0] == pytest.approx(10.0)
    assert distances.loc[1] == pytest.approx(12.0)
