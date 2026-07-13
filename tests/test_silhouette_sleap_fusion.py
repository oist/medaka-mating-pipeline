from __future__ import annotations

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import MaskInstance, bbox_from_mask, write_mask_instances_h5
from scripts.tracking.silhouette_sleap_fusion import (
    FusionConfig,
    annotate_tracklets_with_masks,
    associate_frame,
    build_silhouette_link_candidates,
    split_on_silhouette_conflicts,
    summarize_fused_tracklets,
)


def _row(frame: int, tracklet_id: int, mouth: tuple[float, float], tail: tuple[float, float]) -> dict:
    return {
        "frame": frame,
        "tracklet_id": tracklet_id,
        "raw_detection_index": tracklet_id,
        "mouth_x": mouth[0],
        "mouth_y": mouth[1],
        "tail_x": tail[0],
        "tail_y": tail[1],
        "body_length_px": float(np.hypot(mouth[0] - tail[0], mouth[1] - tail[1])),
    }


def _mask(frame: int, instance_id: int, x0: int, y0: int, x1: int, y1: int) -> MaskInstance:
    array = np.zeros((64, 64), dtype=bool)
    array[y0:y1, x0:x1] = True
    return MaskInstance(
        frame=frame,
        instance_id=instance_id,
        sam2_track_id=-1,
        mask=array,
        bbox=bbox_from_mask(array),
        score=0.95,
    )


def test_unique_silhouettes_strongly_support_two_skeletons():
    rows = pd.DataFrame(
        [
            _row(10, 0, (24, 18), (8, 18)),
            _row(10, 1, (54, 42), (38, 42)),
        ]
    )
    masks = [_mask(10, 0, 5, 13, 27, 23), _mask(10, 1, 35, 37, 57, 47)]
    associated = associate_frame(rows, masks, config=FusionConfig())
    assert set(associated["silhouette_status"]) == {"strong_match"}
    assert set(associated["silhouette_evidence_state"]) == {"support"}
    assert associated["mutual_confidence"].gt(0).all()
    assert associated["n_skeletons_for_mask"].eq(1).all()


def test_endpoints_in_different_masks_flag_tangent_skeleton():
    rows = pd.DataFrame([_row(10, 0, (45, 30), (15, 30))])
    masks = [_mask(10, 0, 9, 24, 22, 37), _mask(10, 1, 39, 24, 52, 37)]
    associated = associate_frame(rows, masks, config=FusionConfig())
    assert associated.iloc[0]["silhouette_status"] == "unknown_split_endpoints"
    assert associated.iloc[0]["silhouette_evidence_state"] == "unknown"
    assert bool(associated.iloc[0]["split_endpoints_across_masks"])


def test_one_mask_covering_two_skeletons_is_unknown_not_contradiction():
    rows = pd.DataFrame(
        [
            _row(10, 0, (24, 27), (8, 27)),
            _row(10, 1, (50, 36), (34, 36)),
        ]
    )
    masks = [_mask(10, 0, 5, 20, 54, 43)]
    associated = associate_frame(rows, masks, config=FusionConfig())
    matched = associated[associated["matched_mask_instance_id"].ge(0)]
    assert len(matched) == 1
    assert matched.iloc[0]["silhouette_status"] == "unknown_mask_merges_skeletons"
    assert matched.iloc[0]["silhouette_evidence_state"] == "unknown"


def test_complete_one_to_one_low_overlap_can_be_strong_contradiction():
    rows = pd.DataFrame(
        [
            _row(10, 0, (24, 18), (8, 18)),
            _row(10, 1, (56, 52), (40, 52)),
        ]
    )
    masks = [
        _mask(10, 0, 5, 13, 27, 23),
        _mask(10, 1, 36, 35, 59, 45),
    ]
    associated = associate_frame(
        rows,
        masks,
        config=FusionConfig(strong_contradiction_min_assignment_margin=0.10),
    )
    contradicted = associated[associated["tracklet_id"].eq(1)].iloc[0]
    assert contradicted["silhouette_status"] == "strong_contradiction_low_overlap"
    assert contradicted["silhouette_evidence_state"] == "contradiction"


def test_incomplete_frame_low_overlap_remains_unknown():
    rows = pd.DataFrame(
        [
            _row(10, 0, (24, 18), (8, 18)),
            _row(10, 1, (56, 52), (40, 52)),
            _row(10, 2, (30, 48), (14, 48)),
        ]
    )
    masks = [
        _mask(10, 0, 5, 13, 27, 23),
        _mask(10, 1, 36, 25, 59, 35),
    ]
    associated = associate_frame(rows, masks, config=FusionConfig())
    assert "contradiction" not in set(associated["silhouette_evidence_state"])


def test_temporal_mask_discontinuity_is_unknown_and_does_not_split(tmp_path):
    rows = pd.DataFrame(
        [
            _row(10, 0, (24, 18), (8, 18)),
            _row(11, 0, (25, 18), (9, 18)),
            _row(12, 0, (50, 50), (40, 50)),
        ]
    )
    masks = [
        _mask(10, 0, 5, 13, 27, 23),
        _mask(11, 0, 6, 13, 28, 23),
        _mask(12, 0, 38, 45, 53, 56),
    ]
    mask_path = tmp_path / "masks.h5"
    write_mask_instances_h5(mask_path, masks, image_shape=(64, 64))
    annotated, _ = annotate_tracklets_with_masks(rows, mask_path)
    assert annotated.loc[annotated.frame == 11, "temporal_mask_iou"].iloc[0] > 0.8
    assert annotated.loc[annotated.frame == 11, "trusted_mask_segment_id"].iloc[0] >= 0
    assert annotated.loc[annotated.frame == 12, "silhouette_soft_reason"].iloc[0] == "mask_temporal_discontinuity"
    assert annotated.loc[annotated.frame == 12, "silhouette_break_reason"].iloc[0] == ""
    assert annotated.loc[annotated.frame == 12, "silhouette_evidence_state"].iloc[0] == "unknown"
    assert annotated.loc[annotated.frame == 12, "trusted_mask_segment_id"].iloc[0] == -1
    fused = split_on_silhouette_conflicts(annotated)
    assert bool(fused.loc[fused.frame == 12, "silhouette_accepted"].iloc[0])
    assert fused["fused_tracklet_id"].nunique() == 1
    summary = summarize_fused_tracklets(fused)
    assert summary["n_rows"].sum() == 3
    assert summary["mean_silhouette_evidence_weight"].between(0, 1).all()


def test_splitter_keeps_unknown_rows_but_removes_contradictions():
    annotated = pd.DataFrame(
        {
            "tracklet_id": [5, 5, 5, 5],
            "frame": [10, 11, 12, 13],
            "silhouette_evidence_state": ["support", "unknown", "contradiction", "support"],
            "silhouette_break_reason": ["", "", "strong_contradiction_low_overlap", ""],
        }
    )
    fused = split_on_silhouette_conflicts(annotated)
    assert fused.silhouette_accepted.tolist() == [True, True, False, True]
    assert fused.loc[fused.frame.isin([10, 11]), "fused_tracklet_id"].nunique() == 1
    assert fused.loc[fused.frame == 13, "fused_tracklet_id"].iloc[0] != fused.loc[fused.frame == 11, "fused_tracklet_id"].iloc[0]


def test_trusted_mask_segment_proposes_fragment_link():
    annotated = pd.DataFrame(
        {
            "trusted_mask_segment_id": [4, 4, 4, 4],
            "tracklet_id": [10, 10, 11, 11],
            "frame": [20, 21, 23, 24],
            "silhouette_evidence_weight": [1.0, 0.7, 1.0, 0.7],
            "mutual_confidence": [0.8, 0.7, 0.9, 0.8],
            "temporal_mask_iou": [np.nan, 0.9, np.nan, 0.85],
        }
    )
    candidates = build_silhouette_link_candidates(annotated)
    assert len(candidates) == 1
    assert candidates.iloc[0]["gap_frames"] == 1
    assert candidates.iloc[0]["silhouette_link_status"] == "candidate"


def test_link_candidates_use_post_split_fused_tracklet_ids():
    fused = pd.DataFrame(
        {
            "trusted_mask_segment_id": [7, 7, 7, 7],
            "tracklet_id": [10, 10, 10, 10],
            "fused_tracklet_id": [101, 101, 102, 102],
            "frame": [30, 31, 33, 34],
            "silhouette_evidence_weight": [1.0, 0.8, 1.0, 0.8],
            "mutual_confidence": [0.9, 0.8, 0.9, 0.8],
            "temporal_mask_iou": [np.nan, 0.9, np.nan, 0.9],
        }
    )
    candidate = build_silhouette_link_candidates(fused).iloc[0]
    assert int(candidate.left_tracklet_id) == 101
    assert int(candidate.right_tracklet_id) == 102


def test_splitter_treats_csv_nan_break_reason_as_no_conflict():
    annotated = pd.DataFrame(
        {
            "tracklet_id": [5, 5],
            "frame": [10, 11],
            "silhouette_break_reason": [np.nan, np.nan],
        }
    )
    fused = split_on_silhouette_conflicts(annotated)
    assert fused.silhouette_accepted.tolist() == [True, True]
    assert fused.fused_tracklet_id.nunique() == 1
