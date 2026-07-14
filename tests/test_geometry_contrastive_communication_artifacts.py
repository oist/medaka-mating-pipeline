from __future__ import annotations

import pandas as pd

from scripts.visualization.build_geometry_contrastive_communication_artifacts import (
    _hex_to_bgr,
    select_complete_local_source_frame,
)
from scripts.visualization.render_geometry_supervision_audit import _positive_components


def _complete_frame(frame: int, *, mask_scores: list[float], residual: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    views = ("top_cam", "side1_cam", "side2_cam")
    for source in range(3):
        for view_index, view in enumerate(views):
            cell = source * 3 + view_index
            rows.append(
                {
                    "sample_id": frame * 100 + cell,
                    "frame": frame,
                    "geometry_source_idx_local": source,
                    "view": view,
                    "matched_mask_score": mask_scores[cell],
                    "residual_px2": residual,
                    "min_fish_separation_mm": 30.0,
                    "split": "test",
                }
            )
    return rows


def test_complete_frame_selection_enforces_per_crop_mask_floor() -> None:
    low_residual_with_one_bad_crop = _complete_frame(
        100,
        mask_scores=[0.50] + [1.0] * 8,
        residual=1.0,
    )
    strict_complete_frame = _complete_frame(
        200,
        mask_scores=[0.90] * 9,
        residual=10.0,
    )
    samples = pd.DataFrame(low_residual_with_one_bad_crop + strict_complete_frame)

    frame, selected = select_complete_local_source_frame(samples)

    assert frame == 200
    assert len(selected) == 9
    assert selected.matched_mask_score.min() == 0.90


def test_hex_to_bgr_matches_opencv_channel_order() -> None:
    assert _hex_to_bgr("#2F6B8A") == (0x8A, 0x6B, 0x2F)


def test_positive_component_labels_follow_explicit_supervision() -> None:
    samples = pd.DataFrame(
        [
            {
                "sample_id": 1,
                "frame": 100,
                "view": "top_cam",
                "window": "heldout",
                "split": "test",
                "local_fragment_id": "top_a",
                "geometry_source_idx_local": 0,
            },
            {
                "sample_id": 2,
                "frame": 100,
                "view": "side1_cam",
                "window": "heldout",
                "split": "test",
                "local_fragment_id": "side_a",
                "geometry_source_idx_local": 0,
            },
            {
                "sample_id": 3,
                "frame": 100,
                "view": "top_cam",
                "window": "heldout",
                "split": "test",
                "local_fragment_id": "top_b",
                "geometry_source_idx_local": 1,
            },
        ]
    )
    positives = pd.DataFrame(
        [
            {
                "sample_a_id": 1,
                "sample_b_id": 2,
                "pair_type": "synchronized_cross_camera_same_source",
                "pair_frame_delta": 0,
            }
        ]
    )
    negatives = pd.DataFrame([{"sample_a_id": 1, "sample_b_id": 3}])

    labeled, components, _events, audit = _positive_components(
        samples, positives, negatives
    )

    labels = labeled.set_index("sample_id").p_label.to_dict()
    assert labels[1] == labels[2]
    assert labels[1] != labels[3]
    assert len(components) == 2
    assert audit["transitive_positive_negative_contradictions"] == 0
