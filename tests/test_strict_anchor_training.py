from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.sleap.train_strict_anchor import (
    apply_post_affine_crop_jitter,
    filter_instances_with_finite_anchor,
    instance_has_finite_anchor,
    instances_have_finite_anchor,
    strict_compute_gt_centroids,
    strict_generate_centroids,
)
from sleap_nn.data.augmentation import apply_geometric_augmentation


class FakeInstance:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def numpy(self):
        return self.points


def test_strict_centroid_does_not_replace_missing_mouth_with_tail() -> None:
    points = torch.tensor(
        [[[[float("nan"), float("nan")], [20.0, 30.0]]]],
        dtype=torch.float32,
    )

    centroid = strict_generate_centroids(points, anchor_ind=0)

    assert torch.isnan(centroid).all()


def test_strict_centroid_uses_finite_mouth_exactly() -> None:
    points = torch.tensor(
        [[[[10.0, 12.0], [20.0, 30.0]]]],
        dtype=torch.float32,
    )

    centroid = strict_generate_centroids(points, anchor_ind=0)

    torch.testing.assert_close(centroid, torch.tensor([[[10.0, 12.0]]]))


def test_strict_policy_requires_configured_anchor() -> None:
    points = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="configured anchor_part"):
        strict_generate_centroids(points, anchor_ind=None)


def test_frame_filter_rejects_any_instance_with_missing_anchor() -> None:
    complete = FakeInstance([[1.0, 2.0], [3.0, 4.0]])
    missing_mouth = FakeInstance([[np.nan, np.nan], [5.0, 6.0]])

    assert instances_have_finite_anchor([complete], anchor_ind=0)
    assert not instances_have_finite_anchor(
        [complete, missing_mouth], anchor_ind=0
    )


def test_instance_filter_keeps_only_finite_semantic_anchors() -> None:
    complete = FakeInstance([[1.0, 2.0], [3.0, 4.0]])
    missing_mouth = FakeInstance([[np.nan, np.nan], [5.0, 6.0]])
    missing_tail = FakeInstance([[7.0, 8.0], [np.nan, np.nan]])

    assert instance_has_finite_anchor(complete, anchor_ind=0)
    assert not instance_has_finite_anchor(missing_mouth, anchor_ind=0)
    assert instance_has_finite_anchor(missing_tail, anchor_ind=0)
    assert filter_instances_with_finite_anchor(
        [complete, missing_mouth, missing_tail], anchor_ind=0
    ) == [complete, missing_tail]


def test_strict_numpy_evaluation_matches_training_policy() -> None:
    points = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[np.nan, np.nan], [8.0, 9.0]],
        ]
    )

    centroids = strict_compute_gt_centroids(points, anchor_ind=0)

    np.testing.assert_allclose(centroids[0], [1.0, 2.0])
    assert np.isnan(centroids[1]).all()


def test_post_affine_crop_jitter_respects_requested_pixel_bound() -> None:
    image = torch.zeros((1, 1, 101, 101), dtype=torch.uint8)
    instances = torch.tensor(
        [[[[50.5, 50.5], [75.0, 50.5]]]], dtype=torch.float32
    )
    kwargs = {
        "rotation_min": -90.0,
        "rotation_max": 90.0,
        "rotation_p": 1.0,
        "scale_min": 0.7,
        "scale_max": 1.3,
        "scale_p": 1.0,
        "translate_width": 8.0 / 101.0,
        "translate_height": 8.0 / 101.0,
        "translate_p": 1.0,
        "affine_p": 0.0,
        "erase_p": 0.0,
        "mixup_p": 0.0,
        "flip_p": 0.0,
    }
    observed = []
    for seed in range(32):
        np.random.seed(seed)
        no_translation = dict(kwargs)
        no_translation["translate_p"] = 0.0
        _, baseline = apply_geometric_augmentation(
            image, instances, **no_translation
        )

        np.random.seed(seed)
        _, augmented = apply_post_affine_crop_jitter(
            apply_geometric_augmentation,
            image,
            instances,
            **kwargs,
        )
        observed.append((augmented - baseline)[0, 0, 0].numpy())

    observed = np.asarray(observed)
    assert np.abs(observed).max() <= 8.0 + 1.0e-3
    assert np.abs(observed).max() > 6.0
