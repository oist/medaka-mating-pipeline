from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.sleap.train_strict_anchor import (
    instances_have_finite_anchor,
    strict_compute_gt_centroids,
    strict_generate_centroids,
)


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
