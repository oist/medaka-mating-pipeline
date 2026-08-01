from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import sleap_io as sio
import torch

from scripts.sleap.train_strict_anchor import (
    apply_post_affine_crop_jitter,
    build_validation_recovery_checkpoint_callback,
    build_grouped_epoch_eval_ground_truth,
    build_grouped_epoch_eval_predictions,
    filter_instances_with_finite_anchor,
    instance_has_finite_anchor,
    instances_have_finite_anchor,
    strict_compute_gt_centroids,
    strict_generate_centroids,
)
from sleap_nn.data.augmentation import apply_geometric_augmentation
from sleap_nn.evaluation import Evaluator
from sleap_nn.training.callbacks import EpochEndEvaluationCallback


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


def test_centered_instance_epoch_eval_groups_all_fish_per_frame() -> None:
    skeleton = sio.Skeleton(["mouth", "tail"])
    video = sio.Video.from_filename("synthetic_side.mp4")
    callback = SimpleNamespace(skeleton=skeleton, videos=[video])
    predictions = []
    ground_truth = []
    for fish_idx in range(3):
        points = np.array(
            [
                [10.0 + fish_idx * 20.0, 12.0],
                [18.0 + fish_idx * 20.0, 14.0],
            ]
        )
        predictions.append(
            {
                "video_idx": 0,
                "frame_idx": 42,
                "pred_peaks": points.reshape(1, 2, 2),
                "pred_scores": np.full((1, 2), 0.9 - fish_idx * 0.1),
            }
        )
        ground_truth.append(
            {
                "video_idx": 0,
                "frame_idx": 42,
                "gt_instances": points.reshape(1, 2, 2),
                "num_instances": 1,
            }
        )

    pred_labels = build_grouped_epoch_eval_predictions(
        callback, predictions, sio, np
    )
    gt_labels = build_grouped_epoch_eval_ground_truth(
        callback, ground_truth, sio, np
    )

    pred_frames = pred_labels.find(video, frame_idx=42)
    gt_frames = gt_labels.find(video, frame_idx=42)
    assert len(pred_frames) == 1
    assert len(gt_frames) == 1
    assert len(pred_frames[0].predicted_instances) == 3
    assert len(gt_frames[0].user_instances) == 3


def test_centered_instance_epoch_eval_keeps_complete_miss_frames() -> None:
    skeleton = sio.Skeleton(["mouth", "tail"])
    video = sio.Video.from_filename("synthetic_misses.mp4")
    callback = EpochEndEvaluationCallback(
        skeleton=skeleton,
        videos=[video],
        eval_frequency=1,
    )
    predictions = [
        {
            "video_idx": 0,
            "frame_idx": 10,
            "pred_peaks": None,
            "pred_scores": None,
        },
        {
            "video_idx": 0,
            "frame_idx": 11,
            "pred_peaks": np.full((1, 2, 2), np.nan),
            "pred_scores": np.full((1, 2), np.nan),
        },
    ]
    ground_truth = [
        {
            "video_idx": 0,
            "frame_idx": frame_idx,
            "gt_instances": np.array(
                [[[10.0, 10.0], [20.0, 20.0]]]
            ),
            "num_instances": 1,
        }
        for frame_idx in (10, 11)
    ]

    pred_labels = build_grouped_epoch_eval_predictions(
        callback, predictions, sio, np
    )
    gt_labels = build_grouped_epoch_eval_ground_truth(
        callback, ground_truth, sio, np
    )

    assert [lf.frame_idx for lf in pred_labels.labeled_frames] == [10, 11]
    assert all(len(lf.instances) == 0 for lf in pred_labels.labeled_frames)
    assert all(
        len(pred_labels.find(video, frame_idx=frame_idx)) == 1
        for frame_idx in (10, 11)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        evaluator = Evaluator(
            ground_truth_instances=gt_labels,
            predicted_instances=pred_labels,
            user_labels_only=False,
        )
        metrics = evaluator.evaluate()
    assert len(evaluator.frame_pairs) == 2
    assert len(evaluator.false_negatives) == 2
    assert np.isnan(metrics["mOKS"]["mOKS"])

    trainer = SimpleNamespace(callback_metrics={}, loggers=[])
    callback._log_metrics(trainer, metrics, epoch=0)
    assert trainer.callback_metrics["eval/val/mOKS"].item() == 0.0


def test_validation_recovery_checkpoint_is_rolling_full_state(
    tmp_path,
) -> None:
    callback = build_validation_recovery_checkpoint_callback(tmp_path / "run")

    assert callback.monitor == "step"
    assert callback.mode == "max"
    assert callback.save_top_k == 1
    assert callback.save_last is False
    assert callback.save_weights_only is False
    assert callback.filename == "recovery-e{epoch:05d}-s{step:09d}"
    assert callback._every_n_epochs == 1
    assert callback._every_n_train_steps == 0
    assert callback._save_on_train_epoch_end is False
    assert callback.format_checkpoint_name(
        {"epoch": 5, "step": 550}
    ).endswith("recovery-e00005-s000000550.ckpt")
