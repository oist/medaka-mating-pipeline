from __future__ import annotations

"""Run ``sleap-nn train`` with strict semantic-anchor supervision.

SLEAP-NN normally replaces a missing configured anchor with the mean of the
remaining visible nodes. That is useful for generic centroid localization, but
it is incorrect when the centroid head has a semantic contract such as
``anchor_part: mouth``: a tail-only instance becomes a tail centroid target.

This wrapper is deliberately process-local; it does not edit the installed
``sleap-nn`` package. Policies are selected with environment variables so the
same patch is installed in Windows DataLoader workers:

``MEDAKA_CENTROID_ANCHOR_POLICY``
    ``frame`` (default) drops a whole frame if any included instance lacks the
    anchor; ``instance`` retains the frame but removes only anchor-missing
    instances; ``disabled`` leaves the centroid dataset unmodified.

``MEDAKA_CENTERED_INSTANCE_ANCHOR_POLICY``
    ``strict`` removes anchor-missing centered-instance samples; ``disabled``
    (default) leaves the centered-instance dataset unmodified.

``MEDAKA_CENTERED_INSTANCE_JITTER_ORDER``
    ``post_affine`` applies configured translation after rotation and scale so
    its pixel bound is not amplified or rotated by those transforms; ``native``
    (default) preserves the upstream composed-affine behavior.

``MEDAKA_RECOVERY_CHECKPOINT_EVERY_VAL``
    ``1`` adds a separate full-state recovery checkpoint callback which retains
    the newest completed validation epoch. Its ``recovery-*.ckpt`` files are
    independent of the monitored ``best*.ckpt`` top-k.

Whenever either strict policy is active, the remaining-node fallback is also
disabled. A missing anchor that somehow reaches target generation stays NaN.
The strict centroid definition is mirrored in centroid evaluation.

The module accepts and forwards the normal ``sleap-nn`` CLI arguments.
"""

import inspect
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch


CENTROID_POLICY_ENV = "MEDAKA_CENTROID_ANCHOR_POLICY"
CENTERED_INSTANCE_POLICY_ENV = "MEDAKA_CENTERED_INSTANCE_ANCHOR_POLICY"
CENTERED_INSTANCE_JITTER_ORDER_ENV = (
    "MEDAKA_CENTERED_INSTANCE_JITTER_ORDER"
)
RECOVERY_CHECKPOINT_ENV = "MEDAKA_RECOVERY_CHECKPOINT_EVERY_VAL"
CENTROID_POLICIES = {"frame", "instance", "disabled"}
CENTERED_INSTANCE_POLICIES = {"strict", "disabled"}
CENTERED_INSTANCE_JITTER_ORDERS = {"native", "post_affine"}


def strict_generate_centroids(
    points: torch.Tensor, anchor_ind: int | None = None
) -> torch.Tensor:
    """Return only the configured anchor, never a remaining-node fallback."""
    if anchor_ind is None:
        raise ValueError(
            "Strict centroid-anchor policy requires a configured anchor_part."
        )
    centroids = points[..., anchor_ind, :].clone()
    invalid = ~torch.isfinite(centroids).all(dim=-1)
    return centroids.masked_fill(invalid.unsqueeze(-1), torch.nan)


def strict_compute_gt_centroids(
    instance_gt_points: np.ndarray, anchor_ind: int | None
) -> np.ndarray:
    """NumPy mirror of :func:`strict_generate_centroids` for evaluation."""
    if anchor_ind is None:
        raise ValueError(
            "Strict centroid-anchor policy requires a configured anchor_part."
        )
    points = np.asarray(instance_gt_points, dtype=np.float64)
    centroids = points[..., anchor_ind, :].copy()
    invalid = ~np.isfinite(centroids).all(axis=-1)
    centroids[invalid] = np.nan
    return centroids


def instances_have_finite_anchor(
    instances: Iterable[Any], anchor_ind: int
) -> bool:
    """Return whether every instance has a finite configured anchor."""
    for instance in instances:
        points = np.asarray(instance.numpy(), dtype=float)
        if points.ndim != 2 or anchor_ind >= points.shape[0]:
            return False
        if not np.isfinite(points[anchor_ind]).all():
            return False
    return True


def instance_has_finite_anchor(instance: Any, anchor_ind: int) -> bool:
    """Return whether one instance has a finite configured anchor."""
    points = np.asarray(instance.numpy(), dtype=float)
    return bool(
        points.ndim == 2
        and anchor_ind < points.shape[0]
        and np.isfinite(points[anchor_ind]).all()
    )


def filter_instances_with_finite_anchor(
    instances: Iterable[Any], anchor_ind: int
) -> list[Any]:
    """Return only instances with a finite configured anchor."""
    return [
        instance
        for instance in instances
        if instance_has_finite_anchor(instance, anchor_ind)
    ]


def build_grouped_epoch_eval_predictions(
    callback: Any,
    predictions: list[dict[str, Any]],
    sio: Any,
    np_module: Any,
) -> Any:
    """Build one prediction frame per source frame for centered-instance eval.

    Centered-instance validation yields one record per instance crop. SLEAP-IO
    does not merge duplicate ``LabeledFrame`` records with the same video/frame
    key, so emitting one frame per crop silently keeps only one fish during
    epoch-end evaluation. Aggregate every crop prediction into a single frame.
    """
    grouped: dict[tuple[int, int], list[Any]] = {}
    for pred in predictions:
        key = (int(pred["video_idx"]), int(pred["frame_idx"]))
        # Register every source frame before inspecting its crop predictions.
        # An empty predicted frame is required for evaluation to count a
        # complete miss and publish the worst-value checkpoint metric.
        instances = grouped.setdefault(key, [])
        pred_peaks = pred["pred_peaks"]
        pred_scores = pred.get("pred_scores")
        if pred_peaks is None:
            continue
        pred_peaks = np_module.asarray(pred_peaks)
        if pred_peaks.size == 0 or np_module.isnan(pred_peaks).all():
            continue
        if pred_peaks.ndim == 2:
            pred_peaks = pred_peaks.reshape(1, -1, 2)
            if pred_scores is not None:
                pred_scores = np_module.asarray(pred_scores).reshape(1, -1)
        elif pred_scores is not None:
            pred_scores = np_module.asarray(pred_scores)

        for inst_idx, inst_points in enumerate(pred_peaks):
            if np_module.isnan(inst_points).all():
                continue
            inst_scores = (
                pred_scores[inst_idx] if pred_scores is not None else None
            )
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=inst_points,
                    skeleton=callback.skeleton,
                    point_scores=(
                        inst_scores
                        if inst_scores is not None
                        else np_module.ones(len(inst_points))
                    ),
                    score=(
                        float(np_module.nanmean(inst_scores))
                        if inst_scores is not None
                        else 1.0
                    ),
                )
            )

    labeled_frames = [
        sio.LabeledFrame(
            video=callback.videos[video_idx],
            frame_idx=frame_idx,
            instances=instances,
        )
        for (video_idx, frame_idx), instances in grouped.items()
    ]
    return sio.Labels(
        videos=callback.videos,
        skeletons=[callback.skeleton],
        labeled_frames=labeled_frames,
    )


def build_grouped_epoch_eval_ground_truth(
    callback: Any,
    ground_truth: list[dict[str, Any]],
    sio: Any,
    np_module: Any,
) -> Any:
    """Build one ground-truth frame per source frame for instance-crop eval."""
    grouped: dict[tuple[int, int], list[Any]] = {}
    for gt in ground_truth:
        gt_instances = np_module.asarray(gt["gt_instances"])
        if gt_instances.ndim == 2:
            gt_instances = gt_instances.reshape(1, -1, 2)

        key = (int(gt["video_idx"]), int(gt["frame_idx"]))
        instances = grouped.setdefault(key, [])
        for inst_data in gt_instances[: int(gt["num_instances"])]:
            if np_module.isnan(inst_data).all():
                continue
            instances.append(
                sio.Instance.from_numpy(
                    points_data=inst_data,
                    skeleton=callback.skeleton,
                )
            )

    labeled_frames = [
        sio.LabeledFrame(
            video=callback.videos[video_idx],
            frame_idx=frame_idx,
            instances=instances,
        )
        for (video_idx, frame_idx), instances in grouped.items()
        if instances
    ]
    return sio.Labels(
        videos=callback.videos,
        skeletons=[callback.skeleton],
        labeled_frames=labeled_frames,
    )


def _policy_from_env(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def build_validation_recovery_checkpoint_callback(run_dir: Path):
    """Return a rolling full-state checkpoint independent of monitored top-k.

    The filename contains epoch and step so Lightning writes the replacement
    before deleting the previous recovery checkpoint. If a write is
    interrupted, resume selection can reject the incomplete new file and fall
    back to the prior completed recovery checkpoint.
    """
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=run_dir.as_posix(),
        filename="recovery-e{epoch:05d}-s{step:09d}",
        monitor="step",
        mode="max",
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        enable_version_counter=True,
    )


def apply_post_affine_crop_jitter(
    original_augmentation,
    image: torch.Tensor,
    instances: torch.Tensor,
    **kwargs,
):
    """Apply translation after rotation/scale without changing its pixel bound.

    Upstream Skia augmentation pre-concatenates translation with rotation and
    scale. Consequently, a nominal +/-8 px translation can become roughly
    +/-12 px with the configured +/-90 degree, 0.7--1.3 affine range. This
    decomposition preserves the same random-draw order while applying the
    translation as a final pure image/keypoint shift.
    """
    translate_p = kwargs.get("translate_p")
    if translate_p is None:
        raise ValueError(
            "Post-affine crop jitter requires an explicit translate_p."
        )
    if float(translate_p) <= 0:
        return original_augmentation(image, instances, **kwargs)

    erase_p = kwargs.get("erase_p", 0.0)
    mixup_p = kwargs.get("mixup_p", 0.0)

    affine_kwargs = dict(kwargs)
    affine_kwargs["translate_p"] = 0.0
    affine_kwargs["erase_p"] = 0.0
    affine_kwargs["mixup_p"] = 0.0
    transformed = original_augmentation(
        image, instances, **affine_kwargs
    )
    if len(transformed) == 3:
        image, instances, masks = transformed
    else:
        image, instances = transformed
        masks = None

    translate_kwargs = dict(kwargs)
    translate_kwargs.update(
        {
            "rotation_p": 0.0,
            "scale_p": 0.0,
            "translate_p": translate_p,
            "affine_p": 0.0,
            "erase_p": 0.0,
            "mixup_p": 0.0,
            "flip_p": 0.0,
            "symmetric_inds": None,
            "masks": masks,
        }
    )
    transformed = original_augmentation(
        image, instances, **translate_kwargs
    )
    if len(transformed) == 3:
        image, instances, masks = transformed
    else:
        image, instances = transformed

    # Preserve upstream ordering: flip -> affine -> erase -> mixup. The crop
    # translation is now the final affine operation, immediately before erase.
    if float(erase_p or 0.0) > 0 or float(mixup_p or 0.0) > 0:
        post_kwargs = dict(kwargs)
        post_kwargs.update(
            {
                "rotation_p": 0.0,
                "scale_p": 0.0,
                "translate_p": 0.0,
                "affine_p": 0.0,
                "erase_p": erase_p,
                "mixup_p": mixup_p,
                "flip_p": 0.0,
                "symmetric_inds": None,
                "masks": masks,
            }
        )
        transformed = original_augmentation(
            image, instances, **post_kwargs
        )
        if len(transformed) == 3:
            image, instances, masks = transformed
        else:
            image, instances = transformed

    if masks is not None:
        return image, instances, masks
    return image, instances


def install_strict_anchor_policy() -> None:
    """Patch SLEAP-NN datasets for this training process only."""
    import sleap_nn.data.custom_datasets as custom_datasets
    import sleap_nn.data.instance_centroids as instance_centroids
    import sleap_nn.evaluation as evaluation
    import sleap_nn.training.callbacks as training_callbacks
    import sleap_nn.training.model_trainer as model_trainer

    if getattr(custom_datasets, "_medaka_strict_anchor_policy", False):
        return

    centroid_policy = _policy_from_env(
        CENTROID_POLICY_ENV, "frame", CENTROID_POLICIES
    )
    centered_instance_policy = _policy_from_env(
        CENTERED_INSTANCE_POLICY_ENV,
        "disabled",
        CENTERED_INSTANCE_POLICIES,
    )
    centered_instance_jitter_order = _policy_from_env(
        CENTERED_INSTANCE_JITTER_ORDER_ENV,
        "native",
        CENTERED_INSTANCE_JITTER_ORDERS,
    )
    if centroid_policy == "disabled" and centered_instance_policy == "disabled":
        raise ValueError("train_strict_anchor.py requires at least one strict policy")
    if (
        centered_instance_jitter_order == "post_affine"
        and centered_instance_policy != "strict"
    ):
        raise ValueError(
            "Post-affine crop jitter requires strict centered-instance policy."
        )

    if centroid_policy != "disabled":
        original_centroid_init = custom_datasets.CentroidDataset.__init__
        centroid_signature = inspect.signature(original_centroid_init)

        def strict_centroid_dataset_init(self, *args, **kwargs) -> None:
            bound = centroid_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            anchor_ind = bound.arguments["anchor_ind"]
            if anchor_ind is None:
                raise ValueError(
                    "Strict centroid-anchor policy requires a non-null anchor_part "
                    "in the centroid model config."
                )

            original_centroid_init(self, *args, **kwargs)
            if centroid_policy == "instance" and self.cache_img is None:
                raise ValueError(
                    "Instance-level strict centroid filtering currently requires "
                    "a memory or disk image cache so filtered instances can be "
                    "stored per sample without modifying the source labels."
                )

            kept = []
            dropped_frames = 0
            dropped_instances = 0
            retained_instances = 0
            for sample in self.lf_idx_list:
                if sample.get("is_negative", False):
                    kept.append(sample)
                    continue

                instances = sample.get("instances")
                if instances is None:
                    labeled_frame = self.labels_list[sample["labels_idx"]][
                        sample["lf_idx"]
                    ]
                    instances = labeled_frame.instances
                instances = list(instances)

                if centroid_policy == "frame":
                    if not instances_have_finite_anchor(
                        instances, int(anchor_ind)
                    ):
                        dropped_frames += 1
                        dropped_instances += len(instances)
                        continue
                    retained_instances += len(instances)
                else:
                    filtered = filter_instances_with_finite_anchor(
                        instances, int(anchor_ind)
                    )
                    dropped_instances += len(instances) - len(filtered)
                    if not filtered:
                        dropped_frames += 1
                        continue
                    retained_instances += len(filtered)
                    # The production configs use disk caching, so storing the
                    # filtered list on the sample is sufficient and avoids
                    # modifying the source SLP. Keep this explicit rather than
                    # mutating LabeledFrame.instances.
                    sample["instances"] = filtered
                kept.append(sample)

            if not kept:
                raise ValueError(
                    "Strict centroid-anchor policy removed every training sample."
                )
            self.lf_idx_list = kept
            self.strict_anchor_policy = {
                "policy": centroid_policy,
                "anchor_ind": int(anchor_ind),
                "kept_frames": len(kept),
                "dropped_frames": dropped_frames,
                "retained_instances": retained_instances,
                "dropped_instances": dropped_instances,
            }
            print(
                "[medaka strict-anchor] "
                f"dataset=centroid; policy={centroid_policy}; "
                f"anchor_ind={anchor_ind}; kept_frames={len(kept)}; "
                f"dropped_frames={dropped_frames}; "
                f"retained_instances={retained_instances}; "
                f"dropped_instances={dropped_instances}; fallback=disabled",
                flush=True,
            )

        custom_datasets.CentroidDataset.__init__ = strict_centroid_dataset_init
        evaluation.compute_gt_centroids = strict_compute_gt_centroids

    if centered_instance_policy == "strict":
        # Upstream appends "-1" to an existing run directory whenever
        # best.ckpt exists, including a genuine Lightning resume. Keep the
        # restored ModelCheckpoint paths live and resume in place only when
        # resume_ckpt_path itself is inside the requested run directory.
        trainer_class = model_trainer.ModelTrainer
        if not getattr(trainer_class, "_medaka_resume_in_place", False):
            original_setup_ckpt_path = trainer_class._setup_ckpt_path

            def resume_safe_setup_ckpt_path(self) -> None:
                requested_run_name = self.config.trainer_config.run_name
                checkpoint_dir = self.config.trainer_config.ckpt_dir
                resume_path = self.config.trainer_config.resume_ckpt_path
                original_setup_ckpt_path(self)
                if not resume_path or not requested_run_name:
                    return
                requested_dir = (
                    Path(checkpoint_dir or ".") / str(requested_run_name)
                ).resolve()
                if Path(str(resume_path)).resolve().parent != requested_dir:
                    return
                if self.config.trainer_config.run_name != requested_run_name:
                    self.config.trainer_config.run_name = requested_run_name
                    print(
                        "[medaka strict-anchor] genuine resume: preserving "
                        f"run_name={requested_run_name} and live top-k files",
                        flush=True,
                    )

            trainer_class._setup_ckpt_path = resume_safe_setup_ckpt_path
            trainer_class._medaka_resume_in_place = True

        if os.environ.get(RECOVERY_CHECKPOINT_ENV, "0") == "1":
            if not getattr(
                trainer_class, "_medaka_validation_recovery", False
            ):
                original_setup_callbacks = (
                    trainer_class._setup_loggers_callbacks
                )

                def setup_callbacks_with_validation_recovery(
                    self, *args, **kwargs
                ):
                    loggers, callbacks = original_setup_callbacks(
                        self, *args, **kwargs
                    )
                    run_dir = (
                        Path(self.config.trainer_config.ckpt_dir)
                        / self.config.trainer_config.run_name
                    )
                    callbacks.append(
                        build_validation_recovery_checkpoint_callback(run_dir)
                    )
                    print(
                        "[medaka strict-anchor] recovery checkpoint enabled: "
                        "one full-state recovery-*.ckpt per completed "
                        "validation epoch",
                        flush=True,
                    )
                    return loggers, callbacks

                trainer_class._setup_loggers_callbacks = (
                    setup_callbacks_with_validation_recovery
                )
                trainer_class._medaka_validation_recovery = True

        # Epoch evaluation receives one record per centered-instance crop.
        # Group those records by source frame so all fish contribute to mOKS.
        epoch_eval_callback = training_callbacks.EpochEndEvaluationCallback
        if not getattr(
            epoch_eval_callback, "_medaka_groups_centered_instances", False
        ):
            epoch_eval_callback._build_pred_labels = (
                build_grouped_epoch_eval_predictions
            )
            epoch_eval_callback._build_gt_labels = (
                build_grouped_epoch_eval_ground_truth
            )
            epoch_eval_callback._medaka_groups_centered_instances = True

        if centered_instance_jitter_order == "post_affine":
            original_geometric_augmentation = (
                custom_datasets.apply_geometric_augmentation
            )

            def post_affine_geometric_augmentation(
                image, instances, **kwargs
            ):
                return apply_post_affine_crop_jitter(
                    original_geometric_augmentation,
                    image,
                    instances,
                    **kwargs,
                )

            custom_datasets.apply_geometric_augmentation = (
                post_affine_geometric_augmentation
            )

        original_centered_init = custom_datasets.CenteredInstanceDataset.__init__
        centered_signature = inspect.signature(original_centered_init)

        def strict_centered_instance_dataset_init(
            self, *args, **kwargs
        ) -> None:
            bound = centered_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            anchor_ind = bound.arguments["anchor_ind"]
            if anchor_ind is None:
                raise ValueError(
                    "Strict centered-instance policy requires a non-null "
                    "anchor_part in the model config."
                )

            original_centered_init(self, *args, **kwargs)
            kept = []
            dropped_instances = 0
            for sample in self.instance_idx_list:
                instances = sample.get("instances")
                if instances is None:
                    labeled_frame = self.labels_list[sample["labels_idx"]][
                        sample["lf_idx"]
                    ]
                    instances = labeled_frame.instances
                instance = list(instances)[sample["inst_idx"]]
                if instance_has_finite_anchor(instance, int(anchor_ind)):
                    kept.append(sample)
                else:
                    dropped_instances += 1

            if not kept:
                raise ValueError(
                    "Strict centered-instance policy removed every training sample."
                )
            self.instance_idx_list = kept
            self.strict_anchor_policy = {
                "policy": centered_instance_policy,
                "anchor_ind": int(anchor_ind),
                "kept_instances": len(kept),
                "dropped_instances": dropped_instances,
            }
            print(
                "[medaka strict-anchor] "
                "dataset=centered_instance; policy=strict; "
                f"anchor_ind={anchor_ind}; kept_instances={len(kept)}; "
                f"dropped_instances={dropped_instances}; fallback=disabled",
                flush=True,
            )

        custom_datasets.CenteredInstanceDataset.__init__ = (
            strict_centered_instance_dataset_init
        )

    # custom_datasets imported the function directly, so patch both bindings.
    instance_centroids.generate_centroids = strict_generate_centroids
    custom_datasets.generate_centroids = strict_generate_centroids
    custom_datasets._medaka_strict_anchor_policy = {
        "centroid": centroid_policy,
        "centered_instance": centered_instance_policy,
        "centered_instance_jitter_order": centered_instance_jitter_order,
    }


def main() -> None:
    """Install the strict policy and invoke the standard SLEAP-NN CLI."""
    install_strict_anchor_policy()
    centroid_policy = os.environ.get(CENTROID_POLICY_ENV, "frame")
    centered_instance_policy = os.environ.get(
        CENTERED_INSTANCE_POLICY_ENV, "disabled"
    )
    centered_instance_jitter_order = os.environ.get(
        CENTERED_INSTANCE_JITTER_ORDER_ENV, "native"
    )
    print(
        "[medaka strict-anchor] policy active: "
        f"centroid={centroid_policy}; "
        f"centered_instance={centered_instance_policy}; "
        f"jitter_order={centered_instance_jitter_order}",
        flush=True,
    )
    from sleap_nn.cli import cli

    cli()


# Windows DataLoader workers execute the launcher as ``__mp_main__``. Install
# the patch there as well so their imported ``custom_datasets.generate_centroids``
# binding cannot revert to the upstream fallback during ``__getitem__``.
if __name__ in {"__main__", "__mp_main__"}:
    install_strict_anchor_policy()


if __name__ == "__main__":
    main()
