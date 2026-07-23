from __future__ import annotations

"""Run ``sleap-nn train`` with strict semantic-anchor supervision.

SLEAP-NN normally replaces a missing configured anchor with the mean of the
remaining visible nodes. That is useful for generic centroid localization, but
it is incorrect when the centroid head has a semantic contract such as
``anchor_part: mouth``: a tail-only instance becomes a tail centroid target.

This wrapper is deliberately process-local; it does not edit the installed
``sleap-nn`` package. For centroid training and validation it:

1. Drops an entire frame if any included instance lacks a finite anchor. This
   avoids both a tail target and dense false-background supervision at the
   location of an animal whose mouth was not labeled.
2. Disables the remaining-node fallback as a defensive check. A missing anchor
   that somehow reaches target generation stays NaN and produces no confmap.
3. Applies the same strict definition to centroid evaluation.

The module accepts and forwards the normal ``sleap-nn`` CLI arguments.
"""

import inspect
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


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


def install_strict_anchor_policy() -> None:
    """Patch SLEAP-NN's centroid dataset for this training process only."""
    import sleap_nn.data.custom_datasets as custom_datasets
    import sleap_nn.data.instance_centroids as instance_centroids
    import sleap_nn.evaluation as evaluation

    if getattr(custom_datasets, "_medaka_strict_anchor_policy", False):
        return

    original_init = custom_datasets.CentroidDataset.__init__
    signature = inspect.signature(original_init)

    def strict_centroid_dataset_init(self, *args, **kwargs) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        anchor_ind = bound.arguments["anchor_ind"]
        if anchor_ind is None:
            raise ValueError(
                "Strict centroid-anchor policy requires a non-null anchor_part "
                "in the centroid model config."
            )

        original_init(self, *args, **kwargs)

        kept = []
        dropped_frames = 0
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

            if instances_have_finite_anchor(instances, int(anchor_ind)):
                kept.append(sample)
            else:
                dropped_frames += 1

        if not kept:
            raise ValueError(
                "Strict centroid-anchor policy removed every training sample."
            )
        self.lf_idx_list = kept
        self.strict_anchor_policy = {
            "anchor_ind": int(anchor_ind),
            "kept_frames": len(kept),
            "dropped_frames": dropped_frames,
        }
        print(
            "[medaka strict-anchor] "
            f"anchor_ind={anchor_ind}; kept_frames={len(kept)}; "
            f"dropped_frames={dropped_frames}; fallback=disabled",
            flush=True,
        )

    # custom_datasets imported the function directly, so patch both bindings.
    instance_centroids.generate_centroids = strict_generate_centroids
    custom_datasets.generate_centroids = strict_generate_centroids
    custom_datasets.CentroidDataset.__init__ = strict_centroid_dataset_init
    custom_datasets._medaka_strict_anchor_policy = True
    evaluation.compute_gt_centroids = strict_compute_gt_centroids


def main() -> None:
    """Install the strict policy and invoke the standard SLEAP-NN CLI."""
    install_strict_anchor_policy()
    print(
        "[medaka strict-anchor] policy active: frames with any missing "
        "configured anchor are excluded from centroid training/validation",
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
