from __future__ import annotations

"""Train a geometry-supervised appearance embedding without identity classes.

Production example (frozen ImageNet backbone, 32-D projection)::

    .venv/Scripts/python.exe -m \
      sam2_id_experiment.scripts.49_train_geometry_contrastive_embedding \
      --dataset runs/geometry_clean_embedding_v1 \
      --output-root runs/appearance_embeddings \
      --run-name geometry_clean_frozen_d32 --embedding-dim 32 --mode frozen

CPU-only end-to-end contract/checkpoint smoke test::

    .venv/Scripts/python.exe -m \
      sam2_id_experiment.scripts.49_train_geometry_contrastive_embedding \
      --synthetic-smoke --output-root .tmp/geometry_embedding_smoke
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.id_model.geometry_contrastive import (
    GeometryContrastiveConfig,
    prepare_run_directory,
    train_geometry_contrastive,
)


def write_synthetic_smoke_contract(dataset_dir: str | Path, *, seed: int = 13) -> Path:
    """Create a tiny deterministic dataset for CPU integration testing.

    Colours are deliberately easy; the result tests plumbing and provenance, not
    biological generalisation. It contains nine whole temporal blocks, three local
    fish per frame, and two cameras. Local fish numbers are used only to construct
    explicit constraints and are not written as classifier targets.
    """
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    rows: list[dict] = []
    sample_lookup: dict[tuple[int, int, int, str], int] = {}
    sample_id = 0
    views = ("top_cam", "side1_cam")
    colours = np.asarray([[210, 60, 35], [40, 190, 70], [45, 80, 215]], dtype=np.uint8)
    for block in range(9):
        for offset in (0, 4, 8, 12):
            frame = block * 100 + offset
            for view_idx, view in enumerate(views):
                for local_fish in range(3):
                    image = np.zeros((64, 64, 3), dtype=np.uint8)
                    yy, xx = np.ogrid[:64, :64]
                    cx = 23 + local_fish * 9 + view_idx * 2
                    cy = 30 + (block % 3) * 2
                    mask = ((xx - cx) / 14) ** 2 + ((yy - cy) / 6) ** 2 <= 1
                    jitter = rng.integers(-5, 6, size=3)
                    image[mask] = np.clip(colours[local_fish].astype(int) + jitter, 0, 255)
                    crop_index = len(images)
                    images.append(image)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "crop_index": crop_index,
                            "frame": frame,
                            "view": view,
                            "split_block_id": f"block_{block:02d}",
                            "fragment_id": f"block_{block:02d}:{view}:{local_fish}",
                        }
                    )
                    sample_lookup[(block, offset, local_fish, view)] = sample_id
                    sample_id += 1

    positives: list[dict] = []
    negatives: list[dict] = []
    for block in range(9):
        # Same-fragment positives are non-adjacent in time.
        for view in views:
            for fish in range(3):
                positives.append(
                    {
                        "sample_a_id": sample_lookup[(block, 0, fish, view)],
                        "sample_b_id": sample_lookup[(block, 12, fish, view)],
                        "pair_type": "same_fragment",
                    }
                )
        # Synchronized, geometry-associated cross-camera positives and exact
        # simultaneous cannot-links.
        for offset in (0, 4, 8, 12):
            for fish in range(3):
                positives.append(
                    {
                        "sample_a_id": sample_lookup[(block, offset, fish, views[0])],
                        "sample_b_id": sample_lookup[(block, offset, fish, views[1])],
                        "pair_type": "cross_camera",
                    }
                )
            for view in views:
                for fish_a, fish_b in ((0, 1), (0, 2), (1, 2)):
                    negatives.append(
                        {
                            "sample_a_id": sample_lookup[(block, offset, fish_a, view)],
                            "sample_b_id": sample_lookup[(block, offset, fish_b, view)],
                            "pair_type": "simultaneous_frame",
                        }
                    )

    with h5py.File(root / "crops.h5", "w") as h5:
        crops = h5.create_dataset("crops", data=np.stack(images), compression="gzip")
        crops.attrs["color_order"] = "RGB"
    pd.DataFrame(rows).to_parquet(root / "samples.parquet", index=False)
    pd.DataFrame(positives).to_parquet(root / "positive_pairs.parquet", index=False)
    pd.DataFrame(negatives).to_parquet(root / "negative_pairs.parquet", index=False)
    (root / "README.txt").write_text(
        "Synthetic geometry-contrastive smoke data only; not biological training data.\n",
        encoding="utf-8",
    )
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a non-clustering ResNet18 metric embedding from explicit geometry constraints."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="directory with crops.h5, samples, and explicit pair tables")
    source.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="generate a tiny black-background RGB contract and run one CPU smoke epoch",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedding-dim", type=int, choices=(8, 32, 128), default=128)
    parser.add_argument("--mode", choices=("frozen", "fine-tune"), default="frozen")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-margin", type=float, default=1.0)
    parser.add_argument("--min-positive-frame-gap", type=int, default=15)
    parser.add_argument("--split-fractions", type=float, nargs=3, metavar=("TRAIN", "CAL", "TEST"), default=(0.70, 0.15, 0.15))
    parser.add_argument("--block-column", default=None)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--temporal-block-frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="CPU, one epoch, random-init, at most two train steps")
    parser.add_argument("--allow-nonsimultaneous-negatives", action="store_true")
    parser.add_argument("--allow-missing-eval-splits", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.synthetic_smoke and args.resume:
        raise SystemExit("--synthetic-smoke cannot be combined with --resume")
    smoke = bool(args.smoke or args.synthetic_smoke)
    run_name = args.run_name or ("synthetic_cpu_smoke" if args.synthetic_smoke else None)
    run_dir = prepare_run_directory(args.output_root, run_name, resume=args.resume)
    dataset = Path(args.dataset) if args.dataset else write_synthetic_smoke_contract(
        run_dir / "synthetic_dataset", seed=args.seed
    )
    fractions = tuple(float(x) for x in args.split_fractions)
    config = GeometryContrastiveConfig(
        embedding_dim=args.embedding_dim,
        train_backbone=args.mode == "fine-tune",
        pretrained=args.pretrained,
        normalize_embeddings=args.normalize_embeddings,
        epochs=1 if smoke else args.epochs,
        batch_size=min(args.batch_size, 12) if smoke else args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        negative_margin=args.negative_margin,
        min_positive_frame_gap=1 if args.synthetic_smoke else args.min_positive_frame_gap,
        train_fraction=fractions[0],
        calibration_fraction=fractions[1],
        test_fraction=fractions[2],
        temporal_block_frames=args.temporal_block_frames,
        seed=args.seed,
        augment=args.augment,
        num_workers=0 if smoke else args.num_workers,
        max_steps_per_epoch=args.max_steps_per_epoch,
        smoke=smoke,
    )
    report = train_geometry_contrastive(
        dataset_dir=dataset,
        run_dir=run_dir,
        config=config,
        device="cpu" if smoke else args.device,
        block_column=args.block_column,
        split_column=args.split_column,
        strict_simultaneous_negatives=not args.allow_nonsimultaneous_negatives,
        resume=args.resume,
        allow_missing_eval_splits=args.allow_missing_eval_splits,
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), "status": report["status"], "evaluation": report["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
