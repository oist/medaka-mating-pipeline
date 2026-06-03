from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.detectron2_adapter.train import train_detectron2_medaka


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Detectron2 Mask R-CNN on medaka COCO labels.")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--base-lr", type=float, default=0.00025)
    parser.add_argument("--ims-per-batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = train_detectron2_medaka(
        train_json=args.train_json,
        train_images=args.train_images,
        output_dir=args.output,
        val_json=args.val_json,
        val_images=args.val_images,
        max_iter=int(args.max_iter),
        base_lr=float(args.base_lr),
        ims_per_batch=int(args.ims_per_batch),
        num_workers=int(args.num_workers),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
