from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import load_config
from sam2_id_experiment.src.detectron2_adapter.coco_export import export_sam2_masks_to_coco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export QC-approved SAM2 masks as COCO instance labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--statuses", default="clean_train")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    statuses = [part.strip() for part in str(args.statuses).split(",") if part.strip()]
    result = export_sam2_masks_to_coco(
        video_path=cfg.video_path,
        mask_path=args.masks,
        tracklets_qc_path=args.tracklets,
        output_dir=args.output,
        statuses=statuses,
        max_images=args.max_images,
        frame_stride=max(1, int(args.frame_stride)),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
