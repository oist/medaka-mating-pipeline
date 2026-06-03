from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.tracklets.link_detections import link_mask_detections_h5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Link per-frame Detectron2 masks into provisional tracklets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output-masks", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--max-frame-gap", type=int, default=20)
    parser.add_argument("--max-centroid-distance-px", type=float, default=150.0)
    parser.add_argument("--min-iou", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    output_masks = Path(args.output_masks) if args.output_masks else out_dir / "detectron2_linked_masks.h5"
    summary_path = Path(args.summary) if args.summary else out_dir / "detectron2_link_summary.json"
    summary = link_mask_detections_h5(
        input_masks=args.masks,
        output_masks=output_masks,
        max_frame_gap=int(args.max_frame_gap),
        max_centroid_distance_px=float(args.max_centroid_distance_px),
        min_iou=float(args.min_iou),
        summary_path=summary_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
