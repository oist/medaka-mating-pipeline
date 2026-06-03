from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.detectron2_adapter.infer import (
    run_detectron2_image_dir_inference,
    run_detectron2_video_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fine-tuned Detectron2 instance segmentation on a video chunk.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--detectron-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-masks", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--output-width", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    output_masks = Path(args.output_masks) if args.output_masks else out_dir / "detectron2_masks.h5"
    frame_stride = int(args.frame_stride or (cfg.raw.get("sam2") or {}).get("frame_stride", 1))
    if args.image_dir:
        output_shape = None
        if args.output_height is not None or args.output_width is not None:
            if args.output_height is None or args.output_width is None:
                raise ValueError("Pass both --output-height and --output-width.")
            output_shape = (int(args.output_height), int(args.output_width))
        result = run_detectron2_image_dir_inference(
            image_dir=args.image_dir,
            config_path=args.detectron_config,
            model_path=args.model,
            output_masks=output_masks,
            start_frame=int(cfg.raw.get("start_frame", 0)),
            frame_stride=frame_stride,
            score_thresh=float(args.score_thresh),
            device=str(args.device),
            output_shape=output_shape,
        )
    else:
        result = run_detectron2_video_inference(
            video_path=Path(args.video_path) if args.video_path else cfg.video_path,
            config_path=args.detectron_config,
            model_path=args.model,
            output_masks=output_masks,
            start_frame=int(cfg.raw.get("start_frame", 0)),
            end_frame=int(cfg.raw.get("end_frame", 0)),
            frame_stride=frame_stride,
            score_thresh=float(args.score_thresh),
            device=str(args.device),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
