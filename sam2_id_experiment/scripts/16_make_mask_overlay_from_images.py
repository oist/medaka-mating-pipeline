from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.evaluation.overlay import make_mask_overlay_video_from_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a sampled-frame mask overlay from cached JPEG frames.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--label-prefix", default="mask")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--assignments", default=None)
    parser.add_argument("--tracks", default=None)
    parser.add_argument("--points", default=None, help="Optional CSV/parquet with frame,x,y or matched_sleap_x/y columns.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    frame_stride = int(args.frame_stride or (cfg.raw.get("sam2") or {}).get("frame_stride", 1))
    output = Path(args.output) if args.output else out_dir / "mask_overlay_sampled.mp4"
    out_path = make_mask_overlay_video_from_images(
        image_dir=args.image_dir,
        mask_path=args.masks,
        output_path=output,
        start_frame=int(cfg.raw.get("start_frame", 0)),
        frame_stride=frame_stride,
        fps=float(args.fps),
        label_prefix=args.label_prefix,
        max_frames=args.max_frames,
        assignments=pd.read_parquet(args.assignments) if args.assignments else None,
        tracks=pd.read_parquet(args.tracks) if args.tracks else None,
        points=_read_table(args.points) if args.points else None,
    )
    print(f"Wrote {out_path}")


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


if __name__ == "__main__":
    main()
