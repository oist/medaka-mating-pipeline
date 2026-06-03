from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.evaluation.overlay import make_qc_overlay_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render SAM2/QC/ID overlay video.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--assignments", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    assignments = pd.read_parquet(args.assignments) if args.assignments else None
    out_path = make_qc_overlay_video(
        video_path=cfg.video_path,
        mask_path=args.masks,
        tracklets_qc=pd.read_parquet(args.tracklets),
        assignments=assignments,
        output_path=out_dir / "qc_overlay.mp4",
        max_frames=args.max_frames,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

