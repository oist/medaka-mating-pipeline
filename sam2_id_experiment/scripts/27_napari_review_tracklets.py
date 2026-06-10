from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import load_config
from sam2_id_experiment.src.review.napari_tracklet_review import (
    launch_napari_review,
    read_review_table,
    write_review_template,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review segmentation tracklets in Napari.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--reference-points", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-reference-distance-px", type=float, default=120.0)
    parser.add_argument(
        "--write-template-only",
        action="store_true",
        help="Write the review CSV template without launching Napari.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    tracklets = pd.read_parquet(args.tracklets)
    reference = read_review_table(args.reference_points) if args.reference_points else None
    output_csv = Path(args.output_csv) if args.output_csv else cfg.output_dir / "napari_tracklet_review.csv"

    if args.write_template_only:
        summary = write_review_template(
            tracklets_qc=tracklets,
            output_csv=output_csv,
            reference_points=reference,
            max_reference_distance_px=float(args.max_reference_distance_px),
        )
    else:
        summary = launch_napari_review(
            video_path=cfg.video_path,
            mask_path=args.masks,
            tracklets_qc=tracklets,
            output_csv=output_csv,
            reference_points=reference,
            max_frames=args.max_frames,
            max_reference_distance_px=float(args.max_reference_distance_px),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
