from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.tracklets.stitch_fragments import stitch_identity_fragments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stitch assigned detector fragments into identity trajectories.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output-tracks", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-gap-frames", type=int, default=300)
    parser.add_argument("--max-speed-px-per-frame", type=float, default=5.0)
    parser.add_argument("--max-assignment-distance", type=float, default=0.35)
    parser.add_argument("--min-assignment-margin", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    output_tracks = Path(args.output_tracks) if args.output_tracks else out_dir / "identity_tracks.parquet"
    output_summary = Path(args.output_summary) if args.output_summary else out_dir / "identity_trajectory_summary.parquet"
    output_json = Path(args.output_json) if args.output_json else out_dir / "identity_stitch_summary.json"
    report = stitch_identity_fragments(
        tracklets_qc=pd.read_parquet(args.tracklets),
        assignments=pd.read_parquet(args.assignments),
        max_gap_frames=int(args.max_gap_frames),
        max_speed_px_per_frame=float(args.max_speed_px_per_frame),
        max_assignment_distance=float(args.max_assignment_distance),
        min_assignment_margin=float(args.min_assignment_margin),
        output_tracks_path=output_tracks,
        output_summary_path=output_summary,
        output_json_path=output_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
