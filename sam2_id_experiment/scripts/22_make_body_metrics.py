from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.sleap_adapter.body_metrics import (
    body_metrics_summary,
    compute_body_metrics_for_tracks,
    load_sleap_body_points,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attach tail/body-axis metrics to already tracked mouth identities. "
            "This does not change identity assignments."
        )
    )
    parser.add_argument("--tracks", required=True, help="Mouth-anchor track CSV.")
    parser.add_argument("--slp", required=True, help="SLEAP .slp with mouth/tail nodes.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--mouth-node", default="mouth")
    parser.add_argument("--tail-node", default="tail")
    parser.add_argument("--max-mouth-match-distance-px", type=float, default=15.0)
    parser.add_argument("--min-body-length-px", type=float, default=2.0)
    parser.add_argument("--max-body-length-px", type=float, default=250.0)
    parser.add_argument("--tail-near-other-mouth-px", type=float, default=5.0)
    parser.add_argument("--smooth-window", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tracks = pd.read_csv(args.tracks)
    if args.start_frame is not None:
        tracks = tracks[tracks["frame"].astype(int) >= int(args.start_frame)].copy()
    if args.end_frame is not None:
        tracks = tracks[tracks["frame"].astype(int) <= int(args.end_frame)].copy()

    body_points = load_sleap_body_points(
        args.slp,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        mouth_node=args.mouth_node,
        tail_node=args.tail_node,
    )
    metrics = compute_body_metrics_for_tracks(
        tracks=tracks,
        body_points=body_points,
        max_mouth_match_distance_px=args.max_mouth_match_distance_px,
        min_body_length_px=args.min_body_length_px,
        max_body_length_px=args.max_body_length_px,
        tail_near_other_mouth_px=args.tail_near_other_mouth_px,
        smooth_window=args.smooth_window,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    summary = body_metrics_summary(metrics)
    summary.update({
        "tracks": str(args.tracks),
        "slp": str(args.slp),
        "output_csv": str(output_csv),
        "mouth_node": args.mouth_node,
        "tail_node": args.tail_node,
    })
    summary_path = Path(args.summary) if args.summary else output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
