from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.sleap_adapter.top_track_export import export_segmentation_top_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export stitched segmentation identities as a top-camera track CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--identity-tracks", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--node-name", default="segmentation_centroid")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "top_cam.segmentation_identity_tracks.csv"
    summary_path = Path(args.summary) if args.summary else out_dir / "top_cam.segmentation_identity_tracks_summary.json"
    summary = export_segmentation_top_tracks(
        identity_tracks=pd.read_parquet(args.identity_tracks),
        output_csv=output_csv,
        output_summary=summary_path,
        node_name=str(args.node_name),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
