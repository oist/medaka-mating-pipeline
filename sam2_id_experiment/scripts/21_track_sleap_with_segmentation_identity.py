from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths

from sam2_id_experiment.src.config import config_section, ensure_output_dir, load_config
from sam2_id_experiment.src.sleap_adapter.points import load_sleap_points
from sam2_id_experiment.src.tracking.hybrid_sleap_segmentation import track_sleap_with_segmentation_identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track top-camera SLEAP points with segmentation identity anchors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--segmentation-tracks", required=True)
    parser.add_argument("--slp", default=None)
    parser.add_argument("--sleap-points-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--events-csv", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--anchor-node", default=None)
    parser.add_argument(
        "--use-visible-node-centroid",
        action="store_true",
        help="Experimental: track visible-node centroids instead of the mouth node.",
    )
    parser.add_argument("--max-points-per-frame", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    tracking_cfg = config_section(cfg, "hybrid_tracking")
    if args.max_points_per_frame is not None:
        tracking_cfg["max_points_per_frame"] = int(args.max_points_per_frame)
    start_frame = cfg.raw.get("start_frame")
    end_frame = cfg.raw.get("end_frame")

    if args.sleap_points_csv:
        sleap_points = pd.read_csv(args.sleap_points_csv)
        slp_path = ""
    else:
        slp_path = str(Path(args.slp) if args.slp else _resolve_slp_path(cfg, tracking_cfg))
        sleap_points = load_sleap_points(
            slp_path,
            start_frame=int(start_frame) if start_frame is not None else None,
            end_frame=int(end_frame) if end_frame is not None else None,
            anchor_node=None if args.use_visible_node_centroid else (
                args.anchor_node or tracking_cfg.get("anchor_node") or "mouth"
            ),
        )

    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "top_cam.hybrid_identity_tracks.csv"
    events_csv = Path(args.events_csv) if args.events_csv else out_dir / "top_cam.hybrid_identity_events.csv"
    summary_json = Path(args.summary) if args.summary else out_dir / "top_cam.hybrid_identity_summary.json"
    _tracks, _events, summary = track_sleap_with_segmentation_identity(
        sleap_points=sleap_points,
        segmentation_tracks=pd.read_parquet(args.segmentation_tracks),
        config=tracking_cfg,
        output_csv=output_csv,
        events_csv=events_csv,
        summary_json=summary_json,
    )
    summary["slp_path"] = slp_path
    summary["n_sleap_points"] = int(len(sleap_points))
    summary["sleap_frame_start"] = int(sleap_points["frame"].min()) if not sleap_points.empty else None
    summary["sleap_frame_end"] = int(sleap_points["frame"].max()) if not sleap_points.empty else None
    summary["n_sleap_frames"] = int(sleap_points["frame"].nunique()) if not sleap_points.empty else 0
    summary["segmentation_tracks"] = str(args.segmentation_tracks)
    summary["output_csv"] = str(output_csv)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _resolve_slp_path(cfg, tracking_cfg: dict) -> Path:
    explicit = tracking_cfg.get("slp_path") or cfg.raw.get("slp_path")
    if explicit:
        return Path(str(explicit))
    experiment = cfg.experiment_name
    view = str(tracking_cfg.get("view") or cfg.view or "top_cam")
    if not experiment:
        raise ValueError("Resolving SLEAP .slp requires experiment_name or --slp.")
    paths_config = cfg.raw.get("paths_config")
    paths_config_path = Path(str(paths_config)) if paths_config else None
    if paths_config_path is not None and not paths_config_path.is_absolute():
        paths_config_path = cfg.path.parent.parents[1] / paths_config_path
    local = load_local_paths(paths_config_path)
    sleap_output_dir = choose_path(
        cfg.raw.get("sleap_output_dir"),
        tracking_cfg.get("sleap_output_dir"),
        local.sleap_output_dir if local else None,
    )
    if sleap_output_dir is None:
        raise ValueError("Resolving SLEAP .slp requires sleap_output_dir in config/paths.local.toml or --slp.")
    slp_dir = Path(sleap_output_dir) / str(experiment)
    subdir = tracking_cfg.get("sleap_subdir") or cfg.raw.get("sleap_subdir")
    if subdir:
        slp_dir = slp_dir / str(subdir)
    return slp_dir / f"{view}.slp"


if __name__ == "__main__":
    main()
