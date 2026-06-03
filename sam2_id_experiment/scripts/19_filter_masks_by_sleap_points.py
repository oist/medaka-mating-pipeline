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
from sam2_id_experiment.src.sleap_adapter.mask_gate import filter_masks_by_sleap_points_h5
from sam2_id_experiment.src.sleap_adapter.points import load_sleap_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter segmentation masks by top-camera SLEAP point coincidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--slp", default=None, help="Explicit top-camera .slp path.")
    parser.add_argument("--sleap-points-csv", default=None, help="Optional pre-exported SLEAP points CSV.")
    parser.add_argument("--output-masks", default=None)
    parser.add_argument("--output-table", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--anchor-node", default=None)
    parser.add_argument(
        "--use-visible-node-centroid",
        action="store_true",
        help="Experimental: validate masks against visible-node centroids instead of the mouth node.",
    )
    parser.add_argument("--include-manual-review", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    gate_cfg = config_section(cfg, "sleap_gate")
    start_frame = cfg.raw.get("start_frame")
    end_frame = cfg.raw.get("end_frame")

    if args.sleap_points_csv:
        points = pd.read_csv(args.sleap_points_csv)
        slp_path = ""
    else:
        slp_path = str(Path(args.slp) if args.slp else _resolve_slp_path(cfg, gate_cfg))
        points = load_sleap_points(
            slp_path,
            start_frame=int(start_frame) if start_frame is not None else None,
            end_frame=int(end_frame) if end_frame is not None else None,
            anchor_node=None if args.use_visible_node_centroid else (
                args.anchor_node or gate_cfg.get("anchor_node") or "mouth"
            ),
        )

    output_masks = Path(args.output_masks) if args.output_masks else out_dir / "sleap_filtered_detectron2_masks.h5"
    output_table = Path(args.output_table) if args.output_table else out_dir / "mask_sleap_coincidence.parquet"
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "mask_sleap_coincidence.csv"
    summary_path = Path(args.summary) if args.summary else out_dir / "mask_sleap_coincidence_summary.json"

    summary = filter_masks_by_sleap_points_h5(
        mask_path=args.masks,
        sleap_points=points,
        output_masks=output_masks,
        output_table=output_table,
        output_csv=output_csv,
        summary_path=summary_path,
        config=gate_cfg,
        include_manual_review=bool(args.include_manual_review),
    )
    summary["slp_path"] = slp_path
    summary["n_sleap_points"] = int(len(points))
    summary["output_masks"] = str(output_masks)
    summary["output_table"] = str(output_table)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _resolve_slp_path(cfg, gate_cfg: dict) -> Path:
    explicit = gate_cfg.get("slp_path") or cfg.raw.get("slp_path")
    if explicit:
        return Path(str(explicit))
    experiment = cfg.experiment_name
    view = str(gate_cfg.get("view") or cfg.view or "top_cam")
    if not experiment:
        raise ValueError("Resolving SLEAP .slp requires experiment_name or --slp.")
    paths_config = cfg.raw.get("paths_config")
    paths_config_path = Path(str(paths_config)) if paths_config else None
    if paths_config_path is not None and not paths_config_path.is_absolute():
        paths_config_path = cfg.path.parent.parents[1] / paths_config_path
    local = load_local_paths(paths_config_path)
    sleap_output_dir = choose_path(
        cfg.raw.get("sleap_output_dir"),
        gate_cfg.get("sleap_output_dir"),
        local.sleap_output_dir if local else None,
    )
    if sleap_output_dir is None:
        raise ValueError("Resolving SLEAP .slp requires sleap_output_dir in config/paths.local.toml or --slp.")
    slp_dir = Path(sleap_output_dir) / str(experiment)
    subdir = gate_cfg.get("sleap_subdir") or cfg.raw.get("sleap_subdir")
    if subdir:
        slp_dir = slp_dir / str(subdir)
    return slp_dir / f"{view}.slp"


if __name__ == "__main__":
    main()
