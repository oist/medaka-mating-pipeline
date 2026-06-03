from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import config_section, ensure_output_dir, load_config
from sam2_id_experiment.src.sam2_adapter.sam2_runner import run_sam2_video_segmentation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM2 video segmentation for a short clip.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without importing/running SAM2.")
    return parser


def _render_config_templates(value, *, cfg):
    if isinstance(value, dict):
        return {k: _render_config_templates(v, cfg=cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_config_templates(v, cfg=cfg) for v in value]
    if isinstance(value, str):
        return value.format(
            experiment_name=cfg.experiment_name or "",
            view=cfg.view or "",
            start_frame=cfg.raw.get("start_frame", ""),
            end_frame=cfg.raw.get("end_frame", ""),
        )
    return value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    masks_path = out_dir / "sam2_masks.h5"
    summary_path = out_dir / "sam2_summary.json"

    summary = {
        "video_path": str(cfg.video_path),
        "output_masks": str(masks_path),
        "start_frame": cfg.raw.get("start_frame"),
        "end_frame": cfg.raw.get("end_frame"),
        "sam2_object_ids_are_final_identity": False,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        runner_config = _render_config_templates(config_section(cfg, "sam2"), cfg=cfg)
        runner_config["opencv_bgsub"] = config_section(cfg, "opencv_bgsub")
        runner_config["start_frame"] = cfg.raw.get("start_frame")
        runner_config["end_frame"] = cfg.raw.get("end_frame")
        runner_config["n_animals_expected"] = cfg.n_animals_expected
        summary.update(run_sam2_video_segmentation(
            video_path=cfg.video_path,
            output_path=masks_path,
            config=runner_config,
        ))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
