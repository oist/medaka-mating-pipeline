from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.production.side_camera_pipeline import (
    SidePilotSpec,
    build_side_pilot_commands,
    write_side_pilot_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write side-camera pilot configs and a command manifest."
    )
    parser.add_argument(
        "--template-config",
        default="sam2_id_experiment/configs/sam2_medaka_test_chunk2.yaml",
        help="Top-camera config to copy shared SAM2/QC/ID settings from.",
    )
    parser.add_argument("--output-dir", default="sam2_id_experiment/configs/side_camera_pilot")
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--start-frame", type=int, default=176300)
    parser.add_argument("--end-frame", type=int, default=181300)
    parser.add_argument("--sleap-subdir", default="round4_260520")
    parser.add_argument("--paths-config", default="configs/paths.local.toml")
    parser.add_argument("--side-anchor-node", default=None)
    parser.add_argument("--identified-3d-csv", required=True)
    parser.add_argument("--best-pre-link-json", required=True)
    parser.add_argument(
        "--detectron-output-dir",
        default="outputs/20251225_K_1_2_FCMC/side_detectron2_model",
    )
    parser.add_argument("--manifest", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = SidePilotSpec(
        experiment_name=str(args.experiment),
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        sleap_subdir=str(args.sleap_subdir),
        paths_config=str(args.paths_config),
    )
    config_paths = write_side_pilot_configs(
        template_config=args.template_config,
        output_dir=args.output_dir,
        spec=spec,
        side_anchor_node=args.side_anchor_node,
    )
    commands = build_side_pilot_commands(
        config_paths=config_paths,
        identified_3d_csv=args.identified_3d_csv,
        best_pre_link_json=args.best_pre_link_json,
        detectron_output_dir=args.detectron_output_dir,
    )
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output_dir) / "side_camera_pilot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": {
            "experiment_name": spec.experiment_name,
            "start_frame": spec.start_frame,
            "end_frame": spec.end_frame,
            "sleap_subdir": spec.sleap_subdir,
            "paths_config": spec.paths_config,
        },
        "config_paths": config_paths,
        "commands": commands,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **payload}, indent=2))


if __name__ == "__main__":
    main()
