from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.sam2_adapter.prompt_seed import seed_prompts_from_mask_h5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed a new SAM2 chunk config from masks in a previous chunk."
    )
    parser.add_argument("--source-masks", required=True, help="Previous chunk sam2_masks.h5.")
    parser.add_argument("--target-config", required=True, help="Config to receive seeded prompts.")
    parser.add_argument(
        "--source-frame",
        type=int,
        default=None,
        help="Absolute source frame in the previous masks. Defaults to target config start_frame.",
    )
    parser.add_argument("--target-local-frame", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--points-per-mask", type=int, default=3)
    parser.add_argument("--output", default=None, help="Prompt YAML output path.")
    parser.add_argument("--update-config", action="store_true")
    parser.add_argument(
        "--replace-prompts",
        action="store_true",
        help="Replace target config sam2.prompts instead of appending.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    target_config = Path(args.target_config)
    cfg = load_config(target_config)
    out_dir = ensure_output_dir(cfg)
    source_frame = int(args.source_frame if args.source_frame is not None else cfg.raw.get("start_frame", 0))
    output_path = Path(args.output) if args.output else out_dir / "sam2_seed_prompts.yaml"

    prompts, selected_frame = seed_prompts_from_mask_h5(
        args.source_masks,
        source_frame=source_frame,
        target_local_frame=int(args.target_local_frame),
        min_area=int(args.min_area),
        points_per_mask=int(args.points_per_mask),
    )
    if not prompts:
        raise RuntimeError(
            f"No usable masks found near source frame {source_frame} in {args.source_masks}."
        )

    payload = {"sam2": {"runner": "sam2_video_predictor", "prompts": prompts}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    if args.update_config:
        with target_config.open("r", encoding="utf-8") as fh:
            config_data = yaml.safe_load(fh) or {}
        sam2 = dict(config_data.get("sam2") or {})
        existing = [] if args.replace_prompts else list(sam2.get("prompts") or [])
        sam2["runner"] = "sam2_video_predictor"
        sam2["prompt_mode"] = "manual_initial_frame"
        sam2["prompts"] = existing + prompts
        config_data["sam2"] = sam2
        target_config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    summary = {
        "source_masks": str(args.source_masks),
        "requested_source_frame": source_frame,
        "selected_source_frame": selected_frame,
        "target_config": str(target_config),
        "target_local_frame": int(args.target_local_frame),
        "n_prompt_objects": len(prompts),
        "prompt_yaml": str(output_path),
        "updated_config": bool(args.update_config),
        "replace_prompts": bool(args.replace_prompts),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
