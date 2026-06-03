from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate SAM2-ID experiment config.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)

    summary = {
        "config": str(cfg.path),
        "video_path": str(cfg.video_path),
        "video_exists": cfg.video_path.exists(),
        "output_dir": str(out_dir),
        "n_animals_expected": cfg.n_animals_expected,
        "sam2_object_ids_are_final_identity": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

