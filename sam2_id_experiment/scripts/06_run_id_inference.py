from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.id_model.infer import embed_crops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed crops with a trained ID model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg) / "id_inference"
    result = embed_crops(
        dataset_dir=args.dataset,
        model_path=args.model,
        output_dir=out_dir,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

