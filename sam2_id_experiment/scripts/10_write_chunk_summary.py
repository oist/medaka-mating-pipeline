from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.evaluation.chunk_summary import write_chunk_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a compact JSON summary for a SAM2 ID chunk.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    summary = write_chunk_summary(output_dir=out_dir, summary_path=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
