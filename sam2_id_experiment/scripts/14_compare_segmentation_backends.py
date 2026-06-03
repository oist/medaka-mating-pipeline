from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.evaluation.backend_compare import write_backend_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SAM2 and Detectron2 segmentation outputs for a chunk.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-masks", required=True)
    parser.add_argument("--candidate-masks", default=None)
    parser.add_argument("--tracklets", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    report = write_backend_comparison(
        output_dir=out_dir,
        reference_masks=args.reference_masks,
        candidate_masks=args.candidate_masks,
        tracklets_qc=args.tracklets,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
