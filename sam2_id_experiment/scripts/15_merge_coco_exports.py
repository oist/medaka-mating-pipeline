from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.detectron2_adapter.coco_merge import merge_coco_exports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge multiple exported COCO directories.")
    parser.add_argument("--coco-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = merge_coco_exports(coco_dirs=[Path(path) for path in args.coco_dir], output_dir=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
