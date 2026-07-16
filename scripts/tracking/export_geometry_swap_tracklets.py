"""Export conservative clean tracklets from geometry swap-review events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tracking.geometry_swap_annotations import export_reviewed_tracklets


def _slot_map(specs: list[str]) -> dict[str, str] | None:
    if not specs:
        return None
    result: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise argparse.ArgumentTypeError(f"--slot-map expects SOURCE=SLOT, got {spec!r}")
        source, slot = spec.split("=", 1)
        result[source] = slot
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="fish_id")
    parser.add_argument("--slot-map", action="append", default=[], metavar="SOURCE=SLOT")
    parser.add_argument("--calibration-model-dir", type=Path)
    parser.add_argument("--max-gap-frames", type=int, default=0)
    parser.add_argument("--write-constraints", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_reviewed_tracklets(
        geometry_csv=args.geometry_csv,
        events_csv=args.events_csv,
        output_dir=args.output_dir,
        calibration_model_dir=args.calibration_model_dir,
        id_column=args.id_column,
        slot_map=_slot_map(args.slot_map),
        max_gap_frames=args.max_gap_frames,
        write_constraints=args.write_constraints,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
