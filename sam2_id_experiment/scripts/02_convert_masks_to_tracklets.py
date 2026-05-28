from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import ensure_output_dir, load_config
from sam2_id_experiment.src.tracklets.mask_to_tracklet import masks_to_instance_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SAM2 masks into raw candidate tracklets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)
    out_path = Path(args.output) if args.output else out_dir / "tracklets_raw.parquet"
    df = masks_to_instance_dataframe(args.masks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} mask instances to {out_path}")


if __name__ == "__main__":
    main()

