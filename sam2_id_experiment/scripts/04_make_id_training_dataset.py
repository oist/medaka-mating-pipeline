from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import config_section, ensure_output_dir, load_config
from sam2_id_experiment.src.id_dataset.crop_extractor import extract_id_crops
from sam2_id_experiment.src.id_dataset.dataset_writer import write_pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ID crops and pair labels from clean tracklets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--masks", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg) / "id_training_dataset"
    ds_cfg = config_section(cfg, "id_dataset")
    debug_cfg = config_section(cfg, "debug")

    tracklets = pd.read_parquet(args.tracklets)
    crops_path, meta_path = extract_id_crops(
        video_path=cfg.video_path,
        mask_path=args.masks,
        tracklets_qc=tracklets,
        output_dir=out_dir,
        crop_size=int(ds_cfg.get("crop_size", 128)),
        max_crops_per_tracklet=int(ds_cfg.get("max_crops_per_tracklet", 500)),
        debug_first_n=100 if bool(debug_cfg.get("write_rejected_examples", False)) else 0,
    )
    pairs_path = write_pairs(meta_path, out_dir, ds_cfg)
    print(f"Wrote {crops_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {pairs_path}")


if __name__ == "__main__":
    main()

