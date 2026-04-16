"""Standalone entry point for training + inference of the cam0 ID network."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.id_network.infer import refine_fish_ids
from scripts.tracking.id_network.train import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train + apply an identity embedder on a cam0 per-clip CSV. "
            "Rewrites fish_id using appearance features; writes a new "
            "*_id_refined.csv next to the input."
        )
    )
    parser.add_argument(
        "id_assigned_csv",
        help="Path to a *_player1_id_assigned.csv produced by the tracking pipeline.",
    )
    parser.add_argument(
        "video",
        help="Path to the cam0 mp4 whose frame indices match the CSV's `frame` column.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--save-model", default=None, help="Where to persist the trained embedder state_dict.")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _ = load_local_paths(args.paths_config)  # reserved for future path resolution

    csv_path = Path(args.id_assigned_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    result = refine_fish_ids(
        df,
        video_path=args.video,
        crop_size=args.crop_size,
        frame_offset=args.frame_offset,
        train_config=train_config,
        min_confidence=args.min_confidence,
        save_model_path=args.save_model,
    )

    out_path = Path(args.output_csv) if args.output_csv else csv_path.with_name(
        csv_path.stem.replace("_id_assigned", "") + "_id_refined.csv"
    )
    result.df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"  cluster_to_fish={json.dumps({int(k): v for k, v in result.cluster_to_fish.items()})}")
    print(f"  confidence={result.confidence:.3f}")


if __name__ == "__main__":
    main()
