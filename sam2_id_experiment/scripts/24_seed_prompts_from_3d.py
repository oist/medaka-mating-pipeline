from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration

from sam2_id_experiment.src.sam2_adapter.prompt_seed_from_3d import (
    mine_3d_seed_prompts,
    write_seed_mining_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine clean 3D identity frames and write SAM2 point prompts."
    )
    parser.add_argument("--identified-3d-csv", required=True, type=Path)
    parser.add_argument("--best-pre-link-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--calibration-model-dir", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-individuals", type=int, default=3)
    parser.add_argument("--min-separation-mm", type=float, default=20.0)
    parser.add_argument("--max-speed-mm-per-frame", type=float, default=8.0)
    parser.add_argument("--max-residual-px2", type=float, default=5000.0)
    parser.add_argument("--max-bind-score-px", type=float, default=60.0)
    parser.add_argument("--allow-geometry-only", action="store_true")
    parser.add_argument("--min-window-frames", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=25)
    parser.add_argument("--negative-points-per-other-fish", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)
    model_dir = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if model_dir is None:
        raise SystemExit("calibration_model_dir required (paths.local.toml or --calibration-model-dir)")

    cfg = {
        "num_individuals": int(args.num_individuals),
        "min_separation_mm": float(args.min_separation_mm),
        "max_speed_mm_per_frame": float(args.max_speed_mm_per_frame),
        "max_residual_px2": float(args.max_residual_px2),
        "max_bind_score_px": float(args.max_bind_score_px),
        "require_top_bind": not bool(args.allow_geometry_only),
        "min_window_frames": int(args.min_window_frames),
        "max_windows": int(args.max_windows),
        "negative_points_per_other_fish": int(args.negative_points_per_other_fish),
    }
    result = mine_3d_seed_prompts(
        identified_3d=args.identified_3d_csv,
        best_pre_link=args.best_pre_link_json,
        calib=Calibration(model_dir),
        config=cfg,
        start_frame=args.start_frame,
    )
    paths = write_seed_mining_outputs(result, output_dir=args.output_dir)
    print(json.dumps({**paths, **result.summary}, indent=2))


if __name__ == "__main__":
    main()
