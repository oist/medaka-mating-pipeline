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
    project_identified_3d_to_view,
    write_seed_mining_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine clean 3D identity frames and write SAM2 point prompts."
    )
    parser.add_argument("--identified-3d-csv", required=True, type=Path)
    parser.add_argument("--best-pre-link-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--view",
        default="top_cam",
        choices=["top_cam", "side1_cam", "side2_cam"],
        help="Camera view for prompt reprojection.",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        type=Path,
        help="Optional output CSV with projected frame,x,y,identity reference rows.",
    )
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
    calib = Calibration(model_dir)
    result = mine_3d_seed_prompts(
        identified_3d=args.identified_3d_csv,
        best_pre_link=args.best_pre_link_json,
        calib=calib,
        config=cfg,
        start_frame=args.start_frame,
        view=args.view,
    )
    paths = write_seed_mining_outputs(result, output_dir=args.output_dir)
    if args.reference_csv is not None:
        reference = project_identified_3d_to_view(
            args.identified_3d_csv,
            calib=calib,
            view=args.view,
        )
        args.reference_csv.parent.mkdir(parents=True, exist_ok=True)
        reference.to_csv(args.reference_csv, index=False)
        paths["reference_csv"] = str(args.reference_csv)
    print(json.dumps({**paths, **result.summary}, indent=2))


if __name__ == "__main__":
    main()
