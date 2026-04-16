from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.file_set_generator import generate_file_sets, save_file_sets
from scripts.tracking.process_tracking import (
    process_from_csv_path,
    process_from_file_set_key,
)
from scripts.tracking.process_tracking_simple import (
    process_from_csv_path_simple,
    process_from_file_set_key_simple,
)
from scripts.tracking.slp_merge import merge_experiment


def _looks_like_csv_path(value: str) -> bool:
    lowered = value.lower()
    return lowered.endswith(".csv") or "\\" in value or "/" in value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2D tracking post-processing from SLEAP CSV exports."
    )
    parser.add_argument(
        "input",
        help="CSV path, set key, or experiment folder name (with --merge-chunks).",
    )
    parser.add_argument(
        "--paths-config",
        default=None,
        help="Optional local TOML path config. Defaults to configs/paths.local.toml if present.",
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        help=(
            "Treat `input` as an experiment folder name (e.g. 20251220_D_1_1_FCMC), "
            "merge the per-view chunk .slp files under sleap_output_dir/<input>/ into "
            "a single .slp per view, and write experiment_metadata.json. "
            "Skips views whose merged file is already newer than every chunk."
        ),
    )
    parser.add_argument(
        "--force-merge",
        action="store_true",
        help="With --merge-chunks, re-merge even if the merged file is fresh.",
    )
    parser.add_argument(
        "--medaka-data-dir",
        default=None,
        help=(
            "Override medaka_data_dir (raw recordings root) for --merge-chunks. "
            "Used to read per-camera metadata.csv."
        ),
    )
    parser.add_argument(
        "--sleap-output-dir",
        default=None,
        help="Root directory containing top_mouth_* and side_mouth_* exports.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for processed CSVs, logs, and visualizations.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use the simplified pipeline without outlier or mirror filtering.",
    )
    parser.add_argument("--max-distance", type=float, default=100.0)
    parser.add_argument("--flow-weight", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=166)
    parser.add_argument("--interp-method", default="linear", choices=["linear", "cubic"])
    parser.add_argument("--interp-limit", type=int, default=None)
    parser.add_argument("--no-smooth", action="store_true", help="Disable trajectory smoothing.")
    parser.add_argument("--no-viz", action="store_true", help="Disable trajectory visualizations.")
    parser.add_argument("--max-jump-distance", type=float, default=200.0)
    parser.add_argument("--max-jump-per-frame", type=float, default=100.0)
    parser.add_argument(
        "--skip-2d-interpolation",
        action="store_true",
        help="Leave 2D gaps as NaN for later 3D-stage interpolation.",
    )
    parser.add_argument(
        "--no-mirror-filter",
        action="store_true",
        help="Disable in-view mirror filtering in the full pipeline.",
    )
    parser.add_argument(
        "--use-id-network",
        action="store_true",
        help=(
            "After Hungarian ID assignment, train a per-clip contrastive "
            "ResNet18 on cam0 crops and rewrite fish_id from its embedding. "
            "Requires medaka_data_dir to resolve cam0_top.mp4."
        ),
    )
    parser.add_argument("--id-network-epochs", type=int, default=5)
    parser.add_argument("--id-network-batch-size", type=int, default=64)
    parser.add_argument("--id-network-lr", type=float, default=1e-3)
    parser.add_argument("--id-network-device", default="cuda")
    parser.add_argument("--id-network-min-confidence", type=float, default=0.6)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    local_paths = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local_paths.sleap_output_dir if local_paths else None,
    )
    output_dir = choose_path(
        args.output_dir,
        local_paths.tracking_output_dir if local_paths else None,
    )

    if sleap_output_dir is None:
        parser.error(
            "SLEAP output root is required. Pass --sleap-output-dir or define "
            "sleap_output_dir in configs/paths.local.toml."
        )

    if args.merge_chunks:
        medaka_data_dir = choose_path(
            args.medaka_data_dir,
            local_paths.medaka_data_dir if local_paths else None,
        )
        if medaka_data_dir is None:
            parser.error(
                "--merge-chunks needs medaka_data_dir (the manifest lives in "
                "the raw recording folder). Pass --medaka-data-dir or set "
                "medaka_data_dir in configs/paths.local.toml."
            )
        merge_experiment(
            experiment_name=args.input,
            sleap_output_dir=sleap_output_dir,
            raw_data_dir=medaka_data_dir,
            force=args.force_merge,
        )
        return

    common_kwargs = {
        "output_dir": output_dir,
        "max_distance": args.max_distance,
        "flow_weight": args.flow_weight,
        "fps": args.fps,
        "interp_method": args.interp_method,
        "interp_limit": args.interp_limit,
        "smooth": not args.no_smooth,
        "visualize": not args.no_viz,
    }

    if args.simple:
        runner_csv = process_from_csv_path_simple
        runner_key = process_from_file_set_key_simple
        extra_kwargs = {}
    else:
        runner_csv = process_from_csv_path
        runner_key = process_from_file_set_key
        extra_kwargs = {
            "max_jump_distance": args.max_jump_distance,
            "max_jump_per_frame": args.max_jump_per_frame,
            "skip_2d_interpolation": args.skip_2d_interpolation,
            "apply_mirror_filter": not args.no_mirror_filter,
            "use_id_network": args.use_id_network,
            "medaka_data_dir": choose_path(
                args.medaka_data_dir,
                local_paths.medaka_data_dir if local_paths else None,
            ),
            "id_network_epochs": args.id_network_epochs,
            "id_network_batch_size": args.id_network_batch_size,
            "id_network_lr": args.id_network_lr,
            "id_network_device": args.id_network_device,
            "id_network_min_confidence": args.id_network_min_confidence,
        }

    input_value = args.input
    if _looks_like_csv_path(input_value):
        runner_csv(
            input_value,
            sleap_output_dir=sleap_output_dir,
            **common_kwargs,
            **extra_kwargs,
        )
        return

    file_sets_json_path = Path(sleap_output_dir) / "file_sets.json"
    if not file_sets_json_path.exists():
        file_sets = generate_file_sets(sleap_output_dir)
        save_file_sets(file_sets, str(file_sets_json_path))

    runner_key(
        input_value,
        file_sets_json_path=str(file_sets_json_path),
        sleap_output_dir=sleap_output_dir,
        **common_kwargs,
        **extra_kwargs,
    )


if __name__ == "__main__":
    main()
