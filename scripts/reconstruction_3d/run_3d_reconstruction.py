"""
Minimal three-view 3D reconstruction entry point with optional cross-frame ID
tracking and trajectory visualization.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.pipeline import run_pipeline


def run_3d_reconstruction(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    output_dir: Optional[str] = None,
    xyz_bounds: Tuple[float, float] | None = None,
    fill_missing_3d: bool = True,
    plot: bool = True,
    first_frame_id_only: bool = False,
    use_fish_id: bool | None = None,
) -> Dict:
    """
    Reconstruct 3D trajectories from the three processed CSV files for one set.
    """
    out = output_dir or processed_dir
    result = run_pipeline(
        processed_dir=processed_dir,
        set_key=set_key,
        model_folder=model_folder,
        output_dir=out,
        xyz_bounds=xyz_bounds,
        use_fish_id=use_fish_id,
        shift_to_positive=True,
        save_best_json=False,
        plot=plot,
        fill_missing_3d=fill_missing_3d,
        first_frame_id_only=first_frame_id_only,
    )
    return {
        "csv_path": result["csv_path"],
        "png_path": result["png_path"],
        "best_tracked": result["best_tracked"],
        "points_by_frame": result["points_by_frame"],
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Three-view 3D reconstruction with identity tracking")
    p.add_argument(
        "--paths-config",
        default=None,
        help="Optional local TOML path config. Defaults to configs/paths.local.toml if present.",
    )
    p.add_argument("--processed_dir", default=None, help="Directory containing the *_processed.csv files")
    p.add_argument("--set_key", required=True, help="Clip set key")
    p.add_argument("--model_folder", default=None, help="Calibration model directory")
    p.add_argument("--output_dir", default=None, help="Output directory; defaults to processed_dir")
    p.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional valid 3D range [LOW, HIGH]. Omit to keep the model output scale unchanged.",
    )
    p.add_argument("--no_fill_missing", action="store_true", help="Do not fill missing frames in 3D")
    p.add_argument("--no_plot", action="store_true", help="Do not generate the 3D trajectory plot")
    p.add_argument(
        "--first_frame_id_only",
        action="store_true",
        help="Fix the a/b/c -> ID_1/2/3 mapping from the first valid frame only",
    )
    fish_id_group = p.add_mutually_exclusive_group()
    fish_id_group.add_argument(
        "--use_fish_id",
        dest="use_fish_id",
        action="store_true",
        help="Force a/b/c ordering from the processed CSV fish_id column when available",
    )
    fish_id_group.add_argument(
        "--ignore_fish_id",
        dest="use_fish_id",
        action="store_false",
        help="Ignore fish_id and order 2D points by score/x instead",
    )
    p.set_defaults(use_fish_id=None)
    args = p.parse_args()

    local_paths = load_local_paths(args.paths_config)
    processed_dir = choose_path(
        args.processed_dir,
        local_paths.processed_dir if local_paths else None,
    )
    model_folder = choose_path(
        args.model_folder,
        local_paths.calibration_model_dir if local_paths else None,
    )
    output_dir = choose_path(
        args.output_dir,
        local_paths.reconstruction_output_dir if local_paths else None,
        processed_dir,
    )

    if processed_dir is None:
        p.error(
            "processed_dir is required. Pass --processed_dir or define processed_dir "
            "in configs/paths.local.toml."
        )
    if model_folder is None:
        p.error(
            "model_folder is required. Pass --model_folder or define "
            "calibration_model_dir in configs/paths.local.toml."
        )

    res = run_3d_reconstruction(
        processed_dir=processed_dir,
        set_key=args.set_key,
        model_folder=model_folder,
        output_dir=output_dir,
        xyz_bounds=(args.xyz_bounds[0], args.xyz_bounds[1]) if args.xyz_bounds else None,
        fill_missing_3d=not args.no_fill_missing,
        plot=not args.no_plot,
        first_frame_id_only=args.first_frame_id_only,
        use_fish_id=args.use_fish_id,
    )
    if res["csv_path"]:
        print("3D CSV:", res["csv_path"])
    if res["png_path"]:
        print("Trajectory plot:", res["png_path"])


if __name__ == "__main__":
    main()
