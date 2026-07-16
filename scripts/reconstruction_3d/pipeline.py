"""Three-view 2D -> 3D reconstruction -> cross-frame identity tracking."""

from pathlib import Path
from typing import Dict, List, Tuple
import json

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d import run_load_2d
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
    apply_first_frame_id_only,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.visualize_3d import run_plot
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _calibration_metadata,
    _print_calibration_resolution,
)


def run_pipeline(
    processed_dir: str,
    set_key: str,
    model_folder: str,
    *,
    calibration_backend: str = "auto",
    output_dir: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
    use_fish_id: bool | None = None,
    allow_permutations: bool | None = None,
    shift_to_positive: bool = True,
    save_best_json: bool = False,
    plot: bool = True,
    fill_missing_3d: bool = False,
    first_frame_id_only: bool = False,
) -> Dict:
    """
    Reconstruct 3D points across views, track identities across frames, and
    optionally export the result as CSV plus a trajectory plot.
    """
    cal = Calibration(model_folder, backend=calibration_backend)
    calibration_metadata = _calibration_metadata(
        cal,
        cal,
        score_gate=None,
        score_gate_unit=cal.score_unit,
        score_gate_option=None,
        scoring_profile="legacy_combined",
    )
    _print_calibration_resolution(calibration_metadata)
    if allow_permutations is None:
        allow_permutations = bool(cal.capabilities.get("hypothesis_scoring", True))
    points_by_frame = run_load_2d(processed_dir, set_key, use_fish_id=use_fish_id)
    candidates_by_frame = compute_candidates(
        cal,
        points_by_frame,
        xyz_bounds=xyz_bounds,
        allow_permutations=allow_permutations,
    )
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)
    if first_frame_id_only:
        best_tracked = apply_first_frame_id_only(best)
    else:
        best_tracked = apply_cross_frame_tracking(best)

    csv_path = None
    png_path = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_best_json:
            with (out_dir / f"{set_key}_best.json").open("w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in best_tracked.items()}, f, indent=2, ensure_ascii=False)
        provenance_path = out_dir / f"{set_key}_reconstruction_provenance.json"
        provenance_path.write_text(
            json.dumps(calibration_metadata, indent=2) + "\n", encoding="utf-8"
        )

        dfs_3d = best_by_frame_to_dfs(
            best_tracked,
            shift_to_positive=shift_to_positive,
            ensure_non_negative=False,  # Leave missing points as NaN.
        )
        if dfs_3d:
            if fill_missing_3d and points_by_frame:
                all_frames = sorted(points_by_frame.keys())
                dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
            csv_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out_dir))
            if plot:
                png_path = run_plot(str(out_dir), set_key, xyz_lim=xyz_bounds)

    return {
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "csv_path": csv_path,
        "png_path": png_path,
        "calibration": calibration_metadata,
    }


def discover_sets(processed_dir: str) -> List[str]:
    """Return set keys that have all three processed CSV files present."""
    root = Path(processed_dir)
    if not root.is_dir():
        return []
    keys = set()
    for p in root.glob("*_player1_processed.csv"):
        name = p.stem.replace("_player1_processed", "")
        if (root / f"{name}_player2_processed.csv").exists() and (root / f"{name}_player3_processed.csv").exists():
            keys.add(name)
    return sorted(keys)
