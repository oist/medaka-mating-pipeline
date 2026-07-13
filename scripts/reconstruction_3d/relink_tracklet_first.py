from __future__ import annotations

"""Re-link a tracklet-first pre-link artifact without rerunning triangulation."""

import argparse
from pathlib import Path

from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _save_best_tracked_json,
    load_best_pre_link_json,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pre_link_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--num-individuals", type=int, default=3)
    parser.add_argument("--max-jump-mm", type=float, default=8.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--no-fill-missing", action="store_true")
    args = parser.parse_args(argv)

    best = load_best_pre_link_json(args.pre_link_json)
    hints = {
        frame: dict(info.get("source_id_hints", {}))
        for frame, info in best.items()
        if info.get("source_id_hints")
    }
    tracked = apply_cross_frame_tracking(
        best,
        num_individuals=args.num_individuals,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        source_id_hints_by_frame=hints,
        source_hints_override_motion_gate=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"{args.output_tag}_best.json"
    _save_best_tracked_json(
        tracked, best_path, num_individuals=args.num_individuals
    )
    dfs = best_by_frame_to_dfs(
        tracked,
        shift_to_positive=False,
        ensure_non_negative=False,
        num_individuals=args.num_individuals,
    )
    if not args.no_fill_missing:
        dfs = ensure_all_frames_and_fill_spline(
            dfs, sorted(best), max_gap_frames=args.max_gap_frames
        )
    csv_path = dfs_to_single_csv(
        dfs, set_key=args.output_tag, output_dir=str(args.output_dir)
    )
    print(f"best JSON: {best_path}")
    print(f"3D CSV   : {csv_path}")


if __name__ == "__main__":
    main()
