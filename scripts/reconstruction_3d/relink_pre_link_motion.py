"""Re-run only the temporal 3D linker on saved pre-link candidates.

This is the one-factor counterpart to :mod:`reconstruct_from_slp`: it reads an
existing ``*_best_pre_link.json``, applies ``apply_cross_frame_tracking`` with
the requested motion settings, and writes an equivalent ``*_best.json`` and
``*_3d.csv``. It never opens SLEAP files or recomputes cross-view geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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


def _derive_output_tag(pre_link_path: Path) -> str:
    stem = pre_link_path.stem
    for suffix in ("_best_pre_link", "_pre_link"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_motion_relink"


def _source_hints(best_by_frame: dict[int, dict]) -> dict[int, dict[int, str]] | None:
    hints = {
        int(frame): {
            int(source_index): str(identity)
            for source_index, identity in info.get("source_id_hints", {}).items()
        }
        for frame, info in best_by_frame.items()
        if info.get("source_id_hints")
    }
    return hints or None


def run_pre_link_motion_relink(
    pre_link_json: Path,
    *,
    output_dir: Path | None = None,
    output_tag: str | None = None,
    num_individuals: int = 3,
    start_frame: int | None = None,
    end_frame: int | None = None,
    max_jump_mm: float | None = None,
    max_jump_hard_cap_mm: float | None = None,
    max_gap_frames: int | None = None,
    motion_velocity_prediction_weight: float = 0.0,
    motion_tangent_weight: float = 0.0,
    motion_acceleration_weight: float = 0.0,
    motion_uncertainty_growth_per_frame: float = 0.25,
    max_motion_gap_frames: int | None = None,
    fill_missing: bool = True,
    shift_to_positive: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Relink one saved candidate sequence without loading SLEAP predictions."""
    input_path = Path(pre_link_json).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Pre-link JSON not found: {input_path}")
    if num_individuals < 1:
        raise ValueError("num_individuals must be at least 1")
    if (
        start_frame is not None
        and end_frame is not None
        and start_frame > end_frame
    ):
        raise ValueError("start_frame must be less than or equal to end_frame")

    tag = output_tag or _derive_output_tag(input_path)
    if not tag or Path(tag).name != tag:
        raise ValueError("output_tag must be a non-empty filename stem")
    out_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    best_json_path = out_dir / f"{tag}_best.json"
    csv_path = out_dir / f"{tag}_3d.csv"
    summary_path = out_dir / f"{tag}_relink_summary.json"
    outputs = (best_json_path, csv_path, summary_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output(s): {joined}")

    best = load_best_pre_link_json(input_path)
    if not best:
        raise ValueError(f"Pre-link JSON contains no frames: {input_path}")
    best = {
        frame: info
        for frame, info in best.items()
        if (start_frame is None or frame >= start_frame)
        and (end_frame is None or frame <= end_frame)
    }
    if not best:
        requested = f"{start_frame if start_frame is not None else 'start'}.."
        requested += f"{end_frame if end_frame is not None else 'end'}"
        raise ValueError(f"No pre-link frames remain after filter {requested}")

    linker_settings = {
        "num_individuals": int(num_individuals),
        "max_jump_mm": max_jump_mm,
        "max_jump_hard_cap_mm": max_jump_hard_cap_mm,
        "max_gap_frames": max_gap_frames,
        "velocity_prediction_weight": motion_velocity_prediction_weight,
        "flow_weight": motion_tangent_weight,
        "acceleration_weight": motion_acceleration_weight,
        "motion_uncertainty_growth_per_frame": (
            motion_uncertainty_growth_per_frame
        ),
        "max_motion_gap_frames": max_motion_gap_frames,
        "source_id_hints_by_frame": _source_hints(best),
        # Match reconstruct_from_slp: hints must not force a gated 3D jump.
        "source_hints_override_motion_gate": False,
    }
    tracked = apply_cross_frame_tracking(best, **linker_settings)

    _save_best_tracked_json(
        tracked,
        best_json_path,
        num_individuals=num_individuals,
    )
    dfs_3d = best_by_frame_to_dfs(
        tracked,
        shift_to_positive=shift_to_positive,
        ensure_non_negative=False,
        num_individuals=num_individuals,
    )
    all_frames = sorted(best)
    if fill_missing:
        dfs_3d = ensure_all_frames_and_fill_spline(
            dfs_3d,
            all_frames,
            max_gap_frames=max_gap_frames,
        )
    written_csv = Path(
        dfs_to_single_csv(dfs_3d, set_key=tag, output_dir=str(out_dir))
    ).resolve()

    summary = {
        "schema_version": "pre_link_motion_relink_v1",
        "input_pre_link_json": str(input_path),
        "num_frames": len(best),
        "requested_frame_range": [start_frame, end_frame],
        "frame_range": [int(all_frames[0]), int(all_frames[-1])],
        "output_tag": tag,
        "outputs": {
            "best_json": str(best_json_path.resolve()),
            "csv_3d": str(written_csv),
        },
        "linker_settings": {
            key: value
            for key, value in linker_settings.items()
            if key != "source_id_hints_by_frame"
        },
        "source_hint_frames": len(linker_settings["source_id_hints_by_frame"] or {}),
        "fill_missing": bool(fill_missing),
        "shift_to_positive": bool(shift_to_positive),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "best_json_path": best_json_path.resolve(),
        "csv_path": written_csv,
        "summary_path": summary_path.resolve(),
        "num_frames": len(best),
        "tracked": tracked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relink a saved *_best_pre_link.json with optional motion costs, "
            "without loading SLEAP or recomputing triangulation."
        )
    )
    parser.add_argument("pre_link_json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default is beside the input JSON.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Output stem; default is <input-tag>_motion_relink.",
    )
    parser.add_argument("--num-individuals", type=int, default=3)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First pre-link frame to include (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last pre-link frame to include (inclusive).",
    )
    parser.add_argument("--max-jump-mm", type=float, default=None)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=None)
    parser.add_argument("--max-gap-frames", type=int, default=None)

    motion = parser.add_argument_group("optional cross-frame motion costs")
    motion.add_argument(
        "--motion-velocity-prediction-weight",
        type=float,
        default=0.0,
    )
    motion.add_argument("--motion-tangent-weight", type=float, default=0.0)
    motion.add_argument("--motion-acceleration-weight", type=float, default=0.0)
    motion.add_argument(
        "--motion-uncertainty-growth-per-frame",
        type=float,
        default=0.25,
    )
    motion.add_argument("--max-motion-gap-frames", type=int, default=None)

    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Leave gaps as NaN instead of interpolating them.",
    )
    parser.add_argument(
        "--no-shift-to-positive",
        action="store_true",
        help="Keep the original coordinate origin.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs with the same output tag.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_pre_link_motion_relink(
            args.pre_link_json,
            output_dir=args.output_dir,
            output_tag=args.output_tag,
            num_individuals=args.num_individuals,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_jump_mm=args.max_jump_mm,
            max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
            max_gap_frames=args.max_gap_frames,
            motion_velocity_prediction_weight=(
                args.motion_velocity_prediction_weight
            ),
            motion_tangent_weight=args.motion_tangent_weight,
            motion_acceleration_weight=args.motion_acceleration_weight,
            motion_uncertainty_growth_per_frame=(
                args.motion_uncertainty_growth_per_frame
            ),
            max_motion_gap_frames=args.max_motion_gap_frames,
            fill_missing=not args.no_fill_missing,
            shift_to_positive=not args.no_shift_to_positive,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Frames          : {result['num_frames']}")
    print(f"3D CSV          : {result['csv_path']}")
    print(f"tracked best    : {result['best_json_path']}")
    print(f"run summary     : {result['summary_path']}")


if __name__ == "__main__":
    main()
