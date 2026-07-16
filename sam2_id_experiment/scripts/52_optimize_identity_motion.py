from __future__ import annotations

"""Globally repair biological-ID swaps using motion-continuity evidence.

The optimizer itself lives in ``sam2_id_experiment.src.id_model``.  This module
is deliberately limited to CSV integration, argument handling, and before/after
quality control so it can also be used to audit alternative optimizers.
"""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FISH_IDS = ("F", "M1", "M2")
REQUIRED_COLUMNS = {"frame", "geometry_slot", "id_final", "X", "Y", "Z"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_csv", type=Path, help="Input 3D identity-track CSV.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("layered", "relocate", "viterbi"),
        default="layered",
        help=(
            "Identity repair strategy. 'layered' is the conservative default; "
            "'relocate' only moves existing boundaries; 'viterbi' globally "
            "re-optimizes all six permutations and is experimental."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="QC report path (default: <output-csv stem>.motion_qc.json).",
    )
    parser.add_argument(
        "--jump-threshold-mm",
        type=float,
        default=8.0,
        help="Per-source-frame 3D displacement counted as a jump in QC.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=166.0,
        help="Source cadence, used only to express speed quantiles in mm/s.",
    )
    parser.add_argument("--overwrite", action="store_true")

    conservative = parser.add_argument_group("conservative repair")
    conservative.add_argument(
        "--max-flicker-frames",
        type=int,
        default=166,
        help="Maximum bracketed permutation run removed by --strategy layered.",
    )
    conservative.add_argument(
        "--search-radius-frames",
        type=int,
        default=350,
        help="Maximum boundary relocation radius in source frames.",
    )
    conservative.add_argument(
        "--local-window-radius-frames",
        type=int,
        default=166,
        help=(
            "Radius for bounded six-state minimum-edit repair around jumps "
            "remaining after relocation (layered strategy only)."
        ),
    )
    conservative.add_argument(
        "--allow-above-threshold-relocation",
        action="store_true",
        help=(
            "Allow a boundary move that improves motion but remains above the "
            "jump threshold. Disabled by default for conservative operation."
        ),
    )

    config = parser.add_argument_group("experimental Viterbi overrides")
    config.add_argument("--position-weight", type=float, default=None)
    config.add_argument("--velocity-weight", type=float, default=None)
    config.add_argument("--tangent-weight", type=float, default=None)
    config.add_argument("--acceleration-weight", type=float, default=None)
    config.add_argument("--appearance-weight", type=float, default=None)
    config.add_argument("--switch-penalty", type=float, default=None)
    config.add_argument("--jump-penalty", type=float, default=None)
    config.add_argument("--hard-jump-mm", type=float, default=None)
    config.add_argument("--max-gap-frames", type=int, default=None)
    return parser


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"p50": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": _finite_float(np.quantile(finite, 0.50)),
        "p95": _finite_float(np.quantile(finite, 0.95)),
        "p99": _finite_float(np.quantile(finite, 0.99)),
        "max": _finite_float(np.max(finite)),
    }


def speed_qc(
    track: pd.DataFrame,
    *,
    id_column: str,
    jump_threshold_mm: float,
    source_fps: float,
) -> dict[str, Any]:
    """Summarize consecutive-frame displacement after grouping rows by fish ID."""
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if jump_threshold_mm < 0:
        raise ValueError("jump_threshold_mm must be non-negative")

    data = track[["frame", id_column, "X", "Y", "Z"]].copy()
    data = data[data[id_column].isin(FISH_IDS)]
    data = data.sort_values([id_column, "frame"], kind="stable")
    duplicate_rows = int(data.duplicated([id_column, "frame"], keep=False).sum())
    data = data.drop_duplicates([id_column, "frame"], keep="first")

    per_id: dict[str, Any] = {}
    all_steps: list[np.ndarray] = []
    all_jump_frames: list[np.ndarray] = []
    for fish_id in FISH_IDS:
        fish = data[data[id_column] == fish_id]
        frames = fish["frame"].to_numpy(dtype=np.int64, copy=False)
        xyz = fish[["X", "Y", "Z"]].to_numpy(dtype=float, copy=False)
        if len(fish) >= 2:
            consecutive = np.diff(frames) == 1
            delta = np.diff(xyz, axis=0)
            valid = consecutive & np.isfinite(delta).all(axis=1)
            delta = delta[valid]
            step_frames = frames[1:][valid]
        else:
            delta = np.empty((0, 3), dtype=float)
            step_frames = np.empty(0, dtype=np.int64)
        step = np.linalg.norm(delta, axis=1)
        jumping = step > jump_threshold_mm
        all_steps.append(step)
        all_jump_frames.append(step_frames[jumping])
        per_id[fish_id] = {
            "n_consecutive_steps": int(len(step)),
            "speed_mm_per_frame": _quantiles(step),
            "speed_mm_per_second": _quantiles(step * source_fps),
            "abs_axis_mm_per_frame": {
                axis: _quantiles(np.abs(delta[:, index]))
                for index, axis in enumerate(("X", "Y", "Z"))
            },
            "jumps_over_threshold": int(jumping.sum()),
            "jump_frames": [int(value) for value in step_frames[jumping]],
        }

    combined_steps = np.concatenate(all_steps) if all_steps else np.empty(0)
    combined_jump_frames = (
        np.concatenate(all_jump_frames) if all_jump_frames else np.empty(0, dtype=np.int64)
    )
    return {
        "units": {
            "native_speed": "mm/source-frame",
            "physical_speed": "mm/s",
            "jump_threshold": "mm/source-frame",
        },
        "source_fps": float(source_fps),
        "jump_threshold_mm": float(jump_threshold_mm),
        "duplicate_id_frame_rows_ignored": duplicate_rows,
        "per_id": per_id,
        "all_ids": {
            "n_consecutive_steps": int(len(combined_steps)),
            "speed_mm_per_frame": _quantiles(combined_steps),
            "speed_mm_per_second": _quantiles(combined_steps * source_fps),
            "jumps_over_threshold": int(len(combined_jump_frames)),
            "unique_jump_frames": int(len(np.unique(combined_jump_frames))),
        },
    }


def one_each_validity(track: pd.DataFrame, *, id_column: str) -> dict[str, Any]:
    """Count frames containing exactly one F, one M1, and one M2 row."""
    frame_sizes = track.groupby("frame", sort=True, observed=True).size()
    counts = (
        track.dropna(subset=[id_column])
        .groupby(["frame", id_column], sort=True, observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(frame_sizes.index, fill_value=0)
    )
    valid = frame_sizes.eq(3)
    for fish_id in FISH_IDS:
        column = counts[fish_id] if fish_id in counts else 0
        valid &= column == 1
    invalid_index = valid.index[~valid]
    total = int(len(valid))
    valid_count = int(valid.sum())
    return {
        "total_frames": total,
        "valid_frames": valid_count,
        "invalid_frames": int(len(invalid_index)),
        "valid_fraction": float(valid_count / total) if total else None,
        "invalid_frame_examples": [int(frame) for frame in invalid_index[:20]],
    }


def permutation_qc(track: pd.DataFrame, *, id_column: str) -> dict[str, Any]:
    """Count adjacent-frame changes in the geometry-slot to biological-ID map."""
    compact = track[["frame", "geometry_slot", id_column]].copy()
    compact["geometry_slot"] = pd.to_numeric(compact["geometry_slot"], errors="coerce")
    duplicate_frame = compact.duplicated(["frame", "geometry_slot"], keep=False)
    duplicate_frames = set(compact.loc[duplicate_frame, "frame"].astype(int))
    compact = compact[
        compact["geometry_slot"].isin((0, 1, 2))
        & compact[id_column].isin(FISH_IDS)
    ]
    pivot = compact.pivot(index="frame", columns="geometry_slot", values=id_column)
    pivot = pivot.reindex(columns=[0, 1, 2])
    code_map = {"F": 0, "M1": 1, "M2": 2}
    codes = pivot.replace(code_map).apply(pd.to_numeric, errors="coerce")
    complete = codes.notna().all(axis=1)
    one_each = np.sort(codes.fillna(-1).to_numpy(dtype=np.int8), axis=1)
    valid = complete.to_numpy() & np.all(one_each == np.array([0, 1, 2]), axis=1)
    if duplicate_frames:
        valid &= ~pivot.index.isin(duplicate_frames)
    valid_codes = codes.loc[valid].astype(np.int8)
    mapping_code = (
        valid_codes[0].to_numpy() * 9
        + valid_codes[1].to_numpy() * 3
        + valid_codes[2].to_numpy()
    )
    valid_frames = valid_codes.index.to_numpy(dtype=np.int64)
    adjacent = np.diff(valid_frames) == 1
    changed = adjacent & (np.diff(mapping_code) != 0)
    changes = valid_frames[1:][changed]
    return {
        "valid_permutation_frames": int(len(valid_frames)),
        "adjacent_frame_comparisons": int(adjacent.sum()),
        "permutation_changes": int(len(changes)),
        "change_frames": [int(frame) for frame in changes],
    }


def changed_spans(
    track: pd.DataFrame,
    *,
    before_column: str,
    after_column: str,
) -> list[dict[str, Any]]:
    """Return contiguous source-frame spans containing at least one relabelled row."""
    before = track[before_column].fillna("<NA>").astype(str)
    after = track[after_column].fillna("<NA>").astype(str)
    changed = track[before != after]
    frames = np.sort(changed["frame"].astype(np.int64).unique())
    if not len(frames):
        return []
    boundaries = np.r_[0, np.flatnonzero(np.diff(frames) > 1) + 1, len(frames)]
    spans: list[dict[str, Any]] = []
    for start_index, end_index in zip(boundaries[:-1], boundaries[1:]):
        start = int(frames[start_index])
        end = int(frames[end_index - 1])
        rows = changed[changed["frame"].between(start, end)]
        spans.append(
            {
                "start_frame": start,
                "end_frame": end,
                "n_frames": int(end_index - start_index),
                "changed_rows": int(len(rows)),
                "labels_before": sorted(rows[before_column].dropna().astype(str).unique()),
                "labels_after": sorted(rows[after_column].dropna().astype(str).unique()),
                "geometry_slots": sorted(rows["geometry_slot"].dropna().astype(str).unique()),
            }
        )
    return spans


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "position_weight",
        "velocity_weight",
        "tangent_weight",
        "acceleration_weight",
        "appearance_weight",
        "switch_penalty",
        "jump_penalty",
        "hard_jump_mm",
        "max_gap_frames",
    )
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _strategy_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.strategy == "viterbi":
        return _config_kwargs(args)
    common = {
        "jump_threshold_mm": args.jump_threshold_mm,
        "search_radius_frames": args.search_radius_frames,
        "require_below_threshold": not args.allow_above_threshold_relocation,
    }
    if args.strategy == "layered":
        common["max_flicker_frames"] = args.max_flicker_frames
        common["local_window_radius_frames"] = args.local_window_radius_frames
    return common


def _find_labels(value: Any, original: pd.DataFrame) -> np.ndarray | None:
    """Extract row-aligned labels from the optimizer's intentionally narrow API seam."""
    if isinstance(value, pd.DataFrame):
        label_column = next(
            (name for name in ("id_final", "optimized_id", "id_optimized") if name in value),
            None,
        )
        if label_column is None:
            return None
        if value.index.equals(original.index):
            return value[label_column].to_numpy(dtype=object)
        keys = ["frame", "geometry_slot"]
        if all(name in value for name in keys):
            if value.duplicated(keys).any():
                raise ValueError("optimizer result has duplicate frame/geometry_slot rows")
            aligned = original[keys].merge(
                value[keys + [label_column]], on=keys, how="left", validate="many_to_one"
            )
            if aligned[label_column].isna().any():
                raise ValueError("optimizer result is missing input frame/geometry_slot rows")
            return aligned[label_column].to_numpy(dtype=object)
        if len(value) == len(original):
            return value[label_column].to_numpy(dtype=object)
        return None
    if isinstance(value, pd.Series) and len(value) == len(original):
        return value.to_numpy(dtype=object)
    if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(original):
        return value.astype(object, copy=False)
    for attribute in (
        "optimized_labels",
        "labels",
        "id_final",
        "optimized_track",
        "track",
    ):
        if hasattr(value, attribute):
            labels = _find_labels(getattr(value, attribute), original)
            if labels is not None:
                return labels
    if isinstance(value, tuple):
        for item in value:
            labels = _find_labels(item, original)
            if labels is not None:
                return labels
    return None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _optimizer_diagnostics(result: Any) -> Any:
    for name in ("diagnostics", "report", "metadata"):
        if hasattr(result, name):
            return _jsonable(getattr(result, name))
    if isinstance(result, tuple) and len(result) > 1:
        return _jsonable(result[1])
    return None


def run_optimizer(
    track: pd.DataFrame,
    config_kwargs: dict[str, Any],
    *,
    strategy: str = "viterbi",
) -> tuple[np.ndarray, Any, Any]:
    """Construct and call the optimizer; keep its evolving interface isolated here."""
    if strategy == "layered":
        from sam2_id_experiment.src.id_model.motion_identity_optimizer import (
            LayeredIdentityConfig,
            repair_identity_track_layered,
        )

        config = LayeredIdentityConfig(**config_kwargs)
        result = repair_identity_track_layered(track, config=config)
    elif strategy == "relocate":
        from sam2_id_experiment.src.id_model.motion_identity_optimizer import (
            TransitionRelocationConfig,
            relocate_identity_transitions,
        )

        config = TransitionRelocationConfig(**config_kwargs)
        result = relocate_identity_transitions(track, config=config)
    elif strategy == "viterbi":
        from sam2_id_experiment.src.id_model.motion_identity_optimizer import (
            MotionIdentityConfig,
            optimize_identity_track,
        )

        config = MotionIdentityConfig(**config_kwargs)
        result = optimize_identity_track(track, config=config)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    labels = _find_labels(result, track)
    if labels is None:
        raise TypeError(
            "identity optimizer must return a row-aligned label array/Series, "
            "a DataFrame, or an object exposing .track/.labels/.optimized_labels"
        )
    if len(labels) != len(track):
        raise ValueError("optimizer returned a different number of labels than input rows")
    return labels, config, result


def _validate_input(track: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(track.columns)
    if missing:
        raise ValueError("input track is missing columns: " + ", ".join(sorted(missing)))
    if track.empty:
        raise ValueError("input track is empty")
    numeric_frame = pd.to_numeric(track["frame"], errors="coerce")
    if numeric_frame.isna().any() or not np.allclose(numeric_frame, np.round(numeric_frame)):
        raise ValueError("frame must contain finite integer values")
    if track.duplicated(["frame", "geometry_slot"]).any():
        raise ValueError("input has duplicate frame/geometry_slot rows")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_csv = args.output_csv
    report_json = args.report_json or output_csv.with_name(
        output_csv.stem + ".motion_qc.json"
    )
    input_resolved = args.track_csv.resolve()
    if output_csv.resolve() == input_resolved:
        raise ValueError("--output-csv must not overwrite the input CSV")
    for path in (output_csv, report_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {path}")

    track = pd.read_csv(args.track_csv, low_memory=False)
    _validate_input(track)
    track["frame"] = pd.to_numeric(track["frame"]).astype(np.int64)
    before_speed = speed_qc(
        track,
        id_column="id_final",
        jump_threshold_mm=args.jump_threshold_mm,
        source_fps=args.source_fps,
    )
    before_validity = one_each_validity(track, id_column="id_final")
    before_permutation = permutation_qc(track, id_column="id_final")

    labels, config, optimizer_result = run_optimizer(
        track.copy(),
        _strategy_config_kwargs(args),
        strategy=args.strategy,
    )
    output = track.copy()
    output["id_final_pre_motion"] = output["id_final"]
    output["id_final"] = labels
    changed = (
        output["id_final_pre_motion"].fillna("<NA>").astype(str)
        != output["id_final"].fillna("<NA>").astype(str)
    )
    if "id_source" not in output:
        output["id_source"] = ""
    source_by_strategy = {
        "layered": "motion_layered",
        "relocate": "motion_relocated",
        "viterbi": "motion_global",
    }
    output.loc[changed, "id_source"] = source_by_strategy[args.strategy]

    after_speed = speed_qc(
        output,
        id_column="id_final",
        jump_threshold_mm=args.jump_threshold_mm,
        source_fps=args.source_fps,
    )
    after_validity = one_each_validity(output, id_column="id_final")
    after_permutation = permutation_qc(output, id_column="id_final")
    spans = changed_spans(
        output, before_column="id_final_pre_motion", after_column="id_final"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    report: dict[str, Any] = {
        "schema_version": "motion_identity_qc_v1",
        "input_csv": str(input_resolved),
        "output_csv": str(output_csv.resolve()),
        "rows": int(len(output)),
        "frame_start": int(output["frame"].min()),
        "frame_end": int(output["frame"].max()),
        "strategy": args.strategy,
        "config": _jsonable(config),
        "changed_rows": int(changed.sum()),
        "changed_frames": int(output.loc[changed, "frame"].nunique()),
        "changed_spans": spans,
        "speed_qc": {"before": before_speed, "after": after_speed},
        "permutation_qc": {"before": before_permutation, "after": after_permutation},
        "one_each_validity": {"before": before_validity, "after": after_validity},
    }
    diagnostics = _optimizer_diagnostics(optimizer_result)
    if diagnostics is not None:
        report["optimizer_diagnostics"] = diagnostics
    report_json.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(_jsonable(report), indent=2))


if __name__ == "__main__":
    main()
